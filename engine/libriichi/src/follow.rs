//! Following many games at once, for a trainer that plays a thousand
//! tables a step and wants Mortal's view of every decision.
//!
//! A `Follower` holds four player states per game. It is fed each game's
//! new mjai events as JSON lines and asked for the observations of whichever
//! players owe a decision, in one sparse batch. Both run across games in
//! parallel without the GIL: the observation's efficiency lookahead costs
//! milliseconds a decision, and a step has hundreds of decisions.
//!
//! This module is not part of Mortal; it was added for the mahjong project
//! that vendors this crate, and is licensed as the rest of it.

use crate::consts::obs_shape;
use crate::mjai::Event;
use crate::py_helper::add_submodule;
use crate::state::{ActionCandidate, PlayerState};
use crate::tile::Tile;

use anyhow::{Context, Result};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// One game: the four players' states and what each may do.
struct Table {
    states: [PlayerState; 4],
    cans: [ActionCandidate; 4],
    oya: u8,
    /// Players told of their own reach ahead of the table (see `tell`),
    /// whose copy of it from the table is therefore to be skipped.
    told_reach: [bool; 4],
}

impl Table {
    fn new() -> Self {
        Self {
            states: std::array::from_fn(|player| PlayerState::new(player as u8)),
            cans: [ActionCandidate::default(); 4],
            oya: 0,
            told_reach: [false; 4],
        }
    }

    fn feed(&mut self, lines: &[String]) -> Result<()> {
        for line in lines {
            let event: Event =
                serde_json::from_str(line).with_context(|| format!("bad mjai line: {line}"))?;
            if let Event::StartKyoku { oya, .. } = &event {
                self.oya = *oya;
            }
            let already_told = match &event {
                Event::Reach { actor } => {
                    let actor = *actor as usize;
                    let told = self.told_reach[actor];
                    self.told_reach[actor] = false;
                    told.then_some(actor)
                }
                _ => None,
            };
            for (player, (state, cans)) in
                self.states.iter_mut().zip(self.cans.iter_mut()).enumerate()
            {
                if already_told == Some(player) {
                    continue;
                }
                *cans = state
                    .update(&event)
                    .with_context(|| format!("player {} rejected: {line}", state.player_id()))?;
            }
        }
        Ok(())
    }
}

/// One decision's observation, kept only where it is not zero: about one
/// value in eighteen is, so this is a twentieth of the dense planes.
struct Sparse {
    indices: Vec<u16>,
    values: Vec<f32>,
    mask: Vec<bool>,
}

/// Many games' player states, fed events and asked for observations in bulk.
#[pyclass]
pub struct Follower {
    tables: Vec<Table>,
    version: u32,
}

#[pymethods]
impl Follower {
    #[new]
    #[pyo3(signature = (games, version = 4))]
    fn new(games: usize, version: u32) -> Self {
        Self {
            tables: (0..games).map(|_| Table::new()).collect(),
            version,
        }
    }

    fn __len__(&self) -> usize {
        self.tables.len()
    }

    /// The observation's shape, `(planes, 34)`.
    fn shape(&self) -> (usize, usize) {
        obs_shape(self.version)
    }

    /// Feeds every game its new events, one list of JSON lines per game in
    /// the order they happened. An empty list is a game with nothing new.
    /// Games are read in parallel.
    fn feed(&mut self, py: Python<'_>, lines: Vec<Vec<String>>) -> PyResult<()> {
        if lines.len() != self.tables.len() {
            return Err(PyValueError::new_err(format!(
                "{} lists of events for {} games",
                lines.len(),
                self.tables.len()
            )));
        }
        let tables = &mut self.tables;
        py.detach(|| {
            tables
                .par_iter_mut()
                .zip(lines.par_iter())
                .try_for_each(|(table, lines)| table.feed(lines))
        })?;
        Ok(())
    }

    /// Whether `player` in `game` may act after what it was last fed.
    fn can_act(&self, game: usize, player: usize) -> bool {
        self.tables[game].cans[player].can_act()
    }

    /// Tells one player about an event ahead of the table. A Mortal that
    /// decides to declare riichi is asked, in a second step, which tile to
    /// throw with it, and answers that from a state in which the reach is
    /// already declared: this feeds it that reach. When the table's own
    /// copy of the event arrives, that player skips it.
    fn tell(&mut self, game: usize, player: usize, line: &str) -> PyResult<()> {
        if game >= self.tables.len() || player >= 4 {
            return Err(PyValueError::new_err(format!(
                "no player {player} in game {game}"
            )));
        }
        let event: Event = serde_json::from_str(line)
            .map_err(|error| PyValueError::new_err(format!("bad mjai line: {line}: {error}")))?;
        let table = &mut self.tables[game];
        if matches!(&event, Event::Reach { actor } if *actor as usize == player) {
            table.told_reach[player] = true;
        }
        table.cans[player] = table.states[player]
            .update(&event)
            .with_context(|| format!("player {player} rejected: {line}"))?;
        Ok(())
    }

    /// The dealer's player number in each game's current hand, which is
    /// what turns a seat into a player: the player at seat `s` is
    /// `(oya + s) % 4`.
    fn dealers(&self) -> Vec<u8> {
        self.tables.iter().map(|table| table.oya).collect()
    }

    /// The observations of the players asked for, as `(game, player)`
    /// pairs, in one batch kept sparse: `indptr` of length rows plus one,
    /// and for each row the flat plane-times-34-plus-position `indices`
    /// and `values` between `indptr[row]` and `indptr[row + 1]`; and the
    /// rows' action masks in Mortal's own action space, dense.
    /// `after_reach` previews the player's own declaration on a clone. A
    /// teacher can describe every riichi discard without changing the
    /// follower when the table ultimately chooses an ordinary discard.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (who, after_reach = false))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        who: Vec<(usize, usize)>,
        after_reach: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<bool>>,
    )> {
        for &(game, player) in &who {
            if game >= self.tables.len() || player >= 4 {
                return Err(PyValueError::new_err(format!(
                    "no player {player} in game {game}"
                )));
            }
        }
        let version = self.version;
        let tables = &self.tables;
        let rows: Vec<Sparse> = py.detach(|| {
            who.par_iter()
                .map(|&(game, player)| -> Result<Sparse> {
                    let original = &tables[game].states[player];
                    let preview = if after_reach {
                        let mut state = original.clone();
                        state.update(&Event::Reach {
                            actor: player as u8,
                        })?;
                        Some(state)
                    } else {
                        None
                    };
                    let state = preview.as_ref().unwrap_or(original);
                    let (obs, mask) = state.encode_obs(version, false);
                    let mut indices = Vec::with_capacity(2048);
                    let mut values = Vec::with_capacity(2048);
                    for (index, &value) in obs.iter().enumerate() {
                        if value != 0. {
                            indices.push(index as u16);
                            values.push(value);
                        }
                    }
                    Ok(Sparse {
                        indices,
                        values,
                        mask: mask.to_vec(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;

        pack(py, rows)
    }
}

/// Many imagined continuations of one real table, each carrying its own
/// copy of a player's state.
///
/// A search deals worlds it cannot see and plays them forward, and wants
/// Mortal's view of every position it reaches. Replaying each from the deal
/// would cost more than the search, and is unnecessary: a continuation
/// starts from a position the follower already holds, so the state is
/// cloned and only the events the continuation invented are applied to the
/// copy. Nothing here touches the real table.
#[pyclass]
pub struct Imagined {
    states: Vec<PlayerState>,
    version: u32,
}

#[pymethods]
impl Imagined {
    /// One state per slot, each copied from the real player named by its
    /// `(game, player)` pair. The same real player may be named by any
    /// number of slots; each gets its own copy.
    #[staticmethod]
    #[pyo3(signature = (follower, who, version = 4))]
    fn from_follower(
        follower: &Follower,
        who: Vec<(usize, usize)>,
        version: u32,
    ) -> PyResult<Self> {
        let mut states = Vec::with_capacity(who.len());
        for (game, player) in who {
            if game >= follower.tables.len() || player >= 4 {
                return Err(PyValueError::new_err(format!(
                    "no player {player} in game {game}"
                )));
            }
            states.push(follower.tables[game].states[player].clone());
        }
        Ok(Self { states, version })
    }

    fn __len__(&self) -> usize {
        self.states.len()
    }

    /// The observation's shape, `(planes, 34)`.
    fn shape(&self) -> (usize, usize) {
        obs_shape(self.version)
    }

    /// Feeds each slot the events its own continuation invented, one list
    /// of JSON lines per slot in the order they happened. Slots are read in
    /// parallel; a slot whose line its state refuses is reported by index,
    /// because a search that has drifted from the rules must be found, not
    /// quietly valued.
    fn feed(&mut self, py: Python<'_>, lines: Vec<Vec<String>>) -> PyResult<()> {
        if lines.len() != self.states.len() {
            return Err(PyValueError::new_err(format!(
                "{} lists of events for {} slots",
                lines.len(),
                self.states.len()
            )));
        }
        let states = &mut self.states;
        py.detach(|| {
            states
                .par_iter_mut()
                .zip(lines.par_iter())
                .enumerate()
                .try_for_each(|(slot, (state, lines))| -> Result<()> {
                    for line in lines {
                        let event: Event = serde_json::from_str(line)
                            .with_context(|| format!("slot {slot}: bad mjai line: {line}"))?;
                        state
                            .update(&event)
                            .with_context(|| format!("slot {slot} rejected: {line}"))?;
                    }
                    Ok(())
                })
        })?;
        Ok(())
    }

    /// Feeds the named slots, each its own list of lines, leaving the rest
    /// as they are: a lookahead asks one seat at a time and tells only the
    /// slots that have moved.
    fn feed_some(
        &mut self,
        py: Python<'_>,
        which: Vec<usize>,
        lines: Vec<Vec<String>>,
    ) -> PyResult<()> {
        if which.len() != lines.len() {
            return Err(PyValueError::new_err(format!(
                "{} lists of events for {} slots",
                lines.len(),
                which.len()
            )));
        }
        for &slot in &which {
            if slot >= self.states.len() {
                return Err(PyValueError::new_err(format!("no slot {slot}")));
            }
        }
        // Distinct slots, so they can be fed in parallel without aliasing:
        // every state is visited once, and only the named ones are fed.
        let mut given: Vec<Option<usize>> = vec![None; self.states.len()];
        for (at, &slot) in which.iter().enumerate() {
            if given[slot].is_some() {
                return Err(PyValueError::new_err(format!("slot {slot} named twice")));
            }
            given[slot] = Some(at);
        }
        let states = &mut self.states;
        py.detach(|| {
            states
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(slot, state)| -> Result<()> {
                    let Some(at) = given[slot] else {
                        return Ok(());
                    };
                    for line in &lines[at] {
                        let event: Event = serde_json::from_str(line)
                            .with_context(|| format!("slot {slot}: bad mjai line: {line}"))?;
                        state
                            .update(&event)
                            .with_context(|| format!("slot {slot} rejected: {line}"))?;
                    }
                    Ok(())
                })
        })?;
        Ok(())
    }

    /// Gives the named slots the concealed tiles their imagined worlds
    /// dealt them, as mjai tile names: a search keeps a copy of every
    /// seat's state per world, and the seats it imagined hold the world's
    /// tiles, not the real seat's. See `PlayerState::replace_concealed`.
    fn replace_concealed(&mut self, which: Vec<usize>, hands: Vec<Vec<String>>) -> PyResult<()> {
        if which.len() != hands.len() {
            return Err(PyValueError::new_err(format!(
                "{} hands for {} slots",
                hands.len(),
                which.len()
            )));
        }
        for (&slot, hand) in which.iter().zip(&hands) {
            let Some(state) = self.states.get_mut(slot) else {
                return Err(PyValueError::new_err(format!("no slot {slot}")));
            };
            let mut counts = [0u8; 34];
            for name in hand {
                let tile: Tile = name
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("slot {slot}: bad tile {name}")))?;
                counts[tile.deaka().as_usize()] += 1;
            }
            state
                .replace_concealed(&counts)
                .map_err(|error| PyValueError::new_err(format!("slot {slot}: {error}")))?;
        }
        Ok(())
    }

    /// Copies of the named slots, as a new set: what a question asked
    /// ahead of the table -- which tile a reach throws -- is put to, so the
    /// slot itself is not told a reach the search may decide against.
    fn clone_some(&self, which: Vec<usize>) -> PyResult<Self> {
        let mut states = Vec::with_capacity(which.len());
        for slot in which {
            let Some(state) = self.states.get(slot) else {
                return Err(PyValueError::new_err(format!("no slot {slot}")));
            };
            states.push(state.clone());
        }
        Ok(Self {
            states,
            version: self.version,
        })
    }

    /// The named slots' observations, in the order named, kept sparse in
    /// the same layout [`Follower::encode`] uses.
    #[allow(clippy::type_complexity)]
    fn encode_some<'py>(
        &self,
        py: Python<'py>,
        which: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<bool>>,
    )> {
        for &slot in &which {
            if slot >= self.states.len() {
                return Err(PyValueError::new_err(format!("no slot {slot}")));
            }
        }
        let version = self.version;
        let states = &self.states;
        let rows: Vec<Sparse> = py.detach(|| {
            which
                .par_iter()
                .map(|&slot| {
                    let (obs, mask) = states[slot].encode_obs(version, false);
                    let mut indices = Vec::with_capacity(2048);
                    let mut values = Vec::with_capacity(2048);
                    for (index, &value) in obs.iter().enumerate() {
                        if value != 0. {
                            indices.push(index as u16);
                            values.push(value);
                        }
                    }
                    Sparse {
                        indices,
                        values,
                        mask: mask.to_vec(),
                    }
                })
                .collect()
        });
        pack(py, rows)
    }

    /// Every slot's observation, in slot order, kept sparse in the same
    /// layout [`Follower::encode`] uses.
    #[allow(clippy::type_complexity)]
    fn encode<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<bool>>,
    )> {
        let version = self.version;
        let rows: Vec<Sparse> = py.detach(|| {
            self.states
                .par_iter()
                .map(|state| {
                    let (obs, mask) = state.encode_obs(version, false);
                    let mut indices = Vec::with_capacity(2048);
                    let mut values = Vec::with_capacity(2048);
                    for (index, &value) in obs.iter().enumerate() {
                        if value != 0. {
                            indices.push(index as u16);
                            values.push(value);
                        }
                    }
                    Sparse {
                        indices,
                        values,
                        mask: mask.to_vec(),
                    }
                })
                .collect()
        });
        pack(py, rows)
    }
}

/// The sparse batch as Python reads it: `indptr` of length rows plus one,
/// the flat `indices` and `values` between consecutive entries, and the
/// rows' masks, dense.
#[allow(clippy::type_complexity)]
fn pack<'py>(
    py: Python<'py>,
    rows: Vec<Sparse>,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<bool>>,
)> {
    let total: usize = rows.iter().map(|row| row.indices.len()).sum();
    let width = rows.first().map_or(0, |row| row.mask.len());
    let mut indptr = Vec::with_capacity(rows.len() + 1);
    let mut indices = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    let mut masks = Vec::with_capacity(rows.len() * width);
    indptr.push(0_i32);
    for row in rows {
        indices.extend_from_slice(&row.indices);
        values.extend_from_slice(&row.values);
        masks.extend_from_slice(&row.mask);
        indptr.push(indices.len() as i32);
    }
    let masks = Array2::from_shape_vec((indptr.len() - 1, width), masks)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, values),
        PyArray2::from_owned_array(py, masks),
    ))
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "follow")?;
    m.add_class::<Follower>()?;
    m.add_class::<Imagined>()?;
    add_submodule(py, prefix, super_mod, &m)
}
