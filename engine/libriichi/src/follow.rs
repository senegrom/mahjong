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
}

impl Table {
    fn new() -> Self {
        Self {
            states: std::array::from_fn(|player| PlayerState::new(player as u8)),
            cans: [ActionCandidate::default(); 4],
            oya: 0,
        }
    }

    fn feed(&mut self, lines: &[String]) -> Result<()> {
        for line in lines {
            let event: Event =
                serde_json::from_str(line).with_context(|| format!("bad mjai line: {line}"))?;
            if let Event::StartKyoku { oya, .. } = &event {
                self.oya = *oya;
            }
            for (state, cans) in self.states.iter_mut().zip(self.cans.iter_mut()) {
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
        py.allow_threads(|| {
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
    #[allow(clippy::type_complexity)]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        who: Vec<(usize, usize)>,
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
        let rows: Vec<Sparse> = py.allow_threads(|| {
            who.par_iter()
                .map(|&(game, player)| {
                    let (obs, mask) = tables[game].states[player].encode_obs(version, false);
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
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "follow")?;
    m.add_class::<Follower>()?;
    add_submodule(py, prefix, super_mod, &m)
}
