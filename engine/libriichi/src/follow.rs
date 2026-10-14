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

use crate::mjai::Event;
use crate::py_helper::add_submodule;
use crate::state::PlayerState;

use anyhow::{Context, Result};
use half::f16;
use half::slice::HalfFloatSliceExt;
use ndarray::Array2;
use numpy::{Element, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// One game: the four players' states.
struct Table {
    states: [PlayerState; 4],
    /// Players told of their own reach ahead of the table (see `tell`),
    /// whose copy of it from the table is therefore to be skipped.
    told_reach: [bool; 4],
}

impl Table {
    fn new() -> Self {
        Self {
            states: std::array::from_fn(|player| PlayerState::new(player as u8)),
            told_reach: [false; 4],
        }
    }

    fn feed(&mut self, lines: &[String]) -> Result<()> {
        for line in lines {
            let event: Event =
                serde_json::from_str(line).with_context(|| format!("bad mjai line: {line}"))?;
            let already_told = match &event {
                Event::Reach { actor } => {
                    let actor = *actor as usize;
                    let told = self.told_reach[actor];
                    self.told_reach[actor] = false;
                    told.then_some(actor)
                }
                _ => None,
            };
            for (player, state) in self.states.iter_mut().enumerate() {
                if already_told == Some(player) {
                    continue;
                }
                state
                    .update(&event)
                    .with_context(|| format!("player {} rejected: {line}", state.player_id()))?;
            }
        }
        Ok(())
    }
}

/// One decision's observation, kept only where it is not zero: about one
/// value in eighteen is, so this is a twentieth of the dense planes. The
/// values are the encoder's float32, or float16 for a caller that asked.
struct Sparse<V = f32> {
    indices: Vec<u16>,
    values: Vec<V>,
    mask: Vec<bool>,
}

impl Sparse {
    /// The same row with its values in float16, rounded to the nearest and
    /// ties to even, as numpy rounds a float32 array cast to float16: the
    /// bits a caller converting the float32 values itself would get, made
    /// here in the worker that encoded the row rather than afterwards on
    /// the caller's one thread.
    fn halved(self) -> Sparse<f16> {
        let mut values = vec![f16::ZERO; self.values.len()];
        values.convert_from_f32_slice(&self.values);
        Sparse {
            indices: self.indices,
            values,
            mask: self.mask,
        }
    }
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
        table.states[player]
            .update(&event)
            .with_context(|| format!("player {player} rejected: {line}"))?;
        Ok(())
    }

    /// The observations of the players asked for, as `(game, player)`
    /// pairs, in one batch kept sparse: `indptr` of length rows plus one,
    /// and for each row the flat plane-times-34-plus-position `indices`
    /// and `values` between `indptr[row]` and `indptr[row + 1]`; and the
    /// rows' action masks in Mortal's own action space, dense.
    /// `after_reach` previews the player's own declaration on a clone. A
    /// teacher can describe every riichi discard without changing the
    /// follower when the table ultimately chooses an ordinary discard.
    /// `half` hands the values back as float16 instead of float32,
    /// converted by the workers (see `Sparse::halved`), for a caller that
    /// keeps them in float16 anyway and would otherwise convert them itself
    /// on one thread.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (who, after_reach = false, half = false))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        who: Vec<(usize, usize)>,
        after_reach: bool,
        half: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyAny>,
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
        let encode_row = |&(game, player): &(usize, usize)| -> Result<Sparse> {
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
        };

        if half {
            let rows: Vec<Sparse<f16>> = py.detach(|| {
                who.par_iter()
                    .map(|pair| encode_row(pair).map(Sparse::halved))
                    .collect::<Result<Vec<_>>>()
            })?;
            let (indptr, indices, values, masks) = pack(py, rows)?;
            Ok((indptr, indices, values.into_any(), masks))
        } else {
            let rows: Vec<Sparse> =
                py.detach(|| who.par_iter().map(encode_row).collect::<Result<Vec<_>>>())?;
            let (indptr, indices, values, masks) = pack(py, rows)?;
            Ok((indptr, indices, values.into_any(), masks))
        }
    }
}

/// The sparse batch as Python reads it: `indptr` of length rows plus one,
/// the flat `indices` and `values` between consecutive entries, and the
/// rows' masks, dense.
#[allow(clippy::type_complexity)]
fn pack<'py, V: Element + Copy>(
    py: Python<'py>,
    rows: Vec<Sparse<V>>,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<V>>,
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
    add_submodule(py, prefix, super_mod, &m)
}
