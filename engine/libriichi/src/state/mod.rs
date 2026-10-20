mod action;
mod agent_helper;
mod getter;
mod item;
mod obs_repr;
mod player_state;
mod sp_tables;
mod update;

// The digests are of the lookahead as the encoder runs it, which
// reproducing the C++ version changes.
#[cfg(all(test, not(feature = "sp_reproduce_cpp_ver")))]
mod golden;
#[cfg(test)]
mod test;

#[cfg(feature = "pymod")]
use crate::py_helper::add_submodule;
pub use action::ActionCandidate;
pub use player_state::PlayerState;
pub use sp_tables::SinglePlayerTables;

#[cfg(feature = "pymod")]
use pyo3::prelude::*;

#[cfg(feature = "pymod")]
pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "state")?;
    m.add_class::<ActionCandidate>()?;
    m.add_class::<PlayerState>()?;
    add_submodule(py, prefix, super_mod, &m)
}
