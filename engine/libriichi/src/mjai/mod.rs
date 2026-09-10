#[cfg(feature = "pymod")]
mod bot;
mod event;

pub use event::{Event, EventExt, EventWithCanAct, Metadata, OutOfBoundError};

#[cfg(feature = "pymod")]
use crate::py_helper::add_submodule;
#[cfg(feature = "pymod")]
use bot::Bot;

#[cfg(feature = "pymod")]
use pyo3::prelude::*;

#[cfg(feature = "pymod")]
pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "mjai")?;
    m.add_class::<Bot>()?;
    add_submodule(py, prefix, super_mod, &m)
}
