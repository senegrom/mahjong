fn main() {
    // Only the Python extension module needs PyO3's link arguments; a wasm
    // build has no interpreter to link against, and no Python to probe for.
    #[cfg(feature = "pymod")]
    pyo3_build_config::add_extension_module_link_args();
}
