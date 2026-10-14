//! A command line the program cannot carry out has to fail.
//!
//! CI runs `riichi-cli fuzz` and `riichi-cli arena` as gates and reads only
//! their exit codes. A mistyped command that exits with success, or a number
//! that quietly turns into the default, passes such a gate while testing far
//! less than it says, so both are refused here before anything is played.

use std::process::{Command, Output};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_riichi-cli"))
        .args(arguments)
        .output()
        .expect("riichi-cli starts")
}

#[test]
fn an_unknown_command_fails_with_the_usage() {
    let output = run(&["fuz", "--games", "500"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "nothing is played: {output:?}");
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(said.contains("\"fuz\""), "the command is named: {said}");
    assert!(said.contains("usage: riichi-cli"), "{said}");
}

#[test]
fn a_number_that_does_not_parse_is_refused() {
    let mistyped: [&[&str]; 6] = [
        &["fuzz", "--games", "5e3"],
        &["fuzz", "--games", "5,000"],
        &["fuzz", "--seed", "1", "--games"],
        &["arena", "--games", "1", "--seed", "one"],
        &["duel", "--games", "1", "--dora-worth", "x"],
        &["duel", "--games", "1", "--defender-dora", "-3"],
    ];
    for arguments in mistyped {
        let output = run(arguments);
        assert_eq!(output.status.code(), Some(2), "{arguments:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{arguments:?} played anyway");
        let said = String::from_utf8_lossy(&output.stderr);
        assert!(
            said.contains("takes a whole number"),
            "{arguments:?}: {said}"
        );
    }
}

#[test]
fn help_and_no_command_print_the_usage_and_succeed() {
    let asking: [&[&str]; 3] = [&[], &["help"], &["--help"]];
    for arguments in asking {
        let output = run(arguments);
        assert!(output.status.success(), "{arguments:?}: {output:?}");
        let said = String::from_utf8_lossy(&output.stdout);
        assert!(
            said.starts_with("usage: riichi-cli"),
            "{arguments:?}: {said}"
        );
    }
}

/// The gates as CI spells them still pass, at a size a test can afford.
#[test]
fn a_well_formed_gate_still_runs() {
    let output = run(&["fuzz", "--games", "2", "--seed", "1"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "fuzz: 2 games, no rule broken\n"
    );
    let output = run(&["arena", "--games", "1", "--seed", "1"]);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("arena: 1 games"));
}
