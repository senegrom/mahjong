//! The Mortal planes the page reads, digested natively, for the browser to
//! match.
//!
//! The trainer encodes natively, and libriichi's golden test holds that
//! encoder to its old bits. The page runs the same encoder compiled to
//! WebAssembly, where a platform's arithmetic can differ: its `expf` was a
//! unit in the last place off once, and the network read wrong planes in a
//! fifth of the rows with every check green. So this plays a few seeded
//! games as `web/tests/mortal-planes.test.js` plays them through the built
//! module, digests every observation and mask the followed player is asked
//! about, and holds them to `testdata/mortal-planes.digests`. The node test
//! holds the browser's to the same file, so the two builds agree or one of
//! the tests fails.
//!
//! `MORTAL_PLANES_BLESS=1` writes the digests instead of checking them,
//! which is right only for a change that is meant to change the planes or
//! the games.

use std::fmt::Write as _;
use std::{env, fs};

use riichi_core::game::{Action, Phase};

use crate::Game;

const DIGESTS: &str = include_str!("../testdata/mortal-planes.digests");
const DIGESTS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/mortal-planes.digests"
);
/// The games, by seed, against the club players.
const SEEDS: [u32; 4] = [1, 2, 3, 4];

/// FNV-1a over 32-bit words, which the node test computes the same way
/// with `Math.imul`.
struct Fnv(u32);

impl Fnv {
    fn word(&mut self, word: u32) {
        self.0 = (self.0 ^ word).wrapping_mul(0x0100_0193);
    }

    fn planes(&mut self, kind: u32, planes: &[f32], mask: &[u8]) {
        self.word(kind);
        self.word(planes.len() as u32);
        for value in planes {
            self.word(value.to_bits());
        }
        self.word(mask.len() as u32);
        for flag in mask {
            self.word(u32::from(*flag));
        }
    }
}

/// One game played by the node test's rule: pass on any claim, otherwise
/// discard the first tile offered. Every decision's observation and mask
/// are digested, and where riichi is offered, the observation after it.
fn play(seed: u32) -> (usize, u32) {
    let mut game = Game::new(f64::from(seed), Some("club".into()));
    let mut digest = Fnv(0x811c_9dc5);
    let mut rows = 0;
    game.play_on().unwrap();
    for _ in 0..20_000 {
        if game.game_is_over() {
            return (rows, digest.0);
        }
        if game.hand_is_over() {
            game.next_hand().unwrap();
            game.play_on().unwrap();
            continue;
        }
        rows += 1;
        let planes = game.agent_observation_mortal();
        digest.planes(1, &planes, &game.agent_mask_mortal());
        match game.hand.phase {
            Phase::CallWindow => game.choose("pass", None).unwrap(),
            Phase::Act => {
                let actions = game.hand.legal_actions();
                if actions
                    .iter()
                    .any(|action| matches!(action, Action::Riichi(_)))
                {
                    let planes = game.agent_observation_after_reach();
                    digest.planes(2, &planes, &game.agent_mask_after_reach());
                }
                let tile = actions
                    .iter()
                    .find_map(|action| match action {
                        Action::Discard(tile) => Some(*tile),
                        _ => None,
                    })
                    .expect("a discard is always offered");
                game.choose("discard", Some(tile.to_string())).unwrap();
            }
            phase => panic!("seed {seed}: asked to decide in {phase:?}"),
        }
        game.play_on().unwrap();
    }
    panic!("seed {seed}: the game did not end");
}

#[test]
fn the_planes_the_page_reads_are_the_ones_written_down() {
    let got: Vec<(u32, usize, u32)> = SEEDS
        .iter()
        .map(|&seed| {
            let (rows, digest) = play(seed);
            (seed, rows, digest)
        })
        .collect();
    let mut text = String::from(
        "# seed rows digest: the Mortal planes and masks of every decision the\n\
         # followed player meets in a game against the club players (see\n\
         # engine/riichi-wasm/src/mortal_planes_digest.rs)\n",
    );
    for (seed, rows, digest) in &got {
        writeln!(text, "{seed} {rows} {digest:08x}").unwrap();
    }
    if env::var_os("MORTAL_PLANES_BLESS").is_some() {
        fs::write(DIGESTS_PATH, &text).expect("the digests are written");
        return;
    }
    let want: Vec<&str> = DIGESTS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let have: Vec<String> = got
        .iter()
        .map(|(seed, rows, digest)| format!("{seed} {rows} {digest:08x}"))
        .collect();
    assert_eq!(have, want, "the planes or the games changed");
    assert!(
        got.iter().all(|(_, rows, _)| *rows >= 100),
        "too few decisions to say much: {got:?}"
    );
}
