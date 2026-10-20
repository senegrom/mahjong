//! The encoder's golden fixture, replayed through the player states.
//!
//! `testdata/encoder-golden/steps.jsonl.gz` holds whole games recorded from
//! the mahjong project's arena, step by step as its self-play feeds them to
//! the follower (`neural/tests/encoder_golden_fixture.py` there records it).
//! This replays them through `PlayerState`, and at every decision self-play
//! asked about digests the observation's planes and mask and the efficiency
//! lookahead's own table, as the player stands and, where our rules let it
//! declare riichi, with its reach previewed on a copy: one digest per hand.
//! `lookahead.digests` holds what this crate gave when the fixture was
//! blessed, so a change meant to be exact has to reproduce them to the bit.
//! The trainer's test of the same games (`neural/tests/test_encoder_golden.py`)
//! holds `Follower.encode`'s output to its old bytes in the same way. Below
//! them, a few hands put to the lookahead directly hold it to its old
//! answers where the games rarely go, and the encoder's table of the river's
//! decay is checked to be correctly rounded, as the native libms give it.
//!
//! `ENCODER_GOLDEN_BLESS=1` writes the digests instead of checking them,
//! which is right only for a change that is meant to change the planes.
//!
//! This module is not part of Mortal; it was added for the mahjong project
//! that vendors this crate, and is licensed as the rest of it.

use super::PlayerState;
use crate::algo::shanten;
use crate::algo::sp::{Candidate, InitState, SPCalculator};
use crate::hand::hand;
use crate::mjai::Event;
use crate::tile::Tile;
use crate::{t, tu8};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Read;
use std::{env, fs, thread};

use flate2::read::GzDecoder;
use serde::Deserialize;

const STEPS: &[u8] = include_bytes!("../../testdata/encoder-golden/steps.jsonl.gz");
const DIGESTS: &str = include_str!("../../testdata/encoder-golden/lookahead.digests");
const DIGESTS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/encoder-golden/lookahead.digests"
);
/// Mortal's observation version 4, the one the trainer and the site encode.
const VERSION: u32 = 4;

/// One line of the fixture: every game's new mjai events, then the players
/// asked for, as `[game, player, preview]`.
#[derive(Deserialize)]
struct Step {
    feed: Vec<(usize, Vec<String>)>,
    who: Vec<(usize, u8, u8)>,
}

/// One game, in the order self-play met it.
enum Item {
    Event(Event),
    Decision { player: u8, preview: bool },
}

/// FNV-1a over 64 bits, written out so that the digests depend on nothing
/// a dependency or the standard library may change between releases.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn count(&mut self, count: usize) {
        self.bytes(&(count as u32).to_le_bytes());
    }

    fn floats<'a>(&mut self, values: impl IntoIterator<Item = &'a f32>) {
        for value in values {
            self.bytes(&value.to_bits().to_le_bytes());
        }
    }
}

/// What the replayed decisions covered, so that a fixture too thin to guard
/// the lookahead's costly rows fails here rather than passing unnoticed.
#[derive(Default)]
struct Coverage {
    rows: usize,
    /// Discards at three shanten, where the lookahead does the most work.
    three_shanten_discards: usize,
    /// Of those, the ones early in the hand, with most of the wall to draw.
    early_three_shanten_discards: usize,
    /// Answers to another player's discard that may chi or pon.
    calls: usize,
    previews: usize,
    tables: usize,
}

impl Coverage {
    const fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.three_shanten_discards += other.three_shanten_discards;
        self.early_three_shanten_discards += other.early_three_shanten_discards;
        self.calls += other.calls;
        self.previews += other.previews;
        self.tables += other.tables;
    }
}

fn games() -> Vec<Vec<Item>> {
    let mut text = String::new();
    GzDecoder::new(STEPS)
        .read_to_string(&mut text)
        .expect("the fixture is gzipped UTF-8");
    let mut games: Vec<Vec<Item>> = vec![];
    for line in text.lines() {
        let step: Step = serde_json::from_str(line).expect("a step of the fixture");
        for (game, lines) in step.feed {
            if games.len() <= game {
                games.resize_with(game + 1, Vec::new);
            }
            for line in lines {
                let event = serde_json::from_str(&line).expect("an mjai line of the fixture");
                games[game].push(Item::Event(event));
            }
        }
        for (game, player, preview) in step.who {
            games[game].push(Item::Decision {
                player,
                preview: preview != 0,
            });
        }
    }
    games
}

fn candidate(digest: &mut Fnv, candidate: &Candidate) {
    digest.bytes(&[candidate.tile.as_u8()]);
    for values in [
        &candidate.tenpai_probs,
        &candidate.win_probs,
        &candidate.exp_values,
    ] {
        digest.count(values.len());
        digest.floats(values.iter());
    }
    digest.count(candidate.required_tiles.len());
    for required in &candidate.required_tiles {
        digest.bytes(&[required.tile.as_u8(), required.count]);
    }
    digest.bytes(&[
        candidate.num_required_tiles,
        u8::from(candidate.shanten_down),
    ]);
}

/// The planes and the mask the encoder makes of a state, and the
/// lookahead's table in full: each candidate's every turn, unrounded, which
/// the planes keep only in part.
fn row(digest: &mut Fnv, state: &PlayerState, coverage: &mut Coverage) {
    let (obs, mask) = state.encode_obs(VERSION, false);
    digest.count(obs.len());
    digest.floats(obs.iter());
    digest.bytes(&mask.iter().map(|&m| u8::from(m)).collect::<Vec<_>>());
    match state.single_player_tables() {
        Ok(tables) => {
            coverage.tables += 1;
            digest.bytes(&[1]);
            digest.count(tables.max_ev_table.len());
            for c in &tables.max_ev_table {
                candidate(digest, c);
            }
        }
        Err(_) => digest.bytes(&[0]),
    }
}

/// Every hand's digest of one game, and what its decisions covered.
fn replay(items: &[Item]) -> (Vec<u64>, Coverage) {
    let mut states: [PlayerState; 4] = std::array::from_fn(|player| PlayerState::new(player as u8));
    let mut hands: Vec<Fnv> = vec![];
    let mut coverage = Coverage::default();
    for item in items {
        match item {
            Item::Event(event) => {
                if matches!(event, Event::StartKyoku { .. }) {
                    hands.push(Fnv::new());
                }
                for state in &mut states {
                    state.update(event).expect("the fixture's events are valid");
                }
            }
            Item::Decision { player, preview } => {
                let (player, preview) = (*player, *preview);
                let state = &states[player as usize];
                let digest = hands.last_mut().expect("decisions come after a deal");
                digest.bytes(&[player, u8::from(preview)]);
                coverage.rows += 1;
                let cans = state.last_cans;
                if cans.can_discard && state.real_time_shanten() == 3 {
                    coverage.three_shanten_discards += 1;
                    if state.tiles_left >= 56 {
                        coverage.early_three_shanten_discards += 1;
                    }
                }
                if !cans.can_discard && (cans.can_chi() || cans.can_pon) {
                    coverage.calls += 1;
                }
                row(digest, state, &mut coverage);
                if preview {
                    coverage.previews += 1;
                    let mut declared = state.clone();
                    match declared.update(&Event::Reach { actor: player }) {
                        Ok(_) => row(digest, &declared, &mut coverage),
                        Err(_) => digest.bytes(&[0xFF]),
                    }
                }
            }
        }
    }
    (hands.into_iter().map(|digest| digest.0).collect(), coverage)
}

/// Replays every game, spread over the processors.
fn replay_games() -> (BTreeMap<(usize, usize), u64>, Coverage) {
    let games = games();
    let count = games.len();
    let workers = thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(count);
    let results: Vec<(usize, Vec<u64>, Coverage)> = thread::scope(|scope| {
        // Every thread is spawned before any is joined; joined as they were
        // spawned, the games would replay one after another.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let games = &games;
                // The lookahead recurses with arrays on the stack, which an
                // unoptimised build does not keep small.
                thread::Builder::new()
                    .stack_size(16 << 20)
                    .spawn_scoped(scope, move || {
                        (worker..count)
                            .step_by(workers)
                            .map(|game| {
                                let (hands, coverage) = replay(&games[game]);
                                (game, hands, coverage)
                            })
                            .collect::<Vec<_>>()
                    })
                    .expect("a replay thread")
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a replay thread"))
            .collect()
    });
    let mut digests = BTreeMap::new();
    let mut coverage = Coverage::default();
    for (game, hands, covered) in results {
        coverage.add(&covered);
        for (hand, digest) in hands.into_iter().enumerate() {
            digests.insert((game, hand), digest);
        }
    }
    (digests, coverage)
}

fn read_digests() -> BTreeMap<(usize, usize), u64> {
    DIGESTS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let game = fields[0].parse().expect("a game number");
            let hand = fields[1].parse().expect("a hand number");
            let digest = u64::from_str_radix(fields[2], 16).expect("a digest");
            ((game, hand), digest)
        })
        .collect()
}

#[test]
fn the_recorded_games_encode_as_they_always_have() {
    if env::var_os("ENCODER_GOLDEN_BLESS").is_some() {
        let (digests, coverage) = replay_games();
        let mut text = String::from(
            "# game hand digest: PlayerState::encode_obs and the lookahead's table at every\n\
             # decision of the hand (see engine/libriichi/src/state/golden.rs)\n",
        );
        for ((game, hand), digest) in &digests {
            writeln!(text, "{game} {hand} {digest:016x}").unwrap();
        }
        fs::write(DIGESTS_PATH, text).expect("the digests are written");
        println!(
            "blessed {} hands of {} rows into {DIGESTS_PATH}",
            digests.len(),
            coverage.rows
        );
        return;
    }

    let (got, coverage) = replay_games();
    let want = read_digests();
    assert!(want.len() >= 150, "{} hands to check", want.len());
    let wrong: Vec<_> = want
        .iter()
        .filter(|&(hand, digest)| got.get(hand) != Some(digest))
        .map(|((game, hand), _)| format!("game {game} hand {hand}"))
        .collect();
    assert!(
        wrong.is_empty() && got.len() == want.len(),
        "{} of {} hands encode differently, first {:?}; a change meant to be exact is not",
        wrong.len(),
        want.len(),
        &wrong[..wrong.len().min(5)],
    );

    // What the replayed games exercise, so that a thinner fixture cannot
    // pass this unnoticed: the lookahead's costliest rows, calls, and
    // previews of a reach. The fixture as recorded has 10,786 rows, 1,455
    // discards at three shanten (906 of them with 56 tiles or more left in
    // the wall), 2,223 answers that may call and 240 previews.
    println!(
        "{} rows: {} with the lookahead's table, {} discards at three shanten ({} early), \
         {} calls, {} reach previews",
        coverage.rows,
        coverage.tables,
        coverage.three_shanten_discards,
        coverage.early_three_shanten_discards,
        coverage.calls,
        coverage.previews,
    );
    assert!(coverage.rows >= 10_000);
    assert!(coverage.three_shanten_discards >= 1_000);
    assert!(coverage.early_three_shanten_discards >= 500);
    assert!(coverage.calls >= 1_500);
    assert!(coverage.previews >= 200);
}

/// The river's decay, exp(-0.2 k) for whole k, is the one transcendental
/// function in version 4's planes, and the encoder reads it from a table
/// (`river_decay`) for k below 64, so that every build writes the same
/// planes. Each entry is the correctly rounded float32: the exponential in
/// double of the float32 argument the encoder used to compute, rounded once,
/// which is exact here because the nearest entry to a rounding tie is 0.0027
/// of a unit in the last place away (k = 26). This platform's float32 `exp`,
/// which the encoder called before the table, gives the same, as glibc's and
/// the Windows runtime's do: the table changed no plane a trainer encodes.
#[test]
fn the_river_decays_correctly_rounded_in_every_build() {
    for k in 0..64 {
        let argument = -0.2 * k as f32;
        let rounded = (argument as f64).exp() as f32;
        let decay = super::obs_repr::river_decay(k);
        assert_eq!(decay.to_bits(), rounded.to_bits(), "k = {k}: {decay:e}, correctly rounded {rounded:e}");
        assert_eq!(
            argument.exp().to_bits(),
            rounded.to_bits(),
            "exp(-0.2 * {k}) is {:e} here, where correctly rounded it is {rounded:e}",
            argument.exp(),
        );
    }
    // Past the table, as no river reaches, the function itself.
    assert_eq!(super::obs_repr::river_decay(64), (-0.2 * 64_f32).exp());
}

/// One hand put to the lookahead directly (see `the_lookahead_answers_as_it_always_has`).
struct Case {
    hand: &'static str,
    pons: &'static [u8],
    dora_indicator: Tile,
    jikaze: u8,
    can_discard: bool,
    tsumos_left: u8,
    /// Tiles seen besides the hand, the dora indicator and the pons.
    seen: &'static str,
    akas_seen: [bool; 3],
    /// Whether to ask with every option, or only as the encoder asks.
    every_option: bool,
}

const CASES: [Case; 8] = [
    // Mortal's own `nanikiru` and `tsumo_only` hands.
    Case {
        hand: "45678m 34789p 3344z",
        pons: &[],
        dora_indicator: t!(P),
        jikaze: tu8!(N),
        can_discard: true,
        tsumos_left: 8,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    Case {
        hand: "3667m 23489p 34688s",
        pons: &[],
        dora_indicator: t!(P),
        jikaze: tu8!(N),
        can_discard: true,
        tsumos_left: 15,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    Case {
        hand: "45677m 456778p 248s",
        pons: &[],
        dora_indicator: t!(6m),
        jikaze: tu8!(E),
        can_discard: true,
        tsumos_left: 15,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    Case {
        hand: "45677m 456778p 48s",
        pons: &[],
        dora_indicator: t!(6m),
        jikaze: tu8!(W),
        can_discard: false,
        tsumos_left: 5,
        seen: "5555s",
        akas_seen: [false, false, true],
        every_option: true,
    },
    // Every unseen tile improves the hand once a 1m is thrown, or as it
    // stands, so not drawing one is impossible after the first draw and
    // `not_tsumo_probs` falls to zero inside the horizon.
    Case {
        hand: "9999m 6677p 88s 335z 1m",
        pons: &[],
        dora_indicator: t!(1m),
        jikaze: tu8!(W),
        can_discard: true,
        tsumos_left: 5,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    Case {
        hand: "9999m 6677p 88s 335z",
        pons: &[],
        dora_indicator: t!(1m),
        jikaze: tu8!(W),
        can_discard: false,
        tsumos_left: 5,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    // After a pon of white dragons: three sets to make, and no riichi.
    Case {
        hand: "234m 58p 678s 113z",
        pons: &[tu8!(P)],
        dora_indicator: t!(1s),
        jikaze: tu8!(S),
        can_discard: true,
        tsumos_left: 12,
        seen: "",
        akas_seen: [false; 3],
        every_option: true,
    },
    // Three shanten with a whole wall to draw: the encoder's costliest kind
    // of row, which tegawari would make slower than a test can wait for.
    Case {
        hand: "1239m 469p 179s 1255z",
        pons: &[],
        dora_indicator: t!(3p),
        jikaze: tu8!(S),
        can_discard: true,
        tsumos_left: 17,
        seen: "",
        akas_seen: [false; 3],
        every_option: false,
    },
];

/// The lookahead asked directly, for what the recorded games rarely or never
/// reach: hands that want every unseen tile, so that `not_tsumo_probs` falls
/// to zero inside the horizon; tegawari, shanten-down discards and ranking by
/// the chance of winning, which Mortal's own tools ask for and the encoder
/// does not; a hand after a call; and three shanten with the whole wall to
/// draw. One digest per hand and set of options, of every candidate in full.
/// `ENCODER_GOLDEN_BLESS=1` prints them instead of checking them.
#[test]
fn the_lookahead_answers_as_it_always_has() {
    #[rustfmt::skip]
    const DIGESTS: [u64; 22] = [
        0x64f41f3683510043, 0x297e9038aa314781, 0xc3007b2dea1a0ca1, 0xffc6693fe7faca23,
        0xea2de147ad66c254, 0x77af458c1f251615, 0xb693bbe6bcafd75a, 0xd5c81263ece60fb1,
        0xe1f7a5b95419a906, 0xe13c55f452971269, 0x6944d782f6a69aa8, 0x774fb6d4713e7e29,
        0x14acea6d9bc17c0b, 0xbfd272f69224c0b0, 0x9351d28e05c159a9, 0x550ac0170050b4a4,
        0xeb474c9fbe05d8d0, 0x37c177d591d575ae, 0x309b059830b25a6b, 0xbf592b4f0b8ddb5c,
        0x20e9b182cbcec1a8, 0x1b137a22cfe29450,
    ];
    let mut got = vec![];
    for case in &CASES {
        let tehai = hand(case.hand).unwrap();
        let mut tiles_seen = tehai;
        tiles_seen[case.dora_indicator.deaka().as_usize()] += 1;
        for &pon in case.pons {
            tiles_seen[pon as usize] += 3;
        }
        if !case.seen.is_empty() {
            for (seen, more) in tiles_seen.iter_mut().zip(hand(case.seen).unwrap()) {
                *seen += more;
            }
        }
        let len_div3 = tehai.iter().sum::<u8>() / 3;
        let cur_shanten = shanten::calc_all(&tehai, len_div3);
        assert!(
            (0..=3).contains(&cur_shanten),
            "{}: shanten {cur_shanten}",
            case.hand
        );
        let is_menzen = case.pons.is_empty();
        // As the encoder asks (`PlayerState::single_player_tables`), then as
        // Mortal's `nanikiru` test does, then ranked by the chance to win.
        let options: &[(bool, bool, bool)] = if case.every_option {
            &[
                (false, false, false),
                (true, true, false),
                (true, true, true),
            ]
        } else {
            &[(false, false, false)]
        };
        for &(tegawari, shanten_down, by_win_prob) in options {
            let calc = SPCalculator {
                tehai_len_div3: len_div3,
                chis: &[],
                pons: case.pons,
                minkans: &[],
                ankans: &[],
                bakaze: tu8!(E),
                jikaze: case.jikaze,
                prefer_riichi: is_menzen,
                is_menzen,
                num_doras_in_fuuro: 0,
                dora_indicators: &[case.dora_indicator],
                calc_double_riichi: tegawari,
                calc_haitei: true,
                sort_result: true,
                maximize_win_prob: by_win_prob,
                calc_tegawari: tegawari,
                calc_shanten_down: shanten_down,
            };
            let state = InitState {
                tehai,
                akas_in_hand: [false; 3],
                tiles_seen,
                akas_seen: case.akas_seen,
            };
            let candidates = calc
                .calc(state, case.can_discard, case.tsumos_left, cur_shanten)
                .unwrap();
            assert!(!candidates.is_empty(), "{}", case.hand);
            let mut digest = Fnv::new();
            digest.count(candidates.len());
            for c in &candidates {
                candidate(&mut digest, c);
            }
            got.push(digest.0);
        }
    }
    if env::var_os("ENCODER_GOLDEN_BLESS").is_some() {
        let listed: Vec<_> = got
            .iter()
            .map(|digest| format!("0x{digest:016x}"))
            .collect();
        println!(
            "const DIGESTS: [u64; {}] = [{}];",
            got.len(),
            listed.join(", ")
        );
        return;
    }
    assert_eq!(got, DIGESTS, "the lookahead answers differently");
}
