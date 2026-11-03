//! The arena's dama codes for the last hand, game by game: the rules
//! engine's own answer for a game whose decision is owed at a turn, zeros
//! for one waiting on a claim or finished, in the order the games are named,
//! and a game the arena does not have refused. The decline codes, row by
//! row, each a game and a player by place: the engine's own answer for the
//! seat that player holds where they owe a decision, at their turn or now or
//! later in a claim window; zeros for any other row; and a game the arena
//! does not have, or a player not one of the four, refused.

use super::{all_last_codes, decline_codes, Seat};
use riichi_core::all_last::{Dama, Decline};
use riichi_core::game::{Discard, Phase};
use riichi_core::tile::{Tile, KINDS};
use riichi_core::wall::Wall;
use riichi_core::Wind;

fn tile(text: &str) -> Tile {
    text.parse().unwrap()
}

/// A game moved on to South 4, North to move on 23456m 456p 678s 33s with
/// the 9p just drawn, 4,000 ahead of South and West: throwing the 9p
/// without declaring keeps it first through every win.
fn last_hand(seed: u64) -> Seat {
    let mut seat = Seat::new(seed, &[]);
    let hand = &mut seat.hand;
    hand.round = Wind::South;
    hand.kyoku = 4;
    hand.wall = Wall::for_analysis(30, &[tile("2z")], 0).unwrap();
    for (player, score) in hand
        .players
        .iter_mut()
        .zip([20_000, 32_000, 32_000, 36_000])
    {
        player.score = score;
    }
    hand.players[Wind::North.index()].hand = "23456m456p678s33s9p".parse().unwrap();
    hand.drawn = Some(tile("9p"));
    hand.turn = Wind::North;
    hand.phase = Phase::Act;
    hand.first_turns_unbroken = false;
    seat
}

/// The codes the engine gives the hand, as the arena hands them over.
fn engine_codes(seat: &Seat) -> Vec<u8> {
    seat.hand
        .all_last_dama()
        .iter()
        .map(|code| *code as u8)
        .collect()
}

#[test]
fn a_fresh_arena_owes_no_last_hand() {
    let seats: Vec<Seat> = (0..3)
        .map(|game| Seat::new(9_100_000 + game, &[]))
        .collect();
    let codes = all_last_codes(&seats, &[0, 1, 2, 1]).expect("every game is the arena's");
    assert_eq!(codes, vec![0; 4 * KINDS]);
    assert_eq!(all_last_codes(&seats, &[]), Ok(Vec::new()));
}

#[test]
fn a_last_hand_is_answered_as_the_engine_answers_it() {
    let seats = vec![Seat::new(9_100_000, &[]), last_hand(9_100_001)];
    let want = engine_codes(&seats[1]);
    assert_eq!(want[tile("9p").idx()], Dama::KeepsFirst as u8);
    assert_eq!(want[tile("2m").idx()], Dama::NotWaiting as u8);
    // Named in any order, and more than once.
    let codes = all_last_codes(&seats, &[1, 0, 1]).expect("every game is the arena's");
    assert_eq!(codes.len(), 3 * KINDS);
    assert_eq!(codes[..KINDS], want[..]);
    assert_eq!(codes[KINDS..2 * KINDS], [0; KINDS]);
    assert_eq!(codes[2 * KINDS..], want[..]);
}

#[test]
fn a_game_waiting_on_a_claim_or_finished_owes_nothing() {
    let mut waiting = last_hand(9_100_002);
    waiting.asking.push_back(Wind::South);
    let mut finished = last_hand(9_100_003);
    finished.finished = true;
    let seats = vec![waiting, finished];
    assert!(seats
        .iter()
        .all(|seat| engine_codes(seat).iter().any(|code| *code != 0)));
    assert_eq!(all_last_codes(&seats, &[0, 1]), Ok(vec![0; 2 * KINDS]));
}

#[test]
fn a_game_the_arena_lacks_is_refused() {
    let seats = vec![last_hand(9_100_004)];
    let refused = all_last_codes(&seats, &[0, 1]).expect_err("there is no game 1");
    assert!(refused.contains("no game 1"), "{refused}");
}

/// A game moved on to South 4, its table with it, the dealer as player 3:
/// the rows the decline rule is asked about name players by place, which the
/// table maps to seats. As `last_hand`, North to move, 4,000 ahead.
fn moved_on(seed: u64) -> Seat {
    let mut seat = last_hand(seed);
    seat.table.round = Wind::South;
    seat.table.dealer = 3;
    seat
}

/// The player sitting at `wind`.
fn at(seat: &Seat, wind: Wind) -> usize {
    seat.table.player_at(wind)
}

/// Hands that wait on nothing, for the seats whose hands nothing reads.
const IDLE: [&str; 2] = ["1z1z2z5z6z7z1p4p7p2s5s8s3m", "3z3z4z4z5z6z7z2p5p8p3s6s9s"];

/// A game moved on to South 4 in which West has just thrown a 1m that North,
/// on 23456m 456p 678s 33s, and South, on 23456m 789p 345s 55s, may both
/// ron, the arena asking South first: North on 5,000, whose win leaves it
/// alone in last, and South on 40,000.
fn two_rons(seed: u64) -> Seat {
    let mut seat = moved_on(seed);
    let hand = &mut seat.hand;
    for (player, score) in hand.players.iter_mut().zip([40_000, 40_000, 15_000, 5_000]) {
        player.score = score;
    }
    hand.players[Wind::East.index()].hand = IDLE[0].parse().unwrap();
    hand.players[Wind::South.index()].hand = "23456m789p345s55s".parse().unwrap();
    hand.players[Wind::West.index()].hand = IDLE[1].parse().unwrap();
    hand.players[Wind::North.index()].hand = "23456m456p678s33s".parse().unwrap();
    hand.players[Wind::West.index()].discards.push(Discard {
        tile: tile("1m"),
        order: 0,
        drawn: false,
        riichi: false,
        claimed: false,
    });
    hand.discards_made = 1;
    hand.turn = Wind::West;
    hand.phase = Phase::CallWindow;
    hand.drawn = None;
    hand.pending_discard = Some((Wind::West, tile("1m")));
    seat.asking.extend([Wind::South, Wind::North]);
    seat
}

#[test]
fn a_fresh_arena_owes_no_late_rule() {
    let seats: Vec<Seat> = (0..3)
        .map(|game| Seat::new(9_100_000 + game, &[]))
        .collect();
    let rows: Vec<(usize, usize)> = (0..3)
        .flat_map(|game| (0..4).map(move |player| (game, player)))
        .collect();
    assert_eq!(decline_codes(&seats, &rows), Ok(vec![0; rows.len()]));
    assert_eq!(decline_codes(&seats, &[]), Ok(Vec::new()));
}

/// A player who owes the game no decision at a turn is given zero.
#[test]
fn a_player_who_owes_no_decision_at_a_turn_is_given_zero() {
    let seats = vec![moved_on(9_100_001)];
    for wind in [Wind::East, Wind::South, Wind::West] {
        let rows = [(0, at(&seats[0], wind))];
        assert_eq!(decline_codes(&seats, &rows), Ok(vec![0]), "{wind:?}");
    }
}

#[test]
fn a_claim_window_answers_every_seat_it_will_ask() {
    let mut seat = two_rons(9_100_005);
    let (east, south, west, north) = (
        at(&seat, Wind::East),
        at(&seat, Wind::South),
        at(&seat, Wind::West),
        at(&seat, Wind::North),
    );
    assert_eq!(
        (
            seat.hand.all_last_decline(Wind::North),
            seat.hand.all_last_decline(Wind::South)
        ),
        (Decline::Last, Decline::NotLast)
    );
    let rows = [(0, north), (0, south), (0, east), (0, west)];
    let want = vec![Decline::Last as u8, Decline::NotLast as u8, 0, 0];
    assert_eq!(decline_codes(std::slice::from_ref(&seat), &rows), Ok(want));
    // South has answered: only North still owes the window an answer.
    seat.asking.pop_front();
    assert_eq!(
        decline_codes(std::slice::from_ref(&seat), &rows),
        Ok(vec![Decline::Last as u8, 0, 0, 0])
    );
}

#[test]
fn a_finished_game_owes_no_late_rule() {
    let mut finished = two_rons(9_100_007);
    finished.finished = true;
    let seats = vec![finished];
    let rows: Vec<(usize, usize)> = (0..4).map(|player| (0, player)).collect();
    assert_eq!(decline_codes(&seats, &rows), Ok(vec![0; rows.len()]));
}

#[test]
fn a_row_the_arena_lacks_is_refused() {
    let seats = vec![moved_on(9_100_009)];
    for (rows, why) in [
        (vec![(0, 2), (1, 0)], "no game 1"),
        (vec![(0, 2), (0, 4)], "no player 4 in game 0"),
    ] {
        let refused = decline_codes(&seats, &rows).expect_err("the row is not the arena's");
        assert!(refused.contains(why), "{refused}");
    }
}
