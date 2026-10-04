//! The arena's legality work, spread across the cores, against the loops
//! that did it one game at a time, which are kept here as the reference:
//! the same masks, the same refusal naming the same game, and a refused
//! batch that moves nothing.

use super::{first_illegal, legal_masks, step_all, Seat};
use riichi_core::encoding::{self, ACTIONS};
use riichi_core::rng::Rng;

/// The masks as the arena wrote them before, one game after another.
fn serial_masks(seats: &[Seat]) -> Vec<bool> {
    let mut mask = vec![false; seats.len() * ACTIONS];
    for (index, seat) in seats.iter().enumerate() {
        let slice = &mut mask[index * ACTIONS..(index + 1) * ACTIONS];
        slice.fill(false);
        if let Some(wind) = seat.pending() {
            encoding::legal_mask(&seat.hand, wind, slice);
        }
    }
    mask
}

/// The strict check as the arena made it before, stopping at the first
/// illegal game, with the message it raised.
fn serial_first_illegal(seats: &[Seat], actions: &[usize]) -> Option<String> {
    for (game, (seat, index)) in seats.iter().zip(actions).enumerate() {
        let Some(wind) = seat.pending() else {
            continue;
        };
        let valid = if seat.asking.is_empty() {
            encoding::decode_action(&seat.hand, *index).is_some()
        } else {
            encoding::decode_call(&seat.hand, wind, *index).is_some()
        };
        if !valid {
            return Some(format!(
                "illegal action {index} in game {game} for {wind:?}"
            ));
        }
    }
    None
}

/// What a game shows of having moved: who owes the decision, how far its
/// hand's log and its unread events go, and how many hands it has played.
fn positions(seats: &[Seat]) -> Vec<(Option<riichi_core::Wind>, usize, usize, u32, bool)> {
    seats
        .iter()
        .map(|seat| {
            (
                seat.pending(),
                seat.hand.log.len(),
                seat.events.len(),
                seat.hands_done,
                seat.finished,
            )
        })
        .collect()
}

/// Tables with the heuristic player in different places, so that claim
/// windows with several askers, bots finishing hands and finished games
/// all turn up among them.
fn tables(count: usize, seed: u64) -> Vec<Seat> {
    let places: [&[usize]; 5] = [&[], &[1, 2, 3], &[0], &[1, 3], &[2]];
    (0..count)
        .map(|game| Seat::new(seed + game as u64, places[game % places.len()]))
        .collect()
}

/// A legal move per game from its mask, chosen at random; zero where the
/// game owes nothing.
fn legal_choices(mask: &[bool], rng: &mut Rng) -> Vec<usize> {
    mask.chunks(ACTIONS)
        .map(|flags| {
            let legal: Vec<usize> = (0..ACTIONS).filter(|&index| flags[index]).collect();
            if legal.is_empty() {
                0
            } else {
                legal[rng.below(legal.len())]
            }
        })
        .collect()
}

#[test]
fn masks_and_refusals_match_the_serial_loops_over_whole_games() {
    let mut rng = Rng::from_seed(20261004);
    let mut seats = tables(12, 700);
    let mut mask = vec![true; seats.len() * ACTIONS];
    let mut refused = 0;
    let mut claims = 0;
    let mut steps = 0;
    while seats.iter().any(|seat| !seat.finished) {
        steps += 1;
        assert!(steps < 8000, "the games should finish");
        claims += seats
            .iter()
            .filter(|seat| seat.pending().is_some() && !seat.asking.is_empty())
            .count();
        legal_masks(&seats, &mut mask);
        assert_eq!(mask, serial_masks(&seats), "masks differ at step {steps}");
        let legal = legal_choices(&mask, &mut rng);
        assert_eq!(first_illegal(&seats, &legal), None);

        // A few games given any index at all, most of them illegal, and
        // then every game: the first illegal game is the one reported, and
        // the batch is refused before any game moves. Every third step
        // only, to keep the unoptimised test build quick.
        if steps % 3 == 0 {
            let mut some = legal.clone();
            for _ in 0..3 {
                some[rng.below(seats.len())] = rng.below(ACTIONS + 2);
            }
            let all: Vec<usize> = (0..seats.len()).map(|_| rng.below(ACTIONS)).collect();
            for actions in [&some, &all] {
                let expected = serial_first_illegal(&seats, actions);
                assert_eq!(first_illegal(&seats, actions), expected);
                if let Some(message) = expected {
                    let before = positions(&seats);
                    assert_eq!(step_all(&mut seats, actions, true), Err(message));
                    assert_eq!(positions(&seats), before);
                    refused += 1;
                }
            }
        }

        assert_eq!(step_all(&mut seats, &legal, true), Ok(()));
    }
    assert!(refused > steps / 6, "the illegal batches were exercised");
    // Claims are checked against the calls offered, not the moves, so the
    // positions compared have to include some.
    assert!(claims > steps / 10, "claim windows were compared too");
    legal_masks(&seats, &mut mask);
    assert!(mask.iter().all(|flag| !flag), "finished games owe nothing");
}

#[test]
fn lenient_steps_play_illegal_indices_as_before() {
    // Without `strict`, an illegal index still moves its game, as the
    // first legal move or a pass, and nothing is refused.
    let mut rng = Rng::from_seed(81);
    let mut seats = tables(8, 900);
    let mut steps = 0;
    while seats.iter().any(|seat| !seat.finished) {
        steps += 1;
        assert!(steps < 8000, "the games should finish");
        let actions: Vec<usize> = (0..seats.len()).map(|_| rng.below(ACTIONS)).collect();
        assert_eq!(step_all(&mut seats, &actions, false), Ok(()));
    }
}
