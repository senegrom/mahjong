//! A world imagined from one seat has to add up, at every point of a game.
//!
//! Two bugs hid here and neither showed at the deal, which is why this walks
//! whole games rather than checking an opening hand. A tile claimed for a
//! set stays in the pond it came from, so counting every discard and every
//! called set counted it twice; and the replacement tiles taken after a quad
//! come off the dead wall without advancing the live draw, so their places
//! looked hidden when their tiles were already in somebody's hand. Both made
//! the tiles and the places disagree by exactly one, which is the sort of
//! thing an assertion catches and a reading of the code does not.

use riichi_core::bot::Bot;
use riichi_core::game::{Call, Phase};
use riichi_core::rng::Rng;
use riichi_core::search::{imagine, seen_by, Belief};
use riichi_core::table::Table;
use riichi_core::tile::{Tile, COPIES};
use riichi_core::Wind;

/// Plays games out and, at every decision, checks from every seat that the
/// tiles nobody has seen are exactly enough to fill the places nobody can
/// see, and that a world imagined from them is one the rules would allow.
#[test]
fn every_seat_can_imagine_a_world_at_every_point_of_a_game() {
    let mut checked = 0;
    let mut with_melds = 0;
    let mut with_quads = 0;

    for seed in 0..12 {
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut bots: Vec<Bot> = (0..4).map(|index| Bot::new(seed * 4 + index)).collect();
        let mut hand = table.deal(&mut rng);
        let mut guard = 0;

        while !matches!(hand.phase, Phase::Over) {
            guard += 1;
            assert!(guard < 600, "a hand ends well before this");

            let melds: usize = Wind::ALL
                .into_iter()
                .map(|seat| hand.players[seat.index()].melds.len())
                .sum();
            if melds > 0 {
                with_melds += 1;
            }
            if hand.wall.dora_indicators().len() > 1 {
                with_quads += 1;
            }

            for seat in Wind::ALL {
                let seen = seen_by(&hand, seat);
                for tile in Tile::all() {
                    assert!(
                        seen.count(tile) <= COPIES,
                        "seed {seed}: {seat:?} can see {} of {tile}",
                        seen.count(tile)
                    );
                }
                let unseen: usize = Tile::all()
                    .map(|tile| COPIES.saturating_sub(seen.count(tile)) as usize)
                    .sum();
                let others: usize = Wind::ALL
                    .into_iter()
                    .filter(|other| *other != seat)
                    .map(|other| hand.players[other.index()].hand.len())
                    .sum();
                let places = hand.wall.hidden_positions().len();
                assert_eq!(
                    unseen.checked_sub(others),
                    Some(places),
                    "seed {seed}: {seat:?} sees {unseen} unseen tiles and {others} in other \
                     hands, leaving {} for {places} hidden places",
                    unseen as i64 - others as i64
                );

                // And the world that comes of it is one the engine accepts.
                let mut rng = Rng::from_seed(seed * 31 + guard as u64);
                let world = imagine(&hand, seat, &Belief::even(), &mut rng);
                assert_eq!(
                    world.players[seat.index()].hand.counts(),
                    hand.players[seat.index()].hand.counts(),
                    "the searching seat's own hand is never imagined"
                );
                for other in Wind::ALL {
                    assert!(
                        world.players[other.index()].hand.is_legal(),
                        "seed {seed}: {other:?} was dealt an impossible hand"
                    );
                }
                checked += 1;
            }

            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    let action = bots[hand.turn.index()].act(&hand);
                    hand.act(action).expect("the bot chose a legal action");
                }
                Phase::CallWindow => {
                    let answers: Vec<(Wind, Call)> = hand
                        .legal_calls()
                        .iter()
                        .map(|(seat, calls)| (*seat, bots[seat.index()].call(&hand, *seat, calls)))
                        .collect();
                    hand.resolve_calls(&answers)
                        .expect("the calls were offered");
                }
                Phase::Over => break,
            }
        }
    }

    assert!(checked > 2000, "only {checked} positions were checked");
    assert!(
        with_melds > 50,
        "hardly any position had a called set: {with_melds}"
    );
    assert!(
        with_quads > 0,
        "no position followed a quad, which is where one bug was"
    );
}
