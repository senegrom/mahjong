//! Accepted declarations constrain every proposal, not just the mean belief.
use riichi_core::bot::Bot;
use riichi_core::game::{Action, Call, Hand, Phase};
use riichi_core::hand::TileSet;
use riichi_core::rng::Rng;
use riichi_core::search::{imagine, seen_by, Belief};
use riichi_core::table::Table;
use riichi_core::tile::Tile;
use riichi_core::Wind;

fn check(world: &Hand, observer: Wind) {
    let mut all = seen_by(world, observer);
    for seat in Wind::ALL {
        if seat == observer {
            continue;
        }
        let mut player = world.players[seat.index()].clone();
        for tile in player.hand.tiles() {
            all.add(tile);
        }
        if player.has_riichi() {
            if player.hand.len() % 3 == 2 {
                assert!(player
                    .hand
                    .remove(world.drawn.expect("a hidden drawn tile is resampled")));
            }
            assert!(
                player.is_tenpai(),
                "riichi proposal is not waiting: {:?}",
                player.hand
            );
        }
    }
    for index in world.wall.hidden_positions() {
        all.add(world.wall.tiles()[index]);
    }
    assert!(
        all.counts().iter().all(|count| *count == 4),
        "tiles must be conserved: {all:?}"
    );
}

#[test]
fn live_riichi_samples_obey_public_facts_without_reading_hidden_hands() {
    let mut checked = 0;
    for seed in 0..8 {
        let mut hand = Table::new().deal(&mut Rng::from_seed(seed));
        let mut bots: Vec<_> = (0..4).map(|i| Bot::new(seed * 4 + i)).collect();
        for step in 0..600 {
            if hand.phase == Phase::Over {
                break;
            }
            let observer = hand.turn;
            if hand.phase == Phase::Act
                && hand
                    .players
                    .iter()
                    .any(|p| p.seat != observer && p.has_riichi())
            {
                let mut secret_changed = hand.clone();
                for seat in Wind::ALL {
                    if seat != observer {
                        let p = &mut secret_changed.players[seat.index()];
                        p.hand = TileSet::from_tiles(vec![Tile::new(0); p.hand.len()]);
                        p.furiten = !p.furiten;
                        p.temporary_furiten = !p.temporary_furiten;
                    }
                }
                for n in 0..8 {
                    let proposal_seed = seed * 10000 + step * 8 + n;
                    let world = imagine(
                        &hand,
                        observer,
                        &Belief::even(),
                        &mut Rng::from_seed(proposal_seed),
                    );
                    let other = imagine(
                        &secret_changed,
                        observer,
                        &Belief::even(),
                        &mut Rng::from_seed(proposal_seed),
                    );
                    check(&world, observer);
                    assert_eq!(
                        world.players[observer.index()],
                        hand.players[observer.index()]
                    );
                    assert_eq!(world.wall, other.wall);
                    for seat in Wind::ALL {
                        assert_eq!(
                            world.players[seat.index()].hand,
                            other.players[seat.index()].hand
                        );
                        if world.players[seat.index()].has_riichi() {
                            assert_eq!(
                                world.players[seat.index()].furiten,
                                other.players[seat.index()].furiten
                            );
                        }
                    }
                    checked += 1;
                }
            }
            match hand.phase {
                Phase::Draw => {
                    hand.draw().unwrap();
                }
                Phase::Act => {
                    let action = hand
                        .legal_actions()
                        .into_iter()
                        .find(|a| matches!(a, Action::Riichi(_)))
                        .unwrap_or_else(|| bots[hand.turn.index()].act(&hand));
                    hand.act(action).unwrap();
                }
                Phase::CallWindow => {
                    let answers: Vec<(Wind, Call)> = hand
                        .legal_calls()
                        .iter()
                        .map(|(seat, calls)| (*seat, bots[seat.index()].call(&hand, *seat, calls)))
                        .collect();
                    hand.resolve_calls(&answers).unwrap();
                }
                Phase::Over => break,
            }
        }
    }
    assert!(
        checked >= 64,
        "must exercise accepted riichi, checked {checked}"
    );
}
