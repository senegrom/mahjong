//! Imagining the hands a seat cannot see.
//!
//! Take everything a player can see, deal the rest at random into the
//! other three hands and the wall in a way that is consistent with it, and
//! the result is one way the hidden tiles might actually lie. Self-play
//! deals one such world at each decision from what the network believes the
//! opponents hold, and a reader of hidden hands learns to tell those
//! imagined hands from the real ones (`riichi-py`'s `imagined_hands_bytes`).
//!
//! Which worlds get imagined is the whole question. Dealing the unseen
//! tiles out evenly assumes every arrangement is as likely as any other,
//! and that is wrong for a reason no amount of counting will fix: the
//! opponents chose what to throw. What is still in their hand is what they
//! wanted to keep. A player who has spent the hand throwing circles is less
//! likely to be holding circles than the count of unseen tiles says, and a
//! player who has thrown nothing but honours is telling you something else
//! again. That is a selection, and reading it is a judgement about this
//! table rather than an arithmetic fact, which is why [`Belief`] is
//! something a network supplies rather than something written down here.
//! Even weights are the fallback when nothing better is on hand, not the
//! model. A hand that has declared riichi is dealt a waiting shape its
//! public actions allow, never one read from the real hand.
//!
//! These worlds were the first half of a search, which made each candidate
//! move in many of them and valued what followed, by playing on with the
//! heuristic player or by asking the network. No version of it played
//! better than the policy alone, and it was removed in September 2026.

mod sampling;

use crate::encoding::{OPPONENTS, POSITIONS};
use crate::game::{Hand, Phase};
use crate::hand::TileSet;
use crate::mjai;
use crate::rng::Rng;
use crate::tile::{Tile, KINDS};
use crate::Wind;

/// How likely each kind of tile is to be in each opponent's hand.
///
/// Three rows of thirty-four, in the same relative seat order the
/// observation uses: row 0 is the player to the imagining seat's right in
/// turn order. A row is a weight, not a probability: it is used to draw tiles
/// from what nobody has seen, so only the ratios between its entries
/// matter, and a zero means "not this one".
///
/// A network trained on [`crate::encoding::opponent_hands`] produces these.
/// Without one, [`Belief::even`] gives every kind the same weight, which is
/// the assumption that opponents discard at random.
#[derive(Clone, PartialEq, Debug)]
pub struct Belief {
    /// Row-major, three rows of [`POSITIONS`] weights.
    pub weights: Vec<f32>,
    /// True for predicted tile-kind mass, false for a per-physical-copy prior.
    pub kind_mass: bool,
}

impl Belief {
    /// Every kind as likely as any other, for want of anything better.
    pub fn even() -> Belief {
        Belief {
            weights: vec![1.0; OPPONENTS * POSITIONS],
            kind_mass: false,
        }
    }

    /// A belief from a network's answer, which must be three rows of
    /// thirty-four. Negative weights are read as zero.
    pub fn from(weights: &[f32]) -> Belief {
        assert_eq!(
            weights.len(),
            OPPONENTS * POSITIONS,
            "a belief is three rows of thirty-four"
        );
        Belief {
            weights: weights
                .iter()
                .map(|weight| {
                    if weight.is_finite() {
                        weight.max(0.0)
                    } else {
                        0.0
                    }
                })
                .collect(),
            kind_mass: true,
        }
    }

    /// The weight this belief puts on `tile` for the opponent `offset`
    /// seats along in turn order, where 1 is the player to the right.
    fn weight(&self, offset: usize, tile: Tile) -> f32 {
        self.weights[(offset - 1) * POSITIONS + tile.idx()]
    }
}

/// One way the hidden tiles might actually lie.
///
/// The player's own hand, everybody's called sets, the discards, the scores
/// and the state of the hand are all kept exactly. Only what `seat` cannot
/// see ([`Hand::unseen_by`]) is dealt again: the other three hands, the rest
/// of the wall, and the dead wall under the indicators that have been turned.
pub fn imagine(hand: &Hand, seat: Wind, belief: &Belief, rng: &mut Rng) -> Hand {
    let mut pool: Vec<Tile> = hand.unseen_by(seat).tiles().collect();
    rng.shuffle(&mut pool);

    let reserved = sampling::reserve(hand, seat, belief, rng, &mut pool);
    let mut world = hand.clone();
    for offset in 1..=OPPONENTS {
        let other = seat.plus(offset);
        let wanted = hand.players[other.index()].hand.len();
        let mut dealt = reserved[other.index()];
        assert!(
            dealt.len() <= wanted,
            "riichi concealed hand has the wrong size"
        );
        let mut drawn = None;
        for _ in dealt.len()..wanted {
            let taken = draw_weighted(&mut pool, belief, offset, rng);
            dealt.add(taken);
            drawn = Some(taken);
        }
        if other == world.turn && matches!(world.phase, Phase::Act) {
            world.drawn = drawn;
        }
        world.players[other.index()].hand = dealt;
        // What they are waiting for and whether they are furiten follow
        // from the tiles, so both are worked out again for the new hand.
        if hand.players[other.index()].has_riichi() {
            let player = &hand.players[other.index()];
            let body = reserved[other.index()];
            let mut visible = body;
            for meld in &player.melds {
                for tile in meld.tiles() {
                    visible.add(tile);
                }
            }
            let waits = crate::shanten::waits(&body, player.melds.len(), &visible);
            let latest = hand
                .players
                .iter()
                .flat_map(|p| &p.discards)
                .map(|d| d.order)
                .max();
            let passed = hand.players.iter().flat_map(|p| &p.discards).any(|d| {
                player.riichi_order.is_some_and(|order| d.order > order)
                    && !(hand.phase == Phase::CallWindow
                        && hand.robbable_quad.is_none()
                        && Some(d.order) == latest)
                    && waits.count(d.tile) > 0
            });
            let from_pond = player.discards.iter().any(|d| waits.count(d.tile) > 0);
            let passed_kan = sampled_riichi_kan_pass(&world, other, &body, &waits);
            world.players[other.index()].furiten = passed || from_pond || passed_kan;
            world.players[other.index()].temporary_furiten = false;
        } else {
            world.players[other.index()].refresh_furiten();
            world.players[other.index()].temporary_furiten =
                sampled_temporary_furiten(&world, other);
        }
    }

    // Whatever nobody was dealt is the rest of the wall. It is already
    // shuffled, and the belief has no say in the order it comes out.
    world.wall = hand.wall.with_hidden(&pool);
    world
}

/// A passed kan-robbery opportunity after an accepted declaration also persists.
/// This supplements discard history using public events, never real private flags.
fn sampled_riichi_kan_pass(world: &Hand, seat: Wind, body: &TileSet, waits: &TileSet) -> bool {
    let Some(declared) = world
        .log
        .iter()
        .rposition(|event| matches!(event, mjai::Event::ReachAccepted { actor } if *actor == seat))
    else {
        return false;
    };
    let unresolved = (world.phase == Phase::CallWindow)
        .then(|| {
            world.log.iter().rposition(|event| {
                matches!(
                    event,
                    mjai::Event::Dahai { .. }
                        | mjai::Event::Kakan { .. }
                        | mjai::Event::Ankan { .. }
                )
            })
        })
        .flatten();
    world
        .log
        .iter()
        .enumerate()
        .skip(declared + 1)
        .any(|(at, event)| {
            if Some(at) == unresolved {
                return false;
            }
            match event {
                mjai::Event::Kakan { actor, tile, .. } => *actor != seat && waits.count(*tile) > 0,
                mjai::Event::Ankan { actor, consumed } if *actor != seat => {
                    consumed.first().is_some_and(|tile| {
                        let mut complete = *body;
                        complete.add(*tile);
                        crate::shanten::thirteen_orphans(
                            &complete,
                            world.players[seat.index()].melds.len(),
                        ) == crate::shanten::COMPLETE
                    })
                }
                _ => false,
            }
        })
}

/// Reconstruct same-cycle furiten from public events and the sampled hand.
/// The original hidden hand's flags are not information the searcher owns.
fn sampled_temporary_furiten(world: &Hand, seat: Wind) -> bool {
    let waits = world.players[seat.index()].waits();
    let unresolved = (world.phase == Phase::CallWindow)
        .then(|| {
            world.log.iter().rposition(|event| {
                matches!(
                    event,
                    mjai::Event::Dahai { .. }
                        | mjai::Event::Kakan { .. }
                        | mjai::Event::Ankan { .. }
                )
            })
        })
        .flatten();
    for (at, event) in world.log.iter().enumerate().rev() {
        match event {
            mjai::Event::Tsumo { actor, .. }
            | mjai::Event::Chi { actor, .. }
            | mjai::Event::Pon { actor, .. }
            | mjai::Event::Daiminkan { actor, .. }
                if *actor == seat =>
            {
                return false
            }
            mjai::Event::Dahai { actor, tile, .. } | mjai::Event::Kakan { actor, tile, .. }
                if *actor != seat && Some(at) != unresolved && waits.count(*tile) > 0 =>
            {
                return true
            }
            mjai::Event::Ankan { actor, consumed } if *actor != seat && Some(at) != unresolved => {
                if let Some(tile) = consumed.first() {
                    let mut completed = world.players[seat.index()].hand;
                    completed.add(*tile);
                    if crate::shanten::thirteen_orphans(
                        &completed,
                        world.players[seat.index()].melds.len(),
                    ) == crate::shanten::COMPLETE
                    {
                        return true;
                    }
                }
            }
            _ => (),
        }
    }
    false
}

/// Takes one tile from `pool`, chosen in proportion to what `belief` says
/// this opponent is holding.
///
/// Falls back to an even draw when the belief puts no weight on anything
/// left in the pool, which can happen when a confident network is wrong
/// about a hand and there is nothing else to deal.
fn draw_weighted(pool: &mut Vec<Tile>, belief: &Belief, offset: usize, rng: &mut Rng) -> Tile {
    assert!(!pool.is_empty(), "there is always a tile left to deal");
    // A network predicts mass per KIND (count / hand size), not per copy.
    // Divide it among the available physical copies. Belief::even deliberately
    // retains a uniform physical-copy prior. Neither is an exact hand posterior.
    let mut available = [0usize; KINDS];
    for tile in pool.iter() {
        available[tile.idx()] += 1;
    }
    let weight = |tile: Tile| {
        let mass = belief.weight(offset, tile);
        if belief.kind_mass {
            mass / available[tile.idx()] as f32
        } else {
            mass
        }
    };
    let total: f32 = pool.iter().map(|tile| weight(*tile)).sum();
    // A belief that puts no weight on anything left, or that has gone to
    // pieces and produced a not-a-number, falls back to an even draw.
    if !total.is_finite() || total <= 0.0 {
        return pool.swap_remove(rng.below(pool.len()));
    }
    // A weighted draw, walking the pool until the running total passes a
    // point chosen along it.
    let mut point = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * total;
    for index in 0..pool.len() {
        point -= weight(pool[index]);
        if point <= 0.0 {
            return pool.swap_remove(index);
        }
    }
    pool.swap_remove(pool.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Table;
    use crate::tile::COPIES;

    /// An imagined world has to be a world: every tile accounted for, none
    /// of them five times over, and everything the player can see left
    /// exactly as it was.
    #[test]
    fn an_imagined_world_is_consistent_with_what_was_seen() {
        let table = Table::new();
        let mut rng = Rng::from_seed(99);
        let hand = table.deal(&mut rng);
        let seat = Wind::East;

        for round in 0..40 {
            let mut rng = Rng::from_seed(1000 + round);
            let world = imagine(&hand, seat, &Belief::even(), &mut rng);

            // What East holds is untouched.
            assert_eq!(
                world.players[seat.index()].hand.counts(),
                hand.players[seat.index()].hand.counts(),
                "the imagining seat's own hand is not imagined"
            );
            // Everybody still holds as many tiles as they did.
            for other in Wind::ALL {
                assert_eq!(
                    world.players[other.index()].hand.len(),
                    hand.players[other.index()].hand.len(),
                    "{other:?} was dealt a different number of tiles"
                );
            }
            // The indicators that were face up are still those tiles.
            assert_eq!(
                world.wall.dora_indicators(),
                hand.wall.dora_indicators(),
                "an indicator that was turned cannot change"
            );

            // Four of every kind, no more, across hands, sets, discards and
            // the whole wall.
            let mut counted = TileSet::new();
            for other in Wind::ALL {
                let player = &world.players[other.index()];
                for tile in player.hand.tiles() {
                    counted.add(tile);
                }
                for meld in &player.melds {
                    for tile in meld.tiles() {
                        counted.add(tile);
                    }
                }
                for discard in &player.discards {
                    counted.add(discard.tile);
                }
            }
            for tile in world.wall.tiles() {
                counted.add(*tile);
            }
            // The wall array still holds the tiles already drawn, which are
            // now also in a hand, so each of those is counted twice.
            for tile in Tile::all() {
                assert!(
                    counted.count(tile) >= COPIES,
                    "{tile} went missing from the imagined world"
                );
            }
        }
    }

    /// A belief has to change what gets imagined, or it is decoration. A
    /// network that has read an opponent's discards and concluded they are
    /// holding circles must produce worlds where they are holding circles.
    #[test]
    fn what_the_belief_says_is_what_gets_dealt() {
        let table = Table::new();
        let mut rng = Rng::from_seed(515);
        let hand = table.deal(&mut rng);
        let seat = Wind::East;

        // Everything on the player to the right, nothing on the other two.
        let mut weights = vec![0.0f32; OPPONENTS * POSITIONS];
        for tile in Tile::all() {
            if matches!(tile.suit(), crate::tile::Suit::Circles) {
                weights[tile.idx()] = 1.0;
            }
        }
        // The other two rows are even, so they are unaffected.
        for offset in 1..OPPONENTS {
            for tile in Tile::all() {
                weights[offset * POSITIONS + tile.idx()] = 1.0;
            }
        }
        let belief = Belief::from(&weights);

        let mut circles = 0;
        let mut held = 0;
        let mut even_circles = 0;
        let mut even_held = 0;
        for round in 0..30 {
            let mut rng = Rng::from_seed(round);
            let world = imagine(&hand, seat, &belief, &mut rng);
            let right = &world.players[seat.plus(1).index()].hand;
            circles += right
                .tiles()
                .filter(|tile| matches!(tile.suit(), crate::tile::Suit::Circles))
                .count();
            held += right.len();

            let mut rng = Rng::from_seed(round);
            let plain = imagine(&hand, seat, &Belief::even(), &mut rng);
            let right = &plain.players[seat.plus(1).index()].hand;
            even_circles += right
                .tiles()
                .filter(|tile| matches!(tile.suit(), crate::tile::Suit::Circles))
                .count();
            even_held += right.len();
        }

        assert_eq!(
            circles, held,
            "told the player holds only circles, every tile dealt them is a circle"
        );
        assert!(
            even_circles * 3 < even_held,
            "and dealt evenly they hold about a quarter circles, not {even_circles} of {even_held}"
        );
    }

    #[test]
    fn sampled_riichi_furiten_includes_passed_but_not_unresolved_kan_robbery() {
        let mut hand = Table::new().deal(&mut Rng::from_seed(9));
        let seat = Wind::South;
        let tile = Tile::new(0);
        let body = hand.players[seat.index()].hand;
        let mut waits = TileSet::new();
        waits.add(tile);
        hand.log.push(mjai::Event::ReachAccepted { actor: seat });
        hand.log.push(mjai::Event::Kakan {
            actor: Wind::East,
            tile,
            consumed: vec![tile; 3],
        });
        hand.phase = Phase::CallWindow;
        assert!(!sampled_riichi_kan_pass(&hand, seat, &body, &waits));
        hand.phase = Phase::Draw;
        assert!(sampled_riichi_kan_pass(&hand, seat, &body, &waits));
        hand.log.push(mjai::Event::ReachAccepted { actor: seat });
        assert!(!sampled_riichi_kan_pass(&hand, seat, &body, &waits));
    }

    #[test]
    fn predicted_kind_mass_is_not_multiplied_by_remaining_copies() {
        let a = Tile::new(0);
        let b = Tile::new(1);
        let mut weights = vec![0.0; OPPONENTS * POSITIONS];
        weights[0] = 0.5;
        weights[1] = 0.5;
        let predicted = Belief::from(&weights);
        let mut rng = Rng::from_seed(941);
        let mut kind_a = 0;
        let mut even_a = 0;
        for _ in 0..10_000 {
            let mut pool = vec![a, a, a, a, b];
            kind_a += usize::from(draw_weighted(&mut pool, &predicted, 1, &mut rng) == a);
            let mut pool = vec![a, a, a, a, b];
            even_a += usize::from(draw_weighted(&mut pool, &Belief::even(), 1, &mut rng) == a);
        }
        assert!((4700..5300).contains(&kind_a), "kind mass: {kind_a}");
        assert!((7700..8300).contains(&even_a), "physical prior: {even_a}");
    }

    #[test]
    fn non_riichi_private_furiten_cannot_leak_into_sampled_worlds() {
        let mut hand = Table::new().deal(&mut Rng::from_seed(991));
        let observer = hand.turn;
        hand.players[observer.index()].temporary_furiten = true;
        let mut changed = hand.clone();
        for seat in Wind::ALL {
            if seat != observer {
                changed.players[seat.index()].temporary_furiten = true;
                changed.players[seat.index()].furiten = true;
            }
        }
        let a = imagine(&hand, observer, &Belief::even(), &mut Rng::from_seed(7));
        let b = imagine(&changed, observer, &Belief::even(), &mut Rng::from_seed(7));
        for seat in Wind::ALL {
            assert_eq!(a.players[seat.index()], b.players[seat.index()]);
        }
        assert!(
            a.players[observer.index()].temporary_furiten,
            "own known state stays"
        );
    }
}
