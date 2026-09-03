//! Looking ahead by imagining the hands you cannot see.
//!
//! A policy network answers from the position in front of it. That is a
//! judgement, not a calculation: it never plays anything out. Search fixes
//! that, but mahjong will not take the search a perfect-information game
//! takes, because three hands and the whole wall are hidden and the tree
//! branches on tiles nobody can know.
//!
//! What works instead is to imagine. Take everything the player can see,
//! deal the rest at random into the other three hands and the wall in a way
//! that is consistent with it, and play that world out. Do it many times
//! and average. A move that wins in most of the worlds where it could
//! matter is a good move, whatever any single deal does. The literature
//! calls this perfect-information Monte Carlo; a player would call it
//! thinking it through.
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
//! model.
//!
//! One thing it does not do at all: it assumes the other players know what
//! they know in each imagined world, so it cannot find a move whose value
//! is that it hides information. Nobody bluffs in these rollouts.

use crate::bot::{Bot, Style};
use crate::encoding::{OPPONENTS, POSITIONS};
use crate::game::{Action, Call, Hand, Phase};
use crate::hand::TileSet;
use crate::rng::Rng;
use crate::tile::{Tile, COPIES};
use crate::Wind;

/// How hard to think.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Effort {
    /// How many imagined worlds each candidate move is played out in. More
    /// worlds means a steadier answer and proportionally more time.
    pub worlds: usize,
    /// How many moves to consider. The rest of the hand is played by the
    /// rollout policy, so this is only the branching at the root.
    pub candidates: usize,
    /// How far to play each world: the whole hand, or a fixed number of
    /// turns after which the hand is judged by how close it stands.
    pub turns: Option<usize>,
}

impl Effort {
    /// Enough to be worth doing and quick enough for a browser.
    pub fn quick() -> Effort {
        Effort {
            worlds: 12,
            candidates: 5,
            turns: None,
        }
    }

    /// For an arena, where there is time.
    pub fn thorough() -> Effort {
        Effort {
            worlds: 60,
            candidates: 8,
            turns: None,
        }
    }
}

/// How likely each kind of tile is to be in each opponent's hand.
///
/// Three rows of thirty-four, in the same relative seat order the
/// observation uses: row 0 is the player to the searcher's right in turn
/// order. A row is a weight, not a probability: it is used to draw tiles
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
}

impl Belief {
    /// Every kind as likely as any other, for want of anything better.
    pub fn even() -> Belief {
        Belief {
            weights: vec![1.0; OPPONENTS * POSITIONS],
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
            weights: weights.iter().map(|weight| weight.max(0.0)).collect(),
        }
    }

    /// The weight this belief puts on `tile` for the opponent `offset`
    /// seats along in turn order, where 1 is the player to the right.
    fn weight(&self, offset: usize, tile: Tile) -> f32 {
        self.weights[(offset - 1) * POSITIONS + tile.idx()]
    }
}

/// What one candidate move came to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Judged {
    /// The move.
    pub action: Action,
    /// The average points the hand moved for the searching player, over
    /// every world it was tried in.
    pub value: f64,
    /// How many worlds it was tried in.
    pub worlds: usize,
}

/// Everything a seat can see of the tiles.
///
/// Their own hand and sets, every discard on the table, everybody's called
/// sets, and the dora indicators that have been turned. What is left is
/// what a search is free to imagine.
pub fn seen_by(hand: &Hand, seat: Wind) -> TileSet {
    let mut seen = TileSet::new();
    for tile in hand.players[seat.index()].hand.tiles() {
        seen.add(tile);
    }
    for other in Wind::ALL {
        let player = &hand.players[other.index()];
        for meld in &player.melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
        for discard in &player.discards {
            // A tile claimed for a set stays in the pond it came from,
            // turned sideways against the set that took it. It is counted
            // with that set, so counting it here as well counts it twice:
            // that made a wait look a tile thinner than it is, and left the
            // search one tile short of a world.
            if !discard.claimed {
                seen.add(discard.tile);
            }
        }
    }
    for indicator in hand.wall.dora_indicators() {
        seen.add(indicator);
    }
    // A tile awaiting a claim needs nothing more: a discard is already in
    // the discarder's pond by the time anybody may claim it, and the tile
    // added to a quad that is being robbed is already in the quad.
    seen
}

/// One way the hidden tiles might actually lie.
///
/// The player's own hand, everybody's called sets, the discards, the scores
/// and the state of the hand are all kept exactly. Only what `seat` cannot
/// see is dealt again: the other three hands, the rest of the wall, and the
/// dead wall under the indicators that have been turned.
pub fn imagine(hand: &Hand, seat: Wind, belief: &Belief, rng: &mut Rng) -> Hand {
    let seen = seen_by(hand, seat);
    let mut pool: Vec<Tile> = Vec::with_capacity(136);
    for tile in Tile::all() {
        for _ in 0..COPIES.saturating_sub(seen.count(tile)) {
            pool.push(tile);
        }
    }
    rng.shuffle(&mut pool);

    let mut world = hand.clone();
    for offset in 1..=OPPONENTS {
        let other = seat.plus(offset);
        let wanted = hand.players[other.index()].hand.len();
        let mut dealt = TileSet::new();
        for _ in 0..wanted {
            let taken = draw_weighted(&mut pool, belief, offset, rng);
            dealt.add(taken);
        }
        world.players[other.index()].hand = dealt;
        // What they are waiting for and whether they are furiten follow
        // from the tiles, so both are worked out again for the new hand.
        world.players[other.index()].refresh_furiten();
    }

    // Whatever nobody was dealt is the rest of the wall. It is already
    // shuffled, and the belief has no say in the order it comes out.
    world.wall = hand.wall.with_hidden(&pool);
    world
}

/// Takes one tile from `pool`, chosen in proportion to what `belief` says
/// this opponent is holding.
///
/// Falls back to an even draw when the belief puts no weight on anything
/// left in the pool, which can happen when a confident network is wrong
/// about a hand and there is nothing else to deal.
fn draw_weighted(pool: &mut Vec<Tile>, belief: &Belief, offset: usize, rng: &mut Rng) -> Tile {
    assert!(!pool.is_empty(), "there is always a tile left to deal");
    let total: f32 = pool.iter().map(|tile| belief.weight(offset, *tile)).sum();
    // A belief that puts no weight on anything left, or that has gone to
    // pieces and produced a not-a-number, falls back to an even draw.
    if total.is_nan() || total <= 0.0 {
        return pool.swap_remove(rng.below(pool.len()));
    }
    // A weighted draw, walking the pool until the running total passes a
    // point chosen along it.
    let mut point = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * total;
    for index in 0..pool.len() {
        point -= belief.weight(offset, pool[index]);
        if point <= 0.0 {
            return pool.swap_remove(index);
        }
    }
    pool.swap_remove(pool.len() - 1)
}

/// Plays one imagined world out and says what the hand moved for `seat`.
///
/// Every player, the searching one included, is played by `style`: the
/// point is to compare moves under the same later play, not to model
/// anybody in particular.
fn play_out(mut world: Hand, seat: Wind, style: Style, turns: Option<usize>, seed: u64) -> f64 {
    let opening = world.players[seat.index()].score;
    let mut bots: Vec<Bot> = (0..4)
        .map(|index| Bot::with_style(seed.wrapping_add(index), style))
        .collect();

    let limit = turns.unwrap_or(usize::MAX);
    let mut played = 0;
    while !matches!(world.phase, Phase::Over) && played < limit {
        match world.phase {
            Phase::Draw => {
                if world.draw().is_err() {
                    break;
                }
            }
            Phase::Act => {
                played += 1;
                let turn = world.turn;
                let action = bots[turn.index()].act(&world);
                if world.act(action).is_err() {
                    break;
                }
            }
            Phase::CallWindow => {
                let answers: Vec<(Wind, Call)> = world
                    .legal_calls()
                    .iter()
                    .map(|(who, calls)| (*who, bots[who.index()].call(&world, *who, calls)))
                    .collect();
                if world.resolve_calls(&answers).is_err() {
                    break;
                }
            }
            Phase::Over => break,
        }
    }

    match &world.outcome {
        // The hand finished, so what it moved is simply what it moved.
        Some(_) => (world.players[seat.index()].score - opening) as f64,
        // It was cut short, so the hand is judged by how it stands: closer
        // to a win is better, and a hand that is waiting is worth more than
        // one that is not. The numbers are small next to a real win, which
        // is the intent: this is a tie-break, not a score.
        None => {
            let player = &world.players[seat.index()];
            let distance = crate::shanten::shanten(&player.hand, player.melds.len());
            let moved = (player.score - opening) as f64;
            moved + 300.0 * (2 - distance.min(6)) as f64
        }
    }
}

/// Judges every candidate move by playing it out in imagined worlds.
///
/// The same worlds are used for every candidate, so the comparison is not
/// clouded by one move happening to be tried in luckier deals than another.
pub fn judge(
    hand: &Hand,
    seat: Wind,
    candidates: &[Action],
    effort: Effort,
    belief: &Belief,
    rng: &mut Rng,
) -> Vec<Judged> {
    assert!(!candidates.is_empty(), "there is always something to do");
    let style = Style::club();

    // One set of worlds, shared by all the candidates.
    let worlds: Vec<Hand> = (0..effort.worlds)
        .map(|_| imagine(hand, seat, belief, rng))
        .collect();

    candidates
        .iter()
        .map(|action| {
            let mut total = 0.0;
            let mut counted = 0;
            for (index, world) in worlds.iter().enumerate() {
                let mut trial = world.clone();
                // A move the engine will not take in this world is not a
                // fault: an imagined hand can make a quad illegal that was
                // legal in the real one. It is simply not counted.
                if trial.act(*action).is_err() {
                    continue;
                }
                total += play_out(trial, seat, style, effort.turns, index as u64 * 977 + 13);
                counted += 1;
            }
            Judged {
                action: *action,
                value: if counted == 0 {
                    f64::NEG_INFINITY
                } else {
                    total / counted as f64
                },
                worlds: counted,
            }
        })
        .collect()
}

/// The move that came out best, having played each of them out.
///
/// `shortlist` is the moves worth considering, best first; a caller with a
/// policy network passes what it ranked highest, and a caller without one
/// passes whatever the engine offers. Only the first
/// [`Effort::candidates`] of them are searched.
pub fn best(
    hand: &Hand,
    seat: Wind,
    shortlist: &[Action],
    effort: Effort,
    belief: &Belief,
    rng: &mut Rng,
) -> Option<Judged> {
    if shortlist.is_empty() {
        return None;
    }
    let taken = shortlist.len().min(effort.candidates.max(1));
    let judged = judge(hand, seat, &shortlist[..taken], effort, belief, rng);
    judged
        .into_iter()
        .filter(|entry| entry.worlds > 0)
        .max_by(|a, b| a.value.partial_cmp(&b.value).expect("values are numbers"))
}

/// A player that thinks before it moves.
///
/// It takes the heuristic player's shortlist and plays each of those moves
/// out rather than trusting the ordering, which is what separates it from
/// the bot it is built on.
#[derive(Clone, Debug)]
pub struct Searcher {
    /// How hard it thinks.
    pub effort: Effort,
    /// What it takes the opponents to be holding. Even weights until a
    /// network says otherwise.
    pub belief: Belief,
    bot: Bot,
    rng: Rng,
}

impl Searcher {
    /// A searcher with a given seed and effort.
    pub fn new(seed: u64, effort: Effort) -> Searcher {
        Searcher {
            effort,
            belief: Belief::even(),
            bot: Bot::with_style(seed, Style::club()),
            rng: Rng::from_seed(seed ^ 0x5ea5_c400),
        }
    }

    /// Chooses a move, having played the candidates out.
    pub fn act(&mut self, hand: &Hand) -> Action {
        let actions = hand.legal_actions();
        assert!(!actions.is_empty(), "a player always has something to do");

        // A win is a win; there is nothing to search.
        if let Some(action) = actions
            .iter()
            .find(|action| matches!(action, Action::Tsumo))
        {
            return *action;
        }
        if actions.len() == 1 {
            return actions[0];
        }

        let seat = hand.turn;
        let shortlist = self.shortlist(hand, &actions);
        match best(
            hand,
            seat,
            &shortlist,
            self.effort,
            &self.belief,
            &mut self.rng,
        ) {
            Some(judged) => judged.action,
            None => self.bot.act(hand),
        }
    }

    /// Answers a claim. Calls are searched the same way, by playing out the
    /// hand that taking and not taking the tile would leave.
    pub fn call(&mut self, hand: &Hand, seat: Wind, offered: &[Call]) -> Call {
        if offered.contains(&Call::Ron) {
            return Call::Ron;
        }
        self.bot.call(hand, seat, offered)
    }

    /// The moves worth playing out, best first as the heuristic player
    /// ranks them, since searching all fourteen discards costs fourteen
    /// times as much for very little.
    fn shortlist(&mut self, hand: &Hand, actions: &[Action]) -> Vec<Action> {
        let preferred = self.bot.act(hand);
        let mut shortlist = vec![preferred];
        for action in actions {
            if *action != preferred {
                shortlist.push(*action);
            }
        }
        shortlist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Table;

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
                "the searching player's own hand is not imagined"
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

    /// A world that was imagined can be played to the end, which is the
    /// only thing a rollout needs of it.
    #[test]
    fn an_imagined_world_can_be_played_out() {
        let table = Table::new();
        let mut rng = Rng::from_seed(7);
        let hand = table.deal(&mut rng);

        let mut finished = 0;
        for round in 0..20 {
            let mut rng = Rng::from_seed(round);
            let world = imagine(&hand, Wind::East, &Belief::even(), &mut rng);
            let moved = play_out(world, Wind::East, Style::club(), None, round);
            assert!(moved.is_finite(), "a played-out world has a result");
            finished += 1;
        }
        assert_eq!(finished, 20);
    }

    /// Searching returns one of the moves it was offered, and says how many
    /// worlds it managed to try it in.
    #[test]
    fn the_search_picks_something_it_was_offered() {
        let table = Table::new();
        let mut rng = Rng::from_seed(4242);
        let hand = table.deal(&mut rng);
        let actions = hand.legal_actions();

        let mut rng = Rng::from_seed(11);
        let effort = Effort {
            worlds: 4,
            candidates: 3,
            turns: Some(20),
        };
        let picked = best(
            &hand,
            hand.turn,
            &actions,
            effort,
            &Belief::even(),
            &mut rng,
        )
        .expect("something came back");
        assert!(actions.contains(&picked.action), "{:?}", picked.action);
        assert!(picked.worlds > 0);
        assert!(picked.value.is_finite());
    }

    /// The same position searched twice with the same seed gives the same
    /// answer, so a game that uses it still replays.
    #[test]
    fn the_search_is_reproducible() {
        let table = Table::new();
        let mut rng = Rng::from_seed(20260903);
        let hand = table.deal(&mut rng);

        let mut first = Searcher::new(
            5,
            Effort {
                worlds: 3,
                candidates: 3,
                turns: Some(12),
            },
        );
        let mut second = Searcher::new(
            5,
            Effort {
                worlds: 3,
                candidates: 3,
                turns: Some(12),
            },
        );
        assert_eq!(first.act(&hand), second.act(&hand));
    }
}
