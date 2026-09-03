//! Looking back at a hand and saying where it went wrong.
//!
//! A player learns from the moves they would not have made, so a review has
//! to do more than mark a move right or wrong. Each note here carries the
//! numbers behind the judgement: how far the hand was from complete, how
//! many tiles would have improved it, and whether the tile was one a player
//! who had declared riichi could win on. The player can then disagree with
//! the advice and see exactly what they are trading away.
//!
//! The adviser is the heuristic bot, which is always available and instant.
//! A trained network can be asked the same question through
//! [`crate::encoding`], and the two answers together are more informative
//! than either alone, because they disagree in interesting places.

use crate::bot::Bot;
use crate::game::{Action, Hand};
use crate::hand::TileSet;
use crate::shanten;
use crate::tile::Tile;
use crate::Wind;

/// How exposed a discard was.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Danger {
    /// Nobody had declared riichi, so nothing was being read.
    Quiet,
    /// The tile could not deal in: it was already through, or the player
    /// waiting for it is furiten on it.
    Safe,
    /// The tile might have dealt in.
    Live,
}

/// Why the adviser would have played something else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    /// The move played is the one advised.
    Agreed,
    /// The advised tile leaves the hand closer to complete.
    Shape,
    /// Both leave the same distance, but more tiles improve the hand after
    /// the advised one.
    Acceptance,
    /// The advised tile is safe where the one played was not.
    Defence,
    /// The advice differs for a reason these numbers do not capture, which
    /// is usually the value of the hand rather than its speed.
    Judgement,
}

impl Reason {
    /// A short line a person can read.
    pub fn line(self) -> &'static str {
        match self {
            Reason::Agreed => "the move the adviser would have made",
            Reason::Shape => "the advised tile leaves the hand closer to complete",
            Reason::Acceptance => "the advised tile leaves more tiles that improve the hand",
            Reason::Defence => "the advised tile could not deal in, the one played could",
            Reason::Judgement => "a trade of speed against value or safety",
        }
    }
}

/// What one decision looked like afterwards.
#[derive(Clone, Debug)]
pub struct Note {
    /// Whose decision it was.
    pub seat: Wind,
    /// Which discard of the hand this was, counting from zero.
    pub turn: u32,
    /// What was actually done.
    pub played: Action,
    /// What the adviser would have done.
    pub advised: Action,
    /// Distance to a complete hand after the move played, where zero is
    /// waiting and minus one is complete.
    pub shanten_played: i32,
    /// The same after the advised move.
    pub shanten_advised: i32,
    /// How many tiles, of those not yet seen, would improve the hand after
    /// the move played.
    pub acceptance_played: u32,
    /// The same after the advised move.
    pub acceptance_advised: u32,
    /// How exposed the tile played was.
    pub danger_played: Danger,
    /// How exposed the advised tile was.
    pub danger_advised: Danger,
    /// Why the two differ.
    pub reason: Reason,
}

impl Note {
    /// Whether the move played is the advised one.
    pub fn agreed(&self) -> bool {
        self.played == self.advised
    }

    /// How much acceptance the move gave up, which is zero when it gave up
    /// none. A hand that is further from complete has given up everything.
    pub fn cost(&self) -> u32 {
        if self.shanten_played > self.shanten_advised {
            return self.acceptance_advised;
        }
        self.acceptance_advised
            .saturating_sub(self.acceptance_played)
    }
}

/// The tile an action discards, if it discards one.
fn discarded(action: Action) -> Option<Tile> {
    match action {
        Action::Discard(tile) | Action::Riichi(tile) => Some(tile),
        Action::Tsumo | Action::ConcealedKan(_) | Action::ExtendedKan(_) => None,
    }
}

/// Everything a seat can see: their own tiles, every discard on the table,
/// the called sets and the dora indicators.
///
/// This is what is left of a tile somebody might be waiting for, so it
/// answers both "how many of these could still come" for the acceptance
/// count and "how thin is this wait" for the player.
pub fn visible_to(hand: &Hand, seat: Wind) -> TileSet {
    let mut seen = hand.players[seat.index()].visible_to_self();
    for other in Wind::ALL {
        if other == seat {
            continue;
        }
        let player = &hand.players[other.index()];
        for discard in &player.discards {
            seen.add(discard.tile);
        }
        for meld in &player.melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
    }
    for indicator in hand.wall.dora_indicators() {
        seen.add(indicator);
    }
    seen
}

/// How exposed a tile is against everyone who has declared riichi.
fn danger_of(hand: &Hand, seat: Wind, tile: Tile) -> Danger {
    let declared: Vec<Wind> = Wind::ALL
        .into_iter()
        .filter(|other| *other != seat && hand.players[other.index()].has_riichi())
        .collect();
    if declared.is_empty() {
        return Danger::Quiet;
    }
    if declared
        .iter()
        .all(|other| hand.safe_against(*other).count(tile) > 0)
    {
        Danger::Safe
    } else {
        Danger::Live
    }
}

/// The hand left after a move, and how it stands.
fn after(hand: &Hand, seat: Wind, action: Action) -> (i32, u32) {
    let player = &hand.players[seat.index()];
    let mut left = player.hand;
    let mut called = player.melds.len();
    match action {
        Action::Discard(tile) | Action::Riichi(tile) => {
            left.remove(tile);
        }
        Action::ConcealedKan(tile) => {
            for _ in 0..4 {
                left.remove(tile);
            }
            called += 1;
        }
        Action::ExtendedKan(tile) => {
            left.remove(tile);
        }
        // A win ends the hand, so there is nothing left to judge.
        Action::Tsumo => return (shanten::COMPLETE, 0),
    }
    let seen = visible_to(hand, seat);
    let distance = shanten::shanten(&left, called);
    let acceptance: u32 = shanten::acceptance(&left, called, &seen)
        .into_iter()
        .map(|(tile, _)| (crate::tile::COPIES as i32 - seen.count(tile) as i32).max(0) as u32)
        .sum();
    (distance, acceptance)
}

/// How many of `tile` nobody has seen yet, from `seat`'s side of the table.
///
/// A wait on a tile of which three are already down is a much thinner hand
/// than the number of waits alone suggests, which is why this is worth
/// showing next to the wait rather than leaving the player to count.
pub fn how_many_left(hand: &Hand, seat: Wind, tile: Tile) -> u8 {
    crate::tile::COPIES.saturating_sub(visible_to(hand, seat).count(tile))
}

/// Judges one decision. `hand` is the position as it stood before the move,
/// and `played` is what was done from it.
pub fn judge(hand: &Hand, played: Action, adviser: &mut Bot) -> Note {
    let seat = hand.turn;
    let advised = adviser.act(hand);
    let (shanten_played, acceptance_played) = after(hand, seat, played);
    let (shanten_advised, acceptance_advised) = after(hand, seat, advised);
    let danger_played = discarded(played)
        .map(|tile| danger_of(hand, seat, tile))
        .unwrap_or(Danger::Quiet);
    let danger_advised = discarded(advised)
        .map(|tile| danger_of(hand, seat, tile))
        .unwrap_or(Danger::Quiet);

    let reason = if played == advised {
        Reason::Agreed
    } else if shanten_advised < shanten_played {
        Reason::Shape
    } else if danger_played == Danger::Live && danger_advised == Danger::Safe {
        Reason::Defence
    } else if shanten_advised == shanten_played && acceptance_advised > acceptance_played {
        Reason::Acceptance
    } else {
        Reason::Judgement
    };

    Note {
        seat,
        turn: hand.discards_made,
        played,
        advised,
        shanten_played,
        shanten_advised,
        acceptance_played,
        acceptance_advised,
        danger_played,
        danger_advised,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Phase;
    use crate::rng::Rng;
    use crate::table::Table;

    /// A note about a move the adviser itself made must agree with it, and
    /// must say so for the right reason.
    #[test]
    fn the_adviser_agrees_with_itself() {
        let table = Table::new();
        let mut rng = Rng::from_seed(5);
        let mut bots: Vec<Bot> = (0..4).map(Bot::new).collect();
        let mut hand = table.deal(&mut rng);
        let mut judged = 0;

        while !matches!(hand.phase, Phase::Over) && judged < 30 {
            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    let seat = hand.turn;
                    let played = bots[table.player_at(seat)].act(&hand);
                    // The same bot, at the same point, asked again.
                    let mut adviser = Bot::new(table.player_at(seat) as u64);
                    let note = judge(&hand, played, &mut adviser);
                    if note.agreed() {
                        assert_eq!(note.reason, Reason::Agreed);
                        assert_eq!(note.cost(), 0, "agreeing costs nothing");
                        judged += 1;
                    }
                    hand.act(played).expect("the bot chose a legal action");
                }
                Phase::CallWindow => {
                    let answers: Vec<(Wind, crate::game::Call)> = hand
                        .legal_calls()
                        .iter()
                        .map(|(seat, calls)| {
                            (
                                *seat,
                                bots[table.player_at(*seat)].call(&hand, *seat, calls),
                            )
                        })
                        .collect();
                    hand.resolve_calls(&answers)
                        .expect("the calls were offered");
                }
                Phase::Over => break,
            }
        }
        assert!(judged > 10, "only {judged} moves were judged");
    }

    /// Throwing away a tile the hand needs is marked as costing shape, and
    /// the numbers back the label up.
    #[test]
    fn a_bad_discard_is_named_as_one() {
        let mut hand = crate::game::Hand::deal(
            &mut Rng::from_seed(20260903),
            Wind::East,
            1,
            0,
            0,
            [30000; 4],
        );
        // Give East a hand one tile from waiting, then throw the wrong tile.
        let player = &mut hand.players[0];
        player.hand = TileSet::new();
        for tile in [
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "9s",
        ] {
            player.hand.add(tile.parse().unwrap());
        }
        hand.drawn = Some("9s".parse().unwrap());
        hand.phase = Phase::Act;
        hand.turn = Wind::East;

        let mut adviser = Bot::new(1);
        // Both discards reach tenpai, so this is not about distance. Letting
        // the lone nine go waits on 1p and 4p, six tiles; breaking the pair
        // waits on the last three nines. The adviser takes the wider wait,
        // and declares riichi on it.
        let note = judge(&hand, Action::Discard("1p".parse().unwrap()), &mut adviser);
        assert_eq!(discarded(note.advised), Some("9s".parse().unwrap()));
        assert!(
            matches!(note.advised, Action::Riichi(_)),
            "{:?}",
            note.advised
        );
        assert!(!note.agreed());
        assert_eq!(
            note.shanten_played, note.shanten_advised,
            "both are waiting"
        );
        assert_eq!(note.acceptance_played, 3, "three nines are left");
        assert_eq!(note.acceptance_advised, 6, "two 1p and four 4p");
        assert_eq!(note.cost(), 3, "the narrower wait gave up three tiles");
        assert_eq!(note.reason, Reason::Acceptance);
    }

    /// A tile the player holds four of is not coming, and one nobody has
    /// touched still has all four out there.
    #[test]
    fn a_wait_says_how_many_are_left() {
        let mut hand =
            crate::game::Hand::deal(&mut Rng::from_seed(4242), Wind::East, 1, 0, 0, [30000; 4]);
        let all_four: Tile = "3p".parse().unwrap();
        let untouched: Tile = "6s".parse().unwrap();

        // Clear the table so the arithmetic is about what is set here.
        for seat in Wind::ALL {
            hand.players[seat.index()].hand = TileSet::new();
            hand.players[seat.index()].discards.clear();
            hand.players[seat.index()].melds.clear();
        }
        for _ in 0..4 {
            hand.players[0].hand.add(all_four);
        }

        assert_eq!(
            how_many_left(&hand, Wind::East, all_four),
            0,
            "holding all four leaves none"
        );

        let indicators = hand.wall.dora_indicators();
        let expected = 4 - indicators.iter().filter(|tile| **tile == untouched).count() as u8;
        assert_eq!(
            how_many_left(&hand, Wind::East, untouched),
            expected,
            "an untouched tile keeps every copy the indicators have not shown"
        );

        // A tile another player has thrown is one fewer, from every seat.
        hand.players[1].discards.push(crate::game::Discard {
            tile: untouched,
            order: 0,
            drawn: false,
            riichi: false,
            claimed: false,
        });
        assert_eq!(how_many_left(&hand, Wind::East, untouched), expected - 1);
        assert_eq!(how_many_left(&hand, Wind::West, untouched), expected - 1);
    }

    /// With somebody waiting on a declared riichi, a tile that is already
    /// through cannot deal in and a fresh one can. The review has to say so
    /// when the two are otherwise the same choice.
    #[test]
    fn throwing_into_a_riichi_is_named_as_the_reason() {
        let mut hand =
            crate::game::Hand::deal(&mut Rng::from_seed(31337), Wind::East, 1, 0, 0, [30000; 4]);

        // South declares, and their own discards are safe against them.
        let through: Tile = "9m".parse().unwrap();
        let fresh: Tile = "5p".parse().unwrap();
        {
            let south = &mut hand.players[Wind::South.index()];
            south.riichi = crate::score::Riichi::Declared;
            south.riichi_order = Some(0);
            south.discards.push(crate::game::Discard {
                tile: through,
                order: 0,
                drawn: false,
                riichi: true,
                claimed: false,
            });
        }

        // East holds two tiles that are equally useless to the hand, so the
        // only thing separating them is whether they can deal in.
        {
            let east = &mut hand.players[Wind::East.index()];
            east.hand = TileSet::new();
            for tile in [
                "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "1p", "1p", "2p",
            ] {
                east.hand.add(tile.parse().unwrap());
            }
            east.hand.add(through);
            east.hand.add(fresh);
        }
        hand.drawn = Some(fresh);
        hand.turn = Wind::East;
        hand.phase = Phase::Act;

        assert_eq!(
            danger_of(&hand, Wind::East, through),
            Danger::Safe,
            "a tile South has already thrown cannot deal into South"
        );
        assert_eq!(
            danger_of(&hand, Wind::East, fresh),
            Danger::Live,
            "a tile nobody has seen might"
        );

        // Playing the live tile when the safe one was there is a defence
        // note, whichever of them the adviser happens to prefer.
        let mut adviser = Bot::new(9);
        let note = judge(&hand, Action::Discard(fresh), &mut adviser);
        if discarded(note.advised) == Some(through) {
            assert_eq!(note.reason, Reason::Defence, "{note:?}");
            assert_eq!(note.danger_played, Danger::Live);
            assert_eq!(note.danger_advised, Danger::Safe);
        }

        // And with nobody waiting, neither tile is dangerous at all.
        hand.players[Wind::South.index()].riichi = crate::score::Riichi::None;
        assert_eq!(danger_of(&hand, Wind::East, fresh), Danger::Quiet);
    }

    /// Every reason has a line, and none of them is empty.
    #[test]
    fn every_reason_reads_as_a_sentence() {
        for reason in [
            Reason::Agreed,
            Reason::Shape,
            Reason::Acceptance,
            Reason::Defence,
            Reason::Judgement,
        ] {
            assert!(reason.line().len() > 10, "{reason:?} has no line");
        }
    }
}
