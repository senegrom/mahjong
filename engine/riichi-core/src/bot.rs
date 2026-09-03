//! A heuristic opponent: efficiency, a little value, and defence.
//!
//! This is the Club tier of the game and the fixed benchmark the trained
//! opponents are measured against. It plays by rules a competent club player
//! would recognise rather than by search:
//!
//! - discard for speed, counting how many tiles would improve the hand and
//!   how many copies of each are still unseen;
//! - keep a hand that can actually be declared, since a complete hand with
//!   no yaku is not a win (EMA 2025 section 3.2);
//! - fold against a declared riichi when the hand is not close, preferring
//!   tiles that player has already discarded, which can never deal in
//!   (section 3.3.9 makes them furiten on their own discards).
//!
//! It is deliberately simple and deterministic given a seed.

use crate::game::{Action, Call, Hand, Player};
use crate::hand::TileSet;
use crate::rng::Rng;
use crate::shanten;
use crate::tile::{Tile, COPIES};
use crate::Wind;

/// How the bot weighs speed against safety.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Style {
    /// How far from a win the hand may be before the bot folds against a
    /// declared riichi. Zero means fold unless already waiting; a large
    /// number means never fold at all.
    pub fold_beyond_shanten: i32,
    /// Whether to declare riichi whenever it is available.
    pub always_riichi: bool,
    /// How often to take a plausible discard rather than the best one, as a
    /// share between zero and one. A little of this is what separates a
    /// beginner from a player who counts.
    pub looseness: f64,
    /// What a dora is worth, counted in tiles of acceptance given up to keep
    /// it. Zero ignores dora entirely, which is how a beginner plays: fast
    /// hands worth nothing. The hand is only worth playing if it scores.
    pub dora_worth: i64,
}

impl Default for Style {
    fn default() -> Style {
        Style::club()
    }
}

impl Style {
    /// Plays for speed, never folds, and is loose about which tile goes.
    pub fn beginner() -> Style {
        Style {
            fold_beyond_shanten: 99,
            always_riichi: true,
            looseness: 0.35,
            dora_worth: 0,
        }
    }

    /// The benchmark: counts its tiles and folds against a riichi.
    pub fn club() -> Style {
        Style {
            fold_beyond_shanten: 1,
            always_riichi: true,
            looseness: 0.0,
            // Left at nothing until the duel says a dora is worth keeping.
            // Sixteen thousand games put it at about 0.018 placement, which
            // is short of two standard errors, so the knob is here and the
            // default is unchanged.
            dora_worth: 0,
        }
    }
}

/// The heuristic player.
#[derive(Clone, Debug)]
pub struct Bot {
    /// Its style.
    pub style: Style,
    rng: Rng,
}

impl Bot {
    /// A bot with the default style.
    pub fn new(seed: u64) -> Bot {
        Bot {
            style: Style::default(),
            rng: Rng::from_seed(seed),
        }
    }

    /// A bot with a chosen style.
    pub fn with_style(seed: u64, style: Style) -> Bot {
        Bot {
            style,
            rng: Rng::from_seed(seed),
        }
    }

    /// Chooses among the actions the engine offers the turn player.
    pub fn act(&mut self, hand: &Hand) -> Action {
        let actions = hand.legal_actions();
        assert!(!actions.is_empty(), "a player always has something to do");

        // Take a win whenever it is there.
        if let Some(action) = actions
            .iter()
            .find(|action| matches!(action, Action::Tsumo))
        {
            return *action;
        }

        let seat = hand.turn;
        let player = &hand.players[seat.index()];
        let threats = threats(hand, seat);

        // Riichi, when the hand is worth locking down.
        if self.style.always_riichi {
            let riichi: Vec<Action> = actions
                .iter()
                .copied()
                .filter(|action| matches!(action, Action::Riichi(_)))
                .collect();
            if !riichi.is_empty() {
                let choice = self.best_discard(
                    hand,
                    player,
                    riichi.iter().filter_map(discarded_tile).collect(),
                    &threats,
                );
                return Action::Riichi(choice);
            }
        }

        // A concealed quad, but never one that costs the hand its shape.
        if let Some(action) = actions.iter().find(|action| match action {
            Action::ConcealedKan(tile) => !self.quad_hurts(player, *tile),
            _ => false,
        }) {
            return *action;
        }

        let candidates: Vec<Tile> = actions.iter().filter_map(discarded_tile).collect();
        Action::Discard(self.best_discard(hand, player, candidates, &threats))
    }

    /// Chooses among the calls the engine offers, or passes.
    pub fn call(&mut self, hand: &Hand, seat: Wind, offered: &[Call]) -> Call {
        // A win is always taken.
        if offered.contains(&Call::Ron) {
            return Call::Ron;
        }
        let player = &hand.players[seat.index()];
        let tile = match hand.pending_discard {
            Some((_, tile)) => tile,
            None => return Call::Pass,
        };
        // Do not open a hand that has nowhere to go, and do not open at all
        // while somebody is waiting on a declared riichi.
        if !threats(hand, seat).is_empty() {
            return Call::Pass;
        }
        let before = shanten::shanten(&player.hand, player.melds.len());
        let mut best = Call::Pass;
        let mut best_shanten = before;
        for call in offered {
            let after = match call {
                Call::Pon | Call::Kan => {
                    let mut probe = player.hand;
                    let copies = if matches!(call, Call::Pon) { 2 } else { 3 };
                    for _ in 0..copies {
                        probe.remove(tile);
                    }
                    shanten::shanten(&probe, player.melds.len() + 1)
                }
                Call::Chii(low) => {
                    let mut probe = player.hand;
                    let second = match low.next_in_suit() {
                        Some(tile) => tile,
                        None => continue,
                    };
                    let third = match second.next_in_suit() {
                        Some(tile) => tile,
                        None => continue,
                    };
                    for member in [*low, second, third] {
                        if member != tile {
                            probe.remove(member);
                        }
                    }
                    shanten::shanten(&probe, player.melds.len() + 1)
                }
                _ => continue,
            };
            // Only call if it brings the hand closer and the hand can still
            // be declared once it is open.
            if after < best_shanten && self.open_hand_has_a_future(hand, seat, tile, call) {
                best_shanten = after;
                best = *call;
            }
        }
        best
    }

    /// Whether an opened hand could still find a yaku: a triplet of dragons
    /// or of a wind that scores, or a hand with no terminals or honours at
    /// all, which All Simples covers.
    fn open_hand_has_a_future(&self, hand: &Hand, seat: Wind, tile: Tile, call: &Call) -> bool {
        let player = &hand.players[seat.index()];
        if matches!(call, Call::Pon | Call::Kan) {
            let scores = tile.is_dragon() || tile == seat.tile() || tile == hand.round.tile();
            if scores {
                return true;
            }
        }
        // Otherwise the hand needs to be heading for All Simples.
        let mut tiles: Vec<Tile> = player.hand.tiles().collect();
        tiles.push(tile);
        for meld in &player.melds {
            tiles.extend(meld.tiles());
        }
        let terminals = tiles
            .iter()
            .filter(|tile| tile.is_terminal_or_honour())
            .count();
        terminals <= 1
    }

    /// Whether declaring a quad would leave the hand further from a win.
    fn quad_hurts(&self, player: &Player, tile: Tile) -> bool {
        let before = shanten::shanten(&player.hand, player.melds.len());
        let mut probe = player.hand;
        probe.counts_mut()[tile.idx()] = 0;
        let after = shanten::shanten(&probe, player.melds.len() + 1);
        after > before
    }

    /// Picks the discard: closest to a win, then widest, then safest.
    fn best_discard(
        &mut self,
        hand: &Hand,
        player: &Player,
        candidates: Vec<Tile>,
        threats: &[Wind],
    ) -> Tile {
        assert!(
            !candidates.is_empty(),
            "there is always something to discard"
        );
        let unseen = unseen_counts(hand, player);
        let dora: Vec<Tile> = hand
            .wall
            .dora_indicators()
            .into_iter()
            .map(|indicator| indicator.dora())
            .collect();
        let current = shanten::shanten(&player.hand, player.melds.len());
        let folding = !threats.is_empty() && current > self.style.fold_beyond_shanten;

        // Two passes: shanten for every candidate first, which is cheap,
        // then the far more expensive acceptance count only for the ones
        // that are actually in the running.
        let mut scored: Vec<(Tile, i32)> = Vec::with_capacity(candidates.len());
        for tile in &candidates {
            let mut probe = player.hand;
            if probe.remove(*tile) {
                scored.push((*tile, shanten::shanten(&probe, player.melds.len())));
            }
        }
        let closest = scored
            .iter()
            .map(|(_, value)| *value)
            .min()
            .unwrap_or(current);
        let shortlist: Vec<(Tile, i32)> = if folding {
            scored.clone()
        } else {
            scored
                .iter()
                .copied()
                .filter(|(_, value)| *value == closest)
                .collect()
        };

        // A loose player sometimes keeps the wrong tile. It still has to be
        // a tile they hold, so the hand stays legal, just not the best.
        if self.style.looseness > 0.0
            && (self.rng.next_u64() as f64 / u64::MAX as f64) < self.style.looseness
        {
            let pick = self.rng.below(candidates.len());
            return candidates[pick];
        }

        let mut best = candidates[0];
        let mut best_key = (i64::MIN, i64::MIN, i64::MIN, 0u64);
        for (tile, after) in shortlist {
            let mut probe = player.hand;
            if !probe.remove(tile) {
                continue;
            }
            // While folding, only safety matters, and the acceptance count
            // is the expensive part of this loop.
            let width = if folding {
                0
            } else {
                acceptance_width(&probe, player, &unseen)
            };
            // Throwing a dora costs a han. Counted here in tiles of
            // acceptance, so a hand only gives one up when what it gets back
            // is worth more than the value it loses. A tile that several
            // indicators point at is worth that much more.
            let value = self.style.dora_worth
                * dora.iter().filter(|marked| **marked == tile).count() as i64;
            let safety = threats
                .iter()
                .map(|threat| safety_of(hand, *threat, tile))
                .min()
                .unwrap_or(100) as i64;

            // Folding puts safety first; otherwise speed leads and safety
            // only breaks ties.
            let key = if folding {
                (safety, -(after as i64), width as i64, self.rng.next_u64())
            } else {
                (
                    -(after as i64),
                    width as i64 - value,
                    safety,
                    self.rng.next_u64(),
                )
            };
            if key > best_key {
                best_key = key;
                best = tile;
            }
        }
        best
    }
}

fn discarded_tile(action: &Action) -> Option<Tile> {
    match action {
        Action::Discard(tile) | Action::Riichi(tile) => Some(*tile),
        _ => None,
    }
}

/// The seats that have declared riichi and are therefore waiting.
fn threats(hand: &Hand, seat: Wind) -> Vec<Wind> {
    Wind::ALL
        .into_iter()
        .filter(|other| *other != seat && hand.players[other.index()].has_riichi())
        .collect()
}

/// How many copies of each kind the player cannot see anywhere.
fn unseen_counts(hand: &Hand, player: &Player) -> TileSet {
    let mut seen = player.visible_to_self();
    for other in &hand.players {
        for discard in &other.discards {
            seen.add(discard.tile);
        }
        if !core::ptr::eq(other, player) {
            for meld in &other.melds {
                for tile in meld.tiles() {
                    seen.add(tile);
                }
            }
        }
    }
    for indicator in hand.wall.dora_indicators() {
        seen.add(indicator);
    }
    let mut unseen = TileSet::new();
    for tile in Tile::all() {
        let count = seen.count(tile).min(COPIES);
        unseen.add_n(tile, COPIES - count);
    }
    unseen
}

/// How many tiles would bring the hand closer, counting the copies left.
fn acceptance_width(hand: &TileSet, player: &Player, unseen: &TileSet) -> u32 {
    let visible = player.visible_to_self();
    shanten::acceptance(hand, player.melds.len(), &visible)
        .into_iter()
        .map(|(tile, _)| unseen.count(tile) as u32)
        .sum()
}

/// A rough safety score against one player who has declared riichi.
///
/// Anything the engine calls safe against them cannot deal in at all: their
/// own discards, and everything that passed after they declared (EMA section
/// 3.3.9). The rest is ranked by the shapes it can complete, which is why
/// honours and terminals are safer than the middle of a suit, and why a tile
/// whose suji partner has been discarded is safer than one that has not.
fn safety_of(hand: &Hand, threat: Wind, tile: Tile) -> u8 {
    if hand.safe_against(threat).count(tile) > 0 {
        return 100;
    }
    let player = &hand.players[threat.index()];
    if tile.is_honour() {
        // Honours cannot sit in a sequence, so only a pair or triplet wait
        // catches them, and every copy already gone makes that less likely.
        let seen = hand
            .players
            .iter()
            .flat_map(|other| other.discards.iter())
            .filter(|discard| discard.tile == tile)
            .count();
        return 60 + 10 * seen.min(3) as u8;
    }
    // Suji: if the tile three away has been discarded by the waiting player,
    // a two-sided wait on this one is ruled out.
    let rank = tile.rank();
    let suji = [
        rank.checked_sub(3),
        Some(rank + 3).filter(|value| *value <= 9),
    ]
    .into_iter()
    .flatten()
    .filter(|value| (1..=9).contains(value))
    .any(|value| {
        let other = Tile::numbered(tile.suit(), value);
        player.discards.iter().any(|discard| discard.tile == other)
    });
    let base = match rank {
        1 | 9 => 40,
        2 | 8 => 30,
        3 | 7 => 20,
        _ => 10,
    };
    if suji {
        base + 20
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Phase;
    use crate::table::Table;

    /// Plays a whole game with four bots, which exercises the engine as much
    /// as it does the bot.
    fn play_a_game(seed: u64) -> Table {
        let mut table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut bots: Vec<Bot> = (0..4).map(|index| Bot::new(seed + index)).collect();
        let mut guard = 0;
        while !table.finished {
            guard += 1;
            assert!(guard < 200, "a game of two rounds should not run this long");
            let mut hand = table.deal(&mut rng);
            let mut turns = 0;
            while !matches!(hand.phase, Phase::Over) {
                turns += 1;
                assert!(turns < 600, "a hand should end well before this");
                match hand.phase {
                    Phase::Draw => {
                        let _ = hand.draw();
                    }
                    Phase::Act => {
                        let seat = hand.turn;
                        let action = bots[table.player_at(seat)].act(&hand);
                        hand.act(action).expect("the bot chose a legal action");
                    }
                    Phase::CallWindow => {
                        let offered = hand.legal_calls();
                        let answers: Vec<(Wind, Call)> = offered
                            .iter()
                            .map(|(seat, calls)| {
                                let player = table.player_at(*seat);
                                (*seat, bots[player].call(&hand, *seat, calls))
                            })
                            .collect();
                        hand.resolve_calls(&answers)
                            .expect("the bots chose legal calls");
                    }
                    Phase::Over => break,
                }
            }
            table.finish(&hand);
        }
        table
    }

    #[test]
    fn four_bots_play_a_whole_game() {
        let table = play_a_game(20260902);
        let total: i32 = table.scores.iter().sum::<i32>() + (table.riichi_sticks * 1000) as i32;
        assert_eq!(total, 120_000, "the table's points are conserved");
        assert!(table.hands_played >= 8);
        assert_eq!(table.final_scores().iter().sum::<i32>(), 0);
    }

    #[test]
    fn games_are_reproducible_from_their_seed() {
        let first = play_a_game(7);
        let second = play_a_game(7);
        assert_eq!(first.scores, second.scores);
        let other = play_a_game(8);
        assert!(
            first.scores != other.scores || first.hands_played != other.hands_played,
            "different seeds should give different games"
        );
    }

    /// EMA 2025 section 3.3.9: a tile the waiting player has already
    /// discarded cannot deal in, so the bot must rank it as safe.
    #[test]
    fn discarded_tiles_are_safe_against_that_player() {
        let mut rng = Rng::from_seed(1);
        let mut hand = Hand::deal(&mut rng, Wind::East, 1, 0, 0, [25000; 4]);
        let tile: Tile = "5p".parse().unwrap();
        hand.players[1].riichi = crate::score::Riichi::Declared;
        hand.players[1].discards.push(crate::game::Discard {
            tile,
            order: 0,
            drawn: true,
            riichi: true,
            claimed: false,
        });
        assert_eq!(safety_of(&hand, Wind::South, tile), 100);
        let unknown: Tile = "4p".parse().unwrap();
        assert!(safety_of(&hand, Wind::South, unknown) < 100);
        // Honours are safer than the middle of a suit.
        assert!(
            safety_of(&hand, Wind::South, "1z".parse().unwrap())
                > safety_of(&hand, Wind::South, unknown)
        );
    }

    #[test]
    fn the_bot_folds_against_a_riichi_when_far_from_a_win() {
        let mut rng = Rng::from_seed(2);
        let mut hand = Hand::deal(&mut rng, Wind::East, 1, 0, 0, [25000; 4]);
        let safe: Tile = "1z".parse().unwrap();
        hand.players[1].riichi = crate::score::Riichi::Declared;
        hand.players[1].discards.push(crate::game::Discard {
            tile: safe,
            order: 0,
            drawn: true,
            riichi: true,
            claimed: false,
        });
        // A hand with nothing going on, holding one tile the riichi player
        // has already discarded.
        hand.players[0].hand = "19m19p19s1234567z".parse().unwrap();
        hand.players[0].hand.remove("7z".parse().unwrap());
        hand.players[0].hand.add(safe);
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = Some(safe);
        let mut bot = Bot::new(3);
        match bot.act(&hand) {
            Action::Discard(tile) => assert_eq!(tile, safe, "the known-safe tile should go"),
            other => panic!("expected a discard, got {other:?}"),
        }
    }
}
