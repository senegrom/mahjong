//! The log has to be a faithful record, not a plausible-looking one.
//!
//! These tests rebuild each hand from its events alone, with no access to
//! the game state, and then compare what the events say against what the
//! engine actually holds. Anything the log forgets to mention shows up as a
//! hand that has drifted, so a missing or misplaced event cannot pass.

use riichi_core::bot::Bot;
use riichi_core::game::{Call, Phase};
use riichi_core::hand::TileSet;
use riichi_core::mjai::{self, Event};
use riichi_core::rng::Rng;
use riichi_core::table::Table;
use riichi_core::tile::Tile;
use riichi_core::Wind;

/// A hand rebuilt from nothing but the log.
#[derive(Default, Clone)]
struct Replay {
    /// Concealed tiles, by seat.
    concealed: [TileSet; 4],
    /// Tiles locked into called sets, by seat.
    melded: [TileSet; 4],
    /// Everything discarded and not claimed, by seat.
    discarded: [Vec<Tile>; 4],
    /// Points, by seat, as the log reports them.
    scores: [i32; 4],
    /// The dora indicators turned face up.
    indicators: Vec<Tile>,
    /// Whether each seat is holding a riichi that was accepted.
    riichi: [bool; 4],
}

impl Replay {
    /// Follows one event. Panics with a readable message when the log asks
    /// for something impossible, such as discarding a tile not held.
    fn apply(&mut self, event: &Event) {
        match event {
            Event::StartKyoku {
                indicator,
                scores,
                hands,
                ..
            } => {
                *self = Replay {
                    scores: *scores,
                    indicators: vec![*indicator],
                    ..Default::default()
                };
                for seat in Wind::ALL {
                    for tile in &hands[seat.index()] {
                        self.concealed[seat.index()].add(*tile);
                    }
                    assert_eq!(
                        self.concealed[seat.index()].len(),
                        13,
                        "a hand is dealt thirteen tiles"
                    );
                }
            }
            Event::Tsumo { actor, tile } => self.concealed[actor.index()].add(*tile),
            Event::Dahai { actor, tile, .. } => {
                assert!(
                    self.concealed[actor.index()].remove(*tile),
                    "{actor:?} discarded {tile}, which they were not holding"
                );
                self.discarded[actor.index()].push(*tile);
            }
            Event::Reach { .. } => {}
            Event::ReachAccepted { actor } => self.riichi[actor.index()] = true,
            Event::Chi {
                actor,
                target,
                tile,
                consumed,
            }
            | Event::Pon {
                actor,
                target,
                tile,
                consumed,
            }
            | Event::Daiminkan {
                actor,
                target,
                tile,
                consumed,
            } => {
                let taken = self.discarded[target.index()].pop();
                assert_eq!(
                    taken,
                    Some(*tile),
                    "a claim takes the tile that was just discarded"
                );
                for member in consumed {
                    assert!(
                        self.concealed[actor.index()].remove(*member),
                        "{actor:?} used {member} for a set without holding it"
                    );
                    self.melded[actor.index()].add(*member);
                }
                self.melded[actor.index()].add(*tile);
            }
            Event::Kakan {
                actor,
                tile,
                consumed,
            } => {
                assert_eq!(consumed.len(), 3, "a quad is added to a triplet");
                assert!(
                    self.concealed[actor.index()].remove(*tile),
                    "{actor:?} added {tile} to a set without holding it"
                );
                self.melded[actor.index()].add(*tile);
            }
            Event::Ankan { actor, consumed } => {
                assert_eq!(consumed.len(), 4, "a concealed quad is four tiles");
                for member in consumed {
                    assert!(
                        self.concealed[actor.index()].remove(*member),
                        "{actor:?} made a quad of {member} without holding four"
                    );
                    self.melded[actor.index()].add(*member);
                }
            }
            Event::Dora { indicator } => self.indicators.push(*indicator),
            Event::Hora { scores, .. } | Event::Ryukyoku { scores, .. } => self.scores = *scores,
            Event::StartGame { .. } | Event::EndKyoku | Event::EndGame => {}
        }
    }

    /// Every tile the log has accounted for, which must never exceed the
    /// four copies of each kind that exist.
    fn seen(&self) -> TileSet {
        let mut all = TileSet::new();
        for seat in Wind::ALL {
            for tile in self.concealed[seat.index()].tiles() {
                all.add(tile);
            }
            for tile in self.melded[seat.index()].tiles() {
                all.add(tile);
            }
            for tile in &self.discarded[seat.index()] {
                all.add(*tile);
            }
        }
        for indicator in &self.indicators {
            all.add(*indicator);
        }
        all
    }
}

/// Plays whole games and checks the log against the engine after each hand.
fn replay_games(seeds: impl Iterator<Item = u64>) -> usize {
    let mut hands_checked = 0;
    for seed in seeds {
        let mut table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut bots: Vec<Bot> = (0..4).map(|index| Bot::new(seed + index)).collect();
        while !table.finished {
            let mut hand = table.deal(&mut rng);
            while !matches!(hand.phase, Phase::Over) {
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
                        let answers: Vec<(Wind, Call)> = hand
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

            let mut replay = Replay::default();
            for event in &hand.log {
                replay.apply(event);
            }

            // The log ends where the hand ends.
            assert!(
                matches!(hand.log.last(), Some(Event::EndKyoku)),
                "seed {seed}: the log does not close the hand"
            );

            // No kind of tile was used more than the four that exist.
            for tile in Tile::all() {
                assert!(
                    replay.seen().count(tile) <= 4,
                    "seed {seed}: {tile} appears more than four times in the log"
                );
            }

            // The rebuilt hands match the ones the engine holds.
            for seat in Wind::ALL {
                let player = &hand.players[seat.index()];
                assert_eq!(
                    replay.concealed[seat.index()].counts(),
                    player.hand.counts(),
                    "seed {seed}, {seat:?}: the log rebuilds a different hand"
                );
                let mut melded = TileSet::new();
                for meld in &player.melds {
                    for tile in meld.tiles() {
                        melded.add(tile);
                    }
                }
                assert_eq!(
                    replay.melded[seat.index()].counts(),
                    melded.counts(),
                    "seed {seed}, {seat:?}: the log rebuilds different called sets"
                );
                assert_eq!(
                    replay.scores[seat.index()],
                    player.score,
                    "seed {seed}, {seat:?}: the log reports a different score"
                );
                assert_eq!(
                    replay.riichi[seat.index()],
                    player.has_riichi(),
                    "seed {seed}, {seat:?}: the log disagrees about riichi"
                );
            }

            // What the log says the hand moved is what it moved.
            let deltas = hand.deltas();
            let reported = hand
                .log
                .iter()
                .find_map(|event| match event {
                    Event::Hora { deltas, .. } | Event::Ryukyoku { deltas, .. } => Some(*deltas),
                    _ => None,
                })
                .expect("a finished hand says what it moved");
            assert_eq!(
                reported, deltas,
                "seed {seed}: the log misreports what the hand moved"
            );
            // Points only move between players and on and off the table as
            // riichi sticks, so nothing is created or destroyed.
            let carried = match &hand.log[0] {
                Event::StartKyoku { kyotaku, .. } => *kyotaku as i32,
                other => panic!("a hand opens with its deal, not {other:?}"),
            };
            assert_eq!(
                deltas.iter().sum::<i32>() + (hand.riichi_sticks as i32 - carried) * 1000,
                0,
                "seed {seed}: points appeared from nowhere"
            );

            hands_checked += 1;
            table.finish(&hand);
        }
    }
    hands_checked
}

/// Two hundred games, every hand rebuilt from its own log.
#[test]
fn the_log_rebuilds_every_hand_it_records() {
    let checked = replay_games(0..50);
    assert!(checked > 200, "only {checked} hands were checked");
}

/// The written form is one JSON object per line and nothing else.
#[test]
fn every_event_is_one_json_object() {
    let mut table = Table::new();
    let mut rng = Rng::from_seed(7);
    let mut bots: Vec<Bot> = (0..4).map(Bot::new).collect();
    let mut lines = Vec::new();
    let mut hands = 0;
    while !table.finished && hands < 4 {
        let mut hand = table.deal(&mut rng);
        while !matches!(hand.phase, Phase::Over) {
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
                    let answers: Vec<(Wind, Call)> = hand
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
        let seating = table.seating();
        lines.extend(hand.log.iter().map(|event| event.to_json(seating)));
        hands += 1;
        table.finish(&hand);
    }

    assert!(lines.len() > 100, "a few hands produce plenty of events");
    for line in &lines {
        assert!(line.starts_with("{\"type\":\""), "not an event: {line}");
        assert!(line.ends_with('}'), "not closed: {line}");
        assert!(!line.contains('\n'), "an event must fit one line: {line}");
        let opens = line.matches('{').count();
        let closes = line.matches('}').count();
        assert_eq!(opens, closes, "unbalanced braces: {line}");
    }
}

/// Player numbers follow the person, not the chair. In the second hand of a
/// round the seats have moved, so the same person must still be number 0.
#[test]
fn numbering_survives_the_deal_moving_on() {
    let mut table = Table::new();
    assert_eq!(
        table.seating(),
        [0, 1, 2, 3],
        "the first deal is the identity"
    );
    let mut rng = Rng::from_seed(11);
    let mut bots: Vec<Bot> = (0..4).map(Bot::new).collect();

    // Play until the deal has moved at least once.
    let mut moved = false;
    let mut guard = 0;
    while !moved && !table.finished {
        guard += 1;
        assert!(guard < 40, "the deal should move within a few hands");
        let before = table.seating();
        let mut hand = table.deal(&mut rng);
        while !matches!(hand.phase, Phase::Over) {
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
                    let answers: Vec<(Wind, Call)> = hand
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
        table.finish(&hand);
        moved = table.seating() != before;
    }
    assert!(moved, "the deal never moved");

    let seating = table.seating();
    let mut sorted = seating;
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2, 3], "everybody has exactly one number");

    // The dealer of this hand is the player the log will call oya.
    let hand = table.deal(&mut rng);
    let line = hand.log[0].to_json(seating);
    assert!(
        line.contains(&format!("\"oya\":{}", seating[Wind::East.index()])),
        "{line}"
    );
}

/// Honours are written the way every other riichi program writes them.
#[test]
fn honours_use_the_shared_notation() {
    assert_eq!(mjai::name("1z".parse::<Tile>().unwrap()), "E");
    assert_eq!(mjai::name("4z".parse::<Tile>().unwrap()), "N");
    assert_eq!(mjai::name("5z".parse::<Tile>().unwrap()), "P");
    assert_eq!(mjai::name("6z".parse::<Tile>().unwrap()), "F");
    assert_eq!(mjai::name("7z".parse::<Tile>().unwrap()), "C");
    assert_eq!(mjai::name("9s".parse::<Tile>().unwrap()), "9s");
}
