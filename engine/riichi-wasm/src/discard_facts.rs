//! The engine's facts about each discard a position offers its seat, which
//! the game review reads to say why Trained AI chose the tile it did, and
//! to say what an edited table changed.
//!
//! For every tile the seat may throw: how far it leaves the hand from
//! waiting, by each hand form and overall; the unseen tiles that would then
//! bring the hand closer; how exposed the tile is against each declared
//! riichi; whether the seat has thrown that kind before, so that a wait on
//! it would be furiten; how many copies the seat cannot see; and what the
//! tile is worth as dora or as a valued honour. For the hand: the dora, how
//! many the seat holds, its valued pairs, and what is safe against each
//! riichi.
//!
//! The distances and improving tiles are counted as the review's notes
//! count them (`riichi_core::review`), against what the seat can see
//! (`Hand::seen_by`), so a table edited in the browser is measured exactly
//! as a played one is. The exposure tiers are the facts study's: a tile
//! safe against the seat (`Hand::safe_against`) first, then honours by how
//! many other copies are in sight, then suit tiles by their suji, ordered
//! by the commonly quoted rates at which such tiles deal into a riichi in
//! the middle of a hand. The names are the study's, which the review's
//! wording is keyed by.

use super::*;
use riichi_core::game::Action;
use riichi_core::shanten;
use riichi_core::tile::COPIES;

/// What the engine knows about a position's discards. A field the page never
/// reads is kept for the engine's own checks and left out of what
/// `review_facts` hands over: the search reads the facts of a few hundred
/// tables, and writing out what nobody read cost each a fifth of its time.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub(super) struct Facts {
    /// How far the hand is from waiting as it stands, before anything is
    /// thrown: zero is waiting, minus one complete. Not handed over.
    #[serde(skip)]
    shanten: i32,
    /// The dora, one for each indicator, so that a kind two indicators
    /// point at is named twice. Not handed over: the page reads each
    /// discard's own.
    #[serde(skip)]
    dora: Vec<String>,
    /// How many han of dora the seat holds, in hand and in its sets.
    dora_held: u32,
    /// The valued honours the seat holds two or more of in hand, each a
    /// yaku once it is a triplet: a dragon, the round's wind, its own seat's.
    valued_pairs: Vec<String>,
    /// Each other seat that has declared riichi, in seat order, with the
    /// tiles that cannot deal into it.
    threats: Vec<Threat>,
    /// Each tile the seat may throw, in tile order. None while the seat is
    /// answering another seat's tile rather than acting.
    discards: Vec<DiscardFacts>,
}

/// A seat that has declared riichi, and what cannot deal into it.
#[derive(Serialize, Debug, PartialEq, Eq)]
struct Threat {
    seat: usize,
    safe: Vec<String>,
}

/// What throwing one tile does.
#[derive(Serialize, Debug, PartialEq, Eq)]
struct DiscardFacts {
    tile: String,
    /// How far the hand is left from waiting, by the nearest form.
    shanten: i32,
    /// The same by each form.
    forms: Forms,
    /// The kinds that would then bring the hand closer, among those the
    /// seat cannot see every copy of. Not handed over: the page reads only
    /// how many copies of them are left (`acceptance`).
    #[serde(skip)]
    improving: Vec<String>,
    /// How many copies of those kinds the seat cannot see.
    acceptance: u32,
    /// How exposed the tile is against each threat, in the same order.
    tiers: Vec<Exposure>,
    /// Whether the seat has thrown this kind already, claimed or not.
    own_river: bool,
    /// How many copies the seat cannot see.
    unseen: u8,
    /// How many copies the seat holds, the tile just drawn included.
    held: u8,
    /// How many han it is worth as dora.
    dora: u8,
    /// Whether it is a valued honour.
    valued: bool,
}

/// How far a hand is from waiting by each form. Seven pairs and thirteen
/// orphans are concealed hands, so a seat with a set has neither.
#[derive(Serialize, Debug, PartialEq, Eq)]
struct Forms {
    regular: i32,
    seven_pairs: Option<i32>,
    thirteen_orphans: Option<i32>,
}

/// How exposed a tile is against one seat in riichi.
#[derive(Serialize, Debug, PartialEq, Eq)]
struct Exposure {
    seat: usize,
    tier: &'static str,
}

/// The engine's facts about every discard a typed-in position offers its
/// seat, for the game review's explanations: see the module's note. Refused,
/// with the validation's reason, for a position the rules refuse.
#[wasm_bindgen]
pub fn review_facts(position: JsValue) -> Result<JsValue, JsValue> {
    let position: Position = serde_wasm_bindgen::from_value(position).map_err(js_error)?;
    let (hand, seat) = position.build().map_err(js_error)?;
    facts(&hand, seat)
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// What `review_facts` answers, before it is handed to the page.
pub(super) fn facts(hand: &Hand, seat: Wind) -> Facts {
    let player = &hand.players[seat.index()];
    let called = player.melds.len();
    let seen = hand.seen_by(seat);
    let doras: Vec<Tile> = hand
        .wall
        .dora_indicators()
        .iter()
        .map(|indicator| indicator.dora())
        .collect();
    let valued = |tile: Tile| tile.is_dragon() || tile == hand.round.tile() || tile == seat.tile();
    let threats: Vec<(Wind, TileSet)> = Wind::ALL
        .into_iter()
        .filter(|other| *other != seat && hand.players[other.index()].has_riichi())
        .map(|other| (other, hand.safe_against(other)))
        .collect();
    // Only the seat to act throws anything: from its hand, less any tile
    // swap-calling bars, and in riichi only the tile it drew.
    let mut thrown: Vec<Tile> = if hand.phase == Phase::Act && hand.turn == seat {
        hand.legal_actions()
            .into_iter()
            .filter_map(|action| match action {
                Action::Discard(tile) | Action::Riichi(tile) => Some(tile),
                _ => None,
            })
            .collect()
    } else {
        Vec::new()
    };
    thrown.sort();
    thrown.dedup();

    let discards = thrown
        .into_iter()
        .map(|tile| {
            let mut left = player.hand;
            left.remove(tile);
            let improving: Vec<Tile> = shanten::acceptance(&left, called, &seen)
                .into_iter()
                .map(|(kind, _)| kind)
                .collect();
            let concealed = called == 0;
            DiscardFacts {
                tile: tile.to_string(),
                shanten: shanten::shanten(&left, called),
                forms: Forms {
                    regular: shanten::standard(&left, called),
                    seven_pairs: concealed.then(|| shanten::seven_pairs(&left, called)),
                    thirteen_orphans: concealed.then(|| shanten::thirteen_orphans(&left, called)),
                },
                acceptance: improving
                    .iter()
                    .map(|kind| u32::from(COPIES.saturating_sub(seen.count(*kind))))
                    .sum(),
                improving: names(improving.iter().copied()),
                tiers: threats
                    .iter()
                    .map(|(other, safe)| Exposure {
                        seat: other.index(),
                        tier: tier(tile, safe, &seen),
                    })
                    .collect(),
                own_river: player.discards.iter().any(|discard| discard.tile == tile),
                unseen: COPIES.saturating_sub(seen.count(tile)),
                held: player.hand.count(tile),
                dora: doras.iter().filter(|dora| **dora == tile).count() as u8,
                valued: valued(tile),
            }
        })
        .collect();

    let own = player.visible_to_self();
    Facts {
        shanten: shanten::shanten(&player.hand, called),
        dora: names(doras.iter().copied()),
        dora_held: doras.iter().map(|dora| u32::from(own.count(*dora))).sum(),
        valued_pairs: names(
            Tile::all().filter(|tile| valued(*tile) && player.hand.count(*tile) >= 2),
        ),
        threats: threats
            .iter()
            .map(|(other, safe)| Threat {
                seat: other.index(),
                safe: names(Tile::all().filter(|tile| safe.count(*tile) > 0)),
            })
            .collect(),
        discards,
    }
}

/// Tiles as the page names them, "7m" or "5z".
fn names(tiles: impl Iterator<Item = Tile>) -> Vec<String> {
    tiles.map(|tile| tile.to_string()).collect()
}

/// How exposed `tile` is against a seat in riichi that `safe` cannot deal
/// into, `seen` counting every copy the deciding seat can see, its own in
/// hand among them. An honour is as live as the copies still out of sight;
/// a suit tile is on suji when the tile three away is safe (a four for a
/// one, a five for a two or an eight, a six for a three), and a four, five
/// or six needs both sides for full suji.
fn tier(tile: Tile, safe: &TileSet, seen: &TileSet) -> &'static str {
    if safe.count(tile) > 0 {
        return "safe";
    }
    if tile.is_honour() {
        let others = seen.count(tile).saturating_sub(1).min(3);
        return [
            "honour, none seen",
            "honour, 1 seen",
            "honour, 2 seen",
            "honour, 3 seen",
        ][usize::from(others)];
    }
    let rank = tile.rank();
    let suji = |other: u8| safe.count(Tile::numbered(tile.suit(), other)) > 0;
    let low = rank > 3 && suji(rank - 3);
    let high = rank + 3 <= 9 && suji(rank + 3);
    match rank {
        1 | 9 if low || high => "suji 1/9",
        1 | 9 => "no-suji 1/9",
        2 | 8 if low || high => "suji 2/8",
        2 | 8 => "no-suji 2/8",
        3 | 7 if low || high => "suji 3-7",
        3 | 7 => "no-suji 3/7",
        _ if low && high => "suji 3-7",
        _ if low || high => "half-suji 4-6",
        _ => "no-suji 4-6",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// East to act on a drawn red dragon in the east round, three turns in,
    /// with North in riichi since its 5 bamboo. North has thrown the 4
    /// characters and the 6 characters; since its declaration East has let
    /// the west wind go, South the 9 circles and West the 2 characters, so
    /// all of those are safe against it. The dora is the 4 bamboo, and East
    /// has already thrown a 3 circles.
    fn east_against_north() -> (Hand, Wind) {
        let thrown = |tile: &str, order: u32| json!({"tile": tile, "order": order, "drawn": false, "riichi": order == 7, "claimed": false});
        let river = |entries: &[(&str, u32)]| -> Vec<_> {
            entries
                .iter()
                .map(|(tile, order)| thrown(tile, *order))
                .collect()
        };
        let seat = |hand: Vec<&str>, discards: Vec<serde_json::Value>, score: i32, riichi: &str| {
            json!({"hand": hand, "melds": [], "discards": discards, "score": score,
                "riichi": riichi, "ippatsu": false, "furiten": false})
        };
        let position: Position = serde_json::from_value(json!({
            "seat": 0, "turn": 0, "phase": "act", "round": 0, "kyoku": 1, "counters": 0,
            "riichi_sticks": 1, "wall": 57, "indicators": ["3s"],
            "players": [
                seat(
                    vec!["1m", "5m", "7m", "3p", "6p", "9p", "2s", "4s", "4s", "8s", "1z", "1z", "5z", "7z"],
                    river(&[("3p", 0), ("2z", 4), ("3z", 8)]),
                    25000,
                    "none",
                ),
                seat(vec![], river(&[("1z", 1), ("9m", 5), ("9p", 9)]), 25000, "none"),
                seat(vec![], river(&[("7z", 2), ("8p", 6), ("2m", 10)]), 25000, "none"),
                seat(vec![], river(&[("4m", 3), ("5s", 7), ("6m", 11)]), 24000, "riichi"),
            ],
            "drawn": "7z", "pending": null, "pending_kind": "discard", "just_claimed": null,
            "after_quad": false, "first_turns": false
        }))
        .unwrap();
        position.build().expect("a valid position")
    }

    fn discard<'a>(facts: &'a Facts, tile: &str) -> &'a DiscardFacts {
        facts
            .discards
            .iter()
            .find(|discard| discard.tile == tile)
            .unwrap_or_else(|| panic!("no discard of {tile}"))
    }

    /// Against North's riichi: what it has thrown, and what anyone threw
    /// after its declaration, cannot deal in. A suit tile is on suji when
    /// the tile three away is safe, a four to six only half on one side;
    /// an honour is as live as the copies still out of sight, East's own
    /// among those in sight.
    #[test]
    fn a_tiles_exposure_follows_the_safe_tiles_and_the_copies_in_sight() {
        let (hand, seat) = east_against_north();
        let facts = facts(&hand, seat);
        assert_eq!(
            facts.threats,
            [Threat {
                seat: 3,
                safe: ["2m", "4m", "6m", "9p", "5s", "3z"]
                    .map(String::from)
                    .to_vec(),
            }]
        );
        let tiers: Vec<(&str, &str)> = facts
            .discards
            .iter()
            .map(|discard| (discard.tile.as_str(), discard.tiers[0].tier))
            .collect();
        assert_eq!(
            tiers,
            [
                ("1m", "suji 1/9"),
                ("5m", "half-suji 4-6"),
                ("7m", "suji 3-7"),
                ("3p", "no-suji 3/7"),
                ("6p", "half-suji 4-6"),
                ("9p", "safe"),
                ("2s", "suji 2/8"),
                ("4s", "no-suji 4-6"),
                ("8s", "suji 2/8"),
                ("1z", "honour, 2 seen"),
                ("5z", "honour, none seen"),
                ("7z", "honour, 1 seen"),
            ]
        );
        assert!(facts.discards.iter().all(|d| d.tiers[0].seat == 3));
    }

    /// What a tile is worth and how many are left: the 4 bamboo is the
    /// dora, held twice; the east wind is East's seat and the round's, held
    /// as a pair, and the dragons are valued too; East has thrown a 3
    /// circles before, and of the 3 circles two are in sight.
    #[test]
    fn a_tiles_worth_and_copies_are_counted_from_the_seats_side() {
        let (hand, seat) = east_against_north();
        let facts = facts(&hand, seat);
        assert_eq!(facts.dora, ["4s"]);
        assert_eq!(facts.dora_held, 2);
        assert_eq!(facts.valued_pairs, ["1z"]);
        let four = discard(&facts, "4s");
        assert_eq!((four.dora, four.held, four.valued), (1, 2, false));
        let east = discard(&facts, "1z");
        assert_eq!((east.held, east.valued, east.unseen), (2, true, 1));
        assert!(discard(&facts, "7z").valued && discard(&facts, "5z").valued);
        let three = discard(&facts, "3p");
        assert_eq!((three.own_river, three.unseen), (true, 2));
        assert!(facts.discards.iter().filter(|d| d.own_river).count() == 1);
    }

    /// The distance and the improving tiles are the review note's own, for
    /// every tile a player could have thrown at every turn of whole games,
    /// in every hand of them; a tile is "safe" against every riichi exactly
    /// where the note calls it safe; its dora is the note's; and the facts
    /// read off the position written for a decision are the facts of the
    /// decision's own hand. The games' players call every set and declare
    /// every quad, or play as the club player does, so the turns include
    /// open hands, which have no seven pairs or thirteen orphans; quads;
    /// a seat in riichi, which throws only the tile it drew; and a turn
    /// just after a claim, which swap-calling bars a tile or two of.
    #[test]
    fn the_facts_agree_with_the_reviews_own_numbers_at_every_turn() {
        use super::review_position::tests::{whole_game, Plays};
        #[derive(Default, Debug)]
        struct Reached {
            turns: usize,
            tiles: usize,
            against_riichi: usize,
            open: usize,
            quads: usize,
            in_riichi: usize,
            after_a_claim: usize,
            beyond_the_first_hand: usize,
        }
        let mut reached = Reached::default();
        let games = [
            (31.0, Plays::Calls),
            (81.0, Plays::Calls),
            (5.0, Plays::Club),
            (16.0, Plays::Club),
        ];
        for (seed, plays) in games {
            whole_game(seed, "club", plays, |game| {
                for (index, decision) in game.decisions.iter().enumerate() {
                    let at = format!(
                        "seed {seed}, hand {}, decision {index}",
                        game.table.hands_played
                    );
                    let (own, seat) = (&decision.position, decision.seat);
                    let facts = facts(own, seat);
                    let position = review_position::written(own, seat).unwrap();
                    let (hand, written_seat) = position.build().unwrap();
                    assert_eq!(facts, super::facts(&hand, written_seat), "{at}");
                    if own.phase != Phase::Act {
                        assert!(facts.discards.is_empty(), "{at}: nothing to throw");
                        continue;
                    }
                    let player = &own.players[seat.index()];
                    let called = !player.melds.is_empty();
                    for discard in &facts.discards {
                        assert_eq!(
                            discard.forms.seven_pairs.is_none()
                                && discard.forms.thirteen_orphans.is_none(),
                            called,
                            "{at}: a hand with a set has neither concealed form"
                        );
                    }
                    if player.has_riichi() {
                        let drawn = own.drawn.map(|tile| tile.to_string());
                        let thrown: Vec<String> =
                            facts.discards.iter().map(|d| d.tile.clone()).collect();
                        assert_eq!(
                            thrown,
                            Vec::from_iter(drawn),
                            "{at}: riichi throws its draw"
                        );
                    }
                    let barred = own.forbidden_discards();
                    assert!(
                        facts
                            .discards
                            .iter()
                            .all(|d| !barred.contains(&d.tile.parse().unwrap())),
                        "{at}: a barred tile is offered"
                    );
                    for discard in &facts.discards {
                        let tile: Tile = discard.tile.parse().unwrap();
                        let note = game
                            .measure_at(index, Action::Discard(tile))
                            .unwrap()
                            .expect("a turn has a note");
                        assert_eq!(
                            (discard.shanten, discard.acceptance, discard.dora),
                            (note.shanten, note.acceptance, note.dora),
                            "{at}: tile {tile}"
                        );
                        let safe = !discard.tiers.is_empty()
                            && discard.tiers.iter().all(|exposure| exposure.tier == "safe");
                        assert_eq!(safe, note.danger == "safe", "{at}: tile {tile}");
                        reached.tiles += 1;
                        reached.against_riichi += usize::from(!discard.tiers.is_empty());
                    }
                    reached.turns += 1;
                    reached.open += usize::from(!player.is_concealed());
                    reached.quads += usize::from(player.melds.iter().any(|m| m.kind.is_kan()));
                    reached.in_riichi += usize::from(player.has_riichi());
                    reached.after_a_claim += usize::from(own.just_claimed.is_some());
                    reached.beyond_the_first_hand +=
                        usize::from((own.round, own.kyoku) != (Wind::East, 1));
                }
            });
        }
        println!("{reached:?}");
        assert!(reached.tiles > 1000, "{reached:?}");
        assert!(reached.against_riichi > 50, "{reached:?}");
        assert!(reached.open > 100, "{reached:?}");
        assert!(reached.quads > 0, "{reached:?}");
        assert!(reached.in_riichi > 0, "{reached:?}");
        assert!(reached.after_a_claim > 20, "{reached:?}");
        assert!(reached.beyond_the_first_hand > 100, "{reached:?}");
    }
}
