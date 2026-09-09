//! Settlement adapter for a real table. All scoring and transfers go through
//! Hand::act / Hand::resolve_calls / Hand::draw (only on an empty wall).
//! Unknown hands and wall tiles are never filled in or simulated.
use super::*;
use riichi_core::game::{Action, Error, Outcome};
use riichi_core::score::{FuReason, Score};

#[derive(Deserialize)]
struct Ending {
    position: Position,
    kind: String,
    #[serde(rename = "nextSeat")]
    next_seat: usize,
    #[serde(rename = "needsDraw")]
    needs_draw: bool,
    winners: Vec<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Input {
    winners: Vec<usize>,
    // Standing concealed tiles, EXCLUDING the winning tile. Known hands
    // cannot be replaced; the guide already has their authoritative tiles.
    hands: [Vec<String>; 4],
    winning_tile: Option<String>,
    ura: Vec<String>,
    tenpai: [bool; 4],
    confirmed_no_furiten: bool,
}

#[derive(Serialize, Debug)]
struct YakuLine {
    name: &'static str,
    han: u8,
    yakuman: bool,
}

#[derive(Serialize, Debug)]
struct WinResult {
    seat: usize,
    winning_tile: String,
    hand: Vec<String>,
    yaku: Vec<YakuLine>,
    han: u8,
    fu: u32,
    fu_detail: Vec<(String, u32)>,
    limit: Option<&'static str>,
    dora: u8,
    ura_dora: u8,
    indicators: Vec<String>,
    ura_indicators: Vec<String>,
    hand_payment: u32,
}

#[derive(Serialize, Debug)]
struct Settlement {
    kind: String,
    before: [i32; 4],
    after: [i32; 4],
    deltas: [i32; 4],
    sticks_before: u32,
    sticks_after: u32,
    repeat: bool,
    next_counters: u32,
    winners: Vec<WinResult>,
    tenpai: Vec<usize>,
    refunded_riichi: Option<usize>,
}

/// Computes an immutable settlement using the same rules as the browser game.
/// The caller persists and applies this result once, not on every render.
#[wasm_bindgen]
pub fn settle_physical(ending: JsValue, input: JsValue) -> Result<JsValue, JsValue> {
    let ending: Ending = serde_wasm_bindgen::from_value(ending).map_err(js_error)?;
    let input: Input = serde_wasm_bindgen::from_value(input).map_err(js_error)?;
    let result = settle(ending, input).map_err(js_error)?;
    result
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

fn settle(ending: Ending, input: Input) -> Result<Settlement, String> {
    let mut p = ending.position;
    wind(p.seat)?;
    wind(p.turn)?;
    wind(ending.next_seat)?;
    let draw = ending.kind == "draw";
    let tsumo = ending.kind == "tsumo";
    if !draw && !tsumo && ending.kind != "ron" {
        return Err("This hand end requires a manual table settlement".into());
    }
    let winners = &input.winners;
    if winners.iter().any(|seat| *seat >= 4)
        || winners
            .iter()
            .enumerate()
            .any(|(i, s)| winners[..i].contains(s))
        || ending.winners.iter().any(|s| !winners.contains(s))
        || (draw && !winners.is_empty())
        || (tsumo && winners.len() != 1)
        || (!draw && !tsumo && (winners.is_empty() || winners.len() > 3))
    {
        return Err("Choose each winning seat once, including the recorded winner".into());
    }
    let before = std::array::from_fn(|i| p.players[i].score);
    let sticks_before = p.riichi_sticks;
    let riichi_before = p.players.each_ref().map(|player| player.riichi != "none");
    if riichi_before.iter().filter(|r| **r).count() as u32 > sticks_before {
        return Err("The riichi pot is smaller than the recorded declarations".into());
    }

    let unknown_ron = !draw && !tsumo && winners.iter().any(|i| p.players[*i].hand.is_empty());
    for (i, tiles) in input.hands.iter().enumerate() {
        let needed = if draw {
            input.tenpai[i]
        } else {
            winners.contains(&i)
        };
        if !tiles.is_empty() {
            if !needed || !p.players[i].hand.is_empty() {
                return Err("Only reveal a previously unknown winning or tenpai hand".into());
            }
            if tiles.len() > 13 {
                return Err("Enter the standing hand without the winning tile or called sets".into());
            }
            p.players[i].hand = tiles.clone();
        }
        if needed && p.players[i].hand.is_empty() {
            return Err(format!("Reveal {}'s concealed tiles", seat_name(i)));
        }
    }

    if unknown_ron && !input.confirmed_no_furiten {
        return Err("Confirm that every ron winner is free of temporary and riichi furiten".into());
    }

    if tsumo {
        let winner = winners[0];
        if p.phase == "draw" {
            if winner != ending.next_seat || winner != p.turn || !ending.needs_draw || p.wall == 0 {
                return Err("Tsumo must be the current player's actual draw".into());
            }
            let winning = input
                .winning_tile
                .as_deref()
                .ok_or("Enter the winning drawn tile")?;
            tile(winning)?;
            p.players[winner].hand.push(winning.to_string());
            p.wall -= 1; // The previously unobserved live/replacement draw, once.
            p.drawn = Some(winning.to_string());
            p.phase = "act".into();
        } else if p.phase != "act" || p.turn != winner || p.drawn.is_none() {
            return Err("Tsumo needs the actual drawn tile, before discarding".into());
        } else if input
            .winning_tile
            .as_ref()
            .is_some_and(|t| Some(t) != p.drawn.as_ref())
        {
            return Err("The winning tile must be the recorded draw".into());
        }
        p.seat = winner;
    } else {
        if p.phase != "call" || p.pending.is_none() {
            return Err("Finish the actual discard or kan response before settling".into());
        }
        if input
            .winning_tile
            .as_ref()
            .is_some_and(|t| Some(t) != p.pending.as_ref())
        {
            return Err("The winning tile must be the pending discard or kan".into());
        }
        if draw {
            if p.wall != 0 || p.pending_kind != "discard" {
                return Err(
                    "An exhaustive draw needs an empty live wall and the final discard".into(),
                );
            }
            p.seat = (p.turn + 1) % 4;
        } else {
            if winners.contains(&p.turn) {
                return Err("The discarder cannot win their own discard".into());
            }
            p.seat = winners[0];
        }
    }

    let (mut hand, _) = p.build_selected(false)?;
    hand.bets_this_hand = riichi_before.map(u32::from);
    // Melds retain chronological call order and source. Reconstruct the
    // responsibility flags that Hand normally records as those calls happen.
    for player in &mut hand.players {
        let mut dragons = 0;
        let mut winds = 0;
        for meld in &player.melds {
            if !meld.kind.opens_hand() || !meld.is_triplet_or_quad() {
                continue;
            }
            let offset = match meld.from {
                ClaimedFrom::Right => 1,
                ClaimedFrom::Across => 2,
                ClaimedFrom::Left => 3,
                ClaimedFrom::SelfDrawn => continue,
            };
            let feeder = wind((player.seat.index() + offset) % 4)?;
            if meld.tile.is_dragon() {
                dragons += 1;
                if dragons == 3 {
                    player.liable_for_dragons = Some(feeder);
                }
            }
            if meld.tile.is_wind() {
                winds += 1;
                if winds == 4 {
                    player.liable_for_winds = Some(feeder);
                }
            }
        }
    }

    let reveal_ura = !draw && winners.iter().any(|i| riichi_before[*i]);
    if input.ura.len() != if reveal_ura { p.indicators.len() } else { 0 } {
        return Err(
            if reveal_ura {
                "Enter every revealed ura indicator, one for each dora indicator"
            } else {
                "Ura indicators are only revealed for a riichi winner"
            }
            .into(),
        );
    }
    if reveal_ura {
        let ura = input
            .ura
            .iter()
            .map(|t| tile(t))
            .collect::<Result<Vec<_>, _>>()?;
        // Include revealed ura in the physical copy budget. For multi-ron,
        // the pending tile is counted ONCE at its source, not per winner.
        let mut visible = TileSet::new();
        for player in &hand.players {
            for t in player.hand.tiles() {
                count_visible(&mut visible, t)?;
            }
            for meld in &player.melds {
                for t in meld.tiles() {
                    count_visible(&mut visible, t)?;
                }
            }
            for discard in player.discards.iter().filter(|d| !d.claimed) {
                count_visible(&mut visible, discard.tile)?;
            }
        }
        for t in hand
            .wall
            .dora_indicators()
            .into_iter()
            .chain(ura.iter().copied())
        {
            count_visible(&mut visible, t)?;
        }
        hand.wall = hand
            .wall
            .with_revealed_ura(&ura)
            .ok_or("Invalid ura indicators")?;
    }

    if draw {
        for (i, player) in hand.players.iter_mut().enumerate() {
            if player.has_riichi() && !input.tenpai[i] {
                return Err("A riichi player must reveal a valid tenpai hand".into());
            }
            if input.tenpai[i] && !player.is_tenpai() {
                return Err(format!("{}'s revealed hand is not tenpai", seat_name(i)));
            }
            // EMA permits a non-riichi player to declare noten. Clearing
            // only this private copy lets the core settle that declaration.
            if !input.tenpai[i] {
                player.hand = TileSet::new();
            }
        }
        hand.phase = Phase::Draw;
        if hand.draw() != Err(Error::Over) {
            return Err("The live wall must be empty; no unknown tile may be drawn".into());
        }
    } else if tsumo {
        hand.act(Action::Tsumo)
            .map_err(|_| "This is not a legal tsumo: check the winning hand and yaku")?;
    } else {
        let calls = winners
            .iter()
            .map(|i| Ok((wind(*i)?, Call::Ron)))
            .collect::<Result<Vec<_>, String>>()?;
        hand.resolve_calls(&calls)
            .map_err(|_| "This is not a legal ron: check the hand, yaku and furiten")?;
    }

    let (results, tenpai, repeat) = match &hand.outcome {
        Some(Outcome::Win { winners, discarder }) => (
            winners
                .iter()
                .map(|(seat, score)| win_result(&hand, *seat, score, discarder.is_some()))
                .collect(),
            Vec::new(),
            winners.iter().any(|(seat, _)| *seat == Wind::East),
        ),
        Some(Outcome::ExhaustiveDraw { tenpai }) => (
            Vec::new(),
            tenpai.iter().map(|s| s.index()).collect(),
            tenpai.contains(&Wind::East),
        ),
        None => return Err("The physical hand did not finish".into()),
    };
    let after = hand.scores();
    if after.iter().map(|s| i64::from(*s)).sum::<i64>() + i64::from(hand.riichi_sticks) * 1000
        != before.iter().map(|s| i64::from(*s)).sum::<i64>() + i64::from(sticks_before) * 1000
    {
        return Err("Settlement does not conserve points and the riichi pot".into());
    }
    Ok(Settlement {
        kind: ending.kind,
        before,
        after,
        deltas: std::array::from_fn(|i| after[i] - before[i]),
        sticks_before,
        sticks_after: hand.riichi_sticks,
        repeat,
        next_counters: if draw || repeat { p.counters + 1 } else { 0 },
        winners: results,
        tenpai,
        refunded_riichi: (0..4).find(|i| riichi_before[*i] && !hand.players[*i].has_riichi()),
    })
}

fn seat_name(index: usize) -> &'static str {
    ["East", "South", "West", "North"][index]
}

fn win_result(hand: &Hand, seat: Wind, score: &Score, ron: bool) -> WinResult {
    let player = &hand.players[seat.index()];
    let mut concealed: Vec<_> = player.hand.tiles().collect();
    if ron {
        concealed.push(score.winning_tile);
    }
    let mut all_tiles = concealed.clone();
    for meld in &player.melds {
        all_tiles.extend(meld.tiles());
    }
    let indicators = hand.wall.dora_indicators();
    let dora = if score.dora == 0 {
        0
    } else {
        indicators
            .iter()
            .map(|indicator| all_tiles.iter().filter(|t| **t == indicator.dora()).count() as u8)
            .sum()
    };
    WinResult {
        seat: seat.index(),
        winning_tile: score.winning_tile.to_string(),
        hand: concealed.iter().map(ToString::to_string).collect(),
        yaku: score
            .yaku
            .iter()
            .map(|(y, han)| YakuLine {
                name: y.name(),
                han: *han,
                yakuman: y.is_yakuman(),
            })
            .collect(),
        han: score.han,
        fu: score.fu,
        fu_detail: score
            .fu_detail
            .iter()
            .map(|(reason, fu)| {
                (
                    match reason {
                        FuReason::Base => "Base".into(),
                        FuReason::ConcealedRon => "Concealed ron".into(),
                        FuReason::SelfDraw => "Self-draw".into(),
                        FuReason::Set(t) => format!("Set of {t}"),
                        FuReason::ValuePair => "Value pair".into(),
                        FuReason::Wait => "Wait".into(),
                        FuReason::OpenPinfu => "Open pinfu".into(),
                    },
                    *fu,
                )
            })
            .collect(),
        limit: score.limit.map(|limit| limit.name()),
        dora,
        ura_dora: score.dora.saturating_sub(dora),
        indicators: indicators.iter().map(ToString::to_string).collect(),
        ura_indicators: if player.has_riichi() {
            hand.wall
                .ura_indicators()
                .iter()
                .map(ToString::to_string)
                .collect()
        } else {
            Vec::new()
        },
        hand_payment: score.payments.total,
    }
}
