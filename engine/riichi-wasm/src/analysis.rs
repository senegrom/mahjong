//! Read-only analysis of a manually entered, partially observed table.
//! Unknown opponents' hands stay empty; only the selected hand is required.
use super::{describe_action, describe_call};
use riichi_core::bot::{Bot, Style};
use riichi_core::encoding::{self, ACTIONS, OBSERVATION};
use riichi_core::game::{Call, Discard, Hand, Phase};
use riichi_core::hand::{ClaimedFrom, Meld, MeldKind, TileSet};
use riichi_core::rng::Rng;
use riichi_core::score::Riichi;
use riichi_core::tile::Tile;
use riichi_core::wall::Wall;
use riichi_core::Wind;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct Choice {
    index: Option<usize>,
    kind: String,
    tile: Option<String>,
    label: String,
    causes_furiten: bool,
}

pub(super) fn observation(hand: &Hand, seat: Wind) -> Vec<f32> {
    let mut out = vec![0.0; OBSERVATION];
    encoding::observe(hand, seat, &mut out);
    out
}

pub(super) fn mask(hand: &Hand, seat: Wind) -> Vec<u8> {
    let mut mask = vec![false; ACTIONS];
    encoding::legal_mask(hand, seat, &mut mask);
    // A physical player can always decline a discard even when no call is
    // available. The game driver normally skips such a decision entirely.
    if hand.phase == Phase::CallWindow && hand.turn != seat {
        mask[encoding::PASS] = true;
    }
    mask.into_iter().map(u8::from).collect()
}

fn choices(hand: &Hand, seat: Wind) -> Vec<Choice> {
    let mut choices: Vec<Choice> = mask(hand, seat)
        .into_iter()
        .enumerate()
        .filter(|(_, allowed)| *allowed != 0)
        .filter_map(|(index, _)| {
            let action = if hand.phase == Phase::CallWindow {
                if index == encoding::PASS {
                    Some(describe_call(Call::Pass))
                } else {
                    encoding::decode_call(hand, seat, index).map(describe_call)
                }
            } else {
                encoding::decode_action(hand, index).map(describe_action)
            }?;
            Some(Choice {
                index: Some(index),
                kind: action.kind,
                tile: action.tile,
                label: action.label,
                // Passing a complete shape also causes furiten without a
                // scoring yaku. Use the engine's waits, not the Ron button.
                causes_furiten: index == encoding::PASS
                    && hand.pending_discard.is_some_and(|(_, tile)| {
                        hand.players[seat.index()].waits().count(tile) > 0
                    }),
            })
        })
        .collect();
    // The trained action space names only the first kan of each kind. Keep
    // other legal kans available to a physical player and to built-in agents,
    // without inventing a separate network weight for them.
    if hand.phase == Phase::Act && hand.turn == seat {
        for action in hand.legal_actions().into_iter().map(describe_action) {
            if !choices
                .iter()
                .any(|c| c.kind == action.kind && c.tile == action.tile)
            {
                choices.push(Choice {
                    index: None,
                    kind: action.kind,
                    tile: action.tile,
                    label: action.label,
                    causes_furiten: false,
                });
            }
        }
    }
    choices
}

pub(super) fn choices_value(hand: &Hand, seat: Wind) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&choices(hand, seat)).map_err(js_error)
}

pub(super) fn pick_value(hand: &Hand, seat: Wind, bot: &mut Bot) -> Result<JsValue, JsValue> {
    let action = match hand.phase {
        Phase::Act if hand.turn == seat => describe_action(bot.act(hand)),
        Phase::CallWindow if hand.turn != seat => {
            let offered = hand.legal_calls().into_iter().find(|(who, _)| *who == seat);
            let call = offered.map_or(Call::Pass, |(_, calls)| bot.call(hand, seat, &calls));
            describe_call(call)
        }
        _ => return Err(JsValue::from_str("no decision for this seat")),
    };
    serde_wasm_bindgen::to_value(&action).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Position {
    seat: usize,
    turn: usize,
    phase: String,
    round: usize,
    kyoku: u8,
    counters: u32,
    riichi_sticks: u32,
    wall: usize,
    indicators: Vec<String>,
    players: [Seat; 4],
    drawn: Option<String>,
    pending: Option<String>,
    #[serde(default)]
    pending_kind: String,
    just_claimed: Option<String>,
    #[serde(default)]
    after_quad: bool,
    #[serde(default)]
    first_turns: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Seat {
    hand: Vec<String>,
    melds: Vec<Set>,
    discards: Vec<Thrown>,
    score: i32,
    riichi: String,
    ippatsu: bool,
    furiten: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Set {
    kind: String,
    tile: String,
    from: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Thrown {
    tile: String,
    order: u32,
    drawn: bool,
    riichi: bool,
    claimed: bool,
}

fn tile(value: &str) -> Result<Tile, String> {
    value.parse().map_err(|_| format!("Invalid tile: {value}"))
}

fn count_visible(seen: &mut TileSet, tile: Tile) -> Result<(), String> {
    if seen.count(tile) >= 4 {
        return Err(format!(
            "More than four copies of {tile}; mark claimed discards to avoid counting them twice"
        ));
    }
    seen.add(tile);
    Ok(())
}

fn wind(value: usize) -> Result<Wind, String> {
    [Wind::East, Wind::South, Wind::West, Wind::North]
        .get(value)
        .copied()
        .ok_or_else(|| "Invalid seat wind".into())
}

impl Position {
    fn build(&self) -> Result<(Hand, Wind), String> {
        let seat = wind(self.seat)?;
        let turn = wind(self.turn)?;
        let round = wind(self.round)?;
        if !(1..=4).contains(&self.kyoku) || self.counters > 100 || self.riichi_sticks > 100 {
            return Err("Invalid round, honba or riichi sticks".into());
        }
        let phase = match self.phase.as_str() {
            "act" if seat == turn => Phase::Act,
            "call" if seat != turn => Phase::CallWindow,
            _ => return Err("Select the acting seat, or another seat answering a discard".into()),
        };
        let scores = std::array::from_fn(|index| self.players[index].score);
        let mut hand = Hand::deal(
            &mut Rng::from_seed(0),
            round,
            self.kyoku,
            self.counters,
            self.riichi_sticks,
            scores,
        );
        hand.log.clear();
        hand.turn = turn;
        hand.phase = phase;
        hand.drawn = self.drawn.as_deref().map(tile).transpose()?;
        hand.just_claimed = self.just_claimed.as_deref().map(tile).transpose()?;
        hand.after_quad = self.after_quad;
        hand.first_turns_unbroken = self.first_turns;
        hand.pending_discard = self
            .pending
            .as_deref()
            .map(tile)
            .transpose()?
            .map(|t| (turn, t));
        let robbing = matches!(self.pending_kind.as_str(), "extended-kan" | "concealed-kan");
        hand.robbable_quad = if robbing {
            hand.pending_discard.map(|(_, t)| t)
        } else {
            None
        };
        hand.robbing_concealed = self.pending_kind == "concealed-kan";
        hand.discards_made = 0;
        if !matches!(
            self.pending_kind.as_str(),
            "" | "discard" | "extended-kan" | "concealed-kan"
        ) {
            return Err("Invalid pending call type".into());
        }
        if (phase == Phase::CallWindow) != hand.pending_discard.is_some()
            || (phase == Phase::CallWindow && (hand.drawn.is_some() || hand.just_claimed.is_some()))
        {
            return Err(
                "A call needs a pending tile; clear the drawn and just-claimed tiles".into(),
            );
        }
        let mut visible = TileSet::new();
        let mut orders = std::collections::HashSet::new();
        let mut quads = 0usize;
        for (index, input) in self.players.iter().enumerate() {
            if input.hand.len() > 14
                || input.melds.len() > 4
                || input.discards.len() > 100
                || !(-100_000..=200_000).contains(&input.score)
            {
                return Err(format!(
                    "Seat {} has too many tiles, sets or discards, or an invalid score",
                    index + 1
                ));
            }
            let player = &mut hand.players[index];
            player.hand = TileSet::new();
            for value in &input.hand {
                let t = tile(value)?;
                player.hand.add(t);
                count_visible(&mut visible, t)?;
            }
            player.melds.clear();
            for set in &input.melds {
                let t = tile(&set.tile)?;
                let from = match set.from {
                    0 => ClaimedFrom::SelfDrawn,
                    1 => ClaimedFrom::Right,
                    2 => ClaimedFrom::Across,
                    3 => ClaimedFrom::Left,
                    _ => return Err("Invalid call direction".into()),
                };
                let kind = match set.kind.as_str() {
                    "chii" if t.rank() <= 7 && !t.is_honour() && set.from == 3 => MeldKind::Chii,
                    "pon" => MeldKind::Pon,
                    "kan" => MeldKind::ClaimedKan,
                    "extended-kan" => MeldKind::ExtendedKan,
                    "concealed-kan" => MeldKind::ConcealedKan,
                    _ => {
                        return Err(
                            "Invalid meld: chii must be a suited sequence from the left".into()
                        )
                    }
                };
                if (kind == MeldKind::ConcealedKan) != (set.from == 0) {
                    return Err("Choose who supplied the call, or Self for a concealed kan".into());
                }
                let meld = Meld {
                    kind,
                    tile: t,
                    from,
                };
                for t in meld.tiles() {
                    count_visible(&mut visible, t)?;
                }
                quads += usize::from(kind.is_kan());
                player.melds.push(meld);
            }
            let expected = 13 - 3 * player.melds.len()
                + usize::from(phase == Phase::Act && index == self.turn);
            if (index == self.seat || !input.hand.is_empty()) && player.hand.len() != expected {
                return Err(format!(
                    "{} needs {expected} concealed tiles, including any drawn tile",
                    ["East", "South", "West", "North"][index]
                ));
            }
            player.discards.clear();
            for entry in &input.discards {
                if entry.order >= 400 || !orders.insert(entry.order) {
                    return Err("Discard order numbers must be unique and below 400".into());
                }
                let t = tile(&entry.tile)?;
                if !entry.claimed {
                    count_visible(&mut visible, t)?;
                }
                hand.discards_made = hand.discards_made.max(entry.order + 1);
                player.discards.push(Discard {
                    tile: t,
                    order: entry.order,
                    drawn: entry.drawn,
                    riichi: entry.riichi,
                    claimed: entry.claimed,
                });
            }
            player.discards.sort_by_key(|entry| entry.order);
            player.riichi = match input.riichi.as_str() {
                "none" => Riichi::None,
                "riichi" => Riichi::Declared,
                "double" => Riichi::Double,
                _ => return Err("Invalid riichi declaration".into()),
            };
            let declarations: Vec<_> = player.discards.iter().filter(|d| d.riichi).collect();
            if declarations.len() != usize::from(player.has_riichi())
                || (player.has_riichi() && !player.is_concealed())
            {
                return Err(
                    "Riichi requires a closed hand and exactly one marked declaration discard"
                        .into(),
                );
            }
            player.riichi_order = declarations.first().map(|d| d.order);
            if input.ippatsu && !player.has_riichi() {
                return Err("Ippatsu requires riichi".into());
            }
            player.ippatsu = input.ippatsu;
            player.furiten = false;
            player.temporary_furiten = input.furiten;
            // Permanent furiten is evaluated on the shape before the draw.
            let mut waiting = player.clone();
            if index == self.turn {
                if let Some(t) = hand.drawn {
                    waiting.hand.remove(t);
                }
            }
            waiting.refresh_furiten();
            if player.has_riichi() && !input.hand.is_empty() && !waiting.is_tenpai() {
                return Err("A declared riichi hand must still be waiting before the draw".into());
            }
            player.furiten = waiting.furiten;
        }
        if phase == Phase::Act {
            if let Some(t) = hand.drawn {
                if hand.current().hand.count(t) == 0 {
                    return Err("The drawn tile must be included in the acting hand".into());
                }
            } else if hand.just_claimed.is_none() {
                return Err("Choose the drawn tile, or the tile just claimed for a set".into());
            }
            if hand.drawn.is_some() && hand.just_claimed.is_some() {
                return Err("A turn starts with either a draw or a call".into());
            }
            if let Some(t) = hand.just_claimed {
                if !hand.current().melds.last().is_some_and(|m| {
                    matches!(m.kind, MeldKind::Chii | MeldKind::Pon) && m.tiles().contains(&t)
                }) {
                    return Err("The just-claimed tile must belong to a chii or pon".into());
                }
            }
        } else if let Some((_, t)) = hand.pending_discard {
            if robbing {
                let expected_kind = if hand.robbing_concealed {
                    MeldKind::ConcealedKan
                } else {
                    MeldKind::ExtendedKan
                };
                if !hand.players[self.turn]
                    .melds
                    .iter()
                    .any(|m| m.kind == expected_kind && m.tile == t)
                {
                    return Err("The pending kan must match its tile and kind in the declaring seat's called sets".into());
                }
            } else if !hand.players[self.turn]
                .discards
                .last()
                .is_some_and(|d| d.tile == t && !d.claimed && d.order + 1 == hand.discards_made)
            {
                return Err("The pending tile must be the most recent unclaimed discard".into());
            }
        }
        let indicators = self
            .indicators
            .iter()
            .map(|value| tile(value))
            .collect::<Result<Vec<_>, _>>()?;
        if quads > 4 {
            return Err("At most four kans can be declared in a hand".into());
        }
        // Before a kan's robbery window closes, its new indicator is not yet
        // exposed and no replacement tile has been taken.
        let completed_quads =
            quads.saturating_sub(usize::from(robbing && phase == Phase::CallWindow));
        hand.wall =
            Wall::for_analysis(self.wall, &indicators, completed_quads).ok_or_else(|| {
                "Enter 0–70 wall tiles and one dora indicator plus one per completed kan"
                    .to_string()
            })?;
        for t in indicators {
            count_visible(&mut visible, t)?;
        }
        Ok((hand, seat))
    }
}

/// Owns an immutable validated position. It cannot draw random tiles or play
/// opponents; the physical table remains the authority for all updates.
#[wasm_bindgen]
pub struct PhysicalAnalysis {
    hand: Hand,
    seat: Wind,
}

#[wasm_bindgen]
impl PhysicalAnalysis {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsValue) -> Result<PhysicalAnalysis, JsValue> {
        let position: Position = serde_wasm_bindgen::from_value(value).map_err(js_error)?;
        let (hand, seat) = position.build().map_err(js_error)?;
        Ok(Self { hand, seat })
    }
    pub fn agent_choices(&self) -> Result<JsValue, JsValue> {
        choices_value(&self.hand, self.seat)
    }
    pub fn agent_observation(&self) -> Vec<f32> {
        observation(&self.hand, self.seat)
    }
    pub fn agent_mask(&self) -> Vec<u8> {
        mask(&self.hand, self.seat)
    }
    pub fn agent_pick(&self, kind: &str) -> Result<JsValue, JsValue> {
        let style = match kind {
            "beginner" => Style::beginner(),
            "club" => Style::club(),
            _ => return Err(JsValue::from_str("choose a built-in agent")),
        };
        pick_value(
            &self.hand,
            self.seat,
            &mut Bot::with_style(self.hand.discards_made as u64, style),
        )
    }
}
