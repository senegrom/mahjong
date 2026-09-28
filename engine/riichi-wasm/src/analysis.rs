//! Read-only analysis of a manually entered, partially observed table.
//! Unknown opponents' hands stay empty; only the selected hand is required.
#[path = "physical_settlement.rs"]
mod settlement;

use super::{describe_action, describe_call, mortal_log};
use riichi::mjai::Event as MortalEvent;
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

/// Our moves that seat may make at this decision: the engine's own mask,
/// and a pass for every seat answering another seat's discard or quad.
///
/// A physical player can always decline a discard even when no call is
/// available. The game driver normally skips such a decision entirely, but a
/// guided or physical table stops at every discard. The listed choices, our
/// mask, Mortal's mask and the translation of Mortal's moves back all start
/// from this one answer, so no adviser is offered a decision it cannot make.
pub(super) fn legal_moves(hand: &Hand, seat: Wind) -> Vec<bool> {
    let mut mask = vec![false; ACTIONS];
    encoding::legal_mask(hand, seat, &mut mask);
    if hand.phase == Phase::CallWindow && hand.turn != seat {
        mask[encoding::PASS] = true;
    }
    mask
}

pub(super) fn mask(hand: &Hand, seat: Wind) -> Vec<u8> {
    legal_moves(hand, seat).into_iter().map(u8::from).collect()
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
                        let player = &hand.players[seat.index()];
                        player.waits().count(tile) > 0 && hand.could_rob_with(seat, tile)
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

// Unknown keys are dropped, not refused: the serde bridge to JavaScript does
// not honour `deny_unknown_fields`, and a draft saved by an older page may
// carry fields this version no longer reads.
#[derive(Deserialize)]
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
struct Set {
    kind: String,
    tile: String,
    from: usize,
}

#[derive(Deserialize)]
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
        return Err(format!("More than four copies of {tile} are visible"));
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
        self.build_selected(true)
    }

    // Settlement may select an unknown seat to validate the final discard.
    // Every supplied hand is still checked; winning/tenpai hands are required
    // separately by the settlement adapter. Advice always requires its hand.
    fn build_selected(&self, require_selected_hand: bool) -> Result<(Hand, Wind), String> {
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
            if input.riichi != "none"
                && player
                    .melds
                    .iter()
                    .any(|m| m.kind != MeldKind::ConcealedKan)
            {
                return Err(
                    "Riichi requires a closed hand: only concealed kans may stand beside it".into(),
                );
            }
            let expected = 13 - 3 * player.melds.len()
                + usize::from(phase == Phase::Act && index == self.turn);
            if ((index == self.seat && require_selected_hand) || !input.hand.is_empty())
                && player.hand.len() != expected
            {
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
        // Every discard marked as claimed must be explained by a called set
        // that took it from that seat; without this, marking extra copies of
        // a tile as claimed put a fifth one on the table.
        if let Err((seat, tile)) = claims(&hand) {
            return Err(format!(
                "{}'s claimed {} needs a called set that took it",
                ["East", "South", "West", "North"][seat.index()],
                tile
            ));
        }
        // The editor keeps the flag through the kan's robbery window, so the
        // replacement that follows is known to be one; only a plain discard
        // window cannot be waiting for a replacement.
        if hand.after_quad && phase != Phase::Act && !robbing {
            return Err("A replacement draw belongs to the seat that declared the kan".into());
        }
        if phase == Phase::Act {
            if hand.after_quad
                && (hand.drawn.is_none() || !hand.current().melds.iter().any(|m| m.kind.is_kan()))
            {
                return Err(
                    "A replacement draw needs a kan in the acting seat's sets and a drawn tile"
                        .into(),
                );
            }
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
                // Swap-calling bars the claimed tile, and for a sequence the
                // tile at its other side (EMA section 3.3.2). A call that
                // leaves nothing else to discard is never offered, so no
                // game reaches this hand, and the engine offers it no move.
                let barred = hand.forbidden_discards();
                let held = &hand.current().hand;
                if !held.is_empty() && held.tiles().all(|tile| barred.contains(&tile)) {
                    return Err(
                        "Swap-calling bars every tile left in this hand, so that call could not have been made"
                            .into(),
                    );
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
        if self.wall + completed_quads > 69 {
            return Err(
                "The live wall holds at most 69 tiles once the dealer has drawn, one fewer for each kan"
                    .into(),
            );
        }
        hand.wall =
            Wall::for_analysis(self.wall, &indicators, completed_quads).ok_or_else(|| {
                "Enter 0–69 wall tiles and one dora indicator plus one per completed kan"
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

    /// This seat's observation as Mortal builds it, rebuilt from the
    /// position by replaying what it says happened.
    pub fn agent_observation_mortal(&self) -> Result<Vec<f32>, JsValue> {
        self.planes(false)
    }

    /// The same seat once a reach is declared, for the second question a
    /// declaration asks. Nothing is declared: it is put to a copy.
    pub fn agent_observation_after_reach(&self) -> Result<Vec<f32>, JsValue> {
        self.planes(true)
    }

    /// Which of Mortal's moves this seat may make, by our rules.
    pub fn agent_mask_mortal(&self) -> Vec<u8> {
        super::mortal_mask_of(&self.hand, self.seat, false)
            .iter()
            .map(|flag| u8::from(*flag))
            .collect()
    }

    /// Which tiles a declaration may discard, in Mortal's numbering.
    pub fn agent_mask_after_reach(&self) -> Vec<u8> {
        super::mortal_mask_of(&self.hand, self.seat, true)
            .iter()
            .map(|flag| u8::from(*flag))
            .collect()
    }

    pub fn agent_action_from_mortal(&self, action: usize, after_reach: bool) -> i32 {
        super::action_from_mortal(&self.hand, self.seat, action, after_reach)
    }

    pub fn mortal_action_of(&self, ours: usize) -> i32 {
        super::mortal_action_for(ours)
    }

    /// How many concealed tiles each of the three other seats holds, in the
    /// order the belief head answers in: the next player, the one across,
    /// then the previous. A guessed hand is a share of these.
    pub fn concealed_counts(&self) -> Vec<u32> {
        concealed_counts(&self.hand, self.seat)
    }
}

impl PhysicalAnalysis {
    fn planes(&self, after_reach: bool) -> Result<Vec<f32>, JsValue> {
        let mut state = mortal_log::state_for(&self.hand, self.seat).ok_or_else(|| {
            JsValue::from_str(
                "This position cannot be replayed as a hand: check the discard order numbers, the called sets and the wall count",
            )
        })?;
        if after_reach && super::may_reach_from(&self.hand, self.seat) {
            // Put to a copy of the state, never to a game: the declaration
            // itself is recorded only when the move is actually made.
            let _ = state.update(&MortalEvent::Reach {
                actor: self.seat.index() as u8,
            });
        }
        let (observation, _mask) = state.encode_obs(super::MORTAL_VERSION, false);
        Ok(observation.iter().copied().collect())
    }
}

/// The three other seats' concealed tile counts, from the seat asked
/// outwards. The drawn tile counts: it is in the hand until it is let go,
/// and that is how the hands the belief head was trained against were
/// counted too.
///
/// A typed-in table leaves the hands nobody has shown empty, but they are
/// hidden, not empty: every seat holds thirteen tiles less three for each
/// set it has called, and one more while it is the seat to act.
pub(super) fn concealed_counts(hand: &Hand, seat: Wind) -> Vec<u32> {
    (1..4)
        .map(|offset| {
            let other = seat.plus(offset);
            let player = &hand.players[other.index()];
            let held = if player.hand.is_empty() {
                13 - 3 * player.melds.len()
                    + usize::from(hand.phase == Phase::Act && hand.turn == other)
            } else {
                player.hand.len()
            };
            held as u32
        })
        .collect()
}

/// A discard marked as claimed, paired with the called set that took it.
pub(super) struct Claim {
    /// Whose river the discard lies in.
    pub(super) target: Wind,
    /// The discard's order number, which is when the set was called.
    pub(super) order: u32,
    /// The tile taken.
    pub(super) tile: Tile,
    /// The seat that called.
    pub(super) claimer: Wind,
    /// Which of the claimer's sets took it.
    pub(super) meld: usize,
}

/// A called set that took a tile from somebody's river.
struct Taker {
    /// Whose river it took from.
    from: Wind,
    /// The tiles it could have taken: any of a chii's three.
    tiles: Vec<Tile>,
    claimer: Wind,
    meld: usize,
}

/// Pairs every discard marked as claimed with a called set that took it:
/// one taken from that seat and holding that tile, no set taking two.
///
/// A chii can explain more than one tile kind, so this is an injective
/// matching rather than a first fit: an earlier pairing may have to move to
/// make room for a later one. It never reorders the recorded meld history.
/// Where more than one pairing would do, it keeps to what the position says
/// happened: a claim gave the turn to its caller, so a claimed discard goes
/// first to the seat that made the next discard, and the tiles one seat took
/// from another go to its sets in the order they were recorded.
///
/// The position's validation and Mortal's replay of it both read this one
/// answer, so every table that passes validation is one the replay can
/// tell. The error names the first claimed discard no set explains.
pub(super) fn claims(hand: &Hand) -> Result<Vec<Claim>, (Wind, Tile)> {
    let mut takers: Vec<Taker> = Vec::new();
    for claimer in Wind::ALL {
        for (meld, set) in hand.players[claimer.index()].melds.iter().enumerate() {
            let offset = match set.from {
                ClaimedFrom::Right => 1,
                ClaimedFrom::Across => 2,
                ClaimedFrom::Left => 3,
                ClaimedFrom::SelfDrawn => continue,
            };
            let tiles = match set.kind {
                MeldKind::Chii => set.tiles(),
                MeldKind::ConcealedKan => continue,
                _ => vec![set.tile],
            };
            takers.push(Taker {
                from: claimer.plus(offset),
                tiles,
                claimer,
                meld,
            });
        }
    }
    let mut played: Vec<(u32, Wind)> = Wind::ALL
        .into_iter()
        .flat_map(|seat| {
            hand.players[seat.index()]
                .discards
                .iter()
                .map(move |discard| (discard.order, seat))
        })
        .collect();
    played.sort_unstable_by_key(|(order, _)| *order);
    let claimed: Vec<(Wind, u32, Tile)> = Wind::ALL
        .into_iter()
        .flat_map(|seat| {
            hand.players[seat.index()]
                .discards
                .iter()
                .filter(|discard| discard.claimed)
                .map(move |discard| (seat, discard.order, discard.tile))
        })
        .collect();
    // For each claimed discard, the sets that could have taken it, the
    // likeliest first. A claim nothing has followed yet belongs to the seat
    // that holds the turn now.
    let candidates: Vec<Vec<usize>> = claimed
        .iter()
        .map(|&(seat, order, tile)| {
            let caller = played
                .iter()
                .find(|(later, _)| *later > order)
                .map_or(hand.turn, |(_, next)| *next);
            let mut sets: Vec<usize> = (0..takers.len())
                .filter(|&at| takers[at].from == seat && takers[at].tiles.contains(&tile))
                .collect();
            sets.sort_by_key(|&at| takers[at].claimer != caller);
            sets
        })
        .collect();
    let mut owners = vec![None; takers.len()];
    for (claim, &(seat, _, tile)) in claimed.iter().enumerate() {
        if !assign_claim(
            claim,
            &candidates,
            &mut owners,
            &mut vec![false; takers.len()],
        ) {
            return Err((seat, tile));
        }
    }
    let mut paired: Vec<Claim> = owners
        .iter()
        .zip(&takers)
        .filter_map(|(owner, taker)| {
            let (target, order, tile) = claimed[(*owner)?];
            Some(Claim {
                target,
                order,
                tile,
                claimer: taker.claimer,
                meld: taker.meld,
            })
        })
        .collect();
    paired.sort_unstable_by_key(|claim| claim.order);
    Ok(paired)
}

/// Finds a set for one claimed discard, moving earlier pairings along where
/// that makes room. A free set is taken before any pairing is disturbed, so
/// one already made moves only when nothing else will do. The visited set
/// bounds a search to the public meld count (at most sixteen), including
/// cycles.
fn assign_claim(
    claim: usize,
    candidates: &[Vec<usize>],
    owners: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    if let Some(&free) = candidates[claim].iter().find(|&&set| owners[set].is_none()) {
        owners[free] = Some(claim);
        return true;
    }
    for &set in &candidates[claim] {
        if visited[set] {
            continue;
        }
        visited[set] = true;
        if let Some(previous) = owners[set] {
            if assign_claim(previous, candidates, owners, visited) {
                owners[set] = Some(claim);
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seat(hand: &str) -> Seat {
        Seat {
            hand: hand
                .split(' ')
                .filter(|t| !t.is_empty())
                .map(String::from)
                .collect(),
            melds: Vec::new(),
            discards: Vec::new(),
            score: 30000,
            riichi: "none".into(),
            ippatsu: false,
            furiten: false,
        }
    }

    fn thrown(tile: &str, order: u32, claimed: bool) -> Thrown {
        Thrown {
            tile: tile.into(),
            order,
            drawn: false,
            riichi: false,
            claimed,
        }
    }

    fn set(kind: &str, tile: &str, from: usize) -> Set {
        Set {
            kind: kind.into(),
            tile: tile.into(),
            from,
        }
    }

    /// East to act after a draw, holding a plain hand with the drawn tile in it.
    fn acting() -> Position {
        Position {
            seat: 0,
            turn: 0,
            phase: "act".into(),
            round: 0,
            kyoku: 1,
            counters: 0,
            riichi_sticks: 0,
            wall: 60,
            indicators: vec!["3s".into()],
            players: [
                seat("1m 2m 3m 4p 5p 6p 7s 8s 9s 1z 1z 2z 3z 5z"),
                seat(""),
                seat(""),
                seat(""),
            ],
            drawn: Some("5z".into()),
            pending: None,
            pending_kind: "discard".into(),
            just_claimed: None,
            after_quad: false,
            first_turns: false,
        }
    }

    fn overlapping_chii() -> Position {
        let mut position = acting();
        position.seat = 1;
        position.turn = 1;
        position.players[0] = seat("");
        position.players[1] = seat("1s 1s 1s 5z 5z");
        position.players[1].melds = vec![
            set("chii", "1p", 3),
            set("chii", "1m", 3),
            set("chii", "3m", 3),
        ];
        position.players[0].discards = vec![
            thrown("1p", 0, true),
            thrown("3m", 4, true),
            thrown("5m", 8, true),
        ];
        position
    }

    #[test]
    fn complete_chii_matching_is_independent_of_meld_and_discard_order() {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for meld_order in permutations {
            for discard_order in permutations {
                let mut position = overlapping_chii();
                let tiles = ["1p", "1m", "3m"];
                position.players[1].melds = meld_order.map(|i| set("chii", tiles[i], 3)).into();
                let claimed = ["1p", "3m", "5m"];
                position.players[0].discards = discard_order
                    .iter()
                    .enumerate()
                    .map(|(i, &kind)| thrown(claimed[kind], (4 * i) as u32, true))
                    .collect();
                // Exercise the position-builder used by physical advice AND settlement.
                for require_hand in [true, false] {
                    let (hand, _) = position.build_selected(require_hand).unwrap();
                    let actual: Vec<_> = hand.players[1].melds.iter().map(|m| m.tile).collect();
                    let expected: Vec<_> = meld_order
                        .iter()
                        .map(|&i| tile(tiles[i]).unwrap())
                        .collect();
                    assert_eq!(actual, expected, "validation must preserve meld history");
                }
            }
        }
    }

    /// The planes the trained network reads for a position, from the replay.
    fn trained_planes(position: &Position) -> Vec<f32> {
        let (hand, seat) = position.build().expect("a valid position");
        let state = mortal_log::state_for(&hand, seat).expect("the replay builds");
        let (planes, _) = state.encode_obs(crate::MORTAL_VERSION, false);
        planes.iter().copied().collect()
    }

    /// A hand as it is really played: South calls East's 1p as 1-2-3p, its
    /// 3m as 1-2-3m and its 5m as 3-4-5m, discarding after each, and now
    /// holds a draw. The sets are entered in `meld_order`.
    fn three_chii(meld_order: [usize; 3]) -> Position {
        let tiles = ["1p", "1m", "3m"];
        let mut position = acting();
        position.seat = 1;
        position.turn = 1;
        position.wall = 59;
        position.players[0] = seat("");
        position.players[0].discards = vec![
            thrown("1p", 0, true),
            thrown("3m", 4, true),
            thrown("5m", 8, true),
            thrown("6z", 12, false),
        ];
        position.players[1] = seat("1s 1s 1s 5z 5z");
        position.players[1].melds = meld_order
            .iter()
            .map(|&at| set("chii", tiles[at], 3))
            .collect();
        position.players[1].discards = vec![
            thrown("9s", 1, false),
            thrown("9s", 5, false),
            thrown("8s", 9, false),
        ];
        position.players[2].discards = vec![
            thrown("1z", 2, false),
            thrown("1z", 6, false),
            thrown("2z", 10, false),
        ];
        position.players[3].discards = vec![
            thrown("3z", 3, false),
            thrown("3z", 7, false),
            thrown("4z", 11, false),
        ];
        position
    }

    #[test]
    fn every_order_of_the_same_sets_replays_to_the_same_planes() {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let first = trained_planes(&three_chii(permutations[0]));
        for meld_order in permutations {
            let planes = trained_planes(&three_chii(meld_order));
            assert!(planes == first, "sets entered as {meld_order:?}");
        }
    }

    /// South called 1-2-3m on East's first 3m and 3-4-5m on its second,
    /// discarding after each, with the two sets entered as `melds`.
    fn twice_claimed(melds: [&str; 2]) -> Position {
        let mut position = acting();
        position.seat = 1;
        position.turn = 1;
        position.wall = 62;
        position.players[0] = seat("");
        position.players[0].discards = vec![
            thrown("3m", 0, true),
            thrown("3m", 4, true),
            thrown("6z", 8, false),
        ];
        position.players[1] = seat("1s 1s 1s 5z 5z 7p 8p 9p");
        position.players[1].melds = melds.iter().map(|low| set("chii", low, 3)).collect();
        position.players[1].discards = vec![thrown("9s", 1, false), thrown("9s", 5, false)];
        position.players[2].discards = vec![thrown("1z", 2, false), thrown("1z", 6, false)];
        position.players[3].discards = vec![thrown("3z", 3, false), thrown("3z", 7, false)];
        position
    }

    #[test]
    fn one_tile_claimed_twice_from_one_seat_replays_in_the_order_entered() {
        for melds in [["1m", "3m"], ["3m", "1m"]] {
            let (hand, seat) = twice_claimed(melds).build().expect("a valid position");
            let paired: Vec<(u32, usize)> = claims(&hand)
                .expect("both discards are explained")
                .iter()
                .map(|claim| (claim.order, claim.meld))
                .collect();
            assert_eq!(paired, vec![(0, 0), (4, 1)], "sets entered as {melds:?}");
            assert!(
                mortal_log::state_for(&hand, seat).is_some(),
                "sets entered as {melds:?}"
            );
        }
    }

    #[test]
    fn a_claimed_discard_goes_to_the_seat_that_played_next() {
        // West pons East's first 3m and discards; North plays; East lets
        // another 3m go, South chiis it as 1-2-3m and discards; South has
        // since drawn. Either set could hold either 3m.
        let mut position = acting();
        position.seat = 1;
        position.turn = 1;
        position.wall = 63;
        position.players[0] = seat("");
        position.players[0].discards = vec![
            thrown("3m", 0, true),
            thrown("3m", 3, true),
            thrown("6z", 7, false),
        ];
        position.players[1] = seat("1s 1s 1s 5z 5z 7p 8p 9p 9p 9p 2z");
        position.players[1].melds = vec![set("chii", "1m", 3)];
        position.players[1].discards = vec![thrown("9s", 4, false)];
        position.players[2].melds = vec![set("pon", "3m", 2)];
        position.players[2].discards = vec![thrown("1z", 1, false), thrown("1z", 5, false)];
        position.players[3].discards = vec![thrown("3z", 2, false), thrown("3z", 6, false)];
        let (hand, seat) = position.build().expect("a valid position");
        let paired: Vec<(u32, Wind)> = claims(&hand)
            .expect("both discards are explained")
            .iter()
            .map(|claim| (claim.order, claim.claimer))
            .collect();
        assert_eq!(paired, vec![(0, Wind::West), (3, Wind::South)]);
        assert!(mortal_log::state_for(&hand, seat).is_some());
    }

    #[test]
    fn complete_matching_still_rejects_reused_unexplained_and_wrong_source_claims() {
        let mut position = overlapping_chii();
        position.players[0].discards.push(thrown("1p", 12, true));
        assert!(position.build().unwrap_err().contains("needs a called set"));
        let mut position = overlapping_chii();
        position.players[0].discards[2].tile = "9p".into();
        assert!(position.build().unwrap_err().contains("needs a called set"));
        let mut position = overlapping_chii();
        position.players[2].discards = std::mem::take(&mut position.players[0].discards);
        assert!(position.build().unwrap_err().contains("needs a called set"));
    }

    #[test]
    fn a_claimed_discard_needs_a_set_that_took_it() {
        let mut position = acting();
        // Four in hand and a fifth on the table, hidden behind "claimed".
        position.players[0] = seat("1m 1m 1m 1m 4p 5p 6p 7s 8s 9s 1z 2z 3z 5z");
        position.players[1].discards.push(thrown("1m", 0, true));
        let error = position.build().unwrap_err();
        assert!(error.contains("claimed 1m"), "{error}");

        // A pon that took it from South explains the claimed discard.
        let mut position = acting();
        position.players[0] = seat("4p 5p 6p 7s 8s 9s 1z 2z 3z 5z 5z");
        position.players[0].melds.push(set("pon", "1m", 1));
        position.players[1].discards.push(thrown("1m", 0, true));
        position
            .build()
            .expect("a claimed discard matched by a pon is fine");
    }

    /// Claiming 4 characters for 4-5-6 bars the 4 and the 7 (EMA section
    /// 3.3.2). A hand left holding nothing else could not have made the
    /// call, so the position is refused rather than offered no move.
    #[test]
    fn a_call_that_leaves_only_barred_tiles_is_refused() {
        let called = || {
            vec![
                set("pon", "1z", 1),
                set("pon", "2z", 2),
                set("chii", "4m", 3),
            ]
        };
        let mut position = acting();
        position.drawn = None;
        position.just_claimed = Some("4m".into());
        position.players[0] = seat("4m 4m 7m 7m 7m");
        position.players[0].melds = called();
        let error = position.build().unwrap_err();
        assert!(error.contains("Swap-calling"), "{error}");

        // With one tile the rule allows, the call stands, and that tile is
        // the only thing to discard.
        position.players[0] = seat("4m 4m 7m 7m 9p");
        position.players[0].melds = called();
        let (hand, _) = position.build().expect("a tile is left to discard");
        assert_eq!(
            hand.legal_actions(),
            [riichi_core::game::Action::Discard(tile("9p").unwrap())]
        );
    }

    #[test]
    fn a_replacement_draw_needs_a_kan() {
        let mut position = acting();
        position.after_quad = true;
        let error = position.build().unwrap_err();
        assert!(error.contains("replacement draw"), "{error}");
    }

    #[test]
    fn the_live_wall_cannot_exceed_what_a_decision_can_see() {
        let mut position = acting();
        position.wall = 70;
        let error = position.build().unwrap_err();
        assert!(error.contains("69"), "{error}");
        position.wall = 69;
        position
            .build()
            .expect("69 is the most the dealer can see after drawing");
    }

    #[test]
    fn riichi_beside_an_open_set_says_so() {
        let mut position = acting();
        position.players[0] = seat("4p 5p 6p 7s 8s 9s 1z 2z 3z 5z 5z");
        position.players[0].melds.push(set("pon", "1m", 1));
        position.players[0].riichi = "riichi".into();
        position.players[1].discards.push(thrown("1m", 0, true));
        let error = position.build().unwrap_err();
        assert!(error.contains("closed hand"), "{error}");
    }

    /// South answers East's concealed kan. Only thirteen orphans could rob
    /// it, so only that shape declines a win by passing.
    fn passing_is_furiten(kan: &str, east: &str, south: &str) -> bool {
        let mut position = acting();
        position.seat = 1;
        position.phase = "call".into();
        position.drawn = None;
        position.pending = Some(kan.into());
        position.pending_kind = "concealed-kan".into();
        position.players[0] = seat(east);
        position.players[0].melds.push(set("concealed-kan", kan, 0));
        position.players[1] = seat(south);
        let (hand, seat) = position.build().expect("a valid robbery window");
        choices(&hand, seat)
            .into_iter()
            .find(|choice| choice.kind == "pass")
            .expect("passing is always offered")
            .causes_furiten
    }

    /// East answers South's 9m, which East can do nothing with: the decision
    /// a guided or physical table stops at after every discard.
    fn pass_only_response() -> Position {
        let mut position = acting();
        position.turn = 1;
        position.phase = "call".into();
        position.drawn = None;
        position.pending = Some("9m".into());
        position.players[0] = seat("1p 2p 3p 4p 5p 6p 7s 8s 9s 1z 2z 3z 5z");
        position.players[0].discards.push(thrown("6z", 0, false));
        position.players[1].discards.push(thrown("9m", 1, false));
        position
    }

    #[test]
    fn a_discard_nothing_can_claim_is_one_the_trained_adviser_answers() {
        let (hand, seat) = pass_only_response().build().expect("a valid response");
        let listed: Vec<(String, Option<usize>)> = choices(&hand, seat)
            .into_iter()
            .map(|choice| (choice.kind, choice.index))
            .collect();
        assert_eq!(listed, vec![("pass".to_string(), Some(encoding::PASS))]);
        // Mortal's mask opens its pass, and its pass is ours, both ways.
        let theirs = crate::mortal_mask_of(&hand, seat, false);
        let open: Vec<usize> = (0..theirs.len()).filter(|&at| theirs[at]).collect();
        assert_eq!(open, vec![crate::MORTAL_PASS]);
        assert_eq!(
            crate::action_from_mortal(&hand, seat, crate::MORTAL_PASS, false),
            encoding::PASS as i32
        );
        assert_eq!(
            crate::mortal_action_for(encoding::PASS),
            crate::MORTAL_PASS as i32
        );
        // And the network can be asked: the replay builds its planes.
        let state = mortal_log::state_for(&hand, seat).expect("the replay builds");
        let (planes, _) = state.encode_obs(crate::MORTAL_VERSION, false);
        assert_eq!(
            planes.nrows(),
            riichi::consts::obs_shape(crate::MORTAL_VERSION).0
        );
    }

    #[test]
    fn a_hidden_hand_is_counted_by_its_sets_not_as_empty() {
        // East answers South's discard; nobody else's hand was typed in, and
        // West has called a pon.
        let mut position = pass_only_response();
        position.players[2].melds.push(set("pon", "7z", 1));
        let (hand, seat) = position.build().expect("a valid response");
        assert_eq!(concealed_counts(&hand, seat), vec![13, 10, 13]);
        // East to act with a typed-in hand of fourteen. Asked from South,
        // East is the previous seat; hidden, it still holds its draw.
        let (mut hand, _) = acting().build().expect("a valid turn");
        assert_eq!(concealed_counts(&hand, Wind::South), vec![13, 13, 14]);
        hand.players[0].hand = TileSet::new();
        assert_eq!(concealed_counts(&hand, Wind::South), vec![13, 13, 14]);
        // The live game knows every hand and counts what is there.
        let game = riichi_core::table::Table::new().deal(&mut Rng::from_seed(3));
        let held: Vec<u32> = (1..4)
            .map(|offset| game.players[offset].hand.len() as u32)
            .collect();
        assert_eq!(concealed_counts(&game, Wind::East), held);
    }

    /// The trained mask, the translation of its moves and the listed choices,
    /// checked against each other for one seat at one decision.
    fn trained_and_listed_agree(hand: &Hand, seat: Wind) {
        let listed = choices(hand, seat);
        let open = crate::mortal_mask_of(hand, seat, false);
        // Our rules have no red fives, so Mortal's are never offered.
        assert!(
            !open[34..37].iter().any(|flag| *flag),
            "a red five is open for {seat:?}"
        );
        for (action, allowed) in open.iter().enumerate() {
            let ours = crate::action_from_mortal(hand, seat, action, false);
            assert_eq!(*allowed, ours >= 0, "Mortal's move {action} at {seat:?}");
            if *allowed {
                assert!(
                    listed
                        .iter()
                        .any(|choice| choice.index == Some(ours as usize)),
                    "Mortal's move {action} means {ours}, which is not listed for {seat:?}"
                );
            }
        }
        for choice in &listed {
            if let Some(index) = choice.index {
                let theirs = crate::mortal_action_for(index);
                assert!(
                    theirs >= 0 && open[theirs as usize],
                    "{} has no open trained move for {seat:?}",
                    choice.kind
                );
            }
        }
    }

    #[test]
    fn the_trained_mask_and_the_listed_choices_agree_at_every_decision() {
        let (mut decisions, mut passes_only) = (0, 0);
        for seed in 1..9u64 {
            let table = riichi_core::table::Table::new();
            let mut rng = Rng::from_seed(seed);
            let mut hand = table.deal(&mut rng);
            let mut bot = Bot::with_style(seed, Style::club());
            for _ in 0..1000 {
                match hand.phase {
                    Phase::Over => break,
                    Phase::Draw => {
                        let _ = hand.draw();
                        continue;
                    }
                    Phase::Act | Phase::CallWindow => {}
                }
                // Every seat a table would ask: the one to act, or all three
                // others once a tile is offered, claim or no claim.
                for seat in Wind::ALL {
                    if (hand.phase == Phase::Act) != (seat == hand.turn) {
                        continue;
                    }
                    trained_and_listed_agree(&hand, seat);
                    decisions += 1;
                    passes_only +=
                        usize::from(choices(&hand, seat).len() == 1 && seat != hand.turn);
                }
                if hand.phase == Phase::Act {
                    let action = bot.act(&hand);
                    hand.act(action).expect("the bot plays legal moves");
                } else {
                    let answers: Vec<_> = hand
                        .legal_calls()
                        .into_iter()
                        .map(|(who, calls)| (who, bot.call(&hand, who, &calls)))
                        .collect();
                    hand.resolve_calls(&answers).expect("the bot calls legally");
                }
            }
        }
        assert!(decisions > 400, "only {decisions} decisions were checked");
        assert!(
            passes_only > 150,
            "only {passes_only} had nothing but a pass"
        );
    }

    #[test]
    fn passing_a_concealed_kan_is_furiten_only_for_thirteen_orphans() {
        assert!(
            passing_is_furiten(
                "1m",
                "2m 3m 4m 4p 5p 6p 7s 8s 9s 1z",
                "9m 9m 1p 9p 1s 9s 1z 2z 3z 4z 5z 6z 7z"
            ),
            "thirteen orphans waiting on the one of characters could have robbed it"
        );
        assert!(
            !passing_is_furiten(
                "5m",
                "1m 2m 3m 4p 5p 6p 7s 8s 9s 1z",
                "1p 2p 3p 4p 5p 6p 7p 8p 9p 1s 1s 3m 4m"
            ),
            "an ordinary shape waiting on the five could not, so it declined nothing"
        );
    }
}
