//! WebAssembly bindings: the same rules engine the training runs on, in the
//! browser.
//!
//! The browser sees one object, [`Game`], which owns a table and the hand in
//! progress and hands out plain data: a view of the table from one seat, the
//! actions that seat may take, and the result of taking one. Nothing about
//! the rules lives on the JavaScript side, so the game a person plays and the
//! game the opponents were trained on cannot drift apart.

use riichi_core::bot::Bot;
use riichi_core::game::{Action, Call, Hand, Outcome, Phase};
use riichi_core::rng::Rng;
use riichi_core::score::Riichi;
use riichi_core::table::Table;
use riichi_core::tile::Tile;
use riichi_core::Wind;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// One tile in a discard row, as the interface needs it.
#[derive(Serialize)]
pub struct DiscardView {
    /// The tile, as `"3p"`.
    pub tile: String,
    /// Whether it was the tile just drawn.
    pub drawn: bool,
    /// Whether it was turned sideways for a riichi declaration.
    pub riichi: bool,
    /// Whether somebody claimed it.
    pub claimed: bool,
}

/// A called set in front of a player.
#[derive(Serialize)]
pub struct MeldView {
    /// `"chii"`, `"pon"`, `"claimed-kan"`, `"extended-kan"` or `"concealed-kan"`.
    pub kind: String,
    /// The tiles of the set, in order.
    pub tiles: Vec<String>,
    /// Which side the claimed tile came from: `"left"`, `"across"`, `"right"`
    /// or `"self"` for a concealed quad.
    pub from: String,
}

/// One player, as the person at the table can see them.
#[derive(Serialize)]
pub struct SeatView {
    /// The seat wind: `"east"`, `"south"`, `"west"` or `"north"`.
    pub seat: String,
    /// How many tiles are in hand.
    pub hand_size: usize,
    /// The concealed tiles, but only for the seat the view belongs to.
    /// The tile just drawn is not among them; it is kept apart below.
    pub hand: Vec<String>,
    /// The tile just drawn, held apart as it would be at the table.
    pub drawn: Option<String>,
    /// Called sets, which everyone can see.
    pub melds: Vec<MeldView>,
    /// The discard row.
    pub discards: Vec<DiscardView>,
    /// Points.
    pub score: i32,
    /// Whether this player has declared riichi.
    pub riichi: bool,
    /// Whether it is this player's turn.
    pub turn: bool,
}

/// Everything the interface needs to draw the table.
#[derive(Serialize)]
pub struct TableView {
    /// The round wind.
    pub round: String,
    /// Counters on the table.
    pub counters: u32,
    /// Riichi bets on the table.
    pub riichi_sticks: u32,
    /// Tiles left to draw.
    pub wall: usize,
    /// The dora indicators face up.
    pub dora_indicators: Vec<String>,
    /// The four seats, starting with the viewer's own.
    pub seats: Vec<SeatView>,
    /// Where the hand is: `"draw"`, `"act"`, `"call"` or `"over"`.
    pub phase: String,
    /// The tile awaiting claims, if any.
    pub pending_discard: Option<String>,
    /// A line describing how the hand ended, once it has.
    pub outcome: Option<String>,
    /// The viewer's waits, when the hand is waiting.
    pub waits: Vec<String>,
    /// How many changes the viewer's hand is from complete.
    pub shanten: i32,
    /// Whether the viewer may not win by discard.
    pub furiten: bool,
}

/// One thing the player may do.
#[derive(Serialize)]
pub struct ActionView {
    /// `"discard"`, `"riichi"`, `"tsumo"`, `"concealed-kan"`, `"extended-kan"`,
    /// `"ron"`, `"pon"`, `"kan"`, `"chii"` or `"pass"`.
    pub kind: String,
    /// The tile the action concerns, where there is one.
    pub tile: Option<String>,
    /// A label the interface can show.
    pub label: String,
}

/// A game against three heuristic opponents.
#[wasm_bindgen]
pub struct Game {
    table: Table,
    hand: Hand,
    rng: Rng,
    bots: Vec<Bot>,
    /// Which of the four people at the table is the person playing. The
    /// seats move between hands; this does not.
    player: usize,
    seat: Wind,
    log: Vec<String>,
}

#[wasm_bindgen]
impl Game {
    /// Starts a game. Which seat the player begins in is drawn by lot, as
    /// the players would draw wind tiles for it (EMA 2025 section 2.3).
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> Game {
        let seed = seed as u64;
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let player = rng.below(4);
        let hand = table.deal(&mut rng);
        let seat = table.seat_of(player);
        Game {
            table,
            hand,
            rng,
            bots: (0..4)
                .map(|index| Bot::new(seed.wrapping_add(index)))
                .collect(),
            player,
            seat,
            log: Vec::new(),
        }
    }

    /// The table as the player sees it.
    pub fn view(&self) -> Result<JsValue, JsValue> {
        let player = &self.hand.players[self.seat.index()];
        let waits = player.waits();
        let view = TableView {
            round: wind_name(self.hand.round).to_string(),
            counters: self.hand.counters,
            riichi_sticks: self.hand.riichi_sticks,
            wall: self.hand.wall.remaining(),
            dora_indicators: self
                .hand
                .wall
                .dora_indicators()
                .iter()
                .map(ToString::to_string)
                .collect(),
            seats: (0..4)
                .map(|offset| {
                    let seat = self.seat.plus(offset);
                    let player = &self.hand.players[seat.index()];
                    SeatView {
                        seat: wind_name(seat).to_string(),
                        hand_size: player.hand.len(),
                        hand: if seat == self.seat {
                            let mut tiles = player.hand;
                            if let Some(drawn) = self.drawn_tile(seat) {
                                tiles.remove(drawn);
                            }
                            tiles.tiles().map(|tile| tile.to_string()).collect()
                        } else {
                            Vec::new()
                        },
                        drawn: if seat == self.seat {
                            self.drawn_tile(seat).map(|tile| tile.to_string())
                        } else {
                            None
                        },
                        melds: player
                            .melds
                            .iter()
                            .map(|meld| MeldView {
                                kind: meld_kind_name(meld.kind).to_string(),
                                tiles: meld.tiles().iter().map(ToString::to_string).collect(),
                                from: claimed_from_name(meld.from).to_string(),
                            })
                            .collect(),
                        discards: player
                            .discards
                            .iter()
                            .map(|discard| DiscardView {
                                tile: discard.tile.to_string(),
                                drawn: discard.drawn,
                                riichi: discard.riichi,
                                claimed: discard.claimed,
                            })
                            .collect(),
                        score: player.score,
                        riichi: player.has_riichi(),
                        turn: self.hand.turn == seat,
                    }
                })
                .collect(),
            phase: match self.hand.phase {
                Phase::Draw => "draw",
                Phase::Act => "act",
                Phase::CallWindow => "call",
                Phase::Over => "over",
            }
            .to_string(),
            pending_discard: self.hand.pending_discard.map(|(_, tile)| tile.to_string()),
            outcome: self.hand.outcome.as_ref().map(describe_outcome),
            waits: waits.tiles().map(|tile| tile.to_string()).collect(),
            shanten: riichi_core::shanten::shanten(&player.hand, player.melds.len()),
            furiten: player.is_furiten(),
        };
        serde_wasm_bindgen::to_value(&view).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// What the player may do right now, which may be nothing while the
    /// opponents are playing.
    pub fn choices(&self) -> Result<JsValue, JsValue> {
        let mut list: Vec<ActionView> = Vec::new();
        match self.hand.phase {
            Phase::Act if self.hand.turn == self.seat => {
                for action in self.hand.legal_actions() {
                    list.push(describe_action(action));
                }
            }
            Phase::CallWindow => {
                for (seat, calls) in self.hand.legal_calls() {
                    if seat != self.seat {
                        continue;
                    }
                    for call in calls {
                        list.push(describe_call(call));
                    }
                }
            }
            _ => {}
        }
        serde_wasm_bindgen::to_value(&list).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Plays the opponents and the draws until the player has something to
    /// decide, or the hand ends. Returns the lines describing what happened.
    pub fn advance(&mut self) -> Result<JsValue, JsValue> {
        self.log.clear();
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 1000 {
                return Err(JsValue::from_str("the hand did not settle"));
            }
            match self.hand.phase {
                Phase::Over => break,
                Phase::Draw => {
                    let _ = self.hand.draw();
                }
                Phase::Act => {
                    if self.hand.turn == self.seat {
                        break;
                    }
                    let seat = self.hand.turn;
                    let who = self.table.player_at(seat);
                    let action = self.bots[who].act(&self.hand);
                    self.note(seat, &describe_action(action).label);
                    self.hand.act(action).map_err(refused)?;
                }
                Phase::CallWindow => {
                    let offered = self.hand.legal_calls();
                    if offered.iter().any(|(seat, _)| *seat == self.seat) {
                        break;
                    }
                    let answers: Vec<(Wind, Call)> = offered
                        .iter()
                        .map(|(seat, calls)| {
                            let who = self.table.player_at(*seat);
                            (*seat, self.bots[who].call(&self.hand, *seat, calls))
                        })
                        .collect();
                    for (seat, call) in &answers {
                        if !matches!(call, Call::Pass) {
                            self.note(*seat, &describe_call(*call).label);
                        }
                    }
                    self.hand.resolve_calls(&answers).map_err(refused)?;
                }
            }
        }
        serde_wasm_bindgen::to_value(&self.log)
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Takes the player's choice, named as `choices` returned it.
    pub fn choose(&mut self, kind: &str, tile: Option<String>) -> Result<(), JsValue> {
        let tile = match tile {
            Some(text) => Some(
                text.parse::<Tile>()
                    .map_err(|error| JsValue::from_str(&format!("not a tile: {error}")))?,
            ),
            None => None,
        };
        match self.hand.phase {
            Phase::Act if self.hand.turn == self.seat => {
                let action = match (kind, tile) {
                    ("discard", Some(tile)) => Action::Discard(tile),
                    ("riichi", Some(tile)) => Action::Riichi(tile),
                    ("tsumo", _) => Action::Tsumo,
                    ("concealed-kan", Some(tile)) => Action::ConcealedKan(tile),
                    ("extended-kan", Some(tile)) => Action::ExtendedKan(tile),
                    _ => return Err(JsValue::from_str("that is not an action")),
                };
                self.note(self.seat, &describe_action(action).label);
                self.hand.act(action).map_err(refused)
            }
            Phase::CallWindow => {
                let call = match (kind, tile) {
                    ("ron", _) => Call::Ron,
                    ("pon", _) => Call::Pon,
                    ("kan", _) => Call::Kan,
                    ("chii", Some(tile)) => Call::Chii(tile),
                    ("pass", _) => Call::Pass,
                    _ => return Err(JsValue::from_str("that is not a call")),
                };
                let offered = self.hand.legal_calls();
                let mut answers: Vec<(Wind, Call)> = vec![(self.seat, call)];
                for (seat, calls) in &offered {
                    if *seat == self.seat {
                        continue;
                    }
                    let who = self.table.player_at(*seat);
                    answers.push((*seat, self.bots[who].call(&self.hand, *seat, calls)));
                }
                if !matches!(call, Call::Pass) {
                    self.note(self.seat, &describe_call(call).label);
                }
                self.hand.resolve_calls(&answers).map_err(refused)
            }
            _ => Err(JsValue::from_str("there is nothing to choose now")),
        }
    }

    /// Whether the hand has finished.
    pub fn hand_is_over(&self) -> bool {
        matches!(self.hand.phase, Phase::Over)
    }

    /// Whether the whole game has finished.
    pub fn game_is_over(&self) -> bool {
        self.table.finished
    }

    /// Settles the finished hand and deals the next one.
    pub fn next_hand(&mut self) -> Result<(), JsValue> {
        if !matches!(self.hand.phase, Phase::Over) {
            return Err(JsValue::from_str("the hand is still being played"));
        }
        self.table.finish(&self.hand);
        // The player keeps their place at the table while the seats move.
        self.seat = self.table.seat_of(self.player);
        if !self.table.finished {
            self.hand = self.table.deal(&mut self.rng);
        }
        Ok(())
    }

    /// The final scores, once the game is over.
    pub fn final_scores(&self) -> Vec<i32> {
        self.table.final_scores().to_vec()
    }

    /// The tile the seat has just drawn and not yet used, if any.
    fn drawn_tile(&self, seat: Wind) -> Option<Tile> {
        if self.hand.turn != seat {
            return None;
        }
        self.hand.drawn
    }

    /// Which of the four people at the table the player is, so the final
    /// scores can be read.
    pub fn player_index(&self) -> usize {
        self.player
    }

    fn note(&mut self, seat: Wind, what: &str) {
        self.log.push(format!("{}: {what}", wind_name(seat)));
    }
}

fn refused(error: riichi_core::game::Error) -> JsValue {
    JsValue::from_str(&format!("the engine refused that: {error:?}"))
}

fn wind_name(wind: Wind) -> &'static str {
    match wind {
        Wind::East => "east",
        Wind::South => "south",
        Wind::West => "west",
        Wind::North => "north",
    }
}

fn meld_kind_name(kind: riichi_core::MeldKind) -> &'static str {
    use riichi_core::MeldKind::*;
    match kind {
        Chii => "chii",
        Pon => "pon",
        ClaimedKan => "claimed-kan",
        ExtendedKan => "extended-kan",
        ConcealedKan => "concealed-kan",
    }
}

fn claimed_from_name(from: riichi_core::ClaimedFrom) -> &'static str {
    use riichi_core::ClaimedFrom::*;
    match from {
        Left => "left",
        Across => "across",
        Right => "right",
        SelfDrawn => "self",
    }
}

fn describe_action(action: Action) -> ActionView {
    match action {
        Action::Discard(tile) => ActionView {
            kind: "discard".into(),
            tile: Some(tile.to_string()),
            label: format!("discards {tile}"),
        },
        Action::Riichi(tile) => ActionView {
            kind: "riichi".into(),
            tile: Some(tile.to_string()),
            label: format!("declares riichi on {tile}"),
        },
        Action::Tsumo => ActionView {
            kind: "tsumo".into(),
            tile: None,
            label: "wins by self-draw".into(),
        },
        Action::ConcealedKan(tile) => ActionView {
            kind: "concealed-kan".into(),
            tile: Some(tile.to_string()),
            label: format!("declares a quad of {tile}"),
        },
        Action::ExtendedKan(tile) => ActionView {
            kind: "extended-kan".into(),
            tile: Some(tile.to_string()),
            label: format!("extends a triplet of {tile}"),
        },
    }
}

fn describe_call(call: Call) -> ActionView {
    match call {
        Call::Ron => ActionView {
            kind: "ron".into(),
            tile: None,
            label: "wins on the discard".into(),
        },
        Call::Pon => ActionView {
            kind: "pon".into(),
            tile: None,
            label: "calls a triplet".into(),
        },
        Call::Kan => ActionView {
            kind: "kan".into(),
            tile: None,
            label: "calls a quad".into(),
        },
        Call::Chii(low) => ActionView {
            kind: "chii".into(),
            tile: Some(low.to_string()),
            label: format!("calls a sequence from {low}"),
        },
        Call::Pass => ActionView {
            kind: "pass".into(),
            tile: None,
            label: "passes".into(),
        },
    }
}

fn describe_outcome(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Win { winners, discarder } => {
            let mut parts: Vec<String> = Vec::new();
            for (seat, score) in winners {
                let how = match discarder {
                    Some(from) => format!("on {}'s discard", wind_name(*from)),
                    None => "by self-draw".to_string(),
                };
                let limit = score
                    .limit
                    .map(|limit| format!(" ({})", limit.name()))
                    .unwrap_or_default();
                let yaku: Vec<String> = score
                    .yaku
                    .iter()
                    .map(|(yaku, han)| format!("{} {han}", yaku.name()))
                    .collect();
                parts.push(format!(
                    "{} wins {how}: {} han, {} fu{limit} [{}]",
                    wind_name(*seat),
                    score.han,
                    score.fu,
                    yaku.join(", ")
                ));
            }
            parts.join("; ")
        }
        Outcome::ExhaustiveDraw { tenpai } => {
            let waiting: Vec<&str> = tenpai.iter().map(|seat| wind_name(*seat)).collect();
            if waiting.is_empty() {
                "exhaustive draw, nobody waiting".to_string()
            } else {
                format!("exhaustive draw, waiting: {}", waiting.join(", "))
            }
        }
    }
}

/// Whether a riichi declaration is showing, used by the interface's hints.
pub fn riichi_label(state: Riichi) -> &'static str {
    match state {
        Riichi::None => "",
        Riichi::Declared => "riichi",
        Riichi::Double => "double riichi",
    }
}
