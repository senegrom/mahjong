//! WebAssembly bindings: the same rules engine the training runs on, in the
//! browser.
//!
//! The browser sees one object, [`Game`], which owns a table and the hand in
//! progress and hands out plain data: a view of the table from one seat, the
//! actions that seat may take, and the result of taking one. Nothing about
//! the rules lives on the JavaScript side, so the game a person plays and the
//! game the opponents were trained on cannot drift apart.

use riichi_core::bot::{Bot, Style};
use riichi_core::encoding::{self, ACTIONS, OBSERVATION};
use riichi_core::game::{Action, Call, Hand, Outcome, Phase};
use riichi_core::review;
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
    /// How the hand ended, once it has.
    pub outcome: Option<OutcomeView>,
    /// The viewer's waits, when the hand is waiting.
    pub waits: Vec<String>,
    /// How many of each wait nobody can see yet, in the same order. A wait
    /// on a tile three of which are already on the table is a much thinner
    /// hand than the number of waits alone suggests.
    pub waits_left: Vec<u8>,
    /// The viewer's tiles that are dora, each worth a han.
    pub dora: Vec<String>,
    /// How many changes the viewer's hand is from complete.
    pub shanten: i32,
    /// Whether the viewer may not win by discard.
    pub furiten: bool,
    /// Tiles that cannot deal into anybody who has declared riichi: their
    /// own discards, and whatever passed after they declared. Empty while
    /// nobody has declared (EMA 2025 section 3.3.9).
    pub safe: Vec<String>,
}

/// A hand that won, with the working shown.
#[derive(Serialize)]
pub struct WinView {
    /// The winner's seat.
    pub seat: String,
    /// Whether it was won by discard or by self-draw.
    pub by: String,
    /// Who let the tile go, when it was not self-drawn.
    pub from: Option<String>,
    /// The winner's concealed tiles, without the winning tile.
    pub hand: Vec<String>,
    /// Their called sets.
    pub melds: Vec<MeldView>,
    /// The tile that completed the hand, shown apart from it.
    pub winning_tile: String,
    /// The yaku, each with the han it was worth here.
    pub yaku: Vec<YakuView>,
    /// Han in total, dora included.
    pub han: u8,
    /// How many of those han came from dora.
    pub dora: u8,
    /// Minipoints.
    pub fu: u32,
    /// The limit reached, if any.
    pub limit: Option<String>,
    /// What the hand is worth, said the way a table would say it.
    pub payment: String,
    /// Riichi bets the winner also collected, which is why the points moved
    /// by more than the hand was worth.
    pub bets: i32,
}

/// One yaku and its han.
#[derive(Serialize)]
pub struct YakuView {
    /// Its name.
    pub name: String,
    /// The han it was worth in this hand.
    pub han: u8,
}

/// How a hand ended.
#[derive(Serialize)]
pub struct OutcomeView {
    /// `"win"` or `"draw"`.
    pub kind: String,
    /// A one-line summary.
    pub line: String,
    /// The winning hands, of which there may be more than one.
    pub wins: Vec<WinView>,
    /// At an exhaustive draw, the seats that were waiting.
    pub tenpai: Vec<String>,
    /// What each seat gained or lost over the hand, in the view's own seat
    /// order, so the first entry is always the player's.
    pub changes: Vec<i32>,
}

/// Where one player finished.
#[derive(Serialize)]
pub struct StandingView {
    /// First to fourth.
    pub place: usize,
    /// The seat they hold in the last hand, which is what the table shows.
    pub seat: String,
    /// Points at the table when the game ended.
    pub score: i32,
    /// The winner bonus their place earned, or the penalty it cost.
    pub uma: i32,
    /// What the game was worth to them: points less the return score, plus
    /// the bonus.
    pub total: i32,
    /// Whether this is the person playing.
    pub you: bool,
}

/// One decision of the player's, looked at again after the hand.
#[derive(Serialize)]
pub struct NoteView {
    /// Which discard of the hand this was, counting from one.
    pub turn: u32,
    /// What the player did, in words.
    pub played: String,
    /// The tile it discarded, if it discarded one.
    pub played_tile: Option<String>,
    /// What the adviser would have done.
    pub advised: String,
    /// The tile that would have discarded.
    pub advised_tile: Option<String>,
    /// Whether the two are the same.
    pub agreed: bool,
    /// Distance to a complete hand after the move played.
    pub shanten_played: i32,
    /// The same after the advised move.
    pub shanten_advised: i32,
    /// Tiles still out there that would improve the hand, after each move.
    pub acceptance_played: u32,
    /// The same after the advised move.
    pub acceptance_advised: u32,
    /// How exposed each tile was: `quiet`, `safe` or `live`.
    pub danger_played: String,
    /// The same for the advised tile.
    pub danger_advised: String,
    /// Why they differ, in words.
    pub reason: String,
    /// A short tag for the reason, for styling.
    pub kind: String,
    /// How many improving tiles the move gave up.
    pub cost: u32,
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
    /// Points each seat held when the hand was dealt, so the score screen
    /// can say what the hand cost or paid.
    opening: [i32; 4],
    /// The position before each of the player's own decisions, with what
    /// they did from it, which is all a review needs.
    decisions: Vec<(Hand, Action)>,
    /// Whether the opponents are answered from outside, which is what a
    /// trained network needs: it runs in the page, not in the engine.
    external: bool,
    /// Seats that still owe an answer to the claim on the table.
    asking: Vec<Wind>,
    /// Answers gathered so far in this claim window.
    gathered: Vec<(Wind, Call)>,
}

#[wasm_bindgen]
impl Game {
    /// Starts a game against opponents of the named strength: `"beginner"`
    /// presses on regardless and is loose about which tile goes, `"club"`
    /// counts its tiles and folds against a declared riichi.
    ///
    /// Which seat the player begins in is drawn by lot, as the players would
    /// draw wind tiles for it (EMA 2025 section 2.3).
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64, difficulty: Option<String>) -> Game {
        let seed = seed as u64;
        let external = matches!(difficulty.as_deref(), Some("neural"));
        let style = match difficulty.as_deref() {
            Some("beginner") => Style::beginner(),
            _ => Style::club(),
        };
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let player = rng.below(4);
        let hand = table.deal(&mut rng);
        let seat = table.seat_of(player);
        let mut game = Game {
            table,
            hand,
            rng,
            bots: (0..4)
                .map(|index| Bot::with_style(seed.wrapping_add(index), style))
                .collect(),
            player,
            seat,
            log: Vec::new(),
            opening: [0; 4],
            decisions: Vec::new(),
            external,
            asking: Vec::new(),
            gathered: Vec::new(),
        };
        game.opening = scores_of(&game.hand);
        game
    }

    /// Whether an opponent owes a decision that the page must answer.
    pub fn needs_opponent_move(&self) -> bool {
        self.external && self.opponent_owing().is_some()
    }

    /// The observation for the opponent who owes a decision, as the network
    /// expects it: planes over the tile kinds, that seat's own view.
    pub fn opponent_observation(&self) -> Vec<f32> {
        let mut out = vec![0.0; OBSERVATION];
        if let Some(seat) = self.opponent_owing() {
            encoding::observe(&self.hand, seat, &mut out);
        }
        out
    }

    /// Which entries of the action space that seat may choose.
    pub fn opponent_mask(&self) -> Vec<u8> {
        let mut mask = vec![false; ACTIONS];
        if let Some(seat) = self.opponent_owing() {
            encoding::legal_mask(&self.hand, seat, &mut mask);
        }
        mask.iter().map(|flag| u8::from(*flag)).collect()
    }

    /// Takes the page's answer for that opponent.
    pub fn play_opponent(&mut self, index: usize) -> Result<(), JsValue> {
        let seat = match self.opponent_owing() {
            Some(seat) => seat,
            None => return Err(JsValue::from_str("no opponent owes a decision")),
        };
        if !self.asking.is_empty() {
            let call = encoding::decode_call(&self.hand, seat, index)
                .or_else(|| encoding::decode_call(&self.hand, seat, encoding::PASS))
                .unwrap_or(Call::Pass);
            self.asking.retain(|other| *other != seat);
            self.gathered.push((seat, call));
            if !matches!(call, Call::Pass) {
                self.note(seat, &describe_call(call).label);
            }
            if self.asking.is_empty() {
                let answers = std::mem::take(&mut self.gathered);
                self.hand.resolve_calls(&answers).map_err(refused)?;
            }
            return Ok(());
        }
        let action = encoding::decode_action(&self.hand, index)
            .or_else(|| self.hand.legal_actions().into_iter().next());
        match action {
            Some(action) => {
                self.note(seat, &describe_action(action).label);
                self.hand.act(action).map_err(refused)
            }
            None => Err(JsValue::from_str("that seat has nothing it may do")),
        }
    }

    /// The table as the player sees it.
    pub fn view(&self) -> Result<JsValue, JsValue> {
        let player = &self.hand.players[self.seat.index()];
        let waits = player.waits();
        let seen = review::visible_to(&self.hand, self.seat);
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
            outcome: self.describe_outcome(),
            waits: waits.tiles().map(|tile| tile.to_string()).collect(),
            waits_left: waits
                .tiles()
                .map(|tile| riichi_core::tile::COPIES.saturating_sub(seen.count(tile)))
                .collect(),
            dora: {
                let mut marked: Vec<String> = Vec::new();
                for indicator in self.hand.wall.dora_indicators() {
                    let tile = indicator.dora();
                    for _ in 0..player.hand.count(tile) {
                        marked.push(tile.to_string());
                    }
                    for meld in &player.melds {
                        for member in meld.tiles() {
                            if member == tile {
                                marked.push(tile.to_string());
                            }
                        }
                    }
                }
                marked.sort();
                marked
            },
            shanten: riichi_core::shanten::shanten(&player.hand, player.melds.len()),
            furiten: player.is_furiten(),
            safe: self.safe_tiles(),
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
        // The log is drained rather than cleared: moves the page answered
        // for the opponents were noted before this call and would otherwise
        // be thrown away unread.
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
                    if self.external {
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
                    if self.external && !offered.is_empty() {
                        // The page answers for the opponents, one at a time.
                        if self.asking.is_empty() {
                            self.asking = offered.iter().map(|(seat, _)| *seat).collect();
                            self.gathered.clear();
                        }
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
        let lines = std::mem::take(&mut self.log);
        serde_wasm_bindgen::to_value(&lines).map_err(|error| JsValue::from_str(&error.to_string()))
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
                self.decisions.push((self.hand.clone(), action));
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
                    // With the page answering for the opponents, a claim it
                    // has already gathered stands; anything not yet asked
                    // passes, since the player's own answer settles the
                    // window either way.
                    if self.external {
                        let already = self
                            .gathered
                            .iter()
                            .find(|(other, _)| other == seat)
                            .map(|(_, call)| *call);
                        answers.push((*seat, already.unwrap_or(Call::Pass)));
                        continue;
                    }
                    let who = self.table.player_at(*seat);
                    answers.push((*seat, self.bots[who].call(&self.hand, *seat, calls)));
                }
                self.asking.clear();
                self.gathered.clear();
                if !matches!(call, Call::Pass) {
                    self.note(self.seat, &describe_call(call).label);
                }
                self.hand.resolve_calls(&answers).map_err(refused)
            }
            _ => Err(JsValue::from_str("there is nothing to choose now")),
        }
    }

    /// Every decision the player made this hand, judged against what the
    /// adviser would have done. Meant to be read once the hand is over.
    pub fn review(&self) -> Result<JsValue, JsValue> {
        // A fresh adviser, so the review does not depend on what the bots
        // happened to roll while they were playing.
        let mut adviser = Bot::with_style(0x5eed_5eed, Style::club());
        let notes: Vec<NoteView> = self
            .decisions
            .iter()
            .map(|(position, played)| {
                let note = review::judge(position, *played, &mut adviser);
                NoteView {
                    turn: note.turn + 1,
                    played: describe_action(note.played).label,
                    played_tile: action_tile(note.played).map(|tile| tile.to_string()),
                    advised: describe_action(note.advised).label,
                    advised_tile: action_tile(note.advised).map(|tile| tile.to_string()),
                    agreed: note.agreed(),
                    shanten_played: note.shanten_played,
                    shanten_advised: note.shanten_advised,
                    acceptance_played: note.acceptance_played,
                    acceptance_advised: note.acceptance_advised,
                    danger_played: danger_name(note.danger_played).to_string(),
                    danger_advised: danger_name(note.danger_advised).to_string(),
                    reason: note.reason.line().to_string(),
                    kind: reason_name(note.reason).to_string(),
                    cost: note.cost(),
                }
            })
            .collect();
        serde_wasm_bindgen::to_value(&notes).map_err(|error| JsValue::from_str(&error.to_string()))
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
            self.opening = scores_of(&self.hand);
            self.decisions.clear();
        }
        Ok(())
    }

    /// The final scores, once the game is over.
    pub fn final_scores(&self) -> Vec<i32> {
        self.table.final_scores().to_vec()
    }

    /// Where everybody finished, best first, once the game is over.
    pub fn standings(&self) -> Result<JsValue, JsValue> {
        let totals = self.table.final_scores();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|player| core::cmp::Reverse(self.table.scores[*player]));

        let standings: Vec<StandingView> = order
            .iter()
            .enumerate()
            .map(|(place, player)| StandingView {
                place: place + 1,
                seat: wind_name(self.table.seat_of(*player)).to_string(),
                score: self.table.scores[*player],
                // What the bonus was worth is the difference between the
                // final total and the points over the return score.
                uma: totals[*player]
                    - (self.table.scores[*player] - riichi_core::table::RETURN_SCORE),
                total: totals[*player],
                you: *player == self.player,
            })
            .collect();
        serde_wasm_bindgen::to_value(&standings)
            .map_err(|error| JsValue::from_str(&error.to_string()))
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

    /// The opponent seat owing a decision, when the page answers for them.
    fn opponent_owing(&self) -> Option<Wind> {
        if !self.external {
            return None;
        }
        if let Some(seat) = self.asking.first() {
            return Some(*seat);
        }
        match self.hand.phase {
            Phase::Act if self.hand.turn != self.seat => Some(self.hand.turn),
            _ => None,
        }
    }

    /// The tiles that cannot deal into anybody currently waiting on a
    /// declared riichi. A tile has to be safe against every one of them.
    fn safe_tiles(&self) -> Vec<String> {
        let declared: Vec<Wind> = Wind::ALL
            .into_iter()
            .filter(|seat| *seat != self.seat)
            .filter(|seat| self.hand.players[seat.index()].has_riichi())
            .collect();
        if declared.is_empty() {
            return Vec::new();
        }
        let sets: Vec<riichi_core::TileSet> = declared
            .iter()
            .map(|seat| self.hand.safe_against(*seat))
            .collect();
        riichi_core::tile::Tile::all()
            .filter(|tile| sets.iter().all(|safe| safe.count(*tile) > 0))
            .map(|tile| tile.to_string())
            .collect()
    }

    /// How the hand ended, with enough to show a score screen.
    fn describe_outcome(&self) -> Option<OutcomeView> {
        let outcome = self.hand.outcome.as_ref()?;
        let now = scores_of(&self.hand);
        let changes = (0..4)
            .map(|offset| {
                let seat = self.seat.plus(offset);
                now[seat.index()] - self.opening[seat.index()]
            })
            .collect();

        Some(match outcome {
            Outcome::Win { winners, discarder } => {
                let wins: Vec<WinView> = winners
                    .iter()
                    .map(|(seat, score)| {
                        let player = &self.hand.players[seat.index()];
                        let mut hand = player.hand;
                        // A hand won by discard does not hold the tile it
                        // won on; one won by self-draw does.
                        if discarder.is_none() {
                            hand.remove(score.winning_tile);
                        }
                        WinView {
                            seat: wind_name(*seat).to_string(),
                            by: if discarder.is_some() {
                                "discard"
                            } else {
                                "self-draw"
                            }
                            .to_string(),
                            from: discarder.map(|from| wind_name(from).to_string()),
                            hand: hand.tiles().map(|tile| tile.to_string()).collect(),
                            melds: player
                                .melds
                                .iter()
                                .map(|meld| MeldView {
                                    kind: meld_kind_name(meld.kind).to_string(),
                                    tiles: meld.tiles().iter().map(ToString::to_string).collect(),
                                    from: claimed_from_name(meld.from).to_string(),
                                })
                                .collect(),
                            winning_tile: score.winning_tile.to_string(),
                            yaku: score
                                .yaku
                                .iter()
                                .map(|(yaku, han)| YakuView {
                                    name: yaku.name().to_string(),
                                    han: *han,
                                })
                                .collect(),
                            han: score.han,
                            dora: score.dora,
                            fu: score.fu,
                            limit: score.limit.map(|limit| limit.name().to_string()),
                            payment: describe_payment(
                                score,
                                matches!(seat, Wind::East),
                                discarder.is_some(),
                            ),
                            bets: (now[seat.index()] - self.opening[seat.index()])
                                - score.payments.total as i32,
                        }
                    })
                    .collect();
                let line = wins
                    .iter()
                    .map(|win| {
                        let how = match &win.from {
                            Some(from) => {
                                let name = match from.as_str() {
                                    "east" => "East",
                                    "south" => "South",
                                    "west" => "West",
                                    _ => "North",
                                };
                                format!("on {name}'s discard")
                            }
                            None => "by self-draw".to_string(),
                        };
                        let seat = match win.seat.as_str() {
                            "east" => "East",
                            "south" => "South",
                            "west" => "West",
                            _ => "North",
                        };
                        format!("{seat} wins {how}")
                    })
                    .collect::<Vec<String>>()
                    .join("; ");
                OutcomeView {
                    kind: "win".to_string(),
                    line,
                    wins,
                    tenpai: Vec::new(),
                    changes,
                }
            }
            Outcome::ExhaustiveDraw { tenpai } => OutcomeView {
                kind: "draw".to_string(),
                line: if tenpai.is_empty() {
                    "Exhaustive draw, nobody waiting".to_string()
                } else {
                    "Exhaustive draw".to_string()
                },
                wins: Vec::new(),
                tenpai: tenpai
                    .iter()
                    .map(|seat| wind_name(*seat).to_string())
                    .collect(),
                changes,
            },
        })
    }

    fn note(&mut self, seat: Wind, what: &str) {
        // Seats read as names in the commentary, not as identifiers.
        let name = seat_title(seat);
        self.log.push(format!("{name} {what}"));
    }
}

fn refused(error: riichi_core::game::Error) -> JsValue {
    JsValue::from_str(&format!("the engine refused that: {error:?}"))
}

/// A tile named the way a person would say it, for the commentary.
fn tile_words(tile: Tile) -> String {
    use riichi_core::Suit;
    if tile.is_honour() {
        return match tile.rank() {
            1 => "east wind",
            2 => "south wind",
            3 => "west wind",
            4 => "north wind",
            5 => "white dragon",
            6 => "green dragon",
            _ => "red dragon",
        }
        .to_string();
    }
    let suit = match tile.suit() {
        Suit::Characters => "characters",
        Suit::Circles => "circles",
        Suit::Bamboo => "bamboo",
        Suit::Honours => "honours",
    };
    format!("{} {suit}", tile.rank())
}

fn wind_name(wind: Wind) -> &'static str {
    match wind {
        Wind::East => "east",
        Wind::South => "south",
        Wind::West => "west",
        Wind::North => "north",
    }
}

/// The same seat, written as a name rather than an identifier.
fn seat_title(wind: Wind) -> &'static str {
    match wind {
        Wind::East => "East",
        Wind::South => "South",
        Wind::West => "West",
        Wind::North => "North",
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
            label: format!("discards the {}", tile_words(tile)),
        },
        Action::Riichi(tile) => ActionView {
            kind: "riichi".into(),
            tile: Some(tile.to_string()),
            label: format!("declares riichi on the {}", tile_words(tile)),
        },
        Action::Tsumo => ActionView {
            kind: "tsumo".into(),
            tile: None,
            label: "wins by self-draw".into(),
        },
        Action::ConcealedKan(tile) => ActionView {
            kind: "concealed-kan".into(),
            tile: Some(tile.to_string()),
            label: format!("declares a quad of {}", tile_words(tile)),
        },
        Action::ExtendedKan(tile) => ActionView {
            kind: "extended-kan".into(),
            tile: Some(tile.to_string()),
            label: format!("extends a triplet of {}", tile_words(tile)),
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
            label: format!("calls a sequence from the {}", tile_words(low)),
        },
        Call::Pass => ActionView {
            kind: "pass".into(),
            tile: None,
            label: "passes".into(),
        },
    }
}

fn scores_of(hand: &Hand) -> [i32; 4] {
    let mut scores = [0; 4];
    for seat in Wind::ALL {
        scores[seat.index()] = hand.players[seat.index()].score;
    }
    scores
}

/// What a hand is worth, said the way a table would say it.
fn describe_payment(score: &riichi_core::score::Score, dealer: bool, by_discard: bool) -> String {
    let payments = score.payments;
    if by_discard {
        return format!("{} from the discarder", payments.from_discarder);
    }
    if dealer {
        format!("{} from each", payments.from_each_other)
    } else {
        format!(
            "{} from the dealer and {} from the others",
            payments.from_dealer, payments.from_each_other
        )
    }
}

/// How exposed a tile was, in a word.
fn danger_name(danger: review::Danger) -> &'static str {
    match danger {
        review::Danger::Quiet => "quiet",
        review::Danger::Safe => "safe",
        review::Danger::Live => "live",
    }
}

/// A short tag for why the adviser differs.
fn reason_name(reason: review::Reason) -> &'static str {
    match reason {
        review::Reason::Agreed => "agreed",
        review::Reason::Shape => "shape",
        review::Reason::Acceptance => "acceptance",
        review::Reason::Defence => "defence",
        review::Reason::Judgement => "judgement",
    }
}

/// The tile an action puts down, if it puts one down.
fn action_tile(action: Action) -> Option<Tile> {
    match action {
        Action::Discard(tile) | Action::Riichi(tile) => Some(tile),
        Action::ConcealedKan(tile) | Action::ExtendedKan(tile) => Some(tile),
        Action::Tsumo => None,
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
