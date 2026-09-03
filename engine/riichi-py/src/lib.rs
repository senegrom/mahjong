//! Python bindings: many games advancing together, for self-play training.
//!
//! Training wants batches, but mahjong does not offer them naturally: a hand
//! is a sequence of decisions by different seats, and a discard can put a
//! decision in front of three players at once. This module smooths that out.
//! Every game is advanced until exactly one seat owes a decision, and the
//! caller is handed one observation and one legality mask per game. Answers
//! to a claim are collected seat by seat and resolved once everybody in that
//! window has replied, which is what the rules describe (EMA 2025 section
//! 3.3.1).
//!
//! Observations come back as bytes, to be read with `numpy.frombuffer`, so
//! nothing is copied through Python objects.
//!
//! ```python
//! import numpy as np, riichi_py
//!
//! arena = riichi_py.Arena(games=256, seed=1)
//! while not arena.all_finished():
//!     seats = np.frombuffer(arena.seats(), dtype=np.int8)
//!     obs = np.frombuffer(arena.observations(), dtype=np.float32)
//!     obs = obs.reshape(-1, riichi_py.PLANES, riichi_py.POSITIONS)
//!     mask = np.frombuffer(arena.legal_mask(), dtype=bool).reshape(-1, riichi_py.ACTIONS)
//!     arena.step(policy(obs, mask))
//! ```

use std::collections::VecDeque;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use riichi_core::bot::Bot;
use riichi_core::encoding::{self, ACTIONS, OBSERVATION, PASS, PLANES, POSITIONS};
use riichi_core::game::{Call, Hand, Outcome, Phase};
use riichi_core::rng::Rng;
use riichi_core::table::Table;
use riichi_core::Wind;

/// One game, and where its next decision sits.
struct Seat {
    table: Table,
    hand: Hand,
    rng: Rng,
    /// Seats that still owe an answer to the claim on the table.
    asking: VecDeque<Wind>,
    /// Answers gathered so far in this claim window.
    answers: Vec<(Wind, Call)>,
    /// Points each seat held when the current hand began, so a hand's
    /// result can be reported as a change.
    opening_scores: [i32; 4],
    /// The change in points over the hand that just ended, by person.
    last_result: [i32; 4],
    /// Whether a hand ended on the most recent step.
    hand_just_ended: bool,
    finished: bool,
    /// The heuristic player for each of the four places, where one sits.
    bots: [Option<Bot>; 4],
    /// A heuristic player kept aside to answer "what would you do here",
    /// which is how a network is taught to imitate it.
    teacher: Bot,
}

impl Seat {
    fn new(seed: u64, bot_places: &[usize]) -> Seat {
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let hand = table.deal(&mut rng);
        let opening_scores = scores_of(&hand);
        let bots = std::array::from_fn(|place| {
            bot_places
                .contains(&place)
                .then(|| Bot::new(seed.wrapping_mul(4).wrapping_add(place as u64)))
        });
        let mut seat = Seat {
            table,
            hand,
            rng,
            asking: VecDeque::new(),
            answers: Vec::new(),
            opening_scores,
            last_result: [0; 4],
            hand_just_ended: false,
            finished: false,
            bots,
            teacher: Bot::new(seed ^ 0x7EAC_4E12),
        };
        seat.settle();
        seat
    }

    /// Whether the place a seat currently holds is played by a bot.
    fn is_bot(&self, seat: Wind) -> bool {
        self.bots[self.table.player_at(seat)].is_some()
    }

    /// The seat that owes a decision, if any.
    fn pending(&self) -> Option<Wind> {
        if self.finished {
            return None;
        }
        if let Some(seat) = self.asking.front() {
            return Some(*seat);
        }
        matches!(self.hand.phase, Phase::Act).then_some(self.hand.turn)
    }

    /// Draws, deals and resolves until a seat owes a decision or the game
    /// is over.
    fn settle(&mut self) {
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 100_000, "a game should settle long before this");
            if self.finished {
                return;
            }
            if let Some(seat) = self.asking.front().copied() {
                if !self.is_bot(seat) {
                    return;
                }
                let place = self.table.player_at(seat);
                let offered = self
                    .hand
                    .legal_calls()
                    .into_iter()
                    .find(|(other, _)| *other == seat)
                    .map(|(_, calls)| calls)
                    .unwrap_or_default();
                let bot = self.bots[place].as_mut().expect("checked above");
                let call = bot.call(&self.hand, seat, &offered);
                self.asking.pop_front();
                self.answers.push((seat, call));
                if self.asking.is_empty() {
                    let answers = std::mem::take(&mut self.answers);
                    self.hand
                        .resolve_calls(&answers)
                        .expect("every answer came from the offered set");
                }
                continue;
            }
            if matches!(self.hand.phase, Phase::Act) && self.is_bot(self.hand.turn) {
                let seat = self.hand.turn;
                let place = self.table.player_at(seat);
                let bot = self.bots[place].as_mut().expect("checked above");
                let action = bot.act(&self.hand);
                self.hand.act(action).expect("the bot chose a legal action");
                continue;
            }
            match self.hand.phase {
                Phase::Act => return,
                Phase::Draw => {
                    let _ = self.hand.draw();
                }
                Phase::CallWindow => {
                    let offered = self.hand.legal_calls();
                    if offered.is_empty() {
                        self.hand
                            .resolve_calls(&[])
                            .expect("an empty window resolves");
                    } else {
                        self.asking = offered.iter().map(|(seat, _)| *seat).collect();
                        self.answers.clear();
                        return;
                    }
                }
                Phase::Over => self.next_hand(),
            }
        }
    }

    fn next_hand(&mut self) {
        // Report the hand's result by person rather than by seat: the
        // seats move between hands, and a trajectory belongs to whoever was
        // sitting there. This has to happen before the deal rotates.
        let closing = scores_of(&self.hand);
        for seat in Wind::ALL {
            let place = self.table.player_at(seat);
            self.last_result[place] = closing[seat.index()] - self.opening_scores[seat.index()];
        }
        self.hand_just_ended = true;
        self.table.finish(&self.hand);
        if self.table.finished {
            self.finished = true;
            return;
        }
        self.hand = self.table.deal(&mut self.rng);
        self.opening_scores = scores_of(&self.hand);
    }

    /// Applies one decision from the seat that owed it.
    fn step(&mut self, index: usize) {
        self.hand_just_ended = false;
        if self.finished {
            return;
        }
        if let Some(seat) = self.asking.pop_front() {
            let call = encoding::decode_call(&self.hand, seat, index)
                .or_else(|| encoding::decode_call(&self.hand, seat, PASS))
                .unwrap_or(Call::Pass);
            self.answers.push((seat, call));
            if self.asking.is_empty() {
                let answers = std::mem::take(&mut self.answers);
                self.hand
                    .resolve_calls(&answers)
                    .expect("every answer came from the offered set");
            }
            self.settle();
            return;
        }
        if matches!(self.hand.phase, Phase::Act) {
            let action = encoding::decode_action(&self.hand, index).unwrap_or_else(|| {
                // A policy that names an illegal action still has to move, so
                // the first legal one is taken. The mask makes this rare.
                self.hand
                    .legal_actions()
                    .into_iter()
                    .next()
                    .expect("a player always has something to do")
            });
            self.hand.act(action).expect("the action was legal");
            self.settle();
        }
    }
}

fn scores_of(hand: &Hand) -> [i32; 4] {
    let mut scores = [0; 4];
    for seat in Wind::ALL {
        scores[seat.index()] = hand.players[seat.index()].score;
    }
    scores
}

impl Seat {
    /// The action the heuristic player would take in this position, as an
    /// index into the flat action space, or `None` where nothing is owed.
    fn teacher_choice(&mut self) -> Option<usize> {
        let seat = self.pending()?;
        if !self.asking.is_empty() {
            let offered = self
                .hand
                .legal_calls()
                .into_iter()
                .find(|(other, _)| *other == seat)
                .map(|(_, calls)| calls)
                .unwrap_or_default();
            let call = self.teacher.call(&self.hand, seat, &offered);
            let claimed = self.hand.pending_discard.map(|(_, tile)| tile)?;
            return Some(call_to_index(call, claimed));
        }
        let action = self.teacher.act(&self.hand);
        Some(action_to_index(action))
    }
}

fn action_to_index(action: riichi_core::game::Action) -> usize {
    use riichi_core::game::Action;
    match action {
        Action::Discard(tile) => tile.idx(),
        Action::Riichi(tile) => 34 + tile.idx(),
        Action::Tsumo => 68,
        Action::ConcealedKan(_) => 76,
        Action::ExtendedKan(_) => 77,
    }
}

fn call_to_index(call: Call, claimed: riichi_core::tile::Tile) -> usize {
    match call {
        Call::Ron => 69,
        Call::Pass => 70,
        Call::Pon => 74,
        Call::Kan => 75,
        Call::Chii(low) => 73 - (claimed.rank().saturating_sub(low.rank())) as usize,
    }
}

/// Many games of riichi, advancing together.
#[pyclass]
pub struct Arena {
    seats: Vec<Seat>,
    observations: Vec<f32>,
    mask: Vec<bool>,
}

#[pymethods]
impl Arena {
    /// Starts `games` games, each seeded from `seed`.
    ///
    /// `bot_places` names the places at every table that the built-in
    /// heuristic player takes; their decisions never reach Python. Leave it
    /// empty for self-play, or pass three places to measure one policy
    /// against the benchmark.
    #[new]
    #[pyo3(signature = (games, seed = 0, bot_places = vec![]))]
    fn new(games: usize, seed: u64, bot_places: Vec<usize>) -> Arena {
        Arena {
            seats: (0..games)
                .map(|index| Seat::new(seed.wrapping_add(index as u64), &bot_places))
                .collect(),
            observations: vec![0.0; games * OBSERVATION],
            mask: vec![false; games * ACTIONS],
        }
    }

    /// How many games are running.
    #[getter]
    fn games(&self) -> usize {
        self.seats.len()
    }

    /// The seat owing a decision in each game, or -1 where the game is over.
    fn seats<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self
            .seats
            .iter()
            .map(|seat| match seat.pending() {
                Some(wind) => wind.index() as u8,
                None => 0xFF,
            })
            .collect();
        PyBytes::new(py, &bytes)
    }

    /// One observation per game, for the seat owing a decision, as float32.
    fn observations<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        for (index, seat) in self.seats.iter().enumerate() {
            let slice = &mut self.observations[index * OBSERVATION..(index + 1) * OBSERVATION];
            match seat.pending() {
                Some(wind) => encoding::observe(&seat.hand, wind, slice),
                None => slice.fill(0.0),
            }
        }
        PyBytes::new(py, bytemuck_cast(&self.observations))
    }

    /// One legality mask per game, as bytes of 0 and 1.
    fn legal_mask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        for (index, seat) in self.seats.iter().enumerate() {
            let slice = &mut self.mask[index * ACTIONS..(index + 1) * ACTIONS];
            slice.fill(false);
            if let Some(wind) = seat.pending() {
                encoding::legal_mask(&seat.hand, wind, slice);
            }
        }
        let bytes: Vec<u8> = self.mask.iter().map(|flag| u8::from(*flag)).collect();
        PyBytes::new(py, &bytes)
    }

    /// What the heuristic player would do in each game, as bytes of the
    /// action index, or 0xFF where no decision is owed. This is the label a
    /// network is taught from before it is left to play on its own.
    fn teacher<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self
            .seats
            .iter_mut()
            .map(|seat| {
                seat.teacher_choice()
                    .map(|index| index as u8)
                    .unwrap_or(0xFF)
            })
            .collect();
        PyBytes::new(py, &bytes)
    }

    /// Applies one action per game. Games that owe no decision ignore theirs.
    fn step(&mut self, actions: Vec<usize>) -> PyResult<()> {
        if actions.len() != self.seats.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.seats.len(),
                actions.len()
            )));
        }
        for (seat, index) in self.seats.iter_mut().zip(actions) {
            seat.step(index);
        }
        Ok(())
    }

    /// Whether every game has finished.
    fn all_finished(&self) -> bool {
        self.seats.iter().all(|seat| seat.finished)
    }

    /// Games where a hand ended on the last step, as bytes of 0 and 1.
    fn hand_ended<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self
            .seats
            .iter()
            .map(|seat| u8::from(seat.hand_just_ended))
            .collect();
        PyBytes::new(py, &bytes)
    }

    /// The change in points over the hand that just ended, four per game,
    /// as int32 by person, not by seat.
    fn hand_result<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut values: Vec<i32> = Vec::with_capacity(self.seats.len() * 4);
        for seat in &self.seats {
            values.extend_from_slice(&seat.last_result);
        }
        PyBytes::new(py, cast_i32(&values))
    }

    /// The final scores of every game, four per game, as int32 by player.
    /// These already carry the winner bonus.
    fn final_scores<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut values: Vec<i32> = Vec::with_capacity(self.seats.len() * 4);
        for seat in &self.seats {
            values.extend_from_slice(&seat.table.final_scores());
        }
        PyBytes::new(py, cast_i32(&values))
    }

    /// Which player holds each seat, four per game, so a game's rewards can
    /// be attributed as the seats move between hands.
    fn seat_players<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut values: Vec<u8> = Vec::with_capacity(self.seats.len() * 4);
        for seat in &self.seats {
            for wind in Wind::ALL {
                values.push(seat.table.player_at(wind) as u8);
            }
        }
        PyBytes::new(py, &values)
    }

    /// What one game is doing right now, for tracking down a stuck table.
    fn debug(&self, game: usize) -> String {
        let seat = match self.seats.get(game) {
            Some(seat) => seat,
            None => return String::new(),
        };
        let pending = seat
            .pending()
            .map(|wind| format!("{wind:?}"))
            .unwrap_or_else(|| "none".to_string());
        let turn_player = seat.table.player_at(seat.hand.turn);
        format!(
            "phase {:?} turn {:?} (place {turn_player}) pending {pending} asking {:?}              finished {} wall {} riichi {:?} drawn {:?} hand {}",
            seat.hand.phase,
            seat.hand.turn,
            seat.asking,
            seat.finished,
            seat.hand.wall.remaining(),
            seat.hand.players[seat.hand.turn.index()].riichi,
            seat.hand.drawn,
            seat.hand.players[seat.hand.turn.index()].hand,
        )
    }

    /// A line describing how the last hand of one game ended, for logs.
    fn describe(&self, game: usize) -> String {
        let seat = match self.seats.get(game) {
            Some(seat) => seat,
            None => return String::new(),
        };
        match &seat.hand.outcome {
            Some(Outcome::Win { winners, discarder }) => winners
                .iter()
                .map(|(wind, score)| {
                    let how = match discarder {
                        Some(_) => "by discard",
                        None => "by self-draw",
                    };
                    let yaku: Vec<&str> =
                        score.yaku.iter().map(|(entry, _)| entry.name()).collect();
                    format!(
                        "{wind:?} wins {how}: {} han {} fu [{}]",
                        score.han,
                        score.fu,
                        yaku.join(", ")
                    )
                })
                .collect::<Vec<String>>()
                .join("; "),
            Some(Outcome::ExhaustiveDraw { tenpai }) => {
                format!("exhaustive draw, waiting: {tenpai:?}")
            }
            None => "in progress".to_string(),
        }
    }
}

fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // Safe: f32 has no padding and any bit pattern is a valid u8.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn cast_i32(values: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

/// The riichi rules engine, as a batched environment.
#[pymodule]
fn riichi_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Arena>()?;
    module.add("PLANES", PLANES)?;
    module.add("POSITIONS", POSITIONS)?;
    module.add("OBSERVATION", OBSERVATION)?;
    module.add("ACTIONS", ACTIONS)?;
    module.add("PASS", PASS)?;
    Ok(())
}
