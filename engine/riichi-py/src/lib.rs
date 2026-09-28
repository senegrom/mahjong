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
use riichi_core::encoding::{
    self, ACTIONS, HANDS, OBSERVATION, OPPONENTS, ORACLE, ORACLE_PLANES, PASS, PLANES, POSITIONS,
};
use riichi_core::game::{Call, Hand, Phase};
use riichi_core::mjai;
use riichi_core::rng::Rng;
use riichi_core::table::Table;
use riichi_core::Wind;

/// One game, and where its next decision sits.
struct Seat {
    table: Table,
    hand: Hand,
    /// Deals the hands. Nothing hypothetical may draw from it.
    rng: Rng,
    /// Seats that still owe an answer to the claim on the table.
    asking: VecDeque<Wind>,
    /// Answers gathered so far in this claim window.
    answers: Vec<(Wind, Call)>,
    /// What each hand that ended since the last step began moved, by
    /// person, in the order they ended; before the first step, those that
    /// ended while the game was made. Native bots can finish a hand without
    /// an outside decision, even two in a row, so one step can end more
    /// than one hand. Only the first can have a decision from outside
    /// waiting on its points: any after it was dealt and finished within
    /// the same step.
    endings: Vec<[i32; 4]>,
    /// How many hands this game has finished. The caller's own count can
    /// only be right if it saw every ending; this one cannot miss any, so
    /// the two disagreeing says an ending went unnoticed.
    hands_done: u32,
    finished: bool,
    /// The heuristic player for each of the four places, where one sits.
    bots: [Option<Bot>; 4],
    /// A heuristic player kept aside to answer "what would you do here",
    /// which is how a network is taught to imitate it.
    teacher: Bot,
    /// The mjai events of this game not yet handed to Python, as JSON
    /// lines, each written under the seating its hand was dealt with.
    events: Vec<String>,
    /// How many of the current hand's log entries have been written out.
    logged: usize,
    /// Whether start_game and end_game have been written.
    started: bool,
    ended: bool,
}

impl Seat {
    fn new(seed: u64, bot_places: &[usize]) -> Seat {
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let hand = table.deal(&mut rng);
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
            endings: Vec::new(),
            hands_done: 0,
            finished: false,
            bots,
            teacher: Bot::new(seed ^ 0x7EAC_4E12),
            events: Vec::new(),
            logged: 0,
            started: false,
            ended: false,
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
                        // Settle bot claims before exporting the next external decision.
                        continue;
                    }
                }
                Phase::Over => self.next_hand(),
            }
        }
    }

    /// Writes the current hand's unwritten events out, under the seating
    /// the hand was dealt with. It has to run before the table rotates,
    /// because the mjai player numbers are the seating and the seating
    /// changes with the deal.
    fn flush_events(&mut self) {
        if !self.started {
            let names = std::array::from_fn(|player| format!("player {player}"));
            self.events
                .push(mjai::Event::StartGame { names }.to_json([0, 1, 2, 3]));
            self.started = true;
        }
        let seating = self.table.seating();
        for event in &self.hand.log[self.logged..] {
            self.events.push(event.to_json(seating));
        }
        self.logged = self.hand.log.len();
    }

    /// The game's mjai events since they were last asked for, as JSON
    /// lines: start_game first, then every event of every hand under the
    /// seating it was dealt with, end_game last. What a player state on
    /// the Python side follows to describe the game the way Mortal's own
    /// bot sees it.
    fn take_events(&mut self) -> Vec<String> {
        self.flush_events();
        if self.finished && !self.ended {
            self.events.push(mjai::Event::EndGame.to_json([0, 1, 2, 3]));
            self.ended = true;
        }
        std::mem::take(&mut self.events)
    }

    fn next_hand(&mut self) {
        self.flush_events();
        // Report the hand's result by person rather than by seat: the
        // seats move between hands, and a trajectory belongs to whoever was
        // sitting there. This has to happen before the deal rotates.
        let moved = self.hand.deltas();
        let mut by_person = [0; 4];
        for seat in Wind::ALL {
            by_person[self.table.player_at(seat)] = moved[seat.index()];
        }
        self.endings.push(by_person);
        self.hands_done += 1;
        self.table.finish(&self.hand);
        if self.table.finished {
            self.finished = true;
            return;
        }
        self.hand = self.table.deal(&mut self.rng);
        self.logged = 0;
    }

    /// Applies one decision from the seat that owed it.
    fn step(&mut self, index: usize) {
        self.endings.clear();
        if self.finished {
            return;
        }
        if let Some(seat) = self.asking.pop_front() {
            // An index naming no call was refused by the arena in strict
            // mode before any table moved; what reaches here is either
            // legal or deliberately lenient, and is answered with a pass.
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
            let claimed = self.hand.pending_discard.map(|(_, tile)| tile);
            return encoding::call_index(call, claimed);
        }
        let action = self.teacher.act(&self.hand);
        Some(encoding::action_index(action))
    }
}

/// Many games of riichi, advancing together.
#[pyclass]
pub struct Arena {
    seats: Vec<Seat>,
    observations: Vec<f32>,
    /// What each deciding seat cannot see, filled in only when asked for.
    oracle: Vec<f32>,
    mask: Vec<bool>,
    /// What the opponents are holding, filled in only when asked for.
    hands: Vec<f32>,
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
    fn new(games: usize, seed: u64, bot_places: Vec<usize>) -> PyResult<Arena> {
        // A place past the fourth would be taken by nobody, and the table
        // would play without the bot its caller asked for.
        if let Some(place) = bot_places.iter().find(|place| **place >= 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bot place {place} is not one of the four places 0 to 3"
            )));
        }
        Ok(Arena {
            seats: (0..games)
                .map(|index| Seat::new(seed.wrapping_add(index as u64), &bot_places))
                .collect(),
            observations: vec![0.0; games * OBSERVATION],
            oracle: vec![0.0; games * ORACLE],
            mask: vec![false; games * ACTIONS],
            hands: vec![0.0; games * HANDS],
        })
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

    /// What each deciding seat cannot see, as float32 planes for the
    /// oracle critic: [`ORACLE_PLANES`] planes of thirty-four per game,
    /// zeros for a game that owes no decision. Only training asks for it;
    /// the network is never shown it when choosing a move.
    fn oracle<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        for (index, seat) in self.seats.iter().enumerate() {
            let slice = &mut self.oracle[index * ORACLE..(index + 1) * ORACLE];
            match seat.pending() {
                Some(wind) => encoding::oracle(&seat.hand, wind, slice),
                None => slice.fill(0.0),
            }
        }
        PyBytes::new(py, bytemuck_cast(&self.oracle))
    }

    /// What the opponents are actually holding, one answer per game, as
    /// float32: three rows of thirty-four, each a distribution over the
    /// kinds, in the same relative seat order as the observation.
    ///
    /// This is the label for the head that reads a table. It is only ever
    /// used to teach: the network is never shown it when choosing a move.
    fn opponent_hands<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        for (index, seat) in self.seats.iter().enumerate() {
            let slice = &mut self.hands[index * HANDS..(index + 1) * HANDS];
            match seat.pending() {
                Some(wind) => encoding::opponent_hands(&seat.hand, wind, slice),
                None => slice.fill(0.0),
            }
        }
        PyBytes::new(py, bytemuck_cast(&self.hands))
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
        // The teacher counts acceptance for every candidate discard, which
        // is the expensive part of the heuristic player, and the warm start
        // asks it at every seat of every game. Across the cores, as the
        // stepping is.
        use rayon::prelude::*;
        let bytes: Vec<u8> = self
            .seats
            .par_iter_mut()
            .map(|seat| {
                seat.teacher_choice()
                    .map(|index| index as u8)
                    .unwrap_or(0xFF)
            })
            .collect();
        PyBytes::new(py, &bytes)
    }

    /// Applies one action per game. Games that owe no decision ignore theirs.
    #[pyo3(signature = (actions, strict=true))]
    fn step(&mut self, actions: Vec<usize>, strict: bool) -> PyResult<()> {
        if actions.len() != self.seats.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.seats.len(),
                actions.len()
            )));
        }
        if strict {
            // Validate the complete batch first: an invalid later row must not
            // leave earlier games advanced with no corresponding training data.
            for (game, (seat, index)) in self.seats.iter().zip(&actions).enumerate() {
                let Some(wind) = seat.pending() else {
                    continue;
                };
                let valid = if seat.asking.is_empty() {
                    encoding::decode_action(&seat.hand, *index).is_some()
                } else {
                    encoding::decode_call(&seat.hand, wind, *index).is_some()
                };
                if !valid {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "illegal action {index} in game {game} for {wind:?}"
                    )));
                }
            }
        }
        // Every game is its own table, hand and generator, and stepping one
        // means running the heuristic players round to the next decision the
        // network owes, which is where the CPU time of a generation goes.
        // The GPU was sitting at ten percent waiting for this loop.
        {
            use rayon::prelude::*;
            self.seats
                .par_iter_mut()
                .zip(actions.par_iter())
                .for_each(|(seat, index)| seat.step(*index));
        }
        Ok(())
    }

    /// Every game's mjai events since they were last asked for, one list
    /// per game (see `Seat::take_events`), so a round's worth of following
    /// costs one call a step rather than one a game.
    fn mjai_all(&mut self) -> Vec<Vec<String>> {
        self.seats.iter_mut().map(Seat::take_events).collect()
    }

    /// Whether every game has finished.
    fn all_finished(&self) -> bool {
        self.seats.iter().all(|seat| seat.finished)
    }

    /// How many hands each game has finished, whether or not the caller
    /// noticed them ending. A collector that credits a hand's points to
    /// the decisions waiting on it has to see every ending: the engine
    /// forgets one as soon as the next step is taken, and a missed ending
    /// is then paid by the hand after, crediting decisions with points
    /// from a hand they took no part in. Counting here is what lets the
    /// collector check its own count against the truth.
    fn hands_done(&self) -> Vec<u32> {
        self.seats.iter().map(|seat| seat.hands_done).collect()
    }

    /// How many hands each game ended on the last step, as bytes; before
    /// the first step, how many ended while the arena was made, which
    /// native bots can do. Usually none or one, but bots can finish a
    /// whole hand before the next outside decision, so a step can end two.
    fn hand_ended<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self
            .seats
            .iter()
            .map(|seat| u8::try_from(seat.endings.len()).unwrap_or(u8::MAX))
            .collect();
        PyBytes::new(py, &bytes)
    }

    /// The change in points over the first hand each game ended on the
    /// last step, four per game, as int32 by person, not by seat; zeros
    /// where none ended. That is the hand every decision still waiting on
    /// a hand was made in. A later hand ended on the same step was dealt
    /// and played out by native bots alone, so nothing waits on it.
    fn hand_result<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut values: Vec<i32> = Vec::with_capacity(self.seats.len() * 4);
        for seat in &self.seats {
            values.extend_from_slice(&seat.endings.first().copied().unwrap_or_default());
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
#[pymodule(gil_used = true)]
fn riichi_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Arena>()?;
    module.add("TRAINING_API_VERSION", 2u32)?;
    module.add("PLANES", PLANES)?;
    // Which encoding of those planes `observations` writes; a checkpoint of
    // the engine's kind records it and is refused where it differs.
    module.add(
        "OBSERVATION_VERSION",
        riichi_core::encoding::OBSERVATION_VERSION,
    )?;
    module.add("POSITIONS", POSITIONS)?;
    module.add("ACTIONS", ACTIONS)?;
    module.add("OPPONENTS", OPPONENTS)?;
    module.add("ORACLE_PLANES", ORACLE_PLANES)?;
    module.add(
        "PLACEMENT_VALUE",
        riichi_core::encoding::PLACEMENT_VALUE.to_vec(),
    )?;
    module.add("HANDS", HANDS)?;
    module.add("PASS", PASS)?;
    Ok(())
}

#[cfg(test)]
mod ownership_tests;
