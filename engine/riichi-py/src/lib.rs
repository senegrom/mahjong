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
    self, ACTIONS, HANDS, HIDDEN_HANDS, HIDDEN_HANDS_PLANES, OBSERVATION, OPPONENTS, ORACLE,
    ORACLE_PLANES, PASS, PLANES, POSITIONS,
};
use riichi_core::game::Action;
use riichi_core::game::{Call, Hand, Outcome, Phase};
use riichi_core::mjai;
use riichi_core::rng::Rng;
use riichi_core::search;
use riichi_core::table::Table;
use riichi_core::Wind;

/// One game, and where its next decision sits.
struct Seat {
    table: Table,
    hand: Hand,
    /// Deals the hands. Nothing hypothetical may draw from it.
    rng: Rng,
    /// Simulation must never consume the generator that deals real hands.
    search_rng: Rng,
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
        // A stream of its own, far from the dealing one, so that how much
        // a search imagines cannot change what is dealt.
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
            search_rng: Rng::from_seed(seed ^ 0x5EA5_C400),
            asking: VecDeque::new(),
            answers: Vec::new(),
            opening_scores,
            last_result: [0; 4],
            hand_just_ended: false,
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

    fn next_hand(&mut self) {
        self.flush_events();
        // Report the hand's result by person rather than by seat: the
        // seats move between hands, and a trajectory belongs to whoever was
        // sitting there. This has to happen before the deal rotates.
        let closing = scores_of(&self.hand);
        for seat in Wind::ALL {
            let place = self.table.player_at(seat);
            self.last_result[place] = closing[seat.index()] - self.opening_scores[seat.index()];
        }
        self.hand_just_ended = true;
        self.hands_done += 1;
        self.table.finish(&self.hand);
        if self.table.finished {
            self.finished = true;
            return;
        }
        self.hand = self.table.deal(&mut self.rng);
        self.opening_scores = scores_of(&self.hand);
        self.logged = 0;
    }

    /// Applies one decision from the seat that owed it.
    fn step(&mut self, index: usize) {
        self.hand_just_ended = false;
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
    /// What each deciding seat cannot see, filled in only when asked for.
    oracle: Vec<f32>,
    mask: Vec<bool>,
    /// What the opponents are holding, filled in only when asked for.
    hands: Vec<f32>,
    /// How the search has spent itself, across every game.
    searched: search::Tally,
    /// For each game, the candidates and leaves of a search whose values
    /// have been asked for and not yet given back.
    pending: Vec<Option<(Vec<Action>, search::Leaves)>>,
    /// For each game, the worlds imagined for a weighed search and not yet
    /// weighed.
    imagined: Vec<Vec<Hand>>,
    /// For each game, a lookahead the caller is playing the other seats
    /// of, with its candidates.
    lookaheads: Vec<Option<(Vec<Action>, search::Lookahead)>>,
}

/// One imagined world per live game from the beliefs given, as the
/// hidden-hand planes: the body of both forms of `Arena::imagined_hands`.
fn imagine_from<'py>(arena: &mut Arena, py: Python<'py>, beliefs: &[f32]) -> Bound<'py, PyBytes> {
    let games = arena.seats.len();
    assert_eq!(beliefs.len(), games * HANDS, "one belief per game");
    let mut planes = vec![0.0f32; games * HIDDEN_HANDS];
    for (game, seat) in arena.seats.iter_mut().enumerate() {
        let Some(wind) = seat.pending() else {
            continue;
        };
        let belief = search::Belief::from(&beliefs[game * HANDS..(game + 1) * HANDS]);
        let world = search::imagine(&seat.hand, wind, &belief, &mut seat.search_rng);
        encoding::hidden_hands(
            &world,
            wind,
            &mut planes[game * HIDDEN_HANDS..(game + 1) * HIDDEN_HANDS],
        );
    }
    PyBytes::new(py, bytemuck_cast(&planes))
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
            oracle: vec![0.0; games * ORACLE],
            mask: vec![false; games * ACTIONS],
            hands: vec![0.0; games * HANDS],
            searched: search::Tally::default(),
            pending: (0..games).map(|_| None).collect(),
            imagined: (0..games).map(|_| Vec::new()).collect(),
            lookaheads: (0..games).map(|_| None).collect(),
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
        if self.hands.len() != self.seats.len() * HANDS {
            self.hands = vec![0.0; self.seats.len() * HANDS];
        }
        for (index, seat) in self.seats.iter().enumerate() {
            let slice = &mut self.hands[index * HANDS..(index + 1) * HANDS];
            match seat.pending() {
                Some(wind) => encoding::opponent_hands(&seat.hand, wind, slice),
                None => slice.fill(0.0),
            }
        }
        PyBytes::new(py, bytemuck_cast(&self.hands))
    }

    /// Searches every live game and returns the move each came to.
    ///
    /// `ranked` holds, for every game, the network's moves in the order it
    /// prefers them, as `ACTIONS` indices; only the first `candidates` of
    /// the legal ones are played out. `beliefs` holds, for every game,
    /// three rows of thirty-four weights saying what the network takes each
    /// opponent to be holding.
    ///
    /// The first move on the list is the one to beat: the network's opinion
    /// is worth something, so a rollout has to beat it by `margin` standard
    /// errors of the paired difference before it is taken. Without that the
    /// search keeps whichever candidate the rollouts smiled on, which is
    /// worse than not searching at all.
    ///
    /// Games where nobody owes a decision come back as [`PASS`].
    #[pyo3(signature = (ranked, beliefs, worlds=10, candidates=4, margin=2.0, hurried=false))]
    fn search(
        &mut self,
        ranked: Vec<Vec<usize>>,
        beliefs: Vec<f32>,
        worlds: usize,
        candidates: usize,
        margin: f64,
        hurried: bool,
    ) -> Vec<usize> {
        let effort = search::Effort {
            worlds,
            candidates,
            turns: None,
            margin,
            hurried,
        };
        let games = self.seats.len();
        assert_eq!(ranked.len(), games, "one ranking per game");
        assert_eq!(beliefs.len(), games * HANDS, "one belief per game");

        let mut overrode = 0usize;
        let mut asked = 0usize;
        let chosen: Vec<usize> = (0..games)
            .map(|game| {
                let seat = &mut self.seats[game];
                let Some(wind) = seat.pending() else {
                    return PASS;
                };
                // Calls are not searched: the branching is small and the
                // rollout would have to answer for three other players at
                // the same moment.
                if !seat.asking.is_empty() {
                    return ranked[game].first().copied().unwrap_or(PASS);
                }
                let belief = search::Belief::from(&beliefs[game * HANDS..(game + 1) * HANDS]);
                // decode_action already refuses anything the engine will
                // not take, so the shortlist is legal by construction.
                let shortlist: Vec<Action> = ranked[game]
                    .iter()
                    .filter_map(|index| encoding::decode_action(&seat.hand, *index))
                    .collect();
                if shortlist.is_empty() {
                    return ranked[game].first().copied().unwrap_or(PASS);
                }
                asked += 1;
                match search::best(
                    &seat.hand,
                    wind,
                    &shortlist,
                    effort,
                    &belief,
                    &mut seat.search_rng,
                ) {
                    Some(judged) => {
                        let picked = action_to_index(judged.action);
                        if Some(picked) != ranked[game].first().copied() {
                            overrode += 1;
                        }
                        picked
                    }
                    None => ranked[game].first().copied().unwrap_or(PASS),
                }
            })
            .collect();
        self.searched.asked += asked;
        self.searched.overrode += overrode;
        chosen
    }

    /// The first half of a search valued by the network. For every live
    /// game, imagines `worlds` worlds, makes each of the first `candidates`
    /// moves of `ranked` in each, runs the other players round to the
    /// deciding seat's next turn, and returns the positions that result.
    ///
    /// Returns, in order: every observation concatenated, as float32; how
    /// many slots each game contributed; what each slot has already
    /// settled, as float32, which is what the hands that ended on the way
    /// moved and the placement if the game ended; and whether each slot's
    /// observation wants the network's value added to that, as bytes of 0
    /// and 1. Slots are numbered candidate-major within a game. A game
    /// that owes no decision, or owes a call, contributes no slots.
    ///
    /// Python values every slot with one forward pass and hands the numbers
    /// to [`Arena::decide`].
    #[pyo3(signature = (ranked, beliefs, worlds=200, candidates=4, hurried=true))]
    fn leaves<'py>(
        &mut self,
        py: Python<'py>,
        ranked: Vec<Vec<usize>>,
        beliefs: Vec<f32>,
        worlds: usize,
        candidates: usize,
        hurried: bool,
    ) -> (Bound<'py, PyBytes>, Vec<usize>, Vec<f32>, Vec<u8>) {
        let effort = search::Effort {
            worlds,
            candidates,
            turns: None,
            margin: 2.0,
            hurried,
        };
        let games = self.seats.len();
        assert_eq!(ranked.len(), games, "one ranking per game");
        assert_eq!(beliefs.len(), games * HANDS, "one belief per game");

        let mut observations: Vec<f32> = Vec::new();
        let mut counts = Vec::with_capacity(games);
        let mut settled: Vec<f32> = Vec::new();
        let mut wanted: Vec<u8> = Vec::new();
        for (game, ranking) in ranked.iter().enumerate() {
            self.pending[game] = None;
            let seat = &mut self.seats[game];
            let Some(wind) = seat.pending() else {
                counts.push(0);
                continue;
            };
            if !seat.asking.is_empty() {
                counts.push(0);
                continue;
            }
            let shortlist: Vec<Action> = ranking
                .iter()
                .filter_map(|index| encoding::decode_action(&seat.hand, *index))
                .take(candidates.max(1))
                .collect();
            if shortlist.is_empty() {
                counts.push(0);
                continue;
            }
            let belief = search::Belief::from(&beliefs[game * HANDS..(game + 1) * HANDS]);
            let got = search::leaves(
                &seat.hand,
                wind,
                &shortlist,
                effort,
                &belief,
                &mut seat.search_rng,
            );
            counts.push(got.counted.len());
            observations.extend_from_slice(&got.observations);
            settled.extend(got.settled.iter().map(|worth| *worth as f32));
            wanted.extend(got.wanted.iter().map(|wants| u8::from(*wants)));
            self.pending[game] = Some((shortlist, got));
        }
        (
            PyBytes::new(py, bytemuck_cast(&observations)),
            counts,
            settled,
            wanted,
        )
    }

    /// The first step of a weighed search. For every live game that owes a
    /// move, imagines `worlds` worlds from the belief's per-tile marginals
    /// and returns what each puts in the three hidden hands, as float32
    /// planes of [`HIDDEN_HANDS_PLANES`] by thirty-four, one world after
    /// another, with how many each game contributed. The marginals are
    /// only a proposal; the reader weighs the worlds and
    /// [`Arena::leaves_from`] takes the ones to keep.
    #[pyo3(signature = (beliefs, worlds=800))]
    fn imagine<'py>(
        &mut self,
        py: Python<'py>,
        beliefs: Vec<f32>,
        worlds: usize,
    ) -> (Bound<'py, PyBytes>, Vec<usize>) {
        let games = self.seats.len();
        assert_eq!(beliefs.len(), games * HANDS, "one belief per game");
        let mut planes: Vec<f32> = Vec::new();
        let mut counts = Vec::with_capacity(games);
        for (game, stored) in self.imagined.iter_mut().enumerate() {
            stored.clear();
            let seat = &mut self.seats[game];
            let Some(wind) = seat.pending() else {
                counts.push(0);
                continue;
            };
            if !seat.asking.is_empty() {
                counts.push(0);
                continue;
            }
            let belief = search::Belief::from(&beliefs[game * HANDS..(game + 1) * HANDS]);
            let imagined =
                search::imagine_worlds(&seat.hand, wind, &belief, &mut seat.search_rng, worlds);
            let start = planes.len();
            planes.resize(start + imagined.len() * HIDDEN_HANDS, 0.0);
            for (index, world) in imagined.iter().enumerate() {
                let slot = start + index * HIDDEN_HANDS;
                encoding::hidden_hands(world, wind, &mut planes[slot..slot + HIDDEN_HANDS]);
            }
            counts.push(imagined.len());
            *stored = imagined;
        }
        (PyBytes::new(py, bytemuck_cast(&planes)), counts)
    }

    /// The second step: takes, per game, which of the imagined worlds to
    /// keep and how much each counts, makes each of the first `candidates`
    /// moves of `ranked` in every kept world, and returns the leaves as
    /// [`Arena::leaves`] does. A game that imagined no worlds, or keeps
    /// none, contributes no slots and comes back as its first move.
    #[pyo3(signature = (ranked, kept, weights, candidates=4, hurried=true))]
    fn leaves_from<'py>(
        &mut self,
        py: Python<'py>,
        ranked: Vec<Vec<usize>>,
        kept: Vec<Vec<usize>>,
        weights: Vec<Vec<f32>>,
        candidates: usize,
        hurried: bool,
    ) -> (Bound<'py, PyBytes>, Vec<usize>, Vec<f32>, Vec<u8>) {
        let games = self.seats.len();
        assert_eq!(ranked.len(), games, "one ranking per game");
        assert_eq!(kept.len(), games, "one list of kept worlds per game");
        assert_eq!(weights.len(), games, "one list of weights per game");
        let effort = search::Effort {
            worlds: 0,
            candidates,
            turns: None,
            margin: 2.0,
            hurried,
        };
        let mut observations: Vec<f32> = Vec::new();
        let mut counts = Vec::with_capacity(games);
        let mut settled: Vec<f32> = Vec::new();
        let mut wanted: Vec<u8> = Vec::new();
        for (game, ranking) in ranked.iter().enumerate() {
            self.pending[game] = None;
            let imagined = std::mem::take(&mut self.imagined[game]);
            let seat = &mut self.seats[game];
            let Some(wind) = seat.pending() else {
                counts.push(0);
                continue;
            };
            if !seat.asking.is_empty() || imagined.is_empty() || kept[game].is_empty() {
                counts.push(0);
                continue;
            }
            let shortlist: Vec<Action> = ranking
                .iter()
                .filter_map(|index| encoding::decode_action(&seat.hand, *index))
                .take(candidates.max(1))
                .collect();
            if shortlist.is_empty() {
                counts.push(0);
                continue;
            }
            assert_eq!(
                kept[game].len(),
                weights[game].len(),
                "one weight per kept world"
            );
            let worlds: Vec<Hand> = kept[game]
                .iter()
                .map(|index| imagined[*index].clone())
                .collect();
            let world_weights: Vec<f64> = weights[game].iter().map(|w| *w as f64).collect();
            let got = search::leaves_from(wind, &shortlist, &worlds, &world_weights, effort);
            counts.push(got.counted.len());
            observations.extend_from_slice(&got.observations);
            settled.extend(got.settled.iter().map(|worth| *worth as f32));
            wanted.extend(got.wanted.iter().map(|wants| u8::from(*wants)));
            self.pending[game] = Some((shortlist, got));
        }
        (
            PyBytes::new(py, bytemuck_cast(&observations)),
            counts,
            settled,
            wanted,
        )
    }

    /// Begins, for every live game that owes a move, a lookahead in which
    /// the caller plays every decision on the way: the weighed search of
    /// [`Arena::leaves_from`] with the network rather than the heuristic
    /// player moving the other seats. Takes, per game, which of the
    /// imagined worlds to keep and how much each counts, makes each of the
    /// first `candidates` moves of `ranked` in every kept world, and
    /// advances every slot to the first decision somebody owes. `depth` is
    /// how many of the deciding player's own turns to act the caller plays
    /// before the position is valued; zero values the next one.
    ///
    /// Returns how many games have a lookahead running. The caller then
    /// alternates [`Arena::lookahead_owed`] and [`Arena::lookahead_apply`]
    /// until nothing is owed, and takes the leaves with
    /// [`Arena::lookahead_leaves`] for [`Arena::decide`].
    #[pyo3(signature = (ranked, kept, weights, candidates=4, depth=0, until_hand_ends=false))]
    fn lookahead_begin(
        &mut self,
        ranked: Vec<Vec<usize>>,
        kept: Vec<Vec<usize>>,
        weights: Vec<Vec<f32>>,
        candidates: usize,
        depth: usize,
        until_hand_ends: bool,
    ) -> usize {
        let games = self.seats.len();
        assert_eq!(ranked.len(), games, "one ranking per game");
        assert_eq!(kept.len(), games, "one list of kept worlds per game");
        assert_eq!(weights.len(), games, "one list of weights per game");
        let mut running = 0;
        for (game, ranking) in ranked.iter().enumerate() {
            self.pending[game] = None;
            self.lookaheads[game] = None;
            let imagined = std::mem::take(&mut self.imagined[game]);
            let seat = &self.seats[game];
            let Some(wind) = seat.pending() else {
                continue;
            };
            if !seat.asking.is_empty() || imagined.is_empty() || kept[game].is_empty() {
                continue;
            }
            let shortlist: Vec<Action> = ranking
                .iter()
                .filter_map(|index| encoding::decode_action(&seat.hand, *index))
                .take(candidates.max(1))
                .collect();
            if shortlist.is_empty() {
                continue;
            }
            assert_eq!(
                kept[game].len(),
                weights[game].len(),
                "one weight per kept world"
            );
            let worlds: Vec<Hand> = kept[game]
                .iter()
                .map(|index| imagined[*index].clone())
                .collect();
            let world_weights: Vec<f64> = weights[game].iter().map(|w| *w as f64).collect();
            let lookahead = search::Lookahead::begin(
                wind,
                &shortlist,
                &worlds,
                &world_weights,
                depth,
                until_hand_ends,
            );
            self.lookaheads[game] = Some((shortlist, lookahead));
            running += 1;
        }
        running
    }

    /// The decisions the lookaheads are waiting on: the observations, as
    /// float32, and the legality masks, as bytes of 0 and 1, of every
    /// waiting slot across every game, in game and slot order, and how
    /// many there are. Nothing waiting means the leaves are ready.
    fn lookahead_owed<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize) {
        let mut observations: Vec<f32> = Vec::new();
        let mut masks: Vec<bool> = Vec::new();
        let mut count = 0;
        for (_, lookahead) in self.lookaheads.iter().flatten() {
            count += lookahead.observe_into(&mut observations, &mut masks);
        }
        let bytes: Vec<u8> = masks.iter().map(|flag| u8::from(*flag)).collect();
        (
            PyBytes::new(py, bytemuck_cast(&observations)),
            PyBytes::new(py, &bytes),
            count,
        )
    }

    /// How many slots each game's lookahead holds, candidates times
    /// worlds, and zero for a game with none: what a caller keeping a copy
    /// of every seat's state per slot needs to lay them out.
    fn lookahead_slots(&self) -> Vec<usize> {
        self.lookaheads
            .iter()
            .map(|entry| entry.as_ref().map_or(0, |(_, lookahead)| lookahead.slots()))
            .collect()
    }

    /// What each slot's world dealt each real player, right after
    /// [`Arena::lookahead_begin`]: per game with a lookahead, per slot, four
    /// lists of mjai tile names indexed by real player. A copy of a seat's
    /// state is given these before the world's events, since the world
    /// dealt the seats the searcher cannot see hands of its own.
    fn lookahead_hands(&self) -> Vec<Vec<[Vec<String>; 4]>> {
        self.lookaheads
            .iter()
            .enumerate()
            .filter_map(|(game, entry)| entry.as_ref().map(|(_, lookahead)| (game, lookahead)))
            .map(|(game, lookahead)| {
                // As the search began, so the seats are the real hand's.
                let real = self.seats[game].table.seating();
                lookahead
                    .hands()
                    .into_iter()
                    .map(|by_seat| {
                        let mut by_player: [Vec<String>; 4] = Default::default();
                        for seat in 0..4 {
                            let player = real[seat];
                            by_player[player] = riichi_core::tile::Tile::all()
                                .flat_map(|tile| {
                                    std::iter::repeat_n(
                                        mjai::name(tile),
                                        by_seat[seat][tile.idx()] as usize,
                                    )
                                })
                                .collect();
                        }
                        by_player
                    })
                    .collect()
            })
            .collect()
    }

    /// The decisions the lookaheads are waiting on, for a network that
    /// reads Mortal's planes: per waiting slot, in the order
    /// [`Arena::lookahead_owed`] gives them, the game, the slot's index in
    /// its game, the real player who owes the decision, and the events the
    /// world invented since that slot was last asked, as mjai lines naming
    /// real players. The legality masks come with them, as bytes of 0 and
    /// 1 over our actions, so the engine's observations need not be built.
    #[allow(clippy::type_complexity)]
    fn lookahead_owed_mjai<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Bound<'py, PyBytes>,
        Vec<Vec<String>>,
    ) {
        let mut games = Vec::new();
        let mut slots = Vec::new();
        let mut players = Vec::new();
        let mut masks: Vec<u8> = Vec::new();
        let mut lines: Vec<Vec<String>> = Vec::new();
        for (game, entry) in self.lookaheads.iter_mut().enumerate() {
            let Some((_, lookahead)) = entry else {
                continue;
            };
            let real = self.seats[game].table.seating();
            let seatings = lookahead.seatings();
            let mut observations: Vec<f32> = Vec::new();
            let mut flags: Vec<bool> = Vec::new();
            lookahead.observe_into(&mut observations, &mut flags);
            masks.extend(flags.iter().map(|flag| u8::from(*flag)));
            for (slot, seat, events) in lookahead.owed_events() {
                let now: [usize; 4] = std::array::from_fn(|s| real[seatings[slot][s]]);
                games.push(game);
                slots.push(slot);
                players.push(now[seat.index()]);
                lines.push(
                    events
                        .iter()
                        .map(|(event, under)| {
                            let then: [usize; 4] = std::array::from_fn(|s| real[under[s]]);
                            event.to_json(then)
                        })
                        .collect(),
                );
            }
        }
        (games, slots, players, PyBytes::new(py, &masks), lines)
    }

    /// Answers them: one action index per waiting slot, in the order
    /// [`Arena::lookahead_owed`] gave them, and every lookahead advances
    /// to the next decision it owes.
    fn lookahead_apply(&mut self, actions: Vec<usize>) -> PyResult<()> {
        let mut offset = 0;
        for (_, lookahead) in self.lookaheads.iter_mut().flatten() {
            let waiting = lookahead.owed().len();
            if offset + waiting > actions.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "expected at least {} actions, got {}",
                    offset + waiting,
                    actions.len()
                )));
            }
            lookahead.apply(&actions[offset..offset + waiting]);
            offset += waiting;
        }
        if offset != actions.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {offset} actions, got {}",
                actions.len()
            )));
        }
        Ok(())
    }

    /// The leaves of every lookahead, as [`Arena::leaves_from`] gives
    /// them, ready for [`Arena::decide`]. A slot still waiting on a
    /// decision does not count. The lookaheads are spent.
    fn lookahead_leaves<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyBytes>, Vec<usize>, Vec<f32>, Vec<u8>) {
        let games = self.seats.len();
        let mut observations: Vec<f32> = Vec::new();
        let mut counts = Vec::with_capacity(games);
        let mut settled: Vec<f32> = Vec::new();
        let mut wanted: Vec<u8> = Vec::new();
        for game in 0..games {
            self.pending[game] = None;
            let Some((shortlist, lookahead)) = self.lookaheads[game].take() else {
                counts.push(0);
                continue;
            };
            let got = lookahead.leaves();
            counts.push(got.counted.len());
            observations.extend_from_slice(&got.observations);
            settled.extend(got.settled.iter().map(|worth| *worth as f32));
            wanted.extend(got.wanted.iter().map(|wants| u8::from(*wants)));
            self.pending[game] = Some((shortlist, got));
        }
        (
            PyBytes::new(py, bytemuck_cast(&observations)),
            counts,
            settled,
            wanted,
        )
    }

    /// For every leaf the last [`Arena::leaves_from`] or
    /// [`Arena::lookahead_leaves`] produced, whose
    /// view it is and what its world invented, in the same slot order.
    ///
    /// A network that reads Mortal's planes cannot be given the engine's
    /// observations. It is given instead the player whose state to copy —
    /// the number the follower knows, not the seat — and the mjai lines to
    /// advance that copy by. A slot with no lines is one nothing can value
    /// this way: it settled, it broke, or its world dealt a new hand and
    /// moved the seats out from under the actor numbers.
    fn leaves_mjai(&self) -> (Vec<usize>, Vec<Vec<String>>) {
        let mut players = Vec::new();
        let mut lines = Vec::new();
        for (game, seat) in self.seats.iter().enumerate() {
            let Some((_, leaves)) = &self.pending[game] else {
                continue;
            };
            // The real table's seating names the people in the hand the
            // search began in; a world that dealt on names them through
            // its own seating, composed with that.
            let real = seat.table.seating();
            for (slot, viewpoint) in leaves.viewpoints.iter().enumerate() {
                let now: [usize; 4] = std::array::from_fn(|s| real[leaves.seatings[slot][s]]);
                players.push(now[viewpoint.index()]);
                let mut told: Vec<String> = leaves.carried[slot]
                    .iter()
                    .map(|(event, under)| {
                        let then: [usize; 4] = std::array::from_fn(|s| real[under[s]]);
                        event.to_json(then)
                    })
                    .collect();
                told.extend(leaves.invented[slot].iter().map(|event| event.to_json(now)));
                lines.push(told);
            }
        }
        (players, lines)
    }

    /// One imagined world per live game, from the belief's marginals, as
    /// the hidden-hand planes the reader is shown: the negatives it learns
    /// to tell from the real hands, which [`Arena::oracle`] carries. Zeros
    /// for a game that owes nothing.
    fn imagined_hands<'py>(&mut self, py: Python<'py>, beliefs: Vec<f32>) -> Bound<'py, PyBytes> {
        imagine_from(self, py, &beliefs)
    }

    /// The same, taking the beliefs as the raw bytes of float32s rather
    /// than a Python list: a round has thousands of steps and a list of a
    /// hundred thousand floats each step cost seconds to build and read.
    fn imagined_hands_bytes<'py>(
        &mut self,
        py: Python<'py>,
        beliefs: &[u8],
    ) -> Bound<'py, PyBytes> {
        assert_eq!(beliefs.len() % 4, 0, "float32 bytes");
        let floats: Vec<f32> = beliefs
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        imagine_from(self, py, &floats)
    }

    /// The second half: takes one value per slot, in the order `leaves`
    /// gave them, and returns the move each game came to. Games that
    /// contributed no slots come back as the first entry of their ranking,
    /// or [`PASS`] when there was none.
    fn decide(&mut self, valued: Vec<f32>, margin: f64, ranked: Vec<Vec<usize>>) -> Vec<usize> {
        let games = self.seats.len();
        assert_eq!(ranked.len(), games, "one ranking per game");
        let mut offset = 0;
        let mut chosen = Vec::with_capacity(games);
        for (game, ranking) in ranked.iter().enumerate() {
            let fallback = ranking.first().copied().unwrap_or(PASS);
            let Some((candidates, leaves)) = self.pending[game].take() else {
                chosen.push(fallback);
                continue;
            };
            let slots = leaves.counted.len();
            let values: Vec<f64> = valued[offset..offset + slots]
                .iter()
                .map(|value| *value as f64)
                .collect();
            offset += slots;
            self.searched.asked += 1;
            match search::decide(&candidates, &leaves, &values, margin) {
                Some(judged) => {
                    let picked = action_to_index(judged.action);
                    if Some(picked) != ranking.first().copied() {
                        self.searched.overrode += 1;
                    }
                    chosen.push(picked);
                }
                None => chosen.push(fallback),
            }
        }
        assert_eq!(offset, valued.len(), "every value was spent");
        chosen
    }

    /// How many decisions the search was asked about, and how many of them
    /// it changed. A search that never changes anything is a no-op dressed
    /// up as a calculation; one that changes everything has a margin that
    /// is not doing its job.
    fn search_tally(&self) -> (usize, usize) {
        (self.searched.asked, self.searched.overrode)
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

    /// One game's mjai events since they were last asked for, as JSON
    /// lines: start_game first, then every event of every hand under the
    /// seating it was dealt with, end_game last. What a player state on
    /// the Python side follows to describe the game the way Mortal's own
    /// bot sees it.
    fn mjai(&mut self, game: usize) -> Vec<String> {
        let seat = &mut self.seats[game];
        seat.flush_events();
        if seat.finished && !seat.ended {
            seat.events.push(mjai::Event::EndGame.to_json([0, 1, 2, 3]));
            seat.ended = true;
        }
        std::mem::take(&mut seat.events)
    }

    /// The same for every game at once, one list per game, so a round's
    /// worth of following costs one call a step rather than one a game.
    fn mjai_all(&mut self) -> Vec<Vec<String>> {
        (0..self.seats.len()).map(|game| self.mjai(game)).collect()
    }

    /// Whether every game has finished.
    fn all_finished(&self) -> bool {
        self.seats.iter().all(|seat| seat.finished)
    }

    /// Games where a hand ended on the last step, as bytes of 0 and 1.
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
#[pymodule(gil_used = true)]
fn riichi_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Arena>()?;
    module.add("TRAINING_API_VERSION", 2u32)?;
    module.add("PLANES", PLANES)?;
    module.add("POSITIONS", POSITIONS)?;
    module.add("OBSERVATION", OBSERVATION)?;
    module.add("ACTIONS", ACTIONS)?;
    module.add("OPPONENTS", OPPONENTS)?;
    module.add("ORACLE_PLANES", ORACLE_PLANES)?;
    module.add("HIDDEN_HANDS_PLANES", HIDDEN_HANDS_PLANES)?;
    module.add("POINTS_PER_UNIT", riichi_core::encoding::POINTS_PER_UNIT)?;
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
