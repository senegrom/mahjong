//! A full game: the east round, the south round, and the final scores.
//!
//! Follows EMA 2025 sections 3.1, 3.4.4, 3.4.5, 3.5, 3.7 and 3.7.1. Four
//! people sit at a table and keep their places; the seat winds move around
//! them between hands, so this module tracks who is currently East and maps
//! the four seats onto the four people.
//!
//! There is no extension round, no agari-yame and no bankruptcy: the game
//! runs to the end of the south round, and a score below zero does not stop
//! it (sections 3.7 and 4.1.4).

use crate::game::{Hand, Outcome};
use crate::rng::Rng;
use crate::Wind;

/// Points every player starts a game with (EMA section 1.4).
pub const STARTING_SCORE: i32 = 30_000;
/// The score every player's total is measured against at the end.
pub const RETURN_SCORE: i32 = 30_000;
/// The winner bonus, best to worst (EMA section 3.7.1).
pub const UMA: [i32; 4] = [15_000, 5_000, -5_000, -15_000];

/// The four people at a table, and where the game has got to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Table {
    /// Points, by player, in seating order around the table.
    pub scores: [i32; 4],
    /// Which player is currently East.
    pub dealer: usize,
    /// Which player was East when the game began, i.e. who marks the round.
    pub first_dealer: usize,
    /// The round wind: east, then south.
    pub round: Wind,
    /// Counters on the table (EMA section 3.4.4).
    pub counters: u32,
    /// Riichi bets left on the table from earlier hands.
    pub riichi_sticks: u32,
    /// How many hands have been played.
    pub hands_played: usize,
    /// Whether the game is over.
    pub finished: bool,
}

impl Table {
    /// A new game with everyone on the starting score.
    pub fn new() -> Table {
        Table {
            scores: [STARTING_SCORE; 4],
            dealer: 0,
            first_dealer: 0,
            round: Wind::East,
            counters: 0,
            riichi_sticks: 0,
            hands_played: 0,
            finished: false,
        }
    }

    /// The seat wind a player currently holds.
    pub fn seat_of(&self, player: usize) -> Wind {
        Wind::ALL[(player + 4 - self.dealer) % 4]
    }

    /// Which player is sitting in a given seat.
    pub fn player_at(&self, seat: Wind) -> usize {
        (self.dealer + seat.index()) % 4
    }

    /// Deals the next hand, with the seats mapped onto the players.
    pub fn deal(&self, rng: &mut Rng) -> Hand {
        let mut seat_scores = [0; 4];
        for seat in Wind::ALL {
            seat_scores[seat.index()] = self.scores[self.player_at(seat)];
        }
        Hand::deal(
            rng,
            self.round,
            self.counters,
            self.riichi_sticks,
            seat_scores,
        )
    }

    /// Takes a finished hand: moves the points home, decides whether the
    /// dealer keeps the deal, and advances the round or ends the game.
    pub fn finish(&mut self, hand: &Hand) {
        for seat in Wind::ALL {
            self.scores[self.player_at(seat)] = hand.players[seat.index()].score;
        }
        self.riichi_sticks = hand.riichi_sticks;
        self.hands_played += 1;

        let dealer_keeps = match &hand.outcome {
            // East stays East by winning, or by being one of several winners
            // (EMA section 3.4.5).
            Some(Outcome::Win { winners, .. }) => {
                winners.iter().any(|(seat, _)| matches!(seat, Wind::East))
            }
            // East stays East when waiting at an exhaustive draw.
            Some(Outcome::ExhaustiveDraw { tenpai }) => tenpai.contains(&Wind::East),
            None => return,
        };

        // A counter is placed after a dealer win and after every exhaustive
        // draw, and cleared when somebody else wins (EMA section 3.4.4).
        match &hand.outcome {
            Some(Outcome::Win { winners, .. }) => {
                if winners.iter().any(|(seat, _)| matches!(seat, Wind::East)) {
                    self.counters += 1;
                } else {
                    self.counters = 0;
                }
            }
            Some(Outcome::ExhaustiveDraw { .. }) => self.counters += 1,
            None => {}
        }

        if !dealer_keeps {
            self.dealer = (self.dealer + 1) % 4;
            // The round turns over when the deal comes back to the player who
            // began the game as East (EMA section 3.5).
            if self.dealer == self.first_dealer {
                match self.round {
                    Wind::East => self.round = Wind::South,
                    _ => self.finished = true,
                }
            }
        }
    }

    /// The final scores: points less the return score, plus uma, with tied
    /// places sharing the bonus for the places they cover, and any riichi
    /// bets left on the table going to the winner (EMA sections 3.7, 3.7.1).
    pub fn final_scores(&self) -> [i32; 4] {
        let mut raw: [i32; 4] = self.scores;

        // Bets still on the table go to the leader, split on a tie with the
        // decimals dropped.
        let leftover = (self.riichi_sticks * 1000) as i32;
        if leftover > 0 {
            let best = *raw.iter().max().expect("four players");
            let leaders: Vec<usize> = (0..4).filter(|index| raw[*index] == best).collect();
            let share = leftover / leaders.len() as i32;
            for index in leaders {
                raw[index] += share;
            }
        }

        // Places, with ties sharing the pooled uma of the places they cover.
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|index| core::cmp::Reverse(raw[*index]));

        let mut result = [0; 4];
        let mut place = 0;
        while place < 4 {
            let score = raw[order[place]];
            let tied: Vec<usize> = order
                .iter()
                .copied()
                .filter(|index| raw[*index] == score)
                .collect();
            let pooled: i32 = (place..place + tied.len()).map(|slot| UMA[slot]).sum();
            let share = pooled / tied.len() as i32;
            for index in &tied {
                result[*index] = raw[*index] - RETURN_SCORE + share;
            }
            place += tied.len();
        }
        result
    }
}

impl Default for Table {
    fn default() -> Table {
        Table::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Action, Phase};

    fn played_out(table: &Table, rng: &mut Rng) -> Hand {
        let mut hand = table.deal(rng);
        let mut guard = 0;
        while !matches!(hand.phase, Phase::Over) {
            guard += 1;
            assert!(guard < 500, "a hand should end well before this");
            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    let actions = hand.legal_actions();
                    let tile = actions
                        .iter()
                        .find_map(|action| match action {
                            Action::Discard(tile) => Some(*tile),
                            _ => None,
                        })
                        .expect("a player can always discard");
                    hand.act(Action::Discard(tile)).unwrap();
                }
                Phase::CallWindow => hand.resolve_calls(&[]).unwrap(),
                Phase::Over => break,
            }
        }
        hand
    }

    /// EMA 2025 section 2.1: the seats move around the table between hands.
    #[test]
    fn seats_follow_the_dealer() {
        let mut table = Table::new();
        assert_eq!(table.seat_of(0), Wind::East);
        assert_eq!(table.seat_of(1), Wind::South);
        table.dealer = 1;
        assert_eq!(table.seat_of(1), Wind::East);
        assert_eq!(table.seat_of(0), Wind::North);
        assert_eq!(table.player_at(Wind::East), 1);
    }

    /// EMA 2025 section 3.4.5: the dealer keeps the deal when waiting at an
    /// exhaustive draw, and a counter goes on the table either way.
    #[test]
    fn a_waiting_dealer_keeps_the_deal() {
        let mut table = Table::new();
        let mut rng = Rng::from_seed(1);
        let mut hand = table.deal(&mut rng);
        hand.outcome = Some(Outcome::ExhaustiveDraw {
            tenpai: vec![Wind::East],
        });
        table.finish(&hand);
        assert_eq!(table.dealer, 0);
        assert_eq!(table.counters, 1);

        let mut hand = table.deal(&mut rng);
        hand.outcome = Some(Outcome::ExhaustiveDraw {
            tenpai: vec![Wind::South],
        });
        table.finish(&hand);
        assert_eq!(table.dealer, 1, "a noten dealer passes the deal on");
        assert_eq!(table.counters, 2, "a draw always adds a counter");
    }

    /// EMA 2025 section 3.4.4: counters are cleared when somebody other than
    /// the dealer wins.
    #[test]
    fn a_non_dealer_win_clears_the_counters() {
        let mut table = Table::new();
        table.counters = 3;
        let mut rng = Rng::from_seed(2);
        let mut hand = table.deal(&mut rng);
        hand.outcome = Some(Outcome::ExhaustiveDraw {
            tenpai: vec![Wind::South],
        });
        // Stand in a win by South rather than a draw.
        hand.outcome = Some(Outcome::Win {
            winners: Vec::new(),
            discarder: None,
        });
        table.finish(&hand);
        assert_eq!(table.counters, 0);
        assert_eq!(table.dealer, 1);
    }

    /// EMA 2025 section 3.5: the south round begins when the deal returns to
    /// the player who started as East, and the game ends the same way.
    #[test]
    fn two_rounds_make_a_game() {
        let mut table = Table::new();
        let mut rng = Rng::from_seed(3);
        for _ in 0..4 {
            let mut hand = table.deal(&mut rng);
            hand.outcome = Some(Outcome::Win {
                winners: Vec::new(),
                discarder: None,
            });
            table.finish(&hand);
        }
        assert_eq!(table.round, Wind::South);
        assert!(!table.finished);
        for _ in 0..4 {
            let mut hand = table.deal(&mut rng);
            hand.outcome = Some(Outcome::Win {
                winners: Vec::new(),
                discarder: None,
            });
            table.finish(&hand);
        }
        assert!(table.finished);
        assert_eq!(table.hands_played, 8);
    }

    /// EMA 2025 section 3.7.1: uma is 15,000, 5,000, -5,000 and -15,000, and
    /// tied places share the bonus for the places they cover.
    #[test]
    fn uma_is_applied_to_the_final_scores() {
        let mut table = Table::new();
        table.scores = [40_000, 30_000, 25_000, 25_000];
        let final_scores = table.final_scores();
        assert_eq!(final_scores[0], 40_000 - 30_000 + 15_000);
        assert_eq!(
            final_scores[1], 5_000,
            "exactly the return score, so only uma"
        );
        // Two players tied for third share the third and fourth bonuses.
        assert_eq!(final_scores[2], 25_000 - 30_000 - 10_000);
        assert_eq!(final_scores[3], 25_000 - 30_000 - 10_000);
        assert_eq!(final_scores.iter().sum::<i32>(), 0);
    }

    /// EMA 2025 section 3.7.1, example 1: two players tied for first each
    /// take half of the pooled first and second bonuses.
    #[test]
    fn a_tie_for_first_splits_twenty_thousand() {
        let mut table = Table::new();
        table.scores = [35_000, 35_000, 20_000, 30_000];
        let final_scores = table.final_scores();
        assert_eq!(final_scores[0], 35_000 - 30_000 + 10_000);
        assert_eq!(final_scores[1], 35_000 - 30_000 + 10_000);
        assert_eq!(final_scores[3], 30_000 - 30_000 - 5_000);
        assert_eq!(final_scores[2], 20_000 - 30_000 - 15_000);
    }

    /// EMA 2025 section 3.7: bets left on the table go to the winner.
    #[test]
    fn leftover_riichi_bets_go_to_the_leader() {
        let mut table = Table::new();
        table.scores = [40_000, 30_000, 20_000, 29_000];
        table.riichi_sticks = 1;
        let final_scores = table.final_scores();
        assert_eq!(final_scores[0], 41_000 - 30_000 + 15_000);
    }

    #[test]
    fn a_whole_game_can_be_played_out() {
        let mut table = Table::new();
        let mut rng = Rng::from_seed(20260902);
        let mut guard = 0;
        while !table.finished {
            guard += 1;
            assert!(guard < 100, "a game of two rounds should not run this long");
            let hand = played_out(&table, &mut rng);
            table.finish(&hand);
        }
        let total: i32 = table.scores.iter().sum::<i32>() + (table.riichi_sticks * 1000) as i32;
        assert_eq!(total, 120_000, "the table's points are conserved");
        assert!(table.hands_played >= 8);
        assert_eq!(table.final_scores().iter().sum::<i32>(), 0);
    }
}
