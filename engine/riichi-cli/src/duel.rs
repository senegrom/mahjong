//! One style against three of another, over the same deals.
//!
//! A change to the bot that sounds sensible is not a change that helps, and
//! average placement over a few hundred games moves by more than a real
//! improvement does. So the challenger sits in each of the four seats in
//! turn over the same deals, which cancels most of the luck, and the spread
//! across the four seatings is the error bar on the answer.

use riichi_core::bot::{Bot, Style};
use riichi_core::game::{Call, Phase};
use riichi_core::rng::Rng;
use riichi_core::table::Table;
use riichi_core::Wind;

/// What one seating came to.
struct Seating {
    placement: f64,
    score: f64,
    wins: f64,
    /// Where the challenger finished each game, in order, so the four
    /// seatings can be paired deal by deal.
    per_game: Vec<f64>,
}

/// Plays `games` whole games with `challenger` as player `chair` and
/// `defender` as the other three.
///
/// `chair` numbers the person, not the seat: seats move as the deal passes
/// round, and the point of this is to follow one player through a whole
/// game rather than whoever happens to be sitting East.
fn play_seating(
    games: usize,
    seed: u64,
    chair: usize,
    challenger: Style,
    defender: Style,
) -> Seating {
    let mut placements = 0.0;
    let mut scores = 0.0;
    let mut firsts = 0.0;
    let mut per_game = Vec::with_capacity(games);

    for game in 0..games {
        // The same seed gives the same deals whichever seat is challenged,
        // so the two styles meet the same tiles.
        let mut rng = Rng::from_seed(seed.wrapping_add(game as u64));
        let mut table = Table::new();
        let mut bots: Vec<Bot> = (0..4)
            .map(|index| {
                let style = if index == chair { challenger } else { defender };
                Bot::with_style(seed.wrapping_add(game as u64 * 4 + index as u64), style)
            })
            .collect();

        let mut guard = 0;
        while !table.finished {
            guard += 1;
            assert!(guard < 200, "a game of two rounds does not run this long");
            let mut hand = table.deal(&mut rng);
            let mut turns = 0;
            while !matches!(hand.phase, Phase::Over) {
                turns += 1;
                assert!(turns < 600, "a hand ends well before this");
                match hand.phase {
                    Phase::Draw => {
                        let _ = hand.draw();
                    }
                    Phase::Act => {
                        let who = table.player_at(hand.turn);
                        let action = bots[who].act(&hand);
                        hand.act(action).expect("the bot chose a legal action");
                    }
                    Phase::CallWindow => {
                        let answers: Vec<(Wind, Call)> = hand
                            .legal_calls()
                            .iter()
                            .map(|(seat, calls)| {
                                let who = table.player_at(*seat);
                                (*seat, bots[who].call(&hand, *seat, calls))
                            })
                            .collect();
                        hand.resolve_calls(&answers)
                            .expect("the calls were offered");
                    }
                    Phase::Over => break,
                }
            }
            table.finish(&hand);
        }

        let final_scores = table.final_scores();
        let mine = final_scores[chair];
        // Ties are broken by player number, which is arbitrary but has to be
        // consistent or two players could both be called third.
        let place = 1 + final_scores
            .iter()
            .enumerate()
            .filter(|(index, score)| **score > mine || (**score == mine && *index < chair))
            .count();
        placements += place as f64;
        per_game.push(place as f64);
        scores += table.scores[chair] as f64;
        if place == 1 {
            firsts += 1.0;
        }
    }

    Seating {
        placement: placements / games as f64,
        score: scores / games as f64,
        wins: firsts / games as f64,
        per_game,
    }
}

/// Runs the four seatings and reports what they say.
pub fn duel(games: usize, seed: u64, challenger: Style, defender: Style) {
    let mut seatings = Vec::new();
    for chair in 0..4 {
        seatings.push(play_seating(games, seed, chair, challenger, defender));
    }

    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let placements: Vec<f64> = seatings.iter().map(|row| row.placement).collect();
    let overall = mean(&placements);

    // One figure per deal: the four placements that deal produced, averaged.
    // Those are independent from deal to deal, where the four seatings are
    // not, since they are the same deals played from different chairs.
    let per_deal: Vec<f64> = (0..games)
        .map(|game| {
            mean(
                &seatings
                    .iter()
                    .map(|row| row.per_game[game])
                    .collect::<Vec<f64>>(),
            )
        })
        .collect();
    let spread = mean(&per_deal);
    let variance = per_deal
        .iter()
        .map(|value| (value - spread).powi(2))
        .sum::<f64>()
        / (per_deal.len() - 1).max(1) as f64;
    let error = (variance / per_deal.len() as f64).sqrt();

    println!("challenger: {challenger:?}");
    println!("defender:   {defender:?}");
    if challenger == defender {
        // Four identical players play identical games whichever seat is
        // called the challenger, and four placements sum to ten, so the
        // average over the four seatings is exactly 2.5 by arithmetic. It
        // is worth running as a check that the harness is not biased, but
        // it says nothing about how strong anybody is.
        println!("(the two styles are the same, so 2.5 here is arithmetic, not evidence)");
    }
    println!();
    for (chair, row) in seatings.iter().enumerate() {
        println!(
            "  player {chair}: placement {:.3}  score {:+.0}  wins {:.3}",
            row.placement, row.score, row.wins
        );
    }
    println!();
    println!(
        "placement {overall:.4} +/- {error:.4} over {} games",
        games * 4
    );
    println!(
        "score {:+.0}, wins {:.3}",
        mean(&seatings.iter().map(|row| row.score).collect::<Vec<f64>>()),
        mean(&seatings.iter().map(|row| row.wins).collect::<Vec<f64>>()),
    );

    // A challenger no stronger than the defender averages 2.5, so that is
    // what the difference is measured against. Saying how many standard
    // errors it comes to is more use than a yes or no, because a result at
    // one and a half is worth another seed rather than a decision.
    let edge = 2.5 - overall;
    if error == 0.0 {
        println!("verdict: not enough seatings to say");
        return;
    }
    let sigmas = edge / error;
    println!("difference from level: {edge:+.4} placement, {sigmas:+.1} standard errors");
    if sigmas > 2.0 {
        println!("verdict: the challenger is stronger");
    } else if sigmas < -2.0 {
        println!("verdict: the challenger is weaker");
    } else {
        println!("verdict: not settled either way, which needs more games");
    }
}
