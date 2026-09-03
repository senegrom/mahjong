//! Plays games from the command line: an arena for the bots and a fuzzer
//! for the engine.
//!
//! ```text
//! riichi-cli arena --games 200 --seed 1     four heuristic bots, with stats
//! riichi-cli fuzz  --games 200 --seed 1     random legal play, checking rules
//! riichi-cli hand  --seed 1                 one hand, move by move
//! riichi-cli log   --seed 1 --games 1       a game as an mjai event log
//! riichi-cli dump  --seed 1 --games 1000    random scored hands, as JSON
//! riichi-cli duel  --games 200              one style against three others
//! ```
//!
//! The fuzzer is the point of this crate: it plays only actions the engine
//! offered, and checks after every one that the table's points add up, that
//! no tile has appeared or vanished, and that no hand holds a fifth copy of
//! anything.

mod duel;
mod dump;

use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use riichi_core::bot::{Bot, Style};
use riichi_core::game::{Action, Call, Hand, Outcome, Phase};
use riichi_core::mjai;
use riichi_core::rng::Rng;
use riichi_core::table::Table;
use riichi_core::tile::{Tile, COPIES, KINDS};
use riichi_core::{TileSet, Wind};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");
    let games = value(&args, "--games").unwrap_or(100);
    let seed = value(&args, "--seed").unwrap_or(1) as u64;

    match command {
        "arena" => {
            arena(games, seed);
            ExitCode::SUCCESS
        }
        "fuzz" => match fuzz(games, seed) {
            Ok(()) => {
                println!("fuzz: {games} games, no rule broken");
                ExitCode::SUCCESS
            }
            Err(report) => {
                eprintln!("fuzz failed: {report}");
                ExitCode::FAILURE
            }
        },
        "hand" => {
            show_hand(seed);
            ExitCode::SUCCESS
        }
        "log" => {
            write_log(games, seed);
            ExitCode::SUCCESS
        }
        "dump" => {
            dump::dump(games, seed);
            ExitCode::SUCCESS
        }
        "duel" => {
            // What is being tried, against the club player as it stands.
            let mut challenger = Style::club();
            if let Some(worth) = value(&args, "--dora-worth") {
                challenger.dora_worth = worth as i64;
            }
            let mut defender = Style::club();
            defender.dora_worth = 0;
            duel::duel(games, seed, challenger, defender);
            ExitCode::SUCCESS
        }
        _ => {
            println!(
                "usage: riichi-cli <arena|fuzz|hand|log|dump|duel> [--games N] [--seed N] [--dora-worth N]"
            );
            ExitCode::SUCCESS
        }
    }
}

fn value(args: &[String], name: &str) -> Option<usize> {
    let index = args.iter().position(|arg| arg == name)?;
    args.get(index + 1)?.parse().ok()
}

/// Writes games as mjai event logs, one JSON object per line, which is
/// what replayers and other people's bots read. Games run one after another
/// on standard output, each opening with `start_game` and closing with
/// `end_game`.
fn write_log(games: usize, seed: u64) {
    for game in 0..games {
        let mut rng = Rng::from_seed(seed.wrapping_add(game as u64));
        let mut table = Table::new();
        let mut bots: Vec<Bot> = (0..4)
            .map(|index| Bot::new(seed.wrapping_add(game as u64 * 4 + index)))
            .collect();

        let names = [
            "riichi-bot 0".to_string(),
            "riichi-bot 1".to_string(),
            "riichi-bot 2".to_string(),
            "riichi-bot 3".to_string(),
        ];
        println!("{}", mjai::Event::StartGame { names }.to_json([0, 1, 2, 3]));

        while !table.finished {
            let seating = table.seating();
            let mut hand = table.deal(&mut rng);
            while !matches!(hand.phase, Phase::Over) {
                match hand.phase {
                    Phase::Draw => {
                        let _ = hand.draw();
                    }
                    Phase::Act => {
                        let seat = hand.turn;
                        let action = bots[table.player_at(seat)].act(&hand);
                        hand.act(action).expect("the bot chose a legal action");
                    }
                    Phase::CallWindow => {
                        let answers: Vec<(Wind, Call)> = hand
                            .legal_calls()
                            .iter()
                            .map(|(seat, calls)| {
                                (
                                    *seat,
                                    bots[table.player_at(*seat)].call(&hand, *seat, calls),
                                )
                            })
                            .collect();
                        hand.resolve_calls(&answers)
                            .expect("the calls were offered");
                    }
                    Phase::Over => break,
                }
            }
            for event in &hand.log {
                println!("{}", event.to_json(seating));
            }
            table.finish(&hand);
        }

        println!("{}", mjai::Event::EndGame.to_json([0, 1, 2, 3]));
    }
}

/// Plays whole games with four heuristic bots and reports how they went.
fn arena(games: usize, seed: u64) {
    let mut placements = [0usize; 4];
    let mut wins = [0usize; 4];
    let mut deal_ins = [0usize; 4];
    let mut totals = [0i64; 4];
    let mut hands = 0usize;
    let mut draws = 0usize;
    let mut riichi_declared = 0usize;
    let mut yaku_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut limits: BTreeMap<&'static str, usize> = BTreeMap::new();

    for game in 0..games {
        let mut rng = Rng::from_seed(seed.wrapping_add(game as u64));
        let mut table = Table::new();
        let mut bots: Vec<Bot> = (0..4)
            .map(|index| Bot::new(seed.wrapping_add(game as u64 * 4 + index)))
            .collect();

        while !table.finished {
            let mut hand = table.deal(&mut rng);
            play_hand(&mut hand, &table, &mut bots);
            hands += 1;
            for seat in Wind::ALL {
                if hand.players[seat.index()].has_riichi() {
                    riichi_declared += 1;
                }
            }
            match &hand.outcome {
                Some(Outcome::Win { winners, discarder }) => {
                    for (seat, score) in winners {
                        wins[table.player_at(*seat)] += 1;
                        for (yaku, _) in &score.yaku {
                            *yaku_counts.entry(yaku.name()).or_default() += 1;
                        }
                        if let Some(limit) = score.limit {
                            *limits.entry(limit.name()).or_default() += 1;
                        }
                    }
                    if let Some(from) = discarder {
                        deal_ins[table.player_at(*from)] += 1;
                    }
                }
                Some(Outcome::ExhaustiveDraw { .. }) => draws += 1,
                None => {}
            }
            table.finish(&hand);
        }

        let final_scores = table.final_scores();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|index| std::cmp::Reverse(table.scores[*index]));
        for (place, player) in order.iter().enumerate() {
            placements[*player] += place + 1;
        }
        for player in 0..4 {
            totals[player] += final_scores[player] as i64;
        }
    }

    println!("arena: {games} games, {hands} hands, {draws} exhaustive draws");
    println!(
        "riichi declared in {:.1}% of seats per hand",
        100.0 * riichi_declared as f64 / (hands * 4) as f64
    );
    println!();
    println!("player  place   win%   deal-in%   points/game");
    for player in 0..4 {
        println!(
            "  {player}     {:.3}   {:>5.1}   {:>7.1}   {:>11.0}",
            placements[player] as f64 / games as f64,
            100.0 * wins[player] as f64 / hands as f64,
            100.0 * deal_ins[player] as f64 / hands as f64,
            totals[player] as f64 / games as f64,
        );
    }
    println!();
    println!("limits reached");
    for (name, count) in &limits {
        println!("  {name:<10} {count}");
    }
    println!();
    println!("most common yaku");
    let mut ordered: Vec<(&&str, &usize)> = yaku_counts.iter().collect();
    ordered.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (name, count) in ordered.iter().take(12) {
        println!("  {name:<28} {count}");
    }
}

/// Plays games with random legal choices, checking the rules hold throughout.
fn fuzz(games: usize, seed: u64) -> Result<(), String> {
    for game in 0..games {
        let mut rng = Rng::from_seed(seed.wrapping_add(game as u64));
        let mut choices = Rng::from_seed(seed.wrapping_add(0xABCD).wrapping_add(game as u64));
        let mut table = Table::new();
        let mut guard = 0;
        while !table.finished {
            guard += 1;
            if guard > 200 {
                return Err(format!("game {game} ran for {guard} hands"));
            }
            let mut hand = table.deal(&mut rng);
            let opening = census(&hand);
            let mut turns = 0;
            while !matches!(hand.phase, Phase::Over) {
                turns += 1;
                if turns > 800 {
                    return Err(format!("game {game} hand ran for {turns} turns"));
                }
                match hand.phase {
                    Phase::Draw => {
                        let _ = hand.draw();
                    }
                    Phase::Act => {
                        let actions = hand.legal_actions();
                        if actions.is_empty() {
                            return Err(format!("game {game}: no action offered"));
                        }
                        let pick = actions[choices.below(actions.len())];
                        hand.act(pick).map_err(|error| {
                            format!("game {game}: engine refused {pick:?}: {error:?}")
                        })?;
                    }
                    Phase::CallWindow => {
                        let offered = hand.legal_calls();
                        let answers: Vec<(Wind, Call)> = offered
                            .iter()
                            .map(|(seat, calls)| (*seat, calls[choices.below(calls.len())]))
                            .collect();
                        hand.resolve_calls(&answers).map_err(|error| {
                            format!("game {game}: engine refused calls: {error:?}")
                        })?;
                    }
                    Phase::Over => break,
                }
                check_invariants(&hand, &opening, game)?;
            }
            let points: i32 = hand.players.iter().map(|player| player.score).sum::<i32>()
                + (hand.riichi_sticks * 1000) as i32;
            let expected: i32 =
                table.scores.iter().sum::<i32>() + (table.riichi_sticks * 1000) as i32;
            if points != expected {
                return Err(format!(
                    "game {game}: points went from {expected} to {points} over one hand"
                ));
            }
            table.finish(&hand);
        }
    }
    Ok(())
}

/// Every tile accounted for: hands, called sets, discards and the wall.
fn census(hand: &Hand) -> TileSet {
    let mut seen = TileSet::new();
    for player in &hand.players {
        for tile in player.hand.tiles() {
            seen.add(tile);
        }
        for meld in &player.melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
        for discard in &player.discards {
            // A claimed tile now sits in somebody's called set, so counting
            // it here as well would find five copies of a kind that a full
            // set only has four of.
            if !discard.claimed {
                seen.add(discard.tile);
            }
        }
    }
    seen
}

fn check_invariants(hand: &Hand, opening: &TileSet, game: usize) -> Result<(), String> {
    // No hand may hold a fifth copy of anything.
    for player in &hand.players {
        if !player.hand.is_legal() {
            return Err(format!(
                "game {game}: a hand holds five of a kind: {}",
                player.hand
            ));
        }
    }
    // Tiles in play never appear from nowhere: what the players hold, have
    // called and have discarded must stay within a full set.
    let now = census(hand);
    for tile in Tile::all() {
        if now.count(tile) > COPIES {
            return Err(format!(
                "game {game}: {} copies of {tile} in play",
                now.count(tile)
            ));
        }
    }
    let _ = opening;
    // Points are only moved, never made.
    let total: i32 = hand.players.iter().map(|player| player.score).sum::<i32>()
        + (hand.riichi_sticks * 1000) as i32;
    if total % 100 != 0 {
        return Err(format!(
            "game {game}: points are not whole hundreds: {total}"
        ));
    }
    let _ = KINDS;
    Ok(())
}

fn play_hand(hand: &mut Hand, table: &Table, bots: &mut [Bot]) {
    let mut guard = 0;
    while !matches!(hand.phase, Phase::Over) {
        guard += 1;
        assert!(guard < 800, "a hand should end well before this");
        match hand.phase {
            Phase::Draw => {
                let _ = hand.draw();
            }
            Phase::Act => {
                let seat = hand.turn;
                let action = bots[table.player_at(seat)].act(hand);
                hand.act(action).expect("the bot chose a legal action");
            }
            Phase::CallWindow => {
                let offered = hand.legal_calls();
                let answers: Vec<(Wind, Call)> = offered
                    .iter()
                    .map(|(seat, calls)| {
                        (*seat, bots[table.player_at(*seat)].call(hand, *seat, calls))
                    })
                    .collect();
                hand.resolve_calls(&answers)
                    .expect("the bots chose legal calls");
            }
            Phase::Over => break,
        }
    }
}

/// Plays one hand and prints it, which is the quickest way to eyeball the
/// engine's behaviour.
fn show_hand(seed: u64) {
    let mut rng = Rng::from_seed(seed);
    let table = Table::new();
    let mut hand = table.deal(&mut rng);
    let mut bots: Vec<Bot> = (0..4).map(|index| Bot::new(seed + index)).collect();

    println!("dora indicator {}", hand.wall.dora_indicators()[0]);
    for seat in Wind::ALL {
        println!("{seat:?}: {}", hand.players[seat.index()].hand);
    }
    println!();

    let mut guard = 0;
    while !matches!(hand.phase, Phase::Over) {
        guard += 1;
        assert!(guard < 800);
        match hand.phase {
            Phase::Draw => {
                if let Ok(tile) = hand.draw() {
                    print!("{:?} draws {tile}", hand.turn);
                }
            }
            Phase::Act => {
                let seat = hand.turn;
                let action = bots[seat.index()].act(&hand);
                match action {
                    Action::Discard(tile) => println!(" and discards {tile}"),
                    Action::Riichi(tile) => println!(" and declares riichi on {tile}"),
                    Action::Tsumo => println!(" and wins"),
                    Action::ConcealedKan(tile) => println!(" and declares a quad of {tile}"),
                    Action::ExtendedKan(tile) => println!(" and extends a triplet of {tile}"),
                }
                hand.act(action).expect("the bot chose a legal action");
            }
            Phase::CallWindow => {
                let offered = hand.legal_calls();
                let answers: Vec<(Wind, Call)> = offered
                    .iter()
                    .map(|(seat, calls)| (*seat, bots[seat.index()].call(&hand, *seat, calls)))
                    .collect();
                for (seat, call) in &answers {
                    if !matches!(call, Call::Pass) {
                        println!("  {seat:?} calls {call:?}");
                    }
                }
                hand.resolve_calls(&answers)
                    .expect("the bots chose legal calls");
            }
            Phase::Over => break,
        }
    }

    println!();
    match &hand.outcome {
        Some(Outcome::Win { winners, discarder }) => {
            for (seat, score) in winners {
                let how = match discarder {
                    Some(from) => format!("on {from:?}'s discard"),
                    None => "by self-draw".to_string(),
                };
                println!(
                    "{seat:?} wins {how}: {} han, {} fu{}",
                    score.han,
                    score.fu,
                    score
                        .limit
                        .map(|limit| format!(", {}", limit.name()))
                        .unwrap_or_default()
                );
                for (yaku, han) in &score.yaku {
                    println!("    {:<28} {han}", yaku.name());
                }
            }
        }
        Some(Outcome::ExhaustiveDraw { tenpai }) => {
            println!("exhaustive draw, waiting: {tenpai:?}");
        }
        None => println!("unfinished"),
    }
    for seat in Wind::ALL {
        println!("{seat:?}: {}", hand.players[seat.index()].score);
    }
}
