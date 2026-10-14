use super::{seeds_by_table, Seat};

#[test]
fn a_seed_deals_its_tables_from_consecutive_seeds() {
    assert_eq!(seeds_by_table(3, 10, None), Ok(vec![10, 11, 12]));
    // As the arena always counted: past the largest seed it wraps to nought.
    assert_eq!(seeds_by_table(2, u64::MAX, None), Ok(vec![u64::MAX, 0]));
    assert_eq!(seeds_by_table(0, 5, None), Ok(vec![]));
}

#[test]
fn a_list_names_every_tables_seed_and_may_repeat_one() {
    assert_eq!(seeds_by_table(3, 0, Some(vec![7, 7, 2])), Ok(vec![7, 7, 2]));
    assert!(seeds_by_table(3, 0, Some(vec![7, 7])).is_err());
    assert!(seeds_by_table(1, 0, Some(vec![7, 7])).is_err());
    // A seed and a list together would leave it unclear which deals.
    assert!(seeds_by_table(2, 9, Some(vec![7, 7])).is_err());
}

/// Every move of a game, played by the heuristic player, and every event
/// it wrote: what makes two tables the same game.
fn played(seed: u64) -> (Vec<String>, Vec<usize>) {
    let mut game = Seat::new(seed, &[]);
    let mut moves = Vec::new();
    let mut events = Vec::new();
    let mut steps = 0;
    while !game.finished && steps < 8000 {
        let action = game
            .teacher_choice()
            .expect("an unfinished game owes a decision");
        moves.push(action);
        game.step(action);
        events.extend(game.take_events());
        steps += 1;
    }
    assert!(game.finished, "the game finishes");
    events.extend(game.take_events());
    (events, moves)
}

/// Tables that share a seed are dealt the same hands and, asked the same
/// moves, play the same game to the end, which is what lets one arena hold
/// a deal's four seatings; another seed deals another game.
#[test]
fn tables_that_share_a_seed_play_the_same_game() {
    let (events, moves) = played(9_100_000);
    assert!(moves.len() > 100);
    assert_eq!(played(9_100_000), (events.clone(), moves));
    let (other, _) = played(9_100_001);
    assert_ne!(other, events);
}
