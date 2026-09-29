use super::{places_by_table, Seat};

#[test]
fn every_exported_decision_belongs_to_an_external_player() {
    let mut saw_rotation = false;
    let mut saw_multiple_claimants = false;
    // Every mix of native bots and external players, including an all-bot
    // game which must finish entirely within settlement.
    for mask in 0..16 {
        let bots: Vec<usize> = (0..4).filter(|player| mask & (1 << player) != 0).collect();
        for seed in [1, 81] {
            let mut game = Seat::new(seed, &bots);
            let mut steps = 0;
            while !game.finished && steps < 8000 {
                let wind = game.pending().expect("unfinished game must owe a decision");
                assert!(
                    !game.is_bot(wind),
                    "native bot escaped settlement: mask={mask}, seed={seed}, wind={wind:?}"
                );
                saw_rotation |= game.table.seating() != [0, 1, 2, 3];
                saw_multiple_claimants |= game.asking.len() + game.answers.len() > 1;
                let action = game
                    .teacher_choice()
                    .expect("external player has a legal move");
                game.step(action);
                steps += 1;
            }
            assert!(
                game.finished,
                "game did not complete: mask={mask}, seed={seed}"
            );
        }
    }
    assert!(saw_rotation);
    assert!(saw_multiple_claimants);
}

/// What every hand of a game moved, by person, as the game's own mjai log
/// has it: the scores after the hand's last win or draw, less the scores it
/// was dealt with. The log is written from the rules engine's events, apart
/// from the bookkeeping under test.
fn results_in_log(game: &Seat) -> Vec<[i32; 4]> {
    fn scores(line: &str) -> [i32; 4] {
        let at = line.find("\"scores\":[").expect("the event carries scores") + 10;
        let end = at + line[at..].find(']').expect("a closed list");
        let values: Vec<i32> = line[at..end]
            .split(',')
            .map(|value| value.parse().expect("a whole number"))
            .collect();
        values.try_into().expect("four scores")
    }
    let mut results = Vec::new();
    let mut dealt = None;
    let mut after = None;
    for line in &game.events {
        if line.contains("\"type\":\"start_kyoku\"") {
            dealt = Some(scores(line));
        } else if line.contains("\"type\":\"hora\"") || line.contains("\"type\":\"ryukyoku\"") {
            after = Some(scores(line));
        } else if line.contains("\"type\":\"end_kyoku\"") {
            let (dealt, after) = (dealt.take().unwrap(), after.take().unwrap());
            results.push(std::array::from_fn(|person| after[person] - dealt[person]));
        }
    }
    results
}

/// Four native bots play a whole game while it is made, so every hand
/// ends before the first step, and each is reported with its own result
/// rather than as one flag and the last hand's points.
#[test]
fn every_hand_that_ends_while_a_game_is_made_is_reported() {
    for seed in [3, 5] {
        let game = Seat::new(seed, &[0, 1, 2, 3]);
        assert!(game.finished);
        assert!(game.hands_done >= 8, "a game of two rounds");
        assert_eq!(game.endings.len(), game.hands_done as usize);
        assert_eq!(game.endings, results_in_log(&game));
    }
}

/// With native bots in three places, a step reports every hand it ended,
/// the first of them being the hand the outside decision was made in, and
/// the next step forgets them.
#[test]
fn each_step_reports_the_hands_it_ended() {
    for (external, seed) in [(0, 11), (2, 12)] {
        let bots: Vec<usize> = (0..4).filter(|place| *place != external).collect();
        let mut game = Seat::new(seed, &bots);
        let mut reported: Vec<[i32; 4]> = game.endings.clone();
        let mut steps = 0;
        while !game.finished {
            steps += 1;
            assert!(steps < 8000);
            let before = game.hands_done as usize;
            let action = game.teacher_choice().expect("an outside decision");
            game.step(action);
            assert_eq!(game.endings.len(), game.hands_done as usize - before);
            reported.extend(game.endings.iter().copied());
        }
        assert_eq!(reported.len(), game.hands_done as usize);
        assert_eq!(reported, results_in_log(&game));
        game.step(0);
        assert!(
            game.endings.is_empty(),
            "a step forgets what the last one ended"
        );
    }
}

#[test]
fn each_table_can_seat_the_bots_in_places_of_its_own() {
    let uniform = places_by_table(3, vec![1, 2], None).expect("the same places everywhere");
    assert_eq!(uniform, vec![vec![1, 2]; 3]);
    let tables = vec![vec![], vec![0], vec![1, 3]];
    assert_eq!(
        places_by_table(3, vec![], Some(tables.clone())),
        Ok(tables.clone())
    );
    for (index, places) in tables.iter().enumerate() {
        let game = Seat::new(40 + index as u64, places);
        for place in 0..4 {
            assert_eq!(game.bots[place].is_some(), places.contains(&place));
        }
    }
    assert!(
        places_by_table(3, vec![1], Some(tables.clone())).is_err(),
        "not both"
    );
    assert!(
        places_by_table(2, vec![], Some(tables)).is_err(),
        "one list a table"
    );
    assert!(
        places_by_table(1, vec![], Some(vec![vec![4]])).is_err(),
        "four places"
    );
    assert!(places_by_table(1, vec![4], None).is_err(), "four places");
}
