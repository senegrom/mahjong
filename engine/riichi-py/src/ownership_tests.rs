use super::Seat;

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
