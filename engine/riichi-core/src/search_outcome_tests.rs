use super::placement_value;
use crate::table::Table;

#[test]
fn terminal_rewards_match_shared_fixtures_under_every_player_permutation() {
    let fixtures = include_str!("../tests/fixtures/placement-rewards.csv");
    let mut cases = 0;
    for row in fixtures.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 8);
        let scores: [i32; 4] = std::array::from_fn(|i| fields[i].parse().unwrap());
        let rewards: [f64; 4] = std::array::from_fn(|i| fields[i + 4].parse().unwrap());
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let order = [a, b, c, d];
                        if (0..4).any(|i| (i + 1..4).any(|j| order[i] == order[j])) {
                            continue;
                        }
                        let mut table = Table::new();
                        table.scores = order.map(|i| scores[i]);
                        table.finished = true;
                        for (player, original) in order.into_iter().enumerate() {
                            assert_eq!(placement_value(&table, player), rewards[original]);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 7 * 24);
}
