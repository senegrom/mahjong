//! Continuation chance is attached to proposals, not their current slot numbers.
use riichi_core::encoding;
use riichi_core::game::{Action, Hand, Phase};
use riichi_core::mjai::Event;
use riichi_core::rng::Rng;
use riichi_core::search::Lookahead;
use riichi_core::table::Table;
use riichi_core::tile::Tile;

fn last_draw(seed: u64) -> Hand {
    let mut hand = Table::new().deal(&mut Rng::from_seed(seed));
    for _ in 0..1000 {
        match hand.phase {
            Phase::Act if hand.wall.remaining() == 0 => return hand,
            Phase::Act => {
                let discard = hand
                    .legal_actions()
                    .into_iter()
                    .find(|action| matches!(action, Action::Discard(_)))
                    .unwrap();
                hand.act(discard).unwrap();
            }
            Phase::Draw => {
                hand.draw().unwrap();
            }
            Phase::CallWindow => hand.resolve_calls(&[]).unwrap(),
            Phase::Over => panic!("fixture must reach the final live-wall draw"),
        }
    }
    panic!("last draw not reached")
}

fn boundary_deals(hand: &Hand, seeds: &[u64]) -> Vec<([Vec<Tile>; 4], Tile)> {
    let candidates: Vec<_> = hand
        .legal_actions()
        .into_iter()
        .filter(|a| matches!(a, Action::Discard(_)))
        .take(2)
        .collect();
    assert_eq!(candidates.len(), 2);
    let worlds = vec![hand.clone(); seeds.len()];
    let mut lookahead = Lookahead::begin(
        hand.turn,
        &candidates,
        &worlds,
        &vec![1.0; seeds.len()],
        0,
        true,
        true,
        seeds,
    );
    for _ in 0..1000 {
        if lookahead.finished() {
            break;
        }
        let (mut observations, mut masks) = (Vec::new(), Vec::new());
        lookahead.observe_into(&mut observations, &mut masks);
        let answers: Vec<_> = masks
            .chunks(encoding::ACTIONS)
            .map(|mask| {
                if mask[encoding::PASS] {
                    encoding::PASS
                } else {
                    mask.iter().position(|legal| *legal).unwrap()
                }
            })
            .collect();
        lookahead.apply(&answers).unwrap();
    }
    lookahead.validate_finished().unwrap();
    lookahead
        .leaves()
        .invented
        .iter()
        .map(|events| {
            events
                .iter()
                .find_map(|event| match event {
                    Event::StartKyoku {
                        hands, indicator, ..
                    } => Some((hands.clone(), *indicator)),
                    _ => None,
                })
                .expect("first boundary must expose the next deal")
        })
        .collect()
}

#[test]
fn fresh_chance_changes_later_deals_while_siblings_remain_paired_and_replayable() {
    let root = last_draw(100);
    let other = last_draw(999);
    let discovery = boundary_deals(&root, &[17, 71]);
    let confirmation = boundary_deals(&other, &[41, 14]);
    assert_eq!(discovery[0], discovery[2]);
    assert_eq!(discovery[1], discovery[3]);
    assert_eq!(confirmation[0], confirmation[2]);
    assert_eq!(confirmation[1], confirmation[3]);
    assert_ne!(discovery[0], discovery[1]);
    assert_ne!(discovery[0], confirmation[0]);
    assert_eq!(discovery, boundary_deals(&root, &[17, 71]));
    let reordered = boundary_deals(&root, &[71, 17]);
    assert_eq!(reordered[0], discovery[1]);
    assert_eq!(reordered[1], discovery[0]);
}
