//! The selected scoring decomposition, not a second guess made by JavaScript.
use riichi_core::agari::{Block, Shape};
use riichi_core::hand::Meld;
use riichi_core::score::Score;
use riichi_core::tile::Tile;
use riichi_core::yaku::Yaku;
use riichi_core::Wind;
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Serialize, Debug)]
pub struct Place {
    tile: String,
    at: String,
}

#[derive(Serialize, Debug)]
pub struct GroupView {
    kind: &'static str,
    concealed: bool,
    tiles: Vec<Place>,
    faces: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct ScoringView {
    sets: Vec<GroupView>,
    pair: GroupView,
}

/// Enum identity remains distinct when several yaku share a display name.
pub fn identity(yaku: Yaku) -> String {
    match yaku {
        Yaku::YakuhaiDragon(tile) => format!("YakuhaiDragon-{tile}"),
        other => format!("{other:?}"),
    }
}

pub fn value_tile(yaku: Yaku, seat: Wind, round: Wind) -> Option<String> {
    match yaku {
        Yaku::YakuhaiDragon(tile) => Some(tile.to_string()),
        Yaku::YakuhaiSeatWind => Some(seat.tile().to_string()),
        Yaku::YakuhaiRoundWind => Some(round.tile().to_string()),
        _ => None,
    }
}

/// `standing` is exactly the tile order rendered as hand:N, without the
/// separately rendered winning tile. The scorer's completed block owns won.
pub fn scoring_view(
    score: &Score,
    standing: &[Tile],
    melds: &[Meld],
    ron: bool,
) -> Option<ScoringView> {
    if score.reading.shape != Shape::Standard {
        return None; // Special whole-hand shapes do not need set attribution.
    }
    let mut available: [VecDeque<usize>; 34] = std::array::from_fn(|_| VecDeque::new());
    for (index, tile) in standing.iter().enumerate() {
        available[tile.index() as usize].push_back(index);
    }
    let mut won = false;
    let mut sets = Vec::new();
    let mut pair = None;
    for (index, block) in score.reading.blocks.iter().enumerate() {
        let faces = block.tiles();
        let tiles = faces
            .iter()
            .map(|tile| {
                let at = if index == score.completed_block && *tile == score.winning_tile && !won {
                    won = true;
                    "won".to_string()
                } else {
                    format!(
                        "hand:{}",
                        available[tile.index() as usize]
                            .pop_front()
                            .expect("scored tile exists")
                    )
                };
                Place {
                    tile: tile.to_string(),
                    at,
                }
            })
            .collect();
        let kind = match block {
            Block::Pair(_) => "pair",
            Block::Sequence(_) => "run",
            Block::Triplet(_) => "triplet",
        };
        let group = GroupView {
            kind,
            concealed: !(ron && index == score.completed_block && kind == "triplet"),
            tiles,
            faces: faces.iter().map(ToString::to_string).collect(),
        };
        if kind == "pair" {
            pair = Some(group);
        } else {
            sets.push(group);
        }
    }
    for (index, meld) in melds.iter().enumerate() {
        let faces = meld.tiles();
        sets.push(GroupView {
            kind: if meld.is_sequence() {
                "run"
            } else if meld.kind.is_kan() {
                "quad"
            } else {
                "triplet"
            },
            concealed: meld.kind.is_concealed_for_fu(),
            tiles: faces
                .iter()
                .enumerate()
                .map(|(slot, tile)| Place {
                    tile: tile.to_string(),
                    at: format!("meld:{index}:{slot}"),
                })
                .collect(),
            faces: faces.iter().map(ToString::to_string).collect(),
        });
    }
    Some(ScoringView {
        sets,
        pair: pair.expect("standard reading has a pair"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use riichi_core::hand::TileSet;
    use riichi_core::score::{score, Situation, WinBy};
    use std::collections::HashSet;

    fn explained(text: &str, winning: &str, by: WinBy) -> (Score, ScoringView) {
        let mut tiles: TileSet = text.parse().unwrap();
        let tile: Tile = winning.parse().unwrap();
        let result = score(
            &tiles,
            &[],
            &Situation::new(Wind::South, Wind::East, by, tile),
        )
        .unwrap();
        tiles.remove(tile);
        let view = scoring_view(
            &result,
            &tiles.tiles().collect::<Vec<_>>(),
            &[],
            by == WinBy::Discard,
        )
        .unwrap();
        let positions: Vec<_> = view
            .sets
            .iter()
            .chain(std::iter::once(&view.pair))
            .flat_map(|group| group.tiles.iter().map(|place| place.at.as_str()))
            .collect();
        assert_eq!(positions.len(), 14);
        assert_eq!(positions.iter().copied().collect::<HashSet<_>>().len(), 14);
        assert!(positions.contains(&"won"));
        (result, view)
    }

    #[test]
    fn value_triplets_keep_identity_and_actual_wind() {
        assert_ne!(
            identity(Yaku::YakuhaiDragon("5z".parse().unwrap())),
            identity(Yaku::YakuhaiDragon("6z".parse().unwrap()))
        );
        assert_eq!(
            value_tile(Yaku::YakuhaiSeatWind, Wind::South, Wind::East).as_deref(),
            Some("2z")
        );
        assert_eq!(
            value_tile(Yaku::YakuhaiRoundWind, Wind::South, Wind::East).as_deref(),
            Some("1z")
        );
    }

    #[test]
    fn ron_completed_triplet_is_not_concealed_but_tsumo_is() {
        for by in [WinBy::Discard, WinBy::SelfDraw] {
            let (score, view) = explained("111m222p333s44455z", "1m", by);
            let expected = if by == WinBy::Discard {
                Yaku::SanAnkou
            } else {
                Yaku::SuuAnkou
            };
            assert!(score.yaku.iter().any(|(y, _)| *y == expected));
            let winning_group = view
                .sets
                .iter()
                .find(|g| g.tiles.iter().any(|t| t.at == "won"))
                .unwrap();
            assert_eq!(winning_group.concealed, by == WinBy::SelfDraw);
        }
    }

    #[test]
    fn ambiguous_hand_exposes_only_the_highest_scoring_decomposition() {
        let (score, view) = explained("111222333m456p55s", "5s", WinBy::Discard);
        assert!(score.yaku.iter().any(|(y, _)| *y == Yaku::SanAnkou));
        assert_eq!(view.sets.iter().filter(|g| g.kind == "triplet").count(), 3);
        let (score, view) = explained("111122223333m55p", "5p", WinBy::Discard);
        assert!(score.yaku.iter().any(|(y, _)| *y == Yaku::Ryanpeikou));
        assert_eq!(view.sets.iter().filter(|g| g.kind == "run").count(), 4);
    }
}
