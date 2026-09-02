//! Reading a complete hand: every way its tiles split into sets and a pair.
//!
//! A hand can often be read in more than one way, and the rules say the
//! reading is not the player's to choose freely: "If there is more than one
//! possible way for the winning tile to complete the hand, the highest-scoring
//! possibility is always chosen" (EMA 2025, section 3.4.3). Scoring therefore
//! needs every reading, not just one, so this module enumerates them.

use crate::hand::TileSet;
use crate::tile::{Tile, KINDS};

/// One group of a read hand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Block {
    /// Three consecutive tiles of one suit, named by the lowest.
    Sequence(Tile),
    /// Three identical tiles.
    Triplet(Tile),
    /// Two identical tiles.
    Pair(Tile),
}

impl Block {
    /// The tiles this block is made of.
    pub fn tiles(self) -> Vec<Tile> {
        match self {
            Block::Sequence(low) => {
                let second = low.next_in_suit().expect("sequence starts below 8");
                let third = second.next_in_suit().expect("sequence starts below 8");
                vec![low, second, third]
            }
            Block::Triplet(tile) => vec![tile; 3],
            Block::Pair(tile) => vec![tile; 2],
        }
    }

    /// Whether the block contains a given tile kind.
    pub fn contains(self, tile: Tile) -> bool {
        self.tiles().contains(&tile)
    }

    /// Whether every tile of the block is a terminal or an honour.
    pub fn is_all_terminal_or_honour(self) -> bool {
        self.tiles().iter().all(|tile| tile.is_terminal_or_honour())
    }

    /// Whether the block contains at least one terminal or honour.
    pub fn has_terminal_or_honour(self) -> bool {
        self.tiles().iter().any(|tile| tile.is_terminal_or_honour())
    }
}

/// Which of the three complete shapes a reading is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Shape {
    /// Four sets and a pair.
    Standard,
    /// Seven different pairs (EMA section 4.2.2).
    SevenPairs,
    /// Thirteen orphans (EMA section 4.2.6).
    ThirteenOrphans,
}

/// One complete reading of a hand's concealed tiles.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Reading {
    /// Which shape this reading is.
    pub shape: Shape,
    /// The blocks of the concealed portion, sorted for comparison. Called
    /// sets are not repeated here; scoring adds them.
    pub blocks: Vec<Block>,
}

impl Reading {
    /// The pair of an ordinary reading, if it has one.
    pub fn pair(&self) -> Option<Tile> {
        self.blocks.iter().find_map(|block| match block {
            Block::Pair(tile) => Some(*tile),
            _ => None,
        })
    }
}

/// Every reading of the concealed tiles, for a hand that is already complete.
///
/// `called` is the number of sets already called, whose tiles are not in
/// `hand`. An empty result means the tiles do not form a complete hand.
pub fn readings(hand: &TileSet, called: usize) -> Vec<Reading> {
    let mut result = Vec::new();
    if called > 4 {
        return result;
    }
    let sets_needed = 4 - called;
    let expected = sets_needed * 3 + 2;

    if hand.len() == expected {
        let mut counts = *hand.counts();
        for index in 0..KINDS {
            if counts[index] < 2 {
                continue;
            }
            let pair = Tile::new(index as u8);
            counts[index] -= 2;
            let mut found = Vec::new();
            split_sets(&mut counts, 0, sets_needed, &mut Vec::new(), &mut found);
            counts[index] += 2;
            for mut blocks in found {
                blocks.push(Block::Pair(pair));
                blocks.sort_unstable();
                result.push(Reading { shape: Shape::Standard, blocks });
            }
        }
        result.sort();
        result.dedup();
    }

    if called == 0 {
        if let Some(reading) = seven_pairs(hand) {
            result.push(reading);
        }
        if let Some(reading) = thirteen_orphans(hand) {
            result.push(reading);
        }
    }
    result
}

/// Whether the tiles form a complete hand in any shape.
pub fn is_complete(hand: &TileSet, called: usize) -> bool {
    !readings(hand, called).is_empty()
}

fn seven_pairs(hand: &TileSet) -> Option<Reading> {
    if hand.len() != 14 {
        return None;
    }
    let mut blocks = Vec::with_capacity(7);
    for tile in Tile::all() {
        match hand.count(tile) {
            0 => {}
            2 => blocks.push(Block::Pair(tile)),
            // Two identical pairs are not allowed (EMA section 4.2.2), so
            // four of a kind cannot stand in for two of the seven.
            _ => return None,
        }
    }
    if blocks.len() == 7 {
        blocks.sort_unstable();
        Some(Reading { shape: Shape::SevenPairs, blocks })
    } else {
        None
    }
}

fn thirteen_orphans(hand: &TileSet) -> Option<Reading> {
    if hand.len() != 14 {
        return None;
    }
    let mut blocks = Vec::with_capacity(13);
    let mut pairs = 0;
    for tile in Tile::all() {
        let count = hand.count(tile);
        if count == 0 {
            continue;
        }
        if !tile.is_terminal_or_honour() {
            return None;
        }
        match count {
            1 => blocks.push(Block::Triplet(tile)),
            2 => {
                pairs += 1;
                blocks.push(Block::Pair(tile));
            }
            _ => return None,
        }
    }
    if blocks.len() == 13 && pairs == 1 {
        blocks.sort_unstable();
        Some(Reading { shape: Shape::ThirteenOrphans, blocks })
    } else {
        None
    }
}

fn split_sets(
    counts: &mut [u8; KINDS],
    start: usize,
    need: usize,
    current: &mut Vec<Block>,
    found: &mut Vec<Vec<Block>>,
) {
    if need == 0 {
        if counts.iter().all(|&n| n == 0) {
            found.push(current.clone());
        }
        return;
    }
    let mut index = start;
    while index < KINDS && counts[index] == 0 {
        index += 1;
    }
    if index == KINDS {
        return;
    }
    let tile = Tile::new(index as u8);

    if counts[index] >= 3 {
        counts[index] -= 3;
        current.push(Block::Triplet(tile));
        split_sets(counts, index, need - 1, current, found);
        current.pop();
        counts[index] += 3;
    }

    if !tile.is_honour()
        && tile.rank() <= 7
        && counts[index + 1] > 0
        && counts[index + 2] > 0
    {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        current.push(Block::Sequence(tile));
        split_sets(counts, index, need - 1, current, found);
        current.pop();
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(text: &str) -> TileSet {
        text.parse().expect("test hand parses")
    }

    #[test]
    fn reads_a_plain_hand_one_way() {
        let readings = readings(&hand("123m456m789m123p11s"), 0);
        assert_eq!(readings.len(), 1);
        assert_eq!(readings[0].shape, Shape::Standard);
        assert_eq!(readings[0].blocks.len(), 5);
        assert_eq!(readings[0].pair().unwrap().to_string(), "1s");
    }

    /// EMA 2025, scoring example 8: 1112223334455 style hands can be read as
    /// sequences or as triplets, and the rules take the better reading, so
    /// both must be offered.
    #[test]
    fn ambiguous_hands_have_several_readings() {
        // Three identical sequences or three triplets, plus a pair.
        let all = readings(&hand("111222333m99p"), 1);
        assert!(all.len() >= 2, "expected several readings, got {all:?}");
        let has_triplets = all.iter().any(|reading| {
            reading.blocks.iter().filter(|b| matches!(b, Block::Triplet(_))).count() >= 3
        });
        let has_sequences = all.iter().any(|reading| {
            reading.blocks.iter().filter(|b| matches!(b, Block::Sequence(_))).count() >= 3
        });
        assert!(has_triplets && has_sequences);
    }

    /// EMA 2025, scoring example 8: a hand that is both seven pairs and two
    /// double sequences must offer both readings so the scorer can pick.
    #[test]
    fn seven_pairs_and_standard_can_coexist() {
        let all = readings(&hand("223344m556677p1z1z"), 0);
        assert!(all.iter().any(|r| r.shape == Shape::SevenPairs));
        assert!(all.iter().any(|r| r.shape == Shape::Standard));
    }

    #[test]
    fn seven_pairs_rejects_a_repeated_pair() {
        // Four of a kind cannot stand in for two of the seven pairs.
        let all = readings(&hand("1111m2233445566p"), 0);
        assert!(all.iter().all(|r| r.shape != Shape::SevenPairs));
    }

    #[test]
    fn thirteen_orphans_is_read() {
        let all = readings(&hand("119m19p19s1234567z"), 0);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].shape, Shape::ThirteenOrphans);
    }

    #[test]
    fn incomplete_hands_have_no_reading() {
        assert!(readings(&hand("123m456m789m11s34p"), 0).is_empty());
        assert!(!is_complete(&hand("123m456m789m11s34p"), 0));
    }

    #[test]
    fn called_sets_reduce_what_the_hand_must_hold() {
        // Two called sets: the concealed part is two sets and a pair.
        assert!(is_complete(&hand("123m456m11p"), 2));
        assert!(!is_complete(&hand("123m456m11p"), 1));
    }

    #[test]
    fn sequences_never_wrap_around_nine() {
        assert!(!is_complete(&hand("891m123m456m789p11s"), 0));
    }
}
