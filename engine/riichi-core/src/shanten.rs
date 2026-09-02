//! Shanten: how many tile exchanges a hand still needs.
//!
//! Shanten counts the changes away from a complete hand, so 0 means tenpai
//! (waiting) and -1 means the hand is already complete. Three shapes are
//! measured and the smallest wins: the ordinary four sets and a pair, Seven
//! Pairs and Thirteen Orphans (EMA 2025, sections 3.2 and 3.3.8).
//!
//! The search is a plain backtracking decomposition. It is exact and easy to
//! check against the rulebook; a table-driven version can replace it later
//! if self-play throughput needs one.

use crate::hand::TileSet;
use crate::tile::{Tile, COPIES, KINDS};

/// Returned when a hand is already complete.
pub const COMPLETE: i32 = -1;
/// Returned when a hand is one tile from complete.
pub const TENPAI: i32 = 0;

/// Shanten of the ordinary shape: four sets and a pair.
///
/// `called` is the number of sets already called, each of which counts as a
/// complete set and whose tiles are not in `hand`.
pub fn standard(hand: &TileSet, called: usize) -> i32 {
    let mut counts = *hand.counts();
    let mut search = Search { best: i32::MAX, called: called as i32 };
    search.walk(&mut counts, 0, 0, 0, false);
    search.best
}

/// Shanten of Seven Pairs (EMA section 4.2.2), which must be concealed and
/// needs seven *different* pairs, so a third and fourth copy of a kind do
/// not help.
pub fn seven_pairs(hand: &TileSet, called: usize) -> i32 {
    if called > 0 {
        return i32::MAX;
    }
    let pairs = hand.counts().iter().filter(|&&n| n >= 2).count() as i32;
    let kinds = hand.distinct() as i32;
    // Six exchanges from no pairs, and a kind must be found for every pair
    // the hand cannot yet form from distinct kinds.
    6 - pairs + (7 - kinds).max(0)
}

/// Shanten of Thirteen Orphans (EMA section 4.2.6): one of each terminal and
/// honour plus a duplicate of any of them. Must be concealed.
pub fn thirteen_orphans(hand: &TileSet, called: usize) -> i32 {
    if called > 0 {
        return i32::MAX;
    }
    let mut kinds = 0;
    let mut has_pair = false;
    for tile in Tile::all().filter(|tile| tile.is_terminal_or_honour()) {
        let count = hand.count(tile);
        if count >= 1 {
            kinds += 1;
        }
        if count >= 2 {
            has_pair = true;
        }
    }
    13 - kinds - i32::from(has_pair)
}

/// Shanten of a hand: the best of the three shapes.
pub fn shanten(hand: &TileSet, called: usize) -> i32 {
    standard(hand, called)
        .min(seven_pairs(hand, called))
        .min(thirteen_orphans(hand, called))
}

/// Whether the hand is complete, i.e. four sets and a pair, Seven Pairs or
/// Thirteen Orphans. This is shape only: a winning hand also needs a yaku
/// (EMA section 3.2), which [`crate::score`] decides.
pub fn is_complete(hand: &TileSet, called: usize) -> bool {
    shanten(hand, called) == COMPLETE
}

/// The tiles that would complete the hand, i.e. its waits.
///
/// `visible` counts every copy of a kind the player can already account for
/// in their own hand and called sets: a hand holding all four copies of a
/// kind cannot wait on a fifth (EMA section 3.3.8).
pub fn waits(hand: &TileSet, called: usize, visible: &TileSet) -> TileSet {
    let mut result = TileSet::new();
    // A hand that is not waiting has no waits, and this one check saves the
    // thirty-four it would otherwise take to find that out.
    if shanten(hand, called) != TENPAI {
        return result;
    }
    let mut probe = *hand;
    for tile in Tile::all() {
        if visible.count(tile) >= COPIES {
            continue;
        }
        probe.add(tile);
        if shanten(&probe, called) == COMPLETE {
            result.add(tile);
        }
        probe.remove(tile);
    }
    result
}

/// Whether the hand is waiting, i.e. tenpai (EMA section 3.3.8).
///
/// A hand whose only candidate is a fifth copy is noten, which is why this
/// asks for the waits rather than the shanten number alone.
pub fn is_tenpai(hand: &TileSet, called: usize, visible: &TileSet) -> bool {
    !waits(hand, called, visible).is_empty()
}

/// The tiles that would reduce the hand's shanten, with the resulting count.
///
/// This is the acceptance ("ukeire") list the efficiency aids and the
/// training oracle both use. The hand is expected to be at a draw boundary,
/// i.e. 13 tiles minus three per called set.
pub fn acceptance(hand: &TileSet, called: usize, visible: &TileSet) -> Vec<(Tile, i32)> {
    let current = shanten(hand, called);
    let mut result = Vec::new();
    let mut probe = *hand;
    for tile in Tile::all() {
        if visible.count(tile) >= COPIES {
            continue;
        }
        probe.add(tile);
        let after = shanten(&probe, called);
        probe.remove(tile);
        if after < current {
            result.push((tile, after));
        }
    }
    result
}

struct Search {
    best: i32,
    called: i32,
}

impl Search {
    /// Records the shanten of one complete decomposition.
    ///
    /// The formula is the standard one: eight exchanges, minus two for every
    /// complete set and one for every partial block, with at most five blocks
    /// in all and an extra exchange needed when five blocks contain no pair.
    fn record(&mut self, sets: i32, partials: i32, pair: bool) {
        let sets = sets + self.called;
        let blocks = sets + partials;
        let mut value = 8 - 2 * sets - partials;
        if blocks == 5 && !pair {
            value += 1;
        }
        if value < self.best {
            self.best = value;
        }
    }

    fn walk(
        &mut self,
        counts: &mut [u8; KINDS],
        start: usize,
        sets: i32,
        partials: i32,
        pair: bool,
    ) {
        let mut index = start;
        while index < KINDS && counts[index] == 0 {
            index += 1;
        }
        if index == KINDS {
            self.record(sets, partials, pair);
            return;
        }

        let tile = Tile::new(index as u8);
        let numbered = !tile.is_honour();
        let rank = tile.rank();
        let blocks = sets + self.called + partials;

        // A triplet.
        if counts[index] >= 3 {
            counts[index] -= 3;
            self.walk(counts, index, sets + 1, partials, pair);
            counts[index] += 3;
        }

        // A sequence, which never wraps past 9 (EMA section 3.2).
        if numbered && rank <= 7 && counts[index + 1] > 0 && counts[index + 2] > 0 {
            counts[index] -= 1;
            counts[index + 1] -= 1;
            counts[index + 2] -= 1;
            self.walk(counts, index, sets + 1, partials, pair);
            counts[index] += 1;
            counts[index + 1] += 1;
            counts[index + 2] += 1;
        }

        if blocks < 5 {
            // A pair, either as the hand's pair or as a partial triplet.
            if counts[index] >= 2 {
                counts[index] -= 2;
                if !pair {
                    self.walk(counts, index, sets, partials + 1, true);
                }
                self.walk(counts, index, sets, partials + 1, pair);
                counts[index] += 2;
            }

            // Two tiles that a third would join: an open or edge shape.
            if numbered && rank <= 8 && counts[index + 1] > 0 {
                counts[index] -= 1;
                counts[index + 1] -= 1;
                self.walk(counts, index, sets, partials + 1, pair);
                counts[index] += 1;
                counts[index + 1] += 1;
            }

            // A closed shape, waiting on the middle tile.
            if numbered && rank <= 7 && counts[index + 2] > 0 {
                counts[index] -= 1;
                counts[index + 2] -= 1;
                self.walk(counts, index, sets, partials + 1, pair);
                counts[index] += 1;
                counts[index + 2] += 1;
            }
        }

        // Or the tile belongs to no block at all.
        counts[index] -= 1;
        self.walk(counts, index, sets, partials, pair);
        counts[index] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(text: &str) -> TileSet {
        text.parse().expect("test hand parses")
    }

    #[test]
    fn complete_ordinary_hands() {
        // Four sequences and a pair.
        assert_eq!(shanten(&hand("123m456m789m123p11s"), 0), COMPLETE);
        // Four triplets and a pair.
        assert_eq!(shanten(&hand("111m222m333m444p55s"), 0), COMPLETE);
        // Two called sets plus two sets and a pair in hand.
        assert_eq!(shanten(&hand("123m456m11p"), 2), COMPLETE);
    }

    #[test]
    fn tenpai_and_one_away() {
        // Waiting on both ends of 3-4 circles.
        assert_eq!(shanten(&hand("123m456m789m11s34p"), 0), TENPAI);
        // Waiting on the middle of 3-5 circles.
        assert_eq!(shanten(&hand("123m456m789m11s35p"), 0), TENPAI);
        // 3 and 6 circles form no block at all, so the hand is a step back.
        assert_eq!(shanten(&hand("123m456m789m11s36p"), 0), 1);
        assert_eq!(shanten(&hand("123m456m789m12s36p"), 0), 1);
    }

    /// EMA 2025 section 4.2.2: Seven Pairs needs seven *different* pairs.
    #[test]
    fn seven_pairs_needs_distinct_kinds() {
        assert_eq!(shanten(&hand("1122334455667m7m"), 0), COMPLETE);
        // Six kinds only, one of them four times: the fourth copy is dead
        // weight for this shape, so the hand is not tenpai on it.
        assert_eq!(seven_pairs(&hand("1111223344556m"), 0), 2);
        // Any call rules the shape out, since it must be concealed.
        assert_eq!(seven_pairs(&hand("1122334455667m"), 1), i32::MAX);
    }

    /// EMA 2025 section 4.2.6: Thirteen Orphans is one of each terminal and
    /// honour plus a duplicate.
    #[test]
    fn thirteen_orphans_shapes() {
        assert_eq!(thirteen_orphans(&hand("19m19p19s1234567z"), 0), TENPAI);
        assert_eq!(thirteen_orphans(&hand("119m19p19s1234567z"), 0), COMPLETE);
        assert_eq!(thirteen_orphans(&hand("19m19p19s123456z"), 0), 1);
    }

    /// EMA 2025 section 3.3.8: a hand holding four copies cannot wait on the
    /// fifth, so such a hand is noten.
    #[test]
    fn no_hand_waits_on_a_fifth_copy() {
        // Three sequences and all four 9 circles: the shape would be
        // complete only with a fifth 9 circles as the pair's partner, so the
        // hand is noten however tempting it looks.
        let concealed = hand("123m456m789m9999p");
        assert_eq!(concealed.len(), 13);
        let waits_now = waits(&concealed, 0, &concealed);
        assert!(waits_now.is_empty(), "found waits {waits_now}");
        assert!(!is_tenpai(&concealed, 0, &concealed));
        // Shape alone would call it tenpai; the fourth-copy rule is what
        // makes the difference.
        assert_eq!(standard(&concealed, 0), TENPAI);
    }

    #[test]
    fn waits_are_found() {
        let concealed = hand("123m456m789m11s34p");
        let list = waits(&concealed, 0, &concealed);
        assert_eq!(list.to_string(), "25p");
        // A three-sided wait, the shape EMA section 3.3.9 uses to explain
        // furiten: every one of the three tiles counts.
        let concealed = hand("99m111z23456789p");
        assert_eq!(concealed.len(), 13);
        let list = waits(&concealed, 0, &concealed);
        assert_eq!(list.to_string(), "147p");
    }

    #[test]
    fn acceptance_lists_useful_draws() {
        let concealed = hand("123m456m789m11s36p");
        let list = acceptance(&concealed, 0, &concealed);
        let tiles: Vec<String> = list.iter().map(|(tile, _)| tile.to_string()).collect();
        assert!(tiles.contains(&"3p".to_string()));
        assert!(tiles.contains(&"6p".to_string()));
        assert!(tiles.contains(&"1s".to_string()));
    }

    #[test]
    fn thirteen_orphans_beats_the_ordinary_shape() {
        let concealed = hand("19m19p19s1234567z");
        assert_eq!(shanten(&concealed, 0), TENPAI);
        assert!(standard(&concealed, 0) > TENPAI);
    }
}
