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

use std::cell::RefCell;
use std::collections::HashMap;

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
    let key = (*hand.counts(), called);
    if let Some(cached) = CACHE.with(|cache| cache.borrow().get(&key).copied()) {
        return cached;
    }
    let value = standard_uncached(&key.0, called);
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        // The same shapes come round again and again while a bot weighs its
        // discards, so they are worth remembering; the cap keeps a long
        // self-play run from growing without bound.
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, value);
    });
    value
}

/// How many decompositions to remember before starting over.
const CACHE_LIMIT: usize = 1 << 20;

thread_local! {
    static CACHE: RefCell<HashMap<([u8; KINDS], usize), i32>> = RefCell::new(HashMap::new());
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

/// What one suit block can contribute: complete sets, partial blocks (the
/// pair among them), and whether one of those partials is a pair.
type Profile = (u8, u8, bool);

/// The shanten of the ordinary shape, computed a suit at a time.
///
/// Each of the three suits and the honours is decomposed on its own into
/// every worthwhile combination of sets and partial blocks, which is cached
/// because the same nine-tile patterns recur constantly. The four results
/// are then combined by a small dynamic program over at most fifty states,
/// which is far cheaper than searching the whole hand at once.
#[allow(clippy::needless_range_loop)]
fn standard_uncached(counts: &[u8; KINDS], called: usize) -> i32 {
    let mut states = [[[false; 2]; 6]; 5];
    states[0][0][0] = true;

    for block in 0..4 {
        let (offset, size) = if block < 3 { (block * 9, 9) } else { (27, 7) };
        let mut pattern = [0u8; 9];
        pattern[..size].copy_from_slice(&counts[offset..offset + size]);
        let profiles = profiles_for(&pattern, size, block == 3);

        let mut next = [[[false; 2]; 6]; 5];
        for sets in 0..5 {
            for partials in 0..6 {
                for pair in 0..2 {
                    if !states[sets][partials][pair] {
                        continue;
                    }
                    for (add_sets, add_partials, add_pair) in &profiles {
                        let sets = (sets + *add_sets as usize).min(4);
                        let partials = partials + *add_partials as usize;
                        if partials > 5 || called + sets + partials > 5 {
                            continue;
                        }
                        let pair = pair | usize::from(*add_pair);
                        next[sets][partials][pair] = true;
                    }
                }
            }
        }
        states = next;
    }

    let mut best = 8 - 2 * called as i32;
    for sets in 0..5 {
        for partials in 0..6 {
            for pair in 0..2 {
                if !states[sets][partials][pair] {
                    continue;
                }
                let sets = sets + called;
                let blocks = sets + partials;
                let mut value = 8 - 2 * sets as i32 - partials as i32;
                if blocks == 5 && pair == 0 {
                    value += 1;
                }
                if value < best {
                    best = value;
                }
            }
        }
    }
    best
}

/// Every worthwhile way one block of tiles can be read.
fn profiles_for(pattern: &[u8; 9], size: usize, honours: bool) -> Vec<Profile> {
    let key = pattern
        .iter()
        .enumerate()
        .fold(0u32, |key, (index, count)| {
            key | ((*count as u32) << (index * 3))
        })
        | ((honours as u32) << 30);
    if let Some(cached) = PROFILES.with(|cache| cache.borrow().get(&key).cloned()) {
        return cached;
    }
    let mut found: Vec<Profile> = Vec::new();
    let mut working = *pattern;
    collect(&mut working, 0, size, honours, 0, 0, false, &mut found);
    // Every distinct reading is kept. Dropping the ones another reading
    // beats would be wrong: a reading with more blocks can be refused later
    // by the five-block cap, and then the smaller one is the answer.
    found.sort_unstable();
    found.dedup();
    let pruned = if found.is_empty() {
        vec![(0, 0, false)]
    } else {
        found
    };
    PROFILES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, pruned.clone());
    });
    pruned
}

#[allow(clippy::too_many_arguments)]
fn collect(
    counts: &mut [u8; 9],
    start: usize,
    size: usize,
    honours: bool,
    sets: u8,
    partials: u8,
    pair: bool,
    found: &mut Vec<Profile>,
) {
    let mut index = start;
    while index < size && counts[index] == 0 {
        index += 1;
    }
    if index == size {
        found.push((sets, partials, pair));
        return;
    }
    if sets + partials > 5 {
        found.push((sets, partials, pair));
        return;
    }

    // A triplet.
    if counts[index] >= 3 {
        counts[index] -= 3;
        collect(
            counts,
            index,
            size,
            honours,
            sets + 1,
            partials,
            pair,
            found,
        );
        counts[index] += 3;
    }

    // A sequence, which honours cannot form and which never wraps (EMA 3.2).
    if !honours && index + 2 < size && counts[index + 1] > 0 && counts[index + 2] > 0 {
        counts[index] -= 1;
        counts[index + 1] -= 1;
        counts[index + 2] -= 1;
        collect(
            counts,
            index,
            size,
            honours,
            sets + 1,
            partials,
            pair,
            found,
        );
        counts[index] += 1;
        counts[index + 1] += 1;
        counts[index + 2] += 1;
    }

    // A pair, which may be the hand's pair or just a partial triplet.
    if counts[index] >= 2 {
        counts[index] -= 2;
        if !pair {
            collect(
                counts,
                index,
                size,
                honours,
                sets,
                partials + 1,
                true,
                found,
            );
        }
        collect(
            counts,
            index,
            size,
            honours,
            sets,
            partials + 1,
            pair,
            found,
        );
        counts[index] += 2;
    }

    if !honours {
        // Two tiles a third would join.
        if index + 1 < size && counts[index + 1] > 0 {
            counts[index] -= 1;
            counts[index + 1] -= 1;
            collect(
                counts,
                index,
                size,
                honours,
                sets,
                partials + 1,
                pair,
                found,
            );
            counts[index] += 1;
            counts[index + 1] += 1;
        }
        // A shape waiting on the tile between.
        if index + 2 < size && counts[index + 2] > 0 {
            counts[index] -= 1;
            counts[index + 2] -= 1;
            collect(
                counts,
                index,
                size,
                honours,
                sets,
                partials + 1,
                pair,
                found,
            );
            counts[index] += 1;
            counts[index + 2] += 1;
        }
    }

    // Or the tile is spare.
    counts[index] -= 1;
    collect(counts, index, size, honours, sets, partials, pair, found);
    counts[index] += 1;
}

thread_local! {
    static PROFILES: RefCell<HashMap<u32, Vec<Profile>>> = RefCell::new(HashMap::new());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tile::COPIES;

    fn hand(text: &str) -> TileSet {
        text.parse().expect("test hand parses")
    }

    /// The straightforward search over the whole hand at once, kept only to
    /// check the fast implementation against.
    fn reference(counts: &[u8; KINDS], called: usize) -> i32 {
        fn walk(
            counts: &mut [u8; KINDS],
            start: usize,
            called: i32,
            sets: i32,
            partials: i32,
            pair: bool,
            best: &mut i32,
        ) {
            let mut index = start;
            while index < KINDS && counts[index] == 0 {
                index += 1;
            }
            if index == KINDS {
                let sets = sets + called;
                let blocks = sets + partials;
                let mut value = 8 - 2 * sets - partials;
                if blocks == 5 && !pair {
                    value += 1;
                }
                *best = (*best).min(value);
                return;
            }
            let tile = Tile::new(index as u8);
            let numbered = !tile.is_honour();
            let rank = tile.rank();
            let blocks = sets + called + partials;

            if counts[index] >= 3 {
                counts[index] -= 3;
                walk(counts, index, called, sets + 1, partials, pair, best);
                counts[index] += 3;
            }
            if numbered && rank <= 7 && counts[index + 1] > 0 && counts[index + 2] > 0 {
                counts[index] -= 1;
                counts[index + 1] -= 1;
                counts[index + 2] -= 1;
                walk(counts, index, called, sets + 1, partials, pair, best);
                counts[index] += 1;
                counts[index + 1] += 1;
                counts[index + 2] += 1;
            }
            if blocks < 5 {
                if counts[index] >= 2 {
                    counts[index] -= 2;
                    if !pair {
                        walk(counts, index, called, sets, partials + 1, true, best);
                    }
                    walk(counts, index, called, sets, partials + 1, pair, best);
                    counts[index] += 2;
                }
                if numbered && rank <= 8 && counts[index + 1] > 0 {
                    counts[index] -= 1;
                    counts[index + 1] -= 1;
                    walk(counts, index, called, sets, partials + 1, pair, best);
                    counts[index] += 1;
                    counts[index + 1] += 1;
                }
                if numbered && rank <= 7 && counts[index + 2] > 0 {
                    counts[index] -= 1;
                    counts[index + 2] -= 1;
                    walk(counts, index, called, sets, partials + 1, pair, best);
                    counts[index] += 1;
                    counts[index + 2] += 1;
                }
            }
            counts[index] -= 1;
            walk(counts, index, called, sets, partials, pair, best);
            counts[index] += 1;
        }

        let mut working = *counts;
        let mut best = i32::MAX;
        walk(&mut working, 0, called as i32, 0, 0, false, &mut best);
        best
    }

    /// The fast decomposition must agree with the straightforward one on
    /// every hand, whatever the shape and however many sets were called.
    #[test]
    fn the_fast_search_matches_the_reference() {
        let mut rng = Rng::from_seed(20260902);
        for round in 0..4000 {
            let called = round % 5;
            let size = 13 - 3 * called;
            let mut counts = [0u8; KINDS];
            let mut drawn = 0;
            while drawn < size {
                let kind = rng.below(KINDS);
                if counts[kind] < COPIES {
                    counts[kind] += 1;
                    drawn += 1;
                }
            }
            let set = TileSet::from_counts(counts);
            assert_eq!(
                standard(&set, called),
                reference(&counts, called),
                "disagreed on {set} with {called} called sets"
            );
        }
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
