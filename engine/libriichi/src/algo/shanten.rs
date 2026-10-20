//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//!
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>

use crate::tuz;
use std::io::prelude::*;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

const JIHAI_TABLE_SIZE: usize = 78_032;
const SUHAI_TABLE_SIZE: usize = 1_940_777;

static JIHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_jihai.bin.gz"),
        JIHAI_TABLE_SIZE,
    )
});
static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = vec![];
    gz.read_to_end(&mut raw).unwrap();

    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);

    ret
}

pub fn ensure_init() {
    assert_eq!(JIHAI_TABLE.len(), JIHAI_TABLE_SIZE);
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();

    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
        }
        lhs[j] = sht;
    }

    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

fn add_jihai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = JIHAI_TABLE.get(index).copied().unwrap_or_default();

    let j = m + 5;
    let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
    }
    lhs[j] = sht;
}

fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

/// `len_div3` must be within [0, 4].
#[must_use]
pub fn calc_normal(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, sum_tiles(&tiles[9..2 * 9]), len_div3);
    add_suhai(&mut ret, sum_tiles(&tiles[2 * 9..3 * 9]), len_div3);
    add_jihai(&mut ret, sum_tiles(&tiles[3 * 9..]), len_div3);

    (ret[5 + len_div3] as i8) - 1
}

#[must_use]
pub fn calc_chitoi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        if c >= 2 {
            pairs += 1;
        }
    });

    let redunct = 7_u8.saturating_sub(kinds) as i8;
    7 - pairs + redunct - 1
}

#[must_use]
pub fn calc_kokushi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;

    tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C]
        .iter()
        .map(|&i| tiles[i])
        .filter(|&c| c > 0)
        .for_each(|c| {
            kinds += 1;
            if c >= 2 {
                pairs += 1;
            }
        });

    let redunct = (pairs > 0) as i8;
    14 - kinds - redunct - 1
}

#[must_use]
pub fn calc_all(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let mut shanten = calc_normal(tiles, len_div3);
    if shanten <= 0 || len_div3 < 4 {
        return shanten;
    }

    shanten = shanten.min(calc_chitoi(tiles));
    if shanten > 0 {
        shanten.min(calc_kokushi(tiles))
    } else {
        shanten
    }
}

/// The shanten of the hands one tile away from a base hand, one more or one
/// fewer of a kind, exactly as `calc_all` gives it. The lookahead asks this
/// for every tile it could draw or discard at every node, some eleven
/// thousand times an observation, and it was most of what the encoder cost.
///
/// `calc_normal` looks the four groups (the three suits and the honours) up
/// in their tables and combines the four rows. A hand one tile away differs
/// from the base in one group only, so this looks the base's groups up once
/// and combines each group's three others once (`others`); a query is then a
/// lookup for the changed group and one entry of its combination with them.
///
/// Why each answer is exactly `calc_all`'s:
///
/// - A row's entry j is the fewest tiles to change for j sets without a
///   pair, and entry 5 + j the fewest for j sets and the pair, for j up to
///   `len_div3` (entries beyond it are never read). Two groups combine by the
///   min-plus product of their rows: each entry the least sum over the ways
///   to share the sets, and the pair, between them. `add_suhai` computes
///   exactly the entries of that product it later reads, and `add_jihai` its
///   last entry, as `product` and `last` do below. That product is
///   commutative and associative, so `calc_normal`'s answer, entry
///   5 + `len_div3` of the four groups' product taken man, pin, sou, then
///   honours, is the same number whichever order the groups are combined in:
///   the same least sum over the same ways to share the sets.
/// - It is whole-number arithmetic. The tables hold four-bit entries, so no
///   sum of four of them passes 60 and a `u8` never overflows.
/// - The changed group's table index is the base's plus or minus the tile's
///   weight in base five, the very number `sum_tiles` makes of the changed
///   counts, even for a fifth tile, which carries into the next digit there
///   as here. The lookup is therefore the same, past the table's end and its
///   default row included.
/// - Seven pairs and thirteen orphans count the kinds held and the pairs; one
///   tile moves those counts in its own kind only, so they are updated from
///   the base's rather than counted again.
pub struct Near {
    len_div3: usize,
    /// Each group's table index: `sum_tiles` of its counts.
    index: [usize; 4],
    /// For each group, the min-plus product of the other three groups' rows.
    others: [[u8; 10]; 4],
    counts: [u8; 34],
    /// What `calc_chitoi` counts in the base hand: kinds held and pairs.
    kinds: i8,
    pairs: i8,
    /// What `calc_kokushi` counts: the same, of terminals and honours only.
    orphan_kinds: i8,
    orphan_pairs: i8,
}

/// What one more of a kind adds to its group's table index: `sum_tiles`
/// reads a group's counts as a number in base five, the first kind highest.
const WEIGHTS: [usize; 34] = {
    let mut weights = [0; 34];
    let mut tid = 0;
    while tid < 34 {
        let (position, length) = if tid < 27 {
            (tid % 9, 9)
        } else {
            (tid - 27, 7)
        };
        let mut weight = 1;
        let mut digit = position + 1;
        while digit < length {
            weight *= 5;
            digit += 1;
        }
        weights[tid] = weight;
        tid += 1;
    }
    weights
};

/// The terminals and honours, the kinds `calc_kokushi` counts.
const ORPHANS: [bool; 34] = {
    let mut orphans = [false; 34];
    let kinds = tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C];
    let mut i = 0;
    while i < kinds.len() {
        orphans[kinds[i]] = true;
        i += 1;
    }
    orphans
};

/// The entries `add_suhai` computes of the min-plus product of two rows:
/// 0..=m, sets without the pair, and 5..=5+m, sets with it, which may come
/// from either row.
fn product(a: &[u8; 10], b: &[u8; 10], m: usize) -> [u8; 10] {
    let mut c = [0; 10];
    for j in 0..=m {
        let mut sht = a[0] + b[j];
        for k in 1..=j {
            sht = sht.min(a[k] + b[j - k]);
        }
        c[j] = sht;
    }
    for j in 0..=m {
        let mut sht = a[5] + b[j];
        for k in 0..=j {
            sht = sht.min(a[5 + k] + b[j - k]).min(a[k] + b[5 + j - k]);
        }
        c[5 + j] = sht;
    }
    c
}

/// Entry 5+m alone of the same product, the one `add_jihai` computes and the
/// one `calc_normal` answers from.
fn last(a: &[u8; 10], b: &[u8; 10], m: usize) -> u8 {
    let mut sht = a[5 + m] + b[0];
    for k in 0..=m {
        sht = sht.min(a[5 + k] + b[m - k]).min(a[k] + b[5 + m - k]);
    }
    sht
}

impl Near {
    /// `len_div3` must be within [0, 4], as for `calc_all`.
    #[must_use]
    pub fn new(tiles: &[u8; 34], len_div3: u8) -> Self {
        let m = len_div3 as usize;
        let index = [
            sum_tiles(&tiles[..9]),
            sum_tiles(&tiles[9..2 * 9]),
            sum_tiles(&tiles[2 * 9..3 * 9]),
            sum_tiles(&tiles[3 * 9..]),
        ];
        let man = SUHAI_TABLE.get(index[0]).copied().unwrap_or_default();
        let pin = SUHAI_TABLE.get(index[1]).copied().unwrap_or_default();
        let sou = SUHAI_TABLE.get(index[2]).copied().unwrap_or_default();
        let honours = JIHAI_TABLE.get(index[3]).copied().unwrap_or_default();
        // Each group's three others, from two shared pairs of groups: six
        // products where combining every hand from scratch takes three for
        // each of some twenty hands.
        let man_pin = product(&man, &pin, m);
        let sou_honours = product(&sou, &honours, m);
        let others = [
            product(&pin, &sou_honours, m),
            product(&man, &sou_honours, m),
            product(&man_pin, &honours, m),
            product(&man_pin, &sou, m),
        ];

        let (mut kinds, mut pairs, mut orphan_kinds, mut orphan_pairs) = (0, 0, 0, 0);
        for (tid, &count) in tiles.iter().enumerate() {
            kinds += i8::from(count > 0);
            pairs += i8::from(count >= 2);
            if ORPHANS[tid] {
                orphan_kinds += i8::from(count > 0);
                orphan_pairs += i8::from(count >= 2);
            }
        }

        Self {
            len_div3: m,
            index,
            others,
            counts: *tiles,
            kinds,
            pairs,
            orphan_kinds,
            orphan_pairs,
        }
    }

    /// `calc_all` of the base hand with one more `tid`.
    #[must_use]
    pub fn with_one_more(&self, tid: usize) -> i8 {
        let count = self.counts[tid];
        self.changed(tid, self.index_of(tid) + WEIGHTS[tid], count, count + 1)
    }

    /// `calc_all` of the base hand with one fewer `tid`, which it must hold.
    #[must_use]
    pub fn with_one_fewer(&self, tid: usize) -> i8 {
        let count = self.counts[tid];
        debug_assert!(count > 0, "no {tid} to take away");
        self.changed(tid, self.index_of(tid) - WEIGHTS[tid], count, count - 1)
    }

    const fn index_of(&self, tid: usize) -> usize {
        self.index[Self::group(tid)]
    }

    const fn group(tid: usize) -> usize {
        if tid < 27 { tid / 9 } else { 3 }
    }

    /// `calc_all` of the base hand with `tid`'s count moved from `before` to
    /// `after`, its group's table index thereby `index`, in `calc_all`'s own
    /// order of tests.
    fn changed(&self, tid: usize, index: usize, before: u8, after: u8) -> i8 {
        let group = Self::group(tid);
        let row = if group < 3 {
            SUHAI_TABLE.get(index).copied().unwrap_or_default()
        } else {
            JIHAI_TABLE.get(index).copied().unwrap_or_default()
        };
        let normal = (last(&self.others[group], &row, self.len_div3) as i8) - 1;
        if normal <= 0 || self.len_div3 < 4 {
            return normal;
        }

        // `calc_chitoi`'s count, with the one kind moved.
        let held = |count: u8| i8::from(count > 0);
        let paired = |count: u8| i8::from(count >= 2);
        let kinds = self.kinds - held(before) + held(after);
        let pairs = self.pairs - paired(before) + paired(after);
        let redunct = 7_u8.saturating_sub(kinds as u8) as i8;
        let shanten = normal.min(7 - pairs + redunct - 1);
        if shanten <= 0 {
            return shanten;
        }

        // `calc_kokushi`'s, likewise.
        let (orphan_kinds, orphan_pairs) = if ORPHANS[tid] {
            (
                self.orphan_kinds - held(before) + held(after),
                self.orphan_pairs - paired(before) + paired(after),
            )
        } else {
            (self.orphan_kinds, self.orphan_pairs)
        };
        let redunct = i8::from(orphan_pairs > 0);
        shanten.min(14 - orphan_kinds - redunct - 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;
    use std::{env, thread};

    /// A small generator of random hands, the same on every platform.
    struct Draws(u64);

    impl Draws {
        const fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        const fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// One more of a random kind that still has a tile to give.
        fn add_any(&mut self, tiles: &mut [u8; 34]) {
            loop {
                let tid = self.below(34);
                if tiles[tid] < 4 {
                    tiles[tid] += 1;
                    return;
                }
            }
        }

        /// One fewer of a random kind held.
        fn remove_any(&mut self, tiles: &mut [u8; 34]) {
            loop {
                let tid = self.below(34);
                if tiles[tid] > 0 {
                    tiles[tid] -= 1;
                    return;
                }
            }
        }
    }

    /// A random hand of the kind `kind` names, with `len_div3` sets to make
    /// and three times that plus one or two tiles, never more than four of a
    /// kind.
    fn random_hand(draws: &mut Draws, kind: usize, len_div3: u8) -> [u8; 34] {
        let size = 3 * len_div3 as usize + 1 + draws.below(2);
        let mut tiles = [0_u8; 34];
        match kind {
            // Drawn at random from the wall: mostly far from ready.
            0 => {}
            // Sets and a pair, then up to three tiles taken out and as many
            // put back at random: hands near ready, where the answers go
            // down to -1.
            1 => {
                for _ in 0..len_div3 {
                    let tid = draws.below(34);
                    if tid < 27 && tid % 9 < 7 && draws.below(2) == 0 {
                        if (0..3).all(|k| tiles[tid + k] < 4) {
                            (0..3).for_each(|k| tiles[tid + k] += 1);
                        }
                    } else if tiles[tid] <= 1 {
                        tiles[tid] += 3;
                    }
                }
                let tid = draws.below(34);
                if tiles[tid] <= 2 {
                    tiles[tid] += 2;
                }
                for _ in 0..draws.below(4) {
                    if tiles.iter().any(|&count| count > 0) {
                        draws.remove_any(&mut tiles);
                    }
                }
            }
            // Pairs: seven pairs decides some answers.
            2 => {
                for _ in 0..3 + draws.below(5) {
                    let tid = draws.below(34);
                    if tiles[tid] <= 2 {
                        tiles[tid] += 2;
                    }
                }
            }
            // Terminals and honours: thirteen orphans decides some answers.
            _ => {
                let orphans: Vec<usize> = (0..34).filter(|&tid| ORPHANS[tid]).collect();
                for _ in 0..size {
                    let tid = orphans[draws.below(orphans.len())];
                    if tiles[tid] < 4 {
                        tiles[tid] += 1;
                    }
                }
            }
        }
        while tiles.iter().map(|&count| count as usize).sum::<usize>() > size {
            draws.remove_any(&mut tiles);
        }
        while tiles.iter().map(|&count| count as usize).sum::<usize>() < size {
            draws.add_any(&mut tiles);
        }
        tiles
    }

    /// How many answers were compared, and which part of `calc_all` gave
    /// them.
    #[derive(Default)]
    struct Compared {
        answers: u64,
        ready: u64,
        by_seven_pairs: u64,
        by_thirteen_orphans: u64,
    }

    fn compare(near: &Near, tiles: &[u8; 34], len_div3: u8, compared: &mut Compared) {
        let mut check = |changed: &[u8; 34], got: i8, what: &str| {
            let want = calc_all(changed, len_div3);
            assert_eq!(got, want, "{tiles:?} {what}, {len_div3} sets to make");
            compared.answers += 1;
            compared.ready += u64::from(want <= 0);
            let normal = calc_normal(changed, len_div3);
            if len_div3 == 4 && normal > 0 {
                let chitoi = calc_chitoi(changed);
                compared.by_seven_pairs += u64::from(chitoi < normal);
                compared.by_thirteen_orphans +=
                    u64::from(calc_kokushi(changed) < normal.min(chitoi));
            }
        };
        for tid in 0..34 {
            // A fifth of a kind too, which no lookahead asks for: the table
            // index carries just as `sum_tiles` carries it.
            let mut more = *tiles;
            more[tid] += 1;
            check(
                &more,
                near.with_one_more(tid),
                &format!("and one more {tid}"),
            );
            if tiles[tid] > 0 {
                let mut fewer = *tiles;
                fewer[tid] -= 1;
                check(
                    &fewer,
                    near.with_one_fewer(tid),
                    &format!("and one fewer {tid}"),
                );
            }
        }
    }

    /// `Near` against `calc_all` itself, for every hand one tile away from
    /// random hands of every length and of four kinds (see `random_hand`).
    /// `NEAR_HANDS` sets how many hands of each kind, 20,000 by default; the
    /// long run is `NEAR_HANDS=1000000 cargo test --release
    /// --no-default-features --lib near_is_calc_all -- --nocapture`.
    #[test]
    fn near_is_calc_all() {
        let hands: u64 = env::var("NEAR_HANDS").map_or(20_000, |n| n.parse().unwrap());
        let counts: Vec<Compared> = thread::scope(|scope| {
            // Every kind's thread is spawned before any is joined.
            #[allow(clippy::needless_collect)]
            let workers: Vec<_> = (0..4)
                .map(|kind| {
                    scope.spawn(move || {
                        let mut draws = Draws(0x9E37_79B9_7F4A_7C15 ^ ((kind as u64 + 1) << 32));
                        let mut compared = Compared::default();
                        for _ in 0..hands {
                            let len_div3 = draws.below(5) as u8;
                            let tiles = random_hand(&mut draws, kind, len_div3);
                            let near = Near::new(&tiles, len_div3);
                            compare(&near, &tiles, len_div3, &mut compared);
                        }
                        compared
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect()
        });
        let total = |part: fn(&Compared) -> u64| counts.iter().map(part).sum::<u64>();
        println!(
            "{} hands, {} answers, every one calc_all's: {} ready or complete, {} given by \
             seven pairs, {} by thirteen orphans",
            4 * hands,
            total(|c| c.answers),
            total(|c| c.ready),
            total(|c| c.by_seven_pairs),
            total(|c| c.by_thirteen_orphans),
        );
        // Every part of `calc_all` gave some of the answers compared.
        assert!(total(|c| c.ready) >= hands);
        assert!(total(|c| c.by_seven_pairs) >= hands / 10);
        assert!(total(|c| c.by_thirteen_orphans) >= hands / 10);
    }

    #[test]
    fn calc_3n_plus_1() {
        let tehai = hand("1111m 333p 222s 444z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("147m 258p 369s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 6);
        let tehai = hand("468m 33346p 7s").unwrap();
        assert_eq!(calc_all(&tehai, 3), 2);
        let tehai = hand("147m 258p 3s").unwrap();
        assert_eq!(calc_all(&tehai, 2), 4);
        let tehai = hand("4455s").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("7z").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("15559m 19p 19s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("9999m 6677p 88s 355z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("19m 19p 159s 123456z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
    }

    #[test]
    fn calc_3n_plus_2() {
        let tehai = hand("2344456m 14p 127s 2z 7p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("2344456m 14p 127s 2z 5p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9m").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("122334m 678p 37s 22z 5s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("122334m 678p 12s 22z 4s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("12223456m 78889p 2m").unwrap();
        assert_eq!(calc_all(&tehai, 4), -1);
        let tehai = hand("34778p").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("34s").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("55m").unwrap();
        assert_eq!(calc_all(&tehai, 0), -1);
    }
}
