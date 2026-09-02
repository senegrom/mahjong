//! A small deterministic generator, so a game is reproducible from its seed.
//!
//! Games are replayed from a seed and a list of actions, which is what makes
//! logs, tests and training data reproducible. The generator is xoshiro256++
//! seeded through SplitMix64: fast, well distributed, and short enough to
//! carry with the engine rather than pull in a dependency. It is not
//! cryptographic and does not need to be.

/// A seeded pseudorandom generator.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Builds a generator from a seed. The same seed always gives the same
    /// sequence, on every platform.
    pub fn from_seed(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            state: [next(), next(), next(), next()],
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// A number in `0..limit`, without modulo bias.
    pub fn below(&mut self, limit: usize) -> usize {
        assert!(limit > 0, "limit must be positive");
        let limit = limit as u64;
        let zone = u64::MAX - u64::MAX % limit;
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % limit) as usize;
            }
        }
    }

    /// Shuffles a slice in place, Fisher and Yates.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = self.below(index + 1);
            items.swap(index, other);
        }
    }

    /// A dice roll of two dice, 2 to 12, as East makes to break the wall
    /// (EMA 2025 section 2.5).
    pub fn roll_dice(&mut self) -> u8 {
        (self.below(6) + self.below(6) + 2) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let mut first = Rng::from_seed(20260902);
        let mut second = Rng::from_seed(20260902);
        for _ in 0..100 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        let mut other = Rng::from_seed(20260903);
        assert_ne!(first.next_u64(), other.next_u64());
    }

    #[test]
    fn shuffling_keeps_every_item() {
        let mut rng = Rng::from_seed(7);
        let mut items: Vec<usize> = (0..136).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..136).collect::<Vec<_>>());
        assert_ne!(items, sorted, "a shuffle of 136 items should move some");
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = Rng::from_seed(11);
        let mut seen = [false; 6];
        for _ in 0..1000 {
            let value = rng.below(6);
            assert!(value < 6);
            seen[value] = true;
        }
        assert!(seen.iter().all(|hit| *hit));
    }

    #[test]
    fn dice_land_between_two_and_twelve() {
        let mut rng = Rng::from_seed(3);
        for _ in 0..1000 {
            let roll = rng.roll_dice();
            assert!((2..=12).contains(&roll));
        }
    }
}
