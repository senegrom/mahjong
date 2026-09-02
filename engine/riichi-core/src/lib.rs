//! A riichi mahjong rules engine following the **EMA Riichi Competition
//! Rules, 2025 edition** (version 1.1, August 2025), in force since
//! 1 January 2026.
//!
//! The rulebook is kept in `docs/rules/` of this repository and the tests
//! cite its section numbers, so a disputed behaviour can be traced to the
//! sentence it comes from.
//!
//! The engine is the single source of the rules for this project: it is
//! compiled to WebAssembly for the browser game and to a Python extension
//! for self-play training, so the opponents learn on exactly the code the
//! humans play against.
//!
//! # Layout
//!
//! - [`tile`] the 34 kinds and the relations between them
//! - [`hand`] counted tiles and called sets
//! - [`shanten`] distance to a complete hand, waits and acceptance
//! - [`agari`] every reading of a complete hand
//! - [`yaku`] the scoring patterns
//! - [`score`] han, minipoints and payments
//! - [`wall`] the wall, dead wall and dora indicators
//! - [`rng`] the seeded generator that makes games reproducible
//!
//! Still to come: the turn state machine.
//!
//! ```
//! use riichi_core::hand::TileSet;
//! use riichi_core::shanten;
//!
//! let waiting: TileSet = "123m456m789m11s34p".parse().unwrap();
//! assert_eq!(shanten::shanten(&waiting, 0), shanten::TENPAI);
//! assert_eq!(shanten::waits(&waiting, 0, &waiting).to_string(), "25p");
//! ```

#![forbid(unsafe_code)]

pub mod agari;
pub mod hand;
pub mod rng;
pub mod score;
pub mod shanten;
pub mod tile;
pub mod wall;
pub mod yaku;

pub use hand::{ClaimedFrom, Meld, MeldKind, TileSet};
pub use tile::{ParseError, Suit, Tile};

/// Which rulebook the engine is following.
///
/// Only [`RuleSet::Ema2025`] is implemented. The value exists so that other
/// rule sets, which differ in red fives, abortive draws, counted yakuman and
/// the winner bonus, can be added later without game logic having to change
/// shape around them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum RuleSet {
    /// EMA Riichi Competition Rules, 2025 edition.
    #[default]
    Ema2025,
}

/// The four seats, in the counter-clockwise turn order east, south, west,
/// north (EMA 2025, section 2.1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Wind {
    /// The dealer's seat.
    East,
    /// To East's right.
    South,
    /// Across from East.
    West,
    /// To East's left.
    North,
}

impl Wind {
    /// All four seats in turn order.
    pub const ALL: [Wind; 4] = [Wind::East, Wind::South, Wind::West, Wind::North];

    /// The seat's index, 0 for East.
    pub const fn index(self) -> usize {
        match self {
            Wind::East => 0,
            Wind::South => 1,
            Wind::West => 2,
            Wind::North => 3,
        }
    }

    /// The seat `steps` places later in turn order.
    pub const fn plus(self, steps: usize) -> Wind {
        Wind::ALL[(self.index() + steps) % 4]
    }

    /// The next seat in turn order.
    pub const fn next(self) -> Wind {
        self.plus(1)
    }

    /// The tile that is this wind, which is what a triplet of it scores on.
    pub const fn tile(self) -> Tile {
        Tile::new(27 + self.index() as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EMA 2025 section 2.1: the order is east, south, west, north, and it
    /// runs counter-clockwise, so South sits at East's right.
    #[test]
    fn seats_run_in_turn_order() {
        assert_eq!(Wind::East.next(), Wind::South);
        assert_eq!(Wind::South.next(), Wind::West);
        assert_eq!(Wind::West.next(), Wind::North);
        assert_eq!(Wind::North.next(), Wind::East);
        assert_eq!(Wind::East.plus(4), Wind::East);
    }

    #[test]
    fn seat_winds_map_to_their_tiles() {
        assert_eq!(Wind::East.tile().to_string(), "1z");
        assert_eq!(Wind::North.tile().to_string(), "4z");
    }
}
