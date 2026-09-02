//! Tiles: the 34 kinds of an EMA riichi set, and the relations between them.
//!
//! A set has four identical tiles of each kind and no others: no red fives,
//! no flowers, no jokers (EMA 2025, section 1.3).
//!
//! Kinds are numbered so that the three suits run 1 to 9 in blocks, then the
//! four winds in the seating order east, south, west, north, then the three
//! dragons in the rulebook's order white, green, red (sections 1.1 and 1.2):
//!
//! ```text
//!  0..8   1m..9m   characters (manzu)
//!  9..17  1p..9p   circles (pinzu)
//! 18..26  1s..9s   bamboo (souzu)
//! 27..30  E S W N
//! 31..33  White Green Red
//! ```

use core::fmt;
use core::str::FromStr;

/// Number of distinct tile kinds in a riichi set.
pub const KINDS: usize = 34;
/// Copies of each kind in a full set.
pub const COPIES: u8 = 4;
/// Tiles in a full set: 34 kinds times 4 copies.
pub const SET_SIZE: usize = KINDS * COPIES as usize;

/// The suit a tile belongs to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Suit {
    /// Characters, manzu, written `m`.
    Characters,
    /// Circles, pinzu, written `p`.
    Circles,
    /// Bamboo, souzu, written `s`.
    Bamboo,
    /// Winds and dragons, written `z`.
    Honours,
}

impl Suit {
    /// The letter used in the standard hand notation.
    pub const fn letter(self) -> char {
        match self {
            Suit::Characters => 'm',
            Suit::Circles => 'p',
            Suit::Bamboo => 's',
            Suit::Honours => 'z',
        }
    }

    /// Whether this suit has numbered tiles that can form sequences.
    pub const fn is_numbered(self) -> bool {
        !matches!(self, Suit::Honours)
    }

    /// Index of the first kind of this suit.
    pub const fn base(self) -> u8 {
        match self {
            Suit::Characters => 0,
            Suit::Circles => 9,
            Suit::Bamboo => 18,
            Suit::Honours => 27,
        }
    }
}

/// One of the 34 tile kinds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tile(u8);

/// East wind.
pub const EAST: Tile = Tile(27);
/// South wind.
pub const SOUTH: Tile = Tile(28);
/// West wind.
pub const WEST: Tile = Tile(29);
/// North wind.
pub const NORTH: Tile = Tile(30);
/// White dragon (haku).
pub const WHITE: Tile = Tile(31);
/// Green dragon (hatsu).
pub const GREEN: Tile = Tile(32);
/// Red dragon (chun).
pub const RED: Tile = Tile(33);

impl Tile {
    /// Builds a tile from its kind index, panicking outside `0..34`.
    pub const fn new(index: u8) -> Tile {
        assert!((index as usize) < KINDS, "tile index out of range");
        Tile(index)
    }

    /// Builds a tile from its kind index, or `None` outside `0..34`.
    pub const fn try_new(index: u8) -> Option<Tile> {
        if (index as usize) < KINDS {
            Some(Tile(index))
        } else {
            None
        }
    }

    /// Builds a numbered tile, e.g. `Tile::numbered(Suit::Circles, 5)` for 5p.
    ///
    /// For [`Suit::Honours`] the rank runs 1 to 7 in the order east, south,
    /// west, north, white, green, red.
    pub const fn numbered(suit: Suit, rank: u8) -> Tile {
        let limit = if suit.is_numbered() { 9 } else { 7 };
        assert!(rank >= 1 && rank <= limit, "rank out of range for suit");
        Tile(suit.base() + rank - 1)
    }

    /// The kind index, `0..34`.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// The kind index as a `usize`, for indexing count arrays.
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    /// The suit this tile belongs to.
    pub const fn suit(self) -> Suit {
        match self.0 {
            0..=8 => Suit::Characters,
            9..=17 => Suit::Circles,
            18..=26 => Suit::Bamboo,
            _ => Suit::Honours,
        }
    }

    /// The rank within the suit: 1 to 9 for numbers, 1 to 7 for honours.
    pub const fn rank(self) -> u8 {
        self.0 - self.suit().base() + 1
    }

    /// Whether this is a wind or a dragon.
    pub const fn is_honour(self) -> bool {
        self.0 >= 27
    }

    /// Whether this is a wind.
    pub const fn is_wind(self) -> bool {
        self.0 >= 27 && self.0 <= 30
    }

    /// Whether this is a dragon.
    pub const fn is_dragon(self) -> bool {
        self.0 >= 31
    }

    /// Whether this is a 1 or a 9 of a numbered suit (EMA section 1.1).
    pub const fn is_terminal(self) -> bool {
        !self.is_honour() && (self.rank() == 1 || self.rank() == 9)
    }

    /// Whether this is a terminal or an honour, the tiles the outside-hand
    /// and all-terminals-and-honours yaku are built from.
    pub const fn is_terminal_or_honour(self) -> bool {
        self.is_honour() || self.is_terminal()
    }

    /// Whether this is a simple, i.e. a 2 to 8 of a numbered suit.
    pub const fn is_simple(self) -> bool {
        !self.is_terminal_or_honour()
    }

    /// Whether this counts as green for All Green (EMA section 4.2.6):
    /// the green dragon and 2, 3, 4, 6 and 8 of bamboo.
    pub const fn is_green(self) -> bool {
        if self.0 == GREEN.0 {
            return true;
        }
        if !matches!(self.suit(), Suit::Bamboo) {
            return false;
        }
        matches!(self.rank(), 2 | 3 | 4 | 6 | 8)
    }

    /// The tile that is dora when this tile is the indicator (section 2.7).
    ///
    /// Numbers advance within their suit and 9 points back to 1; winds follow
    /// east, south, west, north, east; dragons follow red, white, green, red.
    pub const fn dora(self) -> Tile {
        match self.suit() {
            Suit::Honours => {
                if self.is_wind() {
                    // 27..30 cycling.
                    Tile(27 + (self.0 - 27 + 1) % 4)
                } else {
                    // White -> green -> red -> white, i.e. 31..33 cycling.
                    Tile(31 + (self.0 - 31 + 1) % 3)
                }
            }
            suit => {
                let rank = self.rank();
                let next = if rank == 9 { 1 } else { rank + 1 };
                Tile(suit.base() + next - 1)
            }
        }
    }

    /// The next tile up in the same numbered suit, or `None` past 9 and for
    /// honours. Sequences are built with this, so 8-9-1 never forms one
    /// (EMA section 3.2).
    pub const fn next_in_suit(self) -> Option<Tile> {
        if self.is_honour() || self.rank() == 9 {
            None
        } else {
            Some(Tile(self.0 + 1))
        }
    }

    /// Every tile kind, in index order.
    pub fn all() -> impl Iterator<Item = Tile> {
        (0..KINDS as u8).map(Tile)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit().letter())
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// Error returned when a tile or hand string cannot be parsed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

impl FromStr for Tile {
    type Err = ParseError;

    /// Parses one tile in the standard notation, e.g. `3p` or `7z`.
    fn from_str(text: &str) -> Result<Tile, ParseError> {
        let mut chars = text.chars();
        let (rank, letter) = match (chars.next(), chars.next(), chars.next()) {
            (Some(rank), Some(letter), None) => (rank, letter),
            _ => return Err(ParseError(format!("not a tile: {text:?}"))),
        };
        let rank =
            rank.to_digit(10)
                .ok_or_else(|| ParseError(format!("not a tile rank: {rank:?}")))? as u8;
        let suit = match letter {
            'm' => Suit::Characters,
            'p' => Suit::Circles,
            's' => Suit::Bamboo,
            'z' => Suit::Honours,
            other => return Err(ParseError(format!("not a suit letter: {other:?}"))),
        };
        let limit = if suit.is_numbered() { 9 } else { 7 };
        if rank < 1 || rank > limit {
            return Err(ParseError(format!("rank {rank} out of range for {letter}")));
        }
        Ok(Tile::numbered(suit, rank))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_and_names_round_trip() {
        for tile in Tile::all() {
            let text = tile.to_string();
            assert_eq!(text.parse::<Tile>().unwrap(), tile, "round trip of {text}");
        }
        assert_eq!("1m".parse::<Tile>().unwrap().index(), 0);
        assert_eq!("9m".parse::<Tile>().unwrap().index(), 8);
        assert_eq!("1p".parse::<Tile>().unwrap().index(), 9);
        assert_eq!("1s".parse::<Tile>().unwrap().index(), 18);
        assert_eq!("1z".parse::<Tile>().unwrap(), EAST);
        assert_eq!("7z".parse::<Tile>().unwrap(), RED);
    }

    #[test]
    fn bad_tiles_are_rejected() {
        for text in ["", "1", "m1", "0m", "10m", "8z", "0z", "1x", "11m"] {
            assert!(text.parse::<Tile>().is_err(), "{text:?} should not parse");
        }
    }

    /// EMA 2025 section 1.1: ones and nines are terminals; 2 to 8 are simples.
    #[test]
    fn terminals_honours_and_simples() {
        assert!("1m".parse::<Tile>().unwrap().is_terminal());
        assert!("9s".parse::<Tile>().unwrap().is_terminal());
        assert!(!"5p".parse::<Tile>().unwrap().is_terminal());
        assert!("5p".parse::<Tile>().unwrap().is_simple());
        assert!(EAST.is_honour() && EAST.is_wind() && !EAST.is_dragon());
        assert!(RED.is_dragon() && RED.is_terminal_or_honour() && !RED.is_simple());
    }

    /// EMA 2025 section 2.7: a 9 points to the 1 of its suit, winds follow
    /// east to south to west to north to east, dragons red to white to green
    /// to red.
    #[test]
    fn dora_indicator_chains() {
        assert_eq!("6s".parse::<Tile>().unwrap().dora().to_string(), "7s");
        assert_eq!("9m".parse::<Tile>().unwrap().dora().to_string(), "1m");
        assert_eq!("9p".parse::<Tile>().unwrap().dora().to_string(), "1p");
        assert_eq!(EAST.dora(), SOUTH);
        assert_eq!(SOUTH.dora(), WEST);
        assert_eq!(WEST.dora(), NORTH);
        assert_eq!(NORTH.dora(), EAST);
        assert_eq!(RED.dora(), WHITE);
        assert_eq!(WHITE.dora(), GREEN);
        assert_eq!(GREEN.dora(), RED);
    }

    /// EMA 2025 section 4.2.6, All Green: green dragon and 2, 3, 4, 6, 8 bamboo.
    #[test]
    fn green_tiles() {
        let green: Vec<String> = Tile::all()
            .filter(|tile| tile.is_green())
            .map(|tile| tile.to_string())
            .collect();
        assert_eq!(green, ["2s", "3s", "4s", "6s", "8s", "6z"]);
    }

    /// EMA 2025 section 3.2: 8-9-1 is not a sequence.
    #[test]
    fn sequences_do_not_wrap() {
        assert_eq!(
            "8m".parse::<Tile>()
                .unwrap()
                .next_in_suit()
                .unwrap()
                .to_string(),
            "9m"
        );
        assert!("9m".parse::<Tile>().unwrap().next_in_suit().is_none());
        assert!(EAST.next_in_suit().is_none());
    }
}
