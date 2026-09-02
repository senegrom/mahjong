//! Counted collections of tiles, and the sets a hand is built from.
//!
//! A hand's concealed tiles are held as counts per kind, which is what the
//! shanten, winning-hand and scoring code all want. Called sets keep their
//! own shape because scoring distinguishes melded from concealed sets and
//! needs to know who was called from (EMA 2025, sections 3.3.3 to 3.3.6 and
//! 4.1.1).

use core::fmt;
use core::str::FromStr;

use crate::tile::{ParseError, Suit, Tile, COPIES, KINDS};

/// A multiset of tiles, counted by kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileSet {
    counts: [u8; KINDS],
}

impl TileSet {
    /// An empty set.
    pub const fn new() -> TileSet {
        TileSet { counts: [0; KINDS] }
    }

    /// Builds from raw counts.
    pub const fn from_counts(counts: [u8; KINDS]) -> TileSet {
        TileSet { counts }
    }

    /// Collects tiles into a set.
    pub fn from_tiles<I: IntoIterator<Item = Tile>>(tiles: I) -> TileSet {
        let mut set = TileSet::new();
        for tile in tiles {
            set.add(tile);
        }
        set
    }

    /// The counts, indexed by tile kind.
    pub const fn counts(&self) -> &[u8; KINDS] {
        &self.counts
    }

    /// Mutable counts, for the hot paths that rewrite them in place.
    pub fn counts_mut(&mut self) -> &mut [u8; KINDS] {
        &mut self.counts
    }

    /// How many copies of one kind are present.
    pub const fn count(&self, tile: Tile) -> u8 {
        self.counts[tile.idx()]
    }

    /// Total number of tiles.
    pub fn len(&self) -> usize {
        self.counts.iter().map(|&n| n as usize).sum()
    }

    /// Whether the set holds no tiles.
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&n| n == 0)
    }

    /// Number of distinct kinds present.
    pub fn distinct(&self) -> usize {
        self.counts.iter().filter(|&&n| n > 0).count()
    }

    /// Adds one tile.
    pub fn add(&mut self, tile: Tile) {
        self.counts[tile.idx()] += 1;
    }

    /// Adds `n` copies of a tile.
    pub fn add_n(&mut self, tile: Tile, n: u8) {
        self.counts[tile.idx()] += n;
    }

    /// Removes one tile, returning whether it was there.
    pub fn remove(&mut self, tile: Tile) -> bool {
        if self.counts[tile.idx()] == 0 {
            false
        } else {
            self.counts[tile.idx()] -= 1;
            true
        }
    }

    /// Every tile in the set, kind by kind, repeated by its count.
    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.counts
            .iter()
            .enumerate()
            .flat_map(|(index, &count)| {
                (0..count).map(move |_| Tile::new(index as u8))
            })
    }

    /// Whether no kind appears more than the four copies a set contains.
    pub fn is_legal(&self) -> bool {
        self.counts.iter().all(|&n| n <= COPIES)
    }
}

impl Default for TileSet {
    fn default() -> TileSet {
        TileSet::new()
    }
}

impl fmt::Display for TileSet {
    /// The standard notation, suits in the order m, p, s, z, e.g.
    /// `123m456p789s11z`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for suit in [Suit::Characters, Suit::Circles, Suit::Bamboo, Suit::Honours] {
            let limit = if suit.is_numbered() { 9 } else { 7 };
            let mut wrote = false;
            for rank in 1..=limit {
                let tile = Tile::numbered(suit, rank);
                for _ in 0..self.count(tile) {
                    write!(f, "{rank}")?;
                    wrote = true;
                }
            }
            if wrote {
                f.write_str(&suit.letter().to_string())?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for TileSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl FromStr for TileSet {
    type Err = ParseError;

    /// Parses the standard notation, where digits are followed by the suit
    /// letter they belong to, e.g. `123m456p789s11z`.
    fn from_str(text: &str) -> Result<TileSet, ParseError> {
        let mut set = TileSet::new();
        let mut pending: Vec<u8> = Vec::new();
        for character in text.chars() {
            if character.is_ascii_whitespace() {
                continue;
            }
            if let Some(digit) = character.to_digit(10) {
                pending.push(digit as u8);
                continue;
            }
            let suit = match character {
                'm' => Suit::Characters,
                'p' => Suit::Circles,
                's' => Suit::Bamboo,
                'z' => Suit::Honours,
                other => return Err(ParseError(format!("not a suit letter: {other:?}"))),
            };
            if pending.is_empty() {
                return Err(ParseError(format!("suit {character:?} without ranks")));
            }
            let limit = if suit.is_numbered() { 9 } else { 7 };
            for rank in pending.drain(..) {
                if rank < 1 || rank > limit {
                    return Err(ParseError(format!(
                        "rank {rank} out of range for {character}"
                    )));
                }
                set.add(Tile::numbered(suit, rank));
            }
        }
        if !pending.is_empty() {
            return Err(ParseError("ranks without a suit letter".into()));
        }
        Ok(set)
    }
}

/// How a set of three or four tiles came about, which decides both the
/// concealed status of the hand and the minipoints the set is worth.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MeldKind {
    /// Claimed from a discard: three consecutive tiles of one suit.
    Chii,
    /// Claimed from a discard: three identical tiles.
    Pon,
    /// A quad claimed from a discard (EMA section 3.3.4).
    ClaimedKan,
    /// A melded triplet extended to a quad by the fourth tile.
    ExtendedKan,
    /// A quad declared from the hand, which leaves the hand concealed.
    ConcealedKan,
}

impl MeldKind {
    /// Whether this set is a quad.
    pub const fn is_kan(self) -> bool {
        matches!(
            self,
            MeldKind::ClaimedKan | MeldKind::ExtendedKan | MeldKind::ConcealedKan
        )
    }

    /// Whether declaring this set opens the hand. A concealed quad does not
    /// (EMA section 3.3.4).
    pub const fn opens_hand(self) -> bool {
        !matches!(self, MeldKind::ConcealedKan)
    }

    /// Whether the set counts as concealed when minipoints are counted
    /// (EMA section 4.1.1).
    pub const fn is_concealed_for_fu(self) -> bool {
        matches!(self, MeldKind::ConcealedKan)
    }
}

/// Which player a tile was claimed from, relative to the claimer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClaimedFrom {
    /// The player to the left, the only source a sequence may be claimed from.
    Left,
    /// The player across the table.
    Across,
    /// The player to the right.
    Right,
    /// No claim: a concealed quad.
    SelfDrawn,
}

/// A called set in front of a player.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Meld {
    /// What kind of set this is.
    pub kind: MeldKind,
    /// For a triplet or quad, the tile itself; for a sequence, its lowest tile.
    pub tile: Tile,
    /// Who the claimed tile came from.
    pub from: ClaimedFrom,
}

impl Meld {
    /// A claimed sequence, named by its lowest tile.
    pub fn chii(lowest: Tile, from: ClaimedFrom) -> Meld {
        Meld { kind: MeldKind::Chii, tile: lowest, from }
    }

    /// A claimed triplet.
    pub fn pon(tile: Tile, from: ClaimedFrom) -> Meld {
        Meld { kind: MeldKind::Pon, tile, from }
    }

    /// A quad declared from the hand.
    pub fn concealed_kan(tile: Tile) -> Meld {
        Meld { kind: MeldKind::ConcealedKan, tile, from: ClaimedFrom::SelfDrawn }
    }

    /// The tiles this set occupies, four for a quad.
    pub fn tiles(&self) -> Vec<Tile> {
        match self.kind {
            MeldKind::Chii => {
                let second = self.tile.next_in_suit().expect("sequence starts below 8");
                let third = second.next_in_suit().expect("sequence starts below 8");
                vec![self.tile, second, third]
            }
            MeldKind::Pon => vec![self.tile; 3],
            _ => vec![self.tile; 4],
        }
    }

    /// Whether this set is a sequence.
    pub const fn is_sequence(&self) -> bool {
        matches!(self.kind, MeldKind::Chii)
    }

    /// Whether this set is a triplet or a quad.
    pub const fn is_triplet_or_quad(&self) -> bool {
        !self.is_sequence()
    }

    /// Whether every tile of the set is a terminal or an honour.
    pub fn is_all_terminal_or_honour(&self) -> bool {
        self.tiles().iter().all(|tile| tile.is_terminal_or_honour())
    }

    /// Whether the set contains at least one terminal or honour, which is
    /// what the outside-hand yaku ask of every set.
    pub fn has_terminal_or_honour(&self) -> bool {
        self.tiles().iter().any(|tile| tile.is_terminal_or_honour())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_standard_notation() {
        let hand: TileSet = "123m456p789s11z".parse().unwrap();
        assert_eq!(hand.len(), 11);
        assert_eq!(hand.to_string(), "123m456p789s11z");
        assert_eq!(hand.count(Tile::numbered(Suit::Characters, 1)), 1);
        assert_eq!(hand.count(crate::tile::EAST), 2);
        assert_eq!(hand.distinct(), 10);
    }

    #[test]
    fn parsing_rejects_malformed_hands() {
        for text in ["123", "m123", "0m", "8z", "123x"] {
            assert!(text.parse::<TileSet>().is_err(), "{text:?} should not parse");
        }
    }

    #[test]
    fn tiles_iterates_with_repeats() {
        let hand: TileSet = "111m2p".parse().unwrap();
        let tiles: Vec<String> = hand.tiles().map(|tile| tile.to_string()).collect();
        assert_eq!(tiles, ["1m", "1m", "1m", "2p"]);
    }

    #[test]
    fn a_fifth_copy_is_illegal() {
        let mut hand: TileSet = "1111m".parse().unwrap();
        assert!(hand.is_legal());
        hand.add(Tile::numbered(Suit::Characters, 1));
        assert!(!hand.is_legal());
    }

    /// EMA 2025 section 3.3.4: a concealed quad leaves the hand concealed,
    /// every other call opens it.
    #[test]
    fn only_a_concealed_quad_keeps_the_hand_closed() {
        assert!(!MeldKind::ConcealedKan.opens_hand());
        for kind in [
            MeldKind::Chii,
            MeldKind::Pon,
            MeldKind::ClaimedKan,
            MeldKind::ExtendedKan,
        ] {
            assert!(kind.opens_hand(), "{kind:?} opens the hand");
        }
    }

    #[test]
    fn meld_tiles_expand() {
        let chii = Meld::chii("3p".parse().unwrap(), ClaimedFrom::Left);
        let tiles: Vec<String> = chii.tiles().iter().map(|t| t.to_string()).collect();
        assert_eq!(tiles, ["3p", "4p", "5p"]);
        assert_eq!(Meld::pon("5z".parse().unwrap(), ClaimedFrom::Across).tiles().len(), 3);
        assert_eq!(Meld::concealed_kan("1m".parse().unwrap()).tiles().len(), 4);
    }
}
