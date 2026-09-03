//! Yaku: the scoring patterns a winning hand must have at least one of.
//!
//! Values follow the 2025 classification, which lists every yaku at its
//! closed value and marks the ones that lose a han when the hand is open
//! (EMA 2025, section 4.2). Yakuman are not cumulative, and EMA has no
//! counted yakuman: eleven han or more is a sanbaiman.

use crate::hand::MeldKind;
use crate::tile::{Suit, Tile};
use crate::Wind;

/// A scoring pattern.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Yaku {
    /// Riichi: a concealed waiting hand declared with a bet.
    Riichi,
    /// Ippatsu: winning within the first uninterrupted set of turns after riichi.
    Ippatsu,
    /// Fully Concealed Hand: winning by self-draw with a concealed hand.
    MenzenTsumo,
    /// Pinfu: four sequences, a valueless pair and a two-sided wait.
    Pinfu,
    /// Pure Double Sequence: two identical sequences, concealed.
    Iipeiko,
    /// All Simples: no terminals and no honours.
    Tanyao,
    /// A triplet or quad of dragons, one yaku for each.
    YakuhaiDragon(Tile),
    /// A triplet or quad of the player's seat wind.
    YakuhaiSeatWind,
    /// A triplet or quad of the round wind.
    YakuhaiRoundWind,
    /// After a Quad: winning on the replacement tile.
    Rinshan,
    /// Robbing a Quad.
    Chankan,
    /// Under the Sea: self-draw on the last tile in the wall.
    Haitei,
    /// Under the River: winning on the last discard.
    Houtei,
    /// Double Riichi: riichi in the first uninterrupted set of turns.
    DoubleRiichi,
    /// Seven Pairs.
    Chiitoitsu,
    /// Mixed Triple Sequence: the same sequence in each suit.
    Sanshoku,
    /// Pure Straight: 1-2-3, 4-5-6 and 7-8-9 in one suit.
    Ittsuu,
    /// Half Outside Hand: every set and the pair hold a terminal or honour,
    /// the hand holds honours, and at least one set is a sequence.
    Chanta,
    /// Triple Triplet: the same triplet in each suit.
    SanshokuDoukou,
    /// Three Concealed Triplets.
    SanAnkou,
    /// Three Quads.
    SanKantsu,
    /// All Triplets.
    Toitoi,
    /// Little Three Dragons.
    Shousangen,
    /// All Terminals and Honours.
    Honroutou,
    /// Twice Pure Double Sequence, concealed.
    Ryanpeikou,
    /// Half Flush: one suit plus honours.
    Honitsu,
    /// Full Outside Hand: every set and the pair hold a terminal, with at
    /// least one sequence and no honours.
    Junchan,
    /// Blessing of Man: winning by discard before one's first turn.
    Renhou,
    /// Full Flush: one suit and nothing else.
    Chinitsu,
    /// Thirteen Orphans.
    KokushiMusou,
    /// Nine Gates.
    ChuurenPoutou,
    /// Blessing of Heaven: the dealer wins on the starting hand.
    Tenhou,
    /// Blessing of Earth: self-draw in the first uninterrupted set of turns.
    Chiihou,
    /// Four Concealed Triplets.
    SuuAnkou,
    /// Four Quads.
    SuuKantsu,
    /// All Green.
    Ryuuiisou,
    /// All Terminals.
    Chinroutou,
    /// All Honours.
    Tsuuiisou,
    /// Big Three Dragons.
    Daisangen,
    /// Little Four Winds.
    ShouSuushii,
    /// Big Four Winds.
    DaiSuushii,
}

impl Yaku {
    /// Every yaku the rules list, so a test can walk the whole set.
    ///
    /// The dragon yaku carries the tile it was made of, and one of each is
    /// listed here; nothing depends on which, since they share a name and a
    /// value. Adding a yaku without adding it here fails the test that
    /// checks this list against the rulebook, which is the point of it.
    pub const ALL: [Yaku; 41] = [
        Yaku::Riichi,
        Yaku::Ippatsu,
        Yaku::MenzenTsumo,
        Yaku::Pinfu,
        Yaku::Iipeiko,
        Yaku::Tanyao,
        Yaku::YakuhaiDragon(crate::tile::WHITE),
        Yaku::YakuhaiSeatWind,
        Yaku::YakuhaiRoundWind,
        Yaku::Rinshan,
        Yaku::Chankan,
        Yaku::Haitei,
        Yaku::Houtei,
        Yaku::DoubleRiichi,
        Yaku::Chiitoitsu,
        Yaku::Sanshoku,
        Yaku::Ittsuu,
        Yaku::Chanta,
        Yaku::SanshokuDoukou,
        Yaku::SanAnkou,
        Yaku::SanKantsu,
        Yaku::Toitoi,
        Yaku::Shousangen,
        Yaku::Honroutou,
        Yaku::Ryanpeikou,
        Yaku::Honitsu,
        Yaku::Junchan,
        Yaku::Renhou,
        Yaku::Chinitsu,
        Yaku::KokushiMusou,
        Yaku::ChuurenPoutou,
        Yaku::Tenhou,
        Yaku::Chiihou,
        Yaku::SuuAnkou,
        Yaku::SuuKantsu,
        Yaku::Ryuuiisou,
        Yaku::Chinroutou,
        Yaku::Tsuuiisou,
        Yaku::Daisangen,
        Yaku::ShouSuushii,
        Yaku::DaiSuushii,
    ];

    /// Whether this yaku is a yakuman.
    pub const fn is_yakuman(self) -> bool {
        matches!(
            self,
            Yaku::KokushiMusou
                | Yaku::ChuurenPoutou
                | Yaku::Tenhou
                | Yaku::Chiihou
                | Yaku::SuuAnkou
                | Yaku::SuuKantsu
                | Yaku::Ryuuiisou
                | Yaku::Chinroutou
                | Yaku::Tsuuiisou
                | Yaku::Daisangen
                | Yaku::ShouSuushii
                | Yaku::DaiSuushii
        )
    }

    /// Whether the yaku requires a concealed hand, i.e. the ones printed in
    /// italics in the rulebook's yaku list.
    pub const fn requires_concealed(self) -> bool {
        matches!(
            self,
            Yaku::Riichi
                | Yaku::Ippatsu
                | Yaku::MenzenTsumo
                | Yaku::Pinfu
                | Yaku::Iipeiko
                | Yaku::DoubleRiichi
                | Yaku::Chiitoitsu
                | Yaku::Ryanpeikou
                | Yaku::Renhou
                | Yaku::KokushiMusou
                | Yaku::ChuurenPoutou
                | Yaku::Tenhou
                | Yaku::Chiihou
        )
    }

    /// The han this yaku is worth, given whether the hand is open.
    ///
    /// The underlined yaku of the rulebook's list lose one han when open.
    pub const fn han(self, open: bool) -> u8 {
        match self {
            Yaku::Riichi
            | Yaku::Ippatsu
            | Yaku::MenzenTsumo
            | Yaku::Pinfu
            | Yaku::Iipeiko
            | Yaku::Tanyao
            | Yaku::YakuhaiDragon(_)
            | Yaku::YakuhaiSeatWind
            | Yaku::YakuhaiRoundWind
            | Yaku::Rinshan
            | Yaku::Chankan
            | Yaku::Haitei
            | Yaku::Houtei => 1,

            Yaku::DoubleRiichi
            | Yaku::Chiitoitsu
            | Yaku::SanshokuDoukou
            | Yaku::SanAnkou
            | Yaku::SanKantsu
            | Yaku::Toitoi
            | Yaku::Shousangen
            | Yaku::Honroutou => 2,

            Yaku::Sanshoku | Yaku::Ittsuu | Yaku::Chanta => {
                if open {
                    1
                } else {
                    2
                }
            }

            Yaku::Ryanpeikou => 3,
            Yaku::Honitsu | Yaku::Junchan => {
                if open {
                    2
                } else {
                    3
                }
            }

            Yaku::Renhou => 5,
            Yaku::Chinitsu => {
                if open {
                    5
                } else {
                    6
                }
            }

            _ => 13, // Yakuman are paid from their own table, not by han.
        }
    }

    /// The rulebook's name for the yaku, romanised as printed there.
    pub const fn name(self) -> &'static str {
        match self {
            Yaku::Riichi => "Riichi",
            Yaku::Ippatsu => "Ippatsu",
            Yaku::MenzenTsumo => "Fully Concealed Hand",
            Yaku::Pinfu => "Pinfu",
            Yaku::Iipeiko => "Pure Double Sequence",
            Yaku::Tanyao => "All Simples",
            Yaku::YakuhaiDragon(_) => "Dragon Triplet",
            Yaku::YakuhaiSeatWind => "Seat Wind Triplet",
            Yaku::YakuhaiRoundWind => "Round Wind Triplet",
            Yaku::Rinshan => "After a Quad",
            Yaku::Chankan => "Robbing a Quad",
            Yaku::Haitei => "Under the Sea",
            Yaku::Houtei => "Under the River",
            Yaku::DoubleRiichi => "Double Riichi",
            Yaku::Chiitoitsu => "Seven Pairs",
            Yaku::Sanshoku => "Mixed Triple Sequence",
            Yaku::Ittsuu => "Pure Straight",
            Yaku::Chanta => "Half Outside Hand",
            Yaku::SanshokuDoukou => "Triple Triplet",
            Yaku::SanAnkou => "Three Concealed Triplets",
            Yaku::SanKantsu => "Three Quads",
            Yaku::Toitoi => "All Triplets",
            Yaku::Shousangen => "Little Three Dragons",
            Yaku::Honroutou => "All Terminals and Honours",
            Yaku::Ryanpeikou => "Twice Pure Double Sequence",
            Yaku::Honitsu => "Half Flush",
            Yaku::Junchan => "Full Outside Hand",
            Yaku::Renhou => "Blessing of Man",
            Yaku::Chinitsu => "Full Flush",
            Yaku::KokushiMusou => "Thirteen Orphans",
            Yaku::ChuurenPoutou => "Nine Gates",
            Yaku::Tenhou => "Blessing of Heaven",
            Yaku::Chiihou => "Blessing of Earth",
            Yaku::SuuAnkou => "Four Concealed Triplets",
            Yaku::SuuKantsu => "Four Quads",
            Yaku::Ryuuiisou => "All Green",
            Yaku::Chinroutou => "All Terminals",
            Yaku::Tsuuiisou => "All Honours",
            Yaku::Daisangen => "Big Three Dragons",
            Yaku::ShouSuushii => "Little Four Winds",
            Yaku::DaiSuushii => "Big Four Winds",
        }
    }
}

/// One set of a hand as scoring sees it, whether it came from the hand or
/// from a call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Group {
    /// Lowest tile of a sequence, or the tile of a triplet or quad.
    pub tile: Tile,
    /// Whether the group is a sequence.
    pub is_sequence: bool,
    /// Whether the group is a quad.
    pub is_quad: bool,
    /// Whether the group counts as concealed. A triplet completed by a
    /// claimed winning tile does not (EMA section 4.1.1).
    pub concealed: bool,
}

impl Group {
    /// The tiles the group is made of.
    pub fn tiles(self) -> Vec<Tile> {
        if self.is_sequence {
            let second = self.tile.next_in_suit().expect("sequence starts below 8");
            let third = second.next_in_suit().expect("sequence starts below 8");
            vec![self.tile, second, third]
        } else {
            vec![self.tile; if self.is_quad { 4 } else { 3 }]
        }
    }

    /// Whether every tile of the group is a terminal or an honour.
    pub fn is_all_terminal_or_honour(self) -> bool {
        self.tiles().iter().all(|tile| tile.is_terminal_or_honour())
    }

    /// Whether the group holds at least one terminal or honour.
    pub fn has_terminal_or_honour(self) -> bool {
        self.tiles().iter().any(|tile| tile.is_terminal_or_honour())
    }

    /// Whether the group is a triplet or quad, which is what the triplet
    /// yaku count.
    pub const fn is_triplet_like(self) -> bool {
        !self.is_sequence
    }
}

/// Everything the yaku need to know about one reading of one hand.
pub struct Analysis<'a> {
    /// The four sets, or thirteen singles for Thirteen Orphans.
    pub groups: &'a [Group],
    /// The pair.
    pub pair: Tile,
    /// Every tile in the hand, melds included.
    pub all_tiles: &'a [Tile],
    /// Whether the hand is concealed, i.e. no call other than a concealed quad.
    pub concealed: bool,
    /// The player's seat wind.
    pub seat: Wind,
    /// The round wind.
    pub round: Wind,
    /// Kinds of call made, used for the quad yaku.
    pub meld_kinds: &'a [MeldKind],
}

/// The structural yaku of a reading: everything that depends on the tiles
/// rather than on how the hand was won.
pub fn structural(analysis: &Analysis) -> Vec<Yaku> {
    let mut found = Vec::new();
    let groups = analysis.groups;
    let open = !analysis.concealed;

    let sequences: Vec<Group> = groups.iter().copied().filter(|g| g.is_sequence).collect();
    let triplets: Vec<Group> = groups
        .iter()
        .copied()
        .filter(|g| g.is_triplet_like())
        .collect();

    // Value triplets, one yaku each (EMA section 4.2.1).
    for group in &triplets {
        if group.tile.is_dragon() {
            found.push(Yaku::YakuhaiDragon(group.tile));
        }
        if group.tile == analysis.seat.tile() {
            found.push(Yaku::YakuhaiSeatWind);
        }
        if group.tile == analysis.round.tile() {
            found.push(Yaku::YakuhaiRoundWind);
        }
    }

    // All Simples.
    if analysis.all_tiles.iter().all(|tile| tile.is_simple()) {
        found.push(Yaku::Tanyao);
    }

    // Suit composition.
    let suits: Vec<Suit> = analysis
        .all_tiles
        .iter()
        .map(|tile| tile.suit())
        .filter(|suit| suit.is_numbered())
        .collect();
    let honours = analysis.all_tiles.iter().any(|tile| tile.is_honour());
    let one_suit = suits.windows(2).all(|pair| pair[0] == pair[1]);
    if one_suit && !suits.is_empty() {
        if honours {
            found.push(Yaku::Honitsu);
        } else {
            found.push(Yaku::Chinitsu);
        }
    }

    // Terminals and honours everywhere.
    let all_th = analysis
        .all_tiles
        .iter()
        .all(|tile| tile.is_terminal_or_honour());
    if all_th {
        found.push(Yaku::Honroutou);
    }

    // Outside hands: every set and the pair reach a terminal or honour, and
    // at least one set is a sequence. Junchan bars honours; chanta requires
    // them (EMA 2025 sections 4.2.2 and 4.2.3).
    let pair_th = analysis.pair.is_terminal_or_honour();
    let every_group_th = groups.iter().all(|g| g.has_terminal_or_honour());
    if pair_th && every_group_th && !sequences.is_empty() && !all_th {
        if honours {
            found.push(Yaku::Chanta);
        } else {
            found.push(Yaku::Junchan);
        }
    }

    // All Triplets, and the concealed-triplet counts.
    if triplets.len() == 4 {
        found.push(Yaku::Toitoi);
    }
    let concealed_triplets = triplets.iter().filter(|g| g.concealed).count();
    match concealed_triplets {
        3 => found.push(Yaku::SanAnkou),
        4 => found.push(Yaku::SuuAnkou),
        _ => {}
    }

    // Quads.
    let quads = groups.iter().filter(|g| g.is_quad).count();
    match quads {
        3 => found.push(Yaku::SanKantsu),
        4 => found.push(Yaku::SuuKantsu),
        _ => {}
    }

    // Identical sequences, concealed only.
    if analysis.concealed && sequences.len() >= 2 {
        let mut pairs = 0;
        let mut seen: Vec<Tile> = Vec::new();
        for group in &sequences {
            if seen.contains(&group.tile) {
                pairs += 1;
                seen.retain(|tile| *tile != group.tile);
            } else {
                seen.push(group.tile);
            }
        }
        if pairs >= 2 {
            found.push(Yaku::Ryanpeikou);
        } else if pairs == 1 {
            found.push(Yaku::Iipeiko);
        }
    }

    // Mixed Triple Sequence and Triple Triplet.
    if has_all_three_suits(&sequences) {
        found.push(Yaku::Sanshoku);
    }
    if has_all_three_suits(&triplets) {
        found.push(Yaku::SanshokuDoukou);
    }

    // Pure Straight.
    if has_pure_straight(&sequences) {
        found.push(Yaku::Ittsuu);
    }

    // Dragons.
    let dragon_triplets = triplets.iter().filter(|g| g.tile.is_dragon()).count();
    if dragon_triplets == 3 {
        found.push(Yaku::Daisangen);
    } else if dragon_triplets == 2 && analysis.pair.is_dragon() {
        found.push(Yaku::Shousangen);
    }

    // Winds.
    let wind_triplets = triplets.iter().filter(|g| g.tile.is_wind()).count();
    if wind_triplets == 4 {
        found.push(Yaku::DaiSuushii);
    } else if wind_triplets == 3 && analysis.pair.is_wind() {
        found.push(Yaku::ShouSuushii);
    }

    // Whole-hand compositions.
    if analysis.all_tiles.iter().all(|tile| tile.is_honour()) {
        found.push(Yaku::Tsuuiisou);
    }
    if analysis.all_tiles.iter().all(|tile| tile.is_terminal()) {
        found.push(Yaku::Chinroutou);
    }
    if analysis.all_tiles.iter().all(|tile| tile.is_green()) {
        found.push(Yaku::Ryuuiisou);
    }

    let _ = open;
    found.sort_unstable();
    found.dedup();
    found
}

fn has_all_three_suits(groups: &[Group]) -> bool {
    for group in groups {
        if group.tile.is_honour() {
            continue;
        }
        let rank = group.tile.rank();
        let suits_found: Vec<Suit> = groups
            .iter()
            .filter(|other| !other.tile.is_honour() && other.tile.rank() == rank)
            .map(|other| other.tile.suit())
            .collect();
        let has_characters = suits_found.contains(&Suit::Characters);
        let has_circles = suits_found.contains(&Suit::Circles);
        let has_bamboo = suits_found.contains(&Suit::Bamboo);
        if has_characters && has_circles && has_bamboo {
            return true;
        }
    }
    false
}

fn has_pure_straight(sequences: &[Group]) -> bool {
    for suit in [Suit::Characters, Suit::Circles, Suit::Bamboo] {
        let lows: Vec<u8> = sequences
            .iter()
            .filter(|group| group.tile.suit() == suit)
            .map(|group| group.tile.rank())
            .collect();
        if lows.contains(&1) && lows.contains(&4) && lows.contains(&7) {
            return true;
        }
    }
    false
}

/// Nine Gates: a concealed hand of 1112345678999 in one suit plus a
/// duplicate of any tile of that suit (EMA section 4.2.6).
pub fn is_nine_gates(all_tiles: &[Tile], concealed: bool, any_quad: bool) -> bool {
    if !concealed || any_quad || all_tiles.len() != 14 {
        return false;
    }
    let suit = all_tiles[0].suit();
    if !suit.is_numbered() || all_tiles.iter().any(|tile| tile.suit() != suit) {
        return false;
    }
    let mut counts = [0u8; 10];
    for tile in all_tiles {
        counts[tile.rank() as usize] += 1;
    }
    if counts[1] < 3 || counts[9] < 3 {
        return false;
    }
    (2..=8).all(|rank| counts[rank] >= 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(text: &str) -> Tile {
        text.parse().unwrap()
    }

    fn sequence(text: &str) -> Group {
        Group {
            tile: tile(text),
            is_sequence: true,
            is_quad: false,
            concealed: true,
        }
    }

    fn triplet(text: &str) -> Group {
        Group {
            tile: tile(text),
            is_sequence: false,
            is_quad: false,
            concealed: true,
        }
    }

    /// EMA 2025 section 4.2: open hands lose a han on the underlined yaku.
    #[test]
    fn open_hands_lose_a_han_where_marked() {
        // EMA 2025 sections 4.2.2 to 4.2.5, where each yaku that is worth
        // less open says so: "Worth only two han if the hand is open", and
        // Full Flush "Worth only five han if the hand is open".
        assert_eq!(Yaku::Chinitsu.han(false), 6);
        assert_eq!(Yaku::Chinitsu.han(true), 5);
        assert_eq!(Yaku::Honitsu.han(false), 3);
        assert_eq!(Yaku::Honitsu.han(true), 2);
        assert_eq!(Yaku::Sanshoku.han(false), 2);
        assert_eq!(Yaku::Sanshoku.han(true), 1);
        assert_eq!(Yaku::Junchan.han(false), 3);
        assert_eq!(Yaku::Junchan.han(true), 2);
        // Not underlined: the same either way.
        assert_eq!(Yaku::Toitoi.han(false), 2);
        assert_eq!(Yaku::Toitoi.han(true), 2);
        assert_eq!(Yaku::Tanyao.han(true), 1);
    }

    #[test]
    fn pure_straight_and_mixed_triple_sequence() {
        let straight = [
            sequence("1s"),
            sequence("4s"),
            sequence("7s"),
            sequence("2m"),
        ];
        assert!(has_pure_straight(&straight));
        let mixed = [
            sequence("1s"),
            sequence("1m"),
            sequence("1p"),
            sequence("5s"),
        ];
        assert!(!has_pure_straight(&mixed));
        assert!(has_all_three_suits(&mixed));
        assert!(!has_all_three_suits(&straight));
    }

    /// EMA 2025 section 4.2.3: Twice Pure Double Sequence is four sequences
    /// forming two Pure Double Sequences, and "No additional han for Pure
    /// Double Sequence IIPEIKO are counted." Counting both would turn a
    /// three han hand into four.
    #[test]
    fn twice_pure_double_sequence_swallows_the_single_one() {
        let tiles: Vec<Tile> = ["2m", "3m", "4m", "6p", "7p", "8p", "9s"]
            .iter()
            .map(|text| tile(text))
            .collect();

        // Two sequences twice over: 234m 234m 678p 678p.
        let twice = [
            sequence("2m"),
            sequence("2m"),
            sequence("6p"),
            sequence("6p"),
        ];
        let analysis = Analysis {
            groups: &twice,
            pair: tile("9s"),
            all_tiles: &tiles,
            concealed: true,
            seat: Wind::East,
            round: Wind::East,
            meld_kinds: &[],
        };
        let found = structural(&analysis);
        assert!(
            found.contains(&Yaku::Ryanpeikou),
            "four sequences in two pairs: {found:?}"
        );
        assert!(
            !found.contains(&Yaku::Iipeiko),
            "and the single one is not counted on top: {found:?}"
        );

        // One pair of matching sequences is the single yaku, not the double.
        let once = [
            sequence("2m"),
            sequence("2m"),
            sequence("6p"),
            sequence("3s"),
        ];
        let analysis = Analysis {
            groups: &once,
            ..analysis
        };
        let found = structural(&analysis);
        assert!(found.contains(&Yaku::Iipeiko), "{found:?}");
        assert!(!found.contains(&Yaku::Ryanpeikou), "{found:?}");
        assert_eq!(Yaku::Ryanpeikou.han(false), 3);
        assert_eq!(Yaku::Iipeiko.han(false), 1);
    }

    #[test]
    fn triple_triplet_needs_one_of_each_suit() {
        let groups = [triplet("4m"), triplet("4p"), triplet("4s"), triplet("1z")];
        assert!(has_all_three_suits(&groups));
        let groups = [triplet("4m"), triplet("4p"), triplet("5s"), triplet("1z")];
        assert!(!has_all_three_suits(&groups));
    }

    #[test]
    fn nine_gates_shape() {
        let tiles: Vec<Tile> = "11123456789995p"
            .parse::<crate::TileSet>()
            .unwrap()
            .tiles()
            .collect();
        assert!(is_nine_gates(&tiles, true, false));
        // A call rules it out.
        assert!(!is_nine_gates(&tiles, false, false));
        // So does a quad (EMA section 4.2.6).
        assert!(!is_nine_gates(&tiles, true, true));
    }
}
