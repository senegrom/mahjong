//! Scoring a winning hand: han, minipoints and payments.
//!
//! Follows EMA 2025 sections 4.1 to 4.2. Where a hand can be read in more
//! than one way, every reading and every way the winning tile can complete
//! it is scored and the best is taken, as section 3.4.3 requires.

use crate::agari::{self, Block, Reading, Shape};
use crate::hand::{Meld, MeldKind, TileSet};
use crate::tile::Tile;
use crate::yaku::{self, Analysis, Group, Yaku};
use crate::Wind;

/// How the winning tile arrived.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WinBy {
    /// Drawn by the winner (tsumo).
    SelfDraw,
    /// Claimed from a discard, or from a quad being robbed (ron).
    Discard,
}

/// Whether riichi was declared, and how.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Riichi {
    /// No declaration.
    #[default]
    None,
    /// An ordinary declaration.
    Declared,
    /// Declared in the first uninterrupted set of turns (EMA section 4.2.2).
    Double,
}

/// Everything about the win that the tiles alone do not say.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Situation {
    /// The winner's seat wind.
    pub seat: Wind,
    /// The round wind.
    pub round: Wind,
    /// How the hand was won.
    pub win_by: WinBy,
    /// The tile that completed the hand.
    pub winning_tile: Tile,
    /// Riichi state.
    pub riichi: Riichi,
    /// Won within the first uninterrupted set of turns after riichi.
    pub ippatsu: bool,
    /// Won on a replacement tile after a quad.
    pub after_quad: bool,
    /// Won on a tile extending a melded triplet to a quad.
    pub robbing_quad: bool,
    /// Self-drawn the last tile of the wall.
    pub under_the_sea: bool,
    /// Won on the last discard.
    pub under_the_river: bool,
    /// The dealer won on the starting hand.
    pub blessing_of_heaven: bool,
    /// Self-drawn in the very first uninterrupted set of turns.
    pub blessing_of_earth: bool,
    /// Won by discard before the player's first turn.
    pub blessing_of_man: bool,
    /// Dora indicators, the first one and any revealed for quads.
    pub dora_indicators: Vec<Tile>,
    /// Ura dora indicators, which count only for a riichi hand.
    pub ura_indicators: Vec<Tile>,
    /// Counters on the table (EMA section 3.4.4).
    pub counters: u32,
    /// Riichi bets on the table that the winner collects.
    pub riichi_sticks: u32,
}

impl Situation {
    /// A plain win with nothing special about it.
    pub fn new(seat: Wind, round: Wind, win_by: WinBy, winning_tile: Tile) -> Situation {
        Situation {
            seat,
            round,
            win_by,
            winning_tile,
            riichi: Riichi::None,
            ippatsu: false,
            after_quad: false,
            robbing_quad: false,
            under_the_sea: false,
            under_the_river: false,
            blessing_of_heaven: false,
            blessing_of_earth: false,
            blessing_of_man: false,
            dora_indicators: Vec::new(),
            ura_indicators: Vec::new(),
            counters: 0,
            riichi_sticks: 0,
        }
    }

    /// Whether the winner is the dealer, who receives and pays more.
    pub fn is_dealer(&self) -> bool {
        matches!(self.seat, Wind::East)
    }
}

/// A named limit hand (EMA section 4.1.2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Limit {
    /// 5 han, or a smaller hand whose base value reaches the cap.
    Mangan,
    /// 6 to 7 han.
    Haneman,
    /// 8 to 10 han.
    Baiman,
    /// 11 han or more. EMA has no counted yakuman.
    Sanbaiman,
    /// A yakuman hand.
    Yakuman,
}

impl Limit {
    /// The base value the limit pays from.
    pub const fn base(self) -> u32 {
        match self {
            Limit::Mangan => 2000,
            Limit::Haneman => 3000,
            Limit::Baiman => 4000,
            Limit::Sanbaiman => 6000,
            Limit::Yakuman => 8000,
        }
    }

    /// The rulebook's name.
    pub const fn name(self) -> &'static str {
        match self {
            Limit::Mangan => "mangan",
            Limit::Haneman => "haneman",
            Limit::Baiman => "baiman",
            Limit::Sanbaiman => "sanbaiman",
            Limit::Yakuman => "yakuman",
        }
    }
}

/// Why a minipoint was awarded, so the interface can show the working.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FuReason {
    /// The 20 every hand starts from, or 25 for Seven Pairs.
    Base,
    /// Winning by discard with a concealed hand.
    ConcealedRon,
    /// Winning by self-draw.
    SelfDraw,
    /// A triplet or quad.
    Set(Tile),
    /// A pair of dragons, or of the seat or round wind.
    ValuePair,
    /// An edge, closed or pair wait.
    Wait,
    /// An open hand worth exactly 20 minipoints.
    OpenPinfu,
}

/// Who pays what for a scored hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Payments {
    /// On a win by discard, what the discarder pays.
    pub from_discarder: u32,
    /// On a self-draw, what the dealer pays. Zero when the winner is dealer.
    pub from_dealer: u32,
    /// On a self-draw, what each non-dealer pays.
    pub from_each_other: u32,
    /// Everything the winner receives, counters and riichi bets included.
    pub total: u32,
}

/// A fully scored hand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Score {
    /// The yaku found, with the han each is worth in this hand.
    pub yaku: Vec<(Yaku, u8)>,
    /// Han from yaku plus dora.
    pub han: u8,
    /// Han that came from dora, kan dora and ura dora.
    pub dora: u8,
    /// Minipoints, rounded up as the rules require.
    pub fu: u32,
    /// The working behind the minipoints.
    pub fu_detail: Vec<(FuReason, u32)>,
    /// The limit reached, if any.
    pub limit: Option<Limit>,
    /// The base value payments are computed from.
    pub base: u32,
    /// Who pays what.
    pub payments: Payments,
}

/// Why a hand cannot be scored.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScoreError {
    /// The tiles do not form a complete hand.
    NotComplete,
    /// The hand is complete but has no yaku, so it cannot be declared a win
    /// (EMA section 3.2).
    NoYaku,
    /// The winning tile is not part of the hand.
    WinningTileMissing,
}

/// Scores a winning hand, taking the highest-scoring reading.
///
/// `concealed` holds the winner's concealed tiles *including* the winning
/// tile; `melds` holds the called sets.
pub fn score(
    concealed: &TileSet,
    melds: &[Meld],
    situation: &Situation,
) -> Result<Score, ScoreError> {
    if concealed.count(situation.winning_tile) == 0 {
        return Err(ScoreError::WinningTileMissing);
    }
    let readings = agari::readings(concealed, melds.len());
    if readings.is_empty() {
        return Err(ScoreError::NotComplete);
    }

    let mut best: Option<Score> = None;
    let mut saw_hand_without_yaku = false;
    for reading in &readings {
        for candidate in score_reading(reading, concealed, melds, situation) {
            match candidate {
                Ok(score) => {
                    let better = match &best {
                        None => true,
                        Some(current) => {
                            (score.payments.total, score.han, score.fu)
                                > (current.payments.total, current.han, current.fu)
                        }
                    };
                    if better {
                        best = Some(score);
                    }
                }
                Err(ScoreError::NoYaku) => saw_hand_without_yaku = true,
                Err(_) => {}
            }
        }
    }
    match best {
        Some(score) => Ok(score),
        None if saw_hand_without_yaku => Err(ScoreError::NoYaku),
        None => Err(ScoreError::NotComplete),
    }
}

/// Every scoring of one reading, one per way the winning tile completes it.
fn score_reading(
    reading: &Reading,
    concealed: &TileSet,
    melds: &[Meld],
    situation: &Situation,
) -> Vec<Result<Score, ScoreError>> {
    let hand_is_concealed = melds.iter().all(|meld| !meld.kind.opens_hand());
    let mut results = Vec::new();
    let completed: Vec<usize> = reading
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.contains(situation.winning_tile))
        .map(|(index, _)| index)
        .collect();
    if completed.is_empty() {
        return results;
    }
    for index in completed {
        results.push(score_one(
            reading,
            index,
            concealed,
            melds,
            hand_is_concealed,
            situation,
            false,
        ));
        // Blessing of Man is worth five han and combines with nothing, so a
        // hand that qualifies is scored both ways and the better stands
        // (EMA sections 4.2.4 and 4.1).
        if situation.blessing_of_man && hand_is_concealed {
            results.push(score_one(
                reading,
                index,
                concealed,
                melds,
                hand_is_concealed,
                situation,
                true,
            ));
        }
    }
    results
}

#[allow(clippy::too_many_arguments)]
fn score_one(
    reading: &Reading,
    completed: usize,
    concealed: &TileSet,
    melds: &[Meld],
    hand_is_concealed: bool,
    situation: &Situation,
    as_blessing_of_man: bool,
) -> Result<Score, ScoreError> {
    let won_by_discard = matches!(situation.win_by, WinBy::Discard);

    // Every tile of the hand, called sets included, for the whole-hand yaku.
    let mut all_tiles: Vec<Tile> = concealed.tiles().collect();
    for meld in melds {
        all_tiles.extend(meld.tiles());
    }

    // Groups as scoring sees them: the reading's sets, then the called sets.
    let mut groups: Vec<Group> = Vec::new();
    let mut pair = situation.winning_tile;
    let mut wait_fu = 0;
    let mut pair_is_the_wait = false;

    for (index, block) in reading.blocks.iter().enumerate() {
        let is_completed = index == completed;
        match block {
            Block::Pair(tile) => {
                pair = *tile;
                if is_completed {
                    pair_is_the_wait = true;
                    wait_fu = 2;
                }
            }
            Block::Sequence(low) => {
                if is_completed {
                    wait_fu = sequence_wait_fu(*low, situation.winning_tile);
                }
                groups.push(Group {
                    tile: *low,
                    is_sequence: true,
                    is_quad: false,
                    concealed: true,
                });
            }
            Block::Triplet(tile) => {
                // A triplet finished by a claimed tile counts as melded for
                // minipoints and for the concealed-triplet yaku, though the
                // hand itself stays concealed (EMA sections 4.1.1 and 4.2).
                let concealed_set = !(is_completed && won_by_discard);
                groups.push(Group {
                    tile: *tile,
                    is_sequence: false,
                    is_quad: false,
                    concealed: concealed_set,
                });
            }
        }
    }
    for meld in melds {
        groups.push(Group {
            tile: meld.tile,
            is_sequence: meld.is_sequence(),
            is_quad: meld.kind.is_kan(),
            concealed: meld.kind.is_concealed_for_fu(),
        });
    }

    let meld_kinds: Vec<MeldKind> = melds.iter().map(|meld| meld.kind).collect();
    let analysis = Analysis {
        groups: &groups,
        pair,
        all_tiles: &all_tiles,
        concealed: hand_is_concealed,
        seat: situation.seat,
        round: situation.round,
        meld_kinds: &meld_kinds,
    };

    let mut found: Vec<Yaku> = match reading.shape {
        Shape::ThirteenOrphans => vec![Yaku::KokushiMusou],
        Shape::SevenPairs => {
            let mut list = yaku::structural(&analysis);
            // The pair-based shape has no sets, so drop anything that only
            // makes sense for four sets and a pair.
            list.retain(|entry| {
                matches!(
                    entry,
                    Yaku::Tanyao
                        | Yaku::Honitsu
                        | Yaku::Chinitsu
                        | Yaku::Honroutou
                        | Yaku::Tsuuiisou
                )
            });
            list.push(Yaku::Chiitoitsu);
            list
        }
        Shape::Standard => yaku::structural(&analysis),
    };

    // Nine Gates, which the structural pass cannot see because it needs the
    // whole hand rather than its groups.
    let any_quad = groups.iter().any(|group| group.is_quad);
    if yaku::is_nine_gates(&all_tiles, hand_is_concealed, any_quad) {
        found.push(Yaku::ChuurenPoutou);
    }

    // Circumstance yaku.
    if hand_is_concealed {
        match situation.riichi {
            Riichi::Declared => found.push(Yaku::Riichi),
            Riichi::Double => found.push(Yaku::DoubleRiichi),
            Riichi::None => {}
        }
        if situation.ippatsu && !matches!(situation.riichi, Riichi::None) {
            found.push(Yaku::Ippatsu);
        }
        if !won_by_discard {
            found.push(Yaku::MenzenTsumo);
        }
    }
    if situation.after_quad {
        found.push(Yaku::Rinshan);
    }
    if situation.robbing_quad {
        found.push(Yaku::Chankan);
    }
    // Under the Sea does not combine with After a Quad (EMA section 4.2.1).
    if situation.under_the_sea && !situation.after_quad {
        found.push(Yaku::Haitei);
    }
    if situation.under_the_river {
        found.push(Yaku::Houtei);
    }
    if situation.blessing_of_heaven && hand_is_concealed {
        found.push(Yaku::Tenhou);
    }
    if situation.blessing_of_earth && hand_is_concealed {
        found.push(Yaku::Chiihou);
    }

    // Pinfu: four sequences, a valueless pair and a two-sided wait.
    let all_sequences = groups.len() == 4 && groups.iter().all(|group| group.is_sequence);
    let valueless_pair =
        !pair.is_dragon() && pair != situation.seat.tile() && pair != situation.round.tile();
    let pinfu = hand_is_concealed
        && matches!(reading.shape, Shape::Standard)
        && all_sequences
        && valueless_pair
        && !pair_is_the_wait
        && wait_fu == 0;
    if pinfu {
        found.push(Yaku::Pinfu);
    }

    found.sort_unstable();
    found.dedup();

    // Blessing of Man stands alone: no other yaku and no dora.
    if as_blessing_of_man {
        if !(situation.blessing_of_man && hand_is_concealed && won_by_discard) {
            return Err(ScoreError::NoYaku);
        }
        found = vec![Yaku::Renhou];
    }

    if found.is_empty() {
        return Err(ScoreError::NoYaku);
    }

    let open = !hand_is_concealed;
    let yakuman: Vec<Yaku> = found
        .iter()
        .copied()
        .filter(|entry| entry.is_yakuman())
        .collect();

    let (yaku_list, han, dora, limit) = if !yakuman.is_empty() {
        // Yakuman are not cumulative (EMA section 4.2): one yakuman is paid.
        let chosen = yakuman[0];
        (vec![(chosen, 13)], 13u8, 0u8, Some(Limit::Yakuman))
    } else {
        let mut list: Vec<(Yaku, u8)> = found
            .iter()
            .map(|entry| (*entry, entry.han(open)))
            .collect();
        let mut han: u8 = list.iter().map(|(_, han)| *han).sum();
        let mut dora = 0u8;
        if !found.contains(&Yaku::Renhou) {
            dora = count_dora(&all_tiles, situation);
            han = han.saturating_add(dora);
        }
        list.sort_by_key(|(entry, _)| *entry);
        (list, han, dora, None)
    };

    let (fu, fu_detail) = if limit.is_some() {
        (0, Vec::new())
    } else {
        minipoints(
            reading.shape,
            &groups,
            pair,
            wait_fu,
            hand_is_concealed,
            won_by_discard,
            pinfu,
            situation,
        )
    };

    let (base, limit) = base_value(han, fu, limit);
    let payments = payments(base, situation);

    Ok(Score {
        yaku: yaku_list,
        han,
        dora,
        fu,
        fu_detail,
        limit,
        base,
        payments,
    })
}

/// The minipoints for a wait that finishes a sequence.
fn sequence_wait_fu(low: Tile, winning: Tile) -> u32 {
    let position = winning.rank() as i32 - low.rank() as i32;
    match position {
        // The middle tile: a closed wait.
        1 => 2,
        // A 3 finishing 1-2-3 or a 7 finishing 7-8-9: an edge wait.
        2 if low.rank() == 1 => 2,
        0 if low.rank() == 7 => 2,
        _ => 0,
    }
}

fn count_dora(all_tiles: &[Tile], situation: &Situation) -> u8 {
    let mut count = 0u8;
    for indicator in &situation.dora_indicators {
        let dora = indicator.dora();
        count += all_tiles.iter().filter(|tile| **tile == dora).count() as u8;
    }
    if !matches!(situation.riichi, Riichi::None) {
        for indicator in &situation.ura_indicators {
            let dora = indicator.dora();
            count += all_tiles.iter().filter(|tile| **tile == dora).count() as u8;
        }
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn minipoints(
    shape: Shape,
    groups: &[Group],
    pair: Tile,
    wait_fu: u32,
    hand_is_concealed: bool,
    won_by_discard: bool,
    pinfu: bool,
    situation: &Situation,
) -> (u32, Vec<(FuReason, u32)>) {
    let mut detail = Vec::new();
    if matches!(shape, Shape::SevenPairs) {
        detail.push((FuReason::Base, 25));
        return (25, detail);
    }

    let mut total = 20;
    detail.push((FuReason::Base, 20));
    if won_by_discard && hand_is_concealed {
        total += 10;
        detail.push((FuReason::ConcealedRon, 10));
    }
    if !won_by_discard && !pinfu {
        total += 2;
        detail.push((FuReason::SelfDraw, 2));
    }

    for group in groups {
        if group.is_sequence {
            continue;
        }
        let terminal = group.tile.is_terminal_or_honour();
        let value = match (group.is_quad, group.concealed, terminal) {
            (false, false, false) => 2,
            (false, false, true) => 4,
            (false, true, false) => 4,
            (false, true, true) => 8,
            (true, false, false) => 8,
            (true, false, true) => 16,
            (true, true, false) => 16,
            (true, true, true) => 32,
        };
        total += value;
        detail.push((FuReason::Set(group.tile), value));
    }

    // A pair of dragons, or of the seat or round wind. A pair that is both
    // seat and round wind is worth 2, not 4 (EMA 2025 section 4.1.1).
    if pair.is_dragon() || pair == situation.seat.tile() || pair == situation.round.tile() {
        total += 2;
        detail.push((FuReason::ValuePair, 2));
    }

    if wait_fu > 0 {
        total += wait_fu;
        detail.push((FuReason::Wait, wait_fu));
    }

    // An open hand worth exactly 20 minipoints gets 2 for open pinfu.
    if !hand_is_concealed && total == 20 {
        total += 2;
        detail.push((FuReason::OpenPinfu, 2));
    }

    let rounded = total.div_ceil(10) * 10;
    (rounded, detail)
}

/// The base value payments are computed from, and the limit if one applies.
fn base_value(han: u8, fu: u32, yakuman: Option<Limit>) -> (u32, Option<Limit>) {
    if let Some(limit) = yakuman {
        return (limit.base(), Some(limit));
    }
    let limit = match han {
        0..=4 => None,
        5 => Some(Limit::Mangan),
        6..=7 => Some(Limit::Haneman),
        8..=10 => Some(Limit::Baiman),
        _ => Some(Limit::Sanbaiman),
    };
    if let Some(limit) = limit {
        return (limit.base(), Some(limit));
    }
    let raw = fu * 2u32.pow(han as u32 + 2);
    // The base value is capped at a mangan, and anything above 1,900 is paid
    // as one, which is what makes 4 han 30 fu and 3 han 60 fu mangan
    // (EMA 2025 section 4.1.2).
    if raw > 1900 {
        (2000, Some(Limit::Mangan))
    } else {
        (raw, None)
    }
}

fn round_up_100(value: u32) -> u32 {
    value.div_ceil(100) * 100
}

fn payments(base: u32, situation: &Situation) -> Payments {
    let dealer = situation.is_dealer();
    let counters = situation.counters;
    let sticks = situation.riichi_sticks * 1000;
    match situation.win_by {
        WinBy::Discard => {
            let multiplier = if dealer { 6 } else { 4 };
            let from_discarder = round_up_100(base * multiplier) + counters * 300;
            Payments {
                from_discarder,
                from_dealer: 0,
                from_each_other: 0,
                total: from_discarder + sticks,
            }
        }
        WinBy::SelfDraw => {
            if dealer {
                let each = round_up_100(base * 2) + counters * 100;
                Payments {
                    from_discarder: 0,
                    from_dealer: 0,
                    from_each_other: each,
                    total: each * 3 + sticks,
                }
            } else {
                let from_dealer = round_up_100(base * 2) + counters * 100;
                let from_each_other = round_up_100(base) + counters * 100;
                Payments {
                    from_discarder: 0,
                    from_dealer,
                    from_each_other,
                    total: from_dealer + from_each_other * 2 + sticks,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{score as score_hand, *};
    use crate::hand::ClaimedFrom;

    fn hand(text: &str) -> TileSet {
        text.parse().expect("test hand parses")
    }

    fn tile(text: &str) -> Tile {
        text.parse().expect("test tile parses")
    }

    fn names(score: &Score) -> Vec<&'static str> {
        score.yaku.iter().map(|(entry, _)| entry.name()).collect()
    }

    /// EMA 2025, scoring example 1: riichi, fully concealed, pinfu and pure
    /// straight, self-drawn, 5 han, a mangan. A non-dealer receives 4,000
    /// from the dealer and 2,000 from each other player.
    #[test]
    fn example_1_riichi_pinfu_pure_straight_tsumo() {
        let tiles = hand("22p234p123s456s789s");
        let mut situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("9s"));
        situation.riichi = Riichi::Declared;
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert_eq!(score.han, 5);
        assert_eq!(score.limit, Some(Limit::Mangan));
        assert!(names(&score).contains(&"Pinfu"));
        assert!(names(&score).contains(&"Pure Straight"));
        assert!(names(&score).contains(&"Fully Concealed Hand"));
        assert_eq!(score.payments.from_dealer, 4000);
        assert_eq!(score.payments.from_each_other, 2000);
        assert_eq!(score.payments.total, 8000);
    }

    /// EMA 2025, scoring example 2: the same hand won by discard is riichi,
    /// pinfu and pure straight, 4 han 30 fu, which the 2025 edition rounds up
    /// to a mangan: 12,000 from the discarder for the dealer, 8,000 otherwise.
    #[test]
    fn example_2_four_han_thirty_fu_is_a_mangan() {
        let tiles = hand("22p234p123s456s789s");
        let mut situation = Situation::new(Wind::South, Wind::East, WinBy::Discard, tile("9s"));
        situation.riichi = Riichi::Declared;
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert_eq!(score.han, 4);
        assert_eq!(score.fu, 30);
        assert_eq!(score.limit, Some(Limit::Mangan));
        assert_eq!(score.payments.from_discarder, 8000);

        let mut dealer = situation.clone();
        dealer.seat = Wind::East;
        let score = score_hand(&tiles, &[], &dealer).unwrap();
        assert_eq!(score.payments.from_discarder, 12000);
    }

    /// EMA 2025, scoring example 3: an open pure straight with one dora,
    /// 2 han 30 fu including the 2 minipoints for open pinfu. The discarder
    /// pays 2,900 to the dealer and 2,000 to anyone else.
    #[test]
    fn example_3_open_pinfu_and_a_dora() {
        let tiles = hand("22p234p123s789s");
        let melds = [Meld::chii(tile("4s"), ClaimedFrom::Left)];
        let mut situation = Situation::new(Wind::South, Wind::East, WinBy::Discard, tile("9s"));
        // 6 bamboo indicates 7 bamboo, which the hand holds once.
        situation.dora_indicators = vec![tile("6s")];
        let score = score_hand(&tiles, &melds, &situation).unwrap();
        assert_eq!(score.dora, 1);
        assert_eq!(score.han, 2);
        assert_eq!(score.fu, 30);
        assert!(score.fu_detail.contains(&(FuReason::OpenPinfu, 2)));
        assert_eq!(score.payments.from_discarder, 2000);

        let mut dealer = situation.clone();
        dealer.seat = Wind::East;
        let score = score_hand(&tiles, &melds, &dealer).unwrap();
        assert_eq!(score.payments.from_discarder, 2900);
    }

    /// EMA 2025, scoring example 4: four concealed triplets self-drawn is a
    /// yakuman, 16,000 from the dealer and 8,000 from each other player.
    #[test]
    fn example_4_four_concealed_triplets() {
        let tiles = hand("333m444m555m33p888s");
        let situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("8s"));
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert_eq!(score.limit, Some(Limit::Yakuman));
        assert_eq!(names(&score), ["Four Concealed Triplets"]);
        assert_eq!(score.payments.from_dealer, 16000);
        assert_eq!(score.payments.from_each_other, 8000);
        assert_eq!(score.payments.total, 32000);
    }

    /// EMA 2025, scoring example 5: the same tiles won by discard. The
    /// triplet the claimed tile finishes is not concealed, so the hand is
    /// three concealed triplets, all triplets and all simples, 5 han, plus
    /// three dora for a baiman.
    #[test]
    fn example_5_a_claimed_triplet_is_not_concealed() {
        let tiles = hand("333m444m555m33p888s");
        let mut situation = Situation::new(Wind::South, Wind::East, WinBy::Discard, tile("8s"));
        // 3 characters indicates 4 characters, which the hand holds three times.
        situation.dora_indicators = vec![tile("3m")];
        let score = score_hand(&tiles, &melds_none(), &situation).unwrap();
        assert!(names(&score).contains(&"Three Concealed Triplets"));
        assert!(names(&score).contains(&"All Triplets"));
        assert!(names(&score).contains(&"All Simples"));
        assert!(!names(&score).contains(&"Four Concealed Triplets"));
        assert_eq!(score.dora, 3);
        assert_eq!(score.han, 8);
        assert_eq!(score.limit, Some(Limit::Baiman));
        assert_eq!(score.payments.from_discarder, 16000);
    }

    /// EMA 2025, scoring example 7: seven pairs won by discard without
    /// riichi is 2 han 25 fu, and nothing more is added for the dragon pair
    /// or the pair wait: 1,600 to a non-dealer, 2,400 to the dealer.
    #[test]
    fn example_7_seven_pairs_scores_exactly_25_fu() {
        let tiles = hand("22336677p1155s77z");
        let situation = Situation::new(Wind::South, Wind::East, WinBy::Discard, tile("7z"));
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert_eq!(score.fu, 25);
        assert_eq!(score.han, 2);
        assert_eq!(names(&score), ["Seven Pairs"]);
        assert_eq!(score.payments.from_discarder, 1600);

        let mut dealer = situation.clone();
        dealer.seat = Wind::East;
        assert_eq!(
            score_hand(&tiles, &[], &dealer)
                .unwrap()
                .payments
                .from_discarder,
            2400
        );
    }

    /// EMA 2025, scoring example 10: a concealed half flush self-drawn with
    /// an honour triplet and an edge wait, 4 han 40 fu, which pays as a
    /// mangan: 4,000 from the dealer and 2,000 from the others.
    #[test]
    fn example_10_half_flush_edge_wait() {
        // West is neither the seat wind (South) nor the round wind (East).
        let tiles = hand("333z11p234567p789p");
        let situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("7p"));
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert!(names(&score).contains(&"Half Flush"));
        assert!(names(&score).contains(&"Fully Concealed Hand"));
        assert_eq!(score.han, 4);
        assert_eq!(score.fu, 40);
        assert_eq!(score.limit, Some(Limit::Mangan));
        assert_eq!(score.payments.total, 8000);
    }

    /// EMA 2025 section 3.2: a complete hand without a yaku is not a win.
    #[test]
    fn a_hand_without_a_yaku_cannot_win() {
        // An open hand with a terminal sequence: not all simples, no outside
        // hand, no straight, no triple sequence, so nothing to declare on.
        let melds = [Meld::chii(tile("1m"), ClaimedFrom::Left)];
        let concealed = hand("456m789p22s345s");
        let situation = Situation::new(Wind::South, Wind::East, WinBy::Discard, tile("5s"));
        assert_eq!(
            score_hand(&concealed, &melds, &situation),
            Err(ScoreError::NoYaku)
        );
    }

    /// EMA 2025 section 3.4.3: where a hand can be read in more than one way,
    /// the highest-scoring reading is taken.
    #[test]
    fn the_best_reading_wins() {
        // Read as seven pairs it is 2 han 25 fu; read as two double
        // sequences it is worth more, so the scorer must choose the latter.
        let tiles = hand("223344m556677p11z");
        let situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("4m"));
        let score = score_hand(&tiles, &[], &situation).unwrap();
        assert!(names(&score).contains(&"Twice Pure Double Sequence"));
        assert!(!names(&score).contains(&"Seven Pairs"));
    }

    /// EMA 2025 section 4.1.1: a pair that is both the seat and the round
    /// wind is worth 2 minipoints, not 4.
    #[test]
    fn a_double_wind_pair_is_worth_two() {
        let tiles = hand("11z234m567m234p567p");
        let mut situation = Situation::new(Wind::East, Wind::East, WinBy::Discard, tile("7p"));
        // The hand needs a yaku before it can be scored at all.
        situation.riichi = Riichi::Declared;
        let score = score_hand(&tiles, &[], &situation).unwrap();
        let value_pairs: u32 = score
            .fu_detail
            .iter()
            .filter(|(reason, _)| matches!(reason, FuReason::ValuePair))
            .map(|(_, value)| *value)
            .sum();
        assert_eq!(value_pairs, 2);
    }

    fn melds_none() -> Vec<Meld> {
        Vec::new()
    }
}
