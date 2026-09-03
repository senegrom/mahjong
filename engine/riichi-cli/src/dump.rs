//! Random winning hands, scored, written out for a second opinion.
//!
//! The scorer is the part of the rules with the most places to be subtly
//! wrong, and its own tests can only check what its author thought of. So
//! this writes hands and their scores in a form another implementation can
//! read, and `engine/riichi-cli/differential.py` scores the same hands with
//! the MIT-licensed `mahjong` Python library and reports every disagreement.
//!
//! Hands are built rather than dealt. Waiting for real games to produce a
//! million wins would take far too long, and the interesting hands, the
//! yakuman and the odd shapes, would hardly appear at all. Building them
//! from sets and pairs reaches those in proportion to how many there are.

use riichi_core::hand::{ClaimedFrom, Meld, MeldKind, TileSet};
use riichi_core::rng::Rng;
use riichi_core::score::{self, Riichi, Situation, WinBy};
use riichi_core::tile::{Suit, Tile, COPIES, KINDS};
use riichi_core::Wind;

/// A hand under construction, with the four copies of each kind tracked so
/// nothing is used twice.
struct Builder {
    used: [u8; KINDS],
    concealed: TileSet,
    melds: Vec<Meld>,
}

impl Builder {
    fn new() -> Builder {
        Builder {
            used: [0; KINDS],
            concealed: TileSet::new(),
            melds: Vec::new(),
        }
    }

    /// Whether `count` more copies of `tile` are available.
    fn free(&self, tile: Tile, count: u8) -> bool {
        self.used[tile.idx()] + count <= COPIES
    }

    fn take(&mut self, tile: Tile, count: u8) {
        self.used[tile.idx()] += count;
    }
}

/// Builds one complete hand at random, or gives up and returns `None` when
/// the tiles it picked cannot be finished.
fn build(rng: &mut Rng) -> Option<(TileSet, Vec<Meld>, Tile)> {
    let roll = rng.below(100);
    if roll < 6 {
        return build_seven_pairs(rng);
    }
    if roll < 9 {
        return build_thirteen_orphans(rng);
    }
    build_standard(rng)
}

/// Four sets and a pair, some of them possibly called.
fn build_standard(rng: &mut Rng) -> Option<(TileSet, Vec<Meld>, Tile)> {
    let mut builder = Builder::new();
    // How many sets are called rather than concealed. A called set changes
    // the minipoints and closes off the concealed-only yaku, so the mix
    // matters more than the exact proportion.
    let called = match rng.below(10) {
        0..=4 => 0,
        5..=6 => 1,
        7..=8 => 2,
        _ => 3,
    };

    for index in 0..4 {
        let set = pick_set(rng, &mut builder)?;
        if index < called {
            builder.melds.push(set);
        } else {
            // A concealed quad is still a meld, because the hand holds four
            // tiles that are not part of the thirteen.
            if matches!(set.kind, MeldKind::ConcealedKan) {
                builder.melds.push(set);
            } else {
                for tile in set.tiles() {
                    builder.concealed.add(tile);
                }
            }
        }
    }

    let pair = pick_pair(rng, &mut builder)?;
    builder.concealed.add(pair);
    builder.concealed.add(pair);

    let winning = pick_winning_tile(rng, &builder.concealed)?;
    Some((builder.concealed, builder.melds, winning))
}

/// A kind with two copies still free, for the pair.
fn pick_pair(rng: &mut Rng, builder: &mut Builder) -> Option<Tile> {
    for _ in 0..40 {
        let tile = Tile::new(rng.below(KINDS) as u8);
        if builder.free(tile, 2) {
            builder.take(tile, 2);
            return Some(tile);
        }
    }
    None
}

/// One set: a sequence, a triplet, or now and then a quad.
fn pick_set(rng: &mut Rng, builder: &mut Builder) -> Option<Meld> {
    for _ in 0..40 {
        match rng.below(10) {
            0..=4 => {
                // A sequence.
                let suit = [Suit::Characters, Suit::Circles, Suit::Bamboo][rng.below(3)];
                let rank = 1 + rng.below(7) as u8;
                let low = Tile::numbered(suit, rank);
                let mid = low.next_in_suit()?;
                let high = mid.next_in_suit()?;
                if [low, mid, high].iter().all(|tile| builder.free(*tile, 1)) {
                    for tile in [low, mid, high] {
                        builder.take(tile, 1);
                    }
                    return Some(Meld::chii(low, claimed_from(rng)));
                }
            }
            5..=8 => {
                let tile = Tile::new(rng.below(KINDS) as u8);
                if builder.free(tile, 3) {
                    builder.take(tile, 3);
                    return Some(Meld::pon(tile, claimed_from(rng)));
                }
            }
            _ => {
                let tile = Tile::new(rng.below(KINDS) as u8);
                if builder.free(tile, 4) {
                    builder.take(tile, 4);
                    return Some(if rng.below(2) == 0 {
                        Meld::concealed_kan(tile)
                    } else {
                        Meld {
                            kind: MeldKind::ClaimedKan,
                            tile,
                            from: claimed_from(rng),
                        }
                    });
                }
            }
        }
    }
    None
}

/// Seven distinct pairs.
fn build_seven_pairs(rng: &mut Rng) -> Option<(TileSet, Vec<Meld>, Tile)> {
    let mut builder = Builder::new();
    for _ in 0..7 {
        let mut placed = false;
        for _ in 0..40 {
            let tile = Tile::new(rng.below(KINDS) as u8);
            // Seven pairs needs seven different kinds, so two of a kind is
            // not enough: the kind must be unused.
            if builder.used[tile.idx()] == 0 {
                builder.take(tile, 2);
                builder.concealed.add(tile);
                builder.concealed.add(tile);
                placed = true;
                break;
            }
        }
        if !placed {
            return None;
        }
    }
    let winning = pick_winning_tile(rng, &builder.concealed)?;
    Some((builder.concealed, Vec::new(), winning))
}

/// One of each terminal and honour, and a second of one of them.
fn build_thirteen_orphans(rng: &mut Rng) -> Option<(TileSet, Vec<Meld>, Tile)> {
    let mut hand = TileSet::new();
    let kinds: Vec<Tile> = Tile::all()
        .filter(|tile| tile.is_terminal_or_honour())
        .collect();
    for tile in &kinds {
        hand.add(*tile);
    }
    let doubled = kinds[rng.below(kinds.len())];
    hand.add(doubled);
    let winning = pick_winning_tile(rng, &hand)?;
    Some((hand, Vec::new(), winning))
}

/// Any tile of the hand can be the one it was completed by.
fn pick_winning_tile(rng: &mut Rng, hand: &TileSet) -> Option<Tile> {
    let held: Vec<Tile> = Tile::all().filter(|tile| hand.count(*tile) > 0).collect();
    if held.is_empty() {
        return None;
    }
    Some(held[rng.below(held.len())])
}

fn claimed_from(rng: &mut Rng) -> ClaimedFrom {
    match rng.below(3) {
        0 => ClaimedFrom::Left,
        1 => ClaimedFrom::Across,
        _ => ClaimedFrom::Right,
    }
}

/// A situation to score the hand in, with the unusual circumstances turning
/// up often enough to be tested but not so often as to be the norm.
fn pick_situation(rng: &mut Rng, winning: Tile, concealed: bool, seen: &TileSet) -> Situation {
    let mut situation = Situation::new(
        Wind::ALL[rng.below(4)],
        if rng.below(4) == 0 {
            Wind::South
        } else {
            Wind::East
        },
        if rng.below(2) == 0 {
            WinBy::SelfDraw
        } else {
            WinBy::Discard
        },
        winning,
    );
    if concealed && rng.below(3) == 0 {
        situation.riichi = if rng.below(8) == 0 {
            Riichi::Double
        } else {
            Riichi::Declared
        };
        situation.ippatsu = rng.below(4) == 0;
    }
    // The last tile of the wall and a replacement tile after a quad are
    // different tiles, so a hand cannot be won on both; likewise the last
    // discard of the hand is not the tile that extends somebody's triplet.
    // Generating the impossible pair would only test the scorer against a
    // position no game can reach.
    match situation.win_by {
        WinBy::SelfDraw => {
            if rng.below(20) == 0 {
                situation.after_quad = true;
            } else if rng.below(24) == 0 {
                situation.under_the_sea = true;
            }
        }
        WinBy::Discard => {
            if rng.below(20) == 0 {
                situation.robbing_quad = true;
            } else if rng.below(24) == 0 {
                situation.under_the_river = true;
            }
        }
    }
    situation.counters = rng.below(4) as u32;
    // Indicators are drawn from tiles the hand does not hold, which is not
    // required but keeps the dora counts in a believable range.
    let count = 1 + rng.below(4);
    for _ in 0..count {
        for _ in 0..20 {
            let tile = Tile::new(rng.below(KINDS) as u8);
            if seen.count(tile) < COPIES {
                situation.dora_indicators.push(tile);
                break;
            }
        }
    }
    if !matches!(situation.riichi, Riichi::None) {
        for _ in 0..situation.dora_indicators.len() {
            situation
                .ura_indicators
                .push(Tile::new(rng.below(KINDS) as u8));
        }
    }
    situation
}

/// The name the reference library uses for a meld kind.
fn meld_kind(kind: MeldKind) -> &'static str {
    match kind {
        MeldKind::Chii => "chi",
        MeldKind::Pon => "pon",
        MeldKind::ClaimedKan | MeldKind::ExtendedKan => "kan",
        MeldKind::ConcealedKan => "ankan",
    }
}

/// Writes tiles as a JSON array of the plain notation.
fn tiles_json(tiles: impl IntoIterator<Item = Tile>) -> String {
    let names: Vec<String> = tiles
        .into_iter()
        .map(|tile| format!("\"{tile}\""))
        .collect();
    format!("[{}]", names.join(","))
}

/// Writes random winning hands and what this engine scores them, one JSON
/// object per line.
pub fn dump(count: usize, seed: u64) {
    let mut rng = Rng::from_seed(seed);
    let mut written = 0;
    let mut attempts = 0;

    while written < count && attempts < count * 40 {
        attempts += 1;
        let Some((concealed, melds, winning)) = build(&mut rng) else {
            continue;
        };
        let open = melds.iter().any(|meld| meld.kind.opens_hand());
        let mut seen = concealed;
        for meld in &melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
        let situation = pick_situation(&mut rng, winning, !open, &seen);

        // The scorer wants the hand without the winning tile only when the
        // win was by discard; it is given the whole hand and the tile.
        let Ok(scored) = score::score(&concealed, &melds, &situation) else {
            continue;
        };

        let melds_json: Vec<String> = melds
            .iter()
            .map(|meld| {
                format!(
                    "{{\"kind\":\"{}\",\"tiles\":{},\"opened\":{}}}",
                    meld_kind(meld.kind),
                    tiles_json(meld.tiles()),
                    meld.kind.opens_hand()
                )
            })
            .collect();
        let yaku: Vec<String> = scored
            .yaku
            .iter()
            .map(|(yaku, han)| format!("[\"{}\",{}]", yaku.name(), han))
            .collect();

        println!(
            "{{\"concealed\":{},\"melds\":[{}],\"winning\":\"{}\",\"seat\":{},\"round\":{},\
             \"tsumo\":{},\"riichi\":{},\"double_riichi\":{},\"ippatsu\":{},\"after_quad\":{},\
             \"robbing_quad\":{},\"under_the_sea\":{},\"under_the_river\":{},\
             \"dora_indicators\":{},\"ura_indicators\":{},\"counters\":{},\
             \"han\":{},\"fu\":{},\"dora\":{},\"limit\":{},\"yaku\":[{}]}}",
            tiles_json(concealed.tiles()),
            melds_json.join(","),
            winning,
            situation.seat.index(),
            situation.round.index(),
            matches!(situation.win_by, WinBy::SelfDraw),
            !matches!(situation.riichi, Riichi::None),
            matches!(situation.riichi, Riichi::Double),
            situation.ippatsu,
            situation.after_quad,
            situation.robbing_quad,
            situation.under_the_sea,
            situation.under_the_river,
            tiles_json(situation.dora_indicators.clone()),
            tiles_json(situation.ura_indicators.clone()),
            situation.counters,
            scored.han,
            scored.fu,
            scored.dora,
            match scored.limit {
                Some(limit) => format!("\"{}\"", limit.name()),
                None => "null".to_string(),
            },
            yaku.join(","),
        );
        written += 1;
    }
}
