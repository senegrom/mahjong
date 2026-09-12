//! Mortal's reading of a table that was typed in rather than played.
//!
//! The trained network sees the thousand and twelve planes Mortal's encoder
//! builds, and that encoder is driven by events, not by a position: it wants
//! the hand as it happened. A guided game and a physical table give a
//! position instead — this seat's tiles, everyone's discards in order,
//! everyone's called sets, and nothing at all about the hands nobody has
//! shown.
//!
//! That is enough, because the encoder never reads what it cannot see.
//! libriichi's tile type has an explicit unknown and its state loads only
//! its own seat's deal, so the other three hands are dealt face down and
//! everything the observation does use is recovered exactly: the rivers
//! carry their own order numbers, so each discard lands on the turn it was
//! made, with the tedashi and riichi marks that were entered; the called
//! sets say who took what from whom; the wall count fixes how many draws
//! there have been.
//!
//! Two things are invented, and both are invented because they cannot be
//! seen. The first is which of the observed seat's own discards came
//! straight off the draw: the observation does not encode that for the seat
//! itself, only for the other three, so every one of its own discards is
//! replayed as a tile drawn and let go again, which keeps the tiles it
//! holds exactly the tiles that were entered at every point of the replay.
//! The second is when a concealed or added quad was declared, which nothing
//! in a position records; it is placed on one of that seat's own turns.

use std::collections::{HashMap, HashSet};

use riichi::mjai::Event as MortalEvent;
use riichi::state::PlayerState as MortalState;
use riichi_core::game::{Discard, Hand, Phase};
use riichi_core::hand::{ClaimedFrom, MeldKind};
use riichi_core::mjai::name;
use riichi_core::tile::Tile;
use riichi_core::Wind;

/// Tiles in the live wall before the dealer's first draw, which is where
/// libriichi's own count starts.
const WALL_AT_START: usize = 70;

/// A called set matched to the discard it took.
struct Claim {
    /// Who called.
    claimer: Wind,
    /// Which of that seat's sets this is.
    meld: usize,
    /// Whose discard it was.
    target: Wind,
    /// The tile claimed.
    tile: Tile,
}

/// Mortal's state for one seat of a position on its own, having been told
/// everything the position says happened.
///
/// `None` where the position describes a hand this replay cannot express;
/// the caller says so rather than answering from a state built on a guess.
pub(super) fn state_for(hand: &Hand, seat: Wind) -> Option<MortalState> {
    let mut state = MortalState::new(seat.index() as u8);
    for json in events(hand, seat)? {
        let event = serde_json::from_str::<MortalEvent>(&json).ok()?;
        state.update(&event).ok()?;
    }
    Some(state)
}

fn wind_of(index: usize) -> Wind {
    [Wind::East, Wind::South, Wind::West, Wind::North][index % 4]
}

/// Which seat a set was claimed from, given who claimed it.
fn source_of(claimer: usize, from: ClaimedFrom) -> Option<usize> {
    let offset = match from {
        ClaimedFrom::Right => 1,
        ClaimedFrom::Across => 2,
        ClaimedFrom::Left => 3,
        ClaimedFrom::SelfDrawn => return None,
    };
    Some((claimer + offset) % 4)
}

/// Pairs every discard marked as claimed with the set that took it, the way
/// the position's own validation does; that validation has already passed,
/// so a pairing is there to be found.
fn claims(hand: &Hand) -> Option<Vec<Claim>> {
    let mut takers: Vec<(usize, Vec<Tile>, usize, usize)> = Vec::new();
    for (claimer, player) in hand.players.iter().enumerate() {
        for (at, meld) in player.melds.iter().enumerate() {
            let Some(source) = source_of(claimer, meld.from) else {
                continue;
            };
            let could_take = match meld.kind {
                MeldKind::Chii => meld.tiles(),
                MeldKind::ConcealedKan => continue,
                _ => vec![meld.tile],
            };
            takers.push((source, could_take, claimer, at));
        }
    }
    let mut found = Vec::new();
    for (index, player) in hand.players.iter().enumerate() {
        for discard in player.discards.iter().filter(|d| d.claimed) {
            // A set naming exactly this tile first, so a sequence's three
            // possible tiles stay for the discards only it explains.
            let at = takers
                .iter()
                .position(|(who, tiles, ..)| *who == index && tiles == &[discard.tile])
                .or_else(|| {
                    takers
                        .iter()
                        .position(|(who, tiles, ..)| *who == index && tiles.contains(&discard.tile))
                })?;
            let (_, _, claimer, meld) = takers.swap_remove(at);
            found.push(Claim {
                claimer: wind_of(claimer),
                meld,
                target: wind_of(index),
                tile: discard.tile,
            });
        }
    }
    Some(found)
}

/// Every discard of the hand, in the order they were made.
fn ordered(hand: &Hand) -> Vec<(Wind, Discard)> {
    let mut all: Vec<(Wind, Discard)> = hand
        .players
        .iter()
        .enumerate()
        .flat_map(|(index, player)| {
            player
                .discards
                .iter()
                .map(move |discard| (wind_of(index), *discard))
        })
        .collect();
    all.sort_by_key(|(_, discard)| discard.order);
    all
}

fn tiles_json(tiles: &[String]) -> String {
    let mut out = String::from("[");
    for (index, tile) in tiles.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(tile);
        out.push('"');
    }
    out.push(']');
    out
}

/// A draw. Only the seat being observed sees what it drew; for the other
/// three, libriichi reads nothing but the fact that the wall shrank.
fn tsumo(actor: Wind, tile: Option<Tile>) -> String {
    format!(
        "{{\"type\":\"tsumo\",\"actor\":{},\"pai\":\"{}\"}}",
        actor.index(),
        tile.map_or_else(|| "?".to_string(), name),
    )
}

fn dahai(actor: Wind, tile: Tile, tsumogiri: bool) -> String {
    format!(
        "{{\"type\":\"dahai\",\"actor\":{},\"pai\":\"{}\",\"tsumogiri\":{}}}",
        actor.index(),
        name(tile),
        tsumogiri,
    )
}

fn claim_json(kind: &str, actor: Wind, target: Wind, tile: Tile, consumed: &[Tile]) -> String {
    let names: Vec<String> = consumed.iter().copied().map(name).collect();
    format!(
        "{{\"type\":\"{kind}\",\"actor\":{},\"target\":{},\"pai\":\"{}\",\"consumed\":{}}}",
        actor.index(),
        target.index(),
        name(tile),
        tiles_json(&names),
    )
}

fn kakan_json(actor: Wind, tile: Tile) -> String {
    let names = vec![name(tile), name(tile), name(tile)];
    format!(
        "{{\"type\":\"kakan\",\"actor\":{},\"pai\":\"{}\",\"consumed\":{}}}",
        actor.index(),
        name(tile),
        tiles_json(&names),
    )
}

fn dora_json(indicator: Tile) -> String {
    format!(
        "{{\"type\":\"dora\",\"dora_marker\":\"{}\"}}",
        name(indicator)
    )
}

fn ankan_json(actor: Wind, tile: Tile) -> String {
    let names = vec![name(tile), name(tile), name(tile), name(tile)];
    format!(
        "{{\"type\":\"ankan\",\"actor\":{},\"consumed\":{}}}",
        actor.index(),
        tiles_json(&names),
    )
}

/// The tiles a set takes out of the hand its owner was dealt.
///
/// Three for every set, whatever kind it is: a quad's fourth tile comes off
/// the draw, and a sequence or triplet is followed by a discard from hand,
/// which is the third. That is why thirteen tiles always come to thirteen
/// again, however many sets stand in front of a seat.
fn consumed_from_hand(kind: MeldKind, tile: Tile, claimed: Option<Tile>) -> Vec<Tile> {
    match kind {
        MeldKind::Chii => {
            let run = [
                tile,
                tile.next_in_suit().unwrap_or(tile),
                tile.next_in_suit()
                    .and_then(|t| t.next_in_suit())
                    .unwrap_or(tile),
            ];
            let taken = claimed.unwrap_or(tile);
            let mut rest: Vec<Tile> = Vec::new();
            let mut skipped = false;
            for t in run {
                if !skipped && t == taken {
                    skipped = true;
                    continue;
                }
                rest.push(t);
            }
            rest
        }
        MeldKind::Pon | MeldKind::ExtendedKan => vec![tile, tile],
        MeldKind::ClaimedKan | MeldKind::ConcealedKan => vec![tile, tile, tile],
    }
}

/// Whether a set is followed by a discard out of the hand rather than a
/// replacement drawn from the dead wall.
fn discards_after(kind: MeldKind) -> bool {
    matches!(kind, MeldKind::Chii | MeldKind::Pon | MeldKind::ExtendedKan)
}
/// A meld as the replay names it: who called it, what kind, and the tile
/// it was called on.
type Called = (Wind, MeldKind, Tile);

fn events(hand: &Hand, seat: Wind) -> Option<Vec<String>> {
    let taken = claims(hand)?;
    let rows = ordered(hand);
    let by_order: HashMap<u32, usize> = taken
        .iter()
        .enumerate()
        .map(|(at, claim)| {
            let order = hand.players[claim.target.index()]
                .discards
                .iter()
                .find(|d| d.claimed && d.tile == claim.tile)
                .map_or(u32::MAX, |d| d.order);
            (order, at)
        })
        .collect();

    // A set that takes a tile from a river is followed by a discard out of
    // the hand, with no draw between: the claimer's next one.
    let mut from_hand: HashSet<(usize, u32)> = HashSet::new();
    for (order, at) in &by_order {
        let claim = &taken[*at];
        if !discards_after(hand.players[claim.claimer.index()].melds[claim.meld].kind) {
            continue;
        }
        if let Some(next) = hand.players[claim.claimer.index()]
            .discards
            .iter()
            .filter(|d| d.order > *order)
            .find(|d| !from_hand.contains(&(claim.claimer.index(), d.order)))
        {
            from_hand.insert((claim.claimer.index(), next.order));
        }
    }

    // Quads nobody offered a tile for have no moment of their own in a
    // position. They are given one: a turn of that seat's, on which it drew
    // the tile that completed the quad, declared it, and let the
    // replacement go again. The seat acting now, with a replacement in
    // hand, has instead just declared one, so its quad goes at the end.
    let holding_replacement = hand.phase == Phase::Act && hand.after_quad;
    let robbed = hand
        .robbable_quad
        .filter(|_| hand.phase == Phase::CallWindow);
    let mut standing: HashMap<(usize, u32), Vec<Called>> = HashMap::new();
    let mut closing: Vec<Called> = Vec::new();
    for (index, player) in hand.players.iter().enumerate() {
        let who = wind_of(index);
        for meld in &player.melds {
            if !matches!(meld.kind, MeldKind::ConcealedKan | MeldKind::ExtendedKan) {
                continue;
            }
            let last = who == hand.turn
                && (holding_replacement || robbed.is_some_and(|tile| tile == meld.tile));
            if last {
                closing.push((who, meld.kind, meld.tile));
                continue;
            }
            // The seat's own turns are the only place a quad fits; taking
            // the latest keeps the usual shape of a hand, where quads come
            // late, and keeps the tile out of the river before it is drawn.
            let slot = player
                .discards
                .iter()
                .rev()
                .find(|d| !from_hand.contains(&(index, d.order)))
                .map(|d| d.order);
            match slot {
                Some(order) => standing
                    .entry((index, order))
                    .or_default()
                    .push((who, meld.kind, meld.tile)),
                // Declared and not yet discarded from: it can only be now.
                None => closing.push((who, meld.kind, meld.tile)),
            }
        }
    }

    // The thirteen tiles the observed seat must have been dealt for the
    // replay to leave it holding what was entered.
    let mine = &hand.players[seat.index()];
    let mut dealt: Vec<Tile> = Vec::new();
    for tile in Tile::all() {
        for _ in 0..mine.hand.count(tile) {
            dealt.push(tile);
        }
    }
    if hand.turn == seat {
        if let Some(drawn) = hand.drawn {
            let at = dealt.iter().position(|t| *t == drawn)?;
            dealt.remove(at);
        }
    }
    for (at, meld) in mine.melds.iter().enumerate() {
        let claimed = taken
            .iter()
            .find(|claim| claim.claimer == seat && claim.meld == at)
            .map(|claim| claim.tile);
        dealt.extend(consumed_from_hand(meld.kind, meld.tile, claimed));
    }
    for discard in &mine.discards {
        if from_hand.contains(&(seat.index(), discard.order)) {
            dealt.push(discard.tile);
        }
    }
    if dealt.len() != 13 {
        return None;
    }

    // A declaration is paid for when it stands, not when it is made: our
    // engine takes the thousand points at the discard, libriichi at the
    // acceptance that follows. Both must not take it, so the hand is dealt
    // with every bet placed this hand still in front of its owner, and the
    // replay takes them again as it reaches each declaration.
    let bet: Vec<u32> = hand
        .players
        .iter()
        .map(|player| player.discards.iter().filter(|d| d.riichi).count() as u32)
        .collect();
    let scores: Vec<i32> = hand
        .players
        .iter()
        .zip(&bet)
        .map(|(player, placed)| player.score + 1000 * *placed as i32)
        .collect();
    let kyotaku = hand.riichi_sticks.saturating_sub(bet.iter().sum::<u32>());
    let indicators = hand.wall.dora_indicators();
    let first = *indicators.first()?;
    let mut faces = String::from("[");
    for index in 0..4 {
        if index > 0 {
            faces.push(',');
        }
        let names: Vec<String> = if index == seat.index() {
            dealt.iter().copied().map(name).collect()
        } else {
            vec!["?".to_string(); 13]
        };
        faces.push_str(&tiles_json(&names));
    }
    faces.push(']');

    // Indicators after the first are turned over by quads, one each, and a
    // tile discarded before one was turned was not a dora then. Our engine
    // writes the new indicator once the quad stands and before the
    // replacement is drawn, so the replay does the same.
    let mut turning: Vec<Tile> = indicators.iter().skip(1).rev().copied().collect();
    let mut tail: Vec<String> = Vec::new();
    let mut drawn = 0usize;
    let draw = |tail: &mut Vec<String>, drawn: &mut usize, actor: Wind, tile: Option<Tile>| {
        tail.push(tsumo(actor, tile));
        *drawn += 1;
    };

    for (who, discard) in &rows {
        let mine = *who == seat;
        let post = from_hand.contains(&(who.index(), discard.order));
        // A quad on this turn supplies the tile the discard sends out: it
        // was drawn, declared, and the replacement taken in its place.
        let mut supplied = false;
        for (owner, kind, tile) in standing
            .remove(&(who.index(), discard.order))
            .unwrap_or_default()
        {
            draw(
                &mut tail,
                &mut drawn,
                owner,
                (owner == seat).then_some(tile),
            );
            tail.push(if kind == MeldKind::ConcealedKan {
                ankan_json(owner, tile)
            } else {
                kakan_json(owner, tile)
            });
            if let Some(indicator) = turning.pop() {
                tail.push(dora_json(indicator));
            }
            draw(
                &mut tail,
                &mut drawn,
                owner,
                (owner == seat).then_some(discard.tile),
            );
            supplied = true;
        }
        if !post && !supplied {
            draw(&mut tail, &mut drawn, *who, mine.then_some(discard.tile));
        }
        if discard.riichi {
            tail.push(format!("{{\"type\":\"reach\",\"actor\":{}}}", who.index()));
        }
        // Its own discards are replayed as tiles drawn and let go, which is
        // what the extra draw above supplied; the other three keep the mark
        // that was entered for them.
        let tsumogiri = if mine { !post } else { discard.drawn && !post };
        tail.push(dahai(*who, discard.tile, tsumogiri));
        if discard.riichi && !is_pending(hand, *who, discard) {
            tail.push(format!(
                "{{\"type\":\"reach_accepted\",\"actor\":{}}}",
                who.index()
            ));
        }
        if let Some(at) = by_order.get(&discard.order) {
            let claim = &taken[*at];
            let meld = hand.players[claim.claimer.index()].melds[claim.meld];
            let consumed = consumed_from_hand(
                if meld.kind == MeldKind::ExtendedKan {
                    MeldKind::Pon
                } else {
                    meld.kind
                },
                meld.tile,
                Some(claim.tile),
            );
            let kind = match meld.kind {
                MeldKind::Chii => "chi",
                MeldKind::ClaimedKan => "daiminkan",
                _ => "pon",
            };
            tail.push(claim_json(
                kind,
                claim.claimer,
                claim.target,
                claim.tile,
                &consumed,
            ));
            if meld.kind == MeldKind::ClaimedKan {
                if let Some(indicator) = turning.pop() {
                    tail.push(dora_json(indicator));
                }
                // The replacement, which its next discard sends out again.
                let next = hand.players[claim.claimer.index()]
                    .discards
                    .iter()
                    .find(|d| d.order > discard.order)
                    .map(|d| d.tile);
                draw(
                    &mut tail,
                    &mut drawn,
                    claim.claimer,
                    (claim.claimer == seat).then_some(next).flatten(),
                );
            }
        }
    }

    // A quad declared but not yet discarded from, and the seat that holds a
    // replacement now: both belong after everything in the rivers.
    for (owner, kind, tile) in closing {
        draw(
            &mut tail,
            &mut drawn,
            owner,
            (owner == seat).then_some(tile),
        );
        tail.push(if kind == MeldKind::ConcealedKan {
            ankan_json(owner, tile)
        } else {
            kakan_json(owner, tile)
        });
        // A quad still open to robbery has not turned its indicator, and
        // its replacement has not been taken; both wait on the answer.
        if robbed.is_some_and(|robbable| robbable == tile) {
            continue;
        }
        if let Some(indicator) = turning.pop() {
            tail.push(dora_json(indicator));
        }
    }

    // What the seat holding the turn is looking at right now: the tile it
    // drew, or the set it has just taken and must discard from.
    if hand.phase == Phase::Act {
        if let Some(tile) = hand.drawn {
            draw(
                &mut tail,
                &mut drawn,
                hand.turn,
                (hand.turn == seat).then_some(tile),
            );
        }
    }

    // Every draw the position says happened that the replay has not
    // accounted for, so the wall reads the same number it does at the
    // table. They are dealt to a seat whose hand nothing reads.
    let expected = WALL_AT_START.saturating_sub(hand.wall.remaining());
    let filler = expected.saturating_sub(drawn);
    let elsewhere = if seat == Wind::East {
        Wind::South
    } else {
        Wind::East
    };

    let mut out = Vec::with_capacity(tail.len() + filler + 2);
    out.push(
        "{\"type\":\"start_game\",\"names\":[\"player 0\",\"player 1\",\"player 2\",\"player 3\"]}"
            .to_string(),
    );
    out.push(format!(
        "{{\"type\":\"start_kyoku\",\"bakaze\":\"{}\",\"kyoku\":{},\"honba\":{},\"kyotaku\":{},\"oya\":0,\"dora_marker\":\"{}\",\"scores\":[{},{},{},{}],\"tehais\":{}}}",
        name(hand.round.tile()),
        hand.kyoku,
        hand.counters,
        kyotaku,
        name(first),
        scores[0], scores[1], scores[2], scores[3],
        faces,
    ));
    for _ in 0..filler {
        out.push(tsumo(elsewhere, None));
    }
    out.extend(tail);
    Some(out)
}

/// Whether this discard is the one still sitting on the table.
fn is_pending(hand: &Hand, who: Wind, discard: &Discard) -> bool {
    hand.pending_discard == Some((who, discard.tile))
        && hand.players[who.index()]
            .discards
            .iter()
            .all(|other| other.order <= discard.order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use riichi_core::bot::{Bot, Style};
    use riichi_core::game::Action;
    use riichi_core::rng::Rng;
    use riichi_core::table::Table;

    /// Mortal's state for a hand that really was played, told the engine's
    /// own log: the answer the replay has to match.
    fn told(hand: &Hand, seat: Wind) -> MortalState {
        let mut state = MortalState::new(seat.index() as u8);
        let start = riichi_core::mjai::Event::StartGame {
            names: std::array::from_fn(|player| format!("player {player}")),
        };
        for json in std::iter::once(start.to_json([0, 1, 2, 3]))
            .chain(hand.log.iter().map(|event| event.to_json([0, 1, 2, 3])))
        {
            if let Ok(event) = serde_json::from_str::<MortalEvent>(&json) {
                let _ = state.update(&event);
            }
        }
        state
    }

    /// Plays a hand for `steps` decisions and stops wherever that leaves it.
    fn part_way(seed: u64, steps: usize) -> Hand {
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut hand = table.deal(&mut rng);
        let mut bot = Bot::with_style(seed, Style::club());
        let mut taken = 0;
        while taken < steps && !matches!(hand.phase, Phase::Over) {
            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    let action = bot.act(&hand);
                    if matches!(action, Action::Tsumo) {
                        break;
                    }
                    hand.act(action).unwrap();
                    taken += 1;
                }
                Phase::CallWindow => {
                    let answers: Vec<_> = hand
                        .legal_calls()
                        .into_iter()
                        .map(|(who, calls)| (who, bot.call(&hand, who, &calls)))
                        .collect();
                    if answers
                        .iter()
                        .any(|(_, call)| matches!(call, riichi_core::game::Call::Ron))
                    {
                        break;
                    }
                    hand.resolve_calls(&answers).unwrap();
                    taken += 1;
                }
                Phase::Over => break,
            }
        }
        hand
    }

    /// The replay is built from a position, so it must not read anything a
    /// position does not carry: the other three hands are emptied first,
    /// exactly as a guided game holds them.
    fn hidden(mut hand: Hand, seat: Wind) -> Hand {
        for (index, player) in hand.players.iter_mut().enumerate() {
            if index != seat.index() {
                player.hand = riichi_core::hand::TileSet::new();
            }
        }
        hand
    }

    fn planes(state: &MortalState) -> Vec<f32> {
        let (observation, _mask) = state.encode_obs(4, false);
        observation.iter().copied().collect()
    }

    /// The two planes a position cannot carry, and where they sit.
    ///
    /// libriichi freezes which discards keep the shanten number and which
    /// improve it at the moment a hand declares riichi, and never refreshes
    /// them after: from then on they describe the hand one turn before the
    /// declaration. That turn had a draw, and which tile was drawn is the
    /// one thing about its own river a position does not record, so the
    /// replay classifies those discards against the hand it can see rather
    /// than the hand that was held. A seat in riichi has no discard to
    /// choose, so nothing reads them; every other seat matches exactly.
    ///
    /// The index is counted from the encoder: the block that opens with the
    /// discard candidates begins at 874, and these are the two planes after
    /// it. If libriichi's layout moves, this test says so.
    const FROZEN_SHANTEN_DISCARDS: [usize; 2] = [875, 876];

    #[test]
    fn a_position_replays_into_the_observation_it_came_from() {
        let (mut checked, mut called, mut quads, mut reached) = (0, 0, 0, 0);
        for seed in 1..40u64 {
            for steps in [4usize, 9, 15, 22, 28, 36, 44, 52] {
                let hand = part_way(seed, steps);
                if matches!(hand.phase, Phase::Over) {
                    continue;
                }
                for seat in [Wind::East, Wind::South, Wind::West, Wind::North] {
                    if hand.phase == Phase::Act && hand.turn != seat {
                        continue;
                    }
                    called += usize::from(hand.players.iter().any(|p| !p.melds.is_empty()));
                    quads += usize::from(
                        hand.players
                            .iter()
                            .any(|p| p.melds.iter().any(|m| m.kind.is_kan())),
                    );
                    reached += usize::from(hand.players.iter().any(|p| p.has_riichi()));
                    let want = planes(&told(&hand, seat));
                    let position = hidden(hand.clone(), seat);
                    let state = state_for(&position, seat)
                        .unwrap_or_else(|| panic!("seed {seed} step {steps} seat {seat:?}"));
                    let got = planes(&state);
                    assert_eq!(want.len(), got.len());
                    let wrong: Vec<usize> = (0..want.len() / 34)
                        .filter(|plane| {
                            (0..34).any(|at| {
                                (want[plane * 34 + at] - got[plane * 34 + at]).abs() > 1e-6
                            })
                        })
                        .collect();
                    let excused = hand.players[seat.index()].has_riichi()
                        && wrong
                            .iter()
                            .all(|plane| FROZEN_SHANTEN_DISCARDS.contains(plane));
                    assert!(
                        wrong.is_empty() || excused,
                        "seed {seed} step {steps} seat {seat:?} phase {:?}: {} planes differ, first {:?}",
                        hand.phase,
                        wrong.len(),
                        &wrong[..wrong.len().min(12)],
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 200, "only {checked} positions were compared");
        assert!(called > 20, "only {called} of them had a called set");
        assert!(quads > 0, "none of them had a quad");
        assert!(reached > 20, "only {reached} of them had a declared riichi");
    }

    /// A seat not in riichi has nothing excused: every one of the thousand
    /// and twelve planes is the plane the hand really produced.
    #[test]
    fn a_seat_that_has_not_declared_matches_exactly() {
        let mut checked = 0;
        for seed in 1..40u64 {
            for steps in [9usize, 22, 36, 52] {
                let hand = part_way(seed, steps);
                if matches!(hand.phase, Phase::Over) {
                    continue;
                }
                for seat in [Wind::East, Wind::South, Wind::West, Wind::North] {
                    if hand.phase == Phase::Act && hand.turn != seat {
                        continue;
                    }
                    if hand.players[seat.index()].has_riichi() {
                        continue;
                    }
                    let want = planes(&told(&hand, seat));
                    let got = planes(&state_for(&hidden(hand.clone(), seat), seat).unwrap());
                    let wrong: Vec<usize> = (0..want.len() / 34)
                        .filter(|plane| {
                            (0..34).any(|at| {
                                (want[plane * 34 + at] - got[plane * 34 + at]).abs() > 1e-6
                            })
                        })
                        .collect();
                    assert!(
                        wrong.is_empty(),
                        "seed {seed} step {steps} seat {seat:?} phase {:?}: planes {:?}",
                        hand.phase,
                        &wrong[..wrong.len().min(12)],
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 150, "only {checked} positions were compared");
    }
}
