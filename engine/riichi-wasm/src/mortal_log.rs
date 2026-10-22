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
//! The second is when a concealed or added quad was declared, which a table
//! typed in by hand does not record; it is placed on one of that seat's own
//! turns. A position that does record it, as a decision of a game played
//! here does (`History`), has its quads told on the turns they were made.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::analysis::{claims, unclaimed_set};
use riichi::mjai::Event as MortalEvent;
use riichi::state::PlayerState as MortalState;
use riichi_core::game::{Discard, Hand, Phase};
use riichi_core::hand::MeldKind;
use riichi_core::mjai::{name, Event};
use riichi_core::tile::Tile;
use riichi_core::Wind;

/// Tiles in the live wall before the dealer's first draw, which is where
/// libriichi's own count starts.
const WALL_AT_START: usize = 70;

/// What a position can say about its hand that the engine's own state of a
/// hand does not keep, and that Mortal's reading of the hand depends on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct History {
    /// Which player number sits in each seat, East first. Mortal numbers
    /// the players from the game's first dealer, and where two of them hold
    /// the same points it ranks the lower number first, so the four planes
    /// of the observed seat's rank depend on who is who, not only on the
    /// points.
    pub(super) seating: [usize; 4],
    /// For each seat, in the order of its sets, the turn on which each quad
    /// it declared itself was declared, named by the order number of the
    /// discard that turn ended with, or will end with for the turn being
    /// played; `None` where nothing says, and for every set that is not
    /// such a quad. Mortal marks a quad on the discard of the turn that
    /// made it, and a quad's new indicator makes dora of what is thrown
    /// after it and not before, so the turn shows.
    pub(super) quads: [Vec<Option<u32>>; 4],
    /// The tile the observed seat drew on the turn it declared riichi,
    /// where the position says. Mortal sorts a hand's discards into those
    /// that keep its distance from waiting and those that bring it closer
    /// whenever the seat draws, measured against the hand held before the
    /// draw, and stops once a riichi stands, so from then on it shows the
    /// declaring turn's sorting. Told as a tile drawn and let go, as the
    /// observed seat's turns otherwise are, that turn would be measured
    /// against the hand held after the declaration, which is a different
    /// distance wherever the turn's draw is what made the hand ready.
    pub(super) riichi_draw: Option<Tile>,
}

impl History {
    /// What a hand says by itself as the `kyoku`th of its round: the players
    /// numbered as every game numbers them, and no quad's turn. The first
    /// dealer is player 0 and the deal moves on one seat a hand, whichever
    /// round it is, so seat wind `w` of hand `k` is player `w + k − 1`, as
    /// `Table::player_at` seats them.
    pub(super) fn of_kyoku(kyoku: u8) -> History {
        let moved = usize::from(kyoku.max(1)) - 1;
        History {
            seating: std::array::from_fn(|seat| (seat + moved) % 4),
            quads: Default::default(),
            riichi_draw: None,
        }
    }

    /// What a hand played here says about itself, read from its own log,
    /// with `seat` the seat observed: the players numbered by the hand
    /// number, as every game numbers them (`of_kyoku`), the turn of every
    /// quad a seat declared itself, and the tile `seat` drew on the turn it
    /// declared riichi. The log tells a quad before the draw that replaces
    /// it, and the seat's next discard ends that turn, its order number
    /// being how many discards the log told before it. A quad no discard
    /// has followed yet was declared on the turn being played now, whose
    /// discard will carry the next number, which the replay knows to tell
    /// last. A declaration follows the declaring seat's draw.
    pub(super) fn from_log(hand: &Hand, seat: Wind) -> History {
        let mut quads: [Vec<Option<u32>>; 4] =
            std::array::from_fn(|seat| vec![None; hand.players[seat].melds.len()]);
        let mut open: Vec<(Wind, Tile)> = Vec::new();
        let mut made = 0u32;
        let (mut drawn, mut riichi_draw) = (None, None);
        let mark = |quads: &mut [Vec<Option<u32>>; 4], seat: Wind, tile: Tile, made: u32| {
            let at = hand.players[seat.index()].melds.iter().position(|meld| {
                meld.tile == tile
                    && matches!(meld.kind, MeldKind::ConcealedKan | MeldKind::ExtendedKan)
            });
            if let Some(at) = at {
                quads[seat.index()][at] = Some(made);
            }
        };
        for event in &hand.log {
            match event {
                Event::Tsumo { actor, tile } if *actor == seat => drawn = Some(*tile),
                Event::Reach { actor } if *actor == seat => riichi_draw = drawn,
                Event::Ankan { actor, consumed } => {
                    open.extend(consumed.first().map(|tile| (*actor, *tile)));
                }
                Event::Kakan { actor, tile, .. } => open.push((*actor, *tile)),
                Event::Dahai { actor, .. } => {
                    let (ended, waiting): (Vec<_>, Vec<_>) =
                        open.drain(..).partition(|(seat, _)| seat == actor);
                    open = waiting;
                    for (seat, tile) in ended {
                        mark(&mut quads, seat, tile, made);
                    }
                    made += 1;
                }
                _ => {}
            }
        }
        for (seat, tile) in open {
            mark(&mut quads, seat, tile, made);
        }
        History {
            seating: History::of_kyoku(hand.kyoku).seating,
            quads,
            riichi_draw,
        }
    }

    /// The turn on which a seat's set at `meld` was declared, if it is a
    /// quad of its own whose turn is known.
    fn quad_turn(&self, seat: usize, meld: usize) -> Option<u32> {
        self.quads[seat].get(meld).copied().flatten()
    }

    /// The seat each player number sits in, or `None` unless every number
    /// from 0 to 3 is seated exactly once.
    fn seats_by_player(&self) -> Option<[Wind; 4]> {
        let mut seats = [None; 4];
        for seat in Wind::ALL {
            let slot = seats.get_mut(self.seating[seat.index()])?;
            if slot.replace(seat).is_some() {
                return None;
            }
        }
        Some(seats.map(|seat| seat.expect("four numbers fill four slots")))
    }
}

/// Mortal's state for one seat of a position on its own, having been told
/// everything the position says happened.
///
/// `None` where the position describes a hand this replay cannot express;
/// the caller says so rather than answering from a state built on a guess.
pub(super) fn state_for(hand: &Hand, seat: Wind, history: &History) -> Option<MortalState> {
    let mut state = MortalState::new(history.seating[seat.index()] as u8);
    for json in events(hand, seat, history)? {
        let event = serde_json::from_str::<MortalEvent>(&json).ok()?;
        state.update(&event).ok()?;
    }
    Some(state)
}

fn wind_of(index: usize) -> Wind {
    [Wind::East, Wind::South, Wind::West, Wind::North][index % 4]
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

/// An event as the replay tells it, written the way the engine writes its
/// own log, with the players numbered as `seating` seats them.
fn line(event: Event, seating: [usize; 4]) -> String {
    event.to_json(seating)
}

/// A draw. Only the seat being observed sees what it drew; for the other
/// three, libriichi reads nothing but the fact that the wall shrank. The
/// engine's own log always knows the tile, so an unknown one is written
/// here.
fn tsumo(actor: Wind, tile: Option<Tile>, seating: [usize; 4]) -> String {
    format!(
        "{{\"type\":\"tsumo\",\"actor\":{},\"pai\":\"{}\"}}",
        seating[actor.index()],
        tile.map_or_else(|| "?".to_string(), name),
    )
}

/// A quad a seat declared itself: four concealed tiles, or a fourth added
/// to a triplet it claimed.
fn own_quad(actor: Wind, kind: MeldKind, tile: Tile, seating: [usize; 4]) -> String {
    line(
        if kind == MeldKind::ConcealedKan {
            Event::Ankan {
                actor,
                consumed: vec![tile; 4],
            }
        } else {
            Event::Kakan {
                actor,
                tile,
                consumed: vec![tile; 3],
            }
        },
        seating,
    )
}

/// The tiles a set takes out of the hand its owner was dealt.
///
/// Three for every set, whatever kind it is: a quad's fourth tile comes off
/// the draw, and a sequence or triplet is followed by a discard from hand,
/// which is the third. That is why thirteen tiles always come to thirteen
/// again, however many sets stand in front of a seat. `None` for a
/// sequence that cannot hold the tile it was claimed on.
fn consumed_from_hand(kind: MeldKind, tile: Tile, claimed: Option<Tile>) -> Option<Vec<Tile>> {
    Some(match kind {
        MeldKind::Chii => {
            let mut rest = tile.sequence()?.to_vec();
            let taken = claimed.unwrap_or(tile);
            rest.remove(rest.iter().position(|member| *member == taken)?);
            rest
        }
        MeldKind::Pon | MeldKind::ExtendedKan => vec![tile, tile],
        MeldKind::ClaimedKan | MeldKind::ConcealedKan => vec![tile, tile, tile],
    })
}

/// Whether a set is followed by a discard out of the hand rather than a
/// replacement drawn from the dead wall.
fn discards_after(kind: MeldKind) -> bool {
    matches!(kind, MeldKind::Chii | MeldKind::Pon | MeldKind::ExtendedKan)
}
/// A meld as the replay names it: who called it, what kind, and the tile
/// it was called on.
type Called = (Wind, MeldKind, Tile);

fn events(hand: &Hand, seat: Wind, history: &History) -> Option<Vec<String>> {
    let seating = history.seating;
    let seats_by_player = history.seats_by_player()?;
    // The same pairing the position's validation found. Each called set is
    // told at the claimed discard it was paired with, so a set paired with
    // none would never be told, and the state would be a table without it.
    // Validation accepts such a set, since the built-in advisers read the
    // sets themselves; the replay refuses it.
    let taken = claims(hand).ok()?;
    if unclaimed_set(hand, &taken).is_some() {
        return None;
    }
    let rows = ordered(hand);
    // Each claim follows the very discard it was paired with: two sets that
    // took the same tile from the same seat took two different discards.
    let by_order: BTreeMap<u32, usize> = taken
        .iter()
        .enumerate()
        .map(|(at, claim)| (claim.order, at))
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

    // Quads nobody offered a tile for are told on a turn of their seat's,
    // on which it drew the tile that completed the quad, declared it, and
    // let the replacement go again: the turn the position says, where it
    // says one. That turn began with a draw, as no quad is declared on a
    // turn that began with a call, so a discard that followed a call is
    // not one, and the replay refuses it; and an added quad's turn came
    // after the call of the triplet it extends. Where the position does not
    // say, the quad is given a turn. A quad still open to robbery is being
    // declared now, whatever a position says, and so is any the position
    // places on the turn being played, whose discard will carry the next
    // number; the seat acting now with a replacement in hand has just
    // declared one. Those go at the end, in the order of the seat's sets,
    // bar the quad open to robbery, which was declared last of all.
    //
    // The engine clears its mark of a replacement only at the next draw, so
    // a seat that has just claimed the discard after one holds nothing.
    let holding_replacement = hand.phase == Phase::Act && hand.after_quad && hand.drawn.is_some();
    let robbed = hand
        .robbable_quad
        .filter(|_| hand.phase == Phase::CallWindow);
    let mut standing: HashMap<(usize, u32), Vec<Called>> = HashMap::new();
    let mut closing: Vec<Called> = Vec::new();
    for (index, player) in hand.players.iter().enumerate() {
        let who = wind_of(index);
        for (at, meld) in player.melds.iter().enumerate() {
            if !matches!(meld.kind, MeldKind::ConcealedKan | MeldKind::ExtendedKan) {
                continue;
            }
            let now = who == hand.turn && robbed.is_some_and(|tile| tile == meld.tile);
            let playing = who == hand.turn && (holding_replacement || robbed.is_some());
            // The turn the quad is told on, or `None` for the end.
            let slot = match history.quad_turn(index, at) {
                _ if now => None,
                Some(order) if playing && order == hand.discards_made => None,
                Some(order) => {
                    let drawn_turn = player.discards.iter().any(|d| d.order == order)
                        && !from_hand.contains(&(index, order));
                    let after_its_pon = meld.kind != MeldKind::ExtendedKan
                        || taken.iter().any(|claim| {
                            claim.claimer == who && claim.meld == at && claim.order < order
                        });
                    if !drawn_turn || !after_its_pon {
                        return None;
                    }
                    Some(order)
                }
                None if who == hand.turn && holding_replacement => None,
                // The seat's own turns are the only place a quad fits;
                // taking the latest keeps the usual shape of a hand, where
                // quads come late, and keeps the tile out of the river
                // before it is drawn. A quad declared and not yet discarded
                // from can only be now.
                None => player
                    .discards
                    .iter()
                    .rev()
                    .find(|d| !from_hand.contains(&(index, d.order)))
                    .map(|d| d.order),
            };
            match slot {
                Some(order) => standing
                    .entry((index, order))
                    .or_default()
                    .push((who, meld.kind, meld.tile)),
                None => closing.push((who, meld.kind, meld.tile)),
            }
        }
    }
    closing.sort_by_key(|&(who, _, tile)| who == hand.turn && robbed == Some(tile));

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
        dealt.extend(consumed_from_hand(meld.kind, meld.tile, claimed)?);
    }
    for discard in &mine.discards {
        if from_hand.contains(&(seat.index(), discard.order)) {
            dealt.push(discard.tile);
        }
    }
    if dealt.len() != 13 {
        return None;
    }
    // The turn the observed seat declared riichi on is told as it was
    // played where the position says what that turn drew: the tile drawn
    // kept, and the declaring tile thrown from a hand that had held it
    // until then in the drawn tile's place. A drawn tile the hand no longer
    // holds leaves the turn told as a tile drawn and let go.
    let kept = mine
        .discards
        .iter()
        .find(|discard| discard.riichi)
        .zip(history.riichi_draw)
        .filter(|(discard, drawn)| *drawn != discard.tile)
        .and_then(|(discard, drawn)| {
            let at = dealt.iter().position(|tile| *tile == drawn)?;
            dealt[at] = discard.tile;
            Some((discard.order, drawn))
        });

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
    // The deal is dealt by player number, as the points are: the observed
    // seat's thirteen tiles go to whichever number it has.
    let mut faces = String::from("[");
    for (player, at) in seats_by_player.iter().enumerate() {
        if player > 0 {
            faces.push(',');
        }
        let names: Vec<String> = if *at == seat {
            dealt.iter().copied().map(name).collect()
        } else {
            vec!["?".to_string(); 13]
        };
        faces.push_str(&tiles_json(&names));
    }
    faces.push(']');
    let points = seats_by_player.map(|at| scores[at.index()]);

    // Indicators after the first are turned over by quads, one each, and a
    // tile discarded before one was turned was not a dora then. Our engine
    // writes the new indicator once the quad stands and before the
    // replacement is drawn, so the replay does the same.
    let mut turning: Vec<Tile> = indicators.iter().skip(1).rev().copied().collect();
    let mut tail: Vec<String> = Vec::new();
    let mut drawn = 0usize;
    let draw = |tail: &mut Vec<String>, drawn: &mut usize, actor: Wind, tile: Option<Tile>| {
        tail.push(tsumo(actor, tile, seating));
        *drawn += 1;
    };

    for (who, discard) in &rows {
        let mine = *who == seat;
        let post = from_hand.contains(&(who.index(), discard.order));
        // What the turn's last draw brought: the tile it lets go, unless it
        // is the observed seat's declaring turn told as it was played.
        let brought = match kept {
            Some((order, drawn)) if mine && order == discard.order => drawn,
            _ => discard.tile,
        };
        // Quads on this turn supply the tile the discard sends out. The
        // turn's draw made the first, each quad's replacement made the next,
        // and the last one's replacement is what the turn brought: one draw
        // for the turn and one for each quad, however many share it.
        let quads = standing
            .remove(&(who.index(), discard.order))
            .unwrap_or_default();
        for (at, &(owner, kind, tile)) in quads.iter().enumerate() {
            if at == 0 {
                draw(
                    &mut tail,
                    &mut drawn,
                    owner,
                    (owner == seat).then_some(tile),
                );
            }
            tail.push(own_quad(owner, kind, tile, seating));
            if let Some(indicator) = turning.pop() {
                tail.push(line(Event::Dora { indicator }, seating));
            }
            let replacement = quads.get(at + 1).map_or(brought, |&(_, _, next)| next);
            draw(
                &mut tail,
                &mut drawn,
                owner,
                (owner == seat).then_some(replacement),
            );
        }
        if !post && quads.is_empty() {
            draw(&mut tail, &mut drawn, *who, mine.then_some(brought));
        }
        if discard.riichi {
            tail.push(line(Event::Reach { actor: *who }, seating));
        }
        // Its own discards are replayed as tiles drawn and let go, which is
        // what the extra draw above supplied, bar a declaring turn told as
        // it was played; the other three keep the mark that was entered for
        // them.
        let tsumogiri = if mine {
            !post && brought == discard.tile
        } else {
            discard.drawn && !post
        };
        tail.push(line(
            Event::Dahai {
                actor: *who,
                tile: discard.tile,
                drawn: tsumogiri,
            },
            seating,
        ));
        if discard.riichi && !is_pending(hand, *who, discard) {
            tail.push(line(Event::ReachAccepted { actor: *who }, seating));
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
            )?;
            let (actor, target, tile) = (claim.claimer, claim.target, claim.tile);
            let event = match meld.kind {
                MeldKind::Chii => Event::Chi {
                    actor,
                    target,
                    tile,
                    consumed,
                },
                MeldKind::ClaimedKan => Event::Daiminkan {
                    actor,
                    target,
                    tile,
                    consumed,
                },
                _ => Event::Pon {
                    actor,
                    target,
                    tile,
                    consumed,
                },
            };
            tail.push(line(event, seating));
            // An open quad's replacement is not drawn here. The caller's
            // next discard does not come out of the hand (`discards_after`),
            // so the draw told ahead of it is the replacement; a caller that
            // has not discarded since holds it as the tile drawn now, told
            // at the end. Drawing it here as well gave the caller a second
            // tile, one Mortal refuses when it is the caller's own.
            if meld.kind == MeldKind::ClaimedKan {
                if let Some(indicator) = turning.pop() {
                    tail.push(line(Event::Dora { indicator }, seating));
                }
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
        tail.push(own_quad(owner, kind, tile, seating));
        // A quad still open to robbery has not turned its indicator, and
        // its replacement has not been taken; both wait on the answer.
        if robbed.is_some_and(|robbable| robbable == tile) {
            continue;
        }
        if let Some(indicator) = turning.pop() {
            tail.push(line(Event::Dora { indicator }, seating));
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
    out.push(line(
        Event::StartGame {
            names: std::array::from_fn(|player| format!("player {player}")),
        },
        seating,
    ));
    // The deal is written here: the engine's own start_kyoku knows every
    // hand, and this one shows three of them face down. The dealer is
    // whichever player sits East.
    out.push(format!(
        "{{\"type\":\"start_kyoku\",\"bakaze\":\"{}\",\"kyoku\":{},\"honba\":{},\"kyotaku\":{},\"oya\":{},\"dora_marker\":\"{}\",\"scores\":[{},{},{},{}],\"tehais\":{}}}",
        name(hand.round.tile()),
        hand.kyoku,
        hand.counters,
        kyotaku,
        seating[Wind::East.index()],
        name(first),
        points[0], points[1], points[2], points[3],
        faces,
    ));
    for _ in 0..filler {
        out.push(tsumo(elsewhere, None, seating));
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
        told_as(hand, seat, [0, 1, 2, 3])
    }

    /// The same with the players numbered as `seating` seats them, which is
    /// how a game tells Mortal a hand after the first deal has moved on.
    fn told_as(hand: &Hand, seat: Wind, seating: [usize; 4]) -> MortalState {
        let mut state = MortalState::new(seating[seat.index()] as u8);
        let start = riichi_core::mjai::Event::StartGame {
            names: std::array::from_fn(|player| format!("player {player}")),
        };
        for json in std::iter::once(start.to_json(seating))
            .chain(hand.log.iter().map(|event| event.to_json(seating)))
        {
            if let Ok(event) = serde_json::from_str::<MortalEvent>(&json) {
                let _ = state.update(&event);
            }
        }
        state
    }

    /// What a hand says by itself, which is all a table typed in by hand
    /// carries: its hand number's seating, and no quad's turn.
    fn history(hand: &Hand) -> History {
        History::of_kyoku(hand.kyoku)
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

    /// The planes of thirty-four on which two observations differ.
    fn differing(want: &[f32], got: &[f32]) -> Vec<usize> {
        assert_eq!(want.len(), got.len());
        (0..want.len() / 34)
            .filter(|plane| {
                (0..34).any(|at| (want[plane * 34 + at] - got[plane * 34 + at]).abs() > 1e-6)
            })
            .collect()
    }

    /// How many draws a replay tells.
    fn draws_told(lines: &[String]) -> usize {
        lines
            .iter()
            .filter(|line| line.starts_with("{\"type\":\"tsumo\""))
            .count()
    }

    /// How many draws the hand really made, the dealer's fourteenth tile
    /// and every replacement included.
    fn draws_made(hand: &Hand) -> usize {
        hand.log
            .iter()
            .filter(|event| matches!(event, riichi_core::mjai::Event::Tsumo { .. }))
            .count()
    }

    /// Plays the hand `seed` deals, claiming every open quad a seat is
    /// offered and otherwise playing as the club player does, and shows
    /// `look` every decision on the way: the seat to act, and the table
    /// while a tile is on offer. Taking every quad makes them common enough
    /// to test, where the club player alone seldom calls one.
    fn claiming_every_quad(seed: u64, mut look: impl FnMut(&Hand)) {
        use riichi_core::game::Call;
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut hand = table.deal(&mut rng);
        let mut bot = Bot::with_style(seed, Style::club());
        for _ in 0..400 {
            match hand.phase {
                Phase::Over => return,
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    look(&hand);
                    let action = bot.act(&hand);
                    hand.act(action).unwrap();
                }
                Phase::CallWindow => {
                    look(&hand);
                    let answers: Vec<_> = hand
                        .legal_calls()
                        .into_iter()
                        .map(|(who, calls)| {
                            let call = if calls.contains(&Call::Kan) {
                                Call::Kan
                            } else {
                                bot.call(&hand, who, &calls)
                            };
                            (who, call)
                        })
                        .collect();
                    hand.resolve_calls(&answers).unwrap();
                }
            }
        }
        panic!("seed {seed}: the hand did not end");
    }

    /// Whether any seat holds a set of this kind.
    fn stands(hand: &Hand, kind: MeldKind) -> bool {
        hand.players
            .iter()
            .any(|player| player.melds.iter().any(|meld| meld.kind == kind))
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
                    let state = state_for(&position, seat, &history(&position))
                        .unwrap_or_else(|| panic!("seed {seed} step {steps} seat {seat:?}"));
                    let wrong = differing(&want, &planes(&state));
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
                    let got = planes(
                        &state_for(&hidden(hand.clone(), seat), seat, &history(&hand)).unwrap(),
                    );
                    let wrong = differing(&want, &got);
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

    /// An open quad draws one replacement, which a position does not record:
    /// the caller's next discard sends it out again, or the caller holds it
    /// now. Replayed from every later decision of hands that claim every
    /// open quad offered, a position must tell exactly the draws the hand
    /// made and read exactly as Mortal read the hand itself, the caller's
    /// own seat included. Only open quads stand in the positions compared:
    /// theirs is the one moment a position records, where a concealed or
    /// added quad is given one. A seat in riichi is excused its two frozen
    /// planes.
    #[test]
    fn an_open_quad_replays_into_the_observation_it_came_from() {
        let (mut compared, mut callers) = (0, 0);
        let (mut refused, mut miscounted, mut misread) = (Vec::new(), Vec::new(), Vec::new());
        for seed in 1..=60u64 {
            claiming_every_quad(seed, |hand| {
                if !stands(hand, MeldKind::ClaimedKan)
                    || stands(hand, MeldKind::ConcealedKan)
                    || stands(hand, MeldKind::ExtendedKan)
                {
                    return;
                }
                for seat in Wind::ALL {
                    if hand.phase == Phase::Act && hand.turn != seat {
                        continue;
                    }
                    let at = format!("seed {seed} {seat:?} {:?}", hand.phase);
                    compared += 1;
                    callers += usize::from(
                        hand.players[seat.index()]
                            .melds
                            .iter()
                            .any(|meld| meld.kind == MeldKind::ClaimedKan),
                    );
                    let position = hidden(hand.clone(), seat);
                    let lines =
                        events(&position, seat, &history(&position)).expect("the claims pair up");
                    if draws_told(&lines) != draws_made(hand) {
                        miscounted.push(format!(
                            "{at}: {} draws told, {} made",
                            draws_told(&lines),
                            draws_made(hand)
                        ));
                    }
                    let Some(state) = state_for(&position, seat, &history(&position)) else {
                        refused.push(at);
                        continue;
                    };
                    let wrong = differing(&planes(&told(hand, seat)), &planes(&state));
                    let excused = hand.players[seat.index()].has_riichi()
                        && wrong
                            .iter()
                            .all(|plane| FROZEN_SHANTEN_DISCARDS.contains(plane));
                    if !wrong.is_empty() && !excused {
                        misread.push(format!("{at}: planes {:?}", &wrong[..wrong.len().min(8)]));
                    }
                }
            });
        }
        println!(
            "open quads: {compared} replays ({callers} by the caller), {} refused, {} miscounted, {} misread",
            refused.len(),
            miscounted.len(),
            misread.len()
        );
        assert!(
            refused.is_empty() && miscounted.is_empty() && misread.is_empty(),
            "refused {:?}\nmiscounted {:?}\nmisread {:?}",
            &refused[..refused.len().min(4)],
            &miscounted[..miscounted.len().min(4)],
            &misread[..misread.len().min(4)],
        );
        assert!(compared > 200, "only {compared} replays after an open quad");
        assert!(callers > 50, "only {callers} of them by the caller");
    }

    /// The first decision after an open quad is the caller's, holding its
    /// replacement: the replay tells the quad, turns its indicator and
    /// draws that one tile, which only the caller is told.
    #[test]
    fn the_caller_of_an_open_quad_draws_its_replacement_once() {
        let mut found = false;
        for seed in 1..=40u64 {
            claiming_every_quad(seed, |hand| {
                let n = hand.log.len();
                let fresh = hand.phase == Phase::Act
                    && n >= 3
                    && matches!(hand.log[n - 3], riichi_core::mjai::Event::Daiminkan { .. });
                if found || !fresh {
                    return;
                }
                found = true;
                let caller = hand.turn;
                let drawn = hand.drawn.expect("the replacement is in hand");
                for seat in Wind::ALL {
                    let lines = events(&hidden(hand.clone(), seat), seat, &history(hand)).unwrap();
                    let quad = lines
                        .iter()
                        .rposition(|line| line.contains("\"type\":\"daiminkan\""))
                        .expect("the quad is told");
                    assert!(lines[quad + 1].contains("\"type\":\"dora\""), "seed {seed}");
                    let pai = if seat == caller {
                        name(drawn)
                    } else {
                        "?".into()
                    };
                    assert_eq!(
                        lines[quad + 2..],
                        [format!(
                            "{{\"type\":\"tsumo\",\"actor\":{},\"pai\":\"{pai}\"}}",
                            caller.index()
                        )],
                        "seed {seed}, told to {seat:?}"
                    );
                }
                let state = state_for(&hidden(hand.clone(), caller), caller, &history(hand))
                    .expect("the caller's replay is read");
                assert_eq!(
                    differing(&planes(&told(hand, caller)), &planes(&state)),
                    Vec::<usize>::new(),
                    "seed {seed}"
                );
            });
            if found {
                return;
            }
        }
        panic!("no open quad was claimed in forty hands");
    }

    /// The quads a seat declared itself, concealed or added to a triplet.
    fn own_quads(hand: &Hand, seat: Wind) -> usize {
        hand.players[seat.index()]
            .melds
            .iter()
            .filter(|meld| matches!(meld.kind, MeldKind::ConcealedKan | MeldKind::ExtendedKan))
            .count()
    }

    /// However many quads stand, of whatever kind and whoever declared
    /// them, a replay tells exactly the draws the hand made. Where a seat
    /// holds two quads of its own, which the replay gives one turn, Mortal
    /// must also read that seat as holding exactly the tiles it holds.
    #[test]
    fn every_replay_tells_the_draws_the_hand_made() {
        let (mut replays, mut quads, mut shared) = (0, 0, 0);
        let mut wrong = Vec::new();
        let mut doubled = std::collections::BTreeSet::new();
        for seed in 1..=200u64 {
            claiming_every_quad(seed, |hand| {
                let any_quad = hand
                    .players
                    .iter()
                    .any(|player| player.melds.iter().any(|meld| meld.kind.is_kan()));
                if Wind::ALL.into_iter().any(|seat| own_quads(hand, seat) >= 2) {
                    doubled.insert(seed);
                }
                for seat in Wind::ALL {
                    if hand.phase == Phase::Act && hand.turn != seat {
                        continue;
                    }
                    let at = format!("seed {seed} {seat:?} {:?}", hand.phase);
                    replays += 1;
                    quads += usize::from(any_quad);
                    let position = hidden(hand.clone(), seat);
                    let lines =
                        events(&position, seat, &history(&position)).expect("the claims pair up");
                    if draws_told(&lines) != draws_made(hand) {
                        wrong.push(format!(
                            "{at}: {} draws told, {} made",
                            draws_told(&lines),
                            draws_made(hand)
                        ));
                    }
                    if own_quads(hand, seat) < 2 {
                        continue;
                    }
                    shared += 1;
                    match state_for(&position, seat, &history(&position)) {
                        None => wrong.push(format!("{at}: refused")),
                        Some(state)
                            if state.tehai() != *hand.players[seat.index()].hand.counts() =>
                        {
                            wrong.push(format!("{at}: holds other tiles"))
                        }
                        Some(_) => {}
                    }
                }
            });
        }
        println!(
            "draws: {replays} replays, {quads} with a quad standing, {shared} by a seat holding two of its own (seeds {doubled:?}); {} wrong",
            wrong.len()
        );
        assert!(
            wrong.is_empty(),
            "{} wrong: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(6)]
        );
        assert!(quads > 500, "only {quads} replays with a quad standing");
        // Seeds 163 and 195 give a seat two quads of its own.
        assert!(shared > 0, "no seat held two quads of its own");
    }

    /// South has declared two quads of its own, of 7p and 7s, has discarded
    /// since, and is to act on a fresh draw. Nothing in a position says when
    /// either quad was declared, so the replay gives both to South's latest
    /// turn that began with a draw. That turn's draw made the first quad,
    /// the first one's replacement made the second, and the second one's
    /// replacement is the tile the turn let go: two quads, two replacements,
    /// and Mortal reads South as holding exactly what it holds.
    #[test]
    fn two_quads_on_one_turn_draw_one_replacement_each() {
        use riichi_core::hand::{Meld, TileSet};
        use riichi_core::wall::Wall;
        let tile = |text: &str| -> Tile { text.parse().unwrap() };
        let mut hand = Hand::deal(&mut Rng::from_seed(1), Wind::East, 1, 0, 0, [25000; 4]);
        hand.log.clear();
        // The seats discard in turn from East, seventeen tiles so far, none
        // of them claimed; South's are orders 1, 5, 9 and 13.
        let rivers: [&[&str]; 4] = [
            &["1z", "2z", "3z", "4z", "9m"],
            &["1m", "9p", "1s", "9s"],
            &["6z", "7z", "1p", "2p"],
            &["8m", "8p", "8s", "3p"],
        ];
        for (index, river) in rivers.iter().enumerate() {
            let player = &mut hand.players[index];
            player.hand = TileSet::new();
            player.discards = river
                .iter()
                .enumerate()
                .map(|(turn, text)| Discard {
                    tile: tile(text),
                    order: (4 * turn + index) as u32,
                    drawn: true,
                    riichi: false,
                    claimed: false,
                })
                .collect();
        }
        let south = &mut hand.players[Wind::South.index()];
        south.melds = vec![
            Meld::concealed_kan(tile("7p")),
            Meld::concealed_kan(tile("7s")),
        ];
        south.hand = "234m456p55z".parse().unwrap();
        hand.players[Wind::North.index()].hand = "123m567m123p456s1z".parse().unwrap();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = Some(tile("5z"));
        hand.discards_made = 17;
        hand.first_turns_unbroken = false;
        // Twenty draws: East's five, South's five with the one it holds,
        // four each for West and North, and the two replacements. Each quad
        // also moved a live tile into the dead wall.
        hand.wall = Wall::for_analysis(50, &[tile("3m"), tile("6m"), tile("2s")], 2).unwrap();

        // Told to South, whose quads they are, and to North, who sees only
        // that South drew.
        for seat in [Wind::South, Wind::North] {
            let position = hidden(hand.clone(), seat);
            let lines = events(&position, seat, &history(&position)).expect("the position is told");
            assert_eq!(draws_told(&lines), 20, "told to {seat:?}");
            let state = state_for(&position, seat, &history(&position))
                .unwrap_or_else(|| panic!("the replay told to {seat:?} is refused"));
            assert_eq!(
                state.tehai(),
                *hand.players[seat.index()].hand.counts(),
                "told to {seat:?}"
            );
        }
        let lines = events(&hand, Wind::South, &history(&hand)).unwrap();
        let first = lines
            .iter()
            .position(|line| line.contains("\"type\":\"ankan\""))
            .expect("the quads are told");
        assert_eq!(
            lines[first - 1..first + 7],
            [
                "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"7p\"}",
                "{\"type\":\"ankan\",\"actor\":1,\"consumed\":[\"7p\",\"7p\",\"7p\",\"7p\"]}",
                "{\"type\":\"dora\",\"dora_marker\":\"6m\"}",
                "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"7s\"}",
                "{\"type\":\"ankan\",\"actor\":1,\"consumed\":[\"7s\",\"7s\",\"7s\",\"7s\"]}",
                "{\"type\":\"dora\",\"dora_marker\":\"2s\"}",
                "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"9s\"}",
                "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"9s\",\"tsumogiri\":true}",
            ]
        );
    }

    /// Plays the hand `seed` deals, declaring every quad of its own a seat
    /// may, claiming every quad and triplet offered, and otherwise playing
    /// as the club player does, and shows `look` every decision on the way.
    /// The triplets claimed are what added quads are made from, and the
    /// club player alone seldom declares a quad of either kind.
    fn declaring_every_quad(seed: u64, mut look: impl FnMut(&Hand)) {
        use riichi_core::game::Call;
        let table = Table::new();
        let mut rng = Rng::from_seed(seed);
        let mut hand = table.deal(&mut rng);
        let mut bot = Bot::with_style(seed, Style::club());
        for _ in 0..400 {
            match hand.phase {
                Phase::Over => return,
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    look(&hand);
                    let quad = hand.legal_actions().into_iter().find(|action| {
                        matches!(action, Action::ConcealedKan(_) | Action::ExtendedKan(_))
                    });
                    let action = quad.unwrap_or_else(|| bot.act(&hand));
                    hand.act(action).unwrap();
                }
                Phase::CallWindow => {
                    look(&hand);
                    let answers: Vec<_> = hand
                        .legal_calls()
                        .into_iter()
                        .map(|(who, calls)| {
                            let call = [Call::Kan, Call::Pon]
                                .into_iter()
                                .find(|call| calls.contains(call))
                                .unwrap_or_else(|| bot.call(&hand, who, &calls));
                            (who, call)
                        })
                        .collect();
                    hand.resolve_calls(&answers).unwrap();
                }
            }
        }
        panic!("seed {seed}: the hand did not end");
    }

    /// A quad a seat declared itself has no moment of its own in a table
    /// typed in by hand, and the replay gives it one, which reads wrong
    /// where it is not the moment it was made: Mortal marks a quad on the
    /// discard of the turn that made it, and its indicator makes dora of
    /// what is thrown after it. A position that records each quad's turn,
    /// as a game played here can (`History::from_log`), reads exactly as
    /// the hand did, the declarer's own seat included, wherever the guess
    /// read wrong. The guess is counted wrong only beyond the two planes a
    /// seat in riichi reads wrong from a guess of another kind, which the
    /// next test is about; the recorded hand reads those right too.
    #[test]
    fn a_quad_told_on_the_turn_it_was_made_reads_as_the_hand_did() {
        let (mut compared, mut guessed_wrong, mut declarers) = (0, 0, 0);
        let mut misread = Vec::new();
        for seed in 1..=120u64 {
            declaring_every_quad(seed, |hand| {
                if !Wind::ALL.into_iter().any(|seat| own_quads(hand, seat) > 0) {
                    return;
                }
                for seat in Wind::ALL {
                    if hand.phase == Phase::Act && hand.turn != seat {
                        continue;
                    }
                    let position = hidden(hand.clone(), seat);
                    let want = planes(&told(hand, seat));
                    let excused = |wrong: &[usize]| {
                        hand.players[seat.index()].has_riichi()
                            && wrong
                                .iter()
                                .all(|plane| FROZEN_SHANTEN_DISCARDS.contains(plane))
                    };
                    let recorded = History::from_log(hand, seat);
                    let at = format!("seed {seed} {seat:?} {:?}", hand.phase);
                    let Some(state) = state_for(&position, seat, &recorded) else {
                        misread.push(format!("{at}: refused"));
                        continue;
                    };
                    let wrong = differing(&want, &planes(&state));
                    if !wrong.is_empty() {
                        misread.push(format!("{at}: planes {:?}", &wrong[..wrong.len().min(8)]));
                    }
                    compared += 1;
                    declarers += usize::from(own_quads(hand, seat) > 0);
                    let guessed = state_for(&position, seat, &history(hand))
                        .map(|state| differing(&want, &planes(&state)))
                        .unwrap_or_default();
                    guessed_wrong += usize::from(!guessed.is_empty() && !excused(&guessed));
                }
            });
        }
        println!(
            "own quads: {compared} replays ({declarers} by a declarer), {guessed_wrong} read wrong from a guessed turn, {} misread from the recorded one",
            misread.len()
        );
        assert!(
            misread.is_empty(),
            "{} misread: {:?}",
            misread.len(),
            &misread[..misread.len().min(6)]
        );
        assert!(
            compared > 300,
            "only {compared} replays with a quad of a seat's own"
        );
        assert!(declarers > 50, "only {declarers} of them by a declarer");
        assert!(
            guessed_wrong > 0,
            "no guessed turn read wrong, so nothing was tested"
        );
    }

    /// For as long as a riichi stands, Mortal shows the declarer which of
    /// its discards kept its distance from waiting and which brought it
    /// closer on the turn it declared, measured against the hand it held
    /// before that turn's draw. Told as a tile drawn and let go, as a table
    /// typed in by hand tells it, the declaring turn is measured against the
    /// hand held after it, and those two planes read wrong wherever the draw
    /// is what made the hand ready. A position that records the declaring
    /// turn's draw, as a game played here can, reads exactly as the hand did.
    #[test]
    fn a_declaring_turn_told_as_it_was_played_reads_as_the_hand_did() {
        let (mut compared, mut guessed_wrong) = (0, 0);
        let mut misread = Vec::new();
        for seed in 1..80u64 {
            for steps in [28usize, 36, 44, 52, 60] {
                let hand = part_way(seed, steps);
                if matches!(hand.phase, Phase::Over) {
                    continue;
                }
                for seat in Wind::ALL {
                    let deciding = hand.phase != Phase::Act || hand.turn == seat;
                    if !deciding || !hand.players[seat.index()].has_riichi() {
                        continue;
                    }
                    let at = format!("seed {seed} step {steps} seat {seat:?}");
                    let position = hidden(hand.clone(), seat);
                    let want = planes(&told(&hand, seat));
                    let recorded = History::from_log(&hand, seat);
                    match state_for(&position, seat, &recorded) {
                        None => misread.push(format!("{at}: refused")),
                        Some(state) => {
                            let wrong = differing(&want, &planes(&state));
                            if !wrong.is_empty() {
                                misread.push(format!("{at}: planes {wrong:?}"));
                            }
                        }
                    }
                    compared += 1;
                    let guessed = state_for(&position, seat, &history(&hand)).unwrap();
                    guessed_wrong += usize::from(!differing(&want, &planes(&guessed)).is_empty());
                }
            }
        }
        println!(
            "declarers: {compared} replays, {guessed_wrong} read wrong from a declaring turn told as a tile let go, {} misread as played",
            misread.len()
        );
        assert!(misread.is_empty(), "{misread:?}");
        assert!(compared > 30, "only {compared} replays of a seat in riichi");
        assert!(
            guessed_wrong > 0,
            "no declaring turn read wrong, so nothing was tested"
        );
    }

    /// libriichi numbers the players from the game's first dealer and,
    /// where two hold the same points, ranks the lower number first. In the
    /// third hand of a round the dealer is player 2, so with everyone level
    /// East ranks third, behind West and North, players 0 and 1. A position
    /// that numbered the players from its own dealer read East as first.
    #[test]
    fn level_scores_rank_by_the_players_numbers_from_the_first_dealer() {
        let mut hand = Hand::deal(&mut Rng::from_seed(7), Wind::East, 3, 0, 0, [25000; 4]);
        let mut bot = Bot::with_style(7, Style::club());
        for _ in 0..9 {
            if hand.phase == Phase::Draw {
                let _ = hand.draw();
            }
            let action = bot.act(&hand);
            hand.act(action).unwrap();
            while hand.phase == Phase::CallWindow {
                hand.resolve_calls(&[]).unwrap();
            }
        }
        // As `Table::player_at` seats them in the third hand of a round.
        let seating = [2, 3, 0, 1];
        assert_eq!(History::of_kyoku(3).seating, seating);
        let rank_planes = 15..19;
        let mut renumbered = 0;
        for seat in Wind::ALL {
            if hand.phase == Phase::Act && hand.turn != seat {
                continue;
            }
            let want = planes(&told_as(&hand, seat, seating));
            let position = hidden(hand.clone(), seat);
            let state = state_for(&position, seat, &History::of_kyoku(3)).unwrap();
            assert_eq!(
                differing(&want, &planes(&state)),
                Vec::<usize>::new(),
                "{seat:?}"
            );
            // Numbered from the hand's own dealer, the rank reads otherwise.
            let by_seat = History {
                seating: [0, 1, 2, 3],
                quads: Default::default(),
                riichi_draw: None,
            };
            let wrong = differing(
                &want,
                &planes(&state_for(&position, seat, &by_seat).unwrap()),
            );
            assert!(
                wrong.iter().all(|plane| rank_planes.contains(plane)),
                "{seat:?}: {wrong:?}"
            );
            renumbered += usize::from(!wrong.is_empty());
        }
        assert!(renumbered > 0, "no seat read its rank differently");
    }

    /// A position that says when a seat's quad was declared must name a
    /// turn that seat drew on. A discard that followed a call had no draw
    /// before it, and a discard of another seat's is no turn of this one's,
    /// so the replay refuses both rather than tell a hand nobody played.
    #[test]
    fn a_quad_is_told_only_on_a_turn_its_seat_drew_on() {
        use riichi_core::hand::{ClaimedFrom, Meld, TileSet};
        use riichi_core::wall::Wall;
        let tile = |text: &str| -> Tile { text.parse().unwrap() };
        let mut hand = Hand::deal(&mut Rng::from_seed(1), Wind::East, 1, 0, 0, [25000; 4]);
        hand.log.clear();
        // East's 9m at order 4 is ponned by South, which discards the 1m
        // at order 5 straight from its hand; South's 1s at order 1 and its
        // 9s at order 9 began with draws. South has since declared a
        // concealed quad of 7p and now holds a fresh draw.
        let rivers: [&[(&str, bool)]; 4] = [
            &[("1z", false), ("9m", true), ("3z", false)],
            &[("1s", false), ("1m", false), ("9s", false)],
            &[("6z", false), ("2p", false)],
            &[("8m", false), ("3p", false)],
        ];
        let orders: [&[u32]; 4] = [&[0, 4, 8], &[1, 5, 9], &[2, 6], &[3, 7]];
        for (index, river) in rivers.iter().enumerate() {
            let player = &mut hand.players[index];
            player.hand = TileSet::new();
            player.discards = river
                .iter()
                .zip(orders[index])
                .map(|((text, claimed), order)| Discard {
                    tile: tile(text),
                    order: *order,
                    drawn: false,
                    riichi: false,
                    claimed: *claimed,
                })
                .collect();
        }
        let south = &mut hand.players[Wind::South.index()];
        south.melds = vec![
            Meld::pon(tile("9m"), ClaimedFrom::Left),
            Meld::concealed_kan(tile("7p")),
        ];
        south.hand = "234m456s55z".parse().unwrap();
        hand.players[Wind::North.index()].hand = "123m567m123p456s1z".parse().unwrap();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = Some(tile("5z"));
        hand.discards_made = 10;
        hand.first_turns_unbroken = false;
        hand.wall = Wall::for_analysis(57, &[tile("3m"), tile("6m")], 1).unwrap();

        let turn_of_quad = |order: u32| History {
            seating: [0, 1, 2, 3],
            quads: [vec![], vec![None, Some(order)], vec![], vec![]],
            riichi_draw: None,
        };
        for seat in [Wind::South, Wind::North] {
            let position = hidden(hand.clone(), seat);
            for drawn_turn in [1, 9] {
                assert!(
                    state_for(&position, seat, &turn_of_quad(drawn_turn)).is_some(),
                    "a quad on South's turn {drawn_turn}, told to {seat:?}"
                );
            }
            // Order 5 followed South's pon, and order 6 is West's.
            for not_drawn in [5, 6] {
                assert!(
                    state_for(&position, seat, &turn_of_quad(not_drawn)).is_none(),
                    "a quad on order {not_drawn}, told to {seat:?}"
                );
            }
        }
    }
}
