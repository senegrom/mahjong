//! How a position is shown to a network, and how its answer is read back.
//!
//! Both halves live here, in the rules engine, because the browser and the
//! training runs must agree on them exactly. A model exported from training
//! and run in the game sees the same numbers in the same order.
//!
//! **The observation** is a stack of planes over the 34 tile kinds. Counts
//! are unary: a plane per threshold, so three copies of a tile light the
//! first three planes of its group. Everything that is not per-tile, such as
//! the wall count or the scores, is broadcast across all 34 positions as its
//! own plane, which keeps the input a single rectangle.
//!
//! **The action space** is flat, so a policy head is one softmax over
//! [`ACTIONS`] entries with a legality mask.
//!
//! Seats are always relative to the player to move: index 0 is the player
//! themselves, 1 the player to their right in turn order, and so on. A model
//! therefore never has to learn four seats separately.

use crate::game::{Action, Call, Hand};
use crate::shanten;
use crate::tile::{Tile, COPIES, KINDS};
use crate::Wind;

/// Planes in an observation. Planes are only ever added at the end, so a
/// network trained on fewer can be widened with zero weights for the rest
/// and play exactly as it did.
pub const PLANES: usize = 97;
/// Positions in a plane: the 34 tile kinds.
pub const POSITIONS: usize = KINDS;
/// Numbers in one observation.
pub const OBSERVATION: usize = PLANES * POSITIONS;

/// Opponents whose hands a network may be asked to guess at.
pub const OPPONENTS: usize = 3;
/// The points one unit of the value head is worth. The reward a network is
/// trained on is the points a hand moved divided by this, plus a bonus for
/// the place the game ended in, so a value of 0.5 means about two thousand
/// points. A search that mixes valued positions with hands that actually
/// ended needs the two in the same units, and this is the conversion.
pub const POINTS_PER_UNIT: f32 = 4000.0;
/// What finishing a game in each place is worth, best place first, in the
/// value head's units. Training adds this to every decision of a game on
/// top of what each hand moved, so it is the network's idea of what a game
/// is for, and a search that plays an imagined world to the end of the
/// game needs the same numbers.
pub const PLACEMENT_VALUE: [f32; 4] = [1.5, 0.5, -0.5, -1.5];
/// How many of the coming draws the oracle is shown: four rounds of the
/// table, so it knows every seat's next four turns.
pub const ORACLE_DRAWS: usize = 16;
/// Planes in the oracle's view, which is what the player cannot see: the
/// three opponents' concealed tiles as unary counts, the next
/// [`ORACLE_DRAWS`] live draws in order, and the hidden dora indicators as
/// unary counts. For a critic consulted only in training.
pub const ORACLE_PLANES: usize = OPPONENTS * COPIES as usize + ORACLE_DRAWS + COPIES as usize;
/// Numbers in one oracle view.
pub const ORACLE: usize = ORACLE_PLANES * POSITIONS;
/// Planes holding the three opponents' concealed tiles as unary counts, in
/// the observation's relative seat order: the first part of the oracle's
/// view, and what a reader of imagined hands is shown.
pub const HIDDEN_HANDS_PLANES: usize = OPPONENTS * COPIES as usize;
/// Numbers in one set of hidden hands.
pub const HIDDEN_HANDS: usize = HIDDEN_HANDS_PLANES * POSITIONS;
/// Numbers in one answer about what the opponents are holding.
pub const HANDS: usize = OPPONENTS * POSITIONS;
/// Roughly how many discards a hand holds before the wall runs out, used to
/// scale the timing planes into about zero to one.
const DISCARD_SPAN: f32 = 70.0;

/// Entries in the flat action space.
pub const ACTIONS: usize = 78;
/// First index of the plain discards.
pub const DISCARD: usize = 0;
/// First index of the discards that declare riichi.
pub const RIICHI_DISCARD: usize = 34;
/// Declare a win on the drawn tile.
pub const TSUMO: usize = 68;
/// Declare a win on the discard.
pub const RON: usize = 69;
/// Decline a claim.
pub const PASS: usize = 70;
/// Claim a sequence whose lowest tile is two below the claimed tile.
pub const CHII_LOW: usize = 71;
/// Claim a sequence whose lowest tile is one below the claimed tile.
pub const CHII_MIDDLE: usize = 72;
/// Claim a sequence that starts at the claimed tile.
pub const CHII_HIGH: usize = 73;
/// Claim a triplet.
pub const PON: usize = 74;
/// Claim a quad.
pub const CLAIMED_KAN: usize = 75;
/// Declare a quad from four tiles in hand.
pub const CONCEALED_KAN: usize = 76;
/// Add the fourth tile to a melded triplet.
pub const EXTENDED_KAN: usize = 77;

/// What the three opponents were actually holding, as a distribution over
/// the 34 kinds for each, in the same relative seat order as the
/// observation: index 0 is the player to `seat`'s right in turn order.
///
/// This is the label for teaching a network to read a table. During
/// self-play the true hands are known, so it costs nothing to collect, and
/// unlike the result of a game it says something about this position in
/// particular.
///
/// It matters because the tiles an opponent still holds are not a random
/// draw from what nobody has seen. They chose what to throw, so what is
/// left is what they wanted to keep, and a player who has been discarding
/// circles is not equally likely to be holding them. That selection is
/// exactly what a network can learn from the discards and what no amount of
/// counting the unseen tiles will tell you.
///
/// Each opponent's row sums to one when they hold anything, and to zero
/// when they hold nothing, which happens only in states nobody acts from.
pub fn opponent_hands(hand: &Hand, seat: Wind, out: &mut [f32]) {
    assert_eq!(out.len(), HANDS, "the answer is three rows of thirty-four");
    out.fill(0.0);
    for offset in 1..=OPPONENTS {
        let other = seat.plus(offset);
        let held = &hand.players[other.index()].hand;
        let total = held.len() as f32;
        if total == 0.0 {
            continue;
        }
        let base = (offset - 1) * POSITIONS;
        for tile in Tile::all() {
            out[base + tile.idx()] = held.count(tile) as f32 / total;
        }
    }
}

/// Writes the observation for one seat into `out`, which must hold
/// [`OBSERVATION`] numbers.
///
/// The discards carry their history, because in riichi *when* a tile was let
/// go says nearly as much as *that* it was: a tile discarded on the first
/// turn reads very differently from the same tile on the twelfth, a tile
/// taken straight from the draw says the hand had no use for it, and a tile
/// that passed after a riichi declaration can never deal into that player.
pub fn observe(hand: &Hand, seat: Wind, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        OBSERVATION,
        "observation buffer is the wrong size"
    );
    out.fill(0.0);
    let mut plane = 0;
    let me = &hand.players[seat.index()];

    // The player's own concealed tiles.
    unary(out, &mut plane, |tile| me.hand.count(tile));

    // Called sets, every seat, starting with this one.
    for offset in 0..4 {
        let other = seat.plus(offset);
        unary(out, &mut plane, |tile| meld_count(hand, other, tile));
    }

    // Discards, every seat, with their history.
    for offset in 0..4 {
        let other = seat.plus(offset);
        let player = &hand.players[other.index()];
        let discards = &player.discards;

        unary(out, &mut plane, |tile| {
            discards.iter().filter(|entry| entry.tile == tile).count() as u8
        });

        // How late the most recent discard of each kind was.
        value(out, &mut plane, |tile| {
            discards
                .iter()
                .filter(|entry| entry.tile == tile)
                .map(|entry| (entry.order + 1) as f32 / DISCARD_SPAN)
                .fold(0.0, f32::max)
        });
        // Taken straight from the draw.
        mark(out, &mut plane, |tile| {
            discards
                .iter()
                .any(|entry| entry.tile == tile && entry.drawn)
        });
        // Let go after this player declared riichi, so it cannot deal in.
        mark(out, &mut plane, |tile| match player.riichi_order {
            Some(declared) => discards
                .iter()
                .any(|entry| entry.tile == tile && entry.order >= declared),
            None => false,
        });
        // Somebody claimed it, so it never sat in the row.
        mark(out, &mut plane, |tile| {
            discards
                .iter()
                .any(|entry| entry.tile == tile && entry.claimed)
        });
        // The declaration tile itself, the loudest signal a player gives.
        mark(out, &mut plane, |tile| {
            discards
                .iter()
                .any(|entry| entry.tile == tile && entry.riichi)
        });
    }

    // The dora indicators, and how many copies of each kind nobody can see.
    unary(out, &mut plane, |tile| {
        hand.wall
            .dora_indicators()
            .iter()
            .filter(|indicator| **indicator == tile)
            .count() as u8
    });
    let unseen = unseen_counts(hand, seat);
    unary(out, &mut plane, |tile| unseen.count(tile));

    // What this hand is waiting on, and the tile awaiting a claim.
    let waits = me.waits();
    mark(out, &mut plane, |tile| waits.count(tile) > 0);
    let pending = hand.pending_discard.map(|(_, tile)| tile);
    mark(out, &mut plane, |tile| Some(tile) == pending);

    // What is safe against each seat, worked out from the same history.
    for offset in 0..4 {
        let other = seat.plus(offset);
        let safe = hand.safe_against(other);
        mark(out, &mut plane, |tile| safe.count(tile) > 0);
    }

    // Riichi, by relative seat.
    for offset in 0..4 {
        let other = seat.plus(offset);
        let declared = hand.players[other.index()].has_riichi();
        broadcast(out, &mut plane, if declared { 1.0 } else { 0.0 });
    }
    broadcast(out, &mut plane, if me.is_furiten() { 1.0 } else { 0.0 });
    broadcast(out, &mut plane, if me.is_concealed() { 1.0 } else { 0.0 });

    // The winds, one plane each, and the state of the table.
    for offset in 0..4 {
        broadcast(
            out,
            &mut plane,
            if seat.index() == offset { 1.0 } else { 0.0 },
        );
    }
    for offset in 0..4 {
        broadcast(
            out,
            &mut plane,
            if hand.round.index() == offset {
                1.0
            } else {
                0.0
            },
        );
    }
    broadcast(out, &mut plane, hand.wall.remaining() as f32 / DISCARD_SPAN);
    broadcast(out, &mut plane, hand.counters as f32 / 4.0);
    broadcast(out, &mut plane, hand.riichi_sticks as f32 / 4.0);
    for offset in 0..4 {
        let other = seat.plus(offset);
        broadcast(
            out,
            &mut plane,
            hand.players[other.index()].score as f32 / 50_000.0,
        );
    }
    let distance = shanten::shanten(&me.hand, me.melds.len());
    broadcast(out, &mut plane, distance as f32 / 8.0);
    broadcast(out, &mut plane, hand.discards_made as f32 / DISCARD_SPAN);

    // Which hand of the round this is, one plane each. Without it South 1
    // and South 4 look the same, and placement play turns on the
    // difference: how many hands are left to overtake in, or to be
    // overtaken in.
    for number in 1..=4u8 {
        broadcast(
            out,
            &mut plane,
            if hand.kyoku == number { 1.0 } else { 0.0 },
        );
    }

    debug_assert_eq!(plane, PLANES, "the observation must fill every plane");
}

/// Writes what `seat` cannot see into `out`, which must hold [`ORACLE`]
/// numbers: the other three concealed hands, in the same relative seat
/// order as [`observe`], then the coming draws in order, then the hidden
/// dora indicators.
///
/// This is for a critic that is only ever consulted in training. Knowing
/// the hidden tiles makes the return far less of a surprise, so its
/// estimate of a position is far less noisy than one made from what the
/// seat can see, and that estimate is a better baseline for the policy
/// gradient and a quieter target for the public value head. The policy
/// never sees these planes, and nothing at play time asks for them.
pub fn oracle(hand: &Hand, seat: Wind, out: &mut [f32]) {
    assert_eq!(out.len(), ORACLE, "oracle buffer is the wrong size");
    out.fill(0.0);
    hidden_hands(hand, seat, &mut out[..HIDDEN_HANDS]);
    let mut plane = HIDDEN_HANDS_PLANES;
    for (order, tile) in hand.wall.upcoming(ORACLE_DRAWS).iter().enumerate() {
        out[(plane + order) * POSITIONS + tile.idx()] = 1.0;
    }
    plane += ORACLE_DRAWS;
    unary(out, &mut plane, |tile| {
        hand.wall
            .ura_indicators()
            .iter()
            .filter(|indicator| **indicator == tile)
            .count() as u8
    });
    debug_assert_eq!(plane, ORACLE_PLANES, "the oracle must fill every plane");
}

/// Writes the three opponents' concealed tiles into `out`, which must hold
/// [`HIDDEN_HANDS`] numbers, as unary counts in the observation's relative
/// seat order. This is what a reader of hidden hands is shown: the same
/// planes whether the hands are real, in training, or imagined, in a
/// search, so that what it learns of the one it can say of the other.
pub fn hidden_hands(hand: &Hand, seat: Wind, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        HIDDEN_HANDS,
        "hidden hands buffer is the wrong size"
    );
    out.fill(0.0);
    let mut plane = 0;
    for offset in 1..4 {
        let other = &hand.players[seat.plus(offset).index()];
        unary(out, &mut plane, |tile| other.hand.count(tile));
    }
    debug_assert_eq!(plane, HIDDEN_HANDS_PLANES);
}

fn meld_count(hand: &Hand, seat: Wind, tile: Tile) -> u8 {
    hand.players[seat.index()]
        .melds
        .iter()
        .flat_map(|meld| meld.tiles())
        .filter(|member| *member == tile)
        .count() as u8
}

fn unseen_counts(hand: &Hand, seat: Wind) -> crate::TileSet {
    let mut seen = hand.players[seat.index()].visible_to_self();
    for player in &hand.players {
        for discard in &player.discards {
            if !discard.claimed {
                seen.add(discard.tile);
            }
        }
        for meld in &player.melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
    }
    // The player's own tiles were counted twice above; take the smaller of
    // what is seen and a full set, which is what matters here anyway.
    let mut unseen = crate::TileSet::new();
    for tile in Tile::all() {
        let count = seen.count(tile).min(COPIES);
        unseen.add_n(tile, COPIES - count);
    }
    unseen
}

fn unary(out: &mut [f32], plane: &mut usize, count: impl Fn(Tile) -> u8) {
    for threshold in 1..=COPIES {
        let base = *plane * POSITIONS;
        for tile in Tile::all() {
            if count(tile) >= threshold {
                out[base + tile.idx()] = 1.0;
            }
        }
        *plane += 1;
    }
}

fn value(out: &mut [f32], plane: &mut usize, amount: impl Fn(Tile) -> f32) {
    let base = *plane * POSITIONS;
    for tile in Tile::all() {
        out[base + tile.idx()] = amount(tile);
    }
    *plane += 1;
}

fn mark(out: &mut [f32], plane: &mut usize, flag: impl Fn(Tile) -> bool) {
    let base = *plane * POSITIONS;
    for tile in Tile::all() {
        if flag(tile) {
            out[base + tile.idx()] = 1.0;
        }
    }
    *plane += 1;
}

fn broadcast(out: &mut [f32], plane: &mut usize, value: f32) {
    let base = *plane * POSITIONS;
    out[base..base + POSITIONS].fill(value);
    *plane += 1;
}

/// Writes the legal-action mask for the seat to move, or for a seat that has
/// been offered a claim. Returns how many entries are legal.
pub fn legal_mask(hand: &Hand, seat: Wind, out: &mut [bool]) -> usize {
    assert_eq!(out.len(), ACTIONS, "mask buffer is the wrong size");
    out.fill(false);
    let mut count = 0;

    if hand.turn == seat {
        // The engine's own list already leaves out anything the swap-call
        // rule bars, so nothing has to be filtered again here.
        for action in hand.legal_actions() {
            let index = action_index(action);
            if !out[index] {
                out[index] = true;
                count += 1;
            }
        }
    }

    for (other, calls) in hand.legal_calls() {
        if other != seat {
            continue;
        }
        let claimed = hand.pending_discard.map(|(_, tile)| tile);
        for call in calls {
            if let Some(index) = call_index(call, claimed) {
                if !out[index] {
                    out[index] = true;
                    count += 1;
                }
            }
        }
    }
    count
}

fn action_index(action: Action) -> usize {
    match action {
        Action::Discard(tile) => DISCARD + tile.idx(),
        Action::Riichi(tile) => RIICHI_DISCARD + tile.idx(),
        Action::Tsumo => TSUMO,
        Action::ConcealedKan(_) => CONCEALED_KAN,
        Action::ExtendedKan(_) => EXTENDED_KAN,
    }
}

fn call_index(call: Call, claimed: Option<Tile>) -> Option<usize> {
    match call {
        Call::Ron => Some(RON),
        Call::Pon => Some(PON),
        Call::Kan => Some(CLAIMED_KAN),
        Call::Pass => Some(PASS),
        Call::Chii(low) => {
            let claimed = claimed?;
            match claimed.rank().checked_sub(low.rank()) {
                Some(2) => Some(CHII_LOW),
                Some(1) => Some(CHII_MIDDLE),
                Some(0) => Some(CHII_HIGH),
                _ => None,
            }
        }
    }
}

/// Turns an action index back into something the engine will accept.
///
/// The quad entries do not name a tile, so the first legal quad of that kind
/// is taken. Holding two different quads at once is rare enough that the
/// simpler action space is worth it.
pub fn decode_action(hand: &Hand, index: usize) -> Option<Action> {
    let legal = hand.legal_actions();
    match index {
        DISCARD..=33 => {
            let tile = Tile::try_new((index - DISCARD) as u8)?;
            legal
                .contains(&Action::Discard(tile))
                .then_some(Action::Discard(tile))
        }
        RIICHI_DISCARD..=67 => {
            let tile = Tile::try_new((index - RIICHI_DISCARD) as u8)?;
            legal
                .contains(&Action::Riichi(tile))
                .then_some(Action::Riichi(tile))
        }
        TSUMO => legal.contains(&Action::Tsumo).then_some(Action::Tsumo),
        CONCEALED_KAN => legal
            .iter()
            .find(|action| matches!(action, Action::ConcealedKan(_)))
            .copied(),
        EXTENDED_KAN => legal
            .iter()
            .find(|action| matches!(action, Action::ExtendedKan(_)))
            .copied(),
        _ => None,
    }
}

/// Turns an action index back into a claim on the tile awaiting one.
pub fn decode_call(hand: &Hand, seat: Wind, index: usize) -> Option<Call> {
    let claimed = hand.pending_discard.map(|(_, tile)| tile)?;
    let offered = hand
        .legal_calls()
        .into_iter()
        .find(|(other, _)| *other == seat)
        .map(|(_, calls)| calls)?;
    let wanted = match index {
        RON => Call::Ron,
        PON => Call::Pon,
        CLAIMED_KAN => Call::Kan,
        PASS => Call::Pass,
        CHII_LOW | CHII_MIDDLE | CHII_HIGH => {
            let back = (CHII_HIGH - index) as u8;
            let rank = claimed.rank().checked_sub(back)?;
            if !(1..=7).contains(&rank) {
                return None;
            }
            Call::Chii(Tile::numbered(claimed.suit(), rank))
        }
        _ => return None,
    };
    offered.contains(&wanted).then_some(wanted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Phase;
    use crate::rng::Rng;

    fn fresh() -> Hand {
        Hand::deal(
            &mut Rng::from_seed(20260902),
            Wind::East,
            1,
            0,
            0,
            [25000; 4],
        )
    }

    #[test]
    fn an_observation_is_the_size_it_claims() {
        let hand = fresh();
        let mut out = vec![0.0; OBSERVATION];
        observe(&hand, Wind::East, &mut out);
        assert_eq!(out.len(), PLANES * POSITIONS);
        assert!(out.iter().all(|value| value.is_finite()));
        // The dealer holds fourteen tiles, so the first plane has at least
        // as many distinct kinds as a hand can hold twice over.
        let first_plane: f32 = out[..POSITIONS].iter().sum();
        assert!((7.0..=14.0).contains(&first_plane));
    }

    /// The oracle shows what the seat cannot see and nothing of what it
    /// can: each opponent's concealed tiles in the observation's relative
    /// order, then the draws to come in order, then the hidden indicators.
    #[test]
    fn the_oracle_shows_the_hidden_tiles() {
        let hand = fresh();
        let seat = Wind::South;
        let mut out = vec![0.0; ORACLE];
        oracle(&hand, seat, &mut out);

        for offset in 1..4 {
            let other = &hand.players[seat.plus(offset).index()];
            let first = (offset - 1) * COPIES as usize;
            for tile in Tile::all() {
                let lit = (first..first + COPIES as usize)
                    .filter(|plane| out[plane * POSITIONS + tile.idx()] == 1.0)
                    .count() as u8;
                assert_eq!(lit, other.hand.count(tile), "opponent {offset}, {tile:?}");
            }
        }
        let held: f32 = out[..OPPONENTS * COPIES as usize * POSITIONS].iter().sum();
        assert_eq!(held, 40.0, "thirteen tiles twice and the dealer's fourteen");

        let coming = hand.wall.upcoming(ORACLE_DRAWS);
        assert_eq!(coming.len(), ORACLE_DRAWS);
        for (order, tile) in coming.iter().enumerate() {
            let plane = OPPONENTS * COPIES as usize + order;
            let row = &out[plane * POSITIONS..(plane + 1) * POSITIONS];
            assert_eq!(row.iter().sum::<f32>(), 1.0, "one tile per draw");
            assert_eq!(row[tile.idx()], 1.0, "draw {order} is {tile:?}");
        }

        let indicators = OPPONENTS * COPIES as usize + ORACLE_DRAWS;
        let ura = hand.wall.ura_indicators();
        let row = &out[indicators * POSITIONS..(indicators + 1) * POSITIONS];
        assert_eq!(row[ura[0].idx()], 1.0, "the hidden indicator is shown");
        assert_eq!(row.iter().sum::<f32>(), 1.0);
    }

    /// The hidden hands on their own are the first part of the oracle's
    /// view, plane for plane, so a reader shown either sees the same thing.
    #[test]
    fn the_hidden_hands_are_the_front_of_the_oracle() {
        let hand = fresh();
        let mut whole = vec![0.0; ORACLE];
        oracle(&hand, Wind::North, &mut whole);
        let mut part = vec![0.0; HIDDEN_HANDS];
        hidden_hands(&hand, Wind::North, &mut part);
        assert_eq!(part, whole[..HIDDEN_HANDS]);
        assert_eq!(part.iter().sum::<f32>(), 40.0);
    }

    /// The hand's number has a plane of its own, at the end, so a network
    /// can tell the last hand of the game from the first.
    #[test]
    fn the_hand_number_lights_its_own_plane() {
        for number in 1..=4u8 {
            let hand = Hand::deal(
                &mut Rng::from_seed(3),
                Wind::South,
                number,
                0,
                0,
                [30_000; 4],
            );
            let mut out = vec![0.0; OBSERVATION];
            observe(&hand, Wind::West, &mut out);
            for candidate in 1..=4u8 {
                let plane = PLANES - 4 + (candidate as usize - 1);
                let row = &out[plane * POSITIONS..(plane + 1) * POSITIONS];
                let expected = if candidate == number { 1.0 } else { 0.0 };
                assert!(
                    row.iter().all(|value| *value == expected),
                    "hand {number}: the plane for hand {candidate} reads wrong"
                );
            }
        }
    }

    #[test]
    fn seats_are_relative_to_the_player() {
        let hand = fresh();
        let mut east = vec![0.0; OBSERVATION];
        let mut south = vec![0.0; OBSERVATION];
        observe(&hand, Wind::East, &mut east);
        observe(&hand, Wind::South, &mut south);
        assert_ne!(east, south, "different seats see different tables");
        // Every number stays in a range a network can work with.
        for value in east.iter().chain(south.iter()) {
            assert!((-2.0..=2.0).contains(value), "out of range: {value}");
        }
    }

    /// Two tables holding the same tiles but reached in a different order
    /// must not look the same: when a tile was let go is most of what a
    /// discard row says.
    #[test]
    fn the_discards_carry_their_history() {
        let mut early = fresh();
        let mut late = fresh();
        let tile: Tile = "1z".parse().unwrap();

        // The same tile, discarded by South early in one hand and late in
        // the other, with the rest of the row identical.
        early.players[1].discards.push(crate::game::Discard {
            tile,
            order: 1,
            drawn: true,
            riichi: false,
            claimed: false,
        });
        late.players[1].discards.push(crate::game::Discard {
            tile,
            order: 40,
            drawn: true,
            riichi: false,
            claimed: false,
        });

        let mut first = vec![0.0; OBSERVATION];
        let mut second = vec![0.0; OBSERVATION];
        observe(&early, Wind::East, &mut first);
        observe(&late, Wind::East, &mut second);
        assert_ne!(first, second, "the timing of a discard must show");

        // And a tile taken straight from the draw reads differently from one
        // chosen out of the hand.
        let mut chosen = fresh();
        chosen.players[1].discards.push(crate::game::Discard {
            tile,
            order: 1,
            drawn: false,
            riichi: false,
            claimed: false,
        });
        let mut third = vec![0.0; OBSERVATION];
        observe(&chosen, Wind::East, &mut third);
        assert_ne!(first, third, "a tile off the draw must show");
    }

    /// Everything discarded after a riichi declaration is safe against that
    /// player, and the observation says so (EMA section 3.3.9).
    #[test]
    fn what_is_safe_after_a_riichi_is_encoded() {
        let mut hand = fresh();
        let declared: Tile = "5p".parse().unwrap();
        let passed: Tile = "9m".parse().unwrap();
        hand.players[1].riichi = crate::score::Riichi::Declared;
        hand.players[1].riichi_order = Some(3);
        hand.players[1].discards.push(crate::game::Discard {
            tile: declared,
            order: 3,
            drawn: true,
            riichi: true,
            claimed: false,
        });
        hand.players[2].discards.push(crate::game::Discard {
            tile: passed,
            order: 4,
            drawn: true,
            riichi: false,
            claimed: false,
        });

        let safe = hand.safe_against(Wind::South);
        assert!(safe.count(declared) > 0, "their own discard is safe");
        assert!(
            safe.count(passed) > 0,
            "a tile that passed after riichi is safe"
        );
        assert_eq!(
            safe.count("3s".parse().unwrap()),
            0,
            "an unseen tile is not"
        );
    }

    #[test]
    fn the_mask_matches_what_the_engine_offers() {
        let mut hand = fresh();
        let mut mask = vec![false; ACTIONS];
        let count = legal_mask(&hand, Wind::East, &mut mask);
        assert!(count > 0, "the dealer must have something to do");
        assert_eq!(count, mask.iter().filter(|flag| **flag).count());
        // Everything the mask allows must decode to an action the engine
        // takes, and nothing else may.
        for (index, allowed) in mask.iter().enumerate() {
            if *allowed {
                assert!(
                    decode_action(&hand, index).is_some(),
                    "index {index} was allowed but does not decode"
                );
            }
        }
        // A seat that is not to move is offered nothing.
        let mut other = vec![false; ACTIONS];
        assert_eq!(legal_mask(&hand, Wind::South, &mut other), 0);
        let _ = &mut hand;
    }

    #[test]
    fn calls_encode_and_decode() {
        let mut hand = fresh();
        hand.players[1].hand = "45m123p99s".parse().unwrap();
        hand.players[0].hand = "3m".parse().unwrap();
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.act(Action::Discard("3m".parse().unwrap())).unwrap();

        let mut mask = vec![false; ACTIONS];
        let count = legal_mask(&hand, Wind::South, &mut mask);
        assert!(count > 0, "South can claim the sequence");
        assert!(mask[PASS]);
        let chii = (CHII_LOW..=CHII_HIGH).find(|index| mask[*index]);
        let chii = chii.expect("a sequence claim should be offered");
        let call = decode_call(&hand, Wind::South, chii).expect("it should decode");
        assert!(matches!(call, Call::Chii(_)));
    }

    #[test]
    fn a_whole_game_encodes_at_every_step() {
        let mut hand = fresh();
        let mut observation = vec![0.0; OBSERVATION];
        let mut mask = vec![false; ACTIONS];
        let mut guard = 0;
        while !matches!(hand.phase, Phase::Over) {
            guard += 1;
            assert!(guard < 500);
            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    let seat = hand.turn;
                    observe(&hand, seat, &mut observation);
                    let count = legal_mask(&hand, seat, &mut mask);
                    assert!(count > 0);
                    let index = mask.iter().position(|allowed| *allowed).unwrap();
                    let action = decode_action(&hand, index).expect("the mask offered it");
                    hand.act(action)
                        .expect("the engine accepts a masked action");
                }
                Phase::CallWindow => {
                    let answers: Vec<(Wind, Call)> = hand
                        .legal_calls()
                        .iter()
                        .map(|(seat, _)| {
                            observe(&hand, *seat, &mut observation);
                            legal_mask(&hand, *seat, &mut mask);
                            let call = decode_call(&hand, *seat, PASS).unwrap_or(Call::Pass);
                            (*seat, call)
                        })
                        .collect();
                    hand.resolve_calls(&answers).unwrap();
                }
                Phase::Over => break,
            }
        }
        assert!(hand.outcome.is_some());
    }
}
