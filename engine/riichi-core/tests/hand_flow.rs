//! The yaku that depend on when in the hand a win comes, played from the
//! deal through the engine's own turns.
//!
//! every_yaku.rs sets the circumstance flags of a Situation by hand, so it
//! tests only that the scorer pays them. Here the hand's own state sets
//! them: the first set of turns behind the three blessings and Double
//! Riichi (EMA 2025 sections 4.2.2, 4.2.4 and 4.2.6), the one-shot chance
//! behind Ippatsu (section 4.2.1), and a quad declared with one live tile
//! left, whose replacement is both After a Quad and the last tile of the
//! wall (sections 3.4.1 and 4.2.1).
//!
//! Only the tiles are chosen. Each draw is the engine's own, with the tile
//! it gave swapped for the one the test needs, so every flag is left to the
//! engine to set and clear.

use riichi_core::game::{Action, Call, Discard, Hand, Outcome, Phase};
use riichi_core::hand::{ClaimedFrom, Meld, TileSet};
use riichi_core::rng::Rng;
use riichi_core::score::Riichi;
use riichi_core::tile::Tile;
use riichi_core::yaku::Yaku;
use riichi_core::Wind;

fn tile(text: &str) -> Tile {
    text.parse().expect("a test tile parses")
}

fn tiles(text: &str) -> TileSet {
    text.parse().expect("a test hand parses")
}

/// Hands that can claim nothing and win nothing, with no pair and no two
/// tiles of a sequence, for the seats a test does not play.
const SOUTH_IDLE: &str = "147m258p369s1357z";
const WEST_IDLE: &str = "258m369p147s2467z";
const NORTH_IDLE: &str = "369m147p258s1234z";

/// A hand that waits on the 2 circles alone: All Simples, and nothing else
/// before the circumstances add theirs.
const WAITS_ON_TWO_CIRCLES: &str = "234m567m345p678s2p";

/// A freshly dealt East 1 with the four hands replaced: thirteen tiles each,
/// and East's fourteenth, drawn, on top.
fn dealt(hands: [&str; 4], drawn: &str) -> Hand {
    let mut hand = Hand::deal(
        &mut Rng::from_seed(20260902),
        Wind::East,
        1,
        0,
        0,
        [25_000; 4],
    );
    for (seat, text) in Wind::ALL.into_iter().zip(hands) {
        let held = tiles(text);
        assert_eq!(held.len(), 13, "{seat:?} is dealt thirteen tiles");
        hand.players[seat.index()].hand = held;
    }
    hand.players[0].hand.add(tile(drawn));
    hand.drawn = Some(tile(drawn));
    hand
}

/// Hands the tile just drawn back as `tile` instead.
fn replace_drawn(hand: &mut Hand, wanted: &str) {
    let drawn = hand.drawn.expect("a tile was just drawn");
    let player = &mut hand.players[hand.turn.index()];
    assert!(player.hand.remove(drawn));
    player.hand.add(tile(wanted));
    hand.drawn = Some(tile(wanted));
}

/// The turn player draws, as the engine does, and holds `wanted`.
fn draw(hand: &mut Hand, wanted: &str) {
    assert!(matches!(hand.phase, Phase::Draw), "{:?}", hand.phase);
    hand.draw().expect("the wall has tiles");
    replace_drawn(hand, wanted);
}

/// The turn player discards `thrown` and nobody claims it.
fn discard(hand: &mut Hand, thrown: &str) {
    hand.act(Action::Discard(tile(thrown)))
        .expect("the discard is offered");
    if matches!(hand.phase, Phase::CallWindow) {
        hand.resolve_calls(&[]).expect("passing is always allowed");
    }
}

/// The turn player draws `thrown` and discards it.
fn draw_and_discard(hand: &mut Hand, thrown: &str) {
    draw(hand, thrown);
    discard(hand, thrown);
}

/// The yaku of the hand's one win.
fn yaku_won(hand: &Hand) -> Vec<Yaku> {
    match &hand.outcome {
        Some(Outcome::Win { winners, .. }) => {
            assert_eq!(winners.len(), 1, "{winners:?}");
            winners[0].1.yaku.iter().map(|(yaku, _)| *yaku).collect()
        }
        other => panic!("expected a win, got {other:?}"),
    }
}

/// The turn player declares a win on the tile just drawn.
fn tsumo(hand: &mut Hand) -> Vec<Yaku> {
    let offered = hand.legal_actions();
    assert!(offered.contains(&Action::Tsumo), "{offered:?}");
    hand.act(Action::Tsumo).expect("the win is offered");
    yaku_won(hand)
}

/// `seat` claims the tile on the table for a win.
fn ron(hand: &mut Hand, seat: Wind) -> Vec<Yaku> {
    let offered = hand.legal_calls();
    assert!(
        offered
            .iter()
            .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron)),
        "{offered:?}"
    );
    hand.resolve_calls(&[(seat, Call::Ron)])
        .expect("the win is offered");
    yaku_won(hand)
}

/// East opens with the East wind and West claims it for a triplet, then
/// throws `west_throws`: a claim inside the first set of turns, before
/// North has discarded. North holds `north`.
fn west_claims_first(north: &str, west_throws: &str) -> Hand {
    let mut hand = dealt(
        ["147m258p369s3567z", SOUTH_IDLE, "258m2369p147s114z", north],
        "1z",
    );
    hand.act(Action::Discard(tile("1z"))).unwrap();
    assert_eq!(
        hand.legal_calls(),
        vec![(Wind::West, vec![Call::Pon, Call::Pass])]
    );
    hand.resolve_calls(&[(Wind::West, Call::Pon)]).unwrap();
    assert_eq!(hand.turn, Wind::West);
    hand.act(Action::Discard(tile(west_throws))).unwrap();
    hand
}

/// EMA 2025 section 4.2.6: the dealer's dealt hand, complete, is the
/// Blessing of Heaven.
#[test]
fn a_dealt_hand_that_is_complete_is_the_blessing_of_heaven() {
    let mut hand = dealt(
        ["123m456m789m234p5s", SOUTH_IDLE, WEST_IDLE, NORTH_IDLE],
        "5s",
    );
    assert_eq!(tsumo(&mut hand), [Yaku::Tenhou]);
}

/// Section 4.2.6 again: a concealed quad is not allowed in the Blessing of
/// Heaven, so a dealer who declares one and wins on its replacement, with
/// nobody yet having discarded, wins After a Quad and no more.
#[test]
fn a_dealer_who_declared_a_concealed_quad_has_no_blessing_of_heaven() {
    let mut hand = dealt(
        ["1111m456m789m234p", SOUTH_IDLE, WEST_IDLE, NORTH_IDLE],
        "5s",
    );
    hand.act(Action::ConcealedKan(tile("1m"))).unwrap();
    replace_drawn(&mut hand, "5s");
    let yaku = tsumo(&mut hand);
    assert!(yaku.contains(&Yaku::Rinshan), "{yaku:?}");
    assert!(!yaku.contains(&Yaku::Tenhou), "{yaku:?}");
}

/// Section 4.2.6: a non-dealer who completes the hand on their first draw,
/// with the first set of turns unbroken, has the Blessing of Earth.
#[test]
fn a_first_draw_that_completes_the_hand_is_the_blessing_of_earth() {
    let mut hand = dealt(
        [
            "147m258p369s3567z",
            WAITS_ON_TWO_CIRCLES,
            WEST_IDLE,
            NORTH_IDLE,
        ],
        "9m",
    );
    discard(&mut hand, "9m");
    draw(&mut hand, "2p");
    assert_eq!(tsumo(&mut hand), [Yaku::Chiihou]);
}

/// A claim breaks the first set of turns, so North's first draw after West
/// has claimed a tile is an ordinary self-draw.
#[test]
fn a_claim_ends_the_blessing_of_earth() {
    let mut hand = west_claims_first(WAITS_ON_TWO_CIRCLES, "4z");
    draw(&mut hand, "2p");
    assert!(hand.players[Wind::North.index()].discards.is_empty());
    let yaku = tsumo(&mut hand);
    assert!(yaku.contains(&Yaku::MenzenTsumo), "{yaku:?}");
    assert!(!yaku.contains(&Yaku::Chiihou), "{yaku:?}");
}

/// A quad breaks the first set of turns too: after the dealer's concealed
/// quad, South's first draw is an ordinary self-draw.
#[test]
fn a_quad_ends_the_blessing_of_earth() {
    let mut hand = dealt(
        [
            "1111m258p369s357z",
            WAITS_ON_TWO_CIRCLES,
            WEST_IDLE,
            NORTH_IDLE,
        ],
        "6z",
    );
    hand.act(Action::ConcealedKan(tile("1m"))).unwrap();
    discard(&mut hand, "6z");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(yaku.contains(&Yaku::MenzenTsumo), "{yaku:?}");
    assert!(!yaku.contains(&Yaku::Chiihou), "{yaku:?}");
}

/// Section 4.2.4: a win on a discard before the winner's first turn, with
/// the first set of turns unbroken, is the Blessing of Man, which stands
/// alone.
#[test]
fn a_win_before_the_first_turn_is_the_blessing_of_man() {
    let mut hand = dealt(
        [
            "147m258p369s3567z",
            WAITS_ON_TWO_CIRCLES,
            WEST_IDLE,
            NORTH_IDLE,
        ],
        "2p",
    );
    hand.act(Action::Discard(tile("2p"))).unwrap();
    assert_eq!(ron(&mut hand, Wind::South), [Yaku::Renhou]);
}

/// After a claim the same win, by a player who has not yet discarded, is
/// worth what its tiles are worth.
#[test]
fn a_claim_ends_the_blessing_of_man() {
    let mut hand = west_claims_first(WAITS_ON_TWO_CIRCLES, "2p");
    let yaku = ron(&mut hand, Wind::North);
    assert!(yaku.contains(&Yaku::Tanyao), "{yaku:?}");
    assert!(!yaku.contains(&Yaku::Renhou), "{yaku:?}");
}

/// Section 4.2.2: riichi declared in the player's first turn, with the first
/// set of turns unbroken, is Double Riichi.
#[test]
fn riichi_in_the_first_turn_is_double() {
    let mut hand = dealt(
        [WAITS_ON_TWO_CIRCLES, SOUTH_IDLE, WEST_IDLE, NORTH_IDLE],
        "1z",
    );
    hand.act(Action::Riichi(tile("1z"))).unwrap();
    assert_eq!(hand.players[0].riichi, Riichi::Double);
}

/// Riichi in a first turn that a claim interrupted is a plain riichi.
#[test]
fn riichi_after_a_claim_is_not_double() {
    let mut hand = west_claims_first(WAITS_ON_TWO_CIRCLES, "4z");
    draw(&mut hand, "9m");
    assert!(hand.players[Wind::North.index()].discards.is_empty());
    hand.act(Action::Riichi(tile("9m"))).unwrap();
    assert_eq!(hand.players[Wind::North.index()].riichi, Riichi::Declared);
}

/// Riichi in the player's second turn is a plain riichi.
#[test]
fn riichi_in_the_second_turn_is_not_double() {
    let mut hand = dealt(
        [WAITS_ON_TWO_CIRCLES, SOUTH_IDLE, WEST_IDLE, NORTH_IDLE],
        "1z",
    );
    discard(&mut hand, "1z");
    draw_and_discard(&mut hand, "9s");
    draw_and_discard(&mut hand, "9p");
    draw_and_discard(&mut hand, "9m");
    draw(&mut hand, "1s");
    hand.act(Action::Riichi(tile("1s"))).unwrap();
    assert_eq!(hand.players[0].riichi, Riichi::Declared);
}

/// East declares riichi on the East wind, waiting on the 2 circles.
fn east_declares() -> Hand {
    let mut hand = dealt(
        [WAITS_ON_TWO_CIRCLES, SOUTH_IDLE, WEST_IDLE, NORTH_IDLE],
        "1z",
    );
    hand.act(Action::Riichi(tile("1z"))).unwrap();
    if matches!(hand.phase, Phase::CallWindow) {
        hand.resolve_calls(&[]).unwrap();
    }
    assert!(hand.players[0].ippatsu);
    hand
}

/// Section 4.2.1: a win by the declarer within one go-around of the
/// declaration, with no call in between, is Ippatsu.
#[test]
fn a_self_draw_on_the_declarers_next_draw_is_ippatsu() {
    let mut hand = east_declares();
    draw_and_discard(&mut hand, "9s");
    draw_and_discard(&mut hand, "9p");
    draw_and_discard(&mut hand, "9m");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
}

/// Any claim inside the go-around ends the one-shot chance: here North
/// claims South's white dragon for a triplet.
#[test]
fn a_triplet_claimed_by_another_seat_ends_ippatsu() {
    let mut hand = east_declares();
    hand.players[Wind::North.index()].hand = tiles("369m147p258s1255z");
    draw(&mut hand, "9s");
    hand.act(Action::Discard(tile("5z"))).unwrap();
    assert_eq!(
        hand.legal_calls(),
        vec![(Wind::North, vec![Call::Pon, Call::Pass])]
    );
    hand.resolve_calls(&[(Wind::North, Call::Pon)]).unwrap();
    discard(&mut hand, "1z");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(!yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
    assert!(yaku.contains(&Yaku::DoubleRiichi), "{yaku:?}");
}

/// The same for a sequence: West claims South's 3 characters.
#[test]
fn a_sequence_claimed_by_another_seat_ends_ippatsu() {
    let mut hand = east_declares();
    hand.players[Wind::West.index()].hand = tiles("24m369p147s23467z");
    draw(&mut hand, "3m");
    hand.act(Action::Discard(tile("3m"))).unwrap();
    hand.resolve_calls(&[(Wind::West, Call::Chii(tile("2m")))])
        .unwrap();
    discard(&mut hand, "6z");
    draw_and_discard(&mut hand, "9m");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(!yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
}

/// A quad that stands ends every one-shot chance, another seat's concealed
/// quad included.
#[test]
fn another_seats_concealed_quad_ends_ippatsu() {
    let mut hand = east_declares();
    hand.players[Wind::South.index()].hand = tiles("147m258p369s5555z");
    draw(&mut hand, "9s");
    hand.act(Action::ConcealedKan(tile("5z"))).unwrap();
    replace_drawn(&mut hand, "6z");
    discard(&mut hand, "6z");
    draw_and_discard(&mut hand, "9p");
    draw_and_discard(&mut hand, "9m");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(!yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
}

/// The declarer's own next discard ends it as well.
#[test]
fn the_declarers_next_discard_ends_ippatsu() {
    let mut hand = east_declares();
    draw_and_discard(&mut hand, "9s");
    draw_and_discard(&mut hand, "9p");
    draw_and_discard(&mut hand, "9m");
    draw_and_discard(&mut hand, "1s");
    assert!(!hand.players[0].ippatsu);
    draw_and_discard(&mut hand, "9s");
    draw_and_discard(&mut hand, "9p");
    draw_and_discard(&mut hand, "9m");
    draw(&mut hand, "2p");
    let yaku = tsumo(&mut hand);
    assert!(!yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
}

/// Robbing a Quad combines with Ippatsu (section 4.2.1): a quad that is
/// robbed never stood, so it ends no one-shot chance.
#[test]
fn robbing_a_quad_inside_the_go_around_keeps_ippatsu() {
    let mut hand = dealt(
        [
            "234m567m345p67s22p",
            "147m258p369p1357z",
            WEST_IDLE,
            NORTH_IDLE,
        ],
        "1z",
    );
    // South holds a called triplet of 5 bamboo and the fourth one.
    let south = &mut hand.players[Wind::South.index()];
    south.hand = tiles("147m258p369p5s");
    south.melds = vec![Meld::pon(tile("5s"), ClaimedFrom::Left)];
    hand.act(Action::Riichi(tile("1z"))).unwrap();
    if matches!(hand.phase, Phase::CallWindow) {
        hand.resolve_calls(&[]).unwrap();
    }
    draw(&mut hand, "9m");
    hand.act(Action::ExtendedKan(tile("5s"))).unwrap();
    let yaku = ron(&mut hand, Wind::East);
    assert!(yaku.contains(&Yaku::Chankan), "{yaku:?}");
    assert!(yaku.contains(&Yaku::Ippatsu), "{yaku:?}");
}

/// East holds a quad of 1 characters to declare, and four 2 circles that
/// could make another, with one live tile left in the wall.
fn quad_with_one_tile_left() -> Hand {
    let mut hand = dealt(
        [
            "1111m2222p678s99s",
            "99m55z123p456p789p",
            WEST_IDLE,
            NORTH_IDLE,
        ],
        "3p",
    );
    // The hand has run until one live tile is left, and everyone has
    // discarded.
    hand.first_turns_unbroken = false;
    for (order, player) in hand.players.iter_mut().enumerate() {
        player.discards.push(Discard {
            tile: tile("1z"),
            order: order as u32,
            drawn: true,
            riichi: false,
            claimed: false,
        });
    }
    hand.discards_made = 4;
    while hand.wall.remaining() > 1 {
        hand.wall.draw();
    }
    hand.act(Action::ConcealedKan(tile("1m"))).unwrap();
    // The replacement draw took the last live tile into the dead wall.
    assert_eq!(hand.wall.remaining(), 0);
    assert!(hand.after_quad);
    let offered = hand.legal_actions();
    assert!(
        !offered
            .iter()
            .any(|action| matches!(action, Action::ConcealedKan(_))),
        "no quad once the wall is empty, though four 2 circles are held: {offered:?}"
    );
    hand
}

/// A win on the replacement drawn with the wall empty is After a Quad, and
/// not also Under the Sea, which it does not combine with (EMA section
/// 4.2.1).
#[test]
fn a_replacement_drawn_with_the_wall_empty_is_after_a_quad_alone() {
    let mut hand = quad_with_one_tile_left();
    replace_drawn(&mut hand, "4p");
    let yaku = tsumo(&mut hand);
    assert!(yaku.contains(&Yaku::Rinshan), "{yaku:?}");
    assert!(!yaku.contains(&Yaku::Haitei), "{yaku:?}");
}

/// The discard after that replacement is the last of the hand: it may be
/// claimed for a win only (section 3.4.1), and a win on it is Under the
/// River.
#[test]
fn the_discard_after_that_replacement_is_the_last_discard() {
    let mut hand = quad_with_one_tile_left();
    replace_drawn(&mut hand, "9m");
    hand.act(Action::Discard(tile("9m"))).unwrap();
    assert_eq!(
        hand.legal_calls(),
        vec![(Wind::South, vec![Call::Ron, Call::Pass])],
        "South holds a pair of 9 characters, but may only win on the last discard"
    );
    let yaku = ron(&mut hand, Wind::South);
    assert!(yaku.contains(&Yaku::Houtei), "{yaku:?}");
}
