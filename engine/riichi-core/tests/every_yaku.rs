//! Every yaku the rulebook lists, produced from a hand built for it.
//!
//! The million-hand differential exercises the scorer hard, but it only
//! reaches what random construction happens to produce: over a million
//! winning hands it never once made All Terminals, and it can never make
//! the three blessings, which depend on when in the hand the win came
//! rather than on the tiles. Four of the forty-one were therefore untested
//! by anything but their own definition.
//!
//! So this builds a hand for each in turn and checks the scorer finds it,
//! with the han the rulebook gives, open and closed where those differ.
//! Sections 4.2.1 to 4.2.6.

use riichi_core::hand::{ClaimedFrom, Meld, MeldKind, TileSet};
use riichi_core::score::{score, Riichi, Situation, WinBy};
use riichi_core::tile::Tile;
use riichi_core::yaku::Yaku;
use riichi_core::Wind;

fn hand(text: &str) -> TileSet {
    text.parse().expect("a test hand parses")
}

fn tile(text: &str) -> Tile {
    text.parse().expect("a test tile parses")
}

/// A plain concealed win by discard, for a South player in an East round.
fn plain(winning: &str) -> Situation {
    Situation::new(Wind::South, Wind::East, WinBy::Discard, tile(winning))
}

/// Scores the hand and returns the yaku found, or panics saying why not.
fn found(concealed: &TileSet, melds: &[Meld], situation: &Situation) -> Vec<(Yaku, u8)> {
    match score(concealed, melds, situation) {
        Ok(scored) => scored.yaku,
        Err(error) => panic!("the hand did not score: {error:?}"),
    }
}

/// Whether a yaku of this name was found, and for how many han.
fn han_for(yaku: &[(Yaku, u8)], name: &str) -> Option<u8> {
    yaku.iter()
        .find(|(entry, _)| entry.name() == name)
        .map(|(_, han)| *han)
}

/// Checks one yaku: the hand produces it, for the han expected.
#[track_caller]
fn expect(
    name: &str,
    han: u8,
    concealed: &TileSet,
    melds: &[Meld],
    situation: &Situation,
) -> Vec<(Yaku, u8)> {
    let yaku = found(concealed, melds, situation);
    let got = han_for(&yaku, name);
    let listed: Vec<&str> = yaku.iter().map(|(entry, _)| entry.name()).collect();
    assert_eq!(
        got,
        Some(han),
        "{name} was not found for {han} han; the hand scored {listed:?}"
    );
    yaku
}

// ---------------------------------------------------------------- 4.2.1

/// Riichi, and ippatsu on top of it, and double riichi in place of it.
#[test]
fn the_declaration_yaku() {
    let tiles = hand("234m567m234p55s678s");
    let mut situation = plain("8s");
    situation.riichi = Riichi::Declared;
    expect("Riichi", 1, &tiles, &[], &situation);

    situation.ippatsu = true;
    let yaku = expect("Ippatsu", 1, &tiles, &[], &situation);
    assert_eq!(han_for(&yaku, "Riichi"), Some(1), "both, not one instead");

    // A declaration in the first turn is worth two and is not also Riichi.
    situation.riichi = Riichi::Double;
    situation.ippatsu = false;
    let yaku = expect("Double Riichi", 2, &tiles, &[], &situation);
    assert_eq!(han_for(&yaku, "Riichi"), None, "one declaration, not two");
}

/// Winning on your own draw with a hand nobody has called on.
#[test]
fn fully_concealed_hand() {
    let tiles = hand("234m567m234p55s678s");
    let situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("8s"));
    expect("Fully Concealed Hand", 1, &tiles, &[], &situation);
}

/// Four sequences, a pair worth nothing, and a two-sided wait.
#[test]
fn pinfu() {
    let tiles = hand("234m567m234p55s678s");
    expect("Pinfu", 1, &tiles, &[], &plain("6s"));
}

/// Two sequences of the same numbers in the same suit.
#[test]
fn pure_double_sequence() {
    let tiles = hand("223344m567p55s678s");
    expect("Pure Double Sequence", 1, &tiles, &[], &plain("8s"));
}

/// Nothing but twos to eights, which an open hand may also declare.
#[test]
fn all_simples_open_and_closed() {
    let tiles = hand("234m567m234p55s678s");
    expect("All Simples", 1, &tiles, &[], &plain("8s"));

    // EMA allows it with a called set, at the same one han.
    let melds = [Meld::chii(tile("2m"), ClaimedFrom::Left)];
    let open = hand("567m234p55s678s");
    expect("All Simples", 1, &open, &melds, &plain("8s"));
}

/// A triplet of dragons, of the seat wind, and of the round wind.
#[test]
fn the_value_triplets() {
    // Green dragons, which score for anybody.
    let tiles = hand("666z234m567m234p55s");
    expect("Dragon Triplet", 1, &tiles, &[], &plain("5s"));

    // South's own wind, in an East round: the seat wind only.
    let tiles = hand("222z234m567m234p55s");
    let yaku = expect("Seat Wind Triplet", 1, &tiles, &[], &plain("5s"));
    assert_eq!(han_for(&yaku, "Round Wind Triplet"), None);

    // East's wind in an East round, for a South player: the round wind only.
    let tiles = hand("111z234m567m234p55s");
    let yaku = expect("Round Wind Triplet", 1, &tiles, &[], &plain("5s"));
    assert_eq!(han_for(&yaku, "Seat Wind Triplet"), None);

    // East's own wind in an East round is both, one han each.
    let tiles = hand("111z234m567m234p55s");
    let situation = Situation::new(Wind::East, Wind::East, WinBy::Discard, tile("5s"));
    let yaku = expect("Seat Wind Triplet", 1, &tiles, &[], &situation);
    assert_eq!(
        han_for(&yaku, "Round Wind Triplet"),
        Some(1),
        "the double wind is two yaku of one han, not one of two"
    );
}

/// The four yaku that depend on which tile the hand was won with.
#[test]
fn the_circumstance_yaku() {
    let tiles = hand("234m567m234p55s678s");

    let mut after_quad = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("8s"));
    after_quad.after_quad = true;
    expect("After a Quad", 1, &tiles, &[], &after_quad);

    let mut robbing = plain("8s");
    robbing.robbing_quad = true;
    expect("Robbing a Quad", 1, &tiles, &[], &robbing);

    let mut sea = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("8s"));
    sea.under_the_sea = true;
    expect("Under the Sea", 1, &tiles, &[], &sea);

    let mut river = plain("8s");
    river.under_the_river = true;
    expect("Under the River", 1, &tiles, &[], &river);
}

// ---------------------------------------------------------------- 4.2.2

/// Seven different pairs, which is a hand shape of its own.
#[test]
fn seven_pairs() {
    let tiles = hand("1122335577m99p22s");
    expect("Seven Pairs", 2, &tiles, &[], &plain("2s"));
}

/// The same three numbers in each suit, as sequences and as triplets.
#[test]
fn the_three_suit_yaku() {
    let tiles = hand("234m234p234s567m11z");
    expect("Mixed Triple Sequence", 2, &tiles, &[], &plain("7m"));

    let melds = [Meld::chii(tile("2m"), ClaimedFrom::Left)];
    let open = hand("234p234s567m11z");
    expect("Mixed Triple Sequence", 1, &open, &melds, &plain("7m"));

    let tiles = hand("333m333p333s567m11z");
    expect("Triple Triplet", 2, &tiles, &[], &plain("7m"));
}

/// One to nine of a single suit, in three sequences.
#[test]
fn pure_straight() {
    let tiles = hand("123456789m234p55s");
    expect("Pure Straight", 2, &tiles, &[], &plain("5s"));

    let melds = [Meld::chii(tile("1m"), ClaimedFrom::Left)];
    let open = hand("456789m234p55s");
    expect("Pure Straight", 1, &open, &melds, &plain("5s"));
}

/// Every set touching a terminal or an honour, with and without honours.
#[test]
fn the_outside_hands() {
    // Half Outside: at least one set has an honour.
    let tiles = hand("123m123p789s111z99p");
    expect("Half Outside Hand", 2, &tiles, &[], &plain("9p"));

    let melds = [Meld::chii(tile("1m"), ClaimedFrom::Left)];
    let open = hand("123p789s111z99p");
    expect("Half Outside Hand", 1, &open, &melds, &plain("9p"));

    // Full Outside: terminals only, no honours, and at least one sequence.
    let tiles = hand("123m123p789s111m99p");
    let yaku = expect("Full Outside Hand", 3, &tiles, &[], &plain("9p"));
    assert_eq!(
        han_for(&yaku, "Half Outside Hand"),
        None,
        "the fuller hand replaces the lesser one"
    );

    let melds = [Meld::chii(tile("1p"), ClaimedFrom::Left)];
    let open = hand("123m789s111m99p");
    expect("Full Outside Hand", 2, &open, &melds, &plain("9p"));
}

/// Three sets made of the same tile, concealed, and three quads.
#[test]
fn three_concealed_triplets_and_three_quads() {
    let tiles = hand("111m333p555s234m99p");
    expect("Three Concealed Triplets", 2, &tiles, &[], &plain("9p"));

    let melds = [
        Meld::concealed_kan(tile("1m")),
        Meld::concealed_kan(tile("3p")),
        Meld {
            kind: MeldKind::ClaimedKan,
            tile: tile("5s"),
            from: ClaimedFrom::Left,
        },
    ];
    let concealed = hand("234m99p");
    expect("Three Quads", 2, &concealed, &melds, &plain("9p"));
}

/// Four sets of identical tiles and no sequence at all.
///
/// One of them has to be called for. Four triplets nobody called on is
/// Four Concealed Triplets, which is a yakuman and replaces this.
#[test]
fn all_triplets() {
    let melds = [Meld::pon(tile("1m"), ClaimedFrom::Left)];
    let concealed = hand("333p555s777m99p");
    let yaku = expect("All Triplets", 2, &concealed, &melds, &plain("9p"));
    assert_eq!(han_for(&yaku, "Four Concealed Triplets"), None);
}

/// Two dragon sets and a pair of the third.
#[test]
fn little_three_dragons() {
    let tiles = hand("555z666z77z234m567m");
    expect("Little Three Dragons", 2, &tiles, &[], &plain("7m"));
}

/// Nothing but terminals and honours, in a hand that is not all one or the
/// other, so the yakuman above it do not take over.
#[test]
fn all_terminals_and_honours() {
    // Called for, again, so this does not become the concealed yakuman.
    let melds = [Meld::pon(tile("1m"), ClaimedFrom::Left)];
    let concealed = hand("999p111z999s55z");
    let yaku = expect(
        "All Terminals and Honours",
        2,
        &concealed,
        &melds,
        &plain("5z"),
    );
    assert_eq!(
        han_for(&yaku, "All Triplets"),
        Some(2),
        "EMA 4.2.2 says the player adds two han for All Triplets on top"
    );
    assert_eq!(
        han_for(&yaku, "All Terminals"),
        None,
        "there are honours in it"
    );
    assert_eq!(han_for(&yaku, "All Honours"), None, "and terminals too");
}

// ---------------------------------------------------------------- 4.2.3

/// Two pairs of matching sequences, which swallows the single one.
#[test]
fn twice_pure_double_sequence() {
    let tiles = hand("223344m667788p99s");
    let yaku = expect("Twice Pure Double Sequence", 3, &tiles, &[], &plain("9s"));
    assert_eq!(han_for(&yaku, "Pure Double Sequence"), None);
}

/// One suit and honours, then one suit and nothing else.
#[test]
fn the_flushes() {
    let tiles = hand("123456789m11z234m");
    let yaku = expect("Half Flush", 3, &tiles, &[], &plain("4m"));
    assert_eq!(
        han_for(&yaku, "Full Flush"),
        None,
        "honours bar the fuller one"
    );

    let melds = [Meld::chii(tile("1m"), ClaimedFrom::Left)];
    let open = hand("456789m11z234m");
    expect("Half Flush", 2, &open, &melds, &plain("4m"));

    let tiles = hand("123456789m22m345m");
    let yaku = expect("Full Flush", 6, &tiles, &[], &plain("5m"));
    assert_eq!(han_for(&yaku, "Half Flush"), None);

    let melds = [Meld::chii(tile("1m"), ClaimedFrom::Left)];
    let open = hand("456789m22m345m");
    expect("Full Flush", 5, &open, &melds, &plain("5m"));
}

// ------------------------------------------------------ 4.2.4 and 4.2.5

/// Winning by discard before your first turn, which stands alone: EMA
/// 4.2.4 says it "cannot be combined with other yaku or with dora".
#[test]
fn blessing_of_man() {
    let tiles = hand("234m567m234p55s678s");
    let mut situation = plain("8s");
    situation.blessing_of_man = true;
    situation.dora_indicators = vec![tile("1m"), tile("4m")];
    let yaku = expect("Blessing of Man", 5, &tiles, &[], &situation);
    assert_eq!(yaku.len(), 1, "it stands alone: {yaku:?}");

    let scored = score(&tiles, &[], &situation).unwrap();
    assert_eq!(scored.dora, 0, "and takes no dora either");
    assert_eq!(scored.han, 5);
}

// ---------------------------------------------------------------- 4.2.6

/// One of each terminal and honour, plus a second of one of them.
#[test]
fn thirteen_orphans() {
    let tiles = hand("19m19p19s1234567z1z");
    expect("Thirteen Orphans", 13, &tiles, &[], &plain("1z"));
}

/// One suit, held 1112345678999 plus any one of it.
#[test]
fn nine_gates() {
    let tiles = hand("11123456789995m");
    expect("Nine Gates", 13, &tiles, &[], &plain("5m"));
}

/// The dealer's opening hand, and a non-dealer's first draw.
#[test]
fn the_blessings_of_heaven_and_earth() {
    let tiles = hand("234m567m234p55s678s");

    let mut heaven = Situation::new(Wind::East, Wind::East, WinBy::SelfDraw, tile("8s"));
    heaven.blessing_of_heaven = true;
    expect("Blessing of Heaven", 13, &tiles, &[], &heaven);

    let mut earth = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("8s"));
    earth.blessing_of_earth = true;
    expect("Blessing of Earth", 13, &tiles, &[], &earth);
}

/// Four sets of identical tiles, none of them called for.
#[test]
fn four_concealed_triplets() {
    let tiles = hand("111m333p555s777m99p");
    let situation = Situation::new(Wind::South, Wind::East, WinBy::SelfDraw, tile("7m"));
    let yaku = expect("Four Concealed Triplets", 13, &tiles, &[], &situation);
    assert_eq!(
        han_for(&yaku, "All Triplets"),
        None,
        "a yakuman is not padded with the lesser yaku it contains"
    );
}

/// Four quads, which needs four melds and a pair in hand.
#[test]
fn four_quads() {
    let melds = [
        Meld::concealed_kan(tile("1m")),
        Meld::concealed_kan(tile("3p")),
        Meld::concealed_kan(tile("5s")),
        Meld::concealed_kan(tile("7m")),
    ];
    let concealed = hand("99p");
    expect("Four Quads", 13, &concealed, &melds, &plain("9p"));
}

/// Nothing but the six green tiles.
#[test]
fn all_green() {
    let tiles = hand("22334466888s666z");
    expect("All Green", 13, &tiles, &[], &plain("6z"));
}

/// Nothing but ones and nines. Random construction never once made this
/// over a million hands, which is why it is here.
#[test]
fn all_terminals() {
    let tiles = hand("111m999m111p999p99s");
    let yaku = expect("All Terminals", 13, &tiles, &[], &plain("9s"));
    assert_eq!(
        han_for(&yaku, "All Terminals and Honours"),
        None,
        "the yakuman replaces the two han hand it contains"
    );
}

/// Nothing but winds and dragons.
#[test]
fn all_honours() {
    let tiles = hand("111z222z333z444z55z");
    let yaku = expect("All Honours", 13, &tiles, &[], &plain("5z"));
    assert_eq!(han_for(&yaku, "All Terminals and Honours"), None);
}

/// All three dragon sets, and then the winds.
#[test]
fn the_dragon_and_wind_yakuman() {
    let tiles = hand("555z666z777z234m99p");
    let yaku = expect("Big Three Dragons", 13, &tiles, &[], &plain("9p"));
    assert_eq!(han_for(&yaku, "Little Three Dragons"), None);

    // Three wind sets and a pair of the fourth.
    let tiles = hand("111z222z333z44z234m");
    let yaku = expect("Little Four Winds", 13, &tiles, &[], &plain("4m"));
    assert_eq!(han_for(&yaku, "Big Four Winds"), None);

    // All four wind sets.
    let tiles = hand("111z222z333z444z99p");
    let yaku = expect("Big Four Winds", 13, &tiles, &[], &plain("9p"));
    assert_eq!(han_for(&yaku, "Little Four Winds"), None);
}

/// Nothing in the rulebook's list is missing from the engine, and nothing
/// in the engine is missing from the rulebook. The count is the check: the
/// tests above name every one of them, so a yaku added without a test here
/// makes this fail and say so.
#[test]
fn the_engine_knows_exactly_the_rulebooks_yaku() {
    let named = [
        // 4.2.1, one han
        "Riichi",
        "Ippatsu",
        "Fully Concealed Hand",
        "Pinfu",
        "Pure Double Sequence",
        "All Simples",
        "Dragon Triplet",
        "Seat Wind Triplet",
        "Round Wind Triplet",
        "After a Quad",
        "Robbing a Quad",
        "Under the Sea",
        "Under the River",
        // 4.2.2, two han
        "Double Riichi",
        "Seven Pairs",
        "Mixed Triple Sequence",
        "Pure Straight",
        "Half Outside Hand",
        "Triple Triplet",
        "Three Concealed Triplets",
        "Three Quads",
        "All Triplets",
        "Little Three Dragons",
        "All Terminals and Honours",
        // 4.2.3, three han
        "Twice Pure Double Sequence",
        "Half Flush",
        "Full Outside Hand",
        // 4.2.4 and 4.2.5
        "Blessing of Man",
        "Full Flush",
        // 4.2.6, yakuman
        "Thirteen Orphans",
        "Nine Gates",
        "Blessing of Heaven",
        "Blessing of Earth",
        "Four Concealed Triplets",
        "Four Quads",
        "All Green",
        "All Terminals",
        "All Honours",
        "Big Three Dragons",
        "Little Four Winds",
        "Big Four Winds",
    ];
    assert_eq!(named.len(), 41, "the rulebook lists forty-one");

    let engine: Vec<&'static str> = Yaku::ALL.iter().map(|yaku| yaku.name()).collect();
    for name in named {
        assert!(
            engine.contains(&name),
            "the engine has no yaku called {name}"
        );
    }
    for name in &engine {
        assert!(
            named.contains(name),
            "the engine has {name}, which the rulebook does not list"
        );
    }
    assert_eq!(engine.len(), named.len(), "and no more besides");
}
