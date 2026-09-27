//! The concealed quad a riichi player may declare, against the rulebook.
//!
//! EMA 2025 section 3.3.10: a player who declared riichi "may declare a
//! concealed quad if the drawn tile matches a concealed triplet, if this does
//! not change their waiting tiles and if the three tiles to be used for the
//! quad can only be interpreted as a triplet in a hand completed by any
//! waiting tile". Section 6.7.1 gives four hands that fail one of those
//! conditions. Each is here in the engine's notation, with the rulebook's
//! tile-font string beside it: q to o are 1 to 9 characters, a to l 1 to 9
//! circles, and z x c v b n m , . 1 to 9 bamboo.

use riichi_core::game::{Action, Hand, Phase};
use riichi_core::hand::TileSet;
use riichi_core::rng::Rng;
use riichi_core::score::Riichi;
use riichi_core::tile::Tile;
use riichi_core::Wind;

fn tile(text: &str) -> Tile {
    text.parse().expect("a test tile parses")
}

/// What South may do with `drawn`, having declared riichi on the thirteen
/// tiles of `declared`.
fn offered(declared: &str, drawn: &str) -> Vec<Action> {
    let declared: TileSet = declared.parse().expect("a test hand parses");
    assert_eq!(declared.len(), 13, "a riichi hand holds thirteen tiles");
    let drawn = tile(drawn);
    let mut hand = Hand::deal(&mut Rng::from_seed(3), Wind::East, 1, 0, 0, [25_000; 4]);
    let south = &mut hand.players[Wind::South.index()];
    south.hand = declared;
    south.hand.add(drawn);
    south.melds.clear();
    south.discards.clear();
    south.riichi = Riichi::Declared;
    south.riichi_order = Some(0);
    hand.turn = Wind::South;
    hand.phase = Phase::Act;
    hand.drawn = Some(drawn);
    hand.just_claimed = None;
    hand.legal_actions()
}

/// The concealed quads among the actions offered.
fn quads(actions: &[Action]) -> Vec<Tile> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::ConcealedKan(tile) => Some(*tile),
            _ => None,
        })
        .collect()
}

/// Example 1, "eeyusdfccccvb": 33m 67m 234p 3333s 45s. The hand holds all
/// four 3 bamboo, but the drawn 6 bamboo matches no concealed triplet, so
/// there is no quad to declare, of either.
#[test]
fn example_1_the_drawn_tile_must_match_a_concealed_triplet() {
    let actions = offered("3367m234p333345s", "6s");
    assert!(quads(&actions).is_empty(), "{actions:?}");
}

/// Example 2, "wersssdxcvm,.": 234m 2223p 234s 789s, a three-sided wait on
/// 1, 3 and 4 circles. A quad of 2 circles would leave only the 3 circles.
#[test]
fn example_2_a_quad_that_changes_the_waits() {
    let actions = offered("234m2223p234s789s", "2p");
    assert!(quads(&actions).is_empty(), "{actions:?}");
}

/// Example 3, "yyyiooofghzzz": 666m 8m 999m 456p 111s, waiting on 7 and 8
/// characters. The 6 and 9 characters can be read as 6-7-8 or 7-8-9 with
/// the waiting 7, so neither may become a quad; the 1 bamboo may.
#[test]
fn example_3_a_triplet_that_reads_as_part_of_a_sequence() {
    for drawn in ["6m", "9m"] {
        let actions = offered("6668999m456p111s", drawn);
        assert!(quads(&actions).is_empty(), "{drawn}: {actions:?}");
    }
    let actions = offered("6668999m456p111s", "1s");
    assert_eq!(quads(&actions), [tile("1s")], "{actions:?}");
}

/// Example 4, "qwerfffggghhh": 1234m 444p 555p 666p, waiting on 1 and 4
/// characters. Three consecutive triplets can be read as three 4-5-6
/// sequences, so none of the three may become a quad, even though that is
/// the lowest-scoring reading of every completed hand.
#[test]
fn example_4_three_consecutive_triplets() {
    for drawn in ["4p", "5p", "6p"] {
        let actions = offered("1234m444555666p", drawn);
        assert!(quads(&actions).is_empty(), "{drawn}: {actions:?}");
    }
}

/// The same sentence of section 3.3.10, with the drawn tile one of the
/// waits: 1122233455667m waits on 1, 2 and 3 characters, and the fourth 2
/// characters is drawn. It wins the hand, but as a quad it would leave only
/// 1 and 3 characters, since a hand cannot wait on a fifth copy (section
/// 3.3.8). The waits change, so the quad is refused.
#[test]
fn a_quad_of_a_waiting_tile_changes_the_waits() {
    let actions = offered("1122233455667m", "2m");
    assert!(actions.contains(&Action::Tsumo), "{actions:?}");
    assert!(
        actions.contains(&Action::Discard(tile("2m"))),
        "{actions:?}"
    );
    assert!(quads(&actions).is_empty(), "{actions:?}");
}
