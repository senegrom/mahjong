//! The last hand's two rules as the page asks them (riichi_core::all_last):
//! the codes only, for the seat that owes the decision. The page plays the
//! rules over Trained AI's answer (web/src/lib/trained-move.js).

use riichi_core::all_last::Dama;
use riichi_core::game::{Hand, Phase};
use riichi_core::tile::KINDS;
use riichi_core::Wind;

/// The dama codes of a seat that is not asked: absent for every kind.
pub(crate) fn dama_absent() -> Vec<u8> {
    vec![Dama::Absent as u8; KINDS]
}

/// The all-last dama codes of `seat`, a byte a tile kind, numbered as `Dama`
/// is. Asked only at `seat`'s own move, since the core answers for the turn
/// player whoever asks. Zeros anywhere else, a claim window included. A
/// typed-in table can mark the acting seat furiten from a pass, which play
/// clears at the draw and at a claim, so these codes are read as play leaves
/// the hand. Only these: decline reads the hand as it stands.
pub(crate) fn dama_codes(hand: &Hand, seat: Wind) -> Vec<u8> {
    if hand.phase != Phase::Act || hand.turn != seat {
        return dama_absent();
    }
    let codes = if hand.players[seat.index()].temporary_furiten {
        let mut played = hand.clone();
        played.players[seat.index()].temporary_furiten = false;
        played.all_last_dama()
    } else {
        hand.all_last_dama()
    };
    codes.iter().map(|code| *code as u8).collect()
}

/// The decline-last code of `seat`, numbered as `Decline` is: 0 wherever the
/// core does not ask. Read on the hand as it stands, a furiten tick included:
/// a passed ron bars the next, so a ticked seat has no ron to decline.
pub(crate) fn decline_code(hand: &Hand, seat: Wind) -> u8 {
    hand.all_last_decline(seat) as u8
}

/// The two helpers on the core's own tables. The tables are built as
/// `riichi_core::all_last`'s tests build them, their code copied, since a
/// crate's tests cannot be imported; `crate::last_hand_tests` asks a game
/// at the same tables.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{mortal_mask_of, MORTAL_AGARI};
    use riichi_core::all_last::Decline;
    use riichi_core::game::Discard;
    use riichi_core::rng::Rng;
    use riichi_core::tile::Tile;
    use riichi_core::wall::Wall;

    pub(crate) fn tile(text: &str) -> Tile {
        text.parse().unwrap()
    }

    /// A hand of `round` and `kyoku` with `seat` to move on its fourteen
    /// tiles `held`, `drawn` among them, and the points by seat. The wall
    /// holds thirty tiles and its one indicator is the south wind, so the
    /// dora is the west wind, which no hand here holds. The other three hold
    /// what the deal gave them, the dealer's fourteenth tile put back, as
    /// `claim` puts it back; nothing asked here reads them.
    pub(crate) fn position(
        round: Wind,
        kyoku: u8,
        seat: Wind,
        held: &str,
        drawn: &str,
        scores: [i32; 4],
    ) -> Hand {
        let mut hand = Hand::deal(&mut Rng::from_seed(20261030), round, kyoku, 0, 0, scores);
        if let Some(extra) = hand.drawn.take() {
            hand.players[Wind::East.index()].hand.remove(extra);
        }
        hand.wall = Wall::for_analysis(30, &[tile("2z")], 0).unwrap();
        hand.players[seat.index()].hand = held.parse().unwrap();
        hand.drawn = Some(tile(drawn));
        hand.turn = seat;
        hand.phase = Phase::Act;
        hand.first_turns_unbroken = false;
        hand
    }

    /// The probe: South 4, North to move on 23456m 456p 678s 33s with the
    /// 9p just drawn, East on 20,000, South on 36,000 + `gap`, West on
    /// 28,000 - `gap`, North on 36,000. Thrown, the 9p leaves a wait on 1m,
    /// 4m and 7m: a 1m makes pinfu alone, 1,000 by ron and 400/700 by tsumo,
    /// and a 4m or 7m pinfu and tanyao, 2,000 by ron and 700/1,300 by tsumo.
    /// A 1m ron off East or West binds: first only while `gap` is under
    /// 1,000.
    pub(crate) fn probe(gap: i32) -> Hand {
        position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [20_000, 36_000 + gap, 28_000 - gap, 36_000],
        )
    }

    /// A hand of `round` and `kyoku` in which `from` has just thrown the
    /// `thrown` tile and `seat`, on the thirteen tiles `held`, may claim it;
    /// the points by seat. The wall is `position`'s, and the others hold what
    /// the deal gave them, the dealer's fourteenth tile put back.
    pub(crate) fn claim(
        round: Wind,
        kyoku: u8,
        from: Wind,
        thrown: &str,
        seat: Wind,
        held: &str,
        scores: [i32; 4],
    ) -> Hand {
        let mut hand = Hand::deal(&mut Rng::from_seed(20261030), round, kyoku, 0, 0, scores);
        if let Some(extra) = hand.drawn.take() {
            hand.players[Wind::East.index()].hand.remove(extra);
        }
        hand.wall = Wall::for_analysis(30, &[tile("2z")], 0).unwrap();
        hand.players[seat.index()].hand = held.parse().unwrap();
        hand.players[from.index()].discards.push(Discard {
            tile: tile(thrown),
            order: 0,
            drawn: false,
            riichi: false,
            claimed: false,
        });
        hand.discards_made = 1;
        hand.turn = from;
        hand.phase = Phase::CallWindow;
        hand.pending_discard = Some((from, tile(thrown)));
        hand.first_turns_unbroken = false;
        hand
    }

    /// South 4, North to move on 123456m 456p 678s 33s with the 1m just
    /// drawn: a tsumo of pinfu and menzen tsumo, 400/700. North on 10,000,
    /// South on 38,000, West on `west`, and East the rest of 100,000.
    pub(crate) fn tsumo(west: i32) -> Hand {
        position(
            Wind::South,
            4,
            Wind::North,
            "123456m456p678s33s",
            "1m",
            [52_000 - west, 38_000, west, 10_000],
        )
    }

    /// South 4, West has just thrown a 1m that North, on 23456m 456p 678s
    /// 33s, may ron: pinfu, 1,000. North on 10,000, South on 38,000, West on
    /// `west`, and East the rest of 100,000.
    pub(crate) fn ron(west: i32) -> Hand {
        claim(
            Wind::South,
            4,
            Wind::West,
            "1m",
            Wind::North,
            "23456m456p678s33s",
            [52_000 - west, 38_000, west, 10_000],
        )
    }

    fn code(codes: &[u8], text: &str) -> u8 {
        codes[tile(text).idx()]
    }

    /// The probe asks North, the turn player: the 9p keeps first, every
    /// other tile it holds leaves it not waiting, and the kinds it does not
    /// hold are not asked about. Nobody else is asked, whatever the core
    /// would answer for the turn player, and nobody in a claim window.
    #[test]
    fn dama_is_asked_of_the_turn_seat_alone() {
        let hand = probe(0);
        let codes = dama_codes(&hand, Wind::North);
        let core: Vec<u8> = hand
            .all_last_dama()
            .iter()
            .map(|code| *code as u8)
            .collect();
        assert_eq!(codes, core);
        let held = hand.players[Wind::North.index()].hand;
        for kind in Tile::all() {
            let want = if kind == tile("9p") {
                Dama::KeepsFirst
            } else if held.count(kind) > 0 {
                Dama::NotWaiting
            } else {
                Dama::Absent
            };
            assert_eq!(codes[kind.idx()], want as u8, "{kind}");
        }
        for seat in [Wind::East, Wind::South, Wind::West] {
            assert_eq!(dama_codes(&hand, seat), dama_absent(), "{seat:?}");
        }
        let window = ron(12_100);
        for seat in Wind::ALL {
            assert_eq!(dama_codes(&window, seat), dama_absent(), "{seat:?}");
        }
    }

    /// The core's own gaps (`a_tie_for_first_is_short`): first while the
    /// gap is under a thousand, and a tie for first is not first.
    #[test]
    fn the_9p_keeps_first_while_the_gap_is_under_a_thousand() {
        for (gap, want) in [
            (900, Dama::KeepsFirst),
            (1_000, Dama::Short),
            (1_500, Dama::Short),
        ] {
            let codes = dama_codes(&probe(gap), Wind::North);
            assert_eq!(code(&codes, "9p"), want as u8, "a gap of {gap}");
        }
    }

    /// A typed-in table marks the acting seat furiten from its tick, which
    /// play never leaves set at a seat's own move. At a gap of 1,500 only a
    /// 1m ron leaves North short: the core alone, reading the tick, counts
    /// no ron and answers that the 9p keeps first, and the helper answers
    /// as for the table unticked. In a claim window the tick stands: a
    /// ticked seat is offered no ron, so there is none to decline.
    #[test]
    fn a_furiten_tick_is_cleared_for_dama_and_read_for_decline() {
        let plain = probe(1_500);
        let mut ticked = plain.clone();
        ticked.players[Wind::North.index()].temporary_furiten = true;
        assert_eq!(
            ticked.all_last_dama()[tile("9p").idx()],
            Dama::KeepsFirst,
            "the core reads the tick"
        );
        assert_eq!(
            dama_codes(&ticked, Wind::North),
            dama_codes(&plain, Wind::North)
        );
        assert_eq!(
            code(&dama_codes(&ticked, Wind::North), "9p"),
            Dama::Short as u8
        );
        assert!(ticked.players[Wind::North.index()].temporary_furiten);

        let open = ron(12_100);
        assert!(mortal_mask_of(&open, Wind::North, false)[MORTAL_AGARI]);
        assert_eq!(decline_code(&open, Wind::North), Decline::Last as u8);
        let mut ticked = open;
        ticked.players[Wind::North.index()].temporary_furiten = true;
        assert!(!mortal_mask_of(&ticked, Wind::North, false)[MORTAL_AGARI]);
        assert_eq!(decline_code(&ticked, Wind::North), Decline::Absent as u8);
    }

    /// The core's answers, a byte, for every seat: a self-draw and a ron that
    /// leave North alone in last or not
    /// (`a_self_draw_that_leaves_the_player_alone_in_last`,
    /// `a_ron_that_leaves_the_player_alone_in_last`); a seat with no win
    /// open, and the dealer, offered the same ron, never asked.
    #[test]
    fn decline_answers_what_the_core_answers() {
        for (hand, want) in [
            (tsumo(11_900), Decline::NotLast),
            (tsumo(12_000), Decline::Last),
            (ron(12_000), Decline::NotLast),
            (ron(12_100), Decline::Last),
        ] {
            let scores = hand.scores();
            assert_eq!(
                decline_code(&hand, Wind::North),
                want as u8,
                "{scores:?} in {:?}",
                hand.phase
            );
            for seat in Wind::ALL {
                assert_eq!(
                    decline_code(&hand, seat),
                    hand.all_last_decline(seat) as u8,
                    "{seat:?}, {scores:?} in {:?}",
                    hand.phase
                );
            }
            if hand.phase == Phase::Act {
                assert_eq!(decline_code(&hand, Wind::South), 0, "{scores:?}");
            }
        }
        let mut both = ron(12_100);
        both.players[Wind::East.index()].hand = "23456m456p678s33s".parse().unwrap();
        assert!(mortal_mask_of(&both, Wind::East, false)[MORTAL_AGARI]);
        assert_eq!(decline_code(&both, Wind::East), 0);
        assert_eq!(decline_code(&both, Wind::North), Decline::Last as u8);
    }

    /// The page and Python read the codes by number (`DAMA` and `DECLINE`
    /// in web/src/lib/trained-move.js; neural/all_last.py and
    /// neural/late_rules.py), so the numbers are part of the bindings.
    #[test]
    fn the_codes_are_numbered_as_the_page_reads_them() {
        assert_eq!(
            [
                Dama::Absent,
                Dama::KeepsFirst,
                Dama::NoYaku,
                Dama::Short,
                Dama::Dead,
                Dama::NotWaiting
            ]
            .map(|code| code as u8),
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            [Decline::Absent, Decline::Last, Decline::NotLast].map(|code| code as u8),
            [0, 1, 2]
        );
    }
}
