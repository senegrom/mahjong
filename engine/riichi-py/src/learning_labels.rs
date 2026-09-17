//! Training-only immediate-discard labels. Never an acting observation.
//!
//! Rows are relative opponents (next, across, previous). Each has a validity
//! flag, tenpai, 34 evaluated-discard flags, 34 legal-ron flags, and 34 point
//! charges. A point charge is for this opponent claiming alone, including
//! honba/liability but excluding a new riichi deposit and other winners.
//! Call windows are deliberately unlabelled, not labelled safe. Only currently
//! legal ordinary discards are evaluated; unseen/future tiles are not negatives.
use super::Seat;
use riichi_core::game::{Action, Call, Hand, Phase};
use riichi_core::tile::Tile;
use riichi_core::Wind;

pub(super) const WIDTH: usize = 104;

pub(super) fn label(hand: &Hand, actor: Wind) -> [f32; 3 * WIDTH] {
    let mut out = [0.0; 3 * WIDTH];
    if !matches!(hand.phase, Phase::Act) || hand.turn != actor {
        return out;
    }
    for relative in 0..3 {
        let other = actor.plus(relative + 1);
        out[relative * WIDTH] = 1.0;
        out[relative * WIDTH + 1] = f32::from(hand.players[other.index()].is_tenpai());
    }
    for action in hand.legal_actions() {
        let Action::Discard(tile) = action else {
            continue;
        };
        let k = tile.idx();
        let mut discarded = hand.clone();
        discarded
            .act(Action::Discard(Tile::new(k as u8)))
            .expect("listed legal discard");
        let offered = discarded.legal_calls();
        for relative in 0..3 {
            let other = actor.plus(relative + 1);
            let row = relative * WIDTH;
            out[row + 2 + k] = 1.0;
            if !offered
                .iter()
                .any(|(seat, calls)| *seat == other && calls.contains(&Call::Ron))
            {
                continue;
            }
            out[row + 36 + k] = 1.0;
            let mut won = discarded.clone();
            let before = won.players[actor.index()].score;
            let answers: Vec<_> = offered
                .iter()
                .map(|(seat, _)| {
                    (
                        *seat,
                        if *seat == other {
                            Call::Ron
                        } else {
                            Call::Pass
                        },
                    )
                })
                .collect();
            won.resolve_calls(&answers)
                .expect("single listed legal ron");
            out[row + 70 + k] = (before - won.players[actor.index()].score) as f32;
        }
    }
    out
}

pub(super) fn collect(seats: &[Seat]) -> Vec<f32> {
    seats
        .iter()
        .flat_map(|seat| match seat.pending() {
            Some(actor) => label(&seat.hand, actor),
            None => [0.0; 3 * WIDTH],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_read_only_and_known_only_on_legal_discards() {
        let seat = Seat::new(7, &[]);
        let before = format!("{:?}", seat.hand);
        let data = label(&seat.hand, seat.hand.turn);
        assert_eq!(format!("{:?}", seat.hand), before);
        let actions = seat.hand.legal_actions();
        for r in 0..3 {
            for t in 0..34 {
                assert_eq!(
                    data[r * WIDTH + 2 + t] > 0.0,
                    actions.contains(&Action::Discard(Tile::new(t as u8)))
                );
                assert!(data[r * WIDTH + 36 + t] <= data[r * WIDTH + 2 + t]);
                assert!(data[r * WIDTH + 70 + t] >= 0.0);
            }
        }
    }

    #[test]
    fn call_windows_are_missing_labels_not_safe_labels() {
        let mut seat = Seat::new(7, &[]);
        let action = seat
            .hand
            .legal_actions()
            .into_iter()
            .find(|a| matches!(a, Action::Discard(_)))
            .unwrap();
        seat.hand.act(action).unwrap();
        assert!(label(&seat.hand, Wind::South).iter().all(|v| *v == 0.0));
    }
    #[test]
    fn a_legal_ron_has_a_real_charge_and_furiten_is_not_ignored() {
        let mut seat = Seat::new(7, &[]);
        seat.hand.players[0].hand = "123456789m11s249p".parse().unwrap();
        seat.hand.players[1].hand = "123456789m11s34p".parse().unwrap();
        seat.hand.players[1].riichi = riichi_core::score::Riichi::Declared;
        seat.hand.drawn = Some("2p".parse().unwrap());
        seat.hand.counters = 2;
        let before = seat.hand.clone();
        let tile: Tile = "2p".parse().unwrap();
        let data = label(&seat.hand, Wind::East);
        assert_eq!(seat.hand, before);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2 + tile.idx()], 1.0);
        assert_eq!(data[36 + tile.idx()], 1.0);
        assert!(data[70 + tile.idx()] >= 600.0);
        let mut won = seat.hand.clone();
        let starting = won.players[0].score;
        won.act(Action::Discard(tile)).unwrap();
        let claims = won
            .legal_calls()
            .iter()
            .map(|(who, _)| {
                (
                    *who,
                    if *who == Wind::South {
                        Call::Ron
                    } else {
                        Call::Pass
                    },
                )
            })
            .collect::<Vec<_>>();
        won.resolve_calls(&claims).unwrap();
        assert_eq!(
            data[70 + tile.idx()],
            (starting - won.players[0].score) as f32
        );
        seat.hand.players[1].temporary_furiten = true;
        let blocked = label(&seat.hand, Wind::East);
        assert_eq!(blocked[1], 1.0); // Still tenpai, but cannot legally ron.
        assert_eq!(blocked[36 + tile.idx()], 0.0);
        assert_eq!(blocked[70 + tile.idx()], 0.0);
    }
}
