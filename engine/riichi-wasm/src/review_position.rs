//! A decision of a game played here, written as a typed-in position.
//!
//! A review that asks what would change Trained AI's mind about a decision
//! edits the table the player decided at and asks the network again: a tile
//! of the hand swapped, a discard changed, the dora indicator moved. Each
//! edited table is a typed-in position, checked by the rules a physical
//! table is checked by and replayed into Mortal's events (`mortal_log`), so
//! the table it starts from is written in the same form. It is written from
//! the decision's own hand, as the engine kept it for the review, rather
//! than rebuilt from what the page shows, and with the two things the replay
//! needs that a table typed in by hand does not carry (`History`): when each
//! quad a seat declared itself was declared, and what the deciding player
//! drew on the turn they declared riichi. Who the players are follows from
//! the hand number, as it does for any table. Read back, it gives the very
//! planes and mask the review asked the network about.

use super::*;
use riichi_core::game::Player;

/// The table at a decision, from the deciding seat's side. Only that seat's
/// tiles are written: the other hands are what the table hid, as they are
/// at a physical table, and so is whether another seat let a winning tile
/// pass. Refused for a hand nobody is deciding in.
pub(super) fn written(hand: &Hand, seat: Wind) -> Result<Position, &'static str> {
    let acting = match hand.phase {
        Phase::Act if hand.turn == seat => true,
        Phase::CallWindow if hand.turn != seat => false,
        _ => return Err("Nobody decides at this point of the hand"),
    };
    let history = History::from_log(hand, seat);
    let named = |tile: Option<Tile>| tile.map(|tile| tile.to_string());
    Ok(Position {
        seat: seat.index(),
        turn: hand.turn.index(),
        phase: if acting { "act" } else { "call" }.into(),
        round: hand.round.index(),
        kyoku: hand.kyoku,
        counters: hand.counters,
        riichi_sticks: hand.riichi_sticks,
        wall: hand.wall.remaining(),
        indicators: hand
            .wall
            .dora_indicators()
            .iter()
            .map(ToString::to_string)
            .collect(),
        players: std::array::from_fn(|index| seat_written(hand, seat, Wind::ALL[index], &history)),
        drawn: named(hand.drawn.filter(|_| acting)),
        pending: named(
            hand.pending_discard
                .filter(|_| !acting)
                .map(|(_, tile)| tile),
        ),
        pending_kind: match hand.robbable_quad.filter(|_| !acting) {
            Some(_) if hand.robbing_concealed => "concealed-kan",
            Some(_) => "extended-kan",
            None => "discard",
        }
        .into(),
        just_claimed: named(hand.just_claimed.filter(|_| acting)),
        // The engine clears the mark only at the next draw, so it outlives
        // the declaring seat's discard, and a claim of that discard too. A
        // position carries it only while the acting seat holds the
        // replacement it drew, which is all it changes; a seat that has just
        // claimed a tile drew nothing.
        after_quad: acting && hand.after_quad && hand.drawn.is_some(),
        first_turns: hand.first_turns_unbroken,
    })
}

/// One seat as the deciding seat sees it, with what the hand's log says of
/// it: the turn of each of its own quads, and for the deciding seat in
/// riichi the tile it drew on the turn it declared.
fn seat_written(hand: &Hand, deciding: Wind, who: Wind, history: &History) -> Seat {
    let player = &hand.players[who.index()];
    let mine = who == deciding;
    let quads = &history.quads[who.index()];
    Seat {
        hand: if mine {
            player.hand.tiles().map(|tile| tile.to_string()).collect()
        } else {
            Vec::new()
        },
        melds: player
            .melds
            .iter()
            .enumerate()
            .map(|(at, meld)| Set {
                kind: match meld.kind {
                    MeldKind::Chii => "chii",
                    MeldKind::Pon => "pon",
                    MeldKind::ClaimedKan => "kan",
                    MeldKind::ExtendedKan => "extended-kan",
                    MeldKind::ConcealedKan => "concealed-kan",
                }
                .into(),
                tile: meld.tile.to_string(),
                from: match meld.from {
                    ClaimedFrom::SelfDrawn => 0,
                    ClaimedFrom::Right => 1,
                    ClaimedFrom::Across => 2,
                    ClaimedFrom::Left => 3,
                },
                order: quads.get(at).copied().flatten(),
            })
            .collect(),
        discards: player
            .discards
            .iter()
            .map(|discard| Thrown {
                tile: discard.tile.to_string(),
                order: discard.order,
                drawn: discard.drawn,
                riichi: discard.riichi,
                claimed: discard.claimed,
            })
            .collect(),
        score: player.score,
        riichi: match player.riichi {
            Riichi::None => "none",
            Riichi::Declared => "riichi",
            Riichi::Double => "double",
        }
        .into(),
        ippatsu: player.ippatsu,
        furiten: mine && furiten_unshown(hand, who, player),
        riichi_draw: history
            .riichi_draw
            .filter(|_| mine && player.has_riichi())
            .map(|tile| tile.to_string()),
    }
}

/// Whether a seat may not win on a discard for a reason its own discards
/// do not show: it let a winning tile pass, which bars it until its next
/// draw, or for the rest of the hand once it has declared riichi. A
/// position marks only that (`furiten`), as the editor does after a pass,
/// because the validation finds the furiten its discards show by itself,
/// on the same shape it reads here: the hand before the tile just drawn.
fn furiten_unshown(hand: &Hand, who: Wind, player: &Player) -> bool {
    let mut shape = player.clone();
    if hand.turn == who {
        if let Some(drawn) = hand.drawn {
            shape.hand.remove(drawn);
        }
    }
    let waits = shape.waits();
    let shown = player
        .discards
        .iter()
        .any(|discard| waits.count(discard.tile) > 0);
    player.temporary_furiten || (player.furiten && !shown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{describe_action, may_reach_from, mortal_mask_of, Game};
    use riichi_core::game::{Action, Call};

    /// How the player plays a whole game in these tests.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Plays {
        /// As the review check's player: every win and every riichi, and
        /// otherwise the tile just drawn and a pass.
        Draws,
        /// The same, with every quad and every set it may take besides.
        Calls,
        /// As the club player plays, which comes to riichi far more often
        /// than a player throwing its draws.
        Club,
    }

    /// The move the review check's player makes at a turn: a win, then a
    /// riichi, then with `calls` a quad of its own, then the tile just
    /// drawn, then the first tile it may throw.
    fn turn_move(hand: &Hand, calls: bool) -> Action {
        let legal = hand.legal_actions();
        let first = |wanted: &dyn Fn(&Action) -> bool| legal.iter().copied().find(|a| wanted(a));
        first(&|a| *a == Action::Tsumo)
            .or_else(|| first(&|a| matches!(a, Action::Riichi(_))))
            .or_else(|| {
                first(&|a| calls && matches!(a, Action::ConcealedKan(_) | Action::ExtendedKan(_)))
            })
            .or_else(|| first(&|a| hand.drawn.is_some_and(|drawn| *a == Action::Discard(drawn))))
            .or_else(|| first(&|a| matches!(a, Action::Discard(_))))
            .expect("a turn always offers a move")
    }

    /// Plays a whole game against `opponents` as `plays` says, and shows
    /// `look` each hand once it is over and before the next is dealt, which
    /// is when a review reads the hand's decisions. The club player is the
    /// player's own adviser, as the page asks it for a move.
    fn whole_game(seed: f64, opponents: &str, plays: Plays, mut look: impl FnMut(&Game)) {
        let mut game = Game::new(seed, Some(opponents.into()));
        let player = game.player;
        if plays == Plays::Club {
            game.bots[player].style = riichi_core::bot::Style::club();
        }
        let calls = plays == Plays::Calls;
        for _ in 0..20_000 {
            game.play_on().unwrap();
            match game.hand.phase {
                Phase::Over => {
                    look(&game);
                    game.next_hand().unwrap();
                    if game.table.finished {
                        return;
                    }
                }
                Phase::Act => {
                    let action = if plays == Plays::Club {
                        game.bots[player].act(&game.hand)
                    } else {
                        turn_move(&game.hand, calls)
                    };
                    let chosen = describe_action(action);
                    game.choose(&chosen.kind, chosen.tile).unwrap();
                }
                Phase::CallWindow => {
                    let offered = game
                        .hand
                        .legal_calls()
                        .into_iter()
                        .find(|(seat, _)| *seat == game.seat)
                        .map(|(_, offered)| offered)
                        .expect("play stops in a window only for the player");
                    let call = if plays == Plays::Club {
                        game.bots[player].call(&game.hand, game.seat, &offered)
                    } else {
                        [Call::Ron, Call::Kan, Call::Pon]
                            .into_iter()
                            .chain(
                                offered
                                    .iter()
                                    .copied()
                                    .filter(|c| matches!(c, Call::Chii(_))),
                            )
                            .find(|c| offered.contains(c) && (calls || *c == Call::Ron))
                            .unwrap_or(Call::Pass)
                    };
                    let chosen = describe_call(call);
                    game.choose(&chosen.kind, chosen.tile).unwrap();
                }
                Phase::Draw => unreachable!("play goes on through every draw"),
            }
        }
        panic!("seed {seed}: the game did not end");
    }

    /// What a decision's written position reads back as, through JSON as
    /// the page passes it: the hand, the seat, and what the replay reads.
    fn read_back(game: &Game, index: usize) -> (Hand, Wind, History) {
        let decision = &game.decisions[index];
        let position =
            written(&decision.position, decision.seat).expect("a recorded decision is a decision");
        let text = serde_json::to_string(&position).unwrap();
        let position: Position = serde_json::from_str(&text).unwrap();
        let (hand, seat) = position
            .build()
            .unwrap_or_else(|error| panic!("{error}: {text}"));
        (hand, seat, position.history())
    }

    /// Everything the trained adviser is asked at a decision: the planes,
    /// the mask, and where a riichi may be declared the second question's
    /// planes and mask; and the listed choices its answers are read into.
    #[derive(PartialEq, Debug)]
    struct Asked {
        planes: Vec<f32>,
        mask: Vec<bool>,
        reach: Option<(Vec<f32>, Vec<bool>)>,
        choices: Vec<(Option<usize>, String, Option<String>, bool)>,
    }

    fn listed(hand: &Hand, seat: Wind) -> Vec<(Option<usize>, String, Option<String>, bool)> {
        choices(hand, seat)
            .into_iter()
            .map(|choice| {
                (
                    choice.index,
                    choice.kind,
                    choice.tile,
                    choice.causes_furiten,
                )
            })
            .collect()
    }

    /// What a written position asks, read back as `PhysicalAnalysis` reads
    /// it, through the very function that answers the page: Mortal's state
    /// from the replay of the position, and for a riichi's second question
    /// that state once the seat has declared.
    fn asked(hand: &Hand, seat: Wind, history: &History) -> Option<Asked> {
        let reach = if may_reach_from(hand, seat) {
            Some((
                replayed_planes(hand, seat, history, true)?,
                mortal_mask_of(hand, seat, true),
            ))
        } else {
            None
        };
        Some(Asked {
            planes: replayed_planes(hand, seat, history, false)?,
            mask: mortal_mask_of(hand, seat, false),
            reach,
            choices: listed(hand, seat),
        })
    }

    /// The same as the review asked it: Mortal's state from the events the
    /// game told it, and the decision's own hand.
    fn asked_in_play(game: &Game, index: usize) -> Asked {
        let decision = &game.decisions[index];
        let (hand, seat) = (&decision.position, decision.seat);
        let player = game.hand_seating[seat.index()];
        Asked {
            planes: game.mortal_planes_at(decision.told, player, false),
            mask: mortal_mask_of(hand, seat, false),
            reach: may_reach_from(hand, seat).then(|| {
                (
                    game.mortal_planes_at(decision.told, player, true),
                    mortal_mask_of(hand, seat, true),
                )
            }),
            choices: listed(hand, seat),
        }
    }

    /// The planes of thirty-four on which two observations differ.
    fn differing(want: &[f32], got: &[f32]) -> Vec<usize> {
        (0..want.len().min(got.len()) / 34)
            .filter(|plane| want[plane * 34..plane * 34 + 34] != got[plane * 34..plane * 34 + 34])
            .collect()
    }

    /// The gate a review that edits a decision's table stands on: the table
    /// written unedited asks the trained adviser exactly what the review
    /// asked, every one of the thousand and twelve planes, the mask, a
    /// riichi's second question and the listed choices, at every decision
    /// of whole games. The review check's game (seed 5, Beginner opponents,
    /// no calls) is among them; the games that call every set and declare
    /// every quad reach the claims, the quads and the tied scores that a
    /// typed-in table read wrong until it carried each quad's turn and was
    /// numbered by its hand number; seed 17 has the player claim a tile let
    /// go after a quad's replacement; and the games the club player plays
    /// come to riichi questions after the first hand, where the seat's
    /// player number is not its seat's.
    #[test]
    fn every_decision_written_as_a_position_asks_what_the_review_asked() {
        /// How many decisions reached each thing a table typed in by hand
        /// read wrong or could not say, so the test is known to cover it.
        #[derive(Default, Debug)]
        struct Reached {
            decisions: usize,
            answering: usize,
            quad_turns: usize,
            tied: usize,
            renumbered: usize,
            riichi_questions: usize,
            renumbered_riichi_questions: usize,
            riichi_draws: usize,
            claims_after_a_replacement: usize,
        }
        let mut reached = Reached::default();
        let mut wrong = Vec::new();
        let games: [(f64, &str, Plays); 7] = [
            (5.0, "beginner", Plays::Draws),
            (16.0, "beginner", Plays::Calls),
            (31.0, "club", Plays::Calls),
            (81.0, "club", Plays::Calls),
            (17.0, "club", Plays::Calls),
            (5.0, "club", Plays::Club),
            (16.0, "club", Plays::Club),
        ];
        for (seed, opponents, plays) in games {
            let mut in_game = 0;
            whole_game(seed, opponents, plays, |game| {
                for index in 0..game.decisions.len() {
                    let at = format!(
                        "seed {seed} against {opponents}, hand {}, decision {index}",
                        game.table.hands_played
                    );
                    // Who sits where is the hand number's to say.
                    assert_eq!(
                        History::of_kyoku(game.decisions[index].position.kyoku).seating,
                        game.hand_seating,
                        "{at}"
                    );
                    let (hand, seat, history) = read_back(game, index);
                    let want = asked_in_play(game, index);
                    match asked(&hand, seat, &history) {
                        None => wrong.push(format!("{at}: the replay refused it")),
                        Some(got) if got != want => {
                            let planes = differing(&want.planes, &got.planes);
                            wrong.push(format!(
                                "{at}: planes {:?}, same mask {}, same reach {}, same choices {}",
                                &planes[..planes.len().min(8)],
                                got.mask == want.mask,
                                got.reach == want.reach,
                                got.choices == want.choices
                            ));
                        }
                        Some(_) => {}
                    }
                    let scores = hand.players.each_ref().map(|player| player.score);
                    let own = &game.decisions[index].position;
                    in_game += 1;
                    reached.decisions += 1;
                    reached.answering += usize::from(hand.phase == Phase::CallWindow);
                    reached.quad_turns +=
                        usize::from(history.quads.iter().flatten().any(Option::is_some));
                    reached.tied +=
                        usize::from((1..4).any(|at| scores[..at].contains(&scores[at])));
                    reached.renumbered += usize::from(history.seating != [0, 1, 2, 3]);
                    reached.riichi_questions += usize::from(want.reach.is_some());
                    reached.renumbered_riichi_questions += usize::from(
                        want.reach.is_some() && history.seating[seat.index()] != seat.index(),
                    );
                    reached.riichi_draws += usize::from(history.riichi_draw.is_some());
                    reached.claims_after_a_replacement += usize::from(
                        own.phase == Phase::Act && own.after_quad && own.drawn.is_none(),
                    );
                }
            });
            if (seed, plays) == (5.0, Plays::Draws) {
                assert_eq!(in_game, 139, "the review check's game has 139 decisions");
            }
        }
        println!("{reached:?}; {} wrong", wrong.len());
        assert!(
            wrong.is_empty(),
            "{} wrong: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(6)]
        );
        assert!(reached.answering > 50, "{reached:?}");
        assert!(reached.quad_turns > 0, "{reached:?}");
        assert!(reached.tied > 50, "{reached:?}");
        assert!(reached.renumbered > 100, "{reached:?}");
        assert!(reached.renumbered_riichi_questions > 3, "{reached:?}");
        assert!(reached.riichi_draws > 0, "{reached:?}");
        assert!(reached.claims_after_a_replacement > 0, "{reached:?}");
    }

    /// The position is the table as the deciding seat saw it: its own tiles
    /// and what lies face up, and none of what the table hid, whether
    /// another seat's tiles or a winning tile it let pass.
    #[test]
    fn a_written_position_holds_only_what_the_deciding_seat_could_see() {
        let mut checked = 0;
        whole_game(31.0, "club", Plays::Calls, |game| {
            for decision in &game.decisions {
                let position = written(&decision.position, decision.seat).unwrap();
                for (index, seat) in position.players.iter().enumerate() {
                    if index == decision.seat.index() {
                        assert!(!seat.hand.is_empty());
                    } else {
                        assert!(seat.hand.is_empty() && !seat.furiten);
                    }
                }
                checked += 1;
            }
        });
        assert!(checked > 50, "only {checked} decisions");
    }

    /// A seat that let a winning tile pass after declaring riichi may not
    /// win on a discard for the rest of the hand, though nothing in its
    /// river shows it, so the position marks it. A seat that has thrown one
    /// of its own waits is furiten by its river, which the validation reads
    /// again, so that is left unmarked, unless it has also let a winning
    /// tile pass since its last draw.
    #[test]
    fn furiten_is_marked_only_where_the_river_does_not_show_it() {
        let tile = |text: &str| -> Tile { text.parse().unwrap() };
        let mut hand = Hand::deal(&mut Rng::from_seed(3), Wind::East, 1, 0, 0, [25000; 4]);
        hand.phase = Phase::CallWindow;
        hand.drawn = None;
        hand.first_turns_unbroken = false;
        let east = &mut hand.players[Wind::East.index()];
        east.hand = TileSet::new();
        east.discards = vec![Discard {
            tile: tile("9p"),
            order: 0,
            drawn: true,
            riichi: false,
            claimed: false,
        }];
        hand.pending_discard = Some((Wind::East, tile("9p")));
        hand.discards_made = 1;
        // South waits on 3p and 6p, and has thrown neither.
        let south = &mut hand.players[Wind::South.index()];
        south.hand = "123m456m789m45p11s".parse().unwrap();
        south.discards.clear();
        south.riichi = Riichi::Declared;
        south.furiten = true;
        let marked =
            |hand: &Hand| written(hand, Wind::South).unwrap().players[Wind::South.index()].furiten;
        assert!(
            marked(&hand),
            "furiten from a pass, which the river does not show"
        );
        hand.players[Wind::South.index()].discards.push(Discard {
            tile: tile("3p"),
            order: 1,
            drawn: true,
            riichi: true,
            claimed: false,
        });
        assert!(!marked(&hand), "furiten the river shows");
        hand.players[Wind::South.index()].temporary_furiten = true;
        assert!(
            marked(&hand),
            "a pass since the last draw bars a win until the next"
        );
    }

    /// Only a decision has a position: a hand between turns has nobody
    /// deciding, and a seat that is not the one to act has no turn to take.
    #[test]
    fn a_hand_nobody_is_deciding_in_is_refused() {
        let mut hand = Hand::deal(&mut Rng::from_seed(3), Wind::East, 1, 0, 0, [25000; 4]);
        assert!(written(&hand, Wind::East).is_ok());
        assert!(written(&hand, Wind::South).is_err());
        hand.phase = Phase::Draw;
        assert!(written(&hand, Wind::East).is_err());
    }
}
