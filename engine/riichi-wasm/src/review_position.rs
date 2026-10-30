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
pub(super) mod tests {
    use super::*;
    use crate::renamed_rows::not_renamed;
    use crate::{
        describe_action, may_reach_from, mortal_mask_of, planes_of, Game, RecordedDecision,
        MORTAL_VERSION,
    };
    use riichi_core::game::{Action, Call};
    use riichi_core::hand::Meld;
    use riichi_core::mjai::Event;

    /// How the player plays a whole game in these tests.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(in crate::analysis) enum Plays {
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
    pub(in crate::analysis) fn whole_game(
        seed: f64,
        opponents: &str,
        plays: Plays,
        mut look: impl FnMut(&Game),
    ) {
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
    /// planes and mask; the renamed game's planes where those are asked
    /// too; and the listed choices its answers are read into.
    #[derive(PartialEq, Debug)]
    struct Asked {
        planes: Vec<f32>,
        mask: Vec<bool>,
        reach: Option<(Vec<f32>, Vec<bool>)>,
        renamed: Option<Renamed>,
        choices: Vec<(Option<usize>, String, Option<String>, bool)>,
    }

    /// The renamed game's planes at a decision, and where a riichi may be
    /// declared its planes once it is.
    type Renamed = (Vec<f32>, Option<Vec<f32>>);

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
    /// that state once the seat has declared; with `renamed` the renamed
    /// game's too.
    fn asked(hand: &Hand, seat: Wind, history: &History, renamed: bool) -> Option<Asked> {
        let reachable = may_reach_from(hand, seat);
        let reach = if reachable {
            Some((
                replayed_planes(hand, seat, history, true, false)?,
                mortal_mask_of(hand, seat, true),
            ))
        } else {
            None
        };
        let renamed = if renamed {
            let reach = if reachable {
                Some(replayed_planes(hand, seat, history, true, true)?)
            } else {
                None
            };
            Some((replayed_planes(hand, seat, history, false, true)?, reach))
        } else {
            None
        };
        Some(Asked {
            planes: replayed_planes(hand, seat, history, false, false)?,
            mask: mortal_mask_of(hand, seat, false),
            reach,
            renamed,
            choices: listed(hand, seat),
        })
    }

    /// The same as the review asked it: Mortal's state from the events the
    /// game told it, and the decision's own hand.
    fn asked_in_play(game: &Game, index: usize, renamed: bool) -> Asked {
        let decision = &game.decisions[index];
        let (hand, seat) = (&decision.position, decision.seat);
        let reachable = may_reach_from(hand, seat);
        Asked {
            planes: game.mortal_planes_at(decision, false, false),
            mask: mortal_mask_of(hand, seat, false),
            reach: reachable.then(|| {
                (
                    game.mortal_planes_at(decision, true, false),
                    mortal_mask_of(hand, seat, true),
                )
            }),
            renamed: renamed.then(|| {
                (
                    game.mortal_planes_at(decision, false, true),
                    reachable.then(|| game.mortal_planes_at(decision, true, true)),
                )
            }),
            choices: listed(hand, seat),
        }
    }

    /// The deciding player's state told every event of the game, the same of
    /// the renamed game, and how many events they have been told: the replay
    /// from the game's opening that a review made before it replayed the
    /// decision's hand alone, kept as the reference the two agree.
    struct WholeGame {
        plain: riichi::state::PlayerState,
        renamed: riichi::state::PlayerState,
        told: usize,
    }

    impl WholeGame {
        fn new(player: usize) -> WholeGame {
            WholeGame {
                plain: riichi::state::PlayerState::new(player as u8),
                renamed: riichi::state::PlayerState::new(player as u8),
                told: 0,
            }
        }

        /// What the review asked at `decision` as the whole game told reads
        /// it: the planes, and where a riichi may be declared the planes
        /// once it is; with `renamed` the renamed game's too.
        fn asked_at(
            &mut self,
            game: &Game,
            decision: &RecordedDecision,
            renamed: bool,
        ) -> (Vec<f32>, Option<Vec<f32>>, Option<Renamed>) {
            for json in &game.mortal_events[self.told..decision.told] {
                let mut event = serde_json::from_str::<MortalEvent>(json).expect("Mortal reads it");
                self.plain.update(&event).expect("Mortal follows the game");
                event.augment();
                self.renamed
                    .update(&event)
                    .expect("Mortal follows the renamed game");
            }
            self.told = decision.told;
            let (player, reachable) = (
                game.player,
                may_reach_from(&decision.position, decision.seat),
            );
            let read = |state: &riichi::state::PlayerState, after_reach: bool| {
                planes_of(state, player, after_reach)
            };
            (
                read(&self.plain, false),
                reachable.then(|| read(&self.plain, true)),
                renamed.then(|| {
                    (
                        read(&self.renamed, false),
                        reachable.then(|| read(&self.renamed, true)),
                    )
                }),
            )
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
    /// asked, every one of the thousand and twelve planes, the renamed
    /// game's too, the mask, a riichi's second question and the listed
    /// choices, at every decision of whole games. The review check's game
    /// (seed 5, Beginner opponents, no calls) is among them; the games that
    /// call every set and declare every quad reach the claims, the quads and
    /// the tied scores that a typed-in table read wrong until it carried
    /// each quad's turn and was numbered by its hand number; seed 17 has the
    /// player claim a tile let go after a quad's replacement; and the games
    /// the club player plays come to riichi questions after the first hand,
    /// where the seat's player number is not its seat's.
    ///
    /// The review itself replays the decision's hand alone, from its own
    /// deal, and must read as the whole game told reads. Every sixth
    /// decision and every riichi question are asked of the renamed game
    /// too, which the written table and the whole game told must read
    /// alike, and which outside the efficiency lookahead reads as the game
    /// renamed; only some, as an encoding costs about eight milliseconds
    /// unoptimised.
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
            /// Asked of the renamed game too.
            renamed: usize,
            /// Of those, where its lookahead reads otherwise than the
            /// game's renamed, as it may.
            lookahead_renamed_otherwise: usize,
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
            let mut whole: Option<WholeGame> = None;
            whole_game(seed, opponents, plays, |game| {
                let whole = whole.get_or_insert_with(|| WholeGame::new(game.player));
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
                    let decision = &game.decisions[index];
                    // The review's replay opens on the hand's own deal.
                    let deal = |json: &String| json.starts_with(r#"{"type":"start_kyoku""#);
                    let replayed = &game.mortal_events[decision.began..decision.told];
                    if !replayed.first().is_some_and(deal) || replayed[1..].iter().any(deal) {
                        wrong.push(format!("{at}: the review does not replay its own hand"));
                    }
                    let renamed = index.is_multiple_of(6)
                        || may_reach_from(&decision.position, decision.seat);
                    let want = asked_in_play(game, index, renamed);
                    let (planes, reach, named) = whole.asked_at(game, decision, renamed);
                    if (&planes, reach.as_ref(), named.as_ref())
                        != (
                            &want.planes,
                            want.reach.as_ref().map(|(planes, _)| planes),
                            want.renamed.as_ref(),
                        )
                    {
                        wrong.push(format!(
                            "{at}: the hand replayed from its deal reads otherwise than the game"
                        ));
                    }
                    if let Some((named, named_reach)) = &want.renamed {
                        let mut otherwise = not_renamed(&want.planes, named, false);
                        if let (Some((planes, _)), Some(named)) = (&want.reach, named_reach) {
                            otherwise.extend(not_renamed(planes, named, false));
                        }
                        if !otherwise.is_empty() {
                            wrong.push(format!(
                                "{at}: renamed rows {:?} are not the game's rows renamed",
                                &otherwise[..otherwise.len().min(8)]
                            ));
                        }
                        reached.renamed += 1;
                        reached.lookahead_renamed_otherwise +=
                            usize::from(!not_renamed(&want.planes, named, true).is_empty());
                    }
                    match asked(&hand, seat, &history, renamed) {
                        None => wrong.push(format!("{at}: the replay refused it")),
                        Some(got) if got != want => {
                            let planes = differing(&want.planes, &got.planes);
                            wrong.push(format!(
                                "{at}: planes {:?}, same mask {}, same reach {}, same renamed {}, same choices {}",
                                &planes[..planes.len().min(8)],
                                got.mask == want.mask,
                                got.reach == want.reach,
                                got.renamed == want.renamed,
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
        assert!(reached.renamed > 150, "{reached:?}");
        assert!(reached.lookahead_renamed_otherwise > 0, "{reached:?}");
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

    /// A hand told event by event from the four hands dealt, kept as the
    /// engine keeps a hand it plays: what each seat holds, its sets and its
    /// river, the indicators turned, and the draws from the live wall and
    /// the replacements. It is the first hand of a game, everyone level.
    fn told_event_by_event(dealt: [&str; 4], events: Vec<Event>) -> Hand {
        let tile = |text: &str| -> Tile { text.parse().unwrap() };
        let mut hand = Hand::deal(&mut Rng::from_seed(1), Wind::East, 1, 0, 0, [25000; 4]);
        let hands: [Vec<Tile>; 4] = dealt.map(|text| {
            let held: TileSet = text.parse().unwrap();
            held.tiles().collect()
        });
        for (player, held) in hand.players.iter_mut().zip(&hands) {
            player.hand = TileSet::from_tiles(held.iter().copied());
            player.melds.clear();
            player.discards.clear();
        }
        let mut indicators = vec![tile("3m")];
        hand.log = vec![Event::StartKyoku {
            round: Wind::East,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            indicator: indicators[0],
            scores: [25000; 4],
            hands: hands.clone(),
        }];
        let (mut draws, mut replacements, mut made) = (0usize, 0usize, 0u32);
        let (mut after_quad, mut drawn) = (false, None);
        for event in events {
            match &event {
                Event::Tsumo { actor, tile } => {
                    hand.players[actor.index()].hand.add(*tile);
                    if after_quad {
                        replacements += 1;
                    } else {
                        draws += 1;
                    }
                    after_quad = false;
                    drawn = Some(*tile);
                }
                Event::Dahai { actor, tile, .. } => {
                    let player = &mut hand.players[actor.index()];
                    assert!(player.hand.remove(*tile), "{actor:?} holds no {tile}");
                    player.discards.push(Discard {
                        tile: *tile,
                        order: made,
                        drawn: drawn == Some(*tile),
                        riichi: false,
                        claimed: false,
                    });
                    made += 1;
                    drawn = None;
                }
                Event::Pon {
                    actor,
                    target,
                    tile,
                    consumed,
                } => {
                    for used in consumed {
                        assert!(hand.players[actor.index()].hand.remove(*used));
                    }
                    let taken = hand.players[target.index()].discards.last_mut().unwrap();
                    taken.claimed = true;
                    let from = match (target.index() + 4 - actor.index()) % 4 {
                        3 => ClaimedFrom::Left,
                        2 => ClaimedFrom::Across,
                        _ => ClaimedFrom::Right,
                    };
                    hand.players[actor.index()]
                        .melds
                        .push(Meld::pon(*tile, from));
                    drawn = None;
                }
                Event::Ankan { actor, consumed } => {
                    for used in consumed {
                        assert!(hand.players[actor.index()].hand.remove(*used));
                    }
                    hand.players[actor.index()]
                        .melds
                        .push(Meld::concealed_kan(consumed[0]));
                    after_quad = true;
                    drawn = None;
                }
                Event::Kakan { actor, tile, .. } => {
                    let player = &mut hand.players[actor.index()];
                    assert!(player.hand.remove(*tile));
                    let triplet = player
                        .melds
                        .iter_mut()
                        .find(|meld| meld.kind == MeldKind::Pon && meld.tile == *tile)
                        .expect("an added quad extends a triplet");
                    triplet.kind = MeldKind::ExtendedKan;
                    after_quad = true;
                    drawn = None;
                }
                Event::Dora { indicator } => indicators.push(*indicator),
                _ => {}
            }
            hand.log.push(event);
        }
        hand.wall = Wall::for_analysis(70 - draws - replacements, &indicators, replacements)
            .expect("a wall the hand could have");
        hand.discards_made = made;
        hand.first_turns_unbroken = false;
        hand
    }

    /// South claims East's 5 bamboo; later, on one turn, it declares a
    /// concealed quad of 2 circles and adds its fourth 5 bamboo to the
    /// triplet, which West, waiting on the 5 bamboo, may rob. With `apart`
    /// the concealed quad was declared on South's turn before, and
    /// discarded from.
    fn robbing_after_another_quad(apart: bool) -> Hand {
        let tile = |text: &str| -> Tile { text.parse().unwrap() };
        let (east, south, west, north) = (Wind::East, Wind::South, Wind::West, Wind::North);
        let tsumo = |actor: Wind, t: &str| Event::Tsumo {
            actor,
            tile: tile(t),
        };
        let dahai = |actor: Wind, t: &str, drawn: bool| Event::Dahai {
            actor,
            tile: tile(t),
            drawn,
        };
        let ankan = || Event::Ankan {
            actor: south,
            consumed: vec![tile("2p"); 4],
        };
        let kakan = || Event::Kakan {
            actor: south,
            tile: tile("5s"),
            consumed: vec![tile("5s"); 3],
        };
        let turned = || Event::Dora {
            indicator: tile("8p"),
        };
        let mut events = vec![
            tsumo(east, "5z"),
            dahai(east, "5s", false),
            Event::Pon {
                actor: south,
                target: east,
                tile: tile("5s"),
                consumed: vec![tile("5s"), tile("5s")],
            },
            dahai(south, "9m", false),
            tsumo(west, "1s"),
            dahai(west, "1s", true),
            tsumo(north, "5z"),
            dahai(north, "5z", true),
            tsumo(east, "9s"),
            dahai(east, "9s", true),
        ];
        let rest = [
            tsumo(west, "2z"),
            dahai(west, "2z", true),
            tsumo(north, "1z"),
            dahai(north, "1z", true),
            tsumo(east, "2m"),
            dahai(east, "2m", true),
            tsumo(south, "6s"),
        ];
        if apart {
            events.extend([tsumo(south, "3s"), ankan(), turned(), tsumo(south, "4s")]);
            events.push(dahai(south, "4s", true));
            events.extend(rest);
            events.push(kakan());
        } else {
            events.extend([tsumo(south, "3s"), dahai(south, "3s", true)]);
            events.extend(rest);
            events.extend([ankan(), turned(), tsumo(south, "4s"), kakan()]);
        }
        let mut hand = told_event_by_event(
            [
                "5s1p1p3p3p4z4z6m6m7s8s9s2m",
                "2p2p2p2p5s5s5s9m7z7z6z6z1s",
                "1m2m3m4m5m6m7m8m9m1z1z4s6s",
                "2s2s3s3s4p4p6p6p7p7p2z2z3z",
            ],
            events,
        );
        hand.turn = south;
        hand.phase = Phase::CallWindow;
        hand.pending_discard = Some((south, tile("5s")));
        hand.robbable_quad = Some(tile("5s"));
        hand.robbing_concealed = false;
        hand.drawn = None;
        hand.just_claimed = None;
        hand.after_quad = !apart;
        hand
    }

    /// Mortal's planes for a hand really played, told the hand's own log.
    fn told_planes(hand: &Hand, seat: Wind) -> Vec<f32> {
        let mut state = riichi::state::PlayerState::new(seat.index() as u8);
        let start = Event::StartGame {
            names: std::array::from_fn(|player| format!("player {player}")),
        };
        for json in std::iter::once(&start)
            .chain(&hand.log)
            .map(|event| event.to_json([0, 1, 2, 3]))
        {
            let event = serde_json::from_str::<MortalEvent>(&json).expect("Mortal reads it");
            state.update(&event).expect("Mortal follows the hand");
        }
        state
            .encode_obs(MORTAL_VERSION, false)
            .0
            .iter()
            .copied()
            .collect()
    }

    /// A quad open to robbery is being declared now, and so is another the
    /// same seat declared earlier on the same turn, with no discard since.
    /// Written for the seat that may rob it, both are recorded as declared
    /// on the turn being played and told last, the one open to robbery
    /// after the other, and the replay reads as the hand did. Unrecorded,
    /// the earlier one was given the seat's latest turn before, which read
    /// differently. Declared a turn apart, the earlier quad keeps its turn.
    #[test]
    fn two_quads_of_the_turn_being_played_are_told_in_the_order_they_were_declared() {
        for apart in [true, false] {
            let hand = robbing_after_another_quad(apart);
            let seat = Wind::West;
            assert!(
                hand.legal_calls()
                    .iter()
                    .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron)),
                "West may rob the quad"
            );
            let position = written(&hand, seat).expect("West decides");
            let orders: Vec<Option<u32>> = position.players[Wind::South.index()]
                .melds
                .iter()
                .map(|set| set.order)
                .collect();
            // The triplet the fourth 5 bamboo extends is South's first set.
            assert_eq!(
                orders,
                [Some(9), Some(if apart { 5 } else { 9 })],
                "apart {apart}"
            );
            let text = serde_json::to_string(&position).unwrap();
            let position: Position = serde_json::from_str(&text).unwrap();
            let (built, _) = position
                .build()
                .unwrap_or_else(|error| panic!("{error}: {text}"));
            let got = replayed_planes(&built, seat, &position.history(), false, false)
                .expect("the replay builds");
            assert_eq!(
                differing(&told_planes(&hand, seat), &got),
                Vec::<usize>::new(),
                "apart {apart}"
            );
            assert_eq!(
                mortal_mask_of(&built, seat, false),
                mortal_mask_of(&hand, seat, false)
            );
        }
    }
}
