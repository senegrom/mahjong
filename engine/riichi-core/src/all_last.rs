//! The last hand, and what a move there could still buy.
//!
//! South 4 is the last hand of a game: there is no extension round and no
//! agari-yame (see [`crate::table`]), so a win by anyone but the dealer ends
//! the game, and the places are then the points. The dealer is left out: a
//! dealer who wins keeps the deal and South 4 is played again (EMA 2025
//! section 3.4.5), so no win of theirs ends the game.
//!
//! A non-dealer who may declare riichi there has a question a riichi cannot
//! always answer better than silence: if riichi would add no win, and every
//! win the hand can make without it leaves them alone in first, riichi adds
//! no place, and costs a thousand points and the news that the hand is ready.
//! [`Hand::all_last_dama`] answers it for each kind of tile the turn player
//! holds: what discarding it without declaring would leave ([`Dama`]).
//!
//! The question is put to the hand as the plain discard would leave it: the
//! tile out of the hand and in the pond, nothing declared, nothing situational
//! left over (no blessing, since the player has now discarded, and no quad just
//! declared). Its waits are those of [`crate::game::Player::waits`], and a wait
//! counts only while at least one copy is left that the player cannot see
//! ([`Hand::unseen_by`]): a dead wait wins nothing either way. Each live wait
//! is then won every way the hand could win it without riichi: by ron off each
//! of the three others, unless the discard leaves the hand furiten, when
//! neither riichi nor silence could ron (EMA section 3.3.9), and by tsumo,
//! which a concealed hand can always declare. Each win is scored by the
//! engine's own scorer ([`Hand::would_win`]) and settled by its own settlement,
//! the one a real win goes through (the discarder pays a ron, the dealer and
//! the others their shares of a tsumo, the counters with them, and the winner
//! collects every riichi bet on the table), so the answer cannot drift from
//! the rules the game is played by.
//!
//! First means strictly first in points: tied players share the uma of the
//! places they cover (EMA section 3.7.1), and a win empties the table of
//! bets, so none is left over for the leader at the end.
//!
//! The answer holds for the rest of the hand, not only at once. Until the
//! player wins, the others' points can only fall: a riichi bet goes into the
//! pool the winner collects, and comes back only if its declaration is won on
//! (EMA section 3.3.10); a new kan dora only adds han, and the counters do not
//! change within a hand. What is not modelled is two players winning on one
//! discard (EMA section 3.3.1), where the other winner is paid as well and may
//! take the bets: rare, and accepted.
//!
//! # A win that leaves the player last
//!
//! [`Hand::all_last_decline`] asks a non-dealer offered a win in the last
//! hand, a self-draw at their own turn or a ron on what is on the table,
//! whether taking it would leave them strictly below every other player
//! ([`Decline`]). The win is played on a copy through the engine's own
//! [`Hand::act`] or [`Hand::resolve_calls`], which score it with
//! [`Hand::would_win`] and settle it as a real one: the counters, every bet on
//! the table, and the bet of a riichi declared with the very tile won on,
//! which goes back to its owner (EMA section 3.3.10). A riichi winner's ura
//! dora are turned only once the win is declared, so the copy never shows
//! the real ones: its wall shows the same dora and holds as many tiles to
//! draw, with under each dora indicator an ura indicator of the kind the
//! winning hand holds most copies of, the most the ura could add. A larger
//! win only lifts the winner and lowers the payers, so a riichi win that
//! leaves the player alone in last at the most leaves them there whatever
//! lies under the indicators. A non-dealer's win ends the game, so a win
//! that leaves the player alone in last is a fourth place for certain, and
//! where it is the only win, declining it can place them no lower: the hand
//! goes on, and no end of it is worse than last. Another player may win on
//! the same discard (EMA section 3.3.1), and the question does not look:
//! whether another hand can win on the tile is hidden, and at a real table
//! the answers to a discard are given at once, so a rule that read either
//! could not be played honestly. With a second winner, declining can cost: a
//! dealer who wins too keeps the deal, and the declined ron only loses its
//! points; a second non-dealer has the discarder pay twice, which can lift
//! the player off last. Rare, and not modelled.

use crate::game::{Action, Call, Discard, Hand, Phase};
use crate::hand::TileSet;
use crate::score::{Score, ScoreError, WinBy};
use crate::tile::{Tile, KINDS};
use crate::wall::Wall;
use crate::Wind;

/// What declaring riichi with a tile would add, in the last hand, over
/// discarding the same tile without declaring. The numbers are what the
/// Python extension hands over, a byte a kind.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dama {
    /// Not asked: not the last hand, not a non-dealer's own move, no riichi
    /// open, or a kind the hand does not hold.
    Absent = 0,
    /// Riichi would add no win, and every win the hand can make without it
    /// leaves the player alone in first, so riichi would buy no place.
    KeepsFirst = 1,
    /// Some live wait has no yaku to win by without riichi, so riichi would
    /// add that win.
    NoYaku = 2,
    /// Some win would leave another player level with the player or ahead.
    Short = 3,
    /// No wait is live: every copy of every wait is one the player can see.
    Dead = 4,
    /// Riichi with this tile is not legal: discarding it leaves the hand not
    /// waiting.
    NotWaiting = 5,
}

/// Whether the win a non-dealer may declare in the last hand would leave
/// them alone in last (see the module's notes). A byte a seat asked about.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Decline {
    /// Not asked: not the last hand, the dealer, or no win open to the seat.
    Absent = 0,
    /// The win leaves the player strictly below every other: alone in last.
    /// A riichi win does so even at the most its ura dora could add.
    Last = 1,
    /// The win leaves another player level with the player or below them,
    /// or a riichi win could, with the ura dora it may take.
    NotLast = 2,
}

impl Hand {
    /// In the last hand (South 4), at the turn player's own move, when that
    /// player is not the dealer and may declare riichi: for each kind of tile
    /// they hold, indexed by kind, what declaring with it would add over
    /// discarding it without declaring (see [`Dama`] and the module's notes).
    /// [`Dama::Absent`] everywhere otherwise, and for the kinds not held.
    /// Nothing in the hand changes.
    pub fn all_last_dama(&self) -> [Dama; KINDS] {
        let mut codes = [Dama::Absent; KINDS];
        let Some(legal) = self.last_riichi_open() else {
            return codes;
        };
        let held = self.current().hand;
        for tile in Tile::all().filter(|tile| held.count(*tile) > 0) {
            codes[tile.idx()] = if legal.contains(&Action::Riichi(tile)) {
                self.dama_after(tile)
            } else {
                Dama::NotWaiting
            };
        }
        codes
    }

    /// In the last hand (South 4), whether the win `seat` may declare now, a
    /// self-draw at their own turn or a ron on the discard or quad on the
    /// table, would leave them alone in last (see [`Decline`] and the
    /// module's notes). A riichi win is read at the most its ura dora could
    /// add, never at the real ones, which nobody sees before the win is
    /// declared. [`Decline::Absent`] for the dealer, in any other hand, and
    /// where no win is open to `seat`. Nothing in the hand changes.
    pub fn all_last_decline(&self, seat: Wind) -> Decline {
        if self.round != Wind::South || self.kyoku != 4 || seat == Wind::East {
            return Decline::Absent;
        }
        let offered = match self.phase {
            Phase::Act => seat == self.turn && self.legal_actions().contains(&Action::Tsumo),
            Phase::CallWindow => self
                .legal_calls()
                .iter()
                .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron)),
            Phase::Draw | Phase::Over => false,
        };
        if !offered {
            return Decline::Absent;
        }
        let mut won = self.clone();
        // The win writes itself into the log, which nothing reads here.
        won.log.clear();
        let player = &self.players[seat.index()];
        if player.has_riichi() {
            // Nobody sees the ura dora before the win is declared, so it is
            // read at the most they could add to the winning hand: a
            // self-draw's tile is in it already, and a ron's is on the table.
            let mut held = player.visible_to_self();
            if matches!(self.phase, Phase::CallWindow) {
                let (_, tile) = self.pending_discard.expect("a claim window has a tile");
                held.add(tile);
            }
            won.wall = self.most_ura(&held);
        }
        if matches!(self.phase, Phase::Act) {
            won.act(Action::Tsumo).expect("the self-draw was offered");
        } else {
            won.resolve_calls(&[(seat, Call::Ron)])
                .expect("the ron was offered");
        }
        if standing(won.scores(), seat) == (3, 0) {
            Decline::Last
        } else {
            Decline::NotLast
        }
    }

    /// The turn player's legal actions, where the last hand's riichi
    /// question is put: South 4, a non-dealer's own move, riichi open. None
    /// anywhere else.
    fn last_riichi_open(&self) -> Option<Vec<Action>> {
        if !matches!(self.phase, Phase::Act)
            || self.round != Wind::South
            || self.kyoku != 4
            || self.turn == Wind::East
        {
            return None;
        }
        let legal = self.legal_actions();
        legal
            .iter()
            .any(|action| matches!(action, Action::Riichi(_)))
            .then_some(legal)
    }

    /// What discarding `tile` without declaring leaves the turn player: every
    /// win the hand can then make without riichi, scored and settled as the
    /// engine scores and settles a real one, on a copy of the hand.
    fn dama_after(&self, tile: Tile) -> Dama {
        let seat = self.turn;
        let mut after = self.discarded(tile);
        let live = after.live_waits(seat);
        if live.is_empty() {
            return Dama::Dead;
        }
        let Some(wins) = plain_wins(&mut after, seat, &live) else {
            return Dama::NoYaku;
        };
        for win in wins {
            let points = settled(&after, seat, win.score, win.discarder);
            if Wind::ALL
                .iter()
                .any(|other| *other != seat && points[other.index()] >= points[seat.index()])
            {
                return Dama::Short;
            }
        }
        Dama::KeepsFirst
    }

    /// A copy of the hand as the turn player's discard of `tile`, without
    /// declaring, leaves it, with an empty log: the settled copies made of it
    /// write their wins into it, and nothing reads it.
    fn discarded(&self, tile: Tile) -> Hand {
        let seat = self.turn;
        let mut after = self.clone();
        after.log.clear();
        let order = after.discards_made;
        after.discards_made += 1;
        let player = &mut after.players[seat.index()];
        player.hand.remove(tile);
        player.discards.push(Discard {
            tile,
            order,
            drawn: self.drawn == Some(tile),
            riichi: false,
            claimed: false,
        });
        player.ippatsu = false;
        player.refresh_furiten();
        // Nothing situational is left to count: the player has discarded, so
        // no blessing; no quad was just declared, so no win on its
        // replacement and none by robbing it. Nor a win on the last tile,
        // though the wall may run out before the player wins: that only adds
        // a han. And near the end of the wall the player may never draw
        // again, so a self-draw counted here may never come. Either way the
        // answer errs only towards keeping the riichi.
        after.drawn = None;
        after.first_turns_unbroken = false;
        after.after_quad = false;
        after.robbable_quad = None;
        after.robbing_concealed = false;
        after.just_claimed = None;
        after
    }

    /// A wall that shows the same dora as the hand's and holds as many tiles
    /// to draw, which is all scoring reads of it, and shows no ura dora.
    fn without_ura(&self) -> Wall {
        let indicators = self.wall.dora_indicators();
        // A dealt wall holds at most sixty-nine tiles to draw, and one dora
        // indicator and one more for each of at most four quads.
        Wall::for_analysis(self.wall.remaining(), &indicators, indicators.len() - 1)
            .expect("a dealt wall can be shown for analysis")
    }

    /// The wall [`Hand::without_ura`] makes, with under every dora indicator
    /// an ura indicator of the kind `held`, the tiles of a winning hand,
    /// holds most copies of: the most the ura dora could add to a riichi win
    /// with those tiles. The real ura indicators are never read.
    fn most_ura(&self, held: &TileSet) -> Wall {
        let kind = Tile::all()
            .max_by_key(|kind| held.count(*kind))
            .expect("there are tile kinds");
        // The dora runs round each suit, the winds and the dragons, so every
        // kind is the dora of exactly one indicator.
        let under = Tile::all()
            .find(|tile| tile.dora() == kind)
            .expect("every kind is an indicator's dora");
        let ura = vec![under; self.wall.dora_indicators().len()];
        self.without_ura()
            .with_revealed_ura(&ura)
            .expect("an analysis wall takes an ura indicator under each dora indicator")
    }

    /// The waits of `seat`'s hand with a copy left that they cannot see.
    fn live_waits(&self, seat: Wind) -> Vec<Tile> {
        let waits = self.players[seat.index()].waits();
        let unseen = self.unseen_by(seat);
        Tile::all()
            .filter(|wait| waits.count(*wait) > 0 && unseen.count(*wait) > 0)
            .collect()
    }
}

/// A win the turn player could make on a wait without riichi.
struct Win {
    /// Who throws the wait, or None for a self-draw.
    discarder: Option<Wind>,
    score: Score,
}

/// Every win `seat` could make in `after` on the waits `live` without
/// riichi: by ron off each of the three others, unless `seat` is furiten,
/// and by tsumo, each scored by [`Hand::would_win`]. None as soon as a wait
/// has no yaku to win by.
fn plain_wins(after: &mut Hand, seat: Wind, live: &[Tile]) -> Option<Vec<Win>> {
    let furiten = after.players[seat.index()].is_furiten();
    let mut wins = Vec::new();
    for &wait in live {
        if !furiten {
            let ron = after.would_win(seat, wait, WinBy::Discard).ok()?;
            for step in 1..4 {
                wins.push(Win {
                    discarder: Some(seat.plus(step)),
                    score: ron.clone(),
                });
            }
        }
        let tsumo = self_draw(after, seat, wait).ok()?;
        wins.push(Win {
            discarder: None,
            score: tsumo,
        });
    }
    Some(wins)
}

/// `seat`'s self-draw on `wait`, as [`Hand::would_win`] scores it: with the
/// tile in hand, as the player holds it once drawn.
fn self_draw(hand: &mut Hand, seat: Wind, wait: Tile) -> Result<Score, ScoreError> {
    hand.players[seat.index()].hand.add(wait);
    let score = hand.would_win(seat, wait, WinBy::SelfDraw);
    hand.players[seat.index()].hand.remove(wait);
    score
}

/// The points by seat once `score`, `seat`'s win off `discarder` (None for a
/// self-draw), is settled on a copy of `hand` by the settlement a real win
/// goes through.
fn settled(hand: &Hand, seat: Wind, score: Score, discarder: Option<Wind>) -> [i32; 4] {
    let mut settled = hand.clone();
    settled.apply_win(vec![(seat, score)], discarder);
    settled.scores()
}

/// Where `seat` ends the game on `points`: how many others end ahead of them
/// and how many level, which is all their place and its share of the uma
/// turn on (EMA 2025 section 3.7.1).
fn standing(points: [i32; 4], seat: Wind) -> (usize, usize) {
    let mine = points[seat.index()];
    let (mut ahead, mut level) = (0, 0);
    for other in Wind::ALL.into_iter().filter(|other| *other != seat) {
        ahead += usize::from(points[other.index()] > mine);
        level += usize::from(points[other.index()] == mine);
    }
    (ahead, level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::Bot;
    use crate::hand::{ClaimedFrom, Meld};
    use crate::rng::Rng;
    use crate::score::Riichi;
    use crate::table::Table;

    fn tile(text: &str) -> Tile {
        text.parse().unwrap()
    }

    /// A hand of `round` and `kyoku` with `seat` to move on its fourteen
    /// tiles `held`, `drawn` among them, and the points by seat. The wall
    /// holds thirty tiles and its one indicator is the south wind, so the
    /// dora is the west wind, which no hand here holds. The other three hold
    /// what the deal gave them, which nothing asked here reads.
    fn position(
        round: Wind,
        kyoku: u8,
        seat: Wind,
        held: &str,
        drawn: &str,
        scores: [i32; 4],
    ) -> Hand {
        let mut hand = Hand::deal(&mut Rng::from_seed(20261030), round, kyoku, 0, 0, scores);
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
    fn probe(gap: i32) -> Hand {
        position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [20_000, 36_000 + gap, 28_000 - gap, 36_000],
        )
    }

    fn code(hand: &Hand, text: &str) -> Dama {
        hand.all_last_dama()[tile(text).idx()]
    }

    /// Whether `points` leave `seat` alone in first.
    fn first(points: [i32; 4], seat: Wind) -> bool {
        Wind::ALL
            .iter()
            .all(|other| *other == seat || points[other.index()] < points[seat.index()])
    }

    /// What the turn player's discard of `tile`, without declaring, leaves
    /// when every win is played through the real flow rather than on a copy
    /// of the hand: the discard is made and nobody claims it; then, for each
    /// live wait, each other player in turn is handed the wait and throws it,
    /// and the player rons it (unless the discard left them furiten), and
    /// the player draws the wait and declares tsumo. The same codes as
    /// [`Hand::all_last_dama`] where the hand it asks is the hand the real
    /// discard leaves.
    fn played_out(hand: &Hand, tile: Tile) -> Dama {
        let seat = hand.turn;
        let mut after = hand.clone();
        after
            .act(Action::Discard(tile))
            .expect("the discard is legal");
        if matches!(after.phase, Phase::CallWindow) {
            after.resolve_calls(&[]).expect("everyone may pass");
        }
        let waits = after.players[seat.index()].waits();
        let unseen = after.unseen_by(seat);
        let live: Vec<Tile> = Tile::all()
            .filter(|wait| waits.count(*wait) > 0 && unseen.count(*wait) > 0)
            .collect();
        if live.is_empty() {
            return Dama::Dead;
        }
        let furiten = after.players[seat.index()].is_furiten();
        let mut outcomes = Vec::new();
        for &wait in &live {
            if !furiten {
                for step in 1..4 {
                    let from = seat.plus(step);
                    let mut ron = after.clone();
                    ron.turn = from;
                    ron.phase = Phase::Act;
                    ron.pending_discard = None;
                    ron.just_claimed = None;
                    ron.players[from.index()].hand.add(wait);
                    ron.drawn = Some(wait);
                    ron.act(Action::Discard(wait))
                        .expect("the wait is thrown as drawn");
                    let offered = ron
                        .legal_calls()
                        .iter()
                        .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron));
                    if !offered {
                        return Dama::NoYaku;
                    }
                    ron.resolve_calls(&[(seat, Call::Ron)])
                        .expect("the win was offered");
                    outcomes.push(ron.scores());
                }
            }
            let mut tsumo = after.clone();
            tsumo.turn = seat;
            tsumo.phase = Phase::Act;
            tsumo.pending_discard = None;
            tsumo.just_claimed = None;
            tsumo.after_quad = false;
            tsumo.players[seat.index()].hand.add(wait);
            tsumo.players[seat.index()].temporary_furiten = false;
            tsumo.drawn = Some(wait);
            if !tsumo.legal_actions().contains(&Action::Tsumo) {
                return Dama::NoYaku;
            }
            tsumo.act(Action::Tsumo).expect("the win was offered");
            outcomes.push(tsumo.scores());
        }
        if outcomes.iter().all(|points| first(*points, seat)) {
            Dama::KeepsFirst
        } else {
            Dama::Short
        }
    }

    /// A wall that shows the dora of `hand`'s and holds as many tiles to
    /// draw, with under every dora indicator an ura indicator of the kind
    /// `held`, the tiles of a winning hand, holds most copies of.
    fn ura_wall(hand: &Hand, held: TileSet) -> Wall {
        let indicators = hand.wall.dora_indicators();
        let wall =
            Wall::for_analysis(hand.wall.remaining(), &indicators, indicators.len() - 1).unwrap();
        let most_held = Tile::all().map(|kind| held.count(kind)).max().unwrap();
        let kind = Tile::all()
            .find(|kind| held.count(*kind) == most_held)
            .unwrap();
        let under = Tile::all().find(|tile| tile.dora() == kind).unwrap();
        wall.with_revealed_ura(&vec![under; indicators.len()])
            .unwrap()
    }

    /// South 4 with the dealer, East, as player 3: a non-dealer's win there
    /// ends the game.
    fn last_table() -> Table {
        Table {
            round: Wind::South,
            dealer: 3,
            hands_played: 7,
            ..Table::new()
        }
    }

    /// What `seat`'s win does when played for real: won through the engine's
    /// own act or resolve_calls on a copy of the hand, and taken by a copy of
    /// the table, whose game it must end. A riichi win is played on the wall
    /// as dealt, or, with `most`, at the most its ura dora could add: on a
    /// wall that shows the same dora and holds as many tiles to draw, with
    /// under every dora indicator an ura indicator of the kind the winning
    /// hand holds most copies of. [`Decline::Last`] where the table's final
    /// scores leave the player alone in last, [`Decline::NotLast`] where
    /// they do not.
    fn decline_played_out(hand: &Hand, table: &Table, seat: Wind, most: bool) -> Decline {
        let mut won = hand.clone();
        let player = &hand.players[seat.index()];
        if most && player.has_riichi() {
            let mut held = player.visible_to_self();
            if matches!(hand.phase, Phase::CallWindow) {
                held.add(hand.pending_discard.unwrap().1);
            }
            won.wall = ura_wall(hand, held);
        }
        if matches!(hand.phase, Phase::Act) {
            won.act(Action::Tsumo).expect("the self-draw is open");
        } else {
            won.resolve_calls(&[(seat, Call::Ron)])
                .expect("the ron is open");
        }
        let person = table.player_at(seat);
        let mut game = table.clone();
        game.finish(&won);
        assert!(game.finished, "a non-dealer's win in South 4 ends the game");
        let finals = game.final_scores();
        if (0..4).all(|other| other == person || finals[other] > finals[person]) {
            Decline::Last
        } else {
            Decline::NotLast
        }
    }

    /// Any gap under 1,000 keeps North first through every win; every
    /// other tile it holds leaves it not waiting, and the kinds it does not
    /// hold are not asked about.
    #[test]
    fn the_probe_keeps_first_while_the_gap_is_under_a_thousand() {
        for gap in [0, 900] {
            let hand = probe(gap);
            let codes = hand.all_last_dama();
            let held = hand.players[Wind::North.index()].hand;
            for kind in Tile::all() {
                let want = if kind == tile("9p") {
                    Dama::KeepsFirst
                } else if held.count(kind) > 0 {
                    Dama::NotWaiting
                } else {
                    Dama::Absent
                };
                assert_eq!(codes[kind.idx()], want, "{kind} at a gap of {gap}");
            }
        }
    }

    #[test]
    fn a_leader_keeps_first() {
        // South and West on 32,000 each, North 4,000 ahead.
        assert_eq!(code(&probe(-4_000), "9p"), Dama::KeepsFirst);
    }

    /// EMA 2025 section 3.3.10: the winner collects the bets on the table,
    /// which makes the cheapest ron 2,000 to North and a tie at a gap of
    /// 2,000.
    #[test]
    fn the_bets_on_the_table_go_to_the_winner() {
        for (gap, want) in [(1_900, Dama::KeepsFirst), (2_000, Dama::Short)] {
            let mut hand = probe(gap);
            hand.riichi_sticks = 1;
            assert_eq!(code(&hand, "9p"), want, "a gap of {gap}");
        }
    }

    /// A bet placed this hand by another player is in the pool the winner
    /// collects, as a carried one is (EMA 2025 section 3.3.10), and has
    /// cost its owner a thousand: with South's riichi declared, the 1m ron
    /// brings North 2,000 and leaves South on 35,000 + gap, a tie at a gap
    /// of 3,000.
    #[test]
    fn another_players_bet_this_hand_goes_to_the_winner() {
        for (gap, want) in [(2_900, Dama::KeepsFirst), (3_000, Dama::Short)] {
            let mut hand = probe(gap);
            let south = &mut hand.players[Wind::South.index()];
            south.riichi = Riichi::Declared;
            south.riichi_order = Some(0);
            south.score -= 1_000;
            south.discards.push(Discard {
                tile: tile("1z"),
                order: 0,
                drawn: false,
                riichi: true,
                claimed: false,
            });
            hand.discards_made = 1;
            hand.riichi_sticks = 1;
            hand.bets_this_hand[Wind::South.index()] = 1;
            assert_eq!(code(&hand, "9p"), want, "a gap of {gap}");
            assert_eq!(played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
    }

    /// EMA 2025 section 3.4.4: 300 a counter on a ron, so two make the 1m
    /// ron 1,600.
    #[test]
    fn the_counters_are_paid_with_every_win() {
        for (gap, want) in [(1_500, Dama::KeepsFirst), (1_600, Dama::Short)] {
            let mut hand = probe(gap);
            hand.counters = 2;
            assert_eq!(code(&hand, "9p"), want, "a gap of {gap}");
        }
    }

    /// The 4m in North's own pond makes the hand furiten (EMA 2025 section
    /// 3.3.9): neither silence nor riichi could ron, so only the self-draws
    /// count, and a 1m tsumo, North 37,500 and South 36,000 + gap - 400,
    /// ties at a gap of 1,900.
    fn furiten(gap: i32) -> Hand {
        let mut hand = probe(gap);
        hand.players[Wind::North.index()].discards.push(Discard {
            tile: tile("4m"),
            order: 0,
            drawn: false,
            riichi: false,
            claimed: false,
        });
        hand.discards_made = 1;
        hand
    }

    #[test]
    fn a_furiten_hand_counts_its_self_draws_alone() {
        for (gap, want) in [
            (1_500, Dama::KeepsFirst),
            (1_800, Dama::KeepsFirst),
            (1_900, Dama::Short),
        ] {
            assert_eq!(code(&furiten(gap), "9p"), want, "a gap of {gap}");
        }
    }

    /// EMA 2025 section 3.4.4: 100 a counter from each payer of a self-draw,
    /// so with two counters the furiten hand's 1m tsumo is 600/900, North
    /// 38,100 against South's 36,000 + gap - 600, a tie at a gap of 2,700.
    #[test]
    fn the_counters_are_paid_with_a_self_draw_too() {
        for (gap, want) in [(2_600, Dama::KeepsFirst), (2_700, Dama::Short)] {
            let mut hand = furiten(gap);
            hand.counters = 2;
            assert_eq!(code(&hand, "9p"), want, "a gap of {gap}");
            assert_eq!(played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
    }

    /// Right after the player's own concealed quad, a self-draw would win on
    /// the quad's replacement, which the discard spends. North holds a
    /// concealed quad of 5s and 23456m 456p 33s, the 9p drawn from the dead
    /// wall, and the 4m in its own pond: furiten, so only the self-draws
    /// count, and the quad leaves no pinfu. A 1m tsumo is menzen tsumo
    /// alone at 40 fu, 400/700, North 37,500 against South's
    /// 36,000 + gap - 400, a tie at a gap of 1,900; counted as a win on the
    /// replacement it would be 700/1,300, and keep North first there.
    #[test]
    fn the_quads_replacement_is_spent_by_the_discard() {
        for (gap, want) in [(1_800, Dama::KeepsFirst), (1_900, Dama::Short)] {
            let mut hand = furiten(gap);
            let north = &mut hand.players[Wind::North.index()];
            north.hand = "23456m456p33s9p".parse().unwrap();
            north.melds = vec![Meld::concealed_kan(tile("5s"))];
            hand.after_quad = true;
            assert!(hand.legal_actions().contains(&Action::Riichi(tile("9p"))));
            assert_eq!(code(&hand, "9p"), want, "a gap of {gap}");
            assert_eq!(played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
    }

    /// EMA 2025 section 3.7.1: tied players share the uma of the places they
    /// cover, so a tie for first is not first.
    #[test]
    fn a_tie_for_first_is_short() {
        for gap in [1_000, 1_500] {
            assert_eq!(code(&probe(gap), "9p"), Dama::Short, "a gap of {gap}");
        }
    }

    #[test]
    fn only_the_last_hand_is_asked() {
        for (round, kyoku) in [(Wind::East, 4), (Wind::South, 3)] {
            let mut hand = probe(0);
            hand.round = round;
            hand.kyoku = kyoku;
            assert_eq!(
                hand.all_last_dama(),
                [Dama::Absent; KINDS],
                "{round:?} {kyoku}"
            );
        }
    }

    /// The dealer's win keeps the deal and South 4 is played again, so the
    /// dealer is never asked, however far ahead.
    #[test]
    fn the_dealer_is_never_asked() {
        let hand = position(
            Wind::South,
            4,
            Wind::East,
            "23456m456p678s33s9p",
            "9p",
            [60_000, 20_000, 20_000, 20_000],
        );
        assert!(hand.legal_actions().contains(&Action::Riichi(tile("9p"))));
        assert_eq!(hand.all_last_dama(), [Dama::Absent; KINDS]);
    }

    /// 123m 789m 234p 99s 57s waits on the 6s alone, a closed wait with no
    /// yaku: only riichi, or a self-draw, could win with it.
    #[test]
    fn a_wait_with_no_yaku_needs_the_riichi() {
        let hand = position(
            Wind::South,
            4,
            Wind::North,
            "123789m234p5799s4z",
            "4z",
            [25_000, 25_000, 25_000, 45_000],
        );
        assert_eq!(code(&hand, "4z"), Dama::NoYaku);
    }

    /// 234m 678p 222s 55s 78m waits on the 6m, all simples, and on the 9m,
    /// which has no yaku: one such wait is enough.
    #[test]
    fn one_wait_with_no_yaku_is_enough() {
        let hand = position(
            Wind::South,
            4,
            Wind::North,
            "23478m678p22255s4z",
            "4z",
            [25_000, 25_000, 25_000, 45_000],
        );
        assert_eq!(code(&hand, "4z"), Dama::NoYaku);
    }

    /// 234m 678p 345s 57s 88s waits on the 6s alone. With all four in the
    /// others' ponds it can win nothing; with one left it wins all simples,
    /// 1,300 by ron and 500/1,000 by tsumo, and a leader by 20,000 keeps
    /// first.
    #[test]
    fn a_dead_wait_wins_nothing() {
        for (copies, want) in [(4, Dama::Dead), (3, Dama::KeepsFirst)] {
            let mut hand = position(
                Wind::South,
                4,
                Wind::North,
                "234m678p3455788s4z",
                "4z",
                [25_000, 25_000, 25_000, 45_000],
            );
            for order in 0..copies {
                hand.players[order as usize % 3].discards.push(Discard {
                    tile: tile("6s"),
                    order,
                    drawn: false,
                    riichi: false,
                    claimed: false,
                });
            }
            hand.discards_made = copies;
            assert_eq!(code(&hand, "4z"), want, "{copies} in the ponds");
        }
    }

    /// Only the live waits are reckoned: a dead one wins nothing, with
    /// riichi or without. The probe at a gap of 1,500 is short by its 1m
    /// ron alone, worth 1,000; with the four 1m in the others' ponds, its 4m
    /// and 7m, 2,000 by ron, keep North first. 234m 678p 222s 55s 78m lacks
    /// a yaku only on its 9m; with the four 9m gone, the 6m, all simples,
    /// keeps a leader first.
    #[test]
    fn a_dead_wait_is_left_out() {
        let gone = |mut hand: Hand, wait: &str| {
            for order in 0..4 {
                hand.players[order as usize % 3].discards.push(Discard {
                    tile: tile(wait),
                    order,
                    drawn: false,
                    riichi: false,
                    claimed: false,
                });
            }
            hand.discards_made = 4;
            hand
        };
        let no_yaku = position(
            Wind::South,
            4,
            Wind::North,
            "23478m678p22255s4z",
            "4z",
            [25_000, 25_000, 25_000, 45_000],
        );
        for (hand, thrown, wait, live) in [
            (probe(1_500), "9p", "1m", Dama::Short),
            (no_yaku, "4z", "9m", Dama::NoYaku),
        ] {
            assert_eq!(code(&hand, thrown), live, "{thrown} with every wait live");
            let dead = gone(hand, wait);
            assert_eq!(code(&dead, thrown), Dama::KeepsFirst, "{wait} dead");
            assert_eq!(
                played_out(&dead, tile(thrown)),
                Dama::KeepsFirst,
                "{wait} dead"
            );
        }
    }

    /// EMA 2025 section 3.3.10: riichi needs 1,000 points, a tile in the
    /// wall and a concealed hand. Without one there is nothing to ask.
    #[test]
    fn with_no_riichi_open_nothing_is_asked() {
        let mut poor = probe(0);
        poor.players[Wind::North.index()].score = 900;
        let mut empty = probe(0);
        empty.wall = Wall::for_analysis(0, &[tile("2z")], 0).unwrap();
        let mut open = probe(0);
        open.players[Wind::North.index()].hand = "23456m456p33s9p".parse().unwrap();
        open.players[Wind::North.index()].melds = vec![Meld::pon(tile("8s"), ClaimedFrom::Left)];
        for (name, hand) in [
            ("900 points", poor),
            ("an empty wall", empty),
            ("an open hand", open),
        ] {
            assert!(
                !hand
                    .legal_actions()
                    .iter()
                    .any(|action| matches!(action, Action::Riichi(_))),
                "{name}"
            );
            assert_eq!(hand.all_last_dama(), [Dama::Absent; KINDS], "{name}");
        }
    }

    #[test]
    fn far_behind_is_short() {
        // North third on 22,000, 28,000 behind South.
        let hand = position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [20_000, 50_000, 28_000, 22_000],
        );
        assert_eq!(code(&hand, "9p"), Dama::Short);
    }

    #[test]
    fn asking_changes_nothing() {
        for hand in [probe(0), probe(1_500), furiten(1_500)] {
            let before = hand.clone();
            hand.all_last_dama();
            assert_eq!(hand, before);
        }
    }

    /// The copy of the hand the codes are worked out on is the hand the
    /// real discard leaves: every win played through the real flow gives the
    /// same code.
    #[test]
    fn the_real_flow_agrees() {
        let mut cases = vec![
            (probe(0), Dama::KeepsFirst),
            (probe(900), Dama::KeepsFirst),
            (probe(1_000), Dama::Short),
            (probe(1_500), Dama::Short),
            (furiten(1_500), Dama::KeepsFirst),
            (furiten(1_900), Dama::Short),
        ];
        for (gap, want) in [(1_900, Dama::KeepsFirst), (2_000, Dama::Short)] {
            let mut hand = probe(gap);
            hand.riichi_sticks = 1;
            cases.push((hand, want));
        }
        for (hand, want) in cases {
            let scores = hand.scores();
            assert_eq!(code(&hand, "9p"), want, "{scores:?}");
            assert_eq!(played_out(&hand, tile("9p")), want, "{scores:?}");
        }
    }

    /// How the codes fell over some Club games: the turn decisions asked,
    /// those with a tile that keeps first, and the riichi tiles by code; and
    /// the wins open to a non-dealer in the last hand by decline code, and
    /// those coded not last, riichi wins all, that the ura dora as dealt
    /// would have left alone in last. With them, the first games a win that
    /// leaves the player last fell in.
    #[derive(Default, Debug)]
    struct Counted {
        decisions: usize,
        asked: usize,
        keeping: usize,
        tiles: [usize; 6],
        wins: [usize; 3],
        last_as_dealt: usize,
        last_games: Vec<u64>,
    }

    /// Plays one game of four Club players for each of `games`, dealt from
    /// its own seed, and checks the codes at every decision: at a turn, the
    /// dama codes given exactly where it is the last hand, a non-dealer's
    /// move and riichi open, a riichi tile coded 1 to 4 and every other tile
    /// held not waiting, and each riichi tile's code the one the real flow
    /// gives it; and the decline code of every seat given exactly where a
    /// non-dealer of the last hand has a win open, at a turn and in a claim
    /// window, the one the real flow and the table's final scores give it
    /// with a riichi win at the most its ura dora could add, and a code of
    /// last kept on the wall as dealt.
    fn club_games(games: impl IntoIterator<Item = u64>) -> Counted {
        let mut counted = Counted::default();
        for game in games {
            let mut table = Table::new();
            let mut rng = Rng::from_seed(game);
            let mut bots: [Bot; 4] = std::array::from_fn(|place| {
                Bot::new(game.wrapping_mul(4).wrapping_add(place as u64))
            });
            while !table.finished {
                let mut hand = table.deal(&mut rng);
                while !matches!(hand.phase, Phase::Over) {
                    match hand.phase {
                        Phase::Draw => {
                            let _ = hand.draw();
                        }
                        Phase::Act => {
                            check(&hand, &table, game, &mut counted);
                            let action = bots[table.player_at(hand.turn)].act(&hand);
                            hand.act(action).expect("the bot chose a legal action");
                        }
                        Phase::CallWindow => {
                            check_claims(&hand, &table, game, &mut counted);
                            let answers: Vec<(Wind, Call)> = hand
                                .legal_calls()
                                .iter()
                                .map(|(seat, calls)| {
                                    (
                                        *seat,
                                        bots[table.player_at(*seat)].call(&hand, *seat, calls),
                                    )
                                })
                                .collect();
                            hand.resolve_calls(&answers)
                                .expect("the bots chose legal calls");
                        }
                        Phase::Over => break,
                    }
                }
                table.finish(&hand);
            }
        }
        counted
    }

    fn check(hand: &Hand, table: &Table, game: u64, counted: &mut Counted) {
        counted.decisions += 1;
        let codes = hand.all_last_dama();
        let legal = hand.legal_actions();
        let riichi = |kind: Tile| legal.contains(&Action::Riichi(kind));
        let asked = hand.round == Wind::South
            && hand.kyoku == 4
            && hand.turn != Wind::East
            && Tile::all().any(riichi);
        assert_eq!(
            codes.iter().any(|code| *code != Dama::Absent),
            asked,
            "{:?} {} {:?}: {codes:?}",
            hand.round,
            hand.kyoku,
            hand.turn
        );
        for seat in Wind::ALL {
            let open = seat == hand.turn && legal.contains(&Action::Tsumo);
            check_decline(hand, table, game, seat, open, counted);
        }
        if !asked {
            return;
        }
        counted.asked += 1;
        let held = hand.current().hand;
        for kind in Tile::all() {
            let code = codes[kind.idx()];
            if riichi(kind) {
                assert!(
                    matches!(
                        code,
                        Dama::KeepsFirst | Dama::NoYaku | Dama::Short | Dama::Dead
                    ),
                    "{kind}: {code:?}"
                );
                assert_eq!(played_out(hand, kind), code, "{kind} in {held}");
                counted.tiles[code as usize] += 1;
            } else if held.count(kind) > 0 {
                assert_eq!(code, Dama::NotWaiting, "{kind} in {held}");
            } else {
                assert_eq!(code, Dama::Absent, "{kind} not in {held}");
            }
        }
        if codes.contains(&Dama::KeepsFirst) {
            counted.keeping += 1;
        }
    }

    /// The decline code of every seat in a claim window, `open` where the
    /// window offers it a ron.
    fn check_claims(hand: &Hand, table: &Table, game: u64, counted: &mut Counted) {
        let offered = hand.legal_calls();
        for seat in Wind::ALL {
            let open = offered
                .iter()
                .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron));
            check_decline(hand, table, game, seat, open, counted);
        }
    }

    fn check_decline(
        hand: &Hand,
        table: &Table,
        game: u64,
        seat: Wind,
        open: bool,
        counted: &mut Counted,
    ) {
        let code = hand.all_last_decline(seat);
        let asked = hand.round == Wind::South && hand.kyoku == 4 && seat != Wind::East && open;
        assert_eq!(
            code != Decline::Absent,
            asked,
            "{:?} {} {seat:?} in {:?}: {code:?}",
            hand.round,
            hand.kyoku,
            hand.phase
        );
        if !asked {
            return;
        }
        assert_eq!(
            decline_played_out(hand, table, seat, true),
            code,
            "{seat:?} in {:?}, game {game}",
            hand.phase
        );
        // The ura dora as dealt add no more than the most they could: a win
        // coded Last leaves the player alone in last on the wall as dealt
        // too, and one coded NotLast, a riichi win, may.
        let dealt = decline_played_out(hand, table, seat, false);
        assert!(
            code == Decline::NotLast || dealt == Decline::Last,
            "{seat:?} in {:?}, game {game}: {code:?}, and {dealt:?} as dealt",
            hand.phase
        );
        counted.wins[code as usize] += 1;
        counted.last_as_dealt += usize::from(code == Decline::NotLast && dealt == Decline::Last);
        if code == Decline::Last && counted.last_games.len() < 10 {
            counted.last_games.push(game);
        }
    }

    /// Forty games, about a minute unoptimised on the desktop (two hundred
    /// took five and a half): 67 of their turns are a non-dealer's in South
    /// 4 with riichi open, and their riichi tiles fall 13 keeping first, 48
    /// with no yaku, 32 short and 1 dead; 35 wins are open to a non-dealer in
    /// South 4, 5 of them leaving the player alone in last, and 4 more,
    /// riichi wins, that the ura dora as dealt would have left there.
    #[test]
    fn club_games_agree_with_the_real_flow() {
        let began = std::time::Instant::now();
        let counted = club_games(9_100_000..9_100_040);
        println!("{counted:?} in {:.1} s", began.elapsed().as_secs_f64());
        assert!(counted.asked > 20, "{counted:?}");
        for code in [Dama::KeepsFirst, Dama::NoYaku, Dama::Short] {
            assert!(counted.tiles[code as usize] > 0, "{code:?}: {counted:?}");
        }
        for code in [Decline::Last, Decline::NotLast] {
            assert!(counted.wins[code as usize] > 0, "{code:?}: {counted:?}");
        }
    }

    /// The share of the last hand's riichi decisions with a tile that keeps
    /// first, and the riichi tiles by code, over many Club games; and the
    /// wins open to a non-dealer in the last hand by decline code, and the
    /// riichi wins coded not last that the ura dora as dealt would have left
    /// there. Long: run with
    /// `cargo test --release -p riichi-core -- --ignored --nocapture`, which
    /// took 38 minutes on a busy desktop (31 October 2026), checking two
    /// rules since removed as well, and counted 28,719 riichi decisions of
    /// the last hand, 14.72% with a tile that keeps first; and 18,623 wins
    /// open to a non-dealer there, 2,012 of them leaving the player alone in
    /// last, and 868 more, riichi wins, that the ura dora as dealt would
    /// have left there.
    #[test]
    #[ignore]
    fn club_games_counted() {
        let began = std::time::Instant::now();
        let counted = club_games(9_100_000..9_120_000);
        println!(
            "{} decisions at a turn; {} in the last hand with riichi open, {} of them ({:.2}%) with a tile \
             that keeps first",
            counted.decisions,
            counted.asked,
            counted.keeping,
            100.0 * counted.keeping as f64 / counted.asked.max(1) as f64
        );
        for (name, code) in [
            ("keeps first", Dama::KeepsFirst),
            ("no yaku", Dama::NoYaku),
            ("short", Dama::Short),
            ("dead", Dama::Dead),
        ] {
            println!("riichi tiles that {name}: {}", counted.tiles[code as usize]);
        }
        println!(
            "wins open to a non-dealer in the last hand: {} leave the player last, {} do not, {} of these \
             riichi wins the ura as dealt would have left last; first games with one that does: {:?}",
            counted.wins[Decline::Last as usize],
            counted.wins[Decline::NotLast as usize],
            counted.last_as_dealt,
            counted.last_games
        );
        println!("in {:.1} s", began.elapsed().as_secs_f64());
    }

    // A win that leaves the player last.

    /// A hand of `round` and `kyoku` in which `from` has just thrown the
    /// `thrown` tile and `seat`, on the thirteen tiles `held`, may claim it;
    /// the points by seat. The wall is `position`'s, and the others hold what
    /// the deal gave them, the dealer's fourteenth tile put back.
    fn claim(
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
    fn tsumo(west: i32) -> Hand {
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
    fn ron(west: i32) -> Hand {
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

    fn declined(hand: &Hand, seat: Wind) -> Decline {
        let code = hand.all_last_decline(seat);
        if code != Decline::Absent {
            assert_eq!(decline_played_out(hand, &last_table(), seat, true), code);
        }
        code
    }

    /// North's self-draw brings it 1,500 and costs West 400: alone in last
    /// while West stays ahead, level with it at 11,900.
    #[test]
    fn a_self_draw_that_leaves_the_player_alone_in_last() {
        for (west, want) in [(11_900, Decline::NotLast), (12_000, Decline::Last)] {
            assert_eq!(declined(&tsumo(west), Wind::North), want, "West on {west}");
        }
    }

    /// A ron off West brings North 1,000 and costs West as much: a tie at a
    /// gap of 2,000.
    #[test]
    fn a_ron_that_leaves_the_player_alone_in_last() {
        for (west, want) in [(12_000, Decline::NotLast), (12_100, Decline::Last)] {
            assert_eq!(declined(&ron(west), Wind::North), want, "West on {west}");
        }
    }

    /// The winner collects the bets on the table (EMA 2025 section 3.3.10),
    /// and the counters are paid with the win (section 3.4.4): a bet brings
    /// North's self-draw to 2,500, and two counters to 2,100, West paying
    /// 600 of it.
    #[test]
    fn bets_and_counters_are_paid_with_the_win() {
        for (west, want) in [(12_900, Decline::NotLast), (13_000, Decline::Last)] {
            let mut hand = tsumo(west);
            hand.riichi_sticks = 1;
            assert_eq!(declined(&hand, Wind::North), want, "a bet, West on {west}");
        }
        for (west, want) in [(12_700, Decline::NotLast), (12_800, Decline::Last)] {
            let mut hand = tsumo(west);
            hand.counters = 2;
            assert_eq!(
                declined(&hand, Wind::North),
                want,
                "two counters, West on {west}"
            );
        }
    }

    /// EMA 2025 section 3.3.10: a declaration won on never happened, and its
    /// bet goes back. West declared riichi with the 1m North rons: West gets
    /// its thousand back and pays North 1,000, and no bet is left for North,
    /// so North stays alone in last behind West's 11,100; were the bet
    /// North's, it would pass West.
    #[test]
    fn a_riichi_declared_with_the_tile_won_on_is_given_back() {
        for (west, want) in [(11_000, Decline::NotLast), (11_100, Decline::Last)] {
            let mut hand = ron(west);
            let player = &mut hand.players[Wind::West.index()];
            player.riichi = Riichi::Declared;
            player.riichi_order = Some(0);
            player.discards.last_mut().unwrap().riichi = true;
            hand.riichi_sticks = 1;
            hand.bets_this_hand[Wind::West.index()] = 1;
            assert_eq!(declined(&hand, Wind::North), want, "West on {west}");
        }
    }

    /// The same 1m won by ron off the dealer and by tsumo. North on 20,000,
    /// East on 22,100: the ron closes the gap by 2,000, which leaves North
    /// alone in last; the tsumo closes it by 2,200, East paying 700, which
    /// lifts North past East.
    #[test]
    fn a_ron_against_a_tsumo() {
        let scores = [22_100, 30_000, 27_900, 20_000];
        let off_east = claim(
            Wind::South,
            4,
            Wind::East,
            "1m",
            Wind::North,
            "23456m456p678s33s",
            scores,
        );
        assert_eq!(declined(&off_east, Wind::North), Decline::Last);
        let drawn = position(
            Wind::South,
            4,
            Wind::North,
            "123456m456p678s33s",
            "1m",
            scores,
        );
        assert_eq!(declined(&drawn, Wind::North), Decline::NotLast);
    }

    /// `hand` with North in riichi, declared with a 9p before anything else
    /// was thrown, and its bet on the table: North on 9,000 after it. With
    /// `ura`, the tile under the wall's one dora indicator; without, a wall
    /// that shows none.
    fn in_riichi(mut hand: Hand, ura: Option<Tile>) -> Hand {
        for player in hand.players.iter_mut() {
            for discard in player.discards.iter_mut() {
                discard.order += 1;
            }
        }
        hand.discards_made += 1;
        let north = &mut hand.players[Wind::North.index()];
        north.discards.insert(
            0,
            Discard {
                tile: tile("9p"),
                order: 0,
                drawn: false,
                riichi: true,
                claimed: false,
            },
        );
        north.riichi = Riichi::Declared;
        north.riichi_order = Some(0);
        north.score -= 1_000;
        hand.riichi_sticks += 1;
        hand.bets_this_hand[Wind::North.index()] += 1;
        if let Some(ura) = ura {
            hand.wall = hand.wall.with_revealed_ura(&[ura]).unwrap();
        }
        hand
    }

    /// Nobody sees the ura dora before a riichi win is declared, so the win
    /// is read at the most they could add, and its code is the same whatever
    /// lies under the indicator. North, in riichi on 23456m 456p 678s 33s,
    /// wins a 1m with riichi and pinfu, 2,000 by ron off West, and with
    /// menzen tsumo 700/1,300; at the most, two ura dora on the 3s pair make
    /// either a mangan, the ron's 4 han 30 fu paid as one (EMA 2025 section
    /// 4.1.2), 8,000 from West, and the tsumo 2,000/4,000. Against West on
    /// 14,100 the ron with no ura dora would leave North alone in last, on
    /// 12,000 behind West's 12,100, and the tsumo against West on 14,000
    /// too, on 12,700 behind 13,300; but at the most each lifts North past
    /// West, so the code is NotLast under every ura. Against West on 26,100
    /// even the most leaves North on 18,000 behind West's 18,100, and the
    /// tsumo against West on 20,100 too: Last under every ura. A hundred
    /// less, and the most leaves the two level.
    #[test]
    fn a_riichi_win_is_read_at_the_most_the_ura_could_add() {
        for (plain, want) in [
            (ron(14_100), Decline::NotLast),
            (ron(26_000), Decline::NotLast),
            (ron(26_100), Decline::Last),
            (tsumo(14_000), Decline::NotLast),
            (tsumo(20_000), Decline::NotLast),
            (tsumo(20_100), Decline::Last),
        ] {
            let scores = plain.scores();
            for ura in std::iter::once(None).chain(Tile::all().map(Some)) {
                let riichi = in_riichi(plain.clone(), ura);
                assert_eq!(
                    declined(&riichi, Wind::North),
                    want,
                    "{scores:?}, ura {ura:?}"
                );
            }
        }
        // Played for real, the ura decide: with the east wind under the
        // indicator, whose dora North does not hold, either win leaves it
        // alone in last; with the 2s, whose dora is the 3s of its pair, two
        // ura dora lift it past West.
        for plain in [ron(14_100), tsumo(14_000)] {
            for (ura, want) in [("1z", Decline::Last), ("2s", Decline::NotLast)] {
                let riichi = in_riichi(plain.clone(), Some(tile(ura)));
                assert_eq!(
                    decline_played_out(&riichi, &last_table(), Wind::North, false),
                    want,
                    "{:?} in {:?}, ura {ura}",
                    plain.scores(),
                    plain.phase
                );
            }
        }
    }

    /// The most the ura dora could add counts every tile of the winning
    /// hand, a quad's four with the rest, and puts the kind it holds most
    /// under every dora indicator. North, in riichi on 123456m 456p 33s and
    /// a concealed quad of 5s with the 1m just drawn, may declare a tsumo of
    /// riichi and menzen tsumo; the quad has turned a second indicator, and
    /// both are the south wind, whose dora North does not hold. North is on
    /// 9,000 after its bet, East on 36,000, South on 28,000 and West on
    /// 26,000. At the most a 4s lies under each indicator, and the quad's 5s
    /// are eight ura dora: a baiman, 4,000/8,000, which lifts North to
    /// 26,000, past South and West. With a 2s under each, the 3s pair's four
    /// ura dora make a haneman, 3,000/6,000, which leaves North alone in last
    /// on 22,000 behind West's 23,000.
    #[test]
    fn the_most_the_ura_could_add_counts_a_quad_under_every_indicator() {
        let mut hand = position(
            Wind::South,
            4,
            Wind::North,
            "123456m456p33s",
            "1m",
            [36_000, 28_000, 26_000, 10_000],
        );
        hand.players[Wind::North.index()].melds = vec![Meld::concealed_kan(tile("5s"))];
        hand.wall = Wall::for_analysis(30, &[tile("2z"), tile("2z")], 1).unwrap();
        let hand = in_riichi(hand, None);
        let most = hand.most_ura(&hand.players[Wind::North.index()].visible_to_self());
        assert_eq!(most.ura_indicators(), [tile("4s"), tile("4s")]);
        assert_eq!(most.dora_indicators(), hand.wall.dora_indicators());
        assert_eq!(most.remaining(), hand.wall.remaining());
        assert_eq!(declined(&hand, Wind::North), Decline::NotLast);
        // Played for real, the ura decide: with the 4s under each indicator
        // the win lifts North off last, and with the 2s, or the east wind,
        // whose dora North does not hold, it leaves North alone in last.
        for (ura, want) in [
            ("4s", Decline::NotLast),
            ("2s", Decline::Last),
            ("1z", Decline::Last),
        ] {
            let mut real = hand.clone();
            real.wall = hand
                .wall
                .with_revealed_ura(&[tile(ura), tile(ura)])
                .unwrap();
            assert_eq!(
                decline_played_out(&real, &last_table(), Wind::North, false),
                want,
                "ura {ura}"
            );
        }
    }

    /// A claim window asks each seat offered the win about its own win
    /// alone: West's 1m would leave North, on 5,000, alone in last and South,
    /// on 40,000, nowhere near it. The dealer is never asked, though it may
    /// ron too, and neither is the discarder. North is told Last though East
    /// and South may win on the same tile, and with East winning too, East
    /// would keep the deal and North's declining would only lose the ron's
    /// points. That is the price of asking on what the table shows: whether
    /// another hand can win on the tile is hidden, and at a real table the
    /// three answers are given at once, so a question that read them could
    /// not be played honestly (see the module's notes).
    #[test]
    fn each_seat_offered_a_ron_is_asked_about_its_own() {
        let mut hand = claim(
            Wind::South,
            4,
            Wind::West,
            "1m",
            Wind::North,
            "23456m456p678s33s",
            [40_000, 40_000, 15_000, 5_000],
        );
        hand.players[Wind::South.index()].hand = "23456m789p345s55s".parse().unwrap();
        hand.players[Wind::East.index()].hand = "23456m456p678s33s".parse().unwrap();
        let offered: Vec<Wind> = hand
            .legal_calls()
            .iter()
            .filter(|(_, calls)| calls.contains(&Call::Ron))
            .map(|(seat, _)| *seat)
            .collect();
        assert_eq!(offered, vec![Wind::East, Wind::South, Wind::North]);
        assert_eq!(declined(&hand, Wind::North), Decline::Last);
        assert_eq!(declined(&hand, Wind::South), Decline::NotLast);
        assert_eq!(hand.all_last_decline(Wind::East), Decline::Absent);
        assert_eq!(hand.all_last_decline(Wind::West), Decline::Absent);
    }

    /// Nothing is asked where no win is open to the seat: a furiten hand
    /// (EMA 2025 section 3.3.9), a seat whose turn it is not, a hand that is
    /// not complete; nor in any hand but the last, nor of the dealer.
    #[test]
    fn decline_asks_only_a_non_dealer_offered_a_win_in_the_last_hand() {
        let mut furiten = ron(20_000);
        furiten.players[Wind::North.index()].discards.push(Discard {
            tile: tile("4m"),
            order: 1,
            drawn: false,
            riichi: false,
            claimed: false,
        });
        furiten.discards_made = 2;
        furiten.players[Wind::North.index()].refresh_furiten();
        assert_eq!(furiten.all_last_decline(Wind::North), Decline::Absent);
        let drawn = tsumo(20_000);
        assert_eq!(drawn.all_last_decline(Wind::North), Decline::Last);
        for seat in [Wind::East, Wind::South, Wind::West] {
            assert_eq!(drawn.all_last_decline(seat), Decline::Absent, "{seat:?}");
        }
        assert_eq!(probe(0).all_last_decline(Wind::North), Decline::Absent);
        for (round, kyoku) in [(Wind::East, 4), (Wind::South, 3)] {
            for mut hand in [tsumo(20_000), ron(20_000)] {
                hand.round = round;
                hand.kyoku = kyoku;
                assert_eq!(hand.all_last_decline(Wind::North), Decline::Absent);
            }
        }
        let dealer = position(
            Wind::South,
            4,
            Wind::East,
            "123456m456p678s33s",
            "1m",
            [10_000, 30_000, 30_000, 30_000],
        );
        assert!(dealer.legal_actions().contains(&Action::Tsumo));
        assert_eq!(dealer.all_last_decline(Wind::East), Decline::Absent);
    }

    #[test]
    fn asking_about_a_win_changes_nothing() {
        let hands = [
            probe(0),
            probe(1_000),
            tsumo(12_000),
            ron(12_100),
            in_riichi(ron(14_100), Some(tile("2s"))),
        ];
        for hand in hands {
            let before = hand.clone();
            for seat in Wind::ALL {
                hand.all_last_decline(seat);
            }
            assert_eq!(hand, before);
        }
    }
}
