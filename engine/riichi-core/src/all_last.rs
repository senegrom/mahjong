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
//! # Riichi that buys no place
//!
//! [`Hand::all_last_wide_dama`] asks the same question more widely: not
//! whether every win without riichi keeps the player first, but whether
//! riichi would change the place any win ends the game in ([`WideDama`]).
//! The waits and the wins without riichi are the dama question's, scored and
//! settled the same way. Each win is then won again with riichi declared on
//! the same tile, on a copy of the hand as the declaration leaves it: the
//! tile turned sideways, a thousand points paid and the bet in the pool, from
//! which the winner takes it back as a real win does, and a double riichi in
//! the player's first turn with the first set of turns unbroken (EMA section
//! 4.2.2). What riichi adds to a win is not known when it is declared: the
//! one-shot and the ura dora come later. So the riichi win is scored at the
//! least riichi adds, riichi alone, and at the most it could add, riichi, the
//! one-shot, and under each dora indicator an ura indicator of the kind the
//! winning hand holds most copies of. Riichi buys no place where, for every
//! win, the player's place is the plain win's at both. A larger win only
//! moves the winner up and the payers down, so a place both bounds share is
//! the place of every riichi win between them; and a tile that keeps first
//! ([`Dama::KeepsFirst`]) is always one whose riichi buys no place.
//!
//! A place here is how many others end ahead of the player and how many
//! level: tied players share the uma of the places they cover (EMA section
//! 3.7.1), so a tie riichi would break is a place it buys. What adds han to
//! a win with riichi and without alike is not counted: a win on the last
//! tile, on a quad's replacement or by robbing a quad, and a new kan dora,
//! each of which could move one of the two over a line and not the other;
//! nor, as for dama, two winners on one discard. And unlike the dama answer,
//! this one holds at the moment it is asked, not for the rest of the hand:
//! a riichi bet another player places later, which the winner collects, or
//! a new kan dora moves the lines the two wins are read against.
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
//! winning hand holds most copies of, the most the ura could add, as for
//! riichi that buys no place. A larger win only lifts the winner and lowers
//! the payers, so a riichi win that leaves the player alone in last at the
//! most leaves them there whatever lies under the indicators. A non-dealer's
//! win ends the game, so a win that leaves the player alone in last is a
//! fourth place for certain, and where it is the only win, declining it can
//! place them no lower: the hand goes on, and no end of it is worse than
//! last. Another player may win on the same discard (EMA section 3.3.1), and
//! the question does not look: whether another hand can win on the tile is
//! hidden, and at a real table the answers to a discard are given at once,
//! so a rule that read either could not be played honestly. With a second
//! winner, declining can cost: a dealer who wins too keeps the deal, and the
//! declined ron only loses its points; a second non-dealer has the
//! discarder pay twice, which can lift the player off last. Rare, and not
//! modelled.
//!
//! # The leading dealer's last discard
//!
//! [`Hand::all_last_dealer_noten`] asks the dealer of the last hand, at their
//! own discard with at most [`NOTEN_WALL`] tiles left to draw, which discards
//! leave the hand waiting at the exhaustive draw that may follow
//! ([`DealerNoten`]), by the predicate that draw is settled by
//! ([`crate::game::Player::is_tenpai`]). South, West and North draw the
//! tiles left, in turn, and the dealer draws no more, so the hand the discard
//! leaves is the hand the draw finds; a call that hands the dealer another
//! turn meets the question again at that turn's discard. A dealer waiting at
//! the draw keeps the deal and South 4 is played again (EMA section 3.4.5); a
//! dealer not waiting passes it, and the game ends (see [`crate::table`]). It
//! is asked only of a dealer more than [`NOTEN_MARGIN`] ahead of every other
//! player. A dealer who has declared riichi may throw only the tile drawn,
//! which keeps the hand waiting, so for one no discard leaves it not
//! waiting.
//!
//! A discard that leaves the hand not waiting is told apart by whether it
//! could be won on, as far as the table shows: [`DealerNoten::Noten`] where
//! the tile is safe against each of the three others ([`Hand::safe_against`]:
//! in their own pond, which makes them furiten on it, or thrown since their
//! riichi and not won on), [`DealerNoten::Unsafe`] where it is not. Any
//! waiting player who is not furiten may win on such a discard (on the last
//! discard of all, even without another yaku: EMA section 4.2.1), and a ron
//! on it could cost the dealer first.
//!
//! If nobody wins, the lead outlasts the hand. Until it ends nobody's points
//! rise but by a win: a riichi another player declares, legal while a tile is
//! left to draw (EMA section 3.3.10), costs its declarer a thousand, and its
//! bet, which no win then collects, goes to the leader at the end of the
//! game; and the noten payments move at most [`NOTEN_MARGIN`] between the
//! dealer and another player. So a dealer left not waiting ends the game at
//! the draw alone in first. A win by another player after the discard ends
//! the game the same whatever the dealer holds, but for what a dealer left
//! waiting could still do: win on a discard yet to come, which would only
//! play South 4 again, where nothing is decided, though it would end the hand
//! before anyone else could win in it; or win beside another player on one
//! discard (EMA section 3.3.1) and keep the deal. Neither is modelled, and
//! the second is rare. Nor is a claim on the tile thrown, which, with tiles
//! left to draw, changes who draws them, and so who might yet win. Nor a
//! quad: with more than [`NOTEN_WALL`] tiles left at the dealer's discard the
//! question is not put, but a quad another player declares later moves a
//! live tile to the dead wall (EMA section 3.3.4) and can take the dealer's
//! last draw, so that discard, the dealer's real last, goes unasked. And the
//! codes do not weigh what a safe discard gives up: a ron on a tile coded
//! [`DealerNoten::Unsafe`] would end the game at once, possibly with the
//! dealer still first, where a tile coded [`DealerNoten::Noten`] lets the
//! hand run on.

use crate::game::{Action, Call, Discard, Hand, Phase};
use crate::hand::TileSet;
use crate::score::{Riichi, Score, ScoreError, WinBy};
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

/// What declaring riichi with a tile would change, in the last hand, in the
/// place the player ends the game in, over discarding the same tile without
/// declaring (see the module's notes). Numbered as [`Dama`] is, a byte a
/// kind, so that 1 to 4 are the tiles riichi may throw.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WideDama {
    /// Not asked: not the last hand, not a non-dealer's own move, no riichi
    /// open, or a kind the hand does not hold.
    Absent = 0,
    /// Every win the hand can make without riichi ends the game in the place
    /// the same win with riichi would, at the least and at the most riichi
    /// could add, so riichi would buy no place.
    NoPlace = 1,
    /// Some live wait has no yaku to win by without riichi, so riichi would
    /// add that win.
    NoYaku = 2,
    /// Some win would end the game in a better place with riichi.
    BuysPlace = 3,
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

/// What discarding a tile leaves the leading dealer at the exhaustive draw
/// that may end the last hand, and whether another player could win on the
/// tile (see the module's notes). A byte a kind.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DealerNoten {
    /// Not asked: not the dealer's own discard in the last hand with at most
    /// [`NOTEN_WALL`] tiles left to draw, a lead of [`NOTEN_MARGIN`] or less
    /// over someone, or a kind the dealer may not discard.
    Absent = 0,
    /// The hand is left waiting: the dealer keeps the deal at the draw, and
    /// South 4 is played again.
    Tenpai = 1,
    /// The hand is left not waiting, so the deal passes at the draw and the
    /// game ends, and the tile is safe against each of the three others: in
    /// their own pond, or thrown since their riichi.
    Noten = 2,
    /// The hand is left not waiting, and the tile is not safe against some
    /// other player, who might win on it for all the table shows.
    Unsafe = 3,
}

/// The most the noten payments of an exhaustive draw move between the dealer
/// and another player when the dealer is not waiting: a lone waiting player
/// takes 3,000 while the dealer pays 1,000, and three waiting players take
/// 1,000 each while the dealer pays 3,000 (EMA 2025 section 3.4.2). A dealer
/// more than this far ahead of every other player ends a game they are noten
/// in alone in first.
pub const NOTEN_MARGIN: i32 = 4_000;

/// The most tiles the live wall may hold at the dealer's own discard for the
/// dealer noten question to be put: South, West and North draw them in turn,
/// and the dealer draws no more, unless a call hands it another turn (see
/// the module's notes).
pub const NOTEN_WALL: usize = 3;

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

    /// As [`Hand::all_last_dama`], asked more widely: for each kind of tile
    /// the turn player holds, whether declaring riichi with it would change
    /// the place any win ends the game in (see [`WideDama`] and the module's
    /// notes). [`WideDama::Absent`] wherever the dama question is not asked.
    /// Nothing in the hand changes.
    pub fn all_last_wide_dama(&self) -> [WideDama; KINDS] {
        let mut codes = [WideDama::Absent; KINDS];
        let Some(legal) = self.last_riichi_open() else {
            return codes;
        };
        let held = self.current().hand;
        for tile in Tile::all().filter(|tile| held.count(*tile) > 0) {
            codes[tile.idx()] = if legal.contains(&Action::Riichi(tile)) {
                self.wide_dama_after(tile)
            } else {
                WideDama::NotWaiting
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

    /// In the last hand (South 4), at the dealer's own discard with at most
    /// [`NOTEN_WALL`] tiles left to draw, when the dealer leads every other
    /// player by more than [`NOTEN_MARGIN`]: for each kind of tile the
    /// dealer may discard, indexed by kind, whether discarding it leaves the
    /// hand waiting at the exhaustive draw that may follow, and if not,
    /// whether the tile is safe against each of the others (see
    /// [`DealerNoten`] and the module's notes). [`DealerNoten::Absent`]
    /// everywhere otherwise, and for the kinds the dealer may not discard.
    /// Nothing in the hand changes.
    pub fn all_last_dealer_noten(&self) -> [DealerNoten; KINDS] {
        let mut codes = [DealerNoten::Absent; KINDS];
        if !matches!(self.phase, Phase::Act)
            || self.round != Wind::South
            || self.kyoku != 4
            || self.turn != Wind::East
            || self.wall.remaining() > NOTEN_WALL
        {
            return codes;
        }
        let points = self.scores();
        let dealer = points[Wind::East.index()];
        if Wind::ALL
            .iter()
            .any(|other| *other != Wind::East && dealer - points[other.index()] <= NOTEN_MARGIN)
        {
            return codes;
        }
        // What the table shows each of the others cannot win on.
        let safe = [Wind::South, Wind::West, Wind::North].map(|other| self.safe_against(other));
        // The hand as the draw finds it, unless a call hands the dealer
        // another turn: the tile gone, and none drawn after it.
        let mut left = self.current().clone();
        for action in self.legal_actions() {
            let Action::Discard(tile) = action else {
                continue;
            };
            left.hand = self.current().hand;
            left.hand.remove(tile);
            codes[tile.idx()] = if left.is_tenpai() {
                DealerNoten::Tenpai
            } else if safe.iter().all(|against| against.count(tile) > 0) {
                DealerNoten::Noten
            } else {
                DealerNoten::Unsafe
            };
        }
        codes
    }

    /// The turn player's legal actions, where the last hand's riichi
    /// questions are put: South 4, a non-dealer's own move, riichi open.
    /// None anywhere else.
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

    /// What declaring riichi with `tile` would change in the place each win
    /// the hand could make ends the game in: every win without riichi, as
    /// [`Hand::dama_after`] makes them, and the same win with riichi at the
    /// least and the most riichi could add, each settled on a copy.
    fn wide_dama_after(&self, tile: Tile) -> WideDama {
        let seat = self.turn;
        let mut after = self.discarded(tile);
        let live = after.live_waits(seat);
        if live.is_empty() {
            return WideDama::Dead;
        }
        let Some(wins) = plain_wins(&mut after, seat, &live) else {
            return WideDama::NoYaku;
        };
        let declared = self.declared(&after);
        let mut least = declared.bounded(seat, None);
        for win in wins {
            let plain = standing(settled(&after, seat, win.score, win.discarder), seat);
            let mut most = declared.bounded(seat, Some(win.wait));
            for riichi in [&mut least, &mut most] {
                // The tiles are the plain win's, and riichi is one more
                // yaku, so a hand that wins without it wins with it.
                let score = win_on(riichi, seat, win.wait, win.by)
                    .expect("riichi adds a yaku to a hand that wins without it");
                if standing(settled(riichi, seat, score, win.discarder), seat) != plain {
                    return WideDama::BuysPlace;
                }
            }
        }
        WideDama::NoPlace
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

    /// The copy `after`, the hand as the turn player's plain discard leaves
    /// it ([`Hand::discarded`]), with that discard made a riichi declaration,
    /// as [`Hand::act`] makes one: the tile turned sideways, a double riichi
    /// if it is the player's first discard with the first set of turns
    /// unbroken, a thousand points paid and the bet placed. The one-shot is
    /// left to [`Hand::bounded`].
    fn declared(&self, after: &Hand) -> Hand {
        let seat = self.turn;
        let double = self.first_turns_unbroken && self.players[seat.index()].discards.is_empty();
        let mut declared = after.clone();
        let player = &mut declared.players[seat.index()];
        player.riichi = if double {
            Riichi::Double
        } else {
            Riichi::Declared
        };
        player.riichi_order = player.discards.last().map(|discard| discard.order);
        if let Some(discard) = player.discards.last_mut() {
            discard.riichi = true;
        }
        player.score -= 1000;
        declared.riichi_sticks += 1;
        declared.bets_this_hand[seat.index()] += 1;
        declared
    }

    /// The copy `self`, a hand as a riichi declaration leaves it
    /// ([`Hand::declared`]), as scoring sees `seat`'s riichi win at the least
    /// riichi adds, `most` None: no one-shot, and a wall that shows no ura
    /// dora; or at the most it could add to a win on `most`: the one-shot,
    /// and under every dora indicator an ura indicator of the kind the hand
    /// holds most copies of once the wait is in it. The wall shows the same
    /// dora and holds as many tiles to draw, which is all scoring reads of it.
    fn bounded(&self, seat: Wind, most: Option<Tile>) -> Hand {
        let mut bounded = self.clone();
        bounded.wall = match most {
            None => self.without_ura(),
            Some(wait) => {
                let mut held = self.players[seat.index()].visible_to_self();
                held.add(wait);
                self.most_ura(&held)
            }
        };
        bounded.players[seat.index()].ippatsu = most.is_some();
        bounded
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
    wait: Tile,
    by: WinBy,
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
                    wait,
                    by: WinBy::Discard,
                    discarder: Some(seat.plus(step)),
                    score: ron.clone(),
                });
            }
        }
        let tsumo = win_on(after, seat, wait, WinBy::SelfDraw).ok()?;
        wins.push(Win {
            wait,
            by: WinBy::SelfDraw,
            discarder: None,
            score: tsumo,
        });
    }
    Some(wins)
}

/// `seat`'s win on `wait`, by `by`, as [`Hand::would_win`] scores it: a
/// self-draw is scored with the tile in hand, as the player holds it once
/// drawn.
fn win_on(hand: &mut Hand, seat: Wind, wait: Tile, by: WinBy) -> Result<Score, ScoreError> {
    if by == WinBy::Discard {
        return hand.would_win(seat, wait, by);
    }
    hand.players[seat.index()].hand.add(wait);
    let score = hand.would_win(seat, wait, by);
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
    use crate::game::Outcome;
    use crate::hand::{ClaimedFrom, Meld};
    use crate::rng::Rng;
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

    fn wide(hand: &Hand, text: &str) -> WideDama {
        hand.all_last_wide_dama()[tile(text).idx()]
    }

    /// What the turn player's discard of `tile` leaves, without declaring and
    /// with riichi declared on it, when every win is played through the real
    /// flow, as [`played_out`] plays them: for each live wait, each other
    /// player in turn throws it and the player rons it (unless furiten), and
    /// the player draws it and declares tsumo; the player's place is read off
    /// the points after each. The riichi wins are played at each of
    /// `bounds`: `false` with the one-shot gone and a wall that shows no ura
    /// dora, `true` with the one-shot alive and under every dora indicator an
    /// ura indicator of the kind the winning hand holds most copies of. Both
    /// give the codes of [`Hand::all_last_wide_dama`] where the hands it asks
    /// are the hands the real discard and declaration leave.
    fn wide_played(hand: &Hand, tile: Tile, bounds: &[bool]) -> WideDama {
        let plain = match played_places(hand, Action::Discard(tile), None) {
            Ok(places) => places,
            Err(code) => return code,
        };
        for &most in bounds {
            let riichi = played_places(hand, Action::Riichi(tile), Some(most))
                .expect("a hand that wins without riichi wins with it");
            assert_eq!(riichi.len(), plain.len(), "the same wins either way");
            if riichi != plain {
                return WideDama::BuysPlace;
            }
        }
        WideDama::NoPlace
    }

    fn wide_played_out(hand: &Hand, tile: Tile) -> WideDama {
        wide_played(hand, tile, &[false, true])
    }

    /// The turn player's place after each win `action` leaves open, played
    /// through the real flow (see [`wide_played`]), with the riichi wins at
    /// the least riichi adds (`Some(false)`) or the most (`Some(true)`).
    fn played_places(
        hand: &Hand,
        action: Action,
        bound: Option<bool>,
    ) -> Result<Vec<(usize, usize)>, WideDama> {
        let seat = hand.turn;
        let mut after = hand.clone();
        after.act(action).expect("the move is legal");
        if matches!(after.phase, Phase::CallWindow) {
            after.resolve_calls(&[]).expect("everyone may pass");
        }
        let waits = after.players[seat.index()].waits();
        let unseen = after.unseen_by(seat);
        let live: Vec<Tile> = Tile::all()
            .filter(|wait| waits.count(*wait) > 0 && unseen.count(*wait) > 0)
            .collect();
        if live.is_empty() {
            return Err(WideDama::Dead);
        }
        let furiten = after.players[seat.index()].is_furiten();
        let mut places = Vec::new();
        for &wait in &live {
            let mut froms: Vec<Option<Wind>> = if furiten {
                Vec::new()
            } else {
                (1..4).map(|step| Some(seat.plus(step))).collect()
            };
            froms.push(None);
            for from in froms {
                let mut won = after.clone();
                if let Some(most) = bound {
                    bound_riichi(&mut won, seat, wait, most);
                }
                won.phase = Phase::Act;
                won.pending_discard = None;
                won.just_claimed = None;
                won.drawn = Some(wait);
                match from {
                    Some(from) => {
                        won.turn = from;
                        won.players[from.index()].hand.add(wait);
                        won.act(Action::Discard(wait))
                            .expect("the wait is thrown as drawn");
                        let offered = won
                            .legal_calls()
                            .iter()
                            .any(|(who, calls)| *who == seat && calls.contains(&Call::Ron));
                        if !offered {
                            return Err(WideDama::NoYaku);
                        }
                        won.resolve_calls(&[(seat, Call::Ron)])
                            .expect("the win was offered");
                    }
                    None => {
                        won.turn = seat;
                        won.after_quad = false;
                        won.players[seat.index()].hand.add(wait);
                        won.players[seat.index()].temporary_furiten = false;
                        if !won.legal_actions().contains(&Action::Tsumo) {
                            return Err(WideDama::NoYaku);
                        }
                        won.act(Action::Tsumo).expect("the win was offered");
                    }
                }
                places.push(standing(won.scores(), seat));
            }
        }
        Ok(places)
    }

    /// `won`, a hand after a real riichi declaration, set for `seat`'s win on
    /// `wait` at the least riichi adds (`most` false: the one-shot gone, and
    /// a wall that shows no ura dora) or the most (the one-shot alive, and an
    /// ura indicator of the kind the winning hand holds most copies of under
    /// every dora indicator), the wall showing the same dora and holding as
    /// many tiles to draw.
    fn bound_riichi(won: &mut Hand, seat: Wind, wait: Tile, most: bool) {
        let held = most.then(|| {
            let mut held = won.players[seat.index()].visible_to_self();
            held.add(wait);
            held
        });
        won.wall = ura_wall(won, held);
        won.players[seat.index()].ippatsu = most;
    }

    /// A wall that shows the dora of `hand`'s and holds as many tiles to
    /// draw, with under every dora indicator an ura indicator of the kind
    /// `held`, the tiles of a winning hand, holds most copies of, or, with
    /// `held` None, no ura dora shown.
    fn ura_wall(hand: &Hand, held: Option<TileSet>) -> Wall {
        let indicators = hand.wall.dora_indicators();
        let wall =
            Wall::for_analysis(hand.wall.remaining(), &indicators, indicators.len() - 1).unwrap();
        let Some(held) = held else {
            return wall;
        };
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
            won.wall = ura_wall(hand, Some(held));
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

    /// What the dealer's discard did when the rest of the hand was played
    /// quietly (see [`played_quietly`]).
    struct Quiet {
        /// Whether any seat was offered a ron on the dealer's discard.
        ron_offered: bool,
        /// Whether the dealer was waiting at the exhaustive draw.
        waiting: bool,
        /// Whether the table's game ended with the hand.
        ended: bool,
        /// Whether the table's final scores leave the dealer alone in first.
        alone: bool,
    }

    /// The dealer's discard of `tile` played, and the rest of the hand with
    /// it, quietly: every claim on it and on every later discard passed, and
    /// each player who draws after it throwing the tile drawn, with no
    /// quad, no riichi and no win, to the exhaustive draw, which a copy of
    /// the table then takes. The dealer must draw no more, and the hand must
    /// end in the draw.
    fn played_quietly(hand: &Hand, table: &Table, tile: Tile) -> Quiet {
        let mut after = hand.clone();
        after
            .act(Action::Discard(tile))
            .expect("the discard is legal");
        let ron_offered = after
            .legal_calls()
            .iter()
            .any(|(_, calls)| calls.contains(&Call::Ron));
        while !matches!(after.phase, Phase::Over) {
            assert!(
                !(matches!(after.phase, Phase::Draw) && after.turn == Wind::East),
                "the dealer draws no more"
            );
            quiet_step(&mut after);
        }
        let Some(Outcome::ExhaustiveDraw { tenpai }) = &after.outcome else {
            panic!("a hand played quietly ends in an exhaustive draw");
        };
        let waiting = tenpai.contains(&Wind::East);
        let dealer = table.player_at(Wind::East);
        let mut game = table.clone();
        game.finish(&after);
        let finals = game.final_scores();
        Quiet {
            ron_offered,
            waiting,
            ended: game.finished,
            alone: (0..4).all(|other| other == dealer || finals[other] < finals[dealer]),
        }
    }

    /// The dealer's discard of `tile`, coded `code`, played quietly: the
    /// code's promises kept. A discard coded [`DealerNoten::Tenpai`] leaves
    /// the dealer waiting and South 4 to be played again; any other leaves
    /// it not waiting, and the game ends with the dealer alone in first; and
    /// nobody is offered a ron on a discard coded [`DealerNoten::Noten`].
    /// What it did, for the counts.
    fn quiet_finish_agrees(hand: &Hand, table: &Table, tile: Tile, code: DealerNoten) -> Quiet {
        let quiet = played_quietly(hand, table, tile);
        let kept = code != DealerNoten::Absent
            && quiet.waiting == (code == DealerNoten::Tenpai)
            && quiet.ended == !quiet.waiting
            && (quiet.waiting || quiet.alone)
            && !(code == DealerNoten::Noten && quiet.ron_offered);
        assert!(
            kept,
            "{tile} coded {code:?}: ron offered {}, waiting {}, ended {}, alone in first {}",
            quiet.ron_offered, quiet.waiting, quiet.ended, quiet.alone
        );
        quiet
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
    /// those with a tile that keeps first, and the riichi tiles by code; the
    /// riichi tiles by their wide code, and the decisions with a tile whose
    /// riichi buys no place; the wins open to a non-dealer in the last hand
    /// by decline code, and those coded not last, riichi wins all, that the
    /// ura dora as dealt would have left alone in last; the leading dealer's
    /// last discards asked, those with a safe discard that leaves the hand
    /// not waiting, those with one as well as one that keeps it waiting, the
    /// legal discards there by code, and the unsafe ones another player was
    /// offered a ron on when played. With them, the first games a win that
    /// leaves the player last, a leading dealer's last discard, one with a
    /// safe discard that leaves the hand not waiting, and one with that and
    /// a discard that keeps it waiting too fell in.
    #[derive(Default, Debug)]
    struct Counted {
        decisions: usize,
        asked: usize,
        keeping: usize,
        tiles: [usize; 6],
        wide: [usize; 6],
        no_place: usize,
        wins: [usize; 3],
        last_as_dealt: usize,
        noten_asked: usize,
        noten_safe: usize,
        noten_choice: usize,
        noten: [usize; 4],
        unsafe_ron: usize,
        last_games: Vec<u64>,
        noten_games: Vec<u64>,
        safe_games: Vec<u64>,
        choice_games: Vec<u64>,
    }

    /// Plays one game of four Club players for each of `games`, dealt from
    /// its own seed, and checks the codes at every decision: at a turn, the
    /// dama and wide dama codes given exactly where it is the last hand, a
    /// non-dealer's move and riichi open, a riichi tile coded 1 to 4 and
    /// every other tile held not waiting, each riichi tile's codes the ones
    /// the real flow gives it, and the wide code the one the dama code
    /// allows; the decline code of every seat given exactly where a
    /// non-dealer of the last hand has a win open, at a turn and in a claim
    /// window, the one the real flow and the table's final scores give it
    /// with a riichi win at the most its ura dora could add, and a code of
    /// last kept on the wall as dealt; and the dealer noten codes given
    /// exactly at the leading dealer's own discard with at most three tiles
    /// left to draw, on the legal discards, each kept when the rest of the
    /// hand is played quietly (see [`quiet_finish_agrees`]).
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
        let wider = hand.all_last_wide_dama();
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
        assert_eq!(
            wider.iter().any(|code| *code != WideDama::Absent),
            asked,
            "{:?} {} {:?}: {wider:?}",
            hand.round,
            hand.kyoku,
            hand.turn
        );
        for seat in Wind::ALL {
            let open = seat == hand.turn && legal.contains(&Action::Tsumo);
            check_decline(hand, table, game, seat, open, counted);
        }
        check_noten(hand, table, game, &legal, counted);
        if !asked {
            return;
        }
        counted.asked += 1;
        let held = hand.current().hand;
        for kind in Tile::all() {
            let (code, wide) = (codes[kind.idx()], wider[kind.idx()]);
            if riichi(kind) {
                assert!(
                    matches!(
                        code,
                        Dama::KeepsFirst | Dama::NoYaku | Dama::Short | Dama::Dead
                    ),
                    "{kind}: {code:?}"
                );
                assert_eq!(played_out(hand, kind), code, "{kind} in {held}");
                assert_eq!(wide_played_out(hand, kind), wide, "{kind} in {held}");
                counted.tiles[code as usize] += 1;
                counted.wide[wide as usize] += 1;
            } else if held.count(kind) > 0 {
                assert_eq!(code, Dama::NotWaiting, "{kind} in {held}");
            } else {
                assert_eq!(code, Dama::Absent, "{kind} not in {held}");
            }
            let agree = match code {
                Dama::Absent => wide == WideDama::Absent,
                Dama::KeepsFirst => wide == WideDama::NoPlace,
                Dama::NoYaku => wide == WideDama::NoYaku,
                Dama::Short => matches!(wide, WideDama::NoPlace | WideDama::BuysPlace),
                Dama::Dead => wide == WideDama::Dead,
                Dama::NotWaiting => wide == WideDama::NotWaiting,
            };
            assert!(agree, "{kind} in {held}: {code:?} and {wide:?}");
        }
        if codes.contains(&Dama::KeepsFirst) {
            counted.keeping += 1;
        }
        if wider.contains(&WideDama::NoPlace) {
            counted.no_place += 1;
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

    fn check_noten(hand: &Hand, table: &Table, game: u64, legal: &[Action], counted: &mut Counted) {
        let codes = hand.all_last_dealer_noten();
        let points = hand.scores();
        let lead = Wind::ALL
            .iter()
            .filter(|other| **other != Wind::East)
            .map(|other| points[Wind::East.index()] - points[other.index()])
            .min()
            .expect("three others");
        let asked = hand.round == Wind::South
            && hand.kyoku == 4
            && hand.turn == Wind::East
            && hand.wall.remaining() <= NOTEN_WALL
            && lead > NOTEN_MARGIN;
        assert_eq!(
            codes.iter().any(|code| *code != DealerNoten::Absent),
            asked,
            "{:?} {} {:?}, {} to draw, a lead of {lead}: {codes:?}",
            hand.round,
            hand.kyoku,
            hand.turn,
            hand.wall.remaining()
        );
        if !asked {
            return;
        }
        counted.noten_asked += 1;
        if counted.noten_games.len() < 10 {
            counted.noten_games.push(game);
        }
        for kind in Tile::all() {
            let code = codes[kind.idx()];
            if legal.contains(&Action::Discard(kind)) {
                let quiet = quiet_finish_agrees(hand, table, kind, code);
                counted.noten[code as usize] += 1;
                counted.unsafe_ron += usize::from(code == DealerNoten::Unsafe && quiet.ron_offered);
            } else {
                assert_eq!(code, DealerNoten::Absent, "{kind}, game {game}");
            }
        }
        if codes.contains(&DealerNoten::Noten) {
            counted.noten_safe += 1;
            if counted.safe_games.len() < 10 {
                counted.safe_games.push(game);
            }
            if codes.contains(&DealerNoten::Tenpai) {
                counted.noten_choice += 1;
                if counted.choice_games.len() < 10 {
                    counted.choice_games.push(game);
                }
            }
        }
    }

    /// Forty games, about a minute unoptimised on the desktop (two hundred
    /// took five and a half): 67 of their turns are a non-dealer's in South
    /// 4 with riichi open, and their riichi tiles fall 13 keeping first, 48
    /// with no yaku, 32 short and 1 dead, and by the wide question 20 buying
    /// no place, 48 with no yaku, 25 buying one and 1 dead; 35 wins are open
    /// to a non-dealer in South 4, 5 of them leaving the player alone in
    /// last, and 4 more, riichi wins, that the ura dora as dealt would have
    /// left there. A leading dealer's own discard with at most three tiles
    /// left comes 4 times, in games 9,100,017 (twice), 9,100,026 and
    /// 9,100,037, and its 40 legal discards fall 1 keeping the hand waiting,
    /// 4 leaving it not waiting on a safe tile and 35 on an unsafe one, 3 of
    /// which another player was offered a ron on when played; two of the
    /// four have a safe discard that leaves the hand not waiting, and one of
    /// them a discard that keeps it waiting as well.
    #[test]
    fn club_games_agree_with_the_real_flow() {
        let began = std::time::Instant::now();
        let counted = club_games(9_100_000..9_100_040);
        println!("{counted:?} in {:.1} s", began.elapsed().as_secs_f64());
        assert!(counted.asked > 20, "{counted:?}");
        for code in [Dama::KeepsFirst, Dama::NoYaku, Dama::Short] {
            assert!(counted.tiles[code as usize] > 0, "{code:?}: {counted:?}");
        }
        // The wide question finds more tiles whose riichi buys no place than
        // keep first.
        assert!(
            counted.wide[WideDama::NoPlace as usize] > counted.tiles[Dama::KeepsFirst as usize],
            "{counted:?}"
        );
        assert!(
            counted.wide[WideDama::BuysPlace as usize] > 0,
            "{counted:?}"
        );
        for code in [Decline::Last, Decline::NotLast] {
            assert!(counted.wins[code as usize] > 0, "{code:?}: {counted:?}");
        }
        assert!(counted.noten_asked > 0, "{counted:?}");
        for code in [DealerNoten::Tenpai, DealerNoten::Noten, DealerNoten::Unsafe] {
            assert!(counted.noten[code as usize] > 0, "{code:?}: {counted:?}");
        }
        assert!(counted.noten_choice > 0, "{counted:?}");
    }

    /// The share of the last hand's riichi decisions with a tile that keeps
    /// first, and the riichi tiles by code, over many Club games; the riichi
    /// tiles by wide code; the wins open to a non-dealer in the last hand by
    /// decline code, and the riichi wins coded not last that the ura dora as
    /// dealt would have left there; and the leading dealer's last discards,
    /// those with a safe discard that leaves the hand not waiting, and their
    /// legal discards by code. Long: run with
    /// `cargo test --release -p riichi-core -- --ignored --nocapture`, which
    /// took 38 minutes on a busy desktop (31 October 2026) and counted
    /// 28,719 riichi decisions of the last hand, 19.16% with a tile whose
    /// riichi buys no place (14.72% with one that keeps first); 18,623 wins
    /// open to a non-dealer there, 2,012 of them leaving the player alone in
    /// last, and 868 more, riichi wins, that the ura dora as dealt would
    /// have left there; and 1,052 leading dealer's own discards with at most
    /// three tiles left, 230 of them with a safe discard that leaves the
    /// hand not waiting (51 with one that keeps it waiting as well), their
    /// 7,007 legal discards falling 795 waiting, 322 not waiting and safe
    /// and 5,890 not waiting and unsafe, 537 of which another player was
    /// offered a ron on when played.
    #[test]
    #[ignore]
    fn club_games_counted() {
        let began = std::time::Instant::now();
        let counted = club_games(9_100_000..9_120_000);
        println!(
            "{} decisions at a turn; {} in the last hand with riichi open, {} of them ({:.2}%) with a tile \
             that keeps first, {} ({:.2}%) with one whose riichi buys no place",
            counted.decisions,
            counted.asked,
            counted.keeping,
            100.0 * counted.keeping as f64 / counted.asked.max(1) as f64,
            counted.no_place,
            100.0 * counted.no_place as f64 / counted.asked.max(1) as f64
        );
        for (name, code) in [
            ("keeps first", Dama::KeepsFirst),
            ("no yaku", Dama::NoYaku),
            ("short", Dama::Short),
            ("dead", Dama::Dead),
        ] {
            println!("riichi tiles that {name}: {}", counted.tiles[code as usize]);
        }
        for (name, code) in [
            ("buy no place", WideDama::NoPlace),
            ("have no yaku", WideDama::NoYaku),
            ("buy a place", WideDama::BuysPlace),
            ("are dead", WideDama::Dead),
        ] {
            println!("riichi tiles that {name}: {}", counted.wide[code as usize]);
        }
        println!(
            "wins open to a non-dealer in the last hand: {} leave the player last, {} do not, {} of these \
             riichi wins the ura as dealt would have left last; first games with one that does: {:?}",
            counted.wins[Decline::Last as usize],
            counted.wins[Decline::NotLast as usize],
            counted.last_as_dealt,
            counted.last_games
        );
        println!(
            "leading dealer's last discards: {}, first games: {:?}",
            counted.noten_asked, counted.noten_games
        );
        println!(
            "of them with a safe discard that leaves the hand not waiting: {}, first games: {:?}; with one that \
             keeps it waiting too: {}, first games: {:?}",
            counted.noten_safe, counted.safe_games, counted.noten_choice, counted.choice_games
        );
        println!(
            "their legal discards: {} waiting, {} not waiting and safe, {} not waiting and unsafe, {} of these \
             last offered to another player's ron when played",
            counted.noten[DealerNoten::Tenpai as usize],
            counted.noten[DealerNoten::Noten as usize],
            counted.noten[DealerNoten::Unsafe as usize],
            counted.unsafe_ron
        );
        println!("in {:.1} s", began.elapsed().as_secs_f64());
    }

    // Riichi that buys no place.

    /// A tile that keeps first is one whose riichi buys no place: the probe
    /// at gaps under 1,000 and leading by 4,000, and the furiten probe at a
    /// gap of 1,500.
    #[test]
    fn a_tile_that_keeps_first_buys_no_place() {
        for hand in [probe(0), probe(900), probe(-4_000), furiten(1_500)] {
            let scores = hand.scores();
            assert_eq!(code(&hand, "9p"), Dama::KeepsFirst, "{scores:?}");
            assert_eq!(wide(&hand, "9p"), WideDama::NoPlace, "{scores:?}");
            assert_eq!(
                wide_played_out(&hand, tile("9p")),
                WideDama::NoPlace,
                "{scores:?}"
            );
        }
    }

    /// The wide question is put where the dama question is, and answers as
    /// it does wherever its answer is not a place: no yaku, a dead wait, a
    /// tile that leaves the hand not waiting, a kind not held, and nothing
    /// at all for the dealer, in another hand, or with no riichi open.
    #[test]
    fn wide_dama_is_asked_where_dama_is() {
        let north = |held: &str| {
            position(
                Wind::South,
                4,
                Wind::North,
                held,
                "4z",
                [25_000, 25_000, 25_000, 45_000],
            )
        };
        let mut dead = north("234m678p3455788s4z");
        for order in 0..4 {
            dead.players[order as usize % 3].discards.push(Discard {
                tile: tile("6s"),
                order,
                drawn: false,
                riichi: false,
                claimed: false,
            });
        }
        dead.discards_made = 4;
        let dealer = position(
            Wind::South,
            4,
            Wind::East,
            "23456m456p678s33s9p",
            "9p",
            [60_000, 20_000, 20_000, 20_000],
        );
        let mut south_three = probe(0);
        south_three.kyoku = 3;
        let mut poor = probe(0);
        poor.players[Wind::North.index()].score = 900;
        let hands = [
            north("123789m234p5799s4z"),
            north("23478m678p22255s4z"),
            dead,
            dealer,
            south_three,
            poor,
            probe(1_500),
        ];
        for hand in hands {
            let (codes, wider) = (hand.all_last_dama(), hand.all_last_wide_dama());
            for kind in Tile::all() {
                let (code, wide) = (codes[kind.idx()], wider[kind.idx()]);
                let agree = match code {
                    Dama::Absent => wide == WideDama::Absent,
                    Dama::KeepsFirst => wide == WideDama::NoPlace,
                    Dama::NoYaku => wide == WideDama::NoYaku,
                    Dama::Short => matches!(wide, WideDama::NoPlace | WideDama::BuysPlace),
                    Dama::Dead => wide == WideDama::Dead,
                    Dama::NotWaiting => wide == WideDama::NotWaiting,
                };
                assert!(agree, "{kind}: {code:?} and {wide:?}");
            }
        }
    }

    /// North on 22,000, third behind West's 32,000 and South's 36,000, East
    /// on 10,000, on the probe's hand. No win without riichi moves it, nor
    /// any with riichi at the least it adds: the best is a 4m ron of riichi,
    /// pinfu and tanyao, 3,900, and West is 10,000 ahead. But at the most
    /// riichi could add, with the one-shot and the two ura dora the 3s pair
    /// could take, a 1m ron is a mangan, and off West it lifts North past
    /// West: riichi buys a place.
    #[test]
    fn the_most_riichi_could_add_is_counted() {
        let hand = position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [10_000, 36_000, 32_000, 22_000],
        );
        assert_eq!(code(&hand, "9p"), Dama::Short);
        assert_eq!(wide(&hand, "9p"), WideDama::BuysPlace);
        assert_eq!(wide_played(&hand, tile("9p"), &[false]), WideDama::NoPlace);
        assert_eq!(wide_played_out(&hand, tile("9p")), WideDama::BuysPlace);
    }

    /// North far in last on 5,000, 25,000 behind West and South, East on
    /// 35,000: short of first, but no win lifts it, with riichi or without.
    /// At the most riichi could add a 4m or 7m ron is a haneman, 12,000,
    /// which leaves it 1,000 behind its payer.
    #[test]
    fn a_player_no_win_can_lift_buys_no_place() {
        let hand = position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [35_000, 30_000, 30_000, 5_000],
        );
        assert_eq!(code(&hand, "9p"), Dama::Short);
        assert_eq!(wide(&hand, "9p"), WideDama::NoPlace);
        assert_eq!(wide_played_out(&hand, tile("9p")), WideDama::NoPlace);
    }

    /// EMA 2025 section 3.7.1: tied players share the uma, so a tie riichi
    /// would break is a place it buys. At a gap of 1,000 the probe's 1m ron
    /// off East leaves North level with South; with riichi, 2,000, and the
    /// thousand it cost back as the bet the winner takes, alone in first.
    #[test]
    fn a_tie_riichi_would_break_is_a_place_bought() {
        let hand = probe(1_000);
        assert_eq!(code(&hand, "9p"), Dama::Short);
        assert_eq!(wide(&hand, "9p"), WideDama::BuysPlace);
        assert_eq!(
            wide_played(&hand, tile("9p"), &[false]),
            WideDama::BuysPlace
        );
    }

    /// The counters and the bets on the table are paid with every win, with
    /// riichi or without (EMA 2025 sections 3.4.4 and 3.3.10): with two
    /// counters the probe's 1m ron is 1,600, a tie at a gap of 1,600 that
    /// riichi's 2,600 would break; with a bet on the table it is 2,000, a
    /// tie at a gap of 2,000.
    #[test]
    fn counters_and_bets_are_paid_with_riichi_or_without() {
        for (gap, want) in [(1_500, WideDama::NoPlace), (1_600, WideDama::BuysPlace)] {
            let mut hand = probe(gap);
            hand.counters = 2;
            assert_eq!(wide(&hand, "9p"), want, "a gap of {gap}, two counters");
            assert_eq!(wide_played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
        for (gap, want) in [(1_900, WideDama::NoPlace), (2_000, WideDama::BuysPlace)] {
            let mut hand = probe(gap);
            hand.riichi_sticks = 1;
            assert_eq!(wide(&hand, "9p"), want, "a gap of {gap}, a bet");
            assert_eq!(wide_played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
    }

    /// The furiten probe counts its self-draws alone: at a gap of 1,900 its
    /// 1m tsumo ties North with South, and riichi's 700/1,300 would put it
    /// ahead.
    #[test]
    fn a_furiten_hand_counts_its_self_draws_alone_with_riichi_too() {
        for (gap, want) in [(1_500, WideDama::NoPlace), (1_900, WideDama::BuysPlace)] {
            let hand = furiten(gap);
            assert_eq!(wide(&hand, "9p"), want, "a gap of {gap}");
            assert_eq!(wide_played_out(&hand, tile("9p")), want, "a gap of {gap}");
        }
    }

    /// A double riichi, declared with the player's first discard while the
    /// first set of turns is unbroken (EMA 2025 section 4.2.2), counts its
    /// second han. 234m 678p 345s 57s 88s waits on the 6s alone, all simples,
    /// 1,300 by ron. North on 10,000 is 20,000 behind each of the others: at
    /// the most a riichi could add, the one-shot and two ura dora, the ron is
    /// a mangan, 8,000, and lifts it past nobody; a double riichi's is a
    /// haneman, 12,000, which lifts it past its payer.
    #[test]
    fn a_double_riichi_counts_its_second_han() {
        for (unbroken, want) in [(false, WideDama::NoPlace), (true, WideDama::BuysPlace)] {
            let mut hand = position(
                Wind::South,
                4,
                Wind::North,
                "234m678p3455788s4z",
                "4z",
                [30_000, 30_000, 30_000, 10_000],
            );
            hand.first_turns_unbroken = unbroken;
            assert_eq!(wide(&hand, "4z"), want, "first turns unbroken: {unbroken}");
            assert_eq!(
                wide_played_out(&hand, tile("4z")),
                want,
                "first turns unbroken: {unbroken}"
            );
        }
    }

    /// The most riichi could add puts under every dora indicator an ura
    /// indicator of the kind the winning hand holds most copies of, a quad's
    /// four among them, and keeps the one-shot alive; the least shows no ura
    /// dora and no one-shot. North holds a concealed quad of 5s and 23456m
    /// 456p 33s, two dora indicators showing the south wind, and throws the
    /// 9p: its 1m tsumo, with riichi, takes eight ura dora at the most.
    #[test]
    fn the_riichi_bounds_are_the_one_shot_and_the_kind_held_most() {
        let mut hand = probe(0);
        let north = &mut hand.players[Wind::North.index()];
        north.hand = "23456m456p33s9p".parse().unwrap();
        north.melds = vec![Meld::concealed_kan(tile("5s"))];
        hand.wall = Wall::for_analysis(30, &[tile("2z"), tile("2z")], 1).unwrap();
        let after = hand.discarded(tile("9p"));
        let declared = hand.declared(&after);
        let north = &declared.players[Wind::North.index()];
        assert_eq!(north.riichi, Riichi::Declared);
        assert_eq!(north.score, 35_000);
        assert_eq!(
            (declared.riichi_sticks, declared.bets_this_hand),
            (1, [0, 0, 0, 1])
        );
        let mut least = declared.bounded(Wind::North, None);
        let mut most = declared.bounded(Wind::North, Some(tile("1m")));
        let low = win_on(&mut least, Wind::North, tile("1m"), WinBy::SelfDraw).unwrap();
        let high = win_on(&mut most, Wind::North, tile("1m"), WinBy::SelfDraw).unwrap();
        assert_eq!(low.dora, 0);
        assert_eq!(high.dora, 8);
        assert_eq!(high.han, low.han + 1 + 8, "the one-shot and eight ura dora");
        assert!(least.wall.ura_indicators().is_empty());
        assert_eq!(most.wall.dora_indicators(), hand.wall.dora_indicators());
        assert_eq!(most.wall.remaining(), hand.wall.remaining());
    }

    /// The dealer's win keeps the deal, so the dealer is never asked; nor in
    /// any hand but the last.
    #[test]
    fn wide_dama_asks_only_a_non_dealer_in_the_last_hand() {
        let dealer = position(
            Wind::South,
            4,
            Wind::East,
            "23456m456p678s33s9p",
            "9p",
            [60_000, 20_000, 20_000, 20_000],
        );
        assert_eq!(dealer.all_last_wide_dama(), [WideDama::Absent; KINDS]);
        for (round, kyoku) in [(Wind::East, 4), (Wind::South, 3)] {
            let mut hand = probe(0);
            hand.round = round;
            hand.kyoku = kyoku;
            assert_eq!(
                hand.all_last_wide_dama(),
                [WideDama::Absent; KINDS],
                "{round:?} {kyoku}"
            );
        }
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

    // The leading dealer's last discard.

    /// Hands that are not waiting, by seat, for the dealer's opponents.
    const IDLE: [&str; 3] = [
        "1z1z2z5z6z7z1p4p7p2s5s8s3m",
        "3z3z4z4z5z6z7z2p5p8p3s6s9s",
        "1z2z3z4z5z6z7z9s9s3p6p9p2m",
    ];
    /// Hands that are waiting, on 1p and 4p, 1s and 4s, and 4s and 7s.
    const READY: [&str; 3] = [
        "123m789m456s11p23p",
        "456m123p789p11s23s",
        "234p678p345m22s56s",
    ];

    /// South 4, East to move with no tile left to draw, on 23456m 456p 678s
    /// 33s with the 9p just drawn: throwing the 9p keeps the hand waiting on
    /// 1m, 4m and 7m, and any other tile leaves it not waiting. The points by
    /// seat; the opponents named in `waiting` hold waiting hands, the others
    /// hands that are not. Every pond is empty, so no tile is safe against
    /// anyone.
    fn last_discard(scores: [i32; 4], waiting: &[Wind]) -> Hand {
        let mut hand = position(
            Wind::South,
            4,
            Wind::East,
            "23456m456p678s33s9p",
            "9p",
            scores,
        );
        hand.wall = Wall::for_analysis(0, &[tile("2z")], 0).unwrap();
        for (at, seat) in [Wind::South, Wind::West, Wind::North]
            .into_iter()
            .enumerate()
        {
            let held = if waiting.contains(&seat) {
                READY[at]
            } else {
                IDLE[at]
            };
            hand.players[seat.index()].hand = held.parse().unwrap();
        }
        hand
    }

    /// `hand` with `left` tiles left to draw. An analysis wall's tiles are
    /// all 1m, which of these hands only the dealer's waits on, after the
    /// 9p, and which a player who draws one throws again when the hand is
    /// played quietly.
    fn with_left(mut hand: Hand, left: usize) -> Hand {
        hand.wall = Wall::for_analysis(left, &[tile("2z")], 0).unwrap();
        hand
    }

    fn noten(hand: &Hand, text: &str) -> DealerNoten {
        hand.all_last_dealer_noten()[tile(text).idx()]
    }

    /// Every legal discard of the dealer's in `hand` played quietly, each
    /// keeping its code's promises (see [`quiet_finish_agrees`]).
    fn every_discard_agrees(hand: &Hand) {
        let codes = hand.all_last_dealer_noten();
        for action in hand.legal_actions() {
            if let Action::Discard(tile) = action {
                quiet_finish_agrees(hand, &last_table(), tile, codes[tile.idx()]);
            }
        }
    }

    /// `seat` throws `text`, the hand's next discard, as a riichi
    /// declaration if `riichi`.
    fn throw(hand: &mut Hand, seat: Wind, text: &str, riichi: bool) {
        let order = hand.discards_made;
        hand.discards_made += 1;
        hand.players[seat.index()].discards.push(Discard {
            tile: tile(text),
            order,
            drawn: false,
            riichi,
            claimed: false,
        });
    }

    /// `last_discard`'s dealer, led by 4,100 with two tiles left to draw,
    /// after these discards in turn: South, West and North the 2m, East the
    /// 5p, South and West the 3m and the 5p, North a riichi with the north
    /// wind, and South and West the 4p. South waits on 2m and 5m, furiten on
    /// the 2m in its pond; West on 3s and 6s, with the round wind's triplet;
    /// North on 1p and 4p, furiten since it passed the 4p after declaring.
    /// So the 2m is safe against each of the three, in their own ponds, and
    /// so is the 4p, in South's and West's and thrown since North's riichi;
    /// the 3m and the 5p are in South's and West's ponds but were thrown
    /// before North's riichi, and the dealer's other tiles are nowhere.
    fn ponds() -> Hand {
        let mut hand = with_left(last_discard([40_100, 36_000, 12_000, 11_900], &[]), 2);
        for (seat, held) in [
            (Wind::South, "34m789m345s678s99s"),
            (Wind::West, "999m45s111z222z77z"),
            (Wind::North, "567m11p23p456s789s"),
        ] {
            hand.players[seat.index()].hand = held.parse().unwrap();
        }
        for (seat, text) in [
            (Wind::South, "2m"),
            (Wind::West, "2m"),
            (Wind::North, "2m"),
            (Wind::East, "5p"),
            (Wind::South, "3m"),
            (Wind::West, "3m"),
            (Wind::South, "5p"),
            (Wind::West, "5p"),
        ] {
            throw(&mut hand, seat, text, false);
        }
        throw(&mut hand, Wind::North, "4z", true);
        let north = &mut hand.players[Wind::North.index()];
        north.riichi = Riichi::Declared;
        north.riichi_order = Some(8);
        north.score -= 1_000;
        hand.riichi_sticks += 1;
        hand.bets_this_hand[Wind::North.index()] += 1;
        for seat in [Wind::South, Wind::West] {
            throw(&mut hand, seat, "4p", false);
        }
        for seat in [Wind::South, Wind::West, Wind::North] {
            hand.players[seat.index()].refresh_furiten();
        }
        // As the claim windows of the two 4p left it: a riichi player who
        // passes a winning tile is furiten to the end of the hand (EMA 2025
        // section 3.3.9).
        hand.players[Wind::North.index()].furiten = true;
        hand
    }

    /// One step of a hand played quietly: every claim passed, a draw taken,
    /// or the tile drawn thrown.
    fn quiet_step(hand: &mut Hand) {
        match hand.phase {
            Phase::CallWindow => hand.resolve_calls(&[]).expect("everyone may pass"),
            Phase::Draw => {
                hand.draw().expect("a tile is left to draw");
            }
            Phase::Act => {
                let drawn = hand.drawn.expect("a player who drew acts");
                hand.act(Action::Discard(drawn))
                    .expect("the tile drawn may be thrown");
            }
            Phase::Over => panic!("the hand is over"),
        }
    }

    /// Led by 4,100 with up to three tiles left to draw: the 9p keeps the
    /// hand waiting, and every other tile the dealer holds leaves it not
    /// waiting, as the draw finds it, on a tile nothing shows safe; the kinds
    /// it does not hold are not asked about. Played quietly, a dealer left
    /// waiting keeps the deal and one not waiting ends the game alone in
    /// first, whoever else is waiting.
    #[test]
    fn the_leading_dealers_discards_by_the_draws_own_predicate() {
        for left in 0..=NOTEN_WALL {
            for waiting in [
                &[][..],
                &[Wind::South],
                &[Wind::South, Wind::West, Wind::North],
            ] {
                let scores = [40_100, 36_000, 12_000, 11_900];
                let hand = with_left(last_discard(scores, waiting), left);
                let codes = hand.all_last_dealer_noten();
                let held = hand.players[Wind::East.index()].hand;
                for kind in Tile::all() {
                    let want = if kind == tile("9p") {
                        DealerNoten::Tenpai
                    } else if held.count(kind) > 0 {
                        DealerNoten::Unsafe
                    } else {
                        DealerNoten::Absent
                    };
                    assert_eq!(
                        codes[kind.idx()],
                        want,
                        "{kind}, {left} left, {waiting:?} waiting"
                    );
                }
                every_discard_agrees(&hand);
            }
        }
    }

    /// EMA 2025 section 3.4.2: the noten payments move at most 4,000 between
    /// the dealer and another player, so a lead of 4,000 is not enough. Not
    /// waiting, a dealer led by 4,000 ends level with South when South alone
    /// is waiting (South takes 3,000 and the dealer pays 1,000), and when all
    /// three are (each takes 1,000 of the dealer's 3,000); led by 4,100, it
    /// ends alone in first either way.
    #[test]
    fn the_lead_must_be_more_than_four_thousand() {
        for waiting in [&[Wind::South][..], &[Wind::South, Wind::West, Wind::North]] {
            for (scores, asked) in [
                ([40_000, 36_000, 12_000, 12_000], false),
                ([40_100, 36_000, 12_000, 11_900], true),
            ] {
                let hand = last_discard(scores, waiting);
                assert_eq!(
                    noten(&hand, "3s") != DealerNoten::Absent,
                    asked,
                    "{scores:?}"
                );
                let quiet = played_quietly(&hand, &last_table(), tile("3s"));
                assert!(quiet.ended && !quiet.waiting, "{scores:?}");
                assert_eq!(quiet.alone, asked, "{scores:?}, {waiting:?} waiting");
            }
        }
    }

    /// The question is put with up to three tiles left to draw, which South,
    /// West and North draw while the dealer draws no more; with four, the
    /// dealer draws the last, and the question waits for that turn.
    #[test]
    fn the_question_is_put_with_at_most_three_tiles_left() {
        let hand = last_discard([40_100, 36_000, 12_000, 11_900], &[]);
        for left in 0..=NOTEN_WALL {
            let asked = with_left(hand.clone(), left);
            assert_eq!(noten(&asked, "9p"), DealerNoten::Tenpai, "{left} left");
        }
        let mut early = with_left(hand, NOTEN_WALL + 1);
        assert_eq!(early.all_last_dealer_noten(), [DealerNoten::Absent; KINDS]);
        early.act(Action::Discard(tile("3s"))).unwrap();
        while !(matches!(early.phase, Phase::Act) && early.turn == Wind::East) {
            quiet_step(&mut early);
        }
        assert!(early.wall.is_empty());
        assert_ne!(early.all_last_dealer_noten(), [DealerNoten::Absent; KINDS]);
    }

    /// Safe means safe as the table shows it: in each other player's own
    /// pond, or thrown since their riichi (see `ponds`). Played quietly, no
    /// one may win on a safe discard, and West may win on the 3s and the 6s,
    /// which are not.
    #[test]
    fn a_discard_is_safe_only_where_the_table_shows_it() {
        let hand = ponds();
        let codes = hand.all_last_dealer_noten();
        let held = hand.players[Wind::East.index()].hand;
        for kind in Tile::all() {
            let want = if kind == tile("9p") {
                DealerNoten::Tenpai
            } else if [tile("2m"), tile("4p")].contains(&kind) {
                DealerNoten::Noten
            } else if held.count(kind) > 0 {
                DealerNoten::Unsafe
            } else {
                DealerNoten::Absent
            };
            assert_eq!(codes[kind.idx()], want, "{kind}");
        }
        every_discard_agrees(&hand);
        for text in ["3s", "6s"] {
            let quiet = played_quietly(&hand, &last_table(), tile(text));
            assert!(quiet.ron_offered, "{text}");
        }
    }

    /// With three tiles left, South claims the dealer's 3s for a triplet:
    /// West and North draw after it, and the dealer draws the last, a turn
    /// the claim handed it, where the question is put again. It is put to
    /// nobody in between.
    #[test]
    fn a_call_that_hands_the_dealer_another_turn_meets_the_question_again() {
        let scores = [40_100, 36_000, 12_000, 11_900];
        let mut hand = with_left(last_discard(scores, &[]), 3);
        hand.players[Wind::South.index()].hand = "1z1z2z5z6z7z1p4p7p2s5s3s3s".parse().unwrap();
        assert_eq!(noten(&hand, "3s"), DealerNoten::Unsafe);
        hand.act(Action::Discard(tile("3s"))).unwrap();
        hand.resolve_calls(&[(Wind::South, Call::Pon)])
            .expect("South may claim the 3s");
        hand.act(Action::Discard(tile("1p"))).unwrap();
        while !(matches!(hand.phase, Phase::Act) && hand.turn == Wind::East) {
            assert_eq!(
                hand.all_last_dealer_noten(),
                [DealerNoten::Absent; KINDS],
                "{:?} in {:?}",
                hand.turn,
                hand.phase
            );
            quiet_step(&mut hand);
        }
        assert!(hand.wall.is_empty());
        // 123456m 456p 678s 3s 9p: throwing the 3s or the 9p leaves a single
        // wait on the other, and any other discard leaves the hand not
        // waiting, the 1m it drew among them, which West and North threw
        // but South did not.
        assert_eq!(noten(&hand, "3s"), DealerNoten::Tenpai);
        assert_eq!(noten(&hand, "9p"), DealerNoten::Tenpai);
        assert_eq!(noten(&hand, "1m"), DealerNoten::Unsafe);
        every_discard_agrees(&hand);
    }

    /// Nothing is asked of a non-dealer at the last discard, nor in any hand
    /// but the last.
    #[test]
    fn dealer_noten_asks_only_the_dealer_in_the_last_hand() {
        let mut north = position(
            Wind::South,
            4,
            Wind::North,
            "23456m456p678s33s9p",
            "9p",
            [11_900, 12_000, 36_000, 40_100],
        );
        north.wall = Wall::for_analysis(0, &[tile("2z")], 0).unwrap();
        assert_eq!(north.all_last_dealer_noten(), [DealerNoten::Absent; KINDS]);
        for (round, kyoku) in [(Wind::East, 4), (Wind::South, 3)] {
            let mut hand = last_discard([40_100, 36_000, 12_000, 11_900], &[]);
            hand.round = round;
            hand.kyoku = kyoku;
            assert_eq!(
                hand.all_last_dealer_noten(),
                [DealerNoten::Absent; KINDS],
                "{round:?} {kyoku}"
            );
        }
    }

    /// A dealer who declared riichi may only throw the tile drawn (EMA 2025
    /// section 3.3.10), which keeps the hand waiting: no discard leaves it
    /// not waiting, and the kinds it may not throw are not asked about.
    #[test]
    fn a_riichi_dealer_has_no_discard_that_leaves_it_not_waiting() {
        let mut hand = last_discard([40_100, 36_000, 12_000, 11_900], &[]);
        let east = &mut hand.players[Wind::East.index()];
        east.riichi = Riichi::Declared;
        east.riichi_order = Some(0);
        let codes = hand.all_last_dealer_noten();
        for kind in Tile::all() {
            let want = if kind == tile("9p") {
                DealerNoten::Tenpai
            } else {
                DealerNoten::Absent
            };
            assert_eq!(codes[kind.idx()], want, "{kind}");
        }
    }

    #[test]
    fn asking_the_late_questions_changes_nothing() {
        let hands = [
            probe(0),
            probe(1_000),
            tsumo(12_000),
            ron(12_100),
            in_riichi(ron(14_100), Some(tile("2s"))),
            last_discard([40_100, 36_000, 12_000, 11_900], &[Wind::South]),
            ponds(),
        ];
        for hand in hands {
            let before = hand.clone();
            hand.all_last_wide_dama();
            hand.all_last_dealer_noten();
            for seat in Wind::ALL {
                hand.all_last_decline(seat);
            }
            assert_eq!(hand, before);
        }
    }
}
