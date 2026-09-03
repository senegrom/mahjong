//! The turn state machine of a single hand.
//!
//! Follows EMA 2025 chapter 3. A hand runs as draw, act, and a window in
//! which the other players may claim the discard, until somebody declares a
//! win or the wall runs out. Every action a player may take is offered by
//! [`Hand::legal_actions`] and [`Hand::legal_calls`], so nothing has to be
//! trusted: an action that was not offered is refused.
//!
//! Timing at a physical table is a referee matter (section 3.3.1). Software
//! cannot reproduce who spoke first, and does not need to: every player is
//! given the same window on a discard and the claims are resolved by the
//! rulebook's priority, which is the faithful reading of a table where all
//! calls are simultaneous.

use crate::hand::{ClaimedFrom, Meld, MeldKind, TileSet};
use crate::rng::Rng;
use crate::score::{self, Riichi, Score, Situation, WinBy};
use crate::shanten;
use crate::tile::Tile;
use crate::wall::Wall;
use crate::Wind;

/// A tile in a player's discard row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Discard {
    /// The tile discarded.
    pub tile: Tile,
    /// Where this discard falls in the hand, counting every player's
    /// discards together. This is what makes "was it discarded after that
    /// riichi" answerable.
    pub order: u32,
    /// Whether it was the tile just drawn, discarded unchanged.
    pub drawn: bool,
    /// Whether it was turned sideways to declare riichi.
    pub riichi: bool,
    /// Whether another player claimed it. A claimed discard still counts as
    /// this player's discard for furiten (EMA section 3.3.9).
    pub claimed: bool,
}

/// One player's state within a hand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Player {
    /// The seat wind.
    pub seat: Wind,
    /// Concealed tiles, the drawn tile included while it is the player's turn.
    pub hand: TileSet,
    /// Called sets, and any concealed quads.
    pub melds: Vec<Meld>,
    /// The discard row, in order.
    pub discards: Vec<Discard>,
    /// Riichi state.
    pub riichi: Riichi,
    /// Whether the one-shot chance is still alive.
    pub ippatsu: bool,
    /// Points.
    pub score: i32,
    /// Furiten because a wait sits among the player's own discards.
    pub furiten: bool,
    /// Furiten until the next draw or claim, after passing a winning tile.
    pub temporary_furiten: bool,
    /// Where the riichi declaration falls in the hand's order of discards.
    pub riichi_order: Option<u32>,
}

impl Player {
    fn new(seat: Wind, score: i32) -> Player {
        Player {
            seat,
            hand: TileSet::new(),
            melds: Vec::new(),
            discards: Vec::new(),
            riichi: Riichi::None,
            ippatsu: false,
            score,
            furiten: false,
            temporary_furiten: false,
            riichi_order: None,
        }
    }

    /// Whether the hand is concealed, i.e. no call but a concealed quad.
    pub fn is_concealed(&self) -> bool {
        self.melds.iter().all(|meld| !meld.kind.opens_hand())
    }

    /// Whether the player has declared riichi.
    pub fn has_riichi(&self) -> bool {
        !matches!(self.riichi, Riichi::None)
    }

    /// Every tile the player can see of their own: hand plus called sets.
    /// A hand holding all four copies of a kind cannot wait on a fifth
    /// (EMA section 3.3.8).
    pub fn visible_to_self(&self) -> TileSet {
        let mut set = self.hand;
        for meld in &self.melds {
            for tile in meld.tiles() {
                set.add(tile);
            }
        }
        set
    }

    /// The hand's waits, if it is waiting.
    pub fn waits(&self) -> TileSet {
        shanten::waits(&self.hand, self.melds.len(), &self.visible_to_self())
    }

    /// Whether the hand is waiting (EMA section 3.3.8).
    pub fn is_tenpai(&self) -> bool {
        !self.waits().is_empty()
    }

    /// Whether a win by discard is barred (EMA section 3.3.9).
    pub fn is_furiten(&self) -> bool {
        self.furiten || self.temporary_furiten
    }

    /// Recomputes permanent furiten from the discard row.
    ///
    /// A player who has not declared riichi may change their wait to leave
    /// furiten, so the flag is recomputed. A player who has declared cannot
    /// change anything, and furiten they have incurred lasts to the end of
    /// the hand, so theirs is never cleared here (EMA section 3.3.9).
    fn refresh_furiten(&mut self) {
        let waits = self.waits();
        let from_discards = self
            .discards
            .iter()
            .any(|discard| waits.count(discard.tile) > 0);
        self.furiten = if self.has_riichi() {
            self.furiten || from_discards
        } else {
            from_discards
        };
    }
}

/// What the player whose turn it is may do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    /// Discard a tile, ending the turn.
    Discard(Tile),
    /// Declare riichi and discard the tile sideways.
    Riichi(Tile),
    /// Declare a win on the drawn tile.
    Tsumo,
    /// Declare a quad from four concealed tiles.
    ConcealedKan(Tile),
    /// Add the fourth tile to a melded triplet.
    ExtendedKan(Tile),
}

/// What another player may do with a discard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Call {
    /// Claim the discard for a win.
    Ron,
    /// Claim it for a triplet.
    Pon,
    /// Claim it for a quad.
    Kan,
    /// Claim it for a sequence, named by the sequence's lowest tile.
    Chii(Tile),
    /// Decline.
    Pass,
}

impl Call {
    /// Higher wins. A win beats any set call; a triplet or quad beats a
    /// sequence (EMA section 3.3.1).
    const fn priority(self) -> u8 {
        match self {
            Call::Ron => 3,
            Call::Pon | Call::Kan => 2,
            Call::Chii(_) => 1,
            Call::Pass => 0,
        }
    }
}

/// Where a hand is in its cycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// The player to move must draw.
    Draw,
    /// The player to move has drawn and must act.
    Act,
    /// A discard is on the table and the others may claim it.
    CallWindow,
    /// The hand is over.
    Over,
}

/// How a hand ended.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// One or more players declared a win.
    Win {
        /// The winners, in turn order from the discarder, with their scores.
        winners: Vec<(Wind, Score)>,
        /// Who discarded the winning tile, if it was not self-drawn.
        discarder: Option<Wind>,
    },
    /// The wall ran out with no win (EMA section 3.4.2).
    ExhaustiveDraw {
        /// The seats that showed a waiting hand.
        tenpai: Vec<Wind>,
    },
}

/// Why an action was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// The action does not fit the phase the hand is in.
    WrongPhase,
    /// The action was not among those offered.
    NotLegal,
    /// The hand is over.
    Over,
}

/// A single hand in progress.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Hand {
    /// The wall and dead wall.
    pub wall: Wall,
    /// The four players, indexed by seat.
    pub players: [Player; 4],
    /// The dealer's seat, always East within a hand.
    pub round: Wind,
    /// Counters on the table (EMA section 3.4.4).
    pub counters: u32,
    /// Riichi bets carried on the table.
    pub riichi_sticks: u32,
    /// Bets placed during this hand, by seat, so they can go back to their
    /// owners when the rules say they should (EMA section 3.3.10).
    pub bets_this_hand: [u32; 4],
    /// Whose turn it is.
    pub turn: Wind,
    /// Where the hand is in its cycle.
    pub phase: Phase,
    /// The tile just drawn, while the turn player is acting.
    pub drawn: Option<Tile>,
    /// The discard awaiting claims, and who made it.
    pub pending_discard: Option<(Wind, Tile)>,
    /// Whether the first set of turns is still unbroken, which the blessings
    /// and double riichi depend on.
    pub first_turns_unbroken: bool,
    /// Whether the last tile drawn came from the dead wall after a quad.
    pub after_quad: bool,
    /// Whether the most recent quad may be robbed, and its tile.
    pub robbable_quad: Option<Tile>,
    /// Whether that quad was a concealed one, which may only be robbed to
    /// win with Thirteen Orphans (EMA section 3.3.13).
    pub robbing_concealed: bool,
    /// How many discards the hand has seen, which numbers each one.
    pub discards_made: u32,
    /// The tile just claimed for a set, which the claimer may not turn
    /// straight back out (EMA section 3.3.2).
    pub just_claimed: Option<Tile>,
    /// The outcome, once there is one.
    pub outcome: Option<Outcome>,
}

impl Hand {
    /// Deals a new hand. `scores` are the players' points, by seat.
    pub fn deal(
        rng: &mut Rng,
        round: Wind,
        counters: u32,
        riichi_sticks: u32,
        scores: [i32; 4],
    ) -> Hand {
        let mut wall = Wall::shuffled(rng);
        let mut players = [
            Player::new(Wind::East, scores[0]),
            Player::new(Wind::South, scores[1]),
            Player::new(Wind::West, scores[2]),
            Player::new(Wind::North, scores[3]),
        ];
        // Thirteen tiles each, and a fourteenth for the dealer, who then acts
        // without drawing (EMA section 2.8).
        for player in players.iter_mut() {
            for tile in wall.deal(13) {
                player.hand.add(tile);
            }
        }
        let extra = wall.draw().expect("a fresh wall has tiles");
        players[0].hand.add(extra);

        Hand {
            wall,
            players,
            round,
            counters,
            riichi_sticks,
            turn: Wind::East,
            phase: Phase::Act,
            drawn: Some(extra),
            pending_discard: None,
            first_turns_unbroken: true,
            after_quad: false,
            robbable_quad: None,
            robbing_concealed: false,
            discards_made: 0,
            just_claimed: None,
            bets_this_hand: [0; 4],
            outcome: None,
        }
    }

    /// The player whose turn it is.
    pub fn current(&self) -> &Player {
        &self.players[self.turn.index()]
    }

    fn player_mut(&mut self, seat: Wind) -> &mut Player {
        &mut self.players[seat.index()]
    }

    /// The actions the turn player may take.
    pub fn legal_actions(&self) -> Vec<Action> {
        if !matches!(self.phase, Phase::Act) {
            return Vec::new();
        }
        let player = self.current();
        let mut actions = Vec::new();

        // A win on the drawn tile, if the hand is complete and has a yaku.
        // The shape is checked first because it is memoised and cheap, and
        // scoring the hand is neither.
        if let Some(drawn) = self.drawn {
            if shanten::shanten(&player.hand, player.melds.len()) == shanten::COMPLETE
                && self.would_win(self.turn, drawn, WinBy::SelfDraw).is_ok()
            {
                actions.push(Action::Tsumo);
            }
        }

        // After riichi the hand is frozen: the drawn tile is discarded, and
        // only a concealed quad that keeps the waits is allowed
        // (EMA section 3.3.10).
        if player.has_riichi() {
            if let Some(drawn) = self.drawn {
                actions.push(Action::Discard(drawn));
                if self.wall.can_declare_quad() && self.riichi_kan_is_valid(drawn) {
                    actions.push(Action::ConcealedKan(drawn));
                }
            }
            return actions;
        }

        let forbidden = self.forbidden_discards();
        let barred_everything = Tile::all()
            .filter(|tile| player.hand.count(*tile) > 0)
            .all(|tile| forbidden.contains(&tile));
        for tile in Tile::all() {
            if player.hand.count(tile) > 0 && (barred_everything || !forbidden.contains(&tile)) {
                actions.push(Action::Discard(tile));
            }
        }
        debug_assert!(
            !actions.is_empty(),
            "a player always has a tile they may discard"
        );

        // Riichi: concealed, waiting, and at least one tile left in the wall
        // (EMA 2025 section 3.3.10, changed from four in the 2016 edition).
        // The cheap check first: a hand can only be waiting after a discard
        // if the whole hand is at most tenpai already.
        if !player.has_riichi()
            && player.is_concealed()
            && self.wall.remaining() >= 1
            && player.score >= 1000
            && shanten::shanten(&player.hand, player.melds.len()) <= shanten::TENPAI
        {
            let mut probe = player.clone();
            for tile in Tile::all() {
                if player.hand.count(tile) == 0 {
                    continue;
                }
                probe.hand = player.hand;
                probe.hand.remove(tile);
                if probe.is_tenpai() {
                    actions.push(Action::Riichi(tile));
                }
            }
        }

        // Quads, but only in a turn where a tile was drawn: a turn that
        // began with a claim does not allow one (EMA section 3.3.4).
        if self.wall.can_declare_quad() && self.drawn.is_some() {
            for tile in Tile::all() {
                if player.hand.count(tile) == 4 {
                    actions.push(Action::ConcealedKan(tile));
                }
                let has_triplet = player
                    .melds
                    .iter()
                    .any(|meld| matches!(meld.kind, MeldKind::Pon) && meld.tile == tile);
                if has_triplet && player.hand.count(tile) == 1 {
                    actions.push(Action::ExtendedKan(tile));
                }
            }
        }

        actions
    }

    /// Whether the concealed quad a riichi player wants is allowed.
    ///
    /// It must be the fourth copy of a triplet the hand already held, it
    /// must leave the waits exactly as they were, and the three tiles must
    /// read only as a triplet in every hand a waiting tile would complete
    /// (EMA sections 3.3.10 and 6.7.1).
    fn riichi_kan_is_valid(&self, tile: Tile) -> bool {
        let player = self.current();
        let drawn = match self.drawn {
            Some(drawn) if drawn == tile => drawn,
            _ => return false,
        };
        // The hand as it was frozen at the declaration, without the draw.
        let mut frozen = player.hand;
        if !frozen.remove(drawn) {
            return false;
        }
        if frozen.count(tile) != 3 {
            return false;
        }
        let visible = player.visible_to_self();
        let before = shanten::waits(&frozen, player.melds.len(), &visible);
        if before.is_empty() {
            return false;
        }

        // The same hand with the three tiles set aside as a quad.
        let mut without = frozen;
        without.counts_mut()[tile.idx()] = 0;
        let after = shanten::waits(&without, player.melds.len() + 1, &visible);
        if before != after {
            return false;
        }

        // In no completed hand may those tiles be read as part of a sequence.
        for wait in before.tiles() {
            let mut complete = frozen;
            complete.add(wait);
            for reading in crate::agari::readings(&complete, player.melds.len()) {
                let in_a_sequence = reading.blocks.iter().any(|block| {
                    matches!(block, crate::agari::Block::Sequence(_)) && block.contains(tile)
                });
                if in_a_sequence {
                    return false;
                }
            }
        }
        true
    }

    /// Scores the hand as if `seat` won on `tile`, without changing anything.
    pub fn would_win(
        &self,
        seat: Wind,
        tile: Tile,
        win_by: WinBy,
    ) -> Result<Score, score::ScoreError> {
        let player = &self.players[seat.index()];
        let mut concealed = player.hand;
        if matches!(win_by, WinBy::Discard) {
            concealed.add(tile);
        }
        let mut situation = Situation::new(seat, self.round, win_by, tile);
        situation.riichi = player.riichi;
        situation.ippatsu = player.ippatsu;
        situation.counters = self.counters;
        situation.riichi_sticks = self.riichi_sticks;
        situation.dora_indicators = self.wall.dora_indicators();
        if player.has_riichi() {
            situation.ura_indicators = self.wall.ura_indicators();
        }
        situation.after_quad = self.after_quad && matches!(win_by, WinBy::SelfDraw);
        situation.robbing_quad = self.robbable_quad.is_some() && matches!(win_by, WinBy::Discard);
        situation.under_the_sea = self.wall.is_empty() && matches!(win_by, WinBy::SelfDraw);
        situation.under_the_river = self.wall.is_empty() && matches!(win_by, WinBy::Discard);
        // The blessings, all of which need the first set of turns unbroken
        // (EMA section 4.2.6, and Blessing of Man in section 4.2.4).
        let untouched = self.first_turns_unbroken && player.discards.is_empty();
        let nobody_has_discarded = self.players.iter().all(|other| other.discards.is_empty());
        situation.blessing_of_heaven = matches!(seat, Wind::East)
            && matches!(win_by, WinBy::SelfDraw)
            && nobody_has_discarded
            && player.melds.is_empty();
        situation.blessing_of_earth = !matches!(seat, Wind::East)
            && matches!(win_by, WinBy::SelfDraw)
            && untouched
            && player.melds.is_empty();
        situation.blessing_of_man = matches!(win_by, WinBy::Discard) && untouched;
        score::score(&concealed, &player.melds, &situation)
    }

    /// Draws for the player to move.
    pub fn draw(&mut self) -> Result<Tile, Error> {
        if !matches!(self.phase, Phase::Draw) {
            return Err(Error::WrongPhase);
        }
        match self.wall.draw() {
            Some(tile) => {
                self.player_mut(self.turn).hand.add(tile);
                self.player_mut(self.turn).temporary_furiten = false;
                self.drawn = Some(tile);
                self.after_quad = false;
                self.phase = Phase::Act;
                Ok(tile)
            }
            None => {
                self.finish_exhaustive();
                Err(Error::Over)
            }
        }
    }

    /// Takes the turn player's action.
    pub fn act(&mut self, action: Action) -> Result<(), Error> {
        if matches!(self.phase, Phase::Over) {
            return Err(Error::Over);
        }
        if !matches!(self.phase, Phase::Act) {
            return Err(Error::WrongPhase);
        }
        if !self.legal_actions().contains(&action) {
            return Err(Error::NotLegal);
        }
        match action {
            Action::Tsumo => {
                let tile = self.drawn.expect("a self-draw needs a drawn tile");
                let score = self
                    .would_win(self.turn, tile, WinBy::SelfDraw)
                    .expect("the win was offered");
                self.apply_win(vec![(self.turn, score)], None);
            }
            Action::Discard(tile) => self.discard(tile, false),
            Action::Riichi(tile) => {
                let seat = self.turn;
                // Double riichi is a declaration in the player's very first
                // turn, with the first set of turns unbroken (EMA 4.2.2).
                let double =
                    self.first_turns_unbroken && self.players[seat.index()].discards.is_empty();
                let player = self.player_mut(seat);
                player.ippatsu = true;
                self.discard(tile, true);
                let player = self.player_mut(seat);
                player.riichi = if double {
                    Riichi::Double
                } else {
                    Riichi::Declared
                };
                player.score -= 1000;
                self.riichi_sticks += 1;
                self.bets_this_hand[seat.index()] += 1;
            }
            Action::ConcealedKan(tile) => self.declare_quad(tile, MeldKind::ConcealedKan),
            Action::ExtendedKan(tile) => self.declare_quad(tile, MeldKind::ExtendedKan),
        }
        Ok(())
    }

    fn declare_quad(&mut self, tile: Tile, kind: MeldKind) {
        let seat = self.turn;
        {
            let player = self.player_mut(seat);
            let take = if matches!(kind, MeldKind::ConcealedKan) {
                4
            } else {
                1
            };
            for _ in 0..take {
                player.hand.remove(tile);
            }
            if matches!(kind, MeldKind::ExtendedKan) {
                for meld in player.melds.iter_mut() {
                    if matches!(meld.kind, MeldKind::Pon) && meld.tile == tile {
                        meld.kind = MeldKind::ExtendedKan;
                        break;
                    }
                }
            } else {
                player.melds.push(Meld::concealed_kan(tile));
            }
        }
        // A quad breaks the first set of turns. It also ends every one-shot
        // chance, but only once it has actually succeeded: Robbing a Quad
        // may combine with Ippatsu (EMA sections 4.2.1 and 4.2.2), so that
        // is left to finish_quad_draw.
        self.first_turns_unbroken = false;
        // An extended quad may be robbed for a win, and a concealed one only
        // by a hand of Thirteen Orphans (EMA section 3.3.13).
        self.robbable_quad = Some(tile);
        self.robbing_concealed = matches!(kind, MeldKind::ConcealedKan);
        self.pending_discard = Some((seat, tile));
        self.phase = Phase::CallWindow;
        if self.legal_calls().is_empty() {
            self.finish_quad_draw();
        }
    }

    fn finish_quad_draw(&mut self) {
        self.robbable_quad = None;
        self.robbing_concealed = false;
        self.pending_discard = None;
        // The quad stands, so every one-shot chance is gone.
        for player in self.players.iter_mut() {
            player.ippatsu = false;
        }
        let replacement = self.wall.take_replacement();
        match replacement {
            Some(tile) => {
                let seat = self.turn;
                self.player_mut(seat).hand.add(tile);
                self.drawn = Some(tile);
                self.after_quad = true;
                self.phase = Phase::Act;
            }
            None => self.finish_exhaustive(),
        }
    }

    fn discard(&mut self, tile: Tile, riichi: bool) {
        let seat = self.turn;
        let drawn = self.drawn;
        let order = self.discards_made;
        self.discards_made += 1;
        {
            let player = self.player_mut(seat);
            player.hand.remove(tile);
            if riichi {
                player.riichi_order = Some(order);
            }
            player.discards.push(Discard {
                tile,
                order,
                drawn: drawn == Some(tile),
                riichi,
                claimed: false,
            });
            // The one-shot chance runs to the declarer's own next discard
            // (EMA section 4.2.1).
            if !riichi {
                player.ippatsu = false;
            }
            player.refresh_furiten();
        }
        // The first set of turns is over once everyone has discarded once.
        if self
            .players
            .iter()
            .all(|player| !player.discards.is_empty())
        {
            self.first_turns_unbroken = false;
        }
        self.drawn = None;
        self.just_claimed = None;
        self.pending_discard = Some((seat, tile));
        self.phase = Phase::CallWindow;
        if self.legal_calls().is_empty() {
            self.resolve_calls(&[]).expect("no calls to resolve");
        }
    }

    /// The seats whose hand this discard would complete, whether or not
    /// they could declare a win on it. Passing one of these makes a player
    /// temporarily furiten even when the hand has no yaku (EMA 3.3.9).
    fn seats_completed_by(&self, tile: Tile) -> Vec<Wind> {
        let from = match self.pending_discard {
            Some((from, _)) => from,
            None => return Vec::new(),
        };
        Wind::ALL
            .into_iter()
            .filter(|seat| *seat != from)
            .filter(|seat| {
                let player = &self.players[seat.index()];
                let mut completed = player.hand;
                completed.add(tile);
                shanten::shanten(&completed, player.melds.len()) == shanten::COMPLETE
            })
            .collect()
    }

    /// What each other player may do with the discard on the table.
    pub fn legal_calls(&self) -> Vec<(Wind, Vec<Call>)> {
        if !matches!(self.phase, Phase::CallWindow) {
            return Vec::new();
        }
        let (from, tile) = match self.pending_discard {
            Some(pair) => pair,
            None => return Vec::new(),
        };
        let robbing = self.robbable_quad.is_some();
        let last_discard = self.wall.is_empty();
        let mut result = Vec::new();

        for seat in Wind::ALL {
            if seat == from {
                continue;
            }
            let player = &self.players[seat.index()];
            let mut calls = Vec::new();

            // A win, unless furiten (EMA section 3.3.9). The cheap shape
            // check comes first: scoring every discard for every player
            // would otherwise dominate the cost of a hand.
            let mut completed = player.hand;
            completed.add(tile);
            let complete_shape =
                shanten::shanten(&completed, player.melds.len()) == shanten::COMPLETE;
            let shape_allows = !self.robbing_concealed || {
                crate::agari::readings(&completed, player.melds.len())
                    .iter()
                    .any(|reading| matches!(reading.shape, crate::agari::Shape::ThirteenOrphans))
            };
            if complete_shape
                && shape_allows
                && !player.is_furiten()
                && self.would_win(seat, tile, WinBy::Discard).is_ok()
            {
                calls.push(Call::Ron);
            }

            // A quad being robbed offers nothing but the win, and neither
            // does the last discard (EMA sections 3.3.13 and 3.4.1).
            if !robbing && !last_discard && !player.has_riichi() {
                if player.hand.count(tile) >= 2 && self.leaves_a_discard(seat, tile, None) {
                    calls.push(Call::Pon);
                    if player.hand.count(tile) >= 3 && self.wall.can_declare_quad() {
                        calls.push(Call::Kan);
                    }
                }
                // A sequence, only from the player on the left.
                if seat == from.next() && !tile.is_honour() {
                    for low in sequence_starts(tile) {
                        if self.can_form_sequence(seat, low, tile)
                            && self.leaves_a_discard(seat, tile, Some(low))
                        {
                            calls.push(Call::Chii(low));
                        }
                    }
                }
            }

            if !calls.is_empty() {
                calls.push(Call::Pass);
                result.push((seat, calls));
            }
        }
        result
    }

    /// Whether a player who made this call would still have a tile they
    /// are allowed to discard.
    ///
    /// Claiming a tile bars handing it straight back, and for a sequence it
    /// bars the tile from the other side too (EMA section 3.3.2). A player
    /// holding nothing else would be stuck, so that call is not offered.
    fn leaves_a_discard(&self, seat: Wind, claimed: Tile, sequence: Option<Tile>) -> bool {
        let player = &self.players[seat.index()];
        let mut rest = player.hand;
        match sequence {
            None => {
                rest.remove(claimed);
                rest.remove(claimed);
            }
            Some(low) => {
                let second = match low.next_in_suit() {
                    Some(tile) => tile,
                    None => return false,
                };
                let third = match second.next_in_suit() {
                    Some(tile) => tile,
                    None => return false,
                };
                for member in [low, second, third] {
                    if member != claimed {
                        rest.remove(member);
                    }
                }
            }
        }
        let mut barred = vec![claimed];
        if let Some(low) = sequence {
            let rank = claimed.rank();
            let other = if rank == low.rank() {
                Some(rank + 3).filter(|value| *value <= 9)
            } else if rank == low.rank() + 2 {
                rank.checked_sub(3).filter(|value| *value >= 1)
            } else {
                None
            };
            if let Some(rank) = other {
                barred.push(Tile::numbered(claimed.suit(), rank));
            }
        }
        Tile::all()
            .filter(|tile| rest.count(*tile) > 0)
            .any(|tile| !barred.contains(&tile))
    }

    fn can_form_sequence(&self, seat: Wind, low: Tile, claimed: Tile) -> bool {
        let player = &self.players[seat.index()];
        let second = match low.next_in_suit() {
            Some(tile) => tile,
            None => return false,
        };
        let third = match second.next_in_suit() {
            Some(tile) => tile,
            None => return false,
        };
        let needed: Vec<Tile> = [low, second, third]
            .into_iter()
            .filter(|tile| *tile != claimed)
            .collect();
        if needed.len() != 2 {
            return false;
        }
        let mut probe = player.hand;
        needed.iter().all(|tile| probe.remove(*tile))
    }

    /// Applies the players' answers to the discard.
    ///
    /// Every seat offered a call must be represented; a missing answer counts
    /// as a pass. A win beats a set call, and a triplet or quad beats a
    /// sequence (EMA section 3.3.1).
    pub fn resolve_calls(&mut self, answers: &[(Wind, Call)]) -> Result<(), Error> {
        if !matches!(self.phase, Phase::CallWindow) {
            return Err(Error::WrongPhase);
        }
        let offered = self.legal_calls();
        for (seat, call) in answers {
            let allowed = offered
                .iter()
                .find(|(who, _)| who == seat)
                .map(|(_, calls)| calls.contains(call))
                .unwrap_or(false);
            if !allowed {
                return Err(Error::NotLegal);
            }
        }
        let (from, tile) = self.pending_discard.expect("a call window has a discard");

        // Wins first, and several players may win on the same discard
        // (EMA section 3.3.1).
        let winners: Vec<Wind> = answers
            .iter()
            .filter(|(_, call)| matches!(call, Call::Ron))
            .map(|(seat, _)| *seat)
            .collect();
        if !winners.is_empty() {
            // A declaration that is won on never happened: the bet goes
            // back and the hand was never a riichi (EMA section 3.3.10).
            let declaring = self.players[from.index()]
                .discards
                .last()
                .map(|discard| discard.riichi)
                .unwrap_or(false);
            if declaring && self.robbable_quad.is_none() {
                let declarer = self.player_mut(from);
                declarer.score += 1000;
                declarer.riichi = Riichi::None;
                declarer.ippatsu = false;
                self.riichi_sticks = self.riichi_sticks.saturating_sub(1);
                self.bets_this_hand[from.index()] =
                    self.bets_this_hand[from.index()].saturating_sub(1);
                if let Some(discard) = self.player_mut(from).discards.last_mut() {
                    discard.riichi = false;
                }
            }
            let mut scored = Vec::new();
            // In turn order from the discarder, which is how riichi bets and
            // counters are settled.
            for step in 1..4 {
                let seat = from.plus(step);
                if winners.contains(&seat) {
                    let score = self
                        .would_win(seat, tile, WinBy::Discard)
                        .expect("the win was offered");
                    scored.push((seat, score));
                }
            }
            // A robbed quad is not a discard, so there is no discard to
            // mark as claimed (EMA section 3.3.13).
            if self.robbable_quad.is_none() {
                self.mark_claimed(from);
            }
            self.apply_win(scored, Some(from));
            return Ok(());
        }

        // Anyone whose hand this tile completed and who did not take it is
        // furiten until their next draw, even if the hand had no yaku to
        // declare (EMA section 3.3.9).
        for seat in self.seats_completed_by(tile) {
            let player = self.player_mut(seat);
            player.temporary_furiten = true;
            if player.has_riichi() {
                player.furiten = true;
            }
        }

        // A quad that nobody robbed goes ahead.
        if self.robbable_quad.is_some() {
            self.finish_quad_draw();
            return Ok(());
        }

        let best = answers
            .iter()
            .filter(|(_, call)| !matches!(call, Call::Pass))
            .max_by_key(|(_, call)| call.priority());

        match best {
            Some((seat, call)) => {
                let seat = *seat;
                let call = *call;
                self.mark_claimed(from);
                self.take_call(seat, from, tile, call);
            }
            None => {
                if self.wall.is_empty() {
                    self.finish_exhaustive();
                } else {
                    self.pending_discard = None;
                    self.turn = from.next();
                    self.phase = Phase::Draw;
                }
            }
        }
        Ok(())
    }

    fn mark_claimed(&mut self, from: Wind) {
        if let Some(discard) = self.player_mut(from).discards.last_mut() {
            discard.claimed = true;
        }
    }

    fn take_call(&mut self, seat: Wind, from: Wind, tile: Tile, call: Call) {
        let source = match (from.index() + 4 - seat.index()) % 4 {
            3 => ClaimedFrom::Left,
            2 => ClaimedFrom::Across,
            _ => ClaimedFrom::Right,
        };
        // Any call breaks the first set of turns and every one-shot chance.
        self.first_turns_unbroken = false;
        for player in self.players.iter_mut() {
            player.ippatsu = false;
        }
        self.pending_discard = None;
        self.turn = seat;
        {
            let player = self.player_mut(seat);
            player.temporary_furiten = false;
            match call {
                Call::Pon => {
                    player.hand.remove(tile);
                    player.hand.remove(tile);
                    player.melds.push(Meld {
                        kind: MeldKind::Pon,
                        tile,
                        from: source,
                    });
                }
                Call::Kan => {
                    for _ in 0..3 {
                        player.hand.remove(tile);
                    }
                    player.melds.push(Meld {
                        kind: MeldKind::ClaimedKan,
                        tile,
                        from: source,
                    });
                }
                Call::Chii(low) => {
                    let second = low.next_in_suit().expect("a sequence starts below 8");
                    let third = second.next_in_suit().expect("a sequence starts below 8");
                    for member in [low, second, third] {
                        if member != tile {
                            player.hand.remove(member);
                        }
                    }
                    player.melds.push(Meld {
                        kind: MeldKind::Chii,
                        tile: low,
                        from: source,
                    });
                }
                Call::Ron | Call::Pass => unreachable!("handled before"),
            }
        }
        if matches!(call, Call::Kan) {
            self.finish_quad_draw();
        } else {
            self.drawn = None;
            self.just_claimed = Some(tile);
            self.phase = Phase::Act;
        }
    }

    /// Tiles that cannot deal into `seat`.
    ///
    /// A player is furiten on their own discards, so those are always safe
    /// against them (EMA section 3.3.9). Once they have declared riichi they
    /// can no longer change their hand, so everything discarded from that
    /// point on and not claimed for a win is safe as well.
    pub fn safe_against(&self, seat: Wind) -> TileSet {
        let mut safe = TileSet::new();
        let player = &self.players[seat.index()];
        for discard in &player.discards {
            safe.add(discard.tile);
        }
        if let Some(declared) = player.riichi_order {
            for other in &self.players {
                for discard in &other.discards {
                    if discard.order >= declared {
                        safe.add(discard.tile);
                    }
                }
            }
        }
        safe
    }

    /// Tiles the player may not discard after a call, because swap-calling is
    /// forbidden (EMA section 3.3.2).
    pub fn forbidden_discards(&self) -> Vec<Tile> {
        let mut forbidden = Vec::new();
        let claimed = match self.just_claimed {
            Some(tile) if matches!(self.phase, Phase::Act) => tile,
            _ => return forbidden,
        };
        // The claimed tile can never go straight back out.
        forbidden.push(claimed);

        // For a sequence, nor can the tile at the other side of it: claim a
        // 4 for 4-5-6 and the 7 is barred as well, because discarding it
        // would leave the same shape the hand started with.
        let player = self.current();
        if let Some(meld) = player.melds.last() {
            if meld.is_sequence() {
                let low = meld.tile.rank();
                let rank = claimed.rank();
                let other = if rank == low {
                    Some(rank + 3).filter(|value| *value <= 9)
                } else if rank == low + 2 {
                    rank.checked_sub(3).filter(|value| *value >= 1)
                } else {
                    None
                };
                if let Some(rank) = other {
                    forbidden.push(Tile::numbered(claimed.suit(), rank));
                }
            }
        }
        forbidden
    }

    fn apply_win(&mut self, winners: Vec<(Wind, Score)>, discarder: Option<Wind>) {
        for (seat, score) in &winners {
            let payments = score.payments;
            match discarder {
                Some(from) => {
                    self.players[from.index()].score -= payments.from_discarder as i32;
                    self.players[seat.index()].score += payments.from_discarder as i32;
                }
                None => {
                    for other in Wind::ALL {
                        if other == *seat {
                            continue;
                        }
                        let amount = if matches!(other, Wind::East) && !matches!(seat, Wind::East) {
                            payments.from_dealer
                        } else {
                            payments.from_each_other
                        } as i32;
                        self.players[other.index()].score -= amount;
                        self.players[seat.index()].score += amount;
                    }
                }
            }
        }
        // Every winner who declared riichi this hand gets their own bet
        // back; whatever is left, including bets from earlier hands, goes to
        // the winner first in turn order from the discarder (EMA 3.3.10).
        let mut pool = self.riichi_sticks;
        for (seat, _) in &winners {
            let own = self.bets_this_hand[seat.index()].min(pool);
            if own > 0 {
                self.players[seat.index()].score += (own * 1000) as i32;
                pool -= own;
            }
        }
        if let Some((seat, _)) = winners.first() {
            self.players[seat.index()].score += (pool * 1000) as i32;
            pool = 0;
        }
        self.riichi_sticks = pool;
        self.bets_this_hand = [0; 4];
        self.outcome = Some(Outcome::Win { winners, discarder });
        self.phase = Phase::Over;
    }

    fn finish_exhaustive(&mut self) {
        let tenpai: Vec<Wind> = Wind::ALL
            .into_iter()
            .filter(|seat| self.players[seat.index()].is_tenpai())
            .collect();
        // The noten penalty totals 3,000 (EMA section 3.4.2).
        let waiting = tenpai.len();
        if waiting > 0 && waiting < 4 {
            let noten = 4 - waiting;
            let per_winner = (3000 / waiting) as i32;
            let per_loser = (3000 / noten) as i32;
            for seat in Wind::ALL {
                if tenpai.contains(&seat) {
                    self.players[seat.index()].score += per_winner;
                } else {
                    self.players[seat.index()].score -= per_loser;
                }
            }
        }
        self.outcome = Some(Outcome::ExhaustiveDraw { tenpai });
        self.phase = Phase::Over;
    }
}

/// The lowest tiles of the sequences a claimed tile could complete.
fn sequence_starts(tile: Tile) -> Vec<Tile> {
    if tile.is_honour() {
        return Vec::new();
    }
    let rank = tile.rank();
    let suit = tile.suit();
    let mut starts = Vec::new();
    for offset in 0..3u8 {
        if rank < offset + 1 {
            continue;
        }
        let low = rank - offset;
        if (1..=7).contains(&low) {
            starts.push(Tile::numbered(suit, low));
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Hand {
        Hand::deal(&mut Rng::from_seed(20260902), Wind::East, 0, 0, [25000; 4])
    }

    /// EMA 2025 section 2.8: the dealer starts with fourteen tiles and the
    /// others with thirteen, and the dealer acts without drawing.
    #[test]
    fn the_deal_gives_the_dealer_the_extra_tile() {
        let hand = fresh();
        assert_eq!(hand.players[0].hand.len(), 14);
        for seat in 1..4 {
            assert_eq!(hand.players[seat].hand.len(), 13);
        }
        assert_eq!(hand.turn, Wind::East);
        assert!(matches!(hand.phase, Phase::Act));
        assert_eq!(hand.wall.remaining(), 69);
    }

    #[test]
    fn a_discard_passes_the_turn() {
        let mut hand = fresh();
        let tile = match hand
            .legal_actions()
            .into_iter()
            .find_map(|action| match action {
                Action::Discard(tile) => Some(tile),
                _ => None,
            }) {
            Some(tile) => tile,
            None => panic!("the dealer must be able to discard"),
        };
        hand.act(Action::Discard(tile)).unwrap();
        // With nobody calling, the turn moves on and the next player draws.
        if matches!(hand.phase, Phase::CallWindow) {
            hand.resolve_calls(&[]).unwrap();
        }
        assert!(matches!(hand.phase, Phase::Draw) || matches!(hand.phase, Phase::Act));
        assert_eq!(hand.players[0].discards.len(), 1);
    }

    /// EMA 2025 section 3.3.1: a win beats a set call, and a triplet beats a
    /// sequence.
    #[test]
    fn call_priority_is_win_then_triplet_then_sequence() {
        assert!(Call::Ron.priority() > Call::Pon.priority());
        assert!(Call::Pon.priority() > Call::Chii("1m".parse().unwrap()).priority());
        assert!(Call::Kan.priority() == Call::Pon.priority());
    }

    /// EMA 2025 section 3.3.3: a sequence may only be claimed from the player
    /// on the left.
    #[test]
    fn only_the_left_neighbour_may_claim_a_sequence() {
        let mut hand = fresh();
        // Give South a shape that could take 3 characters as a sequence, and
        // make East discard it.
        // Real hands, so the claim leaves something legal to discard.
        hand.players[1].hand = "45m123p99s".parse().unwrap();
        hand.players[2].hand = "45m123p99s".parse().unwrap();
        hand.players[0].hand = "3m".parse().unwrap();
        hand.drawn = None;
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.discard("3m".parse().unwrap(), false);
        let calls = hand.legal_calls();
        let south = calls.iter().find(|(seat, _)| *seat == Wind::South);
        let west = calls.iter().find(|(seat, _)| *seat == Wind::West);
        assert!(
            south.is_some_and(|(_, calls)| calls.iter().any(|call| matches!(call, Call::Chii(_))))
        );
        assert!(
            west.is_none()
                || !west
                    .unwrap()
                    .1
                    .iter()
                    .any(|call| matches!(call, Call::Chii(_)))
        );
    }

    /// EMA 2025 section 3.3.9: a player whose wait sits among their own
    /// discards may not win by discard.
    #[test]
    fn furiten_bars_a_win_by_discard() {
        let mut hand = fresh();
        hand.players[1].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[1].discards.push(Discard {
            tile: "2p".parse().unwrap(),
            order: 0,
            drawn: true,
            riichi: false,
            claimed: false,
        });
        hand.players[1].refresh_furiten();
        assert!(hand.players[1].is_furiten());

        hand.players[0].hand = "2p".parse().unwrap();
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.discard("2p".parse().unwrap(), false);
        let calls = hand.legal_calls();
        let south = calls.iter().find(|(seat, _)| *seat == Wind::South);
        assert!(
            south.is_none() || !south.unwrap().1.contains(&Call::Ron),
            "a furiten player must not be offered the win"
        );
    }

    /// EMA 2025 section 3.3.10, changed in 2025: riichi needs only one tile
    /// left in the wall, where the 2016 edition asked for four.
    #[test]
    fn riichi_is_allowed_with_one_tile_left() {
        let mut hand = fresh();
        hand.players[0].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[0].hand.add("9p".parse().unwrap());
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = Some("9p".parse().unwrap());
        while hand.wall.remaining() > 1 {
            hand.wall.draw();
        }
        assert_eq!(hand.wall.remaining(), 1);
        let actions = hand.legal_actions();
        assert!(actions
            .iter()
            .any(|action| matches!(action, Action::Riichi(_))));
        // With the wall empty there is no declaration to make.
        hand.wall.draw();
        assert!(!hand
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::Riichi(_))));
    }

    /// EMA 2025 section 3.4.1: the last discard may be claimed for a win
    /// only, not for a set.
    #[test]
    fn the_last_discard_is_only_open_to_a_win() {
        let mut hand = fresh();
        while hand.wall.draw().is_some() {}
        hand.players[1].hand = "55m".parse().unwrap();
        hand.players[0].hand = "5m".parse().unwrap();
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.discard("5m".parse().unwrap(), false);
        for (_, calls) in hand.legal_calls() {
            assert!(!calls.contains(&Call::Pon));
            assert!(!calls.contains(&Call::Kan));
        }
    }

    /// EMA 2025 section 3.4.2: the noten penalty totals 3,000 points.
    #[test]
    fn the_noten_penalty_totals_three_thousand() {
        let mut hand = fresh();
        // One waiting hand, three not.
        hand.players[0].hand = "123m456m789m11s34p".parse().unwrap();
        for seat in 1..4 {
            hand.players[seat].hand = "19m19p19s1234z".parse().unwrap();
            hand.players[seat].hand.remove("1z".parse().unwrap());
        }
        let before: i32 = hand.players.iter().map(|player| player.score).sum();
        hand.finish_exhaustive();
        let after: i32 = hand.players.iter().map(|player| player.score).sum();
        assert_eq!(before, after, "the table's points are conserved");
        match &hand.outcome {
            Some(Outcome::ExhaustiveDraw { tenpai }) => {
                assert_eq!(tenpai, &[Wind::East]);
                assert_eq!(hand.players[0].score, 25000 + 3000);
                assert_eq!(hand.players[1].score, 25000 - 1000);
            }
            other => panic!("expected an exhaustive draw, got {other:?}"),
        }
    }

    /// EMA 2025 section 3.3.2, with the rulebook's own examples: a claimed
    /// tile may not be turned straight back out, and neither may the tile
    /// from the other side of a claimed sequence. Claim a 4 for 4-5-6 and
    /// the 7 is barred as well; claim the 6 and the 3 is.
    #[test]
    fn swap_calling_is_barred() {
        let mut hand = fresh();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = None;

        // The 4 was claimed to make 4-5-6.
        hand.players[1]
            .melds
            .push(Meld::chii("4m".parse().unwrap(), ClaimedFrom::Left));
        hand.just_claimed = Some("4m".parse().unwrap());
        let forbidden = hand.forbidden_discards();
        assert!(
            forbidden.contains(&"4m".parse().unwrap()),
            "the claimed tile"
        );
        assert!(forbidden.contains(&"7m".parse().unwrap()), "the other side");
        assert!(!forbidden.contains(&"5m".parse().unwrap()));
        assert!(!forbidden.contains(&"1m".parse().unwrap()));

        // The 6 was claimed for the same sequence.
        hand.just_claimed = Some("6m".parse().unwrap());
        let forbidden = hand.forbidden_discards();
        assert!(
            forbidden.contains(&"6m".parse().unwrap()),
            "the claimed tile"
        );
        assert!(forbidden.contains(&"3m".parse().unwrap()), "the other side");

        // The middle tile has no partner on either side.
        hand.just_claimed = Some("5m".parse().unwrap());
        assert_eq!(hand.forbidden_discards(), vec!["5m".parse().unwrap()]);

        // A claimed triplet bars only the tile itself.
        hand.players[1].melds.clear();
        hand.players[1]
            .melds
            .push(Meld::pon("2p".parse().unwrap(), ClaimedFrom::Across));
        hand.just_claimed = Some("2p".parse().unwrap());
        assert_eq!(hand.forbidden_discards(), vec!["2p".parse().unwrap()]);
    }

    /// However the restriction falls, a player must always be left with
    /// something they may discard. A hand of two tiles both barred by the
    /// old reading of the rule used to leave a player with no move at all.
    #[test]
    fn a_player_always_has_a_tile_they_may_discard() {
        let mut hand = fresh();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.players[1].hand = "99p".parse().unwrap();
        for low in ["1m", "4m", "7m"] {
            hand.players[1]
                .melds
                .push(Meld::chii(low.parse().unwrap(), ClaimedFrom::Left));
        }
        hand.players[1]
            .melds
            .push(Meld::chii("7p".parse().unwrap(), ClaimedFrom::Left));
        hand.just_claimed = Some("7p".parse().unwrap());
        let actions = hand.legal_actions();
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, Action::Discard(_))),
            "the hand holds only 9 circles, which the sequence 7-8-9 must not bar"
        );
    }

    /// EMA 2025 section 3.3.9: a player who declared riichi and passed a
    /// winning discard is furiten "until the end of the hand", so their own
    /// next discard must not clear it.
    #[test]
    fn riichi_furiten_lasts_the_whole_hand() {
        let mut hand = fresh();
        hand.players[1].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[1].riichi = crate::score::Riichi::Declared;
        hand.players[1].furiten = true;
        hand.players[1].temporary_furiten = true;

        // The declarer takes another turn and discards.
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = Some("9p".parse().unwrap());
        hand.players[1].hand.add("9p".parse().unwrap());
        hand.discard("9p".parse().unwrap(), false);
        assert!(
            hand.players[1].furiten,
            "riichi furiten must survive the declarer's own discard"
        );

        // A player who has not declared may leave furiten by changing wait.
        let mut hand = fresh();
        hand.players[2].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[2].furiten = true;
        hand.players[2].refresh_furiten();
        assert!(!hand.players[2].furiten, "an open hand may leave furiten");
    }

    /// EMA 2025 section 3.3.9: passing a tile that completes the hand makes
    /// a player temporarily furiten even when the hand has no yaku.
    #[test]
    fn passing_a_yakuless_completion_is_still_furiten() {
        let mut hand = fresh();
        // West: an open hand with a terminal sequence, so no yaku at all.
        hand.players[2]
            .melds
            .push(Meld::chii("1m".parse().unwrap(), ClaimedFrom::Left));
        hand.players[2].hand = "456m789p22s34s".parse().unwrap();
        hand.players[0].hand = "5s".parse().unwrap();
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.discard("5s".parse().unwrap(), false);

        // Nobody is offered the win, because there is no yaku to declare.
        // Nobody could declare, so the window closed on its own.
        assert!(hand.legal_calls().is_empty());
        assert!(
            hand.players[2].temporary_furiten,
            "the hand was completed and passed, so the player is furiten"
        );
    }

    /// EMA 2025 section 3.3.4: a quad may only be declared in a turn where a
    /// tile was drawn, not in one that began with a claim.
    #[test]
    fn no_quad_in_a_turn_that_began_with_a_claim() {
        let mut hand = fresh();
        hand.players[1].hand = "1111m234p567p9s".parse().unwrap();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;

        // Straight after a claim there is no drawn tile, and no quad.
        hand.drawn = None;
        hand.just_claimed = Some("5z".parse().unwrap());
        hand.players[1]
            .melds
            .push(Meld::pon("5z".parse().unwrap(), ClaimedFrom::Left));
        assert!(!hand
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::ConcealedKan(_))));

        // After an ordinary draw it is offered again.
        hand.just_claimed = None;
        hand.drawn = Some("9s".parse().unwrap());
        assert!(hand
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::ConcealedKan(_))));
    }

    /// EMA 2025 section 3.3.13: a concealed quad may be robbed, but only to
    /// win with Thirteen Orphans.
    #[test]
    fn a_concealed_quad_is_robbable_only_for_thirteen_orphans() {
        let mut hand = fresh();
        // South is waiting on the red dragon for Thirteen Orphans.
        hand.players[1].hand = "119m19p19s123456z".parse().unwrap();
        // West is waiting on it for an ordinary hand, on the pair.
        hand.players[2].hand = "123m456m789m123p7z".parse().unwrap();
        hand.players[0].hand = "7777z".parse().unwrap();
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = Some("7z".parse().unwrap());
        hand.declare_quad("7z".parse().unwrap(), MeldKind::ConcealedKan);

        let offered = hand.legal_calls();
        let south = offered.iter().find(|(seat, _)| *seat == Wind::South);
        let west = offered.iter().find(|(seat, _)| *seat == Wind::West);
        assert!(
            south.is_some_and(|(_, calls)| calls.contains(&Call::Ron)),
            "Thirteen Orphans may rob a concealed quad"
        );
        assert!(
            west.is_none() || !west.unwrap().1.contains(&Call::Ron),
            "an ordinary hand may not"
        );
    }

    /// EMA 2025 section 3.3.10: if the riichi declaration itself is won on,
    /// the declaration is void and the bet goes back.
    #[test]
    fn a_declaration_that_is_won_on_is_void() {
        let mut hand = fresh();
        hand.players[0].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[0].hand.add("2p".parse().unwrap());
        hand.players[1].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[1].riichi = crate::score::Riichi::Declared;
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.drawn = Some("2p".parse().unwrap());

        let before = hand.players[0].score;
        hand.act(Action::Riichi("2p".parse().unwrap())).unwrap();
        assert_eq!(hand.players[0].score, before - 1000, "the bet is placed");

        let offered = hand.legal_calls();
        let south = offered.iter().find(|(seat, _)| *seat == Wind::South);
        assert!(south.is_some_and(|(_, calls)| calls.contains(&Call::Ron)));
        hand.resolve_calls(&[(Wind::South, Call::Ron)]).unwrap();
        assert!(!hand.players[0].has_riichi(), "the declaration never stood");
        assert_eq!(hand.riichi_sticks, 0, "no bet was left on the table");

        // The declarer is out exactly the value of the hand they fed, and
        // not a further thousand for a declaration that never stood.
        let paid = match &hand.outcome {
            Some(Outcome::Win { winners, .. }) => winners[0].1.payments.from_discarder as i32,
            other => panic!("expected a win, got {other:?}"),
        };
        assert_eq!(hand.players[0].score, before - paid);
    }

    #[test]
    fn a_whole_hand_can_be_played_out() {
        let mut hand = fresh();
        let mut guard = 0;
        while !matches!(hand.phase, Phase::Over) {
            guard += 1;
            assert!(guard < 400, "a hand should end well before this");
            match hand.phase {
                Phase::Draw => {
                    let _ = hand.draw();
                }
                Phase::Act => {
                    // Always discard the drawn tile, or the first tile held.
                    let actions = hand.legal_actions();
                    let discard = actions
                        .iter()
                        .find_map(|action| match action {
                            Action::Discard(tile) => Some(*tile),
                            _ => None,
                        })
                        .expect("a player can always discard");
                    hand.act(Action::Discard(discard)).unwrap();
                }
                Phase::CallWindow => {
                    hand.resolve_calls(&[]).unwrap();
                }
                Phase::Over => break,
            }
        }
        assert!(hand.outcome.is_some());
        let total: i32 = hand.players.iter().map(|player| player.score).sum();
        assert_eq!(total, 100_000, "points are conserved across a hand");
    }
}
