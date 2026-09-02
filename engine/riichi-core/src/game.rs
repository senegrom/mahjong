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
    fn refresh_furiten(&mut self) {
        let waits = self.waits();
        self.furiten = self
            .discards
            .iter()
            .any(|discard| waits.count(discard.tile) > 0);
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
    /// Whether the most recent quad was an extended one that may be robbed.
    pub robbable_quad: Option<Tile>,
    /// The outcome, once there is one.
    pub outcome: Option<Outcome>,
}

impl Hand {
    /// Deals a new hand. `scores` are the players' points, by seat.
    pub fn deal(rng: &mut Rng, round: Wind, counters: u32, riichi_sticks: u32, scores: [i32; 4]) -> Hand {
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
        if let Some(drawn) = self.drawn {
            if self.would_win(self.turn, drawn, WinBy::SelfDraw).is_ok() {
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

        for tile in Tile::all() {
            if player.hand.count(tile) > 0 {
                actions.push(Action::Discard(tile));
            }
        }

        // Riichi: concealed, waiting, and at least one tile left in the wall
        // (EMA 2025 section 3.3.10, changed from four in the 2016 edition).
        if !player.has_riichi()
            && player.is_concealed()
            && self.wall.remaining() >= 1
            && player.score >= 1000
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

        // Quads.
        if self.wall.can_declare_quad() {
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

    /// Whether a concealed quad is allowed after riichi: it must not change
    /// the waits, and the three tiles must only read as a triplet
    /// (EMA sections 3.3.10 and 6.7.1).
    fn riichi_kan_is_valid(&self, tile: Tile) -> bool {
        let player = self.current();
        if player.hand.count(tile) != 4 {
            return false;
        }
        let before = player.waits();
        // With the quad set aside, the rest of the hand must wait the same.
        let mut probe = player.clone();
        probe.hand.counts_mut()[tile.idx()] = 0;
        probe.melds.push(Meld::concealed_kan(tile));
        let after = probe.waits();
        if before != after {
            return false;
        }
        // The three tiles must not be readable as part of a sequence in any
        // completed hand, which is what the rulebook's examples turn on.
        for wait in before.tiles() {
            let mut complete = player.hand;
            complete.add(wait);
            for reading in crate::agari::readings(&complete, player.melds.len()) {
                let uses_in_sequence = reading.blocks.iter().any(|block| {
                    matches!(block, crate::agari::Block::Sequence(_)) && block.contains(tile)
                });
                if uses_in_sequence {
                    return false;
                }
            }
        }
        true
    }

    /// Scores the hand as if `seat` won on `tile`, without changing anything.
    pub fn would_win(&self, seat: Wind, tile: Tile, win_by: WinBy) -> Result<Score, score::ScoreError> {
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
                let player = self.player_mut(seat);
                player.riichi = Riichi::None; // set below, once we know which
                player.ippatsu = true;
                self.discard(tile, true);
                let double = self.first_turns_unbroken;
                let player = self.player_mut(seat);
                player.riichi = if double { Riichi::Double } else { Riichi::Declared };
                player.score -= 1000;
                self.riichi_sticks += 1;
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
            let take = if matches!(kind, MeldKind::ConcealedKan) { 4 } else { 1 };
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
        // A quad breaks the first set of turns and every one-shot chance
        // (EMA sections 4.2.1 and 4.2.2).
        self.first_turns_unbroken = false;
        for player in self.players.iter_mut() {
            player.ippatsu = false;
        }
        // An extended quad may be robbed for a win (EMA section 3.3.13).
        if matches!(kind, MeldKind::ExtendedKan) {
            self.robbable_quad = Some(tile);
            self.pending_discard = Some((seat, tile));
            self.phase = Phase::CallWindow;
            if self.legal_calls().is_empty() {
                self.finish_quad_draw();
            }
            return;
        }
        self.finish_quad_draw();
    }

    fn finish_quad_draw(&mut self) {
        self.robbable_quad = None;
        self.pending_discard = None;
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
        {
            let player = self.player_mut(seat);
            player.hand.remove(tile);
            player.discards.push(Discard {
                tile,
                drawn: drawn == Some(tile),
                riichi,
                claimed: false,
            });
            player.refresh_furiten();
        }
        self.drawn = None;
        self.pending_discard = Some((seat, tile));
        self.phase = Phase::CallWindow;
        if self.legal_calls().is_empty() {
            self.resolve_calls(&[]).expect("no calls to resolve");
        }
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

            // A win, unless furiten (EMA section 3.3.9).
            if !player.is_furiten() && self.would_win(seat, tile, WinBy::Discard).is_ok() {
                calls.push(Call::Ron);
            }

            // A quad being robbed offers nothing but the win, and neither
            // does the last discard (EMA sections 3.3.13 and 3.4.1).
            if !robbing && !last_discard && !player.has_riichi() {
                if player.hand.count(tile) >= 2 {
                    calls.push(Call::Pon);
                    if player.hand.count(tile) >= 3 && self.wall.can_declare_quad() {
                        calls.push(Call::Kan);
                    }
                }
                // A sequence, only from the player on the left.
                if seat == from.next() && !tile.is_honour() {
                    for low in sequence_starts(tile) {
                        if self.can_form_sequence(seat, low, tile) {
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
            self.mark_claimed(from);
            self.apply_win(scored, Some(from));
            return Ok(());
        }

        // Anyone who could have won and did not is furiten until their next
        // draw (EMA section 3.3.9).
        for (seat, calls) in &offered {
            if calls.contains(&Call::Ron) {
                let player = self.player_mut(*seat);
                player.temporary_furiten = true;
                if player.has_riichi() {
                    player.furiten = true;
                }
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
                    player.melds.push(Meld { kind: MeldKind::Pon, tile, from: source });
                }
                Call::Kan => {
                    for _ in 0..3 {
                        player.hand.remove(tile);
                    }
                    player.melds.push(Meld { kind: MeldKind::ClaimedKan, tile, from: source });
                }
                Call::Chii(low) => {
                    let second = low.next_in_suit().expect("a sequence starts below 8");
                    let third = second.next_in_suit().expect("a sequence starts below 8");
                    for member in [low, second, third] {
                        if member != tile {
                            player.hand.remove(member);
                        }
                    }
                    player.melds.push(Meld { kind: MeldKind::Chii, tile: low, from: source });
                }
                Call::Ron | Call::Pass => unreachable!("handled before"),
            }
        }
        if matches!(call, Call::Kan) {
            self.finish_quad_draw();
        } else {
            self.drawn = None;
            self.phase = Phase::Act;
        }
    }

    /// Tiles the player may not discard after a call, because swap-calling is
    /// forbidden (EMA section 3.3.2).
    pub fn forbidden_discards(&self) -> Vec<Tile> {
        let mut forbidden = Vec::new();
        if !matches!(self.phase, Phase::Act) || self.drawn.is_some() {
            return forbidden;
        }
        let player = self.current();
        let last = match player.melds.last() {
            Some(meld) => meld,
            None => return forbidden,
        };
        match last.kind {
            MeldKind::Pon => forbidden.push(last.tile),
            MeldKind::Chii => {
                let low = last.tile;
                let second = low.next_in_suit().expect("a sequence starts below 8");
                let third = second.next_in_suit().expect("a sequence starts below 8");
                // The claimed tile itself, and the tile at the other end.
                forbidden.push(low);
                forbidden.push(third);
                let _ = second;
            }
            _ => {}
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
        // Riichi bets go to the first winner in turn order from the discarder,
        // and every winner who declared riichi keeps their own (section 3.3.10).
        if let Some((seat, _)) = winners.first() {
            self.players[seat.index()].score += (self.riichi_sticks * 1000) as i32;
            self.riichi_sticks = 0;
        }
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
        if low >= 1 && low <= 7 {
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
        let tile = match hand.legal_actions().into_iter().find_map(|action| match action {
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
        hand.players[1].hand = "45m".parse().unwrap();
        hand.players[2].hand = "45m".parse().unwrap();
        hand.players[0].hand = "3m".parse().unwrap();
        hand.drawn = None;
        hand.turn = Wind::East;
        hand.phase = Phase::Act;
        hand.discard("3m".parse().unwrap(), false);
        let calls = hand.legal_calls();
        let south = calls.iter().find(|(seat, _)| *seat == Wind::South);
        let west = calls.iter().find(|(seat, _)| *seat == Wind::West);
        assert!(south.is_some_and(|(_, calls)| calls
            .iter()
            .any(|call| matches!(call, Call::Chii(_)))));
        assert!(west.is_none() || !west.unwrap().1.iter().any(|call| matches!(call, Call::Chii(_))));
    }

    /// EMA 2025 section 3.3.9: a player whose wait sits among their own
    /// discards may not win by discard.
    #[test]
    fn furiten_bars_a_win_by_discard() {
        let mut hand = fresh();
        hand.players[1].hand = "123m456m789m11s34p".parse().unwrap();
        hand.players[1].discards.push(Discard {
            tile: "2p".parse().unwrap(),
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
        assert!(actions.iter().any(|action| matches!(action, Action::Riichi(_))));
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

    /// EMA 2025 section 3.3.2: after a call the claimed tile may not be
    /// discarded, nor the tile at the other end of a claimed sequence.
    #[test]
    fn swap_calling_is_barred() {
        let mut hand = fresh();
        hand.turn = Wind::South;
        hand.phase = Phase::Act;
        hand.drawn = None;
        hand.players[1].melds.push(Meld::chii("3m".parse().unwrap(), ClaimedFrom::Left));
        let forbidden = hand.forbidden_discards();
        assert!(forbidden.contains(&"3m".parse().unwrap()));
        assert!(forbidden.contains(&"5m".parse().unwrap()));
        assert!(!forbidden.contains(&"4m".parse().unwrap()));
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
