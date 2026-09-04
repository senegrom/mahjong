//! Looking ahead by imagining the hands you cannot see.
//!
//! A policy network answers from the position in front of it. That is a
//! judgement, not a calculation: it never plays anything out. Search fixes
//! that, but mahjong will not take the search a perfect-information game
//! takes, because three hands and the whole wall are hidden and the tree
//! branches on tiles nobody can know.
//!
//! What works instead is to imagine. Take everything the player can see,
//! deal the rest at random into the other three hands and the wall in a way
//! that is consistent with it, and play that world out. Do it many times
//! and average. A move that wins in most of the worlds where it could
//! matter is a good move, whatever any single deal does. The literature
//! calls this perfect-information Monte Carlo; a player would call it
//! thinking it through.
//!
//! Which worlds get imagined is the whole question. Dealing the unseen
//! tiles out evenly assumes every arrangement is as likely as any other,
//! and that is wrong for a reason no amount of counting will fix: the
//! opponents chose what to throw. What is still in their hand is what they
//! wanted to keep. A player who has spent the hand throwing circles is less
//! likely to be holding circles than the count of unseen tiles says, and a
//! player who has thrown nothing but honours is telling you something else
//! again. That is a selection, and reading it is a judgement about this
//! table rather than an arithmetic fact, which is why [`Belief`] is
//! something a network supplies rather than something written down here.
//! Even weights are the fallback when nothing better is on hand, not the
//! model.
//!
//! One thing it does not do at all: it assumes the other players know what
//! they know in each imagined world, so it cannot find a move whose value
//! is that it hides information. Nobody bluffs in these rollouts.
//!
//! # What the strong programs do
//!
//! Every mahjong program that searches at all, rather than answering from
//! the position, does two things this module also does: it cuts the tree
//! short and has a learned value stand in for the rest. Playing an
//! imagined world out to the end with a heuristic was tried here first and
//! measured worse than not searching, because the heuristic's idea of a
//! finished hand is not the network's, and more worlds only made it surer
//! of the wrong answer. What differs is how the hidden hands are handled.
//!
//! The nearest published design (Liu et al., "Efficient and Robust
//! Imperfect-Information Games Modeling with Fixed-Size Hidden Information
//! Trees", 2023, the search behind LuckyJ) does not sample worlds. It
//! searches a tree of information sets, a few moves deep, with the
//! network's policy guiding which moves to expand, and at each node it
//! weights the possible hidden hands by a learned belief rather than
//! dealing one out. A thousand simulations to a depth of about eight moves
//! is its budget, and the search's verdict is then fed back to the network
//! as an input rather than only used to pick. Tenhou allows about five
//! seconds a decision, which is the budget any of this has to fit.
//!
//! This module deals worlds and then weighs them, which is how the two
//! ideas meet. Dealing from the belief's per-tile marginals is only a
//! proposal: it puts the right tiles in the right hands on average and
//! knows nothing of shape, so it deals a hand of thirteen strays as
//! readily as one that is a turn from winning. What is still in an
//! opponent's hand was kept on purpose, and what is in the wall is what
//! those choices left, so the hidden tiles are not a random draw from the
//! unseen ones and a learned distribution over whole hands has to say
//! which imagined worlds deserve to count. The caller supplies that: a
//! reader trained to tell real hidden hands from imagined ones gives each
//! world a weight, the likelihood ratio between the two, and [`leaves_from`]
//! takes the worlds with their weights. The plausible worlds carry the
//! decision and the rest are dropped or discounted, which is what the
//! information-set search does with its weighted candidates, on an engine
//! that already plays. The paired comparison honours the weights, and even
//! weights give exactly the unweighted numbers, so a search with no reader
//! is the sampled search it grew out of.

use crate::bot::{Bot, Style};
use crate::encoding::{self, OBSERVATION, OPPONENTS, PLACEMENT_VALUE, POINTS_PER_UNIT, POSITIONS};
use crate::game::{Action, Call, Hand, Phase};
use crate::hand::TileSet;
use crate::rng::Rng;
use crate::table::Table;
use crate::tile::{Tile, COPIES};
use crate::Wind;

/// How hard to think.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Effort {
    /// How many imagined worlds each candidate move is played out in. More
    /// worlds means a steadier answer and proportionally more time.
    pub worlds: usize,
    /// How many moves to consider. The rest of the hand is played by the
    /// rollout policy, so this is only the branching at the root.
    pub candidates: usize,
    /// How far to play each world: the whole hand, or a fixed number of
    /// turns after which the hand is judged by how close it stands.
    ///
    /// Stopping early is a false economy. The heuristic player already
    /// counts, exactly, how many tiles would improve the hand; a truncated
    /// rollout judged on the same thing is a noisy version of a calculation
    /// it can do properly. What a rollout knows that counting does not is
    /// what happens at the end: who deals in, what the hand was worth, who
    /// else was waiting. That is only visible if it is played out.
    pub turns: Option<usize>,
    /// How many standard errors better a move must look before the search
    /// overrides the first one on the list. Two is the usual bar, and the
    /// reason it is here at all is that taking the best of several noisy
    /// numbers keeps the luckiest rather than the best.
    pub margin: f64,
    /// Whether to play the imagined worlds out with the hurried player,
    /// which skips counting how many tiles would improve a hand after each
    /// discard. That count is most of what a rollout costs and buys less
    /// inside one than at the root, but whether it can be dropped without
    /// losing anything is a question for the duel, so this is off until
    /// that has been measured.
    pub hurried: bool,
}

impl Effort {
    /// Enough to be worth doing and quick enough for a browser.
    pub fn quick() -> Effort {
        Effort {
            worlds: 12,
            candidates: 5,
            turns: None,
            margin: 2.0,
            hurried: false,
        }
    }

    /// For an arena, where there is time.
    pub fn thorough() -> Effort {
        Effort {
            worlds: 60,
            candidates: 8,
            turns: None,
            margin: 2.0,
            hurried: false,
        }
    }
}

/// How likely each kind of tile is to be in each opponent's hand.
///
/// Three rows of thirty-four, in the same relative seat order the
/// observation uses: row 0 is the player to the searcher's right in turn
/// order. A row is a weight, not a probability: it is used to draw tiles
/// from what nobody has seen, so only the ratios between its entries
/// matter, and a zero means "not this one".
///
/// A network trained on [`crate::encoding::opponent_hands`] produces these.
/// Without one, [`Belief::even`] gives every kind the same weight, which is
/// the assumption that opponents discard at random.
#[derive(Clone, PartialEq, Debug)]
pub struct Belief {
    /// Row-major, three rows of [`POSITIONS`] weights.
    pub weights: Vec<f32>,
}

impl Belief {
    /// Every kind as likely as any other, for want of anything better.
    pub fn even() -> Belief {
        Belief {
            weights: vec![1.0; OPPONENTS * POSITIONS],
        }
    }

    /// A belief from a network's answer, which must be three rows of
    /// thirty-four. Negative weights are read as zero.
    pub fn from(weights: &[f32]) -> Belief {
        assert_eq!(
            weights.len(),
            OPPONENTS * POSITIONS,
            "a belief is three rows of thirty-four"
        );
        Belief {
            weights: weights.iter().map(|weight| weight.max(0.0)).collect(),
        }
    }

    /// The weight this belief puts on `tile` for the opponent `offset`
    /// seats along in turn order, where 1 is the player to the right.
    fn weight(&self, offset: usize, tile: Tile) -> f32 {
        self.weights[(offset - 1) * POSITIONS + tile.idx()]
    }
}

/// What one candidate move came to.
#[derive(Clone, PartialEq, Debug)]
pub struct Judged {
    /// The move.
    pub action: Action,
    /// The average points the hand moved for the searching player, over
    /// every world it was tried in.
    pub value: f64,
    /// What it came to in each world, in the order the worlds were made, so
    /// two candidates can be compared world by world. A world the move was
    /// not legal in holds `None`.
    pub per_world: Vec<Option<f64>>,
    /// How many worlds it was tried in.
    pub worlds: usize,
    /// How much each world counts, in the order the worlds were made:
    /// even when they were only sampled, from a reader of hidden hands
    /// when they were weighed.
    pub weights: Vec<f64>,
}

/// How much better one move looks than another, and how sure that is.
///
/// Both were played out in the same worlds, so the comparison is made world
/// by world: the spread of those differences is far smaller than the spread
/// of either move's own result, because the luck of the deal is the same on
/// both sides of it and cancels.
///
/// The worlds carry weights, so this is a weighted mean of the differences,
/// and its error is the error of a weighted mean of independent draws with
/// the weights' concentration taken out, Kish's effective sample size, so
/// that even weights give the usual n-1 and a few heavy worlds are not
/// mistaken for many.
fn compare(candidate: &Judged, against: &Judged) -> Option<(f64, f64)> {
    let paired: Vec<(f64, f64)> = candidate
        .per_world
        .iter()
        .zip(&against.per_world)
        .zip(&candidate.weights)
        .filter_map(|((mine, theirs), weight)| Some((mine.as_ref()? - theirs.as_ref()?, *weight)))
        .filter(|(_, weight)| *weight > 0.0)
        .collect();
    if paired.len() < 3 {
        return None;
    }
    let total: f64 = paired.iter().map(|(_, weight)| weight).sum();
    let mean = paired
        .iter()
        .map(|(difference, weight)| difference * weight)
        .sum::<f64>()
        / total;
    let concentration: f64 = paired
        .iter()
        .map(|(_, weight)| (weight / total).powi(2))
        .sum();
    if concentration >= 1.0 {
        return None;
    }
    let spread: f64 = paired
        .iter()
        .map(|(difference, weight)| (weight / total).powi(2) * (difference - mean).powi(2))
        .sum();
    Some((mean, (spread / (1.0 - concentration)).sqrt()))
}

/// Everything a seat can see of the tiles.
///
/// Their own hand and sets, every discard on the table, everybody's called
/// sets, and the dora indicators that have been turned. What is left is
/// what a search is free to imagine.
pub fn seen_by(hand: &Hand, seat: Wind) -> TileSet {
    let mut seen = TileSet::new();
    for tile in hand.players[seat.index()].hand.tiles() {
        seen.add(tile);
    }
    for other in Wind::ALL {
        let player = &hand.players[other.index()];
        for meld in &player.melds {
            for tile in meld.tiles() {
                seen.add(tile);
            }
        }
        for discard in &player.discards {
            // A tile claimed for a set stays in the pond it came from,
            // turned sideways against the set that took it. It is counted
            // with that set, so counting it here as well counts it twice:
            // that made a wait look a tile thinner than it is, and left the
            // search one tile short of a world.
            if !discard.claimed {
                seen.add(discard.tile);
            }
        }
    }
    for indicator in hand.wall.dora_indicators() {
        seen.add(indicator);
    }
    // A tile awaiting a claim needs nothing more: a discard is already in
    // the discarder's pond by the time anybody may claim it, and the tile
    // added to a quad that is being robbed is already in the quad.
    seen
}

/// One way the hidden tiles might actually lie.
///
/// The player's own hand, everybody's called sets, the discards, the scores
/// and the state of the hand are all kept exactly. Only what `seat` cannot
/// see is dealt again: the other three hands, the rest of the wall, and the
/// dead wall under the indicators that have been turned.
pub fn imagine(hand: &Hand, seat: Wind, belief: &Belief, rng: &mut Rng) -> Hand {
    let seen = seen_by(hand, seat);
    let mut pool: Vec<Tile> = Vec::with_capacity(136);
    for tile in Tile::all() {
        for _ in 0..COPIES.saturating_sub(seen.count(tile)) {
            pool.push(tile);
        }
    }
    rng.shuffle(&mut pool);

    let mut world = hand.clone();
    for offset in 1..=OPPONENTS {
        let other = seat.plus(offset);
        let wanted = hand.players[other.index()].hand.len();
        let mut dealt = TileSet::new();
        for _ in 0..wanted {
            let taken = draw_weighted(&mut pool, belief, offset, rng);
            dealt.add(taken);
        }
        world.players[other.index()].hand = dealt;
        // What they are waiting for and whether they are furiten follow
        // from the tiles, so both are worked out again for the new hand.
        world.players[other.index()].refresh_furiten();
    }

    // Whatever nobody was dealt is the rest of the wall. It is already
    // shuffled, and the belief has no say in the order it comes out.
    world.wall = hand.wall.with_hidden(&pool);
    world
}

/// Takes one tile from `pool`, chosen in proportion to what `belief` says
/// this opponent is holding.
///
/// Falls back to an even draw when the belief puts no weight on anything
/// left in the pool, which can happen when a confident network is wrong
/// about a hand and there is nothing else to deal.
fn draw_weighted(pool: &mut Vec<Tile>, belief: &Belief, offset: usize, rng: &mut Rng) -> Tile {
    assert!(!pool.is_empty(), "there is always a tile left to deal");
    let total: f32 = pool.iter().map(|tile| belief.weight(offset, *tile)).sum();
    // A belief that puts no weight on anything left, or that has gone to
    // pieces and produced a not-a-number, falls back to an even draw.
    if total.is_nan() || total <= 0.0 {
        return pool.swap_remove(rng.below(pool.len()));
    }
    // A weighted draw, walking the pool until the running total passes a
    // point chosen along it.
    let mut point = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * total;
    for index in 0..pool.len() {
        point -= belief.weight(offset, pool[index]);
        if point <= 0.0 {
            return pool.swap_remove(index);
        }
    }
    pool.swap_remove(pool.len() - 1)
}

/// Plays one imagined world out and says what the hand moved for `seat`.
///
/// Every player, the searching one included, is played by `style`: the
/// point is to compare moves under the same later play, not to model
/// anybody in particular.
fn play_out(mut world: Hand, seat: Wind, style: Style, turns: Option<usize>, seed: u64) -> f64 {
    let opening = world.players[seat.index()].score;
    let mut bots: Vec<Bot> = (0..4)
        .map(|index| Bot::with_style(seed.wrapping_add(index), style))
        .collect();

    let limit = turns.unwrap_or(usize::MAX);
    let mut played = 0;
    while !matches!(world.phase, Phase::Over) && played < limit {
        match world.phase {
            Phase::Draw => {
                if world.draw().is_err() {
                    break;
                }
            }
            Phase::Act => {
                played += 1;
                let turn = world.turn;
                let action = bots[turn.index()].act(&world);
                if world.act(action).is_err() {
                    break;
                }
            }
            Phase::CallWindow => {
                let answers: Vec<(Wind, Call)> = world
                    .legal_calls()
                    .iter()
                    .map(|(who, calls)| (*who, bots[who.index()].call(&world, *who, calls)))
                    .collect();
                if world.resolve_calls(&answers).is_err() {
                    break;
                }
            }
            Phase::Over => break,
        }
    }

    match &world.outcome {
        // The hand finished, so what it moved is simply what it moved.
        Some(_) => (world.players[seat.index()].score - opening) as f64,
        // It was cut short, so the hand is judged by how it stands: closer
        // to a win is better, and a hand that is waiting is worth more than
        // one that is not. The numbers are small next to a real win, which
        // is the intent: this is a tie-break, not a score.
        None => {
            let player = &world.players[seat.index()];
            let distance = crate::shanten::shanten(&player.hand, player.melds.len());
            let moved = (player.score - opening) as f64;
            moved + 300.0 * (2 - distance.min(6)) as f64
        }
    }
}

/// Judges every candidate move by playing it out in imagined worlds.
///
/// The same worlds are used for every candidate, so the comparison is not
/// clouded by one move happening to be tried in luckier deals than another.
pub fn judge(
    hand: &Hand,
    seat: Wind,
    candidates: &[Action],
    effort: Effort,
    belief: &Belief,
    rng: &mut Rng,
) -> Vec<Judged> {
    assert!(!candidates.is_empty(), "there is always something to do");
    let style = if effort.hurried {
        Style::rollout()
    } else {
        Style::club()
    };

    // One set of worlds, shared by all the candidates. They are imagined in
    // order from the one generator, so a searched game replays exactly
    // whatever the threads below get up to.
    let worlds: Vec<Hand> = (0..effort.worlds)
        .map(|_| imagine(hand, seat, belief, rng))
        .collect();

    // Every candidate in every world is its own job, and there are
    // thousands of them when the search is asked to think properly.
    let jobs: Vec<(usize, usize)> = (0..candidates.len())
        .flat_map(|candidate| (0..worlds.len()).map(move |world| (candidate, world)))
        .collect();
    let results: Vec<Option<f64>> = run_all(&jobs, |&(candidate, world)| {
        let mut trial = worlds[world].clone();
        // A move the engine will not take in this world is not a fault: an
        // imagined hand can make a quad illegal that was legal in the real
        // one. It is simply not counted.
        if trial.act(candidates[candidate]).is_err() {
            return None;
        }
        Some(play_out(
            trial,
            seat,
            style,
            effort.turns,
            world as u64 * 977 + 13,
        ))
    });

    candidates
        .iter()
        .enumerate()
        .map(|(candidate, action)| {
            let per_world: Vec<Option<f64>> = (0..worlds.len())
                .map(|world| results[candidate * worlds.len() + world])
                .collect();
            let counted = per_world.iter().flatten().count();
            let total: f64 = per_world.iter().flatten().sum();
            Judged {
                action: *action,
                value: if counted == 0 {
                    f64::NEG_INFINITY
                } else {
                    total / counted as f64
                },
                per_world,
                worlds: counted,
                weights: vec![1.0; worlds.len()],
            }
        })
        .collect()
}

/// Maps `work` over `jobs`, across every core when the crate was built with
/// the `parallel` feature and one after another otherwise. The order of the
/// results is the order of the jobs either way.
#[cfg(feature = "parallel")]
fn run_all<T: Sync, R: Send>(jobs: &[T], work: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    jobs.par_iter().map(work).collect()
}

/// The same, one job after another, for builds without threads.
#[cfg(not(feature = "parallel"))]
fn run_all<T, R>(jobs: &[T], work: impl Fn(&T) -> R) -> Vec<R> {
    jobs.iter().map(work).collect()
}

/// The move that came out best, having played each of them out.
///
/// `shortlist` is the moves worth considering, **best first**: a caller with
/// a policy network passes what it ranked highest, and a caller without one
/// passes what the heuristic player would do. Only the first
/// [`Effort::candidates`] of them are searched.
///
/// The first is the one to beat. It is an opinion worth something, so it is
/// kept unless another move is better by more than the noise in the
/// comparison, which is [`Effort::margin`] standard errors of the paired
/// difference. Without that rule the search keeps whichever candidate the
/// rollouts happened to smile on, which measured more than a placement
/// worse than not searching at all.
pub fn best(
    hand: &Hand,
    seat: Wind,
    shortlist: &[Action],
    effort: Effort,
    belief: &Belief,
    rng: &mut Rng,
) -> Option<Judged> {
    if shortlist.is_empty() {
        return None;
    }
    let taken = shortlist.len().min(effort.candidates.max(1));
    let judged = judge(hand, seat, &shortlist[..taken], effort, belief, rng);
    pick_by_margin(&judged, effort.margin)
}

/// How a search spent itself, for reading rather than for playing.
///
/// A search that never overrides is a no-op dressed up as a calculation,
/// and one that overrides constantly has a margin that is not doing its
/// job. Neither shows in a placement figure until many games have gone by,
/// so both are counted.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Tally {
    /// Decisions the search was asked about.
    pub asked: usize,
    /// Decisions where it took a move other than the first on the list.
    pub overrode: usize,
}

impl Tally {
    /// The share of decisions it changed.
    pub fn share(&self) -> f64 {
        if self.asked == 0 {
            0.0
        } else {
            self.overrode as f64 / self.asked as f64
        }
    }
}

/// The positions a search wants valued: one for each candidate move in
/// each imagined world, after the move is made and the other players have
/// had their turns.
///
/// This is the AlphaZero shape of the search. The move is applied, the
/// world advances to the searching seat's next decision, and the network's
/// value head says what that position is worth from that seat's point of
/// view. Nothing is played out to the end, so the evaluator is the thing
/// that was trained on real outcomes rather than a heuristic standing in
/// for one.
///
/// The value head was trained to predict what the current hand will move,
/// in units of [`POINTS_PER_UNIT`], plus what the game will be worth by
/// the place it ends in ([`PLACEMENT_VALUE`]). A slot whose hand ends on
/// the way has to be brought onto the same scale, or the search compares
/// a win of a quarter of a unit against positions that still carry a
/// whole placement, and a leader would never take a cheap hand. So when a
/// hand ends, what it moved is banked in `settled`, the world is played on
/// into the next hand to the player's first decision there, and the
/// network values that; when the game ends, the placement it ended in is
/// banked instead and there is nothing left to value.
///
/// Slots are numbered `candidate * worlds + world`.
#[derive(Clone, Debug)]
pub struct Leaves {
    /// How many worlds were imagined.
    pub worlds: usize,
    /// How many candidates were tried in each.
    pub candidates: usize,
    /// One observation per slot, [`OBSERVATION`] numbers each, from the
    /// searching player's viewpoint. Slots that want no valuing hold zeros.
    pub observations: Vec<f32>,
    /// What is already known of each slot's worth, in the value head's
    /// units: what the hands that ended on the way moved for the searching
    /// player, and the placement if the game ended.
    pub settled: Vec<f64>,
    /// Whether each slot's observation is a position the network should
    /// value, to be added to `settled`.
    pub wanted: Vec<bool>,
    /// Whether each slot counts for its candidate at all. It does not when
    /// the move could not be made in that world, which an imagined hand
    /// can do to a quad, or when the world could not be played on.
    pub counted: Vec<bool>,
    /// How much each world counts, one per world: what the reader of
    /// hidden hands made of it, or one each when nobody read them.
    pub weights: Vec<f64>,
}

/// Where an imagined world got to after a candidate move.
enum Leaf {
    /// The searching player has a decision to make, seen from `seat`,
    /// which is not the seat the search began in if a hand ended on the
    /// way.
    Position { seat: Wind, settled: f64 },
    /// The game ended, and this is what the player's stake came to.
    Settled(f64),
    /// The world could not be played on. The engine refusing a move or a
    /// hand that will not end would do it, and neither should happen.
    Broken,
}

/// What running the other players did to one hand.
enum Advance {
    /// The searching seat has a decision to make.
    Decision,
    /// The hand ended, and this is what it moved for the searching seat
    /// since it was dealt, in the value head's units. Counting from the
    /// deal rather than from where the search began keeps it on the
    /// value head's scale, whose hand term is measured the same way, even
    /// when a riichi bet was paid on the way.
    HandOver(f64),
    /// The engine refused something, or the hand would not end.
    Broken,
}

/// Runs the other players round to `seat`'s next decision, or to the end
/// of the hand.
fn advance_to_decision(world: &mut Hand, seat: Wind, style: Style, seed: u64) -> Advance {
    let mut bots: Vec<Bot> = (0..4)
        .map(|index| Bot::with_style(seed.wrapping_add(index), style))
        .collect();
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 600 {
            return Advance::Broken;
        }
        match world.phase {
            Phase::Over => break,
            Phase::Act if world.turn == seat => return Advance::Decision,
            Phase::Draw => {
                if world.draw().is_err() {
                    return Advance::Broken;
                }
            }
            Phase::Act => {
                let turn = world.turn;
                let action = bots[turn.index()].act(world);
                if world.act(action).is_err() {
                    return Advance::Broken;
                }
            }
            Phase::CallWindow => {
                // The searching seat's own calls are answered by its bot
                // too: a claim is a small decision and the search is about
                // the discard that was just made.
                let answers: Vec<(Wind, Call)> = world
                    .legal_calls()
                    .iter()
                    .map(|(who, calls)| (*who, bots[who.index()].call(world, *who, calls)))
                    .collect();
                if world.resolve_calls(&answers).is_err() {
                    return Advance::Broken;
                }
            }
        }
    }
    let moved = world.players[seat.index()].score - world.opening[seat.index()];
    Advance::HandOver(moved as f64 / POINTS_PER_UNIT as f64)
}

/// The table a hand is being played at, as far as the hand knows it. Who
/// sits where does not matter to a search, so the seats stand in for the
/// players, and the first dealer is placed so that the hand's number comes
/// out right and the round ends when it should.
fn table_of(hand: &Hand) -> Table {
    let mut table = Table::new();
    for seat in Wind::ALL {
        table.scores[seat.index()] = hand.players[seat.index()].score;
    }
    let number = (hand.kyoku as usize).clamp(1, 4);
    table.first_dealer = (5 - number) % 4;
    table.round = hand.round;
    table.counters = hand.counters;
    table.riichi_sticks = hand.riichi_sticks;
    debug_assert_eq!(table.kyoku() as usize, number);
    table
}

/// What finishing the game is worth to `player`, by the place the final
/// scores put them in. Ties go to the lower seat, as they do when the
/// training target is worked out.
fn placement_value(table: &Table, player: usize) -> f64 {
    let finals = table.final_scores();
    let mine = finals[player];
    let place = finals
        .iter()
        .enumerate()
        .filter(|(other, score)| **score > mine || (**score == mine && *other < player))
        .count();
    PLACEMENT_VALUE[place] as f64
}

/// Plays an imagined world on from just after a candidate move until the
/// searching player has a decision to make, through the end of a hand and
/// into the next if need be, or until the game ends.
fn play_to_leaf(world: &mut Hand, seat: Wind, style: Style, seed: u64) -> Leaf {
    let mut settled = 0.0;
    let mut seat = seat;
    // A game rarely runs past a dozen hands, and the loop is here for the
    // hand that ends before the player's first turn in it, which happens
    // once in a very long while.
    for extra in 0..16u64 {
        let seed = seed.wrapping_add(extra * 4);
        match advance_to_decision(world, seat, style, seed) {
            Advance::Decision => return Leaf::Position { seat, settled },
            Advance::Broken => return Leaf::Broken,
            Advance::HandOver(moved) => {
                settled += moved;
                // The seats of the hand that just ended stand in for the
                // players, so the player is this seat's number.
                let player = seat.index();
                let mut table = table_of(world);
                table.finish(world);
                if table.finished {
                    return Leaf::Settled(settled + placement_value(&table, player));
                }
                let mut rng = Rng::from_seed(seed ^ 0x9e37_79b9);
                *world = table.deal(&mut rng);
                seat = table.seat_of(player);
            }
        }
    }
    Leaf::Broken
}

/// Imagines `count` worlds from the belief: the proposal a reader of
/// hidden hands then weighs.
pub fn imagine_worlds(
    hand: &Hand,
    seat: Wind,
    belief: &Belief,
    rng: &mut Rng,
    count: usize,
) -> Vec<Hand> {
    (0..count)
        .map(|_| imagine(hand, seat, belief, rng))
        .collect()
}

/// Imagines the worlds evenly weighted, makes each candidate move in each,
/// and returns the positions that result for the value head to judge. The
/// sampled search; [`leaves_from`] is the weighed one.
pub fn leaves(
    hand: &Hand,
    seat: Wind,
    candidates: &[Action],
    effort: Effort,
    belief: &Belief,
    rng: &mut Rng,
) -> Leaves {
    let worlds = imagine_worlds(hand, seat, belief, rng, effort.worlds);
    let weights = vec![1.0; worlds.len()];
    leaves_from(seat, candidates, &worlds, &weights, effort)
}

/// Makes each candidate move in each of the given worlds, which the caller
/// has imagined and weighed, and returns the positions that result for the
/// value head to judge. `weights` says how much each world counts.
pub fn leaves_from(
    seat: Wind,
    candidates: &[Action],
    worlds: &[Hand],
    weights: &[f64],
    effort: Effort,
) -> Leaves {
    assert!(!candidates.is_empty(), "there is always something to do");
    assert_eq!(worlds.len(), weights.len(), "one weight per world");
    let style = if effort.hurried {
        Style::rollout()
    } else {
        Style::club()
    };

    let jobs: Vec<(usize, usize)> = (0..candidates.len())
        .flat_map(|candidate| (0..worlds.len()).map(move |world| (candidate, world)))
        .collect();
    // Each job hands back the slot's observation if it has one to value,
    // what is already settled about it, and whether it counts at all.
    let results: Vec<(Vec<f32>, f64, bool)> = run_all(&jobs, |&(candidate, world)| {
        let mut trial = worlds[world].clone();
        if trial.act(candidates[candidate]).is_err() {
            return (Vec::new(), 0.0, false);
        }
        match play_to_leaf(&mut trial, seat, style, world as u64 * 977 + 13) {
            Leaf::Position {
                seat: viewpoint,
                settled,
            } => {
                let mut out = vec![0.0; OBSERVATION];
                encoding::observe(&trial, viewpoint, &mut out);
                (out, settled, true)
            }
            Leaf::Settled(worth) => (Vec::new(), worth, true),
            Leaf::Broken => (Vec::new(), 0.0, false),
        }
    });

    let slots = jobs.len();
    let mut observations = vec![0.0f32; slots * OBSERVATION];
    let mut settled = vec![0.0; slots];
    let mut wanted = vec![false; slots];
    let mut counted = vec![false; slots];
    for (slot, (out, worth, counts)) in results.into_iter().enumerate() {
        if !out.is_empty() {
            observations[slot * OBSERVATION..(slot + 1) * OBSERVATION].copy_from_slice(&out);
            wanted[slot] = true;
        }
        settled[slot] = worth;
        counted[slot] = counts;
    }
    Leaves {
        worlds: worlds.len(),
        candidates: candidates.len(),
        observations,
        settled,
        wanted,
        counted,
        weights: weights.to_vec(),
    }
}

/// Decides from the values the network gave the leaves.
///
/// `valued` holds one number per slot in the value head's units. A slot is
/// worth what was settled on the way plus, where it wants one, the
/// network's value of the position it reached; slots that do not count
/// are left out of their candidate's average. The first candidate is the
/// one to beat, and another is taken only when it wins by `margin`
/// standard errors of the world-by-world difference, for the same reason
/// as in [`best`].
pub fn decide(
    candidates: &[Action],
    leaves: &Leaves,
    valued: &[f64],
    margin: f64,
) -> Option<Judged> {
    assert_eq!(valued.len(), leaves.counted.len(), "one value per slot");
    let judged: Vec<Judged> = (0..leaves.candidates)
        .map(|candidate| {
            let per_world: Vec<Option<f64>> = (0..leaves.worlds)
                .map(|world| {
                    let slot = candidate * leaves.worlds + world;
                    if !leaves.counted[slot] {
                        None
                    } else if leaves.wanted[slot] {
                        Some(leaves.settled[slot] + valued[slot])
                    } else {
                        Some(leaves.settled[slot])
                    }
                })
                .collect();
            let counted = per_world.iter().flatten().count();
            let (total, weight) = per_world
                .iter()
                .zip(&leaves.weights)
                .filter_map(|(value, weight)| Some((value.as_ref()? * weight, *weight)))
                .fold((0.0, 0.0), |(sum, mass), (value, weight)| {
                    (sum + value, mass + weight)
                });
            Judged {
                action: candidates[candidate],
                value: if counted == 0 || weight <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    total / weight
                },
                per_world,
                worlds: counted,
                weights: leaves.weights.clone(),
            }
        })
        .collect();
    pick_by_margin(&judged, margin)
}

/// The first playable candidate, unless another beats it by the margin.
fn pick_by_margin(judged: &[Judged], margin: f64) -> Option<Judged> {
    let incumbent = judged.iter().find(|entry| entry.worlds > 0)?;
    let mut best = incumbent;
    let mut best_edge = 0.0;
    for entry in judged {
        if entry.worlds == 0 || std::ptr::eq(entry, incumbent) {
            continue;
        }
        if let Some((edge, error)) = compare(entry, incumbent) {
            if edge > margin * error && edge > best_edge {
                best = entry;
                best_edge = edge;
            }
        }
    }
    Some(best.clone())
}

/// A player that thinks before it moves.
///
/// It takes the heuristic player's shortlist and plays each of those moves
/// out rather than trusting the ordering, which is what separates it from
/// the bot it is built on.
#[derive(Clone, Debug)]
pub struct Searcher {
    /// How hard it thinks.
    pub effort: Effort,
    /// How often thinking changed its mind.
    pub tally: Tally,
    /// What it takes the opponents to be holding. Even weights until a
    /// network says otherwise.
    pub belief: Belief,
    bot: Bot,
    rng: Rng,
}

impl Searcher {
    /// A searcher with a given seed and effort.
    pub fn new(seed: u64, effort: Effort) -> Searcher {
        Searcher {
            effort,
            tally: Tally::default(),
            belief: Belief::even(),
            bot: Bot::with_style(seed, Style::club()),
            rng: Rng::from_seed(seed ^ 0x5ea5_c400),
        }
    }

    /// Chooses a move, having played the candidates out.
    pub fn act(&mut self, hand: &Hand) -> Action {
        let actions = hand.legal_actions();
        assert!(!actions.is_empty(), "a player always has something to do");

        // A win is a win; there is nothing to search.
        if let Some(action) = actions
            .iter()
            .find(|action| matches!(action, Action::Tsumo))
        {
            return *action;
        }
        if actions.len() == 1 {
            return actions[0];
        }

        let seat = hand.turn;
        let shortlist = self.shortlist(hand, &actions);
        self.tally.asked += 1;
        match best(
            hand,
            seat,
            &shortlist,
            self.effort,
            &self.belief,
            &mut self.rng,
        ) {
            Some(judged) => {
                if Some(judged.action) != shortlist.first().copied() {
                    self.tally.overrode += 1;
                }
                judged.action
            }
            None => self.bot.act(hand),
        }
    }

    /// Answers a claim. Calls are searched the same way, by playing out the
    /// hand that taking and not taking the tile would leave.
    pub fn call(&mut self, hand: &Hand, seat: Wind, offered: &[Call]) -> Call {
        if offered.contains(&Call::Ron) {
            return Call::Ron;
        }
        self.bot.call(hand, seat, offered)
    }

    /// The moves worth playing out, best first as the heuristic player
    /// ranks them, since searching all fourteen discards costs fourteen
    /// times as much for very little.
    fn shortlist(&mut self, hand: &Hand, actions: &[Action]) -> Vec<Action> {
        let preferred = self.bot.act(hand);
        let mut shortlist = vec![preferred];
        for action in actions {
            if *action != preferred {
                shortlist.push(*action);
            }
        }
        shortlist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An imagined world has to be a world: every tile accounted for, none
    /// of them five times over, and everything the player can see left
    /// exactly as it was.
    #[test]
    fn an_imagined_world_is_consistent_with_what_was_seen() {
        let table = Table::new();
        let mut rng = Rng::from_seed(99);
        let hand = table.deal(&mut rng);
        let seat = Wind::East;

        for round in 0..40 {
            let mut rng = Rng::from_seed(1000 + round);
            let world = imagine(&hand, seat, &Belief::even(), &mut rng);

            // What East holds is untouched.
            assert_eq!(
                world.players[seat.index()].hand.counts(),
                hand.players[seat.index()].hand.counts(),
                "the searching player's own hand is not imagined"
            );
            // Everybody still holds as many tiles as they did.
            for other in Wind::ALL {
                assert_eq!(
                    world.players[other.index()].hand.len(),
                    hand.players[other.index()].hand.len(),
                    "{other:?} was dealt a different number of tiles"
                );
            }
            // The indicators that were face up are still those tiles.
            assert_eq!(
                world.wall.dora_indicators(),
                hand.wall.dora_indicators(),
                "an indicator that was turned cannot change"
            );

            // Four of every kind, no more, across hands, sets, discards and
            // the whole wall.
            let mut counted = TileSet::new();
            for other in Wind::ALL {
                let player = &world.players[other.index()];
                for tile in player.hand.tiles() {
                    counted.add(tile);
                }
                for meld in &player.melds {
                    for tile in meld.tiles() {
                        counted.add(tile);
                    }
                }
                for discard in &player.discards {
                    counted.add(discard.tile);
                }
            }
            for tile in world.wall.tiles() {
                counted.add(*tile);
            }
            // The wall array still holds the tiles already drawn, which are
            // now also in a hand, so each of those is counted twice.
            for tile in Tile::all() {
                assert!(
                    counted.count(tile) >= COPIES,
                    "{tile} went missing from the imagined world"
                );
            }
        }
    }

    /// A belief has to change what gets imagined, or it is decoration. A
    /// network that has read an opponent's discards and concluded they are
    /// holding circles must produce worlds where they are holding circles.
    #[test]
    fn what_the_belief_says_is_what_gets_dealt() {
        let table = Table::new();
        let mut rng = Rng::from_seed(515);
        let hand = table.deal(&mut rng);
        let seat = Wind::East;

        // Everything on the player to the right, nothing on the other two.
        let mut weights = vec![0.0f32; OPPONENTS * POSITIONS];
        for tile in Tile::all() {
            if matches!(tile.suit(), crate::tile::Suit::Circles) {
                weights[tile.idx()] = 1.0;
            }
        }
        // The other two rows are even, so they are unaffected.
        for offset in 1..OPPONENTS {
            for tile in Tile::all() {
                weights[offset * POSITIONS + tile.idx()] = 1.0;
            }
        }
        let belief = Belief::from(&weights);

        let mut circles = 0;
        let mut held = 0;
        let mut even_circles = 0;
        let mut even_held = 0;
        for round in 0..30 {
            let mut rng = Rng::from_seed(round);
            let world = imagine(&hand, seat, &belief, &mut rng);
            let right = &world.players[seat.plus(1).index()].hand;
            circles += right
                .tiles()
                .filter(|tile| matches!(tile.suit(), crate::tile::Suit::Circles))
                .count();
            held += right.len();

            let mut rng = Rng::from_seed(round);
            let plain = imagine(&hand, seat, &Belief::even(), &mut rng);
            let right = &plain.players[seat.plus(1).index()].hand;
            even_circles += right
                .tiles()
                .filter(|tile| matches!(tile.suit(), crate::tile::Suit::Circles))
                .count();
            even_held += right.len();
        }

        assert_eq!(
            circles, held,
            "told the player holds only circles, every tile dealt them is a circle"
        );
        assert!(
            even_circles * 3 < even_held,
            "and dealt evenly they hold about a quarter circles, not {even_circles} of {even_held}"
        );
    }

    /// The leaves come back one per candidate per world, each either a
    /// position for the network or a hand that ended with its own answer,
    /// and deciding from made-up values picks what the values say.
    #[test]
    fn leaves_are_one_per_candidate_per_world_and_decide_follows_the_values() {
        let table = Table::new();
        let mut rng = Rng::from_seed(2026);
        let hand = table.deal(&mut rng);
        let seat = hand.turn;
        let candidates: Vec<Action> = hand.legal_actions().into_iter().take(3).collect();
        let effort = Effort {
            worlds: 6,
            candidates: 3,
            turns: None,
            margin: 2.0,
            hurried: true,
        };

        let mut rng = Rng::from_seed(9);
        let got = leaves(&hand, seat, &candidates, effort, &Belief::even(), &mut rng);
        assert_eq!(got.worlds, 6);
        assert_eq!(got.candidates, 3);
        assert_eq!(got.counted.len(), 18);
        assert_eq!(got.observations.len(), 18 * OBSERVATION);
        // Every slot that wants a value carries a real observation: the
        // player's own hand is in it, so the planes are not all zero.
        for slot in (0..18).filter(|slot| got.wanted[*slot]) {
            let planes = &got.observations[slot * OBSERVATION..(slot + 1) * OBSERVATION];
            assert!(
                planes.iter().any(|value| *value != 0.0),
                "slot {slot} was handed back empty"
            );
        }

        // Value the third candidate far above the rest in every world, and
        // it is taken; value them all the same, and the incumbent stands.
        let mut valued = vec![0.0; 18];
        for world in 0..6 {
            valued[2 * 6 + world] = 5.0 + world as f64 * 0.01;
        }
        let chosen = decide(&candidates, &got, &valued, 2.0).expect("a decision");
        if (12..18).all(|slot| got.wanted[slot]) {
            assert_eq!(
                chosen.action, candidates[2],
                "the clearly better move is taken"
            );
        }
        let flat = vec![1.0; 18];
        let kept = decide(&candidates, &got, &flat, 2.0).expect("a decision");
        assert_eq!(kept.action, candidates[0], "nothing beats the incumbent");
    }

    /// Even weights are the unweighted comparison to the last digit, and a
    /// heavier world pulls the mean its way.
    #[test]
    fn even_weights_are_the_unweighted_comparison() {
        let judged = |per_world: Vec<Option<f64>>, weights: Vec<f64>| Judged {
            action: Action::Discard(Tile::new(0)),
            value: 0.0,
            worlds: per_world.len(),
            per_world,
            weights,
        };
        let mine = judged(
            vec![Some(1.0), Some(2.0), Some(4.0), Some(3.0)],
            vec![1.0; 4],
        );
        let theirs = judged(vec![Some(0.0); 4], vec![1.0; 4]);
        let (mean, error) = compare(&mine, &theirs).expect("four worlds compare");
        // Differences 1, 2, 4, 3: mean 2.5, sample variance 5/3, and the
        // error of the mean is the root of that over four.
        assert!((mean - 2.5).abs() < 1e-12);
        assert!((error - (5.0f64 / 3.0 / 4.0).sqrt()).abs() < 1e-12);

        let heavier = judged(
            vec![Some(1.0), Some(2.0), Some(4.0), Some(3.0)],
            vec![1.0, 1.0, 1.0, 2.0],
        );
        let (mean, _) = compare(&heavier, &theirs).expect("still four worlds");
        assert!((mean - (1.0 + 2.0 + 4.0 + 6.0) / 5.0).abs() < 1e-12);

        // A world that counts for nothing is not a world, and two are too
        // few to compare.
        let thin = judged(
            vec![Some(1.0), Some(2.0), Some(4.0), Some(3.0)],
            vec![0.0, 0.0, 1.0, 1.0],
        );
        assert!(compare(&thin, &theirs).is_none());
    }

    /// The leaves carry the weights they were given, the sampled search
    /// weighs every world the same, and the decision is a weighted one:
    /// with all the weight on one world, what that world says goes.
    #[test]
    fn leaves_carry_their_weights_and_decide_by_them() {
        let mut rng = Rng::from_seed(2026);
        let hand = Table::new().deal(&mut rng);
        let seat = hand.turn;
        let candidates: Vec<Action> = hand.legal_actions().into_iter().take(2).collect();
        let effort = Effort {
            worlds: 4,
            candidates: 2,
            turns: None,
            margin: 2.0,
            hurried: true,
        };
        let mut rng = Rng::from_seed(5);
        let worlds = imagine_worlds(&hand, seat, &Belief::even(), &mut rng, 4);
        let given = [0.5, 2.0, 1.0, 0.25];
        let got = leaves_from(seat, &candidates, &worlds, &given, effort);
        assert_eq!(got.weights, given.to_vec());
        assert_eq!(got.worlds, 4);

        let plain = leaves(&hand, seat, &candidates, effort, &Belief::even(), &mut rng);
        assert!(plain.weights.iter().all(|weight| *weight == 1.0));

        // Value the second candidate far above the first in the second
        // world only. Weighed evenly that is one world in four and not
        // enough to override; with the weight on that world it is, but
        // one world cannot be compared, so the answer is the incumbent
        // both ways and the difference shows in the value instead.
        if got.counted.iter().all(|counts| *counts) {
            let mut valued = vec![0.0; 8];
            valued[4 + 1] = 5.0;
            let heavy = leaves_from(seat, &candidates, &worlds, &[0.0, 1.0, 0.0, 0.0], effort);
            let picked = decide(&candidates, &heavy, &valued, 2.0).expect("a decision");
            assert_eq!(
                picked.action, candidates[0],
                "one world is not a comparison"
            );
            let judged = decide(&candidates, &got, &valued, 2.0).expect("a decision");
            assert_eq!(judged.action, candidates[0]);
        }
    }

    /// Plays a hand to its end with the hurried bots.
    fn run_out(world: &mut Hand, seed: u64) {
        let style = Style::rollout();
        let mut bots: Vec<Bot> = (0..4)
            .map(|index| Bot::with_style(seed + index, style))
            .collect();
        let mut guard = 0;
        while !matches!(world.phase, Phase::Over) {
            guard += 1;
            assert!(guard < 600, "a hand ends");
            match world.phase {
                Phase::Draw => {
                    world.draw().expect("a draw");
                }
                Phase::Act => {
                    let turn = world.turn;
                    let action = bots[turn.index()].act(world);
                    world.act(action).expect("a legal move");
                }
                Phase::CallWindow => {
                    let answers: Vec<(Wind, Call)> = world
                        .legal_calls()
                        .iter()
                        .map(|(who, calls)| (*who, bots[who.index()].call(world, *who, calls)))
                        .collect();
                    world.resolve_calls(&answers).expect("calls resolve");
                }
                Phase::Over => {}
            }
        }
    }

    /// A hand that has ended is not a position the network can value. The
    /// search banks what it moved and plays the world into the next hand,
    /// to the same player's first decision there, so that every leaf
    /// carries a placement.
    #[test]
    fn a_finished_hand_is_played_into_the_next_one() {
        let mut rng = Rng::from_seed(11);
        let mut world = Table::new().deal(&mut rng);
        let seat = Wind::South;
        let before = world.players[seat.index()].score;
        run_out(&mut world, 5);
        let after = world.players[seat.index()].score;
        let moved = (after - before) as f64 / POINTS_PER_UNIT as f64;

        // East 1 cannot be the last hand, so the world goes on.
        match play_to_leaf(&mut world, seat, Style::rollout(), 77) {
            Leaf::Position { seat: now, settled } => {
                assert!(
                    (settled - moved).abs() < 1e-9,
                    "what the hand moved is banked"
                );
                assert!(
                    matches!(world.phase, Phase::Act) && world.turn == now,
                    "the player is on turn in the new hand"
                );
                assert!(world.discards_made <= 8, "the new hand has only just begun");
                assert_eq!(
                    world.players[now.index()].score,
                    after,
                    "the player's points came with them"
                );
            }
            Leaf::Settled(_) => panic!("the game cannot end at East 1"),
            Leaf::Broken => panic!("the world could be played on"),
        }
    }

    /// When the hand that ended was the last of the game there is no next
    /// position: the leaf is worth what the hand moved plus the place the
    /// game finished in.
    #[test]
    fn the_last_hand_settles_with_the_placement() {
        let mut settled_once = false;
        let mut went_on_once = false;
        for seed in 0..12u64 {
            let mut table = Table::new();
            table.round = Wind::South;
            table.first_dealer = 1;
            assert_eq!(table.kyoku(), 4, "player 0 is East at South 4");
            let mut rng = Rng::from_seed(100 + seed);
            let mut world = table.deal(&mut rng);
            let seat = Wind::West;
            let before = world.players[seat.index()].score;
            run_out(&mut world, seed);
            let moved =
                (world.players[seat.index()].score - before) as f64 / POINTS_PER_UNIT as f64;
            let mut after = table_of(&world);
            after.finish(&world);

            match play_to_leaf(&mut world.clone(), seat, Style::rollout(), seed) {
                Leaf::Settled(worth) => {
                    assert!(after.finished, "settled only when the game is over");
                    let expected = moved + placement_value(&after, seat.index());
                    assert!((worth - expected).abs() < 1e-9, "the placement is banked");
                    settled_once = true;
                }
                Leaf::Position { .. } => {
                    assert!(
                        !after.finished,
                        "the dealer kept the deal, so the game went on"
                    );
                    went_on_once = true;
                }
                Leaf::Broken => panic!("the world could be played on"),
            }
        }
        assert!(settled_once, "some game ended at South 4");
        assert!(went_on_once, "some dealer kept the deal at South 4");
    }

    /// The placement goes by the final scores, and a tie goes to the lower
    /// seat, as it does when the training target is worked out.
    #[test]
    fn placement_goes_by_final_score_with_ties_to_the_lower_seat() {
        let mut table = Table::new();
        table.scores = [40_000, 30_000, 20_000, 30_000];
        table.finished = true;
        assert_eq!(placement_value(&table, 0), 1.5);
        assert_eq!(placement_value(&table, 1), 0.5);
        assert_eq!(placement_value(&table, 3), -0.5);
        assert_eq!(placement_value(&table, 2), -1.5);
    }

    /// A hand knows enough about its table for the search to play on from
    /// it: the round, the hand's number, the counters, the bets and the
    /// scores all come back.
    #[test]
    fn a_hand_knows_which_table_it_is_at() {
        let mut table = Table::new();
        table.round = Wind::South;
        table.first_dealer = 2;
        table.counters = 2;
        table.riichi_sticks = 1;
        table.scores = [31_000, 29_000, 33_000, 27_000];
        let hand = table.deal(&mut Rng::from_seed(3));
        let seen = table_of(&hand);
        assert_eq!(seen.kyoku(), table.kyoku());
        assert_eq!(seen.round, Wind::South);
        assert_eq!(seen.counters, 2);
        assert_eq!(seen.riichi_sticks, 1);
        assert_eq!(seen.scores, table.scores);
    }

    /// A world that was imagined can be played to the end, which is the
    /// only thing a rollout needs of it.
    #[test]
    fn an_imagined_world_can_be_played_out() {
        let table = Table::new();
        let mut rng = Rng::from_seed(7);
        let hand = table.deal(&mut rng);

        let mut finished = 0;
        for round in 0..20 {
            let mut rng = Rng::from_seed(round);
            let world = imagine(&hand, Wind::East, &Belief::even(), &mut rng);
            let moved = play_out(world, Wind::East, Style::club(), None, round);
            assert!(moved.is_finite(), "a played-out world has a result");
            finished += 1;
        }
        assert_eq!(finished, 20);
    }

    /// Searching returns one of the moves it was offered, and says how many
    /// worlds it managed to try it in.
    #[test]
    fn the_search_picks_something_it_was_offered() {
        let table = Table::new();
        let mut rng = Rng::from_seed(4242);
        let hand = table.deal(&mut rng);
        let actions = hand.legal_actions();

        let mut rng = Rng::from_seed(11);
        let effort = Effort {
            worlds: 4,
            candidates: 3,
            turns: Some(20),
            margin: 2.0,
            hurried: false,
        };
        let picked = best(
            &hand,
            hand.turn,
            &actions,
            effort,
            &Belief::even(),
            &mut rng,
        )
        .expect("something came back");
        assert!(actions.contains(&picked.action), "{:?}", picked.action);
        assert!(picked.worlds > 0);
        assert!(picked.value.is_finite());
    }

    /// The same position searched twice with the same seed gives the same
    /// answer, so a game that uses it still replays.
    #[test]
    fn the_search_is_reproducible() {
        let table = Table::new();
        let mut rng = Rng::from_seed(20260903);
        let hand = table.deal(&mut rng);

        let mut first = Searcher::new(
            5,
            Effort {
                worlds: 3,
                candidates: 3,
                turns: Some(12),
                margin: 2.0,
                hurried: false,
            },
        );
        let mut second = Searcher::new(
            5,
            Effort {
                worlds: 3,
                candidates: 3,
                turns: Some(12),
                margin: 2.0,
                hurried: false,
            },
        );
        assert_eq!(first.act(&hand), second.act(&hand));
    }
}
