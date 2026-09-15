//! Public-information constraints for the concealed part of a riichi hand.
//!
//! This is a constructive proposal, not an exact posterior: random weighted
//! component orders choose among legal waiting shapes. The likelihood reader
//! must be calibrated on this proposal. Never seed it with the real hidden hand.
use super::*;
use crate::shanten;

#[derive(Clone)]
struct Plan {
    seat: Wind,
    called: TileSet,
    melds: usize,
    catalogs: Vec<Vec<TileSet>>,
    shapes: Vec<Vec<usize>>,
}

/// Random weighted order with nonzero support even where a marginal is zero.
fn ordered(
    mut groups: Vec<TileSet>,
    belief: &Belief,
    offset: usize,
    rng: &mut Rng,
) -> Vec<TileSet> {
    let mut keyed: Vec<_> = groups
        .drain(..)
        .map(|group| {
            let log_weight = group
                .tiles()
                .map(|t| {
                    let w = belief.weight(offset, t);
                    if w.is_finite() {
                        w.max(1e-6).ln() as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / group.len().max(1) as f64;
            let u = ((rng.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            (-u.ln() / log_weight.exp(), group)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, group)| group).collect()
}

impl Plan {
    fn accepts(&self, held: &TileSet) -> bool {
        let mut visible = *held;
        for tile in self.called.tiles() {
            visible.add(tile);
        }
        // Public ponds are deliberately excluded: waits may be exhausted.
        shanten::is_tenpai(held, self.melds, &visible)
    }

    fn new(hand: &Hand, seat: Wind, observer: Wind, belief: &Belief, rng: &mut Rng) -> Self {
        let player = &hand.players[seat.index()];
        let melds = player.melds.len();
        assert!(
            player.is_concealed() && melds <= 4,
            "riichi requires a concealed hand"
        );
        let called = TileSet::from_tiles(player.melds.iter().flat_map(|m| m.tiles()));
        let singles: Vec<_> = Tile::all().map(|t| TileSet::from_tiles([t])).collect();
        let pairs: Vec<_> = Tile::all().map(|t| TileSet::from_tiles([t, t])).collect();
        let mut sets: Vec<_> = Tile::all()
            .map(|t| TileSet::from_tiles([t, t, t]))
            .collect();
        let mut partial = pairs.clone();
        for suit in 0..3 {
            for low in 0..7 {
                let a = Tile::new(suit * 9 + low);
                let b = Tile::new(suit * 9 + low + 1);
                let c = Tile::new(suit * 9 + low + 2);
                sets.push(TileSet::from_tiles([a, b, c]));
                partial.push(TileSet::from_tiles([a, c]));
            }
            for low in 0..8 {
                partial.push(TileSet::from_tiles([
                    Tile::new(suit * 9 + low),
                    Tile::new(suit * 9 + low + 1),
                ]));
            }
        }
        let orphans: Vec<_> = Tile::all().filter(|t| t.is_terminal_or_honour()).collect();
        let thirteen = TileSet::from_tiles(orphans.iter().copied());
        let mut kokushi = vec![thirteen];
        for &missing in &orphans {
            for &pair in &orphans {
                if pair != missing {
                    let mut group = thirteen;
                    group.remove(missing);
                    group.add(pair);
                    kokushi.push(group);
                }
            }
        }
        let offset = (seat.index() + 4 - observer.index()) % 4;
        let catalogs = vec![singles, pairs, sets, partial, kokushi]
            .into_iter()
            .map(|groups| ordered(groups, belief, offset, rng))
            .collect();
        // Standard tenpai: either the pair lacks one tile, or one set does.
        let mut shapes = vec![vec![0]];
        shapes[0].extend(vec![2; 4 - melds]);
        if melds < 4 {
            let mut shape = vec![1, 3];
            shape.extend(vec![2; 3 - melds]);
            shapes.push(shape);
        }
        if melds == 0 {
            shapes.push(vec![0, 1, 1, 1, 1, 1, 1]); // seven distinct pairs
            shapes.push(vec![4]); // thirteen orphans, both wait forms
        }
        rng.shuffle(&mut shapes);
        Self {
            seat,
            called,
            melds,
            catalogs,
            shapes,
        }
    }
}

/// Joint backtracking reserves the constrained hands before free marginals.
/// A locally valid first hand must not make a second riichi impossible.
struct Dealer {
    plans: Vec<Plan>,
    available: TileSet,
    hands: [TileSet; 4],
    remaining: Option<usize>,
    exhausted: bool,
}

impl Dealer {
    /// With no spare tiles, choose a shape per hand and cover the pool jointly.
    /// Assigning the scarcest tile first avoids committing a whole early hand
    /// whose locally legal shape strands every possible later allocation.
    fn tight_shapes(
        &mut self,
        at: usize,
        pending: &mut Vec<(usize, usize)>,
        distinct: &mut [bool],
    ) -> bool {
        if self.exhausted {
            return false;
        }
        if at == self.plans.len() {
            return self.cover(pending, distinct);
        }
        for shape in self.plans[at].shapes.clone() {
            let before = pending.len();
            distinct[at] = shape.len() == 7;
            pending.extend(shape.iter().map(|kind| (at, *kind)));
            if self.tight_shapes(at + 1, pending, distinct) {
                return true;
            }
            pending.truncate(before);
            if self.exhausted {
                return false;
            }
        }
        false
    }

    fn cover(&mut self, pending: &mut Vec<(usize, usize)>, distinct: &[bool]) -> bool {
        if let Some(left) = &mut self.remaining {
            if *left == 0 {
                self.exhausted = true;
                return false;
            }
            *left -= 1;
        }
        if pending.is_empty() {
            return self.available.is_empty()
                && self
                    .plans
                    .iter()
                    .all(|p| p.accepts(&self.hands[p.seat.index()]));
        }
        let mut choices: [Vec<(usize, TileSet)>; KINDS] = std::array::from_fn(|_| Vec::new());
        for (position, &(at, kind)) in pending.iter().enumerate() {
            // Identical slots of one hand are interchangeable, not new branches.
            if pending[..position].contains(&(at, kind)) {
                continue;
            }
            let plan = &self.plans[at];
            let held = &self.hands[plan.seat.index()];
            for &group in &plan.catalogs[kind] {
                if group.tiles().any(|t| {
                    group.count(t) > self.available.count(t) || (distinct[at] && held.count(t) > 0)
                }) {
                    continue;
                }
                for tile in Tile::all().filter(|t| group.count(*t) > 0) {
                    choices[tile.idx()].push((position, group));
                }
            }
        }
        let Some(tile) = Tile::all()
            .filter(|t| self.available.count(*t) > 0)
            .min_by_key(|t| choices[t.idx()].len())
        else {
            return false;
        };
        for (position, group) in std::mem::take(&mut choices[tile.idx()]) {
            let part = pending.remove(position);
            let seat = self.plans[part.0].seat.index();
            for tile in group.tiles() {
                self.available.remove(tile);
                self.hands[seat].add(tile);
            }
            if self.cover(pending, distinct) {
                return true;
            }
            for tile in group.tiles() {
                self.available.add(tile);
                self.hands[seat].remove(tile);
            }
            pending.insert(position, part);
            if self.exhausted {
                return false;
            }
        }
        false
    }

    fn next(&mut self, at: usize) -> bool {
        if self.exhausted {
            return false;
        }
        if at == self.plans.len() {
            return true;
        }
        let plan = &self.plans[at];
        if at + 1 == self.plans.len() && self.available.len() == 13 - 3 * plan.melds {
            // With no free tiles left the last hand is forced. Checking it
            // once avoids enumerating every decomposition for each rejected
            // preceding hand in tightly constrained multi-riichi positions.
            if !plan.accepts(&self.available) {
                return false;
            }
            self.hands[plan.seat.index()] = self.available;
            self.available = TileSet::new();
            return true;
        }
        for shape in self.plans[at].shapes.clone() {
            if self.parts(at, &shape, 0, 0, TileSet::new()) {
                return true;
            }
            if self.exhausted {
                break;
            }
        }
        false
    }

    fn parts(
        &mut self,
        at: usize,
        shape: &[usize],
        step: usize,
        first: usize,
        held: TileSet,
    ) -> bool {
        if let Some(left) = &mut self.remaining {
            if *left == 0 {
                self.exhausted = true;
                return false;
            }
            *left -= 1;
        }
        if step == shape.len() {
            let plan = &self.plans[at];
            if !plan.accepts(&held) {
                return false;
            }
            self.hands[plan.seat.index()] = held;
            return self.next(at + 1);
        }
        let catalog = shape[step];
        if shape.len() == 7 && catalog == 1 {
            let pairs = Tile::all()
                .filter(|t| held.count(*t) == 0 && self.available.count(*t) >= 2)
                .count();
            if pairs < shape.len() - step {
                return false;
            }
        }
        for index in first..self.plans[at].catalogs[catalog].len() {
            let group = self.plans[at].catalogs[catalog][index];
            // Seven-pairs components have distinct kinds, not four copies.
            if shape.len() == 7 && group.tiles().any(|t| held.count(t) > 0) {
                continue;
            }
            if group
                .tiles()
                .any(|t| group.count(t) > self.available.count(t))
            {
                continue;
            }
            let mut candidate = held;
            for tile in group.tiles() {
                self.available.remove(tile);
                candidate.add(tile);
            }
            let next = if shape.get(step + 1) == Some(&catalog) {
                index
            } else {
                0
            };
            if self.parts(at, shape, step + 1, next, candidate) {
                return true;
            }
            for tile in group.tiles() {
                self.available.add(tile);
            }
            if self.exhausted {
                return false;
            }
        }
        false
    }
}

pub(super) fn reserve(
    hand: &Hand,
    observer: Wind,
    belief: &Belief,
    rng: &mut Rng,
    pool: &mut Vec<Tile>,
) -> [TileSet; 4] {
    let mut seats: Vec<_> = Wind::ALL
        .into_iter()
        .filter(|seat| *seat != observer && hand.players[seat.index()].has_riichi())
        .collect();
    if seats.is_empty() {
        return [TileSet::new(); 4];
    }
    // An unlucky early shape can make the later hands impossible. Try
    // different public-only orders before exhaustively disproving that shape.
    // Each abort unwinds its reservations; the caller's pool is untouched.
    // The final unbounded attempt retains completeness, including rare waits.
    for attempt in 0..=24 {
        rng.shuffle(&mut seats);
        let plans = seats
            .iter()
            .map(|seat| Plan::new(hand, *seat, observer, belief, rng))
            .collect();
        let mut dealer = Dealer {
            plans,
            available: TileSet::from_tiles(pool.iter().copied()),
            hands: [TileSet::new(); 4],
            remaining: (attempt < 24).then_some(2048 << (attempt / 8)),
            exhausted: false,
        };
        let needed: usize = dealer.plans.iter().map(|p| 13 - 3 * p.melds).sum();
        let tight = dealer.plans.len() > 1 && dealer.available.len() == needed;
        let found = if tight {
            let mut distinct = vec![false; dealer.plans.len()];
            dealer.tight_shapes(0, &mut Vec::new(), &mut distinct)
        } else {
            dealer.next(0)
        };
        if found {
            *pool = dealer.available.tiles().collect();
            rng.shuffle(pool);
            return dealer.hands;
        }
        assert!(
            dealer.exhausted,
            "public riichi constraints have no feasible concealed deal"
        );
    }
    unreachable!("the final exhaustive allocation either succeeds or proves infeasibility")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::Meld;
    use crate::score::Riichi;

    #[test]
    fn interrupted_joint_attempt_restores_every_reserved_tile() {
        let hand = Table::new().deal(&mut Rng::from_seed(5));
        let body: TileSet = "123m123p123s1112z".parse().unwrap();
        let plan = Plan::new(
            &hand,
            Wind::South,
            Wind::East,
            &Belief::even(),
            &mut Rng::from_seed(0),
        );
        let mut dealer = Dealer {
            plans: vec![plan],
            available: body,
            hands: [TileSet::new(); 4],
            remaining: Some(1),
            exhausted: false,
        };
        assert!(!dealer.parts(0, &[0, 2, 2, 2, 2], 0, 0, TileSet::new()));
        assert!(dealer.exhausted);
        assert_eq!(dealer.available, body);
    }

    #[test]
    fn interrupted_exact_cover_restores_hands_pool_and_slots() {
        let hand = Table::new().deal(&mut Rng::from_seed(5));
        let pool: TileSet = "123456789m123456789p123456789s111233345556z"
            .parse()
            .unwrap();
        let mut rng = Rng::from_seed(1);
        let plans = Wind::ALL[1..]
            .iter()
            .map(|seat| Plan::new(&hand, *seat, Wind::East, &Belief::even(), &mut rng))
            .collect();
        let mut dealer = Dealer {
            plans,
            available: pool,
            hands: [TileSet::new(); 4],
            remaining: Some(1),
            exhausted: false,
        };
        let mut pending: Vec<_> = (0..3)
            .flat_map(|at| [0, 2, 2, 2, 2].into_iter().map(move |kind| (at, kind)))
            .collect();
        let original = pending.clone();
        assert!(!dealer.cover(&mut pending, &[false; 3]));
        assert!(dealer.exhausted);
        assert_eq!(dealer.available, pool);
        assert_eq!(dealer.hands, [TileSet::new(); 4]);
        assert_eq!(pending, original);
    }

    #[test]
    fn three_declared_hands_are_reserved_jointly_from_the_same_pool() {
        let pool = [
            "123m123p123s1112z",
            "456m456p456s3334z",
            "789m789p789s5556z",
        ]
        .iter()
        .flat_map(|s| s.parse::<TileSet>().unwrap().tiles().collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let mut hand = Table::new().deal(&mut Rng::from_seed(5));
        for player in &mut hand.players[1..] {
            player.riichi = Riichi::Declared;
        }
        for seed in 0..8 {
            let mut remaining = pool.clone();
            let got = reserve(
                &hand,
                Wind::East,
                &Belief::even(),
                &mut Rng::from_seed(seed),
                &mut remaining,
            );
            assert!(remaining.is_empty());
            assert_eq!(
                TileSet::from_tiles(got.iter().flat_map(|h| h.tiles())),
                TileSet::from_tiles(pool.iter().copied())
            );
            for body in &got[1..] {
                assert!(shanten::is_tenpai(body, 0, body));
            }
        }
    }

    #[test]
    fn every_wait_family_and_concealed_quads_can_be_proposed_without_a_live_wait() {
        for (text, quads) in [
            ("123m123p123s1112z", 0),
            ("112233m445566p7z", 0),
            ("19m19p19s1234567z", 0),
            ("119m19p19s123456z", 0),
            ("123p123s1112z", 1),
            ("2z", 4),
        ] {
            let body: TileSet = text.parse().unwrap();
            let mut hand = Table::new().deal(&mut Rng::from_seed(42));
            let player = &mut hand.players[1];
            player.riichi = Riichi::Declared;
            player.melds = (0..quads)
                .map(|n| Meld::concealed_kan(Tile::new(n)))
                .collect();
            // Deliberately hide a different hand: only public constraints and
            // this candidate pool may determine the result.
            player.hand = TileSet::from_tiles(vec![Tile::new(33); body.len()]);
            for seed in 0..8 {
                let mut pool: Vec<_> = body.tiles().collect();
                let got = reserve(
                    &hand,
                    Wind::East,
                    &Belief::even(),
                    &mut Rng::from_seed(seed),
                    &mut pool,
                );
                assert_eq!(got[1], body, "{text}, {quads} quads");
                assert!(pool.is_empty()); // the completing tile need not remain unseen
                let plan = Plan::new(
                    &hand,
                    Wind::South,
                    Wind::East,
                    &Belief::even(),
                    &mut Rng::from_seed(seed),
                );
                let shape = if text == "112233m445566p7z" {
                    vec![0, 1, 1, 1, 1, 1, 1]
                } else if text == "19m19p19s1234567z" || text == "119m19p19s123456z" {
                    vec![4]
                } else {
                    let mut shape = vec![0];
                    shape.extend(vec![2; 4 - quads as usize]);
                    shape
                };
                let mut dealer = Dealer {
                    plans: vec![plan],
                    available: body,
                    hands: [TileSet::new(); 4],
                    remaining: None,
                    exhausted: false,
                };
                assert!(dealer.parts(0, &shape, 0, 0, TileSet::new()));
            }
        }
    }
}
