//! The wall, the dead wall and the dora indicators.
//!
//! The physical wall is a square of stacks that players break at a rolled
//! position (EMA 2025, sections 2.4 to 2.7). Where the break falls changes
//! which tiles reach which player, but with a shuffled wall it changes
//! nothing about the game, so the wall is modelled as a shuffled sequence
//! with the dead wall taken off the end. The dice are still rolled and kept
//! in the log, so a replay can show the table exactly as it was.
//!
//! The dead wall always holds fourteen tiles: four replacement tiles for
//! quads, then five dora indicators with their five ura dora underneath.
//! After each quad the last live tile joins the dead wall, so the count
//! holds and the number of draws falls by one (section 3.3.4).

use crate::rng::Rng;
use crate::tile::{Tile, COPIES, KINDS, SET_SIZE};

/// Tiles held back from play as the dead wall.
pub const DEAD_WALL: usize = 14;
/// Replacement tiles available, one per quad.
pub const REPLACEMENTS: usize = 4;
/// Quads allowed in a hand (EMA section 3.3.5).
pub const MAX_QUADS: usize = 4;

/// The wall of a single hand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Wall {
    tiles: Vec<Tile>,
    /// Index of the next live draw.
    next_draw: usize,
    /// Index one past the last live tile; falls by one for every quad.
    live_end: usize,
    /// Replacement tiles taken so far.
    replacements_taken: usize,
    /// Dora indicators revealed so far, at least one.
    indicators_revealed: usize,
    /// The dice roll that broke the wall, kept for replays.
    dice: u8,
}

impl Wall {
    /// Builds and shuffles a wall, then reveals the first dora indicator.
    pub fn shuffled(rng: &mut Rng) -> Wall {
        let mut tiles = Vec::with_capacity(SET_SIZE);
        for index in 0..KINDS as u8 {
            for _ in 0..COPIES {
                tiles.push(Tile::new(index));
            }
        }
        rng.shuffle(&mut tiles);
        let dice = rng.roll_dice();
        Wall {
            tiles,
            next_draw: 0,
            live_end: SET_SIZE - DEAD_WALL,
            replacements_taken: 0,
            indicators_revealed: 1,
            dice,
        }
    }

    /// The dice roll East made to break the wall.
    pub const fn dice(&self) -> u8 {
        self.dice
    }

    /// How many tiles are still there to draw. Riichi needs at least one
    /// (EMA 2025 section 3.3.10) and the hand ends when this reaches zero.
    pub const fn remaining(&self) -> usize {
        self.live_end - self.next_draw
    }

    /// Whether the live wall is empty, so the next discard is the last one.
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Draws the next tile, or `None` when the wall is out.
    pub fn draw(&mut self) -> Option<Tile> {
        if self.is_empty() {
            return None;
        }
        let tile = self.tiles[self.next_draw];
        self.next_draw += 1;
        Some(tile)
    }

    /// Deals a hand of `count` tiles.
    pub fn deal(&mut self, count: usize) -> Vec<Tile> {
        (0..count).filter_map(|_| self.draw()).collect()
    }

    /// Whether another quad may be declared: at most four in a hand, and
    /// never once the last live tile has been drawn (sections 3.3.5, 3.4.1).
    pub const fn can_declare_quad(&self) -> bool {
        self.replacements_taken < MAX_QUADS && self.remaining() > 0
    }

    /// Takes a replacement tile after a quad, revealing the new kan dora and
    /// moving the last live tile into the dead wall.
    pub fn take_replacement(&mut self) -> Option<Tile> {
        if self.replacements_taken >= MAX_QUADS {
            return None;
        }
        let tile = self.tiles[SET_SIZE - DEAD_WALL + self.replacements_taken];
        self.replacements_taken += 1;
        self.indicators_revealed += 1;
        // The dead wall keeps fourteen tiles, so the live wall gives one up.
        if self.live_end > self.next_draw {
            self.live_end -= 1;
        }
        Some(tile)
    }

    /// The dora indicators face up on the table.
    pub fn dora_indicators(&self) -> Vec<Tile> {
        (0..self.indicators_revealed)
            .map(|index| self.tiles[SET_SIZE - DEAD_WALL + REPLACEMENTS + index * 2])
            .collect()
    }

    /// The ura dora indicators, which only a riichi winner may look at
    /// (EMA section 3.3.10).
    pub fn ura_indicators(&self) -> Vec<Tile> {
        (0..self.indicators_revealed)
            .map(|index| self.tiles[SET_SIZE - DEAD_WALL + REPLACEMENTS + index * 2 + 1])
            .collect()
    }

    /// Every tile of the wall in order, for replays and tests.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::TileSet;

    fn wall() -> Wall {
        Wall::shuffled(&mut Rng::from_seed(20260902))
    }

    /// EMA 2025 section 1.3: a set is 136 tiles, four of each kind.
    #[test]
    fn a_wall_is_a_full_set() {
        let wall = wall();
        assert_eq!(wall.tiles().len(), SET_SIZE);
        let counted = TileSet::from_tiles(wall.tiles().iter().copied());
        assert!(counted.is_legal());
        assert!(Tile::all().all(|tile| counted.count(tile) == COPIES));
    }

    /// EMA 2025 section 2.6: the dead wall is fourteen tiles, so a hand
    /// deals 53 and leaves 69 draws.
    #[test]
    fn a_full_deal_leaves_sixty_nine_draws() {
        let mut wall = wall();
        assert_eq!(wall.remaining(), SET_SIZE - DEAD_WALL);
        let mut dealt = 0;
        for count in [13, 13, 13, 14] {
            dealt += wall.deal(count).len();
        }
        assert_eq!(dealt, 53);
        assert_eq!(wall.remaining(), 69);
    }

    /// EMA 2025 section 3.3.4: after a quad the last tile of the wall joins
    /// the dead wall, so a replacement costs two draws in all.
    #[test]
    fn a_quad_shortens_the_wall() {
        let mut wall = wall();
        let before = wall.remaining();
        wall.take_replacement().unwrap();
        assert_eq!(wall.remaining(), before - 1);
        // A new kan dora is face up alongside the first indicator.
        assert_eq!(wall.dora_indicators().len(), 2);
        assert_eq!(wall.ura_indicators().len(), 2);
    }

    /// EMA 2025 section 3.3.5: no fifth quad, ever.
    #[test]
    fn at_most_four_quads() {
        let mut wall = wall();
        for _ in 0..MAX_QUADS {
            assert!(wall.can_declare_quad());
            assert!(wall.take_replacement().is_some());
        }
        assert!(!wall.can_declare_quad());
        assert!(wall.take_replacement().is_none());
        assert_eq!(wall.dora_indicators().len(), 5);
        assert_eq!(wall.ura_indicators().len(), 5);
    }

    /// EMA 2025 section 2.7: one indicator is face up at the start, with its
    /// ura underneath, and neither is a live tile.
    #[test]
    fn one_indicator_at_the_start() {
        let wall = wall();
        assert_eq!(wall.dora_indicators().len(), 1);
        assert_eq!(wall.ura_indicators().len(), 1);
        assert_ne!(wall.dora_indicators()[0], wall.ura_indicators()[0]);
    }

    /// EMA 2025 section 3.4.1: no quad may be declared once the last live
    /// tile is gone.
    #[test]
    fn the_wall_runs_out() {
        let mut wall = wall();
        while wall.draw().is_some() {}
        assert!(wall.is_empty());
        assert_eq!(wall.remaining(), 0);
        assert!(!wall.can_declare_quad());
    }

    #[test]
    fn the_same_seed_deals_the_same_wall() {
        let first = Wall::shuffled(&mut Rng::from_seed(5));
        let second = Wall::shuffled(&mut Rng::from_seed(5));
        assert_eq!(first.tiles(), second.tiles());
        assert_eq!(first.dice(), second.dice());
    }
}
