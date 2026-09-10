use super::{ActionCandidate, PlayerState};
use crate::tile::Tile;

#[cfg(feature = "pymod")]
use pyo3::prelude::*;

impl PlayerState {
    #[inline]
    #[must_use]
    pub const fn player_id(&self) -> u8 {
        self.player_id
    }
    #[inline]
    #[must_use]
    pub const fn kyoku(&self) -> u8 {
        self.kyoku
    }
    #[inline]
    #[must_use]
    pub const fn honba(&self) -> u8 {
        self.honba
    }
    #[inline]
    #[must_use]
    pub const fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    #[inline]
    #[must_use]
    pub const fn is_oya(&self) -> bool {
        self.oya == 0
    }

    #[inline]
    #[must_use]
    pub const fn tehai(&self) -> [u8; 34] {
        self.tehai
    }
    #[inline]
    #[must_use]
    pub const fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    #[inline]
    #[must_use]
    pub fn chis(&self) -> &[u8] {
        &self.chis
    }
    #[inline]
    #[must_use]
    pub fn pons(&self) -> &[u8] {
        &self.pons
    }
    #[inline]
    #[must_use]
    pub fn minkans(&self) -> &[u8] {
        &self.minkans
    }
    #[inline]
    #[must_use]
    pub fn ankans(&self) -> &[u8] {
        &self.ankans
    }

    #[inline]
    #[must_use]
    pub const fn at_turn(&self) -> u8 {
        self.at_turn
    }
    #[inline]
    #[must_use]
    pub const fn shanten(&self) -> i8 {
        self.shanten
    }
    #[inline]
    #[must_use]
    pub const fn waits(&self) -> [bool; 34] {
        self.waits
    }

    #[inline]
    #[must_use]
    pub const fn last_cans(&self) -> ActionCandidate {
        self.last_cans
    }

    #[inline]
    #[must_use]
    pub const fn can_w_riichi(&self) -> bool {
        self.can_w_riichi
    }
    #[inline]
    #[must_use]
    pub const fn self_riichi_declared(&self) -> bool {
        self.riichi_declared[0]
    }
    #[inline]
    #[must_use]
    pub const fn self_riichi_accepted(&self) -> bool {
        self.riichi_accepted[0]
    }

    #[inline]
    #[must_use]
    pub const fn at_furiten(&self) -> bool {
        self.at_furiten
    }

    #[inline]
    #[must_use]
    pub const fn last_self_tsumo(&self) -> Option<Tile> {
        self.last_self_tsumo
    }
    #[inline]
    #[must_use]
    pub const fn last_kawa_tile(&self) -> Option<Tile> {
        self.last_kawa_tile
    }

    #[inline]
    #[must_use]
    pub fn ankan_candidates(&self) -> &[Tile] {
        &self.ankan_candidates
    }
    #[inline]
    #[must_use]
    pub fn kakan_candidates(&self) -> &[Tile] {
        &self.kakan_candidates
    }
}

/// The same accessors as Python attributes and methods; they forward to the
/// plain Rust getters above, which are what a non-Python build (wasm) sees.
#[cfg(feature = "pymod")]
#[pymethods]
impl PlayerState {
    #[getter(player_id)]
    #[inline]
    fn player_id_py(&self) -> u8 {
        self.player_id()
    }
    #[getter(kyoku)]
    #[inline]
    fn kyoku_py(&self) -> u8 {
        self.kyoku()
    }
    #[getter(honba)]
    #[inline]
    fn honba_py(&self) -> u8 {
        self.honba()
    }
    #[getter(kyotaku)]
    #[inline]
    fn kyotaku_py(&self) -> u8 {
        self.kyotaku()
    }
    #[getter(is_oya)]
    #[inline]
    fn is_oya_py(&self) -> bool {
        self.is_oya()
    }

    #[getter(tehai)]
    #[inline]
    fn tehai_py(&self) -> [u8; 34] {
        self.tehai()
    }
    #[getter(akas_in_hand)]
    #[inline]
    fn akas_in_hand_py(&self) -> [bool; 3] {
        self.akas_in_hand()
    }

    #[getter(chis)]
    #[inline]
    fn chis_py(&self) -> &[u8] {
        self.chis()
    }
    #[getter(pons)]
    #[inline]
    fn pons_py(&self) -> &[u8] {
        self.pons()
    }
    #[getter(minkans)]
    #[inline]
    fn minkans_py(&self) -> &[u8] {
        self.minkans()
    }
    #[getter(ankans)]
    #[inline]
    fn ankans_py(&self) -> &[u8] {
        self.ankans()
    }

    #[getter(at_turn)]
    #[inline]
    fn at_turn_py(&self) -> u8 {
        self.at_turn()
    }
    #[getter(shanten)]
    #[inline]
    fn shanten_py(&self) -> i8 {
        self.shanten()
    }
    #[getter(waits)]
    #[inline]
    fn waits_py(&self) -> [bool; 34] {
        self.waits()
    }

    #[inline]
    #[pyo3(name = "last_self_tsumo")]
    fn last_self_tsumo_py(&self) -> Option<String> {
        self.last_self_tsumo.map(|t| t.to_string())
    }
    #[inline]
    #[pyo3(name = "last_kawa_tile")]
    fn last_kawa_tile_py(&self) -> Option<String> {
        self.last_kawa_tile.map(|t| t.to_string())
    }

    #[getter(last_cans)]
    #[inline]
    fn last_cans_py(&self) -> ActionCandidate {
        self.last_cans()
    }

    #[inline]
    #[pyo3(name = "ankan_candidates")]
    fn ankan_candidates_py(&self) -> Vec<String> {
        self.ankan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }
    #[inline]
    #[pyo3(name = "kakan_candidates")]
    fn kakan_candidates_py(&self) -> Vec<String> {
        self.kakan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }

    #[getter(can_w_riichi)]
    #[inline]
    fn can_w_riichi_py(&self) -> bool {
        self.can_w_riichi()
    }
    #[getter(self_riichi_declared)]
    #[inline]
    fn self_riichi_declared_py(&self) -> bool {
        self.self_riichi_declared()
    }
    #[getter(self_riichi_accepted)]
    #[inline]
    fn self_riichi_accepted_py(&self) -> bool {
        self.self_riichi_accepted()
    }

    #[getter(at_furiten)]
    #[inline]
    fn at_furiten_py(&self) -> bool {
        self.at_furiten()
    }
}
