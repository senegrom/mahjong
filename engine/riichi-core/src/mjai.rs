//! The mjai event log, the de facto interchange format for riichi programs.
//!
//! A game is written as one JSON object per line, in the order things
//! happened. Other people's tools read it: replayers, reviewers, and rival
//! bots such as Mortal, which is how this engine can be measured against
//! something it did not train on.
//!
//! Two conventions of the format differ from the way the engine works
//! inside, and both are handled when a log is written rather than in the
//! game itself:
//!
//! - Players are numbered, not seated. Number 0 is whoever dealt the very
//!   first hand, and keeps that number all game, while the seat they sit in
//!   moves. [`Event::to_json`] takes the seating and does the translation.
//! - Tiles are written the Japanese way: `1m` to `9s` for the suits, then
//!   `E`, `S`, `W`, `N` for the winds and `P`, `F`, `C` for white, green and
//!   red, in place of this engine's `1z` to `7z`.

use crate::hand::{Meld, MeldKind};
use crate::score::Score;
use crate::tile::{Suit, Tile};
use crate::Wind;
use core::fmt::Write as _;

/// How a tile is written in an mjai log.
///
/// ```
/// use riichi_core::mjai::name;
/// assert_eq!(name("5s".parse().unwrap()), "5s");
/// assert_eq!(name("1z".parse().unwrap()), "E");
/// assert_eq!(name("7z".parse().unwrap()), "C");
/// ```
pub fn name(tile: Tile) -> String {
    match tile.suit() {
        Suit::Honours => ["E", "S", "W", "N", "P", "F", "C"][tile.idx() - 27].to_string(),
        suit => format!("{}{}", tile.rank(), suit.letter()),
    }
}

/// Reads a tile back from the way an mjai log writes it.
///
/// Red fives arrive as `5mr`; this rule set has none, so the tile is taken
/// as the ordinary five rather than refused, which lets logs from other
/// programs be read.
pub fn parse(text: &str) -> Option<Tile> {
    match text {
        "E" => return Some(crate::tile::EAST),
        "S" => return Some(crate::tile::SOUTH),
        "W" => return Some(crate::tile::WEST),
        "N" => return Some(crate::tile::NORTH),
        "P" => return Some(crate::tile::WHITE),
        "F" => return Some(crate::tile::GREEN),
        "C" => return Some(crate::tile::RED),
        _ => {}
    }
    let text = text.strip_suffix('r').unwrap_or(text);
    let mut chars = text.chars();
    let rank = chars.next()?.to_digit(10)? as u8;
    let suit = match chars.next()? {
        'm' => Suit::Characters,
        'p' => Suit::Circles,
        's' => Suit::Bamboo,
        _ => return None,
    };
    if chars.next().is_some() || !(1..=9).contains(&rank) {
        return None;
    }
    Some(Tile::numbered(suit, rank))
}

/// One thing that happened, in the order it happened.
///
/// Seats are held as winds because that is what the game state machine
/// deals in; the numbering the format wants is applied on the way out.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// The game begins.
    StartGame {
        /// What to call the four players, by number.
        names: [String; 4],
    },
    /// A hand is dealt.
    StartKyoku {
        /// The prevailing wind.
        round: Wind,
        /// Which hand of the round this is, from 1.
        kyoku: u8,
        /// Counters carried on the table.
        honba: u32,
        /// Riichi bets carried on the table.
        kyotaku: u32,
        /// The first dora indicator.
        indicator: Tile,
        /// Points held, by seat.
        scores: [i32; 4],
        /// The thirteen tiles dealt to each seat.
        hands: [Vec<Tile>; 4],
    },
    /// A tile is drawn, from the wall or, after a quad, the dead wall.
    Tsumo {
        /// Who drew it.
        actor: Wind,
        /// What they drew.
        tile: Tile,
    },
    /// A tile is discarded.
    Dahai {
        /// Who discarded.
        actor: Wind,
        /// What they discarded.
        tile: Tile,
        /// Whether it is the tile they had just drawn.
        drawn: bool,
    },
    /// A riichi is declared, before the discard that goes with it.
    Reach {
        /// Who declared.
        actor: Wind,
    },
    /// The declaration stands, because nobody won on the discard.
    ReachAccepted {
        /// Who declared.
        actor: Wind,
    },
    /// A sequence is claimed.
    Chi {
        /// Who claimed.
        actor: Wind,
        /// Whose discard it was.
        target: Wind,
        /// The claimed tile.
        tile: Tile,
        /// The two tiles from hand that complete the set.
        consumed: Vec<Tile>,
    },
    /// A triplet is claimed.
    Pon {
        /// Who claimed.
        actor: Wind,
        /// Whose discard it was.
        target: Wind,
        /// The claimed tile.
        tile: Tile,
        /// The two tiles from hand that complete the set.
        consumed: Vec<Tile>,
    },
    /// A quad is claimed from a discard.
    Daiminkan {
        /// Who claimed.
        actor: Wind,
        /// Whose discard it was.
        target: Wind,
        /// The claimed tile.
        tile: Tile,
        /// The three tiles from hand that complete the quad.
        consumed: Vec<Tile>,
    },
    /// A quad is made by adding to a claimed triplet.
    Kakan {
        /// Who declared.
        actor: Wind,
        /// The tile added.
        tile: Tile,
        /// The triplet it was added to.
        consumed: Vec<Tile>,
    },
    /// A quad is made from four concealed tiles.
    Ankan {
        /// Who declared.
        actor: Wind,
        /// The four tiles.
        consumed: Vec<Tile>,
    },
    /// A new dora indicator is turned face up.
    Dora {
        /// The indicator.
        indicator: Tile,
    },
    /// A hand is won.
    Hora {
        /// The winner.
        actor: Wind,
        /// Who let the tile go, which is the winner on a self-draw.
        target: Wind,
        /// The winning tile.
        tile: Tile,
        /// The ura dora indicators, which only a riichi winner sees.
        ura: Vec<Tile>,
        /// Minipoints.
        fu: u32,
        /// Han, dora included.
        han: u8,
        /// What the hand paid, bets aside.
        points: u32,
        /// What each seat gained or lost over the hand.
        deltas: [i32; 4],
        /// Points held afterwards, by seat.
        scores: [i32; 4],
    },
    /// The wall ran out.
    Ryukyoku {
        /// Why the hand ended without a winner. Only `"howanpai"`, the
        /// exhausted wall, occurs here: this rule set has no abortive draws.
        reason: &'static str,
        /// Who was waiting, by seat.
        tenpai: [bool; 4],
        /// What each seat gained or lost over the hand.
        deltas: [i32; 4],
        /// Points held afterwards, by seat.
        scores: [i32; 4],
    },
    /// The hand is over.
    EndKyoku,
    /// The game is over.
    EndGame,
}

/// Writes a list of tiles as a JSON array.
fn tiles_json(tiles: &[Tile]) -> String {
    let mut out = String::from("[");
    for (index, tile) in tiles.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(out, "\"{}\"", name(*tile));
    }
    out.push(']');
    out
}

/// Writes four numbers as a JSON array, in player order.
fn four_json(values: [i32; 4], seats: [usize; 4]) -> String {
    let mut ordered = [0; 4];
    for seat in Wind::ALL {
        ordered[seats[seat.index()]] = values[seat.index()];
    }
    let mut out = String::from("[");
    for (index, value) in ordered.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(out, "{value}");
    }
    out.push(']');
    out
}

/// The seat a player number sits in.
fn seat_of(seats: [usize; 4], player: usize) -> Wind {
    Wind::ALL
        .into_iter()
        .find(|seat| seats[seat.index()] == player)
        .expect("every number is seated")
}

impl Event {
    /// The event as one line of JSON.
    ///
    /// `seats` says which player number sits in each seat, indexed the way
    /// [`Wind::index`] indexes: `seats[Wind::East.index()]` is the number of
    /// this hand's dealer. A game where nobody has moved yet is the identity.
    pub fn to_json(&self, seats: [usize; 4]) -> String {
        let who = |seat: Wind| seats[seat.index()];
        match self {
            Event::StartGame { names } => {
                let mut out = String::from("{\"type\":\"start_game\",\"names\":[");
                for (index, name) in names.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "\"{}\"", name.replace('"', "'"));
                }
                out.push_str("]}");
                out
            }
            Event::StartKyoku {
                round,
                kyoku,
                honba,
                kyotaku,
                indicator,
                scores,
                hands,
            } => {
                let mut dealt = String::from("[");
                for player in 0..4 {
                    if player > 0 {
                        dealt.push(',');
                    }
                    dealt.push_str(&tiles_json(&hands[seat_of(seats, player).index()]));
                }
                dealt.push(']');
                format!(
                    "{{\"type\":\"start_kyoku\",\"bakaze\":\"{}\",\"kyoku\":{},\"honba\":{},\"kyotaku\":{},\"oya\":{},\"dora_marker\":\"{}\",\"scores\":{},\"tehais\":{}}}",
                    name(round.tile()),
                    kyoku,
                    honba,
                    kyotaku,
                    who(Wind::East),
                    name(*indicator),
                    four_json(*scores, seats),
                    dealt,
                )
            }
            Event::Tsumo { actor, tile } => format!(
                "{{\"type\":\"tsumo\",\"actor\":{},\"pai\":\"{}\"}}",
                who(*actor),
                name(*tile)
            ),
            Event::Dahai { actor, tile, drawn } => format!(
                "{{\"type\":\"dahai\",\"actor\":{},\"pai\":\"{}\",\"tsumogiri\":{}}}",
                who(*actor),
                name(*tile),
                drawn
            ),
            Event::Reach { actor } => {
                format!("{{\"type\":\"reach\",\"actor\":{}}}", who(*actor))
            }
            Event::ReachAccepted { actor } => {
                format!("{{\"type\":\"reach_accepted\",\"actor\":{}}}", who(*actor))
            }
            Event::Chi {
                actor,
                target,
                tile,
                consumed,
            } => claim_json("chi", who(*actor), who(*target), *tile, consumed),
            Event::Pon {
                actor,
                target,
                tile,
                consumed,
            } => claim_json("pon", who(*actor), who(*target), *tile, consumed),
            Event::Daiminkan {
                actor,
                target,
                tile,
                consumed,
            } => claim_json("daiminkan", who(*actor), who(*target), *tile, consumed),
            Event::Kakan {
                actor,
                tile,
                consumed,
            } => format!(
                "{{\"type\":\"kakan\",\"actor\":{},\"pai\":\"{}\",\"consumed\":{}}}",
                who(*actor),
                name(*tile),
                tiles_json(consumed)
            ),
            Event::Ankan { actor, consumed } => format!(
                "{{\"type\":\"ankan\",\"actor\":{},\"consumed\":{}}}",
                who(*actor),
                tiles_json(consumed)
            ),
            Event::Dora { indicator } => format!(
                "{{\"type\":\"dora\",\"dora_marker\":\"{}\"}}",
                name(*indicator)
            ),
            Event::Hora {
                actor,
                target,
                tile,
                ura,
                fu,
                han,
                points,
                deltas,
                scores,
            } => format!(
                "{{\"type\":\"hora\",\"actor\":{},\"target\":{},\"pai\":\"{}\",\"uradora_markers\":{},\"fu\":{},\"fan\":{},\"hora_points\":{},\"deltas\":{},\"scores\":{}}}",
                who(*actor),
                who(*target),
                name(*tile),
                tiles_json(ura),
                fu,
                han,
                points,
                four_json(*deltas, seats),
                four_json(*scores, seats),
            ),
            Event::Ryukyoku {
                reason,
                tenpai,
                deltas,
                scores,
            } => {
                let mut waiting = String::from("[");
                for player in 0..4 {
                    if player > 0 {
                        waiting.push(',');
                    }
                    let seat = seat_of(seats, player);
                    waiting.push_str(if tenpai[seat.index()] { "true" } else { "false" });
                }
                waiting.push(']');
                format!(
                    "{{\"type\":\"ryukyoku\",\"reason\":\"{reason}\",\"tenpais\":{waiting},\"deltas\":{},\"scores\":{}}}",
                    four_json(*deltas, seats),
                    four_json(*scores, seats),
                )
            }
            Event::EndKyoku => "{\"type\":\"end_kyoku\"}".to_string(),
            Event::EndGame => "{\"type\":\"end_game\"}".to_string(),
        }
    }
}

/// The three claims share a shape.
fn claim_json(kind: &str, actor: usize, target: usize, tile: Tile, consumed: &[Tile]) -> String {
    format!(
        "{{\"type\":\"{kind}\",\"actor\":{actor},\"target\":{target},\"pai\":\"{}\",\"consumed\":{}}}",
        name(tile),
        tiles_json(consumed)
    )
}

/// The tiles a claim takes out of the claimer's own hand.
pub fn consumed_by(meld: &Meld, claimed: Tile) -> Vec<Tile> {
    match meld.kind {
        MeldKind::Chii => meld
            .tiles()
            .into_iter()
            .filter(|tile| *tile != claimed)
            .collect(),
        MeldKind::Pon => vec![meld.tile; 2],
        MeldKind::ClaimedKan | MeldKind::ExtendedKan => vec![meld.tile; 3],
        MeldKind::ConcealedKan => vec![meld.tile; 4],
    }
}

/// What a winning hand paid, bets aside, as the format reports it.
pub fn hora_points(score: &Score, by_discard: bool) -> u32 {
    if by_discard {
        score.payments.from_discarder
    } else {
        score.payments.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_are_written_the_japanese_way() {
        for tile in Tile::all() {
            assert_eq!(parse(&name(tile)), Some(tile), "{tile} did not survive");
        }
        assert_eq!(name("5z".parse().unwrap()), "P");
        assert_eq!(parse("5mr"), Some("5m".parse().unwrap()));
        assert_eq!(parse("0m"), None);
        assert_eq!(parse("nonsense"), None);
    }

    /// Player numbers stay with the person while the seats move, so a log
    /// from a later hand still names the same four people.
    #[test]
    fn seats_are_translated_to_player_numbers() {
        // South's chair is held by player 3.
        let seats = [1, 3, 0, 2];
        let event = Event::Tsumo {
            actor: Wind::South,
            tile: "3p".parse().unwrap(),
        };
        assert_eq!(
            event.to_json(seats),
            "{\"type\":\"tsumo\",\"actor\":3,\"pai\":\"3p\"}"
        );
    }

    #[test]
    fn scores_are_listed_in_player_order() {
        let seats = [1, 3, 0, 2];
        let event = Event::Ryukyoku {
            reason: "howanpai",
            tenpai: [true, false, false, false],
            deltas: [3000, -1000, -1000, -1000],
            scores: [28000, 24000, 24000, 24000],
        };
        let line = event.to_json(seats);
        // East sits as player 1, so the gain is the second entry.
        assert!(
            line.contains("\"deltas\":[-1000,3000,-1000,-1000]"),
            "{line}"
        );
        assert!(
            line.contains("\"tenpais\":[false,true,false,false]"),
            "{line}"
        );
    }
}
