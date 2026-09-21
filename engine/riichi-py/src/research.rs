//! Opt-in training diagnostics. Labels and snapshots never alter the live game.
use super::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

/// Copy the complete simulation state, not search handles with outstanding work.
#[pyfunction]
fn research_fork(arena: &Arena, rows: Vec<usize>) -> PyResult<Arena> {
    if rows.iter().any(|&i| i >= arena.seats.len()) {
        return Err(PyValueError::new_err("snapshot row outside arena"));
    }
    if arena.pending.iter().any(Option::is_some)
        || arena.lookaheads.iter().any(Option::is_some)
        || arena.imagined.iter().any(|worlds| !worlds.is_empty())
    {
        return Err(PyValueError::new_err("finish outstanding search before snapshotting"));
    }
    // Construct empty scratch buffers; never deal new hands or consume RNG.
    let n = rows.len();
    let mut fork = Arena::new(0, 0, vec![]);
    fork.seats = rows.into_iter().map(|i| arena.seats[i].clone()).collect();
    fork.observations.resize(n * OBSERVATION, 0.0);
    fork.oracle.resize(n * ORACLE, 0.0);
    fork.mask.resize(n * ACTIONS, false);
    fork.hands.resize(n * HANDS, 0.0);
    fork.judgements = (0..n).map(|_| vec![]).collect();
    fork.judgement_weights = (0..n).map(|_| vec![]).collect();
    fork.pending = (0..n).map(|_| None).collect();
    fork.imagined = (0..n).map(|_| vec![]).collect();
    fork.imagined_chance = (0..n).map(|_| vec![]).collect();
    fork.lookaheads = (0..n).map(|_| None).collect();
    Ok(fork)
}

/// Public context only, by immutable player identity (not wind).
/// Columns: round, kyoku, dealer, honba, deposits, discards, actor, phase,
/// four scores, four riichi flags. -1 actor means finished.
#[pyfunction]
fn research_context(arena: &Arena) -> Vec<Vec<i64>> {
    arena.seats.iter().map(|seat| {
        let mut row = vec![seat.hand.round.index() as i64, seat.hand.kyoku as i64,
            seat.table.dealer as i64, seat.hand.counters as i64,
            seat.hand.riichi_sticks as i64, seat.hand.discards_made as i64,
            seat.pending().map(|w| seat.table.player_at(w) as i64).unwrap_or(-1),
            match seat.hand.phase { Phase::Act => 0, Phase::CallWindow => 1,
                                    Phase::Draw => 2, Phase::Over => 3 }];
        for player in 0..4 {
            row.push(seat.hand.players[seat.table.seat_of(player).index()].score as i64);
        }
        for player in 0..4 {
            row.push(i64::from(seat.hand.players[seat.table.seat_of(player).index()].has_riichi()));
        }
        row
    }).collect()
}

/// Exact legal-ron labels, NOT opponent choices. For each legal discard (stage 0)
/// and riichi-discard (stage 1), test each opponent separately on a cloned hand.
/// Labels: [game, stage, relative opponent, tile, (can_ron, payment_points)].
/// Mask: [game, stage, tile]. Payment is the discarder-to-winner liability if that
/// opponent alone claims ron, including honba but excluding riichi deposits.
#[pyfunction]
fn research_danger<'py>(arena: &Arena, py: Python<'py>) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    let blocks: Vec<(Vec<u8>, Vec<f32>)> = py.detach(|| arena.seats.par_iter().map(|seat| {
        let mut mask = vec![0u8; 2 * POSITIONS];
        let mut labels = vec![0f32; 2 * OPPONENTS * POSITIONS * 2];
        if seat.finished || !seat.asking.is_empty() || !matches!(seat.hand.phase, Phase::Act) {
            return (mask, labels);
        }
        let actor = seat.hand.turn;
        for action in seat.hand.legal_actions() {
            let (stage, tile) = match action {
                Action::Discard(t) => (0, t), Action::Riichi(t) => (1, t), _ => continue,
            };
            mask[stage * POSITIONS + tile.idx()] = 1;
            let mut after = seat.hand.clone();
            after.act(action).expect("offered action must be legal");
            let offered = after.legal_calls();
            for distance in 1..4 {
                let opponent = Wind::ALL[(actor.index() + distance) % 4];
                if !offered.iter().any(|(who, calls)| *who == opponent && calls.contains(&Call::Ron)) {
                    continue;
                }
                let mut won = after.clone();
                let answers: Vec<_> = offered.iter().map(|(who, _)|
                    (*who, if *who == opponent { Call::Ron } else { Call::Pass })).collect();
                let before = won.players[actor.index()].score;
                won.resolve_calls(&answers).expect("offered ron must resolve");
                let amount = before - won.players[actor.index()].score;
                let at = ((stage * OPPONENTS + distance - 1) * POSITIONS + tile.idx()) * 2;
                labels[at] = 1.0;
                labels[at + 1] = amount as f32;
            }
        }
        (mask, labels)
    }).collect());
    let mut masks = Vec::new();
    let mut labels = Vec::new();
    for (mask, label) in blocks { masks.extend(mask); labels.extend(label); }
    (PyBytes::new(py, &masks), PyBytes::new(py, bytemuck_cast(&labels)))
}

/// The four INITIAL 13-tile hands, before the dealer's extra draw. Evaluation
/// only: these are chance-control features, never policy observations.
#[pyfunction]
fn research_initial_counts(arena: &Arena) -> PyResult<Vec<Vec<Vec<u8>>>> {
    arena.seats.iter().map(|seat| {
        if seat.hands_done != 0 || seat.hand.discards_made != 0 || seat.hand.kyoku != 1 {
            return Err(PyValueError::new_err("initial counts must be captured before play"));
        }
        let hands = seat.hand.log.iter().find_map(|event| match event {
            mjai::Event::StartKyoku { hands, .. } => Some(hands), _ => None,
        }).ok_or_else(|| PyValueError::new_err("initial deal missing"))?;
        let mut counts = vec![vec![0u8; POSITIONS]; 4];
        for (wind, hand) in hands.iter().enumerate() {
            let player = seat.table.player_at(Wind::ALL[wind]);
            for tile in hand { counts[player][tile.idx()] += 1; }
        }
        Ok(counts)
    }).collect()
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RESEARCH_API_VERSION", 1u32)?;
    module.add_function(wrap_pyfunction!(research_fork, module)?)?;
    module.add_function(wrap_pyfunction!(research_context, module)?)?;
    module.add_function(wrap_pyfunction!(research_danger, module)?)?;
    module.add_function(wrap_pyfunction!(research_initial_counts, module)?)?;
    Ok(())
}
