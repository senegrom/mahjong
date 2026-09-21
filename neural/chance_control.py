"""Experimental chance-only control variates, NEVER promotion evidence.

Inspired by AIVAT (Burch et al., arXiv:1612.06915), but NOT full AIVAT:
no information-set aggregation or player-action correction is claimed here.
A frozen pilot regression uses initial 13-tile-hand moments with exact means
under the engine's uniform 136-tile shuffle. Test outcomes never fit coefficients.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .seed_ledger import atomic_json

VERSION = 1


def finite_correction(probabilities, estimates, outcome):
    """E[v(chance outcome)] - v(observed outcome), for a KNOWN finite law."""
    p, v = np.asarray(probabilities, np.float64), np.asarray(estimates, np.float64)
    if (p.ndim != 1 or not len(p) or p.shape != v.shape or not np.isfinite(p).all()
            or not np.isfinite(v).all() or (p < 0).any() or not np.isclose(p.sum(),1.,atol=1e-12,rtol=0)
            or type(outcome) is not int or not 0 <= outcome < len(p) or p[outcome] <= 0):
        raise ValueError('a known normalized chance law and possible outcome are required')
    return float(p @ v - v[outcome])


def pair_expectation(hand_size, copies, kinds):
    """Expected sum C(count[tile],2) for sampling without replacement."""
    if min(hand_size,copies,kinds) < 1 or hand_size > copies*kinds: raise ValueError('invalid deck')
    if hand_size < 2 or copies*kinds < 2: return 0.
    return math.comb(hand_size,2) * kinds * math.comb(copies,2) / math.comb(copies*kinds,2)


def features(counts):
    """Zero-mean pre-play chance features, by original player, never acting inputs.

    136 tile-count deviations plus four equal-tile-pair deviations. The exact
    mean does not use empirical test means or assumptions about the policies.
    """
    c = np.asarray(counts)
    if (c.ndim != 3 or c.shape[1:] != (4,34) or c.dtype.kind not in 'iu'
            or np.any((c < 0)|(c > 4)) or not np.all(c.sum(2)==13) or (c.sum(1)>4).any()):
        raise ValueError('control features require a physically valid initial deal, 13 tiles per player')
    c = c.astype(np.float64)
    pairs = (c*(c-1)/2).sum(2) - pair_expectation(13,4,34)
    return np.concatenate([(c-13/34).reshape(len(c),-1),pairs],axis=1)


def initial_features(arena):
    import riichi_py
    from .research_state import require_research_engine
    require_research_engine()
    raw = riichi_py.research_initial_counts(arena)
    c = np.asarray([[np.frombuffer(hand,np.uint8) if isinstance(hand,bytes) else hand
                     for hand in game] for game in raw],dtype=np.int64)
    return features(c)


def identity(report):
    return {'policies':[p['sha256'] for p in report['policies']],
            'lineups':[p['lineup'] for p in report['panel']], 'feature_version':VERSION,
            'rotations':report.get('rotations', 4), 'objective':'mean-candidate-placement-utility',
            'training_api_version':report.get('training_api_version'),
            'research_api_version':report.get('research_api_version')}


def fit(pilot_path, out, *, ridge=1.0):
    report = json.loads(Path(pilot_path).read_text())
    if report.get('domain') != 'validation' or report.get('table_panel_version') != 1:
        raise ValueError('control coefficients require a separate validation pilot panel')
    if not np.isfinite(ridge) or ridge <= 0: raise ValueError('ridge must be finite and positive')
    if Path(out).exists(): raise FileExistsError(out)
    coefficients=[]
    for row in report['panel']:
        x = np.asarray(row['control_features'],np.float64); y=np.asarray(row['by_deal_utility'],np.float64)
        if (x.shape != (len(y),140) or len(y)<2 or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('pilot lacks finite chance features')
        xc=x-x.mean(0); yc=y-y.mean()
        beta=np.linalg.solve(xc.T@xc+ridge*np.eye(x.shape[1]),xc.T@yc)
        coefficients.append(beta.tolist())
    saved={'chance_control_version':VERSION,'protocol':identity(report),'coefficients':coefficients,
           'pilot_reservation':report['seed_reservation'],'ridge':ridge,
           'pilot_sha256':hashlib.sha256(Path(pilot_path).read_bytes()).hexdigest(),
           'diagnostic_only':True,'method':'frozen initial-deal control variate; not full AIVAT'}
    atomic_json(Path(out),saved); return saved


def apply(report, saved):
    if saved.get('chance_control_version')!=VERSION or saved['protocol']!=identity(report):
        raise ValueError('control variate belongs to a different policy/table panel')
    a,b=saved['pilot_reservation'],report['seed_reservation']
    if max(a['seed'],b['seed']) < min(a['end'],b['end']):
        raise ValueError('control fitting and evaluation deals overlap')
    if len(saved['coefficients'])!=len(report['panel']): raise ValueError('control panel mismatch')
    output=[]
    bounds=np.array([4.]*(4*34)+[78.] * 4) # Conservative absolute centered-feature limits.
    for row,coefficients in zip(report['panel'],saved['coefficients']):
        beta=np.asarray(coefficients,np.float64); x=np.asarray(row['control_features'],np.float64)
        raw=np.asarray(row['by_deal_utility'],np.float64)
        if beta.shape!=(140,) or not np.isfinite(beta).all() or x.shape!=(len(raw),140):
            raise ValueError('invalid control coefficient shape')
        correction=-(x@beta); adjusted=raw+correction
        if not np.isfinite(adjusted).all(): raise ValueError('nonfinite adjusted results')
        n=len(raw)
        var=float(adjusted.var(ddof=1)) if n>1 else None
        rawvar=float(raw.var(ddof=1)) if n>1 else None
        output.append({'raw_mean':float(raw.mean()),'adjusted_mean':float(adjusted.mean()),
            'raw_standard_error':(rawvar/n)**.5 if n>1 else None,
            'adjusted_standard_error':(var/n)**.5 if n>1 else None,
            'variance_ratio':var/rawvar if rawvar and n>1 else None,
            'mean_correction':float(correction.mean()),'by_deal_adjusted':adjusted.tolist(),
            'conservative_range':[-1.5-float(np.abs(beta)@bounds),1.5+float(np.abs(beta)@bounds)]})
    return {'method':saved['method'],'pilot_sha256':saved['pilot_sha256'],'panels':output,
            'diagnostic_only':True,'promotion':False,
            'warning':'Adjusted outcomes do not have raw placement bounds; do not feed to the promotion gate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pilot_path',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ridge',type=float,default=1.)
    print(json.dumps(fit(**vars(p.parse_args())),allow_nan=False))


if __name__=='__main__':main()
