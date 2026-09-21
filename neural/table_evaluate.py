"""Fixed heterogeneous and two-and-two Mahjong evaluation panels.

Policy 0 is the candidate. Every seat optimizes its own placement; policy
copies are NOT teams. Repeat each lineup in four rotations and cluster on
original deal identity. Reports and chance corrections never promote models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import riichi_py

from . import zoo
from .observe import Views
from .outcomes import placements, validate_budget, require_finished
from .research_state import context, require_research_engine
from .league_state import snapshot
from .seed_ledger import SeedLedger, atomic_json
from .chance_control import initial_features
from .training_safety import TRAINING_API_VERSION


class Diagnostics:
    """Event counts plus game-level late-match cohorts. No synthetic rewards."""
    def __init__(self,games):
        self.riichi=np.zeros((games,4),bool)
        self.dealins=np.zeros((games,4),np.int64);self.pressure_dealins=np.zeros_like(self.dealins)
        self.discards=np.zeros_like(self.dealins);self.pressure_discards=np.zeros_like(self.dealins)
        self.calls=np.zeros_like(self.dealins);self.call_choices=np.zeros_like(self.dealins)
        self.late_leading=np.zeros((games,4),bool);self.late_trailing=np.zeros((games,4),bool)
        self.late_seen=np.zeros(games,bool);self.paid=[set() for _ in range(games)]
        self.last_pressure=np.zeros((games,4),bool)

    def feed(self,events):
        for game,lines in enumerate(events):
            for line in lines:
                e=json.loads(line);kind=e['type']
                if kind=='start_kyoku':
                    self.riichi[game]=False;self.paid[game].clear();self.last_pressure[game]=False
                elif kind=='reach_accepted':self.riichi[game,e['actor']]=True
                elif kind=='dahai':
                    player=e['actor'];self.discards[game,player]+=1
                    self.last_pressure[game,player] = self.riichi[game].sum()-self.riichi[game,player]>0
                    if self.last_pressure[game,player]:self.pressure_discards[game,player]+=1
                elif kind=='hora' and e['actor']!=e['target']:
                    target=e['target']
                    if target not in self.paid[game]:
                        self.dealins[game,target]+=1;self.paid[game].add(target)
                    # Count at most one pressure deal-in per hand/discarder.
                    tag=('pressure',target)
                    if self.last_pressure[game,target] and tag not in self.paid[game]:
                        self.pressure_dealins[game,target]+=1;self.paid[game].add(tag)

    def before(self,public,rows,players,masks,actions):
        for game in np.flatnonzero((public[:,0]>=1)&(public[:,1]>=3)&~self.late_seen):
            scores=public[game,8:12]
            self.late_leading[game]=scores==scores.max();self.late_trailing[game]=scores==scores.min()
            self.late_seen[game]=True
        for game,player in zip(rows,players):
            if masks[game,70] and masks[game,71:76].any():
                self.call_choices[game,player]+=1
                self.calls[game,player]+=int(71<=actions[game]<=75)

    def report(self,candidate_players,final_places):
        pick=np.asarray(candidate_players,np.int64)
        result={name:int(getattr(self,name)[:,pick].sum()) for name in
                ('dealins','pressure_dealins','discards','pressure_discards','calls','call_choices')}
        for name in ('late_leading','late_trailing'):
            selected=getattr(self,name)[:,pick]
            result[name]={'player_games':int(selected.sum()),
                          'placement_sum':float(final_places[:,pick][selected].sum())}
        return result


@torch.no_grad()
def table(players,lineup,*,games,seed,device='cpu',max_steps=4000,want_controls=False):
    require_research_engine();validate_budget(games,max_steps)
    if len(lineup)!=4 or any(type(x) is not int or not 0<=x<len(players) for x in lineup) or 0 not in lineup:
        raise ValueError('lineup must name four policies and include candidate 0')
    arena=riichi_py.Arena(games=games,seed=seed)
    views=Views(arena,games,{p.kind for p in players})
    features=initial_features(arena) if want_controls else None
    diag=Diagnostics(games);steps=0
    for player in players:player.eval()
    def advance():
        events=arena.mjai_all();diag.feed(events)
        if views.observer is not None:views.observer.follower.feed(events)
        views._engine=None;views._step=None
    while not arena.all_finished() and steps<max_steps:
        advance()
        seats=np.frombuffer(arena.seats(),np.uint8);rows=np.flatnonzero(seats!=255)
        if not len(rows):break
        masks=np.frombuffer(arena.legal_mask(),np.uint8).reshape(games,78).astype(bool)
        owners=np.frombuffer(arena.seat_players(),np.uint8).reshape(games,4)[rows,seats[rows]]
        views.prepare(rows,owners)
        assigned=np.asarray(lineup)[owners];actions=np.zeros(games,np.int64)
        for policy in np.unique(assigned):
            select=assigned==policy
            actions[rows[select]]=zoo.choose(players[int(policy)],views,rows[select],owners[select],masks[rows[select]],device)
        diag.before(context(arena),rows,owners,masks,actions)
        arena.step(actions.tolist());steps+=1
    require_finished(arena,steps=steps,context='heterogeneous evaluation');advance()
    scores=np.frombuffer(arena.final_scores(),np.int32).reshape(games,4).copy()
    ranks=placements(scores);mine=[i for i,p in enumerate(lineup) if p==0]
    return {'scores':scores,'utility':(2.5-ranks[:,mine]).mean(1),'diagnostics':diag.report(mine,ranks),'features':features}


def combine_diagnostics(reports):
    sums={k:sum(r[k] for r in reports) for k in ('dealins','pressure_dealins','discards','pressure_discards','calls','call_choices')}
    sums['dealins_per_discard']=sums['dealins']/sums['discards'] if sums['discards'] else None
    sums['pressure_dealins_per_discard']=sums['pressure_dealins']/sums['pressure_discards'] if sums['pressure_discards'] else None
    sums['call_rate']=sums['calls']/sums['call_choices'] if sums['call_choices'] else None
    for key in ('late_leading','late_trailing'):
        count=sum(r[key]['player_games'] for r in reports);total=sum(r[key]['placement_sum'] for r in reports)
        sums[key]={'player_games':count,'mean_placement':total/count if count else None}
    return sums


def evaluate(candidate,opponents,*,ledger,games=512,lineups=None,domain='validation',device='cpu',
             chance_model=None,want_controls=False):
    if domain not in ('validation','test') or type(games) is not int or games<2 or not opponents:
        raise ValueError('a held-out domain, opponents and at least two deals are required')
    if lineups is None:
        # Fixed outside-opponent mixtures plus non-cooperative two-and-two.
        lineups=[[0,1,2,3],[0,0,1,1]] if len(opponents)>=3 else [[0,1,1,1],[0,0,1,1]]
    if not lineups or any(len(row)!=4 or 0 not in row or any(type(i) is not int or not 0<=i<=len(opponents) for i in row) for row in lineups):
        raise ValueError('invalid policy lineup')
    control=json.loads(Path(chance_model).read_text()) if chance_model else None
    want_controls=want_controls or control is not None
    with tempfile.TemporaryDirectory(prefix='table-panel-') as folder:
        folder=Path(folder);entries=[snapshot(Path(p),folder) for p in [candidate,*opponents]]
        players=[zoo.load_player(folder/e['file'],device) for e in entries]
        reservation=SeedLedger(Path(ledger)).reserve(domain,games,'heterogeneous-table-panel')
        report={'table_panel_version':1,'policies':entries,'seed_reservation':reservation,'domain':domain,
                'panel':[],'diagnostic_only':True,'promotion':False,
                'training_api_version': TRAINING_API_VERSION, 'research_api_version': 1,
                'rotations': 4, 'independent_unit': 'initial-deal-seed'}
        began=time.perf_counter()
        for lineup in lineups:
            runs=[table(players,np.roll(lineup,rotation).tolist(),games=games,seed=reservation['seed'],
                        device=device,want_controls=want_controls) for rotation in range(4)]
            by_deal=np.stack([r['utility'] for r in runs]).mean(0)
            mine=np.flatnonzero(np.asarray(lineup)==0)
            # Tied ranks use fractional occupancy of each occupied rank for histograms.
            hist=np.zeros(4)
            for rotation,run in enumerate(runs):
                places=placements(run['scores'])
                candidates=(mine+rotation)%4
                for g in range(games):
                    for person in candidates:
                        same=run['scores'][g]==run['scores'][g,person]
                        better=int((run['scores'][g]>run['scores'][g,person]).sum());ties=int(same.sum())
                        hist[better:better+ties]+=1/ties
            row={'lineup':lineup,'independent_deals':games,'games_total':4*games,
                 'mean_placement':2.5-float(by_deal.mean()),'mean_utility':float(by_deal.mean()),
                 'standard_error':float(by_deal.std(ddof=1)/np.sqrt(games)),
                 'by_deal_utility':by_deal.tolist(),'placement_distribution':(hist/hist.sum()).tolist(),
                 'diagnostics':combine_diagnostics([r['diagnostics'] for r in runs])}
            if want_controls:row['control_features']=runs[0]['features'].tolist()
            report['panel'].append(row)
        report['seconds']=time.perf_counter()-began
    if control is not None:
        from .chance_control import apply
        report['chance_adjustment']=apply(report,control)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path)
    p.add_argument('--opponents',type=Path,nargs='+',required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--games',type=int,default=512)
    p.add_argument('--lineups',nargs='+',help='policy indices, e.g. 0,1,2,3 0,0,1,1')
    p.add_argument('--domain',choices=('validation','test'),default='validation')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--chance-model',type=Path)
    p.add_argument('--want-controls',action='store_true')
    args=vars(p.parse_args());out=args.pop('out')
    if out.exists():raise FileExistsError(out)
    if args['lineups'] is not None:args['lineups']=[list(map(int,row.split(','))) for row in args['lineups']]
    torch.set_num_threads(2);report=evaluate(**args);atomic_json(out,report)
    print(json.dumps({'out':str(out),'panels':len(report['panel']),'promotion':False}))


if __name__=='__main__':main()
