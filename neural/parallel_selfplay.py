"""Synchronous parallel arenas with a bounded, shared inference service.

Every collector finishes against ONE frozen learner version before PPO begins.
Private CPU sampling generators avoid races on Torch's global RNG. GPU inference
is owned by one service thread; no model copies or optimizer run in collectors.
Different collector counts intentionally define different sampling streams.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
import queue
import threading
import time
import numpy as np
import torch

from . import selfplay, mortal_learner, rl_policy, zoo
from .observe import Planes
from .outcomes import placements, validate_budget
from .population import table_matchups


@dataclass
class Request:
    model: int
    planes: torch.Tensor
    legal: torch.Tensor
    future: Future


class InferenceService:
    def __init__(self, models, *, device='cpu', amp=False, max_batch=512, capacity=8, flush_ms=2.):
        if (type(max_batch) is not int or max_batch < 1 or type(capacity) is not int or capacity < 1
                or not np.isfinite(flush_ms) or not 0 <= flush_ms <= 1000 or not models):
            raise ValueError('invalid inference service budget')
        self.models, self.device, self.amp = models, device, amp
        self.max_batch, self.flush = max_batch, flush_ms / 1000
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event(); self.error = None
        self.thread = threading.Thread(target=self._serve, name='mahjong-inference')
        self.metrics = {'inference_seconds': 0., 'inference_batches': 0, 'inference_rows': 0,
                        'max_inference_rows': 0, 'queue_peak': 0}

    def __enter__(self): self.thread.start(); return self

    def abort(self, error):
        self.error = error; self.stop.set()

    def __exit__(self, *exc):
        self.stop.set(); self.thread.join()
        self._drain()

    def _drain(self):
        while True:
            try: req = self.queue.get_nowait()
            except queue.Empty: return
            if not req.future.done(): req.future.set_exception(self.error or RuntimeError('inference service closed'))

    def ask(self, model, planes, legal):
        if planes.device.type != 'cpu' or legal.device.type != 'cpu' or len(planes) != len(legal):
            raise ValueError('collectors submit matching CPU rows')
        if model not in self.models or not len(planes): raise ValueError('invalid inference request')
        req = Request(model, planes.detach(), legal.detach(), Future())
        while not self.stop.is_set():
            try: self.queue.put(req, timeout=.1); break
            except queue.Full: pass
        else: raise RuntimeError('inference aborted') from self.error
        self.metrics['queue_peak'] = max(self.metrics['queue_peak'], self.queue.qsize())
        while True:
            try: return req.future.result(timeout=.1)
            except TimeoutError:
                if self.stop.is_set(): raise RuntimeError('inference aborted') from self.error

    @torch.no_grad()
    def _process(self, requests):
        # Separate policies and shapes; never pad one player's mask into another's.
        groups = {}
        for req in requests: groups.setdefault(req.model, []).append(req)
        for key, group in groups.items():
            pieces = []
            for req in group:
                for start in range(0, len(req.planes), self.max_batch):
                    pieces.append((req, start, min(start+self.max_batch, len(req.planes))))
            outputs = {id(req): torch.empty((len(req.legal), req.legal.shape[1]), dtype=torch.float32)
                       for req in group}
            while pieces:
                taking = []; count = 0
                while pieces and count + pieces[0][2] - pieces[0][1] <= self.max_batch:
                    part = pieces.pop(0); taking.append(part); count += part[2] - part[1]
                began = time.perf_counter()
                planes = torch.cat([r.planes[a:b] for r,a,b in taking]).to(self.device)
                legal = torch.cat([r.legal[a:b] for r,a,b in taking]).to(self.device)
                with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.amp):
                    logits = self.models[key](planes, legal)
                logits = logits.detach().float().cpu()
                if logits.shape != legal.shape or not torch.isfinite(logits[legal.cpu()]).all():
                    raise FloatingPointError('invalid inference output')
                offset = 0
                for req,a,b in taking:
                    outputs[id(req)][a:b] = logits[offset:offset+b-a]; offset += b-a
                self.metrics['inference_seconds'] += time.perf_counter() - began
                self.metrics['inference_batches'] += 1; self.metrics['inference_rows'] += count
                self.metrics['max_inference_rows'] = max(self.metrics['max_inference_rows'], count)
            for req in group: req.future.set_result(outputs[id(req)])

    def _serve(self):
        requests = []
        try:
            while not self.stop.is_set():
                try: first = self.queue.get(timeout=.1)
                except queue.Empty: continue
                requests = [first]; deadline = time.perf_counter() + self.flush
                while sum(len(r.planes) for r in requests) < self.max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0: break
                    try: requests.append(self.queue.get(timeout=remaining))
                    except queue.Empty: break
                self._process(requests); requests = []
        except BaseException as error:
            self.abort(error)
            for req in requests:
                if not req.future.done(): req.future.set_exception(error)
        finally: self._drain()


class RemoteActor:
    kind = 'mortal'
    actions = 46
    def __init__(self, service, generator, key=0):
        self.service, self.generator, self.key = service, generator, key
        self.timing = {'encode':0., 'network':0., 'translate':0.}
    def eval(self): return self
    def decide(self, views, rows, players, legal, greedy=False, **kwargs):
        return mortal_learner.decide_in_mortal_space(
            lambda p,m: self.service.ask(self.key,p,m), views, rows, players, legal,
            greedy=greedy, device='cpu', timing=self.timing, generator=self.generator, **kwargs)


class RemoteOpponent:
    def __init__(self, service, key, kind, actions):
        self.service, self.key, self.kind, self.actions = service, key, kind, actions
    def eval(self): return self
    def choose(self, views, rows, players, legal):
        if self.actions == 46:
            chosen, _ = mortal_learner.decide_in_mortal_space(
                lambda p,m:self.service.ask(self.key,p,m), views, rows, players, legal,
                greedy=True, device='cpu')
            return chosen
        scores = self.service.ask(self.key, views.dense(self.kind, rows, players, 'cpu'),
                                  torch.from_numpy(legal))
        return scores.argmax(1).numpy()


def merge(batches, seed, population=None):
    if not batches: raise ValueError('no completed collectors')
    if len({b.reward_version for b in batches}) != 1:
        raise ValueError('collector reward contracts differ')
    row_fields = ('legal','actions','held','oracle','imagined','returns','placements','log_probs',
                  'explored','behaviour_epsilon','after_exploration','player_of','hand_of','terminal','boundary')
    kwargs = {}
    for name in row_fields:
        values = [getattr(b,name) for b in batches]
        if all(v is None for v in values): kwargs[name] = None
        elif any(v is None for v in values): raise ValueError('collector field mismatch: '+name)
        else: kwargs[name] = torch.cat(values)
    games = decisions = 0; ids = []; links = []
    for b in batches:
        if b.seed != seed+games: raise ValueError('collector seed blocks must be contiguous and ordered')
        ids.append(b.game_of+games)
        links.append(torch.where(b.next_index >= 0, b.next_index+decisions, b.next_index))
        games += b.games; decisions += b.decisions
    scores = np.concatenate([b.final_scores for b in batches]); seats = np.concatenate([b.opponent_seats for b in batches])
    timings = {}
    for b in batches:
        for key,value in b.timing.items(): timings[key] = timings.get(key,0.) + value
    return selfplay.Batch(**kwargs, observations=Planes.cat([b.observations for b in batches]),
        games=games, hands=sum(b.hands for b in batches), decisions=decisions, final_scores=scores,
        game_of=torch.cat(ids), next_index=torch.cat(links), opponent_seats=seats, seed=seed, timing=timings,
        reward_version=batches[0].reward_version,
        matchups=table_matchups(seats, placements(scores), population.members) if population and population.members else [])


def play(net, *, games, seed, collectors=2, inference_batch=512, device='cpu', amp=False,
         opponents=None, population=None, table_mix=(1.,0.,0.), want_oracle=False, max_steps=4000):
    validate_budget(games,max_steps)
    if type(collectors) is not int or not 1 <= collectors <= games:
        raise ValueError('collectors must be between one and games')
    net.eval(); opponents = list(opponents or [])
    models = {0:lambda p,m:rl_policy.logits(net,p,m)}; specs = []
    for key, player in enumerate(opponents,1):
        player.eval(); base = getattr(player,'net',player)
        if isinstance(player,zoo.MortalPlayer):
            models[key] = lambda p,m,b=base:b(p,m)
            specs.append((key,'mortal',46))
        else:
            models[key] = lambda p,m,b=base:rl_policy.logits(b,p,m)
            specs.append((key,player.kind,base.actions))
    counts = [games//collectors + int(i < games%collectors) for i in range(collectors)]
    offsets = np.cumsum([0,*counts[:-1]]).tolist(); results = [None]*collectors
    began = time.perf_counter()
    with InferenceService(models,device=device,amp=amp,max_batch=inference_batch,capacity=collectors*2) as service:
        def collect(index):
            own_seed = seed+offsets[index]
            generator = torch.Generator().manual_seed(own_seed ^ 0xC011EC7)
            actor = RemoteActor(service,generator)
            others = [RemoteOpponent(service,*spec) for spec in specs]
            return selfplay.play(actor,games=counts[index],seed=own_seed,device='cpu',opponents=others,
                population=population,table_mix=tuple(table_mix),want_oracle=want_oracle,max_steps=max_steps)
        with ThreadPoolExecutor(max_workers=collectors,thread_name_prefix='mahjong-collector') as pool:
            futures = {pool.submit(collect,i):i for i in range(collectors)}
            try:
                for future in as_completed(futures): results[futures[future]] = future.result()
            except BaseException as error:
                service.abort(error)
                for future in futures: future.cancel()
                raise
        metrics = dict(service.metrics)
    result = merge(results,seed,population)
    result.timing.update(metrics)
    result.timing['parallel_wall_seconds'] = time.perf_counter()-began
    result.timing['collectors'] = collectors
    wall = result.timing['parallel_wall_seconds']
    result.timing['matches_per_second'] = games / wall
    result.timing['actor_choices_per_second'] = int((result.legal.sum(1) > 1).sum()) / wall
    return result
