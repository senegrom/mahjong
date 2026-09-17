"""Independent, versioned experiment controls; no changes to legacy defaults."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    games: int = 32
    seed: int = 202617
    batch: int = 256
    epochs: int = 2
    critic_epochs: int = 2
    warmup_rounds: int = 2
    learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-4
    clip: float = .2
    entropy: float = .0005
    target_kl: float = .02
    anchor_weight: float = .1
    max_grad_norm: float = 1.
    objective: str = 'hybrid'
    rank_head: str = 'scalar'
    oracle: str = 'none'
    defence_weight: float = 0.
    channels: int = 32
    blocks: int = 2
    fixed: str = 'mortal'
    amp: bool = False
    opponent_share: float = .5
    league_uniform: float = .3
    adaptive_league: bool = False
    role: str = 'main'
    roots: int = 32
    uniform_roots: float = .25
    lesson_weight: float = 0.
    max_steps: int = 4000

    def validate(self):
        for name in ('games', 'batch', 'epochs', 'critic_epochs', 'channels', 'blocks', 'max_steps'):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'warmup_rounds', 'roots'):
            v = getattr(self, name)
            if type(v) is not int or v < 0: raise ValueError(f'invalid {name}')
        if self.channels % 8 or self.seed >= 1_000_000:
            raise ValueError('channels must divide by eight; training seed must be below 1,000,000')
        for name in ('learning_rate', 'critic_learning_rate', 'clip', 'entropy', 'target_kl',
                     'anchor_weight', 'max_grad_norm', 'defence_weight', 'opponent_share',
                     'league_uniform', 'uniform_roots', 'lesson_weight'):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f'invalid {name}')
        if min(self.learning_rate, self.critic_learning_rate, self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError('rates, gradient cap and target KL must be positive')
        if not 0 < self.clip < 1 or not 0 < self.league_uniform <= 1 or not 0 < self.uniform_roots <= 1 or self.opponent_share > 1:
            raise ValueError('invalid probability or clipping control')
        for name, choices in (('objective', ('hybrid', 'placement')), ('rank_head', ('scalar', 'categorical')),
                              ('oracle', ('none', 'hands', 'full')), ('role', ('main', 'exploiter')),
                              ('fixed', ('none', 'mortal', 'ours', 'head', 'mortal+head', 'ours+head'))):
            if getattr(self, name) not in choices: raise ValueError(f'invalid {name}')
        if any(type(getattr(self, name)) is not bool for name in ('amp', 'adaptive_league')):
            raise ValueError('expected boolean controls')
        return self

    def describe(self):
        return asdict(self)
