"""Independent training critics; no hidden input or critic gradient reaches actor.

The public critic sees only observation planes. The optional privileged critic
adds the simulator's PRE-ACTION oracle planes. Both fit the selected actor
objective, not the deployed model's legacy hybrid value head. The old value
head can therefore keep its documented units and search contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class CriticConfig:
    planes: int
    hidden_planes: int = 0
    width: int = 64
    blocks: int = 2

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.planes, self.hidden_planes, self.width, self.blocks))
                or self.planes <= 0 or self.hidden_planes < 0 or self.width < 8
                or self.width % 8 or self.blocks < 1):
            raise ValueError("invalid training critic dimensions")


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, width, 3, padding=1),
                                  nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, width, 3, padding=1))

    def forward(self, planes):
        return planes + self.body(planes)


class TrainingCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Conv1d(config.planes + config.hidden_planes, config.width, 1)
        self.tower = nn.Sequential(*(Residual(config.width) for _ in range(config.blocks)))
        self.value = nn.Sequential(nn.Linear(2 * config.width, config.width), nn.SiLU(),
                                   nn.Linear(config.width, 1))
        # A fresh objective starts at zero, not the old hybrid critic's answer.
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, observation, hidden=None):
        if observation.ndim != 3 or observation.shape[1:] != (self.config.planes, 34):
            raise ValueError("critic observation layout mismatch")
        if self.config.hidden_planes:
            if hidden is None or hidden.shape != (len(observation), self.config.hidden_planes, 34):
                raise ValueError("privileged critic requires pre-action oracle planes")
            observation = torch.cat((observation.detach(), hidden.detach()), dim=1)
        elif hidden is not None:
            raise ValueError("public critic must not receive hidden state")
        features = self.tower(self.stem(observation.detach().float()))
        pooled = torch.cat((features.mean(dim=2), features.amax(dim=2)), dim=1)
        return self.value(pooled).squeeze(1)

    def payload(self):
        return {"config": asdict(self.config), "state": self.state_dict()}

    @classmethod
    def from_payload(cls, payload, device):
        net = cls(CriticConfig(**payload["config"]))
        net.load_state_dict(payload["state"])
        return net.to(device)


@torch.no_grad()
def predict(critic, observations, hidden, batch_size: int, device: str) -> torch.Tensor:
    critic.eval()
    result = torch.empty(len(observations), dtype=torch.float32)
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        planes = observations.slice(start, stop).dense(device)
        secret = hidden[start:stop].to(device).float() if critic.config.hidden_planes else None
        result[start:stop] = critic(planes, secret).float().cpu()
    return result


def fit(critic, optimizer, observations, hidden, targets, *, epochs, batch_size,
        clip_norm, device) -> dict:
    """Current-rollout fitting only; do not treat replayed old-policy returns as current."""
    critic.train()
    total, count, updates, norm = 0.0, 0, 0, 0.0
    for _epoch in range(epochs):
        order = torch.randperm(len(targets))
        for start in range(0, len(order), batch_size):
            picks = order[start:start + batch_size]
            planes = observations.rows(picks.numpy()).dense(device)
            secret = hidden[picks].to(device).float() if critic.config.hidden_planes else None
            optimizer.zero_grad(set_to_none=True)
            wanted = targets[picks].to(device)
            value = critic(planes, secret)
            loss = nn.functional.mse_loss(value, wanted)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(critic.parameters(), clip_norm, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(picks)
            count += len(picks)
            norm += float(grad)
            updates += 1
    return {"loss": total / max(1, count), "updates": updates,
            "gradient_norm": norm / max(1, updates)}
