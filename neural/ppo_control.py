"""Minibatch policy-drift checks and bounded baseline evaluation.

PPO clipping is not a hard trust region. The optional guard stops *before*
applying another update when the sampled reverse KL exceeds target_kl.
See https://spinningup.openai.com/en/latest/algorithms/ppo.html.
"""
from __future__ import annotations

from collections import deque
import math
import time

import torch
from torch import nn

#: What the guard raises on a drift that is not finite, at once or late.
NONFINITE_DRIFT = 'Nonfinite PPO policy drift; no optimizer step was applied'


def add_training_controls(parser) -> None:
    parser.add_argument('--target-kl', type=float, default=0.0,
                        help='stop further PPO updates above this sampled KL; 0 disables the guard')
    parser.add_argument('--baseline-batch', type=int, default=None,
                        help='maximum value-baseline rows per forward; defaults to --batch')


def baseline_batch_size(decisions: int, batch_size: int, requested: int | None = None) -> int:
    """Never pad a tiny rollout to thousands of dense 1,012-plane rows."""
    for name, value in [('decisions', decisions), ('batch', batch_size),
                        ('baseline_batch', requested if requested is not None else batch_size)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    return min(decisions, requested if requested is not None else batch_size)


class PolicyDrift:
    """One guard per collected round; stop all remaining epochs on excess drift.

    exp(delta)-1-delta is nonnegative and its expectation under the old policy
    is KL(old || new) when both policies have the same legal support. This is
    a sampled diagnostic, not exact full-policy KL or a rollback guarantee.
    All forwards and recorded probabilities must use the same precision.
    """
    def __init__(self, target_kl: float = 0.0) -> None:
        if not math.isfinite(target_kl) or target_kl < 0:
            raise ValueError('target_kl must be finite and nonnegative')
        self.target = target_kl
        self.stopped = False
        # Kept as tensors so that, with the guard off, nothing here waits
        # on the card: every other metric is read once at the round's end,
        # and a scalar synchronised on every minibatch would idle the card
        # through the next step's host work. The guard on has to read it
        # when the host decides whether the step is taken (`check`), and
        # need not when the card does (`verdict`).
        self._last: torch.Tensor | float = 0.0
        self._maximum: torch.Tensor | float = 0.0

    @torch.no_grad()
    def estimate(self, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> torch.Tensor:
        """The sampled divergence of a minibatch, where its probabilities are."""
        if old_log_prob.shape != new_log_prob.shape or old_log_prob.numel() == 0:
            raise ValueError('PPO probabilities need matching nonempty shapes')
        delta = new_log_prob.detach().float() - old_log_prob.detach().float()
        # expm1 avoids subtracting two near-equal numbers for small updates.
        return (torch.expm1(delta) - delta).clamp_min(0).mean()

    @torch.no_grad()
    def check(self, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> bool:
        estimate = self.estimate(old_log_prob, new_log_prob)
        if not self.target:
            self._last = estimate
            self._maximum = torch.maximum(torch.as_tensor(self._maximum, device=estimate.device), estimate)
            return False
        last = float(estimate)
        if not math.isfinite(last):
            raise FloatingPointError(NONFINITE_DRIFT)
        self._last = last
        self._maximum = max(float(self._maximum), last)
        self.stopped |= last > self.target
        return self.stopped

    @torch.no_grad()
    def verdict(
        self, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor, refused: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """`check` without waiting on the card (see `StepGuard`): flags on
        the card, whether the drift is not finite, which `check` raises on,
        and whether it is over the target, which stops the learning; None
        with the guard off, which refuses nothing. The figures are noted as
        `check` notes them, in the same arithmetic, but not for an update
        after one already `refused`, which deciding at once never reaches,
        nor for a drift that is not finite, which `check` raises on first.

        The figures stay on the card until the round's end. Where they are
        still numbers they start from on the card itself: making a tensor
        of a number on the host waits on the card to copy it."""
        estimate = self.estimate(old_log_prob, new_log_prob)
        last, maximum = (value if torch.is_tensor(value) else torch.full_like(estimate, value)
                         for value in (self._last, self._maximum))
        if not self.target:
            flags = None
            noted = ~refused
        else:
            bad = ~torch.isfinite(estimate)
            # Compared in double precision, as `check` compares the figure it
            # reads on the host: in single precision a target that rounds up
            # would let through a drift equal to its rounding.
            flags = bad, ~bad & (estimate.double() > self.target)
            noted = ~(refused | bad)
        self._last = torch.where(noted, estimate, last)
        self._maximum = torch.where(noted, torch.maximum(maximum, estimate), maximum)
        return flags

    @property
    def last(self) -> float:
        return float(self._last)

    @property
    def maximum(self) -> float:
        return float(self._maximum)

    def metrics(self) -> dict:
        return {'target_kl': self.target, 'kl_early_stop': self.stopped,
                'sampled_kl_last': self.last, 'sampled_kl_max': self.maximum}


class StepGuard:
    """Whether each of a generation's optimiser updates is taken: what the
    drift guard (`PolicyDrift`) says before it, and the gradient clip,
    which refuses a norm that is not finite.

    Decided on the host, each waits on the card once an update, the guard
    before the backward and the clip before the optimiser's step, and the
    card idles while the host launches what follows: on an L40S, a005's
    updates in mode none took 286.1 ms deciding on the host and 281.2
    deciding late, and in single runs of modes ours and mortal+head about
    4 ms fewer too. Fused AdamW can refuse an update on the card, as a
    gradient scaler has it do: given `found_inf` of one it leaves the
    weights and Adam's moments as they are and takes back the step it
    counted. So when it is `late`, and the optimiser is fused in every
    group, every update is launched without waiting; the card refuses the
    first the guard or the clip refuses and every one after it; and the
    host reads each update's verdict from page-locked memory `lag` updates
    later, by when the card is long done with it. At the first refused it
    stops the learning, or raises the error deciding at once raises, with
    nothing applied in between: the weights, Adam's moments and step
    counts, the sums of the record (`ppo_loop.Totals.add`'s `unless`), the
    drift's figures and the updates counted are bit for bit what deciding
    at once leaves, and each update taken is on the clock as of its launch.
    Every verdict is read before an epoch draws its order (`settle`), which
    leaves torch's generator, which the checkpoint keeps, where deciding at
    once leaves it.

    What the late verdicts cost is the work of the updates launched after
    the first refused, `lag` at most, before the learning stops: it is
    stopped rarely (the guard stopped one generation of a005's, 1100). An
    update refused that would have been the first a parameter ever took
    leaves Adam's state made for it, a step count and moments of nought,
    where deciding at once made none; that happens only in a generation
    that then takes no update at all, which stops the trainer before
    anything is saved.

    Otherwise, with an optimiser that cannot refuse an update on the card
    (not fused, as on the processor) or for a loop that reads something
    after every update on the host anyway (the entropy floor), each is
    decided at once, as it always was."""

    #: How many updates after its own the host reads an update's verdict.
    #: The host cannot run far ahead of the card, whose queue of launches
    #: fills within an update (five thousand-odd kernels in mode none), so
    #: an update two behind the one being launched is long done when read.
    LAG = 2

    def __init__(self, drift: PolicyDrift, optimiser: torch.optim.Optimizer,
                 device: str | torch.device, clock=None, late: bool = True, lag: int = LAG) -> None:
        self.drift = drift
        self.optimiser = optimiser
        #: Where each update taken is noted (see `ppo_loop.LearningClock`).
        self.clock = clock
        self.late = late and all(group.get('fused') for group in optimiser.param_groups)
        self.lag = lag
        #: The updates taken, as far as the host has read.
        self.updates = 0
        #: Whether the update just launched was refused, a flag on the card
        #: when decided late; None when decided at once.
        self.refused: torch.Tensor | None = None
        #: The picks of the minibatch the learning stopped on, when the guard
        #: stopped it late: the first update refused, which need not be the
        #: last launched.
        self.ended: torch.Tensor | None = None
        if not self.late:
            return
        device = torch.device(device)
        card = device.type == 'cuda'
        self._halted = torch.zeros((), dtype=torch.bool, device=device)
        self._clear = torch.zeros((), dtype=torch.bool, device=device)
        self._flags: tuple[torch.Tensor, torch.Tensor] | None = None
        self._norm: torch.Tensor | None = None
        self._stepped = 0
        self._pending: deque = deque()
        # Each update's verdict in a slot of its own until it is read: `lag`
        # are unread when an update is stepped, so with two more a slot is
        # never written before the update that last wrote it has been read.
        self._slots = [(torch.zeros(4, dtype=torch.bool, pin_memory=card),
                        torch.cuda.Event() if card else None) for _ in range(lag + 2)]

    def check(self, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> bool:
        """The drift guard on a minibatch before its update: whether the
        learning stops now. At once, `PolicyDrift.check`, which raises on a
        drift that is not finite; late, its verdict is left on the card for
        `step`, and nothing stops now."""
        if not self.late:
            return self.drift.check(old_log_prob, new_log_prob)
        self._flags = self.drift.verdict(old_log_prob, new_log_prob, self._halted)
        return False

    def clip(self, parameters, max_norm: float) -> torch.Tensor:
        """The gradients' norm, the gradients clipped to `max_norm` as
        `nn.utils.clip_grad_norm_` clips them. A norm that is not finite is
        the clip's error at once; late, the card's refusal of the update."""
        if not self.late:
            return nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        self._norm = nn.utils.clip_grad_norm_(parameters, max_norm)
        return self._norm

    def step(self) -> None:
        """The optimiser's update: taken at once; late, refused on the card
        where the guard or the clip refuses it or an update before it was
        refused, and its verdict copied off the card to be read later."""
        if not self.late:
            self.optimiser.step()
            return
        halted = self._halted
        bad, over = self._flags if self._flags is not None else (self._clear, self._clear)
        broken = ~torch.isfinite(self._norm) if self._norm is not None else self._clear
        refused = halted | bad | over | broken
        # One exactly, as the fused kernels compare it, for a refused update.
        self.optimiser.found_inf = refused.float()
        try:
            self.optimiser.step()
        finally:
            del self.optimiser.found_inf
        slot, ready = self._slots[self._stepped % len(self._slots)]
        slot.copy_(torch.stack((bad, over, broken, halted)), non_blocking=True)
        if ready is not None:
            ready.record()
        self._halted = self.refused = refused

    def launched(self, picks: torch.Tensor) -> None:
        """Notes the update just stepped, on the minibatch `picks` names:
        taken, when decided at once; late, its verdict to be read `lag`
        updates on, and the verdicts that far behind it read now (see
        `settle`)."""
        if not self.late:
            self.updates += 1
            if self.clock is not None:
                self.clock.update()
            return
        self._pending.append((self._stepped, picks, self._norm, time.time()))
        self._stepped += 1
        self._flags = self._norm = None
        self.settle(keep=self.lag)

    def settle(self, keep: int = 0) -> None:
        """Reads the verdicts not yet read, all but the last `keep`. An
        update taken is counted, and noted on the clock as of its launch,
        when deciding at once would have noted it. At the first refused the
        learning stops (`PolicyDrift.stopped`, `ended`), or the error the
        check deciding at once raises is raised; every update after it was
        refused as well, and none of their verdicts is read."""
        if not self.late:
            return
        while len(self._pending) > keep:
            stepped, picks, norm, at = self._pending.popleft()
            slot, ready = self._slots[stepped % len(self._slots)]
            if ready is not None:
                ready.synchronize()
            bad, over, broken, halted = slot.tolist()
            if not (bad or over or broken or halted):
                self.updates += 1
                if self.clock is not None:
                    self.clock.update(at)
                continue
            self._pending.clear()
            if bad:
                raise FloatingPointError(NONFINITE_DRIFT)
            if over:
                self.drift.stopped = True
                self.ended = picks
            elif broken:
                # The clip's own error, raised by the function that raises it
                # in the clip, from the norm the clip found.
                nn.utils.get_total_norm([norm], error_if_nonfinite=True)
                raise RuntimeError(f'the gradient norm {float(norm)} was refused as not finite')
            return
