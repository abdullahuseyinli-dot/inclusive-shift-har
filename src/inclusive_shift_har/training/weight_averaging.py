"""Source-validation-only EMA and SWAD checkpoint averaging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Clone a module state to CPU without retaining a graph."""

    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def average_state_dicts(states: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Average floating tensors and preserve identical non-floating buffers."""

    if not states:
        raise ValueError("at least one state is required for averaging")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("checkpoint states do not share an exact key order")
    result: dict[str, Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if tensors[0].is_floating_point():
            result[key] = (
                torch.stack([tensor.to(torch.float64) for tensor in tensors])
                .mean(0)
                .to(tensors[0].dtype)
            )
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating checkpoint buffer {key!r} differs across states")
            result[key] = tensors[0].clone()
    return result


class EMATracker:
    """Conventional exponential moving average over model states."""

    def __init__(self, decay: float = 0.99) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = decay
        self.state: dict[str, Tensor] | None = None
        self.update_count = 0

    def update(self, module: nn.Module) -> None:
        current = cpu_state_dict(module)
        if self.state is None:
            self.state = current
        else:
            for key, value in current.items():
                if value.is_floating_point():
                    self.state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
                else:
                    self.state[key] = value
        self.update_count += 1


class UniformAveragingTracker:
    """Online uniform model averaging used for a predeclared SWA interval."""

    def __init__(self) -> None:
        self.state: dict[str, Tensor] | None = None
        self.update_count = 0

    def update(self, module: nn.Module) -> None:
        current = cpu_state_dict(module)
        self.update_count += 1
        if self.state is None:
            self.state = current
            return
        for key, value in current.items():
            if value.is_floating_point():
                self.state[key].add_((value - self.state[key]) / self.update_count)
            else:
                self.state[key] = value


@dataclass(frozen=True)
class SWADResult:
    """Dense averaged state and its source-validation epoch interval."""

    state: dict[str, Tensor]
    start_epoch: int
    end_epoch: int
    state_count: int
    threshold: float


class SWADTracker:
    """Overfit-aware dense averaging following the SWAD validation-loss rule."""

    def __init__(
        self,
        *,
        optimum_patience: int = 3,
        overfit_patience: int = 6,
        tolerance_rate: float = 1.2,
    ) -> None:
        if min(optimum_patience, overfit_patience) < 2:
            raise ValueError("SWAD patience values must be at least two")
        if tolerance_rate <= 1:
            raise ValueError("SWAD tolerance rate must exceed one")
        self.optimum_patience = optimum_patience
        self.overfit_patience = overfit_patience
        self.tolerance_rate = tolerance_rate
        self._rolling: deque[tuple[int, float, dict[str, Tensor]]] = deque(
            maxlen=max(optimum_patience, overfit_patience)
        )
        self._candidates: list[tuple[int, float, dict[str, Tensor]]] = []
        self._threshold: float | None = None
        self._finished = False

    def update(self, epoch: int, validation_error: float, module: nn.Module) -> None:
        if epoch < 1 or not 0 <= validation_error <= 2:
            raise ValueError("SWAD epoch/error input is invalid")
        if self._finished:
            return
        state = cpu_state_dict(module)
        self._rolling.append((epoch, validation_error, state))
        if self._threshold is None and len(self._rolling) >= self.optimum_patience:
            window = list(self._rolling)[-self.optimum_patience :]
            errors = [item[1] for item in window]
            if errors[0] == min(errors):
                self._threshold = self.tolerance_rate * sum(errors) / len(errors)
                self._candidates.extend(window)
        elif self._threshold is not None:
            self._candidates.append((epoch, validation_error, state))
            if len(self._rolling) >= self.overfit_patience:
                recent = list(self._rolling)[-self.overfit_patience :]
                if min(item[1] for item in recent) > self._threshold:
                    excluded_epochs = {item[0] for item in recent}
                    self._candidates = [
                        item for item in self._candidates if item[0] not in excluded_epochs
                    ]
                    self._finished = True

    def result(self) -> SWADResult | None:
        if self._threshold is None or not self._candidates:
            return None
        return SWADResult(
            state=average_state_dicts([item[2] for item in self._candidates]),
            start_epoch=self._candidates[0][0],
            end_epoch=self._candidates[-1][0],
            state_count=len(self._candidates),
            threshold=self._threshold,
        )
