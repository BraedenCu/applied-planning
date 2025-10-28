from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import Action, Observation, SimulationAdapter


class IsaacSimAdapter(SimulationAdapter):
    """Placeholder adapter to be implemented when moving to Isaac Sim.

    Intended to mirror the SimulationAdapter contract so upstream code does not change.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "IsaacSimAdapter is a stub. Implement when moving to Isaac Sim."
        )

    def reset(self, seed: Optional[int] = None) -> Observation:  # pragma: no cover
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def set_state(self, state: Dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def render(self, mode: str = "human"):  # pragma: no cover
        raise NotImplementedError

    def attach_camera(self, name: str, cfg) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError
