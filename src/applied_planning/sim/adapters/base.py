from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

# Lightweight type aliases for now
Observation = Dict[str, Any]
Action = Dict[str, Any]
Goal = Dict[str, Any]


@dataclass
class CameraCfg:
    name: str
    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0


class SimulationAdapter(Protocol):
    """Backend-agnostic simulation interface (MuJoCo now, Isaac Sim later)."""

    def reset(self, seed: Optional[int] = None) -> Observation: ...

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]: ...

    def get_state(self) -> Dict[str, Any]: ...

    def set_state(self, state: Dict[str, Any]) -> None: ...

    def render(self, mode: str = "human"): ...

    def attach_camera(self, name: str, cfg: CameraCfg) -> None: ...

    def close(self) -> None: ...
