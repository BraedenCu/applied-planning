from __future__ import annotations

from typing import Any, Dict

# Stubs for future planning integrations (OMPL, TrajOpt, MPPI/MPC, etc.)


def plan_joint_path(start: Any, goal: Any, constraints: Dict[str, Any] | None = None):  # pragma: no cover
    """Placeholder for a joint-space planner."""
    raise NotImplementedError


def plan_cartesian_path(start: Any, goal: Any, constraints: Dict[str, Any] | None = None):  # pragma: no cover
    """Placeholder for a Cartesian-space planner."""
    raise NotImplementedError
