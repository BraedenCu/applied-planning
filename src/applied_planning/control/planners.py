from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PathPlanner:
    """Base path planner for 6DOF robot arms."""

    def __init__(self, model, joint_limits: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize path planner.

        Args:
            model: MuJoCo model or similar robot model
            joint_limits: Dictionary mapping joint indices to (min, max) limits
        """
        self.model = model
        self.joint_limits = joint_limits or {}

    def is_valid_joint_state(self, q: np.ndarray) -> bool:
        """Check if joint configuration is within limits."""
        for idx, (qmin, qmax) in self.joint_limits.items():
            if not (qmin <= q[idx] <= qmax):
                return False
        return True


class RRTPlanner(PathPlanner):
    """RRT (Rapidly-exploring Random Tree) planner for joint-space planning."""

    def __init__(
        self,
        model,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        max_iterations: int = 5000,
        step_size: float = 0.1,
        goal_tolerance: float = 0.05
    ):
        super().__init__(model, joint_limits)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        collision_fn: Optional[callable] = None
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to goal using RRT.

        Args:
            start: Starting joint configuration
            goal: Goal joint configuration
            collision_fn: Optional function to check collisions, takes joint config

        Returns:
            List of waypoints from start to goal, or None if no path found
        """
        nodes = [start]
        parent = {0: None}

        for i in range(self.max_iterations):
            # Sample random configuration or goal
            if np.random.rand() < 0.1:
                q_rand = goal
            else:
                q_rand = self._sample_random_config()

            # Find nearest node
            nearest_idx = self._nearest_node(nodes, q_rand)
            q_near = nodes[nearest_idx]

            # Steer towards random config
            q_new = self._steer(q_near, q_rand)

            # Check if valid and collision-free
            if self.is_valid_joint_state(q_new):
                if collision_fn is None or not collision_fn(q_new):
                    new_idx = len(nodes)
                    nodes.append(q_new)
                    parent[new_idx] = nearest_idx

                    # Check if we reached the goal
                    if np.linalg.norm(q_new - goal) < self.goal_tolerance:
                        return self._extract_path(nodes, parent, new_idx)

        return None  # No path found

    def _sample_random_config(self) -> np.ndarray:
        """Sample a random valid joint configuration."""
        q = np.random.uniform(-np.pi, np.pi, len(self.joint_limits) or 6)
        for idx, (qmin, qmax) in self.joint_limits.items():
            q[idx] = np.random.uniform(qmin, qmax)
        return q

    def _nearest_node(self, nodes: List[np.ndarray], q: np.ndarray) -> int:
        """Find index of nearest node to q."""
        distances = [np.linalg.norm(node - q) for node in nodes]
        return int(np.argmin(distances))

    def _steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Steer from q_near towards q_target by step_size."""
        direction = q_target - q_near
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return q_target
        return q_near + (direction / distance) * self.step_size

    def _extract_path(
        self,
        nodes: List[np.ndarray],
        parent: Dict[int, Optional[int]],
        goal_idx: int
    ) -> List[np.ndarray]:
        """Extract path from start to goal by backtracking."""
        path = []
        current = goal_idx
        while current is not None:
            path.append(nodes[current])
            current = parent[current]
        return list(reversed(path))


class LinearInterpolationPlanner(PathPlanner):
    """Simple linear interpolation planner for quick testing."""

    def __init__(
        self,
        model,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        num_waypoints: int = 50
    ):
        super().__init__(model, joint_limits)
        self.num_waypoints = num_waypoints

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        collision_fn: Optional[callable] = None
    ) -> Optional[List[np.ndarray]]:
        """Plan a straight-line path from start to goal.

        Args:
            start: Starting joint configuration
            goal: Goal joint configuration
            collision_fn: Optional function to check collisions

        Returns:
            List of waypoints from start to goal, or None if invalid
        """
        path = []
        for t in np.linspace(0, 1, self.num_waypoints):
            q = (1 - t) * start + t * goal

            if not self.is_valid_joint_state(q):
                return None

            if collision_fn is not None and collision_fn(q):
                return None

            path.append(q)

        return path


def plan_joint_path(
    start: np.ndarray,
    goal: np.ndarray,
    model: Any,
    constraints: Dict[str, Any] | None = None,
    planner_type: str = "rrt"
) -> Optional[List[np.ndarray]]:
    """Plan a path in joint space from start to goal.

    Args:
        start: Starting joint configuration (6DOF)
        goal: Goal joint configuration (6DOF)
        model: MuJoCo model or similar robot model
        constraints: Optional constraints including joint_limits, collision_fn
        planner_type: Type of planner to use ("rrt" or "linear")

    Returns:
        List of waypoints from start to goal, or None if no path found
    """
    constraints = constraints or {}
    joint_limits = constraints.get("joint_limits")
    collision_fn = constraints.get("collision_fn")

    if planner_type == "linear":
        planner = LinearInterpolationPlanner(model, joint_limits)
    else:
        planner = RRTPlanner(model, joint_limits)

    return planner.plan(start, goal, collision_fn)


def plan_cartesian_path(
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    model: Any,
    constraints: Dict[str, Any] | None = None
) -> Optional[List[np.ndarray]]:
    """Plan a path in Cartesian space from start to goal.

    Args:
        start_pose: Starting end-effector pose [x, y, z, qw, qx, qy, qz]
        goal_pose: Goal end-effector pose
        model: MuJoCo model with forward/inverse kinematics
        constraints: Optional constraints including IK solver, joint_limits

    Returns:
        List of joint waypoints that follow Cartesian path, or None if infeasible
    """
    constraints = constraints or {}
    ik_solver = constraints.get("ik_solver")

    if ik_solver is None:
        raise NotImplementedError(
            "Cartesian planning requires an IK solver in constraints['ik_solver']"
        )

    # Interpolate poses in Cartesian space
    num_waypoints = constraints.get("num_waypoints", 50)
    cartesian_waypoints = []
    for t in np.linspace(0, 1, num_waypoints):
        pose = (1 - t) * start_pose + t * goal_pose
        cartesian_waypoints.append(pose)

    # Convert each Cartesian waypoint to joint space
    joint_waypoints = []
    for pose in cartesian_waypoints:
        q = ik_solver(pose)
        if q is None:
            return None  # IK failed
        joint_waypoints.append(q)

    # Optionally smooth or validate the joint path
    return joint_waypoints
