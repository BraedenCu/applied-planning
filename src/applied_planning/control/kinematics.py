"""Kinematics utilities for 6DOF robot arms."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import mujoco


class MujocoIKSolver:
    """Inverse kinematics solver using MuJoCo's built-in IK capabilities."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "attachment_site",
        max_iterations: int = 500,
        tolerance: float = 1e-3,
        damping: float = 1e-3,
        step_size: float = 0.5
    ):
        """Initialize IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            ee_site_name: Name of end-effector site in the model
            max_iterations: Maximum number of IK iterations
            tolerance: Position/orientation error tolerance (meters/radians)
            damping: Damping factor for numerical stability
            step_size: Step size multiplier for joint updates (0-1)
        """
        self.model = model
        self.data = data
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size

        # Find end-effector site
        try:
            self.ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
            )
            self.use_body = False
            print(f"IK solver using site: {ee_site_name}")
        except Exception:
            # Fallback: try to find by body name
            try:
                self.ee_site_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name
                )
                self.use_body = True
                print(f"IK solver using body: {ee_site_name}")
            except Exception:
                # Last resort: use the last body
                self.ee_site_id = model.nbody - 1
                self.use_body = True
                print(f"Warning: Could not find '{ee_site_name}', using last body (id={self.ee_site_id})")

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        joint_indices: Optional[list] = None
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Optional target quaternion [w, x, y, z]. If None, only position IK
            q_init: Initial joint configuration. If None, uses current state
            joint_indices: Indices of joints to control. If None, uses first 6 DoF

        Returns:
            Joint configuration that achieves target pose, or None if IK failed
        """
        if joint_indices is None:
            joint_indices = list(range(min(6, self.model.nv)))

        # Initialize joint configuration
        if q_init is not None:
            self.data.qpos[:len(q_init)] = q_init
        else:
            q_init = self.data.qpos.copy()

        # MuJoCo IK using Jacobian pseudoinverse
        for iteration in range(self.max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Get current end-effector pose
            if self.use_body:
                current_pos = self.data.xpos[self.ee_site_id].copy()
                current_quat = self.data.xquat[self.ee_site_id].copy()
            else:
                current_pos = self.data.site_xpos[self.ee_site_id].copy()
                current_quat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
                current_quat = self._mat_to_quat(current_quat)

            # Compute position error
            pos_error = target_pos - current_pos

            # Compute orientation error if target quaternion provided
            if target_quat is not None:
                # Quaternion error using MuJoCo's convention [w, x, y, z]
                ori_error = self._quat_error(current_quat, target_quat)
                error = np.concatenate([pos_error, ori_error])
            else:
                error = pos_error

            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                return self.data.qpos[:len(joint_indices)].copy()

            # Compute Jacobian
            if self.use_body:
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_site_id)
            else:
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

            # Select only the joints we're controlling
            jacp = jacp[:, joint_indices]
            jacr = jacr[:, joint_indices]

            if target_quat is not None:
                jac = np.vstack([jacp, jacr])
            else:
                jac = jacp

            # Compute pseudoinverse with damping
            jac_T = jac.T
            damped_inv = jac_T @ np.linalg.inv(
                jac @ jac_T + self.damping * np.eye(jac.shape[0])
            )

            # Update joint positions with step size
            dq = damped_inv @ error
            for i, joint_idx in enumerate(joint_indices):
                self.data.qpos[joint_idx] += self.step_size * dq[i]

        # IK did not converge
        print(f"  IK failed to converge after {self.max_iterations} iterations")
        print(f"  Final error: {np.linalg.norm(error):.6f} (tolerance: {self.tolerance})")
        return None

    def solve_from_pose(
        self,
        target_pose: np.ndarray,
        q_init: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Convenience method to solve IK from 7D pose vector.

        Args:
            target_pose: [x, y, z, qw, qx, qy, qz] pose vector
            q_init: Initial joint configuration

        Returns:
            Joint configuration or None if IK failed
        """
        target_pos = target_pose[:3]
        target_quat = target_pose[3:]  # [qw, qx, qy, qz]
        return self.solve(target_pos, target_quat, q_init)

    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    @staticmethod
    def _quat_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Compute orientation error between quaternions.

        Args:
            q_current: Current quaternion [w, x, y, z]
            q_target: Target quaternion [w, x, y, z]

        Returns:
            3D orientation error vector
        """
        # Compute difference quaternion: q_diff = q_target * q_current^-1
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        q_diff = MujocoIKSolver._quat_multiply(q_target, q_current_conj)

        # Convert to axis-angle (scaled axis)
        # For small angles: error ≈ 2 * [x, y, z]
        return 2.0 * q_diff[1:]

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


def compute_forward_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    ee_site_name: str = "attachment_site"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward kinematics for given joint configuration.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        q: Joint configuration
        ee_site_name: Name of end-effector site (default: "attachment_site" for Lite6)

    Returns:
        Tuple of (position, quaternion) where quaternion is [w, x, y, z]
    """
    # Set joint configuration
    data.qpos[:len(q)] = q
    mujoco.mj_forward(model, data)

    # Get end-effector pose
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        pos = data.site_xpos[site_id].copy()
        mat = data.site_xmat[site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
    except Exception:
        # Fallback to body
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
        except Exception:
            body_id = model.nbody - 1
        pos = data.xpos[body_id].copy()
        quat = data.xquat[body_id].copy()

    return pos, quat
