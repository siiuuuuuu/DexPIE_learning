"""Small task-space MPC tracker for high-rate UR servo commands."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskSpaceMPCConfig:
    horizon: int = 15
    tau: float = 0.18
    iterations: int = 24
    w_track: float = 10.0
    track_decay: float = 1.0
    w_cmd: float = 0.0
    w_yx: float = 0.0
    w_dy: float = 0.0
    w_ddy: float = 20.0
    max_cmd_actual_gap: float = 0.0
    max_velocity: float = 0.5
    max_acceleration: float = 6.0


@dataclass
class TaskSpaceMPCResult:
    command: np.ndarray
    q_pred: np.ndarray
    status: int
    iterations: int
    cost: float
    reference_gap: float
    raw_command_gap: float
    command_gap: float
    tracking_error: float


class TaskSpaceMPCTracker:
    """Plans a constrained xyz servo command under first-order arm lag."""

    def __init__(self, config=None, dt=1.0 / 60.0, workspace_limits=None):
        self.config = config or TaskSpaceMPCConfig()
        self.dt = float(dt)
        self.workspace_limits = workspace_limits or {}
        self._prev_cmd = None
        self._prev_prev_cmd = None
        self._cached_shape = None
        self._cached = None

    def reset(self):
        self._prev_cmd = None
        self._prev_prev_cmd = None

    def plan(self, references, actual_position):
        refs = np.asarray(references, dtype=np.float64)
        if refs.ndim != 2 or refs.shape[1] != 3:
            raise ValueError("references must have shape (N, 3)")
        if refs.shape[0] <= 0:
            raise ValueError("references must not be empty")

        cfg = self.config
        horizon = max(1, int(cfg.horizon))
        if refs.shape[0] < horizon:
            pad = np.repeat(refs[-1:, :], horizon - refs.shape[0], axis=0)
            refs = np.vstack((refs, pad))
        else:
            refs = refs[:horizon]

        actual = np.asarray(actual_position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(actual)):
            actual = refs[0].copy()

        if self._prev_cmd is None or not np.all(np.isfinite(self._prev_cmd)):
            self._prev_cmd = actual.copy()
        if self._prev_prev_cmd is None or not np.all(np.isfinite(self._prev_prev_cmd)):
            self._prev_prev_cmd = self._prev_cmd.copy()

        dynamics = self._dynamics_terms(horizon)
        a = dynamics["a"]
        bmat = dynamics["bmat"]
        weights = dynamics["weights"]
        step_size = dynamics["step_size"]

        reference_gap = float(np.linalg.norm(refs[0] - actual))
        y = self._project(refs.copy(), actual, a)

        for _ in range(max(0, int(cfg.iterations))):
            grad = self._gradient(y, refs, actual, bmat, weights)
            y = self._project(y - step_size * grad, actual, a)

        q_pred = self._predict_actual(y, actual, bmat)
        cost = self._cost(y, refs, actual, bmat, weights)
        command = y[0].copy()

        self._prev_prev_cmd = self._prev_cmd.copy()
        self._prev_cmd = command.copy()
        return TaskSpaceMPCResult(
            command=command,
            q_pred=q_pred,
            status=1,
            iterations=max(0, int(cfg.iterations)),
            cost=float(cost),
            reference_gap=reference_gap,
            raw_command_gap=reference_gap,
            command_gap=float(np.linalg.norm(command - actual)),
            tracking_error=float(np.linalg.norm(q_pred[0] - refs[0])),
        )

    def _dynamics_terms(self, horizon):
        key = (
            horizon,
            self.dt,
            self.config.tau,
            self.config.w_track,
            self.config.track_decay,
            self.config.w_cmd,
            self.config.w_yx,
            self.config.w_dy,
            self.config.w_ddy,
        )
        if self._cached_shape == key and self._cached is not None:
            return self._cached

        tau = max(1e-4, float(self.config.tau))
        a = float(np.exp(-self.dt / tau))
        bmat = np.zeros((horizon, horizon), dtype=np.float64)
        for row in range(horizon):
            for col in range(row + 1):
                bmat[row, col] = (1.0 - a) * (a ** (row - col))

        weights = np.asarray(
            [
                self.config.w_track * (self.config.track_decay ** idx)
                for idx in range(horizon)
            ],
            dtype=np.float64,
        )
        d1 = np.eye(horizon, dtype=np.float64)
        for idx in range(1, horizon):
            d1[idx, idx - 1] = -1.0
        d2 = np.eye(horizon, dtype=np.float64)
        if horizon > 1:
            d2[1, 0] = -2.0
        for idx in range(2, horizon):
            d2[idx, idx - 1] = -2.0
            d2[idx, idx - 2] = 1.0

        hess = 2.0 * (
            bmat.T @ (weights[:, None] * bmat)
            + self.config.w_cmd * np.eye(horizon)
            + self.config.w_yx * np.eye(horizon)
            + self.config.w_dy * (d1.T @ d1)
            + self.config.w_ddy * (d2.T @ d2)
        )
        eig_max = float(np.linalg.eigvalsh(hess).max()) if horizon > 0 else 1.0
        step_size = 0.8 / max(eig_max, 1e-8)
        self._cached_shape = key
        self._cached = {
            "a": a,
            "bmat": bmat,
            "weights": weights,
            "step_size": step_size,
        }
        return self._cached

    def _predict_actual(self, y, actual, bmat):
        dynamics = self._dynamics_terms(y.shape[0])
        const = np.asarray(
            [
                dynamics["a"] ** (idx + 1)
                for idx in range(y.shape[0])
            ],
            dtype=np.float64,
        )[:, None] * actual[None, :]
        return const + bmat @ y

    def _predict_state_before_command(self, y, actual, bmat):
        q_after = self._predict_actual(y, actual, bmat)
        if y.shape[0] == 1:
            return actual.reshape(1, 3)
        return np.vstack((actual.reshape(1, 3), q_after[:-1]))

    def _gradient(self, y, refs, actual, bmat, weights):
        q_pred = self._predict_actual(y, actual, bmat)
        grad = 2.0 * (bmat.T @ (weights[:, None] * (q_pred - refs)))
        grad += 2.0 * self.config.w_cmd * (y - refs)
        if self.config.w_yx > 0.0:
            before = self._predict_state_before_command(y, actual, bmat)
            before_mat = np.vstack(
                (
                    np.zeros((1, y.shape[0]), dtype=np.float64),
                    bmat[:-1],
                )
            )
            yx_op = np.eye(y.shape[0], dtype=np.float64) - before_mat
            grad += 2.0 * self.config.w_yx * (yx_op.T @ (y - before))

        prev = self._prev_cmd
        prev2 = self._prev_prev_cmd
        diff0 = y[0] - prev
        grad[0] += 2.0 * self.config.w_dy * diff0
        for idx in range(1, y.shape[0]):
            diff = y[idx] - y[idx - 1]
            term = 2.0 * self.config.w_dy * diff
            grad[idx] += term
            grad[idx - 1] -= term

        d20 = y[0] - 2.0 * prev + prev2
        grad[0] += 2.0 * self.config.w_ddy * d20
        if y.shape[0] > 1:
            d21 = y[1] - 2.0 * y[0] + prev
            term = 2.0 * self.config.w_ddy * d21
            grad[1] += term
            grad[0] -= 2.0 * term
        for idx in range(2, y.shape[0]):
            diff = y[idx] - 2.0 * y[idx - 1] + y[idx - 2]
            term = 2.0 * self.config.w_ddy * diff
            grad[idx] += term
            grad[idx - 1] -= 2.0 * term
            grad[idx - 2] += term
        return grad

    def _cost(self, y, refs, actual, bmat, weights):
        q_pred = self._predict_actual(y, actual, bmat)
        cost = float(np.sum(weights[:, None] * (q_pred - refs) ** 2))
        cost += float(self.config.w_cmd * np.sum((y - refs) ** 2))
        if self.config.w_yx > 0.0:
            before = self._predict_state_before_command(y, actual, bmat)
            cost += float(self.config.w_yx * np.sum((y - before) ** 2))
        prev = self._prev_cmd
        prev2 = self._prev_prev_cmd
        dy = np.vstack((y[0] - prev, y[1:] - y[:-1]))
        cost += float(self.config.w_dy * np.sum(dy ** 2))
        ddy_parts = [y[0] - 2.0 * prev + prev2]
        if y.shape[0] > 1:
            ddy_parts.append(y[1] - 2.0 * y[0] + prev)
        if y.shape[0] > 2:
            ddy_parts.extend(list(y[2:] - 2.0 * y[1:-1] + y[:-2]))
        cost += float(self.config.w_ddy * np.sum(np.asarray(ddy_parts) ** 2))
        return cost

    def _project(self, y, actual, a):
        y = np.asarray(y, dtype=np.float64).copy()
        cfg = self.config
        dy_max = max(0.0, float(cfg.max_velocity)) * self.dt
        ddy_max = max(0.0, float(cfg.max_acceleration)) * self.dt * self.dt
        dmax = self._command_actual_gap_limit()

        q_prev = actual.copy()
        prev = self._prev_cmd.copy()
        prev2 = self._prev_prev_cmd.copy()
        for idx in range(y.shape[0]):
            cmd = self._clip_workspace(y[idx])
            cmd = self._limit_norm(cmd, prev, dy_max)
            accel_center = 2.0 * prev - prev2
            cmd = self._limit_norm(cmd, accel_center, ddy_max)
            cmd = self._limit_norm(cmd, q_prev, dmax)
            cmd = self._clip_workspace(cmd)
            y[idx] = cmd
            q_prev = a * q_prev + (1.0 - a) * cmd
            prev2 = prev
            prev = cmd
        return y

    def _clip_workspace(self, xyz):
        out = np.asarray(xyz, dtype=np.float64).copy()
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            limit = self.workspace_limits.get(axis)
            if limit is not None and len(limit) == 2:
                out[idx] = min(max(out[idx], float(limit[0])), float(limit[1]))
        return out

    def _command_actual_gap_limit(self):
        configured = max(0.0, float(self.config.max_cmd_actual_gap))
        if configured > 0.0:
            return configured
        vmax = max(0.0, float(self.config.max_velocity))
        if vmax <= 0.0:
            return 0.0
        tau = max(1e-4, float(self.config.tau))
        return tau * vmax

    @staticmethod
    def _limit_norm(value, center, radius):
        value = np.asarray(value, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        radius = max(0.0, float(radius))
        delta = value - center
        norm = float(np.linalg.norm(delta))
        if norm <= radius or norm <= 1e-12:
            return value.copy()
        return center + delta * (radius / norm)
