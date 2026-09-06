"""A posteriori basin, forcing, and primal-recovery certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


Array = np.ndarray


@dataclass(frozen=True)
class DualState:
    y: Array
    q: Array
    x: Array
    g: Array
    conductance: Array
    sigma: float
    x_min: float
    weighted_degree_max: float


@dataclass(frozen=True)
class BasinCertificate:
    eta_upper: float
    decrement_upper: float
    tree_energy: float
    passed: bool


@dataclass(frozen=True)
class ForcingCertificate:
    residual_upper: float
    decrement_lower: float
    eta_lower: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class RecoveryResult:
    x_hat: Array
    correction: Array
    residual_l1: float
    objective_gap_upper: float


def dual_state(problem: ReciprocalTransportProblem, y: Array) -> DualState:
    y = np.asarray(y, dtype=float)
    q = problem.dual_slack(y)
    if np.any(q <= 0) or not np.all(np.isfinite(q)):
        raise ValueError("dual point lies outside q > 0")
    x = np.sqrt(problem.mu / q)
    g = problem.impose_reduced_balance(problem.node_balance(x) - problem.beta)
    conductance = np.sqrt(problem.mu) / (2.0 * q ** 1.5)
    degree = problem.laplacian_diagonal(conductance)
    return DualState(
        y=y.copy(),
        q=q,
        x=x,
        g=g,
        conductance=conductance,
        sigma=float(np.min((problem.mu * q) ** 0.25)),
        x_min=float(np.min(x)),
        weighted_degree_max=float(np.max(degree)),
    )


def hessian(problem: ReciprocalTransportProblem, state: DualState) -> sp.csr_matrix:
    b = problem.incidence
    return (b @ sp.diags(state.conductance) @ b.T).tocsr()


def basin_certificate(
    problem: ReciprocalTransportProblem,
    state: DualState,
    router: TreeRouter,
    threshold: float = 0.1,
) -> BasinCertificate:
    router.validate_for(problem)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 0.1:
        raise ValueError("basin threshold must be finite and in (0, 0.1]")
    energy = router.energy(state.g, state.conductance)
    decrement_upper = float(np.sqrt(max(0.0, energy)))
    eta_upper = float(np.sqrt(2.0) * decrement_upper / state.sigma)
    return BasinCertificate(
        eta_upper=eta_upper,
        decrement_upper=decrement_upper,
        tree_energy=energy,
        passed=bool(eta_upper <= threshold),
    )


def forcing_certificate(
    problem: ReciprocalTransportProblem,
    state: DualState,
    direction: Array,
    router: TreeRouter,
    h: Optional[sp.csr_matrix] = None,
) -> ForcingCertificate:
    router.validate_for(problem)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (problem.n,):
        raise ValueError("direction has the wrong dimension")
    if h is None:
        h_direction = problem.laplacian_matvec(direction, state.conductance)
    else:
        h_direction = np.asarray(h @ direction)
    residual = h_direction + state.g
    residual_upper = float(np.sqrt(max(0.0, router.energy(residual, state.conductance))))
    denom_sq = float(direction @ h_direction)
    if denom_sq <= 0:
        decrement_lower = 0.0
    else:
        decrement_lower = float(
            abs(problem.balanced_pairing(state.g, direction)) / np.sqrt(denom_sq)
        )
    eta_lower = float(np.sqrt(2.0) * decrement_lower / state.sigma)
    threshold = float(eta_lower * decrement_lower / 4.0)
    return ForcingCertificate(
        residual_upper=residual_upper,
        decrement_lower=decrement_lower,
        eta_lower=eta_lower,
        threshold=threshold,
        passed=bool(residual_upper <= threshold),
    )


def exact_decrement(
    problem: ReciprocalTransportProblem,
    state: DualState,
    h: Optional[sp.csr_matrix] = None,
) -> float:
    """Reference-only decrement using a sparse direct reduced solve."""

    if h is None:
        h = hessian(problem, state)
    keep = np.arange(problem.n - 1)
    h_red = h[keep][:, keep].tocsc()
    g_red = state.g[keep]
    z_red = spla.spsolve(h_red, g_red)
    return float(np.sqrt(max(0.0, g_red @ z_red)))


def stopping_radius(
    problem: ReciprocalTransportProblem,
    state: DualState,
    epsilon: float,
) -> float:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    quadratic = np.sqrt(
        epsilon * state.x_min ** 3 /
        (2.0 * float(np.max(problem.mu)) * max(1, problem.n - 1))
    )
    return float(min(state.x_min, quadratic))


def recover_primal(
    problem: ReciprocalTransportProblem,
    state: DualState,
    router: TreeRouter,
) -> RecoveryResult:
    router.validate_for(problem)
    residual_l1 = float(np.linalg.norm(state.g, 1))
    if residual_l1 > state.x_min * (1.0 + 1e-12):
        raise ValueError("residual is too large for certified positive recovery")
    correction = router.route(-state.g)
    x_hat = state.x + correction
    if np.any(x_hat <= 0):
        raise ArithmeticError("floating-point recovery lost strict positivity")
    gap_upper = (
        2.0 * float(np.max(problem.mu)) * max(1, problem.n - 1)
        * residual_l1 ** 2 / state.x_min ** 3
    )
    return RecoveryResult(
        x_hat=x_hat,
        correction=correction,
        residual_l1=residual_l1,
        objective_gap_upper=float(gap_upper),
    )
