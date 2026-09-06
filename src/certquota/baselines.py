"""Reproducible baselines for reciprocal transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pyamg
except ImportError:  # pragma: no cover - optional benchmark dependency
    pyamg = None

from .certificates import dual_state, hessian, recover_primal, stopping_radius
from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


Array = np.ndarray


def _time_exceeded(started: float, limit: Optional[float]) -> bool:
    return bool(limit is not None and perf_counter() - started >= limit)


@dataclass
class BaselineResult:
    status: str
    x: Optional[Array]
    y: Array
    objective: Optional[float]
    history: List[Dict[str, float]] = field(default_factory=list)
    wall_time: float = 0.0


@dataclass
class RSOCBaselineResult(BaselineResult):
    """Clarabel RSOC candidate with its original-unit epigraph variable."""

    epigraph_q: Optional[Array] = None


def _dual_objective(problem: ReciprocalTransportProblem, y: Array) -> float:
    q = problem.dual_slack(y)
    if np.any(q <= 0):
        return float("inf")
    return float(
        -problem.balanced_pairing(problem.beta, y)
        - 2.0 * np.sum(np.sqrt(problem.mu * q))
    )


def _stable_dual_objective_change(
    problem: ReciprocalTransportProblem,
    state,
    direction: Array,
    edge_step: Array,
    step: float,
) -> float:
    """Evaluate ``phi(y+step*d)-phi(y)`` without subtracting large objectives.

    With ``z=step*(B.T d)/q``, exact algebra gives

    ``Delta phi = step*g.T*d + sum sqrt(mu*q) z^2/(1+sqrt(1-z))^2``.

    The summands after the directional derivative are nonnegative, avoiding the
    catastrophic cancellation that makes a conventional Armijo test stall near a
    large optimum.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    ratio = step * np.asarray(edge_step, dtype=float) / state.q
    if np.any(ratio >= 1.0) or not np.all(np.isfinite(ratio)):
        return float("inf")
    root = np.sqrt(1.0 - ratio)
    remainder = np.sqrt(problem.mu * state.q) * (
        ratio / (1.0 + root)
    ) ** 2
    return float(
        step * problem.balanced_pairing(state.g, direction)
        + np.sum(remainder)
    )


def _margin_root_shift(q: Array, mu: Array, margin: float) -> float:
    """Solve ``sum sqrt(mu / (q-t)) = margin`` by safeguarded Newton."""

    if margin <= 0:
        raise ValueError("margins must be positive")
    upper = float(np.min(q))
    safety = max(np.finfo(float).eps * max(1.0, abs(upper)), 1e-15)
    hi = upper - safety

    def value(t: float) -> float:
        return float(np.sum(np.sqrt(mu / (q - t))) - margin)

    lo = min(0.0, float(np.min(q)) - 1.0)
    width = max(1.0, float(np.ptp(q)), abs(lo))
    while value(lo) > 0:
        lo -= width
        width *= 2.0
    t = min(max(0.0, lo), hi)
    for _ in range(60):
        denom = q - t
        f = float(np.sum(np.sqrt(mu / denom)) - margin)
        if abs(f) <= 1e-13 * max(1.0, margin):
            return t
        if f > 0:
            hi = t
        else:
            lo = t
        derivative = float(0.5 * np.sum(np.sqrt(mu) / denom ** 1.5))
        candidate = t - f / derivative
        if not (lo < candidate < hi) or not np.isfinite(candidate):
            candidate = 0.5 * (lo + hi)
        t = candidate
    return 0.5 * (lo + hi)


def dual_feasible_start(
    problem: ReciprocalTransportProblem, minimum_slack: float = 1.0
) -> Array:
    """Construct a dual-domain point for a bipartite instance."""

    if minimum_slack <= 0:
        raise ValueError("minimum_slack must be positive")
    k = max(0.0, 0.5 * (minimum_slack - float(np.min(problem.c))))
    y = np.empty(problem.n, dtype=float)
    y[: problem.n_left] = -k
    y[problem.n_left :] = k
    y -= float(y.mean())
    if float(np.min(problem.dual_slack(y))) < minimum_slack * (1.0 - 1e-12):
        raise ArithmeticError("failed to construct the requested dual slack")
    return y


def nonoptimal_dual_start(
    problem: ReciprocalTransportProblem,
    seed: int = 0,
    relative_slack_step: float = 0.25,
) -> Array:
    """Construct a reproducible interior start that is not an accidental KKT point.

    The construction uses only public problem arrays. It first obtains a domain
    point, then applies a seeded potential perturbation whose maximum relative
    slack change is bounded by ``relative_slack_step``.
    """

    if not 0 < relative_slack_step < 1:
        raise ValueError("relative_slack_step must lie in (0, 1)")
    y = dual_feasible_start(problem)
    q = problem.dual_slack(y)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=problem.n)
    direction -= float(direction.mean())
    edge_change = problem.edge_difference(direction)
    relative = np.max(np.abs(edge_change) / q)
    if not np.isfinite(relative) or relative == 0:
        raise ArithmeticError("failed to construct a nonconstant dual direction")
    direction *= relative_slack_step / relative
    candidate = y + direction
    candidate -= float(candidate.mean())
    if np.any(problem.dual_slack(candidate) <= 0):
        raise ArithmeticError("constructed dual start left the strict domain")
    return candidate


def strictly_feasible_primal_start(
    problem: ReciprocalTransportProblem,
) -> Array:
    """Find a positive feasible flow by a max-min HiGHS linear program."""

    m = problem.m
    objective = np.zeros(m + 1, dtype=float)
    objective[-1] = -1.0
    equality = sp.hstack(
        [problem.incidence[:-1], sp.csr_matrix((problem.n - 1, 1))],
        format="csr",
    )
    inequality = sp.hstack(
        [-sp.eye(m, format="csr"), np.ones((m, 1))], format="csr"
    )
    result = opt.linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(m),
        A_eq=equality,
        b_eq=problem.beta[:-1],
        bounds=[(0.0, None)] * m + [(0.0, None)],
        method="highs",
    )
    if not result.success or result.x[-1] <= 0:
        raise RuntimeError("failed to construct a strictly positive feasible flow")
    x = np.asarray(result.x[:-1], dtype=float)
    if np.any(x <= 0):
        raise ArithmeticError("linear-program start is not strictly positive")
    return x


def coordinate_scaling_sweep(
    problem: ReciprocalTransportProblem,
    y: Array,
    q: Array,
    incident,
) -> None:
    """Apply one in-place cyclic row/column marginal projection sweep."""

    for vertex in range(problem.n):
        indices = np.asarray(incident[vertex], dtype=np.int64)
        target = float(
            problem.beta[vertex]
            if vertex < problem.n_left
            else -problem.beta[vertex]
        )
        shift = _margin_root_shift(q[indices], problem.mu[indices], target)
        sign = 1.0 if vertex < problem.n_left else -1.0
        y[vertex] += sign * shift
        q[indices] -= shift
    y -= float(y.mean())


def _batched_margin_root_shift(
    q: Array,
    mu: Array,
    groups: Array,
    targets: Array,
    max_iterations: int = 60,
    relative_tolerance: float = 2e-13,
    return_iterations: bool = False,
):
    """Solve independent node-coordinate equations in vectorized form."""

    q = np.asarray(q, dtype=float)
    mu = np.asarray(mu, dtype=float)
    groups = np.asarray(groups, dtype=np.int64)
    targets = np.asarray(targets, dtype=float)
    count = targets.size
    if np.any(targets <= 0):
        raise ValueError("strictly positive marginals are required")
    minimum = np.full(count, np.inf, dtype=float)
    np.minimum.at(minimum, groups, q)
    if not np.all(np.isfinite(minimum)):
        raise ValueError("every node must be incident to an edge")
    root_mu = np.sqrt(mu)
    amplitude = np.bincount(groups, weights=root_mu, minlength=count)
    width = np.maximum((amplitude / targets) ** 2, 1.0)
    lo = minimum - width * (1.0 + 8.0 * np.finfo(float).eps)
    hi = np.nextafter(minimum, -np.inf)
    # Coordinate shifts are usually close to zero under a warm start. Starting at
    # the feasible projection of zero preserves safeguarding while avoiding many
    # bisections from a very distant bracket midpoint.
    value = np.minimum(np.maximum(np.zeros(count, dtype=float), lo), hi)

    active = np.ones(count, dtype=bool)
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        denominator = q - value[groups]
        root_term = root_mu / np.sqrt(denominator)
        total = np.bincount(groups, weights=root_term, minlength=count)
        residual = total - targets
        converged = np.abs(residual) <= relative_tolerance * targets
        # A scale-one width test is unsafe when the true coordinate shift is
        # tiny: thousands of individually small marginal errors can then stall
        # the global L1 stopping rule. Only declare a floating bracket exhausted
        # when no representable interior point remains.
        narrow = np.nextafter(lo, hi) >= hi
        active &= ~(converged | narrow)
        if not np.any(active):
            break
        above = (residual > 0) & active
        hi = np.where(above, value, hi)
        lo = np.where(active & ~above, value, lo)
        derivative = 0.5 * np.bincount(
            groups,
            weights=root_mu / denominator**1.5,
            minlength=count,
        )
        candidate = value - residual / derivative
        midpoint = 0.5 * (lo + hi)
        valid = (
            active
            & np.isfinite(candidate)
            & (candidate > lo)
            & (candidate < hi)
        )
        update = np.where(valid, candidate, midpoint)
        value = np.where(active, update, value)
    if return_iterations:
        return value, iterations
    return value


def coordinate_scaling_sweep_vectorized(
    problem: ReciprocalTransportProblem,
    y: Array,
    q: Array,
) -> None:
    """Apply one bipartite row/column sweep using batched root solves."""

    left_shift = _batched_margin_root_shift(
        q,
        problem.mu,
        problem.tails,
        problem.beta[: problem.n_left],
    )
    y[: problem.n_left] += left_shift
    q -= left_shift[problem.tails]

    right_group = problem.heads - problem.n_left
    right_shift = _batched_margin_root_shift(
        q,
        problem.mu,
        right_group,
        -problem.beta[problem.n_left :],
    )
    y[problem.n_left :] -= right_shift
    q -= right_shift[right_group]
    y -= float(y.mean())


def solve_alternating_scaling(
    problem: ReciprocalTransportProblem,
    y0: Array,
    epsilon: float = 1e-10,
    max_sweeps: int = 10000,
    router: Optional[TreeRouter] = None,
    time_limit_seconds: Optional[float] = None,
) -> BaselineResult:
    """Global row/column dual coordinate minimization.

    Each node update exactly enforces its marginal.  This is the reciprocal analogue
    of alternate scaling/Bregman projection and is a required strong baseline.
    """

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        isinstance(max_sweeps, (bool, np.bool_))
        or not isinstance(max_sweeps, (int, np.integer))
        or max_sweeps < 0
    ):
        raise ValueError("max_sweeps must be a nonnegative integer")
    if time_limit_seconds is not None and (
        not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    y = np.asarray(y0, dtype=float).copy()
    if y.shape != (problem.n,) or not np.all(np.isfinite(y)):
        raise ValueError("y0 must be a finite vector of length problem.n")
    y -= float(y.mean())
    q = problem.dual_slack(y)
    if np.any(q <= 0):
        return BaselineResult("invalid_initial_domain", None, y, None)
    if router is None:
        router = TreeRouter.build(problem)
    history: List[Dict[str, float]] = []
    started = perf_counter()

    for sweep in range(max_sweeps + 1):
        if _time_exceeded(started, time_limit_seconds):
            return BaselineResult(
                "timed_out", None, y, None, history, perf_counter() - started
            )
        state = dual_state(problem, y)
        residual_l1 = float(np.linalg.norm(state.g, 1))
        radius = stopping_radius(problem, state, epsilon)
        if residual_l1 <= radius:
            recovered = recover_primal(problem, state, router)
            return BaselineResult(
                "optimal",
                recovered.x_hat,
                y,
                problem.objective(recovered.x_hat),
                history,
                perf_counter() - started,
            )
        if sweep == max_sweeps:
            break

        coordinate_scaling_sweep_vectorized(problem, y, q)
        if sweep < 20 or sweep % 10 == 0:
            history.append(
                {"sweep": float(sweep), "residual_l1": residual_l1, "stop_radius": radius}
            )

    return BaselineResult(
        "max_sweeps", None, y, None, history, perf_counter() - started
    )


def solve_damped_newton(
    problem: ReciprocalTransportProblem,
    y0: Array,
    epsilon: float = 1e-10,
    max_steps: int = 100,
    direct_solve_below: int = 5_000,
    cg_tolerance: float = 1e-10,
    router: Optional[TreeRouter] = None,
    preconditioner: str = "jacobi",
    time_limit_seconds: Optional[float] = None,
) -> BaselineResult:
    """Conventional damped Newton baseline without posterior certificates."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a nonnegative integer")
    if (
        isinstance(direct_solve_below, (bool, np.bool_))
        or not isinstance(direct_solve_below, (int, np.integer))
        or direct_solve_below < 0
    ):
        raise ValueError("direct_solve_below must be a nonnegative integer")
    if not np.isfinite(cg_tolerance) or cg_tolerance <= 0.0:
        raise ValueError("cg_tolerance must be finite and positive")
    if time_limit_seconds is not None and (
        not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    y = np.asarray(y0, dtype=float).copy()
    if y.shape != (problem.n,) or not np.all(np.isfinite(y)):
        raise ValueError("y0 must be a finite vector of length problem.n")
    y -= float(y.mean())
    if np.any(problem.dual_slack(y) <= 0):
        return BaselineResult("invalid_initial_domain", None, y, None)
    if router is None:
        router = TreeRouter.build(problem)
    if preconditioner not in {"jacobi", "tree", "amg"}:
        raise ValueError("preconditioner must be 'jacobi', 'tree', or 'amg'")
    keep = np.arange(problem.n - 1)
    history: List[Dict[str, float]] = []
    started = perf_counter()

    for iteration in range(max_steps + 1):
        if _time_exceeded(started, time_limit_seconds):
            return BaselineResult(
                "timed_out", None, y, None, history, perf_counter() - started
            )
        state = dual_state(problem, y)
        residual_l1 = float(np.linalg.norm(state.g, 1))
        radius = stopping_radius(problem, state, epsilon)
        if residual_l1 <= radius:
            recovered = recover_primal(problem, state, router)
            return BaselineResult(
                "optimal",
                recovered.x_hat,
                y,
                problem.objective(recovered.x_hat),
                history,
                perf_counter() - started,
            )
        if iteration == max_steps:
            break

        rhs = -state.g[keep]
        if problem.n <= direct_solve_below:
            h = hessian(problem, state)
            reduced_h = h[keep][:, keep].tocsr()
            reduced_direction = spla.spsolve(reduced_h.tocsc(), rhs)
            inner_iterations = 0
        else:
            diagonal = np.maximum(
                problem.laplacian_diagonal(state.conductance)[keep],
                np.finfo(float).tiny,
            )

            def reduced_matvec(value):
                full = np.zeros(problem.n, dtype=float)
                full[keep] = value
                return problem.laplacian_matvec(
                    full, state.conductance
                )[keep]

            if preconditioner == "amg":
                if pyamg is None:
                    raise RuntimeError("the AMG preconditioner requires pyamg")
                explicit_h = hessian(problem, state)
                reduced_h = explicit_h[keep][:, keep].tocsr()
            else:
                reduced_h = spla.LinearOperator(
                    (problem.n - 1, problem.n - 1),
                    matvec=reduced_matvec,
                    dtype=float,
                )
            if preconditioner == "jacobi":
                preconditioner_operator = spla.LinearOperator(
                    reduced_h.shape, matvec=lambda value: value / diagonal
                )
            elif preconditioner == "tree":
                def apply_tree(value):
                    full_demand = np.zeros(problem.n, dtype=float)
                    full_demand[keep] = value
                    full_demand[-1] = -float(np.sum(value))
                    potential = router.solve_laplacian(
                        full_demand, state.conductance
                    )
                    potential -= potential[-1]
                    return potential[keep]

                preconditioner_operator = spla.LinearOperator(
                    reduced_h.shape, matvec=apply_tree, dtype=float
                )
            else:
                hierarchy = pyamg.smoothed_aggregation_solver(
                    reduced_h, symmetry="symmetric"
                )
                preconditioner_operator = hierarchy.aspreconditioner(cycle="V")
            counter = {"iterations": 0}

            class LinearSolveTimeLimitReached(Exception):
                pass

            def callback(_):
                counter["iterations"] += 1
                if _time_exceeded(started, time_limit_seconds):
                    raise LinearSolveTimeLimitReached

            try:
                try:
                    reduced_direction, info = spla.cg(
                        reduced_h,
                        rhs,
                        rtol=cg_tolerance,
                        atol=0.0,
                        M=preconditioner_operator,
                        maxiter=max(100, 10 * problem.n),
                        callback=callback,
                    )
                except TypeError:
                    reduced_direction, info = spla.cg(
                        reduced_h,
                        rhs,
                        tol=cg_tolerance,
                        atol=0.0,
                        M=preconditioner_operator,
                        maxiter=max(100, 10 * problem.n),
                        callback=callback,
                    )
            except LinearSolveTimeLimitReached:
                return BaselineResult(
                    "timed_out", None, y, None, history,
                    perf_counter() - started
                )
            if info != 0:
                return BaselineResult(
                    "linear_solve_failed", None, y, None, history,
                    perf_counter() - started
                )
            inner_iterations = counter["iterations"]

        if not np.all(np.isfinite(reduced_direction)):
            return BaselineResult(
                "linear_solve_nonfinite", None, y, None, history,
                perf_counter() - started
            )

        direction = np.zeros(problem.n, dtype=float)
        direction[keep] = reduced_direction
        direction -= float(direction.mean())
        edge_step = problem.edge_difference(direction)
        positive = edge_step > 0
        max_step = 1.0
        if np.any(positive):
            max_step = min(
                1.0,
                0.99 * float(np.min(state.q[positive] / edge_step[positive])),
            )
        directional_derivative = problem.balanced_pairing(state.g, direction)
        if directional_derivative >= 0.0 or not np.isfinite(directional_derivative):
            return BaselineResult(
                "non_descent_direction", None, y, None, history,
                perf_counter() - started
            )
        step = max_step
        accepted = False
        for _ in range(60):
            candidate = y + step * direction
            candidate -= float(candidate.mean())
            objective_change = _stable_dual_objective_change(
                problem, state, direction, edge_step, step
            )
            if objective_change <= 1e-4 * step * directional_derivative:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return BaselineResult(
                "line_search_failed", None, y, None, history,
                perf_counter() - started
            )
        history.append(
            {
                "iteration": float(iteration),
                "residual_l1": residual_l1,
                "step": step,
                "inner_iterations": float(inner_iterations),
            }
        )
        y = candidate

    return BaselineResult(
        "max_steps", None, y, None, history, perf_counter() - started
    )


def solve_slsqp_reference(
    problem: ReciprocalTransportProblem,
    x0: Array,
    max_iterations: int = 2_000,
    tolerance: float = 1e-11,
    time_limit_seconds: Optional[float] = None,
) -> BaselineResult:
    """Small-instance generic nonlinear-programming reference baseline."""

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (problem.m,) or np.any(x0 <= 0):
        raise ValueError("x0 must be a positive edge flow")
    b = problem.incidence[:-1]
    beta = problem.beta[:-1]
    scaled_b = b @ sp.diags(x0)
    constraint = opt.LinearConstraint(scaled_b, beta, beta)
    lower = max(np.finfo(float).tiny, 1e-12)
    bounds = opt.Bounds(np.full(problem.m, lower), np.full(problem.m, np.inf))
    # A far-from-optimal feasible flow can have an enormous reciprocal objective
    # and make SLSQP's scaled termination test vacuous.  A public dual-domain
    # point supplies a problem-scale lower-bound magnitude without using the
    # planted optimum or another solver's output.  Its square root keeps the
    # scaled objective well above one so ftol still resolves matched-gap changes.
    dual_scale = abs(_dual_objective(problem, dual_feasible_start(problem)))
    objective_scale = max(1.0, np.sqrt(dual_scale))
    started = perf_counter()
    latest = {"z": np.ones(problem.m)}

    class TimeLimitReached(Exception):
        pass

    def callback(z):
        latest["z"] = np.asarray(z, dtype=float).copy()
        if _time_exceeded(started, time_limit_seconds):
            raise TimeLimitReached

    try:
        result = opt.minimize(
            lambda z: problem.objective(x0 * z) / objective_scale,
            np.ones(problem.m),
            jac=lambda z: (
                (problem.c - problem.mu / (x0 * z) ** 2)
                * x0
                / objective_scale
            ),
            method="SLSQP",
            constraints=[constraint],
            bounds=bounds,
            callback=callback,
            options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
        )
    except TimeLimitReached:
        y = np.zeros(problem.n)
        return BaselineResult(
            "timed_out", None, y, None,
            [{"iterations": float("nan"), "success": 0.0}],
            perf_counter() - started,
        )
    x = x0 * np.asarray(result.x, dtype=float)
    status = "optimal" if result.success else "failed"
    return BaselineResult(
        status,
        x,
        np.zeros(problem.n),
        problem.objective(x),
        [{"iterations": float(result.nit), "success": float(result.success)}],
        perf_counter() - started,
    )


def solve_lbfgs_dual(
    problem: ReciprocalTransportProblem,
    y0: Array,
    epsilon: float = 1e-10,
    max_iterations: int = 2_000,
    router: Optional[TreeRouter] = None,
    time_limit_seconds: Optional[float] = None,
) -> BaselineResult:
    """Generic L-BFGS dual baseline with a domain-rejecting oracle."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a nonnegative integer")
    if time_limit_seconds is not None and (
        not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    y0 = np.asarray(y0, dtype=float).copy()
    if y0.shape != (problem.n,) or not np.all(np.isfinite(y0)):
        raise ValueError("y0 must be a finite vector of length problem.n")
    y0 -= y0[-1]
    if np.any(problem.dual_slack(y0) <= 0):
        return BaselineResult("invalid_initial_domain", None, y0, None)
    if router is None:
        router = TreeRouter.build(problem)
    evaluations = {"count": 0, "domain_rejects": 0}
    objective_normalization = max(1.0, abs(_dual_objective(problem, y0)))

    def oracle(reduced_y):
        evaluations["count"] += 1
        y = np.zeros(problem.n, dtype=float)
        y[:-1] = reduced_y
        q = problem.dual_slack(y)
        if np.any(q <= 0) or not np.all(np.isfinite(q)):
            evaluations["domain_rejects"] += 1
            return 1e100, np.zeros(problem.n - 1, dtype=float)
        state = dual_state(problem, y)
        return (
            _dual_objective(problem, y) / objective_normalization,
            state.g[:-1] / objective_normalization,
        )

    started = perf_counter()
    latest = {"y": y0[:-1].copy()}

    class TimeLimitReached(Exception):
        pass

    def callback(reduced_y):
        latest["y"] = np.asarray(reduced_y, dtype=float).copy()
        if _time_exceeded(started, time_limit_seconds):
            raise TimeLimitReached

    try:
        result = opt.minimize(
            oracle,
            y0[:-1],
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": max_iterations,
                "ftol": 1e-18,
                "gtol": 1e-18,
                "maxls": 100,
            },
        )
    except TimeLimitReached:
        y = np.zeros(problem.n, dtype=float)
        y[:-1] = latest["y"]
        y -= float(y.mean())
        return BaselineResult(
            "timed_out", None, y, None,
            [{"evaluations": float(evaluations["count"])}],
            perf_counter() - started,
        )
    y = np.zeros(problem.n, dtype=float)
    y[:-1] = np.asarray(result.x, dtype=float)
    y -= float(y.mean())
    if np.any(problem.dual_slack(y) <= 0):
        return BaselineResult(
            "domain_failure",
            None,
            y,
            None,
            [{"evaluations": float(evaluations["count"])}],
            perf_counter() - started,
        )
    state = dual_state(problem, y)
    residual_l1 = float(np.linalg.norm(state.g, 1))
    radius = stopping_radius(problem, state, epsilon)
    if residual_l1 <= radius:
        recovered = recover_primal(problem, state, router)
        status = "optimal"
        x = recovered.x_hat
        objective = problem.objective(x)
    else:
        status = "inaccurate" if result.success else "failed"
        x = None
        objective = None
    return BaselineResult(
        status,
        x,
        y,
        objective,
        [
            {
                "iterations": float(result.nit),
                "evaluations": float(evaluations["count"]),
                "domain_rejects": float(evaluations["domain_rejects"]),
                "residual_l1": residual_l1,
                "stop_radius": radius,
                "scipy_success": float(result.success),
            }
        ],
        perf_counter() - started,
    )


def solve_cvxpy_reference(
    problem: ReciprocalTransportProblem,
    max_iterations: int = 10_000,
    time_limit_seconds: Optional[float] = None,
) -> BaselineResult:
    """Independent CVXPY/Clarabel reciprocal-convex reference."""

    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError("the conic reference requires cvxpy") from error

    started = perf_counter()
    x_scale = max(
        np.finfo(float).tiny,
        float(np.sum(problem.beta[: problem.n_left])) / problem.m,
    )
    z_variable = cp.Variable(problem.m, pos=True)
    x_expression = x_scale * z_variable
    objective_scale = max(
        1.0,
        float(np.sum(np.abs(problem.c) * x_scale + problem.mu / x_scale)),
    )
    objective = cp.Minimize(
        (
            problem.c @ x_expression
            + cp.sum(cp.multiply(problem.mu / x_scale, cp.inv_pos(z_variable)))
        )
        / objective_scale
    )
    row_scale = 1.0 / np.maximum(
        np.abs(problem.beta[:-1]), x_scale
    )
    scaled_incidence = sp.diags(row_scale) @ problem.incidence[:-1]
    constraints = [
        scaled_incidence @ x_expression == row_scale * problem.beta[:-1]
    ]
    model = cp.Problem(objective, constraints)
    try:
        settings = {
            "solver": "CLARABEL",
            "max_iter": max_iterations,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "verbose": False,
        }
        if time_limit_seconds is not None:
            settings["time_limit"] = time_limit_seconds
        model.solve(**settings)
    except Exception as error:  # solver failure is a benchmark outcome
        return BaselineResult(
            f"exception:{type(error).__name__}",
            None,
            np.zeros(problem.n),
            None,
            [{"solver_exception": 1.0}],
            perf_counter() - started,
        )
    if z_variable.value is None:
        return BaselineResult(
            str(model.status), None, np.zeros(problem.n), None,
            [{"solver_iterations": float(getattr(model.solver_stats, "num_iters", 0) or 0)}],
            perf_counter() - started,
        )
    x = x_scale * np.asarray(z_variable.value, dtype=float).reshape(-1)
    status = "optimal" if model.status in {"optimal", "optimal_inaccurate"} else str(model.status)
    return BaselineResult(
        status,
        x,
        np.zeros(problem.n),
        problem.objective(x),
        [
            {
                "solver_iterations": float(
                    getattr(model.solver_stats, "num_iters", 0) or 0
                ),
                "optimal_inaccurate": float(model.status == "optimal_inaccurate"),
            }
        ],
        perf_counter() - started,
    )


def solve_cvxpy_rsoc_candidate(
    problem: ReciprocalTransportProblem,
    max_iterations: int = 10_000,
    time_limit_seconds: Optional[float] = None,
) -> RSOCBaselineResult:
    """Solve the reciprocal epigraph as standard SOCs with CVXPY/Clarabel.

    For every edge the scaled variables ``z=x/x_scale`` and
    ``w=q/q_scale`` use ``q_scale=1/x_scale``.  Hence ``z*w >= 1`` is exactly
    ``x*q >= 1`` and is represented by

    ``norm([2, z-w]) <= z+w``.

    The equality dual returned by CVXPY belongs to the row-scaled constraints
    and the objective divided by ``objective_scale``.  If that dual is
    ``lambda``, the canonical reciprocal-transport potential (root gauge zero)
    is ``y[:-1] = -objective_scale * row_scale * lambda``.  The minus sign is
    required by CVXPY's ``lhs-rhs == 0`` Lagrangian convention.
    """

    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("the RSOC candidate requires cvxpy") from error

    started = perf_counter()
    x_scale = max(
        np.finfo(float).tiny,
        float(np.sum(problem.beta[: problem.n_left])) / problem.m,
    )
    q_scale = 1.0 / x_scale
    z_variable = cp.Variable(problem.m, nonneg=True)
    w_variable = cp.Variable(problem.m, nonneg=True)
    x_expression = x_scale * z_variable
    q_expression = q_scale * w_variable
    objective_scale = max(
        1.0,
        float(
            np.sum(
                np.abs(problem.c) * x_scale + problem.mu * q_scale
            )
        ),
    )
    objective = cp.Minimize(
        (
            problem.c @ x_expression
            + problem.mu @ q_expression
        )
        / objective_scale
    )
    row_scale = 1.0 / np.maximum(np.abs(problem.beta[:-1]), x_scale)
    scaled_incidence = sp.diags(row_scale) @ problem.incidence[:-1]
    equality = (
        scaled_incidence @ x_expression == row_scale * problem.beta[:-1]
    )
    cone = cp.SOC(
        z_variable + w_variable,
        cp.vstack(
            [
                2.0 * np.ones(problem.m, dtype=float),
                z_variable - w_variable,
            ]
        ),
        axis=0,
    )
    model = cp.Problem(objective, [equality, cone])
    try:
        settings = {
            "solver": "CLARABEL",
            "max_iter": max_iterations,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "verbose": False,
        }
        if time_limit_seconds is not None:
            settings["time_limit"] = time_limit_seconds
        model.solve(**settings)
    except Exception as error:  # solver failure is a benchmark outcome
        return RSOCBaselineResult(
            status=f"exception:{type(error).__name__}",
            x=None,
            y=np.zeros(problem.n),
            objective=None,
            history=[{"solver_exception": 1.0}],
            wall_time=perf_counter() - started,
            epigraph_q=None,
        )

    iterations = float(getattr(model.solver_stats, "num_iters", 0) or 0)
    if z_variable.value is None or w_variable.value is None:
        return RSOCBaselineResult(
            status=str(model.status),
            x=None,
            y=np.zeros(problem.n),
            objective=None,
            history=[{"solver_iterations": iterations}],
            wall_time=perf_counter() - started,
            epigraph_q=None,
        )

    x = x_scale * np.asarray(z_variable.value, dtype=float).reshape(-1)
    epigraph_q = q_scale * np.asarray(
        w_variable.value, dtype=float
    ).reshape(-1)
    equality_dual = equality.dual_value
    y = np.zeros(problem.n, dtype=float)
    valid_dual = equality_dual is not None
    if valid_dual:
        dual = np.asarray(equality_dual, dtype=float).reshape(-1)
        valid_dual = bool(
            dual.shape == (problem.n - 1,) and np.all(np.isfinite(dual))
        )
        if valid_dual:
            y[:-1] = -objective_scale * row_scale * dual
            y -= float(np.mean(y))

    finite_positive = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(epigraph_q))
        and np.all(x > 0.0)
        and np.all(epigraph_q > 0.0)
    )
    slack = problem.dual_slack(y)
    stationarity = (
        problem.c - problem.mu / x**2 - problem.edge_difference(y)
        if finite_positive and valid_dual
        else np.full(problem.m, np.nan)
    )
    denominator = max(
        1.0,
        float(np.linalg.norm(problem.c, np.inf)),
        float(np.linalg.norm(problem.mu / x**2, np.inf))
        if finite_positive
        else 1.0,
    )
    history = [
        {
            "solver_iterations": iterations,
            "optimal_inaccurate": float(model.status == "optimal_inaccurate"),
            "x_scale": float(x_scale),
            "q_scale": float(q_scale),
            "objective_scale": float(objective_scale),
            "minimum_epigraph_q": (
                float(np.min(epigraph_q)) if epigraph_q.size else float("nan")
            ),
            "minimum_dual_slack": (
                float(np.min(slack)) if valid_dual else float("nan")
            ),
            "maximum_xq_shortfall": (
                float(np.max(np.maximum(0.0, 1.0 - x * epigraph_q)))
                if finite_positive
                else float("inf")
            ),
            "relative_stationarity_inf": (
                float(np.linalg.norm(stationarity, np.inf) / denominator)
                if np.all(np.isfinite(stationarity))
                else float("inf")
            ),
            "conic_objective": (
                float(problem.c @ x + problem.mu @ epigraph_q)
                if finite_positive
                else float("inf")
            ),
        }
    ]
    if not valid_dual:
        status = "invalid_equality_dual"
    elif not finite_positive:
        status = "invalid_primal"
    else:
        status = (
            "optimal"
            if model.status in {"optimal", "optimal_inaccurate"}
            else str(model.status)
        )
    return RSOCBaselineResult(
        status=status,
        x=x if finite_positive else None,
        y=y,
        objective=problem.objective(x) if finite_positive else None,
        history=history,
        wall_time=perf_counter() - started,
        epigraph_q=epigraph_q if finite_positive else None,
    )
