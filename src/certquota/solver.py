"""Certified inexact Laplacian Newton implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse.linalg as spla

try:
    import pyamg
except ImportError:  # pragma: no cover - optional benchmark dependency
    pyamg = None

from .certificates import (
    BasinCertificate,
    ForcingCertificate,
    basin_certificate,
    dual_state,
    forcing_certificate,
    hessian,
    recover_primal,
    stopping_radius,
)
from .problem import ReciprocalTransportProblem
from .tree import TreeRouter
from .intervals import (
    IntervalBasinCertificate,
    IntervalForcingCertificate,
    IntervalRecoveryCertificate,
    interval_newton_direction,
)
from .strict_backend import (
    prepare_strict_state,
    strict_basin_certificate,
    strict_forcing_certificate,
    strict_linear_residual_candidate,
    strict_recovery_certificate,
    validate_strict_backend,
)


Array = np.ndarray
_CONTRACTION_RATIO_NUMERATOR = 2.0
_CONTRACTION_RATIO_DENOMINATOR = 5.0


def contraction_threshold_lower(decrement_lower: float) -> float:
    """Outward lower bound on ``(2/5) * decrement_lower`` in binary64."""

    if decrement_lower <= 0.0 or not np.isfinite(decrement_lower):
        return 0.0
    doubled_lower = np.nextafter(
        np.float64(decrement_lower) * _CONTRACTION_RATIO_NUMERATOR,
        -np.inf,
    )
    return float(
        max(
            0.0,
            np.nextafter(
                doubled_lower / _CONTRACTION_RATIO_DENOMINATOR,
                -np.inf,
            ),
        )
    )


def strict_forcing_mode(
    certificate: Optional[IntervalForcingCertificate],
) -> Optional[str]:
    """Return the A31-safe acceptance tier for one strict certificate."""

    if certificate is None:
        return None
    if certificate.passed:
        return "quadratic"
    threshold = contraction_threshold_lower(certificate.decrement_lower)
    if (
        np.isfinite(certificate.residual_upper)
        and certificate.residual_upper <= threshold
    ):
        return "contraction"
    return None


@dataclass(frozen=True)
class CertifiedNewtonOptions:
    epsilon: float = 1e-10
    max_outer_steps: int = 30
    max_refinements: int = 12
    initial_cg_tolerance: float = 1e-2
    tolerance_shrink: float = 0.1
    cg_max_iterations: Optional[int] = None
    basin_threshold: float = 0.1
    use_direct_solve_below: int = 0
    preconditioner: str = "jacobi"
    matrix_free: bool = True
    strict_interval: bool = False
    strict_backend: str = "arb"
    interval_decimal_precision: int = 30
    strict_arb_refinement_below: int = 256
    strict_defect_refinements: int = 3
    time_limit_seconds: Optional[float] = None


def _validate_certified_newton_options(
    options: CertifiedNewtonOptions,
) -> None:
    """Validate every option used by a path screen or public solve entry."""

    if not np.isfinite(options.epsilon) or options.epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    integer_fields = (
        ("max_outer_steps", options.max_outer_steps, 0),
        ("max_refinements", options.max_refinements, 0),
        ("use_direct_solve_below", options.use_direct_solve_below, 0),
        (
            "strict_arb_refinement_below",
            options.strict_arb_refinement_below,
            0,
        ),
        ("strict_defect_refinements", options.strict_defect_refinements, 0),
    )
    for name, value, minimum in integer_fields:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if (
        not np.isfinite(options.initial_cg_tolerance)
        or options.initial_cg_tolerance <= 0.0
    ):
        raise ValueError("initial_cg_tolerance must be finite and positive")
    if (
        not np.isfinite(options.tolerance_shrink)
        or not 0.0 < options.tolerance_shrink < 1.0
    ):
        raise ValueError("tolerance_shrink must be finite and in (0, 1)")
    if options.cg_max_iterations is not None and (
        isinstance(options.cg_max_iterations, (bool, np.bool_))
        or not isinstance(options.cg_max_iterations, (int, np.integer))
        or options.cg_max_iterations < 1
    ):
        raise ValueError("cg_max_iterations must be a positive integer")
    if (
        not np.isfinite(options.basin_threshold)
        or not 0.0 < options.basin_threshold <= 0.1
    ):
        raise ValueError("basin_threshold must be finite and in (0, 0.1]")
    if options.preconditioner not in {"tree", "jacobi", "amg"}:
        raise ValueError("preconditioner must be 'tree', 'jacobi', or 'amg'")
    if (
        isinstance(options.interval_decimal_precision, (bool, np.bool_))
        or not isinstance(options.interval_decimal_precision, (int, np.integer))
        or options.interval_decimal_precision < 16
    ):
        raise ValueError("interval_decimal_precision must be at least 16")
    if options.time_limit_seconds is not None and (
        not np.isfinite(options.time_limit_seconds)
        or options.time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")


@dataclass
class CertifiedNewtonResult:
    status: str
    x: Optional[Array]
    y: Array
    objective: Optional[float]
    objective_gap_upper: Optional[float]
    initial_basin: BasinCertificate
    initial_interval_basin: Optional[IntervalBasinCertificate] = None
    final_interval_recovery: Optional[IntervalRecoveryCertificate] = None
    history: List[Dict[str, float | str]] = field(default_factory=list)
    wall_time: float = 0.0
    strict_backend: Optional[str] = None
    strict_preparation_seconds: float = 0.0
    strict_certificate_seconds: float = 0.0
    terminal_recovery_evaluations: int = 0
    terminal_recovery_accepts: int = 0


def _cg_solve(
    h_red,
    rhs: Array,
    tolerance: float,
    max_iterations: int,
    x0: Optional[Array],
    preconditioner_matvec=None,
    diagonal: Optional[Array] = None,
    deadline: Optional[float] = None,
):
    if preconditioner_matvec is None:
        if diagonal is None:
            diagonal = h_red.diagonal()
        diagonal = np.asarray(diagonal, dtype=float)
        safe_diagonal = np.maximum(diagonal, np.finfo(float).tiny)
        preconditioner = spla.LinearOperator(
            h_red.shape, matvec=lambda z: z / safe_diagonal, dtype=float
        )
    else:
        preconditioner = spla.LinearOperator(
            h_red.shape, matvec=preconditioner_matvec, dtype=float
        )
    counter = {"iterations": 0}
    latest = {
        "solution": (
            np.zeros_like(rhs) if x0 is None else np.asarray(x0).copy()
        )
    }

    class LinearSolveDeadlineReached(Exception):
        pass

    def callback(xk):
        counter["iterations"] += 1
        latest["solution"] = np.asarray(xk, dtype=float).copy()
        if deadline is not None and perf_counter() >= deadline:
            raise LinearSolveDeadlineReached

    # SciPy <1.12 uses tol; newer versions accept rtol.  The project supports both.
    try:
        try:
            solution, info = spla.cg(
                h_red,
                rhs,
                x0=x0,
                rtol=tolerance,
                atol=0.0,
                maxiter=max_iterations,
                M=preconditioner,
                callback=callback,
            )
        except TypeError:
            solution, info = spla.cg(
                h_red,
                rhs,
                x0=x0,
                tol=tolerance,
                atol=0.0,
                maxiter=max_iterations,
                M=preconditioner,
                callback=callback,
            )
    except LinearSolveDeadlineReached:
        return latest["solution"], -99, int(counter["iterations"])
    return solution, int(info), int(counter["iterations"])


def solve_certified_newton(
    problem: ReciprocalTransportProblem,
    y0: Array,
    router: Optional[TreeRouter] = None,
    options: Optional[CertifiedNewtonOptions] = None,
) -> CertifiedNewtonResult:
    """Solve from an already certified warm start.

    Ordinary floating-point arithmetic is used here.  Thus ``passed`` means that
    the mathematical certificate was evaluated numerically; a strict machine-level
    certificate additionally requires outward-rounded interval arithmetic.
    """

    if options is None:
        options = CertifiedNewtonOptions()
    _validate_certified_newton_options(options)
    if options.strict_interval:
        validate_strict_backend(options.strict_backend)
    y0_values = np.asarray(y0, dtype=float)
    if y0_values.shape != (problem.n,) or not np.all(np.isfinite(y0_values)):
        raise ValueError("y0 must be a finite vector of length problem.n")
    if router is not None:
        router.validate_for(problem)
    if router is None:
        initial_state = dual_state(problem, y0_values)
        router = TreeRouter.build(
            problem, edge_cost=1.0 / initial_state.conductance
        )
    y = y0_values.copy()
    y -= float(y.mean())
    state = dual_state(problem, y)
    initial_basin = basin_certificate(
        problem, state, router, options.basin_threshold
    )
    initial_interval_basin = None
    prepared_interval_state = None
    strict_preparation_seconds = 0.0
    strict_certificate_seconds = 0.0
    if options.strict_interval:
        preparation_started = perf_counter()
        prepared_interval_state = prepare_strict_state(
            problem,
            y,
            options.strict_backend,
            options.interval_decimal_precision,
        )
        strict_preparation_seconds += perf_counter() - preparation_started
        initial_interval_basin = strict_basin_certificate(
            problem,
            y,
            router,
            options.basin_threshold,
            options.strict_backend,
            options.interval_decimal_precision,
            prepared_interval_state,
        )
        strict_certificate_seconds += initial_interval_basin.wall_time
    result = CertifiedNewtonResult(
        status="uncertified_warm_start",
        x=None,
        y=y,
        objective=None,
        objective_gap_upper=None,
        initial_basin=initial_basin,
        initial_interval_basin=initial_interval_basin,
        strict_backend=(options.strict_backend if options.strict_interval else None),
        strict_preparation_seconds=strict_preparation_seconds,
        strict_certificate_seconds=strict_certificate_seconds,
    )
    if not initial_basin.passed or (
        initial_interval_basin is not None and not initial_interval_basin.passed
    ):
        return result

    started = perf_counter()
    deadline = (
        None
        if options.time_limit_seconds is None
        else started + options.time_limit_seconds
    )
    n = problem.n
    keep = np.arange(n - 1)
    history: List[Dict[str, float | str]] = []

    for outer in range(options.max_outer_steps + 1):
        if (
            options.time_limit_seconds is not None
            and perf_counter() - started >= options.time_limit_seconds
        ):
            result.status = "timed_out"
            result.y = y
            result.history = history
            result.wall_time = perf_counter() - started
            return result
        state = dual_state(problem, y)
        if options.strict_interval and prepared_interval_state is None:
            preparation_started = perf_counter()
            prepared_interval_state = prepare_strict_state(
                problem,
                y,
                options.strict_backend,
                options.interval_decimal_precision,
            )
            result.strict_preparation_seconds += (
                perf_counter() - preparation_started
            )
        residual_l1 = float(np.linalg.norm(state.g, 1))
        stop_radius = stopping_radius(problem, state, options.epsilon)
        if residual_l1 <= stop_radius:
            interval_recovery = None
            if options.strict_interval:
                interval_recovery = strict_recovery_certificate(
                    problem,
                    y,
                    options.epsilon,
                    options.strict_backend,
                    options.interval_decimal_precision,
                    prepared_interval_state,
                    router,
                )
                result.strict_certificate_seconds += interval_recovery.wall_time
                if not interval_recovery.epsilon_optimal_certified:
                    interval_recovery = None
                else:
                    result.final_interval_recovery = interval_recovery
            if options.strict_interval and interval_recovery is None:
                # The binary64 diagnostic is not enough to stop.  Continue to
                # improve the point, or report a forcing failure if machine
                # precision is insufficient for the requested epsilon.
                pass
            else:
                recovered = recover_primal(problem, state, router)
                feasibility = float(
                    np.linalg.norm(problem.primal_residual(recovered.x_hat), 1)
                )
                history.append(
                    {
                        "outer": float(outer),
                        "residual_l1": residual_l1,
                        "stop_radius": stop_radius,
                        "feasibility_l1": feasibility,
                        "objective_gap_upper": recovered.objective_gap_upper,
                        "interval_gap_upper": (
                            interval_recovery.objective_gap_upper
                            if interval_recovery is not None
                            else float("nan")
                        ),
                        "event": 1.0,
                    }
                )
                result.status = "optimal"
                result.x = recovered.x_hat
                result.y = y
                result.objective = problem.objective(recovered.x_hat)
                result.objective_gap_upper = (
                    interval_recovery.objective_gap_upper
                    if interval_recovery is not None
                    else recovered.objective_gap_upper
                )
                result.history = history
                result.wall_time = perf_counter() - started
                return result

        if outer == options.max_outer_steps:
            break

        rhs = -state.g[keep]
        max_iterations = options.cg_max_iterations or max(50, 5 * (n - 1))
        reduced_direction = None
        accepted: Optional[ForcingCertificate] = None
        accepted_interval: Optional[IntervalForcingCertificate] = None
        accepted_mode: Optional[str] = None
        total_cg_iterations = 0
        strict_defect_steps = 0
        strict_interval_evaluations = 0
        binary_screened_candidates = 0
        tolerance = options.initial_cg_tolerance

        use_direct = bool(
            options.use_direct_solve_below
            and n <= options.use_direct_solve_below
        )
        setup_started = perf_counter()
        explicit_h = hessian(problem, state) if (
            use_direct or not options.matrix_free or options.preconditioner == "amg"
        ) else None
        if explicit_h is not None:
            h_red = explicit_h[keep][:, keep].tocsr()
            reduced_diagonal = h_red.diagonal()
        else:
            reduced_diagonal = problem.laplacian_diagonal(
                state.conductance
            )[keep]

            def reduced_matvec(z):
                full = np.zeros(n, dtype=float)
                full[keep] = z
                return problem.laplacian_matvec(
                    full, state.conductance
                )[keep]

            h_red = spla.LinearOperator(
                (n - 1, n - 1), matvec=reduced_matvec, dtype=float
            )
        amg_preconditioner = None
        if options.preconditioner == "amg":
            if pyamg is None:
                raise RuntimeError("the AMG preconditioner requires pyamg")
            hierarchy = pyamg.smoothed_aggregation_solver(
                h_red, symmetry="symmetric"
            )
            amg_operator = hierarchy.aspreconditioner(cycle="V")
            amg_preconditioner = amg_operator.matvec
        linear_setup_seconds = perf_counter() - setup_started

        for refinement in range(options.max_refinements):
            if (
                options.time_limit_seconds is not None
                and perf_counter() - started >= options.time_limit_seconds
            ):
                result.status = "timed_out"
                result.y = y
                result.history = history
                result.wall_time = perf_counter() - started
                return result
            if use_direct:
                if (
                    refinement == 1
                    and options.strict_interval
                    and options.strict_backend == "arb"
                    and problem.n <= options.strict_arb_refinement_below
                ):
                    full_direction = interval_newton_direction(
                        problem,
                        y,
                        options.interval_decimal_precision,
                    )
                    reduced_direction = (
                        full_direction[keep] - full_direction[-1]
                    )
                elif refinement > 0:
                    break
                else:
                    reduced_direction = spla.spsolve(h_red.tocsc(), rhs)
                info, cg_iterations = 0, 0
            else:
                if options.preconditioner == "tree":
                    def apply_tree_preconditioner(z):
                        full_demand = np.zeros(n, dtype=float)
                        full_demand[keep] = z
                        full_demand[-1] = -float(np.sum(z))
                        potential = router.solve_laplacian(
                            full_demand, state.conductance
                        )
                        potential -= potential[-1]
                        return potential[keep]

                    preconditioner_matvec = apply_tree_preconditioner
                elif options.preconditioner == "jacobi":
                    preconditioner_matvec = None
                elif options.preconditioner == "amg":
                    preconditioner_matvec = amg_preconditioner
                else:
                    raise ValueError(
                        "preconditioner must be 'tree', 'jacobi', or 'amg'"
                    )
                reduced_direction, info, cg_iterations = _cg_solve(
                    h_red,
                    rhs,
                    tolerance,
                    max_iterations,
                    reduced_direction,
                    preconditioner_matvec,
                    reduced_diagonal,
                    deadline,
                )
                if info == -99:
                    result.status = "timed_out"
                    result.y = y
                    result.history = history
                    result.wall_time = perf_counter() - started
                    return result
            total_cg_iterations += cg_iterations
            direction = np.zeros(n, dtype=float)
            direction[keep] = reduced_direction
            direction -= float(direction.mean())
            accepted = forcing_certificate(
                problem, state, direction, router, explicit_h
            )
            strict_replay_warranted = bool(
                refinement == options.max_refinements - 1
            )
            if options.strict_interval and strict_replay_warranted:
                accepted_interval = strict_forcing_certificate(
                    problem,
                    y,
                    direction,
                    router,
                    options.strict_backend,
                    options.interval_decimal_precision,
                    prepared_interval_state,
                )
                result.strict_certificate_seconds += accepted_interval.wall_time
                strict_interval_evaluations += 1
                accepted_mode = strict_forcing_mode(accepted_interval)
            elif options.strict_interval:
                # A failed binary64 check is used only as a conservative screen:
                # it can postpone a candidate but can never authorize one.  The
                # explicitly selected strict backend remains authoritative.
                accepted_interval = None
                binary_screened_candidates += 1
            if (
                accepted_mode is not None
                if options.strict_interval
                else accepted.passed
            ):
                break
            tolerance *= options.tolerance_shrink
            if info < 0:
                break

        if (
            options.strict_interval
            and reduced_direction is not None
            and accepted_mode is None
        ):
            for defect_refinement in range(options.strict_defect_refinements):
                if (
                    options.time_limit_seconds is not None
                    and perf_counter() - started >= options.time_limit_seconds
                ):
                    result.status = "timed_out"
                    result.y = y
                    result.history = history
                    result.wall_time = perf_counter() - started
                    return result
                direction = np.zeros(n, dtype=float)
                direction[keep] = reduced_direction
                direction -= float(direction.mean())
                strict_residual = strict_linear_residual_candidate(
                    problem,
                    y,
                    direction,
                    options.strict_backend,
                    options.interval_decimal_precision,
                    prepared_interval_state,
                    state,
                )
                correction_rhs = -strict_residual[keep]
                if use_direct:
                    correction = spla.spsolve(
                        h_red.tocsc(), correction_rhs
                    )
                    info, correction_iterations = 0, 0
                else:
                    if options.preconditioner == "tree":
                        def correction_tree_preconditioner(z):
                            full_demand = np.zeros(n, dtype=float)
                            full_demand[keep] = z
                            full_demand[-1] = -float(np.sum(z))
                            potential = router.solve_laplacian(
                                full_demand, state.conductance
                            )
                            potential -= potential[-1]
                            return potential[keep]

                        correction_preconditioner = (
                            correction_tree_preconditioner
                        )
                    elif options.preconditioner == "amg":
                        correction_preconditioner = amg_preconditioner
                    else:
                        correction_preconditioner = None
                    correction, info, correction_iterations = _cg_solve(
                        h_red,
                        correction_rhs,
                        min(1e-12, options.initial_cg_tolerance),
                        max_iterations,
                        None,
                        correction_preconditioner,
                        reduced_diagonal,
                        deadline,
                    )
                    if info == -99:
                        result.status = "timed_out"
                        result.y = y
                        result.history = history
                        result.wall_time = perf_counter() - started
                        return result
                # A positive CG ``info`` means the requested internal tolerance
                # was not reached within the iteration cap; it does not make the
                # finite candidate unsafe.  As in the main inexact solve, the
                # outward-rounded posterior forcing test is authoritative.
                # Negative status or non-finite arithmetic remains unusable.
                if info < 0 or not np.all(np.isfinite(correction)):
                    break
                reduced_direction = reduced_direction + correction
                total_cg_iterations += correction_iterations
                strict_defect_steps += 1
                direction = np.zeros(n, dtype=float)
                direction[keep] = reduced_direction
                direction -= float(direction.mean())
                accepted = forcing_certificate(
                    problem, state, direction, router, explicit_h
                )
                # Defect correction exists specifically for cases in which
                # binary64 cancellation can disagree with the strict replay.
                # Its bounded candidate set is therefore always checked by the
                # explicitly selected strict backend.
                accepted_interval = strict_forcing_certificate(
                    problem,
                    y,
                    direction,
                    router,
                    options.strict_backend,
                    options.interval_decimal_precision,
                    prepared_interval_state,
                )
                result.strict_certificate_seconds += accepted_interval.wall_time
                strict_interval_evaluations += 1
                accepted_mode = strict_forcing_mode(accepted_interval)
                if accepted_mode is not None:
                    break

        forcing_passed = bool(
            accepted_mode is not None
        ) if options.strict_interval else bool(
            accepted is not None and accepted.passed
        )
        if not options.strict_interval and forcing_passed:
            accepted_mode = "quadratic"
        if accepted is None or not forcing_passed:
            # A forcing rejection forbids continuing from the candidate, but it
            # need not forbid terminating there.  The independent strict primal
            # recovery theorem is sufficient on its own and does not assume that
            # the rejected direction satisfies either Newton recurrence.
            if (
                options.strict_interval
                and accepted is not None
                and reduced_direction is not None
                and np.all(np.isfinite(direction))
            ):
                terminal_y = y + direction
                terminal_y -= float(terminal_y.mean())
                preparation_started = perf_counter()
                terminal_prepared = prepare_strict_state(
                    problem,
                    terminal_y,
                    options.strict_backend,
                    options.interval_decimal_precision,
                )
                result.strict_preparation_seconds += (
                    perf_counter() - preparation_started
                )
                terminal_recovery = strict_recovery_certificate(
                    problem,
                    terminal_y,
                    options.epsilon,
                    options.strict_backend,
                    options.interval_decimal_precision,
                    terminal_prepared,
                    router,
                )
                result.strict_certificate_seconds += terminal_recovery.wall_time
                result.terminal_recovery_evaluations += 1
                if terminal_recovery.epsilon_optimal_certified:
                    terminal_state = dual_state(problem, terminal_y)
                    recovered = recover_primal(
                        problem, terminal_state, router
                    )
                    feasibility = float(
                        np.linalg.norm(
                            problem.primal_residual(recovered.x_hat), 1
                        )
                    )
                    history.append(
                        {
                            "outer": float(outer),
                            "residual_l1": residual_l1,
                            "stop_radius": stop_radius,
                            "forcing_residual_upper": accepted.residual_upper,
                            "forcing_threshold": accepted.threshold,
                            "interval_forcing_residual_upper": (
                                accepted_interval.residual_upper
                                if accepted_interval is not None
                                else float("inf")
                            ),
                            "interval_forcing_threshold_lower": (
                                accepted_interval.threshold_lower
                                if accepted_interval is not None
                                else 0.0
                            ),
                            "interval_contraction_threshold_lower": (
                                contraction_threshold_lower(
                                    accepted_interval.decrement_lower
                                )
                                if accepted_interval is not None
                                else 0.0
                            ),
                            "forcing_mode": "terminal_recovery",
                            "terminal_recovery_residual_l1_upper": (
                                terminal_recovery.residual_l1_upper
                            ),
                            "terminal_recovery_gap_upper": (
                                terminal_recovery.objective_gap_upper
                            ),
                            "feasibility_l1": feasibility,
                            "cg_iterations": float(total_cg_iterations),
                            "strict_defect_refinements": float(
                                strict_defect_steps
                            ),
                            "strict_interval_evaluations": float(
                                strict_interval_evaluations
                            ),
                            "binary_screened_candidates": float(
                                binary_screened_candidates
                            ),
                            "event": 2.0,
                        }
                    )
                    result.status = "optimal"
                    result.x = recovered.x_hat
                    result.y = terminal_y
                    result.objective = problem.objective(recovered.x_hat)
                    result.objective_gap_upper = (
                        terminal_recovery.objective_gap_upper
                    )
                    result.final_interval_recovery = terminal_recovery
                    result.terminal_recovery_accepts += 1
                    result.history = history
                    result.wall_time = perf_counter() - started
                    return result
            history.append(
                {
                    "outer": float(outer),
                    "residual_l1": residual_l1,
                    "stop_radius": stop_radius,
                    "forcing_residual_upper": (
                        accepted.residual_upper
                        if accepted is not None
                        else float("inf")
                    ),
                    "forcing_threshold": (
                        accepted.threshold
                        if accepted is not None
                        else 0.0
                    ),
                    "interval_forcing_residual_upper": (
                        accepted_interval.residual_upper
                        if accepted_interval is not None
                        else float("inf")
                    ),
                    "interval_forcing_threshold_lower": (
                        accepted_interval.threshold_lower
                        if accepted_interval is not None
                        else 0.0
                    ),
                    "interval_contraction_threshold_lower": (
                        contraction_threshold_lower(
                            accepted_interval.decrement_lower
                        )
                        if accepted_interval is not None
                        else 0.0
                    ),
                    "forcing_mode": "rejected",
                    "terminal_recovery_evaluations": float(
                        result.terminal_recovery_evaluations
                    ),
                    "cg_iterations": float(total_cg_iterations),
                    "strict_defect_refinements": float(strict_defect_steps),
                    "strict_interval_evaluations": float(
                        strict_interval_evaluations
                    ),
                    "binary_screened_candidates": float(
                        binary_screened_candidates
                    ),
                    "event": -1.0,
                }
            )
            result.status = "forcing_certificate_failed"
            result.y = y
            result.history = history
            result.wall_time = perf_counter() - started
            return result

        relative_slack_step = float(
            np.max(np.abs(problem.edge_difference(direction)) / state.q)
        )
        y_new = y + direction
        y_new -= float(y_new.mean())
        if np.any(problem.dual_slack(y_new) <= 0):
            result.status = "domain_violation"
            result.y = y
            result.history = history
            result.wall_time = perf_counter() - started
            return result
        history.append(
            {
                "outer": float(outer),
                "residual_l1": residual_l1,
                "stop_radius": stop_radius,
                "forcing_residual_upper": accepted.residual_upper,
                "forcing_threshold": accepted.threshold,
                "interval_forcing_residual_upper": (
                    accepted_interval.residual_upper
                    if accepted_interval is not None
                    else float("nan")
                ),
                "interval_forcing_threshold_lower": (
                    accepted_interval.threshold_lower
                    if accepted_interval is not None
                    else float("nan")
                ),
                "interval_contraction_threshold_lower": (
                    contraction_threshold_lower(
                        accepted_interval.decrement_lower
                    )
                    if accepted_interval is not None
                    else float("nan")
                ),
                "forcing_mode": accepted_mode or "quadratic",
                "decrement_lower": accepted.decrement_lower,
                "eta_lower": accepted.eta_lower,
                "relative_slack_step": relative_slack_step,
                "cg_iterations": float(total_cg_iterations),
                "strict_defect_refinements": float(strict_defect_steps),
                "strict_interval_evaluations": float(
                    strict_interval_evaluations
                ),
                "binary_screened_candidates": float(
                    binary_screened_candidates
                ),
                "linear_setup_seconds": linear_setup_seconds,
                "event": 0.0,
            }
        )
        y = y_new
        prepared_interval_state = None

    result.status = "max_outer_steps"
    result.y = y
    result.history = history
    result.wall_time = perf_counter() - started
    return result
