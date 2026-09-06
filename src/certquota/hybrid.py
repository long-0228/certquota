"""Global alternate-scaling entry followed by certified local Newton steps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np

from .baselines import (
    coordinate_scaling_sweep_vectorized,
    dual_feasible_start,
    solve_damped_newton,
)
from .certificates import (
    basin_certificate,
    dual_state,
    recover_primal,
    stopping_radius,
)
from .problem import ReciprocalTransportProblem
from .solver import (
    CertifiedNewtonOptions,
    CertifiedNewtonResult,
    _validate_certified_newton_options,
    solve_certified_newton,
)
from .tree import TreeRouter
from .intervals import (
    IntervalRecoveryCertificate,
)
from .strict_backend import prepare_strict_state, strict_recovery_certificate


Array = np.ndarray


@dataclass(frozen=True)
class HybridOptions:
    epsilon: float = 1e-10
    max_scaling_sweeps: int = 10_000
    certificate_every: int = 1
    refresh_tree_every: int = 25
    fallback_method: str = "alternate_scaling"
    time_limit_seconds: Optional[float] = None
    newton: CertifiedNewtonOptions = CertifiedNewtonOptions()


@dataclass
class HybridResult:
    status: str
    x: Optional[Array]
    y: Array
    objective: Optional[float]
    objective_gap_upper: Optional[float]
    scaling_sweeps: int
    certificate_checks: int
    certificate_hit: bool
    newton_result: Optional[CertifiedNewtonResult]
    history: List[Dict[str, float]] = field(default_factory=list)
    wall_time: float = 0.0
    final_interval_recovery: Optional[IntervalRecoveryCertificate] = None
    strict_backend: Optional[str] = None
    strict_preparation_seconds: float = 0.0
    strict_certificate_seconds: float = 0.0
    terminal_recovery_evaluations: int = 0
    terminal_recovery_accepts: int = 0
    # The endpoint certificate is tree-specific.  Preserve the actual final
    # router (which may differ from the caller's router after a refresh) so a
    # strict result continues to identify its implicit exact-real recovery.
    router: Optional[TreeRouter] = None


def solve_hybrid(
    problem: ReciprocalTransportProblem,
    y0: Optional[Array] = None,
    router: Optional[TreeRouter] = None,
    options: Optional[HybridOptions] = None,
) -> HybridResult:
    """Use scaling globally and switch only after the basin test passes.

    The paper's near-linear local theorem starts at ``certificate_hit``; the
    preceding coordinate-scaling work is measured separately.
    """

    if options is None:
        options = HybridOptions()
    if not np.isfinite(options.epsilon) or options.epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        isinstance(options.max_scaling_sweeps, (bool, np.bool_))
        or not isinstance(options.max_scaling_sweeps, (int, np.integer))
        or options.max_scaling_sweeps < 0
    ):
        raise ValueError("max_scaling_sweeps must be a nonnegative integer")
    if (
        isinstance(options.certificate_every, (bool, np.bool_))
        or isinstance(options.refresh_tree_every, (bool, np.bool_))
        or not isinstance(options.certificate_every, (int, np.integer))
        or not isinstance(options.refresh_tree_every, (int, np.integer))
        or options.certificate_every < 1
        or options.refresh_tree_every < 1
    ):
        raise ValueError("certificate and tree refresh periods must be positive")
    if options.fallback_method not in {"alternate_scaling", "damped_newton"}:
        raise ValueError(
            "fallback_method must be 'alternate_scaling' or 'damped_newton'"
        )
    if options.time_limit_seconds is not None and (
        not np.isfinite(options.time_limit_seconds)
        or options.time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and positive")
    _validate_certified_newton_options(options.newton)
    if router is not None:
        router.validate_for(problem)
    y = (
        dual_feasible_start(problem)
        if y0 is None
        else np.asarray(y0, dtype=float).copy()
    )
    if y.shape != (problem.n,) or not np.all(np.isfinite(y)):
        raise ValueError("y0 must be a finite vector of length problem.n")
    y -= float(y.mean())
    q = problem.dual_slack(y)
    if np.any(q <= 0):
        return HybridResult(
            "invalid_initial_domain", None, y, None, None, 0, 0, False, None
        )

    history: List[Dict[str, float]] = []
    checks = 0
    started = perf_counter()
    active_router = router
    rejected_strict_preparation_seconds = 0.0
    rejected_strict_certificate_seconds = 0.0
    rejected_terminal_recovery_evaluations = 0
    rejected_terminal_recovery_accepts = 0
    last_rejected_newton: Optional[CertifiedNewtonResult] = None

    for sweep in range(options.max_scaling_sweeps + 1):
        if (
            options.time_limit_seconds is not None
            and perf_counter() - started >= options.time_limit_seconds
        ):
            return HybridResult(
                "timed_out", None, y, None, None, sweep, checks,
                last_rejected_newton is not None,
                last_rejected_newton,
                history,
                perf_counter() - started,
                None,
                (
                    options.newton.strict_backend
                    if options.newton.strict_interval
                    else None
                ),
                rejected_strict_preparation_seconds,
                rejected_strict_certificate_seconds,
                rejected_terminal_recovery_evaluations,
                rejected_terminal_recovery_accepts,
            )
        state = dual_state(problem, y)
        residual_l1 = float(np.linalg.norm(state.g, 1))
        radius = stopping_radius(problem, state, options.epsilon)
        if residual_l1 <= radius:
            if active_router is None:
                active_router = TreeRouter.build(problem)
            strict_recovery = None
            strict_preparation_seconds = 0.0
            strict_certificate_seconds = 0.0
            if options.newton.strict_interval:
                preparation_started = perf_counter()
                prepared = prepare_strict_state(
                    problem,
                    y,
                    options.newton.strict_backend,
                    options.newton.interval_decimal_precision,
                )
                strict_preparation_seconds = perf_counter() - preparation_started
                candidate = strict_recovery_certificate(
                    problem,
                    y,
                    options.epsilon,
                    options.newton.strict_backend,
                    options.newton.interval_decimal_precision,
                    prepared,
                    active_router,
                )
                strict_certificate_seconds = candidate.wall_time
                if candidate.epsilon_optimal_certified:
                    strict_recovery = candidate
            if not options.newton.strict_interval or strict_recovery is not None:
                recovered = recover_primal(problem, state, active_router)
                return HybridResult(
                    "optimal_scaling",
                    recovered.x_hat,
                    y,
                    problem.objective(recovered.x_hat),
                    (
                        strict_recovery.objective_gap_upper
                        if strict_recovery is not None
                        else recovered.objective_gap_upper
                    ),
                    sweep,
                    checks,
                    False,
                    None,
                    history,
                    perf_counter() - started,
                    strict_recovery,
                    (
                        options.newton.strict_backend
                        if options.newton.strict_interval
                        else None
                    ),
                    (
                        rejected_strict_preparation_seconds
                        + strict_preparation_seconds
                    ),
                    (
                        rejected_strict_certificate_seconds
                        + strict_certificate_seconds
                    ),
                    rejected_terminal_recovery_evaluations,
                    rejected_terminal_recovery_accepts,
                    router=active_router,
                )

        if sweep % options.certificate_every == 0:
            if active_router is None or (
                sweep > 0 and sweep % options.refresh_tree_every == 0
            ):
                active_router = TreeRouter.build(
                    problem, edge_cost=1.0 / state.conductance
                )
            certificate = basin_certificate(
                problem,
                state,
                active_router,
                threshold=options.newton.basin_threshold,
            )
            checks += 1
            history.append(
                {
                    "sweep": float(sweep),
                    "residual_l1": residual_l1,
                    "eta_upper": certificate.eta_upper,
                    "certificate_passed": float(certificate.passed),
                }
            )
            if certificate.passed:
                newton_options = CertifiedNewtonOptions(
                    epsilon=options.epsilon,
                    max_outer_steps=options.newton.max_outer_steps,
                    max_refinements=options.newton.max_refinements,
                    initial_cg_tolerance=options.newton.initial_cg_tolerance,
                    tolerance_shrink=options.newton.tolerance_shrink,
                    cg_max_iterations=options.newton.cg_max_iterations,
                    basin_threshold=options.newton.basin_threshold,
                    use_direct_solve_below=options.newton.use_direct_solve_below,
                    preconditioner=options.newton.preconditioner,
                    matrix_free=options.newton.matrix_free,
                    strict_interval=options.newton.strict_interval,
                    strict_backend=options.newton.strict_backend,
                    interval_decimal_precision=options.newton.interval_decimal_precision,
                    strict_arb_refinement_below=(
                        options.newton.strict_arb_refinement_below
                    ),
                    strict_defect_refinements=(
                        options.newton.strict_defect_refinements
                    ),
                    time_limit_seconds=(
                        None
                        if options.time_limit_seconds is None
                        else max(
                            0.0,
                            options.time_limit_seconds - (perf_counter() - started),
                        )
                    ),
                )
                newton = solve_certified_newton(
                    problem, y, active_router, newton_options
                )
                if (
                    newton.status == "forcing_certificate_failed"
                    and newton_options.preconditioner == "amg"
                ):
                    # AMG can be numerically erratic on an extremely narrow
                    # bottleneck even though the point is rigorously inside the
                    # basin.  Retry from the last accepted certified point with
                    # Jacobi-PCG.  This is only a candidate generator: all basin,
                    # forcing, and recovery decisions remain strict and both
                    # attempts stay inside end-to-end time.
                    history[-1]["amg_forcing_retry"] = 1.0
                    history[-1]["amg_failed_newton_seconds"] = newton.wall_time
                    failed_strict_preparation_seconds = (
                        newton.strict_preparation_seconds
                    )
                    failed_strict_certificate_seconds = (
                        newton.strict_certificate_seconds
                    )
                    failed_terminal_recovery_evaluations = (
                        newton.terminal_recovery_evaluations
                    )
                    failed_terminal_recovery_accepts = (
                        newton.terminal_recovery_accepts
                    )
                    retry_options = replace(
                        newton_options,
                        preconditioner="jacobi",
                        strict_defect_refinements=max(
                            3, newton_options.strict_defect_refinements
                        ),
                        time_limit_seconds=(
                            None
                            if options.time_limit_seconds is None
                            else max(
                                0.0,
                                options.time_limit_seconds
                                - (perf_counter() - started),
                            )
                        ),
                    )
                    newton = solve_certified_newton(
                        problem, newton.y, active_router, retry_options
                    )
                    newton.strict_preparation_seconds += (
                        failed_strict_preparation_seconds
                    )
                    newton.strict_certificate_seconds += (
                        failed_strict_certificate_seconds
                    )
                    newton.terminal_recovery_evaluations += (
                        failed_terminal_recovery_evaluations
                    )
                    newton.terminal_recovery_accepts += (
                        failed_terminal_recovery_accepts
                    )
                    history[-1]["jacobi_retry_seconds"] = newton.wall_time
                if newton.status not in {
                    "uncertified_warm_start",
                    "forcing_certificate_failed",
                }:
                    newton.strict_preparation_seconds += (
                        rejected_strict_preparation_seconds
                    )
                    newton.strict_certificate_seconds += (
                        rejected_strict_certificate_seconds
                    )
                    newton.terminal_recovery_evaluations += (
                        rejected_terminal_recovery_evaluations
                    )
                    newton.terminal_recovery_accepts += (
                        rejected_terminal_recovery_accepts
                    )
                    return HybridResult(
                        newton.status,
                        newton.x,
                        newton.y,
                        newton.objective,
                        newton.objective_gap_upper,
                        sweep,
                        checks,
                        True,
                        newton,
                        history,
                        perf_counter() - started,
                        newton.final_interval_recovery,
                        newton.strict_backend,
                        newton.strict_preparation_seconds,
                        newton.strict_certificate_seconds,
                        newton.terminal_recovery_evaluations,
                        newton.terminal_recovery_accepts,
                        router=active_router,
                    )
                rejected_strict_preparation_seconds += (
                    newton.strict_preparation_seconds
                )
                rejected_strict_certificate_seconds += (
                    newton.strict_certificate_seconds
                )
                rejected_terminal_recovery_evaluations += (
                    newton.terminal_recovery_evaluations
                )
                rejected_terminal_recovery_accepts += (
                    newton.terminal_recovery_accepts
                )
                last_rejected_newton = newton
                y = newton.y.copy()
                q = problem.dual_slack(y)
                if newton.status == "forcing_certificate_failed":
                    history[-1]["strict_forcing_rejected"] = 1.0
                    history[-1]["forcing_rejection_seconds"] = (
                        newton.wall_time
                    )
                else:
                    history[-1]["strict_certificate_passed"] = 0.0

        if sweep == options.max_scaling_sweeps:
            break
        if options.fallback_method == "alternate_scaling":
            coordinate_scaling_sweep_vectorized(problem, y, q)
        else:
            fallback = solve_damped_newton(
                problem,
                y,
                epsilon=options.epsilon,
                max_steps=1,
                direct_solve_below=0,
                cg_tolerance=1e-10,
                router=active_router,
                preconditioner=options.newton.preconditioner,
                time_limit_seconds=(
                    None
                    if options.time_limit_seconds is None
                    else max(
                        0.0,
                        options.time_limit_seconds - (perf_counter() - started),
                    )
                ),
            )
            if fallback.status in {
                "linear_solve_failed",
                "linear_solve_nonfinite",
            }:
                # The globalizer is only a candidate generator.  A failed
                # iterative solve must not terminate an otherwise certifiable
                # run when a deterministic sparse solve is affordable.  For a
                # larger graph, make one attempt with an independent
                # preconditioner instead.  In either case the candidate returns
                # to the top of the loop, where the strict certificates alone
                # decide whether a primal solution may be returned.
                failed_fallback = fallback
                use_direct_retry = problem.n <= 5_000
                retry_preconditioner = options.newton.preconditioner
                if not use_direct_retry:
                    retry_preconditioner = (
                        "jacobi"
                        if options.newton.preconditioner in {"amg", "tree"}
                        else "tree"
                    )
                fallback = solve_damped_newton(
                    problem,
                    y,
                    epsilon=options.epsilon,
                    max_steps=1,
                    direct_solve_below=5_000 if use_direct_retry else 0,
                    cg_tolerance=1e-10,
                    router=active_router,
                    preconditioner=retry_preconditioner,
                    time_limit_seconds=(
                        None
                        if options.time_limit_seconds is None
                        else max(
                            0.0,
                            options.time_limit_seconds
                            - (perf_counter() - started),
                        )
                    ),
                )
                retry_seconds = fallback.wall_time
                fallback.wall_time += failed_fallback.wall_time
                if history:
                    history[-1][
                        f"fallback_retry_from_{failed_fallback.status}"
                    ] = 1.0
                    history[-1]["fallback_failed_solve_seconds"] = (
                        failed_fallback.wall_time
                    )
                    history[-1]["fallback_retry_seconds"] = retry_seconds
                    history[-1]["fallback_direct_retry"] = float(
                        use_direct_retry
                    )
                    history[-1]["fallback_preconditioner_retry"] = float(
                        not use_direct_retry
                    )
            if fallback.status not in {"max_steps", "optimal"}:
                return HybridResult(
                    f"fallback_{fallback.status}",
                    None,
                    fallback.y,
                    None,
                    None,
                    sweep,
                    checks,
                    last_rejected_newton is not None,
                    last_rejected_newton,
                    history,
                    perf_counter() - started,
                    None,
                    (
                        options.newton.strict_backend
                        if options.newton.strict_interval
                        else None
                    ),
                    rejected_strict_preparation_seconds,
                    rejected_strict_certificate_seconds,
                    rejected_terminal_recovery_evaluations,
                    rejected_terminal_recovery_accepts,
                )
            if history:
                history[-1]["fallback_newton_seconds"] = fallback.wall_time
                history[-1]["fallback_newton_status"] = float(
                    fallback.status == "optimal"
                )
            if fallback.status == "optimal" and not fallback.history:
                # The binary stopping rule can be looser than the selected
                # outward recovery certificate.  In that case the one-step
                # globalizer has taken no step, so retrying the identical strict
                # state would loop forever.  Fail safely with no primal return.
                if history:
                    history[-1]["fallback_stalled"] = 1.0
                return HybridResult(
                    "fallback_stalled",
                    None,
                    fallback.y,
                    None,
                    None,
                    sweep,
                    checks,
                    last_rejected_newton is not None,
                    last_rejected_newton,
                    history,
                    perf_counter() - started,
                    None,
                    (
                        options.newton.strict_backend
                        if options.newton.strict_interval
                        else None
                    ),
                    rejected_strict_preparation_seconds,
                    rejected_strict_certificate_seconds,
                    rejected_terminal_recovery_evaluations,
                    rejected_terminal_recovery_accepts,
                )
            y = fallback.y.copy()
            q = problem.dual_slack(y)

    return HybridResult(
        "max_scaling_sweeps",
        None,
        y,
        None,
        None,
        options.max_scaling_sweeps,
        checks,
        last_rejected_newton is not None,
        last_rejected_newton,
        history,
        perf_counter() - started,
        None,
        (
            options.newton.strict_backend
            if options.newton.strict_interval
            else None
        ),
        rejected_strict_preparation_seconds,
        rejected_strict_certificate_seconds,
        rejected_terminal_recovery_evaluations,
        rejected_terminal_recovery_accepts,
    )
