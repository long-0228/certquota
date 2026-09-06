"""Generate--certify--repair with an authoritative endpoint verifier.

Every optimizer used here is an untrusted candidate generator.  Only a freshly
prepared strict state followed by :func:`strict_recovery_certificate` can
authorize a fast-path return.  The certified object is the exact-real tree
recovery implicitly represented by ``(problem, result.y, result.router,
result.strict_certificate)``.  ``result.x`` is merely its binary64 numerical
realization and is accompanied by a measured marginal residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable, List, Optional

import numpy as np

from . import baselines as _baselines
from .baselines import (
    BaselineResult,
    dual_feasible_start,
    solve_alternating_scaling,
    solve_damped_newton,
)
from .certificates import dual_state
from .hybrid import HybridOptions, HybridResult, solve_hybrid
from .intervals import IntervalRecoveryCertificate, _immutable_float_snapshot
from .problem import ReciprocalTransportProblem
from .solver import CertifiedNewtonOptions
from .strict_backend import (
    prepare_strict_recovery_state,
    strict_recovery_certificate,
)
from .tree import TreeRouter


Array = np.ndarray


@dataclass(frozen=True)
class VerifiedOptions:
    """Options for the frozen CertQuota-V candidate cascade."""

    epsilon: float = 1e-10
    candidate_margin_divisor: float = 16.0
    repair_margin_divisor: float = 256.0
    candidate_max_steps: int = 100
    candidate_cg_tolerance: float = 1e-10
    sparse_direct_max_reduced_dimension: int = 5_000
    max_scaling_sweeps: int = 10_000
    certificate_every: int = 1
    refresh_tree_every: int = 25
    strict_backend: str = "ieee754"
    interval_decimal_precision: int = 70
    time_limit_seconds: Optional[float] = None
    fallback_newton: CertifiedNewtonOptions = CertifiedNewtonOptions()


@dataclass(frozen=True)
class VerifiedAttempt:
    """One untrusted generation followed by an independent strict replay."""

    method: str
    target_epsilon: float
    candidate_status: str
    verifier_status: str
    candidate_seconds: float
    prepare_seconds: float
    verify_seconds: float
    authorized: bool
    strict_certificate: Optional[IntervalRecoveryCertificate] = None
    error_type: Optional[str] = None
    invoked: bool = True


@dataclass(frozen=True)
class VerifiedResult:
    """CertQuota-V result; ``x`` is not claimed bitwise exactly feasible."""

    status: str
    problem: ReciprocalTransportProblem
    x: Optional[Array]
    y: Array
    objective: Optional[float]
    objective_gap_upper: Optional[float]
    requested_epsilon: float
    method: Optional[str]
    strict_certificate: Optional[IntervalRecoveryCertificate]
    router: Optional[TreeRouter]
    realization_residual_l1: Optional[float]
    candidate_seconds: float
    prepare_seconds: float
    verify_seconds: float
    fallback_seconds: float
    materialization_seconds: float
    wall_time: float
    attempts: tuple[VerifiedAttempt, ...] = field(default_factory=tuple)
    fallback_result: Optional[HybridResult] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, ReciprocalTransportProblem):
            raise TypeError("problem must be a ReciprocalTransportProblem")
        y_snapshot = _immutable_float_snapshot(self.y)
        x_snapshot = (
            None if self.x is None else _immutable_float_snapshot(self.x)
        )
        object.__setattr__(self, "y", y_snapshot)
        object.__setattr__(self, "x", x_snapshot)
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if self.status != "optimal":
            return
        if (
            y_snapshot.shape != (self.problem.n,)
            or not np.all(np.isfinite(y_snapshot))
            or x_snapshot is None
            or x_snapshot.shape != (self.problem.m,)
            or not np.all(np.isfinite(x_snapshot))
            or np.any(x_snapshot <= 0.0)
            or self.router is None
            or self.strict_certificate is None
        ):
            raise ValueError("optimal result lacks a finite bound replay snapshot")
        self.router.validate_for(self.problem)
        if not _certificate_authorizes(
            self.strict_certificate,
            self.requested_epsilon,
            self.problem,
            y_snapshot,
            self.router,
        ):
            raise ValueError("optimal result carries an unbound certificate")
        if (
            self.objective_gap_upper
            != self.strict_certificate.objective_gap_upper
        ):
            raise ValueError("result gap disagrees with its strict certificate")

    @property
    def attempt_count(self) -> int:
        """Return the number of candidate/fallback routines actually invoked."""

        return sum(int(attempt.invoked) for attempt in self.attempts)


@dataclass(frozen=True)
class _AttemptOutcome:
    attempt: VerifiedAttempt
    y_for_next: Array
    x: Optional[Array]
    objective: Optional[float]
    realization_residual_l1: Optional[float]
    materialization_seconds: float


def _remaining_seconds(
    started: float, time_limit_seconds: Optional[float]
) -> Optional[float]:
    if time_limit_seconds is None:
        return None
    return max(0.0, time_limit_seconds - (perf_counter() - started))


def _candidate_y_or_none(
    problem: ReciprocalTransportProblem, result: object
) -> Optional[Array]:
    value = getattr(result, "y", None)
    if value is None:
        return None
    try:
        y = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if y.shape != (problem.n,) or not np.all(np.isfinite(y)):
        return None
    return y.copy()


def _binary_domain_valid(
    problem: ReciprocalTransportProblem, y: Array
) -> bool:
    try:
        q = problem.dual_slack(y)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(q)) and np.all(q > 0.0))


def _certificate_authorizes(
    certificate: Optional[IntervalRecoveryCertificate],
    epsilon: float,
    problem: ReciprocalTransportProblem,
    y: Array,
    router: TreeRouter | None,
) -> bool:
    return bool(
        np.isfinite(epsilon)
        and epsilon > 0.0
        and certificate is not None
        and certificate.positivity_certified
        and certificate.epsilon_optimal_certified
        and np.isfinite(certificate.objective_gap_upper)
        and certificate.objective_gap_upper <= epsilon
        and certificate.is_bound_to(problem, y, epsilon, router)
    )


def _materialize_float_recovery(
    problem: ReciprocalTransportProblem,
    y: Array,
    router: TreeRouter,
) -> tuple[Array, float, float]:
    """Materialize the implicit exact-real recovery for reporting only."""

    router.validate_for(problem)
    state = dual_state(problem, y)
    x = np.asarray(state.x + router.route(-state.g), dtype=float)
    if (
        x.shape != (problem.m,)
        or not np.all(np.isfinite(x))
        or np.any(x <= 0.0)
    ):
        raise ArithmeticError("binary64 recovery is not finite and positive")
    objective = problem.objective(x)
    if not np.isfinite(objective):
        raise ArithmeticError("binary64 recovery has a nonfinite objective")
    residual_l1 = float(np.linalg.norm(problem.primal_residual(x), 1))
    return x.copy(), float(objective), residual_l1


def _run_and_verify(
    problem: ReciprocalTransportProblem,
    start_y: Array,
    router: TreeRouter,
    method: str,
    target_epsilon: float,
    epsilon: float,
    strict_backend: str,
    decimal_precision: int,
    topology_owner: ReciprocalTransportProblem,
    generator: Callable[[], BaselineResult],
) -> _AttemptOutcome:
    candidate_started = perf_counter()
    try:
        generated = generator()
    except Exception as exc:  # an untrusted generator must fail closed
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status="exception",
            verifier_status="not_run",
            candidate_seconds=perf_counter() - candidate_started,
            prepare_seconds=0.0,
            verify_seconds=0.0,
            authorized=False,
            error_type=type(exc).__name__,
        )
        return _AttemptOutcome(attempt, start_y.copy(), None, None, None, 0.0)

    candidate_seconds = perf_counter() - candidate_started
    candidate_status = str(getattr(generated, "status", "unknown"))
    candidate_y = _candidate_y_or_none(problem, generated)
    if candidate_y is None:
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status=candidate_status,
            verifier_status="invalid_candidate",
            candidate_seconds=candidate_seconds,
            prepare_seconds=0.0,
            verify_seconds=0.0,
            authorized=False,
        )
        return _AttemptOutcome(attempt, start_y.copy(), None, None, None, 0.0)
    next_y = (
        candidate_y.copy()
        if _binary_domain_valid(problem, candidate_y)
        else start_y.copy()
    )

    prepare_started = perf_counter()
    try:
        prepared = prepare_strict_recovery_state(
            problem,
            candidate_y,
            strict_backend,
            decimal_precision,
            topology_owner,
        )
    except Exception as exc:
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status=candidate_status,
            verifier_status="preparation_failed",
            candidate_seconds=candidate_seconds,
            prepare_seconds=perf_counter() - prepare_started,
            verify_seconds=0.0,
            authorized=False,
            error_type=type(exc).__name__,
        )
        return _AttemptOutcome(attempt, next_y, None, None, None, 0.0)
    prepare_seconds = perf_counter() - prepare_started

    verify_started = perf_counter()
    try:
        certificate = strict_recovery_certificate(
            problem,
            candidate_y,
            epsilon,
            strict_backend,
            decimal_precision,
            prepared,
            router,
        )
    except Exception as exc:
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status=candidate_status,
            verifier_status="verification_failed",
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=perf_counter() - verify_started,
            authorized=False,
            error_type=type(exc).__name__,
        )
        return _AttemptOutcome(attempt, next_y, None, None, None, 0.0)
    verify_seconds = perf_counter() - verify_started

    if not _certificate_authorizes(
        certificate, epsilon, problem, candidate_y, router
    ):
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status=candidate_status,
            verifier_status="rejected",
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            authorized=False,
            strict_certificate=certificate,
        )
        return _AttemptOutcome(attempt, next_y, None, None, None, 0.0)

    materialization_started = perf_counter()
    try:
        x, objective, residual_l1 = _materialize_float_recovery(
            problem, candidate_y, router
        )
    except Exception as exc:
        materialization_seconds = perf_counter() - materialization_started
        attempt = VerifiedAttempt(
            method=method,
            target_epsilon=target_epsilon,
            candidate_status=candidate_status,
            verifier_status="authorized_realization_failed",
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            authorized=False,
            strict_certificate=certificate,
            error_type=type(exc).__name__,
        )
        return _AttemptOutcome(
            attempt, next_y, None, None, None, materialization_seconds
        )
    materialization_seconds = perf_counter() - materialization_started
    attempt = VerifiedAttempt(
        method=method,
        target_epsilon=target_epsilon,
        candidate_status=candidate_status,
        verifier_status="accepted",
        candidate_seconds=candidate_seconds,
        prepare_seconds=prepare_seconds,
        verify_seconds=verify_seconds,
        authorized=True,
        strict_certificate=certificate,
    )
    return _AttemptOutcome(
        attempt,
        candidate_y.copy(),
        x,
        objective,
        residual_l1,
        materialization_seconds,
    )


def _skipped_attempt(
    method: str, target_epsilon: float, status: str
) -> VerifiedAttempt:
    return VerifiedAttempt(
        method=method,
        target_epsilon=target_epsilon,
        candidate_status=status,
        verifier_status="not_run",
        candidate_seconds=0.0,
        prepare_seconds=0.0,
        verify_seconds=0.0,
        authorized=False,
        invoked=False,
    )


def _validate_options(options: VerifiedOptions) -> None:
    if options.epsilon <= 0.0 or not np.isfinite(options.epsilon):
        raise ValueError("epsilon must be finite and positive")
    if (
        options.candidate_margin_divisor <= 1.0
        or not np.isfinite(options.candidate_margin_divisor)
    ):
        raise ValueError("candidate margin divisor must be finite and greater than one")
    if (
        options.repair_margin_divisor <= options.candidate_margin_divisor
        or not np.isfinite(options.repair_margin_divisor)
    ):
        raise ValueError("repair margin divisor must exceed candidate margin divisor")
    if options.candidate_max_steps < 0:
        raise ValueError("candidate_max_steps must be nonnegative")
    if (
        options.candidate_cg_tolerance <= 0.0
        or not np.isfinite(options.candidate_cg_tolerance)
    ):
        raise ValueError("candidate_cg_tolerance must be finite and positive")
    if options.sparse_direct_max_reduced_dimension < 0:
        raise ValueError("sparse direct dimension must be nonnegative")
    if options.max_scaling_sweeps < 0:
        raise ValueError("max_scaling_sweeps must be nonnegative")
    if options.certificate_every < 1 or options.refresh_tree_every < 1:
        raise ValueError("certificate and tree refresh periods must be positive")
    if options.time_limit_seconds is not None and (
        options.time_limit_seconds <= 0.0
        or not np.isfinite(options.time_limit_seconds)
    ):
        raise ValueError("time_limit_seconds must be finite and positive when supplied")


def solve_verified(
    problem: ReciprocalTransportProblem,
    y0: Optional[Array] = None,
    router: Optional[TreeRouter] = None,
    options: Optional[VerifiedOptions] = None,
) -> VerifiedResult:
    """Run the frozen candidate cascade and return only with strict authority.

    The order is Jacobi damped Newton at ``epsilon/16``, available AMG damped
    Newton at ``epsilon/256``, a sparse direct repair at ``epsilon/256`` when
    the reduced dimension is at most 5,000, alternate scaling at
    ``epsilon/256``, and finally the existing strict hybrid solver.
    """

    options = VerifiedOptions() if options is None else options
    _validate_options(options)
    started = perf_counter()
    initial_y = (
        dual_feasible_start(problem)
        if y0 is None
        else np.asarray(y0, dtype=float).copy()
    )

    def result(
        status: str,
        y: Array,
        *,
        x: Optional[Array] = None,
        objective: Optional[float] = None,
        gap: Optional[float] = None,
        method: Optional[str] = None,
        certificate: Optional[IntervalRecoveryCertificate] = None,
        active_router: Optional[TreeRouter] = None,
        residual_l1: Optional[float] = None,
        candidate_seconds: float = 0.0,
        prepare_seconds: float = 0.0,
        verify_seconds: float = 0.0,
        fallback_seconds: float = 0.0,
        materialization_seconds: float = 0.0,
        attempts: Optional[List[VerifiedAttempt]] = None,
        fallback_result: Optional[HybridResult] = None,
    ) -> VerifiedResult:
        return VerifiedResult(
            status=status,
            problem=problem,
            x=x,
            y=np.asarray(y, dtype=float).copy(),
            objective=objective,
            objective_gap_upper=gap,
            requested_epsilon=options.epsilon,
            method=method,
            strict_certificate=certificate,
            router=active_router,
            realization_residual_l1=residual_l1,
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            fallback_seconds=fallback_seconds,
            materialization_seconds=materialization_seconds,
            wall_time=perf_counter() - started,
            attempts=() if attempts is None else tuple(attempts),
            fallback_result=fallback_result,
        )

    if initial_y.shape != (problem.n,) or not np.all(np.isfinite(initial_y)):
        return result("invalid_initial_domain", initial_y)
    initial_y -= float(initial_y.mean())
    if not _binary_domain_valid(problem, initial_y):
        return result("invalid_initial_domain", initial_y)
    if router is not None:
        router.validate_for(problem)
    if router is None:
        state = dual_state(problem, initial_y)
        router = TreeRouter.build(problem, edge_cost=1.0 / state.conductance)

    attempts: List[VerifiedAttempt] = []
    candidate_seconds = 0.0
    prepare_seconds = 0.0
    verify_seconds = 0.0
    materialization_seconds = 0.0
    working_y = initial_y.copy()
    candidate_target = options.epsilon / options.candidate_margin_divisor
    repair_target = options.epsilon / options.repair_margin_divisor

    def timed_out() -> bool:
        remaining = _remaining_seconds(started, options.time_limit_seconds)
        return bool(remaining is not None and remaining <= 0.0)

    def run_attempt(
        method: str,
        target_epsilon: float,
        generator: Callable[[], BaselineResult],
    ) -> Optional[VerifiedResult]:
        nonlocal working_y
        nonlocal candidate_seconds, prepare_seconds, verify_seconds
        nonlocal materialization_seconds
        outcome = _run_and_verify(
            problem,
            working_y,
            router,
            method,
            target_epsilon,
            options.epsilon,
            options.strict_backend,
            options.interval_decimal_precision,
            router.problem,
            generator,
        )
        attempts.append(outcome.attempt)
        working_y = outcome.y_for_next.copy()
        candidate_seconds += outcome.attempt.candidate_seconds
        prepare_seconds += outcome.attempt.prepare_seconds
        verify_seconds += outcome.attempt.verify_seconds
        materialization_seconds += outcome.materialization_seconds
        if not outcome.attempt.authorized:
            return None
        certificate = outcome.attempt.strict_certificate
        return result(
            "optimal",
            working_y,
            x=outcome.x,
            objective=outcome.objective,
            gap=certificate.objective_gap_upper,
            method=method,
            certificate=certificate,
            active_router=router,
            residual_l1=outcome.realization_residual_l1,
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            materialization_seconds=materialization_seconds,
            attempts=attempts,
        )

    if timed_out():
        return result("timed_out", working_y, active_router=router)
    accepted = run_attempt(
        "damped_newton_jacobi",
        candidate_target,
        lambda: solve_damped_newton(
            problem,
            working_y,
            epsilon=candidate_target,
            max_steps=options.candidate_max_steps,
            direct_solve_below=0,
            cg_tolerance=options.candidate_cg_tolerance,
            router=router,
            preconditioner="jacobi",
            time_limit_seconds=_remaining_seconds(started, options.time_limit_seconds),
        ),
    )
    if accepted is not None:
        return accepted

    if _baselines.pyamg is None:
        attempts.append(_skipped_attempt("damped_newton_amg", repair_target, "unavailable"))
    elif not timed_out():
        accepted = run_attempt(
            "damped_newton_amg",
            repair_target,
            lambda: solve_damped_newton(
                problem,
                working_y,
                epsilon=repair_target,
                max_steps=options.candidate_max_steps,
                direct_solve_below=0,
                cg_tolerance=options.candidate_cg_tolerance,
                router=router,
                preconditioner="amg",
                time_limit_seconds=_remaining_seconds(started, options.time_limit_seconds),
            ),
        )
        if accepted is not None:
            return accepted

    if problem.n - 1 <= options.sparse_direct_max_reduced_dimension:
        if not timed_out():
            accepted = run_attempt(
                "sparse_direct_below_5000",
                repair_target,
                lambda: solve_damped_newton(
                    problem,
                    working_y,
                    epsilon=repair_target,
                    max_steps=options.candidate_max_steps,
                    direct_solve_below=options.sparse_direct_max_reduced_dimension + 1,
                    cg_tolerance=options.candidate_cg_tolerance,
                    router=router,
                    preconditioner="jacobi",
                    time_limit_seconds=_remaining_seconds(started, options.time_limit_seconds),
                ),
            )
            if accepted is not None:
                return accepted
    else:
        attempts.append(_skipped_attempt("sparse_direct_below_5000", repair_target, "ineligible"))

    if not timed_out():
        accepted = run_attempt(
            "alternate_scaling",
            repair_target,
            lambda: solve_alternating_scaling(
                problem,
                working_y,
                epsilon=repair_target,
                max_sweeps=options.max_scaling_sweeps,
                router=router,
                time_limit_seconds=_remaining_seconds(started, options.time_limit_seconds),
            ),
        )
        if accepted is not None:
            return accepted

    if timed_out():
        return result(
            "timed_out",
            working_y,
            active_router=router,
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            materialization_seconds=materialization_seconds,
            attempts=attempts,
        )

    remaining = _remaining_seconds(started, options.time_limit_seconds)
    fallback_newton = replace(
        options.fallback_newton,
        epsilon=options.epsilon,
        use_direct_solve_below=options.sparse_direct_max_reduced_dimension + 1,
        strict_interval=True,
        strict_backend=options.strict_backend,
        interval_decimal_precision=options.interval_decimal_precision,
        time_limit_seconds=remaining,
    )
    fallback_started = perf_counter()
    try:
        fallback = solve_hybrid(
            problem,
            working_y,
            router,
            HybridOptions(
                epsilon=options.epsilon,
                max_scaling_sweeps=options.max_scaling_sweeps,
                certificate_every=options.certificate_every,
                refresh_tree_every=options.refresh_tree_every,
                fallback_method="alternate_scaling",
                time_limit_seconds=remaining,
                newton=fallback_newton,
            ),
        )
    except Exception as exc:
        fallback_seconds = perf_counter() - fallback_started
        attempts.append(
            VerifiedAttempt(
                method="stepwise_certified_newton",
                target_epsilon=options.epsilon,
                candidate_status="exception",
                verifier_status="not_run",
                candidate_seconds=0.0,
                prepare_seconds=0.0,
                verify_seconds=0.0,
                authorized=False,
                error_type=type(exc).__name__,
            )
        )
        return result(
            "fallback_exception",
            working_y,
            active_router=router,
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            fallback_seconds=fallback_seconds,
            materialization_seconds=materialization_seconds,
            attempts=attempts,
        )
    fallback_seconds = perf_counter() - fallback_started
    certificate = fallback.final_interval_recovery
    certifying_router = fallback.router
    authorized = bool(
        fallback.x is not None
        and certifying_router is not None
        and certifying_router.is_compatible(problem)
        and _certificate_authorizes(
            certificate,
            options.epsilon,
            problem,
            fallback.y,
            certifying_router,
        )
    )
    attempts.append(
        VerifiedAttempt(
            method="stepwise_certified_newton",
            target_epsilon=options.epsilon,
            candidate_status=fallback.status,
            verifier_status="accepted" if authorized else "rejected",
            candidate_seconds=0.0,
            prepare_seconds=0.0,
            verify_seconds=0.0,
            authorized=authorized,
            strict_certificate=certificate,
        )
    )
    if not authorized:
        return result(
            f"fallback_{fallback.status}",
            fallback.y,
            active_router=router,
            candidate_seconds=candidate_seconds,
            prepare_seconds=prepare_seconds,
            verify_seconds=verify_seconds,
            fallback_seconds=fallback_seconds,
            materialization_seconds=materialization_seconds,
            attempts=attempts,
            fallback_result=fallback,
        )

    fallback_x = np.asarray(fallback.x, dtype=float).copy()
    residual_l1 = float(np.linalg.norm(problem.primal_residual(fallback_x), 1))
    winning_method = (
        "alternate_scaling_strict_recovery"
        if fallback.status == "optimal_scaling"
        else "stepwise_certified_newton"
    )
    return result(
        "optimal",
        fallback.y,
        x=fallback_x,
        objective=fallback.objective,
        gap=certificate.objective_gap_upper,
        method=winning_method,
        certificate=certificate,
        active_router=certifying_router,
        residual_l1=residual_l1,
        candidate_seconds=candidate_seconds,
        prepare_seconds=prepare_seconds,
        verify_seconds=verify_seconds,
        fallback_seconds=fallback_seconds,
        materialization_seconds=materialization_seconds,
        attempts=attempts,
        fallback_result=fallback,
    )
