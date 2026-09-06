"""Strict outward-rounded certificates backed by FLINT/Arb balls.

Binary64 remains the fast path. At a checkpoint these routines reconstruct every
input float as an exact dyadic Arb value and replay only the finite certificate
calculation. Acceptance always compares an outward upper bound with an outward
lower bound; a numerically ambiguous checkpoint is therefore rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil, log2
from struct import pack
from time import perf_counter
from typing import List, Sequence

import numpy as np

try:
    from flint import arb, arb_mat, ctx
except ImportError:  # pragma: no cover - only in an incomplete installation
    arb = None
    arb_mat = None
    ctx = None

from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


Array = np.ndarray
BACKEND_NAME = "python-flint-arb"
_PREPARED_INTERVAL_TOKEN = object()
_RECOVERY_CERTIFICATE_TOKEN = object()


def _immutable_float_snapshot(values) -> Array:
    raw = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return np.frombuffer(
        raw.tobytes(order="C"), dtype=raw.dtype
    ).reshape(raw.shape)


def _binary64_fingerprint(domain: bytes, values) -> str:
    raw = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    hasher = sha256()
    hasher.update(domain)
    hasher.update(pack("<Q", raw.ndim))
    for dimension in raw.shape:
        hasher.update(pack("<Q", int(dimension)))
    hasher.update(raw.tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class PreparedIntervalState:
    """Immutable Arb state reusable only at one exact outer iterate."""

    problem: ReciprocalTransportProblem
    y: Array
    decimal_precision: int
    x: Sequence | None
    conductance: Sequence | None
    gradient: Sequence | None
    sigma_terms: Sequence | None
    conductance_lower: Array | None
    conductance_upper: Array | None
    sigma_lower: float | None
    sigma_upper: float | None
    x_min_lower: float | None
    _capability: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _identity_seal: tuple | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _problem_fingerprint: str | None = field(
        init=False, repr=False, compare=False, default=None
    )

    @property
    def domain_certified(self) -> bool:
        return self.conductance is not None


@dataclass(frozen=True)
class IntervalBasinCertificate:
    domain_certified: bool
    eta_upper: float
    decrement_upper: float
    sigma_lower: float
    passed: bool
    wall_time: float
    decimal_precision: int
    backend: str = BACKEND_NAME


@dataclass(frozen=True)
class IntervalForcingCertificate:
    residual_upper: float
    decrement_lower: float
    eta_lower: float
    threshold_lower: float
    passed: bool
    wall_time: float
    decimal_precision: int
    backend: str = BACKEND_NAME


@dataclass(frozen=True)
class IntervalRecoveryCertificate:
    residual_l1_upper: float
    x_min_lower: float
    objective_gap_upper: float
    positivity_certified: bool
    epsilon_optimal_certified: bool
    wall_time: float
    decimal_precision: int
    backend: str = BACKEND_NAME
    binding_version: str | None = None
    problem_fingerprint: str | None = None
    router_fingerprint: str | None = None
    iterate_fingerprint: str | None = None
    epsilon_fingerprint: str | None = None
    _capability: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _authority_seal: tuple | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def is_bound_to(
        self,
        problem: ReciprocalTransportProblem,
        y: Array,
        epsilon: float,
        router: TreeRouter | None,
    ) -> bool:
        """Return whether this process-local summary binds the replay inputs."""

        if getattr(self, "_capability", None) is not _RECOVERY_CERTIFICATE_TOKEN:
            return False
        current_authority = (
            self.residual_l1_upper,
            self.x_min_lower,
            self.objective_gap_upper,
            self.positivity_certified,
            self.epsilon_optimal_certified,
            self.decimal_precision,
            self.backend,
            self.binding_version,
            self.problem_fingerprint,
            self.router_fingerprint,
            self.iterate_fingerprint,
            self.epsilon_fingerprint,
        )
        if self._authority_seal != current_authority:
            return False
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            return False
        try:
            if router is not None:
                router.validate_for(problem)
            y_values = np.asarray(y, dtype=float)
            if y_values.shape != (problem.n,) or not np.all(np.isfinite(y_values)):
                return False
            return bool(
                self.binding_version == "certquota-recovery-replay-v1"
                and self.problem_fingerprint == problem.content_fingerprint
                and self.router_fingerprint
                == (None if router is None else router.structure_fingerprint)
                and self.iterate_fingerprint
                == _binary64_fingerprint(b"certquota-y-v1\0", y_values)
                and self.epsilon_fingerprint
                == _binary64_fingerprint(
                    b"certquota-epsilon-v1\0", [epsilon]
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _seal_recovery_certificate(
    certificate: IntervalRecoveryCertificate,
    problem: ReciprocalTransportProblem,
    y: Array,
    epsilon: float,
    router: TreeRouter | None,
) -> IntervalRecoveryCertificate:
    """Bind an authoritative in-process certificate to its replay inputs."""

    object.__setattr__(
        certificate, "binding_version", "certquota-recovery-replay-v1"
    )
    object.__setattr__(
        certificate, "problem_fingerprint", problem.content_fingerprint
    )
    object.__setattr__(
        certificate,
        "router_fingerprint",
        None if router is None else router.structure_fingerprint,
    )
    object.__setattr__(
        certificate,
        "iterate_fingerprint",
        _binary64_fingerprint(b"certquota-y-v1\0", y),
    )
    object.__setattr__(
        certificate,
        "epsilon_fingerprint",
        _binary64_fingerprint(b"certquota-epsilon-v1\0", [epsilon]),
    )
    object.__setattr__(certificate, "_capability", _RECOVERY_CERTIFICATE_TOKEN)
    object.__setattr__(
        certificate,
        "_authority_seal",
        (
            certificate.residual_l1_upper,
            certificate.x_min_lower,
            certificate.objective_gap_upper,
            certificate.positivity_certified,
            certificate.epsilon_optimal_certified,
            certificate.decimal_precision,
            certificate.backend,
            certificate.binding_version,
            certificate.problem_fingerprint,
            certificate.router_fingerprint,
            certificate.iterate_fingerprint,
            certificate.epsilon_fingerprint,
        ),
    )
    return certificate


@dataclass(frozen=True)
class IntervalEtaReference:
    eta_lower: float
    eta_upper: float
    wall_time: float
    decimal_precision: int
    backend: str = BACKEND_NAME


@dataclass(frozen=True)
class IntervalNewtonRecurrenceReference:
    """Rigorous exact-Newton recurrence reference for proof stress only."""

    eta_lower: float
    eta_upper: float
    eta_next_lower: float
    eta_next_upper: float
    next_domain_certified: bool
    wall_time: float
    decimal_precision: int
    backend: str = BACKEND_NAME


def interval_backend_available() -> bool:
    """Return whether the strict Arb replay dependency is importable."""

    return arb is not None and arb_mat is not None and ctx is not None


def _require_backend() -> None:
    if not interval_backend_available():
        raise RuntimeError(
            "strict certificates require python-flint; install benchmark extras"
        )


def _bits(decimal_precision: int) -> int:
    if decimal_precision < 16:
        raise ValueError("decimal_precision must be at least 16")
    return max(80, int(ceil(decimal_precision * log2(10.0))) + 16)


def _point(value: float):
    """Convert the exact stored binary64 value to an Arb dyadic point."""

    return arb(float(value))


def _safe_upper(value) -> float:
    if value.is_nan():
        return float("inf")
    if value.is_zero():
        return 0.0
    return float(np.nextafter(float(value.upper()), np.inf))


def _safe_lower(value) -> float:
    if value.is_nan():
        return float("-inf")
    if value.is_zero():
        return 0.0
    return float(np.nextafter(float(value.lower()), -np.inf))


def _outward_lower_subtract(left: float, right: float) -> float:
    """Subtract exact binary64 dyadics in Arb and return a safe lower bound."""

    return _safe_lower(_point(left) - _point(right))


def _outward_lower_add(left: float, right: float) -> float:
    """Add exact binary64 dyadics in Arb and return a safe lower bound."""

    return _safe_lower(_point(left) + _point(right))


def _absolute_upper(value) -> float:
    return max(0.0, _safe_upper(value.abs_upper()))


def _absolute_lower(value) -> float:
    return max(0.0, _safe_lower(value.abs_lower()))


def _interval_state_from_values(
    problem: ReciprocalTransportProblem, y_values: Sequence
):
    if len(y_values) != problem.n:
        raise ValueError("y has the wrong dimension")
    q: List = []
    x: List = []
    conductance: List = []
    sigma_terms: List = []
    for edge in range(problem.m):
        tail = int(problem.tails[edge])
        head = int(problem.heads[edge])
        q_edge = (
            _point(problem.c[edge])
            - arb(y_values[tail])
            + arb(y_values[head])
        )
        q.append(q_edge)
        if _safe_lower(q_edge) <= 0.0:
            return q, None, None, None, None
        mu = _point(problem.mu[edge])
        root_mu = mu.sqrt()
        root_q = q_edge.sqrt()
        x_edge = (mu / q_edge).sqrt()
        x.append(x_edge)
        conductance.append(root_mu / (_point(2.0) * q_edge * root_q))
        sigma_terms.append((mu * q_edge).sqrt().sqrt())

    gradient = [_point(-value) for value in problem.beta]
    for edge, x_edge in enumerate(x):
        tail = int(problem.tails[edge])
        head = int(problem.heads[edge])
        gradient[tail] += x_edge
        gradient[head] -= x_edge
    # beta[:-1] are the independent quota inputs.  The root component is
    # conservation-implied, so reconstruct the exact balanced Arb covector
    # instead of using the rounded binary64 representative beta[-1].
    gradient[-1] = -sum(gradient[:-1], arb(0))
    return q, x, conductance, gradient, sigma_terms


def _interval_state(problem: ReciprocalTransportProblem, y: Array):
    y = np.asarray(y, dtype=float)
    if y.shape != (problem.n,):
        raise ValueError("y has the wrong dimension")
    return _interval_state_from_values(problem, [_point(value) for value in y])


def _seal_interval_state(
    state: PreparedIntervalState,
) -> PreparedIntervalState:
    object.__setattr__(state, "_capability", _PREPARED_INTERVAL_TOKEN)
    object.__setattr__(
        state, "_problem_fingerprint", state.problem.content_fingerprint
    )
    object.__setattr__(
        state,
        "_identity_seal",
        (
            id(state.problem),
            id(state.y),
            id(state.x),
            id(state.conductance),
            id(state.gradient),
            id(state.sigma_terms),
            id(state.conductance_lower),
            id(state.conductance_upper),
            state.decimal_precision,
            state.sigma_lower,
            state.sigma_upper,
            state.x_min_lower,
        ),
    )
    return state


def prepare_interval_state(
    problem: ReciprocalTransportProblem,
    y: Array,
    decimal_precision: int = 30,
) -> PreparedIntervalState:
    """Build the strict state once for reuse at an unchanged binary64 iterate."""

    _require_backend()
    y_snapshot = _immutable_float_snapshot(y)
    if y_snapshot.shape != (problem.n,):
        raise ValueError("y has the wrong dimension")
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        _, x, conductance, gradient, sigma_terms = _interval_state(
            problem, y_snapshot
        )
        if conductance is None:
            conductance_lower = conductance_upper = None
            sigma_lower = sigma_upper = x_min_lower = None
        else:
            conductance_lower = _immutable_float_snapshot(
                [_safe_lower(value) for value in conductance]
            )
            conductance_upper = _immutable_float_snapshot(
                [_safe_upper(value) for value in conductance]
            )
            sigma_lower = min(_safe_lower(value) for value in sigma_terms)
            sigma_upper = min(_safe_upper(value) for value in sigma_terms)
            x_min_lower = min(_safe_lower(value) for value in x)
        return _seal_interval_state(
            PreparedIntervalState(
                problem,
                y_snapshot,
                decimal_precision,
                None if x is None else tuple(x),
                None if conductance is None else tuple(conductance),
                None if gradient is None else tuple(gradient),
                None if sigma_terms is None else tuple(sigma_terms),
                conductance_lower,
                conductance_upper,
                sigma_lower,
                sigma_upper,
                x_min_lower,
            )
        )
    finally:
        ctx.prec = old_precision


def _prepared_values(
    problem: ReciprocalTransportProblem,
    y: Array,
    decimal_precision: int,
    prepared_state: PreparedIntervalState | None,
):
    if prepared_state is None:
        return _interval_state(problem, y)
    if getattr(prepared_state, "_capability", None) is not _PREPARED_INTERVAL_TOKEN:
        raise ValueError("untrusted or stale prepared interval state")
    expected_seal = (
        id(prepared_state.problem),
        id(prepared_state.y),
        id(prepared_state.x),
        id(prepared_state.conductance),
        id(prepared_state.gradient),
        id(prepared_state.sigma_terms),
        id(prepared_state.conductance_lower),
        id(prepared_state.conductance_upper),
        prepared_state.decimal_precision,
        prepared_state.sigma_lower,
        prepared_state.sigma_upper,
        prepared_state.x_min_lower,
    )
    if (
        prepared_state._identity_seal != expected_seal
        or prepared_state._problem_fingerprint
        != prepared_state.problem.content_fingerprint
    ):
        raise ValueError("untrusted or stale prepared interval state")
    if prepared_state.problem is not problem:
        raise ValueError("prepared interval state belongs to another problem")
    if prepared_state.decimal_precision != decimal_precision:
        raise ValueError("prepared interval state uses another precision")
    y_values = np.asarray(y, dtype=float)
    if y_values.shape != prepared_state.y.shape or not np.array_equal(
        y_values, prepared_state.y
    ):
        raise ValueError("prepared interval state belongs to another iterate")
    return (
        None,
        prepared_state.x,
        prepared_state.conductance,
        prepared_state.gradient,
        prepared_state.sigma_terms,
    )


def _reduced_laplacian(problem: ReciprocalTransportProblem, conductance: Sequence):
    reduced_n = problem.n - 1
    entries = [[arb(0) for _ in range(reduced_n)] for _ in range(reduced_n)]
    for edge, weight in enumerate(conductance):
        tail = int(problem.tails[edge])
        head = int(problem.heads[edge])
        if tail < reduced_n:
            entries[tail][tail] += weight
        if head < reduced_n:
            entries[head][head] += weight
        if tail < reduced_n and head < reduced_n:
            entries[tail][head] -= weight
            entries[head][tail] -= weight
    return arb_mat(entries)


def _eta_bounds(decrement_sq, sigma_terms: Sequence) -> tuple[float, float]:
    decrement_sq_lower = max(0.0, _safe_lower(decrement_sq))
    decrement_sq_upper = max(0.0, _safe_upper(decrement_sq))
    sigma_lower = min(_safe_lower(value) for value in sigma_terms)
    sigma_upper = min(_safe_upper(value) for value in sigma_terms)
    if sigma_lower <= 0.0:
        eta_upper = float("inf")
    else:
        eta_upper = _safe_upper(
            (_point(2.0) * _point(decrement_sq_upper)).sqrt()
            / _point(sigma_lower)
        )
    if sigma_upper <= 0.0:
        eta_lower = 0.0
    else:
        eta_lower = max(
            0.0,
            _safe_lower(
                (_point(2.0) * _point(decrement_sq_lower)).sqrt()
                / _point(sigma_upper)
            ),
        )
    return eta_lower, eta_upper


def _tree_interval_flow(router: TreeRouter, demand: Sequence) -> List:
    subtree = [arb(value) for value in demand]
    flow = [arb(0) for _ in range(router.problem.m)]
    for vertex_value in router.postorder:
        vertex = int(vertex_value)
        if vertex == router.root:
            continue
        edge = int(router.parent_edge[vertex])
        flow[edge] = _point(router.child_incidence_sign[vertex]) * subtree[vertex]
        parent = int(router.parent[vertex])
        subtree[parent] += subtree[vertex]
    return flow


def _arb_tree_recovery_gap_ball(
    problem: ReciprocalTransportProblem,
    x: Sequence,
    correction: Sequence,
    tree_edges: Sequence,
    *,
    signed: bool,
):
    """Return an Arb upper-bound expression for tree recovery.

    ``signed=False`` preserves the former absolute-correction predicate for
    ablation.  The production ``signed=True`` path retains the lower endpoint
    of each correction and therefore converges to the exact primal--dual gap as
    the input balls contract.
    """

    gap_ball = arb(0)
    for edge_value in tree_edges:
        edge = int(edge_value)
        correction_abs_upper = correction[edge].abs_upper()
        x_lower = x[edge].lower()
        if signed:
            corrected_lower = (
                x_lower + correction[edge].lower()
            ).lower()
        else:
            corrected_lower = (
                x_lower - correction_abs_upper
            ).lower()
        if not (corrected_lower > 0):
            return None, False
        numerator_upper = (
            _point(problem.mu[edge])
            * correction_abs_upper
            * correction_abs_upper
        ).upper()
        denominator_lower = (
            x_lower * x_lower * corrected_lower
        ).lower()
        if not (denominator_lower > 0):
            return None, False
        gap_ball += numerator_upper / denominator_lower
    return gap_ball, True


def _arb_absolute_tree_recovery_gap_ball(
    problem: ReciprocalTransportProblem,
    x: Sequence,
    correction: Sequence,
    tree_edges: Sequence,
):
    """Return the legacy absolute-correction Arb bound for ablation."""

    return _arb_tree_recovery_gap_ball(
        problem, x, correction, tree_edges, signed=False
    )


def _arb_signed_tree_recovery_gap_ball(
    problem: ReciprocalTransportProblem,
    x: Sequence,
    correction: Sequence,
    tree_edges: Sequence,
):
    """Return the production signed-correction Arb bound."""

    return _arb_tree_recovery_gap_ball(
        problem, x, correction, tree_edges, signed=True
    )


def _tree_energy_upper(
    router: TreeRouter,
    demand: Sequence,
    conductance: Sequence,
    conductance_lower: Array | None = None,
):
    """Return a ball enclosing the nonnegative tree energy from above."""

    flow = _tree_interval_flow(router, demand)
    energy = arb(0)
    for edge_value in router.tree_edges:
        edge = int(edge_value)
        flow_upper = _absolute_upper(flow[edge])
        lower = (
            _safe_lower(conductance[edge])
            if conductance_lower is None
            else float(conductance_lower[edge])
        )
        if lower <= 0.0:
            return arb("nan")
        energy += _point(flow_upper) * _point(flow_upper) / _point(lower)
    return energy


def interval_basin_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    router: TreeRouter,
    threshold: float = 0.1,
    decimal_precision: int = 30,
    prepared_state: PreparedIntervalState | None = None,
) -> IntervalBasinCertificate:
    router.validate_for(problem)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 0.1:
        raise ValueError("basin threshold must be finite and in (0, 0.1]")
    _require_backend()
    started = perf_counter()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        _, _, conductance, gradient, sigma_terms = _prepared_values(
            problem, y, decimal_precision, prepared_state
        )
        if conductance is None:
            return IntervalBasinCertificate(
                False, float("inf"), float("inf"), 0.0, False,
                perf_counter() - started, decimal_precision
            )
        cached_conductance_lower = (
            prepared_state.conductance_lower
            if prepared_state is not None
            else None
        )
        energy_upper = _safe_upper(
            _tree_energy_upper(
                router, gradient, conductance, cached_conductance_lower
            )
        )
        sigma_lower = (
            float(prepared_state.sigma_lower)
            if prepared_state is not None
            else min(_safe_lower(term) for term in sigma_terms)
        )
        if not np.isfinite(energy_upper) or sigma_lower <= 0.0:
            eta_upper = decrement_upper = float("inf")
        else:
            decrement_upper = _safe_upper(_point(energy_upper).sqrt())
            eta_upper = _safe_upper(
                (_point(2.0) * _point(energy_upper)).sqrt()
                / _point(sigma_lower)
            )
        return IntervalBasinCertificate(
            True,
            eta_upper,
            decrement_upper,
            sigma_lower,
            bool(eta_upper <= threshold),
            perf_counter() - started,
            decimal_precision,
        )
    finally:
        ctx.prec = old_precision


def interval_forcing_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    direction: Array,
    router: TreeRouter,
    decimal_precision: int = 30,
    prepared_state: PreparedIntervalState | None = None,
) -> IntervalForcingCertificate:
    router.validate_for(problem)
    _require_backend()
    started = perf_counter()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (problem.n,):
            raise ValueError("direction has the wrong dimension")
        _, _, conductance, gradient, sigma_terms = _prepared_values(
            problem, y, decimal_precision, prepared_state
        )
        if conductance is None:
            return IntervalForcingCertificate(
                float("inf"), 0.0, 0.0, 0.0, False,
                perf_counter() - started, decimal_precision
            )

        weighted_step: List = []
        denominator_upper_ball = arb(0)
        for edge in range(problem.m):
            tail = int(problem.tails[edge])
            head = int(problem.heads[edge])
            step = _point(direction[tail]) - _point(direction[head])
            weighted_step.append(conductance[edge] * step)
            conductance_upper = (
                float(prepared_state.conductance_upper[edge])
                if prepared_state is not None
                else _safe_upper(conductance[edge])
            )
            step_upper = _absolute_upper(step)
            denominator_upper_ball += (
                _point(conductance_upper)
                * _point(step_upper)
                * _point(step_upper)
            )

        residual = [arb(value) for value in gradient]
        for edge, weighted in enumerate(weighted_step):
            residual[int(problem.tails[edge])] += weighted
            residual[int(problem.heads[edge])] -= weighted
        residual[-1] = -sum(residual[:-1], arb(0))
        cached_conductance_lower = (
            prepared_state.conductance_lower
            if prepared_state is not None
            else None
        )
        residual_energy_upper = _safe_upper(
            _tree_energy_upper(
                router, residual, conductance, cached_conductance_lower
            )
        )
        residual_upper = _safe_upper(_point(residual_energy_upper).sqrt())

        # Pair in the full-rank reduced gauge.  This is algebraically equal to
        # g^T direction because g_root=-sum(g[:-1]), but it avoids treating a
        # rounded root quota as independent and removes interval dependency on
        # the reconstructed root ball.
        numerator = arb(0)
        for vertex in range(problem.n - 1):
            gauge_direction = (
                _point(direction[vertex]) - _point(direction[problem.root])
            )
            numerator += gradient[vertex] * gauge_direction
        numerator_lower = _absolute_lower(numerator)
        denominator_upper = _safe_upper(denominator_upper_ball)
        if denominator_upper <= 0.0:
            decrement_lower = 0.0
        else:
            decrement_lower = max(
                0.0,
                _safe_lower(
                    _point(numerator_lower) / _point(denominator_upper).sqrt()
                ),
            )

        sigma_upper = (
            float(prepared_state.sigma_upper)
            if prepared_state is not None
            else min(_safe_upper(term) for term in sigma_terms)
        )
        if sigma_upper <= 0.0:
            eta_lower = 0.0
        else:
            eta_lower = max(
                0.0,
                _safe_lower(
                    _point(2.0).sqrt()
                    * _point(decrement_lower)
                    / _point(sigma_upper)
                ),
            )
        threshold_lower = max(
            0.0,
            _safe_lower(
                _point(eta_lower) * _point(decrement_lower) / _point(4.0)
            ),
        )
        return IntervalForcingCertificate(
            residual_upper,
            decrement_lower,
            eta_lower,
            threshold_lower,
            bool(residual_upper <= threshold_lower),
            perf_counter() - started,
            decimal_precision,
        )
    finally:
        ctx.prec = old_precision


def interval_recovery_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    epsilon: float,
    decimal_precision: int = 30,
    prepared_state: PreparedIntervalState | None = None,
    router: TreeRouter | None = None,
) -> IntervalRecoveryCertificate:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    _require_backend()
    started = perf_counter()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        _, x, _, gradient, _ = _prepared_values(
            problem, y, decimal_precision, prepared_state
        )
        if x is None:
            return _seal_recovery_certificate(
                IntervalRecoveryCertificate(
                    float("inf"), 0.0, float("inf"), False, False,
                    perf_counter() - started, decimal_precision
                ),
                problem,
                y,
                epsilon,
                router,
            )
        residual_ball = arb(0)
        for value in gradient:
            residual_ball += _point(_absolute_upper(value))
        residual_upper = _safe_upper(residual_ball)
        x_min_lower = (
            float(prepared_state.x_min_lower)
            if prepared_state is not None
            else min(_safe_lower(value) for value in x)
        )
        if router is not None:
            correction = _tree_interval_flow(
                router, [-arb(value) for value in gradient]
            )
            gap_ball, positivity = _arb_signed_tree_recovery_gap_ball(
                problem, x, correction, router.tree_edges
            )
            gap_upper = _safe_upper(gap_ball) if positivity else float("inf")
        else:
            positivity = bool(residual_upper <= x_min_lower)
            if positivity:
                gap_ball = (
                    _point(2.0)
                    * _point(float(np.max(problem.mu)))
                    * _point(float(max(1, problem.n - 1)))
                    * _point(residual_upper)
                    * _point(residual_upper)
                    / (_point(x_min_lower) ** 3)
                )
                gap_upper = _safe_upper(gap_ball)
            else:
                gap_upper = float("inf")
        return _seal_recovery_certificate(
            IntervalRecoveryCertificate(
                residual_upper,
                x_min_lower,
                gap_upper,
                positivity,
                bool(positivity and gap_upper <= epsilon),
                perf_counter() - started,
                decimal_precision,
            ),
            problem,
            y,
            epsilon,
            router,
        )
    finally:
        ctx.prec = old_precision


def interval_exact_eta_reference(
    problem: ReciprocalTransportProblem,
    y: Array,
    decimal_precision: int = 70,
) -> IntervalEtaReference:
    """Enclose the exact Newton eta by a dense reduced Arb solve.

    This is a proof-stress reference for small graphs, not a production solver.
    The last vertex is removed to fix the Laplacian gauge.
    """

    _require_backend()
    started = perf_counter()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        _, _, conductance, gradient, sigma_terms = _interval_state(problem, y)
        if conductance is None:
            return IntervalEtaReference(
                0.0, float("inf"), perf_counter() - started, decimal_precision
            )
        reduced_n = problem.n - 1
        entries = [[arb(0) for _ in range(reduced_n)] for _ in range(reduced_n)]
        for edge, weight in enumerate(conductance):
            tail = int(problem.tails[edge])
            head = int(problem.heads[edge])
            if tail < reduced_n:
                entries[tail][tail] += weight
            if head < reduced_n:
                entries[head][head] += weight
            if tail < reduced_n and head < reduced_n:
                entries[tail][head] -= weight
                entries[head][tail] -= weight
        matrix = arb_mat(entries)
        rhs = arb_mat([[gradient[index]] for index in range(reduced_n)])
        solution = matrix.solve(rhs)
        decrement_sq = arb(0)
        for index in range(reduced_n):
            decrement_sq += gradient[index] * solution[index, 0]

        decrement_sq_lower = max(0.0, _safe_lower(decrement_sq))
        decrement_sq_upper = max(0.0, _safe_upper(decrement_sq))
        sigma_lower = min(_safe_lower(value) for value in sigma_terms)
        sigma_upper = min(_safe_upper(value) for value in sigma_terms)
        if sigma_lower <= 0.0:
            eta_upper = float("inf")
        else:
            eta_upper = _safe_upper(
                (_point(2.0) * _point(decrement_sq_upper)).sqrt()
                / _point(sigma_lower)
            )
        if sigma_upper <= 0.0:
            eta_lower = 0.0
        else:
            eta_lower = max(
                0.0,
                _safe_lower(
                    (_point(2.0) * _point(decrement_sq_lower)).sqrt()
                    / _point(sigma_upper)
                ),
            )
        return IntervalEtaReference(
            eta_lower,
            eta_upper,
            perf_counter() - started,
            decimal_precision,
        )
    finally:
        ctx.prec = old_precision


def interval_exact_newton_recurrence_reference(
    problem: ReciprocalTransportProblem,
    y: Array,
    decimal_precision: int = 70,
) -> IntervalNewtonRecurrenceReference:
    """Enclose an *exact* full Newton step and its next decrement with Arb.

    Unlike evaluating a binary64 sparse-solve direction at high precision, this
    routine solves the reduced Newton system inside Arb and propagates that ball
    solution through the nonlinear next state.  It is intentionally dense and is
    used only on the deterministically selected small proof-stress cases.
    """

    _require_backend()
    started = perf_counter()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        y = np.asarray(y, dtype=float)
        if y.shape != (problem.n,):
            raise ValueError("y has the wrong dimension")
        _, _, conductance, gradient, sigma_terms = _interval_state(problem, y)
        if conductance is None:
            return IntervalNewtonRecurrenceReference(
                0.0,
                float("inf"),
                0.0,
                float("inf"),
                False,
                perf_counter() - started,
                decimal_precision,
            )

        reduced_n = problem.n - 1
        matrix = _reduced_laplacian(problem, conductance)
        rhs = arb_mat([[-gradient[index]] for index in range(reduced_n)])
        step = matrix.solve(rhs)
        decrement_sq = arb(0)
        for index in range(reduced_n):
            decrement_sq -= gradient[index] * step[index, 0]
        eta_lower, eta_upper = _eta_bounds(decrement_sq, sigma_terms)

        updated_y = [_point(value) for value in y]
        for index in range(reduced_n):
            updated_y[index] += step[index, 0]
        _, _, next_conductance, next_gradient, next_sigma_terms = (
            _interval_state_from_values(problem, updated_y)
        )
        if next_conductance is None:
            return IntervalNewtonRecurrenceReference(
                eta_lower,
                eta_upper,
                0.0,
                float("inf"),
                False,
                perf_counter() - started,
                decimal_precision,
            )

        next_matrix = _reduced_laplacian(problem, next_conductance)
        next_rhs = arb_mat(
            [[next_gradient[index]] for index in range(reduced_n)]
        )
        next_solution = next_matrix.solve(next_rhs)
        next_decrement_sq = arb(0)
        for index in range(reduced_n):
            next_decrement_sq += (
                next_gradient[index] * next_solution[index, 0]
            )
        eta_next_lower, eta_next_upper = _eta_bounds(
            next_decrement_sq, next_sigma_terms
        )
        return IntervalNewtonRecurrenceReference(
            eta_lower,
            eta_upper,
            eta_next_lower,
            eta_next_upper,
            True,
            perf_counter() - started,
            decimal_precision,
        )
    finally:
        ctx.prec = old_precision


def interval_newton_direction(
    problem: ReciprocalTransportProblem,
    y: Array,
    decimal_precision: int = 70,
) -> Array:
    """Return the midpoint of an Arb reduced Newton solve for strict refinement.

    The returned binary64 direction is still accepted only if the independent
    outward-rounded posterior forcing certificate passes.  This helper therefore
    improves a candidate; it never substitutes for certification.
    """

    _require_backend()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        y = np.asarray(y, dtype=float)
        if y.shape != (problem.n,):
            raise ValueError("y has the wrong dimension")
        _, _, conductance, gradient, _ = _interval_state(problem, y)
        if conductance is None:
            raise ValueError("dual point lies outside the strict domain")
        reduced_n = problem.n - 1
        matrix = _reduced_laplacian(problem, conductance)
        rhs = arb_mat([[-gradient[index]] for index in range(reduced_n)])
        solution = matrix.solve(rhs)
        direction = np.zeros(problem.n, dtype=float)
        for index in range(reduced_n):
            direction[index] = float(solution[index, 0].mid())
        direction -= float(direction.mean())
        return direction
    finally:
        ctx.prec = old_precision


def interval_linear_residual_midpoint(
    problem: ReciprocalTransportProblem,
    y: Array,
    direction: Array,
    decimal_precision: int = 70,
    prepared_state: PreparedIntervalState | None = None,
) -> Array:
    """Return Arb midpoints of ``H(y) direction + g(y)`` in linear memory."""

    _require_backend()
    old_precision = ctx.prec
    ctx.prec = _bits(decimal_precision)
    try:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (problem.n,):
            raise ValueError("direction has the wrong dimension")
        _, _, conductance, gradient, _ = _prepared_values(
            problem, y, decimal_precision, prepared_state
        )
        if conductance is None:
            raise ValueError("dual point lies outside the strict domain")
        residual = [arb(value) for value in gradient]
        for edge, weight in enumerate(conductance):
            tail = int(problem.tails[edge])
            head = int(problem.heads[edge])
            edge_step = _point(direction[tail]) - _point(direction[head])
            weighted = weight * edge_step
            residual[tail] += weighted
            residual[head] -= weighted
        residual[-1] = -sum(residual[:-1], arb(0))
        return np.asarray(
            [float(value.mid()) for value in residual], dtype=float
        )
    finally:
        ctx.prec = old_precision
