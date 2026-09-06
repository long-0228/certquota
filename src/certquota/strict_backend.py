"""Dispatch between independently validated strict certificate backends."""

from __future__ import annotations

from typing import Union

import numpy as np

from .certificates import DualState, dual_state
from .ieee_intervals import (
    IEEEPreparedState,
    ieee_basin_certificate,
    ieee_forcing_certificate,
    ieee_platform_contract,
    ieee_recovery_certificate,
    prepare_ieee_recovery_state,
    prepare_ieee_state,
)
from .intervals import (
    PreparedIntervalState,
    interval_basin_certificate,
    interval_forcing_certificate,
    interval_linear_residual_midpoint,
    interval_recovery_certificate,
    prepare_interval_state,
)
from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


STRICT_BACKENDS = ("arb", "ieee754")
PreparedStrictState = Union[PreparedIntervalState, IEEEPreparedState]


def validate_strict_backend(backend: str) -> None:
    if backend not in STRICT_BACKENDS:
        raise ValueError(f"strict_backend must be one of {STRICT_BACKENDS}")
    if backend == "ieee754" and not ieee_platform_contract():
        raise RuntimeError("the IEEE-754 strict backend platform contract failed")


def prepare_strict_state(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    backend: str,
    decimal_precision: int,
) -> PreparedStrictState:
    validate_strict_backend(backend)
    if backend == "arb":
        return prepare_interval_state(problem, y, decimal_precision)
    return prepare_ieee_state(problem, y)


def prepare_strict_recovery_state(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    backend: str,
    decimal_precision: int,
    topology_owner: ReciprocalTransportProblem | None = None,
) -> PreparedStrictState:
    """Prepare only terminal-recovery fields when the backend supports it."""

    validate_strict_backend(backend)
    if backend == "arb":
        return prepare_interval_state(problem, y, decimal_precision)
    return prepare_ieee_recovery_state(problem, y, topology_owner)


def strict_basin_certificate(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    router: TreeRouter,
    threshold: float,
    backend: str,
    decimal_precision: int,
    prepared_state: PreparedStrictState | None,
):
    router.validate_for(problem)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 0.1:
        raise ValueError("basin threshold must be finite and in (0, 0.1]")
    validate_strict_backend(backend)
    if backend == "arb":
        if prepared_state is not None and not isinstance(
            prepared_state, PreparedIntervalState
        ):
            raise ValueError("prepared state/backend mismatch")
        return interval_basin_certificate(
            problem,
            y,
            router,
            threshold,
            decimal_precision,
            prepared_state,
        )
    if prepared_state is not None and not isinstance(
        prepared_state, IEEEPreparedState
    ):
        raise ValueError("prepared state/backend mismatch")
    return ieee_basin_certificate(
        problem, y, router, threshold, prepared_state
    )


def strict_forcing_certificate(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    direction: np.ndarray,
    router: TreeRouter,
    backend: str,
    decimal_precision: int,
    prepared_state: PreparedStrictState | None,
):
    router.validate_for(problem)
    validate_strict_backend(backend)
    if backend == "arb":
        if prepared_state is not None and not isinstance(
            prepared_state, PreparedIntervalState
        ):
            raise ValueError("prepared state/backend mismatch")
        return interval_forcing_certificate(
            problem,
            y,
            direction,
            router,
            decimal_precision,
            prepared_state,
        )
    if prepared_state is not None and not isinstance(
        prepared_state, IEEEPreparedState
    ):
        raise ValueError("prepared state/backend mismatch")
    return ieee_forcing_certificate(
        problem, y, direction, router, prepared_state
    )


def strict_recovery_certificate(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    epsilon: float,
    backend: str,
    decimal_precision: int,
    prepared_state: PreparedStrictState | None = None,
    router: TreeRouter | None = None,
):
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    validate_strict_backend(backend)
    if backend == "arb":
        if prepared_state is not None and not isinstance(
            prepared_state, PreparedIntervalState
        ):
            raise ValueError("prepared state/backend mismatch")
        certificate = interval_recovery_certificate(
            problem,
            y,
            epsilon,
            decimal_precision,
            prepared_state,
            router,
        )
    else:
        if prepared_state is not None and not isinstance(
            prepared_state, IEEEPreparedState
        ):
            raise ValueError("prepared state/backend mismatch")
        certificate = ieee_recovery_certificate(
            problem, y, epsilon, prepared_state, router
        )
    if not certificate.is_bound_to(problem, y, epsilon, router):
        raise RuntimeError("strict backend returned an unbound certificate")
    return certificate


def strict_linear_residual_candidate(
    problem: ReciprocalTransportProblem,
    y: np.ndarray,
    direction: np.ndarray,
    backend: str,
    decimal_precision: int,
    prepared_state: PreparedStrictState | None,
    binary_state: DualState | None = None,
) -> np.ndarray:
    """Return a defect-correction candidate residual, never an acceptance test."""

    validate_strict_backend(backend)
    if backend == "arb":
        if prepared_state is not None and not isinstance(
            prepared_state, PreparedIntervalState
        ):
            raise ValueError("prepared state/backend mismatch")
        return interval_linear_residual_midpoint(
            problem,
            y,
            direction,
            decimal_precision,
            prepared_state,
        )
    state = dual_state(problem, y) if binary_state is None else binary_state
    residual = (
        problem.laplacian_matvec(direction, state.conductance) + state.g
    )
    return problem.impose_reduced_balance(residual)
