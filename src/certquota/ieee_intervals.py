"""Vectorized outward binary64 certificates under an explicit IEEE-754 contract.

This module is experimental until its Arb-containment gates pass.  It interprets
every stored float as an exact dyadic real, brackets elementary operations by one
``nextafter`` step, and evaluates grouped reductions with segmented balanced trees
whose every addition is rounded outward.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, TypeVar

import numpy as np

from .intervals import (
    IntervalBasinCertificate,
    IntervalForcingCertificate,
    IntervalRecoveryCertificate,
    _immutable_float_snapshot,
    _seal_recovery_certificate,
)
from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


Array = np.ndarray
BACKEND_NAME = "ieee754-segmented-pairwise-binary64-bounds-experimental"
_NEGATIVE_INFINITY = np.float64(-np.inf)
_POSITIVE_INFINITY = np.float64(np.inf)
_IEEE_PREPARED_TOKEN = object()


@dataclass(frozen=True)
class _PairwiseReductionLevel:
    """One immutable layer of a stable segmented pairwise reduction."""

    output: Array
    left: Array
    right: Array


@dataclass(frozen=True)
class _SegmentedPairwisePlan:
    """Precompiled stable ordering and pairwise-addition DAG."""

    input_size: int
    output_size: int
    workspace_size: int
    order: Array | None
    output_groups: Array
    output_nodes: Array
    levels: tuple[_PairwiseReductionLevel, ...]


@dataclass(frozen=True)
class _ProblemReductionPlan:
    """Fixed reductions induced solely by a problem eligibility graph."""

    clients: _SegmentedPairwisePlan
    models: _SegmentedPairwisePlan
    reduced_nodes: _SegmentedPairwisePlan
    all_nodes: _SegmentedPairwisePlan
    all_edges: _SegmentedPairwisePlan


@dataclass(frozen=True)
class _TreeLevelReductionPlan:
    """Fixed child-to-parent aggregation for one BFS tree level."""

    vertices: Array
    signs: Array
    active_parents: Array
    reduction: _SegmentedPairwisePlan


@dataclass(frozen=True)
class _TreeReductionPlan:
    """Fixed bottom-up reduction schedule for one tree router."""

    levels: tuple[_TreeLevelReductionPlan, ...]
    tree_edges: _SegmentedPairwisePlan


_Plan = TypeVar("_Plan")
_PROBLEM_REDUCTION_CACHE: dict[
    int, tuple[weakref.ReferenceType[object], _ProblemReductionPlan]
] = {}
_TREE_REDUCTION_CACHE: dict[
    int, tuple[weakref.ReferenceType[object], _TreeReductionPlan]
] = {}


@dataclass(frozen=True)
class IEEEPreparedState:
    """Immutable vectorized interval state for one exact binary64 iterate."""

    problem: ReciprocalTransportProblem
    y: Array
    x_lower: Array | None
    x_upper: Array | None
    conductance_lower: Array | None
    conductance_upper: Array | None
    gradient_lower: Array | None
    gradient_upper: Array | None
    sigma_lower: float | None
    sigma_upper: float | None
    x_min_lower: float | None
    platform_contract_passed: bool
    topology_owner: ReciprocalTransportProblem | None = None
    reduction_plan: _ProblemReductionPlan | None = None
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
        return self.conductance_lower is not None

    @property
    def recovery_certified(self) -> bool:
        """Whether the state contains every enclosure used by recovery."""

        return bool(
            self.x_lower is not None
            and self.gradient_lower is not None
            and self.gradient_upper is not None
            and self.x_min_lower is not None
        )


def _seal_ieee_state(state: IEEEPreparedState) -> IEEEPreparedState:
    object.__setattr__(state, "_capability", _IEEE_PREPARED_TOKEN)
    object.__setattr__(
        state, "_problem_fingerprint", state.problem.content_fingerprint
    )
    object.__setattr__(
        state,
        "_identity_seal",
        (
            id(state.problem),
            id(state.y),
            id(state.x_lower),
            id(state.x_upper),
            id(state.conductance_lower),
            id(state.conductance_upper),
            id(state.gradient_lower),
            id(state.gradient_upper),
            state.sigma_lower,
            state.sigma_upper,
            state.x_min_lower,
            state.platform_contract_passed,
            id(state.topology_owner),
            id(state.reduction_plan),
        ),
    )
    return state


@dataclass(frozen=True)
class IEEEPlatformProbe:
    """Results of executable spot checks for the binary64 certificate contract.

    These checks exercise the current process and thread.  They are deliberately
    stronger than inspecting NumPy finfo, but they are not a formal proof of every
    input, compiler path, worker thread, or future floating-point operation.  The
    interval argument therefore remains conditional on conforming IEEE-754 basic
    operations and square root after this necessary, fail-closed gate passes.
    """

    binary64_layout: bool
    nextafter_adjacency: bool
    round_to_nearest_ties_to_even: bool
    gradual_underflow: bool
    basic_operation_spots: bool
    sqrt_rounding_spots: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.binary64_layout
            and self.nextafter_adjacency
            and self.round_to_nearest_ties_to_even
            and self.gradual_underflow
            and self.basic_operation_spots
            and self.sqrt_rounding_spots
        )

    @property
    def failed_checks(self) -> tuple[str, ...]:
        names = (
            "binary64_layout",
            "nextafter_adjacency",
            "round_to_nearest_ties_to_even",
            "gradual_underflow",
            "basic_operation_spots",
            "sqrt_rounding_spots",
        )
        return tuple(name for name in names if not getattr(self, name))


_PLATFORM_PROBE_CACHE: IEEEPlatformProbe | None = None


def _same_binary64(actual, expected) -> bool:
    """Compare binary64 bit patterns, including the sign of zero."""

    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    return bool(
        actual_array.shape == expected_array.shape
        and np.array_equal(
            actual_array.view(np.uint64), expected_array.view(np.uint64)
        )
    )


def _probe_safely(probe: Callable[[], bool]) -> bool:
    """Run one platform probe and fail closed on numerical/runtime errors."""

    try:
        with np.errstate(all="ignore"):
            return bool(probe())
    except Exception:
        return False


def _probe_binary64_layout() -> bool:
    info = np.finfo(np.float64)
    return bool(
        np.dtype(np.float64).kind == "f"
        and np.dtype(np.float64).itemsize == 8
        and info.bits == 64
        and info.nmant == 52
        and info.nexp == 11
        and info.minexp == -1022
        and info.maxexp == 1024
        and _same_binary64(info.eps, float.fromhex("0x1.0000000000000p-52"))
        and _same_binary64(
            info.epsneg, float.fromhex("0x1.0000000000000p-53")
        )
        and _same_binary64(
            info.tiny, float.fromhex("0x1.0000000000000p-1022")
        )
        and _same_binary64(
            info.smallest_subnormal,
            float.fromhex("0x0.0000000000001p-1022"),
        )
        and _same_binary64(
            info.max, float.fromhex("0x1.fffffffffffffp+1023")
        )
    )


def _probe_nextafter_adjacency() -> bool:
    minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000001p-1022")
    )
    minimum_normal = np.float64(
        float.fromhex("0x1.0000000000000p-1022")
    )
    maximum_subnormal = np.float64(
        float.fromhex("0x0.fffffffffffffp-1022")
    )
    maximum_finite = np.float64(
        float.fromhex("0x1.fffffffffffffp+1023")
    )
    starts = np.asarray(
        [
            0.0,
            -0.0,
            minimum_subnormal,
            -minimum_subnormal,
            minimum_normal,
            1.0,
            1.0,
            maximum_finite,
            np.inf,
        ],
        dtype=np.float64,
    )
    directions = np.asarray(
        [
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            0.0,
            np.inf,
            -np.inf,
            np.inf,
            0.0,
        ],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            minimum_subnormal,
            -minimum_subnormal,
            0.0,
            -0.0,
            maximum_subnormal,
            float.fromhex("0x1.0000000000001p+0"),
            float.fromhex("0x1.fffffffffffffp-1"),
            np.inf,
            maximum_finite,
        ],
        dtype=np.float64,
    )
    return _same_binary64(np.nextafter(starts, directions), expected)


def _probe_round_to_nearest_ties_to_even() -> bool:
    half_ulp = np.float64(float.fromhex("0x1.0000000000000p-53"))
    three_half_ulps = np.float64(
        float.fromhex("0x1.8000000000000p-52")
    )
    one_up = np.float64(float.fromhex("0x1.0000000000001p+0"))
    one_up_twice = np.float64(float.fromhex("0x1.0000000000002p+0"))
    left = np.asarray([1.0, -1.0, 1.0, -1.0, one_up, -one_up])
    right = np.asarray(
        [
            half_ulp,
            -half_ulp,
            three_half_ulps,
            -three_half_ulps,
            half_ulp,
            -half_ulp,
        ]
    )
    expected = np.asarray(
        [
            1.0,
            -1.0,
            one_up_twice,
            -one_up_twice,
            one_up_twice,
            -one_up_twice,
        ]
    )
    return _same_binary64(np.add(left, right), expected)


def _probe_gradual_underflow() -> bool:
    minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000001p-1022")
    )
    twice_minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000002p-1022")
    )
    maximum_subnormal = np.float64(
        float.fromhex("0x0.fffffffffffffp-1022")
    )
    minimum_normal = np.float64(
        float.fromhex("0x1.0000000000000p-1022")
    )
    half_minimum_normal = np.float64(
        float.fromhex("0x0.8000000000000p-1022")
    )
    additions = _same_binary64(
        np.add(
            np.asarray([minimum_subnormal, maximum_subnormal]),
            np.asarray([minimum_subnormal, minimum_subnormal]),
        ),
        np.asarray([twice_minimum_subnormal, minimum_normal]),
    )
    subtractions = _same_binary64(
        np.subtract(
            np.asarray([minimum_normal, twice_minimum_subnormal]),
            np.asarray([maximum_subnormal, minimum_subnormal]),
        ),
        np.asarray([minimum_subnormal, minimum_subnormal]),
    )
    multiplications = _same_binary64(
        np.multiply(
            np.asarray(
                [minimum_normal, minimum_subnormal, minimum_subnormal]
            ),
            np.asarray([0.5, 1.0, 0.75]),
        ),
        np.asarray(
            [half_minimum_normal, minimum_subnormal, minimum_subnormal]
        ),
    )
    divisions = _same_binary64(
        np.divide(
            np.asarray(
                [minimum_normal, minimum_subnormal, minimum_subnormal]
            ),
            np.asarray([2.0, minimum_subnormal, 2.0]),
        ),
        np.asarray([half_minimum_normal, 1.0, 0.0]),
    )
    subnormal_sqrt = _same_binary64(
        np.sqrt(minimum_subnormal),
        float.fromhex("0x1.0000000000000p-537"),
    )
    return bool(
        additions
        and subtractions
        and multiplications
        and divisions
        and subnormal_sqrt
    )


def _probe_basic_operation_spots() -> bool:
    one_up = np.float64(float.fromhex("0x1.0000000000001p+0"))
    additions = _same_binary64(
        np.add(np.asarray([0.5, one_up]), np.asarray([0.25, -1.0])),
        np.asarray([0.75, float.fromhex("0x1.0000000000000p-52")]),
    )
    subtractions = _same_binary64(
        np.subtract(
            np.asarray([1.0, -1.0]),
            np.asarray(
                [
                    float.fromhex("0x1.fffffffffffffp-1"),
                    -float.fromhex("0x1.0000000000001p+0"),
                ]
            ),
        ),
        np.asarray(
            [
                float.fromhex("0x1.0000000000000p-53"),
                float.fromhex("0x1.0000000000000p-52"),
            ]
        ),
    )
    multiplications = _same_binary64(
        np.multiply(
            np.asarray([one_up, 1.5, -0.125]),
            np.asarray([one_up, 2.0, 8.0]),
        ),
        np.asarray(
            [float.fromhex("0x1.0000000000002p+0"), 3.0, -1.0]
        ),
    )
    divisions = _same_binary64(
        np.divide(
            np.asarray([1.0, 1.0, -1.0, 7.0]),
            np.asarray([10.0, 3.0, 10.0, 2.0]),
        ),
        np.asarray(
            [
                float.fromhex("0x1.999999999999ap-4"),
                float.fromhex("0x1.5555555555555p-2"),
                -float.fromhex("0x1.999999999999ap-4"),
                3.5,
            ]
        ),
    )
    return bool(additions and subtractions and multiplications and divisions)


def _probe_sqrt_rounding_spots() -> bool:
    minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000001p-1022")
    )
    one_up = np.float64(float.fromhex("0x1.0000000000001p+0"))
    one_up_twice = np.float64(float.fromhex("0x1.0000000000002p+0"))
    values = np.asarray(
        [0.0, -0.0, 4.0, 2.0, one_up, one_up_twice, minimum_subnormal]
    )
    expected = np.asarray(
        [
            0.0,
            -0.0,
            2.0,
            float.fromhex("0x1.6a09e667f3bcdp+0"),
            1.0,
            one_up,
            float.fromhex("0x1.0000000000000p-537"),
        ]
    )
    return _same_binary64(np.sqrt(values), expected)


def ieee_platform_probe() -> IEEEPlatformProbe:
    """Run named, fail-closed probes for the active binary64 environment."""

    return IEEEPlatformProbe(
        binary64_layout=_probe_safely(_probe_binary64_layout),
        nextafter_adjacency=_probe_safely(_probe_nextafter_adjacency),
        round_to_nearest_ties_to_even=_probe_safely(
            _probe_round_to_nearest_ties_to_even
        ),
        gradual_underflow=_probe_safely(_probe_gradual_underflow),
        basic_operation_spots=_probe_safely(_probe_basic_operation_spots),
        sqrt_rounding_spots=_probe_safely(_probe_sqrt_rounding_spots),
    )


def _probe_active_fp_controls() -> bool:
    """Cheap current-thread sentinels for mutable FP controls and primitives."""

    minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000001p-1022")
    )
    minimum_normal = np.float64(
        float.fromhex("0x1.0000000000000p-1022")
    )
    half_minimum_normal = np.float64(
        float.fromhex("0x0.8000000000000p-1022")
    )
    half_ulp = np.float64(float.fromhex("0x1.0000000000000p-53"))
    three_half_ulps = np.float64(
        float.fromhex("0x1.8000000000000p-52")
    )
    return bool(
        np.nextafter(np.float64(0.0), _POSITIVE_INFINITY)
        == minimum_subnormal
        and np.add(np.float64(1.0), half_ulp) == 1.0
        and np.add(np.float64(1.0), three_half_ulps)
        == float.fromhex("0x1.0000000000002p+0")
        and np.subtract(
            np.float64(1.0),
            np.float64(float.fromhex("0x1.fffffffffffffp-1")),
        )
        == half_ulp
        and np.multiply(minimum_normal, np.float64(0.5))
        == half_minimum_normal
        and np.divide(minimum_subnormal, minimum_subnormal) == 1.0
        and np.sqrt(minimum_subnormal)
        == float.fromhex("0x1.0000000000000p-537")
        and np.sqrt(np.float64(2.0))
        == float.fromhex("0x1.6a09e667f3bcdp+0")
    )


def ieee_platform_contract() -> bool:
    """Gate the backend with full probes plus current-thread FP sentinels.

    Layout and implementation spot checks are cached after their first success;
    cheap arithmetic sentinels are repeated because rounding and FTZ/DAZ controls
    can be thread-local and mutable.
    """

    global _PLATFORM_PROBE_CACHE
    if _PLATFORM_PROBE_CACHE is None:
        probe = ieee_platform_probe()
        if not probe.passed:
            return False
        _PLATFORM_PROBE_CACHE = probe
    return _probe_safely(_probe_active_fp_controls)


def _down(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), _NEGATIVE_INFINITY)


def _up(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), _POSITIVE_INFINITY)


def _add_lower(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _down(np.asarray(left, dtype=float) + np.asarray(right, dtype=float))


def _add_upper(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _up(np.asarray(left, dtype=float) + np.asarray(right, dtype=float))


def _subtract_lower(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _down(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))


def _subtract_upper(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _up(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))


def _positive_multiply_lower(left, right):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = _down(np.asarray(left, dtype=float) * np.asarray(right, dtype=float))
    return np.maximum(value, 0.0)


def _positive_multiply_upper(left, right):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return _up(np.asarray(left, dtype=float) * np.asarray(right, dtype=float))


def _positive_divide_lower(numerator, denominator):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        value = _down(
            np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
        )
    return np.maximum(value, 0.0)


def _positive_divide_upper(numerator, denominator):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return _up(
            np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
        )


def _sqrt_lower(value):
    with np.errstate(invalid="ignore", under="ignore"):
        result = _down(np.sqrt(np.asarray(value, dtype=float)))
    return np.maximum(result, 0.0)


def _sqrt_upper(value):
    with np.errstate(invalid="ignore", under="ignore"):
        return _up(np.sqrt(np.asarray(value, dtype=float)))


def _interval_multiply(
    left_lower: Array,
    left_upper: Array,
    right_lower: Array,
    right_upper: Array,
) -> tuple[Array, Array]:
    lower_candidates = np.stack(
        [
            _down(left_lower * right_lower),
            _down(left_lower * right_upper),
            _down(left_upper * right_lower),
            _down(left_upper * right_upper),
        ]
    )
    upper_candidates = np.stack(
        [
            _up(left_lower * right_lower),
            _up(left_lower * right_upper),
            _up(left_upper * right_lower),
            _up(left_upper * right_upper),
        ]
    )
    return np.min(lower_candidates, axis=0), np.max(upper_candidates, axis=0)


def _readonly_indices(
    values: Array, dtype: np.dtype | type | None = None
) -> Array:
    raw = np.asarray(values)
    if dtype is None:
        if raw.size == 0 or (
            np.min(raw) >= np.iinfo(np.int32).min
            and np.max(raw) <= np.iinfo(np.int32).max
        ):
            dtype = np.int32
        else:
            dtype = np.int64
    contiguous = np.ascontiguousarray(raw, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape)


def _compile_segmented_pairwise_plan(
    groups: Array,
    size: int,
    index_dtype: np.dtype | type | None = None,
) -> _SegmentedPairwisePlan:
    """Compile a linear-size stable pairwise-addition DAG for fixed groups.

    Each actual addition creates one immutable workspace node.  Completed
    singleton groups leave the active frontier immediately instead of being
    copied through every remaining level of a larger group.  Consequently the
    plan contains exactly ``input_size - nonempty_groups`` addition nodes and
    has linear storage even for highly skewed degree distributions.
    """

    groups = np.asarray(groups, dtype=np.int64)
    if (
        size < 1
        or groups.ndim != 1
        or np.any(groups < 0)
        or np.any(groups >= size)
    ):
        raise ValueError("invalid segmented-reduction groups")
    if groups.size == 0:
        empty = _readonly_indices(np.empty(0, dtype=np.int64), index_dtype)
        return _SegmentedPairwisePlan(0, int(size), 0, None, empty, empty, ())
    already_grouped = bool(
        groups.size <= 1 or np.all(groups[1:] >= groups[:-1])
    )
    if already_grouped:
        order = None
        current_groups = groups.copy()
    else:
        order = np.argsort(groups, kind="stable")
        current_groups = groups[order]
    current_nodes = np.arange(groups.size, dtype=np.int64)
    next_node = int(groups.size)
    levels: list[_PairwiseReductionLevel] = []
    completed_groups: list[Array] = []
    completed_nodes: list[Array] = []
    while current_groups.size:
        starts = np.empty(current_groups.size, dtype=bool)
        starts[0] = True
        starts[1:] = current_groups[1:] != current_groups[:-1]
        indices = np.arange(current_groups.size, dtype=np.int64)
        group_starts = np.maximum.accumulate(np.where(starts, indices, 0))
        ranks = indices - group_starts
        selected = np.flatnonzero((ranks & 1) == 0)
        partners = selected + 1
        paired = partners < current_groups.size
        paired[paired] = (
            current_groups[partners[paired]] == current_groups[selected[paired]]
        )
        output_positions = np.flatnonzero(paired)
        pair_count = int(output_positions.size)
        next_groups = current_groups[selected]
        next_nodes = current_nodes[selected].copy()
        if pair_count:
            outputs = np.arange(next_node, next_node + pair_count, dtype=np.int64)
            levels.append(
                _PairwiseReductionLevel(
                    _readonly_indices(outputs, index_dtype),
                    _readonly_indices(current_nodes[selected[paired]], index_dtype),
                    _readonly_indices(current_nodes[partners[paired]], index_dtype),
                )
            )
            next_nodes[output_positions] = outputs
            next_node += pair_count

        next_starts = np.empty(next_groups.size, dtype=bool)
        next_starts[0] = True
        next_starts[1:] = next_groups[1:] != next_groups[:-1]
        start_positions = np.flatnonzero(next_starts)
        counts = np.diff(np.append(start_positions, next_groups.size))
        finished = counts == 1
        if np.any(finished):
            finished_positions = start_positions[finished]
            completed_groups.append(next_groups[finished_positions])
            completed_nodes.append(next_nodes[finished_positions])

        active_group = np.repeat(~finished, counts)
        current_groups = next_groups[active_group]
        current_nodes = next_nodes[active_group]

    output_groups = np.concatenate(completed_groups)
    output_nodes = np.concatenate(completed_nodes)
    return _SegmentedPairwisePlan(
        int(groups.size),
        int(size),
        next_node,
        None if order is None else _readonly_indices(order, index_dtype),
        _readonly_indices(output_groups, index_dtype),
        _readonly_indices(output_nodes, index_dtype),
        tuple(levels),
    )


def _execute_segmented_pairwise_plan(
    lower_values: Array,
    upper_values: Array,
    plan: _SegmentedPairwisePlan,
) -> tuple[Array, Array, bool]:
    """Execute a compiled DAG with the same outward additions as the reference."""

    lower_values = np.asarray(lower_values, dtype=float)
    upper_values = np.asarray(upper_values, dtype=float)
    invalid_lower = np.zeros(plan.output_size, dtype=float)
    invalid_upper = np.full(plan.output_size, np.inf, dtype=float)
    if (
        lower_values.ndim != 1
        or lower_values.size != plan.input_size
        or upper_values.shape != lower_values.shape
        or not np.all(np.isfinite(lower_values))
        or not np.all(np.isfinite(upper_values))
        or np.any(lower_values > upper_values)
    ):
        return invalid_lower, invalid_upper, False
    if lower_values.size == 0:
        return invalid_lower.copy(), invalid_lower, True

    workspace_lower = np.empty(plan.workspace_size, dtype=float)
    workspace_upper = np.empty(plan.workspace_size, dtype=float)
    if plan.order is None:
        workspace_lower[: plan.input_size] = lower_values
        workspace_upper[: plan.input_size] = upper_values
    else:
        workspace_lower[: plan.input_size] = lower_values[plan.order]
        workspace_upper[: plan.input_size] = upper_values[plan.order]
    for level in plan.levels:
        workspace_lower[level.output] = _add_lower(
            workspace_lower[level.left], workspace_lower[level.right]
        )
        workspace_upper[level.output] = _add_upper(
            workspace_upper[level.left], workspace_upper[level.right]
        )

    lower = np.zeros(plan.output_size, dtype=float)
    upper = np.zeros(plan.output_size, dtype=float)
    lower[plan.output_groups] = workspace_lower[plan.output_nodes]
    upper[plan.output_groups] = workspace_upper[plan.output_nodes]
    valid = bool(
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(lower <= upper)
    )
    return lower, upper, valid


def _identity_cached_plan(
    cache: dict[int, tuple[weakref.ReferenceType[object], _Plan]],
    owner: object,
    builder: Callable[[], _Plan],
) -> _Plan:
    """Cache a plan by object identity without retaining its owner."""

    key = id(owner)
    record = cache.get(key)
    if record is not None and record[0]() is owner:
        return record[1]
    plan = builder()

    def discard(reference: weakref.ReferenceType[object]) -> None:
        current = cache.get(key)
        if current is not None and current[0] is reference:
            cache.pop(key, None)

    reference = weakref.ref(owner, discard)
    cache[key] = (reference, plan)
    return plan


def _compile_problem_reduction_plan(
    problem: ReciprocalTransportProblem,
) -> _ProblemReductionPlan:
    return _ProblemReductionPlan(
        _compile_segmented_pairwise_plan(problem.tails, problem.n_left),
        _compile_segmented_pairwise_plan(
            problem.heads - problem.n_left, problem.n_right
        ),
        _compile_segmented_pairwise_plan(
            np.zeros(problem.n - 1, dtype=np.int64), 1
        ),
        _compile_segmented_pairwise_plan(
            np.zeros(problem.n, dtype=np.int64), 1
        ),
        _compile_segmented_pairwise_plan(
            np.zeros(problem.m, dtype=np.int64), 1
        ),
    )


def _problem_reduction_plan(
    problem: ReciprocalTransportProblem,
) -> _ProblemReductionPlan:
    return _identity_cached_plan(
        _PROBLEM_REDUCTION_CACHE,
        problem,
        lambda: _compile_problem_reduction_plan(problem),
    )


def _validated_topology_owner(
    problem: ReciprocalTransportProblem,
    topology_owner: ReciprocalTransportProblem | None,
) -> ReciprocalTransportProblem:
    """Return a content-validated immutable owner for reusable graph plans."""

    if topology_owner is None or topology_owner is problem:
        return problem
    compatible = bool(
        topology_owner.n_left == problem.n_left
        and topology_owner.n_right == problem.n_right
        and topology_owner.n == problem.n
        and topology_owner.m == problem.m
        and np.array_equal(topology_owner.tails, problem.tails)
        and np.array_equal(topology_owner.heads, problem.heads)
    )
    if not compatible:
        raise ValueError("topology owner is incompatible with current problem")
    return topology_owner


def _compile_tree_reduction_plan(router: TreeRouter) -> _TreeReductionPlan:
    router.validate_for(router.problem)
    levels: list[_TreeLevelReductionPlan] = []
    for level in range(router.level_offsets.size - 2, 0, -1):
        start = int(router.level_offsets[level])
        end = int(router.level_offsets[level + 1])
        vertices = router.level_order[start:end]
        parents = router.parent[vertices]
        active_parents, compact_groups = np.unique(
            parents, return_inverse=True
        )
        signs = _immutable_float_snapshot(
            router.child_incidence_sign[vertices]
        )
        levels.append(
            _TreeLevelReductionPlan(
                _readonly_indices(vertices, np.int64),
                signs,
                _readonly_indices(active_parents, np.int64),
                _compile_segmented_pairwise_plan(
                    compact_groups, active_parents.size, np.int64
                ),
            )
        )
    return _TreeReductionPlan(
        tuple(levels),
        _compile_segmented_pairwise_plan(
            np.zeros(router.problem.n - 1, dtype=np.int64), 1, np.int64
        ),
    )


def _tree_reduction_plan(router: TreeRouter) -> _TreeReductionPlan:
    router.validate_for(router.problem)
    return _identity_cached_plan(
        _TREE_REDUCTION_CACHE,
        router,
        lambda: _compile_tree_reduction_plan(router),
    )


def _segmented_pairwise_sum_bounds(
    lower_values: Array,
    upper_values: Array,
    groups: Array,
    size: int,
) -> tuple[Array, Array, bool]:
    """Enclose grouped sums with an outward balanced tree per group.

    This compatibility path compiles a fresh plan.  Fixed graph and tree groups
    use cached plans internally, while preserving this uncached private API for
    validation and reference tests.
    """

    lower_values = np.asarray(lower_values, dtype=float)
    upper_values = np.asarray(upper_values, dtype=float)
    groups = np.asarray(groups, dtype=np.int64)
    output_size = max(1, int(size))
    invalid_lower = np.zeros(output_size, dtype=float)
    invalid_upper = np.full(output_size, np.inf, dtype=float)
    if (
        size < 1
        or lower_values.ndim != 1
        or upper_values.shape != lower_values.shape
        or groups.shape != lower_values.shape
        or np.any(groups < 0)
        or np.any(groups >= size)
        or not np.all(np.isfinite(lower_values))
        or not np.all(np.isfinite(upper_values))
        or np.any(lower_values > upper_values)
    ):
        return invalid_lower, invalid_upper, False
    plan = _compile_segmented_pairwise_plan(groups, size)
    return _execute_segmented_pairwise_plan(
        lower_values, upper_values, plan
    )


def _grouped_nonnegative_sum_bounds(
    values: Array, groups: Array, size: int
) -> tuple[Array, Array, bool]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=np.int64)
    if (
        values.ndim != 1
        or groups.shape != values.shape
        or size < 1
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
    ):
        zeros = np.zeros(max(1, size), dtype=float)
        return zeros, np.full_like(zeros, np.inf), False
    lower, upper, valid = _segmented_pairwise_sum_bounds(
        values, values, groups, size
    )
    lower = np.maximum(lower, 0.0)
    return lower, upper, valid


def _grouped_interval_sum_bounds(
    lower_values: Array,
    upper_values: Array,
    groups: Array,
    size: int,
) -> tuple[Array, Array, bool]:
    return _segmented_pairwise_sum_bounds(
        lower_values, upper_values, groups, size
    )


def _planned_nonnegative_sum_bounds(
    values: Array, plan: _SegmentedPairwisePlan
) -> tuple[Array, Array, bool]:
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or values.size != plan.input_size
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
    ):
        zeros = np.zeros(plan.output_size, dtype=float)
        return zeros, np.full_like(zeros, np.inf), False
    lower, upper, valid = _execute_segmented_pairwise_plan(
        values, values, plan
    )
    return np.maximum(lower, 0.0), upper, valid


def _planned_interval_sum_bounds(
    lower_values: Array,
    upper_values: Array,
    plan: _SegmentedPairwisePlan,
) -> tuple[Array, Array, bool]:
    return _execute_segmented_pairwise_plan(
        lower_values, upper_values, plan
    )


def _planned_interval_sum(
    lower: Array,
    upper: Array,
    plan: _SegmentedPairwisePlan,
) -> tuple[float, float, bool]:
    result_lower, result_upper, valid = _planned_interval_sum_bounds(
        np.ravel(lower), np.ravel(upper), plan
    )
    return float(result_lower[0]), float(result_upper[0]), valid


def _planned_nonnegative_sum_upper(
    values: Array, plan: _SegmentedPairwisePlan
) -> tuple[float, bool]:
    _, upper, valid = _planned_nonnegative_sum_bounds(
        np.ravel(values), plan
    )
    return float(upper[0]), valid


def _interval_sum_bounds(lower: Array, upper: Array) -> tuple[float, float, bool]:
    groups = np.zeros(np.asarray(lower).size, dtype=np.int64)
    result_lower, result_upper, valid = _grouped_interval_sum_bounds(
        np.ravel(lower), np.ravel(upper), groups, 1
    )
    return float(result_lower[0]), float(result_upper[0]), valid


def _nonnegative_sum_upper(values: Array) -> tuple[float, bool]:
    groups = np.zeros(np.asarray(values).size, dtype=np.int64)
    _, upper, valid = _grouped_nonnegative_sum_bounds(
        np.ravel(values), groups, 1
    )
    return float(upper[0]), valid


def _invalid_prepared(
    problem: ReciprocalTransportProblem, y: Array, contract: bool
) -> IEEEPreparedState:
    snapshot = _immutable_float_snapshot(y)
    return _seal_ieee_state(
        IEEEPreparedState(
            problem,
            snapshot,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            contract,
        )
    )


def prepare_ieee_state(
    problem: ReciprocalTransportProblem, y: Array
) -> IEEEPreparedState:
    """Build a vectorized outward state for one binary64 dual iterate."""

    return _prepare_ieee_state(
        problem, y, recovery_only=False, topology_owner=problem
    )


def prepare_ieee_recovery_state(
    problem: ReciprocalTransportProblem,
    y: Array,
    topology_owner: ReciprocalTransportProblem | None = None,
) -> IEEEPreparedState:
    """Build only the q/x/gradient enclosures needed by terminal recovery."""

    return _prepare_ieee_state(
        problem,
        y,
        recovery_only=True,
        topology_owner=topology_owner,
    )


def _prepare_ieee_state(
    problem: ReciprocalTransportProblem,
    y: Array,
    *,
    recovery_only: bool,
    topology_owner: ReciprocalTransportProblem | None,
) -> IEEEPreparedState:
    """Shared outward state construction with an explicit lazy recovery path."""

    topology_owner = _validated_topology_owner(problem, topology_owner)
    y = np.asarray(y, dtype=float)
    contract = ieee_platform_contract()
    if y.shape != (problem.n,) or not contract or not np.all(np.isfinite(y)):
        return _invalid_prepared(problem, y, contract)
    snapshot = _immutable_float_snapshot(y)

    first_lower = _subtract_lower(problem.c, y[problem.tails])
    first_upper = _subtract_upper(problem.c, y[problem.tails])
    q_lower = _add_lower(first_lower, y[problem.heads])
    q_upper = _add_upper(first_upper, y[problem.heads])
    if (
        np.any(q_lower <= 0.0)
        or not np.all(np.isfinite(q_lower))
        or not np.all(np.isfinite(q_upper))
    ):
        return _invalid_prepared(problem, snapshot, contract)

    ratio_lower = _positive_divide_lower(problem.mu, q_upper)
    ratio_upper = _positive_divide_upper(problem.mu, q_lower)
    x_lower = _sqrt_lower(ratio_lower)
    x_upper = _sqrt_upper(ratio_upper)

    conductance_lower = conductance_upper = None
    sigma_edge_lower = sigma_edge_upper = None
    if not recovery_only:
        root_mu_lower = _sqrt_lower(problem.mu)
        root_mu_upper = _sqrt_upper(problem.mu)
        root_q_lower = _sqrt_lower(q_lower)
        root_q_upper = _sqrt_upper(q_upper)
        q_three_halves_lower = _positive_multiply_lower(q_lower, root_q_lower)
        q_three_halves_upper = _positive_multiply_upper(q_upper, root_q_upper)
        denominator_lower = _positive_multiply_lower(2.0, q_three_halves_lower)
        denominator_upper = _positive_multiply_upper(2.0, q_three_halves_upper)
        conductance_lower = _positive_divide_lower(
            root_mu_lower, denominator_upper
        )
        conductance_upper = _positive_divide_upper(
            root_mu_upper, denominator_lower
        )

        mu_q_lower = _positive_multiply_lower(problem.mu, q_lower)
        mu_q_upper = _positive_multiply_upper(problem.mu, q_upper)
        sigma_edge_lower = _sqrt_lower(_sqrt_lower(mu_q_lower))
        sigma_edge_upper = _sqrt_upper(_sqrt_upper(mu_q_upper))

    reduction_plan = _problem_reduction_plan(topology_owner)
    row_lower, row_upper, ok_rows = _planned_interval_sum_bounds(
        x_lower, x_upper, reduction_plan.clients
    )
    column_lower, column_upper, ok_columns = _planned_interval_sum_bounds(
        x_lower, x_upper, reduction_plan.models
    )

    gradient_lower = np.empty(problem.n, dtype=float)
    gradient_upper = np.empty(problem.n, dtype=float)
    gradient_lower[: problem.n_left] = _subtract_lower(
        row_lower, problem.beta[: problem.n_left]
    )
    gradient_upper[: problem.n_left] = _subtract_upper(
        row_upper, problem.beta[: problem.n_left]
    )
    model_beta = problem.beta[problem.n_left :]
    gradient_lower[problem.n_left :] = _subtract_lower(-column_upper, model_beta)
    gradient_upper[problem.n_left :] = _subtract_upper(-column_lower, model_beta)
    total_lower, total_upper, ok_balance = _planned_interval_sum(
        gradient_lower[:-1],
        gradient_upper[:-1],
        reduction_plan.reduced_nodes,
    )
    gradient_lower[-1] = -total_upper
    gradient_upper[-1] = -total_lower

    valid = bool(
        ok_rows
        and ok_columns
        and ok_balance
        and np.all(np.isfinite(x_lower))
        and np.all(np.isfinite(x_upper))
        and np.all(x_lower > 0.0)
        and np.all(np.isfinite(gradient_lower))
        and np.all(np.isfinite(gradient_upper))
        and np.all(gradient_lower <= gradient_upper)
        and (
            recovery_only
            or (
                np.all(np.isfinite(conductance_lower))
                and np.all(np.isfinite(conductance_upper))
                and np.all(conductance_lower > 0.0)
                and np.all(conductance_lower <= conductance_upper)
                and np.all(np.isfinite(sigma_edge_lower))
                and np.all(sigma_edge_lower > 0.0)
            )
        )
    )
    if not valid:
        return _invalid_prepared(problem, snapshot, contract)

    stored_x_lower = _immutable_float_snapshot(x_lower)
    stored_x_upper = (
        None if recovery_only else _immutable_float_snapshot(x_upper)
    )
    stored_conductance_lower = (
        None
        if conductance_lower is None
        else _immutable_float_snapshot(conductance_lower)
    )
    stored_conductance_upper = (
        None
        if conductance_upper is None
        else _immutable_float_snapshot(conductance_upper)
    )
    stored_gradient_lower = _immutable_float_snapshot(gradient_lower)
    stored_gradient_upper = _immutable_float_snapshot(gradient_upper)
    return _seal_ieee_state(
        IEEEPreparedState(
            problem,
            snapshot,
            stored_x_lower,
            stored_x_upper,
            stored_conductance_lower,
            stored_conductance_upper,
            stored_gradient_lower,
            stored_gradient_upper,
            None if recovery_only else float(np.min(sigma_edge_lower)),
            None if recovery_only else float(np.min(sigma_edge_upper)),
            float(np.min(x_lower)),
            contract,
            topology_owner,
            reduction_plan,
        )
    )


def _check_prepared(
    problem: ReciprocalTransportProblem,
    y: Array,
    prepared: IEEEPreparedState | None,
    *,
    recovery_only: bool = False,
) -> IEEEPreparedState:
    if prepared is None:
        if recovery_only:
            return prepare_ieee_recovery_state(problem, y)
        return prepare_ieee_state(problem, y)
    if getattr(prepared, "_capability", None) is not _IEEE_PREPARED_TOKEN:
        raise ValueError("untrusted or stale prepared IEEE state")
    expected_seal = (
        id(prepared.problem),
        id(prepared.y),
        id(prepared.x_lower),
        id(prepared.x_upper),
        id(prepared.conductance_lower),
        id(prepared.conductance_upper),
        id(prepared.gradient_lower),
        id(prepared.gradient_upper),
        prepared.sigma_lower,
        prepared.sigma_upper,
        prepared.x_min_lower,
        prepared.platform_contract_passed,
        id(prepared.topology_owner),
        id(prepared.reduction_plan),
    )
    if (
        prepared._identity_seal != expected_seal
        or prepared._problem_fingerprint != prepared.problem.content_fingerprint
    ):
        raise ValueError("untrusted or stale prepared IEEE state")
    if prepared.problem is not problem:
        raise ValueError("prepared IEEE state belongs to another problem")
    values = np.asarray(y, dtype=float)
    if values.shape != prepared.y.shape or not np.array_equal(values, prepared.y):
        raise ValueError("prepared IEEE state belongs to another iterate")
    if not prepared.platform_contract_passed or not ieee_platform_contract():
        # FP controls can change after preparation or differ across threads.
        return _invalid_prepared(problem, values, False)
    return prepared


def _tree_flow_bounds(
    router: TreeRouter,
    demand_lower: Array,
    demand_upper: Array,
    reduction_plan: _TreeReductionPlan | None = None,
) -> tuple[Array, Array, bool]:
    """Enclose the signed tree flow for one balanced demand interval."""

    subtree_lower = np.asarray(demand_lower, dtype=float).copy()
    subtree_upper = np.asarray(demand_upper, dtype=float).copy()
    if (
        subtree_lower.shape != (router.problem.n,)
        or subtree_upper.shape != subtree_lower.shape
        or np.any(subtree_lower > subtree_upper)
    ):
        invalid = np.full(router.problem.n, np.inf)
        return -invalid, invalid, False
    if reduction_plan is None:
        reduction_plan = _tree_reduction_plan(router)
    flow_lower = np.zeros(router.problem.n, dtype=float)
    flow_upper = np.zeros(router.problem.n, dtype=float)
    valid = True
    for level_plan in reduction_plan.levels:
        vertices = level_plan.vertices
        signs = level_plan.signs
        level_flow_lower = np.where(
            signs > 0.0, subtree_lower[vertices], -subtree_upper[vertices]
        )
        level_flow_upper = np.where(
            signs > 0.0, subtree_upper[vertices], -subtree_lower[vertices]
        )
        flow_lower[vertices] = level_flow_lower
        flow_upper[vertices] = level_flow_upper

        increment_lower, increment_upper, ok = _planned_interval_sum_bounds(
            subtree_lower[vertices],
            subtree_upper[vertices],
            level_plan.reduction,
        )
        active_parents = level_plan.active_parents
        subtree_lower[active_parents] = _add_lower(
            subtree_lower[active_parents], increment_lower
        )
        subtree_upper[active_parents] = _add_upper(
            subtree_upper[active_parents], increment_upper
        )
        valid = bool(valid and ok)

    valid = bool(
        valid
        and np.all(np.isfinite(flow_lower))
        and np.all(np.isfinite(flow_upper))
        and np.all(flow_lower <= flow_upper)
    )
    return flow_lower, flow_upper, valid


def _tree_flow_abs_upper(
    router: TreeRouter,
    demand_lower: Array,
    demand_upper: Array,
    reduction_plan: _TreeReductionPlan | None = None,
) -> tuple[Array, bool]:
    """Return the legacy absolute tree-flow enclosure."""

    flow_lower, flow_upper, valid = _tree_flow_bounds(
        router, demand_lower, demand_upper, reduction_plan
    )
    flow_abs = np.maximum(np.abs(flow_lower), np.abs(flow_upper))
    return flow_abs, bool(valid and np.all(np.isfinite(flow_abs)))


def _tree_energy_upper(
    router: TreeRouter,
    demand_lower: Array,
    demand_upper: Array,
    conductance_lower: Array,
    reduction_plan: _TreeReductionPlan | None = None,
) -> tuple[float, bool]:
    if reduction_plan is None:
        reduction_plan = _tree_reduction_plan(router)
    flow_abs, valid = _tree_flow_abs_upper(
        router, demand_lower, demand_upper, reduction_plan
    )
    vertices = router.level_order[1:]
    edges = router.parent_edge[vertices]
    denominator = conductance_lower[edges]
    squared_upper = _positive_multiply_upper(flow_abs[vertices], flow_abs[vertices])
    terms_upper = _positive_divide_upper(squared_upper, denominator)
    energy_upper, sum_ok = _planned_nonnegative_sum_upper(
        terms_upper, reduction_plan.tree_edges
    )
    valid = bool(
        valid
        and sum_ok
        and np.all(denominator > 0.0)
        and np.all(np.isfinite(terms_upper))
        and np.isfinite(energy_upper)
    )
    return (energy_upper if valid else float("inf")), valid


def ieee_basin_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    router: TreeRouter,
    threshold: float = 0.1,
    prepared_state: IEEEPreparedState | None = None,
) -> IntervalBasinCertificate:
    router.validate_for(problem)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 0.1:
        raise ValueError("basin threshold must be finite and in (0, 0.1]")
    started = perf_counter()
    state = _check_prepared(problem, y, prepared_state)
    if not state.domain_certified:
        return IntervalBasinCertificate(
            False,
            float("inf"),
            float("inf"),
            0.0,
            False,
            perf_counter() - started,
            53,
            BACKEND_NAME,
        )
    energy_upper, valid = _tree_energy_upper(
        router,
        state.gradient_lower,
        state.gradient_upper,
        state.conductance_lower,
    )
    decrement_upper = float(_sqrt_upper(energy_upper)) if valid else float("inf")
    eta_upper = (
        float(
            _positive_divide_upper(
                _sqrt_upper(_positive_multiply_upper(2.0, energy_upper)),
                state.sigma_lower,
            )
        )
        if valid
        else float("inf")
    )
    passed = bool(valid and np.isfinite(eta_upper) and eta_upper <= threshold)
    return IntervalBasinCertificate(
        True,
        eta_upper,
        decrement_upper,
        float(state.sigma_lower),
        passed,
        perf_counter() - started,
        53,
        BACKEND_NAME,
    )


def ieee_forcing_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    direction: Array,
    router: TreeRouter,
    prepared_state: IEEEPreparedState | None = None,
) -> IntervalForcingCertificate:
    """Evaluate the posterior forcing inequality with binary64 enclosures."""

    router.validate_for(problem)
    started = perf_counter()
    state = _check_prepared(problem, y, prepared_state)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (problem.n,):
        raise ValueError("direction has the wrong dimension")
    if not state.domain_certified or not np.all(np.isfinite(direction)):
        return IntervalForcingCertificate(
            float("inf"), 0.0, 0.0, 0.0, False,
            perf_counter() - started, 53, BACKEND_NAME
        )

    step_lower = _subtract_lower(
        direction[problem.tails], direction[problem.heads]
    )
    step_upper = _subtract_upper(
        direction[problem.tails], direction[problem.heads]
    )
    weighted_lower, weighted_upper = _interval_multiply(
        state.conductance_lower,
        state.conductance_upper,
        step_lower,
        step_upper,
    )

    problem_plan = (
        state.reduction_plan
        if state.reduction_plan is not None
        else _problem_reduction_plan(problem)
    )
    client_lower, client_upper, ok_client = _planned_interval_sum_bounds(
        weighted_lower, weighted_upper, problem_plan.clients
    )
    model_lower, model_upper, ok_model = _planned_interval_sum_bounds(
        -weighted_upper, -weighted_lower, problem_plan.models
    )
    residual_lower = state.gradient_lower.copy()
    residual_upper = state.gradient_upper.copy()
    residual_lower[: problem.n_left] = _add_lower(
        residual_lower[: problem.n_left], client_lower
    )
    residual_upper[: problem.n_left] = _add_upper(
        residual_upper[: problem.n_left], client_upper
    )
    residual_lower[problem.n_left :] = _add_lower(
        residual_lower[problem.n_left :], model_lower
    )
    residual_upper[problem.n_left :] = _add_upper(
        residual_upper[problem.n_left :], model_upper
    )
    total_lower, total_upper, ok_balance = _planned_interval_sum(
        residual_lower[:-1],
        residual_upper[:-1],
        problem_plan.reduced_nodes,
    )
    residual_lower[-1] = -total_upper
    residual_upper[-1] = -total_lower
    residual_energy_upper, ok_energy = _tree_energy_upper(
        router,
        residual_lower,
        residual_upper,
        state.conductance_lower,
    )
    residual_norm_upper = float(_sqrt_upper(residual_energy_upper))

    step_abs_upper = np.maximum(np.abs(step_lower), np.abs(step_upper))
    denominator_terms = _positive_multiply_upper(
        state.conductance_upper,
        _positive_multiply_upper(step_abs_upper, step_abs_upper),
    )
    denominator_upper, ok_denominator = _planned_nonnegative_sum_upper(
        denominator_terms, problem_plan.all_edges
    )

    gauge_lower = _subtract_lower(direction[:-1], direction[-1])
    gauge_upper = _subtract_upper(direction[:-1], direction[-1])
    product_lower, product_upper = _interval_multiply(
        state.gradient_lower[:-1],
        state.gradient_upper[:-1],
        gauge_lower,
        gauge_upper,
    )
    numerator_lower_endpoint, numerator_upper_endpoint, ok_numerator = (
        _planned_interval_sum(
            product_lower, product_upper, problem_plan.reduced_nodes
        )
    )
    if numerator_lower_endpoint > 0.0:
        absolute_numerator_lower = numerator_lower_endpoint
    elif numerator_upper_endpoint < 0.0:
        absolute_numerator_lower = -numerator_upper_endpoint
    else:
        absolute_numerator_lower = 0.0

    valid = bool(
        ok_client
        and ok_model
        and ok_balance
        and ok_energy
        and ok_denominator
        and ok_numerator
        and denominator_upper > 0.0
        and np.isfinite(residual_norm_upper)
    )
    if valid:
        denominator_root_upper = _sqrt_upper(denominator_upper)
        decrement_lower = float(
            _positive_divide_lower(
                absolute_numerator_lower, denominator_root_upper
            )
        )
        eta_lower = float(
            _positive_divide_lower(
                _positive_multiply_lower(_sqrt_lower(2.0), decrement_lower),
                state.sigma_upper,
            )
        )
        threshold_lower = float(
            _positive_divide_lower(
                _positive_multiply_lower(eta_lower, decrement_lower), 4.0
            )
        )
    else:
        residual_norm_upper = float("inf")
        decrement_lower = eta_lower = threshold_lower = 0.0
    return IntervalForcingCertificate(
        residual_norm_upper,
        decrement_lower,
        eta_lower,
        threshold_lower,
        bool(valid and residual_norm_upper <= threshold_lower),
        perf_counter() - started,
        53,
        BACKEND_NAME,
    )


def _edgewise_recovery_gap_upper(
    mu: Array,
    x_lower: Array,
    correction_abs_upper: Array,
    reduction_plan: _SegmentedPairwisePlan | None = None,
) -> tuple[float, bool, bool]:
    """Return the outward tree-recovery gap bound and positivity status."""

    mu = np.asarray(mu, dtype=float)
    x_lower = np.asarray(x_lower, dtype=float)
    correction_abs_upper = np.asarray(correction_abs_upper, dtype=float)
    valid = bool(
        mu.ndim == 1
        and x_lower.shape == mu.shape
        and correction_abs_upper.shape == mu.shape
        and np.all(np.isfinite(mu))
        and np.all(np.isfinite(x_lower))
        and np.all(np.isfinite(correction_abs_upper))
        and np.all(mu > 0.0)
        and np.all(x_lower > 0.0)
        and np.all(correction_abs_upper >= 0.0)
    )
    if not valid:
        return float("inf"), False, False
    corrected_lower = _subtract_lower(x_lower, correction_abs_upper)
    positivity = bool(
        np.all(np.isfinite(corrected_lower))
        and np.all(corrected_lower > 0.0)
    )
    if not positivity:
        return float("inf"), False, True
    correction_squared = _positive_multiply_upper(
        correction_abs_upper, correction_abs_upper
    )
    numerator = _positive_multiply_upper(mu, correction_squared)
    x_squared_lower = _positive_multiply_lower(x_lower, x_lower)
    denominator = _positive_multiply_lower(x_squared_lower, corrected_lower)
    terms = _positive_divide_upper(numerator, denominator)
    if reduction_plan is None:
        gap_upper, sum_valid = _nonnegative_sum_upper(terms)
    else:
        gap_upper, sum_valid = _planned_nonnegative_sum_upper(
            terms, reduction_plan
        )
    valid = bool(
        sum_valid
        and np.all(denominator > 0.0)
        and np.all(np.isfinite(terms))
        and np.isfinite(gap_upper)
    )
    return (gap_upper if valid else float("inf")), bool(valid), valid


def _edgewise_signed_recovery_gap_upper(
    mu: Array,
    x_lower: Array,
    correction_lower: Array,
    correction_upper: Array,
    reduction_plan: _SegmentedPairwisePlan | None = None,
) -> tuple[float, bool, bool]:
    """Bound recovery with signed correction intervals.

    This is the production tree predicate.  The legacy absolute helper above
    remains available for ablations.  Retaining the signed lower endpoint gives
    ``z_lower = down(x_lower + correction_lower)``, which is never weaker than
    replacing the correction by ``-maxabs`` and converges to the exact recovered
    coordinate as interval widths vanish.
    """

    mu = np.asarray(mu, dtype=float)
    x_lower = np.asarray(x_lower, dtype=float)
    correction_lower = np.asarray(correction_lower, dtype=float)
    correction_upper = np.asarray(correction_upper, dtype=float)
    valid = bool(
        mu.ndim == 1
        and x_lower.shape == mu.shape
        and correction_lower.shape == mu.shape
        and correction_upper.shape == mu.shape
        and np.all(np.isfinite(mu))
        and np.all(np.isfinite(x_lower))
        and np.all(np.isfinite(correction_lower))
        and np.all(np.isfinite(correction_upper))
        and np.all(mu > 0.0)
        and np.all(x_lower > 0.0)
        and np.all(correction_lower <= correction_upper)
    )
    if not valid:
        return float("inf"), False, False
    corrected_lower = _add_lower(x_lower, correction_lower)
    positivity = bool(
        np.all(np.isfinite(corrected_lower))
        and np.all(corrected_lower > 0.0)
    )
    if not positivity:
        return float("inf"), False, True
    correction_abs_upper = np.maximum(
        np.abs(correction_lower), np.abs(correction_upper)
    )
    correction_squared = _positive_multiply_upper(
        correction_abs_upper, correction_abs_upper
    )
    numerator = _positive_multiply_upper(mu, correction_squared)
    x_squared_lower = _positive_multiply_lower(x_lower, x_lower)
    denominator = _positive_multiply_lower(x_squared_lower, corrected_lower)
    terms = _positive_divide_upper(numerator, denominator)
    if reduction_plan is None:
        gap_upper, sum_valid = _nonnegative_sum_upper(terms)
    else:
        gap_upper, sum_valid = _planned_nonnegative_sum_upper(
            terms, reduction_plan
        )
    valid = bool(
        sum_valid
        and np.all(denominator > 0.0)
        and np.all(np.isfinite(terms))
        and np.isfinite(gap_upper)
    )
    return (gap_upper if valid else float("inf")), bool(valid), valid


def ieee_recovery_certificate(
    problem: ReciprocalTransportProblem,
    y: Array,
    epsilon: float,
    prepared_state: IEEEPreparedState | None = None,
    router: TreeRouter | None = None,
) -> IntervalRecoveryCertificate:
    """Evaluate positivity and objective-gap recovery bounds."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if router is not None:
        router.validate_for(problem)
    started = perf_counter()
    state = _check_prepared(
        problem, y, prepared_state, recovery_only=True
    )
    if not state.recovery_certified:
        return _seal_recovery_certificate(
            IntervalRecoveryCertificate(
                float("inf"), 0.0, float("inf"), False, False,
                perf_counter() - started, 53, BACKEND_NAME
            ),
            problem,
            y,
            epsilon,
            router,
        )
    absolute_upper = np.maximum(
        np.abs(state.gradient_lower), np.abs(state.gradient_upper)
    )
    problem_plan = (
        state.reduction_plan
        if state.reduction_plan is not None
        else _problem_reduction_plan(problem)
    )
    residual_upper, valid = _planned_nonnegative_sum_upper(
        absolute_upper, problem_plan.all_nodes
    )
    if router is not None:
        tree_plan = _tree_reduction_plan(router)
        flow_lower, flow_upper, flow_valid = _tree_flow_bounds(
            router,
            state.gradient_lower,
            state.gradient_upper,
            tree_plan,
        )
        vertices = router.level_order[1:]
        edges = router.parent_edge[vertices]
        correction_lower = -flow_upper[vertices]
        correction_upper = -flow_lower[vertices]
        gap_upper, edge_positivity, gap_valid = (
            _edgewise_signed_recovery_gap_upper(
            problem.mu[edges],
            state.x_lower[edges],
            correction_lower,
            correction_upper,
            tree_plan.tree_edges,
            )
        )
        positivity = bool(
            valid
            and flow_valid
            and gap_valid
            and edge_positivity
            and np.isfinite(residual_upper)
        )
        if not positivity:
            gap_upper = float("inf")
    else:
        positivity = bool(
            valid
            and np.isfinite(residual_upper)
            and state.x_min_lower > 0.0
            and residual_upper <= state.x_min_lower
        )
        if positivity:
            gap = _positive_multiply_upper(2.0, float(np.max(problem.mu)))
            gap = _positive_multiply_upper(gap, float(max(1, problem.n - 1)))
            gap = _positive_multiply_upper(gap, residual_upper)
            gap = _positive_multiply_upper(gap, residual_upper)
            denominator = _positive_multiply_lower(
                state.x_min_lower,
                _positive_multiply_lower(state.x_min_lower, state.x_min_lower),
            )
            gap_upper = float(_positive_divide_upper(gap, denominator))
        else:
            gap_upper = float("inf")
    return _seal_recovery_certificate(
        IntervalRecoveryCertificate(
            residual_upper,
            float(state.x_min_lower),
            gap_upper,
            positivity,
            bool(positivity and np.isfinite(gap_upper) and gap_upper <= epsilon),
            perf_counter() - started,
            53,
            BACKEND_NAME,
        ),
        problem,
        y,
        epsilon,
        router,
    )
