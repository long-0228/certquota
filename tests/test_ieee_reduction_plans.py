from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import certquota.ieee_intervals as ieee_module
from certquota.certificates import dual_state
from certquota.ieee_intervals import (
    _PROBLEM_REDUCTION_CACHE,
    _add_lower,
    _add_upper,
    _compile_segmented_pairwise_plan,
    _compile_tree_reduction_plan,
    _execute_segmented_pairwise_plan,
    _grouped_interval_sum_bounds,
    _grouped_nonnegative_sum_bounds,
    _planned_interval_sum_bounds,
    _problem_reduction_plan,
    _segmented_pairwise_sum_bounds,
    _tree_flow_abs_upper,
    _tree_reduction_plan,
    ieee_recovery_certificate,
    prepare_ieee_recovery_state,
    prepare_ieee_state,
)
from certquota.instances import (
    make_graph_family_instance,
    make_planted_instance,
    perturb_dynamic_instance,
)
from certquota.problem import ReciprocalTransportProblem
from certquota.strict_backend import prepare_strict_recovery_state
from certquota.tree import TreeRouter


def _case(seed: int = 8101):
    planted = make_planted_instance(24, 10, density=0.35, seed=seed)
    problem = planted.problem
    state = dual_state(problem, planted.y_star)
    router = TreeRouter.build(problem, edge_cost=1.0 / state.conductance)
    return problem, planted.y_star.copy(), router


def _reference_segmented_pairwise_sum_bounds(
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    groups: np.ndarray,
    size: int,
):
    """Direct level-by-level reference, intentionally independent of plans."""

    order = np.argsort(groups, kind="stable")
    current_groups = np.asarray(groups, dtype=np.int64)[order]
    current_lower = np.asarray(lower_values, dtype=float)[order].copy()
    current_upper = np.asarray(upper_values, dtype=float)[order].copy()
    while current_groups.size > 1:
        starts = np.empty(current_groups.size, dtype=bool)
        starts[0] = True
        starts[1:] = current_groups[1:] != current_groups[:-1]
        if np.all(starts):
            break
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
        next_lower = current_lower[selected].copy()
        next_upper = current_upper[selected].copy()
        next_lower[output_positions] = _add_lower(
            current_lower[selected[paired]], current_lower[partners[paired]]
        )
        next_upper[output_positions] = _add_upper(
            current_upper[selected[paired]], current_upper[partners[paired]]
        )
        current_groups = current_groups[selected]
        current_lower = next_lower
        current_upper = next_upper

    lower = np.zeros(size, dtype=float)
    upper = np.zeros(size, dtype=float)
    lower[current_groups] = current_lower
    upper[current_groups] = current_upper
    return lower, upper, True


def _uncached_tree_flow(router, demand_lower, demand_upper):
    """Reference implementation that rebuilds compact groups at every level."""

    subtree_lower = np.asarray(demand_lower, dtype=float).copy()
    subtree_upper = np.asarray(demand_upper, dtype=float).copy()
    flow_abs = np.zeros(router.problem.n, dtype=float)
    valid = True
    for level in range(router.level_offsets.size - 2, 0, -1):
        start = int(router.level_offsets[level])
        end = int(router.level_offsets[level + 1])
        vertices = router.level_order[start:end]
        signs = router.child_incidence_sign[vertices]
        flow_lower = np.where(
            signs > 0.0, subtree_lower[vertices], -subtree_upper[vertices]
        )
        flow_upper = np.where(
            signs > 0.0, subtree_upper[vertices], -subtree_lower[vertices]
        )
        flow_abs[vertices] = np.maximum(np.abs(flow_lower), np.abs(flow_upper))
        parents = router.parent[vertices]
        active_parents, compact_groups = np.unique(
            parents, return_inverse=True
        )
        increment_lower, increment_upper, ok = _grouped_interval_sum_bounds(
            subtree_lower[vertices],
            subtree_upper[vertices],
            compact_groups,
            active_parents.size,
        )
        subtree_lower[active_parents] = _add_lower(
            subtree_lower[active_parents], increment_lower
        )
        subtree_upper[active_parents] = _add_upper(
            subtree_upper[active_parents], increment_upper
        )
        valid = bool(valid and ok)
    return flow_abs, bool(valid and np.all(np.isfinite(flow_abs)))


def _recovery_payload(certificate):
    return (
        certificate.residual_l1_upper,
        certificate.x_min_lower,
        certificate.objective_gap_upper,
        certificate.positivity_certified,
        certificate.epsilon_optimal_certified,
        certificate.decimal_precision,
        certificate.backend,
    )


def test_compiled_segmented_dag_matches_uncached_endpoints_bit_for_bit() -> None:
    rng = np.random.default_rng(8102)
    groups = rng.integers(0, 37, size=5_003, dtype=np.int64)
    centers = rng.normal(size=groups.size)
    radii = np.exp(rng.uniform(-30.0, -8.0, size=groups.size))
    lower_values = centers - radii
    upper_values = centers + radii

    uncached = _reference_segmented_pairwise_sum_bounds(
        lower_values, upper_values, groups, 37
    )
    plan = _compile_segmented_pairwise_plan(groups, 37)
    compiled = _execute_segmented_pairwise_plan(
        lower_values, upper_values, plan
    )

    assert compiled[2] is uncached[2]
    assert np.array_equal(compiled[0], uncached[0])
    assert np.array_equal(compiled[1], uncached[1])
    assert plan.order is not None
    assert not plan.order.flags.writeable
    assert all(not level.output.flags.writeable for level in plan.levels)
    assert all(not level.left.flags.writeable for level in plan.levels)
    assert plan.order.dtype == np.int32

    identity_plan = _compile_segmented_pairwise_plan(
        np.zeros(128, dtype=np.int64), 1
    )
    assert identity_plan.order is None


def test_segmented_plan_has_linear_size_on_skewed_groups() -> None:
    """A giant group plus many singletons must not be copied per level."""

    giant = 16_384
    groups = np.concatenate(
        [
            np.zeros(giant, dtype=np.int64),
            np.arange(1, giant + 1, dtype=np.int64),
        ]
    )
    plan = _compile_segmented_pairwise_plan(groups, giant + 1)
    pair_count = sum(level.output.size for level in plan.levels)

    assert plan.order is None
    assert pair_count == groups.size - (giant + 1)
    assert plan.workspace_size == groups.size + pair_count
    assert plan.output_groups.size == giant + 1
    assert plan.output_nodes.size == giant + 1

    values = np.linspace(0.5, 1.5, groups.size)
    expected = _reference_segmented_pairwise_sum_bounds(
        values, values, groups, giant + 1
    )
    actual = _execute_segmented_pairwise_plan(values, values, plan)
    assert actual[2] is expected[2]
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_empty_segmented_plan_preserves_zero_output_contract() -> None:
    plan = _compile_segmented_pairwise_plan(np.empty(0, dtype=np.int64), 4)
    lower, upper, valid = _execute_segmented_pairwise_plan(
        np.empty(0), np.empty(0), plan
    )
    assert valid
    assert plan.workspace_size == 0
    assert np.array_equal(lower, np.zeros(4))
    assert np.array_equal(upper, np.zeros(4))


def test_problem_endpoint_plans_match_separate_uncached_reductions() -> None:
    problem, y, _ = _case(8103)
    state = prepare_ieee_state(problem, y)
    plan = _problem_reduction_plan(problem)
    assert plan is _problem_reduction_plan(problem)

    row_lower, _, ok_lower = _grouped_nonnegative_sum_bounds(
        state.x_lower, problem.tails, problem.n_left
    )
    _, row_upper, ok_upper = _grouped_nonnegative_sum_bounds(
        state.x_upper, problem.tails, problem.n_left
    )
    planned_lower, planned_upper, ok_planned = _planned_interval_sum_bounds(
        state.x_lower, state.x_upper, plan.clients
    )
    assert ok_lower and ok_upper and ok_planned
    assert np.array_equal(planned_lower, row_lower)
    assert np.array_equal(planned_upper, row_upper)

    local_heads = problem.heads - problem.n_left
    column_lower, _, ok_lower = _grouped_nonnegative_sum_bounds(
        state.x_lower, local_heads, problem.n_right
    )
    _, column_upper, ok_upper = _grouped_nonnegative_sum_bounds(
        state.x_upper, local_heads, problem.n_right
    )
    planned_lower, planned_upper, ok_planned = _planned_interval_sum_bounds(
        state.x_lower, state.x_upper, plan.models
    )
    assert ok_lower and ok_upper and ok_planned
    assert np.array_equal(planned_lower, column_lower)
    assert np.array_equal(planned_upper, column_upper)


def test_cached_tree_plan_matches_uncached_parent_grouping_bit_for_bit() -> None:
    problem, _, router = _case(8104)
    rng = np.random.default_rng(8104)
    lower = rng.normal(size=problem.n)
    upper = lower + np.exp(rng.uniform(-28.0, -10.0, size=problem.n))

    uncached = _uncached_tree_flow(router, lower, upper)
    cached_plan = _tree_reduction_plan(router)
    cached = _tree_flow_abs_upper(router, lower, upper, cached_plan)
    fresh = _tree_flow_abs_upper(
        router, lower, upper, _compile_tree_reduction_plan(router)
    )

    assert cached_plan is _tree_reduction_plan(router)
    assert cached[1] is uncached[1] is fresh[1]
    assert np.array_equal(cached[0], uncached[0])
    assert np.array_equal(cached[0], fresh[0])


def test_lazy_recovery_state_matches_full_state_and_certificate_bit_for_bit() -> None:
    problem, y, router = _case(8105)
    full = prepare_ieee_state(problem, y)
    lazy = prepare_ieee_recovery_state(problem, y)

    assert full.domain_certified and full.recovery_certified
    assert not lazy.domain_certified and lazy.recovery_certified
    assert lazy.x_upper is None
    assert lazy.conductance_lower is None
    assert lazy.conductance_upper is None
    assert lazy.sigma_lower is None
    assert lazy.sigma_upper is None
    assert np.array_equal(lazy.x_lower, full.x_lower)
    assert np.array_equal(lazy.gradient_lower, full.gradient_lower)
    assert np.array_equal(lazy.gradient_upper, full.gradient_upper)
    assert lazy.x_min_lower == full.x_min_lower

    full_certificate = ieee_recovery_certificate(
        problem, y, 1.0, full, router
    )
    lazy_certificate = ieee_recovery_certificate(
        problem, y, 1.0, lazy, router
    )
    implicit_lazy = ieee_recovery_certificate(problem, y, 1.0, router=router)
    assert _recovery_payload(lazy_certificate) == _recovery_payload(full_certificate)
    assert _recovery_payload(implicit_lazy) == _recovery_payload(full_certificate)

    dispatched = prepare_strict_recovery_state(
        problem, y, "ieee754", decimal_precision=70
    )
    assert dispatched.recovery_certified
    assert not dispatched.domain_certified


def test_problem_and_router_topology_snapshots_are_owned_and_read_only() -> None:
    tails = np.asarray([0, 0, 1, 1], dtype=np.int64)
    heads = np.asarray([2, 3, 2, 3], dtype=np.int64)
    beta = np.asarray([1.0, 1.0, -1.0, -1.0])
    costs = np.asarray([4.0, 5.0, 6.0, 7.0])
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    problem = ReciprocalTransportProblem(
        2, 2, tails, heads, beta, costs, weights
    )
    snapshots = tuple(
        values.copy()
        for values in (
            problem.tails,
            problem.heads,
            problem.beta,
            problem.c,
            problem.mu,
        )
    )

    tails[:] = 1
    heads[:] = 3
    beta[:] = 0.0
    costs[:] = -1.0
    weights[:] = -1.0
    for values, expected in zip(
        (
            problem.tails,
            problem.heads,
            problem.beta,
            problem.c,
            problem.mu,
        ),
        snapshots,
    ):
        assert np.array_equal(values, expected)
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values[0] = values[0]

    initial_router = TreeRouter.build(problem)
    supplied_tree_edges = initial_router.tree_edges.copy()
    router = TreeRouter.build(problem, tree_edges=supplied_tree_edges)
    expected_tree_edges = router.tree_edges.copy()
    supplied_tree_edges[:] = supplied_tree_edges[::-1]
    assert np.array_equal(router.tree_edges, expected_tree_edges)
    for values in (
        router.tree_edges,
        router.parent,
        router.parent_edge,
        router.child_incidence_sign,
        router.postorder,
        router.level_order,
        router.level_offsets,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values[0] = values[0]

    demand = np.asarray([0.25, -0.125, 0.5, -0.625])
    cached = _tree_flow_abs_upper(router, demand, demand)
    fresh = _tree_flow_abs_upper(
        router, demand, demand, _compile_tree_reduction_plan(router)
    )
    assert cached[1] is fresh[1]
    assert np.array_equal(cached[0], fresh[0])


def test_dynamic_problems_reuse_content_validated_topology_owner_plan() -> None:
    base = make_graph_family_instance(
        "client_regular", 64, 8, degree=4, seed=8_201
    )
    current = perturb_dynamic_instance(
        base, scale=1e-3, seed=8_202, max_client_budget=0.2
    )
    assert current.problem is not base.problem
    assert np.array_equal(current.problem.tails, base.problem.tails)
    assert np.array_equal(current.problem.heads, base.problem.heads)
    assert not np.array_equal(current.problem.c, base.problem.c)

    base_state = dual_state(base.problem, base.y_star)
    router = TreeRouter.build(
        base.problem, edge_cost=1.0 / base_state.conductance
    )
    epsilon = 1e-9 * (
        1.0 + abs(current.problem.objective(current.x_star))
    )
    own_prepared = prepare_ieee_recovery_state(
        current.problem, current.y_star
    )
    own_certificate = ieee_recovery_certificate(
        current.problem,
        current.y_star,
        epsilon,
        own_prepared,
        router,
    )

    _PROBLEM_REDUCTION_CACHE.clear()
    with patch.object(
        ieee_module,
        "_compile_problem_reduction_plan",
        wraps=ieee_module._compile_problem_reduction_plan,
    ) as compile_plan:
        base_prepared = prepare_ieee_recovery_state(
            base.problem, base.y_star, base.problem
        )
        ieee_recovery_certificate(
            base.problem,
            base.y_star,
            epsilon,
            base_prepared,
            router,
        )
        reused_prepared = prepare_ieee_recovery_state(
            current.problem, current.y_star, base.problem
        )
        reused_certificate = ieee_recovery_certificate(
            current.problem,
            current.y_star,
            epsilon,
            reused_prepared,
            router,
        )

    assert compile_plan.call_count == 1
    assert reused_prepared.topology_owner is base.problem
    assert reused_prepared.reduction_plan is base_prepared.reduction_plan
    assert _recovery_payload(reused_certificate) == _recovery_payload(
        own_certificate
    )


def test_topology_owner_content_mismatch_fails_closed() -> None:
    current = make_graph_family_instance(
        "client_regular", 64, 8, degree=4, seed=8_211
    )
    wrong_owner = make_graph_family_instance(
        "client_regular", 64, 8, degree=4, seed=8_212
    )
    assert current.problem.n == wrong_owner.problem.n
    assert current.problem.m == wrong_owner.problem.m
    assert not (
        np.array_equal(current.problem.tails, wrong_owner.problem.tails)
        and np.array_equal(current.problem.heads, wrong_owner.problem.heads)
    )
    with pytest.raises(ValueError, match="topology owner is incompatible"):
        prepare_ieee_recovery_state(
            current.problem, current.y_star, wrong_owner.problem
        )
