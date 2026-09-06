from __future__ import annotations

from math import fsum

import numpy as np

from certquota.certificates import dual_state
from certquota.instances import (
    make_client_regular_instance,
    make_graph_family_instance,
    make_planted_instance,
    perturb_dynamic_instance,
)
from certquota.problem import ReciprocalTransportProblem
from certquota.tree import TreeRouter


def test_tree_router_satisfies_balanced_demands() -> None:
    planted = make_planted_instance(9, 7, density=0.35, seed=3)
    problem = planted.problem
    router = TreeRouter.build(problem)
    rng = np.random.default_rng(4)
    demand = rng.normal(size=problem.n)
    demand -= demand.mean()

    flow = router.route(demand)

    np.testing.assert_allclose(problem.incidence @ flow, demand, atol=2e-13)
    assert np.count_nonzero(flow) <= problem.n - 1


def test_tree_laplacian_solver_inverts_tree_system() -> None:
    planted = make_planted_instance(8, 6, density=0.4, seed=5)
    problem = planted.problem
    router = TreeRouter.build(problem)
    rng = np.random.default_rng(6)
    conductance = np.exp(rng.normal(size=problem.m))
    demand = rng.normal(size=problem.n)
    demand -= demand.mean()

    potential = router.solve_laplacian(demand, conductance)
    tree_conductance = np.zeros(problem.m)
    tree_conductance[router.tree_edges] = conductance[router.tree_edges]
    reconstructed = (
        problem.incidence
        @ (tree_conductance * (problem.incidence.T @ potential))
    )

    np.testing.assert_allclose(reconstructed, demand, atol=2e-12)
    assert abs(float(potential.mean())) < 1e-13


def test_planted_solution_satisfies_kkt_conditions() -> None:
    planted = make_planted_instance(8, 6, density=0.4, seed=7)
    problem = planted.problem

    np.testing.assert_allclose(
        problem.primal_residual(planted.x_star), 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        problem.dual_slack(planted.y_star), planted.q_star, rtol=2e-13
    )
    np.testing.assert_allclose(
        problem.c - problem.mu / planted.x_star**2,
        problem.incidence.T @ planted.y_star,
        rtol=2e-13,
        atol=2e-10,
    )


def test_scalable_client_regular_instance_has_known_tree_and_kkt() -> None:
    planted = make_client_regular_instance(
        64, 8, degree=4, seed=19, mu_log10_span=4, quota_ratio=10
    )
    problem = planted.problem
    assert planted.tree_edges is not None
    assert planted.tree_edges.size == problem.n - 1
    router = TreeRouter.build(problem, tree_edges=planted.tree_edges)
    state = dual_state(problem, planted.y_star)
    assert np.linalg.norm(state.g, np.inf) < 1e-10
    assert np.max(problem.beta[: problem.n_left]) <= 0.2 * (1 + 1e-12)
    routed = router.route(problem.beta)
    np.testing.assert_allclose(
        problem.node_balance(routed), problem.beta, atol=1e-10
    )


def test_all_registered_graph_families_are_simple_connected_and_planted() -> None:
    for offset, family in enumerate(
        ("client_regular", "zipf_degree", "community", "bottleneck")
    ):
        planted = make_graph_family_instance(
            family, 256, 16, degree=4, seed=100 + offset
        )
        problem = planted.problem
        endpoints = np.column_stack(
            [problem.tails, problem.heads - problem.n_left]
        )
        assert np.unique(endpoints, axis=0).shape[0] == problem.m
        assert planted.tree_edges is not None
        assert planted.tree_edges.size == problem.n - 1
        router = TreeRouter.build(problem, tree_edges=planted.tree_edges)
        state = dual_state(problem, planted.y_star)
        assert np.linalg.norm(state.g, np.inf) < 2e-10
        routed = router.route(problem.beta)
        np.testing.assert_allclose(
            problem.node_balance(routed), problem.beta, atol=2e-10
        )


def test_bottleneck_has_exactly_one_cross_half_edge() -> None:
    planted = make_graph_family_instance(
        "bottleneck", 512, 20, degree=8, seed=211
    )
    problem = planted.problem
    model = problem.heads - problem.n_left
    # Infer each client's planted half from its first (backbone) endpoint.
    client_side = model.reshape(problem.n_left, 8)[:, 0] >= 10
    edge_side = model >= 10
    cross = edge_side != np.repeat(client_side, 8)
    assert int(np.count_nonzero(cross)) == 1


def test_structurally_impossible_bottleneck_degree_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "smaller model half"):
        make_graph_family_instance(
            "bottleneck", 64, 4, degree=4, seed=212
        )


def test_dynamic_random_walk_preserves_kkt_domain_and_probability_budgets() -> None:
    current = make_graph_family_instance(
        "community", 256, 16, degree=4, seed=220
    )
    for round_index in range(1, 6):
        current = perturb_dynamic_instance(
            current,
            scale=0.1,
            seed=220_000 + round_index,
            max_client_budget=0.2,
        )
        problem = current.problem
        state = dual_state(problem, current.y_star)
        assert np.linalg.norm(state.g, np.inf) < 2e-10
        assert np.min(problem.c) >= 0.0
        assert np.max(problem.beta[: problem.n_left]) <= 0.2 * (1 + 1e-12)
        np.testing.assert_allclose(state.q, current.q_star, rtol=3e-13)


def test_root_quota_is_conservation_implied_not_tolerance_balanced() -> None:
    planted = make_planted_instance(8, 6, density=0.4, seed=229)
    base = planted.problem
    declared_beta = base.beta.copy()
    declared_beta[-1] = np.nextafter(declared_beta[-1], np.inf)
    problem = ReciprocalTransportProblem(
        base.n_left,
        base.n_right,
        base.tails,
        base.heads,
        declared_beta,
        base.c,
        base.mu,
    )

    assert problem.declared_root_beta == declared_beta[-1]
    assert problem.root_beta_adjustment != 0.0
    assert problem.beta[-1] == -fsum(float(value) for value in problem.beta[:-1])

    raw = np.linspace(-1.0, 2.0, problem.n)
    balanced = problem.impose_reduced_balance(raw)
    assert balanced[-1] == -fsum(float(value) for value in balanced[:-1])
    shifted = np.linspace(-0.5, 0.75, problem.n)
    pairing = problem.balanced_pairing(balanced, shifted)
    shifted_pairing = problem.balanced_pairing(balanced, shifted + 123.0)
    assert abs(pairing - shifted_pairing) < 5e-13

    residual = problem.primal_residual(planted.x_star)
    assert residual[-1] == -fsum(float(value) for value in residual[:-1])


def test_reduced_marginal_constructor_has_no_independent_root_input() -> None:
    planted = make_planted_instance(8, 6, density=0.4, seed=230)
    base = planted.problem
    problem = ReciprocalTransportProblem.from_reduced_marginals(
        base.n_left,
        base.n_right,
        base.tails,
        base.heads,
        base.beta[:-1],
        base.c,
        base.mu,
    )

    np.testing.assert_array_equal(problem.beta[:-1], base.beta[:-1])
    assert problem.beta[-1] == -fsum(float(value) for value in base.beta[:-1])
    assert problem.declared_root_beta == problem.beta[-1]
    assert problem.root_beta_adjustment == 0.0


def test_reduced_marginal_constructor_rejects_wrong_shape() -> None:
    planted = make_planted_instance(6, 4, density=0.5, seed=231)
    base = planted.problem
    with np.testing.assert_raises_regex(ValueError, "exactly n-1"):
        ReciprocalTransportProblem.from_reduced_marginals(
            base.n_left,
            base.n_right,
            base.tails,
            base.heads,
            base.beta[:-2],
            base.c,
            base.mu,
        )
