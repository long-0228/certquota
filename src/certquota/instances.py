"""Reproducible planted-optimum sparse bipartite instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .problem import ReciprocalTransportProblem
from .tree import TreeRouter


Array = np.ndarray


@dataclass(frozen=True)
class PlantedInstance:
    problem: ReciprocalTransportProblem
    x_star: Array
    y_star: Array
    q_star: Array
    tree_edges: Optional[Array] = None
    graph_family: str = "unspecified"
    graph_seed: Optional[int] = None
    repair_edges: int = 0
    forced_edges: Optional[Array] = None


def _sample_unique_weighted_heads(
    probabilities: Array,
    degree: int,
    rng: np.random.Generator,
) -> Array:
    """Sample distinct categorical endpoints for many clients.

    ``probabilities`` has one row per client.  The routine uses vectorized
    rejection when the requested degree is sparse and a chunked Gumbel-top-k
    construction for dense small cases.  The latter avoids the pathological
    rejection time that occurs when almost every model must be selected.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    n_left, n_right = probabilities.shape
    if degree < 1 or degree > n_right:
        raise ValueError("degree must lie in [1, n_right]")
    row_sum = probabilities.sum(axis=1)
    if np.any(row_sum <= 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("every categorical row must have positive finite mass")
    probabilities = probabilities / row_sum[:, None]

    # Dense small cases are both faster and safer with exact weighted sampling
    # without replacement.  Chunking caps the largest temporary allocation.
    if degree > n_right // 4 or n_left * n_right <= 1_000_000:
        result = np.empty((n_left, degree), dtype=np.int64)
        chunk_rows = max(1, min(n_left, 1_000_000 // n_right))
        with np.errstate(divide="ignore"):
            log_weights = np.where(probabilities > 0, np.log(probabilities), -np.inf)
        for start in range(0, n_left, chunk_rows):
            stop = min(n_left, start + chunk_rows)
            uniform = np.maximum(
                rng.random((stop - start, n_right)), np.finfo(float).tiny
            )
            scores = log_weights[start:stop] - np.log(-np.log(uniform))
            selected = np.argpartition(scores, -degree, axis=1)[:, -degree:]
            # Deterministic score ordering makes output stable across BLAS builds.
            selected_score = np.take_along_axis(scores, selected, axis=1)
            order = np.argsort(-selected_score, axis=1, kind="stable")
            result[start:stop] = np.take_along_axis(selected, order, axis=1)
        return result

    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    result = np.empty((n_left, degree), dtype=np.int64)
    rows = np.arange(n_left, dtype=np.int64)
    for layer in range(degree):
        candidate = np.sum(
            rng.random(n_left)[:, None] > cumulative, axis=1, dtype=np.int64
        )
        duplicate = np.any(candidate[:, None] == result[:, :layer], axis=1)
        attempts = 0
        while np.any(duplicate):
            active = np.flatnonzero(duplicate)
            fresh = np.sum(
                rng.random(active.size)[:, None] > cumulative[active],
                axis=1,
                dtype=np.int64,
            )
            candidate[active] = fresh
            duplicate[active] = np.any(
                fresh[:, None] == result[active, :layer], axis=1
            )
            attempts += 1
            if attempts > 10_000:
                raise ArithmeticError("weighted endpoint rejection did not terminate")
        result[rows, layer] = candidate
    return result


def _sample_shared_unique_heads(
    probabilities: Array,
    n_rows: int,
    degree: int,
    rng: np.random.Generator,
) -> Array:
    """Sample distinct endpoints when all requested rows share one law."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("shared probabilities must be one-dimensional")
    n_right = probabilities.size
    if n_rows < 0 or degree < 1 or degree > n_right:
        raise ValueError("invalid shared categorical dimensions")
    if n_rows == 0:
        return np.empty((0, degree), dtype=np.int64)
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("categorical weights must be finite and nonnegative")
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("the categorical law must have positive mass")
    probabilities = probabilities / total

    if degree > n_right // 4 or n_rows * n_right <= 1_000_000:
        result = np.empty((n_rows, degree), dtype=np.int64)
        chunk_rows = max(1, min(n_rows, 1_000_000 // n_right))
        with np.errstate(divide="ignore"):
            log_weights = np.where(probabilities > 0, np.log(probabilities), -np.inf)
        for start in range(0, n_rows, chunk_rows):
            stop = min(n_rows, start + chunk_rows)
            uniform = np.maximum(
                rng.random((stop - start, n_right)), np.finfo(float).tiny
            )
            scores = log_weights[None, :] - np.log(-np.log(uniform))
            selected = np.argpartition(scores, -degree, axis=1)[:, -degree:]
            selected_score = np.take_along_axis(scores, selected, axis=1)
            order = np.argsort(-selected_score, axis=1, kind="stable")
            result[start:stop] = np.take_along_axis(selected, order, axis=1)
        return result

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    result = np.empty((n_rows, degree), dtype=np.int64)
    for layer in range(degree):
        candidate = np.searchsorted(cumulative, rng.random(n_rows), side="right")
        duplicate = np.any(candidate[:, None] == result[:, :layer], axis=1)
        attempts = 0
        while np.any(duplicate):
            active = np.flatnonzero(duplicate)
            fresh = np.searchsorted(
                cumulative, rng.random(active.size), side="right"
            )
            candidate[active] = fresh
            duplicate[active] = np.any(
                fresh[:, None] == result[active, :layer], axis=1
            )
            attempts += 1
            if attempts > 10_000:
                raise ArithmeticError("shared endpoint rejection did not terminate")
        result[:, layer] = candidate
    return result


def _install_chain_backbone(local_heads: Array, model_order: Array) -> Array:
    """Install and return a known spanning-tree backbone in an endpoint table."""

    n_left, degree = local_heads.shape
    n_right = model_order.size
    if n_left < n_right or degree < 2:
        raise ValueError("the chain backbone needs n_left >= n_right and degree >= 2")
    local_heads[:n_right, 0] = model_order
    local_heads[: n_right - 1, 1] = model_order[1:]
    # The last model also needs a distinct second endpoint, although it is not a
    # tree edge.  This assignment keeps every client row simple.
    local_heads[n_right - 1, 1] = model_order[0]
    client_tree = degree * np.arange(n_left, dtype=np.int64)
    model_tree = degree * np.arange(n_right - 1, dtype=np.int64) + 1
    return np.concatenate([client_tree, model_tree])


def _refill_backbone_rows(
    local_heads: Array,
    model_order: Array,
    row_weights: Array,
    rng: np.random.Generator,
) -> Array:
    """Install a chain and refill its rows without introducing parallel edges."""

    tree_edges = _install_chain_backbone(local_heads, model_order)
    n_right = model_order.size
    degree = local_heads.shape[1]
    row_weights = np.asarray(row_weights, dtype=float)
    if row_weights.shape != (n_right, n_right):
        raise ValueError("backbone row weights have the wrong shape")
    # The final row's second edge is not part of the chain.  Keep it inside the
    # row's declared support (important for the exactly-one-bridge family).
    final_row = n_right - 1
    final_weights = row_weights[final_row].copy()
    final_weights[local_heads[final_row, 0]] = 0.0
    if float(final_weights.sum()) <= 0:
        raise ValueError("the final backbone row has no distinct allowed endpoint")
    local_heads[final_row, 1] = rng.choice(
        n_right, p=final_weights / final_weights.sum()
    )
    for row in range(n_right):
        fixed = local_heads[row, :2]
        if degree == 2:
            continue
        weights = row_weights[row].copy()
        weights[fixed] = 0.0
        if np.count_nonzero(weights > 0) < degree - 2:
            raise ValueError("not enough allowed models to refill a backbone row")
        local_heads[row, 2:] = rng.choice(
            n_right, size=degree - 2, replace=False, p=weights / weights.sum()
        )
    return tree_edges


def _plant_on_endpoint_table(
    local_heads: Array,
    tree_edges: Array,
    seed: int,
    mu_log10_span: float,
    quota_ratio: float,
    cost_scale: float,
    max_client_budget: float,
    graph_family: str,
    repair_edges: int = 0,
) -> PlantedInstance:
    """Attach a planted KKT solution to a simple bipartite endpoint table."""

    local_heads = np.asarray(local_heads, dtype=np.int64)
    if local_heads.ndim != 2:
        raise ValueError("local_heads must be two-dimensional")
    n_left, degree = local_heads.shape
    n_right = int(local_heads.max()) + 1
    sorted_heads = np.sort(local_heads, axis=1)
    if np.any(sorted_heads[:, 1:] == sorted_heads[:, :-1]):
        raise ValueError("parallel client-model eligibility edges are not allowed")
    rng = np.random.default_rng(seed + 97_409)
    tails = np.repeat(np.arange(n_left, dtype=np.int64), degree)
    flat_heads = local_heads.reshape(-1)
    heads = (n_left + flat_heads).astype(np.int64, copy=False)

    row_budget = max_client_budget * rng.uniform(0.25, 1.0, size=n_left)
    model_rank = np.linspace(0.0, 1.0, n_right)
    model_multiplier = np.exp(np.log(quota_ratio) * model_rank)
    raw_flow = np.exp(rng.uniform(-0.5, 0.5, size=tails.size))
    raw_flow *= model_multiplier[flat_heads]
    row_sum = np.bincount(tails, weights=raw_flow, minlength=n_left)
    x_star = raw_flow * (row_budget / row_sum)[tails]

    log_mu = rng.uniform(
        -0.5 * mu_log10_span, 0.5 * mu_log10_span, size=tails.size
    )
    if tails.size >= 2:
        log_mu[:2] = (-0.5 * mu_log10_span, 0.5 * mu_log10_span)
    mu = np.power(10.0, log_mu)
    q_star = mu / x_star**2
    y_star = rng.normal(size=n_left + n_right)
    y_star -= float(y_star.mean())

    probe = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=np.zeros(n_left + n_right),
        c=np.ones(tails.size),
        mu=mu,
    )
    edge_difference = probe.edge_difference(y_star)
    if cost_scale == 0 or np.max(np.abs(edge_difference)) == 0:
        y_star.fill(0.0)
    else:
        admissible = 0.8 * float(np.min(q_star)) / float(
            np.max(np.abs(edge_difference))
        )
        y_star *= min(float(cost_scale), 1.0) * admissible
    c = q_star + probe.edge_difference(y_star)
    beta = probe.node_balance(x_star)
    problem = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=beta,
        c=c,
        mu=mu,
    )
    return PlantedInstance(
        problem=problem,
        x_star=x_star,
        y_star=y_star,
        q_star=q_star,
        tree_edges=np.asarray(tree_edges, dtype=np.int64),
        graph_family=graph_family,
        graph_seed=seed,
        repair_edges=repair_edges,
        forced_edges=np.asarray(tree_edges[n_left:], dtype=np.int64),
    )


def _connected_bipartite_edges(
    n_left: int,
    n_right: int,
    density: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    if n_left < 1 or n_right < 1:
        raise ValueError("both vertex sides must be nonempty")
    max_edges = n_left * n_right
    target = int(round(density * max_edges))
    target = min(max_edges, max(n_left + n_right - 1, target))

    edges = set()
    # A random alternating backbone makes every vertex reachable.
    for i in range(n_left):
        edges.add((i, int(rng.integers(n_right))))
    for j in range(n_right):
        edges.add((int(rng.integers(n_left)), j))

    # The two star-like sets above need not be connected.  Join components with a
    # deterministic chain of valid bipartite edges.
    for i in range(1, n_left):
        edges.add((i, i % n_right))
        edges.add((i - 1, i % n_right))

    while len(edges) < target:
        edges.add((int(rng.integers(n_left)), int(rng.integers(n_right))))

    pairs = sorted(edges)
    tails = np.asarray([u for u, _ in pairs], dtype=np.int64)
    heads = np.asarray([n_left + v for _, v in pairs], dtype=np.int64)
    return tails, heads


def make_planted_instance(
    n_left: int,
    n_right: int,
    density: float = 0.1,
    seed: int = 0,
    mu_log_range: float = 2.0,
    flow_log_range: float = 1.0,
    dual_scale: float = 0.05,
    mean_client_budget: float = 0.2,
) -> PlantedInstance:
    """Generate an MMFL-shaped instance with a known exact optimum.

    Each left marginal is an expected client activation budget below one.  The
    right/model quotas are induced by the planted positive edge probabilities.
    """

    if not 0 < mean_client_budget <= 0.8:
        raise ValueError("mean_client_budget must lie in (0, 0.8]")

    rng = np.random.default_rng(seed)
    tails, heads = _connected_bipartite_edges(n_left, n_right, density, rng)
    m = tails.size
    mu = np.exp(rng.uniform(-mu_log_range, mu_log_range, size=m))
    x_star = np.exp(rng.uniform(-flow_log_range, flow_log_range, size=m))
    row_budgets = mean_client_budget * rng.uniform(0.75, 1.25, size=n_left)
    for left in range(n_left):
        mask = tails == left
        x_star[mask] *= row_budgets[left] / float(np.sum(x_star[mask]))
    n = n_left + n_right
    probe = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=np.zeros(n),
        c=np.ones(m),
        mu=mu,
    )
    beta = probe.node_balance(x_star)
    y_star = rng.normal(scale=dual_scale, size=n)
    y_star -= float(y_star.mean())
    q_star = mu / x_star ** 2
    c = q_star + probe.edge_difference(y_star)
    problem = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=beta,
        c=c,
        mu=mu,
    )
    return PlantedInstance(problem, x_star, y_star, q_star)


def perturb_planted_instance(
    planted: PlantedInstance,
    scale: float,
    seed: int,
    change_marginals: bool = False,
) -> PlantedInstance:
    """Create a nearby planted problem on the same eligibility graph."""

    if scale < 0:
        raise ValueError("scale must be nonnegative")
    old = planted.problem
    rng = np.random.default_rng(seed)
    mu = old.mu * np.exp(scale * rng.normal(size=old.m))
    x_star = planted.x_star * np.exp(scale * rng.normal(size=old.m))
    if change_marginals:
        x_star *= float(planted.x_star.sum()) / float(x_star.sum())
    else:
        # Preserve beta by retaining the planted flow.  Only objective parameters
        # and the optimal dual move; dynamic beta experiments use a separate path.
        x_star = planted.x_star.copy()
    y_star = planted.y_star + scale * rng.normal(size=old.n)
    y_star -= float(y_star.mean())
    beta = old.node_balance(x_star)
    q_star = mu / x_star ** 2
    c = q_star + old.edge_difference(y_star)
    problem = ReciprocalTransportProblem(
        old.n_left, old.n_right, old.tails, old.heads, beta, c, mu
    )
    return PlantedInstance(
        problem=problem,
        x_star=x_star,
        y_star=y_star,
        q_star=q_star,
        tree_edges=planted.tree_edges,
        graph_family=planted.graph_family,
        graph_seed=planted.graph_seed,
        repair_edges=planted.repair_edges,
        forced_edges=planted.forced_edges,
    )


def perturb_dynamic_instance(
    planted: PlantedInstance,
    scale: float,
    seed: int,
    max_client_budget: float = 0.2,
    clip_standard_deviations: float = 3.0,
) -> PlantedInstance:
    """Advance the pre-registered planted Gaussian random walk by one round.

    ``log(x_star)``, ``log(q_star)``, and the zero-mean dual potential receive
    independent clipped Gaussian increments. Row masses are projected only when
    needed to retain the MMFL probability interpretation. The KKT recipe then
    reconstructs ``mu``, ``c``, and ``beta`` on the unchanged eligibility graph.
    """

    if scale < 0:
        raise ValueError("scale must be nonnegative")
    if not 0 < max_client_budget <= 0.8:
        raise ValueError("max_client_budget must lie in (0, 0.8]")
    if clip_standard_deviations <= 0:
        raise ValueError("clip_standard_deviations must be positive")
    old = planted.problem
    rng = np.random.default_rng(seed)
    bound = clip_standard_deviations * scale
    log_x_step = np.clip(scale * rng.normal(size=old.m), -bound, bound)
    log_q_step = np.clip(scale * rng.normal(size=old.m), -bound, bound)
    x_star = planted.x_star * np.exp(log_x_step)
    row_sum = np.bincount(old.tails, weights=x_star, minlength=old.n_left)
    row_factor = np.minimum(1.0, max_client_budget / row_sum)
    x_star *= row_factor[old.tails]
    q_star = planted.q_star * np.exp(log_q_step)
    y_star = planted.y_star + np.clip(
        scale * rng.normal(size=old.n), -bound, bound
    )
    y_star -= float(y_star.mean())

    edge_difference = old.edge_difference(y_star)
    if float(np.min(q_star + edge_difference)) <= 0.0:
        admissible = 0.8 * float(np.min(q_star)) / max(
            float(np.max(np.abs(edge_difference))), np.finfo(float).tiny
        )
        y_star *= min(1.0, admissible)
        y_star -= float(y_star.mean())
    mu = x_star**2 * q_star
    c = q_star + old.edge_difference(y_star)
    beta = old.node_balance(x_star)
    problem = ReciprocalTransportProblem(
        old.n_left, old.n_right, old.tails, old.heads, beta, c, mu
    )
    return PlantedInstance(
        problem=problem,
        x_star=x_star,
        y_star=y_star,
        q_star=q_star,
        tree_edges=planted.tree_edges,
        graph_family=planted.graph_family,
        graph_seed=planted.graph_seed,
        repair_edges=planted.repair_edges,
        forced_edges=planted.forced_edges,
    )


def make_client_regular_instance(
    n_left: int,
    n_right: int,
    degree: int = 8,
    seed: int = 0,
    mu_log10_span: float = 4.0,
    quota_ratio: float = 1.0,
    cost_scale: float = 0.0,
    max_client_budget: float = 0.2,
) -> PlantedInstance:
    """Generate a scalable connected client-regular instance and known tree.

    The construction uses only ``O(m+n)`` arrays.  Its first two edge layers
    contain a model ring, which supplies a spanning tree without running a
    general-purpose minimum-spanning-tree routine on multi-million-edge graphs.
    Remaining endpoint layers are randomized modular permutations.
    """

    if n_left < n_right:
        raise ValueError("n_left must be at least n_right for the planted backbone")
    if not 2 <= degree <= n_right:
        raise ValueError("degree must lie in [2, n_right]")
    if mu_log10_span < 0 or quota_ratio < 1 or cost_scale < 0:
        raise ValueError("invalid nonnegative factor")
    if not 0 < max_client_budget <= 0.8:
        raise ValueError("max_client_budget must lie in (0, 0.8]")

    rng = np.random.default_rng(seed)
    clients = np.arange(n_left, dtype=np.int64)
    tails = np.repeat(clients, degree)
    base = rng.integers(0, n_right, size=n_left, dtype=np.int64)
    base[:n_right] = np.arange(n_right, dtype=np.int64)

    # A stride coprime with n_right produces distinct endpoints for each client.
    candidates = np.asarray(
        [value for value in range(1, n_right) if np.gcd(value, n_right) == 1],
        dtype=np.int64,
    )
    strides = rng.choice(candidates, size=n_left, replace=True)
    strides[:n_right] = 1
    layers = np.arange(degree, dtype=np.int64)
    local_heads = (base[:, None] + strides[:, None] * layers[None, :]) % n_right
    model_permutation = rng.permutation(n_right)
    local_heads = model_permutation[local_heads]
    heads = (n_left + local_heads.reshape(-1)).astype(np.int64, copy=False)

    # One incident edge for every client plus n_right-1 ring links is a tree.
    client_tree = degree * clients
    model_tree = degree * np.arange(n_right - 1, dtype=np.int64) + 1
    tree_edges = np.concatenate([client_tree, model_tree])

    row_budget = max_client_budget * rng.uniform(0.25, 1.0, size=n_left)
    model_rank = np.linspace(0.0, 1.0, n_right)
    model_multiplier = np.exp(np.log(quota_ratio) * model_rank)
    raw_flow = np.exp(rng.uniform(-0.5, 0.5, size=tails.size))
    raw_flow *= model_multiplier[local_heads.reshape(-1)]
    row_sum = np.bincount(tails, weights=raw_flow, minlength=n_left)
    x_star = raw_flow * (row_budget / row_sum)[tails]

    log_mu = rng.uniform(
        -0.5 * mu_log10_span, 0.5 * mu_log10_span, size=tails.size
    )
    mu = np.power(10.0, log_mu)
    q_star = mu / x_star**2
    y_star = rng.normal(size=n_left + n_right)
    y_star -= float(y_star.mean())

    probe = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=np.zeros(n_left + n_right),
        c=np.ones(tails.size),
        mu=mu,
    )
    edge_difference = probe.edge_difference(y_star)
    if cost_scale == 0 or np.max(np.abs(edge_difference)) == 0:
        y_star.fill(0.0)
    else:
        admissible = 0.8 * float(np.min(q_star)) / float(
            np.max(np.abs(edge_difference))
        )
        y_star *= min(float(cost_scale), 1.0) * admissible
    c = q_star + probe.edge_difference(y_star)
    beta = probe.node_balance(x_star)
    problem = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=beta,
        c=c,
        mu=mu,
    )
    return PlantedInstance(
        problem=problem,
        x_star=x_star,
        y_star=y_star,
        q_star=q_star,
        tree_edges=tree_edges,
        graph_family="client_regular",
        graph_seed=seed,
        repair_edges=n_right - 1,
        forced_edges=model_tree,
    )


def make_graph_family_instance(
    graph_family: str,
    n_left: int,
    n_right: int,
    degree: int = 8,
    seed: int = 0,
    mu_log10_span: float = 4.0,
    quota_ratio: float = 1.0,
    cost_scale: float = 0.0,
    max_client_budget: float = 0.2,
    zipf_exponent: float = 1.2,
    community_intra_probability: float = 0.9,
) -> PlantedInstance:
    """Generate one of the four pre-registered sparse graph families.

    All rows are simple (no parallel client--model pairs), a deterministic chain
    backbone makes connectivity auditable, and the returned instance includes a
    known spanning tree and exact planted optimum.  In the bottleneck family the
    chain contains exactly one cross-half edge.
    """

    families = {"client_regular", "zipf_degree", "community", "bottleneck"}
    if graph_family not in families:
        raise ValueError(f"unknown graph family {graph_family!r}")
    if n_left < n_right:
        raise ValueError("n_left must be at least n_right for the planted backbone")
    if not 2 <= degree <= n_right:
        raise ValueError("degree must lie in [2, n_right]")
    if mu_log10_span < 0 or quota_ratio < 1 or cost_scale < 0:
        raise ValueError("invalid nonnegative factor")
    if not 0 < max_client_budget <= 0.8:
        raise ValueError("max_client_budget must lie in (0, 0.8]")
    if zipf_exponent <= 0:
        raise ValueError("zipf_exponent must be positive")
    if not 0 < community_intra_probability < 1:
        raise ValueError("community_intra_probability must lie in (0, 1)")
    if graph_family == "client_regular":
        return make_client_regular_instance(
            n_left=n_left,
            n_right=n_right,
            degree=degree,
            seed=seed,
            mu_log10_span=mu_log10_span,
            quota_ratio=quota_ratio,
            cost_scale=cost_scale,
            max_client_budget=max_client_budget,
        )

    rng = np.random.default_rng(seed)
    local_heads = np.empty((n_left, degree), dtype=np.int64)

    if graph_family == "zipf_degree":
        model_order = rng.permutation(n_right)
        rank_weights = np.arange(1, n_right + 1, dtype=float) ** (-zipf_exponent)
        weights = np.empty(n_right, dtype=float)
        weights[model_order] = rank_weights
        local_heads[:] = _sample_shared_unique_heads(weights, n_left, degree, rng)
        backbone_weights = np.broadcast_to(weights, (n_right, n_right)).copy()
        tree_edges = _refill_backbone_rows(
            local_heads, model_order, backbone_weights, rng
        )
        repair_edges = n_right - 1

    elif graph_family == "community":
        if n_right < 4:
            raise ValueError("the four-community family needs at least four models")
        block_count = 4
        shuffled_models = rng.permutation(n_right)
        model_groups = np.array_split(shuffled_models, block_count)
        model_block = np.empty(n_right, dtype=np.int64)
        for block, models in enumerate(model_groups):
            model_block[models] = block
        model_order = np.concatenate(model_groups)
        client_block = rng.integers(0, block_count, size=n_left, dtype=np.int64)
        client_block[:n_right] = model_block[model_order]
        outside_probability = (1.0 - community_intra_probability) / (block_count - 1)
        laws = np.empty((block_count, n_right), dtype=float)
        for block in range(block_count):
            sizes = np.bincount(model_block, minlength=block_count)
            laws[block] = outside_probability / sizes[model_block]
            laws[block, model_block == block] = (
                community_intra_probability / sizes[block]
            )
            rows = np.flatnonzero(client_block == block)
            local_heads[rows] = _sample_shared_unique_heads(
                laws[block], rows.size, degree, rng
            )
        backbone_weights = laws[client_block[:n_right]]
        tree_edges = _refill_backbone_rows(
            local_heads, model_order, backbone_weights, rng
        )
        repair_edges = block_count - 1

    else:
        left_models = rng.permutation(np.arange(0, n_right // 2, dtype=np.int64))
        right_models = rng.permutation(
            np.arange(n_right // 2, n_right, dtype=np.int64)
        )
        if degree > min(left_models.size, right_models.size):
            raise ValueError(
                "bottleneck degree cannot exceed the smaller model half"
            )
        model_order = np.concatenate([left_models, right_models])
        model_side = np.zeros(n_right, dtype=np.int64)
        model_side[right_models] = 1
        client_side = rng.integers(0, 2, size=n_left, dtype=np.int64)
        client_side[:n_right] = model_side[model_order]
        laws = np.zeros((2, n_right), dtype=float)
        laws[0, left_models] = 1.0
        laws[1, right_models] = 1.0
        for side in (0, 1):
            rows = np.flatnonzero(client_side == side)
            local_heads[rows] = _sample_shared_unique_heads(
                laws[side], rows.size, degree, rng
            )
        backbone_weights = laws[client_side[:n_right]]
        tree_edges = _refill_backbone_rows(
            local_heads, model_order, backbone_weights, rng
        )
        repair_edges = 1

    return _plant_on_endpoint_table(
        local_heads=local_heads,
        tree_edges=tree_edges,
        seed=seed,
        mu_log10_span=mu_log10_span,
        quota_ratio=quota_ratio,
        cost_scale=cost_scale,
        max_client_budget=max_client_budget,
        graph_family=graph_family,
        repair_edges=repair_edges,
    )


def perturb_feasible_flow(
    planted: PlantedInstance,
    seed: int,
    boundary_fraction: float = 0.5,
    cycle_count: int = 20,
) -> Array:
    """Return a positive nonoptimal flow with exactly the planted marginals."""

    if not 0 < boundary_fraction < 1:
        raise ValueError("boundary_fraction must lie in (0, 1)")
    problem = planted.problem
    router = TreeRouter.build(problem)
    tree_set = set(int(edge) for edge in router.tree_edges)
    non_tree = [edge for edge in range(problem.m) if edge not in tree_set]
    if not non_tree:
        return planted.x_star.copy()
    rng = np.random.default_rng(seed)
    circulation = np.zeros(problem.m, dtype=float)
    selected = rng.choice(
        non_tree, size=min(cycle_count, len(non_tree)), replace=False
    )
    b = problem.incidence
    for edge in selected:
        coefficient = float(rng.normal())
        unit = np.zeros(problem.m, dtype=float)
        unit[int(edge)] = 1.0
        demand = np.asarray(b @ unit).reshape(-1)
        cycle = unit + router.route(-demand)
        circulation += coefficient * cycle
    if np.max(np.abs(circulation)) == 0:
        return planted.x_star.copy()
    negative = circulation < 0
    positive = circulation > 0
    limits = []
    if np.any(negative):
        limits.append(float(np.min(planted.x_star[negative] / -circulation[negative])))
    if np.any(positive):
        limits.append(float(np.min(planted.x_star[positive] / circulation[positive])))
    step = boundary_fraction * min(limits)
    if rng.random() < 0.5:
        step = -step
        # Recompute the valid distance in the opposite direction.
        active = circulation > 0
        step = -boundary_fraction * float(
            np.min(planted.x_star[active] / circulation[active])
        )
    x = planted.x_star + step * circulation
    if np.any(x <= 0):
        raise ArithmeticError("cycle perturbation left the positive orthant")
    return x
