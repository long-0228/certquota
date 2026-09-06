"""Expected-quota categorical sampling and MMFL estimator utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import ReciprocalTransportProblem


Array = np.ndarray


def marginal_residuals(
    problem: ReciprocalTransportProblem, probabilities: Array
) -> tuple[Array, Array]:
    """Return client-row and model-column residuals for an edge allocation."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (problem.m,):
        raise ValueError("probabilities have the wrong dimension")
    residual = problem.primal_residual(probabilities)
    return residual[: problem.n_left], -residual[problem.n_left :]


def validate_expected_quota_allocation(
    problem: ReciprocalTransportProblem,
    probabilities: Array,
    tolerance: float = 1e-9,
) -> None:
    """Validate positivity, categorical row masses, and both expected marginals."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (problem.m,):
        raise ValueError("probabilities have the wrong dimension")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("every eligible edge probability must be finite and positive")
    row_residual, column_residual = marginal_residuals(problem, probabilities)
    scale = 1.0 + float(np.linalg.norm(problem.beta, 1))
    if (
        float(np.linalg.norm(row_residual, 1) + np.linalg.norm(column_residual, 1))
        > tolerance * scale
    ):
        raise ValueError("allocation does not satisfy the requested expected quotas")
    actual_row_mass = np.bincount(
        problem.tails, weights=probabilities, minlength=problem.n_left
    )
    if np.any(actual_row_mass > 1.0):
        raise ValueError(
            "materialized categorical row probability exceeds one"
        )
    row_mass = problem.beta[: problem.n_left]
    if np.any(row_mass > 1.0 + tolerance):
        raise ValueError("categorical client sampling requires every row mass <= 1")


def maximum_entropy_allocation(
    problem: ReciprocalTransportProblem,
    base_measure: Array | None = None,
    tolerance: float = 1e-12,
    max_sweeps: int = 100_000,
) -> tuple[Array, int]:
    """Compute the positive sparse maximum-entropy allocation by IPFP.

    With a unit base measure this is the most diffuse feasible distribution on
    the declared eligibility graph.  The routine is a sampling-policy baseline,
    not an optimization baseline for the reciprocal objective.
    """

    if tolerance <= 0.0 or max_sweeps <= 0:
        raise ValueError("tolerance and max_sweeps must be positive")
    if base_measure is None:
        allocation = np.ones(problem.m, dtype=float)
    else:
        allocation = np.asarray(base_measure, dtype=float).copy()
        if allocation.shape != (problem.m,):
            raise ValueError("base_measure has the wrong dimension")
        if np.any(allocation <= 0.0) or not np.all(np.isfinite(allocation)):
            raise ValueError("base_measure must be finite and positive")

    row_target = problem.beta[: problem.n_left]
    column_target = -problem.beta[problem.n_left :]
    local_heads = problem.heads - problem.n_left
    target_scale = 1.0 + float(np.sum(row_target) + np.sum(column_target))

    for sweep in range(1, max_sweeps + 1):
        row_sum = np.bincount(
            problem.tails, weights=allocation, minlength=problem.n_left
        )
        if np.any(row_sum <= 0.0) or not np.all(np.isfinite(row_sum)):
            raise ArithmeticError("IPFP lost a positive client marginal")
        allocation *= (row_target / row_sum)[problem.tails]

        column_sum = np.bincount(
            local_heads, weights=allocation, minlength=problem.n_right
        )
        if np.any(column_sum <= 0.0) or not np.all(np.isfinite(column_sum)):
            raise ArithmeticError("IPFP lost a positive model marginal")
        allocation *= (column_target / column_sum)[local_heads]

        if sweep == 1 or sweep % 10 == 0:
            row_residual, column_residual = marginal_residuals(problem, allocation)
            residual = float(
                np.linalg.norm(row_residual, 1)
                + np.linalg.norm(column_residual, 1)
            )
            if residual <= tolerance * target_scale:
                validate_expected_quota_allocation(
                    problem, allocation, tolerance=max(1e-10, 10.0 * tolerance)
                )
                return allocation, sweep

    raise RuntimeError("maximum-entropy IPFP did not converge within max_sweeps")


@dataclass(frozen=True)
class IndependentCategoricalSampler:
    """Precomputed independent per-client categorical laws.

    Each client draws at most one eligible edge.  The missing materialized row
    mass is the probability of remaining idle.  Target-quota residuals are
    validated but never hidden by renormalizing to the target row mass.  Draws
    are independent across clients, exactly as in the paper's variance theorem.
    """

    problem: ReciprocalTransportProblem
    probabilities: Array
    edge_order: Array
    offsets: Array
    conditional_cdf: Array
    row_mass: Array

    @classmethod
    def build(
        cls,
        problem: ReciprocalTransportProblem,
        probabilities: Array,
        tolerance: float = 1e-9,
    ) -> "IndependentCategoricalSampler":
        probabilities = np.asarray(probabilities, dtype=float)
        validate_expected_quota_allocation(problem, probabilities, tolerance)
        edge_order = np.argsort(problem.tails, kind="stable")
        sorted_tails = problem.tails[edge_order]
        counts = np.bincount(sorted_tails, minlength=problem.n_left)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        row_mass = np.bincount(
            problem.tails, weights=probabilities, minlength=problem.n_left
        )
        conditional_cdf = np.empty(problem.m, dtype=float)
        for client in range(problem.n_left):
            start, end = int(offsets[client]), int(offsets[client + 1])
            values = probabilities[edge_order[start:end]] / row_mass[client]
            cdf = np.cumsum(values)
            cdf[-1] = 1.0
            conditional_cdf[start:end] = cdf
        return cls(
            problem=problem,
            probabilities=probabilities.copy(),
            edge_order=edge_order,
            offsets=offsets,
            conditional_cdf=conditional_cdf,
            row_mass=row_mass,
        )

    def draw(self, rng: np.random.Generator) -> Array:
        """Return selected edge indices, with at most one edge per client."""

        active_uniform = rng.random(self.problem.n_left)
        category_uniform = rng.random(self.problem.n_left)
        active = np.flatnonzero(active_uniform < self.row_mass)
        selected = np.empty(active.size, dtype=np.int64)
        for position, client_value in enumerate(active):
            client = int(client_value)
            start, end = int(self.offsets[client]), int(self.offsets[client + 1])
            local = int(
                np.searchsorted(
                    self.conditional_cdf[start:end],
                    category_uniform[client],
                    side="right",
                )
            )
            selected[position] = self.edge_order[start + min(local, end - start - 1)]
        return selected


def ht_model_estimate(
    problem: ReciprocalTransportProblem,
    probabilities: Array,
    selected_edges: Array,
    edge_updates: Array,
) -> Array:
    """Return one Horvitz--Thompson update estimate per model."""

    probabilities = np.asarray(probabilities, dtype=float)
    selected_edges = np.asarray(selected_edges, dtype=np.int64)
    edge_updates = np.asarray(edge_updates, dtype=float)
    if probabilities.shape != (problem.m,):
        raise ValueError("probabilities have the wrong dimension")
    if edge_updates.ndim != 2 or edge_updates.shape[0] != problem.m:
        raise ValueError("edge_updates must have shape (m, vector_dimension)")
    if np.any(selected_edges < 0) or np.any(selected_edges >= problem.m):
        raise ValueError("selected edge index is out of range")
    estimate = np.zeros((problem.n_right, edge_updates.shape[1]), dtype=float)
    local_heads = problem.heads[selected_edges] - problem.n_left
    contributions = edge_updates[selected_edges] / probabilities[selected_edges, None]
    np.add.at(estimate, local_heads, contributions)
    return estimate


def exact_conditional_mse_by_model(
    problem: ReciprocalTransportProblem,
    probabilities: Array,
    edge_updates: Array,
    model_weights: Array | None = None,
) -> tuple[Array, float]:
    """Evaluate the exact conditional HT MSE from the theorem."""

    probabilities = np.asarray(probabilities, dtype=float)
    edge_updates = np.asarray(edge_updates, dtype=float)
    if probabilities.shape != (problem.m,) or np.any(probabilities <= 0.0):
        raise ValueError("probabilities must be a positive edge vector")
    if edge_updates.ndim != 2 or edge_updates.shape[0] != problem.m:
        raise ValueError("edge_updates must have shape (m, vector_dimension)")
    squared_norm = np.einsum("ij,ij->i", edge_updates, edge_updates)
    edge_mse = squared_norm * (1.0 / probabilities - 1.0)
    local_heads = problem.heads - problem.n_left
    by_model = np.bincount(
        local_heads, weights=edge_mse, minlength=problem.n_right
    )
    if model_weights is None:
        total = float(np.sum(by_model))
    else:
        model_weights = np.asarray(model_weights, dtype=float)
        if model_weights.shape != (problem.n_right,) or np.any(model_weights < 0.0):
            raise ValueError("model_weights must be a nonnegative model vector")
        total = float(model_weights @ by_model)
    return by_model, total
