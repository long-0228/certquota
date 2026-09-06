"""Problem representation for sparse bipartite reciprocal transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import fsum
from struct import pack
from typing import Iterable, Tuple

import numpy as np
import scipy.sparse as sp


Array = np.ndarray
_PROBLEM_TOKEN = object()


def _immutable_array(values: Array, dtype) -> Array:
    """Return a canonical array backed by immutable ``bytes`` storage."""

    contiguous = np.ascontiguousarray(values, dtype=dtype)
    snapshot = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return snapshot.reshape(contiguous.shape)


def _array_digest(hasher, values: Array, dtype) -> None:
    """Append a canonical shape-and-content encoding to ``hasher``."""

    canonical = np.ascontiguousarray(values, dtype=dtype)
    hasher.update(pack("<Q", canonical.ndim))
    for dimension in canonical.shape:
        hasher.update(pack("<Q", int(dimension)))
    hasher.update(canonical.tobytes(order="C"))


@dataclass(frozen=True)
class ReciprocalTransportProblem:
    """A connected bipartite instance with left-to-right oriented edges.

    The incidence convention is ``+1`` at a left endpoint and ``-1`` at a
    right endpoint.  Consequently ``B @ x = (a, -d)`` encodes row and column
    marginals.
    """

    n_left: int
    n_right: int
    tails: Array
    heads: Array
    beta: Array
    c: Array
    mu: Array
    declared_root_beta: float = field(init=False)
    root_beta_adjustment: float = field(init=False)
    _topology_fingerprint: str = field(init=False, repr=False, compare=False)
    _content_fingerprint: str = field(init=False, repr=False, compare=False)
    _capability: object = field(
        init=False, repr=False, compare=False, default=None
    )
    _validation_seal: tuple | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        # These arrays define both the mathematical instance and cached
        # topology-dependent certificate plans.  Own immutable snapshots so a
        # caller cannot invalidate either through an aliased input array.
        if (
            isinstance(self.n_left, (bool, np.bool_))
            or isinstance(self.n_right, (bool, np.bool_))
            or not isinstance(self.n_left, (int, np.integer))
            or not isinstance(self.n_right, (int, np.integer))
            or self.n_left <= 0
            or self.n_right <= 0
        ):
            raise ValueError("n_left and n_right must be positive integers")
        tails = np.array(self.tails, dtype=np.int64, copy=True)
        heads = np.array(self.heads, dtype=np.int64, copy=True)
        beta = np.array(self.beta, dtype=float, copy=True)
        c = np.array(self.c, dtype=float, copy=True)
        mu = np.array(self.mu, dtype=float, copy=True)
        m = tails.size
        n = self.n_left + self.n_right

        if any(
            values.ndim != 1 for values in (tails, heads, beta, c, mu)
        ):
            raise ValueError("problem arrays must be one-dimensional")

        if heads.size != m or c.size != m or mu.size != m:
            raise ValueError("edge arrays, c, and mu must have the same length")
        if beta.size != n:
            raise ValueError("beta has the wrong dimension")
        if m == 0:
            raise ValueError("the eligibility graph must contain at least one edge")
        if np.any(tails < 0) or np.any(tails >= self.n_left):
            raise ValueError("tails must index left vertices")
        if np.any(heads < self.n_left) or np.any(heads >= n):
            raise ValueError("heads must index right vertices in global indexing")
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            raise ValueError("all reciprocal weights mu must be finite and positive")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(beta)):
            raise ValueError("c and beta must be finite")
        declared_balance = fsum(float(value) for value in beta)
        beta_l1 = fsum(abs(float(value)) for value in beta)
        if abs(declared_balance) > 1e-9 * max(1.0, beta_l1):
            raise ValueError("beta must be balanced")

        # The incidence constraints have rank n-1.  Treat beta[:-1] as the
        # independent binary64 inputs and define the last component in exact
        # real arithmetic as their negative sum.  ``beta[-1]`` stores only the
        # nearest binary64 representative; strict routines reconstruct the
        # implied component from beta[:-1] instead of trusting this rounding.
        declared_root_beta = float(beta[-1])
        implied_root_beta = -fsum(float(value) for value in beta[:-1])
        beta[-1] = implied_root_beta

        tails = _immutable_array(tails, np.dtype("<i8"))
        heads = _immutable_array(heads, np.dtype("<i8"))
        beta = _immutable_array(beta, np.dtype("<f8"))
        c = _immutable_array(c, np.dtype("<f8"))
        mu = _immutable_array(mu, np.dtype("<f8"))

        topology_hasher = sha256()
        topology_hasher.update(b"certquota-problem-topology-v1\0")
        topology_hasher.update(pack("<QQ", int(self.n_left), int(self.n_right)))
        _array_digest(topology_hasher, tails, np.dtype("<i8"))
        _array_digest(topology_hasher, heads, np.dtype("<i8"))
        topology_fingerprint = topology_hasher.hexdigest()

        content_hasher = sha256()
        content_hasher.update(b"certquota-problem-content-v1\0")
        content_hasher.update(bytes.fromhex(topology_fingerprint))
        _array_digest(content_hasher, beta, np.dtype("<f8"))
        _array_digest(content_hasher, c, np.dtype("<f8"))
        _array_digest(content_hasher, mu, np.dtype("<f8"))

        object.__setattr__(self, "tails", tails)
        object.__setattr__(self, "heads", heads)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "declared_root_beta", declared_root_beta)
        object.__setattr__(
            self,
            "root_beta_adjustment",
            float(implied_root_beta - declared_root_beta),
        )
        object.__setattr__(self, "_topology_fingerprint", topology_fingerprint)
        object.__setattr__(
            self, "_content_fingerprint", content_hasher.hexdigest()
        )
        object.__setattr__(self, "_capability", _PROBLEM_TOKEN)
        object.__setattr__(
            self,
            "_validation_seal",
            (
                self.n_left,
                self.n_right,
                id(tails),
                id(heads),
                id(beta),
                id(c),
                id(mu),
                topology_fingerprint,
                content_hasher.hexdigest(),
                declared_root_beta,
                float(implied_root_beta - declared_root_beta),
            ),
        )

    @property
    def n(self) -> int:
        return self.n_left + self.n_right

    @property
    def m(self) -> int:
        return int(self.tails.size)

    @property
    def root(self) -> int:
        """Return the vertex whose quota equation is conservation-implied."""

        return self.n - 1

    @property
    def topology_fingerprint(self) -> str:
        """Return the immutable topology snapshot fingerprint."""

        self.validate()
        return self._topology_fingerprint

    @property
    def content_fingerprint(self) -> str:
        """Return the immutable mathematical-input fingerprint."""

        self.validate()
        return self._content_fingerprint

    def validate(self) -> "ReciprocalTransportProblem":
        """Fail closed if a frozen field was replaced after construction."""

        try:
            current = (
                self.n_left,
                self.n_right,
                id(self.tails),
                id(self.heads),
                id(self.beta),
                id(self.c),
                id(self.mu),
                self._topology_fingerprint,
                self._content_fingerprint,
                self.declared_root_beta,
                self.root_beta_adjustment,
            )
            valid = bool(
                self._capability is _PROBLEM_TOKEN
                and current == self._validation_seal
            )
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise ValueError("problem validation seal is missing or stale")
        return self

    def __reduce__(self):
        """Re-run construction and validation when unpickling."""

        self.validate()
        declared_beta = np.array(self.beta, dtype=float, copy=True)
        declared_beta[-1] = self.declared_root_beta
        return (
            type(self),
            (
                self.n_left,
                self.n_right,
                self.tails,
                self.heads,
                declared_beta,
                self.c,
                self.mu,
            ),
        )

    @property
    def relative_root_beta_adjustment(self) -> float:
        """Return the rounded declared-to-implied root diagnostic.

        ``root_beta_adjustment`` is computed from ``math.fsum`` and binary64
        representatives; it is not the exact-real ``Delta_root`` used by the
        theorem.  Strict routines reconstruct that quantity outward from the
        independent quota entries.
        """

        return float(
            abs(self.root_beta_adjustment)
            / max(1.0, abs(self.declared_root_beta))
        )

    def impose_reduced_balance(self, values: Array) -> Array:
        """Return a numerical representative of an exactly balanced covector.

        Mathematically the last entry is ``-sum(values[:-1])`` in exact real
        arithmetic.  The returned last binary64 value is only its high-accuracy
        floating representative; outward-rounded code performs the sum itself.
        """

        values = np.asarray(values, dtype=float)
        if values.shape != (self.n,):
            raise ValueError("node vector has the wrong dimension")
        balanced = values.copy()
        balanced[-1] = -fsum(float(value) for value in balanced[:-1])
        return balanced

    def balanced_pairing(self, values: Array, y: Array) -> float:
        """Pair an implicitly balanced covector with a potential.

        Only ``values[:-1]`` are independent.  The formula is invariant under
        adding a constant to ``y`` and never uses a rounded stored root value.
        """

        values = np.asarray(values, dtype=float)
        y = np.asarray(y, dtype=float)
        if values.shape != (self.n,) or y.shape != (self.n,):
            raise ValueError("node vector has the wrong dimension")
        return float(values[:-1] @ (y[:-1] - y[-1]))

    @property
    def incidence(self) -> sp.csr_matrix:
        edges = np.arange(self.m, dtype=np.int64)
        rows = np.concatenate([self.tails, self.heads])
        cols = np.concatenate([edges, edges])
        data = np.concatenate([np.ones(self.m), -np.ones(self.m)])
        return sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.m)).tocsr()

    @property
    def edges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(zip(self.tails.tolist(), self.heads.tolist()))

    def objective(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m,) or np.any(x <= 0):
            return float("inf")
        return float(self.c @ x + np.sum(self.mu / x))

    def edge_difference(self, y: Array) -> Array:
        """Return ``B.T @ y`` without materializing the incidence matrix."""

        y = np.asarray(y, dtype=float)
        if y.shape != (self.n,):
            raise ValueError("y has the wrong dimension")
        return y[self.tails] - y[self.heads]

    def node_balance(self, edge_flow: Array) -> Array:
        """Return ``B @ edge_flow`` in linear time and memory."""

        edge_flow = np.asarray(edge_flow, dtype=float)
        if edge_flow.shape != (self.m,):
            raise ValueError("edge_flow has the wrong dimension")
        positive = np.bincount(
            self.tails, weights=edge_flow, minlength=self.n
        )
        negative = np.bincount(
            self.heads, weights=edge_flow, minlength=self.n
        )
        return positive - negative

    def laplacian_matvec(self, y: Array, conductance: Array) -> Array:
        """Apply ``B diag(conductance) B.T`` without a sparse matrix."""

        conductance = np.asarray(conductance, dtype=float)
        if conductance.shape != (self.m,):
            raise ValueError("conductance has the wrong dimension")
        return self.node_balance(conductance * self.edge_difference(y))

    def laplacian_diagonal(self, conductance: Array) -> Array:
        """Return the weighted graph degree vector."""

        conductance = np.asarray(conductance, dtype=float)
        if conductance.shape != (self.m,):
            raise ValueError("conductance has the wrong dimension")
        return (
            np.bincount(self.tails, weights=conductance, minlength=self.n)
            + np.bincount(self.heads, weights=conductance, minlength=self.n)
        )

    def primal_residual(self, x: Array) -> Array:
        residual = self.node_balance(np.asarray(x, dtype=float)) - self.beta
        return self.impose_reduced_balance(residual)

    def dual_slack(self, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        if y.shape != (self.n,):
            raise ValueError("y has the wrong dimension")
        return self.c - self.edge_difference(y)

    @classmethod
    def from_edges(
        cls,
        n_left: int,
        n_right: int,
        edges: Iterable[Tuple[int, int]],
        beta: Array,
        c: Array,
        mu: Array,
    ) -> "ReciprocalTransportProblem":
        pairs = list(edges)
        tails = np.asarray([u for u, _ in pairs], dtype=np.int64)
        raw_heads = np.asarray([v for _, v in pairs], dtype=np.int64)
        if raw_heads.size and raw_heads.max() < n_right:
            heads = raw_heads + n_left
        else:
            heads = raw_heads
        return cls(n_left, n_right, tails, heads, beta, c, mu)

    @classmethod
    def from_reduced_marginals(
        cls,
        n_left: int,
        n_right: int,
        tails: Array,
        heads: Array,
        beta_nonroot: Array,
        c: Array,
        mu: Array,
    ) -> "ReciprocalTransportProblem":
        """Construct an instance from its ``n-1`` independent marginals.

        The incidence equations have rank ``n-1``.  This constructor is the
        unambiguous public interface for exact represented feasibility: callers
        provide only the independent binary64 marginals, and the distinguished
        root marginal is derived with :func:`math.fsum`.  No supplied root value
        is silently substituted into the mathematical instance.
        """

        if (
            isinstance(n_left, (bool, np.bool_))
            or isinstance(n_right, (bool, np.bool_))
            or not isinstance(n_left, (int, np.integer))
            or not isinstance(n_right, (int, np.integer))
            or n_left <= 0
            or n_right <= 0
        ):
            raise ValueError("n_left and n_right must be positive integers")
        independent = np.asarray(beta_nonroot, dtype=float)
        expected = int(n_left) + int(n_right) - 1
        if independent.ndim != 1 or independent.size != expected:
            raise ValueError("beta_nonroot must contain exactly n-1 entries")
        if not np.all(np.isfinite(independent)):
            raise ValueError("beta_nonroot must be finite")
        beta = np.empty(expected + 1, dtype=float)
        beta[:-1] = independent
        beta[-1] = -fsum(float(value) for value in independent)
        return cls(n_left, n_right, tails, heads, beta, c, mu)
