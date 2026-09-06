"""Linear-time tree routing used by all certificates and primal recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from struct import pack
from threading import Lock
from typing import List, Optional
from weakref import ref

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .problem import ReciprocalTransportProblem


Array = np.ndarray
_COMPATIBILITY_CACHE_LOCK = Lock()
_LAST_COMPATIBLE_PROBLEM: dict[int, tuple[object, object]] = {}


def _is_last_compatible(router: "TreeRouter", problem) -> bool:
    """Return whether exact comparison previously accepted this object pair.

    The entry is keyed by identity and contains weak references to both objects.
    Thus object-id reuse cannot produce a hit, dead objects are not retained, and
    the cache is never a substitute for the first exact topology comparison.
    """

    with _COMPATIBILITY_CACHE_LOCK:
        entry = _LAST_COMPATIBLE_PROBLEM.get(id(router))
        return bool(
            entry is not None
            and entry[0]() is router
            and entry[1]() is problem
        )


def _remember_compatible(router: "TreeRouter", problem) -> None:
    """Memoize one exactly checked cross-object pair without retaining it."""

    key = id(router)

    def discard(router_reference) -> None:
        with _COMPATIBILITY_CACHE_LOCK:
            current = _LAST_COMPATIBLE_PROBLEM.get(key)
            if current is not None and current[0] is router_reference:
                _LAST_COMPATIBLE_PROBLEM.pop(key, None)

    router_reference = ref(router, discard)
    problem_reference = ref(problem)
    with _COMPATIBILITY_CACHE_LOCK:
        _LAST_COMPATIBLE_PROBLEM[key] = (
            router_reference,
            problem_reference,
        )


def _readonly_copy(values: Array, dtype=None) -> Array:
    """Return a canonical array backed by immutable ``bytes`` storage."""

    contiguous = np.ascontiguousarray(values, dtype=dtype)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    )
    return result


def _index_vector(values: Array, name: str) -> Array:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.asarray(raw, dtype=np.int64)


def _update_digest(hasher, values: Array, dtype) -> None:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    hasher.update(pack("<Q", canonical.size))
    hasher.update(canonical.tobytes(order="C"))


@dataclass(frozen=True)
class TreeRouter:
    problem: ReciprocalTransportProblem
    tree_edges: Array
    root: int
    parent: Array
    parent_edge: Array
    child_incidence_sign: Array
    postorder: Array
    level_order: Array
    level_offsets: Array
    _validated_topology_fingerprint: str = field(
        init=False, repr=False, compare=False
    )
    _structure_fingerprint: str = field(init=False, repr=False, compare=False)
    _validation_seal: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Own and fully validate a rooted spanning-tree representation."""

        if not isinstance(self.problem, ReciprocalTransportProblem):
            raise TypeError("problem must be a ReciprocalTransportProblem")
        n, m = self.problem.n, self.problem.m
        if isinstance(self.root, (bool, np.bool_)) or not isinstance(
            self.root, (int, np.integer)
        ):
            raise ValueError("root must be an integer vertex")
        root = int(self.root)
        if not 0 <= root < n:
            raise ValueError("invalid root")

        tree_edges = _index_vector(self.tree_edges, "tree_edges")
        parent = _index_vector(self.parent, "parent")
        parent_edge = _index_vector(self.parent_edge, "parent_edge")
        postorder = _index_vector(self.postorder, "postorder")
        level_order = _index_vector(self.level_order, "level_order")
        level_offsets = _index_vector(self.level_offsets, "level_offsets")
        signs = np.asarray(self.child_incidence_sign, dtype=float)

        if (
            tree_edges.shape != (n - 1,)
            or np.any(tree_edges < 0)
            or np.any(tree_edges >= m)
            or np.unique(tree_edges).size != n - 1
        ):
            raise ValueError("tree_edges must contain n-1 unique valid edges")
        if (
            parent.shape != (n,)
            or parent_edge.shape != (n,)
            or signs.shape != (n,)
        ):
            raise ValueError("parent, parent_edge, and signs must have length n")
        if postorder.shape != (n,) or level_order.shape != (n,):
            raise ValueError("tree traversal arrays must have length n")
        if not np.all(np.isfinite(signs)):
            raise ValueError("incidence signs must be finite")

        vertices = np.arange(n, dtype=np.int64)
        nonroot = vertices != root
        if (
            parent[root] != -1
            or parent_edge[root] != -1
            or signs[root] != 0.0
        ):
            raise ValueError("root metadata must use parent=-1, edge=-1, sign=0")
        if (
            np.any(parent[nonroot] < 0)
            or np.any(parent[nonroot] >= n)
            or np.any(parent[nonroot] == vertices[nonroot])
        ):
            raise ValueError(
                "every non-root vertex must have a valid distinct parent"
            )
        chosen = parent_edge[nonroot]
        if (
            np.any(chosen < 0)
            or np.any(chosen >= m)
            or np.unique(chosen).size != n - 1
            or not np.array_equal(np.sort(chosen), np.sort(tree_edges))
        ):
            raise ValueError(
                "every tree edge must be the unique parent edge of one "
                "non-root vertex"
            )

        tail = self.problem.tails[chosen]
        head = self.problem.heads[chosen]
        child = vertices[nonroot]
        expected_sign = np.empty(n - 1, dtype=float)
        tail_child = (tail == child) & (head == parent[nonroot])
        head_child = (head == child) & (tail == parent[nonroot])
        if not np.all(tail_child | head_child):
            raise ValueError(
                "parent_edge endpoints disagree with parent metadata"
            )
        expected_sign[tail_child] = 1.0
        expected_sign[head_child] = -1.0
        if not np.array_equal(signs[nonroot], expected_sign):
            raise ValueError(
                "child incidence signs disagree with oriented graph endpoints"
            )

        # Independently verify that the declared edges form a spanning tree.
        union_parent = np.arange(n, dtype=np.int64)

        def find(vertex: int) -> int:
            while union_parent[vertex] != vertex:
                union_parent[vertex] = union_parent[union_parent[vertex]]
                vertex = int(union_parent[vertex])
            return vertex

        for edge_value in tree_edges:
            edge = int(edge_value)
            left = int(self.problem.tails[edge])
            right = int(self.problem.heads[edge])
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                raise ValueError("tree_edges contains a cycle")
            union_parent[right_root] = left_root
        if len({find(vertex) for vertex in range(n)}) != 1:
            raise ValueError("tree_edges does not span the graph")

        if (
            np.unique(level_order).size != n
            or np.any(level_order < 0)
            or np.any(level_order >= n)
            or level_order[0] != root
        ):
            raise ValueError("level_order must be a root-first vertex permutation")
        positions = np.empty(n, dtype=np.int64)
        positions[level_order] = np.arange(n, dtype=np.int64)
        if np.any(positions[parent[nonroot]] >= positions[nonroot]):
            raise ValueError("level_order must place every parent before its children")
        depth = np.zeros(n, dtype=np.int64)
        for vertex_value in level_order[1:]:
            vertex = int(vertex_value)
            depth[vertex] = depth[int(parent[vertex])] + 1
        order_depth = depth[level_order]
        if np.any(order_depth[1:] < order_depth[:-1]):
            raise ValueError(
                "level_order must group vertices by nondecreasing depth"
            )
        counts = np.bincount(
            depth, minlength=int(np.max(depth)) + 1
        )
        expected_offsets = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(counts, dtype=np.int64),
            )
        )
        if not np.array_equal(level_offsets, expected_offsets):
            raise ValueError("level_offsets disagrees with rooted-tree depths")
        if not np.array_equal(postorder, level_order[::-1]):
            raise ValueError("postorder must be the reverse level traversal")

        snapshots = (
            _readonly_copy(tree_edges, np.dtype("<i8")),
            _readonly_copy(parent, np.dtype("<i8")),
            _readonly_copy(parent_edge, np.dtype("<i8")),
            _readonly_copy(signs, np.dtype("<f8")),
            _readonly_copy(postorder, np.dtype("<i8")),
            _readonly_copy(level_order, np.dtype("<i8")),
            _readonly_copy(level_offsets, np.dtype("<i8")),
        )
        names = (
            "tree_edges",
            "parent",
            "parent_edge",
            "child_incidence_sign",
            "postorder",
            "level_order",
            "level_offsets",
        )
        for name, value in zip(names, snapshots):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root", root)

        digest = sha256()
        digest.update(b"certquota-tree-router-v1\0")
        digest.update(bytes.fromhex(self.problem.topology_fingerprint))
        digest.update(pack("<q", root))
        for value in snapshots:
            _update_digest(digest, value, value.dtype)
        structure = digest.hexdigest()
        object.__setattr__(
            self,
            "_validated_topology_fingerprint",
            self.problem.topology_fingerprint,
        )
        object.__setattr__(self, "_structure_fingerprint", structure)
        object.__setattr__(
            self,
            "_validation_seal",
            (
                id(self.problem),
                root,
                structure,
                *(id(value) for value in snapshots),
            ),
        )

    @property
    def structure_fingerprint(self) -> str:
        """Return a digest of the validated rooted-tree schedule."""

        return self._structure_fingerprint

    def validate_for(self, problem: ReciprocalTransportProblem) -> "TreeRouter":
        """Fail closed unless this sealed router is valid for ``problem``.

        Full structural validation occurs once in :meth:`__post_init__`.  A
        same-owner check is constant work because bytes-backed arrays cannot be
        made writeable and the identity seal detects attribute replacement.
        The first validation of a cross-problem pair additionally compares the
        immutable endpoint arrays exactly in linear work; a digest is never
        authoritative by itself.  A process-local weak identity cache makes
        repeated validation of that same already-checked pair constant work.
        """

        try:
            arrays = (
                self.tree_edges,
                self.parent,
                self.parent_edge,
                self.child_incidence_sign,
                self.postorder,
                self.level_order,
                self.level_offsets,
            )
            current = (
                id(self.problem),
                self.root,
                self._structure_fingerprint,
                *(id(value) for value in arrays),
            )
            sealed = bool(
                current == self._validation_seal
                and self.problem.topology_fingerprint
                == self._validated_topology_fingerprint
            )
        except (AttributeError, TypeError):
            sealed = False
        if not sealed:
            raise ValueError("router validation seal is missing or stale")
        if not isinstance(problem, ReciprocalTransportProblem):
            raise ValueError("router is incompatible with problem topology")
        # Cache hits rely on the target object's immutable snapshots.  Validate
        # its constant-work identity seal before consulting the memoized pair so
        # post-validation field replacement can never inherit old authority.
        problem.validate()
        # A digest is only a fast mismatch filter, never the authority for a
        # zero-error topology check.  Dynamic instances may legitimately reuse
        # a router owned by another problem object, so compare the immutable
        # endpoint arrays exactly before accepting that cross-object reuse.
        # This keeps validation linear in the declared input size without
        # introducing a cryptographic collision-resistance premise.
        if problem is not self.problem:
            if _is_last_compatible(self, problem):
                return self
            compatible = bool(
                problem.n_left == self.problem.n_left
                and problem.n_right == self.problem.n_right
                and problem.n == self.problem.n
                and problem.m == self.problem.m
                and problem.topology_fingerprint
                == self._validated_topology_fingerprint
                and np.array_equal(problem.tails, self.problem.tails)
                and np.array_equal(problem.heads, self.problem.heads)
            )
            if not compatible:
                raise ValueError("router is incompatible with problem topology")
            _remember_compatible(self, problem)
        return self

    def is_compatible(self, problem: ReciprocalTransportProblem) -> bool:
        """Return whether the validated router matches ``problem`` topology."""

        try:
            self.validate_for(problem)
        except (TypeError, ValueError):
            return False
        return True

    def __reduce__(self):
        """Re-run full structural validation when unpickling."""

        return (
            type(self),
            (
                self.problem,
                self.tree_edges,
                self.root,
                self.parent,
                self.parent_edge,
                self.child_incidence_sign,
                self.postorder,
                self.level_order,
                self.level_offsets,
            ),
        )

    @classmethod
    def build(
        cls,
        problem: ReciprocalTransportProblem,
        tree_edges: Optional[Array] = None,
        root: Optional[int] = None,
        edge_cost: Optional[Array] = None,
    ) -> "TreeRouter":
        """Build a BFS tree or a minimum-cost spanning tree.

        ``edge_cost`` is used only to choose a Kruskal tree.  It does not enter
        the certificate itself.
        """

        n, m = problem.n, problem.m
        if root is None:
            root = n - 1
        if not 0 <= root < n:
            raise ValueError("invalid root")

        if tree_edges is None:
            if edge_cost is not None and m >= 50_000:
                cost = np.asarray(edge_cost, dtype=float)
                if (
                    cost.shape != (m,)
                    or np.any(cost <= 0)
                    or not np.all(np.isfinite(cost))
                ):
                    raise ValueError("edge_cost must be finite and positive")
                graph = sp.coo_matrix(
                    (
                        np.concatenate([cost, cost]),
                        (
                            np.concatenate([problem.tails, problem.heads]),
                            np.concatenate([problem.heads, problem.tails]),
                        ),
                    ),
                    shape=(n, n),
                ).tocsr()
                mst = minimum_spanning_tree(graph, overwrite=True).tocoo()
                low = np.minimum(mst.row, mst.col).astype(np.int64, copy=False)
                high = np.maximum(mst.row, mst.col).astype(np.int64, copy=False)
                mst_keys = low * np.int64(n) + high
                edge_low = np.minimum(problem.tails, problem.heads)
                edge_high = np.maximum(problem.tails, problem.heads)
                edge_keys = edge_low * np.int64(n) + edge_high
                order = np.argsort(edge_keys, kind="mergesort")
                sorted_keys = edge_keys[order]
                positions = np.searchsorted(sorted_keys, mst_keys)
                if (
                    mst_keys.size != n - 1
                    or np.any(positions >= sorted_keys.size)
                    or np.any(sorted_keys[positions] != mst_keys)
                ):
                    raise ValueError("the graph is disconnected or MST mapping failed")
                tree_edges = order[positions].astype(np.int64, copy=False)
            elif edge_cost is None:
                order = np.arange(m, dtype=np.int64)
            else:
                cost = np.asarray(edge_cost, dtype=float)
                if (
                    cost.shape != (m,)
                    or np.any(cost <= 0)
                    or not np.all(np.isfinite(cost))
                ):
                    raise ValueError("edge_cost must be finite and positive")
                order = np.argsort(cost, kind="mergesort")
            if tree_edges is None:
                uf_parent = np.arange(n, dtype=np.int64)
                uf_rank = np.zeros(n, dtype=np.int8)

                def find(a: int) -> int:
                    while uf_parent[a] != a:
                        uf_parent[a] = uf_parent[uf_parent[a]]
                        a = int(uf_parent[a])
                    return a

                chosen: List[int] = []
                for e in order:
                    u, v = int(problem.tails[e]), int(problem.heads[e])
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        continue
                    if uf_rank[ru] < uf_rank[rv]:
                        ru, rv = rv, ru
                    uf_parent[rv] = ru
                    if uf_rank[ru] == uf_rank[rv]:
                        uf_rank[ru] += 1
                    chosen.append(int(e))
                    if len(chosen) == n - 1:
                        break
                tree_edges = np.asarray(chosen, dtype=np.int64)
        else:
            tree_edges = np.asarray(tree_edges, dtype=np.int64)

        if tree_edges.size != n - 1:
            raise ValueError("the graph is disconnected or tree_edges is not a tree")
        if np.any(tree_edges < 0) or np.any(tree_edges >= m):
            raise ValueError("invalid tree edge index")

        tree_tail = problem.tails[tree_edges]
        tree_head = problem.heads[tree_edges]
        adjacency = sp.coo_matrix(
            (
                np.ones(2 * tree_edges.size, dtype=np.int8),
                (
                    np.concatenate([tree_tail, tree_head]),
                    np.concatenate([tree_head, tree_tail]),
                ),
            ),
            shape=(n, n),
        ).tocsr()
        traversal, predecessors = breadth_first_order(
            adjacency,
            i_start=root,
            directed=False,
            return_predecessors=True,
        )
        traversal = np.asarray(traversal, dtype=np.int64)
        parent = np.asarray(predecessors, dtype=np.int64)
        if traversal.size != n or np.any(parent[np.arange(n) != root] < 0):
            raise ValueError("tree does not span the graph")
        parent[root] = -1

        low = np.minimum(tree_tail, tree_head)
        high = np.maximum(tree_tail, tree_head)
        keys = low * np.int64(n) + high
        key_order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[key_order]
        vertices = np.arange(n, dtype=np.int64)
        nonroot = vertices != root
        parent_low = np.minimum(vertices[nonroot], parent[nonroot])
        parent_high = np.maximum(vertices[nonroot], parent[nonroot])
        parent_keys = parent_low * np.int64(n) + parent_high
        positions = np.searchsorted(sorted_keys, parent_keys)
        if np.any(positions >= sorted_keys.size) or np.any(
            sorted_keys[positions] != parent_keys
        ):
            raise ArithmeticError("failed to recover parent-edge indices")
        parent_edge = np.full(n, -1, dtype=np.int64)
        parent_edge[nonroot] = tree_edges[key_order[positions]]
        child_sign = np.zeros(n, dtype=float)
        chosen = parent_edge[nonroot]
        child_sign[nonroot] = np.where(
            problem.tails[chosen] == vertices[nonroot], 1.0, -1.0
        )
        child_count = np.bincount(
            parent[nonroot], minlength=n
        ).astype(np.int64, copy=False)
        offsets = [0, 1]
        level_start, level_end = 0, 1
        while level_end < n:
            next_count = int(
                np.sum(child_count[traversal[level_start:level_end]])
            )
            if next_count <= 0:
                raise ArithmeticError("invalid BFS level structure")
            level_start, level_end = level_end, level_end + next_count
            offsets.append(level_end)

        return cls(
            problem=problem,
            tree_edges=_readonly_copy(tree_edges, np.int64),
            root=root,
            parent=_readonly_copy(parent, np.int64),
            parent_edge=_readonly_copy(parent_edge, np.int64),
            child_incidence_sign=_readonly_copy(child_sign, float),
            postorder=_readonly_copy(traversal[::-1], np.int64),
            level_order=_readonly_copy(traversal, np.int64),
            level_offsets=_readonly_copy(offsets, np.int64),
        )

    def route(self, demand: Array) -> Array:
        """Return the unique tree-supported flow satisfying ``B f = demand``.

        Floating-point imbalance is placed at the root.  For exact input data the
        correction is zero; interval-certified code must instead track it explicitly.
        """

        self.validate_for(self.problem)
        r = np.asarray(demand, dtype=float).copy()
        if r.shape != (self.problem.n,):
            raise ValueError("demand has the wrong dimension")
        r[self.root] -= float(r.sum())
        subtree = r.copy()
        flow = np.zeros(self.problem.m, dtype=float)
        for level in range(self.level_offsets.size - 2, 0, -1):
            start = int(self.level_offsets[level])
            end = int(self.level_offsets[level + 1])
            vertices = self.level_order[start:end]
            edges = self.parent_edge[vertices]
            values = subtree[vertices]
            flow[edges] = self.child_incidence_sign[vertices] * values
            np.add.at(subtree, self.parent[vertices], values)
        return flow

    def energy(self, demand: Array, conductance: Array) -> float:
        self.validate_for(self.problem)
        conductance = np.asarray(conductance, dtype=float)
        if conductance.shape != (self.problem.m,) or np.any(conductance <= 0):
            raise ValueError("conductances must be positive")
        flow = self.route(demand)
        e = self.tree_edges
        return float(np.sum(flow[e] ** 2 / conductance[e]))

    def solve_laplacian(self, demand: Array, conductance: Array) -> Array:
        """Solve the weighted tree Laplacian in linear time.

        The returned potential is mean zero.  It satisfies
        ``B diag(conductance) B.T potential = demand`` up to floating-point
        roundoff, with only the spanning-tree conductances retained.
        """

        self.validate_for(self.problem)
        conductance = np.asarray(conductance, dtype=float)
        if conductance.shape != (self.problem.m,) or np.any(conductance <= 0):
            raise ValueError("conductances must be positive")
        flow = self.route(demand)
        potential = np.zeros(self.problem.n, dtype=float)
        for level in range(1, self.level_offsets.size - 1):
            start = int(self.level_offsets[level])
            end = int(self.level_offsets[level + 1])
            vertices = self.level_order[start:end]
            edges = self.parent_edge[vertices]
            voltage_drop = flow[edges] / conductance[edges]
            potential[vertices] = (
                potential[self.parent[vertices]]
                + self.child_incidence_sign[vertices] * voltage_drop
            )
        potential -= float(potential.mean())
        return potential
