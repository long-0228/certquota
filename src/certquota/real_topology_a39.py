"""Local-only MovieLens-1M topology loader and frozen A39 KKT planting.

This module intentionally contains no downloader.  MovieLens supplies only the
real sparse user--movie support; all optimization quantities are planted and do
not define or evaluate a recommender system.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
from math import fsum
from pathlib import Path
import re
import struct
from typing import Any
import zipfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .instances import PlantedInstance
from .problem import ReciprocalTransportProblem


Array = np.ndarray

A39_PROTOCOL = "certquota-design-v1-a39-real-topology"
MOVIELENS_1M_PAGE = "https://grouplens.org/datasets/movielens/1m/"
MOVIELENS_1M_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
)
MOVIELENS_1M_MD5_URL = f"{MOVIELENS_1M_URL}.md5"
MOVIELENS_1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"
MOVIELENS_1M_RATINGS_MEMBER = "ml-1m/ratings.dat"

CONFIRMATORY_NUMERIC_SEEDS = (39_001, 39_002, 39_003, 39_004, 39_005)
ROW_BUDGET = 0.2
RATING_LOG_COEFFICIENT = 0.15
FLOW_NOISE_HALF_WIDTH = 0.25
MU_LOG10_HALF_WIDTH = 2.0
POTENTIAL_TO_MIN_SLACK = 0.1
WARM_START_TO_MIN_SLACK = 0.01
WARM_START_SEED_OFFSET = 1_000_003

_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TOPOLOGY_DOMAIN = b"certquota-a39-movielens-topology-v1\0"


def _readonly(values: Array, dtype: Any) -> Array:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _archive_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_sha256(
    original_users: Array,
    original_movies: Array,
    n_left: int,
    n_right: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(_TOPOLOGY_DOMAIN)
    digest.update(struct.pack("<QQQ", n_left, n_right, int(original_users.size)))
    pairs = np.empty((original_users.size, 2), dtype="<i8")
    pairs[:, 0] = original_users
    pairs[:, 1] = original_movies
    digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MovieLens1MTopology:
    """Canonical largest-component topology derived from ``ratings.dat``."""

    original_user_ids: Array
    original_movie_ids: Array
    tails: Array
    heads_local: Array
    ratings: Array
    timestamps: Array
    archive_md5: str
    member_crc32: str
    member_uncompressed_bytes: int
    raw_rows: int
    duplicate_rows_removed: int
    component_count: int
    dropped_edges: int
    dropped_users: int
    dropped_movies: int
    topology_sha256: str

    def __post_init__(self) -> None:
        users = _readonly(self.original_user_ids, np.int64)
        movies = _readonly(self.original_movie_ids, np.int64)
        tails = _readonly(self.tails, np.int64)
        heads = _readonly(self.heads_local, np.int64)
        ratings = _readonly(self.ratings, np.int8)
        timestamps = _readonly(self.timestamps, np.int64)
        m = tails.size
        if not (
            heads.size == ratings.size == timestamps.size == m
            and users.ndim == movies.ndim == tails.ndim == heads.ndim == 1
        ):
            raise ValueError("MovieLens topology arrays have incompatible shapes")
        if users.size == 0 or movies.size == 0 or m == 0:
            raise ValueError("MovieLens selected component must be nonempty")
        if np.any(tails < 0) or np.any(tails >= users.size):
            raise ValueError("MovieLens tail index is out of range")
        if np.any(heads < 0) or np.any(heads >= movies.size):
            raise ValueError("MovieLens head index is out of range")
        if np.any(ratings < 1) or np.any(ratings > 5):
            raise ValueError("MovieLens ratings must lie in {1,...,5}")
        if self.archive_md5 != self.archive_md5.lower() or not _MD5_PATTERN.fullmatch(
            self.archive_md5
        ):
            raise ValueError("archive_md5 must be lowercase MD5")
        if not re.fullmatch(r"[0-9a-f]{8}", self.member_crc32):
            raise ValueError("member_crc32 must be eight lowercase hex digits")
        object.__setattr__(self, "original_user_ids", users)
        object.__setattr__(self, "original_movie_ids", movies)
        object.__setattr__(self, "tails", tails)
        object.__setattr__(self, "heads_local", heads)
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "timestamps", timestamps)

    @property
    def n_left(self) -> int:
        return int(self.original_user_ids.size)

    @property
    def n_right(self) -> int:
        return int(self.original_movie_ids.size)

    @property
    def n(self) -> int:
        return self.n_left + self.n_right

    @property
    def m(self) -> int:
        return int(self.tails.size)

    @property
    def heads_global(self) -> Array:
        result = self.heads_local + self.n_left
        result.setflags(write=False)
        return result


def _parse_ratings_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> tuple[Array, Array, Array, Array, int]:
    if member.flag_bits & 0x1:
        raise ValueError("encrypted MovieLens members are not supported")
    users = array("q")
    movies = array("q")
    ratings = array("b")
    timestamps = array("q")
    try:
        with archive.open(member, "r") as binary:
            with io.TextIOWrapper(binary, encoding="ascii", errors="strict") as text:
                for line_number, raw_line in enumerate(text, start=1):
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        raise ValueError(
                            f"blank ratings.dat line at one-based line {line_number}"
                        )
                    fields = line.split("::")
                    if len(fields) != 4:
                        raise ValueError(
                            f"malformed ratings.dat line at one-based line {line_number}"
                        )
                    try:
                        user_id, movie_id, rating, timestamp = map(int, fields)
                    except ValueError as error:
                        raise ValueError(
                            "noninteger ratings.dat field at one-based line "
                            f"{line_number}"
                        ) from error
                    if user_id <= 0 or movie_id <= 0:
                        raise ValueError("MovieLens user and movie IDs must be positive")
                    if rating < 1 or rating > 5:
                        raise ValueError("MovieLens rating must be an integer in {1,...,5}")
                    if timestamp < 0 or timestamp > np.iinfo(np.int64).max:
                        raise ValueError("MovieLens timestamp is outside int64 range")
                    users.append(user_id)
                    movies.append(movie_id)
                    ratings.append(rating)
                    timestamps.append(timestamp)
    except UnicodeDecodeError as error:
        raise ValueError("ratings.dat must be ASCII under the A39 protocol") from error
    if not users:
        raise ValueError("ratings.dat contains no observations")
    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(movies, dtype=np.int64),
        np.asarray(ratings, dtype=np.int8),
        np.asarray(timestamps, dtype=np.int64),
        len(users),
    )


def _deduplicate_latest(
    users: Array, movies: Array, ratings: Array, timestamps: Array
) -> tuple[Array, Array, Array, Array, int]:
    # lexsort uses the last key as primary: pair order first, then descending
    # timestamp and rating so the first row of a duplicate block is retained.
    order = np.lexsort(
        (
            -ratings.astype(np.int16, copy=False),
            -timestamps,
            movies,
            users,
        )
    )
    users = users[order]
    movies = movies[order]
    ratings = ratings[order]
    timestamps = timestamps[order]
    first = np.ones(users.size, dtype=bool)
    first[1:] = (users[1:] != users[:-1]) | (movies[1:] != movies[:-1])
    removed = int(users.size - np.count_nonzero(first))
    return (
        users[first],
        movies[first],
        ratings[first],
        timestamps[first],
        removed,
    )


def _select_component(
    users: Array, movies: Array, ratings: Array, timestamps: Array
) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, int]]:
    unique_users = np.unique(users)
    unique_movies = np.unique(movies)
    n_left = int(unique_users.size)
    n_right = int(unique_movies.size)
    tails = np.searchsorted(unique_users, users).astype(np.int64, copy=False)
    heads = np.searchsorted(unique_movies, movies).astype(np.int64, copy=False)
    global_heads = heads + n_left
    adjacency = sp.coo_matrix(
        (
            np.ones(2 * users.size, dtype=np.int8),
            (
                np.concatenate([tails, global_heads]),
                np.concatenate([global_heads, tails]),
            ),
        ),
        shape=(n_left + n_right, n_left + n_right),
    ).tocsr()
    component_count, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    if np.any(labels[tails] != labels[global_heads]):
        raise ArithmeticError("component labels disagree across an edge")
    edge_counts = np.bincount(labels[tails], minlength=component_count)
    vertex_counts = np.bincount(labels, minlength=component_count)
    sentinel = np.iinfo(np.int64).max
    min_user = np.full(component_count, sentinel, dtype=np.int64)
    min_movie = np.full(component_count, sentinel, dtype=np.int64)
    np.minimum.at(min_user, labels[:n_left], unique_users)
    np.minimum.at(min_movie, labels[n_left:], unique_movies)
    selected = min(
        range(component_count),
        key=lambda index: (
            -int(edge_counts[index]),
            -int(vertex_counts[index]),
            int(min_user[index]),
            int(min_movie[index]),
        ),
    )
    edge_mask = labels[tails] == selected
    kept_users = unique_users[labels[:n_left] == selected]
    kept_movies = unique_movies[labels[n_left:] == selected]
    selected_users = users[edge_mask]
    selected_movies = movies[edge_mask]
    selected_ratings = ratings[edge_mask]
    selected_timestamps = timestamps[edge_mask]
    selected_tails = np.searchsorted(kept_users, selected_users).astype(
        np.int64, copy=False
    )
    selected_heads = np.searchsorted(kept_movies, selected_movies).astype(
        np.int64, copy=False
    )
    metadata = {
        "component_count": int(component_count),
        "dropped_edges": int(users.size - selected_users.size),
        "dropped_users": int(unique_users.size - kept_users.size),
        "dropped_movies": int(unique_movies.size - kept_movies.size),
    }
    return (
        kept_users,
        kept_movies,
        selected_tails,
        selected_heads,
        selected_ratings,
        selected_timestamps,
        metadata,
    )


def load_movielens_1m_zip(
    path: str | Path,
    *,
    expected_md5: str = MOVIELENS_1M_MD5,
) -> MovieLens1MTopology:
    """Load the canonical MovieLens-1M topology from a verified local zip.

    ``expected_md5`` exists only so unit tests can use tiny local fixtures.  The
    A39 confirmatory runner requires the frozen official value.
    """

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"MovieLens-1M archive not found at {archive_path}; obtain it from "
            f"{MOVIELENS_1M_URL} and pass the local zip path explicitly"
        )
    if not isinstance(expected_md5, str) or not _MD5_PATTERN.fullmatch(
        expected_md5
    ):
        raise ValueError("expected_md5 must be a lowercase 32-digit MD5")
    observed_md5 = _archive_md5(archive_path)
    if observed_md5 != expected_md5:
        raise ValueError(
            "MovieLens-1M archive MD5 mismatch: "
            f"expected {expected_md5}, observed {observed_md5}"
        )
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            matches = [
                item
                for item in archive.infolist()
                if item.filename == MOVIELENS_1M_RATINGS_MEMBER
            ]
            if len(matches) != 1:
                raise ValueError(
                    "MovieLens archive must contain exactly one "
                    f"{MOVIELENS_1M_RATINGS_MEMBER!r} member"
                )
            member = matches[0]
            users, movies, ratings, timestamps, raw_rows = _parse_ratings_member(
                archive, member
            )
    except zipfile.BadZipFile as error:
        raise ValueError("MovieLens archive is not a valid CRC-clean zip") from error
    users, movies, ratings, timestamps, removed = _deduplicate_latest(
        users, movies, ratings, timestamps
    )
    (
        kept_users,
        kept_movies,
        tails,
        heads,
        ratings,
        timestamps,
        component_metadata,
    ) = _select_component(users, movies, ratings, timestamps)
    topology_hash = _topology_sha256(
        kept_users[tails], kept_movies[heads], kept_users.size, kept_movies.size
    )
    return MovieLens1MTopology(
        original_user_ids=kept_users,
        original_movie_ids=kept_movies,
        tails=tails,
        heads_local=heads,
        ratings=ratings,
        timestamps=timestamps,
        archive_md5=observed_md5,
        member_crc32=f"{member.CRC:08x}",
        member_uncompressed_bytes=int(member.file_size),
        raw_rows=raw_rows,
        duplicate_rows_removed=removed,
        component_count=component_metadata["component_count"],
        dropped_edges=component_metadata["dropped_edges"],
        dropped_users=component_metadata["dropped_users"],
        dropped_movies=component_metadata["dropped_movies"],
        topology_sha256=topology_hash,
    )


def make_movielens_planted_instance(
    topology: MovieLens1MTopology,
    seed: int,
) -> PlantedInstance:
    """Attach the frozen A39 synthetic KKT data to a real topology."""

    seed = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    flow_noise = rng.uniform(
        -FLOW_NOISE_HALF_WIDTH, FLOW_NOISE_HALF_WIDTH, topology.m
    )
    raw_flow = np.exp(
        RATING_LOG_COEFFICIENT * (topology.ratings.astype(float) - 3.0)
        + flow_noise
    )
    row_sums = np.bincount(
        topology.tails, weights=raw_flow, minlength=topology.n_left
    )
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ArithmeticError("MovieLens planted row normalization failed")
    x_star = ROW_BUDGET * raw_flow / row_sums[topology.tails]
    mu = np.power(
        10.0,
        rng.uniform(-MU_LOG10_HALF_WIDTH, MU_LOG10_HALF_WIDTH, topology.m),
    )
    q_star = mu / np.square(x_star)
    raw_y = rng.standard_normal(topology.n)
    raw_y -= raw_y[-1]
    heads_global = topology.heads_local + topology.n_left
    raw_difference = raw_y[topology.tails] - raw_y[heads_global]
    maximum_difference = float(np.max(np.abs(raw_difference)))
    if maximum_difference == 0.0:
        y_star = np.zeros(topology.n, dtype=float)
    else:
        y_star = raw_y * (
            POTENTIAL_TO_MIN_SLACK * float(np.min(q_star)) / maximum_difference
        )
        y_star -= y_star[-1]
    edge_difference = y_star[topology.tails] - y_star[heads_global]
    c = q_star + edge_difference
    left_beta = np.bincount(
        topology.tails, weights=x_star, minlength=topology.n_left
    )
    right_beta = np.bincount(
        topology.heads_local, weights=x_star, minlength=topology.n_right
    )
    beta = np.concatenate([left_beta, -right_beta])
    beta[-1] = -fsum(float(value) for value in beta[:-1])
    problem = ReciprocalTransportProblem(
        n_left=topology.n_left,
        n_right=topology.n_right,
        tails=topology.tails,
        heads=heads_global,
        beta=beta,
        c=c,
        mu=mu,
    )
    return PlantedInstance(
        problem=problem,
        x_star=_readonly(x_star, float),
        y_star=_readonly(y_star, float),
        q_star=_readonly(q_star, float),
        graph_family="movielens_1m_largest_component",
        graph_seed=seed,
    )


def make_movielens_warm_start(
    planted: PlantedInstance,
    seed: int,
) -> Array:
    """Return the frozen dual-feasible A39 warm start."""

    problem = planted.problem
    rng = np.random.Generator(np.random.PCG64(int(seed) + WARM_START_SEED_OFFSET))
    noise = rng.standard_normal(problem.n)
    noise -= noise[-1]
    difference = problem.edge_difference(noise)
    maximum_difference = float(np.max(np.abs(difference)))
    if maximum_difference > 0.0:
        noise *= (
            WARM_START_TO_MIN_SLACK
            * float(np.min(planted.q_star))
            / maximum_difference
        )
    else:
        noise.fill(0.0)
    result = np.asarray(planted.y_star, dtype=float) + noise
    result -= result[-1]
    return result


def movielens_topology_summary(topology: MovieLens1MTopology) -> dict[str, Any]:
    """Return aggregate-only topology fields safe for result artifacts."""

    left_degree = np.bincount(topology.tails, minlength=topology.n_left)
    right_degree = np.bincount(topology.heads_local, minlength=topology.n_right)

    def degree_fields(prefix: str, values: Array) -> dict[str, float | int]:
        quantiles = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0))
        return {
            f"{prefix}_degree_min": int(values.min()),
            f"{prefix}_degree_q25": float(quantiles[1]),
            f"{prefix}_degree_median": float(quantiles[2]),
            f"{prefix}_degree_q75": float(quantiles[3]),
            f"{prefix}_degree_q95": float(quantiles[4]),
            f"{prefix}_degree_max": int(values.max()),
        }

    return {
        "dataset_page": MOVIELENS_1M_PAGE,
        "dataset_archive_url": MOVIELENS_1M_URL,
        "dataset_checksum_url": MOVIELENS_1M_MD5_URL,
        "dataset_expected_md5": MOVIELENS_1M_MD5,
        "dataset_observed_md5": topology.archive_md5,
        "ratings_member": MOVIELENS_1M_RATINGS_MEMBER,
        "ratings_member_crc32": topology.member_crc32,
        "ratings_member_uncompressed_bytes": topology.member_uncompressed_bytes,
        "raw_rating_rows": topology.raw_rows,
        "duplicate_rows_removed": topology.duplicate_rows_removed,
        "component_count": topology.component_count,
        "dropped_edges": topology.dropped_edges,
        "dropped_users": topology.dropped_users,
        "dropped_movies": topology.dropped_movies,
        "n_left": topology.n_left,
        "n_right": topology.n_right,
        "n_vertices": topology.n,
        "edges": topology.m,
        "density": float(topology.m / (topology.n_left * topology.n_right)),
        "topology_sha256": topology.topology_sha256,
        **degree_fields("left", left_degree),
        **degree_fields("right", right_degree),
    }


def planted_kkt_summary(planted: PlantedInstance) -> dict[str, float]:
    """Return finite diagnostics for the frozen planted optimum."""

    problem = planted.problem
    slack = problem.dual_slack(planted.y_star)
    target = problem.mu / np.square(planted.x_star)
    stationarity = np.abs(slack - target) / np.maximum(1.0, np.abs(target))
    residual = problem.primal_residual(planted.x_star)
    return {
        "planted_objective": float(problem.objective(planted.x_star)),
        "planted_x_min": float(np.min(planted.x_star)),
        "planted_x_max": float(np.max(planted.x_star)),
        "planted_mu_min": float(np.min(problem.mu)),
        "planted_mu_max": float(np.max(problem.mu)),
        "planted_q_min": float(np.min(planted.q_star)),
        "planted_q_max": float(np.max(planted.q_star)),
        "planted_c_min": float(np.min(problem.c)),
        "planted_c_max": float(np.max(problem.c)),
        "planted_scaled_kkt_residual_max": float(np.max(stationarity)),
        "planted_binary64_marginal_residual_l1": float(np.sum(np.abs(residual))),
        "planted_left_budget_deviation_max": float(
            np.max(
                np.abs(
                    np.bincount(
                        problem.tails,
                        weights=planted.x_star,
                        minlength=problem.n_left,
                    )
                    - ROW_BUDGET
                )
            )
        ),
    }

