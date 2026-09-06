"""Portable, self-contained replay bundles for endpoint certificates.

The in-process :class:`~certquota.intervals.IntervalRecoveryCertificate` is a
capability-bound summary and is intentionally not serializable as authority.
This module instead serializes the exact binary64 problem, candidate, tree, and
tolerance inputs.  A fresh process can reconstruct those inputs and rerun the
authoritative strict backend.  The newly produced in-process certificate is the
authority; the serialized summary is only a comparison aid.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .problem import ReciprocalTransportProblem
from .strict_backend import (
    prepare_strict_recovery_state,
    strict_recovery_certificate,
)
from .tree import TreeRouter


SCHEMA_VERSION = "certquota-endpoint-replay-v1"
_MAX_ARRAY_BYTES = 8 * 1024**3


class ReplayBundleError(ValueError):
    """Raised when a replay bundle is malformed or fails its content binding."""


@dataclass(frozen=True)
class ReplayVerification:
    """Result of independently replaying one portable endpoint bundle."""

    accepted: bool
    claim_matches: bool
    backend: str
    decimal_precision: int
    objective_gap_upper: float
    residual_l1_upper: float
    x_min_lower: float
    positivity_certified: bool
    epsilon_optimal_certified: bool
    epsilon: float
    problem_fingerprint: str
    router_fingerprint: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible public summary."""

        def finite_or_none(value: float) -> float | None:
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None

        return {
            "schema": SCHEMA_VERSION,
            "accepted": self.accepted,
            "claim_matches": self.claim_matches,
            "backend": self.backend,
            "decimal_precision": self.decimal_precision,
            "objective_gap_upper": finite_or_none(self.objective_gap_upper),
            "residual_l1_upper": finite_or_none(self.residual_l1_upper),
            "x_min_lower": finite_or_none(self.x_min_lower),
            "positivity_certified": self.positivity_certified,
            "epsilon_optimal_certified": self.epsilon_optimal_certified,
            "epsilon": self.epsilon,
            "problem_fingerprint": self.problem_fingerprint,
            "router_fingerprint": self.router_fingerprint,
            "payload_sha256": self.payload_sha256,
        }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _encode_array(values: Any, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    if array.ndim != 1:
        raise ReplayBundleError("replay arrays must be one-dimensional")
    if np.issubdtype(array.dtype, np.floating) and not np.all(
        np.isfinite(array)
    ):
        raise ReplayBundleError("replay arrays must contain only finite values")
    return {
        "dtype": dtype,
        "length": int(array.size),
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode(
            "ascii"
        ),
    }


def _decode_array(
    value: Any,
    *,
    dtype: str,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ReplayBundleError(f"{name} must be an encoded array")
    if set(value) != {"dtype", "length", "data_base64"}:
        raise ReplayBundleError(f"{name} has unexpected encoded-array fields")
    if value.get("dtype") != dtype:
        raise ReplayBundleError(f"{name} has the wrong dtype")
    length = value.get("length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ReplayBundleError(f"{name} has an invalid length")
    if expected_length is not None and length != expected_length:
        raise ReplayBundleError(f"{name} has the wrong length")
    itemsize = np.dtype(dtype).itemsize
    byte_count = length * itemsize
    if byte_count > _MAX_ARRAY_BYTES:
        raise ReplayBundleError(f"{name} exceeds the replay size limit")
    encoded = value.get("data_base64")
    if not isinstance(encoded, str):
        raise ReplayBundleError(f"{name} data must be base64 text")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ReplayBundleError(f"{name} is not valid base64") from error
    if len(raw) != byte_count:
        raise ReplayBundleError(f"{name} byte count disagrees with its length")
    array = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()
    if np.issubdtype(array.dtype, np.floating) and not np.all(
        np.isfinite(array)
    ):
        raise ReplayBundleError(f"{name} contains a nonfinite value")
    return array


def _float_hex(value: Any, name: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ReplayBundleError(f"{name} must be finite")
    return numeric.hex()


def _parse_float_hex(value: Any, name: str) -> float:
    if not isinstance(value, str):
        raise ReplayBundleError(f"{name} must be a hexadecimal float string")
    try:
        numeric = float.fromhex(value)
    except ValueError as error:
        raise ReplayBundleError(f"{name} is not a hexadecimal float") from error
    if not math.isfinite(numeric):
        raise ReplayBundleError(f"{name} must be finite")
    return numeric


def _backend_key(certificate_backend: str) -> str:
    if certificate_backend.startswith("ieee754-"):
        return "ieee754"
    if certificate_backend == "python-flint-arb":
        return "arb"
    raise ReplayBundleError(
        f"unsupported certificate backend {certificate_backend!r}"
    )


def replay_bundle(result: Any) -> dict[str, Any]:
    """Create a content-bound portable replay document from an optimal result.

    The returned mapping contains no trusted serialized capability.  It contains
    all exact binary64 inputs needed for a fresh authoritative replay.
    """

    problem = getattr(result, "problem", None)
    router = getattr(result, "router", None)
    certificate = getattr(result, "strict_certificate", None)
    y = np.asarray(getattr(result, "y", None), dtype=float)
    epsilon = float(getattr(result, "requested_epsilon", math.nan))
    if (
        getattr(result, "status", None) != "optimal"
        or not isinstance(problem, ReciprocalTransportProblem)
        or not isinstance(router, TreeRouter)
        or certificate is None
        or y.shape != (problem.n,)
        or not np.all(np.isfinite(y))
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not certificate.is_bound_to(problem, y, epsilon, router)
        or not certificate.positivity_certified
        or not certificate.epsilon_optimal_certified
        or not math.isfinite(float(certificate.objective_gap_upper))
        or float(certificate.objective_gap_upper) > epsilon
    ):
        raise ReplayBundleError(
            "only an accepted, input-bound VerifiedResult can be exported"
        )
    problem.validate()
    router.validate_for(problem)
    declared_beta = np.array(problem.beta, dtype=float, copy=True)
    declared_beta[-1] = problem.declared_root_beta
    backend_key = _backend_key(str(certificate.backend))
    payload = {
        "schema": SCHEMA_VERSION,
        "problem": {
            "n_left": int(problem.n_left),
            "n_right": int(problem.n_right),
            "tails": _encode_array(problem.tails, "<i8"),
            "heads": _encode_array(problem.heads, "<i8"),
            "declared_beta": _encode_array(declared_beta, "<f8"),
            "c": _encode_array(problem.c, "<f8"),
            "mu": _encode_array(problem.mu, "<f8"),
            "content_fingerprint": problem.content_fingerprint,
        },
        "candidate_y": _encode_array(y, "<f8"),
        "tree": {
            "root": int(router.root),
            "tree_edges": _encode_array(router.tree_edges, "<i8"),
            "structure_fingerprint": router.structure_fingerprint,
        },
        "epsilon_hex": _float_hex(epsilon, "epsilon"),
        "replay": {
            "backend": backend_key,
            "decimal_precision": int(certificate.decimal_precision),
        },
        "claimed_certificate": {
            "authorized": True,
            "backend": str(certificate.backend),
            "objective_gap_upper_hex": _float_hex(
                certificate.objective_gap_upper, "objective_gap_upper"
            ),
            "residual_l1_upper_hex": _float_hex(
                certificate.residual_l1_upper, "residual_l1_upper"
            ),
            "x_min_lower_hex": _float_hex(
                certificate.x_min_lower, "x_min_lower"
            ),
            "problem_fingerprint": certificate.problem_fingerprint,
            "router_fingerprint": certificate.router_fingerprint,
            "iterate_fingerprint": certificate.iterate_fingerprint,
            "epsilon_fingerprint": certificate.epsilon_fingerprint,
        },
    }
    digest = _payload_digest(payload)
    return {"payload": payload, "payload_sha256": digest}


def write_replay_bundle(result: Any, destination: str | Path) -> Path:
    """Write one accepted result as canonical strict JSON."""

    path = Path(destination)
    document = replay_bundle(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(document) + "\n", encoding="ascii")
    return path


def _read_document(source: str | bytes | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, bytes):
        try:
            decoded = source.decode("ascii")
        except UnicodeDecodeError as error:
            raise ReplayBundleError("bundle bytes must be ASCII JSON") from error
    elif isinstance(source, Path):
        decoded = source.read_text(encoding="ascii")
    elif isinstance(source, str):
        stripped = source.lstrip()
        decoded = source if stripped.startswith("{") else Path(source).read_text(
            encoding="ascii"
        )
    else:
        raise TypeError("source must be a path, JSON text/bytes, or mapping")
    try:
        document = json.loads(decoded)
    except json.JSONDecodeError as error:
        raise ReplayBundleError("bundle is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise ReplayBundleError("bundle top level must be an object")
    return document


def verify_replay_bundle(
    source: str | bytes | Path | Mapping[str, Any],
    *,
    backend: str | None = None,
    decimal_precision: int | None = None,
) -> ReplayVerification:
    """Reconstruct and authoritatively replay a portable endpoint bundle."""

    document = _read_document(source)
    if set(document) != {"payload", "payload_sha256"}:
        raise ReplayBundleError("bundle has unexpected top-level fields")
    payload = document.get("payload")
    expected_digest = document.get("payload_sha256")
    if not isinstance(payload, Mapping) or not isinstance(expected_digest, str):
        raise ReplayBundleError("bundle payload or digest has the wrong type")
    actual_digest = _payload_digest(payload)
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ReplayBundleError("bundle payload hash mismatch")
    if payload.get("schema") != SCHEMA_VERSION:
        raise ReplayBundleError("unsupported replay-bundle schema")

    problem_data = payload.get("problem")
    tree_data = payload.get("tree")
    replay_data = payload.get("replay")
    claim = payload.get("claimed_certificate")
    if not all(
        isinstance(value, Mapping)
        for value in (problem_data, tree_data, replay_data, claim)
    ):
        raise ReplayBundleError("bundle sections must be objects")
    assert isinstance(problem_data, Mapping)
    assert isinstance(tree_data, Mapping)
    assert isinstance(replay_data, Mapping)
    assert isinstance(claim, Mapping)

    n_left = problem_data.get("n_left")
    n_right = problem_data.get("n_right")
    if (
        isinstance(n_left, bool)
        or isinstance(n_right, bool)
        or not isinstance(n_left, int)
        or not isinstance(n_right, int)
        or n_left <= 0
        or n_right <= 0
    ):
        raise ReplayBundleError("problem dimensions must be positive integers")
    n = n_left + n_right
    tails = _decode_array(problem_data.get("tails"), dtype="<i8", name="tails")
    m = int(tails.size)
    heads = _decode_array(
        problem_data.get("heads"), dtype="<i8", name="heads", expected_length=m
    )
    beta = _decode_array(
        problem_data.get("declared_beta"),
        dtype="<f8",
        name="declared_beta",
        expected_length=n,
    )
    c = _decode_array(
        problem_data.get("c"), dtype="<f8", name="c", expected_length=m
    )
    mu = _decode_array(
        problem_data.get("mu"), dtype="<f8", name="mu", expected_length=m
    )
    y = _decode_array(
        payload.get("candidate_y"),
        dtype="<f8",
        name="candidate_y",
        expected_length=n,
    )
    tree_edges = _decode_array(
        tree_data.get("tree_edges"),
        dtype="<i8",
        name="tree_edges",
        expected_length=n - 1,
    )
    root = tree_data.get("root")
    if isinstance(root, bool) or not isinstance(root, int):
        raise ReplayBundleError("tree root must be an integer")
    epsilon = _parse_float_hex(payload.get("epsilon_hex"), "epsilon_hex")
    if epsilon <= 0.0:
        raise ReplayBundleError("epsilon must be positive")

    problem = ReciprocalTransportProblem(
        n_left=n_left,
        n_right=n_right,
        tails=tails,
        heads=heads,
        beta=beta,
        c=c,
        mu=mu,
    )
    expected_problem_fingerprint = problem_data.get("content_fingerprint")
    if (
        not isinstance(expected_problem_fingerprint, str)
        or problem.content_fingerprint != expected_problem_fingerprint
    ):
        raise ReplayBundleError("problem fingerprint mismatch")
    router = TreeRouter.build(problem, tree_edges=tree_edges, root=root)
    expected_router_fingerprint = tree_data.get("structure_fingerprint")
    if (
        not isinstance(expected_router_fingerprint, str)
        or router.structure_fingerprint != expected_router_fingerprint
    ):
        raise ReplayBundleError("tree-router fingerprint mismatch")

    selected_backend = replay_data.get("backend") if backend is None else backend
    if selected_backend not in {"ieee754", "arb"}:
        raise ReplayBundleError("replay backend must be 'ieee754' or 'arb'")
    stored_precision = replay_data.get("decimal_precision")
    selected_precision = stored_precision if decimal_precision is None else decimal_precision
    if (
        isinstance(selected_precision, bool)
        or not isinstance(selected_precision, int)
        or selected_precision < 16
    ):
        raise ReplayBundleError("decimal precision must be an integer at least 16")

    prepared = prepare_strict_recovery_state(
        problem,
        y,
        selected_backend,
        selected_precision,
        problem,
    )
    certificate = strict_recovery_certificate(
        problem,
        y,
        epsilon,
        selected_backend,
        selected_precision,
        prepared,
        router,
    )
    accepted = bool(
        certificate.positivity_certified
        and certificate.epsilon_optimal_certified
        and math.isfinite(float(certificate.objective_gap_upper))
        and float(certificate.objective_gap_upper) <= epsilon
        and certificate.is_bound_to(problem, y, epsilon, router)
    )
    claim_matches = bool(
        claim.get("authorized") is accepted
        and claim.get("backend") == str(certificate.backend)
        and claim.get("objective_gap_upper_hex")
        == _float_hex(certificate.objective_gap_upper, "objective_gap_upper")
        and claim.get("residual_l1_upper_hex")
        == _float_hex(certificate.residual_l1_upper, "residual_l1_upper")
        and claim.get("x_min_lower_hex")
        == _float_hex(certificate.x_min_lower, "x_min_lower")
        and claim.get("problem_fingerprint") == problem.content_fingerprint
        and claim.get("router_fingerprint") == router.structure_fingerprint
        and claim.get("iterate_fingerprint") == certificate.iterate_fingerprint
        and claim.get("epsilon_fingerprint") == certificate.epsilon_fingerprint
    )
    return ReplayVerification(
        accepted=accepted,
        claim_matches=claim_matches,
        backend=str(certificate.backend),
        decimal_precision=int(certificate.decimal_precision),
        objective_gap_upper=float(certificate.objective_gap_upper),
        residual_l1_upper=float(certificate.residual_l1_upper),
        x_min_lower=float(certificate.x_min_lower),
        positivity_certified=bool(certificate.positivity_certified),
        epsilon_optimal_certified=bool(certificate.epsilon_optimal_certified),
        epsilon=epsilon,
        problem_fingerprint=problem.content_fingerprint,
        router_fingerprint=router.structure_fingerprint,
        payload_sha256=actual_digest,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for independent replay verification."""

    parser = argparse.ArgumentParser(
        description="Re-run a CertQuota endpoint replay bundle"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--backend", choices=("ieee754", "arb"))
    parser.add_argument("--decimal-precision", type=int)
    args = parser.parse_args(argv)
    try:
        result = verify_replay_bundle(
            args.bundle,
            backend=args.backend,
            decimal_precision=args.decimal_precision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "status": "invalid_bundle",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0 if result.accepted and result.claim_matches else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
