"""Portable endpoint replay bundle coverage."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from certquota import (
    ReplayBundleError,
    ReplayVerification,
    VerifiedOptions,
    make_planted_instance,
    replay_bundle,
    solve_verified,
    verify_replay_bundle,
    write_replay_bundle,
)


def _accepted_result():
    planted = make_planted_instance(
        n_left=8,
        n_right=4,
        density=0.75,
        seed=43001,
        mu_log_range=2.0,
        mean_client_budget=0.2,
    )
    return solve_verified(
        planted.problem,
        y0=planted.y_star,
        options=VerifiedOptions(epsilon=1e-8, strict_backend="ieee754"),
    )


def test_portable_bundle_replays_in_fresh_objects(tmp_path) -> None:
    result = _accepted_result()
    assert result.status == "optimal"
    target = write_replay_bundle(result, tmp_path / "endpoint.json")

    replayed = verify_replay_bundle(target)

    assert replayed.accepted
    assert replayed.claim_matches
    assert replayed.positivity_certified
    assert replayed.epsilon_optimal_certified
    assert replayed.objective_gap_upper <= replayed.epsilon
    assert replayed.problem_fingerprint == result.problem.content_fingerprint
    assert replayed.router_fingerprint == result.router.structure_fingerprint


def test_bundle_preserves_declared_root_and_canonical_json(tmp_path) -> None:
    result = _accepted_result()
    first = write_replay_bundle(result, tmp_path / "first.json")
    second = write_replay_bundle(result, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="ascii"))
    assert document == replay_bundle(result)


def test_payload_tampering_is_rejected_before_replay() -> None:
    document = replay_bundle(_accepted_result())
    document["payload"]["epsilon_hex"] = float(1e-2).hex()

    with pytest.raises(ReplayBundleError, match="payload hash mismatch"):
        verify_replay_bundle(document)


def test_unaccepted_result_cannot_be_exported() -> None:
    failed = replace(_accepted_result(), status="failed")
    with pytest.raises(ReplayBundleError, match="only an accepted"):
        replay_bundle(failed)


def test_rejected_replay_summary_remains_strict_json() -> None:
    verification = ReplayVerification(
        accepted=False,
        claim_matches=False,
        backend="ieee754-directed",
        decimal_precision=70,
        objective_gap_upper=math.inf,
        residual_l1_upper=math.inf,
        x_min_lower=-math.inf,
        positivity_certified=False,
        epsilon_optimal_certified=False,
        epsilon=1e-8,
        problem_fingerprint="0" * 64,
        router_fingerprint="1" * 64,
        payload_sha256="2" * 64,
    )

    encoded = json.dumps(verification.to_dict(), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["objective_gap_upper"] is None
    assert decoded["residual_l1_upper"] is None
    assert decoded["x_min_lower"] is None
