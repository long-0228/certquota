from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from threading import get_ident

import numpy as np

import certquota.ieee_intervals as ieee
from certquota.certificates import dual_state
from certquota.instances import make_planted_instance
from certquota.tree import TreeRouter


def test_platform_probe_reports_every_required_category() -> None:
    probe = ieee.ieee_platform_probe()

    assert probe.binary64_layout
    assert probe.nextafter_adjacency
    assert probe.round_to_nearest_ties_to_even
    assert probe.gradual_underflow
    assert probe.basic_operation_spots
    assert probe.sqrt_rounding_spots
    assert probe.passed
    assert probe.failed_checks == ()
    assert ieee.ieee_platform_contract()


def test_failed_cold_probe_is_not_cached_and_can_retry(monkeypatch) -> None:
    good = ieee.ieee_platform_probe()
    failed = replace(good, binary64_layout=False)
    responses = iter([failed, good])
    monkeypatch.setattr(ieee, "_PLATFORM_PROBE_CACHE", None)
    monkeypatch.setattr(ieee, "ieee_platform_probe", lambda: next(responses))
    monkeypatch.setattr(ieee, "_probe_active_fp_controls", lambda: True)

    assert not ieee.ieee_platform_contract()
    assert ieee._PLATFORM_PROBE_CACHE is None

    assert ieee.ieee_platform_contract()
    assert ieee._PLATFORM_PROBE_CACHE is good


def test_platform_probe_rejects_broken_nextafter(monkeypatch) -> None:
    def stuck_nextafter(value, direction):
        del direction
        return np.asarray(value, dtype=np.float64)

    monkeypatch.setattr(ieee.np, "nextafter", stuck_nextafter)

    probe = ieee.ieee_platform_probe()
    assert not probe.nextafter_adjacency
    assert "nextafter_adjacency" in probe.failed_checks
    assert not probe.passed
    assert not ieee.ieee_platform_contract()


def test_platform_probe_rejects_non_even_tie_rounding(monkeypatch) -> None:
    real_add = ieee.np.add
    half_ulp = np.float64(float.fromhex("0x1.0000000000000p-53"))
    one_up = np.float64(float.fromhex("0x1.0000000000001p+0"))

    def ties_up_add(left, right):
        result = np.asarray(real_add(left, right), dtype=np.float64)
        mask = (np.asarray(left) == 1.0) & (np.asarray(right) == half_ulp)
        return np.where(mask, one_up, result)

    monkeypatch.setattr(ieee.np, "add", ties_up_add)

    probe = ieee.ieee_platform_probe()
    assert not probe.round_to_nearest_ties_to_even
    assert probe.gradual_underflow
    assert probe.basic_operation_spots
    assert not probe.passed
    assert not ieee.ieee_platform_contract()


def test_platform_probe_rejects_flush_to_zero(monkeypatch) -> None:
    real_multiply = ieee.np.multiply
    minimum_normal = np.float64(float.fromhex("0x1.0000000000000p-1022"))

    def flushing_multiply(left, right):
        result = np.asarray(real_multiply(left, right), dtype=np.float64)
        subnormal = (np.abs(result) < minimum_normal) & (result != 0.0)
        signed_zero = np.copysign(np.zeros_like(result), result)
        return np.where(subnormal, signed_zero, result)

    monkeypatch.setattr(ieee.np, "multiply", flushing_multiply)

    probe = ieee.ieee_platform_probe()
    assert not probe.gradual_underflow
    assert probe.basic_operation_spots
    assert "gradual_underflow" in probe.failed_checks
    assert not probe.passed
    assert not ieee.ieee_platform_contract()


def test_platform_probe_rejects_bad_sqrt_spot(monkeypatch) -> None:
    real_sqrt = ieee.np.sqrt
    real_nextafter = ieee.np.nextafter

    def perturbed_sqrt(value):
        values = np.asarray(value, dtype=np.float64)
        result = np.asarray(real_sqrt(values), dtype=np.float64)
        perturbed = real_nextafter(result, np.float64(np.inf))
        return np.where(values == 2.0, perturbed, result)

    monkeypatch.setattr(ieee.np, "sqrt", perturbed_sqrt)

    probe = ieee.ieee_platform_probe()
    assert probe.gradual_underflow
    assert not probe.sqrt_rounding_spots
    assert "sqrt_rounding_spots" in probe.failed_checks
    assert not probe.passed
    assert not ieee.ieee_platform_contract()


def test_individual_probe_exceptions_fail_closed(monkeypatch) -> None:
    def exploding_nextafter(value, direction):
        del value, direction
        raise RuntimeError("simulated unsupported primitive")

    monkeypatch.setattr(ieee.np, "nextafter", exploding_nextafter)

    probe = ieee.ieee_platform_probe()
    assert not probe.nextafter_adjacency
    assert not probe.passed


def _prepared_certificate_case():
    planted = make_planted_instance(5, 4, density=0.65, seed=8812)
    problem = planted.problem
    y = planted.y_star.copy()
    state = dual_state(problem, y)
    router = TreeRouter.build(problem, edge_cost=1.0 / state.conductance)
    prepared = ieee.prepare_ieee_state(problem, y)
    assert prepared.domain_certified
    assert prepared.recovery_certified
    return problem, y, router, prepared


def test_prepared_state_rechecks_changed_active_controls(monkeypatch) -> None:
    problem, y, router, prepared = _prepared_certificate_case()
    monkeypatch.setattr(ieee, "_probe_active_fp_controls", lambda: False)

    basin = ieee.ieee_basin_certificate(
        problem, y, router, prepared_state=prepared
    )
    forcing = ieee.ieee_forcing_certificate(
        problem,
        y,
        np.zeros(problem.n),
        router,
        prepared_state=prepared,
    )
    recovery = ieee.ieee_recovery_certificate(
        problem,
        y,
        1.0,
        prepared_state=prepared,
        router=router,
    )

    assert not basin.domain_certified
    assert not basin.passed
    assert np.isinf(basin.eta_upper)
    assert not forcing.passed
    assert np.isinf(forcing.residual_upper)
    assert not recovery.positivity_certified
    assert not recovery.epsilon_optimal_certified
    assert np.isinf(recovery.objective_gap_upper)


def test_prepared_state_rechecks_worker_thread_controls(monkeypatch) -> None:
    main_thread = get_ident()
    monkeypatch.setattr(
        ieee,
        "_probe_active_fp_controls",
        lambda: get_ident() == main_thread,
    )
    problem, y, router, prepared = _prepared_certificate_case()

    def certify_on_worker():
        return ieee.ieee_basin_certificate(
            problem, y, router, prepared_state=prepared
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        certificate = executor.submit(certify_on_worker).result()

    assert not certificate.domain_certified
    assert not certificate.passed
    assert np.isinf(certificate.eta_upper)


def test_outward_primitives_enclose_boundary_exact_values() -> None:
    minimum_subnormal = np.float64(
        float.fromhex("0x0.0000000000001p-1022")
    )
    minimum_normal = np.float64(
        float.fromhex("0x1.0000000000000p-1022")
    )
    one_up = np.float64(float.fromhex("0x1.0000000000001p+0"))
    maximum_finite = np.float64(
        float.fromhex("0x1.fffffffffffffp+1023")
    )

    multiply_left = np.asarray(
        [minimum_subnormal, minimum_subnormal, minimum_normal, one_up]
    )
    multiply_right = np.asarray([0.5, 0.75, 0.5, one_up])
    multiply_lower = ieee._positive_multiply_lower(
        multiply_left, multiply_right
    )
    multiply_upper = ieee._positive_multiply_upper(
        multiply_left, multiply_right
    )
    for left, right, lower, upper in zip(
        multiply_left, multiply_right, multiply_lower, multiply_upper
    ):
        exact = Fraction.from_float(float(left)) * Fraction.from_float(
            float(right)
        )
        assert Fraction.from_float(float(lower)) <= exact
        assert exact <= Fraction.from_float(float(upper))

    divide_numerator = np.asarray(
        [minimum_subnormal, minimum_normal, 1.0, maximum_finite]
    )
    divide_denominator = np.asarray([2.0, 3.0, 10.0, 2.0])
    divide_lower = ieee._positive_divide_lower(
        divide_numerator, divide_denominator
    )
    divide_upper = ieee._positive_divide_upper(
        divide_numerator, divide_denominator
    )
    for numerator, denominator, lower, upper in zip(
        divide_numerator, divide_denominator, divide_lower, divide_upper
    ):
        exact = Fraction.from_float(float(numerator)) / Fraction.from_float(
            float(denominator)
        )
        assert Fraction.from_float(float(lower)) <= exact
        assert exact <= Fraction.from_float(float(upper))

    sqrt_values = np.asarray(
        [minimum_subnormal, minimum_normal, one_up, 2.0, maximum_finite]
    )
    sqrt_lower = ieee._sqrt_lower(sqrt_values)
    sqrt_upper = ieee._sqrt_upper(sqrt_values)
    for value, lower, upper in zip(sqrt_values, sqrt_lower, sqrt_upper):
        exact_input = Fraction.from_float(float(value))
        assert Fraction.from_float(float(lower)) ** 2 <= exact_input
        assert exact_input <= Fraction.from_float(float(upper)) ** 2
