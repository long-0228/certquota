"""Stable public API for certified reciprocal transport solvers."""

from importlib.metadata import PackageNotFoundError, version as _package_version


try:
    __version__ = _package_version("certquota")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0.2.0"

from .certificates import (
    BasinCertificate,
    DualState,
    ForcingCertificate,
    basin_certificate,
    dual_state,
    forcing_certificate,
    recover_primal,
)
from .instances import (
    make_client_regular_instance,
    make_graph_family_instance,
    make_planted_instance,
    perturb_planted_instance,
    perturb_dynamic_instance,
)
from .hybrid import HybridOptions, HybridResult, solve_hybrid
from .intervals import (
    IntervalBasinCertificate,
    IntervalForcingCertificate,
    IntervalEtaReference,
    IntervalRecoveryCertificate,
    interval_basin_certificate,
    interval_backend_available,
    interval_forcing_certificate,
    interval_exact_eta_reference,
    interval_recovery_certificate,
)
from .problem import ReciprocalTransportProblem
from .replay import (
    ReplayBundleError,
    ReplayVerification,
    replay_bundle,
    verify_replay_bundle,
    write_replay_bundle,
)
from .sampling import (
    IndependentCategoricalSampler,
    exact_conditional_mse_by_model,
    ht_model_estimate,
    marginal_residuals,
    maximum_entropy_allocation,
    validate_expected_quota_allocation,
)
from .solver import CertifiedNewtonOptions, CertifiedNewtonResult, solve_certified_newton
from .verified import VerifiedAttempt, VerifiedOptions, VerifiedResult, solve_verified

__all__ = [
    "__version__",
    "BasinCertificate",
    "CertifiedNewtonOptions",
    "CertifiedNewtonResult",
    "DualState",
    "ForcingCertificate",
    "HybridOptions",
    "HybridResult",
    "VerifiedAttempt",
    "VerifiedOptions",
    "VerifiedResult",
    "IntervalBasinCertificate",
    "IntervalForcingCertificate",
    "IntervalEtaReference",
    "IntervalRecoveryCertificate",
    "ReciprocalTransportProblem",
    "ReplayBundleError",
    "ReplayVerification",
    "IndependentCategoricalSampler",
    "exact_conditional_mse_by_model",
    "ht_model_estimate",
    "marginal_residuals",
    "maximum_entropy_allocation",
    "validate_expected_quota_allocation",
    "basin_certificate",
    "dual_state",
    "forcing_certificate",
    "make_client_regular_instance",
    "make_graph_family_instance",
    "make_planted_instance",
    "interval_basin_certificate",
    "interval_backend_available",
    "interval_forcing_certificate",
    "interval_exact_eta_reference",
    "interval_recovery_certificate",
    "perturb_planted_instance",
    "perturb_dynamic_instance",
    "recover_primal",
    "replay_bundle",
    "solve_certified_newton",
    "solve_hybrid",
    "solve_verified",
    "verify_replay_bundle",
    "write_replay_bundle",
]
