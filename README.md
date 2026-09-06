# CertQuota-V

**A Solver-Independent, Fixed-Topology Linear-Work Endpoint Certificate for Sparse Reciprocal Transportation**

Long Li · Independent Researcher · [lilong0228@gmail.com](mailto:lilong0228@gmail.com)

[Paper website](https://long-0228.github.io/certquota/) ·
[Preprint PDF](https://long-0228.github.io/certquota/assets/certquota-preprint.pdf) ·
[Dated release](https://github.com/long-0228/certquota/releases/tag/preprint-2026-09-06) ·
[Clean-install CI](https://github.com/long-0228/certquota/actions/workflows/clean-install.yml)

CertQuota-V checks a numerical candidate for
`min sum(c[e] * z[e] + mu[e] / z[e]), Bz = beta, z > 0`.
It either rejects or certifies positivity, exact feasibility of an implicit
exact-real recovered allocation, and an objective gap at most epsilon.
The certificate does not trust the candidate solver's status. Verification is
linear-work on precompiled, fixed-topology schedules; this is not a claim that
the full optimization pipeline is globally linear-time.

## Install and try

Python 3.10 or later:

```sh
git clone https://github.com/long-0228/certquota.git
cd certquota
python -m pip install .
python examples/solve_and_replay.py
```

The production code remains version 0.2.0. The dated preprint adds author
metadata, public documentation, portable installation tests, and independent
research checkers; it does not silently change numerical algorithms.

Only `result.status == "optimal"` authorizes a certificate. The materialized
binary64 vector is an approximation to the certified implicit real allocation,
not a literally exact-feasible vector. A separate rational decision format
provides explicit exact-feasible fractions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Evidence, with boundaries

- On 18 matched problems at the same saved accuracy targets: 18/18 CertQuota
  public cold pipelines and 18/18 same-SeDuMi-dual transfers pass; the
  experimental VSDP/Octave-interval port passes 15/18. All 18 port brackets
  are finite and independently witnessed. **This is not stock VSDP/INTLAB.**
- 114 primary proof objects were checked locally with Python's standard
  library, without importing CertQuota, NumPy, Arb, or VSDP. The public archive
  distributes the 90 synthetic-problem proofs; the 24 MovieLens-derived
  objects are excluded conservatively under the dataset redistribution terms.
- 12 historical-only MovieLens workloads yield independently checked explicit
  rational budget decisions. Costs and quotas are designed proxies on one
  dataset, not production latency or recommendation-quality measurements.
- The original difficult-start budget passes 26/36 valid starts. All 36 pass
  only after explicitly additional exploratory retry stages; 18 invalid
  controls are rejected.
- Rebuild-on-rejection matches 216/216 dynamic-tree cells with 66 builds,
  versus 216 builds for always-rebuild. These are correlated cells from
  nine graphs, not 216 independent datasets.

The full local development suite passed 404 tests on Windows. The public
installation check uses a clearly named 35-test core subset and three
independently rechecked 1024-edge examples. See actual CI runs for platform
results; a workflow definition is not evidence that a run passed.

## What is in this public snapshot?

This repository contains core source, a runnable example, the isolated-install
test subset, A45 independent checkers, and the paper source. The release also
provides a separately hash-manifested numerical witness archive. Original local
A42-A45 archives are preserved separately and are **not** silently rewritten by
publication. This curated snapshot is not the full historical experiment
environment and does not reproduce every timing panel with one command.

No raw MovieLens archive, INTLAB, third-party solver binaries, local virtual
environment, credentials, or machine-specific runtime logs are redistributed.
The paper is a preprint, not a claim of journal acceptance or completed
independent human proof review. An arXiv identifier will be linked only after
one has actually been assigned.

## License and disclosure

Original software is [BSD-3-Clause](LICENSE), copyright 2026 Long Li.
The manuscript and its figures remain copyright Long Li; making them available
here does not apply the software license to the paper. Unmodified Springer
LaTeX support files retain their own notices; see [THIRD_PARTY.md](THIRD_PARTY.md).
No specific external grant funded this research; the author declares no
competing interests. OpenAI Codex assisted implementation, experiment
orchestration, manuscript restructuring, and consistency checks. It is not an
author; Long Li is responsible for the work and the published text.

## Citation

```bibtex
@misc{li2026certquota,
  author = {Li, Long},
  title = {A Solver-Independent, Fixed-Topology Linear-Work Endpoint Certificate
           for Sparse Reciprocal Transportation},
  year = {2026},
  howpublished = {Preprint},
  url = {https://long-0228.github.io/certquota/}
}
```
