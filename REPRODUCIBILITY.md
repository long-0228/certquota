# Reproducing the public preprint snapshot

This publication snapshot has a deliberately narrower executable scope than
the complete historical local reviewer bundle. Numerical code under `src/` and
the A45 independent checker files are copied byte-for-byte. Public package
metadata, documentation, and author-identified manuscript files are later
publication changes.

## Isolated wheel verification (Windows or Linux)

```sh
python -m pip install build
python research/a45/clean_wheel_check.py --output results/new-clean
```

Use a new output path. The script builds a wheel, creates a fresh virtual
environment, pins NumPy 1.26.4 / SciPy 1.15.3 / pytest 8.4.2, installs the wheel
non-editably, and runs 35 core tests outside the checkout's import path. It
generates and independently checks three 1024-edge certificates. Public CI uses
Python 3.11 on Ubuntu 24.04 and Windows 2022. It is a portability smoke check,
**not** a full rerun of the historical Windows timing measurements or all
404 historical research tests.

Results, platform details, wheel hash, and logs are uploaded as CI artifacts.
An IEEE platform probe is a runtime safeguard, not a formal proof of all
arithmetic on every CPU.

## Standalone proof verification (no numerical dependencies)

Download `certquota-a45-public-evidence.zip` from the dated release and extract
it into a new directory. From that directory run:

```sh
python -I -S verify_public_evidence.py
```

The runner first verifies the manifest, then rechecks 90 synthetic primary proof
objects with integer-directed and exact-rational arithmetic. The 18 VSDP-port
brackets are all mathematically witnessed but only 15 meet the shared epsilon;
the runner expects and distinguishes these outcomes. It never treats a finite
bracket as matched-accuracy acceptance. The checkers have a 200,000-edge audit
limit and are not the production linear-work implementation.

The 24 MovieLens-derived endpoint/rational objects from the 114-object local
audit are not redistributed, because the dataset terms require separate
permission to redistribute data. Local availability of derived arrays is not
treated as permission to publish them. The rational checker itself is included.

The archive contains the original proof JSON bytes and a mapping back to
their original relative paths and SHA-256 hashes. It omits host logs and
runtime installations. It is independently recheckable evidence, not a
complete recipe for regenerating VSDP candidates. The historical external
baseline used an experimental Octave interval compatibility port with dense
paths, not the standard VSDP/INTLAB stack; do not interpret its time or memory
as standard VSDP performance.

## Exactness contract

Inputs are represented binary64 arrays. The independent marginal inputs are
the first `n-1` values; the last is their negative **exact** sum. The stored
binary64 root value is diagnostic. Production replay certifies an implicit
exact-real tree-recovered allocation. Its returned binary64 vector has a
measured residual and is not literally exact-feasible. Rational decision
objects serialize numerator/denominator pairs, have their own direct
feasibility and objective checks, and need not equal the implicit allocation.
Converting these fractions to binary64 loses literal exact feasibility.

## Paper build

In `paper/`, run `pdflatex main`, `bibtex main`, then `pdflatex main` twice.
All TeX inputs and eight figures are included, with no parent-directory input
paths. The release's `certquota-arxiv-source.tar.gz` has the same portable
source tree, with `main.tex` at the archive root. It is prepared for arXiv's
own processor; successful local compilation is not a substitute for checking
the arXiv preview.

## Scope and data

The 12 application cells use one historical MovieLens-1M dataset with designed
cost and quota proxies, not measured end-user benefits. GroupLens distributes
the original dataset under its own terms at
https://grouplens.org/datasets/movielens/1m/ . It is not bundled here.
No independent human proof-review completion or journal acceptance is claimed.
The local frozen A45 audit predates this public snapshot, including any later
CI run, and is not overwritten to retroactively claim a release existed.
