"""Portable integer/rational verifier. Python standard library only.

This intentionally does NOT import certquota or reproduce its stable gap
identity. Endpoint verification directly encloses P(X) and D(y). A VSDP-port
proof is checked using exact conic dual inequalities and a rational primal
flow, without trusting an interval linear solve or an optimizer status.
"""
from __future__ import annotations

import argparse
import base64
from fractions import Fraction as F
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import time

BITS = 160
SCALE = 1 << BITS
MAX_EDGES = 200_000  # audit tool, not the production linear-time verifier


def ceildiv(a, b):
    return -((-a) // b)


class I:
    """Closed dyadic enclosure [lo,hi]/2**BITS, all operations integer-directed."""
    __slots__ = ("lo", "hi")

    def __init__(self, value=0, *, bounds=None):
        if bounds is not None:
            self.lo, self.hi = bounds
        else:
            v = F(value)
            self.lo = (v.numerator * SCALE) // v.denominator
            self.hi = ceildiv(v.numerator * SCALE, v.denominator)
        if self.lo > self.hi:
            raise ValueError("empty enclosure")

    def __add__(self, other):
        other = other if isinstance(other, I) else I(other)
        return I(bounds=(self.lo + other.lo, self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(bounds=(-self.hi, -self.lo))

    def __sub__(self, other):
        return self + (-other if isinstance(other, I) else -I(other))

    def __mul__(self, other):
        other = other if isinstance(other, I) else I(other)
        terms = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return I(bounds=(min(terms) // SCALE, ceildiv(max(terms), SCALE)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, I) else I(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("division enclosure contains zero")
        terms = [F(a * SCALE, b) for a in (self.lo, self.hi)
                 for b in (other.lo, other.hi)]
        low, high = min(terms), max(terms)
        return I(bounds=(low.numerator // low.denominator,
                         ceildiv(high.numerator, high.denominator)))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("negative square root enclosure")
        low = math.isqrt(self.lo * SCALE)
        high = math.isqrt(self.hi * SCALE)
        if high * high < self.hi * SCALE:
            high += 1
        return I(bounds=(low, high))


def lower_float(value: F) -> float:
    out = float(value)
    return math.nextafter(out, -math.inf) if F(out) > value else out


def upper_float(value: F) -> float:
    out = float(value)
    return math.nextafter(out, math.inf) if F(out) < value else out


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def encode(values, integer=False):
    """Producer convenience; decoding/checking never relies on a producer hash."""
    values = list(values)
    return {"dtype": "<i8" if integer else "<f8", "length": len(values),
            "data_base64": base64.b64encode(struct.pack(
                "<" + ("q" if integer else "d") * len(values), *values)).decode("ascii")}


def decode(value, length=None, integer=False):
    if not isinstance(value, dict) or set(value) != {"dtype", "length", "data_base64"}:
        raise ValueError("malformed encoded array")
    size = value["length"]
    if type(size) is not int or not 0 <= size <= 6 * MAX_EDGES:
        raise ValueError("array size outside audit limit")
    if length is not None and size != length:
        raise ValueError("array length mismatch")
    if value["dtype"] != ("<i8" if integer else "<f8"):
        raise ValueError("array dtype mismatch")
    raw = base64.b64decode(value["data_base64"], validate=True)
    if len(raw) != 8 * size:
        raise ValueError("array byte length mismatch")
    out = list(struct.unpack("<" + ("q" if integer else "d") * size, raw))
    if not integer and not all(math.isfinite(x) for x in out):
        raise ValueError("nonfinite array")
    return out


def read_problem(p):
    nl, nr = p["n_left"], p["n_right"]
    if type(nl) is not int or type(nr) is not int or min(nl, nr) < 1:
        raise ValueError("invalid dimensions")
    n = nl + nr
    tails = decode(p["tails"], integer=True)
    m = len(tails)
    if not 0 < m <= MAX_EDGES or n > m + 1:
        raise ValueError("invalid graph size")
    heads = decode(p["heads"], m, integer=True)
    if any(not 0 <= a < nl or not nl <= b < n for a, b in zip(tails, heads)):
        raise ValueError("invalid bipartite endpoints")
    beta_float = decode(p["declared_beta"], n)
    c_float, mu_float = decode(p["c"], m), decode(p["mu"], m)
    if any(x <= 0 for x in mu_float):
        raise ValueError("nonpositive reciprocal weight")
    beta = list(map(F, beta_float))
    beta[-1] = -sum(beta[:-1], F(0))  # the documented n-1 input contract
    beta_float[-1] = float(beta[-1])
    # Independently implement the public, versioned content fingerprint.
    topology = sha256(b"certquota-problem-topology-v1\0")
    topology.update(struct.pack("<QQ", nl, nr))
    def append_array(digest, values, code):
        digest.update(struct.pack("<QQ", 1, len(values)))
        digest.update(struct.pack("<" + code * len(values), *values))
    append_array(topology, tails, "q")
    append_array(topology, heads, "q")
    content = sha256(b"certquota-problem-content-v1\0")
    content.update(topology.digest())
    for values in (beta_float, c_float, mu_float):
        append_array(content, values, "d")
    if p.get("content_fingerprint") != content.hexdigest():
        raise ValueError("problem fingerprint mismatch")
    return n, tails, heads, beta, list(map(F, c_float)), list(map(F, mu_float))


def tree_order(n, tails, heads, edges=None, root=None):
    """Independent adjacency/BFS implementation; validates spanning-tree inputs."""
    root = n - 1 if root is None else root
    if type(root) is not int or not 0 <= root < n:
        raise ValueError("invalid root")
    if edges is not None and (len(edges) != n - 1 or len(set(edges)) != n - 1):
        raise ValueError("tree must have n-1 distinct edges")
    adjacency = [[] for _ in range(n)]
    for e in (range(len(tails)) if edges is None else edges):
        if not 0 <= e < len(tails):
            raise ValueError("invalid tree edge")
        a, b = tails[e], heads[e]
        adjacency[a].append((b, e, 1))
        adjacency[b].append((a, e, -1))
    parent = {root: None}
    order = [root]
    for v in order:
        for w, e, sign_at_v in adjacency[v]:
            if w not in parent:
                parent[w] = (v, e, -sign_at_v)
                order.append(w)
    if len(order) != n:
        raise ValueError("disconnected graph/tree")
    return order, parent


def recover(n, tails, heads, beta, x, order, parent, zero):
    """Exact identity B h=beta-B x by leaf elimination, for F or I arithmetic."""
    residual = [zero for _ in range(n)]
    for e, (a, b) in enumerate(zip(tails, heads)):
        residual[a] = residual[a] + x[e]
        residual[b] = residual[b] - x[e]
    demand = [beta[v] - residual[v] for v in range(n)]
    out = list(x)
    for v in reversed(order[1:]):
        p, e, sign = parent[v]
        out[e] = out[e] + sign * demand[v]
        demand[p] = demand[p] + demand[v]
    return out


def objective(c, mu, flow):
    total = I()
    for cost, weight, value in zip(c, mu, flow):
        value = value if isinstance(value, I) else I(value)
        if value.lo <= 0:
            raise ValueError("positive primal feasibility not established")
        total = total + I(cost) * value + I(weight) / value
    return total


def summarize(lower, upper, epsilon, **extra):
    width = upper - lower
    if width < 0:
        raise ValueError("invalid objective ordering")
    return {"accepted": bool(width <= epsilon),
            "lower": lower_float(lower), "upper": upper_float(upper),
            "gap_upper": upper_float(width),
            "lower_exact": str(lower), "upper_exact": str(upper),
            "epsilon_exact": str(epsilon), "precision_bits": BITS, **extra}


def check_endpoint(payload):
    n, tails, heads, beta, c, mu = read_problem(payload["problem"])
    y = list(map(F, decode(payload["candidate_y"], n)))
    eps_float = float.fromhex(payload["epsilon_hex"])
    if not math.isfinite(eps_float) or eps_float <= 0:
        raise ValueError("invalid tolerance")
    epsilon = F(eps_float)
    tree = payload["tree"]
    order, parent = tree_order(n, tails, heads,
                              decode(tree["tree_edges"], n - 1, True), tree["root"])
    q = [cost - y[a] + y[b] for cost, a, b in zip(c, tails, heads)]
    if any(v <= 0 for v in q):
        raise ValueError("dual domain not established")
    x = [(I(w) / I(s)).sqrt() for w, s in zip(mu, q)]
    flow = recover(n, tails, heads, list(map(I, beta)), x, order, parent, I())
    primal = objective(c, mu, flow)
    dual = I(sum((b * v for b, v in zip(beta, y)), F(0)))
    for w, s in zip(mu, q):
        dual = dual + 2 * I(w * s).sqrt()
    return summarize(F(dual.lo, SCALE), F(primal.hi, SCALE), epsilon,
                     kind="direct_objective_endpoint", exact_feasible_object="implicit_real",
                     x_min_lower=lower_float(F(min(x.lo for x in flow), SCALE)),
                     problem_fingerprint=payload["problem"]["content_fingerprint"])


def check_conic(payload):
    """Verify a port bracket for these original inputs, not VSDP in general.

    The lifted equations are reconstructed from the original inputs and scales;
    no serialized matrix, cone residual or optimizer status is trusted.
    """
    n, tails, heads, beta, c, mu = read_problem(payload["problem"])
    m = len(tails)
    lift = payload["lift"]
    xs, os = F(float.fromhex(lift["x_scale_hex"])), F(float.fromhex(lift["objective_scale_hex"]))
    if xs <= 0 or os <= 0:
        raise ValueError("invalid conic scaling")
    rs = list(map(F, decode(lift["row_scale"], n - 1)))
    if any(v <= 0 for v in rs):
        raise ValueError("invalid row scale")
    y = list(map(F, decode(payload["dual_witness"], n - 1 + 3 * m)))
    # Reconstruct c_lift - A_lift^T y, never trust a serialized residual or At.
    d_z, d_w, d_soc = [], [], []
    for e, (a, b) in enumerate(zip(tails, heads)):
        by = rs[a] * y[a] - (rs[b] * y[b] if b < n - 1 else 0)
        yu, ys, yv = y[n - 1 + e], y[n - 1 + m + e], y[n - 1 + 2 * m + e]
        d_z.append(c[e] * xs / os - xs * by + ys + yv)
        d_w.append(mu[e] / xs / os + ys - yv)
        d_soc.append((-ys, -yu, -yv))
    if any(v < 0 for v in d_z + d_w):
        raise ValueError("exact LP dual cone violation")
    if any(s < 0 or s*s < u*u + v*v for s, u, v in d_soc):
        raise ValueError("exact SOC dual cone violation")
    dual = os * (sum((beta[i] * rs[i] * y[i] for i in range(n - 1)), F(0))
                 + 2 * sum(y[n - 1:n - 1 + m], F(0)))
    x0 = list(map(F, decode(payload["primal_original_midpoint"], m)))
    order, parent = tree_order(n, tails, heads)
    primal_flow = recover(n, tails, heads, beta, x0, order, parent, F(0))
    # Explicit exact residual check, independent of the elimination argument.
    residual = [F(0) for _ in range(n)]
    for a, b, value in zip(tails, heads, primal_flow):
        residual[a] += value
        residual[b] -= value
    if residual != beta or any(v <= 0 for v in primal_flow):
        raise ValueError("exact primal feasibility failed")
    primal_upper = F(objective(c, mu, primal_flow).hi, SCALE)
    claimed_l = F(float.fromhex(payload["claimed_lower_hex"]))
    claimed_u = F(float.fromhex(payload["claimed_upper_hex"]))
    if claimed_l > dual or claimed_u < primal_upper:
        raise ValueError("exported witnesses do not prove the claimed bracket")
    eps = F(float.fromhex(payload["epsilon_hex"]))
    if eps <= 0:
        raise ValueError("invalid tolerance")
    return summarize(claimed_l, claimed_u, eps, kind="exact_conic_dual_rational_primal",
                     bracket_valid=True, exact_feasible_object="explicit_rational",
                     rational_primal_objective_upper=upper_float(primal_upper),
                     exact_dual_objective_lower=lower_float(dual),
                     problem_fingerprint=payload["problem"]["content_fingerprint"])


def check(document):
    if set(document) != {"payload", "payload_sha256"}:
        raise ValueError("invalid proof document")
    p = document["payload"]
    digest = sha256(canonical(p).encode("ascii")).hexdigest()
    if digest != document["payload_sha256"]:
        raise ValueError("payload checksum mismatch")
    if p["schema"] == "certquota-endpoint-replay-v1":
        result = check_endpoint(p)
    elif p["schema"] == "certquota-conic-witness-a45-v1":
        result = check_conic(p)
    else:
        raise ValueError("unknown proof schema")
    result["payload_sha256"] = digest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = check(json.loads(args.proof.read_text(encoding="ascii")))
        result["seconds"] = time.perf_counter() - started
        print(canonical(result))
        return 0 if result["accepted"] else 2
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        print(canonical({"accepted": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
