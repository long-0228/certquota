"""Check an explicit rational allocation and a budget decision; stdlib only."""
from fractions import Fraction as F
from hashlib import sha256
import argparse
import json
from pathlib import Path

try:
    from .independent_check import I, SCALE, canonical, read_problem, decode, objective, summarize
except ImportError:  # python -I -S decision_check.py proof.json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from independent_check import I, SCALE, canonical, read_problem, decode, objective, summarize


def check_decision(doc):
    p = doc['payload']
    if sha256(canonical(p).encode()).hexdigest() != doc['payload_sha256']:
        raise ValueError('payload checksum mismatch')
    if p['schema'] != 'certquota-rational-decision-a45-v1':
        raise ValueError('wrong schema')
    n, tails, heads, beta, c, mu = read_problem(p['problem'])
    if len(p['rational_flow']) != len(tails):
        raise ValueError('wrong flow dimension')
    x = [F(int(num), int(den)) for num, den in p['rational_flow']]
    if any(v <= 0 for v in x):
        raise ValueError('nonpositive flow')
    residual = [F(0)] * n
    for a, b, v in zip(tails, heads, x):
        residual[a] += v
        residual[b] -= v
    if residual != beta:
        raise ValueError('rational marginal equality failed')
    y = list(map(F, decode(p['candidate_y'], n)))
    dual = I(sum((b*v for b, v in zip(beta, y)), F(0)))
    for a, b, cost, weight in zip(tails, heads, c, mu):
        q = cost - y[a] + y[b]
        if q <= 0:
            raise ValueError('nonpositive dual slack')
        dual = dual + 2 * I(weight*q).sqrt()
    upper = F(objective(c, mu, x).hi, SCALE)
    lower = F(dual.lo, SCALE)
    epsilon = F(float.fromhex(p['epsilon_hex']))
    if epsilon <= 0:
        raise ValueError('invalid epsilon')
    cap = F(p['cap_exact'])
    decision = ('certified_below_policy_cap' if upper <= cap else
                'certified_cap_infeasible' if lower > cap else 'undecided')
    return summarize(lower, upper, epsilon, exact_feasible_object='explicit_rational',
                     decision=decision, cap_exact=str(cap), exact_marginals=True,
                     problem_fingerprint=p['problem']['content_fingerprint'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proof', type=Path)
    args = parser.parse_args()
    try:
        result = check_decision(json.loads(args.proof.read_text(encoding='ascii')))
        print(canonical(result))
        raise SystemExit(0 if result['accepted'] else 2)
    except (ValueError, KeyError, TypeError) as exc:
        print(canonical({'accepted': False, 'error': str(exc)}))
        raise SystemExit(1)
