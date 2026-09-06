"""Public cold start, strict acceptance, and portable certificate replay."""
from pathlib import Path
from certquota import make_graph_family_instance, solve_verified, VerifiedOptions
from certquota.baselines import nonoptimal_dual_start
from certquota.replay import write_replay_bundle, verify_replay_bundle

problem = make_graph_family_instance('community', 64, 16, degree=4, seed=46001).problem
candidate = nonoptimal_dual_start(problem, seed=46002)
result = solve_verified(problem, candidate, options=VerifiedOptions(epsilon=1e-7))
print('Solver/verifier status:', result.status)
if result.status == 'optimal':
    output = Path('certificate.json')
    if output.exists():
        raise SystemExit('certificate.json already exists; use a new working directory')
    write_replay_bundle(result, output)
    replay = verify_replay_bundle(output)
    assert replay.accepted and replay.claim_matches
    print('Independent process can run: certquota-verify certificate.json')
    print('Binary64 output is approximate; exact feasibility belongs to the implicit real recovery.')
else:
    raise SystemExit('No certificate was authorized on this run.')
