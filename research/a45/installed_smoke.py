"""Run with python -I after installing a wheel; never add the source tree."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import certquota
import numpy as np
import scipy
from certquota.baselines import nonoptimal_dual_start
from certquota.ieee_intervals import ieee_platform_probe
from certquota.replay import write_replay_bundle, verify_replay_bundle


def main():
    package = Path(certquota.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Package did not come from the isolated installed environment')
    probe = ieee_platform_probe()
    assert probe.passed
    rows = []
    for index, family in enumerate(('client_regular', 'community', 'bottleneck')):
        problem = certquota.make_graph_family_instance(family, 256, 32, degree=4, seed=45901+index).problem
        y0 = nonoptimal_dual_start(problem, seed=45911+index)
        result = certquota.solve_verified(problem, y0, options=certquota.VerifiedOptions(epsilon=1e-7))
        assert result.status == 'optimal'
        path = Path.cwd()/f'installed_{family}.json'
        write_replay_bundle(result, path)
        replay = verify_replay_bundle(path)
        assert replay.accepted and replay.claim_matches
        rows.append({'family': family, 'edges': problem.m, 'accepted': True,
                     'problem_fingerprint': problem.content_fingerprint})
    print(json.dumps({'platform': platform.platform(), 'python': sys.version,
        'package_file': str(package), 'certquota_version': certquota.__version__,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'probe': asdict(probe),
        'installed_wheel_only': True, 'cases': rows}, sort_keys=True))


if __name__ == '__main__':
    main()
