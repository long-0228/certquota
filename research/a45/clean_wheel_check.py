"""Build/install a non-editable wheel in a fresh venv; portable Windows/Linux."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    def run(command, name, cwd=out):
        with (out/(name+'.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {name}.log')
    run([sys.executable, '-m', 'build', '--wheel', '--outdir', out/'dist', ROOT], 'build', ROOT)
    wheels = list((out/'dist').glob('*.whl'))
    assert len(wheels) == 1
    venv.EnvBuilder(with_pip=True).create(out/'env')
    python = out/'env'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run([python, '-m', 'pip', 'install', '--disable-pip-version-check',
         wheels[0], 'numpy==1.26.4', 'scipy==1.15.3', 'pytest==8.4.2'], 'install')
    run([python, '-I', ROOT/'research/a45/installed_smoke.py'], 'installed_smoke')
    # No checkout conftest.py: that file deliberately prepends ROOT/src.
    selected = ('test_problem_and_tree.py', 'test_ieee_platform_contract.py',
                'test_ieee_reduction_plans.py', 'test_replay_bundle.py')
    tests = out/'tests'
    tests.mkdir()
    for name in selected:
        shutil.copy2(ROOT/'tests'/name, tests/name)
    run([python, '-I', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', tests], 'installed_tests')
    for proof in sorted(out.glob('installed_*.json')):
        run([python, '-I', '-S', ROOT/'research/a45/independent_check.py', proof], proof.stem+'_independent')
    smoke = json.loads((out/'installed_smoke.log').read_text(encoding='utf-8').strip().splitlines()[-1])
    result = {'passed': True, 'installed_wheel_only': True, 'runtime': smoke,
        'wheel_sha256': sha256(wheels[0].read_bytes()).hexdigest(),
        'pytest_summary': (out/'installed_tests.log').read_text(encoding='utf-8').strip().splitlines()[-1],
        'is_linux': sys.platform.startswith('linux'), 'source_checkout_conftest_used': False}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
