"""
Tests of the command line (dies/cli.py, Phase 3, Nov 2026): `dies run` and `dies check`,
called through main(argv) and, once, as `python -m dies` in a separate process.
"""

import copy
import os
import subprocess
import sys
import warnings

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import dies
from dies.cli import main
from results import DiESResults, DynamicResults

DIKE = os.path.join(ROOT, 'examples', 'dike_example.yaml')


def write_scenario(tmp_path, change=None, name='scenario.yaml'):
    """The dike example on a coarse grid (fast), with an optional change, written to tmp_path."""
    with open(DIKE) as f:
        raw = yaml.safe_load(f)
    raw['grid']['cell'] = ['500 m', '500 m']
    raw['output']['dpi'] = 50
    if change:
        change(raw)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_run_writes_the_output(tmp_path, capsys):
    path = write_scenario(tmp_path)
    out = tmp_path / 'results'
    assert main(['run', path, '--out', str(out)]) == 0
    files = sorted(os.listdir(out))
    # Results, the copy of the scenario, 7 static figures and 8 dynamic steps (4, 8, ..., 32)
    assert {'scenario.yaml', 'static_results.npz', 'static_results.csv', 'dynamic_results.npz',
            'dynamic_final_step.csv'} <= set(files)
    assert len([f for f in files if f.startswith('dynamic_step_')]) == 8 and len(files) == 20
    # The same results as running the scenario from Python
    run = dies.load_scenario(path).run()
    dyn = DynamicResults.load(str(out / 'dynamic_results.npz'))
    assert (dyn.final_mode() == run.dynamic.final_mode()).all()
    static = DiESResults.load(str(out / 'static_results.npz'))
    assert (static['sxx'] == run.static['sxx']).all()
    text = capsys.readouterr().out
    assert '32 dynamic steps' in text and '20 files written' in text


def test_run_default_folder_and_overwrite(tmp_path, monkeypatch, capsys):
    path = write_scenario(tmp_path, lambda r: r.update(analysis={'type': 'static'},
                                                       output={'figures': {'static': ['s1']}}),
                          name='dike.yaml')
    monkeypatch.chdir(tmp_path)
    # Default folder: <scenario file name>_results in the current folder
    assert main(['run', path, '-q']) == 0
    assert sorted(os.listdir(tmp_path / 'dike_results')) == ['s1.png', 'scenario.yaml', 'static_results.npz']
    assert capsys.readouterr().out == ''          # --quiet prints nothing when all goes well
    # A folder that is not empty is not overwritten without --overwrite
    assert main(['run', path, '-q']) == 1
    assert 'is not empty; use --overwrite' in capsys.readouterr().err
    assert main(['run', path, '-q', '--overwrite']) == 0


def test_invalid_scenario_lists_every_problem(tmp_path, capsys):
    def break_it(raw):
        raw['layers'][0]['E'] = '30e9'
        raw['layers'][0]['frictoin'] = raw['layers'][0].pop('friction')
        raw['cracks'][0]['sigma_I'] = '8 MPa'
    path = write_scenario(tmp_path, break_it)
    assert main(['run', path, '--out', str(tmp_path / 'out')]) == 1
    err = capsys.readouterr().err
    assert 'Invalid scenario (4 problems)' in err and 'Traceback' not in err
    assert "layers[0].E: '30e9' has no unit" in err
    assert "did you mean 'friction'?" in err
    assert not (tmp_path / 'out').exists()        # nothing is written
    assert main(['check', path]) == 1
    assert main(['check', str(tmp_path / 'missing.yaml')]) == 1
    assert 'cannot be read' in capsys.readouterr().err


def test_check_summary(tmp_path, capsys):
    assert main(['check', DIKE]) == 0
    text = capsys.readouterr().out
    assert 'Grid: x -5 km to 5 km, y -6 km to -0.01 km, cells 100 m x 100 m (100 x 60 = 6000 nodes)' in text
    assert 'sigma_I 8 MPa (from the aperture of 2 m)' in text
    assert 'Analysis: dynamic, 32 steps of 0.25 MPa' in text
    assert text.strip().endswith('The scenario is valid.')
    # The node count is that of the model that run() builds
    run = dies.Scenario.from_dict(dies.load_scenario(DIKE).raw).run()
    assert run.static.shape == (60, 100)


def test_python_m_dies(tmp_path):
    """`python -m dies` works from the repository without installing anything."""
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, '-m', 'dies', 'check', DIKE], capture_output=True, text=True,
                            env=env, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert 'The scenario is valid.' in result.stdout
    result = subprocess.run([sys.executable, '-m', 'dies', '--version'], capture_output=True, text=True,
                            env=env, cwd=str(tmp_path))
    assert result.returncode == 0 and result.stdout.startswith('DiES ')


def test_entry_point_declared():
    """pyproject.toml installs the `dies` command, pointing at dies.cli:main."""
    try:
        import tomllib
    except ImportError:
        pytest.skip('tomllib needs Python 3.11')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        project = tomllib.load(f)['project']
    assert project['scripts'] == {'dies': 'dies.cli:main'}
