"""Offline regression tests: packaging, relocatable inputs, and hint transport."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import package
import plot
import setup
import audit_recipe

ROOT = Path(__file__).resolve().parents[2]


def load_worker():
    spec = importlib.util.spec_from_file_location('portable_api_worker', ROOT / 'src/opsd/api_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortabilityTests(unittest.TestCase):
    def test_public_layout_matches_original_template(self):
        names = {path.relative_to(ROOT).as_posix() for path in package.public_paths()}
        self.assertEqual({name.split('/')[0] for name in names}, {
            'README.md', 'project.yaml', 'TODO.md', 'LICENSE', '.gitignore', '.env.example',
            'docs', 'src', 'configs', 'checkpoints', 'data', 'results', 'scripts', '.github',
        })
        self.assertEqual({name for name in names if name.startswith('scripts/')}, {
            'scripts/train.sh', 'scripts/eval.sh', 'scripts/reproduce.sh',
        })
        self.assertEqual({name for name in names if name.startswith('docs/')}, {
            'docs/idea.md', 'docs/related_work.md', 'docs/experiment_log.md',
        })
        self.assertEqual({name for name in names if name.startswith('.github/')}, {
            '.github/PULL_REQUEST_TEMPLATE.md', '.github/ISSUE_TEMPLATE/weekly_report.md',
            '.github/ISSUE_TEMPLATE/bug_report.md', '.github/ISSUE_TEMPLATE/experiment_request.md',
            '.github/workflows/cpu.yml',
        })

    def test_public_archive_runs_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive = folder / 'source.zip'
            package.export(archive)
            with zipfile.ZipFile(archive) as z:
                names = set(z.namelist())
                self.assertNotIn('.env', names)
                self.assertEqual({n for n in names if n.startswith('data/')},
                                 {'data/README.md', 'data/frozen_data.zip'})
                self.assertEqual({n for n in names if n.startswith('checkpoints/')}, {'checkpoints/.gitkeep'})
                self.assertFalse(any(n.startswith(('.deps/', 'runs/', 'reports/')) for n in names))
                self.assertTrue(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names))
                z.extractall(folder / 'standalone')
            clean = folder / 'standalone'
            env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            for key in ('RETHINK_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY'):
                env.pop(key, None)
            # Different cwd, no .env, no weights/dependency checkout, no API.
            for relative, options in (
                ('src/tools/setup.py', ['--bundled-data']),
                ('src/tools/check.py', []), ('src/tools/setup.py', ['--help']),
                ('src/tools/plot.py', ['--help']),
                ('src/tools/package.py', []), ('src/opsd/api_worker.py', ['--help']),
                ('src/opsd/prepare.py', ['--help']), ('src/opsd/run.py', ['--help']),
                ('src/tools/run_pair.py', ['--help']), ('src/tools/audit_recipe.py', []),
                ('src/evaluation/run_seen140.py', ['--help']),
            ):
                result = subprocess.run([sys.executable, str(clean / relative), *options], cwd=folder,
                                        env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, relative + ': ' + result.stderr)

    def test_privacy_scan_redacts_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for original in package.public_paths():
                target = root / original.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            fake = 'sk-' + 'A1' * 20
            with (root / 'README.md').open('a', encoding='utf-8') as stream:
                stream.write('\n' + fake + '\n')
            findings = package.privacy_findings(root)
            self.assertIn(('README.md', 'credential'), [(name, category) for name, _, category in findings])
            self.assertNotIn(fake, repr(findings))
            with self.assertRaises(RuntimeError):
                package.export(root / 'blocked.zip', root)
            self.assertFalse((root / 'blocked.zip').exists())

    def test_data_import_standard_layout_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            source = Path(directory) / 'dataset'
            (root / 'configs').mkdir(parents=True)
            (source / 'route_sft').mkdir(parents=True)
            (source / 'route_sft/train.jsonl').write_bytes(b'{"test":true}\n')
            (source / 'selection_ids.json').write_bytes(b'[]\n')
            names = ('route_sft/train.jsonl', 'selection_ids.json')
            lock = {'files': {name: setup.sha(source / name) for name in names}}
            (root / 'configs/data.json').write_text(json.dumps(lock), encoding='utf-8')
            with patch.object(setup, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                setup.import_data(source)
                setup.import_data(source)  # idempotent verified import
                self.assertTrue(all(setup.sha(root / 'data' / n) == lock['files'][n] for n in names))
                (root / 'data/selection_ids.json').write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    setup.import_data(source)
                self.assertEqual((root / 'data/selection_ids.json').read_bytes(), b'changed')

    def test_data_import_validates_every_file_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            source = Path(directory) / 'data'
            (root / 'configs').mkdir(parents=True)
            source.mkdir()
            (source / 'first.json').write_bytes(b'{}')
            lock = {'files': {'first.json': setup.sha(source / 'first.json'), 'missing.json': '0' * 64}}
            (root / 'configs/data.json').write_text(json.dumps(lock), encoding='utf-8')
            with patch.object(setup, 'ROOT', root), self.assertRaises(RuntimeError):
                setup.import_data(source)
            self.assertFalse((root / 'data').exists())

    def test_api_endpoint_requires_https_without_embedded_credentials(self):
        worker = load_worker()
        self.assertEqual(worker.validate_endpoint('https://api.example/v1/chat/completions'),
                         'https://api.example/v1/chat/completions')
        for endpoint in ('', 'http://api.example', 'https://user:password@api.example',
                         'https://api.example?token=value', 'https://api.example#secret'):
            with self.assertRaises(RuntimeError):
                worker.validate_endpoint(endpoint)

    def test_api_redirect_is_refused(self):
        worker = load_worker()
        with self.assertRaisesRegex(RuntimeError, 'API_REDIRECT_REFUSED'):
            worker.NoRedirect().redirect_request(None, None, None, None, None, None)

    def test_ray_runtime_directory_is_local_unique_and_configurable(self):
        spec = importlib.util.spec_from_file_location('runtime_paths', ROOT / 'src/runtime_paths.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'RETHINK_RUNTIME_ROOT': directory}):
                first, second = runtime.new_ray_directory(), runtime.new_ray_directory()
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())
            self.assertEqual(first.parent, Path(directory).resolve())

    def test_cached_hint_is_remapped_without_network(self):
        worker = load_worker()
        # A transport-only unit test; protocol semantics are tested separately.
        worker.protocol = SimpleNamespace(VERSION='test', request_body=lambda _: {}, digest=lambda _: 'cached',
            parse_hints=lambda content, group, finish: {group['id']: content})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / 'cached.json').write_text(json.dumps({'request_sha256': 'cached', 'content': 'hint',
                'finish_reason': 'stop', 'outputs': {'old': 'hint'}}), encoding='utf-8')
            with patch.object(worker.urllib.request, 'build_opener', side_effect=AssertionError('No network')):
                value, called = worker.execute({'id': 'new'}, cache, 'unused', 'https://api.example')
            self.assertFalse(called)
            self.assertEqual(value['outputs'], {'new': 'hint'})

    @unittest.skipUnless((ROOT / '.deps/sdar/.rethink_setup.json').is_file(), 'Run setup.py --framework first')
    def test_pinned_hint_protocol_keeps_actor_target_and_checks_action(self):
        worker = load_worker()
        protocol = worker.load_protocol()
        step = {'step': 0, 'observation': 'A desk is visible.', 'admissible_actions': ['go to desk 1'],
                'reasoning': 'Inspect the desk.', 'action': 'go to desk 1', 'feedback': 'A book is here.'}
        request = {'key': 'row', 'target': 0, 'episode': {'task': 'find a book', 'steps': [step]}}
        group, = protocol.group_requests([request])
        good = protocol.parse_hints(json.dumps({'hints': {'0': "Next command: 'go to desk 1'. Inspect the desk."}}), group, 'stop')
        self.assertIsNotNone(good['row']['payload'])  # no Analyzer Good/Bad gate
        bad = protocol.parse_hints(json.dumps({'hints': {'0': "Next command: 'take book 1'."}}), group, 'stop')
        self.assertIsNone(bad['row']['payload'])
        self.assertEqual(set(bad), {'row'})  # invalid hint does not delete the Actor route


class ResultTests(unittest.TestCase):
    @staticmethod
    def summary(successes):
        return dict(status='passed', step=150, split='valid_seen', tasks=140,
                    successes=successes, success_rate=successes / 140,
                    task_manifest_sha256=setup.sha(ROOT / 'configs/seen140_tasks.json'),
                    analyzer_called=False, teacher_called=False, training_updates=0)

    def test_fresh_results_replace_static_scores_without_private_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            summaries = [folder / name for name in ('baseline.json', 'method.json')]
            for path, wins in zip(summaries, (70, 84)):
                result = self.summary(wins)
                result['private_metadata'] = 'must not appear in the public table'
                path.write_text(json.dumps(result), encoding='utf-8')
            rows = plot.rows_from_summaries(*summaries)
            self.assertEqual([row['successes'] for row in rows], [70, 84])
            self.assertEqual([row['success_rate'] for row in rows], [.5, .6])
            self.assertTrue(all(set(row) == set(plot.FIELDS) for row in rows))
            table = folder / 'results/tables/seen140.csv'
            plot.write_table(rows, table)
            self.assertNotIn('private_metadata', table.read_text())
            self.assertIn('0.5',table.read_text())
            self.assertIn('0.6',table.read_text())

    def test_unmatched_or_incomplete_evaluations_are_rejected(self):
        changes = ({'status': 'running'}, {'task_manifest_sha256': '0' * 64},
                   {'step': 140}, {'split': 'valid_unseen'}, {'tasks': 134},
                   {'teacher_called': True}, {'analyzer_called': True}, {'training_updates': 1},
                   {'success_rate': .99}, {'success_rate': float('nan')}, {'successes': 70.5})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            baseline, method = folder / 'baseline.json', folder / 'method.json'
            baseline.write_text(json.dumps(self.summary(70)), encoding='utf-8')
            for change in changes:
                with self.subTest(change=change):
                    method.write_text(json.dumps(dict(self.summary(84), **change)), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        plot.rows_from_summaries(baseline, method)

    def test_no_unrun_model_results_are_published(self):
        published = {p.relative_to(ROOT).as_posix() for p in package.public_paths()}
        self.assertNotIn('results/tables/seen140.csv', published)
        self.assertNotIn('results/figures/seen140.png', published)
        self.assertIn('results/tables/.gitkeep', published)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Install configs/requirements-sft.txt')
    def test_result_cli_generates_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            baseline, method = folder / 'baseline.json', folder / 'method.json'
            baseline.write_text(json.dumps(self.summary(70)), encoding='utf-8')
            method.write_text(json.dumps(self.summary(84)), encoding='utf-8')
            destination = folder / 'results'
            result = subprocess.run([sys.executable, str(ROOT / 'src/tools/plot.py'),
                '--baseline', str(baseline), '--method', str(method), '--out', str(destination)],
                cwd=folder, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((destination / 'tables/seen140.csv').is_file())
            self.assertTrue((destination / 'figures/seen140.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertFalse((destination / 'demos').exists())  # no fabricated Demo


class AblationContractTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('bash'), 'bash required; exercised in Linux CI')
    def test_assigned_evaluation_gpu_wins_over_dotenv_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'src/evaluation').mkdir(parents=True)
            shutil.copyfile(ROOT / 'scripts/eval.sh', root / 'scripts/eval.sh')
            (root / '.env').write_text(
                'EVAL_GPU=0\nALFWORLD_DATA=unused\nOPSD_PYTHON=bash\n', encoding='utf-8')
            # The shell fixture replaces the evaluator, not the GPU allocator.
            (root / 'src/evaluation/run_seen140.py').write_text(
                'printf "%s" "$CUDA_VISIBLE_DEVICES"\n', encoding='utf-8')
            result = subprocess.run(['bash', str(root / 'scripts/eval.sh'), '--model', 'unused'],
                env=dict(os.environ, EVAL_GPU='4'), cwd=directory, capture_output=True,
                text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '4')

    def test_recipe_matches_frozen_3b_except_explicit_delta(self):
        self.assertTrue(audit_recipe.audit()['passed'])

    def test_only_one_explicit_gpu_is_allocated(self):
        sys.path.insert(0, str(ROOT / 'src'))
        from gpu_layout import layout
        self.assertEqual(layout({}), {'method': ['0']})
        self.assertEqual(layout({'TRAIN_GPUS':'3'}), {'method':['3']})
        with self.assertRaises(ValueError):
            layout({'TRAIN_GPUS':'0,1'})
        with self.assertRaises(ValueError):
            layout({'TRAIN_GPUS':'0,0'})
        with self.assertRaises(ValueError):
            layout({'TRAIN_GPUS':'-1'})

    def test_shared_opsd_math_and_original_single_rank_scaling(self):
        import yaml
        current = yaml.safe_load((ROOT / 'configs/opsd.yaml').read_text())
        old = json.loads((ROOT / 'configs/reference_3b.json').read_text())['opsd']
        current['algorithm'].pop('external_router',None)
        self.assertEqual(current['algorithm'], old['algorithm'])
        self.assertEqual(current['actor_rollout_ref'], old['actor_rollout_ref'])
        self.assertEqual(current['data'], old['data'])
        self.assertEqual(current['env'], old['env'])
        # Preserve the original 3B single-rank token-mean/microbatch objective.
        global_batch = current['actor_rollout_ref']['actor']['ppo_mini_batch_size']
        micro = current['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu']
        self.assertEqual((global_batch, micro), (256, 32))
        self.assertEqual(global_batch // micro, 8)

    def test_one_arm_queue_uses_declared_start_and_final150(self):
        import run_pair
        from ablation import contract
        with patch.object(run_pair, 'sft_complete', return_value=False), \
             patch.object(Path, 'exists', return_value=False):
            queue = run_pair.commands('opsd')
        expected=(['sft_preflight','sft'] if contract()['sft_enabled'] else [])+['training','export','evaluation']
        self.assertEqual([x[0] for x in queue],expected)
        self.assertIn('--allow-api-cost',next(x[1] for x in queue if x[0]=='training'))
        self.assertIn('global_step_150',' '.join(queue[-2][1]))
        self.assertEqual(queue[-1][1][-2:],['--step','150'])

    def test_eval_explicitly_disables_both_api_paths(self):
        import ast
        tree=ast.parse((ROOT/'src/evaluation/run_seen140.py').read_text())
        settings={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                for k,v in zip(node.keys,node.values):
                    if isinstance(k,ast.Constant) and isinstance(v,ast.Constant):settings[k.value]=v.value
        self.assertIs(settings['algorithm.local_hints.enabled'],False)
        self.assertIs(settings['algorithm.external_router.enabled'],False)
        self.assertIs(settings['algorithm.external_router.evaluation_enabled'],False)


if __name__ == '__main__':
    unittest.main()
