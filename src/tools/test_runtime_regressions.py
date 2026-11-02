"""Offline regressions for interpreter lookup and cross-repository seen140 identity."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from runtime_paths import resolve_python
from evaluation.identity import LEGACY_RAW_SHA256, require_seen140
import plot
import run_pair


class InterpreterTests(unittest.TestCase):
    def test_bare_python_uses_path(self):
        with patch('runtime_paths.shutil.which', return_value=sys.executable) as lookup:
            self.assertEqual(resolve_python('python', ROOT), str(Path(sys.executable).resolve()))
            lookup.assert_called_once_with('python')

    def test_relative_path_is_root_relative_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / 'env with spaces/python'
            interpreter.parent.mkdir()
            interpreter.touch()
            interpreter.chmod(0o700)
            self.assertEqual(resolve_python('env with spaces/python', root), str(interpreter.resolve()))
            self.assertEqual(resolve_python(str(interpreter.resolve()), ROOT), str(interpreter.resolve()))

    def test_missing_command_and_file_fail_before_training(self):
        with patch('runtime_paths.shutil.which', return_value=None):
            with self.assertRaises(ValueError):
                resolve_python('missing-python', ROOT)
        with self.assertRaises(ValueError):
            resolve_python('./missing-env/python', ROOT)
        with patch.dict(os.environ, {'OPSD_PYTHON': './missing-env/python'}), \
                patch.object(run_pair, 'sft_complete', side_effect=AssertionError('Not reached')):
            with self.assertRaises(ValueError):
                run_pair.commands('opsd')


class EvaluationIdentityTests(unittest.TestCase):
    def test_both_verified_legacy_serializations_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('full.json', 'ablation.json')]
            for path, checksum in zip(paths, sorted(LEGACY_RAW_SHA256)):
                summary = dict(status='passed', step=150, split='valid_seen', tasks=140,
                    successes=0, success_rate=0., task_manifest_sha256=checksum,
                    analyzer_called=False, teacher_called=False, training_updates=0)
                path.write_text(json.dumps(summary), encoding='utf-8')
            self.assertEqual(len(plot.rows_from_summaries(*paths)), 2)

    def test_changed_order_seed_or_game_is_rejected(self):
        original = json.loads((ROOT / 'configs/seen140_tasks.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'tasks.json'
            for key in ('order', 'seed', 'game_sha256', 'split'):
                rows = json.loads(json.dumps(original))
                if key == 'order':
                    rows[0], rows[1] = rows[1], rows[0]
                elif key == 'seed':
                    rows[0][key] += 1
                else:
                    rows[0][key] = 'changed'
                manifest.write_text(json.dumps(rows), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    require_seen140({'task_manifest_sha256': next(iter(LEGACY_RAW_SHA256))}, manifest)

    def test_unknown_or_conflicting_identity_is_rejected(self):
        for summary in ({'task_manifest_sha256': '0' * 64},
                        {'task_manifest_sha256': next(iter(LEGACY_RAW_SHA256)),
                         'task_manifest_semantic_sha256': '0' * 64}):
            with self.assertRaises(ValueError):
                require_seen140(summary, ROOT / 'configs/seen140_tasks.json')


if __name__ == '__main__':
    unittest.main()
