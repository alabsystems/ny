"""Failure-oriented tests using tiny Python children; no solver or compiler launches."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import safenlp_benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def test_real_producer_tags_preserve_floating_bounds_and_mask_counts(self):
        raw = ('[input-head] status=open classes=1 calls=2 replayed=3 lower_bits=c000000000000000 elapsed_s=0.125000\n'
               '[input-head-triangle-delta] before_bits=c008000000000000 after_bits=c000000000000000\n'
               '[input-head-large-negative-group-replay] rows=2 certified_masks=64\n'
               '"domains_explored": 19\n')
        observed = bench.diagnostics(raw)
        self.assertEqual(observed['root_bound_deltas'][0]['before'], -3.0)
        self.assertEqual(observed['source_root_observations'][0]['lower'], -2.0)
        self.assertEqual(observed['large_group_replay'], [dict(rows=2, certified_masks=64)])
        self.assertEqual(observed['engine_domains_explored_observations'], [19])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='safenlp-runner-test-')
        self.root = Path(self.temp.name)
        self.limits = dict(cpu_affinity=[min(os.sched_getaffinity(0))],
                           address_space_bytes=None, max_open_files=256)

    def tearDown(self):
        self.temp.cleanup()

    def run_child(self, code, budget=1.5):
        directory = self.root / ('run-' + str(len(list(self.root.iterdir()))))
        directory.mkdir()
        # Other sessions' real builders are outside these synthetic unit tests.
        with patch.object(bench, 'BUSY', re.compile(r'(?!)')):
            execution = bench.execute([sys.executable, '-B', '-c', code], str(directory),
                {'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1'},
                self.limits, directory, budget, sample_seconds=0.01)
        stdout = (directory / 'stdout.log').read_text()
        stderr = (directory / 'stderr.log').read_text()
        classified = bench.classify(execution, directory / 'result.txt', stdout, stderr)
        return directory, execution, classified

    def test_early_result_does_not_hide_late_process_completion(self):
        directory, execution, classified = self.run_child(
            "from pathlib import Path; import time; Path('result.txt').write_text('unsat\\n'); time.sleep(0.6)", 0.15)
        self.assertTrue(execution['timed_out'])
        self.assertEqual(classified['outcome'], 'timeout')
        self.assertTrue(classified['valid_for_comparison'])
        self.assertEqual((directory / 'result.txt').read_text(), 'unsat\n')
        self.assertFalse(execution['live_group_after_cleanup'])
        self.assertGreaterEqual(execution['cleanup_seconds'], 0)

    def test_early_result_does_not_hide_live_descendant(self):
        _, execution, classified = self.run_child(
            "import subprocess,sys; from pathlib import Path; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.8)']); "
            "Path('result.txt').write_text('unsat\\n')", 0.2)
        self.assertTrue(execution['timed_out'])
        self.assertEqual(classified['outcome'], 'timeout')
        self.assertFalse(execution['live_group_after_cleanup'])

    def test_nonzero_exit_cannot_publish_conclusive_result(self):
        _, execution, classified = self.run_child(
            "from pathlib import Path; Path('result.txt').write_text('sat\\n'); raise SystemExit(7)")
        self.assertEqual(execution['returncode'], 7)
        self.assertEqual(classified['outcome'], 'error')
        self.assertFalse(classified['independently_validated_sat'])

    def test_sat_is_only_a_claim_and_witness_is_retained(self):
        directory, _, classified = self.run_child(
            "from pathlib import Path; Path('result.txt').write_text('sat\\n'); "
            "Path('witness.txt').write_text('untrusted witness')")
        self.assertEqual(classified['outcome'], 'claimed_sat')
        self.assertFalse(classified['independently_validated_sat'])
        self.assertEqual((directory / 'witness.txt').read_text(), 'untrusted witness')

    def test_absent_or_malformed_artifact_never_falls_back_to_stdout(self):
        for code in ["print('unsat')", "from pathlib import Path; Path('result.txt').write_text('broken'); print('unsat')"]:
            with self.subTest(code=code):
                _, _, classified = self.run_child(code)
                self.assertEqual(classified['outcome'], 'error')

    def test_conflicting_artifact_and_log_statuses_are_rejected(self):
        _, _, classified = self.run_child(
            "from pathlib import Path; Path('result.txt').write_text('unsat\\n'); print('Result: sat')")
        self.assertEqual(classified['outcome'], 'error')
        result = self.root / 'ambiguous.txt'
        result.write_text('{"status":"sat","verdict":"unsat"}')
        self.assertIsNotNone(bench.result_status(result)[1])
        for malformed in ['{"status":"sat","status":"unsat"}', '{"status":"unsat","time":NaN}']:
            result.write_text(malformed)
            self.assertIsNotNone(bench.result_status(result)[1])

    def test_cached_verdict_is_not_a_fresh_measurement(self):
        _, _, classified = self.run_child(
            "from pathlib import Path; Path('result.txt').write_text('unsat\\n'); print('cache: 1 served, 0 measured')")
        self.assertEqual(classified['outcome'], 'error')
        self.assertTrue(classified['cached_verdict_detected'])

    def test_private_input_caches_do_not_leak_between_queries(self):
        model, prop = self.root / 'original.onnx', self.root / 'original.vnnlib'
        model.write_bytes(b'unchanged model')
        prop.write_bytes(b'unchanged property')
        row = dict(model=str(model), property=str(prop), model_sha256=bench.sha(model), property_sha256=bench.sha(prop))
        first, second = self.root / 'first', self.root / 'second'
        first.mkdir()
        second.mkdir()
        a = bench.private_inputs(row, first)
        Path(a['property'] + '.compiled').write_bytes(b'ABC side cache')
        b = bench.private_inputs(row, second)
        self.assertFalse(Path(b['property'] + '.compiled').exists())
        self.assertEqual(Path(a['property']).read_bytes(), Path(b['property']).read_bytes())
        self.assertFalse(Path(str(prop) + '.compiled').exists())

    def test_balanced_sequence_and_exclusive_artifacts(self):
        rows = [dict(id='row0'), dict(id='row1')]
        order = list(bench.sequence(rows, 3))
        for row in rows:
            positions = {arm: [] for arm in bench.ARMS}
            for rep in range(3):
                arms = [a for r, n, a in order if r is row and n == rep]
                for index, arm in enumerate(arms):
                    positions[arm].append(index)
            self.assertTrue(all(sorted(p) == [0, 1, 2] for p in positions.values()))
        artifact = self.root / 'receipt.json'
        bench.immutable_json(artifact, {'original': True})
        with self.assertRaises(FileExistsError):
            bench.immutable_json(artifact, {'original': False})
        self.assertEqual(json.loads(artifact.read_text()), {'original': True})

    def test_external_builder_detection_excludes_current_process_group(self):
        snapshot = {1: dict(name='rustc', pgid=99, state='R', rss=4),
                    2: dict(name='ny', pgid=42, state='R', rss=8),
                    3: dict(name='targo', pgid=88, state='Z', rss=0)}
        live, rss, busy = bench.monitor(snapshot, 42, set())
        self.assertTrue(live)
        self.assertEqual(rss, 8)
        self.assertEqual(busy, [dict(pid=1, name='rustc')])

    def test_stdout_intermediate_safe_is_never_a_terminal_result(self):
        execution = dict(launch_error=None, interrupted=False, contaminated=False,
            live_group_after_cleanup=False, timed_out=False, process_seconds=0.1,
            timeout_seconds=20, returncode=0)
        result = bench.classify(execution, self.root / 'absent.txt',
                                'safe\ncontinuing expensive verification\n', '', 'stdout')
        self.assertEqual(result['outcome'], 'error')
        self.assertFalse(result['valid_for_comparison'])

    def test_symlink_result_is_rejected(self):
        directory, _, classified = self.run_child(
            "from pathlib import Path; Path('real-result.txt').write_text('unsat\\n'); "
            "Path('result.txt').symlink_to('real-result.txt')")
        self.assertTrue((directory / 'result.txt').is_symlink())
        self.assertEqual(classified['outcome'], 'error')

    def fake_sweep(self, *, live_group=False, symlink=False, identity_failure=False, refusal=False):
        model, prop = self.root / 'm.onnx', self.root / 'p.vnnlib'
        model.write_bytes(b'model')
        prop.write_bytes(b'property')
        row = dict(id='example', model=str(model), property=str(prop),
                   model_sha256=bench.sha(model), property_sha256=bench.sha(prop))
        plan = dict(schema='ny_safenlp_benchmark_v1', rows=[row], repetitions=1,
            budget_seconds=20, environment={}, resource_limits=self.limits, identities=[], sources=[],
            arms={name: dict(cwd=str(self.root), argv=[sys.executable]) for name in bench.ARMS})
        outside = self.root / 'outside.txt'
        outside.write_text('untrusted stale witness')

        def execute(_argv, _cwd, _env, _limits, directory, _budget):
            if refusal:
                raise ValueError('outside compiler present before launch')
            for name, text in [('stdout.log', ''), ('stderr.log', ''), ('result.txt', 'unsat\n')]:
                (directory / name).write_text(text)
            if symlink:
                (directory / 'witness.txt').symlink_to(outside)
            return dict(launch_error=None, interrupted=False, contaminated=False,
                live_group_after_cleanup=live_group, timed_out=False, process_seconds=0.01,
                timeout_seconds=20, returncode=0)

        checks = [[], [], ValueError('identity changed after query'), ValueError('identity still changed')]
        output = self.root / 'experiment'
        with patch.object(bench, 'validate_plan'), patch.object(bench, 'check_identities',
                side_effect=checks if identity_failure else None, return_value=[]), \
                patch.object(bench, 'execute', side_effect=execute) as worker, \
                patch.dict(os.environ, {'NY_SAFENLP_BOX_PROVENANCE': 'ny_box_measure'}):
            rc = bench.run(plan, output)
        return rc, worker.call_count, output, json.loads((output / 'summary.json').read_text())

    def test_surviving_owned_group_stops_before_next_arm(self):
        rc, calls, _, summary = self.fake_sweep(live_group=True)
        self.assertEqual((rc, calls, summary['attempted_runs']), (1, 1, 1))
        self.assertFalse(summary['complete'])

    def test_generated_symlink_is_recorded_and_stops_sweep(self):
        rc, calls, output, summary = self.fake_sweep(symlink=True)
        self.assertEqual((rc, calls), (1, 1))
        record = json.loads(next(output.glob('*/measurement.json')).read_text())
        self.assertEqual(record['refused_symlink_artifacts'], ['witness.txt'])
        self.assertEqual(record['artifacts']['witness.txt']['kind'], 'symlink')
        self.assertFalse(summary['complete'])

    def test_identity_failure_always_leaves_error_summary(self):
        rc, calls, output, summary = self.fake_sweep(identity_failure=True)
        self.assertEqual((rc, calls), (1, 1))
        self.assertIn('still changed', summary['final_identity_error'])
        self.assertTrue((output / 'identities-after-error.json').is_file())
        self.assertFalse(summary['complete'])

    def test_prelaunch_builder_refusal_leaves_zero_run_summary(self):
        rc, calls, output, summary = self.fake_sweep(refusal=True)
        self.assertEqual((rc, calls, summary['attempted_runs']), (1, 1, 0))
        self.assertIn('outside compiler', summary['refusal_reason'])
        self.assertFalse(json.loads(next(output.glob('*/refusal.json')).read_text())['solver_launched'])
        self.assertFalse(summary['complete'])


if __name__ == '__main__':
    unittest.main()
