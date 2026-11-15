#!/usr/bin/env python3
"""Freeze and measure the AY proposal prototype; never issue competition verdicts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from benchmark_paired_difference import PROOF_SOURCES, validate_export


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--benchmark-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--memory-mib', type=int, default=2048)
    ap.add_argument('--lane', choices=['exact','float','reduced','reduced-lean'], default='reduced')
    args = ap.parse_args()
    if not 0 < args.seconds <= 100 or not 16 <= args.memory_mib <= 2048:
        ap.error('seconds must be in (0,100], memory in [16,2048] MiB')
    rows = [int(r) for r in args.rows.split(',')]
    if not rows or len(set(rows)) != len(rows) or any(r < 0 or r >= 50 for r in rows):
        ap.error('rows must be distinct indices in [0,49]')
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / args.binary.name
    binary.write_bytes(args.binary.read_bytes())
    binary.chmod(0o755)
    source_files = list(PROOF_SOURCES)
    source_files += ['tools/paired-difference-ay/Cargo.toml', 'tools/paired-difference-ay/Cargo.lock']
    source_files += sorted(str(p.relative_to(args.source_root)) for p in
                           (args.source_root / 'tools/paired-difference-ay').rglob('*.rs'))
    source_files += ['scripts/benchmark_paired_difference_ay.py',
                     'crates/ny-mip/src/paired_difference_ay.rs',
                     'crates/ny-mip/src/paired_difference_ay_tests.rs',
                     'crates/ny-mip/src/paired_difference_ay_reduced.rs',
                     'crates/ny-mip/src/paired_difference_ay_reduced_tests.rs',
                     'crates/ny-mip/Cargo.toml']
    source_hashes = {r: sha(args.source_root / r) for r in source_files}
    evidence = {
        'schema': 'paired-ay-prototype-measure-v1', 'complete': False,
        'binary_sha256': sha(binary), 'binary': str(binary.resolve()),
        'source_root': str(args.source_root.resolve()), 'source_sha256': source_hashes,
        'ay_commit': '55d14404c302d3cdeb9911ed947d41e1dc8188df',
        'seconds': args.seconds, 'memory_mib_requested': args.memory_mib, 'lane': args.lane,
        'claims': {'competition_verdicts': 0, 'additional_competition_points': 0}, 'rows': []}
    result_path = args.output / 'results.json'
    result_path.write_text(json.dumps(evidence, indent=2) + '\n')
    for index in rows:
        path = args.inputs / f'instance_{index}.paired'
        export = validate_export(index, path, args.benchmark_root)
        if export['source_hash_validation'] != 'verified':
            raise ValueError('source identities must be verified')
        dual = args.output / f'instance_{index}.dual'
        command = [str(binary.resolve()), str(path.resolve()), str(args.seconds),
                   str(dual.resolve()), str(args.memory_mib), args.lane]
        started = time.monotonic()
        timed_out = False
        try:
            run = subprocess.run(command, text=True, capture_output=True,
                                 timeout=args.seconds + 2)
            stdout, stderr, rc = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as ex:
            timed_out = True
            stdout = ex.stdout or ''
            stderr = ex.stderr or ''
            if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
            rc = None
        elapsed = time.monotonic() - started
        log = args.output / f'instance_{index}.log'
        log.write_text(stdout + '\nSTDERR:\n' + stderr)
        report = None
        try:
            report = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            pass
        row = {'index': index, 'command': command, 'export': export,
               'elapsed_seconds_external': elapsed, 'return_code': rc,
               'outer_timeout': timed_out, 'late_for_requested_budget': elapsed > args.seconds,
               'stdout': stdout, 'stderr': stderr, 'report': report,
               'log_sha256': sha(log),
               'dual_sha256': sha(dual) if dual.is_file() else None}
        evidence['rows'].append(row)
        result_path.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps({'index': index, 'elapsed': elapsed, 'rc': rc,
                          'report_present': report is not None}), flush=True)
    if {r: sha(args.source_root / r) for r in source_files} != source_hashes:
        raise ValueError('source changed during measurement; no completed receipt')
    if sha(binary) != evidence['binary_sha256']:
        raise ValueError('binary changed during measurement')
    evidence['complete'] = True
    result_path.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
