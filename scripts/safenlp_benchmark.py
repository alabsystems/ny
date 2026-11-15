#!/usr/bin/env python3
"""Serial, whole-process SafeNLP development comparison. No proof or score claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import string
import struct
import subprocess
import sys
import time

ARMS = ('ny_off', 'ny_on', 'abc')
ENV_KEYS = {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TMPDIR', 'XDG_RUNTIME_DIR',
            'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
            'OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
            'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TOKENIZERS_PARALLELISM',
            'OMP_PROC_BIND', 'PYTHONPATH', 'XDG_CACHE_HOME'}
FIELDS = {'model', 'property', 'run_dir', 'result', 'witness', 'row_id', 'budget'}
ALIASES = {'sat': 'sat', 'violated': 'sat', 'unsafe': 'sat', 'unsafe-pgd': 'sat',
           'unsafe-bab': 'sat', 'unsat': 'unsat', 'verified': 'unsat', 'safe': 'unsat',
           'safe-incomplete': 'unsat', 'safe-complete': 'unsat', 'unknown': 'unknown',
           'timeout': 'timeout'}
STATUS = re.compile(r'^\s*(?:(?:final result|verification result|result)\s*:\s*)?'
                    r'(unsat|sat|unknown|timeout|verified|violated|safe(?:-incomplete|-complete)?|'
                    r'unsafe(?:-pgd|-bab)?)(?:\s+in\s+[0-9.eE+-]+\s+seconds?\.?)?\s*$', re.I)
BUSY = re.compile(r'^(?:ny(?:[-_].*)?|cargo|targo|rustc|trustc|nvcc|ninja|cmake|cc1plus|cc1|clang.*|gcc.*)$')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def git(path, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(path), *args],
                                   text=True, stderr=subprocess.PIPE).strip()


def check_identities(plan):
    observed = []
    for item in plan['identities']:
        path = Path(item['path'])
        actual = sha(path)
        require(actual == item['sha256'], 'identity changed: ' + str(path))
        observed.append(dict(path=str(path), resolved=str(path.resolve()), sha256=actual,
                             bytes=path.stat().st_size, role=item['role']))
    for row in plan['rows']:
        for kind in ('model', 'property'):
            require(sha(row[kind]) == row[kind + '_sha256'], 'row input changed: ' + row['id'])
    for source in plan['sources']:
        require(git(source['path'], 'rev-parse', 'HEAD') == source['head'], 'source HEAD changed')
        require(source['clean'] is True and not git(source['path'], 'status', '--porcelain'),
                'source checkout is dirty: ' + source['path'])
    return observed


def validate_plan(plan):
    require(plan['schema'] == 'ny_safenlp_benchmark_v1', 'wrong plan schema')
    budget = plan['budget_seconds']
    require(isinstance(budget, (int, float)) and math.isfinite(budget) and budget > 0, 'invalid budget')
    require(type(plan['repetitions']) is int and 1 <= plan['repetitions'] <= 30, 'invalid repetitions')
    require(set(plan['arms']) == set(ARMS), 'exactly ny_off, ny_on and abc are required')
    require(plan['arms']['ny_off']['argv'][0] == plan['arms']['ny_on']['argv'][0], 'NY OFF/ON must share one binary')
    require(set(plan['environment']) <= ENV_KEYS and all(isinstance(v, str) and '\0' not in v
            for v in plan['environment'].values()), 'environment must be explicitly whitelisted')
    limits = plan['resource_limits']
    require(set(limits) == {'cpu_affinity', 'address_space_bytes', 'max_open_files'}, 'resource-limit schema')
    require(limits['cpu_affinity'] and all(type(v) is int for v in limits['cpu_affinity'])
            and set(limits['cpu_affinity']) <= os.sched_getaffinity(0), 'invalid CPU affinity')
    require(limits['address_space_bytes'] is None or type(limits['address_space_bytes']) is int
            and limits['address_space_bytes'] > 0, 'invalid address-space limit')
    require(type(limits['max_open_files']) is int and limits['max_open_files'] >= 64, 'invalid descriptor limit')
    require(plan['identities'] and plan['sources'] and plan['rows'], 'missing identity, source or input pins')
    pinned = {str(Path(p['path'])) for p in plan['identities']}
    for item in plan['identities']:
        require(Path(item['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}', item['sha256'])
                and isinstance(item['role'], str), 'invalid identity')
    for arm in plan['arms'].values():
        require(Path(arm['cwd']).is_absolute() and Path(arm['cwd']).is_dir(), 'invalid working directory')
        require(arm.get('status_source', 'result') == 'result', 'a text result artifact is required; stdout is diagnostic')
        require(arm['argv'] and arm['argv'][0] in pinned and Path(arm['argv'][0]).is_absolute(), 'unpinned executable')
        for arg in arm['argv']:
            require(isinstance(arg, str) and '\0' not in arg, 'invalid argv')
            for _, field, spec, conversion in string.Formatter().parse(arg):
                require(field is None or field in FIELDS and not spec and conversion is None, 'invalid argv template')
    require(len({r['id'] for r in plan['rows']}) == len(plan['rows']), 'duplicate row IDs')
    for row in plan['rows']:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', row['id']) is not None, 'invalid row ID')
        for kind in ('model', 'property'):
            require(Path(row[kind]).is_absolute() and re.fullmatch('[0-9a-f]{64}', row[kind + '_sha256']), 'invalid row pin')


def sequence(rows, repetitions):
    """Latin orders, with reverse direction on alternating rows; no outcome-based selection."""
    orders = [ARMS, ARMS[1:] + ARMS[:1], ARMS[2:] + ARMS[:2]]
    for rep in range(repetitions):
        for index, row in enumerate(rows):
            order = orders[(index + rep) % 3]
            if index % 2:
                order = tuple(reversed(order))
            for arm in order:
                yield row, rep, arm


def census():
    """Sample Linux process identities/RSS; never inspect environment or print command lines."""
    found = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
            left, rest = raw.rsplit(')', 1)
            fields = rest.split()
            found[int(entry.name)] = dict(name=left.split('(', 1)[1], state=fields[0],
                ppid=int(fields[1]), pgid=int(fields[2]), rss=max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            continue
    return found


def ancestors(snapshot):
    result, pid = set(), os.getpid()
    while pid and pid not in result:
        result.add(pid)
        pid = snapshot.get(pid, {}).get('ppid', 0)
    return result


def monitor(snapshot, pgid, exempt):
    own = [v for v in snapshot.values() if v['pgid'] == pgid and v['state'] != 'Z']
    busy = [{'pid': p, 'name': v['name']} for p, v in snapshot.items()
            if p not in exempt and v['pgid'] != pgid and v['state'] != 'Z' and BUSY.fullmatch(v['name'])]
    return bool(own), sum(v['rss'] for v in own), busy


def child_limits(limits):
    os.sched_setaffinity(0, limits['cpu_affinity'])
    if limits['address_space_bytes'] is not None:
        resource.setrlimit(resource.RLIMIT_AS, (limits['address_space_bytes'],) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits['max_open_files'],) * 2)


def terminate_group(pgid):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + 0.2
    while time.monotonic() < end:
        if not any(v['pgid'] == pgid and v['state'] != 'Z' for v in census().values()):
            return
        time.sleep(0.01)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    end = time.monotonic() + 0.2
    while time.monotonic() < end:
        if not any(v['pgid'] == pgid and v['state'] != 'Z' for v in census().values()):
            return
        time.sleep(0.01)


def execute(argv, cwd, env, limits, directory, budget, sample_seconds=0.1):
    """Deadline covers startup through complete group quiescence; forced cleanup is separate."""
    before = census()
    exempt = ancestors(before)
    _, _, busy = monitor(before, -1, exempt)
    require(not busy, 'outside compiler/NY process present before run: ' + repr(busy))
    started = time.monotonic()
    deadline = started + budget
    proc = usage = None
    timed_out = interrupted = False
    peak = samples = 0
    contamination = {}
    error = None
    ended = started
    cleanup_seconds = 0.0
    previous = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with (directory / 'stdout.log').open('xb') as out, (directory / 'stderr.log').open('xb') as err:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err,
                start_new_session=True, shell=False, preexec_fn=lambda: child_limits(limits))
            while True:
                if proc.returncode is None:
                    waited, status, observed = os.wait4(proc.pid, os.WNOHANG)
                    if waited:
                        usage = observed
                        proc.returncode = os.waitstatus_to_exitcode(status)
                now = time.monotonic()
                snapshot = census()
                live, rss, busy = monitor(snapshot, proc.pid, exempt)
                peak, samples = max(peak, rss), samples + 1
                contamination.update((v['pid'], v) for v in busy)
                ended = time.monotonic()
                if ended > deadline:
                    timed_out = True
                    break
                if proc.returncode is not None and not live:
                    break
                time.sleep(min(sample_seconds, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
        ended = time.monotonic()
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
        ended = time.monotonic()
    finally:
        cleanup_start = time.monotonic()
        if proc is not None:
            if timed_out or interrupted or error:
                terminate_group(proc.pid)
            if proc.returncode is None:
                reap_end = time.monotonic() + 0.2
                while time.monotonic() < reap_end:
                    waited, status, observed = os.wait4(proc.pid, os.WNOHANG)
                    if waited:
                        usage = observed
                        proc.returncode = os.waitstatus_to_exitcode(status)
                        break
                    time.sleep(0.01)
        cleanup_seconds = time.monotonic() - cleanup_start
        signal.signal(signal.SIGTERM, previous)
    after = census()
    leftovers, _, busy = monitor(after, -1 if proc is None else proc.pid, exempt)
    contamination.update((v['pid'], v) for v in busy)
    return dict(argv=argv, pid=None if proc is None else proc.pid, returncode=None if proc is None else proc.returncode,
        process_seconds=ended - started, cleanup_seconds=cleanup_seconds, timeout_seconds=budget,
        timed_out=timed_out, interrupted=interrupted, launch_error=error, live_group_after_cleanup=leftovers,
        contaminated=bool(contamination), outside_processes=list(contamination.values()),
        contamination_scope='sampled outside compiler/NY process names only; not general Python/CPU/GPU load',
        sampled_group_peak_rss_bytes=peak, rss_samples=samples, sampling_scope='Linux live process group; sampled, not exact peak',
        wait4_usage=None if usage is None else dict(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
            max_rss_bytes=usage.ru_maxrss * 1024, scope='leader and its waited descendants; not simultaneous group peak'))


def statuses(text):
    values = []
    for line in text.splitlines():
        match = STATUS.fullmatch(line)
        if match:
            values.append(ALIASES[match.group(1).lower()])
    return set(values)


def result_status(path):
    if path.is_symlink():
        return None, 'symlink result artifact refused'
    if not path.is_file():
        return None, 'missing result artifact'
    try:
        text = path.read_text()
        if text.lstrip().startswith(('{', '[', '"')):
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    require(key not in value, 'duplicate JSON key')
                    value[key] = item
                return value
            def nonfinite(value):
                raise ValueError('nonfinite JSON: ' + value)
            value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
            if isinstance(value, dict):
                fields = [value[k] for k in ('status', 'result', 'verdict') if k in value]
                require(fields and all(isinstance(v, str) for v in fields), 'invalid JSON status')
                found = set(ALIASES.get(v.lower(), 'invalid') for v in fields)
            else:
                found = {ALIASES.get(value.lower(), 'invalid')} if isinstance(value, str) else {'invalid'}
        else:
            first = next((x.strip() for x in text.splitlines() if x.strip()), '')
            require(STATUS.fullmatch(first) is not None, 'result must begin with a verdict')
            found = statuses(text)
        require(len(found) == 1 and 'invalid' not in found, 'malformed or conflicting result status')
        return found.pop(), None
    except (OSError, UnicodeError, ValueError) as error:
        return None, str(error)


def classify(execution, result, stdout, stderr, status_source='result'):
    reported, issue = result_status(result)
    logged = statuses(stdout) | statuses(stderr)
    if status_source != 'result':
        issue = 'stdout verdict acceptance is disabled'
    elif reported is not None and logged and logged != {reported}:
        issue = 'result artifact conflicts with log statuses'
    cached = bool(re.search(r'cache:\s*[1-9][0-9]*\s+served|["\']from_cache["\']\s*:\s*true', stdout + stderr, re.I))
    invalid = execution['launch_error'] or execution['interrupted'] or execution['contaminated'] or execution['live_group_after_cleanup'] or cached
    if invalid:
        outcome = 'error'
    elif execution['timed_out'] or execution['process_seconds'] > execution['timeout_seconds']:
        outcome = 'timeout'
    elif execution['returncode'] != 0 or issue:
        outcome = 'error'
    else:
        outcome = {'sat': 'claimed_sat', 'unsat': 'claimed_unsat', 'unknown': 'open', 'timeout': 'timeout'}.get(reported, 'error')
    return dict(outcome=outcome, reported_status=reported, status_issue=issue, cached_verdict_detected=cached,
        valid_for_comparison=outcome != 'error', independently_validated_sat=False,
        independent_unsat_proof=False, witness_postprocessing='required externally for every claimed_sat')


def diagnostics(text):
    tags = ('[input-head', '[g3]', '[cover-gate]', '[frontier]', '[phase]', '[root-', '[gpu-', '[cuda')
    lines = [line for line in text.splitlines() if any(tag in line for tag in tags)]
    def decoded(bits):
        value = struct.unpack('>d', bytes.fromhex(bits))[0]
        return value if math.isfinite(value) else None
    deltas = [dict(before=decoded(a), after=decoded(b), before_bits=a, after_bits=b)
              for a, b in re.findall(r'\[input-head-triangle-delta\] before_bits=([0-9a-f]{16}) after_bits=([0-9a-f]{16})', text)]
    groups = [dict(rows=int(a), certified_masks=int(b)) for a, b in re.findall(
        r'\[input-head-large-negative-group-replay\] rows=(\d+) certified_masks=(\d+)', text)]
    roots = [dict(status=s, lower=decoded(bits) if bits != 'none' else None, lower_bits=bits,
                  seconds=float(seconds)) for s, bits, seconds in re.findall(
        r'\[input-head\] status=(\w+) classes=1 calls=\d+ replayed=\d+ lower_bits=(none|[0-9a-f]{16}) elapsed_s=([0-9.]+)', text)]
    return dict(lines=lines, source_admission_observed='[input-head-admission]' in text,
        source_bridge_complete_observed=bool(re.search(r'\[input-head-bridge\] status=complete\b', text)),
        large_group_stage_observed=bool(groups), large_group_replay=groups,
        root_bound_deltas=deltas, source_root_observations=roots,
        engine_domains_explored_observations=[int(v) for v in re.findall(r'"domains_explored"\s*:\s*(\d+)', text)],
        native_child_engagement='unavailable: no dedicated producer telemetry',
        source_owned_contribution='not inferred; requires original Instance/G2/G3 adjudication')


def private_inputs(row, directory):
    inputs = directory / 'inputs'
    inputs.mkdir()
    result = {}
    for kind, filename in [('model', 'model.onnx'), ('property', 'property.vnnlib')]:
        target = inputs / filename
        shutil.copyfile(row[kind], target)
        require(sha(target) == row[kind + '_sha256'], 'private input copy differs')
        result[kind] = str(target)
    return result


def artifacts(directory):
    paths = list(directory.rglob('*'))
    links = [str(p.relative_to(directory)) for p in paths if p.is_symlink()]
    inventory = {str(p.relative_to(directory)): (dict(kind='symlink', target=os.readlink(p)) if p.is_symlink()
                 else dict(sha256=sha(p), bytes=p.stat().st_size)) for p in paths if p.is_symlink() or p.is_file()}
    return inventory, links


def run(plan, output):
    require(os.environ.get('NY_SAFENLP_BOX_PROVENANCE') == 'ny_box_measure', 'run inside root-owned ny_box measure wrapper')
    validate_plan(plan)
    initial = check_identities(plan)
    output.mkdir(parents=False, exist_ok=False)
    immutable_json(output / 'plan.json', plan)
    immutable_json(output / 'identities-before.json', initial)
    results, refusal, final_error = [], None, None
    interrupted = False
    for index, (row, repetition, arm_name) in enumerate(sequence(plan['rows'], plan['repetitions'])):
        try:
            check_identities(plan)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            refusal = str(error)
            break
        directory = output / f'{index:03d}-{row["id"]}-{arm_name}-r{repetition}'
        directory.mkdir()
        copied = private_inputs(row, directory)
        cache = directory / 'cache'
        cache.mkdir()
        env = dict(plan['environment'], PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', XDG_CACHE_HOME=str(cache))
        values = dict(**copied, row_id=row['id'],
            run_dir=str(directory), result=str(directory / 'result.txt'), witness=str(directory / 'witness.txt'),
            budget=str(plan['budget_seconds']))
        arm = plan['arms'][arm_name]
        argv = [a.format_map(values) for a in arm['argv']]
        started = dict(row_id=row['id'], arm=arm_name, repetition=repetition, argv=argv,
                       resource_limits=plan['resource_limits'], environment=env)
        immutable_json(directory / 'launch.json', started)
        try:
            execution = execute(argv, arm['cwd'], env, plan['resource_limits'], directory, plan['budget_seconds'])
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            refusal = str(error)
            immutable_json(directory / 'refusal.json', dict(reason=refusal, solver_launched=False))
            break
        stdout, stderr = [(directory / n).read_text(errors='replace') for n in ('stdout.log', 'stderr.log')]
        result = classify(execution, directory / 'result.txt', stdout, stderr, arm.get('status_source', 'result'))
        identity_error = None
        try:
            check_identities(plan)
            for kind, path in copied.items():
                require(sha(path) == row[kind + '_sha256'], 'solver changed private original input')
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            identity_error = str(error)
            result.update(outcome='error', valid_for_comparison=False)
        inventory, links = artifacts(directory)
        if links:
            result.update(outcome='error', valid_for_comparison=False)
        record = dict(**started, execution=execution, result=result, identity_error=identity_error,
                      diagnostics=diagnostics(stdout + stderr), private_inputs=copied,
                      original_inputs={k: row[k] for k in ('model', 'property', 'model_sha256', 'property_sha256')},
                      row_metadata=row.get('metadata', {}), selection_role=row.get('role', 'development'),
                      artifacts=inventory, refused_symlink_artifacts=links)
        immutable_json(directory / 'measurement.json', record)
        results.append(dict(directory=directory.name, row_id=row['id'], arm=arm_name, repetition=repetition, **result))
        print(json.dumps(results[-1]), flush=True)
        interrupted = execution['interrupted']
        if interrupted or execution['contaminated'] or execution['live_group_after_cleanup'] or identity_error or links:
            break
    try:
        immutable_json(output / 'identities-after.json', check_identities(plan))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        final_error = str(error)
        immutable_json(output / 'identities-after-error.json', dict(reason=final_error))
    expected = len(plan['rows']) * len(ARMS) * plan['repetitions']
    rc = 0 if len(results) == expected and not refusal and not final_error and all(r['valid_for_comparison'] for r in results) else 1
    if interrupted:
        rc = 130
    immutable_json(output / 'summary.json', dict(schema='ny_safenlp_benchmark_result_v1', complete=rc == 0,
        refusal_reason=refusal, final_identity_error=final_error, attempted_runs=len(results), planned_runs=expected,
        returncode=rc, results=results, box_provenance_attestation='ny_box_measure; root must retain actual wrapper log',
        selection='row roles are declared in frozen plan; diagnostic/structural selection is not a competition score',
        official_score_claim=False, independent_unsat_proof=False,
        timing_scope='process start through observed group quiescence; forced cleanup recorded separately',
        cache_policy='prehashed warm OS page cache; fresh process, private input copies and private XDG artifact cache per query'))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        require(sys.platform == 'linux', 'Linux process census/resource semantics required')
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        if args.check_only:
            check_identities(plan)
            print(json.dumps(dict(valid=True, planned_runs=len(plan['rows']) * 3 * plan['repetitions'])))
            return 0
        require(args.output is not None and args.output.is_absolute(), '--output must be a new absolute directory')
        return run(plan, args.output)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
