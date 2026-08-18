import re, sys, collections
path = sys.argv[1]
tool = None; bench = None
tally = collections.defaultdict(lambda: collections.Counter())
per_bench = collections.defaultdict(lambda: collections.Counter())
for line in open(path, errors='replace'):
    m = re.match(r'checking counterexample for (\S+)', line)
    if m: tool = m.group(1).strip(); continue
    m = re.match(r'Checking ce path: \S+, (\S+)', line)
    if m: bench = m.group(1).strip(); continue
    m = re.match(r'CE result ([a-z_]+)', line)
    if m and tool:
        tally[tool][m.group(1)] += 1
        per_bench[(tool,bench)][m.group(1)] += 1
print(f"=== {path} ===")
print(f"{'tool':<20}{'correct':>9}{'up_to_tol':>11}{'not_vio':>9}{'exec_bad':>9}{'strict%':>9}")
for t, c in sorted(tally.items(), key=lambda kv: -sum(kv[1].values())):
    ok=c['correct']; tol=c['correct_up_to_tolerance']; nv=c['spec_not_violated']; eb=c['exec_doesnt_match']
    tot=ok+tol
    pct = f"{100.0*ok/tot:.1f}" if tot else "-"
    print(f"{t:<20}{ok:>9}{tol:>11}{nv:>9}{eb:>9}{pct:>9}")
