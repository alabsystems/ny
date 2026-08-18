"""MOAT CHECK: rows where NY says `unsat` while some tool holds a STRICTLY-CORRECT
counterexample. If any exist, NY is provably wrong there.

Repo root comes from this script's own location, or from $NY_REPO_ROOT."""
import csv, os, re, collections, pathlib
def t2(p):
    q=[c for c in p.split('/') if c not in ('','.')]
    return '/'.join(q[-2:]) if len(q)>=2 else (q[-1] if q else p)
def repo_root():
    env=os.environ.get('NY_REPO_ROOT')
    root=pathlib.Path(env).expanduser().resolve() if env else pathlib.Path(__file__).resolve().parents[2]
    if not (root/'Cargo.toml').is_file():
        raise SystemExit(f"FATAL: {root} is not the ny repo (no Cargo.toml)"
                         f"{' — $NY_REPO_ROOT is wrong' if env else ''}. Set NY_REPO_ROOT to the checkout root.")
    return str(root)
REPO=repo_root(); OFF=f'{REPO}/external_tools/vnncomp2025_results'
if not os.path.isdir(OFF):
    raise SystemExit(f"FATAL: official VNN-COMP results tree missing: {OFF}\n"
                     "  This audit IS the organizers' results; without them it checks nothing.\n"
                     "  Fetch external_tools/vnncomp2025_results before running it.")
TOOLS=['alpha_beta_crown','neuralsat','pyrat','nnenum','sobolbox','cora','nnv']
# 1) organizer CE classification per (benchmark, instance-index, tool)
LOG=f'{OFF}/SCORING-ZERO-TOL/results.txt'
idx=None;bench=None;tool=None
ce=collections.defaultdict(dict)
for line in open(LOG,errors='replace'):
    m=re.match(r'Checking counterexamples\s*for index (\d+)',line)
    if m: idx=int(m.group(1)); continue
    m=re.match(r'checking counterexample for (\S+)',line)
    if m: tool=m.group(1).strip(); continue
    m=re.match(r'Checking ce path: \S+, (\S+)',line)
    if m: bench=m.group(1).strip(); continue
    m=re.match(r'CE result ([a-z_]+)',line)
    if m and idx is not None and bench: ce[(bench,idx)][tool]=m.group(1)
strict=collections.Counter(); wrong=[]
for cat_dir in sorted(d for d in os.listdir(f"{OFF}/alpha_beta_crown") if d.startswith("2025_") and os.path.isdir(f"{OFF}/alpha_beta_crown/{d}")):
    cat=cat_dir.replace('2025_','',1)
    base=[r for r in csv.reader(open(f'{OFF}/alpha_beta_crown/2025_{cat}/results.csv')) if len(r)>=6]
    ny={}
    bank=f'{REPO}/reports/measured/{cat}.csv'
    if os.path.exists(bank):
        for r in csv.reader(open(bank)):
            if len(r)>=5: ny[(t2(r[1]),t2(r[2]))]=r[4].lower()
    for i,b in enumerate(base):
        key=(t2(b[1]),t2(b[2]))
        if 'test_nano' in key[0] or 'test_tiny' in key[0]: continue
        cls=ce.get((f'2025_{cat}',i),{})
        strict_tools=[t for t,c in cls.items() if c=='correct']
        if not strict_tools: continue
        strict[cat]+=1
        nyv=ny.get(key)
        if nyv=='unsat':
            wrong.append((cat,key,strict_tools))
print(f"instances with >=1 STRICTLY-correct CE, by benchmark:")
for c,k in strict.most_common(): print(f"  {k:>5}  {c}")
print(f"\n*** NY says UNSAT where a STRICT counterexample exists: {len(wrong)} ***")
for cat,key,ts in wrong[:20]: print(f"  {cat:<22} {key[1][:52]:<54} strict_by={ts}")
if not wrong: print("  NONE — the moat holds against every strictly-validated field counterexample.")
