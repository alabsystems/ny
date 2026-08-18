"""Rows where the official field DISAGREES. Identity = (onnx, vnnlib) pair,
joined by ROW INDEX in each tool's results.csv (which follows instances.csv order)
— NOT vnnlib basename, which is not unique and produced a false alarm last time.

Repo root comes from this script's own location, or from $NY_REPO_ROOT."""
import csv, os, collections, pathlib
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
rows=[]
for d in sorted(os.listdir(OFF)):
    if not d.startswith('alpha_beta_crown'): continue
for cat_dir in sorted(os.listdir(f'{OFF}/alpha_beta_crown')):
    cat = cat_dir.replace('2025_','',1)
    per_tool={}
    for t in TOOLS:
        p=f'{OFF}/{t}/2025_{cat}/results.csv'
        if os.path.exists(p):
            per_tool[t]=[r for r in csv.reader(open(p)) if len(r)>=6]
    if 'alpha_beta_crown' not in per_tool: continue
    n=len(per_tool['alpha_beta_crown'])
    # NY bank keyed by (onnx basename, vnnlib basename)
    ny={}
    bank=f'{REPO}/reports/measured/{cat}.csv'
    if os.path.exists(bank):
        for r in csv.reader(open(bank)):
            if len(r)>=5: ny[(r[1].split('/')[-1], r[2].split('/')[-1])]=r[4].lower()
    for i in range(n):
        base=per_tool['alpha_beta_crown'][i]
        key=(base[1].split('/')[-1], base[2].split('/')[-1])
        if 'test_nano' in key[0] or 'test_tiny' in key[0]: continue
        sats=[];unsats=[]
        for t,tr in per_tool.items():
            if i>=len(tr): continue
            if tr[i][1].split('/')[-1]!=key[0] or tr[i][2].split('/')[-1]!=key[1]: continue  # alignment guard
            v=tr[i][4].lower()
            if v=='sat': sats.append((t,float(tr[i][5])))
            elif v=='unsat': unsats.append((t,float(tr[i][5])))
        if sats and unsats:
            rows.append((cat,key,sorted(sats,key=lambda x:x[1]),unsats,ny.get(key,'NO-ROW')))
print(f"SPLIT-FIELD INSTANCES (>=1 sat AND >=1 unsat): {len(rows)}\n")
byc=collections.Counter(r[0] for r in rows)
for c,k in byc.most_common(): print(f"  {k:>4}  {c}")
cand=[r for r in rows if r[4] not in ('sat','unsat')]
print(f"\nNY UNDECIDED on {len(cand)} of them  <- conversion candidates")
print(f"NY DECIDED on {len(rows)-len(cand)}  <- moat cross-check")
print("\nCANDIDATES, easiest CE first:")
for cat,key,sats,unsats,nyv in sorted(cand,key=lambda r:r[2][0][1])[:20]:
    print(f"  {sats[0][1]:7.2f}s {sats[0][0]:<16} {cat:<24} {key[1][:46]:<48} ny={nyv} unsat_by={[u[0] for u in unsats]}")
