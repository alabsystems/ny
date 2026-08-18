import re, sys, csv, collections, os
LOG='<home>/ny/external_tools/vnncomp2025_results/SCORING-ZERO-TOL/results.txt'
BENCH='<home>/ny/benchmarks/vnncomp2025/benchmarks'
MEAS='<home>/ny/reports/measured'

# 1. pool: (bench, idx) with CEs but no strict one
tool=None; bench=None; idx=None
res=collections.defaultdict(dict)
for line in open(LOG, errors='replace'):
    m=re.match(r'Checking counterexamples\s*for index (\d+)', line)
    if m: idx=int(m.group(1)); continue
    m=re.match(r'checking counterexample for (\S+)', line)
    if m: tool=m.group(1).strip(); continue
    m=re.match(r'Checking ce path: \S+, (\S+)', line)
    if m: bench=m.group(1).strip(); continue
    m=re.match(r'CE result ([a-z_]+)', line)
    if m and idx is not None and bench: res[(bench,idx)][tool]=m.group(1)

pool=[k for k,v in res.items() if 'correct' not in v.values() and 'correct_up_to_tolerance' in v.values()]
print(f"pool size: {len(pool)}")

# 2. index -> vnnlib via instances.csv
def inst_list(b):
    cat=b.replace('2025_','',1)
    p=os.path.join(BENCH,cat,'instances.csv')
    if not os.path.exists(p): return None
    rows=[r for r in csv.reader(open(p)) if r]
    return [r[1].split('/')[-1] for r in rows]

# 3. ny verdict by vnnlib basename
def ny_map(b):
    cat=b.replace('2025_','',1)
    p=os.path.join(MEAS,cat+'.csv')
    if not os.path.exists(p): return None
    d={}
    for r in csv.reader(open(p)):
        if len(r)>=5: d[r[2].split('/')[-1]]=r[4]
    return d

agg=collections.defaultdict(collections.Counter)
missing=collections.Counter()
for b,i in pool:
    il=inst_list(b); nm=ny_map(b)
    if il is None or nm is None or i>=len(il): missing[b]+=1; continue
    v=nm.get(il[i])
    agg[b][v if v else 'NO-NY-ROW']+=1

print(f"{'benchmark':<38}{'ny=sat':>8}{'timeout':>9}{'unsat':>7}{'other':>7}")
tot=collections.Counter()
for b in sorted(agg, key=lambda x:-sum(agg[x].values())):
    c=agg[b]; o=sum(v for k,v in c.items() if k not in ('sat','timeout','unsat'))
    print(f"{b:<38}{c['sat']:>8}{c['timeout']:>9}{c['unsat']:>7}{o:>7}")
    tot['sat']+=c['sat']; tot['timeout']+=c['timeout']; tot['unsat']+=c['unsat']; tot['other']+=o
print(f"{'TOTAL':<38}{tot['sat']:>8}{tot['timeout']:>9}{tot['unsat']:>7}{tot['other']:>7}")
if missing: print("unjoinable:", dict(missing))
