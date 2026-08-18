import re, collections
LOG='external_tools/vnncomp2025_results/SCORING-ZERO-TOL/results.txt'
idx=None; bench=None; holds=0
res=collections.defaultdict(dict); meta={}
for line in open(LOG, errors='replace'):
    m=re.match(r'Checking counterexamples\s*for index (\d+)\. Violated: (\d+) \((.*?)\), Holds: (\d+) \((.*?)\)', line)
    if m:
        idx=int(m.group(1)); holds=int(m.group(4)); continue
    m=re.match(r'checking counterexample for (\S+)', line)
    if m: tool=m.group(1).strip(); continue
    m=re.match(r'Checking ce path: \S+, (\S+)', line)
    if m: bench=m.group(1).strip(); continue
    m=re.match(r'CE result ([a-z_]+)', line)
    if m and idx is not None and bench:
        res[(bench,idx)][tool]=m.group(1); meta[(bench,idx)]=holds
pool=[k for k,v in res.items() if 'correct' not in v.values() and 'correct_up_to_tolerance' in v.values()]
disputed=[k for k in pool if meta.get(k,0)>0]
print(f"pooled (no strict CE anywhere)          : {len(pool)}")
print(f"  ...of those, some tool claims HOLDS   : {len(disputed)}   <- a strict NY CE would inflict -150 on each holds-claimer")
print(f"  ...no tool claims holds (undisputed)  : {len(pool)-len(disputed)}")
allc=[k for k,v in res.items()]
disp_all=[k for k in allc if meta.get(k,0)>0]
print(f"\nall CE-checked instances                : {len(allc)}")
print(f"  ...disputed (holds AND violated)      : {len(disp_all)}")
