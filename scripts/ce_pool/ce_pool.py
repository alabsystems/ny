import re, sys, collections
path = sys.argv[1]
tool=None; bench=None; inst=None
# instance -> {tool: result}
inst_res = collections.defaultdict(dict)
for line in open(path, errors='replace'):
    m = re.match(r'checking counterexample for (\S+)', line)
    if m: tool = m.group(1).strip(); continue
    m = re.match(r'Checking ce path: (\S+), (\S+)', line)
    if m:
        p = m.group(1); bench = m.group(2).strip()
        inst = (bench, p.rsplit('/',1)[-1])
        continue
    m = re.match(r'CE result ([a-z_]+)', line)
    if m and tool and inst:
        inst_res[inst][tool] = m.group(1)

tot = len(inst_res)
has_strict = sum(1 for v in inst_res.values() if 'correct' in v.values())
only_tol  = sum(1 for v in inst_res.values() if 'correct' not in v.values()
                and 'correct_up_to_tolerance' in v.values())
print(f"instances with at least one submitted CE : {tot}")
print(f"  ... at least one STRICTLY correct CE   : {has_strict}")
print(f"  ... ONLY tolerance-only CEs (GT-unknown, scores 0 for EVERYONE): {only_tol}")
print()
print("GT-UNKNOWN POOL BY BENCHMARK (addressable by a strict CE):")
per = collections.Counter()
for (b,_),v in inst_res.items():
    if 'correct' not in v.values() and 'correct_up_to_tolerance' in v.values():
        per[b]+=1
for b,c in per.most_common():
    print(f"  {c:>5}  {b}")
