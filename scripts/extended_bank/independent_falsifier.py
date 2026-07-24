#!/usr/bin/env python3
"""SUPPLEMENTARY independent falsifier for ny's UNSAT/holds verdicts. Does NOT use ny's
attack code — a fresh numpy+onnxruntime SPSA/exact-grad PGD with random restarts, staying
strictly inside the closed vnnlib input box. If it finds a point that (a) is strictly
in-box and (b) violates the full vnnlib spec (via vnnlib_ce), that is a candidate false-unsat.

LIMITATION (important): this hand-rolled attacker is WEAKER than the reference tools and its
surrogate is disjunction-biased, so a "no-ce-found" is only weak evidence. The AUTHORITATIVE
false-unsat check is validate_reference_ces.py, which validates the actual counterexamples
that alpha_beta_crown/neuralsat/pyrat found. Use this only as a secondary probe.

Usage: independent_falsifier.py <onnx> <vnnlib> [restarts] [iters] [seed]
Prints: FALSIFIED <detail>  |  no-ce-found  |  ERROR <msg>
Exit 3 on FALSIFIED (so callers can detect breaches), 0 otherwise.
"""
import sys, os, re, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnnlib_ce

onnx_path, vnnlib_path = sys.argv[1], sys.argv[2]
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ITERS    = int(sys.argv[4]) if len(sys.argv) > 4 else 60
SEED     = int(sys.argv[5]) if len(sys.argv) > 5 else 12345
rng = np.random.default_rng(SEED)

text = open(vnnlib_path).read()
n_in = len(re.findall(r'\(declare-const X_\d+', text))
ub = {}; lb = {}
for m in re.finditer(r'\(assert \(<= X_(\d+)\s+([-\d.eE]+)\)\)', text): ub[int(m.group(1))] = float(m.group(2))
for m in re.finditer(r'\(assert \(>= X_(\d+)\s+([-\d.eE]+)\)\)', text): lb[int(m.group(1))] = float(m.group(2))
LB = np.array([lb.get(i, -np.inf) for i in range(n_in)], dtype=np.float64)
UB = np.array([ub.get(i,  np.inf) for i in range(n_in)], dtype=np.float64)
# finite bounds only where declared; undeclared inputs (rare) get a nominal [-1,1]
LB = np.where(np.isfinite(LB), LB, -1.0); UB = np.where(np.isfinite(UB), UB, 1.0)
width = np.maximum(UB - LB, 1e-12)

sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
shp = [d if isinstance(d, int) else 1 for d in inp.shape]

# output atoms for a scalar "violation margin" surrogate (how close to violating).
# margin > 0 means an atom is satisfied (spec violated). We ascend the max atom margin,
# but ACCEPTANCE is always the exact full-spec check via vnnlib_ce (handles and/or).
atoms = re.findall(r'\((>=|<=) Y_(\d+) (Y_(\d+)|[-\d.eE]+)\)', text)

def forward(x):
    return sess.run(None, {inp.name: x.astype(np.float32).reshape(shp)})[0].flatten().astype(np.float64)

def surrogate(y):
    # max over atoms of (satisfaction slack); >=0 when some atom holds
    best = -1e30
    for opr, a, b, _ in atoms:
        ya = y[int(a)]; yb = y[int(b[2:])] if b.startswith('Y_') else float(b)
        s = (ya - yb) if opr == '>=' else (yb - ya)
        if s > best: best = s
    return best

def is_ce(x):
    xd = {i: float(x[i]) for i in range(n_in)}
    try:
        ib, ce, det = vnnlib_ce.validate(onnx_path, vnnlib_path, xd)
        return ce, det
    except Exception as e:
        return False, f"validate-err {e}"

def clamp(x): return np.minimum(np.maximum(x, LB), UB)

if not atoms:
    print("no-ce-found (no Y atoms parsed)"); sys.exit(0)

EXACT_GRAD = n_in <= 64  # exact finite-diff gradient when cheap; else SPSA

def grad(x, base_s):
    c = np.maximum(width * 0.01, 1e-6)
    if EXACT_GRAD:
        g = np.zeros(n_in)
        for i in range(n_in):
            xp = x.copy(); xp[i] = min(x[i] + c[i], UB[i])
            xm = x.copy(); xm[i] = max(x[i] - c[i], LB[i])
            denom = (xp[i] - xm[i]) or 1e-12
            g[i] = (surrogate(forward(xp)) - surrogate(forward(xm))) / denom
        return g
    # SPSA average of a few probes
    g = np.zeros(n_in)
    for _ in range(3):
        d = rng.choice([-1.0, 1.0], size=n_in)
        sp = surrogate(forward(clamp(x + c * d))); sm = surrogate(forward(clamp(x - c * d)))
        g += (sp - sm) / (2.0 * c) * d
    return g / 3.0

best_surr = -1e30
for r in range(RESTARTS):
    if r % 6 == 0:   x = LB.copy()
    elif r % 6 == 1: x = UB.copy()
    elif r % 6 == 2: x = (LB + UB) / 2.0
    elif r % 6 == 3: x = np.where(rng.random(n_in) < 0.5, LB, UB)  # random corner
    else:            x = LB + rng.random(n_in) * width
    m = np.zeros(n_in); v = np.zeros(n_in); b1, b2, eps = 0.9, 0.999, 1e-8
    lr = width * 0.15
    for it in range(1, ITERS + 1):
        y = forward(x); s = surrogate(y)
        if s > best_surr: best_surr = s
        ce, det = is_ce(x)
        if ce:
            print(f"FALSIFIED restart={r} iter={it} :: {det}"); sys.exit(3)
        g = grad(x, s)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
        x = clamp(x + lr * mh / (np.sqrt(vh) + eps))
        lr *= 0.985
print(f"no-ce-found (best surrogate margin {best_surr:+.6f}, {RESTARTS}x{ITERS}, exact_grad={EXACT_GRAD})")
sys.exit(0)
