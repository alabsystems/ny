#!/usr/bin/env python3
"""Sign/BNN falsifier for traffic_signs_recognition_2023. The net is a binarized net
(binary conv/dense weights, Sign activations) -> piecewise constant, zero gradient, so
PGD on the true net fails. Technique: a differentiable SOFT-SIGN surrogate tanh(alpha*.)
whose alpha is ramped up toward the true Sign, giving a usable gradient; every iterate is
checked against the TRUE net (onnxruntime) for a genuine in-box counterexample.

Reconstructs the exact forward (conv1 3x3 stride1 valid, Sign, conv2 2x2 stride1 valid,
Sign, flatten, matmul, softmax; the Sign(Sign(x)+0.1) pairs collapse to one Sign). All
weights are +/-1. Input is NHWC [30,30,3] flattened C-order to X_0..X_2699.

Usage: bnn_falsifier.py <onnx> <vnnlib> [restarts] [iters] [seed]  -> FALSIFIED/no-ce
"""
import sys, os, re, numpy as np, onnxruntime as ort
from onnx import numpy_helper
import onnx as onnxmod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnnlib_ce

ONNX, VNNLIB = sys.argv[1], sys.argv[2]
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 120
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(SEED)

# ---- weights (net-1 architecture: 30x30, conv 16/3x3 + conv 32/2x2 + dense; by shape) ----
m = onnxmod.load(ONNX)
by_shape = {}
for i in m.graph.initializer:
    a = numpy_helper.to_array(i)
    by_shape.setdefault(a.shape, a)
try:
    W1 = by_shape[(16, 3, 3, 3)].astype(np.float64)
    W2 = by_shape[(32, 16, 2, 2)].astype(np.float64)
    WD = by_shape[(23328, 43)].astype(np.float64)
except KeyError:
    print(f"skip: {os.path.basename(ONNX)} is not the net-1 (30x30 2-conv) architecture this falsifier handles")
    sys.exit(0)

def im2col(x, KH, KW):  # x [C,H,W] -> [C*KH*KW, OH*OW]
    C, H, Wd = x.shape; OH, OW = H-KH+1, Wd-KW+1
    cols = np.empty((C, KH, KW, OH, OW))
    for ki in range(KH):
        for kj in range(KW):
            cols[:, ki, kj] = x[:, ki:ki+OH, kj:kj+OW]
    return cols.reshape(C*KH*KW, OH*OW), OH, OW

def conv_fwd(x, w):  # x [C,H,W], w [OC,IC,KH,KW]
    OC, IC, KH, KW = w.shape
    cols, OH, OW = im2col(x, KH, KW)
    out = (w.reshape(OC, -1) @ cols).reshape(OC, OH, OW)
    return out, cols, OH, OW

def conv_bwd(gout, w, cols_shape_in, OH, OW):
    # gout [OC,OH,OW]; returns grad wrt input [C,H,W]
    OC, IC, KH, KW = w.shape
    C, H, Wd = cols_shape_in
    gcols = (w.reshape(OC, -1).T @ gout.reshape(OC, -1)).reshape(C, KH, KW, OH, OW)
    gin = np.zeros((C, H, Wd))
    for ki in range(KH):
        for kj in range(KW):
            gin[:, ki:ki+OH, kj:kj+OW] += gcols[:, ki, kj]
    return gin

def forward(x_nhwc, alpha, hard=False):
    """x_nhwc [30,30,3] -> logits[43]; returns (logits, cache) for backward (soft only)."""
    x = np.transpose(x_nhwc, (2, 0, 1))  # NCHW [3,30,30]
    c1, cols1, OH1, OW1 = conv_fwd(x, W1)  # [16,28,28]
    s1 = np.sign(c1) if hard else np.tanh(alpha*c1)
    s1c = np.transpose(s1, (1, 2, 0))  # NHWC [28,28,16]
    x2 = np.transpose(s1c, (2, 0, 1))  # NCHW [16,28,28]
    c2, cols2, OH2, OW2 = conv_fwd(x2, W2)  # [32,27,27]
    s2 = np.sign(c2) if hard else np.tanh(alpha*c2)
    s2c = np.transpose(s2, (1, 2, 0))  # NHWC [27,27,32]
    flat = s2c.reshape(-1)  # 23328, C-order NHWC
    s3 = np.sign(flat) if hard else np.tanh(alpha*flat)
    logits = WD.T @ s3  # [43]
    cache = (x, c1, OH1, OW1, c2, OH2, OW2, s2c.shape, flat, alpha)
    return logits, cache

def backward(gl, cache):  # gl grad wrt logits [43] -> grad wrt x_nhwc [30,30,3]
    x, c1, OH1, OW1, c2, OH2, OW2, s2shape, flat, alpha = cache
    gs3 = WD @ gl  # [23328]
    gflat = gs3 * alpha*(1-np.tanh(alpha*flat)**2)
    gs2c = gflat.reshape(s2shape)  # NHWC [27,27,32]
    gc2 = np.transpose(gs2c, (2, 0, 1))  # [32,27,27]
    gc2 = gc2 * alpha*(1-np.tanh(alpha*c2)**2)
    gx2 = conv_bwd(gc2, W2, (16, 28, 28), OH2, OW2)  # [16,28,28]
    gs1c = np.transpose(gx2, (1, 2, 0))  # NHWC [28,28,16]
    gc1 = np.transpose(gs1c, (2, 0, 1))  # [16,28,28]
    gc1 = gc1 * alpha*(1-np.tanh(alpha*c1)**2)
    gx = conv_bwd(gc1, W1, (3, 30, 30), OH1, OW1)  # [3,30,30]
    return np.transpose(gx, (1, 2, 0))  # NHWC [30,30,3]

# ---- box + spec ----
text = open(VNNLIB).read(); n = len(re.findall(r'\(declare-const X_\d+', text))
ub = {}; lb = {}
for mm in re.finditer(r'\(assert \(<= X_(\d+)\s+([-\d.eE]+)\)\)', text): ub[int(mm.group(1))] = float(mm.group(2))
for mm in re.finditer(r'\(assert \(>= X_(\d+)\s+([-\d.eE]+)\)\)', text): lb[int(mm.group(1))] = float(mm.group(2))
LB = np.array([lb.get(i, 0.0) for i in range(n)]).reshape(30, 30, 3)
UB = np.array([ub.get(i, 1.0) for i in range(n)]).reshape(30, 30, 3)
# true class: the vnnlib asserts (>= Y_j Y_C) for j != C  (disjunction). Find C = the index
# that appears on the RHS of every atom.
atoms = re.findall(r'\(>= Y_(\d+) Y_(\d+)\)', text)
from collections import Counter
rhs = Counter(int(b) for _, b in atoms)
TRUE_C = rhs.most_common(1)[0][0] if rhs else 0
others = sorted({int(a) for a, _ in atoms})

sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
def ort_logits(x_nhwc):
    return sess.run(None, {inp.name: x_nhwc.astype(np.float32).reshape(1, 30, 30, 3)})[0].flatten()
def is_ce(x_nhwc):
    xd = {i: float(x_nhwc.reshape(-1)[i]) for i in range(n)}
    ib, ce, det = vnnlib_ce.validate(ONNX, VNNLIB, xd)
    return ce, det

def clamp(x): return np.minimum(np.maximum(x, LB), UB)
mid = (LB+UB)/2; W = np.maximum(UB-LB, 1e-9)

for r in range(RESTARTS):
    if r == 0: x = mid.copy()
    elif r % 3 == 1: x = clamp(mid + (rng.random(mid.shape)-0.5)*W)
    else: x = LB + rng.random(LB.shape)*W
    # pick a target: the runner-up class under ORT at the start
    ol = ort_logits(x); tgt = int(np.argsort(ol)[::-1][0] if np.argmax(ol) != TRUE_C else np.argsort(ol)[::-1][1])
    lr = W*0.3
    for it in range(ITERS):
        alpha = 2.0 + 18.0*it/ITERS  # ramp 2 -> 20
        # objective: maximize logit[tgt] - logit[TRUE_C] on the surrogate
        logits, cache = forward(x, alpha)
        gl = np.zeros(43); gl[tgt] = 1.0; gl[TRUE_C] = -1.0
        g = backward(gl, cache)
        x = clamp(x + lr*np.sign(g))
        lr *= 0.99
        if it % 8 == 0 or it == ITERS-1:
            ce, det = is_ce(x)
            if ce:
                print(f"FALSIFIED restart={r} it={it} tgt={tgt} trueC={TRUE_C} :: {det[:60]}"); sys.exit(3)
        # also retarget occasionally to the current ORT runner-up
        if it % 20 == 19:
            ol = ort_logits(x); order = np.argsort(ol)[::-1]
            tgt = int(order[0] if order[0] != TRUE_C else order[1])
print(f"no-ce ({RESTARTS}x{ITERS}, trueC={TRUE_C})"); sys.exit(0)
