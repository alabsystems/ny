#!/usr/bin/env python3
"""Soft-sign BNN falsifier for the LARGER traffic_signs_recognition_2023 net
   3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43.

The net is a binarized net (binary +/-1 conv/dense weights, Sign activations) -> piecewise
constant, zero gradient, so PGD on the true net has no usable signal. Technique (extends
bnn_falsifier.py from the 30x30 2-conv net): a differentiable SOFT-SIGN surrogate whose
sharpness alpha is ramped up toward the true Sign gives a usable gradient; every iterate is
checked against the TRUE net (onnxruntime) + vnnlib_ce.validate for a genuine in-box CE.

ONNX graph (studied node-by-node) -> reconstructed forward (all in NCHW; BN/Sign per-chan):
  Transpose NHWC->NCHW
  Conv1  5x5 valid, BINARY +/-1 weights            -> [32,44,44]
  MaxPool 2x2/2                                     -> [32,22,22]
  BatchNorm3 (scale=1, beta, mean, var; eps=1e-3)   affine per-channel
  Sign  (soft: tanh)                                            -- Sign(Sign(x)+0.1) collapses
  Conv2  5x5 valid, BINARY +/-1 weights            -> [64,18,18]
  MaxPool 2x2/2                                     -> [64,9,9]
  BatchNorm4 (scale=1, beta, mean, var; eps=1e-3)   affine per-channel
  Sign  (soft)
  Conv3  3x3 valid, REAL weights (BN5 folded in) + bias  -> [64,7,7]
  flatten NHWC (transpose->[7,7,64]->3136)
  Sign  (soft)
  Dense1 3136x256, BINARY +/-1 weights             -> [256]
  BatchNorm6 (Mul by rsqrt, Add sub) affine
  Sign  (soft)
  Dense2 256x43, BINARY +/-1 weights               -> [43]
  Softmax (monotone; CE ordering handled by ORT check)

Extra ops vs net-1 handled here:
  - MaxPool  : forward = window max; backward = route-to-max subgradient (scatter to argmax)
  - BatchNorm: folded to per-channel affine z*(scale/sqrt(var+eps)) + (beta - mean*...)
  - 3 convs (incl. a real-weight fused-BN conv3 with bias) + 2 dense
  - RMS-normalised soft-sign tanh(alpha * z / rms(z))  (rms detached) so the gradient survives
    through 4 stacked saturating Sign layers; dividing by a positive scalar preserves sign().

Usage: bnn_falsifier_net2.py <onnx> <vnnlib> [restarts] [iters] [seed]  -> FALSIFIED / no-ce
Exit codes: 3 = genuine in-box CE found, 0 = none found, 2 = wrong architecture.
"""
import sys, os, re, numpy as np, onnxruntime as ort
from onnx import numpy_helper
import onnx as onnxmod
np.seterr(all="ignore")  # Accelerate BLAS emits spurious FP flags on exact +/-1 matmuls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnnlib_ce

ONNX, VNNLIB = sys.argv[1], sys.argv[2]
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 160
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- weights (by exact name) ---
m = onnxmod.load(ONNX)
init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
def find(shape, uniq=None):
    for a in init.values():
        if a.shape == shape and (uniq is None or set(np.unique(a)) <= uniq):
            return a
    return None
try:
    W1 = find((32, 3, 5, 5), {-1.0, 1.0})     # conv1 binary
    W2 = find((64, 32, 5, 5), {-1.0, 1.0})    # conv2 binary
    W3 = init['sequential_1/quant_conv2d_5/QuantConv2D_weights_fused_bn']  # conv3 real (BN5 folded)
    B3 = init['sequential_1/quant_conv2d_5/QuantConv2D_bias_fused_bn']
    WD1 = find((3136, 256), {-1.0, 1.0})      # dense1 binary
    WD2 = find((256, 43), {-1.0, 1.0})        # dense2 binary
    bn3_scale = init['sequential_1/batch_normalization_3/Const:0']
    bn3_beta  = init['sequential_1/batch_normalization_3/ReadVariableOp:0']
    bn3_mean  = init['sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:0']
    bn3_var   = init['sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:0']
    bn4_scale = init['sequential_1/batch_normalization_5/Const:0']
    bn4_beta  = init['sequential_1/batch_normalization_4/ReadVariableOp:0']
    bn4_mean  = init['sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:0']
    bn4_var   = init['sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:0']
    bn6_rsqrt = init['sequential_1/batch_normalization_6/batchnorm/Rsqrt:0']
    bn6_sub   = init['sequential_1/batch_normalization_6/batchnorm/sub:0']
    assert all(w is not None for w in (W1, W2, WD1, WD2))
except (KeyError, AssertionError):
    print(f"skip: {os.path.basename(ONNX)} is not the 48x48 3-conv traffic_signs net this falsifier handles")
    sys.exit(2)

EPS = 0.0010000000474974513
# fold BN3/BN4 to per-channel affine  z*a + b   (a = scale/sqrt(var+eps), b = beta - mean*a)
a3 = (bn3_scale / np.sqrt(bn3_var + EPS))[:, None, None]; b3 = (bn3_beta - bn3_mean*bn3_scale/np.sqrt(bn3_var+EPS))[:, None, None]
a4 = (bn4_scale / np.sqrt(bn4_var + EPS))[:, None, None]; b4 = (bn4_beta - bn4_mean*bn4_scale/np.sqrt(bn4_var+EPS))[:, None, None]

# --------------------------------------------------------------------------------- kernels ---
def conv_fwd(x, w):  # x [C,H,W], w [OC,IC,KH,KW] -> [OC,OH,OW], valid stride1
    OC, IC, KH, KW = w.shape; C, H, Wd = x.shape; OH, OW = H-KH+1, Wd-KW+1
    cols = np.empty((C, KH, KW, OH, OW))
    for ki in range(KH):
        for kj in range(KW):
            cols[:, ki, kj] = x[:, ki:ki+OH, kj:kj+OW]
    return (w.reshape(OC, -1) @ cols.reshape(C*KH*KW, OH*OW)).reshape(OC, OH, OW)

def conv_bwd(gout, w, in_shape):  # gout [OC,OH,OW] -> grad wrt input [C,H,W]
    OC, IC, KH, KW = w.shape; C, H, Wd = in_shape; OH, OW = gout.shape[1], gout.shape[2]
    gcols = (w.reshape(OC, -1).T @ gout.reshape(OC, -1)).reshape(C, KH, KW, OH, OW)
    gin = np.zeros((C, H, Wd))
    for ki in range(KH):
        for kj in range(KW):
            gin[:, ki:ki+OH, kj:kj+OW] += gcols[:, ki, kj]
    return gin

def maxpool2(x):  # [C,H,W] 2x2 stride2 -> (pooled [C,OH,OW], argwin [C,OH,OW] in 0..3)
    C, H, Wd = x.shape; OH, OW = H//2, Wd//2
    xp = x[:, :OH*2, :OW*2].reshape(C, OH, 2, OW, 2).transpose(0, 1, 3, 2, 4).reshape(C, OH, OW, 4)
    return xp.max(axis=3), xp.argmax(axis=3)

def maxpool2_bwd(g, arg, in_shape):  # scatter g to argmax positions -> [C,H,W]
    C, H, Wd = in_shape; OH, OW = H//2, Wd//2
    buf = np.zeros((C, OH, OW, 4))
    ci, oi, oj = np.indices((C, OH, OW))
    buf[ci, oi, oj, arg] = g
    buf = buf.reshape(C, OH, OW, 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, OH*2, OW*2)
    out = np.zeros((C, H, Wd)); out[:, :OH*2, :OW*2] = buf
    return out

def dsign(x):  # true collapsed Sign(Sign(x)+0.1): x<0 -> -1 else +1
    return np.where(x < 0, -1.0, 1.0)

def softsign(z, alpha):  # RMS-normalised surrogate; returns (t, deriv) with deriv = d t / d z
    c = np.sqrt(np.mean(z*z)) + 1e-9
    t = np.tanh(alpha * z / c)
    return t, (alpha / c) * (1.0 - t*t)

# --------------------------------------------------------------------------------- forward ---
def forward_hard(x_nhwc):  # exact net -> logits[43]
    x = np.transpose(x_nhwc, (2, 0, 1))
    s1 = dsign(maxpool2(conv_fwd(x, W1))[0] * a3 + b3)
    s2 = dsign(maxpool2(conv_fwd(s1, W2))[0] * a4 + b4)
    p3 = conv_fwd(s2, W3) + B3[:, None, None]
    s3 = dsign(np.transpose(p3, (1, 2, 0)).reshape(-1))
    s4 = dsign((s3 @ WD1) * bn6_rsqrt + bn6_sub)
    return s4 @ WD2

def forward_soft(x_nhwc, alpha):  # surrogate -> logits[43], cache for backward
    x = np.transpose(x_nhwc, (2, 0, 1))
    p1 = conv_fwd(x, W1); m1, arg1 = maxpool2(p1); b1 = m1 * a3 + b3
    s1, d1g = softsign(b1, alpha)
    p2 = conv_fwd(s1, W2); m2, arg2 = maxpool2(p2); b2 = m2 * a4 + b4
    s2, d2g = softsign(b2, alpha)
    p3 = conv_fwd(s2, W3) + B3[:, None, None]
    f3 = np.transpose(p3, (1, 2, 0)).reshape(-1)
    s3, d3g = softsign(f3, alpha)
    d1 = s3 @ WD1; b6 = d1 * bn6_rsqrt + bn6_sub
    s4, d4g = softsign(b6, alpha)
    logits = s4 @ WD2
    cache = (arg1, d1g, p1.shape, arg2, d2g, p2.shape, d3g, p3.shape, d4g)
    return logits, cache

def backward(gl, cache):  # gl [43] -> grad wrt x_nhwc [48,48,3]
    arg1, d1g, p1shape, arg2, d2g, p2shape, d3g, p3shape, d4g = cache
    gs4 = WD2 @ gl                                  # [256]
    gb6 = gs4 * d4g                                 # [256]
    gd1 = gb6 * bn6_rsqrt                            # [256]
    gs3 = WD1 @ gd1                                  # [3136]
    gf3 = gs3 * d3g                                  # [3136]
    gp3 = gf3.reshape(p3shape[1], p3shape[2], p3shape[0]).transpose(2, 0, 1)  # [64,7,7]
    gs2 = conv_bwd(gp3, W3, (64, 9, 9))              # [64,9,9]
    gb2 = gs2 * d2g                                  # d2g already alpha/c*(1-t^2)
    gm2 = gb2 * a4
    gp2 = maxpool2_bwd(gm2, arg2, p2shape)           # [64,18,18]
    gs1 = conv_bwd(gp2, W2, (32, 22, 22))            # [32,22,22]
    gb1 = gs1 * d1g
    gm1 = gb1 * a3
    gp1 = maxpool2_bwd(gm1, arg1, p1shape)           # [32,44,44]
    gx = conv_bwd(gp1, W1, (3, 48, 48))              # [3,48,48]
    return np.transpose(gx, (1, 2, 0))               # [48,48,3]

# ----------------------------------------------------------------------------- box + spec ---
text = open(VNNLIB).read(); n = len(re.findall(r'\(declare-const X_\d+', text))
ub = {}; lb = {}
for mm in re.finditer(r'\(assert \(<= X_(\d+)\s+([-\d.eE]+)\)\)', text): ub[int(mm.group(1))] = float(mm.group(2))
for mm in re.finditer(r'\(assert \(>= X_(\d+)\s+([-\d.eE]+)\)\)', text): lb[int(mm.group(1))] = float(mm.group(2))
LB = np.array([lb.get(i, 0.0) for i in range(n)]).reshape(48, 48, 3)
UB = np.array([ub.get(i, 255.0) for i in range(n)]).reshape(48, 48, 3)
atoms = re.findall(r'\(>= Y_(\d+) Y_(\d+)\)', text)
from collections import Counter
rhs = Counter(int(b) for _, b in atoms)
TRUE_C = rhs.most_common(1)[0][0] if rhs else 0
OTHERS = sorted({int(a) for a, _ in atoms})

sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0].name
def ort_logits(x_nhwc):
    return sess.run(None, {inp: x_nhwc.astype(np.float32).reshape(1, 48, 48, 3)})[0].flatten()
def is_ce(x_nhwc):
    xd = {i: float(x_nhwc.reshape(-1)[i]) for i in range(n)}
    ib, ce, det = vnnlib_ce.validate(ONNX, VNNLIB, xd)
    return ce, det

def clamp(x): return np.minimum(np.maximum(x, LB), UB)
mid = (LB+UB)/2.0; Wd = np.maximum(UB-LB, 1e-9)

print(f"net2 48x48 3-conv  trueC={TRUE_C}  inputs={n}  box_width mean={Wd.mean():.2f}", flush=True)
best_margin = -1e30
for r in range(RESTARTS):
    if r == 0:              x = mid.copy()
    elif r % 3 == 1:        x = clamp(mid + (rng.random(mid.shape)-0.5)*Wd)
    else:                   x = LB + rng.random(LB.shape)*Wd
    ol = ort_logits(x); order = np.argsort(ol)[::-1]
    tgt = int(order[0] if order[0] != TRUE_C else order[1])
    lr = Wd * 0.30
    for it in range(ITERS):
        alpha = 2.0 + 18.0*it/ITERS               # ramp 2 -> 20
        logits, cache = forward_soft(x, alpha)
        gl = np.zeros(43); gl[tgt] = 1.0; gl[TRUE_C] = -1.0
        g = backward(gl, cache)
        x = clamp(x + lr*np.sign(g))
        lr = lr * 0.99
        if it % 6 == 0 or it == ITERS-1:
            ol = ort_logits(x); order = np.argsort(ol)[::-1]
            margin = ol[order[0] if order[0] != TRUE_C else order[1]] - ol[TRUE_C]
            best_margin = max(best_margin, margin)
            if order[0] != TRUE_C:                # ORT already flips class -> confirm genuine CE
                ce, det = is_ce(x)
                if ce:
                    print(f"FALSIFIED restart={r} it={it} alpha={alpha:.1f} tgt={tgt} trueC={TRUE_C} winC={int(order[0])} :: {det[:80]}", flush=True)
                    out = os.environ.get("NET2_CE_OUT")
                    if out:  # dump witness so an independent vnnlib_ce.py CLI run can re-confirm
                        xr = x.reshape(-1)
                        with open(out, "w") as fh:
                            fh.write("(\n" + "\n".join(f"(X_{i} {float(xr[i]):.8g})" for i in range(n)) + "\n)\n")
                        print(f"  wrote witness -> {out}", flush=True)
                    sys.exit(3)
            tgt = int(order[0] if order[0] != TRUE_C else order[1])
    print(f"  restart {r}: best_margin so far (max_j!=C logit - logit_C) = {best_margin:.4f}", flush=True)
print(f"no-ce ({RESTARTS}x{ITERS}, trueC={TRUE_C}, best_margin={best_margin:.4f})", flush=True)
sys.exit(0)
