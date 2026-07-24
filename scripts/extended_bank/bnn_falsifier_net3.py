#!/usr/bin/env python3
"""Soft-sign BNN falsifier for the LARGE traffic_signs_recognition_2023 net
    3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx

Extends the 30x30 technique in bnn_falsifier.py to a 64x64, three-conv-block binarized
net. Each block is QConv(+/-1 weights) -> MaxPool2x2 -> BatchNorm -> Sign(Sign(.)+0.1),
then flatten -> Dense_1024(+/-1) -> BatchNorm -> Sign -> Dense_43(+/-1) -> Softmax.

The true net is piecewise-constant (Sign activations) => zero gradient, so PGD on it
fails. Technique: a differentiable SOFT-SIGN surrogate tanh(alpha*.) with alpha ramped
2->20 toward the true Sign, giving a usable gradient; every iterate is screened on an
EXACT hard forward (verified bit-identical to onnxruntime, max logit |diff|=0) and any
apparent CE is CONFIRMED against the TRUE net via onnxruntime + vnnlib_ce.validate.

Handled ops (forward + numpy backward):
  * Conv (binary +/-1 weights, valid, im2col)          -> exact linear, transposed conv grad
  * MaxPool 2x2 stride2 (floor)                         -> route-to-max subgradient (argmax scatter)
  * BatchNormalization (scale=1) folded to s*x+t        -> per-channel affine (positive scale)
  * dense BatchNorm already folded (mul,sub)            -> x*mul+sub
  * Sign(Sign(x)+0.1)  == (x>=0 ? +1 : -1)              -> soft tanh(alpha*z); hard np.where screen
  * MatMul (+/-1 weights)                               -> linear
  * Softmax                                             -> monotonic, order-preserving (ignored; we
                                                          compare raw logits, which decide the spec)

Input is NHWC [1,64,64,3] pixels in [0,255]; X_i (C-order) reshapes to (64,64,3)=HWC.
Spec: OR_j (Y_j >= Y_TRUEC); a CE is any strictly-in-box x with logit_j >= logit_TRUEC.

Usage: bnn_falsifier_net3.py <onnx> <vnnlib> [restarts] [iters] [seed] -> FALSIFIED/no-ce
"""
import sys, os, re, numpy as np
import onnxruntime as ort
import onnx as onnxmod
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnnlib_ce

np.seterr(all="ignore")  # numpy/BLAS emits spurious matmul warnings on this platform

ONNX, VNNLIB = sys.argv[1], sys.argv[2]
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ weights ----
m = onnxmod.load(ONNX)
init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
by_shape = {}
for name, a in init.items():
    by_shape.setdefault(a.shape, a)
try:
    W1 = by_shape[(32, 3, 5, 5)]      # conv1 weights (+/-1)
    W2 = by_shape[(64, 32, 5, 5)]     # conv2 weights (+/-1)
    W3 = by_shape[(64, 64, 3, 3)]     # conv3 weights (+/-1)
    WD1 = by_shape[(1600, 1024)]      # dense1 weights (+/-1)
    WD2 = by_shape[(1024, 43)]        # dense2 weights (+/-1)
except KeyError:
    print(f"skip: {os.path.basename(ONNX)} is not the 64x64 3-conv net this falsifier handles")
    sys.exit(0)

EPS_BN = 0.0010000000474974513


def bnfold(scale, bias, mean, var):
    """BatchNormalization y = scale*(x-mean)/sqrt(var+eps)+bias  ->  z = s*x + t (s>0)."""
    inv = 1.0 / np.sqrt(var + EPS_BN)
    return scale * inv, bias - scale * mean * inv


s1, t1 = bnfold(init['sequential_4/batch_normalization_15/Const:0'],
                init['sequential_4/batch_normalization_15/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:0'])
s2, t2 = bnfold(init['sequential_4/batch_normalization_17/Const:0'],
                init['sequential_4/batch_normalization_16/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:0'])
s3, t3 = bnfold(init['sequential_4/batch_normalization_17/Const:0'],
                init['sequential_4/batch_normalization_17/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:0'],
                init['sequential_4/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:0'])
mul4 = init['sequential_4/batch_normalization_18/batchnorm/Rsqrt:0']   # dense BN (already folded)
sub4 = init['sequential_4/batch_normalization_18/batchnorm/sub:0']
s1 = s1[:, None, None]; t1 = t1[:, None, None]
s2 = s2[:, None, None]; t2 = t2[:, None, None]
s3 = s3[:, None, None]; t3 = t3[:, None, None]

# ------------------------------------------------------------------ layers -----
def conv_fwd(x, w):  # x [IC,H,W], w [OC,IC,KH,KW] -> out [OC,OH,OW]
    OC, IC, KH, KW = w.shape
    C, H, Wd = x.shape
    OH, OW = H - KH + 1, Wd - KW + 1
    cols = np.empty((IC, KH, KW, OH, OW))
    for ki in range(KH):
        for kj in range(KW):
            cols[:, ki, kj] = x[:, ki:ki + OH, kj:kj + OW]
    out = (w.reshape(OC, -1) @ cols.reshape(IC * KH * KW, OH * OW)).reshape(OC, OH, OW)
    return out


def conv_bwd(gout, w, in_shape):  # gout [OC,OH,OW] -> gin [IC,H,W]
    OC, IC, KH, KW = w.shape
    C, H, Wd = in_shape
    OH, OW = H - KH + 1, Wd - KW + 1
    gcols = (w.reshape(OC, -1).T @ gout.reshape(OC, OH * OW)).reshape(IC, KH, KW, OH, OW)
    gin = np.zeros((C, H, Wd))
    for ki in range(KH):
        for kj in range(KW):
            gin[:, ki:ki + OH, kj:kj + OW] += gcols[:, ki, kj]
    return gin


def maxpool_fwd(x):  # [C,H,W] 2x2 stride2 floor -> out [C,OH,OW], argmax [C,OH,OW]
    C, H, W = x.shape
    OH, OW = H // 2, W // 2
    win = x[:, :2 * OH, :2 * OW].reshape(C, OH, 2, OW, 2).transpose(0, 1, 3, 2, 4).reshape(C, OH, OW, 4)
    out = win.max(axis=-1)
    arg = win.argmax(axis=-1)
    return out, arg


def maxpool_bwd(gout, arg, in_shape):  # scatter gout to argmax positions -> gin [C,H,W]
    C, H, W = in_shape
    OH, OW = gout.shape[1], gout.shape[2]
    gwin = np.zeros((C, OH, OW, 4))
    ci, hi, wi = np.indices((C, OH, OW))
    gwin[ci, hi, wi, arg] = gout
    full = gwin.reshape(C, OH, OW, 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, 2 * OH, 2 * OW)
    gin = np.zeros((C, H, W))
    gin[:, :2 * OH, :2 * OW] = full
    return gin


def sign2(x):  # exact Sign(Sign(x)+0.1)
    return np.where(x >= 0, 1.0, -1.0)


def forward(x_nhwc, alpha, hard=False):
    """x_nhwc [64,64,3] -> logits[43]; returns (logits, cache) (cache used by backward)."""
    x = np.transpose(x_nhwc, (2, 0, 1))          # NCHW [3,64,64]
    c1 = conv_fwd(x, W1)                          # [32,60,60]
    mp1, arg1 = maxpool_fwd(c1)                   # [32,30,30]
    z1 = s1 * mp1 + t1
    a1 = sign2(z1) if hard else np.tanh(alpha * z1)
    c2 = conv_fwd(a1, W2)                         # [64,26,26]
    mp2, arg2 = maxpool_fwd(c2)                   # [64,13,13]
    z2 = s2 * mp2 + t2
    a2 = sign2(z2) if hard else np.tanh(alpha * z2)
    c3 = conv_fwd(a2, W3)                         # [64,11,11]
    mp3, arg3 = maxpool_fwd(c3)                   # [64,5,5]
    z3 = s3 * mp3 + t3
    a3 = sign2(z3) if hard else np.tanh(alpha * z3)
    a3n = np.transpose(a3, (1, 2, 0))            # NHWC [5,5,64]
    flat = a3n.reshape(-1)                        # 1600 (C-order NHWC)
    d1 = flat @ WD1                              # [1024]
    z4 = d1 * mul4 + sub4
    a4 = sign2(z4) if hard else np.tanh(alpha * z4)
    logits = a4 @ WD2                            # [43]
    cache = (z1, arg1, z2, arg2, z3, arg3, a3n.shape, z4, alpha)
    return logits, cache


def backward(gl, cache):  # gl grad wrt logits [43] -> grad wrt x_nhwc [64,64,3]
    z1, arg1, z2, arg2, z3, arg3, a3shape, z4, alpha = cache
    ga4 = WD2 @ gl                                        # [1024]
    gz4 = ga4 * alpha * (1 - np.tanh(alpha * z4) ** 2)
    gflat = WD1 @ gz4                                     # [1600]
    ga3n = gflat.reshape(a3shape)                        # NHWC [5,5,64]
    ga3 = np.transpose(ga3n, (2, 0, 1))                  # [64,5,5]
    gz3 = ga3 * alpha * (1 - np.tanh(alpha * z3) ** 2)
    gmp3 = gz3 * s3
    gc3 = maxpool_bwd(gmp3, arg3, (64, 11, 11))
    ga2 = conv_bwd(gc3, W3, (64, 13, 13))               # [64,13,13]
    gz2 = ga2 * alpha * (1 - np.tanh(alpha * z2) ** 2)
    gmp2 = gz2 * s2
    gc2 = maxpool_bwd(gmp2, arg2, (64, 26, 26))
    ga1 = conv_bwd(gc2, W2, (32, 30, 30))               # [32,30,30]
    gz1 = ga1 * alpha * (1 - np.tanh(alpha * z1) ** 2)
    gmp1 = gz1 * s1
    gc1 = maxpool_bwd(gmp1, arg1, (32, 60, 60))
    gx = conv_bwd(gc1, W1, (3, 64, 64))                 # [3,64,64]
    return np.transpose(gx, (1, 2, 0))                  # NHWC [64,64,3]


# --------------------------------------------------------------- box + spec ----
text = open(VNNLIB).read()
n = len(re.findall(r'\(declare-const X_\d+', text))
ub = {}; lb = {}
for mm in re.finditer(r'\(assert \(<= X_(\d+)\s+([-\d.eE]+)\)\)', text):
    ub[int(mm.group(1))] = float(mm.group(2))
for mm in re.finditer(r'\(assert \(>= X_(\d+)\s+([-\d.eE]+)\)\)', text):
    lb[int(mm.group(1))] = float(mm.group(2))
side = int(round((n / 3) ** 0.5))                        # 64
LB = np.array([lb.get(i, 0.0) for i in range(n)]).reshape(side, side, 3)
UB = np.array([ub.get(i, 255.0) for i in range(n)]).reshape(side, side, 3)
atoms = re.findall(r'\(>= Y_(\d+) Y_(\d+)\)', text)
from collections import Counter
rhs = Counter(int(b) for _, b in atoms)
TRUE_C = rhs.most_common(1)[0][0] if rhs else 0
others = sorted({int(a) for a, _ in atoms})

# --------------------------------------------------------------- oracles -------
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]


def hard_logits(x_nhwc):
    """EXACT (verified bit-identical to ORT) integer logits from the hard forward."""
    lo, _ = forward(x_nhwc, 0.0, hard=True)
    return lo


def is_ce_true(x_nhwc):
    """Final confirmation against the TRUE net via onnxruntime + full vnnlib assert check."""
    xd = {i: float(x_nhwc.reshape(-1)[i]) for i in range(n)}
    ib, ce, det = vnnlib_ce.validate(ONNX, VNNLIB, xd)
    return ce, det


def clamp(x):
    return np.minimum(np.maximum(x, LB), UB)


mid = (LB + UB) / 2.0
Wd = np.maximum(UB - LB, 1e-9)


def screen_and_confirm(x, r, it, tgt):
    """Screen with the exact hard forward; only pay for ORT+vnnlib when it shows a CE."""
    lo = hard_logits(x)
    other = np.delete(lo, TRUE_C)
    if other.max() >= lo[TRUE_C]:                        # apparent CE on the true net
        ce, det = is_ce_true(x)
        if ce:
            print(f"FALSIFIED restart={r} it={it} tgt={tgt} trueC={TRUE_C} "
                  f"margin={lo[TRUE_C]-other.max():.0f} :: {det[:80]}")
            return True
    return False


# ------------------------------------------------------------------- PGD --------
best_margin = 1e18
for r in range(RESTARTS):
    if r == 0:
        x = mid.copy()
    elif r % 3 == 1:
        x = clamp(mid + (rng.random(mid.shape) - 0.5) * Wd)
    else:
        x = LB + rng.random(LB.shape) * Wd
    lo = hard_logits(x)
    tgt = int(np.delete(np.arange(43), TRUE_C)[np.delete(lo, TRUE_C).argmax()])
    lr = Wd * 0.35
    for it in range(ITERS):
        alpha = 2.0 + 18.0 * it / max(1, ITERS - 1)      # ramp 2 -> 20
        logits, cache = forward(x, alpha)
        gl = np.zeros(43); gl[tgt] = 1.0; gl[TRUE_C] = -1.0
        g = backward(gl, cache)
        x = clamp(x + lr * np.sign(g))
        lr *= 0.985
        if it % 5 == 0 or it == ITERS - 1:
            lo = hard_logits(x)
            other = np.delete(lo, TRUE_C)
            margin = lo[TRUE_C] - other.max()
            if margin < best_margin:
                best_margin = margin
            if other.max() >= lo[TRUE_C]:
                if screen_and_confirm(x, r, it, tgt):
                    sys.exit(3)
            tgt = int(np.delete(np.arange(43), TRUE_C)[other.argmax()])  # retarget best-other
print(f"no-ce ({RESTARTS}x{ITERS}, trueC={TRUE_C}, best_margin={best_margin:.0f})")
sys.exit(0)
