#!/usr/bin/env python3
"""
Numerically validate the joint alpha-gradient adjoint derived in
docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md against central finite differences.

Model: CROWN lower bound of  c^T f(x)  over an input box, where
  f(x) = W3 . relu2( W2 . relu1( W1 x + b1 ) + b2 ) + b3
Two ReLU layers so the adjoint must propagate through an intermediate linear
(W2) -- the regime where the local gradient (l_i . sum max) diverges from the
true joint gradient.

Pure Python (no numpy). Deterministic weights via a fixed-seed LCG.
"""

# ---------- deterministic pseudo-random (no imports needed) ----------
_state = [123456789]
def rnd():
    # LCG -> uniform in [-1, 1)
    _state[0] = (1103515245 * _state[0] + 12345) & 0x7fffffff
    return (_state[0] / 0x7fffffff) * 2.0 - 1.0

def mat(r, c, scale=1.0):
    return [[rnd()*scale for _ in range(c)] for _ in range(r)]
def vec(n, scale=1.0):
    return [rnd()*scale for _ in range(n)]

def matvec_rowmajor(W, x):
    # W is rows x cols, returns W . x  (len rows)
    return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]

# ---------- network definition ----------
d0, d1, d2, d3 = 4, 5, 5, 3          # input, hidden1, hidden2, output
W1 = mat(d1, d0); b1 = vec(d1)
W2 = mat(d2, d1); b2 = vec(d2)
W3 = mat(d3, d2); b3 = vec(d3)
c  = vec(d3)                          # spec seed (single spec row): minimize c^T out

x_l = [-1.0]*d0
x_u = [ 1.0]*d0

# Fixed pre-activation bounds for the two ReLU layers, all unstable (l<0<u).
l1 = [-1.0 - 0.3*i for i in range(d1)]; u1 = [1.0 + 0.2*i for i in range(d1)]
l2 = [-1.2 - 0.2*i for i in range(d2)]; u2 = [0.9 + 0.3*i for i in range(d2)]

def relu_chord(l, u):
    ubar = [u[i]/(u[i]-l[i]) for i in range(len(l))]   # upper slope
    t    = [-ubar[i]*l[i]    for i in range(len(l))]   # upper intercept
    return ubar, t
ubar1, t1 = relu_chord(l1, u1)
ubar2, t2 = relu_chord(l2, u2)

# ---------- forward CROWN lower-bound fold, output -> input ----------
# Layers in fold order: Linear W3, ReLU2, Linear W2, ReLU1, Linear W1, concretize.
# alpha1, alpha2 are the free lower slopes for ReLU1, ReLU2 (each in [0,1]).
def crown_lower_bound(alpha1, alpha2, record=False):
    ops = []   # (kind, data) recorded in fold order for the adjoint
    A = c[:]                       # coeff at output (len d3)
    b = 0.0
    # Linear W3:  A_after = A . W3   (len d2);  b += A . b3
    b += sum(A[i]*b3[i] for i in range(d3))
    A_after = [sum(A[i]*W3[i][j] for i in range(d3)) for j in range(d2)]
    if record: ops.append(('lin', W3, b3, d3, d2))
    A = A_after
    # ReLU2 (bounds l2,u2, slope alpha2):
    A_before = A[:]
    A_after = [0.0]*d2
    sig2 = [0.0]*d2; tau2 = [0.0]*d2
    for i in range(d2):
        if A_before[i] >= 0.0:
            sig, tau = alpha2[i], 0.0
        else:
            sig, tau = ubar2[i], t2[i]
        sig2[i], tau2[i] = sig, tau
        b += A_before[i]*tau
        A_after[i] = A_before[i]*sig
    if record: ops.append(('relu', A_before, sig2, tau2, 'r2'))
    A = A_after
    # Linear W2:  A_after = A . W2  (len d1); b += A . b2
    b += sum(A[i]*b2[i] for i in range(d2))
    A_after = [sum(A[i]*W2[i][j] for i in range(d2)) for j in range(d1)]
    if record: ops.append(('lin', W2, b2, d2, d1))
    A = A_after
    # ReLU1 (bounds l1,u1, slope alpha1):
    A_before = A[:]
    A_after = [0.0]*d1
    sig1 = [0.0]*d1; tau1 = [0.0]*d1
    for i in range(d1):
        if A_before[i] >= 0.0:
            sig, tau = alpha1[i], 0.0
        else:
            sig, tau = ubar1[i], t1[i]
        sig1[i], tau1[i] = sig, tau
        b += A_before[i]*tau
        A_after[i] = A_before[i]*sig
    if record: ops.append(('relu', A_before, sig1, tau1, 'r1'))
    A = A_after
    # Linear W1:  A_after = A . W1  (len d0); b += A . b1
    b += sum(A[i]*b1[i] for i in range(d1))
    A_after = [sum(A[i]*W1[i][j] for i in range(d1)) for j in range(d0)]
    if record: ops.append(('lin', W1, b1, d1, d0))
    A = A_after
    # concretize over input box:  bound = b + sum_j phi(A[j])
    bound = b
    for j in range(d0):
        bound += A[j]*x_l[j] if A[j] >= 0.0 else A[j]*x_u[j]
    A0 = A[:]
    if record:
        return bound, ops, A0
    return bound

# ---------- closed-form JOINT gradient via the adjoint ----------
def joint_grad(alpha1, alpha2):
    bound, ops, A0 = crown_lower_bound(alpha1, alpha2, record=True)
    # seed adjoint at input:  xi[j] = x_l[j] if A0[j]>=0 else x_u[j]
    Abar = [ (x_l[j] if A0[j] >= 0.0 else x_u[j]) for j in range(d0) ]
    g = {'r1':[0.0]*d1, 'r2':[0.0]*d2}
    # adj_b = d(bound)/db = 1 throughout (b feeds the bound with coeff 1).
    # walk ops in REVERSE fold order (input -> output)
    for kind, *data in reversed(ops):
        if kind == 'lin':
            W, bias, dout, din = data     # forward: A_after(din)=A_before(dout).W ; b += A_before.bias
            # adj(A_before)[i] = sum_j adj(A_after)[j].W[i][j]  +  bias[i].adj_b
            Abar = [ sum(Abar[j]*W[i][j] for j in range(din)) + bias[i] for i in range(dout) ]
        else:  # relu:  A_after[i]=A_before[i].sigma_i ; b += A_before[i].tau_i
            A_before, sigma, tau, tag = data
            n = len(A_before)
            # harvest (alpha enters only A_after, only on positive coeff):
            gg = g[tag]
            for i in range(n):
                gg[i] += Abar[i]*max(A_before[i], 0.0)
            # propagate: adj(A_before)[i] = adj(A_after)[i].sigma_i + tau_i.adj_b
            Abar = [ Abar[i]*sigma[i] + tau[i] for i in range(n) ]
    return g['r1'], g['r2']

# ---------- LOCAL gradient (the refuted rule): l_i . max(A_before[i],0) ----------
def local_grad(alpha1, alpha2):
    _, ops, _ = crown_lower_bound(alpha1, alpha2, record=True)
    g = {'r1':[0.0]*d1, 'r2':[0.0]*d2}
    lmap = {'r1': l1, 'r2': l2}
    for kind, *data in ops:
        if kind == 'relu':
            A_before, sigma, tau, tag = data
            lv = lmap[tag]
            for i in range(len(A_before)):
                g[tag][i] += lv[i]*max(A_before[i], 0.0)
    return g['r1'], g['r2']

# ---------- finite-difference reference ----------
def fd_grad(alpha1, alpha2, eps=1e-6):
    g1 = [0.0]*d1; g2 = [0.0]*d2
    for i in range(d1):
        a = alpha1[:]; a[i]+=eps; hp = crown_lower_bound(a, alpha2)
        a = alpha1[:]; a[i]-=eps; hm = crown_lower_bound(a, alpha2)
        g1[i] = (hp-hm)/(2*eps)
    for i in range(d2):
        a = alpha2[:]; a[i]+=eps; hp = crown_lower_bound(alpha1, a)
        a = alpha2[:]; a[i]-=eps; hm = crown_lower_bound(alpha1, a)
        g2[i] = (hp-hm)/(2*eps)
    return g1, g2

# ---------- run at a few interior alpha points ----------
def relerr(a, b):
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a-b)/denom

def run(alpha1, alpha2, label):
    jg1, jg2 = joint_grad(alpha1, alpha2)
    lg1, lg2 = local_grad(alpha1, alpha2)
    fg1, fg2 = fd_grad(alpha1, alpha2)
    je = max([relerr(jg1[i], fg1[i]) for i in range(d1)] +
             [relerr(jg2[i], fg2[i]) for i in range(d2)])
    le = max([relerr(lg1[i], fg1[i]) for i in range(d1)] +
             [relerr(lg2[i], fg2[i]) for i in range(d2)])
    print(f"[{label}] max rel.err  JOINT vs FD = {je:.3e}   LOCAL vs FD = {le:.3e}")
    # per-neuron detail for ReLU1 (the deep layer where local diverges most)
    print(f"    ReLU1 neuron 0:  FD={fg1[0]:+.6f}  joint={jg1[0]:+.6f}  local={lg1[0]:+.6f}")
    return je, le

print("=== joint alpha-gradient adjoint validation (docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md) ===")
worst_joint = 0.0
for k, (a1v, a2v) in enumerate([(0.5,0.5),(0.3,0.7),(0.8,0.2),(0.5,0.9),(0.1,0.4)]):
    a1 = [a1v]*d1; a2 = [a2v]*d2
    je, le = run(a1, a2, f"alpha1={a1v} alpha2={a2v}")
    worst_joint = max(worst_joint, je)

print()
print(f"WORST joint-vs-FD relative error across all points: {worst_joint:.3e}")
print("PASS" if worst_joint < 1e-4 else "FAIL -- derivation error!")
