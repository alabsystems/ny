#!/usr/bin/env python3
"""
Validate the RESIDUAL-block fan-out in the joint alpha-gradient adjoint
(docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md, the "Residual merge" rule) against
central finite differences. Resnets are the flagship cifar100/tinyimagenet
topology, and the residual fan-out (one Ā driving BOTH the skip and the F
branch) is the most error-prone part of the adjoint.

Net:  x -> W1 -> ReLU1 -> a1 -> [ a1 + F(a1) ] -> Wout -> out,
      F(a1) = Wg . ReLU_f( Wf.a1 + bf ) + bg      (dims: F maps dh->dh)

We harvest alpha1 (ReLU1, on the trunk) AND alphaf (ReLU_f, INSIDE the residual
branch). If BOTH match FD, the fan-out adjoint is correct. Pure Python.
"""

_state = [987654321]
def rnd():
    _state[0] = (1103515245 * _state[0] + 12345) & 0x7fffffff
    return (_state[0] / 0x7fffffff) * 2.0 - 1.0
def mat(r, c): return [[rnd() for _ in range(c)] for _ in range(r)]
def vec(n):    return [rnd() for _ in range(n)]

d0, dh, dout = 3, 4, 2
W1 = mat(dh, d0); b1 = vec(dh)     # x -> z1
Wf = mat(dh, dh); bf = vec(dh)     # a1 -> zf   (inside F)
Wg = mat(dh, dh); bg = vec(dh)     # af -> g    (inside F)
Wout = mat(dout, dh); bout = vec(dout)
c = vec(dout)
x_l = [-1.0]*d0; x_u = [1.0]*d0

l1 = [-1.0-0.3*i for i in range(dh)]; u1 = [1.0+0.2*i for i in range(dh)]
lf = [-1.1-0.2*i for i in range(dh)]; uf = [0.8+0.3*i for i in range(dh)]
def chord(l,u):
    ub=[u[i]/(u[i]-l[i]) for i in range(len(l))]; t=[-ub[i]*l[i] for i in range(len(l))]
    return ub,t
ub1,t1 = chord(l1,u1); ubf,tf = chord(lf,uf)

def dotcols(A, W):   # A (rows of W) times W (rows x cols) -> len cols
    R=len(W); C=len(W[0])
    return [sum(A[i]*W[i][j] for i in range(R)) for j in range(C)]
def dotrows(Abar, W): # adjoint: Abar (cols of W) times W^T -> len rows
    R=len(W); C=len(W[0])
    return [sum(Abar[j]*W[i][j] for j in range(C)) for i in range(R)]

def relu_fold(A, alpha, ub, t):   # returns (A_post, sigma, tau, bias_add)
    n=len(A); A2=[0.0]*n; sig=[0.0]*n; tau=[0.0]*n; badd=0.0
    for i in range(n):
        if A[i]>=0.0: s,ta = alpha[i], 0.0
        else:         s,ta = ub[i], t[i]
        sig[i],tau[i]=s,ta; badd+=A[i]*ta; A2[i]=A[i]*s
    return A2, sig, tau, badd

def bound_and_rec(alpha1, alphaf, rec=False):
    A=c[:]; b=0.0
    # Linear Wout
    b += sum(A[i]*bout[i] for i in range(dout)); A = dotcols(A, Wout)   # at out_block (dh)
    # residual split
    A_skip = A[:]; A_g = A[:]
    # F backward: Linear Wg (g->af)
    b += sum(A_g[i]*bg[i] for i in range(dh)); A_g = dotcols(A_g, Wg)   # at af
    A_af = A_g[:]
    A_g, sigf, tauf, ba = relu_fold(A_g, alphaf, ubf, tf); b += ba      # at zf
    # Linear Wf (zf->a1)
    b += sum(A_g[i]*bf[i] for i in range(dh)); A_g = dotcols(A_g, Wf)   # at a1 (F contrib)
    # merge
    A_a1 = [A_skip[i]+A_g[i] for i in range(dh)]
    A_pre1 = A_a1[:]
    A, sig1, tau1, ba = relu_fold(A_a1, alpha1, ub1, t1); b += ba       # at z1
    # Linear W1
    b += sum(A[i]*b1[i] for i in range(dh)); A = dotcols(A, W1)         # A0 (d0)
    A0 = A[:]
    bound = b + sum((A0[j]*x_l[j] if A0[j]>=0 else A0[j]*x_u[j]) for j in range(d0))
    if rec: return bound, A0, A_pre1, sig1, tau1, A_af, sigf, tauf
    return bound

def joint_grad(alpha1, alphaf):
    _, A0, A_pre1, sig1, tau1, A_af, sigf, tauf = bound_and_rec(alpha1, alphaf, rec=True)
    Abar = [ (x_l[j] if A0[j]>=0 else x_u[j]) for j in range(d0) ]      # xi ; adj_b=1
    # Linear W1 adjoint (+ bias channel b1)
    Abar_z1 = [ dotrows(Abar, W1)[i] + b1[i] for i in range(dh) ]
    # ReLU1 adjoint: harvest alpha1, propagate to a1
    g1 = [ Abar_z1[i]*max(A_pre1[i],0.0) for i in range(dh) ]
    Abar_a1 = [ Abar_z1[i]*sig1[i] + tau1[i] for i in range(dh) ]
    # RESIDUAL FAN-OUT: Abar_a1 drives the F branch (and the skip; skip only matters
    # for adjoint upstream of the block, of which there is none here).
    # F_adjoint(Abar_a1): Linear Wf adjoint (+bf) -> ReLU_f harvest -> ...
    Abar_zf = [ dotrows(Abar_a1, Wf)[i] + bf[i] for i in range(dh) ]
    gf = [ Abar_zf[i]*max(A_af[i],0.0) for i in range(dh) ]
    return g1, gf

def local_grad(alpha1, alphaf):
    _, A0, A_pre1, sig1, tau1, A_af, sigf, tauf = bound_and_rec(alpha1, alphaf, rec=True)
    g1 = [ l1[i]*max(A_pre1[i],0.0) for i in range(dh) ]
    gf = [ lf[i]*max(A_af[i],0.0) for i in range(dh) ]
    return g1, gf

def fd(alpha1, alphaf, eps=1e-6):
    g1=[0.0]*dh; gf=[0.0]*dh
    for i in range(dh):
        a=alpha1[:]; a[i]+=eps; hp=bound_and_rec(a,alphaf)
        a=alpha1[:]; a[i]-=eps; hm=bound_and_rec(a,alphaf); g1[i]=(hp-hm)/(2*eps)
    for i in range(dh):
        a=alphaf[:]; a[i]+=eps; hp=bound_and_rec(alpha1,a)
        a=alphaf[:]; a[i]-=eps; hm=bound_and_rec(alpha1,a); gf[i]=(hp-hm)/(2*eps)
    return g1, gf

def relerr(a,b): return abs(a-b)/max(abs(a),abs(b),1e-9)

print("=== residual-block fan-out validation (joint alpha-gradient adjoint) ===")
worst=0.0
for a1v,afv in [(0.5,0.5),(0.3,0.7),(0.8,0.2),(0.2,0.9),(0.6,0.1)]:
    a1=[a1v]*dh; af=[afv]*dh
    jg1,jgf = joint_grad(a1,af); lg1,lgf = local_grad(a1,af); fg1,fgf = fd(a1,af)
    je = max([relerr(jg1[i],fg1[i]) for i in range(dh)]+[relerr(jgf[i],fgf[i]) for i in range(dh)])
    le = max([relerr(lg1[i],fg1[i]) for i in range(dh)]+[relerr(lgf[i],fgf[i]) for i in range(dh)])
    # restrict to the residual branch (alphaf) to prove the fan-out is exercised
    jef = max(relerr(jgf[i],fgf[i]) for i in range(dh))
    nz  = sum(1 for i in range(dh) if abs(fgf[i])>1e-6)
    maxf= max(abs(fgf[i]) for i in range(dh))
    print(f"[a1={a1v} af={afv}] JOINT vs FD={je:.3e}  LOCAL vs FD={le:.3e}  | "
          f"residual-branch: {nz}/{dh} nonzero, max|grad_f|={maxf:.4f}, joint-vs-FD={jef:.3e}")
    worst=max(worst,je)
print()
print(f"WORST joint-vs-FD (incl. residual-branch alpha) : {worst:.3e}")
print("PASS" if worst<1e-4 else "FAIL -- residual fan-out adjoint error!")
raise SystemExit(0 if worst < 1e-4 else 1)
