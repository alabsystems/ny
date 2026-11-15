// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane stage 2 — the JOINT QUADRATIC BOUND at a `ReduceSum(Mul(a, b))`
//! site of a lowered lane program (`lsnc_relu`'s Lyapunov value `V = x'Px`
//! and its decrease `dV = (f(x) - x)' P (f(x) + x)`).
//!
//! # Why
//!
//! McCormick relaxes each product `a_i * b_i` on the operands' BOX, as if `a`
//! and `b` were independent. At `dV` both factors are functions of the same
//! six inputs, and the certificate holds only BECAUSE of that correlation.
//! The numpy decision gate (`scratchpad/lsncproto/gate_engine.py`, variant
//! `allmax`, 2026-09-24) measured this bound as the difference between
//! ~5.6e7 domains per unsat row and 2e5-1.4e6; on row 2's root box the
//! published `dV` upper bound drops from 28.04 (plain) to 13.12.
//!
//! # What it computes — per domain, in f64, every step rounded outward
//!
//! 1. Each summand's two operand elements are reduced to a MIDPOINT AFFINE
//!    in `dx = x - xc` plus a residual radius, from the operand's forward
//!    forms and (when armed) its two CROWN certificates — backward passes
//!    seeded at the operand with `+e_i` / `-e_i` over the SAME forward pass —
//!    keeping, per element, the lower/upper pair with the smallest RIGOROUS
//!    radius (`reduce`).
//! 2. `sum_i w_i ma_i(dx) mb_i(dx)` is ONE quadratic `q + g.dx + dx'H dx`;
//!    the residual cross terms are charged by interval arithmetic, and the
//!    assembly rounding by a Higham term (`assemble`).
//! 3. `H` is split by a cyclic Jacobi sweep into `sum_l lambda_l (u_l.dx)^2`
//!    plus the residual `E = H - U diag(lambda) U'`. That is an identity for
//!    ANY real `U` and `lambda` (no orthogonality or accuracy is assumed), and
//!    `E` is CHARGED IN FULL as `sum_jk |E_jk| xr_j xr_k`, so a poor
//!    decomposition only loosens the bound (`deviation`; the negative control
//!    in `joint_tests` shows the charge is load-bearing).
//! 4. The convex part (`lambda > 0`) is maximised EXACTLY over the `2^d` box
//!    vertices (a convex function attains its maximum over a box at a
//!    vertex); the concave part is either dropped (it is `<= 0`) or replaced
//!    by its tangent at a projected-gradient point (a concave function lies
//!    below every tangent, wherever the point is). The smaller is kept, and
//!    both are capped by the `O(d^2)` bound `|g|.xr + sum lambda+ (|u|.xr)^2`.
//! 5. The site's concrete interval is INTERSECTED with the result, and the
//!    planes handed to the backward pass (and later to clipping) are the
//!    TANGENT AT THE BOX CENTRE `q + g.dx` plus the rigorous remainder — from
//!    this same pass. A plane is never deconstructed from a max'd bound and
//!    never carried to another box: it lives in this box's `NodeVal`.
//!
//! Soundness never depends on the heuristic parts (the pair selection, the
//! Jacobi convergence, the tangent point): each only changes which valid
//! bound is published.

use super::{gslack, next_down, next_up, Chunk, Fam, NodeVal, Op, MAX_INPUT_DIM};

/// Jacobi sweeps. Convergence is a tightness question only (the residual of
/// the decomposition is charged), so a cap is always safe.
pub(super) const JACOBI_SWEEPS: usize = 32;

/// Largest input dimension whose `2^d` box vertices are enumerated; above it
/// only the `O(d^2)` eigen bound is used.
pub(super) const VERTEX_MAX_DIM: usize = 8;

/// Projected-gradient steps that pick a tangent point (heuristic only: ANY
/// point gives a valid tangent of a concave function).
const PGA_STEPS: usize = 12;

const MD: usize = MAX_INPUT_DIM;

/// A `ReduceSum(Mul(a, b))` site of a lowered program: `node` is a width-1
/// [`Op::Sparse`] whose every term reads a DISTINCT element of the
/// [`Op::Mul`] at `mul`, whose operands are `a` and `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct JointSite {
    pub(super) node: usize,
    pub(super) mul: usize,
    pub(super) a: usize,
    pub(super) b: usize,
    /// Graph node name of the reduction (observability only).
    pub(super) name: String,
}

/// Every site of `ops`. Structural only: a site's term coefficients (with
/// their error bounds) and its constant are read per domain from the
/// domain's own program, so a family shares one site list.
pub(super) fn detect_sites(ops: &[Op], sizes: &[usize]) -> Vec<JointSite> {
    let mut sites = Vec::new();
    for (node, op) in ops.iter().enumerate() {
        let Op::Sparse { terms, .. } = op else {
            continue;
        };
        if sizes[node] != 1 || terms.len() != 1 || terms[0].is_empty() {
            continue;
        }
        let mul = terms[0][0].src as usize;
        if mul >= node {
            continue;
        }
        let Some(Op::Mul { a, b }) = ops.get(mul) else {
            continue;
        };
        let mut seen = vec![false; sizes[mul]];
        let distinct = terms[0].iter().all(|t| {
            let i = t.idx as usize;
            let fresh = t.src as usize == mul && i < seen.len() && !seen[i];
            if fresh {
                seen[i] = true;
            }
            fresh
        });
        if distinct {
            sites.push(JointSite {
                node,
                mul,
                a: *a,
                b: *b,
                name: String::new(),
            });
        }
    }
    sites
}

/// One operand element over a domain as a midpoint affine in `dx = x - xc`
/// plus a radius: `|value(x) - (m . dx + p)| <= rad` for every `x` in the
/// box; `mag >= max |m . dx + p|` and `vmag >= max |value(x)|` there.
#[derive(Debug, Clone, Copy)]
pub(super) struct Affine {
    pub(super) m: [f64; MD],
    pub(super) p: f64,
    pub(super) rad: f64,
    pub(super) mag: f64,
    pub(super) vmag: f64,
}

/// Reduce one operand element to its tightest midpoint affine over every
/// (lower, upper) pair of candidate forms. Each form is `d + 1` long,
/// constant last, and VALID over the whole box in its direction (`low(x) <=
/// value(x) <= up(x)`). `vmag` must bound `|value|` over the box. `None` when
/// nothing finite results.
///
/// For a pair `(L, U)` the midpoint `m = (L + U) / 2`, `p = m . xc + (L_c +
/// U_c) / 2` may be ANY floats: the radius is the rigorous maximum over the
/// box of the actual deviations `U(x) - (m.dx + p) = (U - m).dx + (U.xc +
/// U_c - p)` and `(m.dx + p) - L(x) = (m - L).dx + (p - L.xc - L_c)`,
/// computed with a Higham charge on their magnitudes.
pub(super) fn reduce(
    lows: &[&[f64]],
    ups: &[&[f64]],
    xc: &[f64],
    xr: &[f64],
    vmag: f64,
) -> Option<Affine> {
    let d = xc.len();
    if !vmag.is_finite() || d > MD {
        return None;
    }
    let mut best: Option<Affine> = None;
    for low in lows {
        for up in ups {
            let mut m = [0.0f64; MD];
            let mut p = 0.5 * low[d] + 0.5 * up[d];
            for k in 0..d {
                m[k] = 0.5 * low[k] + 0.5 * up[k];
                p += m[k] * xc[k];
            }
            let (mut eu, mut su) = (up[d] - p, up[d].abs() + p.abs());
            let (mut ed, mut sd) = (p - low[d], low[d].abs() + p.abs());
            for k in 0..d {
                eu += (up[k] - m[k]).abs() * xr[k] + up[k] * xc[k];
                su += (up[k].abs() + m[k].abs()) * xr[k] + (up[k] * xc[k]).abs();
                ed += (m[k] - low[k]).abs() * xr[k] - low[k] * xc[k];
                sd += (low[k].abs() + m[k].abs()) * xr[k] + (low[k] * xc[k]).abs();
            }
            let ru = next_up(eu + gslack(3 * d + 6, su));
            let rd = next_up(ed + gslack(3 * d + 6, sd));
            // `f64::max` DROPS a NaN operand: test finiteness first.
            if !(ru.is_finite() && rd.is_finite() && p.is_finite()) {
                continue;
            }
            let rad = ru.max(rd).max(0.0);
            let better = match &best {
                None => true,
                Some(b) => rad < b.rad,
            };
            if better {
                best = Some(Affine {
                    m,
                    p,
                    rad,
                    mag: 0.0,
                    vmag,
                });
            }
        }
    }
    let mut a = best?;
    let mut s = a.p.abs();
    for k in 0..d {
        s += a.m[k].abs() * xr[k];
    }
    let own = next_up(s + gslack(d + 3, s));
    let via_value = next_up(vmag + a.rad);
    if !(own.is_finite() && via_value.is_finite()) {
        return None;
    }
    a.mag = own.min(via_value);
    Some(a)
}

/// One summand `w * a * b` of a site. The true coefficient lies in `[w -
/// w_err, w + w_err]`; `z_mag >= max |a * b|` over the box (charged only when
/// `w_err > 0`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Term {
    pub(super) w: f64,
    pub(super) w_err: f64,
    pub(super) z_mag: f64,
    pub(super) a: Affine,
    pub(super) b: Affine,
}

/// `sum_t w_t ma_t(dx) mb_t(dx) = q + g.dx + dx'H dx` (float coefficients),
/// plus the three charges that relate it to the site's value.
struct Quad {
    h: [f64; MD * MD],
    g: [f64; MD],
    q: f64,
    /// Assembly rounding: bounds `|exact form - float form|` over the box.
    e_asm: f64,
    /// Residual cross terms `sum |w| min(ra vb + ma rb, rb va + mb ra)`.
    res: f64,
    /// Coefficient error `sum w_err z_mag`.
    e_coef: f64,
}

/// Assemble the quadratic of the midpoint affines.
///
/// ROUNDING: every `H_jk`, `g_j` and `q` is a sum over the `n` summands of a
/// product of at most four floats, so its error is at most `gamma_{n+4}`
/// times the sum of the absolute values of its products; weighted by the box
/// half-widths and summed, those errors are bounded by `gamma_{n+4} sum_t
/// |w_t| (|ma_t|.xr + |pa_t|)(|mb_t|.xr + |pb_t|)` — which is what `e_asm`
/// charges (`H` is filled on its upper triangle and mirrored, and `dx'Hdx`
/// only sees the symmetric part).
fn assemble(terms: &[Term], xr: &[f64]) -> Option<Quad> {
    let d = xr.len();
    let mut h = [0.0f64; MD * MD];
    let mut g = [0.0f64; MD];
    let mut q = 0.0f64;
    let (mut s_asm, mut res, mut e_coef) = (0.0f64, 0.0f64, 0.0f64);
    for t in terms {
        let (a, b) = (&t.a, &t.b);
        let hw = 0.5 * t.w;
        for j in 0..d {
            for k in j..d {
                h[j * d + k] += hw * (a.m[j] * b.m[k] + a.m[k] * b.m[j]);
            }
            g[j] += t.w * (a.p * b.m[j] + b.p * a.m[j]);
        }
        q += t.w * (a.p * b.p);
        let (mut ma, mut mb) = (a.p.abs(), b.p.abs());
        for k in 0..d {
            ma += a.m[k].abs() * xr[k];
            mb += b.m[k].abs() * xr[k];
        }
        s_asm += t.w.abs() * ma * mb;
        // a b - ma mb = (a - ma) b + ma (b - mb) = (b - mb) a + mb (a - ma).
        let r1 = a.rad * b.vmag + a.mag * b.rad;
        let r2 = b.rad * a.vmag + b.mag * a.rad;
        res += t.w.abs() * r1.min(r2);
        if t.w_err > 0.0 {
            e_coef += t.w_err * t.z_mag;
        }
    }
    for j in 0..d {
        for k in 0..j {
            h[j * d + k] = h[k * d + j];
        }
    }
    let n = terms.len();
    let e_asm = gslack(n + 8, s_asm);
    let res = next_up(res + gslack(2 * n + 8, res));
    let e_coef = next_up(e_coef + gslack(2 * n + 8, e_coef));
    let finite = h[..d * d].iter().chain(&g[..d]).all(|v| v.is_finite())
        && [q, e_asm, res, e_coef].iter().all(|v| v.is_finite());
    finite.then_some(Quad {
        h,
        g,
        q,
        e_asm,
        res,
        e_coef,
    })
}

/// Cyclic Jacobi eigen-sweep of a symmetric `n x n` row-major matrix, ported
/// from `joint_bilinear::jacobi_eigh` with a relative stop. Returns `(lambda,
/// u)` with `u[l * n + j]` the `j`-th component of the `l`-th vector. No
/// accuracy or orthogonality is claimed or needed: the caller charges the
/// residual `H - U diag(lambda) U'` in full. `sweeps = 0` returns the
/// diagonal and the identity.
pub(super) fn jacobi_eigh(matrix: &[f64], n: usize, sweeps: usize) -> ([f64; MD], [f64; MD * MD]) {
    let mut a = [0.0f64; MD * MD];
    a[..n * n].copy_from_slice(&matrix[..n * n]);
    let mut v = [0.0f64; MD * MD];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..sweeps {
        let (mut off, mut diag) = (0.0f64, 0.0f64);
        for p in 0..n {
            diag += a[p * n + p] * a[p * n + p];
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if !off.is_finite() || off.partial_cmp(&(1e-32 * diag)) != Some(std::cmp::Ordering::Greater)
        {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                if !t.is_finite() {
                    continue;
                }
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut lambda = [0.0f64; MD];
    for i in 0..n {
        lambda[i] = a[i * n + i];
    }
    let mut u = [0.0f64; MD * MD];
    for l in 0..n {
        for j in 0..n {
            u[l * n + j] = v[j * n + l];
        }
    }
    (lambda, u)
}

/// Rigorous deviations of `phi(dx) = g.dx + dx'H dx` over `|dx| <= xr`, with
/// the decomposition residual kept apart so a caller can never forget it:
/// `up + ech >= max phi`, `dn + ech >= max(-phi)`, `ur + ech >= max dx'Hdx`,
/// `lr + ech >= max(-dx'Hdx)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Dev {
    pub(super) up: f64,
    pub(super) dn: f64,
    pub(super) ur: f64,
    pub(super) lr: f64,
    /// `sum_jk |E_jk| xr_j xr_k` (rounded up), `E = H - U diag(lambda) U'`.
    /// Zero only when `charge_residual` is false (the negative control).
    pub(super) ech: f64,
}

/// See [`Dev`]. `sweeps` and `charge_residual` exist for the negative
/// control; production passes [`JACOBI_SWEEPS`] and `true`.
pub(super) fn deviation(
    h: &[f64],
    g: &[f64],
    xr: &[f64],
    sweeps: usize,
    charge_residual: bool,
) -> Option<Dev> {
    let d = xr.len();
    if d == 0 || d > MD || h.len() < d * d || g.len() < d {
        return None;
    }
    let inputs_finite = h[..d * d]
        .iter()
        .chain(&g[..d])
        .chain(xr)
        .all(|v| v.is_finite());
    if !inputs_finite || xr.iter().any(|&r| r < 0.0) {
        return None;
    }
    let (mut lam, mut u) = jacobi_eigh(h, d, sweeps);
    if !lam[..d].iter().chain(&u[..d * d]).all(|v| v.is_finite()) {
        // Fall back to the trivially valid split: lambda = diag, U = I.
        let fallback = jacobi_eigh(h, d, 0);
        lam = fallback.0;
        u = fallback.1;
    }
    // |u_l| . xr
    let mut pu = [0.0f64; MD];
    for l in 0..d {
        let mut s = 0.0f64;
        for j in 0..d {
            s += u[l * d + j].abs() * xr[j];
        }
        pu[l] = s;
    }
    let (mut sq, mut qpos, mut qneg) = (0.0f64, 0.0f64, 0.0f64);
    for l in 0..d {
        let t = lam[l].abs() * pu[l] * pu[l];
        sq += t;
        if lam[l] > 0.0 {
            qpos += t;
        } else {
            qneg += t;
        }
    }
    let mut sg = 0.0f64;
    for j in 0..d {
        sg += g[j].abs() * xr[j];
    }
    let (mut sh, mut ech) = (0.0f64, 0.0f64);
    for j in 0..d {
        for k in 0..d {
            let w = xr[j] * xr[k];
            sh += h[j * d + k].abs() * w;
            let mut recon = 0.0f64;
            for l in 0..d {
                recon += lam[l] * u[l * d + j] * u[l * d + k];
            }
            ech += (h[j * d + k] - recon).abs() * w;
        }
    }
    // Magnitude guard: every quantity below is a small multiple of these, so
    // nothing can overflow into an inf - inf NaN that `max` would drop.
    if (sg + sq + sh + ech).partial_cmp(&1e200) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    // `ech` is the float image of sum |E_jk| xr_j xr_k; the exact residual of
    // the FLOAT decomposition differs from the computed one by at most
    // gamma * (|H_jk| + sum_l |lambda_l u_lj u_lk|) per entry, which sums to
    // gamma * (sh + sq) over the box.
    let ech = if charge_residual {
        next_up(ech + gslack(2 * d * d + 16, ech + sh + sq))
    } else {
        0.0
    };
    // O(d^2) bound, valid at any d: (u.dx)^2 <= (|u|.xr)^2.
    let nslack = gslack(2 * d + 8, sg + sq);
    let mut up = next_up(sg + qpos + nslack);
    let mut dn = next_up(sg + qneg + nslack);
    let mut ur = next_up(qpos + nslack);
    let mut lr = next_up(qneg + nslack);
    if d <= VERTEX_MAX_DIM {
        let v = vertex_bounds(h, g, xr, &lam, &u, sg, sq);
        // Both sides are rigorous; keep the smaller (a NaN vertex value is
        // dropped by `min`, which keeps the O(d^2) bound).
        up = up.min(v.up);
        dn = dn.min(v.dn);
        ur = ur.min(v.ur);
        lr = lr.min(v.lr);
    }
    [up, dn, ur, lr, ech]
        .iter()
        .all(|v| v.is_finite())
        .then_some(Dev {
            up,
            dn,
            ur,
            lr,
            ech,
        })
}

/// The vertex bounds of [`deviation`] (`d <= VERTEX_MAX_DIM`), residual
/// excluded. With `P(dx) = sum_{lambda>0} lambda (u.dx)^2` (convex) and `N(dx)
/// = sum_{lambda<0} lambda (u.dx)^2` (concave, `<= 0`):
///
/// * `up = min( max_v g.v + P(v),  max_v (g + t).v + P(v) + kap )` where
///   `t = grad N(d0)`, `kap = -N(d0)`: `N(dx) <= N(d0) + t.(dx - d0) = t.dx -
///   N(d0)` because `t.d0 = 2 N(d0)` for a quadratic form;
/// * `dn` mirrors it on `-phi = -g.dx - N - P` with the tangent of `-P`;
/// * `ur = max_v P(v)` and `lr = max_v -N(v)`.
///
/// Every maximum is of a CONVEX function over the box, attained at a vertex.
/// ROUNDING: every evaluated expression is a sum of at most `4d + 8` rounded
/// products whose magnitudes are bounded by `sg + 3 sq` (`|t|.xr <= 2 sq`,
/// `kap <= sq`, `P, -N <= sq`), and the float `t`, `kap` differ from the
/// exact tangent of the float `N` by at most `gamma * 2 sq` over the box; one
/// `gslack(4d + 16, 2 (sg + 3 sq))` covers both.
fn vertex_bounds(
    h: &[f64],
    g: &[f64],
    xr: &[f64],
    lam: &[f64; MD],
    u: &[f64; MD * MD],
    sg: f64,
    sq: f64,
) -> Dev {
    const NV: usize = 1 << VERTEX_MAX_DIM;
    let d = xr.len();
    let nv = 1usize << d;
    let corner = |vi: usize, j: usize| {
        if (vi >> j) & 1 == 1 {
            xr[j]
        } else {
            -xr[j]
        }
    };
    let mut lin = [0.0f64; NV];
    let mut pos = [0.0f64; NV];
    let mut neg = [0.0f64; NV];
    let (mut up1, mut dn1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let (mut pmax, mut nmax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let (mut arg_up, mut arg_dn) = (0usize, 0usize);
    for vi in 0..nv {
        let mut li = 0.0f64;
        for j in 0..d {
            li += g[j] * corner(vi, j);
        }
        let (mut p, mut n) = (0.0f64, 0.0f64);
        for l in 0..d {
            if lam[l] == 0.0 {
                continue;
            }
            let mut t = 0.0f64;
            for j in 0..d {
                t += u[l * d + j] * corner(vi, j);
            }
            let e = lam[l] * t * t;
            if lam[l] > 0.0 {
                p += e;
            } else {
                n += e;
            }
        }
        lin[vi] = li;
        pos[vi] = p;
        neg[vi] = n;
        if li + p > up1 {
            up1 = li + p;
            arg_up = vi;
        }
        if -li - n > dn1 {
            dn1 = -li - n;
            arg_dn = vi;
        }
        pmax = pmax.max(p);
        nmax = nmax.max(-n);
    }
    let start = |vi: usize| {
        let mut s = [0.0f64; MD];
        for j in 0..d {
            s[j] = corner(vi, j);
        }
        s
    };
    let lam_max = lam[..d].iter().fold(0.0f64, |acc, &l| acc.max(l.abs()));
    let d_up = pga(h, g, xr, start(arg_up), 1.0, lam_max);
    let d_dn = pga(h, g, xr, start(arg_dn), -1.0, lam_max);
    let (mut t_up, mut t_dn) = ([0.0f64; MD], [0.0f64; MD]);
    let (mut kap_up, mut kap_dn) = (0.0f64, 0.0f64);
    for l in 0..d {
        if lam[l] == 0.0 {
            continue;
        }
        let (mut pu, mut pd) = (0.0f64, 0.0f64);
        for j in 0..d {
            pu += u[l * d + j] * d_up[j];
            pd += u[l * d + j] * d_dn[j];
        }
        if lam[l] < 0.0 {
            // grad N(d0) and -N(d0) >= 0.
            for j in 0..d {
                t_up[j] += 2.0 * lam[l] * pu * u[l * d + j];
            }
            kap_up -= lam[l] * pu * pu;
        } else {
            // grad P(d0') and P(d0') >= 0.
            for j in 0..d {
                t_dn[j] += 2.0 * lam[l] * pd * u[l * d + j];
            }
            kap_dn += lam[l] * pd * pd;
        }
    }
    let (mut up2, mut dn2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for vi in 0..nv {
        let (mut tu, mut td) = (0.0f64, 0.0f64);
        for j in 0..d {
            let c = corner(vi, j);
            tu += t_up[j] * c;
            td += t_dn[j] * c;
        }
        up2 = up2.max(lin[vi] + tu + pos[vi]);
        dn2 = dn2.max(-lin[vi] - td - neg[vi]);
    }
    let up2 = up2 + kap_up;
    let dn2 = dn2 + kap_dn;
    let vslack = gslack(4 * d + 16, 2.0 * (sg + 3.0 * sq));
    Dev {
        up: next_up(up1.min(up2) + vslack),
        dn: next_up(dn1.min(dn2) + vslack),
        ur: next_up(pmax + vslack),
        lr: next_up(nmax + vslack),
        ech: 0.0,
    }
}

/// Projected-gradient ascent of `sign * (g.d + d'Hd)` over `|d| <= xr` from
/// `x` (only picks a tangent point; the result stays inside the box).
fn pga(h: &[f64], g: &[f64], xr: &[f64], mut x: [f64; MD], sign: f64, lam_max: f64) -> [f64; MD] {
    let d = xr.len();
    let eta = 1.0 / (2.0 * lam_max + 1e-300);
    if !eta.is_finite() {
        return x;
    }
    for _ in 0..PGA_STEPS {
        let mut next = x;
        for j in 0..d {
            let mut gr = g[j];
            for k in 0..d {
                gr += 2.0 * h[j * d + k] * x[k];
            }
            let y = x[j] + eta * sign * gr;
            next[j] = if y.is_finite() {
                y.clamp(-xr[j], xr[j])
            } else {
                x[j]
            };
        }
        x = next;
    }
    x
}

/// The joint result for one site on one domain: `lo <= site(x) <= hi`, and
/// the planes `g . x + c_lo <= site(x) <= g . x + c_hi`, for every `x` in the
/// box (the box must lie inside `xc +- xr`).
#[derive(Debug, Clone, Copy)]
pub(super) struct SiteBound {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) g: [f64; MD],
    pub(super) c_lo: f64,
    pub(super) c_hi: f64,
}

/// The joint quadratic bound of `sum_t w_t a_t b_t + k`, `k in [k_lo, k_hi]`.
///
/// With `Q(dx) = sum_t w_t ma_t mb_t`, `Q = q + g.dx + P + N + dx'E dx`:
///
/// ```text
///   site >= Q - res - e_coef + k_lo >= q + g.dx - (lr + ech) - tot' + k_lo
///   site <= Q + res + e_coef + k_hi <= q + g.dx + (ur + ech) + tot' + k_hi
/// ```
///
/// with `tot' = e_asm + res + e_coef`; the interval uses `dn`/`up` in place of
/// the `g.dx` + remainder. The plane constants are those same expressions
/// moved to `x` (`g.dx = g.x - g.xc`), with a Higham charge on the few
/// additions.
pub(super) fn site_bound(
    terms: &[Term],
    k_lo: f64,
    k_hi: f64,
    xc: &[f64],
    xr: &[f64],
    sweeps: usize,
    charge_residual: bool,
) -> Option<SiteBound> {
    let d = xc.len();
    if terms.is_empty() || xr.len() != d || !(k_lo.is_finite() && k_hi.is_finite() && k_lo <= k_hi)
    {
        return None;
    }
    let quad = assemble(terms, xr)?;
    let dev = deviation(&quad.h[..d * d], &quad.g[..d], xr, sweeps, charge_residual)?;
    let tot = next_up(next_up(next_up(quad.e_asm + quad.res) + quad.e_coef) + dev.ech);
    let q = quad.q;
    let hi = next_up(q + dev.up + tot + k_hi + gslack(6, q.abs() + dev.up + tot + k_hi.abs()));
    let lo = next_down(q - dev.dn - tot + k_lo - gslack(6, q.abs() + dev.dn + tot + k_lo.abs()));
    let (mut gx, mut sgx) = (0.0f64, 0.0f64);
    for k in 0..d {
        gx += quad.g[k] * xc[k];
        sgx += (quad.g[k] * xc[k]).abs();
    }
    let c_lo = next_down(
        q - gx - dev.lr - tot + k_lo - gslack(d + 8, q.abs() + sgx + dev.lr + tot + k_lo.abs()),
    );
    let c_hi = next_up(
        q - gx + dev.ur + tot + k_hi + gslack(d + 8, q.abs() + sgx + dev.ur + tot + k_hi.abs()),
    );
    let ok = [lo, hi, c_lo, c_hi].iter().all(|v| v.is_finite()) && lo <= hi;
    ok.then_some(SiteBound {
        lo,
        hi,
        g: quad.g,
        c_lo,
        c_hi,
    })
}

/// Box centre and a rigorous half-width: the box `[bl, bh]` lies inside
/// `xc +- xr` exactly. `false` on a non-finite box.
pub(super) fn centre_radius(bl: &[f64], bh: &[f64], xc: &mut [f64], xr: &mut [f64]) -> bool {
    let mut ok = true;
    for k in 0..bl.len() {
        let c = 0.5 * bl[k] + 0.5 * bh[k];
        xc[k] = c;
        xr[k] = next_up((c - bl[k]).max(bh[k] - c));
        ok &= c.is_finite() && xr[k].is_finite() && xr[k] >= 0.0;
    }
    ok
}

/// Apply the joint bound at `site` to its node value `v` (the site's own
/// `Op::Sparse`, width 1, already computed and concretized) over every
/// domain of the chunk: intersect its concrete interval and store the
/// joint planes (`v.joint_lo` / `v.joint_hi`, `[m][d + 1]`, `v.joint_ok`).
/// A domain the pass cannot bound (poisoned, non-finite) keeps its bounds and
/// gets no planes: the backward pass then falls back to the node's own forms.
pub(super) fn apply_site(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    site: &JointSite,
    v: &mut NodeVal,
) {
    let (m, d) = (ch.m, ch.d);
    let d1 = d + 1;
    v.joint_lo = vec![0.0; m * d1];
    v.joint_hi = vec![0.0; m * d1];
    v.joint_ok = vec![false; m];
    let get = |i: usize| vals.get(i).and_then(Option::as_ref);
    let (Some(va), Some(vb), Some(vz)) = (get(site.a), get(site.b), get(site.mul)) else {
        return;
    };
    if d > MD {
        return;
    }
    let (mut xc, mut xr) = ([0.0f64; MD], [0.0f64; MD]);
    let mut terms: Vec<Term> = Vec::new();
    for b in 0..m {
        if v.bad[b] || va.bad[b] || vb.bad[b] || vz.bad[b] {
            continue;
        }
        let (bl, bh) = (&ch.lo[b * d..(b + 1) * d], &ch.hi[b * d..(b + 1) * d]);
        if !centre_radius(bl, bh, &mut xc[..d], &mut xr[..d]) {
            continue;
        }
        let (rows, k_lo, k_hi, _) = fam.sparse(site.node, b);
        terms.clear();
        let mut ok = !rows.is_empty() && !k_lo.is_empty() && !k_hi.is_empty();
        if ok {
            for t in &rows[0] {
                let k = t.idx as usize * m + b;
                let (Some(a), Some(bb)) = (
                    operand_affine(va, k, d, &xc[..d], &xr[..d]),
                    operand_affine(vb, k, d, &xc[..d], &xr[..d]),
                ) else {
                    ok = false;
                    break;
                };
                terms.push(Term {
                    w: t.coef,
                    w_err: t.err,
                    z_mag: vz.lo[k].abs().max(vz.hi[k].abs()),
                    a,
                    b: bb,
                });
            }
        }
        if !ok || terms.is_empty() {
            continue;
        }
        let Some(sb) = site_bound(
            &terms,
            k_lo[0],
            k_hi[0],
            &xc[..d],
            &xr[..d],
            JACOBI_SWEEPS,
            true,
        ) else {
            continue;
        };
        // Width 1: the site's element at domain b is index b.
        v.lo[b] = v.lo[b].max(sb.lo);
        v.hi[b] = v.hi[b].min(sb.hi);
        if !matches!(
            v.lo[b].partial_cmp(&v.hi[b]),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            // Two valid enclosures of one value always overlap: an empty
            // meet can only be numerical, and it poisons the domain.
            v.bad[b] = true;
            continue;
        }
        v.zmax[b] = v.lo[b].abs().max(v.hi[b].abs());
        let r = b * d1;
        v.joint_lo[r..r + d].copy_from_slice(&sb.g[..d]);
        v.joint_lo[r + d] = sb.c_lo;
        v.joint_hi[r..r + d].copy_from_slice(&sb.g[..d]);
        v.joint_hi[r + d] = sb.c_hi;
        v.joint_ok[b] = true;
    }
}

/// Element `k = i * m + b` of an operand node as a midpoint affine: the
/// forward forms, plus the CROWN certificates when the forward pass stored
/// them (`NodeVal::cert_lo` / `cert_hi`, same layout as the forms).
fn operand_affine(val: &NodeVal, k: usize, d: usize, xc: &[f64], xr: &[f64]) -> Option<Affine> {
    let d1 = d + 1;
    let r = k * d1;
    if val.lo_form.len() < r + d1 || val.lo.len() <= k {
        return None;
    }
    let vmag = val.lo[k].abs().max(val.hi[k].abs());
    let (fl, fh) = (&val.lo_form[r..r + d1], &val.hi_form[r..r + d1]);
    let certified =
        val.cert_lo.len() == val.lo_form.len() && val.cert_hi.len() == val.hi_form.len();
    if certified {
        reduce(
            &[fl, &val.cert_lo[r..r + d1]],
            &[fh, &val.cert_hi[r..r + d1]],
            xc,
            xr,
            vmag,
        )
    } else {
        reduce(&[fl], &[fh], xc, xr, vmag)
    }
}

#[cfg(test)]
#[path = "joint_tests.rs"]
mod tests;
