// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sum-to-1-aware ("simplex water-filling") IBP tightening for `softmax @ V`.
//!
//! In attention, `Y = P @ V` where `P` is a row-stochastic softmax output:
//! each row `P[i, :]` satisfies `P[i,k] >= 0` and `sum_k P[i,k] = 1`. Term-wise
//! interval IBP keeps only `P[i,k] in [0,1]` and DROPS the `sum_k = 1`
//! constraint, so it over-counts the dot product `Y[i,j] = sum_k P[i,k] V[k,j]`
//! (e.g. it bounds the upper by `sum_k Ph[i,k] Vh[k,j]`, with the `Ph` mass far
//! exceeding 1). Restoring the simplex constraint bounds each output by an exact
//! LP over the box-intersected simplex, which is strictly tighter.
//!
//! This is the same lever as the constant-weight "DFL envelope"
//! (`network::core::graph::ibp::dfl_envelope`), generalised to a PERTURBED `V`
//! (the attention value matrix). It is used both by the explicit
//! `Softmax -> MatMul(V)` graph pattern (via dfl_envelope) and by the fused
//! [`crate::layers::SelfAttentionLayer`] IBP path.
//!
//! # Soundness
//!
//! For the true `(P, V)`: `P[i,:]` lies in the box-intersected simplex
//! `S_i = {p : Pl[i,:] <= p <= Ph[i,:], sum_k p_k = 1}` (the IBP box is sound and
//! softmax rows sum to exactly 1), and `V[k,j] in [Vl[k,j], Vh[k,j]]`. Since
//! `p_k >= 0`:
//!   `Y[i,j] = sum_k p_k V[k,j] <= sum_k p_k Vh[k,j] <= max_{p in S_i} sum_k p_k Vh[k,j]`,
//!   `Y[i,j] = sum_k p_k V[k,j] >= sum_k p_k Vl[k,j] >= min_{p in S_i} sum_k p_k Vl[k,j]`.
//! The two extremal LPs over `S_i` (a box ∩ one hyperplane) are solved exactly by
//! greedy water-filling ([`simplex_lp_max`]). We round the f64 LP value OUTWARD on
//! the f32 cast so the stored interval still encloses the true real value.
//! Intersecting the per-element envelope with the (also-sound) term-wise IBP
//! output only tightens — never widens.

use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Maximize `sum_k p_k * a_k` over the box-intersected simplex
/// `{pl_k <= p_k <= ph_k, sum_k p_k = 1}`. Exact greedy water-filling: start at
/// the lower box corner `p = pl` and pour the residual mass `1 - sum(pl)` into
/// coordinates in DESCENDING `a_k`, each up to its cap `ph_k`.
///
/// Optimality: the feasible set is a box intersected with one equality
/// hyperplane; the LP optimum saturates the highest-coefficient coordinates
/// first (continuous-knapsack / transportation argument). Computed in f64.
///
/// SOUNDNESS of the thresholds: a sound IBP box always has `sum pl <= 1 <= sum ph`
/// (the true row sums to 1 and lies in the box). We reject STRICTLY when
/// `sum pl > 1.0` (the lower floor would already oversum, so starting at `pl`
/// could use mass `> 1` and the result need not upper-bound the true value), and
/// reject when `sum ph < 1 - 1e-5` (only rounding noise; a sound IBP has
/// `sum ph >= 1`). Both rejections return `None` so the caller keeps the existing
/// sound bound (never widening).
pub(crate) fn simplex_lp_max(pl: &[f32], ph: &[f32], a: &[f32]) -> Option<f64> {
    let n = pl.len();
    debug_assert_eq!(ph.len(), n);
    debug_assert_eq!(a.len(), n);
    if n == 0 {
        return None;
    }
    let s0: f64 = pl.iter().map(|&x| x as f64).sum();
    let smax: f64 = ph.iter().map(|&x| x as f64).sum();
    if s0 > 1.0 || smax < 1.0 - 1e-5 {
        return None;
    }
    let mut p: Vec<f64> = pl.iter().map(|&x| x as f64).collect();
    let mut budget = (1.0 - s0).max(0.0);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap_or(std::cmp::Ordering::Equal));
    for &j in &order {
        if budget <= 0.0 {
            break;
        }
        let room = (ph[j] as f64 - p[j]).max(0.0);
        let add = room.min(budget);
        p[j] += add;
        budget -= add;
    }
    Some((0..n).map(|j| p[j] * a[j] as f64).sum())
}

/// Minimize `sum_k p_k * a_k` over the same box-intersected simplex.
/// `min sum p_k a_k = -max sum p_k (-a_k)`.
pub(crate) fn simplex_lp_min(pl: &[f32], ph: &[f32], a: &[f32]) -> Option<f64> {
    let neg: Vec<f32> = a.iter().map(|&x| -x).collect();
    simplex_lp_max(pl, ph, &neg).map(|v| -v)
}

/// Tighten the term-wise IBP output of `P @ V` using the simplex (sum-to-1)
/// structure of the softmax rows `P`. Operates on the **2-D last-two-axes**
/// batch slices: `probs` and `out_ibp` share leading batch dims and trailing
/// `(M, K)` / `(M, N)`; `v` shares the same leading batch dims with trailing
/// `(K, N)` (or `(N, K)` if `v_transposed`).
///
/// Returns the tightened tensor (tighten-or-equal vs `out_ibp`). Falls back to a
/// clone of `out_ibp` when shapes/preconditions do not match, so it is always a
/// safe no-op on unsupported inputs.
///
/// `scale` (e.g. attention `1/sqrt(d)`) is NOT applied here — `probs @ V` carries
/// no scale in standard attention. Pass already-scaled `out_ibp`/`v` if needed.
pub(crate) fn tighten_softmax_v_ibp(
    probs: &BoundedTensor,
    v: &BoundedTensor,
    out_ibp: &BoundedTensor,
    v_transposed: bool,
) -> BoundedTensor {
    let p_shape = probs.shape();
    let v_shape = v.shape();
    let o_shape = out_ibp.shape();
    let pnd = p_shape.len();
    let vnd = v_shape.len();
    let ond = o_shape.len();
    // Require >= 2-D and matching batch ranks.
    if pnd < 2 || vnd != pnd || ond != pnd {
        return out_ibp.clone();
    }
    let m = p_shape[pnd - 2];
    let k = p_shape[pnd - 1];
    let (n, contract) = if v_transposed {
        (v_shape[vnd - 2], v_shape[vnd - 1])
    } else {
        (v_shape[vnd - 1], v_shape[vnd - 2])
    };
    if contract != k || o_shape[ond - 2] != m || o_shape[ond - 1] != n {
        return out_ibp.clone();
    }
    // Batch dims must match across all three.
    if p_shape[..pnd - 2] != v_shape[..vnd - 2] || p_shape[..pnd - 2] != o_shape[..ond - 2] {
        return out_ibp.clone();
    }
    if m == 0 || k == 0 || n == 0 {
        return out_ibp.clone();
    }
    let batch: usize = p_shape[..pnd - 2].iter().product();

    let (pl, ph) = probs.lower_upper();
    let (vl, vh) = v.lower_upper();
    let (ol, ou) = out_ibp.lower_upper();

    // Flatten to (batch, rows, cols) views via raw slices; ndarray stores
    // row-major (C order), so element [b, r, c] is at b*R*C + r*C + c.
    let pl_s = match pl.as_slice() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };
    let ph_s = match ph.as_slice() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };
    let vl_s = match vl.as_slice() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };
    let vh_s = match vh.as_slice() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };
    let mut new_l = ol.to_owned();
    let mut new_u = ou.to_owned();
    let nl = match new_l.as_slice_mut() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };
    let nu = match new_u.as_slice_mut() {
        Some(s) => s,
        None => return out_ibp.clone(),
    };

    // V slice layout per batch: (K, N) normal or (N, K) transposed.
    let v_rows = if v_transposed { n } else { k };
    let v_cols = if v_transposed { k } else { n };

    let mut p_lo = vec![0.0f32; k];
    let mut p_hi = vec![0.0f32; k];
    let mut vcol_lo = vec![0.0f32; k];
    let mut vcol_hi = vec![0.0f32; k];

    for b in 0..batch {
        let p_base = b * m * k;
        let v_base = b * v_rows * v_cols;
        let o_base = b * m * n;

        // V column getter: V[k_idx, j] (normal) or V[j, k_idx] (transposed).
        let vget = |arr: &[f32], kk: usize, j: usize| -> f32 {
            if v_transposed {
                arr[v_base + j * v_cols + kk] // (N, K): row j, col kk
            } else {
                arr[v_base + kk * v_cols + j] // (K, N): row kk, col j
            }
        };

        for i in 0..m {
            // Per-row probability interval.
            let mut bad = false;
            for kk in 0..k {
                let lo = pl_s[p_base + i * k + kk];
                let hi = ph_s[p_base + i * k + kk];
                if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < lo {
                    bad = true;
                    break;
                }
                p_lo[kk] = lo;
                p_hi[kk] = hi;
            }
            if bad {
                continue;
            }
            // Strict lower-sum feasibility (see simplex_lp_max soundness note).
            let sum_lo: f64 = p_lo.iter().map(|&x| x as f64).sum();
            let sum_hi: f64 = p_hi.iter().map(|&x| x as f64).sum();
            if sum_lo > 1.0 || sum_hi < 1.0 - 1e-5 {
                continue;
            }

            for j in 0..n {
                let mut vbad = false;
                for kk in 0..k {
                    let lo = vget(vl_s, kk, j);
                    let hi = vget(vh_s, kk, j);
                    if !lo.is_finite() || !hi.is_finite() {
                        vbad = true;
                        break;
                    }
                    vcol_lo[kk] = lo;
                    vcol_hi[kk] = hi;
                }
                if vbad {
                    continue;
                }
                let hi = simplex_lp_max(&p_lo, &p_hi, &vcol_hi);
                let lo = simplex_lp_min(&p_lo, &p_hi, &vcol_lo);
                let (Some(hi), Some(lo)) = (hi, lo) else {
                    continue;
                };
                let env_hi = next_up_f32(hi as f32);
                let env_lo = next_down_f32(lo as f32);
                if !env_hi.is_finite() || !env_lo.is_finite() {
                    continue;
                }
                let oidx = o_base + i * n + j;
                let tl = nl[oidx].max(env_lo);
                let tu = nu[oidx].min(env_hi);
                if tl <= tu {
                    nl[oidx] = tl;
                    nu[oidx] = tu;
                } else {
                    // Disjoint after intersection: the envelope is the
                    // authoritative sound range (true value lies in it).
                    nl[oidx] = env_lo;
                    nu[oidx] = env_hi;
                }
            }
        }
    }

    BoundedTensor::new(new_l, new_u).unwrap_or_else(|_| out_ibp.clone())
}

#[cfg(test)]
mod tests;
