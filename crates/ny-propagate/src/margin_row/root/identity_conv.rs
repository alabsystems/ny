// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact specialization of the first f32 Conv applied to an identity tableau.
//!
//! Injective gathers give each coefficient at most one nonzero product: a
//! normal f32 weight times one. All other terms are zeros. Under nearest-even
//! arithmetic the dense accumulation and direct scatter therefore have the
//! same coefficient bits. Its functional error is the outward sum of actual
//! weight-conversion errors; no coefficient dot product remains to charge.
//! Fold-ball and bias charges remain shared. Unsupported inputs retain the
//! dense path.

use super::*;

pub(super) struct Plan {
    p: usize,
    k: usize,
    n_out: usize,
    weights: Vec<f32>,
}

fn qualified_thread() -> bool {
    use std::hint::black_box;
    let one = black_box(1.0_f32);
    let above = black_box(f32::from_bits((127 - 24) << 23));
    let below = black_box(f32::from_bits((127 - 25) << 23));
    ny_core::has_f64_interval_proof_environment()
        && black_box(one + above).to_bits() == 1.0_f32.to_bits()
        && black_box(one - below).to_bits() == 1.0_f32.to_bits()
}

pub(super) fn plan(
    c: &ConvOp,
    n_in: usize,
    n_out: usize,
    xabs: &[f64],
    s_xabs: f64,
) -> Option<Plan> {
    if c.input != 0
        || n_in == 0
        || xabs.len() != n_in
        || !s_xabs.is_finite()
        || s_xabs < 0.0
        || !xabs
            .iter()
            .all(|&v| v.to_bits() == 0 || (v.is_normal() && v > 0.0))
    {
        return None;
    }
    let (co, ci, kh, kw) = c.kernel;
    let p = c.oshape.1.checked_mul(c.oshape.2)?;
    let k = ci.checked_mul(kh)?.checked_mul(kw)?;
    let src_len = c
        .ishape
        .0
        .checked_mul(c.ishape.1)?
        .checked_mul(c.ishape.2)?;
    n_in.checked_add(3)?;
    k.checked_add(8)?;
    if co == 0
        || p == 0
        || k == 0
        || c.oshape.0 != co
        || c.ishape.0 != ci
        || src_len != n_in
        || c.k_fwd != k
        || co.checked_mul(p)? != n_out
        || co.checked_mul(k)? != c.wmat.len()
        || k.checked_mul(p)? != c.gather.len()
        || c.bias.len() != co
        || c.bias_err.len() != co
    {
        return None;
    }
    // FP state is thread-local. Qualify the actual worker pool as well as the
    // caller, without changing either one's rounding or denormal controls.
    if !qualified_thread()
        || !rayon::broadcast(|_| qualified_thread())
            .into_iter()
            .all(|ok| ok)
    {
        return None;
    }
    let mut weights = Vec::with_capacity(c.wmat.len());
    for &w in &c.wmat {
        if !w.is_finite() {
            return None;
        }
        let v = w as f32;
        // Bit-based zero test cannot mistake a subnormal for zero under DAZ.
        if v.to_bits() & 0x7fff_ffff != 0 && !v.is_normal() {
            return None;
        }
        weights.push(v);
    }
    let mut seen = vec![usize::MAX; n_in];
    for sp in 0..p {
        for t in 0..k {
            let gi = c.gather[t * p + sp];
            if gi == usize::MAX {
                continue;
            }
            if gi >= n_in || seen[gi] == sp {
                return None;
            }
            seen[gi] = sp;
        }
    }
    Some(Plan {
        p,
        k,
        n_out,
        weights,
    })
}

pub(super) fn packed_center<T: TabElem>(
    c: &ConvOp,
    plan: &Plan,
    n_in: usize,
    deadline: Option<Instant>,
) -> Result<sparse_stem::PackedRows<T>> {
    sparse_stem::PackedRows::from_identity(c, n_in, plan.p, plan.k, &plan.weights, deadline)
}

#[cfg(test)]
pub(super) fn capture_packed<T: TabElem>(
    rows: &sparse_stem::PackedRows<T>,
    g: &[f64],
    support: &[u64],
) {
    if TRACE.with(|slot| slot.borrow().is_some()) {
        let (m, d) = rows.materialize();
        capture(true, &m, &d, g, support);
    }
}

pub(super) fn center<T: TabElem>(c: &ConvOp, plan: &Plan, dst: &mut Array2<T>) {
    debug_assert_eq!(dst.nrows(), plan.n_out);
    dst.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut row)| {
            let oc = j / plan.p;
            let sp = j % plan.p;
            for t in 0..plan.k {
                let gi = c.gather[t * plan.p + sp];
                if gi == usize::MAX {
                    continue;
                }
                let w = plan.weights[oc * plan.k + t];
                if w.to_bits() & 0x7fff_ffff != 0 {
                    row[gi] = T::from_acc(w);
                }
            }
        });
}

/// Functional error of direct identity scatter, before the shared bias fold.
///
/// Each valid gather scatters one captured f32 weight q, exact in either storage
/// lane, with D=0. The functional charge is sum_t |w_t-q_t| H_t, where H_t
/// rounds the stored input magnitude upward to enclose |mid_t| + rad_t.
/// No coefficient accumulation, input conversion or directed D store remains.
///
/// Round each nonzero gap, product and sum upward, including underflow to zero.
/// Include nonzero weights that cast to zero, and preserve overflow as infinity.
/// Keep one entry per output even for zero error: a nonempty vector enables
/// the shared f32 bias-store charge.
pub(super) fn error(c: &ConvOp, plan: &Plan, xabs: &[f64]) -> Vec<f64> {
    let gaps: Vec<f64> = c
        .wmat
        .iter()
        .zip(&plan.weights)
        .map(|(&w, &captured)| {
            let q = f64::from(captured);
            if w == q {
                0.0
            } else {
                next_up((w - q).abs())
            }
        })
        .collect();
    let magnitudes: Vec<f64> = xabs.iter().map(|&v| next_up(v)).collect();
    let mut error = vec![0.0; plan.n_out];
    error.par_iter_mut().enumerate().for_each(|(j, bound)| {
        let oc = j / plan.p;
        let sp = j % plan.p;
        for t in 0..plan.k {
            let gi = c.gather[t * plan.p + sp];
            if gi == usize::MAX {
                continue;
            }
            let gap = gaps[oc * plan.k + t];
            if gap != 0.0 {
                let term = next_up(gap * magnitudes[gi]);
                *bound = if *bound == 0.0 {
                    term
                } else {
                    next_up(*bound + term)
                };
            }
        }
    });
    error
}

#[cfg(not(test))]
pub(super) const fn force_dense() -> bool {
    false
}

#[cfg(test)]
pub(super) fn force_dense() -> bool {
    TRACE.with(|slot| slot.borrow().as_ref().is_some_and(|state| state.0))
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct IdentityConvTrace {
    pub implicit: bool,
    pub shape: (usize, usize),
    pub m: Vec<u64>,
    pub d: Vec<u64>,
    pub g: Vec<u64>,
    pub support: Vec<u64>,
}

#[cfg(test)]
thread_local! {
    static TRACE: std::cell::RefCell<Option<(bool, Vec<IdentityConvTrace>)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_identity_conv_trace_for_test<R>(
    force_dense: bool,
    f: impl FnOnce() -> R,
) -> (R, Vec<IdentityConvTrace>) {
    struct Restore(Option<(bool, Vec<IdentityConvTrace>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TRACE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let restore = Restore(TRACE.with(|slot| slot.replace(Some((force_dense, Vec::new())))));
    let result = f();
    let trace = TRACE.with(|slot| slot.borrow_mut().take().expect("armed trace").1);
    drop(restore);
    (result, trace)
}

#[cfg(test)]
pub(super) fn capture<T: TabElem>(
    implicit: bool,
    m: &Array2<T>,
    d: &Array2<T>,
    g: &[f64],
    support: &[u64],
) {
    TRACE.with(|slot| {
        if let Some((_, traces)) = slot.borrow_mut().as_mut() {
            traces.push(IdentityConvTrace {
                implicit,
                shape: m.dim(),
                m: m.iter().map(|x| x.wide().to_bits()).collect(),
                d: d.iter().map(|x| x.wide().to_bits()).collect(),
                g: g.iter().map(|x| x.to_bits()).collect(),
                support: support.to_vec(),
            });
        }
    });
}

#[cfg(test)]
mod tests;
