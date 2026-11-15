// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-ibp-seed: certified interval (IBP) boxes intersected SHRINK-ONLY with
//! the margin-row root tableau's trunk boxes (FRONTIER_PLAN §7.F2).
//!
//! # Why
//!
//! On IBP-trained nets (the `challenging_certified_training_2026` eps8 CNN7s)
//! interval propagation DOMINATES linear relaxation: a vanilla CROWN
//! back-substitution tightened 0 of 4,378 IBP-unstable neurons on idx8072, and
//! a 0.1 s CROWN-IBP proves idx5323 at the root (+0.086) where the lane's root
//! sits at -4.818 with 8 classes open. The lane built its trunk boxes from the
//! forward DeepPoly (M, D) tableau intersected with `#backward-interm` only and
//! never looked at IBP, so on these nets it started several units below the
//! bound a 0.1 s probe reaches.
//!
//! # What
//!
//! The root walk (`root.rs::walk_ops`) carries, next to the tableau, one
//! CERTIFIED interval `[lo_t, hi_t]` per live tensor `t`, starting from the
//! declared input box:
//!
//! * Conv / ChannelAffine: `y = W v + b` from `v in [mid - rad, mid + rad]`,
//!   centre `fl(W mid + b)`, radius `|W| rad` (one `conv_apply_forward` with
//!   `abs_w = true`), plus every rounding and fold charge (below), rounded
//!   OUTWARD;
//! * Flatten: identity; Add: `[next_down(la + lb), next_up(ha + hb)]` (only the
//!   pure diagnostic chain uses it — the seed declines residual trunks, below);
//! * Relu (trunk layer `k`): `[max(l_k, 0), max(u_k, 0)]` of layer `k`'s FINAL
//!   published box (after this intersection AND `#backward-interm`), so every
//!   layer's IBP starts from the tightest predecessor available.
//!
//! At each trunk ReLU the tableau box is concretized, the `#backward-interm`
//! `bi_cal` capture takes the FORWARD-ONLY box, and only then this module
//! intersects `l = max(l, lo)`, `u = min(u, hi)`, BEFORE `bi.tighten_layer`
//! (the ordering and its reasons are documented at the call site in
//! `root.rs`).
//!
//! # Soundness
//!
//! Claim, per tensor: for EVERY input in the root box and EVERY member of the
//! certified fold ball (a folded op's true parameters anywhere inside
//! `|W - W'| <= rel |W'|`, `|b - b'| <= bias_err`, `|s - s'| <= srel |s'|`,
//! `|t - t'| <= shift_err` — the same ball the tableau's #fold-ball charge
//! covers), the exact real-arithmetic value lies in `[lo_t, hi_t]`. By
//! induction over the ops (it holds for the input box by definition, and on a
//! Tier-2 epoch rebuild the root box is a superset of the split subdomain):
//!
//! * the input tensor `v` lies in `[mid - rad, mid + rad]` exactly:
//!   `mid = fl((lo + hi) / 2)`, `rad = next_up(max(fl(hi - mid), fl(mid - lo)))`
//!   and `next_up` of a correctly rounded non-negative difference dominates
//!   the exact difference — the tableau's own input-box formula;
//! * Conv: `y - (W' mid + b') = W' d + (W - W') v + (b - b')` with `|d| <= rad`,
//!   so `|y - (W' mid + b')| <= |W'| rad + rel |W'| (|mid| + rad) + bias_err`.
//!   The float centre `c = fl(fl(sum W' mid) + b')` misses `W' mid + b'` by at
//!   most `gamma_k |W'| |mid|` (Higham, ANY summation order, FMA included) plus
//!   gradual-underflow dust, and `u |c| / (1 - u) <= 2u |c|` for the bias add.
//!   `A = cert(fl(sum |W'| rad))` and `B = cert(fl(sum |W'| |mid|))` dominate
//!   the exact non-negative sums with the #fold-ball `cert` (`next_up(v (1 +
//!   2 gamma_{k+2}) + MIN_POSITIVE)`, `root.rs::conv_fold_ball`). The radius is
//!   `R = A + gamma_{k+2} B + rel (A + B) + bias_err + 2u |c| + MIN_POSITIVE`,
//!   every addition rounded up (each is one correctly rounded op on
//!   non-negative upper bounds, so its `next_up` dominates the exact sum), and
//!   the box is `[next_down(fl(c - R)), next_up(fl(c + R))]`, which contains
//!   `[c - R, c + R]` exactly.
//! * ChannelAffine: the same with `k = 1`: `c = fl(fl(s' mid) + t')`,
//!   `R = next_up(|s'| rad) + 2u |fl(s' mid)| + srel next_up(|s'|
//!   next_up(|mid| + rad)) + shift_err + 2u |c| + MIN_POSITIVE`.
//! * The CHARGES ARE SIGN-SAFE: every term is a non-negative magnitude added
//!   symmetrically to a radius, so none depends on the sign of an input or a
//!   coefficient — the class of leak `#root-rounding` and `#fold-ball` closed
//!   in the tableau (a coefficient widening that moves a wall INWARD at
//!   negative inputs) cannot arise in a centre/radius interval.
//! * Relu: `relu` is monotone and `max` is exact.
//! * The intersection of two certified enclosures is a certified enclosure;
//!   `max` of lowers / `min` of uppers is exact and SHRINK-ONLY. Two sound
//!   enclosures of a non-empty set cannot cross, so a crossing is a DEFECT
//!   signal (or an empty split subdomain inside an epoch rebuild): it is NEVER
//!   an emptiness certificate. One crossed neuron refuses the seed for its
//!   WHOLE layer ([`intersect_layer_or_refuse`]): every neuron of that layer
//!   keeps its tableau box, the event is counted and printed loudly, and the
//!   build goes on (fail closed).
//! * Downstream, `gates_from_box` + `repair_upper_lines` re-certify the
//!   frozen gates on the PUBLISHED box, exactly as after `#backward-interm`.
//!
//! The false-`unsat` risk of this module is an UNDER-charged radius. It is
//! pinned by the exact dyadic-rational containment tests
//! (`tests/ibp_seed_tests.rs`), including a code-mutation pin that drops the
//! fold charge and must be caught.
//!
//! # Scope (phase 1)
//!
//! Chain trunks only (the CNN7 family: a conv chain plus the FC-BN head). A
//! trunk with a residual `Add` DECLINES with `reason=residual` and intersects
//! nothing, so the shared lane stays byte-identical on the cifar100 /
//! tinyimagenet resnets even with the key armed. Parity mode declines
//! (`reason=parity`): IBP is a verdict-mode (Outward) certificate. The head
//! y-box (the first head Gemm's output) is NOT seeded here — the root walk
//! stops at the last trunk ReLU.
//!
//! # Observable (every armed build, unconditional, fixed schema)
//!
//! One line per trunk ReLU, then one terminal line:
//!
//! ```text
//! [mr-ibp-seed] layer=<k> neurons=<n> tightened=<t> stabilized=<s> crossed=<c> declined=<reason|-> unst_before=<a> unst_after=<b> nonfinite=<x>
//! [mr-ibp-seed] done layers=<L> tightened=<T> stabilized=<S> crossed=<C> nonfinite=<X> unst_before=<A> unst_after=<B> declined=<reason|-> rss_need_mb=<f> secs=<f> complete=<0|1>
//! ```
//!
//! `unst_before` / `unst_after` are the layer's unstable counts before and
//! after THIS intersection (backward-interm may stabilize more afterwards).
//! The terminal line is printed on EVERY exit of an armed walk: `complete=1`
//! when the walk finished, `complete=0` from `Drop` when it returned early
//! (deadline, error), counting only the layers it reached. An armed run
//! without these lines is VOID. Dark (the default) prints nothing and the walk
//! is byte-identical.

use std::time::Instant;

use ndarray::Array2;

use super::net::{conv_apply_forward, ConvOp, TwinNet, TwinOp};
use super::rounding::{gamma_n, next_down, next_up, RoundMode, UNIT};

/// Admission ceiling when the memory headroom is unobservable (competition runs
/// uncapped; `root_certify_once::UNOBSERVED_HEADROOM_CAP` is the sibling).
const UNOBSERVED_HEADROOM_CAP: u64 = 4 << 30;

/// Why an ARMED seed intersects nothing on this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// A residual `Add` before the last trunk ReLU (phase 1 is chain-only).
    Residual,
    /// Not the certified-outward verdict mode.
    Parity,
    /// The interval state was refused RSS admission.
    Rss,
    /// An op or shape the chain does not describe (unreachable on a compiled
    /// net; fail closed rather than guess).
    Op,
}

impl Decline {
    /// Token printed as `declined=<token>`.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Residual => "residual",
            Self::Parity => "parity",
            Self::Rss => "rss",
            Self::Op => "op",
        }
    }
}

/// One trunk layer's intersection counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerSeed {
    /// Layer width.
    pub neurons: usize,
    /// Neurons whose box strictly shrank on either side.
    pub tightened: usize,
    /// Neurons unstable before the intersection and stable after it.
    pub stabilized: usize,
    /// Crossed intersections (defect signal): the tableau box was kept.
    pub crossed: usize,
    /// Non-finite IBP entries skipped (the tableau box was kept).
    pub nonfinite: usize,
    /// Unstable neurons before the intersection.
    pub unst_before: usize,
    /// Unstable neurons after the intersection.
    pub unst_after: usize,
}

/// A certified interval `[lo, hi]` per neuron of one tensor.
///
/// `pub(super)` (with the four box builders below) for `#root-box-source`
/// (`root_box.rs`), whose patch-backward root walk carries the same certified
/// reference boxes; visibility only.
pub(super) type IBox = (Vec<f64>, Vec<f64>);

/// Is this `(l, u)` pair a ReLU-unstable neuron (the `gates_from_box` rule)?
#[inline]
fn unstable(l: f64, u: f64) -> bool {
    l < 0.0 && u > 0.0
}

/// SHRINK-ONLY intersection of one layer's tableau box `(l, u)` with its IBP
/// box `(bl, bu)`, in place. Crossed and non-finite entries keep the tableau
/// box and are only counted (fail closed). Shapes must agree; a mismatch
/// returns `None` and touches nothing.
pub fn intersect_layer(l: &mut [f64], u: &mut [f64], bl: &[f64], bu: &[f64]) -> Option<LayerSeed> {
    let n = l.len();
    if u.len() != n || bl.len() != n || bu.len() != n {
        return None;
    }
    let mut st = LayerSeed {
        neurons: n,
        ..LayerSeed::default()
    };
    for j in 0..n {
        let was_unstable = unstable(l[j], u[j]);
        st.unst_before += usize::from(was_unstable);
        let (lb, ub) = (bl[j], bu[j]);
        if !(lb.is_finite() && ub.is_finite()) {
            st.nonfinite += 1;
            st.unst_after += usize::from(was_unstable);
            continue;
        }
        let nl = l[j].max(lb);
        let nu = u[j].min(ub);
        if nl > nu {
            // Two certified enclosures of a non-empty range cannot cross: a
            // defect signal, NOT an emptiness certificate. Keep the tableau box.
            st.crossed += 1;
            st.unst_after += usize::from(was_unstable);
            continue;
        }
        if nl > l[j] || nu < u[j] {
            st.tightened += 1;
            if was_unstable && !unstable(nl, nu) {
                st.stabilized += 1;
            }
            l[j] = nl;
            u[j] = nu;
        }
        st.unst_after += usize::from(unstable(l[j], u[j]));
    }
    Some(st)
}

/// LAYER-LEVEL commit of [`intersect_layer`] (what the root walk publishes):
/// the intersection is computed on a copy and written back only when NO
/// neuron of the layer crossed. One crossing refuses the seed for the WHOLE
/// layer, which keeps its tableau box (FRONTIER_PLAN §7.F2: "a crossing
/// refuses the seed for that layer, keeps the tableau box"). Two certified
/// enclosures of a non-empty range cannot cross, so a crossed neuron is
/// evidence that one side is defective on this layer — and the other
/// neurons' IBP tightenings came out of the same conv / affine charge as the
/// one that crossed, so they are no more trustworthy than it is (fail
/// closed). A refused layer reports its crossings and non-finite skips with
/// `tightened = stabilized = 0` and `unst_after = unst_before`. Shapes must
/// agree; a mismatch returns `None` and touches nothing.
pub fn intersect_layer_or_refuse(
    l: &mut [f64],
    u: &mut [f64],
    bl: &[f64],
    bu: &[f64],
) -> Option<LayerSeed> {
    let (mut nl, mut nu) = (l.to_vec(), u.to_vec());
    let st = intersect_layer(&mut nl, &mut nu, bl, bu)?;
    if st.crossed > 0 {
        return Some(LayerSeed {
            tightened: 0,
            stabilized: 0,
            unst_after: st.unst_before,
            ..st
        });
    }
    l.copy_from_slice(&nl);
    u.copy_from_slice(&nu);
    Some(st)
}

/// Certified centre and radius of an interval: `mid +- rad` covers
/// `[lo, hi]` in exact arithmetic (the tableau's own input-box formula).
/// `None` on a non-finite or reversed interval.
#[inline]
fn mid_rad(lo: f64, hi: f64) -> Option<(f64, f64)> {
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return None;
    }
    // Parity-critical formula shared with `root.rs` (`mid = (lo + hi) / 2`).
    #[allow(clippy::manual_midpoint)]
    let mid = (lo + hi) / 2.0;
    let rad = next_up((hi - mid).max(mid - lo));
    (mid.is_finite() && rad.is_finite()).then_some((mid, rad))
}

/// `>=` the exact sum of the non-negative products whose float sum is `v`
/// (`g` = `gamma_{k+2}` of the product count): the #fold-ball `cert`
/// (`root.rs::conv_fold_ball`) verbatim.
#[inline]
fn cert_sum(v: f64, g: f64) -> f64 {
    next_up(v * (1.0 + 2.0 * g) + f64::MIN_POSITIVE)
}

/// Close a certified centre/radius pair into an outward box.
#[inline]
fn close(c: f64, r: f64) -> (f64, f64) {
    (next_down(c - r), next_up(c + r))
}

#[cfg(test)]
thread_local! {
    /// TEST-ONLY mutation switch: drop the FOLD charges (`weight_rel_err`,
    /// `bias_err`, `scale_rel_err`, `shift_err`) from every IBP box, so the
    /// containment tests can prove they see an under-charged radius.
    static FOLD_CHARGE_DROPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// TEST-ONLY per-layer tap (see [`with_ibp_seed_tap_for_test`]).
    static TAP: std::cell::RefCell<Option<Vec<LayerTap>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn fold_charge_dropped() -> bool {
    FOLD_CHARGE_DROPPED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
const fn fold_charge_dropped() -> bool {
    false
}

/// TEST-ONLY: run `f` with every IBP box built WITHOUT its fold charges (the
/// code mutation the exact containment test must catch). Thread-local and
/// reset on exit (also on unwind); no production build can reach it.
#[cfg(test)]
pub(crate) fn with_ibp_fold_charge_dropped_for_test<Out>(f: impl FnOnce() -> Out) -> Out {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FOLD_CHARGE_DROPPED.with(|c| c.set(false));
        }
    }
    FOLD_CHARGE_DROPPED.with(|c| c.set(true));
    let _reset = Reset;
    f()
}

/// TEST-ONLY record of one trunk layer's seed step.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct LayerTap {
    /// Trunk layer index.
    pub layer: usize,
    /// The box handed IN (the tableau box, forward-only).
    pub fwd_l: Vec<f64>,
    /// Upper side of the box handed in.
    pub fwd_u: Vec<f64>,
    /// The layer's certified IBP box (empty when declined).
    pub ibp_l: Vec<f64>,
    /// Upper side of the IBP box.
    pub ibp_u: Vec<f64>,
    /// The box after the intersection (before `#backward-interm`).
    pub out_l: Vec<f64>,
    /// Upper side after the intersection.
    pub out_u: Vec<f64>,
    /// The printed counters.
    pub stats: LayerSeed,
    /// `declined=` state at this layer.
    pub declined: Option<Decline>,
}

/// TEST-ONLY: run `f` and return every seed step it took on THIS thread (the
/// root walk runs on its caller's thread; only its inner kernels fan out).
#[cfg(test)]
pub(crate) fn with_ibp_seed_tap_for_test<Out>(f: impl FnOnce() -> Out) -> (Out, Vec<LayerTap>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TAP.with(|t| t.borrow_mut().take());
        }
    }
    TAP.with(|t| *t.borrow_mut() = Some(Vec::new()));
    let reset = Reset;
    let out = f();
    let taps = TAP.with(|t| t.borrow_mut().take()).unwrap_or_default();
    drop(reset);
    (out, taps)
}

/// Certified IBP box of a conv output (see the module soundness notes).
/// `None` on any non-finite input interval or result (the caller fails closed).
pub(super) fn conv_box(c: &ConvOp, lo: &[f64], hi: &[f64], n_out: usize) -> Option<IBox> {
    let n_src = lo.len();
    if hi.len() != n_src {
        return None;
    }
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    if co * p != n_out || c.bias.len() != co || c.bias_err.len() != co || p == 0 {
        return None;
    }
    let mut mid = Array2::<f64>::zeros((n_src, 1));
    let mut mag = Array2::<f64>::zeros((n_src, 2));
    for i in 0..n_src {
        let (m, r) = mid_rad(lo[i], hi[i])?;
        mid[[i, 0]] = m;
        mag[[i, 0]] = r;
        mag[[i, 1]] = m.abs();
    }
    // Centre lane `W' mid` and the two magnitude lanes `|W'| rad`, `|W'| |mid|`
    // (no bias in either: the bias is folded below).
    let mut ym = Array2::<f64>::zeros((n_out, 1));
    conv_apply_forward(c, &mid, &mut ym, false);
    let mut ya = Array2::<f64>::zeros((n_out, 2));
    conv_apply_forward(c, &mag, &mut ya, true);
    let g = gamma_n(c.k_fwd + 2);
    let drop_fold = fold_charge_dropped();
    let rel = if drop_fold { 0.0 } else { c.weight_rel_err };
    let mut out_lo = Vec::with_capacity(n_out);
    let mut out_hi = Vec::with_capacity(n_out);
    for j in 0..n_out {
        let ch = j / p;
        let a = cert_sum(ya[[j, 0]], g);
        let b = cert_sum(ya[[j, 1]], g);
        let centre = ym[[j, 0]] + c.bias[ch];
        // |W'| rad, then the rounding of the centre dot product.
        let mut r = next_up(a + next_up(g * b));
        // The fold ball: rel |W'| (|mid| + rad), then the bias error.
        if rel > 0.0 {
            r = next_up(r + next_up(rel * next_up(a + b)));
        }
        if !drop_fold {
            r = next_up(r + c.bias_err[ch]);
        }
        // The bias add (`u |s| <= 2u |fl(s)|`) and gradual-underflow dust.
        r = next_up(r + next_up(2.0 * UNIT * centre.abs()));
        r = next_up(r + f64::MIN_POSITIVE);
        let (bl, bu) = close(centre, r);
        if !(bl.is_finite() && bu.is_finite()) {
            return None;
        }
        out_lo.push(bl);
        out_hi.push(bu);
    }
    Some((out_lo, out_hi))
}

/// Certified IBP box of a per-neuron affine `s v + t` (the compiled
/// `TwinOp::ChannelAffine`, per-NEURON vectors).
pub(super) fn affine_box(
    scale: &[f64],
    shift: &[f64],
    scale_rel_err: f64,
    shift_err: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> Option<IBox> {
    let n = scale.len();
    if shift.len() != n || shift_err.len() != n || lo.len() != n || hi.len() != n {
        return None;
    }
    let drop_fold = fold_charge_dropped();
    let srel = if drop_fold { 0.0 } else { scale_rel_err };
    let mut out_lo = Vec::with_capacity(n);
    let mut out_hi = Vec::with_capacity(n);
    for j in 0..n {
        let (m, r) = mid_rad(lo[j], hi[j])?;
        let s = scale[j];
        let sa = s.abs();
        let prod = s * m;
        let centre = prod + shift[j];
        // |s'| rad, then the rounding of `s' mid`.
        let mut rr = next_up(next_up(sa * r) + next_up(2.0 * UNIT * prod.abs()));
        // The fold ball: srel |s'| (|mid| + rad), then the shift error.
        if srel > 0.0 {
            rr = next_up(rr + next_up(srel * next_up(sa * next_up(m.abs() + r))));
        }
        if !drop_fold {
            rr = next_up(rr + shift_err[j]);
        }
        rr = next_up(rr + next_up(2.0 * UNIT * centre.abs()));
        rr = next_up(rr + f64::MIN_POSITIVE);
        let (bl, bu) = close(centre, rr);
        if !(bl.is_finite() && bu.is_finite()) {
            return None;
        }
        out_lo.push(bl);
        out_hi.push(bu);
    }
    Some((out_lo, out_hi))
}

/// Certified IBP box of an elementwise sum.
pub(super) fn add_box(a: &IBox, b: &IBox) -> Option<IBox> {
    let n = a.0.len();
    if a.1.len() != n || b.0.len() != n || b.1.len() != n {
        return None;
    }
    let mut out_lo = Vec::with_capacity(n);
    let mut out_hi = Vec::with_capacity(n);
    for j in 0..n {
        let (bl, bu) = (next_down(a.0[j] + b.0[j]), next_up(a.1[j] + b.1[j]));
        if !(bl.is_finite() && bu.is_finite()) {
            return None;
        }
        out_lo.push(bl);
        out_hi.push(bu);
    }
    Some((out_lo, out_hi))
}

/// ReLU of a certified pre-activation box (exact).
pub(super) fn relu_box(l: &[f64], u: &[f64]) -> IBox {
    (
        l.iter().map(|&v| v.max(0.0)).collect(),
        u.iter().map(|&v| v.max(0.0)).collect(),
    )
}

/// Pure IBP (no tableau) through the trunk: one certified pre-activation box
/// per trunk ReLU, in execution order. A DIAGNOSTIC and test oracle (the
/// real-ONNX containment gates in ny-cli); it never feeds a bound. Supports
/// residual `Add`. `None` on a malformed input box or a non-finite result.
#[must_use]
pub fn pure_ibp_trunk_boxes(net: &TwinNet, lo: &[f64], hi: &[f64]) -> Option<Vec<IBox>> {
    if lo.len() != net.n_in || hi.len() != net.n_in {
        return None;
    }
    let last_relu = *net.trunk_relus.last()?;
    let mut boxes: Vec<Option<IBox>> = vec![None; net.ops.len() + 1];
    boxes[0] = Some((lo.to_vec(), hi.to_vec()));
    let mut out = Vec::with_capacity(net.trunk_relus.len());
    for (k, op) in net.ops.iter().enumerate().take(last_relu + 1) {
        let next = match op {
            TwinOp::Conv(c) => {
                let (l, u) = boxes[c.input].as_ref()?;
                conv_box(c, l, u, net.tsize[k + 1])?
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                let (l, u) = boxes[*input].as_ref()?;
                affine_box(scale, shift, *scale_rel_err, shift_err, l, u)?
            }
            TwinOp::Flatten { input } => boxes[*input].clone()?,
            TwinOp::Add { lhs, rhs } => add_box(boxes[*lhs].as_ref()?, boxes[*rhs].as_ref()?)?,
            TwinOp::Relu { input, .. } => {
                let (l, u) = boxes[*input].as_ref()?;
                out.push((l.clone(), u.clone()));
                relu_box(l, u)
            }
            TwinOp::Gemm { .. } => return None,
        };
        boxes[k + 1] = Some(next);
    }
    Some(out)
}

/// The per-build seed state carried through the root walk. `None` (from
/// [`IbpSeed::new`]) is the dark state: the walk is then byte-identical.
pub(crate) struct IbpSeed {
    /// Certified box per tensor id; dropped after its last consumer.
    boxes: Vec<Option<IBox>>,
    /// Remaining in-walk consumers per tensor id.
    left: Vec<usize>,
    /// `Some` = armed but intersecting nothing on this build.
    declined: Option<Decline>,
    /// Per trunk layer counters, in execution order.
    layers: Vec<LayerSeed>,
    /// Admission estimate, bytes.
    need_bytes: u64,
    t0: Instant,
    /// Set by [`Self::finish`]. An armed state dropped without it (the walk
    /// returned early: deadline, error) prints its terminal line from `Drop`
    /// with `complete=0`, so the observable exists on every exit.
    finished: bool,
}

/// Admitted bytes for the interval state: half the observed memory headroom
/// (cgroup- or `MemAvailable`-derived), or [`UNOBSERVED_HEADROOM_CAP`].
fn admitted_bytes() -> u64 {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => headroom / 2,
        None => UNOBSERVED_HEADROOM_CAP,
    }
}

impl IbpSeed {
    /// Arm the seed for one root walk over ops `..=last_relu`. `armed ==
    /// false` returns `None` (dark, byte-identical, silent). `consumers` is the
    /// walk's own per-tensor consumer count (copied; the walk mutates its own).
    /// `lo`/`hi` is the declared input box, already validated by the walk.
    pub(crate) fn new(
        net: &TwinNet,
        last_relu: usize,
        mode: RoundMode,
        consumers: &[usize],
        lo: &[f64],
        hi: &[f64],
        armed: bool,
    ) -> Option<Self> {
        if !armed {
            return None;
        }
        // Worst case every trunk tensor alive at once (two f64 per neuron),
        // plus one conv's centre/magnitude scratch.
        let sizes = &net.tsize[..=(last_relu + 1).min(net.tsize.len() - 1)];
        let total: u64 = sizes.iter().map(|&s| s as u64).sum();
        let widest: u64 = sizes.iter().map(|&s| s as u64).max().unwrap_or(0);
        let need_bytes = total
            .saturating_mul(16)
            .saturating_add(widest.saturating_mul(48));
        let mut seed = Self {
            boxes: vec![None; net.ops.len() + 1],
            left: consumers.to_vec(),
            declined: None,
            layers: Vec::new(),
            need_bytes,
            t0: Instant::now(),
            finished: false,
        };
        if !mode.outward() {
            seed.declined = Some(Decline::Parity);
        } else if net.ops[..=last_relu]
            .iter()
            .any(|op| matches!(op, TwinOp::Add { .. }))
        {
            seed.declined = Some(Decline::Residual);
        } else if need_bytes > admitted_bytes() {
            seed.declined = Some(Decline::Rss);
        } else if lo.len() != net.n_in || hi.len() != net.n_in {
            seed.declined = Some(Decline::Op);
        } else {
            seed.boxes[0] = Some((lo.to_vec(), hi.to_vec()));
        }
        Some(seed)
    }

    /// Armed AND intersecting (not declined).
    pub(crate) fn active(&self) -> bool {
        self.declined.is_none()
    }

    /// Decline for the rest of this build (fail closed: every later layer
    /// keeps its tableau box) and free the state.
    fn decline(&mut self, why: Decline) {
        if self.declined.is_none() {
            self.declined = Some(why);
        }
        self.boxes.fill(None);
    }

    /// Consume one in-walk use of tensor `id` (drops it after its last one).
    fn release(&mut self, id: usize) {
        if let Some(n) = self.left.get_mut(id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                if let Some(b) = self.boxes.get_mut(id) {
                    *b = None;
                }
            }
        }
    }

    /// Advance the interval state over op `k` (tensor `k + 1`). ReLU is a
    /// no-op here: its output is set by [`Self::post_relu`] from the FINAL
    /// published box.
    pub(crate) fn forward(&mut self, net: &TwinNet, k: usize, op: &TwinOp) {
        if !self.active() {
            return;
        }
        let next = match op {
            TwinOp::Relu { .. } => return,
            TwinOp::Conv(c) => {
                let out = self.boxes[c.input]
                    .as_ref()
                    .and_then(|(l, u)| conv_box(c, l, u, net.tsize[k + 1]));
                self.release(c.input);
                out
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
            } => {
                let out = self.boxes[*input]
                    .as_ref()
                    .and_then(|(l, u)| affine_box(scale, shift, *scale_rel_err, shift_err, l, u));
                self.release(*input);
                out
            }
            TwinOp::Flatten { input } => {
                let out = self.boxes[*input].clone();
                self.release(*input);
                out
            }
            // Declined at `new` (residual) / rejected by the walk (Gemm);
            // unreachable here, and fail closed if it ever is not.
            TwinOp::Add { .. } | TwinOp::Gemm { .. } => None,
        };
        match next {
            Some(b) => {
                if let Some(slot) = self.boxes.get_mut(k + 1) {
                    *slot = Some(b);
                } else {
                    self.decline(Decline::Op);
                }
            }
            None => self.decline(Decline::Op),
        }
    }

    /// Intersect trunk layer `li`'s tableau box `(l, u)` (the pre-activation
    /// tensor `input`) with its IBP box, shrink-only, and print the layer's
    /// fixed-schema line. Declined: prints the line and touches nothing.
    pub(crate) fn seed_layer(&mut self, li: usize, input: usize, l: &mut [f64], u: &mut [f64]) {
        #[cfg(test)]
        let (fwd_l, fwd_u) = (l.to_vec(), u.to_vec());
        let mut st = LayerSeed {
            neurons: l.len(),
            ..LayerSeed::default()
        };
        #[cfg(test)]
        let mut ibp_tap: (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut shape_ok = true;
        if self.active() {
            match self.boxes[input].as_ref() {
                Some((bl, bu)) => {
                    #[cfg(test)]
                    {
                        ibp_tap = (bl.clone(), bu.clone());
                    }
                    match intersect_layer_or_refuse(l, u, bl, bu) {
                        Some(s) => st = s,
                        None => shape_ok = false,
                    }
                }
                None => shape_ok = false,
            }
        }
        if !shape_ok {
            self.decline(Decline::Op);
        }
        if !self.active() || !shape_ok {
            // Nothing was intersected: report the layer's own unstable count.
            let unst = l
                .iter()
                .zip(u.iter())
                .filter(|(&a, &b)| unstable(a, b))
                .count();
            st = LayerSeed {
                neurons: l.len(),
                unst_before: unst,
                unst_after: unst,
                ..LayerSeed::default()
            };
        }
        self.release(input);
        let declined = self.declined.map_or("-", Decline::token);
        eprintln!(
            "[mr-ibp-seed] layer={li} neurons={} tightened={} stabilized={} crossed={} \
             declined={declined} unst_before={} unst_after={} nonfinite={}",
            st.neurons,
            st.tightened,
            st.stabilized,
            st.crossed,
            st.unst_before,
            st.unst_after,
            st.nonfinite
        );
        if st.crossed > 0 {
            eprintln!(
                "[mr-ibp-seed] CROSSED layer={li} count={} — two certified enclosures \
                 disagree: DEFECT SIGNAL (or an empty split subdomain); seed REFUSED for the \
                 whole layer, tableau box kept",
                st.crossed
            );
        }
        self.layers.push(st);
        #[cfg(test)]
        TAP.with(|t| {
            if let Some(v) = t.borrow_mut().as_mut() {
                v.push(LayerTap {
                    layer: li,
                    fwd_l,
                    fwd_u,
                    ibp_l: ibp_tap.0,
                    ibp_u: ibp_tap.1,
                    out_l: l.to_vec(),
                    out_u: u.to_vec(),
                    stats: st,
                    declined: self.declined,
                });
            }
        });
    }

    /// Set the ReLU output box of tensor `k + 1` from layer `k`'s FINAL
    /// published pre-activation box (after `#backward-interm`). Skipped when
    /// the output has no in-walk consumer (the last trunk ReLU).
    pub(crate) fn post_relu(&mut self, k: usize, l: &[f64], u: &[f64]) {
        if !self.active() {
            return;
        }
        if self.left.get(k + 1).copied().unwrap_or(0) == 0 {
            return;
        }
        let b = relu_box(l, u);
        if let Some(slot) = self.boxes.get_mut(k + 1) {
            *slot = Some(b);
        } else {
            self.decline(Decline::Op);
        }
    }

    /// The terminal fixed-schema line of an armed build that completed its
    /// walk (`complete=1`). An early exit prints it from `Drop` instead.
    pub(crate) fn finish(&mut self) {
        self.print_done(true);
        self.finished = true;
    }

    /// The terminal line. Written with `writeln!` (errors ignored) rather than
    /// `eprintln!` because it also runs from `Drop`, possibly while unwinding,
    /// where a panicking stderr write would abort the process.
    fn print_done(&self, complete: bool) {
        use std::io::Write as _;
        let mut sum = LayerSeed::default();
        for s in &self.layers {
            sum.tightened += s.tightened;
            sum.stabilized += s.stabilized;
            sum.crossed += s.crossed;
            sum.nonfinite += s.nonfinite;
            sum.unst_before += s.unst_before;
            sum.unst_after += s.unst_after;
        }
        #[allow(clippy::cast_precision_loss)]
        let need_mb = self.need_bytes as f64 / f64::from(1u32 << 20);
        let _ = writeln!(
            std::io::stderr(),
            "[mr-ibp-seed] done layers={} tightened={} stabilized={} crossed={} nonfinite={} \
             unst_before={} unst_after={} declined={} rss_need_mb={need_mb:.1} secs={:.3} \
             complete={}",
            self.layers.len(),
            sum.tightened,
            sum.stabilized,
            sum.crossed,
            sum.nonfinite,
            sum.unst_before,
            sum.unst_after,
            self.declined.map_or("-", Decline::token),
            self.t0.elapsed().as_secs_f64(),
            u8::from(complete)
        );
    }
}

impl Drop for IbpSeed {
    /// An armed walk that returned early (deadline, error, unwind) still
    /// prints its terminal line, with `complete=0` — "observable only on
    /// success" would make a truncated armed build indistinguishable from a
    /// dark one.
    fn drop(&mut self) {
        if !self.finished {
            self.print_done(false);
        }
    }
}
