// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-errlane — the ENVELOPE ERROR LANE (EEL), margin-row throughput plan B3.
//!
//! # What it replaces
//!
//! In `Outward` mode every backward pass carries, beside the coefficient
//! matrix `L` (`n x R`), a certified elementwise error matrix `E` (`n x R`)
//! with the invariant that the exact-real backward recursion applied to the
//! stored upstream values stays within `L ± E` (see the `engine.rs` module
//! docs). Maintaining `E` is a second full `conv_apply_backward` on `|W|` per
//! conv, plus `combined_err` and a `certify_up` sweep: measured by the B1
//! probe (`col-scaling-v2`, all three nets) as exactly HALF of every pass at
//! every width (`outward_over_parity` = 2.00-2.04 at 1 thread).
//!
//! # The mechanism (plan F3)
//!
//! Every relative part of a per-op update is MONOTONE NON-DECREASING in
//! (`E`, `|L|`) with coefficients that do not depend on the column:
//!
//! | op | per-column update | coefficient |
//! |---|---|---|
//! | ReLU gate | `slack16((e + 4u|v|)·ms_j + 4u|lv|)` | `ms_j >= 0` |
//! | ReLU exception | `slack16((e + 4u|v|)·max(|a2|,|s2|) + 4u|lnew|)` | `>= 0` |
//! | beta charge | `slack16(e + 2u|lv|)` | `1` |
//! | ChannelAffine | `slack16((e + rel|v| + 4u|v|)·|s_j| + (rel+4u)|lnew|)` | `|s_j|` |
//! | Conv | `certify_up(|W|ᵀ(e + g(|L| + e)), g2)` | `|W| >= 0` |
//! | merge | `slack16(e_a + e_b + 2u|l_a + l_b|)` | `1` |
//!
//! Each positive error product is additionally rounded upward when it is
//! subnormal, including a nonzero exact product rounded to zero. This repaired
//! product is monotone, including the transition to the smallest normal value.
//! A nominal product can lose at most `eta/2`, where `eta = 2^-1074`; the
//! per-column lane adds `eta` when a nonzero nominal product is subnormal.
//! That conditional indicator is NOT monotone in the nominal magnitude: a
//! larger group maximum can produce a normal product while a smaller column
//! underflows. The envelope therefore adds the monotone majorant `eta` whenever
//! both the group maximum and the multiplier are nonzero. This additional
//! local addition is rounded upward before the existing accumulation.
//!
//! ONE envelope `Ê[g] >= E[j, c]` for every row `j` in group `g` and EVERY
//! column `c` uses these repaired formulas with `|L|` replaced by its group
//! maximum `Λ[g] = max_{j∈g, c} |L[j, c]|`. Each term then dominates its
//! per-column counterpart, so the envelope dominates the per-column lane
//! BIT-FOR-BIT, not merely in exact arithmetic — `eel_dominates_percol_
//! every_op` asserts `>=` with no tolerance after every op. The bias lane
//! `eb` stays per column and reads `Ê` wherever it read `E[j, c]`; the
//! coefficient lane `L` and bias `b` are NOT TOUCHED, so a pass's `a` and `b`
//! are bit-identical with the lane on or off and only the certified penalty
//! can grow.
//!
//! # Soundness
//!
//! The per-column lane's argument is local: given `|L_true - L| <= E_in`, the
//! op's certified evaluation yields an `E_out` bounding the new deviation,
//! because the corrected formula upper-bounds an exact-real bound of
//! (`E_in`, `|L|`). The envelope feeds a MONOTONE MAJORANT of that formula with
//! elementwise larger inputs (`Ê >= E`, `Λ >= |L|`), so its output bounds the
//! deviation of every column simultaneously. Nothing downstream changes: the
//! final envelope is BROADCAST back to an `n_in x R` matrix, which every
//! consumer (`concretize_*`, the GPU shadows, the parent lift) reads exactly as
//! before — a looser `E` can cost a proof and cannot manufacture one.
//!
//! # Two groupings
//!
//! * `envelope` (per ROW, group size 1): one value per neuron. The conv step
//!   uses the CPU error contraction at `R = 1`, with the same term and fold
//!   order as the repaired per-column CPU lane and the grouped nominal dust
//!   majorant. The per-output fold order is independent of `R` and column
//!   blocking, so dominance is bit-exact. Engaged
//!   only on passes with `R >= ENV_MIN_COLS` (score, y-refresh,
//!   backward-interm, tinyimagenet's wide evals): at narrow `R` the per-column
//!   lane is already an `R`-wide `|W|` conv and a 1-wide one saves little
//!   (plan F1/F2).
//! * `per_channel` (group = one channel's spatial positions): one value per
//!   CHANNEL. The conv step becomes `A[ci][tap] = Σ_co |w| comb[co]` followed by
//!   `max_isp Σ_{taps(isp)} A[ci][tap]` — `O(ci·co·taps + ci·ip·taps)` instead
//!   of an `R = 1` conv over every position, i.e. nearly free even at `R = 1`,
//!   so this grouping engages on EVERY pass. SOUND for the same reason plus
//!   one line: summing every tap listed in `back_taps[isp]` (multiplicity
//!   included) against the channel maximum dominates that position's own sum,
//!   and the max over positions dominates each position. The nested sum has a
//!   different association than the per-column fold, so its certificate uses
//!   `gamma_n(2·k_bwd + max_taps + 16)` instead of `gamma_n(k_bwd + 8)`: the
//!   extra `k_bwd + max_taps + 8` ulps are exactly what makes the per-channel
//!   value dominate the per-column CERTIFIED value bit-for-bit even where the
//!   envelope is tight (first-order: `k_bwd + max_taps + 10 >= co`, true since
//!   `k_bwd = max_taps · co`). Its LOOSENESS is unmeasured — that is what the
//!   `audit` arm is for.
//!
//! # The `audit` arm
//!
//! Publishes the per-column lane (every published bound is the `off` bound;
//! only wall clock, and so any time-driven policy, differs) and SHADOWS both
//! envelopes on the same pass — the row envelope wherever `envelope` would
//! publish it (`R >= ENV_MIN_COLS`), the per-channel envelope on every pass —
//! then records, per column: the penalty difference, the concretized-bound
//! loosening, positive lower bounds the envelope would have lost, and any
//! elementwise dominance violation. It costs about two extra passes per pass;
//! it is a MEASUREMENT arm, never a treatment.
//!
//! # Observables (rule R9)
//!
//! * `[mr-errlane] armed mode=<m> min_cols=16` / `[mr-errlane] dark mode=off`
//!   once per lane, from the frozen policy;
//! * `[mr-errlane] done mode= passes= eligible= env_passes= pc_passes=
//!   audited= max_pen_env= max_pen_col= ... aud_row_* aud_pc_*` on every lane
//!   stop (printed with the `[mr-census]` stop receipts, so a snapshot line
//!   exists for a lane that never returns);
//! * `errlane_mode= errlane_env_passes= errlane_pc_passes=` on the
//!   `[mr-census]` line itself.
//!
//! An armed run whose `env_passes` (or `pc_passes`) is zero is VOID, not a
//! negative: the key did not arrive at the engine.

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

use super::engine::PassOut;
use super::net::{
    envelope_error_conv_without_deadline, underflow_product_up, ConvOp, TwinNet, TwinOp,
};
use super::rounding::{certify_up, gamma_n, next_up};

/// Pass-width floor for the ROW envelope (plan §3 B3 correction: narrow passes
/// keep the per-column lane).
pub const ENV_MIN_COLS: usize = 16;

/// Which certified coefficient-error lane a [`super::engine::BackwardEngine`]
/// carries. Chosen ONCE per lane from the frozen `margin_row.err_envelope`
/// policy (`RootPrecisionRuntimePolicy::err_lane`) and copied into every
/// engine at construction; never read from a global.
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ErrLane {
    /// Per-column lane (`off`, the default), including underflow correction.
    #[default]
    #[serde(rename = "off")]
    PerCol,
    /// Row envelope on passes with `R >= ENV_MIN_COLS` (`envelope`).
    #[serde(rename = "envelope")]
    Envelope,
    /// Per-channel envelope on every pass (`per_channel`).
    #[serde(rename = "per_channel")]
    PerChannel,
    /// Publish per-column, shadow both envelopes, record looseness (`audit`).
    #[serde(rename = "audit")]
    Audit,
}

impl ErrLane {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 4] = ["off", "envelope", "per_channel", "audit"];

    /// Parse one canonical token (exact spelling; the lever declaration is an
    /// `Enum` over [`Self::TOKENS`], so a resolved value always parses).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "off" => Some(Self::PerCol),
            "envelope" => Some(Self::Envelope),
            "per_channel" => Some(Self::PerChannel),
            "audit" => Some(Self::Audit),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::PerCol => "off",
            Self::Envelope => "envelope",
            Self::PerChannel => "per_channel",
            Self::Audit => "audit",
        }
    }

    /// What one Outward pass with `r` seed columns PUBLISHES.
    pub(crate) fn publish(self, r: usize, min_cols: usize) -> PassLane {
        match self {
            Self::PerCol | Self::Audit => PassLane::PerCol,
            Self::Envelope if r >= min_cols.max(1) => PassLane::Envelope { per_channel: false },
            Self::Envelope => PassLane::PerCol,
            Self::PerChannel if r >= 1 => PassLane::Envelope { per_channel: true },
            Self::PerChannel => PassLane::PerCol,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::PerCol => 1,
            Self::Envelope => 2,
            Self::PerChannel => 3,
            Self::Audit => 4,
        }
    }
}

/// The lane one backward pass actually runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassLane {
    /// Today's `n x R` error matrix.
    PerCol,
    /// One envelope value per group (`per_channel = false`: per row).
    Envelope {
        /// Group rows by channel instead of by neuron.
        per_channel: bool,
    },
}

/// NaN-propagating max (a NaN anywhere reaches the pass's non-finite
/// firewall instead of being silently dropped, which `f64::max` would do).
#[inline]
pub(crate) fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// `grp` if it partitions `n` rows exactly, else `1` (per row: always sound).
#[inline]
fn checked_grp(n: usize, grp: usize) -> usize {
    if grp >= 1 && n.is_multiple_of(grp) {
        grp
    } else {
        1
    }
}

/// An envelope error: `v[g] >= E[j, c]` for every row `j` in group
/// `g = j / grp` and every column `c`.
#[derive(Clone, Debug)]
pub(crate) struct EnvErr {
    pub(crate) v: Vec<f64>,
    pub(crate) grp: usize,
}

impl EnvErr {
    /// All-zero envelope over `n` rows (an exact seed).
    pub(crate) fn zeros(n: usize, grp: usize) -> Self {
        let grp = checked_grp(n, grp);
        Self {
            v: vec![0.0; n / grp],
            grp,
        }
    }

    /// Envelope of a per-column error matrix (group max over rows AND
    /// columns; the entries are non-negative, so `|.|` is the identity).
    pub(crate) fn from_matrix(e: &Array2<f64>, grp: usize) -> Self {
        let grp = checked_grp(e.nrows(), grp);
        Self {
            v: group_absmax(e, grp),
            grp,
        }
    }

    /// Envelope value for row `j`.
    #[inline]
    pub(crate) fn at(&self, j: usize) -> f64 {
        self.v[j / self.grp]
    }

    /// Row count covered.
    pub(crate) fn rows(&self) -> usize {
        self.v.len() * self.grp
    }

    /// Re-express over groups of `grp_to` rows. Coarsening takes the max of the
    /// merged groups; refining copies the coarse value down. Both dominate.
    pub(crate) fn regroup(self, grp_to: usize) -> Self {
        let n = self.rows();
        let grp_to = checked_grp(n, grp_to);
        if grp_to == self.grp {
            return self;
        }
        if grp_to.is_multiple_of(self.grp) {
            let k = grp_to / self.grp;
            let v = self
                .v
                .chunks(k)
                .map(|c| c.iter().fold(0.0f64, |a, &x| nan_max(a, x)))
                .collect();
            return Self { v, grp: grp_to };
        }
        let v = (0..n / grp_to)
            .map(|h| (h * grp_to..(h + 1) * grp_to).fold(0.0f64, |a, j| nan_max(a, self.at(j))))
            .collect();
        Self { v, grp: grp_to }
    }

    /// Materialize as the `n x r` matrix every consumer of `PassOut::e` reads.
    pub(crate) fn broadcast(&self, r: usize) -> Array2<f64> {
        let n = self.rows();
        let mut out = Array2::<f64>::zeros((n, r));
        if r == 0 || n == 0 {
            return out;
        }
        let dst = out.as_slice_mut().expect("standard layout");
        let min_len = (1usize << 12).div_ceil(r).max(1);
        dst.par_chunks_mut(r)
            .with_min_len(min_len)
            .enumerate()
            .for_each(|(j, row)| row.fill(self.at(j)));
        out
    }
}

/// `Λ[g] = max_{j ∈ g, c} |m[j, c]|` (NaN-propagating).
pub(crate) fn group_absmax(m: &Array2<f64>, grp: usize) -> Vec<f64> {
    let n = m.nrows();
    let r = m.ncols();
    let grp = checked_grp(n, grp);
    if n == 0 {
        return Vec::new();
    }
    if r == 0 {
        return vec![0.0; n / grp];
    }
    let s = m.as_slice().expect("standard layout");
    let chunk = grp * r;
    let min_len = (1usize << 12).div_ceil(chunk).max(1);
    s.par_chunks(chunk)
        .with_min_len(min_len)
        .map(|c| c.iter().fold(0.0f64, |a, &x| nan_max(a, x.abs())))
        .collect()
}

/// Per-group max of `|x|` over one per-row vector (ChannelAffine scale).
pub(crate) fn group_vec_absmax(x: &[f64], grp: usize) -> Vec<f64> {
    let grp = checked_grp(x.len(), grp);
    x.chunks(grp)
        .map(|c| c.iter().fold(0.0f64, |a, &v| nan_max(a, v.abs())))
        .collect()
}

/// Per-group ReLU gate contraction bound `M[g] >= max(|alpha_j|, |s_j|,
/// |ms_j|)` over `j ∈ g`, folded into `acc` (so the domain-stacked arm can max
/// over every block's gates).
///
/// Taking `|alpha|` and `|s|` as well as `ms` makes `M` bound BOTH factors the
/// per-column update uses: the contraction `ms_j`, and (through
/// `|lv| = fl(|v| · slope) <= fl(Λv · M)`) the fresh-rounding magnitude, so the
/// envelope needs no second sweep over the transformed coefficients.
pub(crate) fn fold_gate_max(acc: &mut [f64], alpha: &[f64], s: &[f64], ms: &[f64], grp: usize) {
    let n = alpha.len();
    let grp = checked_grp(n, grp);
    for (g, slot) in acc.iter_mut().enumerate().take(n / grp) {
        let mut m = *slot;
        for j in g * grp..(g + 1) * grp {
            m = nan_max(m, alpha[j].abs());
            m = nan_max(m, s[j].abs());
            m = nan_max(m, ms[j].abs());
        }
        *slot = m;
    }
}

/// Envelope ReLU gate update (bit-dominates the per-column
/// `slack16((e + 4u|v|)·ms_j + 4u|lv|)` for every `j ∈ g` and column).
pub(crate) fn env_gate_update(env: &mut EnvErr, lam_v: &[f64], m: &[f64]) {
    use super::rounding::{slack16, UNIT};
    for ((ev, &lv), &mg) in env.v.iter_mut().zip(lam_v).zip(m) {
        let lam_out = lv * mg;
        let charge = underflow_product_up(*ev + underflow_product_up(4.0 * UNIT, lv), mg)
            + underflow_product_up(4.0 * UNIT, lam_out);
        *ev = slack16(env_nominal_dust_up(charge, lv, mg));
    }
}

/// Monotone upper budget for every grouped column's nominal product loss.
/// Testing whether `lambda * magnitude` itself underflows is insufficient:
/// a smaller column can underflow even when the group maximum does not.
/// One eta doubles the maximal RN product loss eta/2, covering its internal
/// dot amplification under the engine's preflight gamma <= 0.25 premise.
#[inline]
pub(crate) fn env_nominal_dust_up(charge: f64, lambda: f64, magnitude: f64) -> f64 {
    if lambda != 0.0 && magnitude != 0.0 {
        next_up(charge + f64::from_bits(1))
    } else {
        charge
    }
}

/// Per-tensor group sizes for the per-channel envelope: a conv's input and
/// output tensors group by their spatial size `H·W`; ReLU / ChannelAffine /
/// Add inherit through (shape-preserving); everything else, and any size that
/// does not partition its tensor, groups per row. Any grouping is sound (every
/// op reads the group size from the envelope it holds and merges regroup), so
/// a conflict just costs tightness.
pub(crate) fn channel_groups(net: &TwinNet) -> Vec<usize> {
    let nt = net.tsize.len();
    let mut g: Vec<Option<usize>> = vec![None; nt];
    for (k, op) in net.ops.iter().enumerate() {
        if let TwinOp::Conv(cv) = op {
            if cv.input < nt && g[cv.input].is_none() {
                g[cv.input] = Some(cv.ishape.1 * cv.ishape.2);
            }
            if k + 1 < nt && g[k + 1].is_none() {
                g[k + 1] = Some(cv.oshape.1 * cv.oshape.2);
            }
        }
    }
    for _ in 0..3 {
        for (k, op) in net.ops.iter().enumerate() {
            let out = k + 1;
            if out >= nt {
                continue;
            }
            let ins: [Option<usize>; 2] = match op {
                TwinOp::Relu { input, .. } | TwinOp::ChannelAffine { input, .. } => {
                    [Some(*input), None]
                }
                TwinOp::Add { lhs, rhs } => [Some(*lhs), Some(*rhs)],
                _ => [None, None],
            };
            for inp in ins.into_iter().flatten() {
                if inp >= nt || net.tsize[inp] != net.tsize[out] {
                    continue;
                }
                match (g[inp], g[out]) {
                    (Some(a), None) => g[out] = Some(a),
                    (None, Some(b)) => g[inp] = Some(b),
                    _ => {}
                }
            }
        }
    }
    g.into_iter()
        .enumerate()
        .map(|(t, v)| checked_grp(net.tsize[t], v.unwrap_or(1)))
        .collect()
}

/// Envelope conv backward: the error lane of `TwinOp::Conv` for one envelope.
///
/// `l` is the conv OUTPUT coefficient matrix, `env` its envelope, `g` the
/// per-column arm's `combined_err` factor (accumulation gamma + folded-weight
/// relative error), `grp_in` the input tensor's target grouping.
///
/// Both groupings use the corrected CPU error products. The device error
/// product lacks these absolute dust terms and is ineligible for this lane.
pub(crate) fn env_conv_backward(
    cv: &ConvOp,
    l: &Array2<f64>,
    env: &EnvErr,
    g: f64,
    grp_in: usize,
) -> Result<EnvErr> {
    if !(g.is_finite() && g >= 0.0) || env.v.iter().any(|&e| !(e.is_finite() && e >= 0.0)) {
        return Err(NyError::InvalidSpec(
            "margin_row: non-finite or negative convolution error envelope".into(),
        ));
    }
    let co = cv.kernel.0;
    let op_ = cv.oshape.1 * cv.oshape.2;
    let ip = cv.ishape.1 * cv.ishape.2;
    if env.grp == op_ && env.v.len() == co && grp_in == ip && (op_ > 1 || ip > 1) {
        let lam = group_absmax(l, op_);
        if lam.iter().any(|&x| !x.is_finite()) {
            return Err(NyError::InvalidSpec(
                "margin_row: non-finite convolution coefficient envelope".into(),
            ));
        }
        return Ok(env_conv_per_channel(cv, &lam, env, g));
    }
    let row = env.clone().regroup(1);
    env_conv_per_row(cv, l, &row, g)
}

/// Row envelope: the repaired `R = 1` `|W|` backward contraction on
/// `comb = Ê + UP_subnormal(g(Λ + Ê))`, using a monotone nominal dust
/// majorant and the same post-contraction certifier as the per-column arm.
fn env_conv_per_row(cv: &ConvOp, l: &Array2<f64>, env: &EnvErr, g: f64) -> Result<EnvErr> {
    debug_assert_eq!(env.grp, 1);
    let n_out = l.nrows();
    let lam = group_absmax(l, 1);
    let mut comb = Array2::<f64>::zeros((n_out, 1));
    {
        let cs = comb.as_slice_mut().expect("standard layout");
        for (j, cv_) in cs.iter_mut().enumerate() {
            let ev = env.v[j];
            // Same expression shape and repaired product as combined_err.
            *cv_ = ev + underflow_product_up(g, lam[j] + ev);
        }
    }
    let n_in = cv.ishape.0 * cv.ishape.1 * cv.ishape.2;
    let mut en = Array2::<f64>::zeros((n_in, 1));
    let lambda = Array2::from_shape_vec((n_out, 1), lam).expect("known row envelope shape");
    // Refusal propagates without publishing any partial error rows.
    envelope_error_conv_without_deadline(cv, &comb, &lambda, &mut en)?;
    let g2 = gamma_n(cv.k_bwd + 8);
    Ok(EnvErr {
        v: en.iter().map(|&x| certify_up(x, g2)).collect(),
        grp: 1,
    })
}

/// Per-channel envelope: `O(ci·co·taps + ci·ip·taps)`. See the module docs for
/// the dominance argument and the widened certificate.
fn env_conv_per_channel(cv: &ConvOp, lam: &[f64], env: &EnvErr, g: f64) -> EnvErr {
    let (co, ci, kh, kw) = cv.kernel;
    let taps = kh * kw;
    let comb: Vec<f64> = (0..co)
        .map(|oc| {
            let ev = env.v[oc];
            ev + underflow_product_up(g, lam[oc] + ev)
        })
        .collect();
    // wt is [ci][kh][kw][co]: entry (ic, ky, kx, oc) sits at
    // ((ic*kh + ky)*kw + kx)*co + oc = (ic*taps + ky*kw + kx)*co + oc.
    let a: Vec<f64> = (0..ci * taps)
        .into_par_iter()
        .map(|it| {
            let base = it * co;
            let mut acc = 0.0f64;
            for (oc, &c) in comb.iter().enumerate() {
                let w = cv.wt[base + oc].abs();
                if w != 0.0 {
                    let term = underflow_product_up(w, c);
                    acc += env_nominal_dust_up(term, lam[oc], w);
                }
            }
            acc
        })
        .collect();
    let max_taps = cv.back_taps.iter().map(Vec::len).max().unwrap_or(0);
    let g2 = gamma_n(2 * cv.k_bwd + max_taps + 16);
    let v: Vec<f64> = (0..ci)
        .into_par_iter()
        .map(|ic| {
            let arow = &a[ic * taps..(ic + 1) * taps];
            let mut mx = 0.0f64;
            for tl in &cv.back_taps {
                let mut s = 0.0f64;
                for &(ky, kx, _osp) in tl {
                    s += arow[ky * kw + kx];
                }
                mx = nan_max(mx, s);
            }
            certify_up(mx, g2)
        })
        .collect();
    EnvErr {
        v,
        grp: cv.ishape.1 * cv.ishape.2,
    }
}

// ---------------------------------------------------------------------------
//  Observables
// ---------------------------------------------------------------------------

const F64_INF_BITS: u64 = 0x7ff0_0000_0000_0000;

/// Lane mode announced by the current lane (0 = none announced).
static MODE: AtomicU8 = AtomicU8::new(0);
static PASSES: AtomicU64 = AtomicU64::new(0);
static ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static ENV_PASSES: AtomicU64 = AtomicU64::new(0);
static PC_PASSES: AtomicU64 = AtomicU64::new(0);
static AUDITED: AtomicU64 = AtomicU64::new(0);
/// `max_c (Σ_i Ê_i·xabs_i + eb_c)` over PUBLISHED row-envelope passes.
static MAX_PEN_ENV: AtomicU64 = AtomicU64::new(0);
static MAX_PEN_PC: AtomicU64 = AtomicU64::new(0);

/// Per-variant audit accumulators (see [`record_audit`]).
struct AuditAcc {
    passes: AtomicU64,
    cols: AtomicU64,
    max_pen_env: AtomicU64,
    max_pen_col: AtomicU64,
    max_dpen: AtomicU64,
    max_loosen: AtomicU64,
    max_rel: AtomicU64,
    min_pos: AtomicU64,
    pos_lost: AtomicU64,
    dom_viol: AtomicU64,
    lane_mismatch: AtomicU64,
    errors: AtomicU64,
}

impl AuditAcc {
    const fn new() -> Self {
        Self {
            passes: AtomicU64::new(0),
            cols: AtomicU64::new(0),
            max_pen_env: AtomicU64::new(0),
            max_pen_col: AtomicU64::new(0),
            max_dpen: AtomicU64::new(0),
            max_loosen: AtomicU64::new(0),
            max_rel: AtomicU64::new(0),
            min_pos: AtomicU64::new(F64_INF_BITS),
            pos_lost: AtomicU64::new(0),
            dom_viol: AtomicU64::new(0),
            lane_mismatch: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        for a in [
            &self.passes,
            &self.cols,
            &self.max_pen_env,
            &self.max_pen_col,
            &self.max_dpen,
            &self.max_loosen,
            &self.max_rel,
            &self.pos_lost,
            &self.dom_viol,
            &self.lane_mismatch,
            &self.errors,
        ] {
            a.store(0, Relaxed);
        }
        self.min_pos.store(F64_INF_BITS, Relaxed);
    }

    fn fields(&self, prefix: &str) -> String {
        let n = self.passes.load(Relaxed);
        let f = |a: &AtomicU64| {
            if n == 0 {
                "na".to_string()
            } else {
                format!("{:.3e}", f64::from_bits(a.load(Relaxed)))
            }
        };
        let min_pos = self.min_pos.load(Relaxed);
        let min_pos = if n == 0 || min_pos == F64_INF_BITS {
            "na".to_string()
        } else {
            format!("{:.6e}", f64::from_bits(min_pos))
        };
        format!(
            " {prefix}_passes={n} {prefix}_cols={} {prefix}_max_pen_env={} \
             {prefix}_max_pen_col={} {prefix}_max_dpen={} {prefix}_max_loosen={} \
             {prefix}_max_rel={} {prefix}_min_pos={min_pos} {prefix}_pos_lost={} \
             {prefix}_dom_viol={} {prefix}_lane_mismatch={} {prefix}_errors={}",
            self.cols.load(Relaxed),
            f(&self.max_pen_env),
            f(&self.max_pen_col),
            f(&self.max_dpen),
            f(&self.max_loosen),
            f(&self.max_rel),
            self.pos_lost.load(Relaxed),
            self.dom_viol.load(Relaxed),
            self.lane_mismatch.load(Relaxed),
            self.errors.load(Relaxed),
        )
    }
}

static AUD_ROW: AuditAcc = AuditAcc::new();
static AUD_PC: AuditAcc = AuditAcc::new();

/// Which shadow an audit record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    /// Row envelope (`envelope`).
    Row,
    /// Per-channel envelope (`per_channel`).
    Channel,
}

impl Variant {
    fn acc(self) -> &'static AuditAcc {
        match self {
            Self::Row => &AUD_ROW,
            Self::Channel => &AUD_PC,
        }
    }
}

/// Record a NON-NEGATIVE finite value as a running max (non-negative f64 bit
/// patterns order like the values).
#[inline]
fn fmax(a: &AtomicU64, v: f64) {
    if v.is_finite() && v >= 0.0 {
        a.fetch_max(v.to_bits(), Relaxed);
    }
}

#[inline]
fn fmin(a: &AtomicU64, v: f64) {
    if v.is_finite() && v >= 0.0 {
        a.fetch_min(v.to_bits(), Relaxed);
    }
}

/// Reset every counter (lane start; called by the census reset).
pub(crate) fn reset() {
    MODE.store(0, Relaxed);
    for a in [
        &PASSES,
        &ELIGIBLE,
        &ENV_PASSES,
        &PC_PASSES,
        &AUDITED,
        &MAX_PEN_ENV,
        &MAX_PEN_PC,
    ] {
        a.store(0, Relaxed);
    }
    AUD_ROW.reset();
    AUD_PC.reset();
}

/// Print (and return) the lane-start engagement line for the frozen policy.
pub(crate) fn announce(lane: ErrLane) -> String {
    MODE.store(lane.code(), Relaxed);
    let line = if lane == ErrLane::PerCol {
        "[mr-errlane] dark mode=off".to_string()
    } else {
        format!(
            "[mr-errlane] armed mode={} min_cols={ENV_MIN_COLS} source=policy",
            lane.token()
        )
    };
    eprintln!("{line}");
    line
}

/// Per-column penalty maximum `max_c (Σ_i e[i,c]·xabs_i + eb_c)` of a pass.
pub(crate) fn max_penalty(out: &PassOut, xabs: &[f64]) -> Option<f64> {
    let e = out.e.as_ref()?;
    let r = e.ncols();
    let es = e.as_slice()?;
    let mut pen = vec![0.0f64; r];
    for (i, &xa) in xabs.iter().enumerate().take(e.nrows()) {
        for (p, &ev) in pen.iter_mut().zip(&es[i * r..(i + 1) * r]) {
            *p += ev * xa;
        }
    }
    Some(
        pen.iter()
            .zip(&out.eb)
            .fold(0.0f64, |m, (&p, &b)| nan_max(m, p + b)),
    )
}

/// Count one Outward pass. `pen` is supplied only for published envelopes.
pub(crate) fn note_pass(lane: PassLane, r: usize, pen: Option<f64>) {
    PASSES.fetch_add(1, Relaxed);
    if r >= ENV_MIN_COLS {
        ELIGIBLE.fetch_add(1, Relaxed);
    }
    match lane {
        PassLane::PerCol => {}
        PassLane::Envelope { per_channel: false } => {
            ENV_PASSES.fetch_add(1, Relaxed);
            if let Some(p) = pen {
                fmax(&MAX_PEN_ENV, p);
            }
        }
        PassLane::Envelope { per_channel: true } => {
            PC_PASSES.fetch_add(1, Relaxed);
            if let Some(p) = pen {
                fmax(&MAX_PEN_PC, p);
            }
        }
    }
}

/// Count one audited (published per-column) pass.
pub(crate) fn note_audited() {
    AUDITED.fetch_add(1, Relaxed);
}

/// A shadow pass failed (non-finite, shape): counted, never propagated.
pub(crate) fn record_audit_error(variant: Variant) {
    variant.acc().errors.fetch_add(1, Relaxed);
}

/// Compare a published per-column pass with one envelope shadow of the SAME
/// pass. `bc`/`be` are the two passes' concretized bounds in the pass's own
/// direction (`lower`).
pub(crate) fn record_audit(
    variant: Variant,
    col: &PassOut,
    env: &PassOut,
    bc: &[f64],
    be: &[f64],
    lower: bool,
    xabs: &[f64],
) {
    let acc = variant.acc();
    acc.passes.fetch_add(1, Relaxed);
    let r = bc.len();
    acc.cols.fetch_add(r as u64, Relaxed);
    // The coefficient lane must be bit-identical: EEL never touches `L` or `b`.
    let lane_same = col.a.raw_dim() == env.a.raw_dim()
        && col
            .a
            .iter()
            .zip(env.a.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && col
            .b
            .iter()
            .zip(&env.b)
            .all(|(x, y)| x.to_bits() == y.to_bits());
    if !lane_same {
        acc.lane_mismatch.fetch_add(1, Relaxed);
    }
    // Elementwise dominance of the certified error and bias-error lanes.
    let dom = match (col.e.as_ref(), env.e.as_ref()) {
        (Some(ec), Some(ee)) if ec.raw_dim() == ee.raw_dim() => {
            ec.iter().zip(ee.iter()).all(|(c, e)| e >= c)
                && col.eb.iter().zip(&env.eb).all(|(c, e)| e >= c)
        }
        _ => false,
    };
    if !dom {
        acc.dom_viol.fetch_add(1, Relaxed);
    }
    // Per-column penalties (`Σ_i e·xabs + eb`) in both lanes.
    let pen_of = |out: &PassOut| -> Vec<f64> {
        let mut pen = out.eb.clone();
        if let Some(e) = out.e.as_ref() {
            let rr = e.ncols();
            if let Some(es) = e.as_slice() {
                for (i, &xa) in xabs.iter().enumerate().take(e.nrows()) {
                    for (p, &ev) in pen.iter_mut().zip(&es[i * rr..(i + 1) * rr]) {
                        *p += ev * xa;
                    }
                }
            }
        }
        pen
    };
    let pc = pen_of(col);
    let pe = pen_of(env);
    for c in 0..r.min(pc.len()).min(pe.len()) {
        fmax(&acc.max_pen_col, pc[c]);
        fmax(&acc.max_pen_env, pe[c]);
        fmax(&acc.max_dpen, (pe[c] - pc[c]).max(0.0));
        let loosen = if lower { bc[c] - be[c] } else { be[c] - bc[c] };
        fmax(&acc.max_loosen, loosen.max(0.0));
        if lower && bc[c] > 0.0 {
            fmin(&acc.min_pos, bc[c]);
            fmax(&acc.max_rel, loosen.max(0.0) / bc[c]);
            if be[c] <= 0.0 {
                acc.pos_lost.fetch_add(1, Relaxed);
            }
        }
    }
}

fn mode_token() -> &'static str {
    match MODE.load(Relaxed) {
        1 => "off",
        2 => "envelope",
        3 => "per_channel",
        4 => "audit",
        _ => "unannounced",
    }
}

fn pen_field(n: u64, a: &AtomicU64) -> String {
    if n == 0 {
        "na".to_string()
    } else {
        format!("{:.3e}", f64::from_bits(a.load(Relaxed)))
    }
}

/// The `[mr-errlane] done ...` stop receipt (fixed schema: every field on
/// every line, `na` where the arm did not measure it).
///
/// `max_pen_env` / `max_pen_col` are the looseness pair: in `audit` mode they
/// are the row envelope's and the per-column lane's maxima over the SAME
/// audited passes (`aud_row_*`); in `envelope` mode `max_pen_env` is the
/// published maximum and `max_pen_col` is `na` (no shadow ran).
pub fn stop_line() -> String {
    let env_passes = ENV_PASSES.load(Relaxed);
    let pc_passes = PC_PASSES.load(Relaxed);
    let aud_row_n = AUD_ROW.passes.load(Relaxed);
    let max_pen_env = if env_passes > 0 {
        pen_field(env_passes, &MAX_PEN_ENV)
    } else {
        pen_field(aud_row_n, &AUD_ROW.max_pen_env)
    };
    let max_pen_col = pen_field(aud_row_n, &AUD_ROW.max_pen_col);
    let aud_pc_n = AUD_PC.passes.load(Relaxed);
    let max_pen_pc = if pc_passes > 0 {
        pen_field(pc_passes, &MAX_PEN_PC)
    } else {
        pen_field(aud_pc_n, &AUD_PC.max_pen_env)
    };
    format!(
        "[mr-errlane] done mode={} min_cols={ENV_MIN_COLS} passes={} eligible={} \
         env_passes={env_passes} pc_passes={pc_passes} audited={} max_pen_env={max_pen_env} \
         max_pen_col={max_pen_col} max_pen_pc={max_pen_pc}{}{}",
        mode_token(),
        PASSES.load(Relaxed),
        ELIGIBLE.load(Relaxed),
        AUDITED.load(Relaxed),
        AUD_ROW.fields("aud_row"),
        AUD_PC.fields("aud_pc"),
    )
}

/// Fields appended to the `[mr-census]` line.
pub(crate) fn census_fields() -> String {
    format!(
        " errlane_mode={} errlane_env_passes={} errlane_pc_passes={} errlane_audited={}",
        mode_token(),
        ENV_PASSES.load(Relaxed),
        PC_PASSES.load(Relaxed),
        AUDITED.load(Relaxed),
    )
}

// ---------------------------------------------------------------------------
//  Test-only per-op trace (the `eel_dominates_percol_every_op` oracle)
// ---------------------------------------------------------------------------

/// Records the certified error lane after every op (and after each ReLU
/// sub-step) of passes run on THIS thread while a capture is open. Compiled
/// only for unit tests; production passes carry no trace code.
#[cfg(test)]
pub(crate) mod trace {
    use ndarray::Array2;
    use std::cell::RefCell;

    /// One recorded step.
    #[derive(Clone, Debug)]
    pub(crate) struct Step {
        /// Op index (`usize::MAX` = seed).
        pub(crate) op: usize,
        /// Sub-step tag.
        pub(crate) tag: &'static str,
        /// Materialized error lane (`E`, or the broadcast envelope).
        pub(crate) e: Option<Array2<f64>>,
        /// Coefficients at the step (must match across lanes bit-for-bit).
        pub(crate) l: Array2<f64>,
        /// Bias-error lane at the step.
        pub(crate) eb: Vec<f64>,
    }

    thread_local! {
        static TRACE: RefCell<Option<Vec<Step>>> = const { RefCell::new(None) };
    }

    /// Is a capture open on this thread?
    pub(crate) fn active() -> bool {
        TRACE.with(|t| t.borrow().is_some())
    }

    /// Append one step (no-op without an open capture).
    pub(crate) fn push(step: Step) {
        TRACE.with(|t| {
            if let Some(v) = t.borrow_mut().as_mut() {
                v.push(step);
            }
        });
    }

    /// Run `f` with a capture open and return its result plus the steps.
    pub(crate) fn capture<R>(f: impl FnOnce() -> R) -> (R, Vec<Step>) {
        TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
        let out = f();
        let steps = TRACE.with(|t| t.borrow_mut().take()).unwrap_or_default();
        (out, steps)
    }
}
