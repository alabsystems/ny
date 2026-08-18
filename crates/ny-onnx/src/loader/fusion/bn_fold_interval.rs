// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified interval WIDTH REPORT for the BatchNormalization fold
//! (#bn-interval-fold, W0.2 proposal — reporting increment only).
//!
//! # What this module does, and what it deliberately does not do
//!
//! The W0.2 proposal is to carry folded BN weights as f64 INTERVALS
//! `[fl_down(W*s), fl_up(W*s)]` so a single enclosure covers both the folded
//! expression tree and the unfolded two-step tree. NY's weight representation
//! cannot hold an interval today (see the module-level assessment in
//! `batch_norm_fold.rs`), so this module implements the measurement half of
//! that proposal WITHOUT changing any stored weight:
//!
//!   * it computes, in rigorous outward-rounded f64 interval arithmetic, an
//!     enclosure of the EXACT real-arithmetic fold coefficient
//!     `W_ij · γ_c / sqrt(var_c + ε)` and bias `b_c · s_c + t_c`;
//!   * it takes the hull of that enclosure with the f32 value the fold
//!     ACTUALLY stored;
//!   * it reports the maximum absolute and relative hull width per folded node.
//!
//! The hull width is exactly the quantity a future interval-weight fold would
//! have to carry. Its dominant term is the f32 rounding of the fold equations —
//! the same slack the ORT enclosure property test (`batch_norm_ort_prop.rs`)
//! currently absorbs with a hand-picked `FOLD_TOL_REL = 1e-4`. Reporting it
//! turns that constant into a measured number.
//!
//! # Soundness
//!
//! This module is observational. It never mutates `WeightStore`, never returns
//! a value that gates a fold, and never participates in bound propagation, so
//! it cannot weaken (or strengthen) any bound. It is reached only when the
//! `NY_BN_FOLD_INTERVAL_REPORT=1` dark gate is set; with the gate unset or set
//! to anything else, no function here is called and the fold is byte-identical
//! to the pre-change behaviour.

use ndarray::{ArrayD, Dimension};
use ny_core::dd::{next_down_f64, next_up_f64};
use tracing::info;

#[cfg(test)]
thread_local! {
    /// Thread-local test override for the dark gate.
    ///
    /// Tests must be able to exercise both gate states, but mutating the
    /// process-global environment races with every other test in the binary
    /// that loads a BN graph. A thread-local is race-free because each test
    /// runs on its own thread. It exists only under `cfg(test)`, so production
    /// code has exactly one gate source: the environment variable.
    static FORCE_INTERVAL_REPORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII enabler for the thread-local override.
#[cfg(test)]
pub(super) struct ForceIntervalReport;

#[cfg(test)]
impl ForceIntervalReport {
    pub fn enable() -> Self {
        FORCE_INTERVAL_REPORT.set(true);
        // Start from an empty sink. The test harness REUSES threads across
        // tests, and the sink is thread-local, so a sibling test that emitted
        // reports without draining them would otherwise leak its rows into this
        // test's `take_reports()` and break any exact-count assertion. Draining
        // here (rather than in each test) makes the count deterministic
        // regardless of test order, harness thread reuse, or -j.
        let _ = take_reports();
        Self
    }
}

#[cfg(test)]
impl Drop for ForceIntervalReport {
    fn drop(&mut self) {
        FORCE_INTERVAL_REPORT.set(false);
    }
}

/// Dark gate for the interval width report (#bn-interval-fold). Default OFF:
/// only the exact string `1` enables it, so a stray value cannot silently turn
/// on extra work in a competition run.
pub(super) fn interval_report_enabled() -> bool {
    #[cfg(test)]
    if FORCE_INTERVAL_REPORT.get() {
        return true;
    }
    std::env::var("NY_BN_FOLD_INTERVAL_REPORT").ok().as_deref() == Some("1")
}

/// An outward-rounded f64 interval.
///
/// Every operation rounds the computed endpoint one f64 ULP outward. IEEE-754
/// requires `+`, `-`, `*`, `/` and `sqrt` to be correctly rounded, so the exact
/// real result of each operation lies within half an ULP of the computed f64
/// endpoint and therefore strictly inside the widened endpoint. That makes each
/// method a valid enclosure step regardless of the host rounding mode having no
/// directed-rounding control in stable Rust.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// A degenerate interval holding one exactly-representable value.
    ///
    /// Used for the f32 BN parameters: widening an f32 to f64 is exact, so the
    /// authored `γ`, `β`, `mean`, `var`, `ε` and `W` enter the computation with
    /// zero width and all reported width is genuinely fold-induced.
    pub fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    /// The whole real line — the only honest enclosure when a channel's affine
    /// cannot be bounded. Reported as `unenclosable`, never as a small width.
    pub fn whole() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    pub fn contains(self, value: f64) -> bool {
        value >= self.lo && value <= self.hi
    }

    /// Width, itself rounded UP so a reported width is never an understatement.
    pub fn width(self) -> f64 {
        if !self.lo.is_finite() || !self.hi.is_finite() {
            return f64::INFINITY;
        }
        next_up_f64(self.hi - self.lo)
    }

    /// Smallest interval containing `self` and the point `value`.
    pub fn hull_with(self, value: f64) -> Self {
        Self {
            lo: self.lo.min(value),
            hi: self.hi.max(value),
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            lo: next_down_f64(self.lo + other.lo),
            hi: next_up_f64(self.hi + other.hi),
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: next_down_f64(self.lo - other.hi),
            hi: next_up_f64(self.hi - other.lo),
        }
    }

    /// All four endpoint products, then outward rounding — the textbook
    /// sign-agnostic interval product. Endpoint enumeration (rather than a
    /// sign-case table) keeps the mixed-sign cases correct without a branch
    /// matrix to get wrong.
    pub fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let mut lo = products[0];
        let mut hi = products[0];
        for product in &products[1..] {
            lo = lo.min(*product);
            hi = hi.max(*product);
        }
        Self {
            lo: next_down_f64(lo),
            hi: next_up_f64(hi),
        }
    }

    /// Interval division. `None` when the divisor straddles zero (or is not
    /// finite), so a degenerate BN denominator fails closed rather than
    /// reporting an unbounded width as if it were a measurement.
    pub fn div(self, other: Self) -> Option<Self> {
        if !other.lo.is_finite() || !other.hi.is_finite() || (other.lo <= 0.0 && other.hi >= 0.0) {
            return None;
        }
        let quotients = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        let mut lo = quotients[0];
        let mut hi = quotients[0];
        for quotient in &quotients[1..] {
            lo = lo.min(*quotient);
            hi = hi.max(*quotient);
        }
        Some(Self {
            lo: next_down_f64(lo),
            hi: next_up_f64(hi),
        })
    }

    /// Interval square root. `None` for a negative lower endpoint, matching the
    /// fold's own rejection of a non-positive denominator.
    pub fn sqrt(self) -> Option<Self> {
        if self.lo < 0.0 || !self.hi.is_finite() {
            return None;
        }
        Some(Self {
            lo: next_down_f64(self.lo.sqrt()),
            hi: next_up_f64(self.hi.sqrt()),
        })
    }
}

/// Rigorous enclosures of the BN affine for one channel, in exact-real terms.
///
/// `scale` encloses `γ / sqrt(var + ε)` and `shift` encloses
/// `β − (γ · mean) / sqrt(var + ε)` — the same expressions
/// `batch_norm_affine` evaluates in f32, but with every rounding accounted for.
#[derive(Clone, Copy, Debug)]
pub(super) struct ChannelAffineInterval {
    pub scale: Interval,
    pub shift: Interval,
}

impl ChannelAffineInterval {
    /// The vacuous enclosure used when a channel's affine cannot be bounded.
    pub fn unenclosable() -> Self {
        Self {
            scale: Interval::whole(),
            shift: Interval::whole(),
        }
    }
}

/// Enclose one channel's BN affine. Returns `None` on the same degenerate
/// denominators the f32 fold rejects.
pub(super) fn channel_affine_interval(
    gamma: f32,
    beta: f32,
    mean: f32,
    var: f32,
    epsilon: f32,
) -> Option<ChannelAffineInterval> {
    // f32 -> f64 widening is exact, so these carry zero width.
    let gamma = Interval::point(f64::from(gamma));
    let beta = Interval::point(f64::from(beta));
    let mean = Interval::point(f64::from(mean));
    let var = Interval::point(f64::from(var));
    let epsilon = Interval::point(f64::from(epsilon));

    let denominator = var.add(epsilon).sqrt()?;
    if denominator.lo <= 0.0 {
        return None;
    }
    let scale = gamma.div(denominator)?;
    // Grouped as `(γ·mean)/d` to match the f32 evaluation order in
    // `batch_norm_affine`. Real multiplication/division is associative, so any
    // grouping encloses the same real value; matching the source keeps the
    // reported width attributable term by term.
    let shift = beta.sub(gamma.mul(mean).div(denominator)?);
    Some(ChannelAffineInterval { scale, shift })
}

/// Per-node summary of the certified fold interval (#bn-interval-fold).
///
/// All widths are hull widths: the enclosure of the exact real fold value
/// UNIONED with the f32 value actually stored. `weight_*` covers the fused
/// weight tensor, `bias_*` the fused bias vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct FoldIntervalReport {
    pub weight_elements: usize,
    pub bias_elements: usize,
    /// Largest absolute hull width over fused weight elements.
    pub weight_max_abs_width: f64,
    /// Largest hull width relative to `max(1, |stored|)` over weight elements.
    /// Comparable head-to-head with `batch_norm_ort_prop::FOLD_TOL_REL`.
    pub weight_max_rel_width: f64,
    pub bias_max_abs_width: f64,
    pub bias_max_rel_width: f64,
    /// Stored f32 values whose exact-value enclosure did NOT already contain
    /// them, i.e. elements where the fold's own f32 rounding is visible. This
    /// is expected to be nonzero and is not an error; it is the measurement.
    pub stored_outside_exact_enclosure: usize,
    /// Elements for which no enclosure could be formed (degenerate denominator
    /// or non-finite intermediate). A nonzero count means the report is partial.
    pub unenclosable_elements: usize,
}

impl FoldIntervalReport {
    fn record(&mut self, exact: Option<Interval>, stored: f64, is_bias: bool) {
        if is_bias {
            self.bias_elements += 1;
        } else {
            self.weight_elements += 1;
        }
        let Some(exact) = exact else {
            self.unenclosable_elements += 1;
            return;
        };
        if !exact.contains(stored) {
            self.stored_outside_exact_enclosure += 1;
        }
        let hull = exact.hull_with(stored);
        let abs_width = hull.width();
        let rel_width = abs_width / stored.abs().max(1.0);
        if is_bias {
            self.bias_max_abs_width = self.bias_max_abs_width.max(abs_width);
            self.bias_max_rel_width = self.bias_max_rel_width.max(rel_width);
        } else {
            self.weight_max_abs_width = self.weight_max_abs_width.max(abs_width);
            self.weight_max_rel_width = self.weight_max_rel_width.max(rel_width);
        }
    }
}

/// How the fold's per-channel scale maps onto an axis of the fused weight.
///
/// The direct Conv fold scales axis 0 by channel, ConvTranspose scales axis 1,
/// and Gemm scales axis 0 or 1 depending on `transB` — all three are "index
/// along one axis selects the channel", which is what this describes.
#[derive(Clone, Copy, Debug)]
pub(super) struct ChannelAxis {
    pub axis: usize,
    /// Features per channel. `1` for the direct folds; `block` for the
    /// across-Reshape folds where `channel = feature / block`.
    pub block: usize,
}

/// Compute the certified interval width report for one fired fold.
///
/// `original_weight` is the pre-fold tensor, `fused_weight` the f32 tensor the
/// fold stored, and `original_bias` / `fused_bias` the same for the bias (the
/// synthesized-bias case passes `original_bias = None`, matching `fuse_bias`'s
/// zero-bias branch).
///
/// Returns `None` only when the shapes are inconsistent with `channel_axis` —
/// which the caller has already validated, so `None` here means "do not report"
/// rather than "the fold is wrong".
pub(super) fn fold_interval_report(
    original_weight: &ArrayD<f32>,
    fused_weight: &ArrayD<f32>,
    original_bias: Option<&ArrayD<f32>>,
    fused_bias: &ArrayD<f32>,
    affine: &[ChannelAffineInterval],
    channel_axis: ChannelAxis,
) -> Option<FoldIntervalReport> {
    if original_weight.shape() != fused_weight.shape()
        || channel_axis.block == 0
        || channel_axis.axis >= original_weight.ndim()
    {
        return None;
    }
    let axis_len = original_weight.shape()[channel_axis.axis];
    if axis_len != affine.len().checked_mul(channel_axis.block)? {
        return None;
    }

    let mut report = FoldIntervalReport::default();

    // Iterate by multi-index so the channel of each element is unambiguous for
    // any rank and any channel axis, rather than relying on a flat-index
    // stride computation that would have to mirror ndarray's layout.
    for (index, stored) in fused_weight.indexed_iter() {
        let channel = index.as_array_view()[channel_axis.axis] / channel_axis.block;
        let exact = affine.get(channel).map(|channel_affine| {
            Interval::point(f64::from(original_weight[&index])).mul(channel_affine.scale)
        });
        report.record(exact, f64::from(*stored), false);
    }

    // Fused bias: `b'[k] = b[k] * scale[c(k)] + shift[c(k)]`, with `b = 0` when
    // the fold synthesized the bias. The direct folds have one bias entry per
    // channel; the Gemm->Reshape->BN fold has `block` entries per channel.
    let bias_block = if fused_bias.len() == affine.len() {
        1
    } else if fused_bias.len() == affine.len().checked_mul(channel_axis.block)? {
        channel_axis.block
    } else {
        // A bias whose length matches neither layout (e.g. the
        // BN->Reshape->Gemm tail, whose bias is an inner-product over features
        // rather than a per-channel affine) is reported as weight-only.
        return Some(report);
    };
    for (position, stored) in fused_bias.iter().enumerate() {
        let channel = position / bias_block;
        let exact = affine.get(channel).map(|channel_affine| {
            let base = original_bias
                .and_then(|bias| bias.iter().nth(position).copied())
                .unwrap_or(0.0);
            Interval::point(f64::from(base))
                .mul(channel_affine.scale)
                .add(channel_affine.shift)
        });
        report.record(exact, f64::from(*stored), true);
    }

    Some(report)
}

#[cfg(test)]
thread_local! {
    /// Thread-local capture of emitted reports, so tests can assert on the
    /// actual numbers rather than scraping log output. Test-only, like the gate
    /// override above.
    static REPORT_SINK: std::cell::RefCell<Vec<FoldIntervalReport>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Drain and return every report emitted on this thread.
#[cfg(test)]
pub(super) fn take_reports() -> Vec<FoldIntervalReport> {
    REPORT_SINK.with_borrow_mut(std::mem::take)
}

/// Emit one report line. Structured `tracing` fields so a run can be swept with
/// a field filter; `info` (not `debug`) because the gate is explicitly opt-in
/// and its whole purpose is to produce this output.
pub(super) fn emit_report(
    op_type: &str,
    node_idx: usize,
    bn_idx: usize,
    report: &FoldIntervalReport,
) {
    #[cfg(test)]
    REPORT_SINK.with_borrow_mut(|sink| sink.push(*report));
    info!(
        target: "ny::bn_fold_interval",
        op_type,
        node_idx,
        bn_idx,
        weight_elements = report.weight_elements,
        bias_elements = report.bias_elements,
        weight_max_abs_width = report.weight_max_abs_width,
        weight_max_rel_width = report.weight_max_rel_width,
        bias_max_abs_width = report.bias_max_abs_width,
        bias_max_rel_width = report.bias_max_rel_width,
        stored_outside_exact_enclosure = report.stored_outside_exact_enclosure,
        unenclosable_elements = report.unenclosable_elements,
        "BN fold certified interval width"
    );
}
