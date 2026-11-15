// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! G2: containment sampling, generic over a lane's IR (JS F9).
//!
//! A certified frame (a root, a state, a sub-domain) publishes enclosures:
//! pre-activation boxes, certified lower bounds of objective rows, and the
//! claim the verdict rests on (every margin strictly positive). At a concrete
//! point of the frame's input box, the lane's own f64 (or double-double)
//! reference semantics must sit inside every one of them. A point that sits
//! OUTSIDE by more than the float band proves the enclosure wrong, and the
//! verdict resting on it is refused.
//!
//! # The band
//!
//! Both sides describe the same exact real function, but they are not the same
//! arithmetic: the reference is plain f64 over the lane's STORED (possibly
//! BatchNorm-folded) weights, while the certified side charges the fold's
//! coefficient error (~2^-24 relative per folded layer). So a comparison must
//! clear [`CONTAINMENT_REL_BAND`] relative to the magnitudes involved — the
//! same band the graph-lane domain fence was calibrated with, two orders of
//! magnitude above the fold error. The defect class this guard exists for is
//! off by the whole range (±1.18e-38 published for a true [-1, 1]).
//!
//! # Cost admission
//!
//! The guard never spends more than the caller's budget: the frame states a
//! conservative per-point cost and working set, and the guard evaluates only
//! as many points as both admit (possibly none — a SKIPPED guard is recorded,
//! never read as a pass or a failure). Skipping is always safe: the guard is
//! downgrade-only.

use std::time::{Duration, Instant};

use ndarray::Array2;

use crate::certify::{GuardId, GuardReport};

/// Relative band for comparing a certified enclosure with an f64 point
/// evaluation of the same function. See the module docs.
pub const CONTAINMENT_REL_BAND: f64 = crate::containment_fence::DOMAIN_FENCE_REL_BAND;

/// Relative band (to the logit scale) for the margin-positivity claim itself;
/// the constant the margin fence was calibrated with.
pub const MARGIN_CLAIM_REL_BAND: f64 = crate::containment_fence::MARGIN_FENCE_REL_BAND;

/// Largest working set the guard may allocate for one batch of points.
pub const MAX_GUARD_BYTES: u64 = 256 * 1024 * 1024;

/// A certified frame the guard can sample, in the lane's own IR.
pub trait CertifiedFrame {
    /// The input box the frame's enclosures cover (the lane's OWN arrays).
    fn input_box(&self) -> (&[f64], &[f64]);

    /// A conservative estimate of the reference evaluation's cost for ONE
    /// point, in seconds. Admission only.
    fn point_cost_secs(&self) -> f64;

    /// Bytes the reference evaluation holds per point. Admission only.
    fn point_bytes(&self) -> u64;

    /// Evaluate the reference semantics at the COLUMNS of `points` and record
    /// every certified enclosure the frame publishes into `audit`. An `Err`
    /// means the evaluator could not run: the guard is then skipped, never
    /// failed.
    fn audit(&self, points: &Array2<f64>, audit: &mut ContainmentAudit) -> Result<(), String>;
}

/// Accumulates the comparisons of one guard run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContainmentAudit {
    checks: u64,
    blind: u64,
    violations: u64,
    first: Option<String>,
}

fn band(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    CONTAINMENT_REL_BAND * (1.0 + scale)
}

impl ContainmentAudit {
    fn violate(&mut self, detail: String) {
        self.violations += 1;
        if self.first.is_none() {
            self.first = Some(detail);
        }
    }

    /// `lo <= value <= hi` up to the band. An infinite side is vacuous; a NaN
    /// value is uninformative (counted blind, never a violation).
    pub fn check_box(
        &mut self,
        what: &str,
        index: usize,
        sample: usize,
        lo: f64,
        hi: f64,
        value: f64,
    ) {
        self.checks += 1;
        if value.is_nan() || lo.is_nan() || hi.is_nan() {
            self.blind += 1;
            return;
        }
        let b = band(&[lo, hi, value]);
        if lo.is_finite() && value < lo - b {
            self.violate(format!(
                "{what}[{index}] sample {sample}: value {value:e} below certified lower {lo:e} \
                 (band {b:e})"
            ));
        } else if hi.is_finite() && value > hi + b {
            self.violate(format!(
                "{what}[{index}] sample {sample}: value {value:e} above certified upper {hi:e} \
                 (band {b:e})"
            ));
        }
    }

    /// `certified_lower <= value` up to the band.
    pub fn check_lower(
        &mut self,
        what: &str,
        index: usize,
        sample: usize,
        certified_lower: f64,
        value: f64,
    ) {
        self.checks += 1;
        if value.is_nan() || certified_lower.is_nan() {
            self.blind += 1;
            return;
        }
        let b = band(&[certified_lower, value]);
        if certified_lower.is_finite() && value < certified_lower - b {
            self.violate(format!(
                "{what}[{index}] sample {sample}: value {value:e} below certified lower bound \
                 {certified_lower:e} (band {b:e})"
            ));
        }
    }

    /// The verdict's own claim `value > 0`, refuted only below
    /// `-MARGIN_CLAIM_REL_BAND · (1 + scale)` (`scale` = the logit magnitude
    /// at the point). Inside the band a violation is undetectable and is
    /// counted blind, never hidden.
    pub fn check_positive(
        &mut self,
        what: &str,
        index: usize,
        sample: usize,
        value: f64,
        scale: f64,
    ) {
        self.checks += 1;
        if value.is_nan() || !scale.is_finite() {
            self.blind += 1;
            return;
        }
        let b = MARGIN_CLAIM_REL_BAND * (1.0 + scale.abs());
        if value < -b {
            self.violate(format!(
                "{what}[{index}] sample {sample}: claimed-positive value is {value:e} \
                 (band {b:e}) — a counterexample to the certified claim"
            ));
        } else if value <= b {
            self.blind += 1;
        }
    }

    /// Comparisons made.
    #[must_use]
    pub fn checks(&self) -> u64 {
        self.checks
    }

    /// Comparisons inside the band (undetectable either way).
    #[must_use]
    pub fn blind(&self) -> u64 {
        self.blind
    }

    /// Violations beyond the band.
    #[must_use]
    pub fn violations(&self) -> u64 {
        self.violations
    }
}

/// What one guard run did.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainmentOutcome {
    /// Points evaluated.
    pub points: usize,
    /// The audit.
    pub audit: ContainmentAudit,
    /// Why the guard did not run (`None` = it ran).
    pub skipped: Option<String>,
    /// Wall seconds spent.
    pub secs: f64,
}

impl ContainmentOutcome {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            points: 0,
            audit: ContainmentAudit::default(),
            skipped: Some(reason.into()),
            secs: 0.0,
        }
    }

    /// The guard's trips (empty unless a violation was seen).
    #[must_use]
    pub fn report(&self) -> GuardReport {
        let mut report = GuardReport::new();
        if self.audit.violations > 0 {
            report.trip(
                GuardId::Containment,
                format!(
                    "{} of {} checks outside the certified enclosure; first: {}",
                    self.audit.violations,
                    self.audit.checks,
                    self.audit.first.as_deref().unwrap_or("?")
                ),
            );
        }
        report
    }

    /// The summary-line fragment
    /// `g2=<ran|skipped(reason)> points= checks= blind= violations= secs=`.
    #[must_use]
    pub fn fragment(&self) -> String {
        let state = self
            .skipped
            .as_ref()
            .map_or_else(|| "ran".to_string(), |r| format!("skipped({r})"));
        format!(
            "g2={state} g2_points={} g2_checks={} g2_blind={} g2_violations={} g2_secs={:.3}",
            self.points, self.audit.checks, self.audit.blind, self.audit.violations, self.secs
        )
    }
}

/// Deterministic sample points of a box: the centre, the two extreme
/// vertices, alternating vertices at doubling strides, then deterministic
/// interior points. Never more than `want`.
#[must_use]
pub fn sample_points(lo: &[f64], hi: &[f64], want: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(want);
    if want == 0 || lo.len() != hi.len() || lo.is_empty() {
        return out;
    }
    out.push(lo.iter().zip(hi).map(|(l, h)| 0.5 * (l + h)).collect());
    if out.len() < want {
        out.push(lo.to_vec());
    }
    if out.len() < want {
        out.push(hi.to_vec());
    }
    let mut stride = 1usize;
    while out.len() < want && stride <= lo.len() {
        out.push(
            lo.iter()
                .zip(hi)
                .enumerate()
                .map(|(i, (l, h))| if (i / stride) % 2 == 1 { *h } else { *l })
                .collect(),
        );
        stride = stride.saturating_mul(2);
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    while out.len() < want {
        out.push(
            lo.iter()
                .zip(hi)
                .map(|(l, h)| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // 53 random bits -> [0, 1).
                    #[allow(clippy::cast_precision_loss)]
                    let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                    let x = l + u * (h - l);
                    // Stay inside the box without `clamp`, which panics on an
                    // inverted or NaN interval (`run_containment` refuses
                    // those, but this function is public).
                    if x < *l {
                        *l
                    } else if x > *h {
                        *h
                    } else {
                        x
                    }
                })
                .collect(),
        );
    }
    out
}

/// Run G2 on `frame` with at most `max_points` points inside `budget`
/// (`None` = no time limit, for tests and proof-mode audits).
pub fn run_containment(
    frame: &dyn CertifiedFrame,
    max_points: u32,
    budget: Option<Duration>,
) -> ContainmentOutcome {
    let t0 = Instant::now();
    let (lo, hi) = frame.input_box();
    if lo.is_empty() || lo.len() != hi.len() {
        return ContainmentOutcome::skipped("bad_box");
    }
    // `!(l <= h)` is true on NaN, which must also refuse.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if lo
        .iter()
        .zip(hi)
        .any(|(l, h)| !(l <= h) || !l.is_finite() || !h.is_finite())
    {
        return ContainmentOutcome::skipped("bad_box");
    }
    let mut admitted = usize::try_from(max_points).unwrap_or(usize::MAX);
    let per_bytes = frame.point_bytes().max(1);
    admitted = admitted.min(usize::try_from(MAX_GUARD_BYTES / per_bytes).unwrap_or(usize::MAX));
    if let Some(budget) = budget {
        let per = frame.point_cost_secs();
        if per.is_finite() && per > 0.0 {
            let fit = (budget.as_secs_f64() / per).floor();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let fit = if fit.is_finite() && fit > 0.0 {
                fit.min(1e9) as usize
            } else {
                0
            };
            admitted = admitted.min(fit);
        }
    }
    if admitted == 0 {
        return ContainmentOutcome::skipped("budget");
    }
    let pts = sample_points(lo, hi, admitted);
    let n = lo.len();
    let mut x = Array2::<f64>::zeros((n, pts.len()));
    for (s, p) in pts.iter().enumerate() {
        for (i, &v) in p.iter().enumerate() {
            x[[i, s]] = v;
        }
    }
    let mut audit = ContainmentAudit::default();
    match frame.audit(&x, &mut audit) {
        Ok(()) => ContainmentOutcome {
            points: pts.len(),
            audit,
            skipped: None,
            secs: t0.elapsed().as_secs_f64(),
        },
        Err(reason) => {
            let mut out = ContainmentOutcome::skipped(format!("evaluator: {reason}"));
            out.secs = t0.elapsed().as_secs_f64();
            out
        }
    }
}
