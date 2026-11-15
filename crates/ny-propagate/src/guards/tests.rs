// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Guard-library tests: every guard is downgrade-only, samples
//! deterministically, admits its own cost, and trips on the defect class it
//! exists for while staying quiet inside the float band.

use std::time::Duration;

use ndarray::Array2;

use super::containment::{
    run_containment, sample_points, CertifiedFrame, ContainmentAudit, CONTAINMENT_REL_BAND,
};
use super::margin_hist::{MarginHistogram, DECADES};
use super::spec_agreement::{spec_agreement, Agreement, SpecAgreementAudit};
use crate::certify::GuardId;

/// A toy frame: `f(x) = Σ x_i` on the box, with a certified enclosure
/// `[lo_sum − slack, hi_sum + slack]` and a claimed-positive margin `f + shift`.
struct SumFrame {
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Added to the certified lower bound: `> 0` makes the enclosure WRONG.
    lower_error: f64,
    shift: f64,
    cost: f64,
}

impl CertifiedFrame for SumFrame {
    fn input_box(&self) -> (&[f64], &[f64]) {
        (&self.lo, &self.hi)
    }

    fn point_cost_secs(&self) -> f64 {
        self.cost
    }

    fn point_bytes(&self) -> u64 {
        64
    }

    fn audit(&self, points: &Array2<f64>, audit: &mut ContainmentAudit) -> Result<(), String> {
        let lo_sum: f64 = self.lo.iter().sum();
        let hi_sum: f64 = self.hi.iter().sum();
        for s in 0..points.ncols() {
            let v: f64 = points.column(s).sum();
            audit.check_box("sum", 0, s, lo_sum + self.lower_error, hi_sum, v);
            audit.check_lower(
                "row",
                0,
                s,
                lo_sum + self.lower_error + self.shift,
                v + self.shift,
            );
            audit.check_positive("margin", 0, s, v + self.shift, v.abs());
        }
        Ok(())
    }
}

fn frame(lower_error: f64, shift: f64) -> SumFrame {
    SumFrame {
        lo: vec![-1.0, 0.5, 2.0],
        hi: vec![1.0, 0.75, 3.0],
        lower_error,
        shift,
        cost: 1e-3,
    }
}

#[test]
fn g2_containment_passes_a_sound_frame_and_trips_on_a_wrong_enclosure() {
    let ok = run_containment(&frame(0.0, 10.0), 16, None);
    assert!(ok.skipped.is_none());
    assert!(ok.points > 0);
    assert_eq!(ok.audit.violations(), 0);
    assert!(ok.report().is_clear());

    // The certified lower bound is raised by 0.5: the centre and every vertex
    // with small coordinates now sits below it.
    let bad = run_containment(&frame(0.5, 10.0), 16, None);
    assert!(bad.audit.violations() > 0);
    let report = bad.report();
    assert!(!report.is_clear());
    assert_eq!(report.trips()[0].guard, GuardId::Containment);

    // A claimed-positive margin that is negative at a sample is a counterexample.
    let neg = run_containment(&frame(0.0, -3.0), 16, None);
    assert!(neg.audit.violations() > 0);
}

#[test]
fn g2_containment_stays_quiet_inside_the_band() {
    // An enclosure wrong by far less than the band is not reported: the band
    // absorbs the uncharged fold error between the two arithmetics.
    let tiny = CONTAINMENT_REL_BAND * 1e-3;
    let out = run_containment(&frame(tiny, 10.0), 16, None);
    assert_eq!(out.audit.violations(), 0);
}

#[test]
fn g2_containment_admits_its_own_cost_and_skips_rather_than_overrun() {
    let mut f = frame(0.5, 10.0);
    f.cost = 1.0;
    // A 0.5 s budget cannot pay for one 1 s point: skipped, not failed.
    let out = run_containment(&f, 16, Some(Duration::from_millis(500)));
    assert_eq!(out.skipped.as_deref(), Some("budget"));
    assert!(out.report().is_clear(), "a skipped guard never trips");
    // 3.5 s buys three points.
    let out = run_containment(&f, 16, Some(Duration::from_millis(3500)));
    assert_eq!(out.points, 3);
    // Bad boxes are skipped.
    let mut inverted = frame(0.0, 10.0);
    inverted.lo[0] = 2.0;
    assert_eq!(
        run_containment(&inverted, 16, None).skipped.as_deref(),
        Some("bad_box")
    );
    let mut nan = frame(0.0, 10.0);
    nan.hi[1] = f64::NAN;
    assert_eq!(
        run_containment(&nan, 16, None).skipped.as_deref(),
        Some("bad_box")
    );
}

#[test]
fn g2_sample_points_are_deterministic_bounded_and_inside_the_box() {
    let lo = vec![-1.0, 0.0, 5.0, -2.0, 0.25];
    let hi = vec![1.0, 0.0, 6.0, 2.0, 0.5];
    for want in [0usize, 1, 2, 3, 7, 32] {
        let a = sample_points(&lo, &hi, want);
        let b = sample_points(&lo, &hi, want);
        assert_eq!(a, b, "deterministic");
        assert!(a.len() <= want);
        if want >= 3 {
            assert_eq!(a[1], lo);
            assert_eq!(a[2], hi);
        }
        for p in &a {
            for (i, v) in p.iter().enumerate() {
                assert!(lo[i] <= *v && *v <= hi[i], "sample outside the box");
            }
        }
    }
    assert_eq!(sample_points(&lo, &hi, 32).len(), 32);
}

#[test]
fn g3_spec_agreement_trips_only_outside_the_band() {
    assert_eq!(spec_agreement(0.5, false, 1.0), Agreement::Agree);
    assert_eq!(spec_agreement(-0.5, true, 1.0), Agreement::Agree);
    assert_eq!(spec_agreement(0.5, true, 1.0), Agreement::Disagree);
    assert_eq!(spec_agreement(-0.5, false, 1.0), Agreement::Disagree);
    assert_eq!(spec_agreement(1e-9, true, 1.0), Agreement::Blind);
    assert_eq!(spec_agreement(f64::NAN, true, 1.0), Agreement::Blind);

    let mut audit = SpecAgreementAudit::default();
    audit.note(0, 0.5, false, 1.0);
    audit.note(1, 1e-12, true, 1.0);
    assert!(audit.report().is_clear());
    audit.note(2, 0.5, true, 1.0);
    let report = audit.report();
    assert_eq!(report.trips()[0].guard, GuardId::SpecAgreement);
    assert_eq!(audit.disagreements(), 1);
    assert_eq!(audit.points(), 3);
}

#[test]
fn g5_margin_histogram_buckets_by_decade_and_tracks_the_minimum() {
    let mut h = MarginHistogram::new();
    for v in [5e-13, 1e-12, 3e-11, 2e-10, 1e-5, 9.9e-5, 1e-4, 0.5] {
        h.note(v);
    }
    // Not closes: ignored.
    for v in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        h.note(v);
    }
    assert_eq!(h.total(), 8);
    assert_eq!(h.min().map(f64::to_bits), Some(5e-13f64.to_bits()));
    let c = h.counts();
    assert_eq!(c.len(), DECADES);
    assert_eq!(c[0], 1, "< 1e-12");
    assert_eq!(c[1], 1, "[1e-12, 1e-11)");
    assert_eq!(c[2], 1, "[1e-11, 1e-10)");
    assert_eq!(c[3], 1, "[1e-10, 1e-9)");
    assert_eq!(c[8], 2, "[1e-5, 1e-4)");
    assert_eq!(c[9], 2, ">= 1e-4");
    assert_eq!(
        h.fragment(),
        "closes=8 min_margin=5.000e-13 margin_hist=1,1,1,1,0,0,0,0,2,2"
    );
    assert_eq!(
        MarginHistogram::new().fragment(),
        "closes=0 min_margin=none margin_hist=0,0,0,0,0,0,0,0,0,0"
    );
}
