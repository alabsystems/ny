// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! G3: spec agreement (JS F7).
//!
//! The spec translation (VNN-LIB → objective rows → input box) is TCB, and a
//! misread passes every forward self-check: the engine then proves the wrong
//! property perfectly. At sampled points the engine's objective rows and the
//! SAT GATE's own spec evaluator (the full VNN-LIB predicate the organizer's
//! checker mirrors) must agree in sign wherever the margin is clear of the
//! float band:
//!
//! * engine margin `> band` (the engine says "safe here") and the spec says
//!   "violated" — the engine is proving a different property;
//! * engine margin `< −band` (the engine says "violated here") and the spec
//!   says "safe" — likewise.
//!
//! Inside the band either answer is possible and the point is counted blind.
//! This module is the arithmetic; the adapter that evaluates the spec lives
//! where the VNN-LIB evaluator lives (`ny-cli`).

use crate::certify::{GuardId, GuardReport};

/// Relative band (to the logit scale) inside which a sign cannot be trusted.
pub const SPEC_AGREEMENT_REL_BAND: f64 = crate::containment_fence::DOMAIN_FENCE_REL_BAND;

/// One point's comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agreement {
    /// Both sides agree.
    Agree,
    /// The margin is inside the band (or not finite): undecidable here.
    Blind,
    /// The engine's objective rows and the spec evaluator disagree.
    Disagree,
}

/// Compare the engine's smallest objective margin at a point with the spec
/// evaluator's violation predicate at the same point.
#[must_use]
pub fn spec_agreement(engine_min_margin: f64, spec_violated: bool, scale: f64) -> Agreement {
    if !engine_min_margin.is_finite() || !scale.is_finite() {
        return Agreement::Blind;
    }
    let band = SPEC_AGREEMENT_REL_BAND * (1.0 + scale.abs());
    if engine_min_margin > band {
        if spec_violated {
            Agreement::Disagree
        } else {
            Agreement::Agree
        }
    } else if engine_min_margin < -band {
        if spec_violated {
            Agreement::Agree
        } else {
            Agreement::Disagree
        }
    } else {
        Agreement::Blind
    }
}

/// Accumulates G3 over the sampled points.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecAgreementAudit {
    points: u64,
    agree: u64,
    blind: u64,
    disagree: u64,
    first: Option<String>,
}

impl SpecAgreementAudit {
    /// Record one point.
    pub fn note(&mut self, sample: usize, engine_min_margin: f64, spec_violated: bool, scale: f64) {
        self.points += 1;
        match spec_agreement(engine_min_margin, spec_violated, scale) {
            Agreement::Agree => self.agree += 1,
            Agreement::Blind => self.blind += 1,
            Agreement::Disagree => {
                self.disagree += 1;
                if self.first.is_none() {
                    self.first = Some(format!(
                        "sample {sample}: engine min margin {engine_min_margin:e} vs spec \
                         violated={spec_violated}"
                    ));
                }
            }
        }
    }

    /// Points recorded.
    #[must_use]
    pub fn points(&self) -> u64 {
        self.points
    }

    /// Disagreements.
    #[must_use]
    pub fn disagreements(&self) -> u64 {
        self.disagree
    }

    /// The guard's trips.
    #[must_use]
    pub fn report(&self) -> GuardReport {
        let mut report = GuardReport::new();
        if self.disagree > 0 {
            report.trip(
                GuardId::SpecAgreement,
                format!(
                    "{} of {} points disagree with the spec evaluator; first: {}",
                    self.disagree,
                    self.points,
                    self.first.as_deref().unwrap_or("?")
                ),
            );
        }
        report
    }

    /// The summary-line fragment `g3_points= g3_agree= g3_blind= g3_disagree=`.
    #[must_use]
    pub fn fragment(&self) -> String {
        format!(
            "g3_points={} g3_agree={} g3_blind={} g3_disagree={}",
            self.points, self.agree, self.blind, self.disagree
        )
    }
}
