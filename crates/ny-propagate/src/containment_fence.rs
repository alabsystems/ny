// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime containment fence for unsat-emitting paths (`#containment-fence`).
//!
//! # Why this exists
//!
//! On 2026-08-25 an n-ary `Concat` input-pairing defect published **finite,
//! plausible, non-containing** bounds on the default path: `±1.18e-38` for a
//! true range of `[-1, 1]`. Nothing caught it at runtime. `root_bounds_are_finite()`
//! was green, no lever was involved, and it silently minted 21 `unsat` verdicts
//! on `vit_2023` that had to be retracted (`531b658e1`, `0c8e7e4f7`). The fix
//! for that specific bug was type-level and lane-specific. The CLASS had — and
//! until this module, still has — no runtime guard at all.
//!
//! Every false proof of that class shares one signature: **the certified
//! enclosure does not contain the network's own behaviour.** That is checkable
//! directly, cheaply, and without trusting any part of the bound pipeline.
//!
//! # The check
//!
//! Evaluate the network at concrete points drawn from the input box through a
//! SOUND evaluator, and compare against the certified output enclosure. A
//! degenerate interval (`lower == upper`) makes the interval forward a sound
//! enclosure of a single point, so for a sample `p` the evaluator returns
//! `got` with `got.lower <= f(p) <= got.upper`.
//!
//! The certified enclosure claims `cert.lower <= f(x) <= cert.upper` for EVERY
//! `x` in the box — in particular for `p`. So the two intervals must overlap.
//! If they are strictly disjoint on any coordinate, the certified enclosure
//! provably does not contain `f(p)`, and any verdict resting on it is unsound.
//!
//! No tolerance is applied, and none is correct here: both sides are sound
//! enclosures, so a strict separation is a genuine contradiction rather than a
//! rounding artifact. That is what makes this a fence and not a heuristic.
//!
//! # Why it is safe to run unconditionally
//!
//! It is **downgrade-only by construction**: the only thing it can do is turn a
//! (wrong) `Verified` into `Unknown`. It can never manufacture a proof, tighten
//! a bound, or change which lane runs. A false alarm therefore costs at most
//! one row's `+10`, while a miss of the class it detects costs `-150` and the
//! moat. That asymmetry is the entire argument for running it on every unsat.
//!
//! # What it is independent OF, precisely
//!
//! It shares no code with CROWN, the tableau, the alpha ascent, or the MIP
//! seam — the pipeline that computes the certified bound — so a defect in any
//! of those cannot mask itself here. That is the independence the fence needs
//! and the only one it claims.
//!
//! It is NOT independent of everything. It evaluates through
//! `graph_ibp_f64_cell`, which is also what loads and interprets the model, so
//! a defect in op SEMANTICS shared by both sides (a Conv that means the wrong
//! thing in the same way twice) is invisible to it. The 2026-08-25 defect was
//! not of that kind — it was a wiring error in bound propagation, correctly
//! evaluated on the point side — which is why the fence catches it. State the
//! blind spot rather than overclaiming: a fence whose advertised coverage
//! exceeds its real coverage retires a question it never answered.

use ndarray::ArrayD;
use ny_core::Result;

use crate::network::graph_ibp_f64_cell::Interval64;

/// How many sample points the fence may evaluate. Each costs one interval
/// forward pass at a point, which is the cheapest evaluation the engine has.
pub const DEFAULT_FENCE_SAMPLES: usize = 8;

/// The outcome of a containment check.
#[derive(Debug, Clone, PartialEq)]
pub enum FenceOutcome {
    /// Every sampled point's sound enclosure overlaps the certified enclosure.
    /// This is NOT a proof that the bound is correct — it is the absence of a
    /// detected contradiction.
    Consistent {
        /// Points actually evaluated (an evaluator refusal reduces this).
        samples: usize,
    },
    /// A sampled point's sound enclosure is strictly disjoint from the
    /// certified enclosure. The certified bound does not contain the network's
    /// behaviour; any verdict resting on it must be refused.
    Contradiction(Box<FenceContradiction>),
    /// The fence could not run (evaluator refused every sample, shape
    /// mismatch, empty box). Carries no information either way — callers must
    /// NOT read this as consistent.
    Inconclusive {
        /// Why the fence could not reach a conclusion.
        reason: String,
    },
}

/// The specific coordinate and values that falsify a certified enclosure.
#[derive(Debug, Clone, PartialEq)]
pub struct FenceContradiction {
    /// Flat index of the output coordinate that separates.
    pub index: usize,
    /// The certified enclosure at that coordinate.
    pub certified: (f64, f64),
    /// The sound point enclosure at that coordinate.
    pub observed: (f64, f64),
    /// Which sample produced it (index into the generated sample list).
    pub sample: usize,
}

impl FenceContradiction {
    /// A one-line, log-ready statement of the falsification.
    pub fn describe(&self) -> String {
        format!(
            "output[{}] certified [{:e}, {:e}] excludes observed [{:e}, {:e}] at sample {}",
            self.index,
            self.certified.0,
            self.certified.1,
            self.observed.0,
            self.observed.1,
            self.sample
        )
    }
}

/// Deterministic sample points for a box. Order matters only for reproducibility.
///
/// The first three are the classic witnesses (centre and the two extreme
/// vertices); the rest are alternating vertices chosen by index parity against a
/// varying stride, which reaches distinct corners of a high-dimensional box
/// without enumerating `2^n` of them. Deterministic by construction — the fence
/// must give the same answer twice on the same input.
fn sample_points(lo: &ArrayD<f64>, hi: &ArrayD<f64>, want: usize) -> Vec<ArrayD<f64>> {
    let mut out = Vec::with_capacity(want);
    if want == 0 || lo.shape() != hi.shape() {
        return out;
    }
    // Centre, then the two extremes.
    let mid = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, h)| 0.5 * (l + h))
        .collect::<Vec<_>>();
    out.push(ArrayD::from_shape_vec(lo.raw_dim(), mid).expect("centre shape"));
    if want > 1 {
        out.push(lo.clone());
    }
    if want > 2 {
        out.push(hi.clone());
    }
    // Alternating vertices: stride k selects `hi` where `(i / k) % 2 == 1`.
    let mut stride = 1usize;
    while out.len() < want {
        let v = lo
            .iter()
            .zip(hi.iter())
            .enumerate()
            .map(|(i, (l, h))| if (i / stride) % 2 == 1 { *h } else { *l })
            .collect::<Vec<_>>();
        out.push(ArrayD::from_shape_vec(lo.raw_dim(), v).expect("vertex shape"));
        stride = stride.saturating_mul(2);
        if stride > lo.len().max(1) {
            break;
        }
    }
    out
}

/// Check that a certified output enclosure contains the network's behaviour at
/// sampled points of the input box.
///
/// `eval_point` must be a SOUND evaluator: given a point it returns an interval
/// guaranteed to contain the true output. A degenerate `Interval64` through the
/// engine's f64 cell forward satisfies this.
///
/// Returns [`FenceOutcome::Contradiction`] only when a sound point enclosure is
/// STRICTLY disjoint from the certified enclosure on some coordinate — which is
/// a genuine falsification, not a tolerance question.
pub fn check_certified_bound_contains_samples<F>(
    lo: &ArrayD<f64>,
    hi: &ArrayD<f64>,
    certified: &Interval64,
    max_samples: usize,
    mut eval_point: F,
) -> FenceOutcome
where
    F: FnMut(&ArrayD<f64>) -> Result<Interval64>,
{
    if lo.shape() != hi.shape() {
        return FenceOutcome::Inconclusive {
            reason: "input box lower/upper shape mismatch".into(),
        };
    }
    // BOTH conditions, checked explicitly. An earlier version of this guard
    // tested only `l <= h`, which (-inf, +inf) SATISFIES — the centre sample
    // then computed `0.5 * (-inf + inf)` = NaN, every comparison against NaN is
    // false, and the fence returned `Consistent` having compared nothing. That
    // is a false PASS inside the guard against false PASSes, on exactly the
    // input shape most likely to indicate an upstream defect (an unbounded box
    // is what a failed bound propagation looks like). Found by adversarial
    // review; pinned by `an_unbounded_box_is_inconclusive_not_consistent`.
    // The NEGATION is the point, so clippy's `partial_cmp` rewrite is declined
    // here: `!(l <= h)` is TRUE when the two are incomparable (NaN), which is
    // exactly the case the comment above says must reach `Inconclusive`. A
    // `l > h` spelling is FALSE on NaN and reinstates the false PASS.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if lo.iter().zip(hi.iter()).any(|(l, h)| !(l <= h)) {
        return FenceOutcome::Inconclusive {
            reason: "input box is empty or inverted".into(),
        };
    }
    if lo.iter().chain(hi.iter()).any(|v| !v.is_finite()) {
        return FenceOutcome::Inconclusive {
            reason: "input box carries a non-finite endpoint; no sample point is well defined"
                .into(),
        };
    }
    let points = sample_points(lo, hi, max_samples);
    if points.is_empty() {
        return FenceOutcome::Inconclusive {
            reason: "no sample points could be generated".into(),
        };
    }

    let cert_lo = certified.lower.as_slice().map(|s| s.to_vec());
    let cert_hi = certified.upper.as_slice().map(|s| s.to_vec());
    let (Some(cert_lo), Some(cert_hi)) = (cert_lo, cert_hi) else {
        return FenceOutcome::Inconclusive {
            reason: "certified enclosure is not in standard layout".into(),
        };
    };

    let mut evaluated = 0usize;
    for (sample, p) in points.iter().enumerate() {
        let got = match eval_point(p) {
            Ok(g) => g,
            // A refusal is not evidence of anything. Skip it; if EVERY sample
            // refuses we report Inconclusive rather than Consistent, because
            // "nothing contradicted me" and "I never looked" must not be the
            // same answer (the absence-as-evidence trap).
            Err(_) => continue,
        };
        let (Some(g_lo), Some(g_hi)) = (got.lower.as_slice(), got.upper.as_slice()) else {
            continue;
        };
        if g_lo.len() != cert_lo.len() || g_hi.len() != cert_hi.len() {
            continue;
        }
        evaluated += 1;
        for i in 0..cert_lo.len() {
            // Strict separation in either direction falsifies containment.
            // Non-finite values are skipped: an infinite certified bound
            // contains everything, and a non-finite observation is a different
            // defect that this fence does not adjudicate.
            if !cert_lo[i].is_finite() || !cert_hi[i].is_finite() {
                continue;
            }
            if !g_lo[i].is_finite() || !g_hi[i].is_finite() {
                continue;
            }
            if g_lo[i] > cert_hi[i] || g_hi[i] < cert_lo[i] {
                return FenceOutcome::Contradiction(Box::new(FenceContradiction {
                    index: i,
                    certified: (cert_lo[i], cert_hi[i]),
                    observed: (g_lo[i], g_hi[i]),
                    sample,
                }));
            }
        }
    }

    if evaluated == 0 {
        return FenceOutcome::Inconclusive {
            reason: "every sample evaluation refused".into(),
        };
    }
    FenceOutcome::Consistent { samples: evaluated }
}

// ===================== margin-positivity fence (#containment-fence) =========

/// Relative band inside which a sampled margin cannot refute anything.
///
/// Reuses the constant the repo already calibrated for exactly this comparison
/// — `margin_row::gpu_seam::PROBE_REL_SLACK` (`gpu_seam.rs:188`), applied there
/// as `tol = PROBE_REL_SLACK * (1.0 + f.abs())` (`gpu_seam.rs:1525`,
/// `gpu_backward/cache.rs:438`) so a realization probe refuses to trip inside
/// the band. The same shape of check deserves the same band.
pub const MARGIN_FENCE_REL_BAND: f64 = 1e-9;

/// Outcome of a margin-positivity check.
#[derive(Debug, Clone, PartialEq)]
pub enum MarginFenceOutcome {
    /// No sampled point refuted the claim.
    ///
    /// `min_margin` is the smallest margin seen and `blind_samples` counts the
    /// (sample, class) pairs that landed inside the band — where a genuine
    /// violation could NOT have been detected. A `Consistent` with a large
    /// `blind_samples` is much weaker evidence than one without, and callers
    /// that log a coverage claim must say which they got.
    Consistent {
        /// Points actually evaluated.
        samples: usize,
        /// Smallest margin observed across all samples and classes.
        min_margin: f64,
        /// (sample, class) pairs inside the undetectable band.
        blind_samples: usize,
    },
    /// A sampled point's margin is negative by more than the float band. The
    /// claim `margin > 0 on the whole box` is false, and the verdict resting on
    /// it must be refused.
    Contradiction(Box<MarginContradiction>),
    /// The fence could not run. Carries no information either way.
    Inconclusive {
        /// Why.
        reason: String,
    },
}

/// The specific class and value that falsify a margin-row `Unsat`.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginContradiction {
    /// Adversarial class whose margin went non-positive.
    pub class: usize,
    /// The margin observed there.
    pub margin: f64,
    /// The band it had to clear to count.
    pub band: f64,
    /// Which sample produced it.
    pub sample: usize,
}

impl MarginContradiction {
    /// A one-line, log-ready statement of the falsification.
    pub fn describe(&self) -> String {
        format!(
            "margin against class {} is {:e} at sample {} (band {:e}) — the lane certified it \
             POSITIVE over the whole box, so this point is a counterexample",
            self.class, self.margin, self.sample, self.band
        )
    }
}

/// Check a margin-row `Unsat` against concrete points of its own input box.
///
/// # What the lane claims
///
/// `MarginRowOutcome::Unsat` means: for every adversarial class `j` and every
/// `x` in the declared box, `out_t(x) > out_j(x)`. The compiled net's tail is
/// validated to be exactly `Flatten -> Gemm1 -> Relu -> Gemm2(last)`
/// (`margin_row/net.rs`), so the margin row
/// `(W2[t,:] - W2[j,:]) . relu(y) + (b2[t] - b2[j])` IS the output difference,
/// not a proxy. Sampling it is therefore a direct test of the claim, not a
/// heuristic — which is what makes this worth running on the highest-volume
/// unsat lane instead of recording a vacuous "fenced" disposition for it.
///
/// # Why the caller must pass the lane's OWN box
///
/// `lo`/`hi` must be the very arrays the lane was handed. Do NOT re-parse or
/// re-derive them. Two live reasons:
///
/// * the lane accepts up to `1e-6` per-coordinate disagreement between its hand
///   parser and `load_vnnlib` before failing closed, so a re-derived box can sit
///   up to `1e-6` OUTSIDE the proven region — and the corners, which are the
///   most valuable samples, are exactly where that bites;
/// * `spec.input_bounds` is widened to the clause UNION for mixed disjunctive
///   properties, so re-deriving can hand back a strictly larger box.
///
/// Given the lane's own arrays the sampling is safe by construction: the lane
/// proves over `mid ± rad` with `rad` rounded OUTWARD (`next_up`), so `[lo,hi]`
/// sits strictly inside the certified region.
///
/// # Why this is one-sided, and NOT `check_certified_bound_contains_samples`
///
/// That function compares two two-sided `Interval64`s and requires a SOUND
/// point evaluator. Neither holds here: the claim is the one-sided
/// `margin ∈ (0, +inf)`, and the natural evaluator (`TwinNet::forward_points`)
/// is documented "exact f64 forward evaluation" — plain accumulation, no
/// interval, no error term.
///
/// # The band, which is the whole false-alarm argument
///
/// A fence alarming on `margin <= 0` from an exact-f64 evaluation WOULD false-
/// alarm on correct `unsat`s, because the two sides are not the same arithmetic:
///
/// * the point side ignores the BatchNorm-fold error (`bias_err`, `w_rel`) that
///   the certified side charges — and cifar100's ResNet folds 20 BNs;
/// * the certified side may run f32 kernels (`root_f32`) or f32 storage;
/// * `closed()` has NO epsilon (`b > 0.0`), so on a hard row the certified
///   margin can legitimately be `1e-12` — the same order as the float error.
///
/// So a refutation must clear a band relative to the LOGIT SCALE — the
/// evaluator returns `(margins, max|logit|)` at each point and the band is
/// `REL * (1 + scale)`, because the error being tolerated grows with `|W||z|`
/// and not with how close the margin sits to zero. Only a margin BELOW it
/// counts. Above the band a violation is real; inside it we cannot tell, and
/// the count is reported rather than silently folded into a pass. This costs
/// detection power only within ~1e-9 relative of zero, while the defect class
/// the fence exists for is off by orders of magnitude — the 2026-08-25 Concat
/// bug published `±1.18e-38` for a true range of `[-1, 1]`.
///
/// Downgrade-only: the worst outcome is a correct proof becoming `Unknown`
/// (-10); a missed false `unsat` costs -150 and the moat.
pub fn check_margin_positive_at_samples<F>(
    lo: &[f64],
    hi: &[f64],
    adv: &[usize],
    max_samples: usize,
    mut eval_margins: F,
) -> MarginFenceOutcome
where
    F: FnMut(&[f64]) -> Result<(Vec<f64>, f64)>,
{
    if lo.len() != hi.len() {
        return MarginFenceOutcome::Inconclusive {
            reason: "input box lower/upper length mismatch".into(),
        };
    }
    if lo.is_empty() {
        return MarginFenceOutcome::Inconclusive {
            reason: "empty input box".into(),
        };
    }
    // The NEGATION is the point, so clippy's `partial_cmp` rewrite is declined
    // here: `!(l <= h)` is TRUE when the two are incomparable (NaN), which is
    // exactly the case the comment above says must reach `Inconclusive`. A
    // `l > h` spelling is FALSE on NaN and reinstates the false PASS.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if lo.iter().zip(hi.iter()).any(|(l, h)| !(l <= h)) {
        return MarginFenceOutcome::Inconclusive {
            reason: "input box is empty or inverted".into(),
        };
    }
    if lo.iter().chain(hi.iter()).any(|v| !v.is_finite()) {
        return MarginFenceOutcome::Inconclusive {
            reason: "input box carries a non-finite endpoint; no sample point is well defined"
                .into(),
        };
    }
    if adv.is_empty() {
        return MarginFenceOutcome::Inconclusive {
            reason: "no adversarial classes to check".into(),
        };
    }

    let pts = margin_sample_points(lo, hi, max_samples);
    let mut evaluated = 0usize;
    let mut min_margin = f64::INFINITY;
    let mut blind = 0usize;

    for (s, p) in pts.iter().enumerate() {
        let Ok((margins, scale)) = eval_margins(p) else {
            continue;
        };
        if !scale.is_finite() || scale < 0.0 {
            return MarginFenceOutcome::Inconclusive {
                reason: "evaluator reported a non-finite or negative logit scale".into(),
            };
        }
        if margins.len() != adv.len() {
            return MarginFenceOutcome::Inconclusive {
                reason: format!(
                    "evaluator returned {} margins for {} adversarial classes",
                    margins.len(),
                    adv.len()
                ),
            };
        }
        if margins.iter().any(|m| m.is_nan()) {
            return MarginFenceOutcome::Inconclusive {
                reason: "evaluator produced a NaN margin".into(),
            };
        }
        evaluated += 1;
        for (r, &m) in margins.iter().enumerate() {
            if m < min_margin {
                min_margin = m;
            }
            // Relative to the LOGIT SCALE, not to the margin. The error being
            // tolerated is the accumulated f64 error of the forward pass, which
            // grows with |W||z| — the magnitude of the logits — and is entirely
            // independent of how close the margin happens to sit to zero. An
            // earlier version scaled by `m.abs()`, which degenerates: solving
            // `m < -REL*(1+|m|)` gives `|m| > ~REL` whatever the network, so the
            // band collapsed to a fixed ~1e-9 absolute floor and would have been
            // too tight on any model with large logits (at |logit| ~ 1e4 the
            // Higham envelope alone is ~5e-9). Caught by
            // `the_band_scales_with_magnitude`.
            let band = MARGIN_FENCE_REL_BAND * (1.0 + scale);
            if m < -band {
                return MarginFenceOutcome::Contradiction(Box::new(MarginContradiction {
                    class: adv[r],
                    margin: m,
                    band,
                    sample: s,
                }));
            }
            if m <= band {
                // Inside the band: a genuine violation here would have been
                // indistinguishable from float error. Counted, never hidden.
                blind += 1;
            }
        }
    }

    if evaluated == 0 {
        return MarginFenceOutcome::Inconclusive {
            reason: "the evaluator refused every sample; nothing was compared".into(),
        };
    }
    MarginFenceOutcome::Consistent {
        samples: evaluated,
        min_margin,
        blind_samples: blind,
    }
}

/// Deterministic sample points for a flat box. Centre, the two extreme
/// vertices, then alternating vertices by index parity against a varying
/// stride. Deterministic by construction — the fence must give the same answer
/// twice on the same input.
fn margin_sample_points(lo: &[f64], hi: &[f64], want: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(want);
    if want == 0 {
        return out;
    }
    out.push(
        lo.iter()
            .zip(hi.iter())
            .map(|(l, h)| 0.5 * (l + h))
            .collect(),
    );
    if want > 1 {
        out.push(lo.to_vec());
    }
    if want > 2 {
        out.push(hi.to_vec());
    }
    let mut stride = 1usize;
    while out.len() < want {
        out.push(
            lo.iter()
                .zip(hi.iter())
                .enumerate()
                .map(|(i, (l, h))| if (i / stride) % 2 == 1 { *h } else { *l })
                .collect(),
        );
        stride = stride.saturating_mul(2);
        if stride > lo.len().max(1) {
            break;
        }
    }
    out
}

// ===================== graph-BaB domain fence (#vit-bank-gate) ==============
//
// The two checks above compare ONE certified output enclosure against the
// network's behaviour over the WHOLE input box. The 2026-08-25 defect did not
// live there: it lived in the branch-and-bound CHILD backward
// (`batched/backward_core.rs`, `constraints/backward/dispatch.rs`), which
// certifies a lower bound over a SUB-region — the input box intersected with
// the domain's ReLU split literals. A child's bound is only obliged to contain
// the network at points of THAT sub-region, so a whole-box comparison against
// it would false-alarm, and a root-only comparison never sees the defect.
//
// The functions below make the domain-scoped check expressible without any
// graph machinery (so they are unit-testable in isolation):
//
// * a sample is COMPARED against a domain only when its own sound
//   pre-activation enclosure puts it STRICTLY inside every split literal (and
//   inside the domain's input box), so membership is proved, not guessed;
// * only the certified LOWER bound of each objective row is compared: on a
//   beta-constrained child the published UPPER endpoint is not a certified
//   bound (see `#violdrop` in `batched_multi.rs`), so comparing it would be a
//   false alarm by construction;
// * a contradiction needs a separation beyond a relative band sized for f32
//   certified arithmetic against an f64 enclosure.
//
// Downgrade-only, exactly like the rest of this module.

/// Relative band for comparing an f32-computed certified bound against a
/// sound f64 point enclosure. Both sides are sound enclosures of the same
/// exact real function, so a correct bound never separates at all; the band
/// only absorbs error that the certified side is KNOWN not to charge (the BN
/// fold, ~2^-24 relative per folded layer — memory
/// `ny-DECIDED-charging-the-BN-fold-error-is-FREE`), with two orders of
/// magnitude of headroom. The defect class this fence exists for is off by
/// the whole range (`±1.18e-38` for a true `[-1, 1]`), so detection power is
/// lost only within `1e-5 * (1 + |bound| + |value|)` of a bound.
pub const DOMAIN_FENCE_REL_BAND: f64 = 1e-5;

/// One ReLU split literal of a branch-and-bound domain, as the fence tests it:
/// the ReLU node the split was taken on, the flat neuron index into that
/// ReLU's pre-activation, and the phase (`active` = pre-activation >= 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainSplitLiteral {
    /// Name of the split ReLU node (the constraint's `node_name`).
    pub relu_node: String,
    /// Flat index into that ReLU's pre-activation.
    pub neuron: usize,
    /// `true` = pre-activation >= 0 on the domain; `false` = <= 0.
    pub active: bool,
}

/// One sample point's SOUND f64 evaluation, reduced to what a domain check
/// needs.
#[derive(Debug, Clone)]
pub struct DomainFenceSample {
    /// The sample point, flat (row-major).
    pub point: Vec<f64>,
    /// Sound enclosure of every objective row `c_i . y` at the point.
    pub objectives: Vec<(f64, f64)>,
    /// Sound enclosure of each ReLU node's PRE-activation at the point, flat,
    /// keyed by the ReLU node's name: `(lower, upper)` slices.
    pub relu_pre_activations: std::collections::HashMap<String, (Vec<f64>, Vec<f64>)>,
}

impl DomainFenceSample {
    /// Whether this sample is PROVABLY inside a domain.
    ///
    /// `input_box` is the domain's input box, or `None` when the domain shares
    /// the very box the samples were drawn from (then membership in it holds
    /// by construction). Every literal must be decided STRICTLY by the
    /// sample's sound pre-activation enclosure (`lower > 0` for an active
    /// literal, `upper < 0` for an inactive one); a literal the enclosure
    /// straddles, an unknown ReLU node, or an out-of-range neuron makes the
    /// answer `false`. So a `true` is a proof of membership, and a `false`
    /// merely means "not usable for this domain".
    pub fn lies_in_domain(
        &self,
        input_box: Option<(&[f64], &[f64])>,
        literals: &[DomainSplitLiteral],
    ) -> bool {
        if let Some((lo, hi)) = input_box {
            if lo.len() != self.point.len() || hi.len() != self.point.len() {
                return false;
            }
            let inside = self
                .point
                .iter()
                .zip(lo.iter().zip(hi.iter()))
                .all(|(p, (l, h))| (*l..=*h).contains(p));
            if !inside {
                return false;
            }
        }
        literals.iter().all(|literal| {
            let Some((lower, upper)) = self.relu_pre_activations.get(&literal.relu_node) else {
                return false;
            };
            let (Some(&l), Some(&u)) = (lower.get(literal.neuron), upper.get(literal.neuron))
            else {
                return false;
            };
            if literal.active {
                l > 0.0
            } else {
                u < 0.0
            }
        })
    }
}

/// Outcome of checking one domain's certified objective lower bounds.
#[derive(Debug, Clone, PartialEq)]
pub enum DomainFenceOutcome {
    /// At least one sample was proved inside the domain and no certified
    /// lower bound exceeded the network's value there beyond the band.
    Consistent {
        /// Samples proved inside the domain.
        samples_in_domain: usize,
        /// (sample, row) pairs actually compared.
        rows_compared: usize,
        /// Smallest `observed_upper - certified_lower` seen (the proof slack).
        min_slack: f64,
        /// Compared pairs whose slack sat inside the band (undetectable).
        blind: usize,
    },
    /// No sample could be proved inside the domain. Carries no information
    /// either way — callers must NOT read it as consistent.
    NoSampleInDomain,
    /// A sample proved inside the domain has a network value strictly below
    /// the domain's certified lower bound beyond the band: the certified bound
    /// does not contain the network, and any verdict resting on it is unsound.
    Contradiction(Box<DomainFenceContradiction>),
}

/// The row, bound and value that falsify a domain's certified lower bound.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainFenceContradiction {
    /// Objective row whose certified lower bound is violated.
    pub row: usize,
    /// The certified lower bound.
    pub certified_lower: f64,
    /// The sound enclosure of the row's value at the sample.
    pub observed: (f64, f64),
    /// The band the separation had to clear.
    pub band: f64,
    /// Index of the sample (into the fence's sample list).
    pub sample: usize,
}

impl DomainFenceContradiction {
    /// A one-line, log-ready statement of the falsification.
    pub fn describe(&self) -> String {
        format!(
            "objective row {} certified lower {:e} exceeds the network's value [{:e}, {:e}] at \
             sample {} proved inside the domain (band {:e})",
            self.row,
            self.certified_lower,
            self.observed.0,
            self.observed.1,
            self.sample,
            self.band
        )
    }
}

/// Check a domain's certified objective LOWER bounds against every sample
/// proved inside the domain.
///
/// Non-finite certified lowers (`-inf` is a vacuous lower bound) and
/// non-finite observations are skipped, as in
/// [`check_certified_bound_contains_samples`]. A row with no sample and no
/// comparison contributes nothing; a domain with no sample inside it returns
/// [`DomainFenceOutcome::NoSampleInDomain`].
pub fn check_domain_lower_bounds_at_samples(
    samples: &[DomainFenceSample],
    input_box: Option<(&[f64], &[f64])>,
    literals: &[DomainSplitLiteral],
    certified_lowers: &[f64],
) -> DomainFenceOutcome {
    let mut samples_in_domain = 0usize;
    let mut rows_compared = 0usize;
    let mut min_slack = f64::INFINITY;
    let mut blind = 0usize;
    for (sample_index, sample) in samples.iter().enumerate() {
        if !sample.lies_in_domain(input_box, literals) {
            continue;
        }
        samples_in_domain += 1;
        for (row, (&certified_lower, &(observed_lower, observed_upper))) in certified_lowers
            .iter()
            .zip(sample.objectives.iter())
            .enumerate()
        {
            if !certified_lower.is_finite() || !observed_upper.is_finite() {
                continue;
            }
            rows_compared += 1;
            let band = DOMAIN_FENCE_REL_BAND * (1.0 + certified_lower.abs() + observed_upper.abs());
            let slack = observed_upper - certified_lower;
            if slack < -band {
                return DomainFenceOutcome::Contradiction(Box::new(DomainFenceContradiction {
                    row,
                    certified_lower,
                    observed: (observed_lower, observed_upper),
                    band,
                    sample: sample_index,
                }));
            }
            if slack <= band {
                blind += 1;
            }
            if slack < min_slack {
                min_slack = slack;
            }
        }
    }
    if samples_in_domain == 0 {
        return DomainFenceOutcome::NoSampleInDomain;
    }
    DomainFenceOutcome::Consistent {
        samples_in_domain,
        rows_compared,
        min_slack,
        blind,
    }
}

/// First coordinate at which a certified two-sided enclosure and a sound
/// observation are separated beyond `rel_band * (1 + |bound| + |value|)`, or
/// `None`. Slices must have equal length (a length mismatch returns `None`;
/// callers count it as not compared). Non-finite entries on either side are
/// skipped: an infinite certified bound contains everything, and a
/// non-finite observation is a different defect this fence does not
/// adjudicate.
pub fn first_band_separation(
    certified_lower: &[f64],
    certified_upper: &[f64],
    observed_lower: &[f64],
    observed_upper: &[f64],
    rel_band: f64,
    sample: usize,
) -> Option<FenceContradiction> {
    let n = certified_lower.len();
    if certified_upper.len() != n || observed_lower.len() != n || observed_upper.len() != n {
        return None;
    }
    for i in 0..n {
        let (cl, ch) = (certified_lower[i], certified_upper[i]);
        let (ol, oh) = (observed_lower[i], observed_upper[i]);
        if !cl.is_finite() || !ch.is_finite() || !ol.is_finite() || !oh.is_finite() {
            continue;
        }
        let above = ol - ch;
        let below = cl - oh;
        let band_above = rel_band * (1.0 + ch.abs() + ol.abs());
        let band_below = rel_band * (1.0 + cl.abs() + oh.abs());
        if above > band_above || below > band_below {
            return Some(FenceContradiction {
                index: i,
                certified: (cl, ch),
                observed: (ol, oh),
                sample,
            });
        }
    }
    None
}

/// Deterministic sample points for a flat box, for the domain fence.
///
/// The first half are the centre, the two extreme vertices and alternating
/// vertices (as [`check_margin_positive_at_samples`] uses); the rest are
/// INTERIOR points at the 1/4 and 3/4 blends of the same stride patterns. A
/// split child's region is typically decided by activation signs that
/// extreme vertices rarely realise together, so interior points raise the
/// chance that a sample is PROVED inside a child. Every point lies in the box
/// (a blend that fails to lands on the lower corner). Callers must pass a
/// finite box with `lo <= hi`.
pub fn domain_fence_sample_points(lo: &[f64], hi: &[f64], want: usize) -> Vec<Vec<f64>> {
    if want == 0 || lo.is_empty() || lo.len() != hi.len() {
        return Vec::new();
    }
    let mut out = margin_sample_points(lo, hi, want.div_ceil(2));
    let mut stride = 1usize;
    let mut flipped = false;
    while out.len() < want {
        out.push(
            lo.iter()
                .zip(hi.iter())
                .enumerate()
                .map(|(i, (l, h))| {
                    let high_side = ((i / stride) % 2 == 1) != flipped;
                    let fraction = if high_side { 0.75 } else { 0.25 };
                    let value = l + fraction * (h - l);
                    if value.is_finite() && (*l..=*h).contains(&value) {
                        value
                    } else {
                        *l
                    }
                })
                .collect(),
        );
        stride = stride.saturating_mul(2);
        if stride > lo.len() {
            if flipped {
                break;
            }
            flipped = true;
            stride = 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn arr(v: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v.to_vec()).unwrap()
    }

    /// A sound evaluator for `f(x) = 2x`, returned as a degenerate interval.
    fn double(p: &ArrayD<f64>) -> Result<Interval64> {
        let v = p.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
        Ok(Interval64::point(arr(&v)))
    }

    #[test]
    fn a_valid_enclosure_passes_and_reports_the_sample_count() {
        // f(x) = 2x over x in [-1, 1] has range [-2, 2].
        let lo = arr(&[-1.0]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[-2.0]),
            upper: arr(&[2.0]),
        };
        match check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double) {
            FenceOutcome::Consistent { samples } => assert!(samples >= 3),
            other => panic!("a valid enclosure must pass, got {other:?}"),
        }
    }

    /// THE REGRESSION THIS MODULE EXISTS FOR: the n-ary Concat shape — a
    /// finite, plausible, NON-CONTAINING bound (`±1.18e-38` for a true range of
    /// `[-1, 1]`). Every runtime check in the tree passed this in August 2026.
    #[test]
    fn the_nary_concat_shape_is_caught() {
        let lo = arr(&[-1.0]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[-1.18e-38]),
            upper: arr(&[1.18e-38]),
        };
        match check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double) {
            FenceOutcome::Contradiction(c) => {
                assert_eq!(c.index, 0);
                assert!(
                    c.observed.0 > c.certified.1 || c.observed.1 < c.certified.0,
                    "the reported contradiction must actually be a strict separation: {c:?}"
                );
            }
            other => panic!("a non-containing bound MUST be caught, got {other:?}"),
        }
    }

    /// An over-wide bound is not a contradiction. The fence detects unsoundness
    /// (a bound too NARROW to contain reality), never mere looseness.
    #[test]
    fn a_loose_but_sound_enclosure_is_not_flagged() {
        let lo = arr(&[-1.0]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[-1e6]),
            upper: arr(&[1e6]),
        };
        assert!(matches!(
            check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double),
            FenceOutcome::Consistent { .. }
        ));
    }

    /// "Nothing contradicted me" and "I never looked" must not be the same
    /// answer — the absence-as-evidence trap this repo keeps hitting.
    #[test]
    fn total_evaluator_refusal_is_inconclusive_not_consistent() {
        let lo = arr(&[-1.0]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[-1e-38]),
            upper: arr(&[1e-38]),
        };
        let out = check_certified_bound_contains_samples(&lo, &hi, &cert, 4, |_| {
            Err(ny_core::NyError::InternalError("refused".into()))
        });
        match out {
            FenceOutcome::Inconclusive { .. } => {}
            other => panic!("total refusal must be Inconclusive, got {other:?}"),
        }
    }

    /// An infinite certified bound contains everything; it is loose, not unsound.
    #[test]
    fn an_infinite_certified_bound_is_never_a_contradiction() {
        let lo = arr(&[-1.0]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[f64::NEG_INFINITY]),
            upper: arr(&[f64::INFINITY]),
        };
        assert!(matches!(
            check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double),
            FenceOutcome::Consistent { .. }
        ));
    }

    /// Determinism: the fence must give the same answer twice on the same input.
    #[test]
    fn the_sample_set_is_deterministic() {
        let lo = arr(&[-1.0, -2.0, -3.0]);
        let hi = arr(&[1.0, 2.0, 3.0]);
        assert_eq!(sample_points(&lo, &hi, 6), sample_points(&lo, &hi, 6));
    }

    /// A multi-coordinate defect is reported against the coordinate that
    /// actually separates, so the log names the real culprit.
    #[test]
    fn the_reported_index_is_the_separating_coordinate() {
        let lo = arr(&[-1.0, -1.0]);
        let hi = arr(&[1.0, 1.0]);
        // Coordinate 0 is contained; coordinate 1 is not.
        let cert = Interval64 {
            lower: arr(&[-2.0, -1e-30]),
            upper: arr(&[2.0, 1e-30]),
        };
        match check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double) {
            FenceOutcome::Contradiction(c) => assert_eq!(c.index, 1),
            other => panic!("expected a contradiction on coordinate 1, got {other:?}"),
        }
    }

    /// THE FALSE-PASS THE GUARD ITSELF HAD. `(-inf, +inf)` satisfies `l <= h`,
    /// so the original guard admitted it; the centre sample was then
    /// `0.5 * (-inf + inf)` = NaN, every comparison against NaN is false, and
    /// the fence returned `Consistent` having compared nothing — on exactly the
    /// shape that signals a failed bound propagation upstream.
    #[test]
    fn an_unbounded_box_is_inconclusive_not_consistent() {
        let lo = arr(&[f64::NEG_INFINITY]);
        let hi = arr(&[f64::INFINITY]);
        let cert = Interval64 {
            lower: arr(&[-1e-38]),
            upper: arr(&[1e-38]),
        };
        match check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double) {
            FenceOutcome::Inconclusive { .. } => {}
            other => panic!("an unbounded box must be Inconclusive, never a pass: {other:?}"),
        }
    }

    /// A single non-finite endpoint is enough to make every sample ill-defined.
    #[test]
    fn a_half_infinite_box_is_inconclusive() {
        let lo = arr(&[0.0]);
        let hi = arr(&[f64::INFINITY]);
        let cert = Interval64 {
            lower: arr(&[-1e-38]),
            upper: arr(&[1e-38]),
        };
        assert!(matches!(
            check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double),
            FenceOutcome::Inconclusive { .. }
        ));
    }

    /// A NaN endpoint must also refuse rather than silently compare nothing.
    #[test]
    fn a_nan_endpoint_is_inconclusive() {
        let lo = arr(&[f64::NAN]);
        let hi = arr(&[1.0]);
        let cert = Interval64 {
            lower: arr(&[-2.0]),
            upper: arr(&[2.0]),
        };
        assert!(matches!(
            check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double),
            FenceOutcome::Inconclusive { .. }
        ));
    }

    /// An empty or inverted box carries no information and must not be read as
    /// a pass.
    #[test]
    fn an_inverted_box_is_inconclusive() {
        let lo = arr(&[1.0]);
        let hi = arr(&[-1.0]);
        let cert = Interval64 {
            lower: arr(&[-2.0]),
            upper: arr(&[2.0]),
        };
        assert!(matches!(
            check_certified_bound_contains_samples(&lo, &hi, &cert, 4, double),
            FenceOutcome::Inconclusive { .. }
        ));
    }

    // ---------------- margin-positivity fence ----------------

    /// A real violation, orders of magnitude outside the band, is caught.
    #[test]
    fn a_negative_margin_refutes_the_unsat() {
        let out =
            check_margin_positive_at_samples(&[-1.0], &[1.0], &[3], 8, |_p| Ok((vec![-0.5], 1.0)));
        match out {
            MarginFenceOutcome::Contradiction(c) => {
                assert_eq!(
                    c.class, 3,
                    "must name the adversarial class, not the row index"
                );
                assert!(c.margin < 0.0);
            }
            other => panic!("a -0.5 margin must refute an unsat: {other:?}"),
        }
    }

    /// THE FALSE-ALARM GUARD. The point side is exact f64 while the certified
    /// side charges BatchNorm-fold error and may be f32, and `closed()` has no
    /// epsilon — so a certified margin of ~1e-12 is legitimate. A tiny negative
    /// sample inside the band must NOT refuse a correct proof: that would cost
    /// a real +10 per affected row.
    #[test]
    fn a_margin_inside_the_float_band_does_not_refute() {
        let out = check_margin_positive_at_samples(&[-1.0], &[1.0], &[3], 8, |_p| {
            Ok((vec![-1e-12], 1.0))
        });
        match out {
            MarginFenceOutcome::Consistent {
                blind_samples,
                min_margin,
                ..
            } => {
                assert!(
                    blind_samples > 0,
                    "an in-band sample must be COUNTED, not hidden"
                );
                assert!(
                    min_margin < 0.0,
                    "the observed minimum is reported honestly"
                );
            }
            other => panic!("an in-band margin must not refute: {other:?}"),
        }
    }

    /// The band scales with the LOGIT MAGNITUDE the evaluator reports, so it
    /// stays correct on networks whose logits are large — where the forward
    /// pass's own f64 error is correspondingly larger.
    #[test]
    fn the_band_scales_with_magnitude() {
        // scale = 1e6 -> band ~= 1e-9 * (1 + 1e6) ~= 1e-3, so -1e-4 is INSIDE it
        // and must not refute.
        let inside = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0], 4, |_p| {
            Ok((vec![-1.0e-4_f64], 1.0e6))
        });
        assert!(
            matches!(inside, MarginFenceOutcome::Consistent { .. }),
            "a large logit scale must widen the band: {inside:?}"
        );
        // Same margin, ordinary logit scale -> band ~= 2e-9, so it REFUTES.
        let outside = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0], 4, |_p| {
            Ok((vec![-1.0e-4_f64], 1.0))
        });
        assert!(
            matches!(outside, MarginFenceOutcome::Contradiction(_)),
            "at ordinary scale the same margin must refute: {outside:?}"
        );
    }

    /// A non-finite or negative scale is a broken evaluator, not a pass.
    #[test]
    fn a_bad_logit_scale_is_inconclusive() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let out = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0], 4, |_p| {
                Ok((vec![1.0], bad))
            });
            assert!(
                matches!(out, MarginFenceOutcome::Inconclusive { .. }),
                "scale {bad} must be refused: {out:?}"
            );
        }
    }

    /// A comfortably positive margin is a clean pass with nothing blind.
    #[test]
    fn a_positive_margin_is_a_clean_pass() {
        match check_margin_positive_at_samples(&[-1.0], &[1.0], &[0, 1], 5, |_p| {
            Ok((vec![2.0, 3.0], 1.0))
        }) {
            MarginFenceOutcome::Consistent {
                samples,
                min_margin,
                blind_samples,
            } => {
                assert!(samples > 0);
                assert_eq!(blind_samples, 0, "nothing was undetectable here");
                assert!((min_margin - 2.0).abs() < 1e-12);
            }
            other => panic!("expected a clean pass: {other:?}"),
        }
    }

    /// The 2026-08-25 n-ary Concat shape: a bound wrong by ~38 orders of
    /// magnitude is nowhere near the band.
    #[test]
    fn the_nary_concat_magnitude_is_far_outside_the_band() {
        let out =
            check_margin_positive_at_samples(&[-1.0], &[1.0], &[7], 8, |_p| Ok((vec![-1.0], 1.0)));
        assert!(
            matches!(out, MarginFenceOutcome::Contradiction(_)),
            "the class this fence exists for must be caught: {out:?}"
        );
    }

    /// An evaluator that refuses everything proves nothing and must never read
    /// as a pass.
    #[test]
    fn margin_total_refusal_is_inconclusive() {
        let out = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0], 8, |_p| {
            Err(ny_core::NyError::InternalError("refused".into()))
        });
        assert!(
            matches!(out, MarginFenceOutcome::Inconclusive { .. }),
            "{out:?}"
        );
    }

    /// An unbounded box makes every sample ill-defined — the same false-PASS
    /// this module already had once.
    #[test]
    fn margin_unbounded_box_is_inconclusive() {
        let out = check_margin_positive_at_samples(
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &[0],
            8,
            |_p| Ok((vec![1.0], 1.0)),
        );
        assert!(
            matches!(out, MarginFenceOutcome::Inconclusive { .. }),
            "{out:?}"
        );
    }

    /// A shape disagreement between the evaluator and the class list is a bug,
    /// not a pass.
    #[test]
    fn margin_arity_mismatch_is_inconclusive() {
        let out = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0, 1], 4, |_p| {
            Ok((vec![1.0], 1.0))
        });
        assert!(
            matches!(out, MarginFenceOutcome::Inconclusive { .. }),
            "{out:?}"
        );
    }

    /// A NaN margin must refuse rather than compare nothing (every comparison
    /// against NaN is false, which would otherwise read as a pass).
    #[test]
    fn margin_nan_is_inconclusive() {
        let out = check_margin_positive_at_samples(&[-1.0], &[1.0], &[0], 4, |_p| {
            Ok((vec![f64::NAN], 1.0))
        });
        assert!(
            matches!(out, MarginFenceOutcome::Inconclusive { .. }),
            "{out:?}"
        );
    }

    /// Same input, same answer — twice.
    #[test]
    fn margin_sampling_is_deterministic() {
        let lo = vec![-1.0, -2.0, -3.0];
        let hi = vec![1.0, 2.0, 3.0];
        let a = margin_sample_points(&lo, &hi, 6);
        let b = margin_sample_points(&lo, &hi, 6);
        assert_eq!(a, b);
        assert_eq!(a[0], vec![0.0, 0.0, 0.0], "first sample is the centre");
        assert_eq!(a[1], lo);
        assert_eq!(a[2], hi);
    }

    // ---------------- #vit-bank-gate: graph-BaB domain fence ----------------

    /// A sample whose ReLU `r` pre-activation is `pre` (one neuron) and whose
    /// two objective rows evaluate to `objectives`.
    fn domain_sample(point: f64, pre: (f64, f64), objectives: &[(f64, f64)]) -> DomainFenceSample {
        let mut relu_pre_activations = std::collections::HashMap::new();
        relu_pre_activations.insert("r".to_string(), (vec![pre.0], vec![pre.1]));
        DomainFenceSample {
            point: vec![point],
            objectives: objectives.to_vec(),
            relu_pre_activations,
        }
    }

    fn literal(active: bool) -> DomainSplitLiteral {
        DomainSplitLiteral {
            relu_node: "r".to_string(),
            neuron: 0,
            active,
        }
    }

    /// A child whose certified lower bounds sit below the network's value at
    /// every sample proved inside it is consistent, and the report counts
    /// what was actually compared.
    #[test]
    fn vit_bank_gate_domain_fence_passes_a_sound_child_bound() {
        let samples = vec![
            domain_sample(0.5, (0.25, 0.25), &[(1.0, 1.0), (2.0, 2.0)]),
            domain_sample(-0.5, (-0.25, -0.25), &[(-3.0, -3.0), (0.0, 0.0)]),
        ];
        let outcome =
            check_domain_lower_bounds_at_samples(&samples, None, &[literal(true)], &[0.5, 1.5]);
        match outcome {
            DomainFenceOutcome::Consistent {
                samples_in_domain,
                rows_compared,
                min_slack,
                blind,
            } => {
                assert_eq!(
                    samples_in_domain, 1,
                    "only the active-side sample is in the child"
                );
                assert_eq!(rows_compared, 2);
                assert!((min_slack - 0.5).abs() < 1e-12);
                assert_eq!(blind, 0);
            }
            other => panic!("a sound child bound must pass: {other:?}"),
        }
    }

    /// THE CLASS THIS EXISTS FOR, on a CHILD: a certified lower bound far above
    /// the network's value at a sample proved inside the child.
    #[test]
    fn vit_bank_gate_domain_fence_catches_a_non_containing_child_lower_bound() {
        let samples = vec![domain_sample(
            0.5,
            (0.25, 0.25),
            &[(1.0, 1.0), (-1.0, -1.0)],
        )];
        match check_domain_lower_bounds_at_samples(&samples, None, &[literal(true)], &[0.5, 0.75]) {
            DomainFenceOutcome::Contradiction(c) => {
                assert_eq!(c.row, 1, "must name the row that separates");
                assert_eq!(c.sample, 0);
                assert!(c.certified_lower > c.observed.1 + c.band);
            }
            other => panic!("a non-containing child bound must be caught: {other:?}"),
        }
    }

    /// THE FALSE-ALARM GUARD: a child's bound only covers its own sub-region.
    /// A sample on the OTHER side of the split, however low its value, must
    /// never be compared.
    #[test]
    fn vit_bank_gate_domain_fence_never_compares_a_sample_outside_the_split() {
        let samples = vec![domain_sample(0.5, (0.25, 0.25), &[(-100.0, -100.0)])];
        assert_eq!(
            check_domain_lower_bounds_at_samples(&samples, None, &[literal(false)], &[0.0]),
            DomainFenceOutcome::NoSampleInDomain
        );
    }

    /// Membership must be PROVED: a pre-activation enclosure that straddles 0
    /// decides neither phase, so the sample is unusable for either child.
    #[test]
    fn vit_bank_gate_domain_fence_a_straddling_pre_activation_is_not_membership() {
        let samples = vec![domain_sample(0.0, (-1e-9, 1e-9), &[(-100.0, -100.0)])];
        for active in [true, false] {
            assert_eq!(
                check_domain_lower_bounds_at_samples(&samples, None, &[literal(active)], &[0.0]),
                DomainFenceOutcome::NoSampleInDomain,
                "active={active}"
            );
        }
        // An unknown ReLU node is likewise not membership.
        let unknown = DomainSplitLiteral {
            relu_node: "missing".to_string(),
            neuron: 0,
            active: true,
        };
        assert_eq!(
            check_domain_lower_bounds_at_samples(&samples, None, &[unknown], &[0.0]),
            DomainFenceOutcome::NoSampleInDomain
        );
    }

    /// A domain whose INPUT box excludes the sample never compares it.
    #[test]
    fn vit_bank_gate_domain_fence_enforces_the_domain_input_box() {
        let samples = vec![domain_sample(0.5, (0.25, 0.25), &[(-100.0, -100.0)])];
        let lo: &[f64] = &[-1.0];
        let hi: &[f64] = &[0.25];
        assert_eq!(
            check_domain_lower_bounds_at_samples(
                &samples,
                Some((lo, hi)),
                &[literal(true)],
                &[0.0]
            ),
            DomainFenceOutcome::NoSampleInDomain
        );
    }

    /// A separation inside the f32 band is counted blind, never a
    /// contradiction; `-inf` lowers are vacuous and skipped.
    #[test]
    fn vit_bank_gate_domain_fence_band_and_vacuous_rows() {
        let samples = vec![domain_sample(0.5, (0.25, 0.25), &[(1.0, 1.0), (1.0, 1.0)])];
        let inside_band = 1.0 + 0.5 * DOMAIN_FENCE_REL_BAND;
        match check_domain_lower_bounds_at_samples(
            &samples,
            None,
            &[literal(true)],
            &[inside_band, f64::NEG_INFINITY],
        ) {
            DomainFenceOutcome::Consistent {
                rows_compared,
                blind,
                ..
            } => {
                assert_eq!(rows_compared, 1, "the -inf row is vacuous and skipped");
                assert_eq!(blind, 1);
            }
            other => panic!("an in-band slack must not refute: {other:?}"),
        }
    }

    /// Two-sided node separation: the Concat-defect shape separates, a loose
    /// but sound enclosure does not, and an in-band miss does not.
    #[test]
    fn vit_bank_gate_first_band_separation_is_two_sided_and_banded() {
        let hit = first_band_separation(
            &[-2.0, -1.18e-38],
            &[2.0, 1.18e-38],
            &[0.5, 0.75],
            &[0.5, 0.75],
            DOMAIN_FENCE_REL_BAND,
            3,
        )
        .expect("the n-ary Concat shape must separate");
        assert_eq!(hit.index, 1);
        assert_eq!(hit.sample, 3);
        assert!(
            first_band_separation(&[-1e6], &[1e6], &[0.5], &[0.5], DOMAIN_FENCE_REL_BAND, 0)
                .is_none()
        );
        let just_above = 1.0 + 0.5 * DOMAIN_FENCE_REL_BAND;
        assert!(first_band_separation(
            &[0.0],
            &[1.0],
            &[just_above],
            &[just_above],
            DOMAIN_FENCE_REL_BAND,
            0
        )
        .is_none());
        let below = first_band_separation(&[0.0], &[1.0], &[-0.5], &[-0.5], 1e-5, 0)
            .expect("a value below the certified lower bound separates");
        assert_eq!(below.index, 0);
        assert!(first_band_separation(&[0.0], &[1.0], &[0.5, 0.5], &[0.5], 1e-5, 0).is_none());
    }

    /// Deterministic, the requested count on a wide box, every point inside
    /// the box, and the second half strictly interior on a non-degenerate box.
    #[test]
    fn vit_bank_gate_domain_fence_samples_are_deterministic_and_in_the_box() {
        let lo: Vec<f64> = (0..64_i32).map(|i| -1.0 - f64::from(i % 3)).collect();
        let hi: Vec<f64> = (0..64_i32).map(|i| 1.0 + f64::from(i % 5)).collect();
        let a = domain_fence_sample_points(&lo, &hi, 16);
        let b = domain_fence_sample_points(&lo, &hi, 16);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        for point in &a {
            assert_eq!(point.len(), lo.len());
            for (k, value) in point.iter().enumerate() {
                assert!((lo[k]..=hi[k]).contains(value));
            }
        }
        for point in &a[8..] {
            for (k, value) in point.iter().enumerate() {
                assert!(
                    lo[k] < *value && *value < hi[k],
                    "blend points are interior"
                );
            }
        }
        assert_eq!((domain_fence_sample_points(&lo, &hi[..3], 4)).len(), 0);
        assert_eq!((domain_fence_sample_points(&[], &[], 4)).len(), 0);
    }
}
