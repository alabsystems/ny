// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-bank-gate: the containment fence, wired into the multi-objective graph
//! BaB (typed key `bab.mo_containment_fence` -> `BetaCrownConfig::
//! mo_containment_fence`, default OFF).
//!
//! # What it checks
//!
//! `crate::containment_fence` had zero production callers. This driver gives
//! it two, both on the lane vit_2023 takes (disjunctive multi-objective graph
//! BaB, shared/batched executor):
//!
//! 1. **Root.** At arm time, `MO_FENCE_SAMPLES` deterministic points of the
//!    root input box are evaluated through the SOUND f64 fence walk
//!    (`GraphNetwork::propagate_f64_fence_nodes`), and EVERY root node bound
//!    the lane will build children from is compared two-sided against the
//!    point enclosure of that node — including the CLS+pos `Concat` path — as
//!    are the root objective LOWER bounds.
//! 2. **Children.** Every child a wave returns is compared, before the queue
//!    fold, at every sample PROVED inside it (input box plus every ReLU split
//!    literal decided strictly by the sample's own sound pre-activation), on
//!    its certified objective LOWER bounds only — the upper endpoint of a
//!    beta-constrained child is not certified (`#violdrop`). That is the
//!    stage the 2026-08-25 N-ary Concat defect lived in
//!    (`batched/backward_core.rs`, `constraints/backward/dispatch.rs`), which a
//!    root-only fence never sees.
//! 3. **Root-terminal `Verified`** (#vit-margin-alpha-complex, P8). When the
//!    root closes every objective before any BaB, there is no domain to arm
//!    against. [`fence_root_terminal_verified`] compares, at the same samples,
//!    the published output enclosure two-sided AND the certified objective
//!    LOWER bounds the verdict actually rests on (recorded by the closing root
//!    site in `GraphBabLifecycle::root_verified_objective_lowers`), and — when
//!    `bab.mo_root_ort_adjudication` is armed — the same lowers against an
//!    INDEPENDENT forward of the original model (ONNX Runtime, via
//!    [`RootAdjudicationOracle`]). It FAILS CLOSED: an authority it could not
//!    check (lowers not recorded, no evaluable sample, an armed ORT
//!    adjudication with no oracle or no adjudicated sample) returns `Unknown`.
//! 4. **Emptiness claims** (P8). A region closed by `InfeasibleDomain` or an
//!    empty half-space carries no bound, so no sample can check it. Every such
//!    site on the multi-objective lanes counts itself
//!    ([`note_emptiness_claim`]), and [`apply_emptiness_gate`] turns a fenced
//!    run's `Verified` that rested on ANY such claim into `Unknown`. The
//!    2026-08-30 alarm's silent double-`Infeasible` fold (`Children([])` then
//!    `Verified`) is refused on EVERY run, fenced or not, by the queue fold
//!    itself (#spine-legacy-lanes, the M2c certified-infeasibility rule); the
//!    fence additionally refuses the single-sided kinds (sibling, construct,
//!    NoUnstable-empty, sequential, multi-depth, per-disjunct).
//!
//! Not covered (stated, not hidden): the sequential multi-objective lane
//! (conjunctive properties, or no GEMM engine / `batch_size == 1`) folds its
//! children without passing them here, so the summary line reports
//! `children_checked=0` on that lane (its emptiness claims ARE counted).
//! Also outside the fence (review, 2026-09-24): `NoUnstable` closes that are
//! bound proofs, whose result carries no bound; children's intermediate node
//! bounds; and any OTHER lane the caller runs after this one returns `Unknown`
//! (the refusal is lane-local, not a run-level veto). The kFSB lookahead's
//! simulated empty sides are NOT scoring-only (P8 review): a committed kFSB
//! cover omits an `Infeasible` side, and a parent-cover certificate may rest on
//! one, so both are counted as `construct` emptiness claims.
//!
//! # What it does on a contradiction
//!
//! Downgrade only: the verifier returns `Unknown` with the contradiction as the
//! reason, before the offending wave is folded. It never tightens a bound,
//! closes a domain, or publishes anything. A false alarm costs one row's +10;
//! a missed false unsat costs -150 and the moat.
//!
//! # When it refuses to arm
//!
//! When any mechanism is active that makes a domain's certified bound valid
//! only on a PROPERTY-CONDITIONED sub-region (cut pool, conflict-clause store,
//! interm-domain clipping, relaxed clipping), or on a conjunctive run. Those
//! bounds are allowed to exclude safe points, so comparing them at arbitrary
//! samples would false-alarm by construction. The refusal is printed, never
//! silent.
//!
//! # Observable
//!
//! `[mo-containment-fence] armed=1 ...` (or `armed=0 reason=...`) once per
//! BaB, `[mo-containment-fence] CONTRADICTION ...` on a refusal, and one
//! `[mo-containment-fence] summary ...` line when the BaB returns (printed on
//! drop, so every return path emits it). A measurement that cannot find the
//! summary line did not run the fence. A root-terminal `Verified` prints one
//! `[mo-containment-fence] root-verified outcome=… objective_lowers=… ort=…`
//! line instead, and every fenced verification prints one
//! `[mo-containment-fence] emptiness-gate verdict=… claims=N … outcome=…`
//! line on return. A vit bank requires `root-verified outcome=consistent
//! objective_lowers=fenced` (plus `ort=adjudicated` when that key is armed) on
//! a root-terminal row, and `emptiness-gate … claims=0 … outcome=pass` on
//! every row.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};

use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::beta_crown::root_adjudication::RootAdjudicationOracle;
use crate::containment_fence::{
    check_domain_lower_bounds_at_samples, domain_fence_sample_points, first_band_separation,
    DomainFenceOutcome, DomainFenceSample, DomainSplitLiteral, DOMAIN_FENCE_REL_BAND,
};
use crate::{BoundedTensor, GraphNetwork, Interval64, Layer, NETWORK_INPUT};

/// Sample points drawn from the root input box. Each costs one sound f64 walk
/// at arm time; child checks reuse them and cost no network evaluation.
pub(super) const MO_FENCE_SAMPLES: usize = 16;

/// Wall-clock cap on the arm-time walks. A fence that cannot evaluate a single
/// sample inside it disarms (and says so) rather than eat the BaB budget.
const MO_FENCE_ARM_CAP: Duration = Duration::from_secs(5);

/// Why the fence refused to arm on this run, or `None` when it may arm.
///
/// Every reason names a mechanism whose bounds are valid only on a
/// property-conditioned sub-region, where an arbitrary sample need not be
/// contained.
pub(super) fn mo_containment_fence_refusal(
    conjunctive: bool,
    cuts_enabled: bool,
    clause_store_enabled: bool,
    clip_interm_domain: bool,
    relaxed_clip: bool,
) -> Option<&'static str> {
    if conjunctive {
        Some("conjunctive")
    } else if cuts_enabled {
        Some("cuts")
    } else if clause_store_enabled {
        Some("conflict-clauses")
    } else if clip_interm_domain {
        Some("clip-interm-domain")
    } else if relaxed_clip {
        Some("relaxed-clip")
    } else {
        None
    }
}

/// Flatten a certified f32 tensor to exact f64 `(lower, upper)` vectors in
/// logical (row-major) order — the order the f64 walk's arrays iterate in.
fn flat_f64(bounds: &BoundedTensor) -> (Vec<f64>, Vec<f64>) {
    (
        bounds.lower().iter().map(|v| f64::from(*v)).collect(),
        bounds.upper().iter().map(|v| f64::from(*v)).collect(),
    )
}

/// Whether two tensor shapes describe the same row-major layout once leading
/// singleton (batch) axes are ignored. Only such pairs are compared
/// element-by-element: equal element counts alone could pair a transposed
/// layout coordinate-for-coordinate and manufacture a false alarm.
fn same_layout(a: &[usize], b: &[usize]) -> bool {
    let strip = |shape: &[usize]| -> Vec<usize> {
        shape.iter().copied().skip_while(|&dim| dim == 1).collect()
    };
    strip(a) == strip(b)
}

/// Sound enclosure of `c . y` for one objective row against the output's
/// point enclosure: every product and partial sum rounded outward.
fn objective_enclosure(row: &[f32], y_lower: &[f64], y_upper: &[f64]) -> Option<(f64, f64)> {
    if row.len() != y_lower.len() || row.len() != y_upper.len() {
        return None;
    }
    let mut lower = 0.0_f64;
    let mut upper = 0.0_f64;
    for ((&c, &yl), &yu) in row.iter().zip(y_lower.iter()).zip(y_upper.iter()) {
        let c = f64::from(c);
        if !c.is_finite() || !yl.is_finite() || !yu.is_finite() {
            return None;
        }
        let a = c * yl;
        let b = c * yu;
        let (product_lower, product_upper) = if a <= b { (a, b) } else { (b, a) };
        lower = (lower + product_lower.next_down()).next_down();
        upper = (upper + product_upper.next_up()).next_up();
    }
    Some((lower, upper))
}

/// Relative band for comparing a certified objective lower bound against an
/// objective value computed from an INDEPENDENT forward (ONNX Runtime, f32).
///
/// Unlike the f64 fence walk, an ORT output is not a sound enclosure of the
/// real function: it carries ORT's own f32 rounding. Ten times the f64-walk
/// band ([`DOMAIN_FENCE_REL_BAND`]) absorbs that with margin to spare, while
/// the defect class the adjudication exists for (a graph ny built differently
/// from the model) is off by far more than `1e-4 * (1 + |bound| + |value|)`.
/// A separation inside the band is counted `blind`, never a pass it did not
/// earn.
pub(super) const ORT_ADJUDICATION_REL_BAND: f64 = 1e-4;

/// What the root-terminal fence concluded (#vit-margin-alpha-complex, P8).
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct RootTerminalFence {
    /// A certified bound that the network (f64 walk) or the independent
    /// forward (ORT) violates beyond its band.
    pub(super) contradiction: Option<String>,
    /// The verdict's AUTHORITY could not be checked at all. FAIL-CLOSED: the
    /// caller must not publish the verdict as fenced.
    pub(super) unfenced: Option<String>,
}

impl RootTerminalFence {
    /// The `Unknown` reason the caller must publish instead of `Verified`, or
    /// `None` when the root-terminal verdict was fenced and consistent.
    pub(super) fn downgrade_reason(&self) -> Option<String> {
        if let Some(contradiction) = &self.contradiction {
            return Some(format!("containment fence: {contradiction}"));
        }
        self.unfenced.as_ref().map(|why| {
            format!("containment fence: root-terminal verdict authority unfenced ({why})")
        })
    }
}

/// #vit-bank-gate + #vit-margin-alpha-complex (P8): fence a ROOT-terminal
/// `Verified` — the root closed every objective before any BaB, so no domain
/// exists to arm against. At the fence's deterministic samples of `input`:
///
/// 1. the published OUTPUT enclosure is compared two-sided against the sound
///    f64 walk (the 2026-08-25 `Concat` shape);
/// 2. the certified objective LOWER bounds the verdict actually rests on
///    (`objective_lowers`, recorded by the root site that closed —
///    `GraphBabLifecycle::root_verified_objective_lowers`) are compared
///    against the walk's sound objective enclosures — the verdict's
///    AUTHORITY, which the output-enclosure check alone left
///    `objective_lowers=UNFENCED`;
/// 3. when `ort_requested`, the same lowers are compared against the
///    objective values of an INDEPENDENT forward of the original model
///    (`oracle`, ONNX Runtime), which catches a defect in the graph ny built —
///    one the f64 walk shares with the bounds and so can never see.
///
/// FAIL-CLOSED: `unfenced` is set (the caller downgrades) when the lowers
/// were not recorded or do not match the objectives, when no sample could be
/// evaluated or no row compared, or when ORT adjudication was requested but
/// no oracle was attached or no sample could be adjudicated. A fence that
/// refused to arm (`refusal`: property-conditioned bounds) prints
/// `armed=0 reason=…` and returns an empty report, as before.
///
/// Prints `[mo-containment-fence] root-verified CONTRADICTION …` on a
/// separation, then EXACTLY ONE summary line (fixed schema):
/// `[mo-containment-fence] root-verified outcome=<consistent|CONTRADICTION|UNFENCED>
/// samples=N walks_failed=N output_enclosure=<checked|UNCOMPARED|absent> objective_lowers=<fenced|UNFENCED>
/// objective_rows=N min_objective_slack=E blind=N ort=<off|adjudicated|UNADJUDICATED>
/// ort_oracle=<name|none> ort_samples=N ort_failed=N ort_min_slack=E ort_blind=N
/// reason=<token|none> secs=S`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn fence_root_terminal_verified(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_bounds: Option<&BoundedTensor>,
    objectives: &[Vec<f32>],
    objective_lowers: Option<&[f32]>,
    ort_requested: bool,
    oracle: Option<&dyn RootAdjudicationOracle>,
    refusal: Option<&'static str>,
) -> RootTerminalFence {
    if let Some(reason) = refusal {
        eprintln!("[mo-containment-fence] root-verified armed=0 reason={reason}");
        return RootTerminalFence::default();
    }
    let started = Instant::now();
    let mut report = RootTerminalFence::default();
    let (lower, upper) = flat_f64(input);
    let box_ok = !lower.is_empty()
        && lower.iter().chain(upper.iter()).all(|v| v.is_finite())
        && lower.iter().zip(upper.iter()).all(|(l, h)| l <= h);

    // The authority, validated before any work: exactly one finite-or-not
    // certified lower per objective row.
    let lowers: Option<Vec<f64>> = objective_lowers
        .filter(|lowers| !objectives.is_empty() && lowers.len() == objectives.len())
        .map(|lowers| lowers.iter().map(|v| f64::from(*v)).collect());
    let ort_oracle = if ort_requested { oracle } else { None };

    let mut compared = 0usize;
    let mut walks_failed = 0usize;
    // Samples at which the published OUTPUT enclosure was actually compared
    // (a layout mismatch skips the comparison), so `output_enclosure=checked`
    // is printed only when it was earned.
    let mut output_compared = 0usize;
    let mut samples: Vec<DomainFenceSample> = Vec::new();
    let mut ort_samples = 0usize;
    let mut ort_failed = 0usize;
    let mut ort_blind = 0usize;
    let mut ort_min_slack = f64::INFINITY;
    let mut first_ort_error: Option<String> = None;
    if box_ok {
        let certified_output = output_bounds.map(flat_f64);
        let input_shape = input.shape().to_vec();
        'samples: for (sample, point) in
            domain_fence_sample_points(&lower, &upper, MO_FENCE_SAMPLES)
                .into_iter()
                .enumerate()
        {
            if started.elapsed() > MO_FENCE_ARM_CAP {
                break;
            }
            // (3) The INDEPENDENT forward first: it needs only the point.
            if let (Some(oracle), Some(lowers)) = (ort_oracle, lowers.as_ref()) {
                let point32: Vec<f32> = point.iter().map(|v| *v as f32).collect();
                match oracle.evaluate(&point32) {
                    Ok(y) if objectives.iter().all(|row| row.len() == y.len()) => {
                        let y64: Vec<f64> = y.iter().map(|v| f64::from(*v)).collect();
                        let mut usable = false;
                        for (row_index, (row, &certified_lower)) in
                            objectives.iter().zip(lowers.iter()).enumerate()
                        {
                            let Some((_, observed_upper)) = objective_enclosure(row, &y64, &y64)
                            else {
                                continue;
                            };
                            if !certified_lower.is_finite() {
                                continue;
                            }
                            usable = true;
                            let band = ORT_ADJUDICATION_REL_BAND
                                * (1.0 + certified_lower.abs() + observed_upper.abs());
                            let slack = observed_upper - certified_lower;
                            if slack < -band {
                                let what = format!(
                                    "root objectives vs ONNX Runtime: objective row {row_index} \
                                     certified lower {certified_lower:e} exceeds the independent \
                                     forward's value {observed_upper:e} at sample {sample} \
                                     (band {band:e})"
                                );
                                eprintln!(
                                    "[mo-containment-fence] root-verified CONTRADICTION {what}"
                                );
                                report.contradiction = Some(what);
                                break 'samples;
                            }
                            if slack <= band {
                                ort_blind += 1;
                            }
                            ort_min_slack = ort_min_slack.min(slack);
                        }
                        if usable {
                            ort_samples += 1;
                        } else {
                            ort_failed += 1;
                        }
                    }
                    Ok(y) => {
                        ort_failed += 1;
                        if first_ort_error.is_none() {
                            first_ort_error = Some(format!(
                                "output width {} does not match the objective rows",
                                y.len()
                            ));
                        }
                    }
                    Err(error) => {
                        ort_failed += 1;
                        if first_ort_error.is_none() {
                            first_ort_error = Some(error);
                        }
                    }
                }
            }

            // (1) + (2): the sound f64 walk.
            let Ok(array) = ArrayD::from_shape_vec(IxDyn(&input_shape), point.clone()) else {
                walks_failed += 1;
                continue;
            };
            let Ok(walk) = graph.propagate_f64_fence_nodes(&Interval64::point(array)) else {
                walks_failed += 1;
                continue;
            };
            let Some(observed) = walk.get(graph.output_name()) else {
                walks_failed += 1;
                continue;
            };
            let observed_lower: Vec<f64> = observed.lower.iter().copied().collect();
            let observed_upper: Vec<f64> = observed.upper.iter().copied().collect();
            compared += 1;
            if let (Some(bounds), Some((certified_lower, certified_upper))) =
                (output_bounds, certified_output.as_ref())
            {
                if observed_lower.len() == certified_lower.len()
                    && same_layout(bounds.shape(), observed.lower.shape())
                {
                    output_compared += 1;
                    if let Some(separation) = first_band_separation(
                        certified_lower,
                        certified_upper,
                        &observed_lower,
                        &observed_upper,
                        DOMAIN_FENCE_REL_BAND,
                        sample,
                    ) {
                        let what = format!("root output: {}", separation.describe());
                        eprintln!("[mo-containment-fence] root-verified CONTRADICTION {what}");
                        report.contradiction = Some(what);
                        break;
                    }
                }
            }
            samples.push(DomainFenceSample {
                point,
                objectives: objectives
                    .iter()
                    .map(|row| {
                        objective_enclosure(row, &observed_lower, &observed_upper)
                            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
                    })
                    .collect(),
                relu_pre_activations: HashMap::new(),
            });
        }
    }

    // (2) The verdict's authority against every evaluated sample.
    let mut objective_rows = 0usize;
    let mut min_objective_slack = f64::INFINITY;
    let mut blind = 0usize;
    if report.contradiction.is_none() {
        if let Some(lowers) = lowers.as_ref() {
            match check_domain_lower_bounds_at_samples(&samples, None, &[], lowers) {
                DomainFenceOutcome::Contradiction(contradiction) => {
                    let what = format!("root objectives: {}", contradiction.describe());
                    eprintln!("[mo-containment-fence] root-verified CONTRADICTION {what}");
                    report.contradiction = Some(what);
                }
                DomainFenceOutcome::Consistent {
                    rows_compared,
                    min_slack,
                    blind: blind_pairs,
                    ..
                } => {
                    objective_rows = rows_compared;
                    min_objective_slack = min_slack;
                    blind = blind_pairs;
                }
                DomainFenceOutcome::NoSampleInDomain => {}
            }
        }
    }

    // FAIL-CLOSED decisions, in order of precedence.
    if report.contradiction.is_none() {
        report.unfenced = if !box_ok {
            Some("root-box-empty-or-non-finite".to_string())
        } else if objective_lowers.is_none() {
            Some("objective-lowers-not-recorded".to_string())
        } else if lowers.is_none() {
            Some("objective-lowers-shape-mismatch".to_string())
        } else if compared == 0 {
            Some(format!("no-evaluable-sample walks_failed={walks_failed}"))
        } else if objective_rows == 0 {
            Some("no-objective-row-compared".to_string())
        } else if ort_requested && oracle.is_none() {
            Some("ort-requested-but-no-oracle-attached".to_string())
        } else if ort_requested && ort_samples == 0 {
            Some(format!(
                "ort-no-sample-adjudicated first_error={}",
                sanitize_token(first_ort_error.as_deref().unwrap_or("none"))
            ))
        } else {
            None
        };
    }

    let outcome = if report.contradiction.is_some() {
        "CONTRADICTION"
    } else if report.unfenced.is_some() {
        "UNFENCED"
    } else {
        "consistent"
    };
    let ort_state = if !ort_requested {
        "off"
    } else if ort_samples > 0 && report.unfenced.is_none() {
        "adjudicated"
    } else {
        "UNADJUDICATED"
    };
    eprintln!(
        "[mo-containment-fence] root-verified outcome={outcome} samples={compared} \
         walks_failed={walks_failed} output_enclosure={} objective_lowers={} objective_rows={} \
         min_objective_slack={:e} blind={} ort={ort_state} ort_oracle={} ort_samples={} \
         ort_failed={} ort_min_slack={:e} ort_blind={} reason={} secs={:.3}",
        match (output_bounds.is_some(), output_compared > 0) {
            (false, _) => "absent",
            (true, true) => "checked",
            (true, false) => "UNCOMPARED",
        },
        if objective_rows > 0 {
            "fenced"
        } else {
            "UNFENCED"
        },
        objective_rows,
        min_objective_slack,
        blind,
        oracle
            .filter(|_| ort_requested)
            .map_or("none", |oracle| oracle.name()),
        ort_samples,
        ort_failed,
        ort_min_slack,
        ort_blind,
        sanitize_token(report.unfenced.as_deref().unwrap_or("none")),
        started.elapsed().as_secs_f64()
    );
    report
}

/// One whitespace-free token for a `reason=` field (a free-form error would
/// otherwise break the line's `key=value` schema).
fn sanitize_token(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_whitespace() || ch.is_control() || ch == '"' {
                '_'
            } else {
                ch
            }
        })
        .take(160)
        .collect()
}

/// #vit-bank-gate: the OUTPUT-ENCLOSURE half of the root-terminal fence only
/// (the pre-P8 contract): the contradiction, if any. Kept for the historical
/// unit test; production calls [`fence_root_terminal_verified`], which also
/// fences the verdict's objective lowers and fails closed.
#[cfg(test)]
pub(super) fn root_finished_output_contradiction(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_bounds: Option<&BoundedTensor>,
    refusal: Option<&'static str>,
) -> Option<String> {
    fence_root_terminal_verified(graph, input, output_bounds, &[], None, false, None, refusal)
        .contradiction
}

// ---------------------------------------------------------------------------
// #vit-margin-alpha-complex (P8): the EMPTINESS gate.
// ---------------------------------------------------------------------------

/// One kind of emptiness claim a multi-objective BaB can close a region with.
///
/// An emptiness claim (`InfeasibleDomain`, or a half-space `with_constraint`
/// proves empty) closes its sub-region WITHOUT a bound, so no fence can check
/// it. The 2026-08-30 vit alarm's one genuinely silent path was exactly this:
/// both children raised `InfeasibleDomain`, the fold recorded `Children([])`,
/// and the lane published `Verified`. Every site that closes a region this
/// way on the multi-objective lanes calls [`note_emptiness_claim`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EmptinessClaim {
    /// Queue fold: `Children([])` (every evaluated child infeasible).
    FoldPair = 0,
    /// Queue fold: `InfeasibleEmpty` (a NoUnstable domain proved empty).
    FoldEmpty = 1,
    /// Queue fold: `InfeasibleCoverRefused` (already unresolved; counted).
    FoldCover = 2,
    /// Batched fold: some, not all, children infeasible (survivors queued).
    Sibling = 3,
    /// Batched child construction: a half-space proved empty and skipped.
    /// Also a committed kFSB cover that omits an `Infeasible` side, and a kFSB
    /// parent-cover certificate that may rest on one (`kfsb_multi.rs`).
    Construct = 4,
    /// Sequential lane: any of its emptiness dispositions.
    Sequential = 5,
    /// Multi-depth split: an empty leaf half-space dropped.
    MultiDepth = 6,
    /// Per-disjunct child evaluation proved empty.
    PerDisjunct = 7,
}

const EMPTINESS_CLAIM_KINDS: usize = 8;
const EMPTINESS_CLAIM_TOKENS: [&str; EMPTINESS_CLAIM_KINDS] = [
    "fold_pair",
    "fold_empty",
    "fold_cover",
    "sibling",
    "construct",
    "sequential",
    "multi_depth",
    "per_disjunct",
];

/// Process-cumulative claim counters. Process-wide because the claim sites
/// (Rayon workers, the fold, the sequential lane) share no lifecycle handle;
/// the gate reads DELTAS across one verification call, so an unrelated claim
/// elsewhere in the process can only make it MORE conservative.
static MO_EMPTINESS_CLAIMS: [AtomicU64; EMPTINESS_CLAIM_KINDS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Count one emptiness claim. Pure bookkeeping: callers keep their fold
/// exactly as it was.
pub(super) fn note_emptiness_claim(kind: EmptinessClaim) {
    MO_EMPTINESS_CLAIMS[kind as usize].fetch_add(1, Ordering::Relaxed);
}

/// A snapshot of (or a delta between snapshots of) the claim counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EmptinessClaimCounts([u64; EMPTINESS_CLAIM_KINDS]);

impl EmptinessClaimCounts {
    /// The process-cumulative counters now.
    pub(super) fn snapshot() -> Self {
        let mut counts = [0u64; EMPTINESS_CLAIM_KINDS];
        for (slot, counter) in counts.iter_mut().zip(MO_EMPTINESS_CLAIMS.iter()) {
            *slot = counter.load(Ordering::Relaxed);
        }
        Self(counts)
    }

    /// Claims made after `before` was taken.
    pub(super) fn since(self, before: Self) -> Self {
        let mut counts = [0u64; EMPTINESS_CLAIM_KINDS];
        for ((slot, now), then) in counts.iter_mut().zip(self.0).zip(before.0) {
            *slot = now.saturating_sub(then);
        }
        Self(counts)
    }

    pub(super) fn total(self) -> u64 {
        self.0.iter().sum()
    }

    fn describe(self) -> String {
        EMPTINESS_CLAIM_TOKENS
            .iter()
            .zip(self.0)
            .map(|(token, count)| format!("{token}={count}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// The emptiness gate's decision: the `Unknown` reason that must replace a
/// `Verified` resting on any emptiness claim, or `None`. Pure, so the unit
/// test exercises exactly the predicate production calls.
pub(super) fn emptiness_gate_downgrade(
    verified: bool,
    claims: EmptinessClaimCounts,
) -> Option<String> {
    (verified && claims.total() > 0).then(|| {
        format!(
            "containment fence: verified rests on {} unfenced emptiness claim(s) ({})",
            claims.total(),
            claims.describe()
        )
    })
}

/// Apply the emptiness gate to one fenced multi-objective verification and
/// print its UNCONDITIONAL line (fixed schema):
/// `[mo-containment-fence] emptiness-gate verdict=<verified|other|error> claims=N
/// fold_pair=N fold_empty=N fold_cover=N sibling=N construct=N sequential=N
/// multi_depth=N per_disjunct=N outcome=<pass|DOWNGRADED|n/a>` (`verdict=error`
/// comes from [`emit_emptiness_gate_error_line`] on an `Err` return).
///
/// Downgrade-only: a `Verified` with any claim since `before` becomes
/// `Unknown`; every other status is untouched.
pub(super) fn apply_emptiness_gate(result: &mut BetaCrownResult, before: EmptinessClaimCounts) {
    let claims = EmptinessClaimCounts::snapshot().since(before);
    let verified = matches!(result.result, BabVerificationStatus::Verified);
    let downgrade = emptiness_gate_downgrade(verified, claims);
    eprintln!(
        "[mo-containment-fence] emptiness-gate verdict={} claims={} {} outcome={}",
        if verified { "verified" } else { "other" },
        claims.total(),
        claims.describe(),
        match (verified, downgrade.is_some()) {
            (true, true) => "DOWNGRADED",
            (true, false) => "pass",
            (false, _) => "n/a",
        }
    );
    if let Some(reason) = downgrade {
        result.result = BabVerificationStatus::Unknown { reason };
    }
}

/// The emptiness gate's line for a fenced verification that returned an
/// ERROR (P8 review): there is no verdict to gate, but the line is still
/// printed, in the same fixed schema (`verdict=error … outcome=n/a`), so every
/// fenced verification emits exactly one gate line and a missing line always
/// means the gate never ran.
pub(super) fn emit_emptiness_gate_error_line(before: EmptinessClaimCounts) {
    let claims = EmptinessClaimCounts::snapshot().since(before);
    eprintln!(
        "[mo-containment-fence] emptiness-gate verdict=error claims={} {} outcome=n/a",
        claims.total(),
        claims.describe()
    );
}

/// The armed fence for one multi-objective BaB run.
pub(super) struct MoContainmentFence {
    samples: Vec<DomainFenceSample>,
    root_input: Arc<BoundedTensor>,
    walks_failed: usize,
    root_nodes_checked: usize,
    root_elements_checked: usize,
    root_nodes_skipped: usize,
    children_checked: usize,
    children_in_domain: usize,
    children_without_sample: usize,
    children_genbab_skipped: usize,
    child_rows_compared: usize,
    child_blind: usize,
    min_child_slack: f64,
    contradiction: Option<String>,
    started: Instant,
    /// Set when arming failed after construction; suppresses the drop
    /// summary, which would otherwise read like a run that checked something.
    disarmed: bool,
}

impl MoContainmentFence {
    /// Arm the fence against the ROOT domain, or print why it cannot.
    ///
    /// `refusal` is [`mo_containment_fence_refusal`]'s answer for this run.
    /// A returned fence may already carry a root contradiction; the caller
    /// must read [`Self::contradiction`] before using the root.
    pub(super) fn arm(
        graph: &GraphNetwork,
        root: &MultiObjectiveGraphBabDomain,
        objectives: &[Vec<f32>],
        refusal: Option<&'static str>,
    ) -> Option<Self> {
        if let Some(reason) = refusal {
            eprintln!("[mo-containment-fence] armed=0 reason={reason}");
            return None;
        }
        let started = Instant::now();
        let (root_lower, root_upper) = flat_f64(&root.input_bounds);
        // Finiteness first, so the ordering test below never sees a NaN.
        let box_ok = !root_lower.is_empty()
            && root_lower
                .iter()
                .chain(root_upper.iter())
                .all(|v| v.is_finite())
            && root_lower
                .iter()
                .zip(root_upper.iter())
                .all(|(l, h)| l <= h);
        if !box_ok {
            eprintln!("[mo-containment-fence] armed=0 reason=root-box-empty-or-non-finite");
            return None;
        }
        let input_shape = root.input_bounds.shape().to_vec();

        // ReLU node -> the node whose output is its pre-activation.
        let mut relu_pre: Vec<(String, String)> = Vec::new();
        for name in graph.node_names() {
            if let Some(node) = graph.node(name) {
                if matches!(node.layer(), Layer::ReLU(_)) {
                    if let Some(pre) = node.inputs().first() {
                        relu_pre.push((name.clone(), pre.clone()));
                    }
                }
            }
        }

        let mut fence = Self {
            samples: Vec::new(),
            root_input: Arc::clone(&root.input_bounds),
            walks_failed: 0,
            root_nodes_checked: 0,
            root_elements_checked: 0,
            root_nodes_skipped: 0,
            children_checked: 0,
            children_in_domain: 0,
            children_without_sample: 0,
            children_genbab_skipped: 0,
            child_rows_compared: 0,
            child_blind: 0,
            min_child_slack: f64::INFINITY,
            contradiction: None,
            started,
            disarmed: false,
        };
        let mut first_walk_error: Option<String> = None;
        let mut checked_node_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // The root's certified node bounds, flattened ONCE (exact f32 -> f64).
        let certified_nodes: Vec<(String, Vec<usize>, Vec<f64>, Vec<f64>)> = root
            .node_bounds
            .iter()
            .map(|(name, bounds)| {
                let (lower, upper) = flat_f64(bounds);
                (name.clone(), bounds.shape().to_vec(), lower, upper)
            })
            .collect();

        let points = domain_fence_sample_points(&root_lower, &root_upper, MO_FENCE_SAMPLES);
        for (sample_index, point) in points.iter().enumerate() {
            if started.elapsed() > MO_FENCE_ARM_CAP {
                break;
            }
            let Ok(array) = ArrayD::from_shape_vec(IxDyn(&input_shape), point.clone()) else {
                fence.walks_failed += 1;
                continue;
            };
            let walk = match graph.propagate_f64_fence_nodes(&Interval64::point(array)) {
                Ok(walk) => walk,
                Err(error) => {
                    fence.walks_failed += 1;
                    if first_walk_error.is_none() {
                        first_walk_error = Some(error.to_string());
                    }
                    continue;
                }
            };

            // (1) Root node bounds, two-sided, every node both sides know.
            for (name, shape, certified_lower, certified_upper) in &certified_nodes {
                let Some(observed) = walk.get(name.as_str()) else {
                    continue;
                };
                let observed_lower: Vec<f64> = observed.lower.iter().copied().collect();
                let observed_upper: Vec<f64> = observed.upper.iter().copied().collect();
                if observed_lower.len() != certified_lower.len()
                    || !same_layout(shape, observed.lower.shape())
                {
                    if sample_index == 0 {
                        fence.root_nodes_skipped += 1;
                    }
                    continue;
                }
                if checked_node_names.insert(name.clone()) {
                    fence.root_nodes_checked += 1;
                }
                fence.root_elements_checked += certified_lower.len();
                if let Some(separation) = first_band_separation(
                    certified_lower,
                    certified_upper,
                    &observed_lower,
                    &observed_upper,
                    DOMAIN_FENCE_REL_BAND,
                    sample_index,
                ) {
                    fence.trip(format!("root node '{name}': {}", separation.describe()));
                    fence.print_armed_line(first_walk_error.as_deref());
                    return Some(fence);
                }
            }

            // (2) Objective enclosures at this point.
            let Some(output) = walk.get(graph.output_name()) else {
                fence.walks_failed += 1;
                continue;
            };
            let y_lower: Vec<f64> = output.lower.iter().copied().collect();
            let y_upper: Vec<f64> = output.upper.iter().copied().collect();
            let objective_values: Vec<(f64, f64)> = objectives
                .iter()
                .map(|row| {
                    objective_enclosure(row, &y_lower, &y_upper)
                        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
                })
                .collect();

            // (3) ReLU pre-activations, for child membership.
            let mut relu_pre_activations: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
            for (relu, pre) in &relu_pre {
                if pre == NETWORK_INPUT {
                    relu_pre_activations.insert(relu.clone(), (point.clone(), point.clone()));
                } else if let Some(pre_value) = walk.get(pre.as_str()) {
                    relu_pre_activations.insert(
                        relu.clone(),
                        (
                            pre_value.lower.iter().copied().collect(),
                            pre_value.upper.iter().copied().collect(),
                        ),
                    );
                }
            }
            fence.samples.push(DomainFenceSample {
                point: point.clone(),
                objectives: objective_values,
                relu_pre_activations,
            });
        }

        if fence.samples.is_empty() {
            eprintln!(
                "[mo-containment-fence] armed=0 reason=no-evaluable-sample walks_failed={} \
                 first_error={}",
                fence.walks_failed,
                first_walk_error.as_deref().unwrap_or("none")
            );
            fence.disarmed = true;
            return None;
        }

        // (4) Root objective LOWER bounds against every sample.
        let root_lowers: Vec<f64> = root
            .objective_bounds
            .iter()
            .map(|(lower, _)| f64::from(*lower))
            .collect();
        if let DomainFenceOutcome::Contradiction(contradiction) =
            check_domain_lower_bounds_at_samples(&fence.samples, None, &[], &root_lowers)
        {
            fence.trip(format!("root objectives: {}", contradiction.describe()));
        }
        fence.print_armed_line(first_walk_error.as_deref());
        Some(fence)
    }

    fn print_armed_line(&self, first_walk_error: Option<&str>) {
        eprintln!(
            "[mo-containment-fence] armed=1 samples={} walks_failed={} root_nodes_checked={} \
             root_elements={} root_nodes_skipped={} root={} arm_secs={:.3} first_walk_error={}",
            self.samples.len(),
            self.walks_failed,
            self.root_nodes_checked,
            self.root_elements_checked,
            self.root_nodes_skipped,
            if self.contradiction.is_some() {
                "CONTRADICTION"
            } else {
                "consistent"
            },
            self.started.elapsed().as_secs_f64(),
            first_walk_error.unwrap_or("none")
        );
    }

    fn trip(&mut self, what: String) {
        eprintln!("[mo-containment-fence] CONTRADICTION {what}");
        if self.contradiction.is_none() {
            self.contradiction = Some(what);
        }
    }

    /// The first contradiction found, if any. A `Some` means the lane must not
    /// publish a verdict that rests on the bound pipeline.
    pub(super) fn contradiction(&self) -> Option<&str> {
        self.contradiction.as_deref()
    }

    /// Check one child domain's certified objective LOWER bounds at every
    /// sample proved inside it.
    pub(super) fn check_child(&mut self, child: &MultiObjectiveGraphBabDomain) {
        if self.contradiction.is_some() {
            return;
        }
        self.children_checked += 1;
        // GenBaB literals restrict the region at arbitrary split points on
        // non-ReLU nodes; the fence does not model them, so such a child is
        // never compared (fail-safe: less coverage, never a false alarm).
        if !child.history.genbab_constraints.is_empty() {
            self.children_genbab_skipped += 1;
            return;
        }
        let literals: Vec<DomainSplitLiteral> = child
            .history
            .constraints
            .iter()
            .map(|constraint| DomainSplitLiteral {
                relu_node: constraint.node_name.clone(),
                neuron: constraint.neuron_idx,
                active: constraint.is_active,
            })
            .collect();
        let own_box = if Arc::ptr_eq(&child.input_bounds, &self.root_input) {
            None
        } else {
            Some(flat_f64(&child.input_bounds))
        };
        let input_box = own_box
            .as_ref()
            .map(|(lower, upper)| (lower.as_slice(), upper.as_slice()));
        let lowers: Vec<f64> = child
            .objective_bounds
            .iter()
            .map(|(lower, _)| f64::from(*lower))
            .collect();
        match check_domain_lower_bounds_at_samples(&self.samples, input_box, &literals, &lowers) {
            DomainFenceOutcome::Consistent {
                rows_compared,
                min_slack,
                blind,
                ..
            } => {
                self.children_in_domain += 1;
                self.child_rows_compared += rows_compared;
                self.child_blind += blind;
                if min_slack < self.min_child_slack {
                    self.min_child_slack = min_slack;
                }
            }
            DomainFenceOutcome::NoSampleInDomain => {
                self.children_without_sample += 1;
            }
            DomainFenceOutcome::Contradiction(contradiction) => {
                self.trip(format!(
                    "child depth={} literals={}: {}",
                    child.depth,
                    literals.len(),
                    contradiction.describe()
                ));
            }
        }
    }
}

impl Drop for MoContainmentFence {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        eprintln!(
            "[mo-containment-fence] summary outcome={} samples={} root_nodes_checked={} \
             children_checked={} children_in_domain={} children_without_sample={} \
             children_genbab_skipped={} child_rows_compared={} child_blind={} \
             min_child_slack={:e} secs={:.3}",
            if self.contradiction.is_some() {
                "CONTRADICTION"
            } else {
                "consistent"
            },
            self.samples.len(),
            self.root_nodes_checked,
            self.children_checked,
            self.children_in_domain,
            self.children_without_sample,
            self.children_genbab_skipped,
            self.child_rows_compared,
            self.child_blind,
            self.min_child_slack,
            self.started.elapsed().as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    use super::*;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::GraphNode;

    /// x in [-1,1]^2 -> fc1 = x -> relu1 -> fc2 = [r0 + r1, r0 - r1].
    fn tiny_relu_graph() -> GraphNetwork {
        let identity = LinearLayer::new(
            arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
            Some(arr1(&[0.0_f32, 0.0])),
        )
        .unwrap();
        let head = LinearLayer::new(
            arr2(&[[1.0_f32, 1.0], [1.0, -1.0]]),
            Some(arr1(&[0.0_f32, 0.0])),
        )
        .unwrap();
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("fc1", Layer::Linear(identity)));
        g.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["fc1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "fc2",
            Layer::Linear(head),
            vec!["relu1".to_string()],
        ));
        g.set_output("fc2");
        g
    }

    fn unit_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
        )
        .unwrap()
    }

    /// Objectives y0 and y1 with thresholds far below both ranges.
    fn objectives() -> Vec<Vec<f32>> {
        vec![vec![1.0, 0.0], vec![0.0, 1.0]]
    }

    const THRESHOLDS: [f32; 2] = [-10.0, -10.0];

    fn sound_root(graph: &GraphNetwork, input: &BoundedTensor) -> MultiObjectiveGraphBabDomain {
        let node_bounds: HashMap<String, BoundedTensor> =
            graph.collect_node_bounds(input).expect("IBP node bounds");
        // y0 = r0 + r1 in [0, 2]; y1 = r0 - r1 in [-1, 1].
        MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 2.0), (-1.0, 1.0)],
            input,
            &THRESHOLDS,
            false,
        )
        .expect("root domain")
    }

    fn active_child(
        graph: &GraphNetwork,
        root: &MultiObjectiveGraphBabDomain,
    ) -> MultiObjectiveGraphBabDomain {
        root.with_constraint(
            graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: true,
                score: 0.0,
            },
            false,
            &THRESHOLDS,
        )
        .expect("child construction")
        .expect("the active half-space of an unstable neuron is feasible")
    }

    /// A sound root and a sound child pass, and the child is actually
    /// compared (a sample is PROVED inside it) — a fence that compares nothing
    /// must not read as a pass.
    #[test]
    fn vit_bank_gate_lane_fence_passes_sound_root_and_child_bounds() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let root = sound_root(&graph, &input);
        let mut fence = MoContainmentFence::arm(&graph, &root, &objectives(), None)
            .expect("the fence arms on a supported graph");
        assert!(
            fence.contradiction().is_none(),
            "{:?}",
            fence.contradiction()
        );
        assert!(
            fence.root_nodes_checked >= 3,
            "fc1, relu1 and fc2 are all compared"
        );

        let mut child = active_child(&graph, &root);
        // On x0 >= 0: y0 = x0 + max(x1, 0) >= 0, y1 = x0 - max(x1, 0) >= -1.
        child
            .update_bounds(vec![(0.0, 2.0), (-1.0, 1.0)], &THRESHOLDS, false)
            .unwrap();
        fence.check_child(&child);
        assert!(
            fence.contradiction().is_none(),
            "{:?}",
            fence.contradiction()
        );
        assert_eq!(fence.children_checked, 1);
        assert_eq!(
            fence.children_in_domain, 1,
            "a sample must be proved inside the child"
        );
        assert!(fence.child_rows_compared > 0);
    }

    /// THE CLASS THIS EXISTS FOR: a CHILD whose certified lower bound does
    /// not contain the network on its own sub-region is refused.
    #[test]
    fn vit_bank_gate_lane_fence_refuses_a_non_containing_child() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let root = sound_root(&graph, &input);
        let mut fence = MoContainmentFence::arm(&graph, &root, &objectives(), None)
            .expect("the fence arms on a supported graph");
        let mut child = active_child(&graph, &root);
        // y0 = x0 + max(x1, 0) is 0.5 at the interior sample (0.5, -0.5),
        // which lies in the child (x0 > 0): a certified lower of 1.5 is false.
        child
            .update_bounds(vec![(1.5, 2.0), (-1.0, 1.0)], &THRESHOLDS, false)
            .unwrap();
        fence.check_child(&child);
        let reason = fence
            .contradiction()
            .expect("a non-containing child lower bound must be refused");
        assert!(reason.contains("child depth=1"), "{reason}");
        assert!(reason.contains("objective row 0"), "{reason}");
    }

    /// SCOPING, the false-alarm guard: a child's lower bound only covers its
    /// own half-space. On x0 <= 0, -y0 = -max(x1, 0) >= -1 is TRUE, while on
    /// x0 > 0 it is false (-2 at (1, 1)). A whole-box comparison would refuse
    /// this correct child; the scoped check compares it only at samples
    /// proved inside x0 < 0 — and does compare it there.
    #[test]
    fn vit_bank_gate_lane_fence_scopes_child_checks_to_the_child_region() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let objectives3: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let thresholds3 = [-10.0_f32, -10.0, -10.0];
        let node_bounds: HashMap<String, BoundedTensor> =
            graph.collect_node_bounds(&input).expect("IBP node bounds");
        let root = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 2.0), (-1.0, 1.0), (-2.0, 0.0)],
            &input,
            &thresholds3,
            false,
        )
        .unwrap();
        let mut fence = MoContainmentFence::arm(&graph, &root, &objectives3, None)
            .expect("the fence arms on a supported graph");
        assert!(
            fence.contradiction().is_none(),
            "{:?}",
            fence.contradiction()
        );
        let mut child = root
            .with_constraint(
                &graph,
                GraphNeuronConstraint {
                    node_name: "relu1".to_string(),
                    neuron_idx: 0,
                    is_active: false,
                    score: 0.0,
                },
                false,
                &thresholds3,
            )
            .unwrap()
            .unwrap();
        child
            .update_bounds(
                vec![(0.0, 1.0), (-1.0, 0.0), (-1.0, 0.0)],
                &thresholds3,
                false,
            )
            .unwrap();
        fence.check_child(&child);
        assert!(
            fence.contradiction().is_none(),
            "a bound true on its own half-space must not be refused: {:?}",
            fence.contradiction()
        );
        assert_eq!(
            fence.children_in_domain, 1,
            "the child must still be compared"
        );
    }

    /// The 2026-08-25 shape at the ROOT: a finite, plausible, non-containing
    /// node bound (`±1.18e-38` for a true `[-1, 1]`) is caught at arm time.
    #[test]
    fn vit_bank_gate_lane_fence_catches_the_nary_concat_shape_at_the_root() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let mut node_bounds: HashMap<String, BoundedTensor> =
            graph.collect_node_bounds(&input).expect("IBP node bounds");
        node_bounds.insert(
            "fc1".to_string(),
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.18e-38_f32, -1.18e-38]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.18e-38_f32, 1.18e-38]).unwrap(),
            )
            .unwrap(),
        );
        let root = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 2.0), (-1.0, 1.0)],
            &input,
            &THRESHOLDS,
            false,
        )
        .unwrap();
        let fence = MoContainmentFence::arm(&graph, &root, &objectives(), None)
            .expect("the fence arms and reports the root contradiction");
        let reason = fence
            .contradiction()
            .expect("the n-ary Concat shape must be caught at the root");
        assert!(reason.contains("root node 'fc1'"), "{reason}");
    }

    /// A false root OBJECTIVE lower bound is caught at arm time.
    #[test]
    fn vit_bank_gate_lane_fence_catches_a_false_root_objective_lower_bound() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let node_bounds: HashMap<String, BoundedTensor> =
            graph.collect_node_bounds(&input).expect("IBP node bounds");
        // y1 = r0 - r1 reaches -1 at (-1, 1); a certified lower of 0.5 is false.
        let root = MultiObjectiveGraphBabDomain::root(
            node_bounds,
            vec![(0.0, 2.0), (0.5, 1.0)],
            &input,
            &THRESHOLDS,
            false,
        )
        .unwrap();
        let fence = MoContainmentFence::arm(&graph, &root, &objectives(), None).unwrap();
        let reason = fence
            .contradiction()
            .expect("a false root lower must be caught");
        assert!(reason.contains("root objectives"), "{reason}");
        assert!(reason.contains("objective row 1"), "{reason}");
    }

    /// Every property-conditioned mechanism refuses to arm, and names itself.
    #[test]
    fn vit_bank_gate_lane_fence_refuses_property_conditioned_runs() {
        assert_eq!(
            mo_containment_fence_refusal(false, false, false, false, false),
            None
        );
        assert_eq!(
            mo_containment_fence_refusal(true, false, false, false, false),
            Some("conjunctive")
        );
        assert_eq!(
            mo_containment_fence_refusal(false, true, false, false, false),
            Some("cuts")
        );
        assert_eq!(
            mo_containment_fence_refusal(false, false, true, false, false),
            Some("conflict-clauses")
        );
        assert_eq!(
            mo_containment_fence_refusal(false, false, false, true, false),
            Some("clip-interm-domain")
        );
        assert_eq!(
            mo_containment_fence_refusal(false, false, false, false, true),
            Some("relaxed-clip")
        );
        let graph = tiny_relu_graph();
        let input = unit_box();
        let root = sound_root(&graph, &input);
        assert!(MoContainmentFence::arm(&graph, &root, &objectives(), Some("cuts")).is_none());
    }

    /// A ROOT-terminal `Verified` is fenced on its published output enclosure:
    /// a sound enclosure passes, the Concat-defect shape is refused, and a
    /// missing enclosure is reported UNFENCED rather than passing silently.
    #[test]
    fn vit_bank_gate_root_finished_output_is_fenced() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        // fc2 = [r0 + r1, r0 - r1] over [-1, 1]^2: [0, 2] x [-1, 1].
        let sound = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 1.0]).unwrap(),
        )
        .unwrap();
        assert!(root_finished_output_contradiction(&graph, &input, Some(&sound), None).is_none());

        let collapsed = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.18e-38_f32, -1.18e-38]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.18e-38_f32, 1.18e-38]).unwrap(),
        )
        .unwrap();
        let reason = root_finished_output_contradiction(&graph, &input, Some(&collapsed), None)
            .expect("a collapsed root output enclosure must be refused");
        assert!(reason.contains("root output"), "{reason}");

        assert!(root_finished_output_contradiction(&graph, &input, None, None).is_none());
        assert!(
            root_finished_output_contradiction(&graph, &input, Some(&collapsed), Some("cuts"))
                .is_none(),
            "a refused run is not fenced"
        );
    }

    /// The objective enclosure is sound and point-tight.
    #[test]
    fn vit_bank_gate_objective_enclosure_is_outward() {
        let (lower, upper) =
            objective_enclosure(&[1.0, -2.0, 0.5], &[0.1, 0.2, 0.3], &[0.1, 0.2, 0.3]).unwrap();
        let exact = 0.1 - 0.4 + 0.15;
        assert!(lower <= exact && exact <= upper);
        assert!(upper - lower < 1e-12);
        assert!(objective_enclosure(&[1.0], &[0.0, 1.0], &[0.0, 1.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // #vit-margin-alpha-complex (P8): the root-terminal authority, the ORT
    // adjudication, and the emptiness gate.
    // -----------------------------------------------------------------------

    /// The sound output enclosure of `tiny_relu_graph` over `unit_box`.
    fn sound_output() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0_f32, 1.0]).unwrap(),
        )
        .unwrap()
    }

    /// An exact forward of `tiny_relu_graph` shifted by `y1_shift` on the
    /// second output: `0.0` is the model ny built; anything else simulates an
    /// ONNX model ny's graph does NOT faithfully represent (the defect class
    /// only an independent forward can see).
    struct TinyOracle {
        y1_shift: f32,
    }

    impl RootAdjudicationOracle for TinyOracle {
        fn name(&self) -> &'static str {
            "tiny"
        }
        fn evaluate(&self, x: &[f32]) -> Result<Vec<f32>, String> {
            if x.len() != 2 {
                return Err(format!("expected 2 inputs, got {}", x.len()));
            }
            let (r0, r1) = (x[0].max(0.0), x[1].max(0.0));
            Ok(vec![r0 + r1, r0 - r1 + self.y1_shift])
        }
    }

    struct FailingOracle;

    impl RootAdjudicationOracle for FailingOracle {
        fn name(&self) -> &'static str {
            "failing"
        }
        fn evaluate(&self, _x: &[f32]) -> Result<Vec<f32>, String> {
            Err("session could not be built".to_string())
        }
    }

    struct WideOracle;

    impl RootAdjudicationOracle for WideOracle {
        fn name(&self) -> &'static str {
            "wide"
        }
        fn evaluate(&self, _x: &[f32]) -> Result<Vec<f32>, String> {
            Ok(vec![0.0, 0.0, 0.0])
        }
    }

    /// The root-terminal verdict's AUTHORITY is fenced: sound certified lowers
    /// pass with every row compared; a FALSE certified lower (0.5 on
    /// y1 = r0 − r1, which reaches −1 at the vertex (−1, 1)) is refused; and a
    /// verdict whose lowers were never recorded, or do not match the
    /// objectives, FAILS CLOSED instead of passing as fenced.
    #[test]
    fn vit_bank_gate_root_terminal_objective_lowers_are_fenced() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let output = sound_output();
        let objectives = objectives();

        let sound = fence_root_terminal_verified(
            &graph,
            &input,
            Some(&output),
            &objectives,
            Some(&[0.0, -1.0]),
            false,
            None,
            None,
        );
        assert_eq!(
            sound,
            RootTerminalFence::default(),
            "sound lowers must pass: {sound:?}"
        );
        assert!(sound.downgrade_reason().is_none());

        let false_lower = fence_root_terminal_verified(
            &graph,
            &input,
            Some(&output),
            &objectives,
            Some(&[0.0, 0.5]),
            false,
            None,
            None,
        );
        let reason = false_lower
            .downgrade_reason()
            .expect("a false root objective lower must be refused");
        assert!(reason.contains("root objectives"), "{reason}");
        assert!(reason.contains("objective row 1"), "{reason}");
        assert!(false_lower.contradiction.is_some());

        let unrecorded = fence_root_terminal_verified(
            &graph,
            &input,
            Some(&output),
            &objectives,
            None,
            false,
            None,
            None,
        );
        assert!(unrecorded.contradiction.is_none());
        assert_eq!(
            unrecorded.unfenced.as_deref(),
            Some("objective-lowers-not-recorded")
        );
        assert!(
            unrecorded
                .downgrade_reason()
                .is_some_and(|reason| reason.contains("unfenced")),
            "an unrecorded authority must fail CLOSED"
        );

        let mismatched = fence_root_terminal_verified(
            &graph,
            &input,
            Some(&output),
            &objectives,
            Some(&[0.0]),
            false,
            None,
            None,
        );
        assert_eq!(
            mismatched.unfenced.as_deref(),
            Some("objective-lowers-shape-mismatch")
        );

        // The collapsed output enclosure (the 2026-08-25 shape) is still
        // refused even when the lowers are sound.
        let collapsed = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.18e-38_f32, -1.18e-38]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.18e-38_f32, 1.18e-38]).unwrap(),
        )
        .unwrap();
        let concat_shape = fence_root_terminal_verified(
            &graph,
            &input,
            Some(&collapsed),
            &objectives,
            Some(&[0.0, -1.0]),
            false,
            None,
            None,
        );
        assert!(concat_shape
            .contradiction
            .as_deref()
            .is_some_and(|what| what.contains("root output")));

        // A property-conditioned run is not fenced (and not downgraded).
        assert_eq!(
            fence_root_terminal_verified(
                &graph,
                &input,
                Some(&output),
                &objectives,
                None,
                true,
                None,
                Some("cuts"),
            ),
            RootTerminalFence::default()
        );
    }

    /// The ORT adjudication is INDEPENDENT of ny's graph: with a forward of
    /// the model ny built it passes; with a forward of a DIFFERENT model (y1
    /// shifted by −2, so y1 reaches −3) it refuses lowers the f64 walk of ny's
    /// graph accepts — the defect class only an independent forward can see.
    /// Armed, it FAILS CLOSED when no oracle is attached, when every sample
    /// errors, or when the output width does not match the objectives; dark
    /// (`ort_requested = false`) an attached oracle is never consulted.
    #[test]
    fn vit_bank_gate_root_terminal_ort_adjudication_is_independent_and_fails_closed() {
        let graph = tiny_relu_graph();
        let input = unit_box();
        let output = sound_output();
        let objectives = objectives();
        let lowers = [0.0_f32, -1.0];
        let run = |oracle: Option<&dyn RootAdjudicationOracle>, requested: bool| {
            fence_root_terminal_verified(
                &graph,
                &input,
                Some(&output),
                &objectives,
                Some(&lowers),
                requested,
                oracle,
                None,
            )
        };

        let faithful = TinyOracle { y1_shift: 0.0 };
        assert_eq!(
            run(Some(&faithful), true),
            RootTerminalFence::default(),
            "a faithful independent forward must adjudicate sound lowers"
        );

        let divergent = TinyOracle { y1_shift: -2.0 };
        let refused = run(Some(&divergent), true);
        let what = refused
            .contradiction
            .as_deref()
            .expect("a certified lower above the independent forward's value must be refused");
        assert!(what.contains("ONNX Runtime"), "{what}");
        assert!(what.contains("objective row 1"), "{what}");
        // ...while ny's own walk (ORT dark) accepts the very same lowers.
        assert_eq!(run(Some(&divergent), false), RootTerminalFence::default());

        let no_oracle = run(None, true);
        assert_eq!(
            no_oracle.unfenced.as_deref(),
            Some("ort-requested-but-no-oracle-attached")
        );
        assert!(no_oracle.downgrade_reason().is_some());

        let failing = run(Some(&FailingOracle), true);
        assert!(failing.contradiction.is_none());
        assert!(
            failing
                .unfenced
                .as_deref()
                .is_some_and(|why| why.starts_with("ort-no-sample-adjudicated")),
            "{failing:?}"
        );
        assert!(
            failing.downgrade_reason().is_some(),
            "armed + unadjudicated fails closed"
        );

        let wide = run(Some(&WideOracle), true);
        assert!(
            wide.unfenced
                .as_deref()
                .is_some_and(|why| why.starts_with("ort-no-sample-adjudicated")),
            "{wide:?}"
        );

        // DARK: an attached (even failing) oracle is never consulted.
        assert_eq!(
            run(Some(&FailingOracle), false),
            RootTerminalFence::default()
        );
    }

    /// The EMPTINESS gate: a `Verified` resting on ANY emptiness claim becomes
    /// `Unknown` (the 2026-08-30 alarm's silent double-`Infeasible` fold can no
    /// longer publish on a fenced run); a claim-free `Verified` and every
    /// non-verified status are untouched.
    #[test]
    fn vit_bank_gate_emptiness_gate_refuses_a_verified_resting_on_any_emptiness_claim() {
        let none = EmptinessClaimCounts::default();
        assert_eq!(none.total(), 0);
        assert!(emptiness_gate_downgrade(true, none).is_none());

        for kind in [
            EmptinessClaim::FoldPair,
            EmptinessClaim::FoldEmpty,
            EmptinessClaim::FoldCover,
            EmptinessClaim::Sibling,
            EmptinessClaim::Construct,
            EmptinessClaim::Sequential,
            EmptinessClaim::MultiDepth,
            EmptinessClaim::PerDisjunct,
        ] {
            let mut counts = [0u64; EMPTINESS_CLAIM_KINDS];
            counts[kind as usize] = 1;
            let claims = EmptinessClaimCounts(counts);
            let reason = emptiness_gate_downgrade(true, claims)
                .unwrap_or_else(|| panic!("{kind:?}: a verified resting on it must downgrade"));
            assert!(
                reason.contains(&format!("{}=1", EMPTINESS_CLAIM_TOKENS[kind as usize])),
                "{kind:?}: {reason}"
            );
            assert!(
                emptiness_gate_downgrade(false, claims).is_none(),
                "{kind:?}: a non-verified status is never touched"
            );
        }

        // The live counters: a claim noted after the snapshot is seen, and the
        // fenced result is downgraded. (Other tests in this process can only
        // ADD claims, which keeps this direction deterministic.)
        let before = EmptinessClaimCounts::snapshot();
        note_emptiness_claim(EmptinessClaim::FoldPair);
        let seen = EmptinessClaimCounts::snapshot().since(before);
        assert!(seen.0[EmptinessClaim::FoldPair as usize] >= 1);
        let mut result = BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 3,
            time_elapsed: Duration::from_millis(1),
            max_depth_reached: 1,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 1,
        };
        apply_emptiness_gate(&mut result, before);
        assert!(
            matches!(&result.result, BabVerificationStatus::Unknown { reason }
                if reason.contains("emptiness")),
            "a fenced Verified resting on an emptiness claim must publish Unknown: {:?}",
            result.result
        );
        let mut timeout = BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            ..result.clone()
        };
        apply_emptiness_gate(&mut timeout, before);
        assert!(matches!(timeout.result, BabVerificationStatus::Timeout));
    }

    /// DARK BY DEFAULT: both P8 keys ship off, so no verifier fences,
    /// adjudicates or runs the emptiness gate unless a preset arms them.
    #[test]
    fn vit_bank_gate_p8_keys_default_dark() {
        let config = crate::beta_crown::config::BetaCrownConfig::default();
        assert!(!config.mo_containment_fence);
        assert!(!config.mo_root_ort_adjudication);
    }
}
