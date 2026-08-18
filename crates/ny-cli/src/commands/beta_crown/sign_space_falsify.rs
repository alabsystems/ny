// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Front end for the LP-guided sign-space falsifier
//! ([`ny_mip::falsify_bnn_sign_suffix_unwired`]).
//!
//! `ny-cli` is the only crate that depends on BOTH `ny-onnx` and `ny-mip`, so
//! this is where a real ONNX graph and a real VNN-LIB property can be turned
//! into the plain-data [`SignSpaceRequest`] the search core wants. The core
//! does no I/O, no environment reads and no ONNX parsing; all of that lives
//! here.
//!
//! # What this module is allowed to conclude: nothing
//!
//! [`SignSpaceOutcome`] has no `Verified`/`Unsat` variant BY CONSTRUCTION — a
//! falsifier can only ever exhibit a witness — and
//! [`the_lane_cannot_produce_a_verified_outcome`] pins that here as well as in
//! `ny-mip`'s own tests. So this lane is structurally incapable of causing a
//! false `unsat`.
//!
//! The other direction is not structural, it is a routing obligation, and it is
//! discharged at the CALL SITE rather than here: `run_sign_space_lane` returns
//! a candidate INPUT VECTOR, never a verdict. `commands/vnncomp.rs` renders it
//! into an SMT-LIB witness, re-forwards it through the ORIGINAL model with
//! `rehydrate_original_witness_outputs`, and hands it to the EXISTING,
//! UNCHANGED `gate_sat_with_trusted_oracle` — the same real ONNX-Runtime
//! forward plus true-`f64` recheck every other `sat` source goes through. The
//! search's own arithmetic is never published. Every non-`Candidate` outcome
//! maps to `None` at that call site, which means the ordinary verification path
//! runs exactly as it would have.
//!
//! # Why there is no `reference_forward`
//!
//! [`SignSpaceRequest::reference_forward`] is an optional layout self-check:
//! the core compares its own integer logits against an independent pre-Softmax
//! forward at the box centre and at sampled vertices, to catch a flatten or
//! transposition that would otherwise produce arithmetically plausible garbage.
//!
//! We pass `None`, deliberately, because the only oracle available here is ONNX
//! Runtime on the ORIGINAL model — whose output is POST-Softmax. On this net
//! that view is useless: 41-43 of 43 Softmax outputs underflow to exactly
//! `0.0f` at every sampled point (measured; see
//! `docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` §6 and
//! `ny-onnx/tests/traffic_terminal_softmax_strip.rs`), so it cannot be compared
//! against logits. Obtaining pre-Softmax values needs the separate, separately
//! dark terminal-Softmax peel, and wiring that in here would be a second
//! transform on the path rather than the wiring this module is.
//!
//! Nothing is lost on the soundness side. The core's OTHER self-check — the
//! incremental delta path against a from-scratch recompute — still runs, and a
//! layout error cannot survive publication anyway: it would produce a candidate
//! that the unchanged trusted-oracle gate's real ORT forward REJECTS, and the
//! run falls through to the normal path. The reference forward would buy search
//! efficiency and an earlier, clearer refusal, not soundness.
//!
//! # The shape the graph walker admits
//!
//! All three staged traffic-sign nets, at 15 / 31 / 33 nodes:
//!
//! ```text
//! graph      := ConvBlock{2,}  DenseBlock*  MatMul  [Softmax]
//! ConvBlock  := Transpose(0,3,1,2) Conv [MaxPool] [BatchNormalization]
//!               Transpose(0,2,3,1) [Reshape] SignPair
//! DenseBlock := MatMul [Mul(const) Add(const)] SignPair
//! SignPair   := Sign Add(scalar) Sign
//! ```
//!
//! ADMISSION IS STRUCTURAL, NEVER BY FILENAME OR CATEGORY. Everything the
//! deeper nets need — the `MaxPool`, the `BatchNormalization`, the folded-BN
//! third convolution whose `|W|` is a positive per-channel scale, the dense
//! head — is read out of the real tensors and PROVED in the core:
//!
//! * `|W|` constant within an output channel, BITWISE, and every channel scale
//!   strictly positive ([`ny_mip::SignSpaceRefusal::ChannelScaleNotConstant`],
//!   [`ny_mip::SignSpaceRefusal::NonPositiveChannelScale`]);
//! * every folded BatchNorm slope strictly positive, per channel
//!   ([`ny_mip::SignSpaceRefusal::BatchNormNotFoldable`]) — a negative slope
//!   inverts the downstream bit, so it is refused, not absorbed;
//! * `MaxPool` before its BatchNorm and `Sign`, giving
//!   `sign(BN(max_w z_w)) = OR_w [z_w >= t]`, with the pool geometry itself
//!   re-derived rather than assumed;
//! * conv1 patch-index distinctness at the ACTUAL geometry (`5x5x3` on the
//!   deeper nets, `3x3x3` on the shallow one), recomputed per position, which
//!   is what makes the FIXED/FREE prepass exact.
//!
//! The first convolution is still held to exact `+/-1` with no bias, because
//! the `f32` replay bound is derived for unit taps; the final dense is still
//! held to exact `+/-1`, because a per-CLASS scale would reorder the logits.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use ny_mip::{
    BinaryStage, ConvSpec, InputGeometry, PoolSpec, SegmentMove, SignSpaceActivation,
    SignSpaceAffine, SignSpaceCandidate, SignSpaceLimits, SignSpaceOutcome, SignSpaceRequest,
    TrustRegion,
};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_onnx::OnnxModel;

/// Counts every call that actually reaches
/// [`ny_mip::falsify_bnn_sign_suffix_unwired`].
///
/// This exists so the default-off contract can be asserted on a CODE PATH
/// rather than on a log line: with the lever unset the counter must not move,
/// no matter what is passed in. It is incremented immediately before the core
/// call and nowhere else.
static CORE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// How many times the search core has been entered in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn core_call_count() -> usize {
    CORE_CALLS.load(Ordering::Relaxed)
}

/// Whether the `NY_BNN_SIGN_SPACE` gate is armed, given the typed preset
/// answer for this category (`attack.bnn_sign_space`).
///
/// TWO LAYERS, ONE RULE, and the rule is the lever crate's — not a local
/// re-implementation of it (`ny_levers::read_over_config`):
///
/// * the ENVIRONMENT still wins wherever it is present, in both directions.
///   Exact `"1"` arms whatever the preset asked for and exact `"0"` disarms
///   it; any OTHER byte sequence is a recorded rejection that suppresses the
///   preset and falls back to the declaration default (`false`). So the
///   exact-`"1"` semantics are unchanged, and a typo is a kill switch rather
///   than a silent promotion of the preset;
/// * with the variable ABSENT, `config` decides: `Some(true)` from a category
///   preset arms the lane on the scored path with no environment at all,
///   `Some(false)` or `None` leaves it dark.
///
/// A `config` value that the declaration would reject cannot happen for a
/// `Bool` lever handed a `Bool`; if it somehow did, this fails CLOSED.
pub(crate) fn sign_space_falsify_armed(config: Option<bool>) -> bool {
    ny_levers::read_over_config(
        &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_LANE,
        config.map(ny_levers::LeverValue::Bool),
    )
    .map(|resolved| resolved.value.as_bool())
    .unwrap_or(false)
}

/// The arming rule as a pure predicate over one raw environment string and one
/// typed preset answer.
///
/// Same declaration, same parser, same chokepoint as
/// [`sign_space_falsify_armed`] — only the lookup is injected — so a test of
/// this function is a test of the production arming rule and not of a
/// re-implementation of it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn sign_space_falsify_armed_from(raw: Option<&str>, config: Option<bool>) -> bool {
    let owned = raw.map(str::to_owned);
    ny_levers::read_over_config_with(
        &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_LANE,
        move |_| owned,
        config.map(ny_levers::LeverValue::Bool),
    )
    .map(|resolved| resolved.value.as_bool())
    .unwrap_or(false)
}

/// The trust-region arm named by one resolved lever value.
///
/// Split out from the reader so a test can exercise the MAPPING without an
/// environment, exactly as `sign_space_falsify_armed_from` does for the lane
/// gate. `None` — and, defensively, any string the declaration's `Enum` parser
/// would have rejected — is the shipped full box.
///
/// The fractions are of the box's WIDEST half-width, not absolute pixels: the
/// fragment's boxes run `eps = 1..10` in raw pixel units and the core's own
/// fixtures are narrower than 1, so an absolute radius would mean a different
/// experiment on every row.
fn trust_region_from_value(raw: Option<&str>) -> TrustRegion {
    match raw {
        Some("box") => TrustRegion::Doubling {
            initial_fraction: 0.125,
        },
        Some("tight") => TrustRegion::Doubling {
            initial_fraction: 0.015_625,
        },
        Some("linf") => TrustRegion::Nearest {
            initial_fraction: 0.015_625,
            refine: 4,
        },
        _ => SignSpaceLimits::default().trust_region,
    }
}

/// The trust-region arm one raw environment string selects.
///
/// Same declaration, same `Enum` parser, same chokepoint as
/// [`SignSpaceProblem::trust_region`] — only the lookup is injected — so a test
/// of this is a test of the production rule, not of a re-implementation.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn trust_region_from(raw: Option<&str>) -> TrustRegion {
    let owned = raw.map(str::to_owned);
    trust_region_from_value(
        ny_levers::read_with(
            &ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRUST_REGION,
            move |_| owned,
        )
        .value
        .as_str(),
    )
}

/// What one consultation of the lane produced.
///
/// There is deliberately no verdict-shaped variant. [`Self::Candidate`] carries
/// a CLAIMED counterexample that the caller must route through the unchanged
/// witness-validation chain; everything else means "the ordinary path runs
/// unchanged".
#[derive(Debug)]
pub(crate) enum SignSpaceLaneOutcome {
    /// Neither the environment nor the category preset armed the lane (or the
    /// environment explicitly disarmed it). Returned BEFORE any model load,
    /// property parse or request construction.
    Disarmed,
    /// The lane declined before the core was consulted: the graph or the
    /// property is outside the shape this front end can extract, or there was
    /// not enough budget left to be worth starting.
    NotAdmitted(String),
    /// The core refused the request as outside its admitted fragment.
    Refused(String),
    /// The search ran out of realizable improving flips, LP budget or time.
    Exhausted {
        /// Best pre-Softmax margin reached in pattern space (`<= 0`).
        best_logit_margin: i64,
        /// FREE first-layer unit count from the exact prepass.
        free_units: usize,
        /// Accepted sign flips.
        flips: usize,
        /// Realizability LPs solved.
        lp_solves: usize,
    },
    /// The realizability LP machinery itself failed.
    SolverError(String),
    /// A CLAIMED counterexample. Not a verdict.
    Candidate(Box<SignSpaceCandidate>),
}

impl SignSpaceLaneOutcome {
    /// The claimed counterexample input, if any.
    ///
    /// This is the ONLY accessor that can yield something publishable, and it
    /// yields it for exactly one variant. Every other outcome — including every
    /// refusal — returns `None`, which is what makes "a refusal falls through to
    /// the existing solver path unchanged" a property of the type rather than of
    /// the caller's discipline.
    pub(crate) fn candidate_input(&self) -> Option<&[f64]> {
        match self {
            Self::Candidate(candidate) => Some(&candidate.input),
            Self::Disarmed
            | Self::NotAdmitted(_)
            | Self::Refused(_)
            | Self::Exhausted { .. }
            | Self::SolverError(_) => None,
        }
    }

    /// One-line description for the flight receipt / stderr.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Disarmed => {
                "disarmed (neither attack.bnn_sign_space nor NY_BNN_SIGN_SPACE armed the lane)"
                    .to_string()
            }
            Self::NotAdmitted(reason) => format!("not admitted: {reason}"),
            Self::Refused(reason) => format!("core refused: {reason}"),
            Self::Exhausted {
                best_logit_margin,
                free_units,
                flips,
                lp_solves,
            } => format!(
                "exhausted (best pattern-space margin {best_logit_margin}, \
                 {free_units} free units, {flips} flips, {lp_solves} LP solves)"
            ),
            Self::SolverError(error) => format!("realizability LP failed: {error}"),
            Self::Candidate(candidate) => format!(
                "candidate (logit margin +{}, argmax {}, slack {:.4}, {} free units, \
                 {} flips, {} LP solves, {:.1}s) — PENDING the trusted-oracle gate",
                candidate.logit_margin,
                candidate.argmax,
                candidate.lp_slack,
                candidate.free_units,
                candidate.flips,
                candidate.lp_solves,
                candidate.elapsed.as_secs_f64(),
            ),
        }
    }
}

/// Minimum remaining wall clock worth starting the lane with.
///
/// The three banked `model_30` rows are recovered in 58-110 s on a 2026 laptop
/// (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md`), and the exact interval
/// prepass plus the layout/engine self-check alone cost a few seconds. Starting
/// under this floor spends budget the ordinary lanes could have used and cannot
/// realistically finish.
const MIN_LANE_BUDGET: Duration = Duration::from_secs(20);

/// Slack left to the caller between the lane's deadline and the instance
/// deadline, so the trusted-oracle gate (model load + ORT forward + true-`f64`
/// recheck) and results publication still fit.
const LANE_SAFETY_MARGIN: Duration = Duration::from_secs(45);

/// Hard cap on one lane consultation regardless of how long the instance is.
const LANE_WALL_CAP: Duration = Duration::from_mins(4);

/// Fraction of the remaining instance budget the lane may take.
const LANE_BUDGET_FRACTION: f64 = 0.5;

/// Wall-clock budget for one lane consultation, or `None` when the remaining
/// budget is too small to be worth spending.
fn lane_budget(remaining: Option<Duration>) -> Option<Duration> {
    let Some(remaining) = remaining else {
        return Some(LANE_WALL_CAP);
    };
    let usable = remaining.checked_sub(LANE_SAFETY_MARGIN)?;
    let budget = usable.mul_f64(LANE_BUDGET_FRACTION).min(LANE_WALL_CAP);
    (budget >= MIN_LANE_BUDGET).then_some(budget)
}

/// Consult the sign-space falsification lane.
///
/// Returns a CLAIMED counterexample or a reason the lane declined. NEVER a
/// verdict: see the module docs for the routing obligation this leaves with the
/// caller.
///
/// `config` is the category preset's `attack.bnn_sign_space`, resolved UNDER
/// the environment by [`sign_space_falsify_armed`]. On the disarmed arm this
/// returns [`SignSpaceLaneOutcome::Disarmed`] before touching `onnx` or
/// `vnnlib` at all, so an unarmed category pays nothing and behaves
/// identically.
pub(crate) fn run_sign_space_lane(
    onnx: &Path,
    vnnlib: &Path,
    remaining: Option<Duration>,
    config: Option<bool>,
) -> SignSpaceLaneOutcome {
    // THE GATE. Everything below this line — model load, property parse,
    // request construction, the core call — is unreachable on the dark arm.
    if !sign_space_falsify_armed(config) {
        return SignSpaceLaneOutcome::Disarmed;
    }
    let Some(budget) = lane_budget(remaining) else {
        return SignSpaceLaneOutcome::NotAdmitted(format!(
            "remaining budget {:?} leaves less than {MIN_LANE_BUDGET:?} after the \
             {LANE_SAFETY_MARGIN:?} publication margin",
            remaining
        ));
    };
    // PROPERTY FIRST, ON PURPOSE. Parsing the VNN-LIB is far cheaper than
    // loading and shape-inferring an ONNX graph, and the argmax-complement test
    // rejects most benchmark families outright. Ordering it first means an
    // ARMED run on an unrelated category pays a property parse, not a model
    // load, before falling through.
    let spec = match ny_onnx::vnnlib::load_vnnlib(vnnlib) {
        Ok(spec) => spec,
        Err(error) => {
            return SignSpaceLaneOutcome::NotAdmitted(format!("property parse failed: {error}"))
        }
    };
    let property = match ExtractedProperty::extract(&spec) {
        Ok(property) => property,
        Err(reason) => return SignSpaceLaneOutcome::NotAdmitted(reason),
    };
    let model = match ny_onnx::load_onnx(onnx) {
        Ok(model) => model,
        Err(error) => {
            return SignSpaceLaneOutcome::NotAdmitted(format!("model load failed: {error}"))
        }
    };
    let problem = match SignSpaceProblem::assemble(&model, &spec, property) {
        Ok(problem) => problem,
        Err(reason) => return SignSpaceLaneOutcome::NotAdmitted(reason),
    };
    problem.solve(budget)
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// A folded per-channel affine (a BatchNorm, or a `Mul`/`Add` constant pair).
#[derive(Debug, Clone)]
struct FoldedAffine {
    scale: Vec<f64>,
    offset: Vec<f64>,
}

/// Owned tensors for one BINARY stage beyond `conv2`.
#[derive(Debug)]
enum StageTensors {
    Conv {
        weights: Vec<f32>,
        // Boxed purely to keep the variant spread under clippy.toml's
        // `enum-variant-size-threshold = 64`: inline this made Conv 176 bytes
        // against Dense's 96. Field access auto-derefs, so no read site changes.
        geometry: Box<ConvGeometry>,
        bias: Option<Vec<f64>>,
        pool: Option<PoolSpec>,
        affine: Option<FoldedAffine>,
        add: f64,
    },
    Dense {
        weights: Vec<f32>,
        in_dim: usize,
        out_dim: usize,
        affine: Option<FoldedAffine>,
        add: f64,
    },
}

/// Owned tensors for one extracted problem, so the borrowed
/// [`SignSpaceRequest`] has something to point at.
struct SignSpaceProblem {
    input: InputGeometry,
    conv1_weights: Vec<f32>,
    conv1: ConvGeometry,
    pool1: Option<PoolSpec>,
    affine1: Option<FoldedAffine>,
    add1: f64,
    conv2_weights: Vec<f32>,
    conv2: ConvGeometry,
    pool2: Option<PoolSpec>,
    affine2: Option<FoldedAffine>,
    add2: f64,
    stages: Vec<StageTensors>,
    dense: Vec<f32>,
    num_classes: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    target_class: usize,
    challengers: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct ConvGeometry {
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
}

impl SignSpaceProblem {
    /// The acceptance tolerance THIS geometry needs.
    ///
    /// The core refuses any tolerance below `f32_replay_slack_floor(taps,
    /// max|pixel|)`, and that floor is geometry-specific: `~6.6e-3` at the
    /// shallow net's `3x3x3` (so the historical `0.05` stands, unchanged) and
    /// `~7.3e-2` at the deeper nets' `5x5x3`, where `0.05` is UNSOUND. Rather
    /// than hard-code a second constant, re-derive the floor from the tensors
    /// and keep a 1.5x margin over it.
    fn tolerance(&self) -> f64 {
        let taps = self.conv1.kernel_h * self.conv1.kernel_w * self.conv1.in_channels;
        let max_abs = self
            .lo
            .iter()
            .chain(self.hi.iter())
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        let floor = ny_mip::f32_replay_slack_floor(taps, max_abs);
        SignSpaceLimits::default().tolerance.max(floor * 1.5)
    }

    /// Which point each realizability round adopts from its LP primal.
    ///
    /// DARK BY DEFAULT: absent the override this is whatever the core ships
    /// (`SegmentMove::Vertex`), so the scored path is byte-identical to the
    /// configuration every banked row was measured on. `NY_BNN_SIGN_SPACE_
    /// MINIMAL_MOVE=1` selects the closed-form minimal segment move instead, so
    /// the two shapes can be A/B'd on the same row.
    ///
    /// Deliberately NOT a preset key: it is an A/B instrument, and promoting it
    /// to the config layer would let it reach a scored run before it has a
    /// measurement to stand on.
    fn segment_move(&self) -> SegmentMove {
        if ny_levers::read(&ny_levers::decls::dark_probes::BNN_SIGN_SPACE_MINIMAL_MOVE)
            .value
            .as_bool()
        {
            SegmentMove::MinimalTheta
        } else {
            SignSpaceLimits::default().segment_move
        }
    }

    /// Where the realizability LP is allowed to put the pixel vector.
    ///
    /// DARK BY DEFAULT: absent the lever this is whatever the core ships
    /// (`TrustRegion::FullBox` — the vnnlib box on every pixel column), so the
    /// scored path is byte-identical to the configuration every banked row was
    /// measured on. The three arms are A/B instruments for the SIDEWAYS half of
    /// the §10 wall, and like `MINIMAL_MOVE` this is deliberately NOT a preset
    /// key: promoting it to the config layer would let it reach a scored run
    /// before it has a measurement to stand on.
    fn trust_region(&self) -> TrustRegion {
        trust_region_from_value(
            ny_levers::read(&ny_levers::decls::dark_probes::BNN_SIGN_SPACE_TRUST_REGION)
                .value
                .as_str(),
        )
    }

    /// Build the request and hand it to the core.
    fn solve(&self, budget: Duration) -> SignSpaceLaneOutcome {
        let conv1 = ConvSpec::valid_unit_stride(
            &self.conv1_weights,
            self.conv1.out_channels,
            self.conv1.in_channels,
            self.conv1.kernel_h,
            self.conv1.kernel_w,
        );
        let conv2 = ConvSpec::valid_unit_stride(
            &self.conv2_weights,
            self.conv2.out_channels,
            self.conv2.in_channels,
            self.conv2.kernel_h,
            self.conv2.kernel_w,
        );
        let stages: Vec<BinaryStage<'_>> = self
            .stages
            .iter()
            .map(|stage| match stage {
                StageTensors::Conv {
                    weights,
                    geometry,
                    bias,
                    pool,
                    affine,
                    add,
                } => {
                    let mut conv = ConvSpec::valid_unit_stride(
                        weights,
                        geometry.out_channels,
                        geometry.in_channels,
                        geometry.kernel_h,
                        geometry.kernel_w,
                    );
                    if let Some(bias) = bias {
                        conv = conv.with_bias(bias);
                    }
                    BinaryStage::Conv {
                        conv,
                        pool: *pool,
                        affine: affine.as_ref().map(FoldedAffine::borrow),
                        activation: SignSpaceActivation::SignAddSign { add_constant: *add },
                    }
                }
                StageTensors::Dense {
                    weights,
                    in_dim,
                    out_dim,
                    affine,
                    add,
                } => BinaryStage::Dense {
                    weights,
                    in_dim: *in_dim,
                    out_dim: *out_dim,
                    affine: affine.as_ref().map(FoldedAffine::borrow),
                    activation: SignSpaceActivation::SignAddSign { add_constant: *add },
                },
            })
            .collect();
        let request = SignSpaceRequest {
            input: self.input,
            conv1,
            conv1_pool: self.pool1,
            // A folded BatchNorm is the deeper nets' shape. The extractor
            // hands over `gamma / sqrt(var + eps)` and
            // `beta - scale * mean` UNCHECKED; the CORE proves `scale > 0` per
            // channel and refuses otherwise, because a negative slope inverts
            // the downstream bit.
            conv1_affine: self.affine1.as_ref().map(FoldedAffine::borrow),
            activation1: SignSpaceActivation::SignAddSign {
                add_constant: self.add1,
            },
            conv2,
            conv2_pool: self.pool2,
            conv2_affine: self.affine2.as_ref().map(FoldedAffine::borrow),
            activation2: SignSpaceActivation::SignAddSign {
                add_constant: self.add2,
            },
            stages: &stages,
            dense: &self.dense,
            num_classes: self.num_classes,
            lo: &self.lo,
            hi: &self.hi,
            target_class: self.target_class,
            challengers: &self.challengers,
            // Box midpoint. Explicitly NOT reconstructed from a centre and an
            // epsilon: 2 of the 3 banked rows have non-integral centres and
            // widths in {1, 2}, and the core re-derives the midpoint from the
            // lo/hi we read straight out of the VNN-LIB.
            reference_input: None,
            // See the module docs: no usable pre-Softmax oracle exists here.
            reference_forward: None,
        };
        let limits = SignSpaceLimits {
            max_wall_time: budget,
            tolerance: self.tolerance(),
            segment_move: self.segment_move(),
            trust_region: self.trust_region(),
            ..SignSpaceLimits::default()
        };
        // The one and only entry to the search core.
        CORE_CALLS.fetch_add(1, Ordering::Relaxed);
        match ny_mip::falsify_bnn_sign_suffix_unwired(&request, &limits) {
            Ok(SignSpaceOutcome::Candidate(candidate)) => {
                SignSpaceLaneOutcome::Candidate(candidate)
            }
            Ok(SignSpaceOutcome::Exhausted {
                best_logit_margin,
                free_units,
                flips,
                lp_solves,
                ..
            }) => SignSpaceLaneOutcome::Exhausted {
                best_logit_margin,
                free_units,
                flips,
                lp_solves,
            },
            Ok(SignSpaceOutcome::Refused(refusal)) => {
                SignSpaceLaneOutcome::Refused(format!("{refusal:?}"))
            }
            Err(error) => SignSpaceLaneOutcome::SolverError(error.to_string()),
        }
    }

    /// Join the already-checked property to the binarized conv suffix of a
    /// loaded ONNX graph.
    ///
    /// FAIL-CLOSED throughout: every deviation from the expected shape is an
    /// `Err`, which the caller turns into a fall-through. Being strict here can
    /// only ever LOSE a `sat`; it cannot produce one.
    fn assemble(
        model: &OnnxModel,
        spec: &VnnLibSpec,
        property: ExtractedProperty,
    ) -> Result<Self, String> {
        let ExtractedProperty {
            target_class,
            challengers,
            num_classes,
        } = property;
        let input = extract_input_geometry(model, spec)?;
        let graph = extract_graph(model)?;
        if graph.num_classes != num_classes {
            return Err(format!(
                "dense layer produces {} classes, the property names {num_classes}",
                graph.num_classes
            ));
        }
        let pixels = input.height * input.width * input.channels;
        if spec.input_bounds.len() != pixels {
            return Err(format!(
                "property declares {} inputs, geometry {input:?} needs {pixels}",
                spec.input_bounds.len()
            ));
        }
        let lo: Vec<f64> = spec.input_bounds.iter().map(|&(lo, _)| lo).collect();
        let hi: Vec<f64> = spec.input_bounds.iter().map(|&(_, hi)| hi).collect();
        let ExtractedGraph {
            conv1_weights,
            conv1,
            pool1,
            affine1,
            add1,
            conv2_weights,
            conv2,
            pool2,
            affine2,
            add2,
            stages,
            dense,
            num_classes,
        } = graph;
        Ok(Self {
            input,
            conv1_weights,
            conv1,
            pool1,
            affine1,
            add1,
            conv2_weights,
            conv2,
            pool2,
            affine2,
            add2,
            stages,
            dense,
            num_classes,
            lo,
            hi,
            target_class,
            challengers,
        })
    }
}

impl FoldedAffine {
    fn borrow(&self) -> SignSpaceAffine<'_> {
        SignSpaceAffine {
            scale: &self.scale,
            offset: &self.offset,
        }
    }
}

/// The property side of the request: the argmax-complement target and its
/// challengers.
struct ExtractedProperty {
    target_class: usize,
    challengers: Vec<usize>,
    num_classes: usize,
}

impl ExtractedProperty {
    /// The strict argmax-complement disjunction, or a reason this is not one.
    ///
    /// The admitted shape is exactly the one the staged corpus produces and the
    /// one `ny-onnx`'s terminal-Softmax predicate already pins: a top-level
    /// disjunction of SINGLETON clauses, every atom a non-strict
    /// `GreaterEq(challenger, target)` over one shared target, and the challenger
    /// set exactly every other class.
    fn extract(spec: &VnnLibSpec) -> Result<Self, String> {
        // #witness-declared-bounds: a top-level assert constrains EVERY clause,
        // so a per-clause box would make `input_bounds` a widened union rather
        // than the real box. Refuse rather than search a superset of the box —
        // a witness outside the declared box is a false `sat` waiting to be
        // caught by the gate, and there is no reason to generate one.
        if !spec.per_clause_input_bounds.is_empty() {
            return Err("property carries per-clause input boxes".to_string());
        }
        if !spec.declared_input_bounds.is_empty() && spec.declared_input_bounds != spec.input_bounds
        {
            return Err(
                "declared input bounds differ from the effective bounds (clause union widening)"
                    .to_string(),
            );
        }
        if spec.dual_network.is_some() {
            return Err("property declares a dual network".to_string());
        }
        let (target_class, challengers, num_classes) = extract_argmax_complement(spec)?;
        Ok(Self {
            target_class,
            challengers,
            num_classes,
        })
    }
}

/// `(target, challengers, num_classes)` for a strict argmax-complement
/// disjunction, or a reason it is not one.
fn extract_argmax_complement(spec: &VnnLibSpec) -> Result<(usize, Vec<usize>, usize), String> {
    if !spec.is_disjunction {
        return Err("property is not a top-level disjunction".to_string());
    }
    let num_classes = spec.num_outputs;
    if num_classes < 2 {
        return Err(format!("property declares {num_classes} outputs"));
    }
    if spec.output_constraint_clauses.len() != num_classes - 1 {
        return Err(format!(
            "property has {} clauses, an argmax complement over {num_classes} classes needs {}",
            spec.output_constraint_clauses.len(),
            num_classes - 1
        ));
    }
    let mut target: Option<usize> = None;
    let mut challengers: Vec<usize> = Vec::with_capacity(num_classes - 1);
    for clause in &spec.output_constraint_clauses {
        let [OutputConstraint::GreaterEq(challenger, class)] = clause[..] else {
            return Err(format!(
                "clause {clause:?} is not a single `Y_i >= Y_t` atom"
            ));
        };
        match target {
            None => target = Some(class),
            Some(existing) if existing == class => {}
            Some(existing) => {
                return Err(format!(
                    "clauses name two different targets ({existing} and {class})"
                ))
            }
        }
        if challenger >= num_classes || class >= num_classes {
            return Err(format!(
                "clause names class index out of range for {num_classes} outputs"
            ));
        }
        challengers.push(challenger);
    }
    let target = target.ok_or_else(|| "property has no clauses".to_string())?;
    let mut sorted = challengers.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let expected: Vec<usize> = (0..num_classes).filter(|&c| c != target).collect();
    if sorted != expected {
        return Err(format!(
            "challenger set is not every class other than the target {target}"
        ));
    }
    // The flat list must be exactly the clause concatenation, or some atom
    // outside the disjunction constrains the property in a way we would ignore.
    if spec.output_constraints.len() != spec.output_constraint_clauses.len() {
        return Err("flat constraint list is not the clause concatenation".to_string());
    }
    Ok((target, challengers, num_classes))
}

/// NHWC input geometry from the graph's declared input tensor.
fn extract_input_geometry(model: &OnnxModel, spec: &VnnLibSpec) -> Result<InputGeometry, String> {
    let tensor = model
        .network
        .inputs
        .first()
        .ok_or_else(|| "graph declares no input".to_string())?;
    if model.network.inputs.len() != 1 {
        return Err(format!(
            "graph declares {} inputs; the lane admits exactly one",
            model.network.inputs.len()
        ));
    }
    // `[N, H, W, C]`; `N` may be symbolic (`-1`) but must not be a real batch.
    let [batch, height, width, channels] = tensor.shape[..] else {
        return Err(format!("input shape {:?} is not rank 4 NHWC", tensor.shape));
    };
    if batch > 1 {
        return Err(format!("input declares batch {batch}"));
    }
    let dims = [height, width, channels];
    if dims.iter().any(|&d| d <= 0) {
        return Err(format!(
            "input shape {:?} has a non-positive HWC",
            tensor.shape
        ));
    }
    let geometry = InputGeometry {
        height: height as usize,
        width: width as usize,
        channels: channels as usize,
    };
    let pixels = geometry.height * geometry.width * geometry.channels;
    if pixels != spec.num_inputs {
        return Err(format!(
            "graph input {geometry:?} has {pixels} elements, the property declares {}",
            spec.num_inputs
        ));
    }
    Ok(geometry)
}

/// The tensors of the admitted binarized suffix.
struct ExtractedGraph {
    conv1_weights: Vec<f32>,
    conv1: ConvGeometry,
    pool1: Option<PoolSpec>,
    affine1: Option<FoldedAffine>,
    add1: f64,
    conv2_weights: Vec<f32>,
    conv2: ConvGeometry,
    pool2: Option<PoolSpec>,
    affine2: Option<FoldedAffine>,
    add2: f64,
    stages: Vec<StageTensors>,
    dense: Vec<f32>,
    num_classes: usize,
}

/// One `Sign -> Add(c) -> Sign` composite, plus the optional per-channel
/// affine and pool that precede it.
struct ConvBlock {
    conv: usize,
    pool: Option<PoolSpec>,
    affine: Option<FoldedAffine>,
    add: f64,
    reshaped: bool,
}

struct DenseBlock {
    matmul: usize,
    affine: Option<FoldedAffine>,
    add: f64,
}

/// Hard cap on how many blocks the walker will parse before declining, so a
/// pathological graph cannot turn admission into a long walk.
const MAX_PARSED_BLOCKS: usize = 16;

/// Walk the loaded graph and match the binarized suffix BLOCK BY BLOCK.
///
/// The grammar, which is exactly what the three staged traffic-sign ONNX files
/// produce (15, 31 and 33 nodes) and nothing wider:
///
/// ```text
/// graph      := ConvBlock{2,}  DenseBlock*  MatMul  [Softmax]
/// ConvBlock  := Transpose(0,3,1,2) Conv [MaxPool] [BatchNormalization]
///               Transpose(0,2,3,1) [Reshape] SignPair
/// DenseBlock := MatMul [Mul(const) Add(const)] SignPair
/// SignPair   := Sign Add(scalar) Sign
/// ```
///
/// with the extra structural conditions that the `Reshape` occurs in EXACTLY
/// the last `ConvBlock` (it is the flatten into the dense head) and that the
/// trailing `MatMul` is the final classifier.
///
/// The NCHW/NHWC transpose pairs cancel: the core works in NHWC throughout and
/// the ONNX weights are already in `[out, in, kh, kw]`, which is exactly the
/// layout [`ConvSpec`] documents. Per-channel ops (`MaxPool` over H/W,
/// `BatchNormalization` over C) commute with that pair, which is why the
/// BatchNorm sitting on the NCHW side can be folded into an NHWC per-channel
/// affine unchanged. The `Reshape` flattens NHWC row-major, which is exactly
/// the core's `(row * W + col) * C + chan` flat index — this is the
/// transposition risk the module docs discuss, and it is checked end-to-end by
/// the corpus traffic test, not asserted here.
fn extract_graph(model: &OnnxModel) -> Result<ExtractedGraph, String> {
    use ny_core::LayerType as T;

    let layers = &model.network.layers;
    let mut at = 0usize;
    let mut conv_blocks: Vec<ConvBlock> = Vec::new();
    while at < layers.len() && layers[at].layer_type == T::Transpose {
        if conv_blocks.len() >= MAX_PARSED_BLOCKS {
            return Err(format!("more than {MAX_PARSED_BLOCKS} convolution blocks"));
        }
        let block = parse_conv_block(model, &mut at)?;
        conv_blocks.push(block);
    }
    if conv_blocks.len() < 2 {
        return Err(format!(
            "the fragment needs at least 2 convolution blocks, found {}",
            conv_blocks.len()
        ));
    }
    for (index, block) in conv_blocks.iter().enumerate() {
        let last = index + 1 == conv_blocks.len();
        if block.reshaped != last {
            return Err(format!(
                "convolution block {index} {} a Reshape; the flatten belongs to the last block only",
                if block.reshaped { "carries" } else { "lacks" }
            ));
        }
    }

    let mut dense_blocks: Vec<DenseBlock> = Vec::new();
    // A `MatMul` starts a dense block only when a `Sign` pair follows it; the
    // FINAL classifier is the one that is not followed by a `Sign`.
    while at < layers.len() && layers[at].layer_type == T::MatMul && !is_final_matmul(model, at) {
        if dense_blocks.len() >= MAX_PARSED_BLOCKS {
            return Err(format!("more than {MAX_PARSED_BLOCKS} dense blocks"));
        }
        dense_blocks.push(parse_dense_block(model, &mut at)?);
    }

    let final_matmul = expect_node(model, &mut at, T::MatMul)?;
    // A terminal Softmax is monotone and cannot change the argmax, so it may be
    // present or absent; nothing else may follow.
    if at < layers.len() && layers[at].layer_type == T::Softmax {
        at += 1;
    }
    if at != layers.len() {
        return Err(format!(
            "graph has {} trailing node(s) after the binarized suffix",
            layers.len() - at
        ));
    }
    // The node LIST being in the right order is not the same as the DATA
    // flowing through it in that order: a topologically ordered graph can still
    // fan out, skip, or feed a node from somewhere other than its predecessor.
    // Extracting weights from a graph that only looks like the chain would give
    // arithmetically plausible logits for a network that is not this one, so
    // require a real single-producer chain from the declared input to the
    // declared output.
    check_linear_chain(model)?;

    let (conv1_weights, conv1) = conv_tensor(model, conv_blocks[0].conv, "conv1")?;
    if model.network.layers[conv_blocks[0].conv].inputs.len() != 2 {
        return Err("conv1 carries a bias; the f32 replay bound is derived for unit taps".into());
    }
    let (conv2_weights, conv2) = conv_tensor(model, conv_blocks[1].conv, "conv2")?;
    let mut stages: Vec<StageTensors> = Vec::new();
    for block in &conv_blocks[2..] {
        let (weights, geometry) = conv_tensor(model, block.conv, "stage conv")?;
        let bias = conv_bias(model, block.conv)?;
        stages.push(StageTensors::Conv {
            weights,
            geometry: Box::new(geometry),
            bias,
            pool: block.pool,
            affine: block.affine.clone(),
            add: block.add,
        });
    }
    for block in &dense_blocks {
        let (weights, in_dim, out_dim) = matmul_tensor(model, block.matmul, "stage dense")?;
        stages.push(StageTensors::Dense {
            weights,
            in_dim,
            out_dim,
            affine: block.affine.clone(),
            add: block.add,
        });
    }
    let (dense, _, num_classes) = matmul_tensor(model, final_matmul, "dense")?;
    if conv2.in_channels != conv1.out_channels {
        return Err(format!(
            "conv2 takes {} channels, conv1 produces {}",
            conv2.in_channels, conv1.out_channels
        ));
    }
    Ok(ExtractedGraph {
        conv1_weights,
        conv1,
        pool1: conv_blocks[0].pool,
        affine1: conv_blocks[0].affine.clone(),
        add1: conv_blocks[0].add,
        conv2_weights,
        conv2,
        pool2: conv_blocks[1].pool,
        affine2: conv_blocks[1].affine.clone(),
        add2: conv_blocks[1].add,
        stages,
        dense,
        num_classes,
    })
}

/// Is the `MatMul` at `index` the terminal classifier (i.e. NOT followed by a
/// `Sign` composite)?
fn is_final_matmul(model: &OnnxModel, index: usize) -> bool {
    use ny_core::LayerType as T;
    let layers = &model.network.layers;
    let mut probe = index + 1;
    // Skip the optional `Mul`/`Add` BatchNorm pair.
    if probe + 1 < layers.len()
        && layers[probe].layer_type == T::Mul
        && layers[probe + 1].layer_type == T::Add
    {
        probe += 2;
    }
    probe >= layers.len() || layers[probe].layer_type != T::Sign
}

fn expect_node(
    model: &OnnxModel,
    at: &mut usize,
    want: ny_core::LayerType,
) -> Result<usize, String> {
    let layers = &model.network.layers;
    let index = *at;
    let layer = layers
        .get(index)
        .ok_or_else(|| format!("graph ends before the expected {want:?} at node {index}"))?;
    if layer.layer_type != want {
        return Err(format!(
            "node {index} is {:?}, expected {want:?}",
            layer.layer_type
        ));
    }
    *at += 1;
    Ok(index)
}

/// `Sign -> Add(scalar) -> Sign`, returning the scalar.
fn parse_sign_pair(model: &OnnxModel, at: &mut usize, which: &str) -> Result<f64, String> {
    use ny_core::LayerType as T;
    expect_node(model, at, T::Sign)?;
    let add = expect_node(model, at, T::Add)?;
    expect_node(model, at, T::Sign)?;
    scalar_initializer(model, add, which)
}

fn parse_conv_block(model: &OnnxModel, at: &mut usize) -> Result<ConvBlock, String> {
    use ny_core::LayerType as T;
    let layers = &model.network.layers;
    let t_in = expect_node(model, at, T::Transpose)?;
    check_perm(model, t_in, &[0, 3, 1, 2])?;
    let conv = expect_node(model, at, T::Conv2d)?;
    let pool = if *at < layers.len() && layers[*at].layer_type == T::MaxPool {
        let index = expect_node(model, at, T::MaxPool)?;
        Some(pool_spec(model, index)?)
    } else {
        None
    };
    let affine = if *at < layers.len() && layers[*at].layer_type == T::BatchNorm {
        let index = expect_node(model, at, T::BatchNorm)?;
        Some(batch_norm_affine(model, index)?)
    } else {
        None
    };
    let t_out = expect_node(model, at, T::Transpose)?;
    check_perm(model, t_out, &[0, 2, 3, 1])?;
    let reshaped = if *at < layers.len() && layers[*at].layer_type == T::Reshape {
        expect_node(model, at, T::Reshape)?;
        true
    } else {
        false
    };
    let add = parse_sign_pair(model, at, "conv block activation")?;
    Ok(ConvBlock {
        conv,
        pool,
        affine,
        add,
        reshaped,
    })
}

fn parse_dense_block(model: &OnnxModel, at: &mut usize) -> Result<DenseBlock, String> {
    use ny_core::LayerType as T;
    let layers = &model.network.layers;
    let matmul = expect_node(model, at, T::MatMul)?;
    let affine = if *at + 1 < layers.len()
        && layers[*at].layer_type == T::Mul
        && layers[*at + 1].layer_type == T::Add
    {
        let mul = expect_node(model, at, T::Mul)?;
        let add = expect_node(model, at, T::Add)?;
        Some(mul_add_affine(model, mul, add)?)
    } else {
        None
    };
    let add = parse_sign_pair(model, at, "dense block activation")?;
    Ok(DenseBlock {
        matmul,
        affine,
        add,
    })
}

/// Every node consumes its predecessor's single output, the first consumes the
/// declared graph input, and the last produces the declared graph output.
///
/// Inputs beyond the first are initializers (conv/dense weights, BatchNorm
/// statistics, the `Add` constant, the `Reshape` target shape) and are checked
/// where they are read.
fn check_linear_chain(model: &OnnxModel) -> Result<(), String> {
    let layers = &model.network.layers;
    let graph_input = &model
        .network
        .inputs
        .first()
        .ok_or_else(|| "graph declares no input".to_string())?
        .name;
    for (index, layer) in layers.iter().enumerate() {
        let produced = match layer.outputs.as_slice() {
            [single] => single,
            other => {
                return Err(format!(
                    "node {index} produces {} outputs; the chain admits exactly one",
                    other.len()
                ))
            }
        };
        let consumed = layer
            .inputs
            .first()
            .ok_or_else(|| format!("node {index} consumes nothing"))?;
        let expected = match index.checked_sub(1) {
            None => graph_input,
            Some(previous) => &layers[previous].outputs[0],
        };
        if consumed != expected {
            return Err(format!(
                "node {index} consumes `{consumed}`, not its predecessor's `{expected}`"
            ));
        }
        let _ = produced;
    }
    let last = layers
        .last()
        .ok_or_else(|| "graph has no nodes".to_string())?;
    let graph_output = &model
        .network
        .outputs
        .first()
        .ok_or_else(|| "graph declares no output".to_string())?
        .name;
    if &last.outputs[0] != graph_output {
        return Err(format!(
            "the chain ends at `{}`, not the declared graph output `{graph_output}`",
            last.outputs[0]
        ));
    }
    Ok(())
}

fn check_perm(model: &OnnxModel, index: usize, want: &[i64]) -> Result<(), String> {
    let layer = &model.network.layers[index];
    match layer.attributes.get("perm") {
        Some(ny_onnx::AttributeValue::Ints(perm)) if perm.as_slice() == want => Ok(()),
        other => Err(format!(
            "node {index} transposes with {other:?}, expected perm {want:?}"
        )),
    }
}

/// A `MaxPool` this lane can reason about: VALID, no dilation, `ceil_mode = 0`,
/// row-major storage order.
///
/// The pool sits BEFORE its BatchNorm and `Sign`, so it is a plain monotone max
/// over the pre-activations — that, plus a positive BatchNorm slope (proven per
/// channel in the core), is what makes `sign(BN(max_w z_w)) = OR_w [z_w >= t]`
/// hold. A pool AFTER the `Sign` would be a max over `+/-1`, which is a
/// different (and still monotone, but differently indexed) object, and this
/// walker never sees one because the grammar puts the pool inside the block.
fn pool_spec(model: &OnnxModel, index: usize) -> Result<PoolSpec, String> {
    let layer = &model.network.layers[index];
    let kernel = match layer.attributes.get("kernel_shape") {
        Some(ny_onnx::AttributeValue::Ints(values)) if values.len() == 2 => values.clone(),
        other => return Err(format!("MaxPool node {index} has kernel_shape {other:?}")),
    };
    let strides = match layer.attributes.get("strides") {
        None => vec![1, 1],
        Some(ny_onnx::AttributeValue::Ints(values)) if values.len() == 2 => values.clone(),
        other => return Err(format!("MaxPool node {index} has strides {other:?}")),
    };
    match layer.attributes.get("pads") {
        None => {}
        Some(ny_onnx::AttributeValue::Ints(values)) if values.iter().all(|&v| v == 0) => {}
        other => {
            return Err(format!(
                "MaxPool node {index} has pads {other:?}, expected VALID"
            ))
        }
    }
    match layer.attributes.get("dilations") {
        None => {}
        Some(ny_onnx::AttributeValue::Ints(values)) if values.iter().all(|&v| v == 1) => {}
        other => return Err(format!("MaxPool node {index} has dilations {other:?}")),
    }
    match layer.attributes.get("auto_pad") {
        None => {}
        Some(ny_onnx::AttributeValue::String(mode)) if mode == "NOTSET" || mode == "VALID" => {}
        other => return Err(format!("MaxPool node {index} has auto_pad {other:?}")),
    }
    for name in ["ceil_mode", "storage_order"] {
        match layer.attributes.get(name) {
            None => {}
            Some(ny_onnx::AttributeValue::Int(0)) => {}
            other => return Err(format!("MaxPool node {index} has {name} {other:?}")),
        }
    }
    if layer.inputs.len() != 1 || layer.outputs.len() != 1 {
        return Err(format!(
            "MaxPool node {index} has {}/{} inputs/outputs; the lane admits 1/1 (an Indices \
             output means the argmax is consumed somewhere)",
            layer.inputs.len(),
            layer.outputs.len()
        ));
    }
    let dims: Vec<usize> = kernel
        .iter()
        .chain(strides.iter())
        .map(|&v| usize::try_from(v).unwrap_or(0))
        .collect();
    if dims.contains(&0) {
        return Err(format!(
            "MaxPool node {index} has non-positive kernel/stride {kernel:?}/{strides:?}"
        ));
    }
    Ok(PoolSpec {
        kernel_h: dims[0],
        kernel_w: dims[1],
        stride_h: dims[2],
        stride_w: dims[3],
    })
}

/// Fold an ONNX `BatchNormalization` into a per-channel affine.
///
/// Input order is `(X, scale, B, mean, var)` and inference-mode semantics are
/// `y = scale * (x - mean) / sqrt(var + eps) + B`, so
///
/// ```text
///   slope_c  = scale_c / sqrt(var_c + eps)
///   offset_c = B_c - slope_c * mean_c
/// ```
///
/// NOTHING is asserted about the SIGN of `slope_c` here: the core proves
/// `slope_c > 0` per channel and refuses otherwise. A negative slope inverts
/// the bit and would silently corrupt every threshold downstream, which is
/// exactly why the proof obligation lives with the module that owns the
/// threshold algebra rather than with this reader.
fn batch_norm_affine(model: &OnnxModel, index: usize) -> Result<FoldedAffine, String> {
    let layer = &model.network.layers[index];
    if layer.inputs.len() != 5 {
        return Err(format!(
            "BatchNormalization node {index} has {} inputs, expected 5",
            layer.inputs.len()
        ));
    }
    // Training-mode BatchNorm has three outputs and a different meaning.
    if layer.outputs.len() != 1 {
        return Err(format!(
            "BatchNormalization node {index} has {} outputs; only inference mode is admitted",
            layer.outputs.len()
        ));
    }
    let epsilon = match layer.attributes.get("epsilon") {
        None => 1e-5f64,
        Some(ny_onnx::AttributeValue::Float(value)) => f64::from(*value),
        other => {
            return Err(format!(
                "BatchNormalization node {index} has epsilon {other:?}"
            ))
        }
    };
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(format!(
            "BatchNormalization node {index} has epsilon {epsilon}"
        ));
    }
    let read = |slot: usize, what: &str| -> Result<Vec<f64>, String> {
        let name = &layer.inputs[slot];
        let tensor = model.weights.get(name).ok_or_else(|| {
            format!("BatchNormalization node {index} {what} `{name}` is not an initializer")
        })?;
        Ok(tensor.iter().copied().map(f64::from).collect())
    };
    let gamma = read(1, "scale")?;
    let beta = read(2, "B")?;
    let mean = read(3, "mean")?;
    let var = read(4, "var")?;
    let channels = gamma.len();
    if channels == 0 || beta.len() != channels || mean.len() != channels || var.len() != channels {
        return Err(format!(
            "BatchNormalization node {index} has inconsistent statistic lengths \
             {}/{}/{}/{}",
            gamma.len(),
            beta.len(),
            mean.len(),
            var.len()
        ));
    }
    let mut scale = Vec::with_capacity(channels);
    let mut offset = Vec::with_capacity(channels);
    for c in 0..channels {
        let denominator = (var[c] + epsilon).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(format!(
                "BatchNormalization node {index} channel {c} has var {} (+ eps {epsilon})",
                var[c]
            ));
        }
        let slope = gamma[c] / denominator;
        let shift = beta[c] - slope * mean[c];
        if !slope.is_finite() || !shift.is_finite() {
            return Err(format!(
                "BatchNormalization node {index} channel {c} folds to a non-finite affine"
            ));
        }
        scale.push(slope);
        offset.push(shift);
    }
    Ok(FoldedAffine { scale, offset })
}

/// Fold a `Mul(const) -> Add(const)` pair into a per-channel affine.
///
/// This is how the exporter emits the BatchNorm that follows a dense layer:
/// `Mul` carries `gamma / sqrt(var + eps)` and `Add` carries
/// `beta - mean * gamma / sqrt(var + eps)`, already combined. Both must be
/// rank-1 vectors of the same width; a scalar or a broadcast of a different
/// shape is refused rather than guessed at.
fn mul_add_affine(model: &OnnxModel, mul: usize, add: usize) -> Result<FoldedAffine, String> {
    let read = |index: usize, what: &str| -> Result<Vec<f64>, String> {
        let layer = &model.network.layers[index];
        if layer.inputs.len() != 2 {
            return Err(format!(
                "{what} node {index} has {} inputs, expected 2",
                layer.inputs.len()
            ));
        }
        let name = &layer.inputs[1];
        let tensor = model.weights.get(name).ok_or_else(|| {
            format!("{what} node {index} constant `{name}` is not an initializer")
        })?;
        if tensor.shape().len() != 1 {
            return Err(format!(
                "{what} node {index} constant has shape {:?}, expected a per-channel vector",
                tensor.shape()
            ));
        }
        Ok(tensor.iter().copied().map(f64::from).collect())
    };
    let scale = read(mul, "Mul")?;
    let offset = read(add, "Add")?;
    if scale.is_empty() || scale.len() != offset.len() {
        return Err(format!(
            "Mul/Add BatchNorm at nodes {mul}/{add} has {}/{} channels",
            scale.len(),
            offset.len()
        ));
    }
    Ok(FoldedAffine { scale, offset })
}

/// The convolution's initializer weights plus its geometry, refusing anything
/// that is not stride-1 / VALID / dilation-1 / single-group.
///
/// The core re-checks all of this (and the per-channel `|W|` test) from the
/// `ConvSpec` it is handed; doing it here too means an unsupported geometry
/// declines with a message naming the ONNX node rather than a tensor.
fn conv_tensor(
    model: &OnnxModel,
    index: usize,
    which: &str,
) -> Result<(Vec<f32>, ConvGeometry), String> {
    let layer = &model.network.layers[index];
    for (name, want) in [("strides", 1i64), ("dilations", 1i64)] {
        match layer.attributes.get(name) {
            None => {}
            Some(ny_onnx::AttributeValue::Ints(values)) if values.iter().all(|&v| v == want) => {}
            Some(other) => {
                return Err(format!("{which} has {name} {other:?}, expected all {want}"))
            }
        }
    }
    match layer.attributes.get("pads") {
        None => {}
        Some(ny_onnx::AttributeValue::Ints(values)) if values.iter().all(|&v| v == 0) => {}
        Some(other) => return Err(format!("{which} has pads {other:?}, expected VALID")),
    }
    match layer.attributes.get("auto_pad") {
        None => {}
        Some(ny_onnx::AttributeValue::String(mode)) if mode == "NOTSET" || mode == "VALID" => {}
        Some(other) => return Err(format!("{which} has auto_pad {other:?}")),
    }
    match layer.attributes.get("group") {
        None => {}
        Some(ny_onnx::AttributeValue::Int(1)) => {}
        Some(other) => return Err(format!("{which} has group {other:?}, expected 1")),
    }
    if layer.inputs.len() != 2 && layer.inputs.len() != 3 {
        return Err(format!(
            "{which} has {} inputs; a convolution has 2 (bias-free) or 3",
            layer.inputs.len()
        ));
    }
    let tensor = model.weights.get(&layer.inputs[1]).ok_or_else(|| {
        format!(
            "{which} weights `{}` are not an initializer",
            layer.inputs[1]
        )
    })?;
    let [out_channels, in_channels, kernel_h, kernel_w] = tensor.shape()[..] else {
        return Err(format!(
            "{which} weights have shape {:?}, expected rank 4",
            tensor.shape()
        ));
    };
    let weights = tensor
        .as_slice()
        .ok_or_else(|| format!("{which} weights are not contiguous"))?
        .to_vec();
    Ok((
        weights,
        ConvGeometry {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
        },
    ))
}

/// The convolution's per-output-channel bias, if it declares one.
fn conv_bias(model: &OnnxModel, index: usize) -> Result<Option<Vec<f64>>, String> {
    let layer = &model.network.layers[index];
    let Some(name) = layer.inputs.get(2) else {
        return Ok(None);
    };
    let tensor = model
        .weights
        .get(name)
        .ok_or_else(|| format!("conv bias `{name}` is not an initializer"))?;
    if tensor.shape().len() != 1 {
        return Err(format!(
            "conv bias `{name}` has shape {:?}, expected a per-channel vector",
            tensor.shape()
        ));
    }
    Ok(Some(tensor.iter().copied().map(f64::from).collect()))
}

/// The `Sign -> Add(c) -> Sign` constant. Must be an exact scalar initializer:
/// the core admits the composite only for `0 < c < 1`, which is precisely the
/// condition making `B(v) = +1 iff v >= 0`.
fn scalar_initializer(model: &OnnxModel, index: usize, which: &str) -> Result<f64, String> {
    let layer = &model.network.layers[index];
    if layer.inputs.len() != 2 {
        return Err(format!("{which} Add has {} inputs", layer.inputs.len()));
    }
    let tensor = model
        .weights
        .get(&layer.inputs[1])
        .ok_or_else(|| format!("{which} Add constant is not an initializer"))?;
    if tensor.len() != 1 {
        return Err(format!(
            "{which} Add constant has {} elements, expected a scalar",
            tensor.len()
        ));
    }
    Ok(f64::from(tensor.iter().next().copied().unwrap_or(f32::NAN)))
}

/// A `[in_dim, out_dim]` initializer, row-major — exactly the layout
/// [`SignSpaceRequest::dense`] and [`BinaryStage::Dense`] document.
fn matmul_tensor(
    model: &OnnxModel,
    index: usize,
    which: &str,
) -> Result<(Vec<f32>, usize, usize), String> {
    let layer = &model.network.layers[index];
    if layer.inputs.len() != 2 {
        return Err(format!("{which} MatMul has {} inputs", layer.inputs.len()));
    }
    let tensor = model
        .weights
        .get(&layer.inputs[1])
        .ok_or_else(|| format!("{which} weights are not an initializer"))?;
    let [in_dim, out_dim] = tensor.shape()[..] else {
        return Err(format!(
            "{which} weights have shape {:?}, expected rank 2",
            tensor.shape()
        ));
    };
    let weights = tensor
        .as_slice()
        .ok_or_else(|| format!("{which} weights are not contiguous"))?
        .to_vec();
    Ok((weights, in_dim, out_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) THE UNARMED CONTRACT, asserted on a code path.
    ///
    /// With neither the lever nor a preset the lane must decline BEFORE it
    /// constructs anything. Two independent witnesses of that here, neither of
    /// them a log line: the returned variant is `Disarmed`, which is reachable
    /// from exactly one `return` in `run_sign_space_lane` — the one above every
    /// model/property/request construction — and the search core's own
    /// invocation counter does not move.
    ///
    /// The paths handed in DO NOT EXIST. If the gate were bypassed the lane
    /// would have to reach `load_onnx` and come back `NotAdmitted`, so this
    /// also rules out an accidental "armed but failed early" pass.
    #[test]
    fn unset_lever_never_constructs_the_falsifier() {
        assert!(
            !sign_space_falsify_armed_from(None, None),
            "an absent NY_BNN_SIGN_SPACE and a silent preset must leave the lane dark"
        );
        for config in [None, Some(false)] {
            let before = core_call_count();
            let outcome = run_sign_space_lane(
                Path::new("/nonexistent/sign-space/model.onnx"),
                Path::new("/nonexistent/sign-space/property.vnnlib"),
                Some(Duration::from_mins(8)),
                config,
            );
            assert!(
                matches!(outcome, SignSpaceLaneOutcome::Disarmed),
                "unarmed lane must short-circuit before any load; got {}",
                outcome.describe()
            );
            assert_eq!(
                core_call_count(),
                before,
                "the search core must not be entered with the lane unarmed"
            );
        }
    }

    /// (b) EXACT-`"1"` SEMANTICS. Anything else is OFF — not trimmed, not
    /// case-folded, not truthy-parsed. These are the exact strings the lever
    /// chokepoint's own contract names, re-asserted through THIS reader so a
    /// future local parser cannot widen them.
    ///
    /// Both halves are asserted against a SILENT preset (the historical shape)
    /// and against a preset that ASKED for the lane. The second half is the
    /// one that matters now that a preset can arm it: a near-miss token must
    /// not ride the preset's `true` into an armed lane, and `"0"` must be able
    /// to kill it.
    #[test]
    fn only_the_single_character_one_arms_the_lane() {
        for config in [None, Some(false)] {
            assert!(sign_space_falsify_armed_from(Some("1"), config));
            for off in [
                "", "0", "01", "true", "on", " 1", "1 ", "TRUE", "yes", "2", "-1",
            ] {
                assert!(
                    !sign_space_falsify_armed_from(Some(off), config),
                    "{off:?} must not arm NY_BNN_SIGN_SPACE (preset {config:?})"
                );
            }
        }
        // A preset that armed the lane does not widen the token set, and the
        // one admissible off-token still kills it.
        assert!(sign_space_falsify_armed_from(Some("1"), Some(true)));
        assert!(
            !sign_space_falsify_armed_from(Some("0"), Some(true)),
            "NY_BNN_SIGN_SPACE=0 must disarm a preset-armed lane"
        );
        for off in ["", "01", "true", "on", " 1", "1 ", "TRUE", "yes", "2", "-1"] {
            assert!(
                !sign_space_falsify_armed_from(Some(off), Some(true)),
                "{off:?} must suppress the preset and fall back to the declaration default"
            );
        }
    }

    /// (b'') THE TRUST-REGION LEVER IS DARK BY DEFAULT AND HAS NO PRESET ROUTE.
    ///
    /// With the variable absent the LP gets the vnnlib box on every pixel
    /// column, which is the configuration every banked traffic-signs row was
    /// measured on. Exactly three tokens select a region; everything else —
    /// including the near misses that a hand-rolled `contains`/`trim` parser
    /// would have accepted — is a recorded rejection that falls back to it.
    #[test]
    fn the_trust_region_lever_is_dark_by_default_and_only_three_tokens_arm_it() {
        assert_eq!(
            trust_region_from(None),
            TrustRegion::FullBox,
            "with NY_BNN_SIGN_SPACE_TRUST_REGION absent the LP must get the whole box"
        );
        assert_eq!(
            SignSpaceLimits::default().trust_region,
            TrustRegion::FullBox
        );
        assert_eq!(
            trust_region_from(Some("box")),
            TrustRegion::Doubling {
                initial_fraction: 0.125
            }
        );
        assert_eq!(
            trust_region_from(Some("tight")),
            TrustRegion::Doubling {
                initial_fraction: 0.015_625
            }
        );
        assert_eq!(
            trust_region_from(Some("linf")),
            TrustRegion::Nearest {
                initial_fraction: 0.015_625,
                refine: 4
            }
        );
        for off in [
            "", "1", "0", "BOX", "Box", " box", "box ", "boxes", "tight ", "l-inf", "linf1",
            "true", "full", "FullBox",
        ] {
            assert_eq!(
                trust_region_from(Some(off)),
                TrustRegion::FullBox,
                "{off:?} must not arm a trust region"
            );
        }
    }

    /// (b') THE PRESET ROUTE ARMS THE LANE WITH NO ENVIRONMENT AT ALL.
    ///
    /// The whole point of the typed key: a scored competition run exports no
    /// `NY_*`, so this is the arming path that produces a real score. The
    /// companion end-to-end assertion — that the SHIPPED
    /// `traffic_signs_recognition_2023` preset carries the key and the real
    /// binary honours it — is
    /// `tests/bnn_sign_space_traffic.rs::preset_arms_the_lane_on_the_scored_path_with_no_env_var`.
    #[test]
    fn the_category_preset_arms_the_lane_with_no_env_var() {
        assert!(
            sign_space_falsify_armed_from(None, Some(true)),
            "attack.bnn_sign_space: true must arm the lane with NY_BNN_SIGN_SPACE absent"
        );
        // And it really reaches the lane, not just the predicate: with the
        // preset armed the gate is passed and the lane goes on to decline these
        // nonexistent paths rather than short-circuiting at `Disarmed`.
        let before = core_call_count();
        let outcome = run_sign_space_lane(
            Path::new("/nonexistent/sign-space/model.onnx"),
            Path::new("/nonexistent/sign-space/property.vnnlib"),
            Some(Duration::from_mins(8)),
            Some(true),
        );
        assert!(
            matches!(outcome, SignSpaceLaneOutcome::NotAdmitted(_)),
            "a preset-armed lane must reach the extraction and decline there; got {}",
            outcome.describe()
        );
        assert_eq!(
            core_call_count(),
            before,
            "a property that cannot even be parsed must not reach the search core"
        );
    }

    /// (c) A REFUSAL IS NOT A VERDICT AND CANNOT BECOME ONE.
    ///
    /// `candidate_input` is the only way anything reaches the witness gate, and
    /// it is `None` for every outcome but `Candidate`. So a `Refused` (or
    /// `Exhausted`, or `NotAdmitted`, or `SolverError`, or `Disarmed`) leaves
    /// the caller with exactly what the unwired path gave it. The call-site half
    /// of this — that `None` means the gate is never consulted and the normal
    /// verification path runs — is pinned in `commands/vnncomp.rs`'s
    /// `sign_space_lane_is_verdict_neutral_unless_it_has_a_candidate`.
    #[test]
    fn no_outcome_but_a_candidate_can_reach_the_witness_gate() {
        let outcomes = [
            SignSpaceLaneOutcome::Disarmed,
            SignSpaceLaneOutcome::NotAdmitted("budget".into()),
            SignSpaceLaneOutcome::Refused("CompositeActivationNotBinary".into()),
            SignSpaceLaneOutcome::Exhausted {
                best_logit_margin: -102,
                free_units: 488,
                flips: 250,
                lp_solves: 1234,
            },
            SignSpaceLaneOutcome::SolverError("lowering failed".into()),
        ];
        for outcome in &outcomes {
            assert!(
                outcome.candidate_input().is_none(),
                "{} must not yield a publishable witness",
                outcome.describe()
            );
        }
    }

    /// The type itself cannot express a proof. Mirrors `ny-mip`'s
    /// `the_outcome_enum_has_no_verified_variant`, on this side of the wiring:
    /// if someone adds a verdict-shaped variant to `SignSpaceLaneOutcome`, this
    /// match stops compiling.
    #[test]
    fn the_lane_cannot_produce_a_verified_outcome() {
        fn is_falsification_only(outcome: &SignSpaceLaneOutcome) -> bool {
            match outcome {
                SignSpaceLaneOutcome::Candidate(_)
                | SignSpaceLaneOutcome::Disarmed
                | SignSpaceLaneOutcome::NotAdmitted(_)
                | SignSpaceLaneOutcome::Refused(_)
                | SignSpaceLaneOutcome::Exhausted { .. }
                | SignSpaceLaneOutcome::SolverError(_) => true,
            }
        }
        assert!(is_falsification_only(&SignSpaceLaneOutcome::Disarmed));
    }

    /// The budget policy never starts a run it cannot finish and never eats the
    /// publication margin.
    #[test]
    fn lane_budget_respects_the_publication_margin() {
        assert_eq!(lane_budget(None), Some(LANE_WALL_CAP));
        // 480 s scored budget: (480 - 45) / 2 = 217.5 s, under the 240 s cap.
        assert_eq!(
            lane_budget(Some(Duration::from_mins(8))),
            Some(Duration::from_secs_f64(217.5))
        );
        // Long budget clamps to the wall cap.
        assert_eq!(
            lane_budget(Some(Duration::from_hours(1))),
            Some(LANE_WALL_CAP)
        );
        // Not enough left after the margin to reach the floor.
        assert_eq!(lane_budget(Some(Duration::from_secs(80))), None);
        assert_eq!(lane_budget(Some(Duration::from_secs(45))), None);
        assert_eq!(lane_budget(Some(Duration::from_secs(1))), None);
    }
}
