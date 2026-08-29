// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Laddered model-level verification driver.
//!
//! `verify_model` runs a single graph network through an escalating ladder of
//! propagation methods, from the cheapest/loosest to the most expensive/tightest,
//! stopping as soon as a rung produces a decisive verdict:
//!
//! ```text
//! IBP -> alpha-CROWN -> CROWN -> beta-CROWN -> (MIP, feature = "complete")
//! ```
//!
//! Escalation is *demand-driven*: a higher rung is only attempted when the
//! previous rung left the property unproven, the observed worst-case bound width
//! exceeds [`LadderConfig::escalation_width_threshold`], and the rung is allowed
//! by [`LadderConfig::max_method`]. The driver records a [`RungOutcome`] per
//! attempt so callers can audit exactly which methods ran and how tight they got.
//!
//! Soundness is preserved end-to-end: the ladder never reports `Verified` more
//! strongly than the underlying method achieved, it threads
//! [`SoundnessProvenance`] through every rung with [`SoundnessProvenance::combine`],
//! and a complete method (beta-CROWN / MIP) returning `Violated` short-circuits
//! the ladder with that definitive counterexample.
//!
//! ```rust,no_run
//! # #[cfg(feature = "propagate")]
//! # fn run() -> ny_core::Result<()> {
//! use ny_api::graph::GraphNetwork;
//! use ny_api::ladder::{verify_model, LadderConfig};
//! use ny_core::VerificationSpec;
//! # let net: GraphNetwork = unimplemented!();
//! # let spec: VerificationSpec = unimplemented!();
//! let outcome = verify_model(&net, &spec, &LadderConfig::default())?;
//! println!("verdict via {}: {} rungs", outcome.method_used, outcome.rungs.len());
//! # Ok(())
//! # }
//! ```

use ny_core::{
    nan_propagating_max, Bound, MethodUsed, NyError, Result, SoundnessProvenance,
    VerificationResult, VerificationSpec,
};

use crate::graph::GraphNetwork;
use crate::verify::{PropagationConfig, PropagationMethod, Verifier};

/// Configuration for the laddered model-level driver.
///
/// The defaults run the full propagation ladder up to beta-CROWN, escalate only
/// when bounds remain very loose, and leave the MIP terminal disabled.
#[derive(Debug, Clone)]
pub struct LadderConfig {
    /// Highest rung the ladder is permitted to climb to.
    ///
    /// The driver never attempts a method tighter than this. For example,
    /// `max_method = PropagationMethod::Crown` caps escalation at CROWN even when
    /// the property is still unproven.
    pub max_method: PropagationMethod,
    /// Worst-case bound width above which escalation to the next rung is allowed.
    ///
    /// After a rung leaves the property unproven, the driver inspects the widest
    /// output bound it produced. Escalation proceeds only if that width strictly
    /// exceeds this threshold (i.e. the bounds are still loose enough that a
    /// tighter method might help). A small/zero threshold makes the ladder
    /// essentially always escalate; a very large threshold makes it stop early.
    pub escalation_width_threshold: f32,
    /// Whether to attempt the complete MIP terminal after beta-CROWN.
    ///
    /// Only has an effect when the crate is built with `feature = "complete"`.
    /// When the feature is off this flag is accepted but ignored (the MIP rung
    /// gracefully degrades to a no-op).
    ///
    /// # This flag can make [`verify_model`] fail
    ///
    /// The exported MIP encoder covers only a sequential `Linear (ReLU Linear)*`
    /// chain. Asking for the complete terminal on a graph outside that fragment
    /// is a request the driver cannot honour, and it REFUSES with
    /// [`NyError::UnsupportedOp`] rather than recording a rung it never ran.
    /// Leave this `false` (the default) for callers that want the propagation
    /// verdict regardless of topology.
    pub use_complete: bool,
    /// Optional per-rung timeout in milliseconds.
    ///
    /// When set, it is applied to the spec used for each rung (overriding the
    /// spec's own timeout) so no single method can stall the ladder.
    ///
    /// It also becomes the per-solve wall-clock budget of the complete MIP
    /// terminal: it is the only time preference the caller expresses, so the
    /// terminal honours it rather than silently running to `ny-mip`'s much
    /// larger default. Callers that need a MIP-specific budget should drive
    /// [`crate::complete::verify_graph_complete_mip`] directly with their own
    /// [`CompleteMipOptions`](crate::complete::CompleteMipOptions).
    pub timeout_ms: Option<u64>,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            max_method: PropagationMethod::BetaCrown,
            escalation_width_threshold: 1e6,
            use_complete: false,
            timeout_ms: None,
        }
    }
}

/// Outcome of a single rung (method attempt) in the ladder.
#[derive(Debug, Clone)]
pub struct RungOutcome {
    /// The method this rung actually ran (as reported by the verifier, which may
    /// differ from the requested method when the verifier falls back internally).
    pub method: MethodUsed,
    /// Whether this rung proved the property.
    pub verified: bool,
    /// Widest output-bound width observed at this rung, if bounds were produced.
    pub max_width: Option<f32>,
    /// Human-readable note describing what happened at this rung.
    pub note: String,
}

/// Result of a laddered verification run.
#[derive(Debug, Clone)]
pub struct LadderedResult {
    /// The decisive verification result returned by the ladder.
    pub result: VerificationResult,
    /// The method that produced [`Self::result`].
    pub method_used: MethodUsed,
    /// One [`RungOutcome`] per method attempted, in escalation order.
    pub rungs: Vec<RungOutcome>,
}

/// Compute the widest bound width across a slice of output bounds.
///
/// Returns `None` for an empty slice. Non-finite widths (from infinite bounds)
/// are propagated as-is so callers can recognize exploded bounds.
fn max_bound_width(bounds: &[Bound]) -> Option<f32> {
    bounds
        .iter()
        .map(Bound::width)
        .fold(None, |acc, width| match acc {
            Some(previous) => Some(nan_propagating_max(previous, width)),
            None => Some(width),
        })
}

/// Extract the bounds a result carries (certified, best-effort, or partial).
fn result_bounds(result: &VerificationResult) -> Option<&[Bound]> {
    match result {
        VerificationResult::Verified { output_bounds, .. } => Some(output_bounds.as_slice()),
        VerificationResult::Unknown { bounds, .. } => Some(bounds.as_slice()),
        VerificationResult::Timeout { partial_bounds, .. } => {
            partial_bounds.as_deref().map(|b| b as &[Bound])
        }
        VerificationResult::Violated { .. } => None,
    }
}

/// Map a requested [`PropagationMethod`] to its [`MethodUsed`] tag.
///
/// `PropagationMethod` does not expose a public `Into<MethodUsed>` conversion,
/// so the ladder mirrors the mapping locally (kept in sync with
/// `ny_propagate::types::PropagationMethod::method_used`).
fn method_used_for(method: PropagationMethod) -> MethodUsed {
    match method {
        PropagationMethod::Ibp => MethodUsed::Ibp,
        PropagationMethod::Crown => MethodUsed::Crown,
        PropagationMethod::AlphaCrown => MethodUsed::AlphaCrown,
        PropagationMethod::SdpCrown => MethodUsed::SdpCrown,
        PropagationMethod::BetaCrown => MethodUsed::BetaCrown,
    }
}

/// The method tag a result reports, falling back to the requested method when the
/// verifier recorded none.
fn result_method(result: &VerificationResult, requested: MethodUsed) -> MethodUsed {
    result.actual_method_tag().cloned().unwrap_or(requested)
}

/// A `Violated` verdict from a *complete* method is a definitive refutation that
/// must short-circuit the ladder (a counterexample exists).
fn is_definitive_violation(result: &VerificationResult) -> bool {
    matches!(result, VerificationResult::Violated { .. })
}

/// Build the per-rung spec, applying the ladder-level timeout override if set.
fn rung_spec(spec: &VerificationSpec, cfg: &LadderConfig) -> Result<VerificationSpec> {
    match cfg.timeout_ms {
        Some(ms) => VerificationSpec::from_parts(
            spec.input_bounds().to_vec(),
            spec.output_bounds().to_vec(),
            Some(ms),
            spec.input_shape().map(<[usize]>::to_vec),
        ),
        None => Ok(spec.clone()),
    }
}

/// Run one rung: verify the graph with `method` and turn the verdict into a
/// `(result, RungOutcome)` pair.
fn run_rung(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    method: PropagationMethod,
) -> Result<(VerificationResult, RungOutcome)> {
    let verifier = Verifier::new(PropagationConfig {
        method,
        ..Default::default()
    });
    let result = verifier.verify_graph(net, spec)?;

    let actual = result_method(&result, method_used_for(method));
    let verified = result.is_verified();
    let max_width = result_bounds(&result).and_then(max_bound_width);
    let note = match &result {
        VerificationResult::Verified { .. } => "verified".to_string(),
        VerificationResult::Violated { .. } => "violated (counterexample found)".to_string(),
        VerificationResult::Unknown { reason, .. } => format!("unknown: {reason}"),
        VerificationResult::Timeout { .. } => "timeout".to_string(),
    };

    let outcome = RungOutcome {
        method: actual,
        verified,
        max_width,
        note,
    };
    Ok((result, outcome))
}

/// Convert a `PropagationMethod` to a stable rank for ladder ordering / capping.
fn rung_rank(method: PropagationMethod) -> u8 {
    match method {
        PropagationMethod::Ibp => 0,
        PropagationMethod::Crown => 2,
        PropagationMethod::AlphaCrown => 1,
        PropagationMethod::SdpCrown => 2,
        PropagationMethod::BetaCrown => 3,
    }
}

/// Whether `candidate` is allowed under the configured ceiling `max_method`.
fn rung_allowed(candidate: PropagationMethod, max_method: PropagationMethod) -> bool {
    rung_rank(candidate) <= rung_rank(max_method)
}

/// Whether the observed bound width justifies escalating to a tighter method.
///
/// Escalates when there are no bounds to inspect (decisively unknown, give the
/// tighter method a chance) or when the widest bound strictly exceeds the
/// threshold. A `NaN` width (corrupt/exploded) also escalates.
fn should_escalate(max_width: Option<f32>, threshold: f32) -> bool {
    match max_width {
        None => true,
        Some(w) => w.is_nan() || w > threshold,
    }
}

fn validate_config(cfg: &LadderConfig) -> Result<()> {
    if cfg.escalation_width_threshold.is_nan() || cfg.escalation_width_threshold < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "escalation_width_threshold must be non-negative and not NaN, got {}",
            cfg.escalation_width_threshold
        )));
    }
    Ok(())
}

/// The complete terminal is a true next rung: it is eligible only after every
/// propagation rung ran and beta-CROWN's remaining width still justifies
/// escalation under the same threshold policy.
fn should_run_complete(
    cfg: &LadderConfig,
    rungs: &[RungOutcome],
    propagation_rung_count: usize,
) -> bool {
    cfg.use_complete
        && rungs.len() == propagation_rung_count
        && should_escalate(
            rungs.last().and_then(|rung| rung.max_width),
            cfg.escalation_width_threshold,
        )
}

/// Run the laddered verification driver on a single graph network.
///
/// Starts at IBP. If a rung verifies the property, the ladder returns
/// immediately. Otherwise, when the observed bound width exceeds
/// [`LadderConfig::escalation_width_threshold`] and
/// [`LadderConfig::max_method`] permits, it escalates through alpha-CROWN, CROWN,
/// beta-CROWN, and (under `feature = "complete"` with
/// [`LadderConfig::use_complete`]) the complete MIP terminal. A `Violated`
/// verdict from a complete method short-circuits with that counterexample.
/// Soundness provenance is combined across every rung that contributed.
///
/// # ENSURES
/// - The returned [`LadderedResult::result`] is never `Verified` unless some rung
///   actually proved the property.
/// - [`LadderedResult::rungs`] is non-empty (IBP always runs first).
///
/// # Errors
/// Returns [`NyError::InvalidSpec`] when
/// [`LadderConfig::escalation_width_threshold`] is NaN or negative, and
/// propagates specification/verification errors from an attempted rung.
pub fn verify_model(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    cfg: &LadderConfig,
) -> Result<LadderedResult> {
    validate_config(cfg)?;
    let spec = rung_spec(spec, cfg)?;
    let mut rungs: Vec<RungOutcome> = Vec::new();
    let mut combined_provenance = SoundnessProvenance::sound();

    // The propagation ladder, in escalation order. SDP-CROWN is intentionally
    // omitted: it is a specialized L2-ball path, not a general escalation rung.
    let ladder = [
        PropagationMethod::Ibp,
        PropagationMethod::AlphaCrown,
        PropagationMethod::Crown,
        PropagationMethod::BetaCrown,
    ];

    // Track the tightest Unknown/Timeout seen so we can return it as the fallback.
    let mut best_fallback: Option<(VerificationResult, MethodUsed)> = None;

    for (idx, &method) in ladder.iter().enumerate() {
        // IBP (idx 0) always runs. Higher rungs require both an allowance from
        // max_method and a wide-enough previous bound to be worth the cost.
        if idx > 0 {
            if !rung_allowed(method, cfg.max_method) {
                break;
            }
            let prev_width = rungs.last().and_then(|r| r.max_width);
            if !should_escalate(prev_width, cfg.escalation_width_threshold) {
                break;
            }
        }

        let (result, outcome) = run_rung(net, &spec, method)?;
        combined_provenance = combined_provenance.combine(result.provenance());
        let actual_method = outcome.method.clone();
        let verified = outcome.verified;
        rungs.push(outcome);

        if verified {
            // Re-attach the combined provenance so downstream consumers see the
            // soundness contributions of every rung that ran.
            let result = result.with_provenance(combined_provenance);
            return Ok(LadderedResult {
                result,
                method_used: actual_method,
                rungs,
            });
        }

        if is_definitive_violation(&result) {
            // Complete method found a real counterexample: decisive refutation.
            let result = result.with_provenance(combined_provenance);
            return Ok(LadderedResult {
                result,
                method_used: actual_method,
                rungs,
            });
        }

        // Keep the tightest fallback (smallest max_width) seen so far.
        let this_width = result_bounds(&result).and_then(max_bound_width);
        let keep = match &best_fallback {
            None => true,
            Some((prev, _)) => {
                let prev_width = result_bounds(prev).and_then(max_bound_width);
                fallback_is_tighter(this_width, prev_width)
            }
        };
        if keep {
            best_fallback = Some((result, actual_method));
        }
    }

    // Optional complete MIP terminal (only when feature + flag are both on).
    if should_run_complete(cfg, &rungs, ladder.len()) {
        if let Some((result, method, outcome)) = run_complete_terminal(net, &spec, cfg)? {
            combined_provenance = combined_provenance.combine(result.provenance());
            let verified = outcome.verified;
            let definitive_violation = is_definitive_violation(&result);
            rungs.push(outcome);
            if verified || definitive_violation {
                let result = result.with_provenance(combined_provenance);
                return Ok(LadderedResult {
                    result,
                    method_used: method,
                    rungs,
                });
            }
            let this_width = result_bounds(&result).and_then(max_bound_width);
            let keep = match &best_fallback {
                None => true,
                Some((prev, _)) => {
                    let prev_width = result_bounds(prev).and_then(max_bound_width);
                    fallback_is_tighter(this_width, prev_width)
                }
            };
            if keep {
                best_fallback = Some((result, method));
            }
        }
    }

    // Nothing verified: return the tightest inconclusive result, carrying the
    // combined provenance across all rungs.
    let (result, method_used) = best_fallback.expect("ladder always runs at least the IBP rung");
    let result = result.with_provenance(combined_provenance);
    Ok(LadderedResult {
        result,
        method_used,
        rungs,
    })
}

/// Tightness comparison for fallback selection: smaller widths win; a present
/// width beats an absent one; ties keep the existing choice.
fn fallback_is_tighter(candidate: Option<f32>, current: Option<f32>) -> bool {
    if candidate.is_some_and(f32::is_nan) {
        return false;
    }
    match (candidate, current) {
        (Some(candidate), Some(current)) => {
            if current.is_nan() {
                true
            } else {
                candidate < current
            }
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => false,
    }
}

/// Run the complete MIP terminal.
///
/// When built with `feature = "complete"` this performs a REAL complete MIP
/// verification of `net` against `spec` via
/// [`crate::complete::verify_graph_complete_mip`]: the graph is reduced to the
/// sequential `Linear (ReLU Linear)*` fragment the exported encoder accepts, its
/// hidden pre-activation boxes are taken from ny's sound node-bound collection
/// and inflated outward, and every finite output-bound side is refuted as a
/// separate MIP feasibility question.
///
/// It has exactly two outcomes:
///
/// * a rung backed by an actual solve — `Verified` when every negated-property
///   disjunct was proven infeasible, otherwise `Unknown` carrying the solver's
///   own reason; or
/// * a LOUD REFUSAL: `Err(NyError::UnsupportedOp)` naming the node that put the
///   graph outside the encodable fragment. It never records a `Mip` rung it did
///   not actually run — a verdict-shaped non-answer is indistinguishable from a
///   real result to a downstream consumer, and that is the worst failure this
///   driver can produce.
///
/// A `Sat` disjunct is deliberately NOT reported as `Violated`; see the
/// soundness contract on [`crate::complete`].
///
/// When the feature is off this is a no-op (returns `Ok(None)`), so default
/// `propagate` builds compile and behave identically.
#[cfg(feature = "complete")]
fn run_complete_terminal(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    _cfg: &LadderConfig,
) -> Result<Option<(VerificationResult, MethodUsed, RungOutcome)>> {
    use crate::complete::{CompleteMipOptions, MipConfig};
    use ny_core::UnknownReason;

    let mut options = CompleteMipOptions::default();
    // `rung_spec` has already folded `LadderConfig::timeout_ms` into the spec,
    // so this is the caller's own budget when they expressed one.
    if let Some(ms) = spec.timeout_ms() {
        options.mip = MipConfig {
            timeout_secs: (ms as f64) / 1000.0,
            ..options.mip
        };
    }

    // Propagates the loud refusal for a graph outside the encodable fragment.
    let report = crate::complete::verify_graph_complete_mip(net, spec, &options)?;

    if report.verified {
        let output_bounds = spec.output_bounds().to_vec();
        let note = format!(
            "complete MIP terminal: all {} negated-property disjunct(s) proven infeasible{}",
            report.disjuncts,
            if report.certified {
                " (exact certificate evidence verified)"
            } else {
                ""
            }
        );
        let result = VerificationResult::Verified {
            provenance: SoundnessProvenance::sound(),
            output_bounds: output_bounds.clone(),
            proof: None,
            actual_method: Some(MethodUsed::Mip),
        };
        let outcome = RungOutcome {
            method: MethodUsed::Mip,
            verified: true,
            max_width: max_bound_width(&output_bounds),
            note,
        };
        return Ok(Some((result, MethodUsed::Mip, outcome)));
    }

    let reason = report
        .inconclusive
        .unwrap_or_else(|| "complete MIP terminal reached no decision".to_string());
    let message = format!(
        "complete MIP terminal: {} of {} disjunct(s) refuted; {reason}",
        report.disjuncts_refuted, report.disjuncts
    );
    // The terminal proved nothing, so it contributes no bounds of its own. The
    // vacuous box keeps it out of the ladder's tightest-fallback selection.
    let bounds: Vec<Bound> = spec
        .output_bounds()
        .iter()
        .map(|_| Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY))
        .collect();
    let result = VerificationResult::Unknown {
        provenance: SoundnessProvenance::sound(),
        bounds: bounds.clone(),
        reason: UnknownReason::Other {
            message: message.clone(),
        },
        actual_method: Some(MethodUsed::Mip),
    };
    let outcome = RungOutcome {
        method: MethodUsed::Mip,
        verified: false,
        max_width: max_bound_width(&bounds),
        note: message,
    };
    Ok(Some((result, MethodUsed::Mip, outcome)))
}

/// No-op complete terminal for builds without `feature = "complete"`.
#[cfg(not(feature = "complete"))]
fn run_complete_terminal(
    _net: &GraphNetwork,
    _spec: &VerificationSpec,
    _cfg: &LadderConfig,
) -> Result<Option<(VerificationResult, MethodUsed, RungOutcome)>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphNetwork, GraphNode};
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};
    use ny_core::{Bound, VerificationSpec};

    /// Tiny 1->1 identity-ish linear graph: out = x (weight 1, bias 0), then a
    /// second linear that scales by 1. No ReLU branch needed for the easy case.
    fn identity_graph() -> GraphNetwork {
        let w = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("1x1 weight");
        let linear = LinearLayer::new(w, Some(Array1::zeros(1))).expect("valid linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.set_output("linear");
        graph
    }

    /// Tiny linear+ReLU+linear graph with a 2->2 expansion. Used for the
    /// escalation case where a loose output spec keeps IBP from proving.
    fn relu_graph() -> GraphNetwork {
        let w1 = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(Array1::zeros(2))).expect("valid linear1");
        let w2 = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(Array1::zeros(2))).expect("valid linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    #[test]
    fn ibp_proves_at_rung_zero() {
        let graph = identity_graph();
        // Input in [0, 1] -> output in [0, 1]. Spec allows [-1, 2]: trivially IBP-provable.
        let spec = VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(-1.0, 2.0)])
            .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &LadderConfig::default())
            .expect("ladder should run on identity graph");

        assert!(
            outcome.result.is_verified(),
            "loose spec should verify, got {:?}",
            outcome.result
        );
        assert_eq!(
            outcome.method_used,
            MethodUsed::Ibp,
            "the easy case must be proven by the first (IBP) rung"
        );
        assert_eq!(
            outcome.rungs.len(),
            1,
            "verifying at rung 0 must not escalate; rungs = {:?}",
            outcome.rungs
        );
        assert!(outcome.rungs[0].verified, "rung 0 must be marked verified");
        assert_eq!(outcome.rungs[0].method, MethodUsed::Ibp);
    }

    #[test]
    fn escalates_past_ibp_when_threshold_low() {
        let graph = relu_graph();
        // Input in [-1, 1]^2. Pick an output spec that IBP's loose interval
        // arithmetic cannot prove but a tighter method (or more rungs) attempts.
        // Output of relu_graph for x in [-1,1]^2 spans a non-trivial box, so a
        // narrow spec keeps IBP from proving.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        )
        .expect("valid spec");

        // Threshold 0 forces escalation whenever the property is unproven and a
        // non-trivial width remains; cap at AlphaCrown to keep the test fast.
        let cfg = LadderConfig {
            max_method: PropagationMethod::AlphaCrown,
            escalation_width_threshold: 0.0,
            use_complete: false,
            timeout_ms: Some(5_000),
        };

        let outcome = verify_model(&graph, &spec, &cfg).expect("ladder should run on relu graph");

        // IBP cannot prove this narrow spec, so the ladder must have climbed at
        // least one rung beyond IBP (to AlphaCrown).
        assert!(
            outcome.rungs.len() >= 2,
            "escalation should attempt more than the IBP rung; rungs = {:?}",
            outcome.rungs
        );
        assert_eq!(
            outcome.rungs[0].method,
            MethodUsed::Ibp,
            "first rung must always be IBP"
        );
        assert!(
            !outcome.rungs[0].verified,
            "the narrow spec must not be IBP-provable (precondition of this test)"
        );
        // The escalated rung must advance past IBP in the method ladder.
        let second = &outcome.rungs[1];
        assert!(
            matches!(
                second.method,
                MethodUsed::AlphaCrown | MethodUsed::Crown | MethodUsed::Ibp
            ),
            "second rung should be the escalated AlphaCrown attempt (verifier may relabel on \
             internal fallback), got {:?}",
            second.method
        );
    }

    #[test]
    fn max_method_ceiling_caps_escalation() {
        let graph = relu_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            // Impossible-to-prove tiny spec to force maximum escalation pressure.
            vec![Bound::new(0.0, 0.0), Bound::new(0.0, 0.0)],
        )
        .expect("valid spec");

        // Cap the ladder at IBP: even with a zero threshold, no escalation may occur.
        let cfg = LadderConfig {
            max_method: PropagationMethod::Ibp,
            escalation_width_threshold: 0.0,
            use_complete: false,
            timeout_ms: Some(5_000),
        };

        let outcome = verify_model(&graph, &spec, &cfg).expect("ladder should run");
        assert_eq!(
            outcome.rungs.len(),
            1,
            "max_method = Ibp must prevent any escalation; rungs = {:?}",
            outcome.rungs
        );
        assert_eq!(outcome.method_used, MethodUsed::Ibp);
        assert!(
            !outcome.result.is_verified(),
            "the impossible spec must not be reported as verified"
        );
    }

    #[test]
    fn max_bound_width_picks_widest() {
        let bounds = [
            Bound::new(0.0, 1.0),
            Bound::new(-2.0, 3.0),
            Bound::new(1.0, 1.5),
        ];
        assert_eq!(max_bound_width(&bounds), Some(5.0));
        assert_eq!(max_bound_width(&[]), None);

        let nan_width_then_finite = [
            Bound::new_allow_infinite(f32::INFINITY, f32::INFINITY),
            Bound::new(0.0, 1.0),
        ];
        assert!(
            max_bound_width(&nan_width_then_finite)
                .expect("non-empty bounds")
                .is_nan(),
            "a later finite width must not absorb an earlier NaN width"
        );
    }

    #[test]
    fn should_escalate_logic() {
        // No bounds -> escalate (give the tighter method a chance).
        assert!(should_escalate(None, 1.0));
        // Width above threshold -> escalate.
        assert!(should_escalate(Some(2.0), 1.0));
        // Width at/below threshold -> stop.
        assert!(!should_escalate(Some(1.0), 1.0));
        assert!(!should_escalate(Some(0.5), 1.0));
        // NaN width (exploded) -> escalate.
        assert!(should_escalate(Some(f32::NAN), 1.0));
    }

    #[test]
    fn invalid_thresholds_are_rejected() {
        for threshold in [f32::NAN, -1.0, f32::NEG_INFINITY] {
            let cfg = LadderConfig {
                escalation_width_threshold: threshold,
                ..LadderConfig::default()
            };
            assert!(
                validate_config(&cfg).is_err(),
                "{threshold} must be invalid"
            );
        }
        let cfg = LadderConfig {
            escalation_width_threshold: f32::INFINITY,
            ..LadderConfig::default()
        };
        assert!(
            validate_config(&cfg).is_ok(),
            "+infinity is a valid 'never escalate for finite widths' policy"
        );
    }

    #[test]
    fn complete_terminal_obeys_ladder_progress_and_threshold() {
        let cfg = LadderConfig {
            use_complete: true,
            escalation_width_threshold: 1.0,
            ..LadderConfig::default()
        };
        let rung = |max_width| RungOutcome {
            method: MethodUsed::Ibp,
            verified: false,
            max_width,
            note: String::new(),
        };

        let all_wide = vec![rung(Some(2.0)); 4];
        assert!(should_run_complete(&cfg, &all_wide, 4));
        assert!(!should_run_complete(&cfg, &all_wide[..3], 4));

        let mut beta_tight = all_wide;
        beta_tight[3].max_width = Some(0.5);
        assert!(!should_run_complete(&cfg, &beta_tight, 4));
    }

    #[test]
    fn fallback_tightness_prefers_smaller_width() {
        assert!(fallback_is_tighter(Some(1.0), Some(2.0)));
        assert!(!fallback_is_tighter(Some(2.0), Some(1.0)));
        assert!(fallback_is_tighter(Some(1.0), Some(f32::NAN)));
        assert!(!fallback_is_tighter(Some(f32::NAN), Some(1.0)));
        assert!(
            !fallback_is_tighter(Some(f32::NAN), None),
            "a corrupt width must not replace a no-bounds timeout"
        );
        assert!(fallback_is_tighter(Some(1.0), None));
        assert!(!fallback_is_tighter(None, Some(1.0)));
        assert!(!fallback_is_tighter(None, None));
    }

    #[test]
    fn rung_allowance_respects_ceiling() {
        assert!(rung_allowed(PropagationMethod::Ibp, PropagationMethod::Ibp));
        assert!(rung_allowed(
            PropagationMethod::AlphaCrown,
            PropagationMethod::BetaCrown
        ));
        assert!(!rung_allowed(
            PropagationMethod::BetaCrown,
            PropagationMethod::AlphaCrown
        ));
        assert!(!rung_allowed(
            PropagationMethod::Crown,
            PropagationMethod::AlphaCrown
        ));
    }
}

/// The complete MIP terminal has exactly two honest outcomes: a rung backed by a
/// real solve, or a loud refusal. These tests pin both, and pin that it never
/// records a `Mip` rung it did not run.
#[cfg(all(test, feature = "complete"))]
mod complete_terminal_tests {
    use super::*;
    use crate::graph::{GraphNetwork, GraphNode};
    use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
    use ndarray::{Array1, Array2};
    use ny_core::{Bound, NyError, VerificationSpec};

    fn linear(weight: Vec<f32>, rows: usize, cols: usize, bias: Vec<f32>) -> LinearLayer {
        LinearLayer::new(
            Array2::from_shape_vec((rows, cols), weight).expect("weight shape"),
            Some(Array1::from_vec(bias)),
        )
        .expect("valid linear layer")
    }

    /// `Linear -> ReLU -> Linear -> ReLU -> Linear` computing
    /// `relu(|x| - 0.5) - relu(|x| - 0.5)`, i.e. exactly `0` on every input.
    ///
    /// Both second-layer neurons carry the SAME pre-activation, so per-neuron
    /// CROWN relaxes them independently and cannot see the cancellation: every
    /// propagation rung stops at roughly `[-0.5, 0.5]` (asserted below). The MIP
    /// encoding ties them with an equality row and closes the gap, so this graph
    /// separates a REAL terminal from a stub that returns a non-answer.
    fn cancelling_chain() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(linear(vec![1.0, 1.0, 1.0, 1.0], 2, 2, vec![-0.5, -0.5])),
            vec!["r1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "r2",
            Layer::ReLU(ReLULayer::new()),
            vec!["l2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l3",
            Layer::Linear(linear(vec![1.0, -1.0], 1, 2, vec![0.0])),
            vec!["r2".to_string()],
        ));
        graph.set_output("l3");
        graph
    }

    /// A residual `Add` join — a two-input node, outside the fragment the
    /// exported MIP encoder covers.
    fn residual_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "l1",
            Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
        ));
        graph.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "l1".to_string(),
            "r1".to_string(),
        ));
        graph.set_output("add");
        graph
    }

    /// Force every propagation rung to be attempted and to fail.
    fn escalate_to_terminal(use_complete: bool) -> LadderConfig {
        LadderConfig {
            max_method: PropagationMethod::BetaCrown,
            escalation_width_threshold: 0.0,
            use_complete,
            timeout_ms: Some(30_000),
        }
    }

    /// Precondition of the terminal tests: no propagation rung can prove this.
    #[test]
    fn no_propagation_rung_proves_the_cancelling_chain() {
        let graph = cancelling_chain();
        let spec =
            VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-0.01, 0.01)])
                .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &escalate_to_terminal(false))
            .expect("the propagation ladder runs");
        assert!(
            !outcome.result.is_verified(),
            "precondition broken: propagation now proves this, so the terminal test below would \
             no longer exercise the MIP rung"
        );
        assert_eq!(
            outcome.rungs.len(),
            4,
            "all four propagation rungs must run: {:?}",
            outcome.rungs
        );
        assert!(
            outcome.rungs.iter().all(|rung| !rung.verified),
            "no propagation rung may prove the cancellation: {:?}",
            outcome.rungs
        );
        assert!(
            outcome
                .rungs
                .iter()
                .all(|rung| rung.method != MethodUsed::Mip),
            "the MIP rung must not appear with use_complete = false: {:?}",
            outcome.rungs
        );
    }

    #[test]
    fn complete_terminal_performs_a_real_mip_verification() {
        let graph = cancelling_chain();
        let spec =
            VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-0.01, 0.01)])
                .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &escalate_to_terminal(true))
            .expect("the fragment graph is encodable");

        assert!(
            outcome.result.is_verified(),
            "the MIP terminal must decide what propagation could not: {:?}",
            outcome.result
        );
        assert_eq!(
            outcome.method_used,
            MethodUsed::Mip,
            "the verdict must be attributed to the MIP terminal"
        );
        assert_eq!(
            outcome.rungs.len(),
            5,
            "four propagation rungs then the terminal: {:?}",
            outcome.rungs
        );
        let terminal = outcome.rungs.last().expect("terminal rung");
        assert_eq!(terminal.method, MethodUsed::Mip);
        assert!(terminal.verified);
        assert!(
            terminal.note.contains("disjunct(s) proven infeasible"),
            "the rung note must describe the solve that produced it: {}",
            terminal.note
        );

        // Cross-check against the terminal's own entry point on the same
        // inputs: a stub could satisfy the assertions above only by agreeing
        // with a genuine solver run.
        let report = crate::complete::verify_graph_complete_mip(
            &graph,
            &spec,
            &crate::complete::CompleteMipOptions::default(),
        )
        .expect("encodable graph");
        assert!(report.verified);
        assert_eq!(report.disjuncts, 2);
        assert_eq!(report.disjuncts_refuted, 2);
    }

    #[test]
    fn complete_terminal_does_not_overclaim_on_a_false_property() {
        let graph = cancelling_chain();
        // The true output is exactly 0, so this spec is false.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new_allow_infinite(0.5, f32::INFINITY)],
        )
        .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &escalate_to_terminal(true))
            .expect("the fragment graph is encodable");

        assert!(
            !outcome.result.is_verified(),
            "a false property must never be reported verified: {:?}",
            outcome.result
        );
        assert!(
            !matches!(outcome.result, VerificationResult::Violated { .. }),
            "a feasible RELAXED negation is not a counterexample and must not be dressed up as \
             one: {:?}",
            outcome.result
        );
        let terminal = outcome
            .rungs
            .last()
            .expect("the terminal rung is still recorded");
        assert_eq!(terminal.method, MethodUsed::Mip);
        assert!(!terminal.verified);
        assert!(
            terminal.note.contains("is feasible"),
            "the rung must carry the solver's own reason: {}",
            terminal.note
        );
    }

    #[test]
    fn complete_terminal_refuses_a_graph_outside_the_encodable_fragment() {
        let graph = residual_graph();
        // Impossible spec: every propagation rung runs and fails, so the ladder
        // reaches the terminal.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new(0.0, 0.0), Bound::new(0.0, 0.0)],
        )
        .expect("valid spec");

        let error = verify_model(&graph, &spec, &escalate_to_terminal(true))
            .expect_err("a residual Add join cannot be encoded, so the terminal must refuse");
        assert!(
            matches!(error, NyError::UnsupportedOp(_)),
            "the refusal must be a typed UnsupportedOp, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("'add'"),
            "the refusal must name the blocking node: {message}"
        );
        assert!(
            message.contains("Refusing rather than returning a verdict-shaped non-answer"),
            "the refusal must say why it is an error and not a rung: {message}"
        );
    }

    /// The refusal is scoped to the opt-in: with the flag off the same graph
    /// still returns its propagation verdict.
    #[test]
    fn the_refusal_is_scoped_to_use_complete() {
        let graph = residual_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0)],
            vec![Bound::new(0.0, 0.0), Bound::new(0.0, 0.0)],
        )
        .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &escalate_to_terminal(false))
            .expect("without the opt-in the ladder is unchanged");
        assert!(!outcome.result.is_verified());
        assert!(
            outcome
                .rungs
                .iter()
                .all(|rung| rung.method != MethodUsed::Mip),
            "no MIP rung may be recorded when the terminal was not requested: {:?}",
            outcome.rungs
        );
    }

    /// A `Mip` rung is only ever recorded by a run that actually reached the
    /// solver: the ladder must not manufacture one from the config alone.
    #[test]
    fn no_mip_rung_without_reaching_the_terminal() {
        let graph = cancelling_chain();
        // Provable at rung 0, so the ladder returns before the terminal.
        let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-9.0, 9.0)])
            .expect("valid spec");

        let outcome = verify_model(&graph, &spec, &escalate_to_terminal(true))
            .expect("an easy spec never reaches the terminal");
        assert!(outcome.result.is_verified());
        assert_eq!(outcome.method_used, MethodUsed::Ibp);
        assert_eq!(outcome.rungs.len(), 1, "{:?}", outcome.rungs);
    }
}
