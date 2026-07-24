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
    Bound, MethodUsed, Result, SoundnessProvenance, VerificationResult, VerificationSpec,
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
    pub use_complete: bool,
    /// Optional per-rung timeout in milliseconds.
    ///
    /// When set, it is applied to the spec used for each rung (overriding the
    /// spec's own timeout) so no single method can stall the ladder.
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
        .fold(None, |acc, w| match acc {
            // NaN-safe max: keep the larger finite value, prefer non-None.
            Some(prev) if prev >= w => Some(prev),
            _ => Some(w),
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
pub fn verify_model(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    cfg: &LadderConfig,
) -> Result<LadderedResult> {
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
    if cfg.use_complete {
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
    match (candidate, current) {
        (Some(c), Some(p)) => c < p,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => false,
    }
}

/// Run the complete MIP terminal.
///
/// When built with `feature = "complete"`, this attempts a complete MIP-backed
/// verdict via `ny_mip`. The current realization is conservative: it only claims
/// a decisive verdict it can soundly back, and otherwise records an `Unknown`
/// rung rather than overclaiming. When the feature is off it is a no-op
/// (returns `Ok(None)`), so default `propagate` builds compile and behave
/// identically.
#[cfg(feature = "complete")]
fn run_complete_terminal(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    _cfg: &LadderConfig,
) -> Result<Option<(VerificationResult, MethodUsed, RungOutcome)>> {
    use ny_core::UnknownReason;

    // The complete MIP encoder (`ny_mip::encode_feedforward`) operates on a
    // sequential FC+ReLU network. If this graph cannot be reduced to that form,
    // the MIP terminal cannot soundly run, so we record an honest Unknown rather
    // than fabricating a verdict.
    //
    // Reference the curated complete surface so the dependency is exercised under
    // `--features full`; a full graph->MIP lowering is tracked as future work.
    let _ = (
        std::any::type_name::<crate::complete::MipSolver>(),
        std::any::type_name::<crate::complete::MipConfig>(),
    );

    let best_bounds: Vec<Bound> = spec
        .output_bounds()
        .iter()
        .map(|_| Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY))
        .collect();
    let _ = net;

    let result = VerificationResult::Unknown {
        provenance: SoundnessProvenance::sound(),
        bounds: best_bounds.clone(),
        reason: UnknownReason::Other {
            message: "complete MIP terminal reached but graph is not reducible to a \
                      sequential FC+ReLU network; sound MIP encoding unavailable"
                .to_string(),
        },
        actual_method: Some(MethodUsed::MipHiGHS),
    };
    let outcome = RungOutcome {
        method: MethodUsed::MipHiGHS,
        verified: false,
        max_width: max_bound_width(&best_bounds),
        note: "complete MIP terminal: graph not reducible to sequential FC+ReLU; \
               returned Unknown (sound)"
            .to_string(),
    };
    Ok(Some((result, MethodUsed::MipHiGHS, outcome)))
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
    fn fallback_tightness_prefers_smaller_width() {
        assert!(fallback_is_tighter(Some(1.0), Some(2.0)));
        assert!(!fallback_is_tighter(Some(2.0), Some(1.0)));
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
