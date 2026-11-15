// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #relusplitter-lp-refine (wave-6 C7) — the ROOT intermediate-bound refiner
//! seam.
//!
//! The multi-objective graph BaB root may hand its stored per-node boxes to an
//! externally attached refiner that returns TIGHTER certified enclosures for
//! some nodes, which the root intersects shrink-only into the stored map BEFORE
//! the root objective pass and the BaB setup consume it — so the root bound and
//! every descendant domain inherit them. The refiner that exists (ny-cli's
//! rigorous LP refiner over ny-mip's `LpTightener`, Vibecheck's recipe:
//! pre-activation layers `1..=max_layer`, layer-progressive, window 1 — only
//! the LP relaxation, never a MILP) lives in ny-cli/ny-mip, which ny-propagate
//! cannot depend on — hence this trait seam, mirroring the
//! [`GraphMipLeafOracle`](super::graph_mip_leaf::GraphMipLeafOracle) and
//! [`RootAdjudicationOracle`](super::root_adjudication::RootAdjudicationOracle)
//! precedents (a runtime-attached `Arc` on `BetaCrownVerifier`, preserved by
//! `with_config_from`, `None` = byte-identical default).
//!
//! # Soundness contract (the implementor's obligations)
//!
//! - Every candidate `(lower, upper)` returned for a node MUST enclose that
//!   node's TRUE output over the request's input box, for the graph in the
//!   request — a rigorous LP bound (Neumaier–Shcherbina or exact rational),
//!   never a float LP optimum.
//! - Anything the implementor cannot certify must simply be absent (a missing
//!   node keeps its stored box).
//!
//! # What the root does with it
//!
//! `multi_objective::root::intersect_interm_into_stored` (the
//! `#dd-zono-interm` publication primitive) — per node, all-or-nothing:
//! arity mismatch, a non-finite candidate or a crossed intersection REFUSES the
//! whole node; otherwise `max(stored_l, next_down(cand_l))` /
//! `min(stored_u, next_up(cand_u))`. It can only shrink a box, so the outcome
//! set can move `unknown -> unsat` and never `unsat -> sat`.

use std::collections::HashMap;
use std::time::Instant;

use ny_tensor::BoundedTensor;

use crate::beta_crown::config::MipRefineIntermediates;
use crate::GraphNetwork;

/// Everything the refiner needs from the root.
pub struct RootIntermRefineRequest<'a> {
    /// The graph the BaB verifies (the LP must encode exactly this graph).
    pub graph: &'a GraphNetwork,
    /// The root input box.
    pub input_bounds: &'a BoundedTensor,
    /// The root's stored sound per-node output boxes, keyed by node name
    /// (a Linear node's box is the following ReLU's PRE-activation box).
    pub node_bounds: &'a HashMap<String, BoundedTensor>,
    /// The root tightening ladder's deadline. The refiner budgets its own
    /// share of what remains and must never run past it.
    pub deadline: Option<Instant>,
}

/// Telemetry for the root's fixed-schema `[mip-refine]` line. Counts only —
/// nothing here is a bound.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RootIntermRefineTelemetry {
    /// Whether the refiner ran any LP at all.
    pub engaged: bool,
    /// `ok`, or the stable decline reason.
    pub reason: &'static str,
    /// Pre-activation layers the pass attempted.
    pub layers: usize,
    /// Unstable neurons over the attempted layers before the pass.
    pub unstable_before: usize,
    /// Unstable neurons over the attempted layers after the pass.
    pub unstable_after: usize,
    /// Rigorous-bound requests issued.
    pub lp_solves: usize,
    /// Requests that returned a RIGOROUS finite bound.
    pub rigorous: usize,
    /// Neurons whose box shrank.
    pub tightened_neurons: usize,
    /// The refiner's own slice ended before every unstable neuron ran.
    pub deadline_hit: bool,
    /// Wall seconds inside the refiner.
    pub secs: f64,
}

/// What the refiner returns.
#[derive(Debug, Clone, Default)]
pub struct RootIntermRefineOutcome {
    /// Certified per-node candidate enclosures (binary64 lower/upper), keyed by
    /// graph node name. Empty on any decline.
    pub candidates: HashMap<String, (Vec<f64>, Vec<f64>)>,
    /// The run's telemetry.
    pub telemetry: RootIntermRefineTelemetry,
}

/// The root intermediate-bound refiner. Implementations live outside
/// ny-propagate (ny-cli's rigorous LP refiner); the signature is infallible by
/// design — implementors map every internal failure to an empty outcome with a
/// decline reason.
pub trait RootIntermRefiner: Send + Sync {
    /// Short stable token for the observable (`refiner=<name>`).
    fn name(&self) -> &'static str;

    /// Refine the stored boxes. See the module-level soundness contract.
    fn refine(&self, req: &RootIntermRefineRequest<'_>) -> RootIntermRefineOutcome;
}

/// The root's fixed-schema observable, printed on EVERY run where a refiner
/// is attached and the root reaches the phase (engaged or declined). Every
/// field is always present, so a grep never has to distinguish "absent" from
/// "zero".
#[must_use]
pub fn root_interm_refine_line(
    refiner: &str,
    telemetry: &RootIntermRefineTelemetry,
    nodes_intersected: usize,
) -> String {
    format!(
        "[mip-refine] root refiner={refiner} engaged={} reason={} layers={} unstable_before={} \
         unstable_after={} lp_solves={} rigorous={} tightened={} nodes_intersected={} \
         deadline_hit={} secs={:.3}",
        u8::from(telemetry.engaged),
        if telemetry.reason.is_empty() {
            "unset"
        } else {
            telemetry.reason
        },
        telemetry.layers,
        telemetry.unstable_before,
        telemetry.unstable_after,
        telemetry.lp_solves,
        telemetry.rigorous,
        telemetry.tightened_neurons,
        nodes_intersected,
        u8::from(telemetry.deadline_hit),
        telemetry.secs,
    )
}

/// The CLI's fixed-schema attach-time line, printed on EVERY run whose
/// resolved config carries `bab.mip_refine_intermediates` — attached or not —
/// so an armed key that never reached a refiner reads as `attached=0`, never
/// as a missing line.
#[must_use]
pub fn mip_refine_armed_line(
    config: &MipRefineIntermediates,
    attached: bool,
    reason: &str,
) -> String {
    format!(
        "[mip-refine] armed layers={} budget_frac={:.3} attached={} reason={reason}",
        config.layers,
        config.budget_frac,
        u8::from(attached),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armed_line_has_a_fixed_schema() {
        let line = mip_refine_armed_line(&MipRefineIntermediates::default(), true, "ok");
        assert_eq!(
            line,
            "[mip-refine] armed layers=3 budget_frac=0.300 attached=1 reason=ok"
        );
        let off = mip_refine_armed_line(
            &MipRefineIntermediates {
                layers: 2,
                budget_frac: 0.5,
            },
            false,
            "mip_feature_off",
        );
        assert_eq!(
            off,
            "[mip-refine] armed layers=2 budget_frac=0.500 attached=0 reason=mip_feature_off"
        );
    }

    #[test]
    fn observable_has_a_fixed_schema_for_engaged_and_declined_runs() {
        let fields = [
            "refiner=",
            "engaged=",
            "reason=",
            "layers=",
            "unstable_before=",
            "unstable_after=",
            "lp_solves=",
            "rigorous=",
            "tightened=",
            "nodes_intersected=",
            "deadline_hit=",
            "secs=",
        ];
        let declined = root_interm_refine_line(
            "lp",
            &RootIntermRefineTelemetry {
                reason: "not_relu_chain",
                ..RootIntermRefineTelemetry::default()
            },
            0,
        );
        let engaged = root_interm_refine_line(
            "lp",
            &RootIntermRefineTelemetry {
                engaged: true,
                reason: "ok",
                layers: 3,
                unstable_before: 301,
                unstable_after: 280,
                lp_solves: 590,
                rigorous: 588,
                tightened_neurons: 290,
                deadline_hit: false,
                secs: 12.5,
            },
            6,
        );
        let unset = root_interm_refine_line("lp", &RootIntermRefineTelemetry::default(), 0);
        for line in [&declined, &engaged, &unset] {
            assert!(line.starts_with("[mip-refine] root "), "{line}");
            for field in fields {
                assert_eq!(line.matches(field).count(), 1, "{field} in {line}");
            }
        }
        assert!(declined.contains("engaged=0 reason=not_relu_chain"));
        assert!(engaged.contains("engaged=1 reason=ok layers=3"));
        assert!(engaged.contains("nodes_intersected=6"));
        assert!(unset.contains("reason=unset"));
    }
}
