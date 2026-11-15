// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark QDQ activation-relaxation lane: engagement observable + report-only
//! verdict latch (#smart-turn-qdq).
//!
//! When `model.qdq_activation_relaxation` is armed and the loaded graph
//! carries `QdqPerturbation` layers, this module prints the lever's OWN
//! observable — the relaxation count and, per relaxed quantizer, the computed
//! PRE-QUANTIZER interval bounds (the smart_turn head's line is the sound
//! bound the bucket-bisect report names: `U < s/2 = +0.004727089` closes a
//! row) — and latches a process-local flag the vnncomp verdict funnel reads
//! to keep this round REPORT-ONLY: a bound is a finding, never a verdict,
//! until the mechanism survives review and the family adjudication policy is
//! settled.

use std::sync::atomic::{AtomicBool, Ordering};

use ny_core::GemmEngine;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

use super::BetaCrownModel;

/// Latched when the armed gate met a graph that actually contains relaxation
/// layers, i.e. the dark lane is live for the current instance.
static QDQ_RELAXATION_ENGAGED: AtomicBool = AtomicBool::new(false);

/// Reset the engagement latch (call before each in-process instance).
pub(crate) fn reset_qdq_relaxation_engaged() {
    QDQ_RELAXATION_ENGAGED.store(false, Ordering::SeqCst);
}

/// Whether the dark QDQ relaxation lane engaged for the current instance.
pub(crate) fn qdq_relaxation_engaged() -> bool {
    QDQ_RELAXATION_ENGAGED.load(Ordering::SeqCst)
}

/// Emit the ARMED half of the engagement observable and latch the
/// report-only flag. Called as soon as the graph exists, BEFORE input-bounds
/// construction, so the count line survives spec-shape failures (smart_turn
/// is multi-input; the single-input verify lane refuses its vnnlib box after
/// the model has already loaded and fused).
pub(super) fn emit_qdq_relaxation_armed(model: &BetaCrownModel) -> bool {
    let BetaCrownModel::Graph(graph) = model else {
        return false;
    };
    let relaxed = qdq_node_names(graph);
    if relaxed.is_empty() {
        return false;
    }
    QDQ_RELAXATION_ENGAGED.store(true, Ordering::SeqCst);
    eprintln!(
        "[qdq_relax] armed: {} activation quantizer pair(s) relaxed to certified envelopes; \
         verdicts from this lane are REPORT-ONLY this round",
        relaxed.len()
    );
    true
}

/// Emit the BOUNDS half of the observable: one root IBP pass printing every
/// relaxed quantizer's pre-activation interval. IBP through the deep
/// staircase is expected to be loose on intermediate nodes; the number is the
/// finding either way (a bound above the threshold prices the tightness gap).
/// Never fails the run: any bound-collection error is printed and swallowed.
pub(super) fn emit_qdq_relaxation_bounds(
    model: &BetaCrownModel,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) {
    let BetaCrownModel::Graph(graph) = model else {
        return;
    };
    let relaxed: Vec<&str> = qdq_node_names(graph);
    if relaxed.is_empty() {
        return;
    }
    match graph.collect_node_bounds_with_engine(input, engine) {
        Ok(bounds) => {
            for name in &relaxed {
                let Some(node) = graph.node(name) else {
                    continue;
                };
                let Some(pre_name) = node.inputs().first() else {
                    continue;
                };
                let pre_bounds = if pre_name.as_str() == ny_propagate::NETWORK_INPUT {
                    Some(input)
                } else {
                    bounds.get(pre_name.as_str())
                };
                match pre_bounds {
                    Some(pre) => {
                        let lo = pre.lower().iter().copied().fold(f32::INFINITY, f32::min);
                        let up = pre
                            .upper()
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        eprintln!(
                            "[qdq_relax] pre-quantizer ibp node='{name}' input='{pre_name}' \
                             elems={} lower_min={lo:.9e} upper_max={up:.9e}",
                            pre.lower().len()
                        );
                    }
                    None => eprintln!(
                        "[qdq_relax] pre-quantizer ibp node='{name}' input='{pre_name}' \
                         bounds=<not collected>"
                    ),
                }
            }
        }
        Err(error) => {
            eprintln!("[qdq_relax] pre-quantizer ibp pass failed (probe only): {error}");
        }
    }
}

/// Names of every graph node whose layer is the QDQ relaxation, in the
/// graph's node order.
fn qdq_node_names(graph: &GraphNetwork) -> Vec<&str> {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .is_some_and(|node| node.layer().layer_type() == "QdqPerturbation")
        })
        .map(String::as_str)
        .collect()
}
