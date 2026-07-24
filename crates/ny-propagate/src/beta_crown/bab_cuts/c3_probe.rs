// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN increment C3 — DARK diagnostic probe (`NY_C3_PROBE=1`,
//! default off, stderr-only, read-only; `docs/CERTIFIED_CUT_CROWN_DESIGN.md`
//! §C3).
//!
//! The C3 hypothesis is scoped to BaB subdomains (root-level L1 cuts are
//! measured dead — §C2/C2b): on a deep resistant branch, split premises may
//! collapse a cut's bound `B` far enough that folding it finally pays. Before
//! wiring any fold, this probe answers the two decision questions for every
//! deep domain the wide multi-objective lane processes:
//!
//! 1. **Which ReLU layers do the branch's split premises live on?** If none
//!    are on the FIRST ReLU (the only layer whose pre-activations are exactly
//!    affine in the input — the L1 cut generation domain), L1-C3 is moot and
//!    the answer routes to L2+ generation (CROWN-upper-bounded rows, the
//!    DeepPair pattern).
//! 2. **Where L1 splits exist: how far does the split re-derivation collapse
//!    each candidate group's `B`?** (`generate_l1_cuts_for_splits`: root-B vs
//!    strengthened-B per same-position group anchored on the split neurons.)

use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::GraphSplitHistory;
use crate::layers::{Conv2dLayer, Layer};
use crate::{GraphNetwork, NETWORK_INPUT};

use super::multi_relu_cut_gen::generate_l1_cuts_for_splits;

/// The dark gate: C3 probing is active only when `NY_C3_PROBE=1`.
/// Read per call (cheap at per-domain frequency, test-friendly).
pub(crate) fn c3_probe_enabled() -> bool {
    matches!(std::env::var("NY_C3_PROBE").ok().as_deref(), Some("1"))
}

/// Minimum domain depth to probe (`NY_C3_PROBE_DEPTH`, default 8 — the depth
/// band where the prop885 resistant branch starts oscillating).
fn c3_probe_min_depth() -> usize {
    std::env::var("NY_C3_PROBE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
}

/// Resolve the network's FIRST ReLU (execution order) and its producer conv,
/// requiring the conv to read the network input directly (the L1 exact-affine
/// premise). `None` ⇒ the architecture is outside the L1 cut domain.
fn first_relu_conv(graph: &GraphNetwork) -> Option<(String, &Conv2dLayer)> {
    let order = graph.exec_order().ok()?;
    let relu_name = order.iter().find(|name| {
        graph
            .node(name)
            .is_some_and(|node| matches!(node.layer(), Layer::ReLU(_)))
    })?;
    let conv_name = graph.node(relu_name)?.inputs().first()?;
    let conv_node = graph.node(conv_name)?;
    let Layer::Conv2d(conv) = conv_node.layer() else {
        return None;
    };
    if conv_node.inputs() != [NETWORK_INPUT.to_string()] {
        return None;
    }
    Some((relu_name.clone(), conv))
}

/// Probe one BaB subdomain: log its split-layer histogram and, for premises
/// on the first ReLU, each candidate group's root-B vs split-strengthened B.
/// No-op below the depth threshold or with no ReLU split premises.
pub(crate) fn c3_probe_domain(
    graph: &GraphNetwork,
    depth: usize,
    history: &GraphSplitHistory,
    input: &BoundedTensor,
) {
    if depth < c3_probe_min_depth() {
        return;
    }
    let cons = &history.constraints;
    if cons.is_empty() {
        return;
    }

    // (1) Split-layer histogram: node → (+active / −inactive premise counts).
    let mut hist: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for c in cons {
        let e = hist.entry(c.node_name()).or_default();
        if c.is_active() {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }
    let hist_s = hist
        .iter()
        .map(|(n, (a, i))| format!("{n}=+{a}/-{i}"))
        .collect::<Vec<_>>()
        .join(" ");

    let Some((first_relu, conv)) = first_relu_conv(graph) else {
        eprintln!(
            "[c3-probe] depth={depth} splits={} first_relu=UNRESOLVED hist: {hist_s}",
            cons.len()
        );
        return;
    };
    let l1: Vec<(u32, bool)> = cons
        .iter()
        .filter(|c| c.node_name() == first_relu)
        .map(|c| (c.neuron_idx() as u32, c.is_active()))
        .collect();
    eprintln!(
        "[c3-probe] depth={depth} splits={} L1_splits={} first_relu={first_relu} hist: {hist_s}",
        cons.len(),
        l1.len()
    );
    if l1.is_empty() {
        return;
    }

    // (2) Candidate groups over split + neighboring unstable L1 neurons.
    let xl: Vec<f32> = input.lower().iter().copied().collect();
    let xu: Vec<f32> = input.upper().iter().copied().collect();
    for d in generate_l1_cuts_for_splits(conv, &xl, &xu, 3, &l1) {
        eprintln!(
            "[c3-probe]   group neurons={:?} states={:?} rootB={:.4} splitB={:.4} dB={:.4} n_split={} unstable_split={}",
            d.cut.cut.neurons,
            d.states,
            d.root_b,
            d.strengthened_b,
            d.root_b - d.strengthened_b,
            d.n_split,
            d.n_split_unstable,
        );
    }
}
