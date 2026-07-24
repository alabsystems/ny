// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Site data for backward CROWN dispatch coverage tests.
//!
//! This module holds `include_str!` constants and site expectation arrays
//! used by [`super::dispatch_coverage_tests`]. Separated to keep each file
//! under the 500-line limit.
//!
//! After the canonical dispatch sites were made exhaustive (commit 6b872d2,
//! #3424), the coverage test tracks each site's *explicit* Layer variant
//! references rather than the missing-from-canonical delta. This keeps
//! arrays small (10-20 items per site) and directly documents what each
//! dispatch site handles.

use std::{fs, path::Path, sync::OnceLock};

// === Source text constants ===
pub(super) const GRAPH_CROWN_PROPAGATION_SRC: &str = include_str!("graph_crown/propagation.rs");
pub(super) const GRAPH_CROWN_BACKWARD_NODE_DISPATCH_SRC: &str =
    include_str!("graph_crown/backward_node_dispatch.rs");
pub(super) const GRAPH_ALPHA_BACKWARD_SRC: &str = concat!(
    include_str!("graph_alpha/backward/mod.rs"),
    include_str!("graph_alpha/backward/nonlinear.rs"),
    include_str!("graph_alpha/backward/gradients.rs"),
);
pub(super) const GRAPH_ALPHA_BOUNDS_SRC: &str = concat!(
    include_str!("graph_alpha/bounds/mod.rs"),
    include_str!("graph_alpha/bounds/target_backward.rs"),
);
pub(super) const GRAPH_ALPHA_PROPAGATE_SEQ_BACKWARD_SRC: &str =
    include_str!("graph_alpha/propagate_sequential/backward.rs");
pub(super) const CROWN_NETWORK_SRC: &str = concat!(
    include_str!("core/sequential/crown.rs"),
    include_str!("core/sequential/crown/patches_step.rs"),
    include_str!("core/sequential/crown/backward_step.rs"),
    // `propagate_sdp_crown` was moved into its own module (core/sequential/crown/sdp.rs);
    // include it so the dispatch-coverage parser can still find the `pub fn
    // propagate_sdp_crown(` marker and its two `match layer` dispatch sites.
    include_str!("core/sequential/crown/sdp.rs"),
);
pub(super) const GRAPH_CROWN_BATCHED_SRC: &str = include_str!("core/graph/crown_batched.rs");
pub(super) const ALPHA_CROWN_BACKWARD_SRC: &str = include_str!("alpha_crown/backward.rs");
pub(super) const CONSTRAINTS_BACKWARD_SRC: &str = concat!(
    include_str!("../beta_crown/engine/graph/propagation/constraints/backward/mod.rs"),
    include_str!("../beta_crown/engine/graph/propagation/constraints/backward/setup.rs"),
    include_str!("../beta_crown/engine/graph/propagation/constraints/backward/linear.rs"),
    include_str!("../beta_crown/engine/graph/propagation/constraints/backward/finalize.rs"),
    // `process_shared_dispatch_node` (the generic operator backward dispatch site)
    // was extracted into backward/dispatch.rs; include it so the dispatch-coverage
    // parser can locate the marker and confirm it delegates to the shared dispatcher.
    include_str!("../beta_crown/engine/graph/propagation/constraints/backward/dispatch.rs"),
);
pub(super) const BATCHED_BACKWARD_SRC: &str =
    include_str!("../beta_crown/engine/graph/propagation/batched/backward_core.rs");
pub(super) const BETA_CROWN_ENGINE_BACKWARD_SRC: &str = concat!(
    include_str!("../beta_crown/engine/backward/layer_dispatch.rs"),
    include_str!("../beta_crown/engine/backward/legacy.rs"),
);
pub(super) const BETA_CROWN_GRADIENTS_JOINT_SRC: &str =
    include_str!("../beta_crown/engine/optimization/gradients/joint.rs");
pub(super) const BETA_CROWN_GRADIENTS_BETA_ONLY_SRC: &str =
    include_str!("../beta_crown/engine/optimization/gradients/beta_only.rs");
pub(super) const SHARED_DISPATCH_SRC: &str = include_str!("backward_dispatch/dispatch.rs");

#[derive(Clone, Copy)]
pub(super) enum SourceText {
    Static(&'static str),
    Loader(fn() -> &'static str),
}

impl SourceText {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Static(source) => source,
            Self::Loader(loader) => loader(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum ExpectedLayerRefs {
    Static(&'static [&'static str]),
    Loader(fn() -> &'static [&'static str]),
}

impl ExpectedLayerRefs {
    pub(super) fn as_slice(self) -> &'static [&'static str] {
        match self {
            Self::Static(layer_refs) => layer_refs,
            Self::Loader(loader) => loader(),
        }
    }
}

fn graph_crown_spec_src() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE
        .get_or_init(|| {
            let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/network/graph_crown");
            let monolith = base.join("spec_propagation.rs");
            if monolith.exists() {
                return read_source_text(&monolith);
            }

            let mod_path = base.join("spec_propagation/mod.rs");
            let core_path = base.join("spec_propagation/core.rs");
            format!(
                "{}{}",
                read_source_text(&mod_path),
                read_source_text(&core_path)
            )
        })
        .as_str()
}

fn read_source_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| {
        format!(
            "/* dispatch coverage failed to read {}: {} */",
            path.display(),
            err
        )
    })
}

const GRAPH_CROWN_SPEC_LAYER_REFS_MONOLITH: &[&str] = &["Div", "MulBinary", "ReLU"];
const GRAPH_CROWN_SPEC_LAYER_REFS_SPLIT: &[&str] = &["Div", "Linear", "MulBinary", "ReLU"];

fn graph_crown_spec_expected_layer_refs() -> &'static [&'static str] {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/network/graph_crown");
    if base.join("spec_propagation.rs").exists() {
        GRAPH_CROWN_SPEC_LAYER_REFS_MONOLITH
    } else {
        GRAPH_CROWN_SPEC_LAYER_REFS_SPLIT
    }
}

// === Site expectation types ===

#[derive(Clone, Copy)]
pub(super) struct DispatchSiteExpectation {
    pub(super) name: &'static str,
    pub(super) source: SourceText,
    pub(super) fn_marker: &'static str,
    pub(super) match_index: usize,
    /// The Layer variants explicitly referenced in this site's dispatch match.
    /// Compared against the actual parsed set to detect drift.
    /// When `exhaustive` is true, this field is ignored and the site is
    /// checked against the canonical set instead.
    pub(super) expected_explicit: &'static [&'static str],
    /// When true, the site is fully exhaustive (no catch-all) and must match
    /// the canonical dispatch's explicit set. `expected_explicit` is ignored.
    pub(super) exhaustive: bool,
}

pub(super) struct DelegatingSiteExpectation {
    pub(super) name: &'static str,
    pub(super) source: SourceText,
    pub(super) fn_marker: &'static str,
    pub(super) expected_site_specific: &'static [&'static str],
}

pub(super) struct SiteSpecificOnlyExpectation {
    pub(super) name: &'static str,
    pub(super) source: SourceText,
    pub(super) fn_marker: &'static str,
    pub(super) expected_layer_refs: ExpectedLayerRefs,
}

// === Site arrays ===

// Shared dispatcher is the canonical layer coverage source (exhaustive, no catch-all).
// The exhaustive `match ctx.layer` lives in `dispatch_backward_layer_inner`; the public
// `dispatch_backward_layer` is a thin wrapper that first carries/discharges the certified
// coefficient error (#vnncomp-aw-soundness) and then delegates to the inner match.
pub(super) const CANONICAL_SITE: DispatchSiteExpectation = DispatchSiteExpectation {
    name: "backward_dispatch::shared_core",
    source: SourceText::Static(SHARED_DISPATCH_SRC),
    fn_marker: "fn dispatch_backward_layer_inner(",
    match_index: 0,
    expected_explicit: &[],
    exhaustive: true,
};

// Sites that still use direct `match &node.layer` dispatch blocks.
// Each site's expected_explicit lists the Layer variants it references explicitly.
// Sites with `exhaustive: true` must match the canonical set (no catch-all).
// Sites with `exhaustive: false` have a catch-all for unlisted variants.
pub(super) const MATCH_BASED_SITES: &[DispatchSiteExpectation] = &[
    // crown_backward_step: removed — now delegates to dispatch_backward_layer
    // with pre-filters for ReLU, multi-input ops, and SkipMerge (commit 920096df1).
    // Tracked in DELEGATING_SITES instead.
    //
    // Patches-aware wrapper: exhaustive — no catch-all since #3424.
    DispatchSiteExpectation {
        name: "network::crown::crown_backward_step_patches",
        source: SourceText::Static(CROWN_NETWORK_SRC),
        fn_marker: "pub(crate) fn crown_backward_step_patches(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // SDP-CROWN (arXiv:2506.06665) only supports sequential Linear/ReLU networks,
    // so both its forward Lipschitz pass (match_index 0) and backward CROWN pass
    // (match_index 1) explicitly handle just `Linear` and `ReLU` and reject any
    // other layer via `unreachable!`/validate_linear_relu_only. These sites are
    // intentionally NON-exhaustive — not a drop-in for the canonical dispatcher.
    DispatchSiteExpectation {
        name: "network::crown::propagate_sdp_crown::forward",
        source: SourceText::Static(CROWN_NETWORK_SRC),
        fn_marker: "pub fn propagate_sdp_crown(",
        match_index: 0,
        expected_explicit: &["Linear", "ReLU"],
        exhaustive: false,
    },
    DispatchSiteExpectation {
        name: "network::crown::propagate_sdp_crown::backward",
        source: SourceText::Static(CROWN_NETWORK_SRC),
        fn_marker: "pub fn propagate_sdp_crown(",
        match_index: 1,
        expected_explicit: &["Linear", "ReLU"],
        exhaustive: false,
    },
    // network::crown::propagate_crown_batched removed: thin wrapper delegating
    // through Layer::propagate_crown_backward_batched (exhaustive since 6b872d2).
    //
    // propagate_crown_fast/propagate_crown_with_linear delegate to
    // crown_backward_step_patches (tracked above). Removed after dispatch dedup.
    //
    // backward_pass_core: exhaustive — no catch-all since #3424.
    DispatchSiteExpectation {
        name: "network::alpha_crown::backward_pass_core",
        source: SourceText::Static(ALPHA_CROWN_BACKWARD_SRC),
        fn_marker: "pub(super) fn backward_pass_core(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // The outer sequential_backward_pass loop now delegates per-layer dispatch to
    // propagate_sequential_backward_node. Track the helper that still owns the
    // exhaustive Layer match.
    DispatchSiteExpectation {
        name: "network::graph_alpha::propagate_sequential_backward_node",
        source: SourceText::Static(GRAPH_ALPHA_PROPAGATE_SEQ_BACKWARD_SRC),
        fn_marker: "fn propagate_sequential_backward_node(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // propagate_alpha_crown_single_pass_sequential_graph is now a thin wrapper
    // that delegates to sequential_backward_pass (already tracked above). Removed.
    DispatchSiteExpectation {
        name: "beta_crown::engine::propagate_layer_backward_with_alpha_beta",
        source: SourceText::Static(BETA_CROWN_ENGINE_BACKWARD_SRC),
        fn_marker: "pub(in crate::beta_crown::engine) fn propagate_layer_backward_with_alpha_beta(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // Graph batched CROWN inner loop: exhaustive — no catch-all since #3424.
    DispatchSiteExpectation {
        name: "network::graph_crown_batched::propagate_crown_batched_inner",
        source: SourceText::Static(GRAPH_CROWN_BATCHED_SRC),
        fn_marker: "fn propagate_crown_batched_inner(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // #[cfg(test)] legacy beta-only backward dispatch (same arms as production).
    DispatchSiteExpectation {
        name: "beta_crown::engine::propagate_layer_backward_with_beta",
        source: SourceText::Static(BETA_CROWN_ENGINE_BACKWARD_SRC),
        fn_marker: "pub(in crate::beta_crown::engine) fn propagate_layer_backward_with_beta(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // Production joint α/β/λ gradient backward dispatch.
    DispatchSiteExpectation {
        name: "beta_crown::optimization::compute_joint_gradients::backward",
        source: SourceText::Static(BETA_CROWN_GRADIENTS_JOINT_SRC),
        fn_marker: "pub(crate) fn compute_joint_gradients(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // Production joint α/β/λ gradient forward sensitivity dispatch.
    DispatchSiteExpectation {
        name: "beta_crown::optimization::compute_joint_gradients::forward",
        source: SourceText::Static(BETA_CROWN_GRADIENTS_JOINT_SRC),
        fn_marker: "pub(crate) fn compute_joint_gradients(",
        match_index: 1,
        expected_explicit: &[],
        exhaustive: true,
    },
    // #[cfg(test)] analytical β gradient backward dispatch.
    DispatchSiteExpectation {
        name: "beta_crown::optimization::compute_beta_gradients::backward",
        source: SourceText::Static(BETA_CROWN_GRADIENTS_BETA_ONLY_SRC),
        fn_marker: "pub(crate) fn compute_beta_gradients(",
        match_index: 0,
        expected_explicit: &[],
        exhaustive: true,
    },
    // #[cfg(test)] analytical β gradient forward sensitivity dispatch.
    DispatchSiteExpectation {
        name: "beta_crown::optimization::compute_beta_gradients::forward",
        source: SourceText::Static(BETA_CROWN_GRADIENTS_BETA_ONLY_SRC),
        fn_marker: "pub(crate) fn compute_beta_gradients(",
        match_index: 1,
        expected_explicit: &[],
        exhaustive: true,
    },
];

// Sites that delegate to `dispatch_backward_layer` and only override specific
// layer handling in the caller.
pub(super) const DELEGATING_SITES: &[DelegatingSiteExpectation] = &[
    // crown_backward_step pre-filters ReLU (alpha/beta not available in sequential
    // CROWN), multi-input ops (no graph edges), and SkipMerge (identity pass-through).
    // Everything else delegates to dispatch_backward_layer (commit 920096df1, #3079).
    DelegatingSiteExpectation {
        name: "network::crown::crown_backward_step",
        source: SourceText::Static(CROWN_NETWORK_SRC),
        fn_marker: "fn crown_backward_step(",
        expected_site_specific: &[
            "Add",
            "Atan2",
            "BilinearCrown",
            "Concat",
            "Div",
            "ExpandLikeLastAxis",
            "IndexAdd",
            "MatMul",
            "MaxBinary",
            "MinBinary",
            "MulBinary",
            "NonZero",
            "ReLU",
            "ScatterAdd",
            "ScatterNd",
            "SelfAttention",
            "SkipMerge",
            "Sub",
            "Where",
        ],
    },
    // graph_crown shared helper after #3935/#3960 dedup. Coordinator-specific
    // overrides stay in SITE_SPECIFIC_ONLY_SITES below.
    DelegatingSiteExpectation {
        name: "graph_crown::dispatch_shared_core",
        source: SourceText::Static(GRAPH_CROWN_BACKWARD_NODE_DISPATCH_SRC),
        fn_marker: "pub(super) fn dispatch_shared_core(",
        expected_site_specific: &[],
    },
    // constraints/backward.rs: backward_crown_constrained → process_constrained_backward_node
    // (Linear/ReLU pre-filters) → process_shared_dispatch_node (dispatch call).
    // Track the innermost helper that owns the dispatch_backward_layer call.
    // The extracted Linear/ReLU pre-filters stay covered in
    // SITE_SPECIFIC_ONLY_SITES below.
    DelegatingSiteExpectation {
        name: "constraints::backward_crown_constrained",
        source: SourceText::Static(CONSTRAINTS_BACKWARD_SRC),
        fn_marker: "fn process_shared_dispatch_node(",
        expected_site_specific: &[],
    },
    // After #4186 nonlinear extraction, dag_alpha_backward_pass_core only keeps
    // the Conv2d Patches-mode init check (#3293). ReLU/Sigmoid/Sqrt/Tanh handling
    // was moved to handle_nonlinear_node in nonlinear.rs and is tracked below.
    DelegatingSiteExpectation {
        name: "graph_alpha::dag_alpha_backward_pass_core",
        source: SourceText::Static(GRAPH_ALPHA_BACKWARD_SRC),
        fn_marker: "fn dag_alpha_backward_pass_core(",
        expected_site_specific: &["Conv2d"],
    },
    // graph_alpha/bounds keeps alpha/non-alpha ReLU logic, MulBinary relaxation
    // site-specific (#2094, #3396), deterministic/mixed Where handling (#3676),
    // and Sigmoid/Sqrt/Tanh alpha path overrides. The backward loop carrying
    // these site-specific `Layer::*` arms was extracted from
    // `propagate_crown_to_node_core` into the shared per-seed helper
    // `run_target_backward_pass_core` so the concrete and input-linear wrappers,
    // plus the objective-chunking (#patches-obj-chunk) path, drive one identical
    // loop body.
    DelegatingSiteExpectation {
        name: "graph_alpha::bounds::run_target_backward_pass_core",
        source: SourceText::Static(GRAPH_ALPHA_BOUNDS_SRC),
        fn_marker: "fn run_target_backward_pass_core(",
        expected_site_specific: &[
            "Div",
            "MulBinary",
            "ReLU",
            "Reciprocal",
            "Sigmoid",
            "Sqrt",
            "Tanh",
            "Where",
        ],
    },
];

// Coordinators that no longer call `dispatch_backward_layer` directly after
// helper extraction, but still own site-specific `Layer::*` handling that must
// stay aligned with the shared dispatcher split.
pub(super) const SITE_SPECIFIC_ONLY_SITES: &[SiteSpecificOnlyExpectation] = &[
    // batched/backward_core.rs overrides Linear, ReLU, and Div (Div is rewritten
    // to reciprocal-plus-multiply, matching alpha-beta-CROWN graph optimization).
    // It was a DELEGATING site until the per-domain loop moved the shared call
    // into the `generic_backward_domain` helper in the SAME file — the helper
    // still calls `dispatch_backward_layer`, so coverage is intact, but the
    // delegating check (which only reads the coordinator's own body) could no
    // longer see it. Same refactor, and same reclassification, as
    // `constraints::process_constrained_backward_node` below. The site-specific
    // layer set is still pinned here, so a NEW bypass would still fail the gate.
    SiteSpecificOnlyExpectation {
        name: "batched::dispatch_node_backward",
        source: SourceText::Static(BATCHED_BACKWARD_SRC),
        fn_marker: "fn dispatch_node_backward(",
        expected_layer_refs: ExpectedLayerRefs::Static(&["Div", "Linear", "ReLU"]),
    },
    // constraints/backward.rs extracted the dispatch call into
    // process_shared_dispatch_node, but process_constrained_backward_node still
    // owns the Linear/ReLU overrides that bypass the shared dispatcher.
    SiteSpecificOnlyExpectation {
        name: "constraints::process_constrained_backward_node",
        source: SourceText::Static(CONSTRAINTS_BACKWARD_SRC),
        fn_marker: "fn process_constrained_backward_node(",
        expected_layer_refs: ExpectedLayerRefs::Static(&["Linear", "ReLU"]),
    },
    // graph_crown/propagation.rs owns ReLU/MulBinary/Div/Where special cases
    // around dispatch_shared_core. Conv2d patches-mode gating no longer matches
    // `Layer::Conv2d` directly — it is driven by the precomputed `plan.has_conv2d`
    // flag, and per-node Conv2d Patches backward is handled by the exhaustive
    // crown_backward_step_patches site. Linear handling was extracted to
    // backward_node_dispatch (#3935). The short-name truncation entry is now a
    // thin delegate; track the real implementation that owns the Layer:: refs
    // (#dedup-root-collections Fix B).
    SiteSpecificOnlyExpectation {
        name: "graph_crown::propagation::coordinator",
        source: SourceText::Static(GRAPH_CROWN_PROPAGATION_SRC),
        fn_marker:
            "fn crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(",
        expected_layer_refs: ExpectedLayerRefs::Static(&["Div", "MulBinary", "ReLU", "Where"]),
    },
    // After #3960 split, the short-name wrapper in mod.rs is a thin delegate.
    // Track the real implementation in core.rs which owns the Layer:: refs.
    SiteSpecificOnlyExpectation {
        name: "graph_crown::spec_propagation::coordinator",
        source: SourceText::Loader(graph_crown_spec_src),
        fn_marker: "pub(crate) fn propagate_crown_with_specs_and_engine_with_linear_and_reference_bounds_and_deadline_and_truncation(",
        expected_layer_refs: ExpectedLayerRefs::Loader(graph_crown_spec_expected_layer_refs),
    },
    // #4186 extracted nonlinear alpha-CROWN handling into nonlinear.rs. Keep
    // the explicit ReLU/Reciprocal/Sigmoid/Sqrt/Tanh site-specific split covered
    // even though dag_alpha_backward_pass_core now only retains the dispatch call.
    SiteSpecificOnlyExpectation {
        name: "graph_alpha::handle_nonlinear_node",
        source: SourceText::Static(GRAPH_ALPHA_BACKWARD_SRC),
        fn_marker: "pub(super) fn handle_nonlinear_node(",
        expected_layer_refs: ExpectedLayerRefs::Static(&[
            "ReLU",
            "Reciprocal",
            "Sigmoid",
            "Sqrt",
            "Tanh",
        ]),
    },
];
