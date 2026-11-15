// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Margin-row twin-wall BaB lane (#twinwall).
//!
//! Production Rust implementation of THE verified twin-wall mechanism
//! (validated against an exact-rational falsifier harness during
//! development): for the
//! cifar100/tinyimagenet resnet family, per BaB domain, ONE sparse CROWN
//! backward pass of `nf + a few` MARGIN rows — the margin functional rows
//! seeded at the output, composed through per-domain-refreshed HEAD gates and
//! FROZEN root trunk gates with PIECE-FIXED overrides for split neurons —
//! concretized against the input box. Best-first BaB with exact batched
//! candidate scoring closes the failing classes.
//!
//! Every verdict-feeding bound runs in [`RoundMode::Outward`]: f64 accumulate
//! with certified error carry + directed rounding toward -inf (the moat:
//! never overstate a lower bound; NaN/Inf/unsupported fail closed to
//! Unknown). `RoundMode::Parity` reproduces the Python reference bit-nearly
//! for differential tests only.
//!
//! Production authority was granted 2026-07-18 after the directed-rounding and
//! strict ONNX/VNNLIB-adapter proof obligations were discharged.  The trusted
//! constant and detailed evidence trail live with [`margin_row_bab_enabled`];
//! no environment variable can promote an unauthorised lane to an `Unsat`
//! verdict.

/// #alpha-opt: frozen-policy gated (typed preset or exact legacy
/// `NY_MARGIN_ROW_ALPHA_OPT=1`, declaration default OFF) projected supergradient
/// optimization of the root trunk alpha before the tree starts.
/// See the module docs for the gradient derivation and the `ms` soundness
/// invariant.
pub mod alpha_opt;
/// `#auto-alloc` — per-instance selection of the root phase parameters from the
/// LOADED MODEL, replacing constants that were calibrated on one model of a
/// category and shipped to both. See the module docs for the four on record.
pub mod auto_alloc;
pub mod bab;
/// #backward-interm: declaration-default OFF with a legacy environment
/// override, but explicitly armed by the measured CIFAR100 and TinyImageNet
/// typed presets. Each trunk ReLU's input box is re-derived by the lane's own
/// backward engine over frozen prefix gates and intersected (shrink-only) with
/// the forward tableau box. See the module docs for the soundness and the
/// clip-calibration ordering trap.
pub(crate) mod backward_interm;
/// #margin-row-beta: env-gated (`NY_MARGIN_ROW_BETA=1`, default OFF) β-CROWN
/// split Lagrangians for the lane's backward pass, with a one-step projected
/// supergradient proposal re-scored by the unchanged certified concretize.
pub(crate) mod beta;
/// #bi-gpu: the `#backward-interm` (and `#joint-refresh`) prefix pass PAIRS on
/// the certified sound-f64 GPU conv route of `#mr-gpu-eval`, through a scoped
/// route with its own per-pair accounting and RSS admission. Typed-preset
/// gated (`margin_row.bi_gpu`, declaration default OFF = byte-identical); see
/// the module docs.
pub(crate) mod bi_gpu;
pub mod bounds;
/// #mr-census: the ALWAYS-ON, verdict-free cost census of this lane — exactly
/// one `[mr-census]` stderr line per lane invocation, on every stop path.
/// Telemetry only; see the module docs for what each field does and does not
/// mean.
pub mod census;
/// #clip-and-verify: split-derived halfspaces + the closed-form dual that
/// re-concretizes over `box INTERSECT halfspaces`. See the module docs for the
/// frontier-explosion measurement that motivates it.
pub mod clip;
/// #mr-close-telemetry: ALWAYS-ON, verdict-free PROOF-INTEGRITY telemetry —
/// the certified value of every close a verdict rests on (root `dj`, tree
/// bound closes by stage, emptiness slacks, retirements), as exact bits and a
/// decimal, into `BabStats` (the flight record) and one `[mr-close]` stderr
/// line per lane exit. Telemetry only; see the module docs.
pub mod close_integrity;
/// #closure-adaptive: the runtime policy that picks this lane's closure
/// mechanisms from the child-closure fraction the lane measures about ITSELF.
pub mod closure;
/// #soundness-spine M2d: G2 containment sampling of the margin-row root frame.
pub(crate) mod containment_guard;
/// #cct-margin-row-admission: the CERTIFIED `Gemm -> BatchNormalization` head
/// fold (f64, with a carried relative weight error and absolute bias error),
/// used by the ny-cli extractor for CNN7's FC-BN head. See the module docs
/// for the derivation; the exact-rational oracle lives in its tests.
pub mod dense_bn_fold;
/// #margin-row-domain-alpha: default-OFF PER-DOMAIN alpha. The lower-line
/// slope is re-optimized against each BaB domain's own gates (clip tightening
/// included) instead of being frozen at the root value, and each child
/// warm-starts from its parent's accepted alpha. Tightness-only — any
/// `alpha in [0, 1]` is a valid lower line pointwise — with the certified pass
/// as the only scorer. See the module docs for the soundness obligations and
/// the clip-pairing measurement contract.
pub(crate) mod domain_alpha;
pub mod engine;
/// #mr-errlane (margin-row throughput plan B3): the ENVELOPE certified error
/// lane. One envelope value per row (or per channel) dominates the per-column
/// error matrix by monotone induction, so a wide pass carries its error lane at
/// the cost of a 1-wide pass. Typed-preset gated (`margin_row.err_envelope`,
/// declaration default `off` = byte-identical); see the module docs.
pub mod errlane;
/// review(eng14): the test-only EXACT oracle (rational arrangement-vertex
/// enumeration of one ReLU head layer over an input box of at most four
/// dimensions) that the input-head and direct-head transport soundness tests
/// compare every certified bound with.
#[cfg(test)]
pub(crate) mod exact_relu_oracle;
pub(crate) mod gpu_backward;
/// #mr-gpu-eval: the certified pass's conv backward products (signed, per-column
/// error lane, row envelope) on the process-global SOUND f64 GEMM engine
/// (cuBLAS `Dgemm`), certified by the lane's unchanged order-independent
/// charges. Typed-preset gated (`margin_row.gpu_conv`, declaration default OFF
/// = byte-identical); see the module docs.
pub(crate) mod gpu_conv;
pub(crate) mod gpu_seam;
/// #head-factor stage 1: the head seed in FACTORED form. Dark; equivalence-tested
/// only: no production caller exists yet, so it compiles for tests alone (an
/// arming change removes this `cfg(test)` together with `head_factor_armed`).
#[cfg(test)]
pub(crate) mod head_factor;
/// Source-owned bounded joint ReLU refinements on the final head.
/// Enabled by default; frozen `margin_row.head_groups: false` disables the route.
pub(crate) mod head_group;
pub(super) mod head_group_source;
pub mod hyperplane_probe;
/// #mr-ibp-seed: certified interval (IBP) boxes intersected SHRINK-ONLY with
/// the root tableau's trunk boxes, after the forward-only `bi_cal` capture and
/// before `#backward-interm`. Typed-preset gated (`margin_row.ibp_seed`,
/// declaration default OFF = byte-identical and silent), carried on the root's
/// frozen [`RootPrecisionRuntimePolicy`]; see the module docs.
pub mod ibp_seed;
pub(super) mod input_clipping;
/// Source-owned input-relative rows; no public proof ingestion or default route.
pub(crate) mod input_rows;
pub(crate) mod joint_grad;
pub(crate) mod marginal;
pub mod net;
/// #bi-patch: PER-COLUMN RECEPTIVE-FIELD (patch-mode) prefix passes for
/// `#backward-interm` and `#joint-refresh` — each trunk-seeded column walks
/// only its own receptive field, in virtual per-column frames, with the dense
/// arm's certified arithmetic restricted to that support. Typed-preset gated
/// (`margin_row.bi_patch`, declaration default OFF = byte-identical); see the
/// module docs.
pub(crate) mod patch;
pub mod prof;
/// #rebake: frozen-policy gated (typed preset or exact legacy
/// `NY_MARGIN_ROW_REBAKE=1`, declaration default OFF) joint forward re-bake —
/// iterate alpha ascent -> forward tableau rebuild with the
/// optimized alpha baked into `apply_gates` -> backward-interm re-run, with
/// round-level monotone accept on the certified `root_eval` bound. See the
/// module docs for the soundness argument and the one re-derived widening.
pub mod rebake;
pub mod root;
/// #root-box-source (ENGINE_PROGRAM M5b): the CERTIFIED patch-backward box
/// source for the root (`margin_row.root_box_source: tableau |
/// patch_backward`, default `tableau` = the forward tableau, bit-identical and
/// never entering this module) — a certified IBP reference per tensor, then a
/// certified `#bi-patch` receptive-field backward (dense fallback) over only
/// the reference-unstable neurons, intersected shrink-only. Carried on the
/// root's frozen [`RootPrecisionRuntimePolicy`]; see the module docs.
pub mod root_box;
/// #root-certify-once: Phase-0.5 root alpha, SEARCH UNTRUSTED / CERTIFY ONCE —
/// the root alpha ascent (optionally with abc-style per-round intermediate
/// refresh) run on a Parity scratch root, then ONE certified evaluation of the
/// chosen alpha, kept only if it strictly beats today's certified root.
/// Typed-preset gated (`margin_row.root_certify_once`, declaration default
/// OFF = byte-identical); see the module docs.
pub mod root_certify_once;
/// #attack-open-disjuncts: the lane's certified OUTER-root per-class bounds,
/// published (only when armed) for the attack-only open-disjunct stage.
/// Scheduling input for the attack; never read by any bound or verdict.
pub mod root_open;
/// #bbe-joint-root (ENGINE_PROGRAM M5a): the search selector of
/// `#root-certify-once` (`margin_row.root_search: ascent | joint_f32`, default
/// `ascent` = bit-identical) — the plain-data lowering of the lane IR, the
/// injected UNTRUSTED `ny_core::search::SearchBackend` (registered by ny-cli;
/// this crate never depends on `ny-bbe`) and the `[bbe]` observable. Its
/// slopes reach the certified root only through the existing ingest gate.
pub mod root_search;
pub mod rounding;
pub mod spec; // #margin-row-gpu-eft: dark skeleton, docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md
/// #spec-alpha: the per-START-NODE (spec) alpha axis at the root direct
/// pass. Typed-preset gated (`margin_row.spec_alpha`, declaration default
/// OFF), byte-identical when dark. See the module docs for the soundness
/// argument -- it adds zero lines of certified arithmetic.
pub mod spec_alpha;
/// Admitted original model/reference ownership and producer-sealed prefixes.
pub(super) mod support_program;
/// #frozen-dual-bab (V1 stage B): the certified BFS over stage A's frozen
/// alpha-zonotope state, run as a TIER between the root and the tree on the
/// classes the root leaves open (`margin_row.frozen_dual_bab`, declaration
/// default OFF). A class closes only when every leaf is certified and the
/// level ledger closes; everything else is handed back to the tree unchanged.
/// `margin_row.frozen_dual_bab_search` (#wide-fd-search, M4; default `f64`)
/// can hand each level's scoring to an injected UNTRUSTED backend whose
/// proposals the tier certifies in f64 and splits otherwise.
pub mod zono_bab;
/// #frozen-dual-state (V1 stage A): the CERTIFIED query state of the frozen
/// alpha-zonotope dual-ascent BaB — per open class, the generator-space
/// linearisation of the margin and of every unstable pre-activation, the
/// rank-1 split substitution and the certified node bound. Typed key
/// `margin_row.frozen_dual_bab` (declaration default OFF); stage A has no
/// consumer, so an armed lane only builds, reports and drops the states.
pub mod zono_state;

#[cfg(test)]
mod tests;

/// #cct-margin-row-admission: the lane charges a BN-folded head's certified
/// fold error (sampled containment over the whole error ball).
#[cfg(test)]
mod folded_head_tests;

#[cfg(test)]
mod direct_root_lane_tests;

use std::time::Instant;

pub use alpha_opt::RootAlphaExecutionEvidence;
/// #wave-beta-bab (ENGINE_PROGRAM M6): the `margin_row.tree_mode` selector
/// and the injected wave-search backend's registration (ny-cli calls it once
/// at startup; the factory runs only on the first ARMED wave tree).
pub use bab::wave::{
    set_wave_cuda_search_backend_factory, set_wave_search_backend_factory, take_last_wave_summary,
    TreeMode, WaveBackendKind, WavePolicy,
};
pub use bab::{
    BabConfig, BabStats, ClassBabStats, EpochExecutionEvidence, EpochFrameRoute,
    FrontierExecutionEvidence, FrontierRoute, MarginRowBab, MarginRowOutcome,
    PolicyExecutionEvidence, WidthExecutionEvidence,
};
pub use backward_interm::BackwardIntermExecutionEvidence;
pub use domain_alpha::Preset as DomainAlphaPreset;
pub use errlane::ErrLane;
/// #margin-row-gpu-eft: register the injected device executor for the
/// certified GPU backward lane (ny-cli calls this once at startup; the
/// closure is only invoked on the first ARMED admission).
pub use gpu_backward::device::set_margin_row_lane_gpu_factory;
/// #binary-identity: the implementation-generation marker `ny --build-id`
/// compiles into the binary's canonical identity, so the R8 provenance check
/// is a field comparison rather than a remembered `strings | grep`.
pub use gpu_backward::PROVENANCE_MARKER as GPU_BACKWARD_PROVENANCE_MARKER;
#[cfg(test)]
pub(crate) use head_group::input_direct::test_verify_input_head_retained_with_provider;
pub use head_group::input_direct::{
    verify_input_head, verify_input_head_retained, InputHeadContinuation, InputHeadOutcome,
    InputHeadReport, InputHeadSpec,
};
pub use head_group::{register_head_phase_advice_factory, HeadGroupPolicy, PhaseCoverPolicy};
pub use net::TwinNet;
pub use root::RootGates;
pub use root::RootPrecisionExecutionEvidence;
/// #root-box-source: the `margin_row.root_box_source` selector, and the
/// composition conflicts that refuse it (root build and preset validation).
pub use root_box::{PatchBackwardConflicts, RootBoxSource};
/// #bbe-joint-root: the `margin_row.root_search` selector and the injected
/// backend's registration (ny-cli calls it once at startup; the factory runs
/// only on the first ARMED `joint_f32` phase).
pub use root_search::{set_root_search_backend_factory, RootSearch};
pub use rounding::RoundMode;
pub use spec::{TwinOpSpec, TwinSpec};
pub use support_program::{
    PreparedHeadPreactivationRow, PreparedHeadPreactivationRows, PreparedInputHalfspaceBatch,
    PreparedInputRows, PreparedInputSplit, PreparedInputSplitSide, PreparedModelDagLiveness,
    PreparedRankedInputRow, PreparedRankedRowAdvice, PreparedRankedRowAdviceLayer,
    PreparedReferenceArithmetic, PreparedReferenceReceipt, PreparedReplayLeaf,
    PreparedReplayLiteral, PreparedReplayStats, PreparedRootAttemptReceipt,
    PreparedRootCompilation, PreparedRootConsumerOptions, PreparedRootOpen, PreparedRootOutcome,
    PreparedRootProof, PreparedRootProofOptions, PreparedRootRetention, PreparedRootSession,
    PreparedRowStorageReceipt,
};
pub use support_program::{
    PreparedSparseInputHalfspace, PreparedSparseInputRow, PreparedSparseInputRows,
};
/// #wide-fd-search (ENGINE_PROGRAM M4): the `margin_row.frozen_dual_bab_search`
/// selector and the injected level-search backends' registration (ny-cli calls
/// it once per kind at startup; a factory runs only on the first ARMED class).
pub use zono_bab::{set_frozen_dual_search_backend_factory, FdBackendKind, FrozenDualSearch};

/// Immutable entry-time policy for the complete `#epoch-bab` family.
///
/// The cadence, attempt cap, Tier-0 widths, retained-row budget, incremental
/// parent width, and diagnostic gate are one coupled policy: retention and
/// generation are selected from these values before the root is built, while
/// nested epochs must inherit the same resolved choices. Scored runs derive
/// the policy from their receipt-owning [`ny_levers::LeverSet`]; compatibility
/// callers may resolve the legacy environment/preset layers once at entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochRuntimePolicy {
    pub depth: usize,
    pub max_attempts: usize,
    pub tier0_exact: usize,
    pub tier0_universe: usize,
    pub retain_per_layer: usize,
    pub parent_topk: usize,
    /// Historical `std::env::var(...).is_ok()` semantics: every present UTF-8
    /// value arms the trace (including `"0"` and `""`), while non-UTF-8 is
    /// present in the receipt but remains dark.
    pub debug_trace: bool,
}

impl Default for EpochRuntimePolicy {
    fn default() -> Self {
        Self {
            depth: 0,
            max_attempts: 2,
            tier0_exact: 0,
            tier0_universe: 64,
            retain_per_layer: 128,
            parent_topk: 0,
            debug_trace: false,
        }
    }
}

impl EpochRuntimePolicy {
    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let resolved = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let platform_usize = |decl: &'static ny_levers::LeverDecl| {
            resolved(decl)?
                .value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
        };
        let debug = resolved(&d::EPOCH_DEBUG)?;
        Ok(Self {
            depth: platform_usize(&d::EPOCH_DEPTH)?,
            max_attempts: platform_usize(&d::EPOCH_ATTEMPTS)?,
            tier0_exact: platform_usize(&d::EPOCH_TIER0)?,
            tier0_universe: platform_usize(&d::EPOCH_UNIVERSE)?,
            retain_per_layer: platform_usize(&d::EPOCH_RETAIN)?,
            parent_topk: platform_usize(&d::MARGIN_ROW_EPOCH_PARENT_TOPK)?,
            debug_trace: debug.env_utf8 == Some(true),
        })
    }

    /// Compatibility resolver for callers without a scored frozen lever set.
    /// This is the sole compatibility read of every epoch preset/environment
    /// input; the lane and any nested trees receive only these plain values.
    fn from_layered_process() -> Self {
        use ny_levers::decls::margin_row as d;

        let (depth_preset, attempts_preset, parent_topk_preset) = epoch_preset();
        Self {
            depth: epoch_over_preset(&d::EPOCH_DEPTH, depth_preset, 0),
            max_attempts: epoch_over_preset(&d::EPOCH_ATTEMPTS, attempts_preset, 2),
            tier0_exact: crate::preset_gates::epoch_tier0(),
            tier0_universe: crate::preset_gates::epoch_universe(),
            retain_per_layer: crate::preset_gates::epoch_retain(),
            parent_topk: epoch_over_preset(&d::MARGIN_ROW_EPOCH_PARENT_TOPK, parent_topk_preset, 0),
            debug_trace: ny_levers::read_raw(&d::EPOCH_DEBUG).is_some(),
        }
    }
}

/// Immutable entry-time policy for the ordinary margin-row branch widths.
///
/// `k_head`/`k_trunk` and `adaptive` are the frozen effective treatment.
/// Their contextual peers are the exact values exposed by
/// [`ny_levers::Resolved::lower_value`], so execution evidence can compare the
/// same ranked identity population without reopening environment or preset
/// state after the run starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarginRowWidthRuntimePolicy {
    pub k_head: usize,
    pub contextual_k_head: usize,
    pub k_trunk: usize,
    pub contextual_k_trunk: usize,
    pub adaptive: bool,
    pub contextual_adaptive: bool,
}

impl Default for MarginRowWidthRuntimePolicy {
    fn default() -> Self {
        Self {
            k_head: 8,
            contextual_k_head: 8,
            k_trunk: 8,
            contextual_k_trunk: 8,
            adaptive: false,
            contextual_adaptive: false,
        }
    }
}

impl MarginRowWidthRuntimePolicy {
    fn platform_usize(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<usize, String> {
        value
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
    }

    fn width(
        decl: &'static ny_levers::LeverDecl,
        value: &ny_levers::LeverValue,
    ) -> Result<usize, String> {
        // Historical typed delivery used `0` as the process-cell's silent
        // sentinel, so a typed Some(0) selected the built-in width 8. The new
        // declaration intentionally admits config zero to preserve that
        // distinction while environment zero remains an ignored override.
        Self::platform_usize(decl, value).map(|value| if value == 0 { 8 } else { value })
    }

    fn from_resolved(
        head: &ny_levers::Resolved,
        trunk: &ny_levers::Resolved,
        adaptive: &ny_levers::Resolved,
    ) -> Result<Self, String> {
        Ok(Self {
            k_head: Self::width(head.decl, &head.value)?,
            contextual_k_head: Self::width(head.decl, &head.lower_value)?,
            k_trunk: Self::width(trunk.decl, &trunk.value)?,
            contextual_k_trunk: Self::width(trunk.decl, &trunk.lower_value)?,
            adaptive: adaptive.value.as_bool(),
            contextual_adaptive: adaptive.lower_value.as_bool(),
        })
    }

    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::{margin_row as d, sound_channel as s};

        let resolved = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        Self::from_resolved(
            resolved(&d::MARGIN_ROW_K_HEAD)?,
            resolved(&d::MARGIN_ROW_K_TRUNK)?,
            resolved(&s::MARGIN_ROW_K_ADAPT)?,
        )
    }

    /// Compatibility resolver for non-scored callers. Legacy process input
    /// and preset cells are resolved at lane entry; downstream and nested
    /// drivers receive only the resulting plain values.
    fn from_layered_process() -> Self {
        use ny_levers::decls::{margin_row as d, sound_channel as s};

        let head = ny_levers::read_over_config(
            &d::MARGIN_ROW_K_HEAD,
            k_preset(&K_HEAD_PRESET).map(|value| {
                ny_levers::LeverValue::U64(
                    u64::try_from(value).expect("branch-head preset must fit u64"),
                )
            }),
        )
        .expect("the branch-head preset must satisfy its declaration");
        let trunk = ny_levers::read_over_config(
            &d::MARGIN_ROW_K_TRUNK,
            k_preset(&K_TRUNK_PRESET).map(|value| {
                ny_levers::LeverValue::U64(
                    u64::try_from(value).expect("branch-trunk preset must fit u64"),
                )
            }),
        )
        .expect("the branch-trunk preset must satisfy its declaration");
        let adaptive = ny_levers::read_over_config(
            &s::MARGIN_ROW_K_ADAPT,
            k_adaptive_preset().map(ny_levers::LeverValue::Bool),
        )
        .expect("the adaptive-width preset must satisfy its declaration");
        Self::from_resolved(&head, &trunk, &adaptive)
            .expect("the width declarations must resolve to their declared types")
    }
}

#[cfg(test)]
mod margin_row_width_runtime_policy_tests {
    use super::MarginRowWidthRuntimePolicy;

    fn resolved(
        decl: &'static ny_levers::LeverDecl,
        raw: Option<&str>,
        config: Option<ny_levers::LeverValue>,
    ) -> ny_levers::Resolved {
        let raw = raw.map(str::to_owned);
        ny_levers::read_over_config_with(decl, move |_| raw, config)
            .expect("test value must satisfy its declaration")
    }

    fn policy(
        head_raw: Option<&str>,
        head_config: Option<u64>,
        trunk_raw: Option<&str>,
        trunk_config: Option<u64>,
        adaptive_raw: Option<&str>,
        adaptive_config: Option<bool>,
    ) -> MarginRowWidthRuntimePolicy {
        use ny_levers::decls::{margin_row as d, sound_channel as s};
        let head = resolved(
            &d::MARGIN_ROW_K_HEAD,
            head_raw,
            head_config.map(ny_levers::LeverValue::U64),
        );
        let trunk = resolved(
            &d::MARGIN_ROW_K_TRUNK,
            trunk_raw,
            trunk_config.map(ny_levers::LeverValue::U64),
        );
        let adaptive = resolved(
            &s::MARGIN_ROW_K_ADAPT,
            adaptive_raw,
            adaptive_config.map(ny_levers::LeverValue::Bool),
        );
        MarginRowWidthRuntimePolicy::from_resolved(&head, &trunk, &adaptive)
            .expect("declared width types")
    }

    #[test]
    fn typed_zero_stays_silent_in_effective_and_contextual_widths() {
        let silent = policy(None, Some(0), None, Some(0), None, None);
        assert_eq!(silent.k_head, 8);
        assert_eq!(silent.contextual_k_head, 8);
        assert_eq!(silent.k_trunk, 8);
        assert_eq!(silent.contextual_k_trunk, 8);

        let env_over_silent = policy(Some("16"), Some(0), Some("32"), Some(0), None, None);
        assert_eq!(env_over_silent.k_head, 16);
        assert_eq!(env_over_silent.contextual_k_head, 8);
        assert_eq!(env_over_silent.k_trunk, 32);
        assert_eq!(env_over_silent.contextual_k_trunk, 8);
    }

    #[test]
    fn invalid_env_falls_through_typed_width_and_bool_lower_is_exact() {
        let frozen = policy(
            Some("0"),
            Some(16),
            Some("bad"),
            Some(32),
            Some("0"),
            Some(true),
        );
        assert_eq!(frozen.k_head, 16);
        assert_eq!(frozen.contextual_k_head, 8);
        assert_eq!(frozen.k_trunk, 32);
        assert_eq!(frozen.contextual_k_trunk, 8);
        assert!(!frozen.adaptive);
        assert!(frozen.contextual_adaptive);
    }
}

/// Immutable entry-time policy for the margin-row controls migrated away from
/// process-global environment latches.
///
/// Scored VNN-COMP runs construct this from the same [`ny_levers::LeverSet`]
/// that owns their flight receipt. Compatibility and research callers still
/// accept legacy environment input, but freeze it once at lane entry and carry
/// these plain values through the complete tree (including nested epochs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginRowRuntimePolicy {
    /// Frozen ordinary branch-width and adaptive-width policy.
    pub width: MarginRowWidthRuntimePolicy,
    /// Frozen root-tableau compute and storage precision policy.
    pub root_precision: RootPrecisionRuntimePolicy,
    /// Frozen backward-intermediate arming and tuning policy.
    pub backward_interm: BackwardIntermRuntimePolicy,
    /// Frozen standalone-alpha and joint-rebake policy.
    pub root_alpha: RootAlphaRuntimePolicy,
    /// #joint-refresh: frozen backward-only joint intermediate refresh policy
    /// (default `rounds: 0` = dark, byte-identical).
    pub joint_refresh: JointRefreshRuntimePolicy,
    /// #root-certify-once: frozen search-untrusted / certify-once root alpha
    /// policy (default `enabled: false` = dark, byte-identical).
    pub root_certify_once: RootCertifyOnceRuntimePolicy,
    /// Optional complete head-group relaxation on the unconditional outer
    /// frame; the same frozen lever is serialized in the scored receipt.
    pub head_groups: Option<HeadGroupPolicy>,
    /// Frozen legacy classwise schedule; shared by scored route admission and
    /// actual lane dispatch, with no later process-environment lookup.
    pub classwise: bool,
    /// Frozen per-domain alpha gate and ascent schedule.
    pub domain_alpha: DomainAlphaRuntimePolicy,
    /// Frozen Tier-0/Tier-2 epoch, retention, and trace policy.
    pub epoch: EpochRuntimePolicy,
    /// Give selected failing output columns independent beta state.
    pub beta_per_column: bool,
    /// Maximum number of independently optimized beta columns.
    pub beta_columns: usize,
    /// Inherit certificate-closed margin columns into descendants.
    pub column_retire: bool,
    /// Prebuild y-packs for executable multi-domain frontier batches.
    pub parallel_prebuild: bool,
    /// #wave-beta-bab (ENGINE_PROGRAM M6): the outer frame's tree engine
    /// (`margin_row.tree_mode` / `tree_wave_batch` / `tree_wave_cert_threads`),
    /// resolved from the SAME frozen LeverSet the flight receipt serialises, so
    /// the lane and its receipt have one authority. Default `legacy`: the
    /// established tree, bit for bit.
    pub wave: WavePolicy,
}

/// Immutable entry-time policy for per-domain alpha re-optimization.
///
/// The scored route resolves the whole family from its receipt-owning lever
/// set and carries the values through the complete tree. Compatibility callers
/// may still use the historical preset cells, but those cells are consumed
/// once at lane entry rather than reopened from each domain evaluation.
///
/// The three SCHEDULE fields decide WHICH admitted domains pay for the ascent
/// (`domain_alpha::schedule_admits_with`); they are carried here RESOLVED, so
/// `gap` is `f64::INFINITY` when no closure window is set (the declaration's
/// `0.0` default and every non-positive or non-finite input all mean "no
/// window", the `MARGIN_ROW_DOMAIN_ALPHA_ETA` convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainAlphaRuntimePolicy {
    /// Arm per-domain lower-line slope optimization.
    pub enabled: bool,
    /// Maximum certified ascent trials per eligible domain.
    pub iters: usize,
    /// Initial normalized ascent step in `[0, 1]` slope units.
    pub eta: f64,
    /// Schedule: skip the ascent at domains shallower than this split depth.
    pub min_depth: usize,
    /// Schedule: pay the ascent only where `depth % stride == 0`.
    pub stride: usize,
    /// Schedule: pay only within this much of closure (`b >= -gap`);
    /// `f64::INFINITY` = no window.
    pub gap: f64,
}

impl Default for DomainAlphaRuntimePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            iters: 2,
            eta: 0.25,
            min_depth: 0,
            stride: 1,
            gap: f64::INFINITY,
        }
    }
}

impl DomainAlphaRuntimePolicy {
    pub(crate) fn normalized(self) -> Self {
        let defaults = Self::default();
        Self {
            enabled: self.enabled,
            iters: self.iters.clamp(1, 8),
            eta: if self.eta.is_finite() && self.eta > 0.0 {
                self.eta.clamp(domain_alpha::ETA_MIN, domain_alpha::ETA_MAX)
            } else {
                defaults.eta
            },
            min_depth: self.min_depth.min(64),
            stride: self.stride.clamp(1, 16),
            gap: if self.gap.is_finite() && self.gap > 0.0 {
                self.gap
            } else {
                f64::INFINITY
            },
        }
    }

    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let iters = value(&d::MARGIN_ROW_DOMAIN_ALPHA_ITERS)?
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_DOMAIN_ALPHA_ITERS.name
                )
            })?
            .clamp(1, 8);
        let eta = value(&d::MARGIN_ROW_DOMAIN_ALPHA_ETA)?
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                format!(
                    "{} is not a finite number",
                    d::MARGIN_ROW_DOMAIN_ALPHA_ETA.name
                )
            })?;
        let unsigned = |decl: &'static ny_levers::LeverDecl| -> Result<usize, String> {
            value(decl)?
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
        };
        let min_depth = unsigned(&d::MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH)?;
        let stride = unsigned(&d::MARGIN_ROW_DOMAIN_ALPHA_STRIDE)?;
        // The declaration's `0.0` default (and any non-positive value) means
        // "no window"; `normalized` maps those onto `f64::INFINITY`.
        let gap = value(&d::MARGIN_ROW_DOMAIN_ALPHA_GAP)?
            .as_f64()
            .ok_or_else(|| format!("{} is not a number", d::MARGIN_ROW_DOMAIN_ALPHA_GAP.name))?;
        Ok(Self {
            enabled: value(&d::MARGIN_ROW_DOMAIN_ALPHA)?.as_bool(),
            iters,
            eta,
            min_depth,
            stride,
            gap,
        }
        .normalized())
    }

    /// Compatibility resolver for non-scored callers using the historical
    /// layered preset cells. This is the only compatibility read; domain
    /// evaluation consumes the resulting plain values.
    fn from_layered_process() -> Self {
        Self {
            enabled: domain_alpha::enabled(),
            iters: domain_alpha::iters(),
            eta: domain_alpha::eta0(),
            min_depth: domain_alpha::min_depth(),
            stride: domain_alpha::stride(),
            gap: domain_alpha::gap(),
        }
        .normalized()
    }
}

/// Immutable entry-time policy for the complete root-alpha/rebake family.
///
/// `NY_MARGIN_ROW_ALPHA_LR` and `NY_MARGIN_ROW_ALPHA_COLS` are shared by the
/// standalone optimizer and every rebake round, so the two mechanisms must be
/// frozen as one policy.  Scored runs derive these values from the same lever
/// set as the flight receipt and then pass plain values through both routes;
/// neither optimizer may reopen process-global preset or environment state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootAlphaRuntimePolicy {
    pub alpha_enabled: bool,
    pub alpha_iters: usize,
    pub alpha_secs: f64,
    pub alpha_lr: f64,
    pub alpha_cols: usize,
    pub alpha_guard: bool,
    pub rebake_enabled: bool,
    pub rebake_rounds: usize,
    pub rebake_secs: f64,
    /// #spec-alpha: arm the per-START-NODE (spec) alpha axis at the root
    /// direct pass. Default OFF; armed by no shipped yaml.
    pub spec_alpha_enabled: bool,
    /// `W`: hardest failing classes that receive candidates.
    pub spec_alpha_width: usize,
    /// `E`: candidates per selected class.
    pub spec_alpha_cands: usize,
    /// `1` = closed-form ladder only; `>= 2` adds gradient rounds.
    pub spec_alpha_rounds: usize,
    /// Hard cap on committed exception entries. `0` refuses and says so.
    pub spec_alpha_entries: usize,
}

impl Default for RootAlphaRuntimePolicy {
    fn default() -> Self {
        Self {
            alpha_enabled: false,
            alpha_iters: 20,
            alpha_secs: 20.0,
            alpha_lr: 0.1,
            alpha_cols: 8,
            alpha_guard: true,
            rebake_enabled: false,
            rebake_rounds: 2,
            rebake_secs: 30.0,
            spec_alpha_enabled: false,
            spec_alpha_width: 3,
            spec_alpha_cands: 3,
            spec_alpha_rounds: 1,
            spec_alpha_entries: 65_536,
        }
    }
}

impl RootAlphaRuntimePolicy {
    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::{margin_row as m, sound_channel as s};

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let platform_usize = |decl: &'static ny_levers::LeverDecl| {
            value(decl)?
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
        };
        let finite = |decl: &'static ny_levers::LeverDecl| {
            value(decl)?
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| format!("{} is not a finite number", decl.name))
        };

        let alpha_secs = finite(&s::MARGIN_ROW_ALPHA_SECS)?;
        let rebake_secs = finite(&m::MARGIN_ROW_REBAKE_SECS)?;
        Ok(Self {
            alpha_enabled: value(&s::MARGIN_ROW_ALPHA_OPT)?.as_bool(),
            alpha_iters: platform_usize(&s::MARGIN_ROW_ALPHA_ITERS)?.clamp(1, 64),
            alpha_secs: if alpha_secs > 0.0 { alpha_secs } else { 20.0 },
            alpha_lr: finite(&s::MARGIN_ROW_ALPHA_LR)?.clamp(0.0, 1.0),
            alpha_cols: platform_usize(&s::MARGIN_ROW_ALPHA_COLS)?.clamp(1, 64),
            alpha_guard: value(&s::MARGIN_ROW_ALPHA_GUARD)?.as_bool(),
            rebake_enabled: value(&m::MARGIN_ROW_REBAKE)?.as_bool(),
            rebake_rounds: platform_usize(&m::MARGIN_ROW_REBAKE_ROUNDS)?.clamp(1, 8),
            rebake_secs: if rebake_secs >= 0.0 {
                rebake_secs
            } else {
                30.0
            },
            spec_alpha_enabled: value(&s::MARGIN_ROW_SPEC_ALPHA)?.as_bool(),
            spec_alpha_width: platform_usize(&s::MARGIN_ROW_SPEC_ALPHA_WIDTH)?.clamp(1, 64),
            spec_alpha_cands: platform_usize(&s::MARGIN_ROW_SPEC_ALPHA_CANDS)?.clamp(1, 8),
            spec_alpha_rounds: platform_usize(&s::MARGIN_ROW_SPEC_ALPHA_ROUNDS)?.clamp(1, 3),
            spec_alpha_entries: platform_usize(&s::MARGIN_ROW_SPEC_ALPHA_ENTRIES)?.min(4_194_304),
        })
    }

    /// Compatibility resolver for callers without a scored frozen lever set.
    /// This is the sole compatibility read of the legacy preset/env cells.
    fn from_layered_process() -> Self {
        let alpha_secs = crate::preset_gates::margin_row_alpha_secs();
        let rebake_secs = crate::preset_gates::margin_row_rebake_secs();
        Self {
            alpha_enabled: crate::preset_gates::margin_row_alpha_opt_enabled(),
            alpha_iters: crate::preset_gates::margin_row_alpha_iters().clamp(1, 64),
            alpha_secs: if alpha_secs.is_finite() && alpha_secs > 0.0 {
                alpha_secs
            } else {
                20.0
            },
            alpha_lr: crate::preset_gates::margin_row_alpha_lr().clamp(0.0, 1.0),
            alpha_cols: crate::preset_gates::margin_row_alpha_cols().clamp(1, 64),
            alpha_guard: crate::preset_gates::margin_row_alpha_guard_enabled(),
            rebake_enabled: crate::preset_gates::margin_row_rebake_enabled(),
            rebake_rounds: crate::preset_gates::margin_row_rebake_rounds().clamp(1, 8),
            rebake_secs: if rebake_secs.is_finite() && rebake_secs >= 0.0 {
                rebake_secs
            } else {
                30.0
            },
            spec_alpha_enabled: crate::preset_gates::margin_row_spec_alpha_enabled(),
            spec_alpha_width: crate::preset_gates::margin_row_spec_alpha_width().clamp(1, 64),
            spec_alpha_cands: crate::preset_gates::margin_row_spec_alpha_cands().clamp(1, 8),
            spec_alpha_rounds: crate::preset_gates::margin_row_spec_alpha_rounds().clamp(1, 3),
            spec_alpha_entries: crate::preset_gates::margin_row_spec_alpha_entries().min(4_194_304),
        }
    }
}

/// Immutable entry-time policy for root-tableau compute and storage precision.
///
/// The two values must travel together because f32 storage is meaningful only
/// when f32 compute is effective. Keeping them in the receipt-owned runtime
/// policy also prevents rebakes and nested epoch roots from observing a later
/// process-global mutation than the initial root did.
///
/// #mr-errlane: `err_lane` rides here because this is the one receipt-owned
/// precision authority that EVERY root carries — the initial build, rebakes,
/// nested epoch roots, the parent-frame refresh child and the backward-interm
/// scratch root — and every [`engine::BackwardEngine`] copies it at
/// construction. It selects how the certified coefficient-error lane is
/// carried (per column, or as a dominating envelope), never what is computed.
///
/// #mr-ibp-seed: `ibp_seed` rides here for the same reason as `err_lane`. The
/// seed is a property of the ROOT TABLEAU WALK, and this is the one frozen
/// authority every walk receives: the initial root build, every `#rebake`
/// round (`root.root_precision_policy`) and every Tier-2 epoch rebuild
/// (`BabConfig::root_precision_policy`). Carrying it anywhere else would let a
/// rebake or an epoch rebuild silently drop the seed and publish a looser
/// frame than the root it replaces. The backward-interm scratch root builds no
/// tableau and keeps the default (`false`).
///
/// #root-box-source: `box_source` rides here for the same reason as
/// `ibp_seed`: it selects how EVERY root walk builds its trunk boxes — the
/// initial build, each `#rebake` round and each Tier-2 epoch rebuild — so a
/// descendant root can never silently switch source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RootPrecisionRuntimePolicy {
    pub root_f32: bool,
    pub tableau_f32: bool,
    /// Certified error-lane representation (`margin_row.err_envelope`).
    pub err_lane: ErrLane,
    /// #mr-ibp-seed (`margin_row.ibp_seed`): intersect every trunk box of the
    /// root tableau walk with its certified IBP box (`ibp_seed.rs`). `false`
    /// (the declaration default) never builds the interval state: the walk is
    /// byte-identical and prints nothing.
    pub ibp_seed: bool,
    /// #root-box-source (`margin_row.root_box_source`, ENGINE_PROGRAM M5b):
    /// which certified construction supplies the trunk boxes. `Tableau` (the
    /// declaration default) is today's forward tableau, bit for bit and
    /// silent; `PatchBackward` is `root_box.rs` (Outward only; a Parity build
    /// or a decline runs the tableau).
    pub box_source: RootBoxSource,
}

impl RootPrecisionRuntimePolicy {
    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let resolved_bool = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.as_bool())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let err_lane = levers
            .resolved(&d::MARGIN_ROW_ERR_ENVELOPE)
            .ok_or_else(|| {
                format!(
                    "{} is missing from the frozen LeverSet",
                    d::MARGIN_ROW_ERR_ENVELOPE.name
                )
            })?
            .value
            .as_str()
            .and_then(ErrLane::from_token)
            .ok_or_else(|| {
                format!(
                    "{} is not one of {:?}",
                    d::MARGIN_ROW_ERR_ENVELOPE.name,
                    ErrLane::TOKENS
                )
            })?;
        let box_source = levers
            .resolved(&d::MARGIN_ROW_ROOT_BOX_SOURCE)
            .ok_or_else(|| {
                format!(
                    "{} is missing from the frozen LeverSet",
                    d::MARGIN_ROW_ROOT_BOX_SOURCE.name
                )
            })?
            .value
            .as_str()
            .and_then(RootBoxSource::from_token)
            .ok_or_else(|| {
                format!(
                    "{} is not one of {:?}",
                    d::MARGIN_ROW_ROOT_BOX_SOURCE.name,
                    RootBoxSource::TOKENS
                )
            })?;
        Ok(Self {
            root_f32: resolved_bool(&d::MARGIN_ROW_ROOT_F32)?,
            tableau_f32: resolved_bool(&d::MARGIN_ROW_TABLEAU_F32)?,
            err_lane,
            ibp_seed: resolved_bool(&d::MARGIN_ROW_IBP_SEED)?,
            box_source,
        })
    }

    /// Compatibility resolver for callers without a scored frozen LeverSet.
    /// This is the only place the compatibility preset cells and environment
    /// may be consulted; every root build receives the resulting plain values.
    fn from_layered_process() -> Self {
        use ny_levers::decls::margin_row as d;

        let root_f32 = ny_levers::read_over_config(
            &d::MARGIN_ROW_ROOT_F32,
            Some(ny_levers::LeverValue::Bool(root_f32_preset())),
        )
        .map(|resolved| resolved.value.as_bool())
        .unwrap_or(false);
        // #mr-errlane: env > typed preset cell > declaration (`off`), through
        // the governed `text_gate!`; an inadmissible token cannot reach here
        // (the `Enum` declaration rejects it to the default).
        let err_lane = crate::preset_gates::margin_row_err_envelope()
            .as_deref()
            .and_then(ErrLane::from_token)
            .unwrap_or_default();
        Self {
            root_f32,
            tableau_f32: crate::preset_gates::margin_row_tableau_f32_enabled(),
            err_lane,
            // #mr-ibp-seed: env > typed preset cell > declaration (`false`),
            // through the governed `bool_gate!`.
            ibp_seed: crate::preset_gates::margin_row_ibp_seed_enabled(),
            // #root-box-source: env > typed preset cell > declaration
            // (`tableau`), through the governed `text_gate!`; an inadmissible
            // token cannot reach here (the `Enum` declaration rejects it).
            box_source: crate::preset_gates::margin_row_root_box_source()
                .as_deref()
                .and_then(RootBoxSource::from_token)
                .unwrap_or_default(),
        }
    }
}

/// Immutable entry-time policy for the complete backward-intermediate family.
///
/// Keeping the arming gate and all four tuning values together prevents a
/// scored run from mixing the receipt's frozen values with later process-wide
/// environment mutations.  The policy owns only configuration; execution
/// counters live in [`BackwardIntermExecutionEvidence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardIntermRuntimePolicy {
    pub enabled: bool,
    pub secs: f64,
    pub chunk: usize,
    pub chunk_secs: f64,
    pub topk: usize,
    /// #interm-alpha: per-start-node prefix alpha candidate dose (0 = the
    /// pre-#interm-alpha phase, byte-identical; clamped to the policy count
    /// at the reader).
    pub interm_alpha: usize,
    /// #bi-window (`margin_row.bi_window`): run every prefix pass pair of this
    /// phase — and of `#joint-refresh`, which re-runs it — on a
    /// receptive-field CROP of the prefix net (`backward_interm::window`).
    /// Execution strategy only: the crop is a certified pass of the unchanged
    /// engine on a smaller net that computes exactly the seeded
    /// pre-activations. `false` (the declaration default) never enters the
    /// crop code and is byte-identical. NOT carried by `BiCfg`: a `#rebake`
    /// round's explicit config (`from_config`) runs dense.
    pub window: bool,
    /// #bi-gpu (`margin_row.bi_gpu`): run every prefix pass pair of this
    /// phase — and of `#joint-refresh` — on the certified sound-f64 GPU conv
    /// route (`bi_gpu` module), with per-pair RSS admission and the per-op
    /// bit-identical CPU fallback. Execution strategy only: the lane's
    /// unchanged order-independent charges certify the device sum. `false`
    /// (the declaration default) builds no route and is byte-identical. NOT
    /// carried by `BiCfg`: a `#rebake` round's explicit config runs on the CPU.
    pub gpu: bool,
    /// #bi-patch (`margin_row.bi_patch` / `_ratio` / `_mb`): run every prefix
    /// pass pair of this phase — and of `#joint-refresh` — as a PER-COLUMN
    /// RECEPTIVE-FIELD walk (`patch` module), falling back to the crop or the
    /// dense pair wherever the plan refuses. Execution strategy only: each
    /// patch pair is the dense pair restricted to its support (never looser).
    /// Disabled (the declaration default) never enters the patch code and is
    /// byte-identical. NOT carried by `BiCfg`: a `#rebake` round runs dense.
    pub patch: BiPatchRuntimePolicy,
}

/// #bi-patch: the frozen per-column receptive-field policy of the
/// backward-intermediate phase (see [`BackwardIntermRuntimePolicy::patch`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiPatchRuntimePolicy {
    /// `margin_row.bi_patch`: arm patch-mode prefix pairs (default false).
    /// Takes precedence over `#bi-window` on every layer it can plan.
    pub enabled: bool,
    /// `margin_row.bi_patch_ratio`: the crossover `rho` — walking backward,
    /// the first tensor whose per-column frame exceeds `rho * n` rows (and
    /// every tensor below it) runs as the dense tail. Clamped to
    /// `[0.05, 1.0]`; default 0.5.
    pub ratio: f64,
    /// `margin_row.bi_patch_mb`: the per-pass working set in MiB that caps a
    /// patch chunk's width (default 256).
    pub mb: usize,
}

impl Default for BiPatchRuntimePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            ratio: patch::DEFAULT_RATIO,
            mb: patch::DEFAULT_MB,
        }
    }
}

impl BiPatchRuntimePolicy {
    /// The phase's frozen patch configuration: `Some` only when armed.
    pub(crate) fn cfg(self) -> Option<patch::PatchCfg> {
        self.enabled
            .then(|| patch::PatchCfg::new(self.ratio, self.mb))
    }

    /// Clamp a resolved crossover ratio into its admissible range (a
    /// non-finite value takes the declaration default).
    fn clamp_ratio(ratio: f64) -> f64 {
        if ratio.is_finite() {
            ratio.clamp(patch::RATIO_MIN, patch::RATIO_MAX)
        } else {
            patch::DEFAULT_RATIO
        }
    }
}

pub(crate) const DEFAULT_BACKWARD_INTERM_CHUNK: usize = 256;

impl Default for BackwardIntermRuntimePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            secs: 20.0,
            chunk: DEFAULT_BACKWARD_INTERM_CHUNK,
            chunk_secs: 0.0,
            topk: 1024,
            interm_alpha: 0,
            window: false,
            gpu: false,
            patch: BiPatchRuntimePolicy::default(),
        }
    }
}

impl BackwardIntermRuntimePolicy {
    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let enabled = value(&d::MARGIN_ROW_BACKWARD_INTERM)?.as_bool();
        let secs = value(&d::MARGIN_ROW_BI_SECS)?.as_f64().ok_or_else(|| {
            format!(
                "{} is not a finite non-negative number",
                d::MARGIN_ROW_BI_SECS.name
            )
        })?;
        let chunk = value(&d::MARGIN_ROW_BI_CHUNK)?
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_BI_CHUNK.name
                )
            })?
            .clamp(1, 4096);
        let chunk_secs = value(&d::MARGIN_ROW_BI_CHUNK_SECS)?
            .as_f64()
            .ok_or_else(|| {
                format!(
                    "{} is not a finite non-negative number",
                    d::MARGIN_ROW_BI_CHUNK_SECS.name
                )
            })?;
        let topk = value(&d::MARGIN_ROW_BI_TOPK)?
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_BI_TOPK.name
                )
            })?
            .max(1);
        let interm_alpha = value(&d::MARGIN_ROW_INTERM_ALPHA)?
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_INTERM_ALPHA.name
                )
            })?
            .min(backward_interm::INTERM_ALPHA_POLICIES);
        let window = value(&d::MARGIN_ROW_BI_WINDOW)?.as_bool();
        let gpu = value(&d::MARGIN_ROW_BI_GPU)?.as_bool();
        let patch = BiPatchRuntimePolicy {
            enabled: value(&d::MARGIN_ROW_BI_PATCH)?.as_bool(),
            ratio: BiPatchRuntimePolicy::clamp_ratio(
                value(&d::MARGIN_ROW_BI_PATCH_RATIO)?
                    .as_f64()
                    .ok_or_else(|| {
                        format!(
                            "{} is not a finite number",
                            d::MARGIN_ROW_BI_PATCH_RATIO.name
                        )
                    })?,
            ),
            mb: value(&d::MARGIN_ROW_BI_PATCH_MB)?
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    format!(
                        "{} is not a platform-sized unsigned integer",
                        d::MARGIN_ROW_BI_PATCH_MB.name
                    )
                })?
                .max(1),
        };
        Ok(Self {
            enabled,
            secs: if secs == 0.0 { 0.0 } else { secs },
            chunk,
            chunk_secs: if chunk_secs == 0.0 { 0.0 } else { chunk_secs },
            topk,
            interm_alpha,
            window,
            gpu,
            patch,
        })
    }

    /// Compatibility resolver for callers that do not own the scored frozen
    /// [`ny_levers::LeverSet`].  Every process-global input is consumed once,
    /// here, before the lane starts; no root/rebake reader reopens it later.
    fn from_layered_process() -> Self {
        use ny_levers::decls::margin_row as d;

        let defaults = Self::default();
        let enabled = ny_levers::read_over_config(
            &d::MARGIN_ROW_BACKWARD_INTERM,
            Some(ny_levers::LeverValue::Bool(backward_interm_preset())),
        )
        .map(|resolved| resolved.value.as_bool())
        .unwrap_or(false);
        let secs = ny_levers::read_over_config(
            &d::MARGIN_ROW_BI_SECS,
            bi_secs_preset().map(ny_levers::LeverValue::F64),
        )
        .ok()
        .and_then(|resolved| resolved.value.as_f64())
        .unwrap_or(defaults.secs);
        Self {
            enabled,
            secs: if secs == 0.0 { 0.0 } else { secs },
            chunk: crate::preset_gates::margin_row_bi_chunk().clamp(1, 4096),
            chunk_secs: {
                let value = crate::preset_gates::margin_row_bi_chunk_secs();
                if value == 0.0 {
                    0.0
                } else {
                    value
                }
            },
            topk: crate::preset_gates::margin_row_bi_topk().max(1),
            interm_alpha: crate::preset_gates::margin_row_interm_alpha()
                .min(backward_interm::INTERM_ALPHA_POLICIES),
            window: crate::preset_gates::margin_row_bi_window_enabled(),
            gpu: crate::preset_gates::margin_row_bi_gpu_enabled(),
            patch: BiPatchRuntimePolicy {
                enabled: crate::preset_gates::margin_row_bi_patch_enabled(),
                ratio: BiPatchRuntimePolicy::clamp_ratio(
                    crate::preset_gates::margin_row_bi_patch_ratio(),
                ),
                mb: crate::preset_gates::margin_row_bi_patch_mb().max(1),
            },
        }
    }

    pub(crate) fn config(self, mode: RoundMode) -> Option<backward_interm::BiCfg> {
        (self.enabled && mode.outward()).then_some(backward_interm::BiCfg {
            // #bi-yield ARM (experiment build only — the shipped default is 0/0.0):
            // evaluate the marginal tightening rate every 4 completed chunks and
            // stop when a window falls below 25% of the best window seen.
            // A window of 1 = evaluate after EVERY layer. Measured: this phase
            // completes only FOUR layers inside its whole budget, so any window
            // >= 4 makes the first evaluation land on the last layer and the
            // check is inert. See `a_window_at_the_layer_count_is_inert`.
            yield_window: 0,
            yield_ratio: 0.25,
            secs: self.secs,
            chunk: self.chunk,
            chunk_secs: self.chunk_secs,
            topk: self.topk,
            interm_alpha: self.interm_alpha,
        })
    }

    pub(crate) fn from_config(config: Option<backward_interm::BiCfg>) -> Self {
        config.map_or_else(Self::default, |config| Self {
            enabled: true,
            secs: config.secs,
            chunk: config.chunk.clamp(1, 4096),
            chunk_secs: config.chunk_secs,
            topk: config.topk.max(1),
            interm_alpha: config
                .interm_alpha
                .min(backward_interm::INTERM_ALPHA_POLICIES),
            // `BiCfg` carries no crop flag: an explicit (rebake-round) config
            // runs the dense phase, exactly as before #bi-window existed.
            window: false,
            // Nor a device flag (#bi-gpu): rebake rounds run on the CPU.
            gpu: false,
            // Nor patch mode (#bi-patch): rebake rounds run the dense pairs.
            patch: BiPatchRuntimePolicy::default(),
        })
    }
}

/// #joint-refresh (cct plan P5 step 2): immutable entry-time policy for the
/// backward-only JOINT intermediate refresh (`rebake::maybe_joint_refresh_root`).
///
/// Each round re-derives the root's intermediate boxes from its CURRENT gates
/// (the committed alpha included) with the lane's own certified backward engine
/// — shrink-only, re-certified gates — then runs a short alpha ascent whose
/// frozen head frame (y-box, head gates, m1/m2v) is rebuilt from the refreshed
/// gates, and keeps the round only if the unchanged certified `root_eval`
/// bound strictly improves. The per-round selection width, chunk and crop flag
/// are the frozen `#backward-interm` policy's; the refresh refuses unless that
/// phase is armed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointRefreshRuntimePolicy {
    /// Refresh rounds (`margin_row.joint_refresh`). `0` — the declaration
    /// default — disarms the phase: the entry returns before touching
    /// anything and prints nothing, so the lane is byte-identical.
    pub rounds: usize,
    /// Wall request in seconds (`margin_row.joint_refresh_secs`, default 30),
    /// further capped to 40% of the remaining deadline.
    pub secs: f64,
}

impl Default for JointRefreshRuntimePolicy {
    fn default() -> Self {
        Self {
            rounds: 0,
            secs: 30.0,
        }
    }
}

impl JointRefreshRuntimePolicy {
    /// Hard ceiling on rounds; the wall request is the binding cap in practice.
    pub(crate) const MAX_ROUNDS: usize = 16;

    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let rounds = value(&d::MARGIN_ROW_JOINT_REFRESH)?
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_JOINT_REFRESH.name
                )
            })?
            .min(Self::MAX_ROUNDS);
        let secs = value(&d::MARGIN_ROW_JOINT_REFRESH_SECS)?
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                format!(
                    "{} is not a finite number",
                    d::MARGIN_ROW_JOINT_REFRESH_SECS.name
                )
            })?;
        Ok(Self {
            rounds,
            secs: if secs >= 0.0 { secs } else { 30.0 },
        })
    }

    /// Compatibility resolver for callers without a scored frozen LeverSet.
    fn from_layered_process() -> Self {
        let secs = crate::preset_gates::margin_row_joint_refresh_secs();
        Self {
            rounds: crate::preset_gates::margin_row_joint_refresh().min(Self::MAX_ROUNDS),
            secs: if secs.is_finite() && secs >= 0.0 {
                secs
            } else {
                30.0
            },
        }
    }
}

/// #root-certify-once (VIBECHECK_PLAN "Phase-0.5 root alpha, search-untrusted /
/// certify-once"): immutable entry-time policy for
/// `root_certify_once::maybe_certify_once_root`.
///
/// Armed, the root alpha ascent runs UNCERTIFIED on a Parity scratch copy of
/// the root (optionally alternating with an untrusted refresh of its
/// intermediate boxes), then ONE certified evaluation of the chosen alpha is
/// published only if it strictly beats the certified incumbent — the root the
/// lane would publish without the phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootCertifyOnceRuntimePolicy {
    /// Arm the phase (`margin_row.root_certify_once`). `false` — the
    /// declaration default — returns before touching anything and prints
    /// nothing, so the lane is byte-identical.
    pub enabled: bool,
    /// Untrusted spec-phase ascent iterations
    /// (`margin_row.root_certify_once_iters`, default 20, clamped to
    /// `1..=MAX_ITERS`).
    pub iters: usize,
    /// Untrusted joint-intermediate rounds before the spec phase
    /// (`margin_row.root_certify_once_refresh`, default 0, clamped to
    /// `MAX_REFRESH`); each is an untrusted refresh of every trunk box plus one
    /// untrusted ascent step. Needs `#backward-interm` armed.
    pub refresh: usize,
    /// Wall request in seconds (`margin_row.root_certify_once_secs`, default
    /// 20), further capped to 40% of the remaining deadline.
    pub secs: f64,
    /// #bbe-joint-root: which UNTRUSTED search proposes the slopes
    /// (`margin_row.root_search`, default `ascent` = today's phase, bit for
    /// bit; `joint_f32` = the injected joint-alpha root search, M5a). The
    /// certify half is the same either way.
    pub search: RootSearch,
}

impl Default for RootCertifyOnceRuntimePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            iters: 20,
            refresh: 0,
            secs: 20.0,
            search: RootSearch::Ascent,
        }
    }
}

impl RootCertifyOnceRuntimePolicy {
    /// Hard ceiling on untrusted spec iterations; the wall request binds first
    /// in practice.
    pub(crate) const MAX_ITERS: usize = 128;
    /// Hard ceiling on untrusted joint rounds.
    pub(crate) const MAX_REFRESH: usize = 16;

    fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let unsigned = |decl: &'static ny_levers::LeverDecl| -> Result<usize, String> {
            value(decl)?
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
        };
        let secs = value(&d::MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS)?
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                format!(
                    "{} is not a finite number",
                    d::MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS.name
                )
            })?;
        let search = value(&d::MARGIN_ROW_ROOT_SEARCH)?
            .as_str()
            .and_then(RootSearch::from_token)
            .ok_or_else(|| {
                format!(
                    "{} is not one of {:?}",
                    d::MARGIN_ROW_ROOT_SEARCH.name,
                    RootSearch::TOKENS
                )
            })?;
        Ok(Self {
            enabled: value(&d::MARGIN_ROW_ROOT_CERTIFY_ONCE)?.as_bool(),
            iters: unsigned(&d::MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS)?.clamp(1, Self::MAX_ITERS),
            refresh: unsigned(&d::MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH)?.min(Self::MAX_REFRESH),
            secs: if secs >= 0.0 { secs } else { 20.0 },
            search,
        })
    }

    /// Compatibility resolver for callers without a scored frozen LeverSet.
    fn from_layered_process() -> Self {
        let secs = crate::preset_gates::margin_row_root_certify_once_secs();
        Self {
            enabled: crate::preset_gates::margin_row_root_certify_once_enabled(),
            iters: crate::preset_gates::margin_row_root_certify_once_iters()
                .clamp(1, Self::MAX_ITERS),
            refresh: crate::preset_gates::margin_row_root_certify_once_refresh()
                .min(Self::MAX_REFRESH),
            secs: if secs.is_finite() && secs >= 0.0 {
                secs
            } else {
                20.0
            },
            // #bbe-joint-root: env > typed preset cell > declaration
            // (`ascent`), through the governed `text_gate!`; an inadmissible
            // token cannot reach here (the `Enum` declaration rejects it).
            search: crate::preset_gates::margin_row_root_search()
                .as_deref()
                .and_then(RootSearch::from_token)
                .unwrap_or_default(),
        }
    }
}

impl Default for MarginRowRuntimePolicy {
    fn default() -> Self {
        Self {
            width: MarginRowWidthRuntimePolicy::default(),
            root_precision: RootPrecisionRuntimePolicy::default(),
            backward_interm: BackwardIntermRuntimePolicy::default(),
            root_alpha: RootAlphaRuntimePolicy::default(),
            joint_refresh: JointRefreshRuntimePolicy::default(),
            root_certify_once: RootCertifyOnceRuntimePolicy::default(),
            head_groups: HeadGroupPolicy::new(5.0, 2, 4),
            classwise: false,
            domain_alpha: DomainAlphaRuntimePolicy::default(),
            epoch: EpochRuntimePolicy::default(),
            beta_per_column: false,
            beta_columns: 8,
            column_retire: false,
            parallel_prebuild: false,
            wave: WavePolicy::default(),
        }
    }
}

impl MarginRowRuntimePolicy {
    /// Resolve from the run's already-frozen, receipt-owning lever set.
    ///
    /// Missing or ill-typed entries are rejected instead of silently falling
    /// back to a second authority. A normal full workspace set is total, so an
    /// error indicates a programming/configuration boundary mismatch.
    pub fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        Self::from_levers_inner(levers, true)
    }

    fn from_levers_inner(
        levers: &ny_levers::LeverSet,
        reject_invalid_phase_cover: bool,
    ) -> Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let value = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let beta_per_column = value(&d::MARGIN_ROW_BETA_PERCOL)?.as_bool();
        let beta_columns = value(&d::MARGIN_ROW_BETA_COLS)?
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_BETA_COLS.name
                )
            })?
            .clamp(1, 32);
        let column_retire = value(&d::MARGIN_ROW_COL_RETIRE)?.as_bool();
        let parallel_prebuild = value(&d::MARGIN_ROW_PARALLEL_PREBUILD)?.as_bool();
        let classwise = margin_row_classwise_from_env(value(&d::MARGIN_ROW_CLASSWISE)?.as_str());
        let phase_cover_count = value(&d::MARGIN_ROW_HEAD_PHASE_COVER)?
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| {
                format!(
                    "{} is not a platform-sized unsigned integer",
                    d::MARGIN_ROW_HEAD_PHASE_COVER.name
                )
            })?;
        let phase_cover = PhaseCoverPolicy::new(phase_cover_count);
        if phase_cover_count != 0 && phase_cover.is_none() && reject_invalid_phase_cover {
            return Err(format!(
                "{} must be 0 (off) or 1..=7",
                d::MARGIN_ROW_HEAD_PHASE_COVER.name
            ));
        }
        let directional_cuts = value(&d::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS)?.as_bool();
        let head_groups = value(&d::MARGIN_ROW_HEAD_GROUPS)?.as_bool().then(|| {
            let policy = HeadGroupPolicy::new(5.0, 2, 4)
                .expect("fixed head-group policy is bounded")
                .with_directional_cuts(directional_cuts);
            phase_cover.map_or(policy, |cover| policy.with_phase_cover(cover))
        });
        Ok(Self {
            width: MarginRowWidthRuntimePolicy::from_levers(levers)?,
            root_precision: RootPrecisionRuntimePolicy::from_levers(levers)?,
            backward_interm: BackwardIntermRuntimePolicy::from_levers(levers)?,
            root_alpha: RootAlphaRuntimePolicy::from_levers(levers)?,
            joint_refresh: JointRefreshRuntimePolicy::from_levers(levers)?,
            root_certify_once: RootCertifyOnceRuntimePolicy::from_levers(levers)?,
            head_groups,
            classwise,
            domain_alpha: DomainAlphaRuntimePolicy::from_levers(levers)?,
            epoch: EpochRuntimePolicy::from_levers(levers)?,
            beta_per_column,
            beta_columns,
            column_retire,
            parallel_prebuild,
            wave: WavePolicy::from_levers(levers)?,
        })
    }

    fn from_process_environment() -> Self {
        let levers = ny_levers::LeverSet::freeze(ny_levers::all());
        // Legacy compatibility refuses only an invalid optional cover; scored
        // routes above reject that configuration before any lane can run.
        let mut policy = Self::from_levers_inner(&levers, false)
            .expect("the complete workspace LeverSet must contain the margin-row policy");
        policy.width = MarginRowWidthRuntimePolicy::from_layered_process();
        policy.root_precision = RootPrecisionRuntimePolicy::from_layered_process();
        policy.backward_interm = BackwardIntermRuntimePolicy::from_layered_process();
        policy.root_alpha = RootAlphaRuntimePolicy::from_layered_process();
        policy.joint_refresh = JointRefreshRuntimePolicy::from_layered_process();
        policy.root_certify_once = RootCertifyOnceRuntimePolicy::from_layered_process();
        policy.head_groups = crate::preset_gates::margin_row_head_groups_enabled().then(|| {
            let base = HeadGroupPolicy::new(5.0, 2, 4)
                .expect("fixed head-group policy is bounded")
                .with_directional_cuts(
                    crate::preset_gates::margin_row_head_directional_cuts_enabled(),
                );
            PhaseCoverPolicy::new(crate::preset_gates::margin_row_head_phase_cover())
                .map_or(base, |cover| base.with_phase_cover(cover))
        });
        policy.domain_alpha = DomainAlphaRuntimePolicy::from_layered_process();
        policy.epoch = EpochRuntimePolicy::from_layered_process();
        // #wave-beta-bab: compatibility callers only; the scored lane takes
        // `from_levers`'s value above.
        policy.wave = WavePolicy::from_gates();
        policy
    }
}

/// Production gate for the vnncomp wiring.
///
/// Authority granted 2026-07-18 after BOTH quarantine proof obligations were
/// discharged (adversarial + formal verification, ~850k zero-violation enclosure
/// checks; commits 5628ed6f/55455cbb/b7342fdf/8841eaa5):
///   1. margin-coefficient directed-rounding proof — the backward-bias running-
///      accumulator under-count (a REAL false-UNSAT, 351-witness oracle) is fixed
///      (2u*|b_running| per step), and the root-tableau outward headroom is now
///      depth-scaled (gamma_n(n_in+16+8*trunk_relus)) so soundness is depth-
///      independent by construction. Both guarded by discriminating oracles.
///   2. strict ONNX/vnnlib adapter validation — the input box AND the output
///      robustness property are cross-checked against the trusted load_vnnlib spec
///      (set-equality of the adversarial set), fail-closed on any divergence.
///
/// Kept a trusted code constant (NOT an env var) so no untrusted process can flip
/// the authority switch. The lane is strictly-additive (only turns timeout/unknown
/// into a certified `unsat`, never `sat`), so enabling it cannot regress a verdict
/// except via the budget reserve, which only applies to the twin-wall net family.
///
/// Measured behind this gate (#epoch-bab, docs/EPOCH_BAB_STATE_2026-07-18.md):
/// +5 `metaroom_2023` UNSAT instances the internal verifier times out on, each
/// closed at the ROOT in ~10s, every one confirmed against the official 2025
/// results (>=6 tools unsat, 0 sat). The shipped path defaults to a 45 s
/// reserve. Concurrent execution is a separate performance option that requires
/// `NY_MARGIN_ROW_CONCURRENT=1` and a GPU-device preset; it is off by default.
pub fn margin_row_bab_enabled() -> bool {
    true
}

/// Compatibility preset cell for the SOUND f32 root tableau
/// (`margin_row.root_f32`).
///
/// Set once by the CLI before any lane runs. Scored VNN-COMP roots consume the
/// value projected into their frozen LeverSet; compatibility entry points read
/// this cell once while constructing [`MarginRowRuntimePolicy`]. Root/rebake
/// builders never read it directly.
///
/// SOUNDNESS: arming it can only LOOSEN the root tableau — the f32 rounding is
/// charged into a certified additive concretize slack that is subtracted from
/// every lower endpoint and added to every upper one. A wrong value here costs
/// proofs; it cannot manufacture one. The lane itself remains fail-closed
/// (`Unsat` or `Unknown` only).
static ROOT_F32_PRESET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Arm/disarm the typed preset route for the f32 root tableau.
pub fn set_root_f32_preset(on: bool) {
    ROOT_F32_PRESET.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Is the typed preset route armed?
pub(crate) fn root_f32_preset() -> bool {
    ROOT_F32_PRESET.load(std::sync::atomic::Ordering::Relaxed)
}

/// #margin-row-branch-width: compatibility preset cells for the candidate budget.
///
/// `0` = preset silent, use the built-in default. Scored VNN-COMP execution
/// bypasses these cells and consumes its receipt-owned
/// [`MarginRowWidthRuntimePolicy`]. They remain for non-scored callers of the
/// compatibility entry that do not provide a frozen [`ny_levers::LeverSet`].
///
/// SOUNDNESS: width chooses WHICH domains to split, never how a bound is
/// computed. Every candidate is scored by the same certified batched pass, so
/// a wrong width costs proofs (measured: k=4 does 5x the search of k=8 and
/// LOSES one); it cannot manufacture one.
static K_HEAD_PRESET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static K_TRUNK_PRESET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Arm the non-scored compatibility preset route. `None` leaves the default.
pub fn set_branch_width_preset(k_head: Option<usize>, k_trunk: Option<usize>) {
    K_HEAD_PRESET.store(k_head.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    K_TRUNK_PRESET.store(k_trunk.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// One optional integer carried across the CLI -> propagate boundary.
///
/// Presence is separate from value because all three epoch controls accept an
/// explicit `0` kill switch. Reusing `0` as "preset silent" would make a typed
/// preset unable to disarm a process-level default or earlier selection.
struct EpochPresetCell {
    value: std::sync::atomic::AtomicUsize,
    present: std::sync::atomic::AtomicBool,
}

impl EpochPresetCell {
    const fn new() -> Self {
        Self {
            value: std::sync::atomic::AtomicUsize::new(0),
            present: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn store(&self, value: Option<usize>) {
        if let Some(value) = value {
            self.value
                .store(value, std::sync::atomic::Ordering::Relaxed);
            self.present
                .store(true, std::sync::atomic::Ordering::Release);
        } else {
            self.present
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    fn load(&self) -> Option<usize> {
        self.present
            .load(std::sync::atomic::Ordering::Acquire)
            .then(|| self.value.load(std::sync::atomic::Ordering::Relaxed))
    }
}

static EPOCH_DEPTH_PRESET: EpochPresetCell = EpochPresetCell::new();
static EPOCH_ATTEMPTS_PRESET: EpochPresetCell = EpochPresetCell::new();
static EPOCH_PARENT_TOPK_PRESET: EpochPresetCell = EpochPresetCell::new();

fn store_epoch_preset(
    cells: (&EpochPresetCell, &EpochPresetCell, &EpochPresetCell),
    values: (Option<usize>, Option<usize>, Option<usize>),
) {
    cells.0.store(values.0);
    cells.1.store(values.1);
    cells.2.store(values.2);
}

/// Deliver typed margin-row Tier-2 epoch controls from the category preset.
///
/// Called before any margin-row lane starts. Each `None` clears that typed
/// selection and restores env/default resolution; each `Some(0)` is preserved
/// as an explicit kill switch. Legacy environment inputs win when present.
pub fn set_epoch_preset(depth: Option<usize>, attempts: Option<usize>, parent_topk: Option<usize>) {
    store_epoch_preset(
        (
            &EPOCH_DEPTH_PRESET,
            &EPOCH_ATTEMPTS_PRESET,
            &EPOCH_PARENT_TOPK_PRESET,
        ),
        (depth, attempts, parent_topk),
    );
}

fn epoch_preset() -> (Option<usize>, Option<usize>, Option<usize>) {
    (
        EPOCH_DEPTH_PRESET.load(),
        EPOCH_ATTEMPTS_PRESET.load(),
        EPOCH_PARENT_TOPK_PRESET.load(),
    )
}

/// Preserve the historical legacy integer parser while adding a typed
/// fallback. A PRESENT environment value has precedence: admissible integers
/// win, malformed/non-Unicode values suppress the preset and select the old
/// built-in default. Only true absence falls through to the typed value.
///
/// NO LONGER A PRODUCTION READER, and deliberately kept anyway. The scored
/// path consumes [`EpochRuntimePolicy::from_levers`] from the flight receipt;
/// [`epoch_over_preset`] remains only in the compatibility constructor. This
/// function stays as the EXECUTABLE SPEC of the parser that migration had to
/// preserve: `declared_epoch_resolution_matches_the_legacy_parser` runs both
/// over the same case table and demands they agree. Deleting it would leave
/// the claim "semantics unchanged" resting on review alone.
#[cfg(test)]
fn legacy_epoch_over_preset(
    raw: Result<String, std::env::VarError>,
    preset: Option<usize>,
    default: usize,
) -> usize {
    match raw {
        Ok(raw) => raw.parse().unwrap_or(default),
        Err(std::env::VarError::NotPresent) => preset.unwrap_or(default),
        Err(std::env::VarError::NotUnicode(_)) => default,
    }
}

/// Compatibility-only resolution of one `#epoch-bab` control through its
/// declaration, layered over the typed preset.
///
/// The declared form of `legacy_epoch_over_preset`, which is kept above as a
/// test-only spec. `read_over_config`
/// implements the identical three-way rule — an admissible environment integer
/// wins over the preset in BOTH directions, a PRESENT malformed or non-UTF-8
/// value suppresses the preset and lands on the declaration default, and only
/// true absence falls through to the preset and then to the default.
///
/// `default` is passed rather than read out of the declaration so the call site
/// still shows the historical number it is required to preserve;
/// `epoch_declarations_carry_the_historical_defaults` pins the two against each
/// other so they cannot drift apart.
fn epoch_over_preset(
    decl: &'static ny_levers::LeverDecl,
    preset: Option<usize>,
    default: usize,
) -> usize {
    let config = preset
        .and_then(|value| u64::try_from(value).ok())
        .map(ny_levers::LeverValue::U64);
    ny_levers::read_over_config(decl, config)
        .ok()
        .and_then(|resolved| resolved.value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn epoch_parent_generation_budget(topk: usize, depth: usize, attempts: usize) -> usize {
    if topk > 0 && depth > 0 && attempts > 0 {
        bab::EPOCH_PARENT_GENERATION_CAP
    } else {
        0
    }
}

/// Whether the root build retains Tier-0 rows (`#epoch-bab`): Tier-0 itself,
/// or the historical full epoch (depth without the incremental parent arm).
/// Public so preset validation decides `#root-box-source`'s `retain`
/// conflict with the lane's own rule.
#[must_use]
pub fn epoch_retention_requested(
    tier0_exact: usize,
    epoch_depth: usize,
    epoch_parent_topk: usize,
) -> bool {
    tier0_exact > 0 || (epoch_depth > 0 && epoch_parent_topk == 0)
}

#[cfg(test)]
mod epoch_preset_delivery_tests {
    use super::{
        legacy_epoch_over_preset, store_epoch_preset, EpochPresetCell, EpochRuntimePolicy,
    };

    #[test]
    fn bridge_preserves_zero_silence_and_replacement_without_globals() {
        let depth = EpochPresetCell::new();
        let attempts = EpochPresetCell::new();
        let topk = EpochPresetCell::new();
        let cells = (&depth, &attempts, &topk);

        store_epoch_preset(cells, (Some(0), Some(5), Some(16)));
        assert_eq!(
            (depth.load(), attempts.load(), topk.load()),
            (Some(0), Some(5), Some(16))
        );

        store_epoch_preset(cells, (None, Some(0), None));
        assert_eq!(
            (depth.load(), attempts.load(), topk.load()),
            (None, Some(0), None)
        );
    }

    #[test]
    fn legacy_epoch_env_is_explicitly_above_typed_preset() {
        assert_eq!(
            legacy_epoch_over_preset(Err(std::env::VarError::NotPresent), Some(3), 0),
            3,
            "true absence falls through to the typed preset"
        );
        assert_eq!(legacy_epoch_over_preset(Ok("7".into()), Some(3), 0), 7);
        assert_eq!(
            legacy_epoch_over_preset(Ok("malformed".into()), Some(3), 0),
            0,
            "a present malformed legacy value suppresses the preset and keeps the old default"
        );
        assert_eq!(
            legacy_epoch_over_preset(Err(std::env::VarError::NotPresent), None, 2),
            2,
            "silence preserves the historical attempts default"
        );
    }

    /// The call sites still name the historical defaults; the declarations must
    /// agree, or the receipt would report a number the reader never used.
    #[test]
    fn epoch_declarations_carry_the_historical_defaults() {
        use ny_levers::{decls::margin_row, DefaultSpec};

        for (decl, default) in [
            (&margin_row::EPOCH_DEPTH, 0_u64),
            (&margin_row::EPOCH_ATTEMPTS, 2),
            (&margin_row::EPOCH_TIER0, 0),
            (&margin_row::EPOCH_UNIVERSE, 64),
            (&margin_row::EPOCH_RETAIN, 128),
        ] {
            assert_eq!(
                decl.default,
                DefaultSpec::U64(default),
                "{} must keep its historical default",
                decl.name
            );
        }
    }

    /// R9 in test form: demand the migrated reader's own observable.
    ///
    /// `epoch_over_preset` replaced `legacy_epoch_over_preset` at the live call
    /// site, and "the parser is unchanged" is a claim that has to be checked
    /// rather than asserted. Both are run over the same case table through
    /// `read_over_config_with`, the declaration's real layering path, and must
    /// agree on every case — including the two that historically diverge from a
    /// naive reading: a PRESENT malformed value suppresses the preset (it is a
    /// kill switch, not a promotion), and an explicit `"0"` is an admissible
    /// disarm rather than a rejection.
    #[test]
    fn declared_epoch_resolution_matches_the_legacy_parser() {
        use ny_levers::{decls::margin_row, read_over_config_with, LeverValue};

        let declared = |decl: &'static ny_levers::LeverDecl,
                        raw: Option<&str>,
                        preset: Option<usize>,
                        default: usize| {
            let owned = raw.map(str::to_owned);
            let config = preset
                .and_then(|value| u64::try_from(value).ok())
                .map(LeverValue::U64);
            read_over_config_with(decl, move |_| owned, config)
                .ok()
                .and_then(|resolved| resolved.value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(default)
        };

        for (decl, default) in [
            (&margin_row::EPOCH_DEPTH, 0_usize),
            (&margin_row::EPOCH_ATTEMPTS, 2),
            (&margin_row::EPOCH_TIER0, 0),
            (&margin_row::EPOCH_UNIVERSE, 64),
            (&margin_row::EPOCH_RETAIN, 128),
        ] {
            for raw in [
                None,
                Some("0"),
                Some("1"),
                Some("7"),
                Some("malformed"),
                Some(""),
                Some(" 4"),
                Some("-1"),
            ] {
                for preset in [None, Some(0), Some(3), Some(16)] {
                    let legacy = legacy_epoch_over_preset(
                        raw.map_or(Err(std::env::VarError::NotPresent), |v| Ok(v.to_owned())),
                        preset,
                        default,
                    );
                    assert_eq!(
                        declared(decl, raw, preset, default),
                        legacy,
                        "{} disagrees on raw={raw:?} preset={preset:?}",
                        decl.name
                    );
                }
            }
        }
    }

    #[test]
    fn frozen_epoch_policy_preserves_defaults_and_parser_shapes() {
        let defaults = ny_levers::LeverSet::freeze_with(ny_levers::all(), |_| None);
        assert_eq!(
            EpochRuntimePolicy::from_levers(&defaults).unwrap(),
            EpochRuntimePolicy::default()
        );

        let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| match name {
            // The five historical epoch integers are deliberately untrimmed.
            "NY_EPOCH_DEPTH" => Some(" 3".to_owned()),
            "NY_EPOCH_ATTEMPTS" => Some(" 5".to_owned()),
            "NY_EPOCH_TIER0" => Some(" 8".to_owned()),
            "NY_EPOCH_UNIVERSE" => Some(" 128".to_owned()),
            "NY_EPOCH_RETAIN" => Some(" 256".to_owned()),
            // The newer incremental width has always used the trimmed parser.
            "NY_MARGIN_ROW_EPOCH_PARENT_TOPK" => Some(" 16 ".to_owned()),
            // This gate is presence-armed, not truth-value parsed.
            "NY_EPOCH_DEBUG" => Some("0".to_owned()),
            _ => None,
        });
        assert_eq!(
            EpochRuntimePolicy::from_levers(&levers).unwrap(),
            EpochRuntimePolicy {
                parent_topk: 16,
                debug_trace: true,
                ..EpochRuntimePolicy::default()
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn frozen_epoch_debug_keeps_non_utf8_presence_dark() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let raw = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |name| {
            (name == "NY_EPOCH_DEBUG").then(|| OsString::from_vec(vec![0x80]))
        });
        let levers = ny_levers::LeverSet::resolve(ny_levers::all(), &raw);
        let resolved = levers
            .resolved(&ny_levers::decls::margin_row::EPOCH_DEBUG)
            .unwrap();
        assert_eq!(resolved.env_utf8, Some(false));
        assert!(matches!(resolved.value, ny_levers::LeverValue::Text(_)));
        assert!(
            !EpochRuntimePolicy::from_levers(&levers)
                .unwrap()
                .debug_trace
        );
    }

    #[test]
    fn parent_refresh_engagement_requires_width_cadence_and_attempt_budget() {
        use super::epoch_parent_generation_budget;

        assert_eq!(epoch_parent_generation_budget(16, 3, 2), 4);
        assert_eq!(epoch_parent_generation_budget(0, 3, 2), 0);
        assert_eq!(epoch_parent_generation_budget(16, 0, 2), 0);
        assert_eq!(epoch_parent_generation_budget(16, 3, 0), 0);
    }

    #[test]
    fn retention_engagement_preserves_the_tier0_or_full_epoch_disjunction() {
        use super::epoch_retention_requested;

        assert!(epoch_retention_requested(4, 0, 0));
        assert!(epoch_retention_requested(4, 3, 16));
        assert!(epoch_retention_requested(0, 3, 0));
        assert!(!epoch_retention_requested(0, 3, 16));
        assert!(!epoch_retention_requested(0, 0, 0));
    }
}

/// #margin-row-adaptive-width: compatibility preset cell for the
/// frontier-driven width ratchet. Scored VNN-COMP execution instead freezes
/// this choice beside both widths in [`MarginRowWidthRuntimePolicy`].
static K_ADAPTIVE_PRESET: ny_levers::gate::BoolCell = ny_levers::gate::BoolCell::new();

/// Arm the non-scored adaptive-width compatibility route.
pub fn set_k_adaptive_preset(on: bool) {
    K_ADAPTIVE_PRESET.set(on);
}

pub(crate) fn k_adaptive_preset() -> Option<bool> {
    K_ADAPTIVE_PRESET.get()
}

/// #closure-adaptive: typed preset route for the closure-driven escalation
/// policy (`margin_row.closure_adaptive`). Same bridge shape as
/// [`set_k_adaptive_preset`]; `NY_MARGIN_ROW_CLOSURE_ADAPTIVE` wins in both
/// directions where it is present.
///
/// THE BRIDGE IS THE ONE LINE THAT MAKES THIS SHIPPABLE. Everything below the
/// CLI is built and tested; until a caller in `commands::vnncomp` does
///
/// ```text
///     if let Some(on) = preset.margin_row.closure_adaptive {
///         ny_propagate::margin_row::set_closure_adaptive_preset(on);
///     }
/// ```
///
/// the yaml key parses and does nothing, because `run_instance.sh` exports
/// no `NY_*` and the env override cannot fire in competition.
pub fn set_closure_adaptive_preset(on: bool) {
    closure::set_closure_adaptive_preset(on);
}

/// #branch-reduceop: how a candidate's TWO child bounds reduce to one ranking
/// score. `false` (default) = `min`, the FSB balanced score and ny's historical
/// behaviour; `true` = `max`, the original BaBSR s_SR score, which is what
/// alpha-beta-CROWN's TUNED cifar100 and tinyimagenet configs select
/// (`branching: {reduceop: max}`) in deliberate override of its own global
/// `min` default — on exactly the two benchmarks ny fails.
///
/// RANKER-ONLY and therefore sound either way: it chooses which candidate to
/// split on, while every child bound still comes from the same certified pass
/// and the closure test is untouched. A bad choice costs tree shape, never a
/// proof. Default `false` keeps the lane byte-identical.
static BRANCH_REDUCE_MAX: ny_levers::gate::BoolCell = ny_levers::gate::BoolCell::new();

/// Arm the typed `reduceop: max` branching route (#branch-reduceop).
pub fn set_branch_reduce_max_preset(on: bool) {
    BRANCH_REDUCE_MAX.set(on);
}

fn resolve_branch_reduce_max(preset: Option<bool>) -> bool {
    resolve_branch_reduce_max_from_layered(ny_levers::read_over_config(
        &ny_levers::decls::margin_row::MARGIN_ROW_BRANCH_REDUCE_MAX,
        preset.map(ny_levers::LeverValue::Bool),
    ))
}

fn resolve_branch_reduce_max_from_layered(
    layered: Result<ny_levers::Resolved, ny_levers::ResolveError>,
) -> bool {
    layered
        .map(|resolved| resolved.value.as_bool())
        .unwrap_or(false)
}

/// Read the branch reduction. Goes through the declared lever rather than a
/// raw `env::var` — the ratchet caught exactly that on the first draft of this
/// function, which is what the ratchet is for. Env wins in BOTH directions
/// where present (exact `1` arms, exact `0` kills), matching every other
/// margin-row bridge; absent, the typed preset decides.
pub(crate) fn branch_reduce_max() -> bool {
    static RESOLVED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RESOLVED.get_or_init(|| resolve_branch_reduce_max(BRANCH_REDUCE_MAX.get()))
}

/// #branch-parent-lift: arm the recovery of discarded candidate evidence.
///
/// The expansion scores every shortlisted candidate with a certified pass and
/// then keeps only the pair it splits on. Each discarded pair still lower-bounds
/// the PARENT (see `bab::parent_lift`), so the best of them can be folded into
/// the child floor for free. Default `false` keeps the lane byte-identical.
static PARENT_LIFT: ny_levers::gate::BoolCell = ny_levers::gate::BoolCell::new();

/// Arm the typed parent-lift route (#branch-parent-lift).
pub fn set_parent_lift_preset(on: bool) {
    PARENT_LIFT.set(on);
}

fn resolve_parent_lift(preset: Option<bool>) -> bool {
    resolve_parent_lift_from_layered(ny_levers::read_over_config(
        &ny_levers::decls::margin_row::MARGIN_ROW_PARENT_LIFT,
        preset.map(ny_levers::LeverValue::Bool),
    ))
}

fn resolve_parent_lift_from_layered(
    layered: Result<ny_levers::Resolved, ny_levers::ResolveError>,
) -> bool {
    layered
        .map(|resolved| resolved.value.as_bool())
        .unwrap_or(false)
}

/// Read the parent-lift gate through the declared lever, exactly like
/// [`branch_reduce_max`]: env wins in both directions where present, otherwise
/// the typed preset decides.
pub(crate) fn parent_lift_enabled() -> bool {
    static RESOLVED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RESOLVED.get_or_init(|| resolve_parent_lift(PARENT_LIFT.get()))
}

/// #margin-row-domain-alpha: compatibility preset route for PER-DOMAIN alpha
/// and its two ascent knobs.
///
/// Scored runs do not use these process-wide cells: generic preset projection
/// and environment layering are frozen in the receipt-owning [`ny_levers::LeverSet`],
/// then [`MarginRowRuntimePolicy::from_levers`] supplies the plain values to
/// every domain. This bridge remains for research and compatibility callers
/// that enter without that frozen set. `None` leaves a cell empty, which
/// resolves exactly as the declaration default does; the environment still
/// wins over a set cell in both directions.
///
/// Compatibility callers must invoke this before any margin-row lane starts:
/// the readers latch in `OnceLock`s and a later write cannot be seen.
pub fn set_domain_alpha_preset(p: DomainAlphaPreset) {
    domain_alpha::set_preset(p);
}

/// #backward-interm: typed preset route (see `backward_interm.rs`). Same
/// bridge shape as [`set_root_f32_preset`]; env wins in both directions.
static BACKWARD_INTERM_PRESET: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Arm the typed backward-interm preset route.
pub fn set_backward_interm_preset(on: bool) {
    BACKWARD_INTERM_PRESET.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn backward_interm_preset() -> bool {
    BACKWARD_INTERM_PRESET.load(std::sync::atomic::Ordering::Relaxed)
}

/// #clip-stateful: typed preset route (`margin_row.clip_stateful`).
static CLIP_STATEFUL_PRESET: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Arm the typed STATEFUL (accumulate + inherit) Clip-and-Verify route.
pub fn set_clip_stateful_preset(on: bool) {
    CLIP_STATEFUL_PRESET.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn clip_stateful_preset_armed() -> bool {
    CLIP_STATEFUL_PRESET.load(std::sync::atomic::Ordering::Relaxed)
}

/// #backward-interm budget: the typed preset route for the phase's seconds cap.
///
/// MEASURED 2026-08-22 (64 banked-timeout resnet_large rows, 3/arm on the
/// decisive ones): at the shipped 20 s default the phase is TRUNCATED
/// (`exhausted=true`) on 10/10 resnet_LARGE rows, which need 24-35 s, and on
/// 0/9 resnet_MEDIUM rows, which finish in 6.6-8.4 s. So raising the cap is
/// free on medium (byte-identical no-op, verified) and strictly better on
/// large (tighter root => SMALLER tree: idx_5410 39 -> 28 expansions, idx_496
/// 2 -> 1, with no wall cost since those rows finish ~28 s inside budget).
/// Converts idx_9883 (unknown 3/3 -> unsat 3/3) and de-randomises idx_5308,
/// whose 20 s arm lands nondeterministically on two different roots. Moat 7/7
/// preserved. NOTE the effective value is `min(secs, 0.4 * remaining)`, so 40
/// resolves to ~37.2 s at the scored 100 s budget.
static BI_SECS_PRESET: ny_levers::gate::F64Cell = ny_levers::gate::F64Cell::new();

fn store_bi_secs_preset(cell: &ny_levers::gate::F64Cell, secs: u64) {
    cell.set(secs as f64);
}

/// Arm the typed backward-interm seconds cap. Explicit `0` is retained as a
/// real zero-second budget; only an unwritten cell means absent.
pub fn set_bi_secs_preset(secs: u64) {
    store_bi_secs_preset(&BI_SECS_PRESET, secs);
}

pub(crate) fn bi_secs_preset() -> Option<f64> {
    BI_SECS_PRESET.get()
}

#[cfg(test)]
mod layered_preset_delivery_tests {
    use super::{
        resolve_branch_reduce_max_from_layered, store_bi_secs_preset, BackwardIntermRuntimePolicy,
        DomainAlphaRuntimePolicy, EpochRuntimePolicy, ErrLane, JointRefreshRuntimePolicy,
        MarginRowRuntimePolicy, MarginRowWidthRuntimePolicy, RootAlphaRuntimePolicy, RootBoxSource,
        RootCertifyOnceRuntimePolicy, RootPrecisionRuntimePolicy, RootSearch, TreeMode, WavePolicy,
    };

    #[test]
    fn head_groups_freeze_only_the_receipt_owned_choice() {
        let key = ny_levers::decls::margin_row::MARGIN_ROW_HEAD_GROUPS.name;
        let freeze = |raw: Option<&str>| {
            let values = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
                (name == key).then(|| raw.map(str::to_owned)).flatten()
            });
            MarginRowRuntimePolicy::from_levers(&values).unwrap()
        };
        let armed = freeze(None);
        assert_eq!(armed, MarginRowRuntimePolicy::default());
        assert_eq!(freeze(Some("1")), armed);
        // Governed Bool rejects these tokens and uses its true default.
        assert_eq!(freeze(Some("true")), armed);
        let dark = freeze(Some("0"));
        assert_eq!(dark.head_groups, None);
        let policy = armed
            .head_groups
            .expect("frozen true reaches the native lane");
        assert_eq!(policy.max_seconds(), 5.0);
        assert_eq!(policy.groups_per_margin(), 2);
        assert_eq!(policy.proposal_rounds(), 4);
        assert!(policy.directional_cuts());
        assert_eq!(policy.phase_cover(), None);
        assert_eq!(
            MarginRowRuntimePolicy {
                head_groups: None,
                ..armed
            },
            dark
        );
    }

    #[test]
    fn head_directional_cuts_freeze_independently_inside_the_parent_route() {
        use ny_levers::decls::margin_row as d;
        let freeze = |groups: bool, cuts: Option<bool>, raw: Option<&str>| {
            let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |name| {
                (name == d::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS.name)
                    .then(|| raw.map(std::ffi::OsString::from))
                    .flatten()
            });
            let levers =
                ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
                    if decl.name == d::MARGIN_ROW_HEAD_GROUPS.name {
                        Some(ny_levers::LeverValue::Bool(groups))
                    } else if decl.name == d::MARGIN_ROW_HEAD_DIRECTIONAL_CUTS.name {
                        cuts.map(ny_levers::LeverValue::Bool)
                    } else {
                        None
                    }
                })
                .unwrap();
            MarginRowRuntimePolicy::from_levers(&levers).unwrap()
        };
        let enabled = freeze(true, None, None);
        assert_eq!(enabled, MarginRowRuntimePolicy::default());
        let disabled = freeze(true, Some(false), None);
        let policy = disabled
            .head_groups
            .expect("only the new stage is disabled");
        assert!(!policy.directional_cuts());
        assert_eq!(
            policy.with_directional_cuts(true),
            enabled.head_groups.unwrap()
        );
        assert_eq!(policy.phase_cover(), None);
        assert_eq!(freeze(true, Some(true), Some("0")), disabled);
        assert_eq!(freeze(true, Some(false), Some("1")), enabled);
        assert_eq!(freeze(true, Some(false), Some("malformed")), enabled);
        for cuts in [false, true] {
            assert_eq!(freeze(false, Some(cuts), None).head_groups, None);
        }
    }

    #[test]
    fn head_phase_cover_freezes_only_bounded_receipt_owned_requests() {
        use ny_levers::decls::margin_row as d;
        let freeze = |groups: bool, count: Option<u64>, raw: Option<&str>| {
            let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |name| {
                (name == d::MARGIN_ROW_HEAD_PHASE_COVER.name)
                    .then(|| raw.map(std::ffi::OsString::from))
                    .flatten()
            });
            let levers =
                ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
                    if decl.name == d::MARGIN_ROW_HEAD_GROUPS.name {
                        Some(ny_levers::LeverValue::Bool(groups))
                    } else if decl.name == d::MARGIN_ROW_HEAD_PHASE_COVER.name {
                        count.map(ny_levers::LeverValue::U64)
                    } else {
                        None
                    }
                })
                .unwrap();
            MarginRowRuntimePolicy::from_levers(&levers)
        };
        assert_eq!(
            freeze(false, None, None).unwrap(),
            MarginRowRuntimePolicy {
                head_groups: None,
                ..Default::default()
            }
        );
        let baseline = freeze(true, None, None).unwrap();
        assert_eq!(baseline.head_groups.unwrap().phase_cover(), None);
        assert_eq!(freeze(true, Some(0), None).unwrap(), baseline);
        for count in 1..=7 {
            let policy = freeze(true, Some(count), None).unwrap();
            assert_eq!(
                policy
                    .head_groups
                    .unwrap()
                    .phase_cover()
                    .unwrap()
                    .max_heads(),
                count as usize
            );
            assert_eq!(policy.head_groups.unwrap().max_seconds(), 5.0);
            assert_eq!(freeze(false, Some(count), None).unwrap().head_groups, None);
        }
        assert_eq!(freeze(true, Some(7), Some("0")).unwrap(), baseline);
        assert_eq!(freeze(true, Some(7), Some("malformed")).unwrap(), baseline);
        assert_eq!(
            freeze(true, Some(1), Some("7"))
                .unwrap()
                .head_groups
                .unwrap()
                .phase_cover()
                .unwrap()
                .max_heads(),
            7
        );
        for count in [8, 128, u64::MAX] {
            assert!(freeze(true, Some(count), None).is_err());
        }
        assert!(freeze(true, Some(1), Some("8")).is_err());
    }

    #[test]
    fn compatibility_refuses_unbounded_phase_cover_without_changing_the_base_policy() {
        use ny_levers::decls::margin_row as d;
        let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            if name == d::MARGIN_ROW_HEAD_GROUPS.name {
                Some("1".to_owned())
            } else if name == d::MARGIN_ROW_HEAD_PHASE_COVER.name {
                Some("8".to_owned())
            } else {
                None
            }
        });
        assert!(MarginRowRuntimePolicy::from_levers(&levers).is_err());
        let policy = MarginRowRuntimePolicy::from_levers_inner(&levers, false).unwrap();
        let group = policy.head_groups.unwrap();
        assert_eq!(group.phase_cover(), None);
        assert_eq!(group.max_seconds(), 5.0);
        assert_eq!(group.groups_per_margin(), 2);
        assert_eq!(group.proposal_rounds(), 4);
    }

    #[test]
    fn branch_reduce_env_wins_both_directions_and_malformed_suppresses_preset() {
        let resolve = |raw: Option<&str>, preset: Option<bool>| {
            let raw = raw.map(str::to_owned);
            resolve_branch_reduce_max_from_layered(ny_levers::read_over_config_with(
                &ny_levers::decls::margin_row::MARGIN_ROW_BRANCH_REDUCE_MAX,
                move |_| raw,
                preset.map(ny_levers::LeverValue::Bool),
            ))
        };
        assert!(resolve(None, Some(true)));
        assert!(!resolve(None, Some(false)));
        assert!(!resolve(Some("0"), Some(true)));
        assert!(resolve(Some("1"), Some(false)));
        assert!(
            !resolve(Some("true"), Some(true)),
            "malformed env must land on declaration default false"
        );
    }

    #[test]
    fn bi_secs_cell_distinguishes_absence_from_explicit_zero() {
        let cell = ny_levers::gate::F64Cell::new();
        assert_eq!(cell.get(), None);
        store_bi_secs_preset(&cell, 0);
        assert_eq!(cell.get(), Some(0.0));
        store_bi_secs_preset(&cell, 40);
        assert_eq!(cell.get(), Some(40.0));
    }

    #[test]
    fn margin_row_runtime_policy_is_frozen_and_typed() {
        let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| match name {
            "NY_MARGIN_ROW_ROOT_F32" => Some(" on ".to_owned()),
            "NY_MARGIN_ROW_TABLEAU_F32" => Some("1".to_owned()),
            "NY_MARGIN_ROW_BACKWARD_INTERM" => Some("1".to_owned()),
            "NY_MARGIN_ROW_BI_SECS" => Some(" 40 ".to_owned()),
            "NY_MARGIN_ROW_BI_CHUNK" => Some("0".to_owned()),
            "NY_MARGIN_ROW_BI_CHUNK_SECS" => Some("2.5".to_owned()),
            "NY_MARGIN_ROW_BI_TOPK" => Some("0".to_owned()),
            "NY_MARGIN_ROW_BETA_PERCOL" => Some("1".to_owned()),
            "NY_MARGIN_ROW_BETA_COLS" => Some(" 64 ".to_owned()),
            "NY_MARGIN_ROW_COL_RETIRE" => Some("1".to_owned()),
            "NY_MARGIN_ROW_PARALLEL_PREBUILD" => Some("0".to_owned()),
            "NY_MARGIN_ROW_ALPHA_OPT" => Some("1".to_owned()),
            "NY_MARGIN_ROW_ALPHA_ITERS" => Some(" 96 ".to_owned()),
            "NY_MARGIN_ROW_ALPHA_SECS" => Some(" 12.5 ".to_owned()),
            "NY_MARGIN_ROW_ALPHA_LR" => Some(" 2 ".to_owned()),
            "NY_MARGIN_ROW_ALPHA_COLS" => Some(" 4 ".to_owned()),
            "NY_MARGIN_ROW_ALPHA_GUARD" => Some("0".to_owned()),
            "NY_MARGIN_ROW_REBAKE" => Some("1".to_owned()),
            "NY_MARGIN_ROW_REBAKE_ROUNDS" => Some(" 0 ".to_owned()),
            "NY_MARGIN_ROW_REBAKE_SECS" => Some(" 45 ".to_owned()),
            "NY_MARGIN_ROW_DOMAIN_ALPHA" => Some("1".to_owned()),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS" => Some(" 12 ".to_owned()),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA" => Some(" 2 ".to_owned()),
            "NY_EPOCH_DEPTH" => Some("3".to_owned()),
            "NY_EPOCH_ATTEMPTS" => Some("5".to_owned()),
            "NY_EPOCH_TIER0" => Some("8".to_owned()),
            "NY_EPOCH_UNIVERSE" => Some("128".to_owned()),
            "NY_EPOCH_RETAIN" => Some("256".to_owned()),
            "NY_MARGIN_ROW_EPOCH_PARENT_TOPK" => Some(" 16 ".to_owned()),
            "NY_EPOCH_DEBUG" => Some(String::new()),
            // #spec-alpha: these five keys are FED here because they are
            // ASSERTED below. The point of
            // this test is that every field resolves FROM the frozen LeverSet,
            // which only a NON-DEFAULT value demonstrates (the declaration
            // defaults are false / 3 / 3 / 1 / 65536).
            "NY_MARGIN_ROW_SPEC_ALPHA" => Some("1".to_owned()),
            "NY_MARGIN_ROW_SPEC_ALPHA_WIDTH" => Some(" 5 ".to_owned()),
            "NY_MARGIN_ROW_SPEC_ALPHA_CANDS" => Some("2".to_owned()),
            "NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS" => Some(" 3 ".to_owned()),
            "NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES" => Some("1024".to_owned()),
            _ => None,
        });
        assert_eq!(
            MarginRowRuntimePolicy::from_levers(&levers).unwrap(),
            MarginRowRuntimePolicy {
                width: MarginRowWidthRuntimePolicy::default(),
                root_precision: RootPrecisionRuntimePolicy {
                    root_f32: true,
                    tableau_f32: true,
                    // #mr-errlane: not set by this fixture -> declaration `off`.
                    err_lane: ErrLane::PerCol,
                    // #mr-ibp-seed: not set by this fixture -> declaration false.
                    ibp_seed: false,
                    // #root-box-source: not set by this fixture -> `tableau`.
                    box_source: RootBoxSource::Tableau,
                },
                backward_interm: BackwardIntermRuntimePolicy {
                    enabled: true,
                    secs: 40.0,
                    chunk: 1,
                    chunk_secs: 2.5,
                    topk: 1,
                    interm_alpha: 0,
                    // #bi-window: not set by this fixture -> declaration false.
                    window: false,
                    // #bi-gpu: not set by this fixture -> declaration false.
                    gpu: false,
                    // #bi-patch: not set by this fixture -> declaration defaults.
                    patch: super::BiPatchRuntimePolicy::default(),
                },
                root_alpha: RootAlphaRuntimePolicy {
                    alpha_enabled: true,
                    alpha_iters: 64,
                    alpha_secs: 12.5,
                    alpha_lr: 1.0,
                    alpha_cols: 4,
                    alpha_guard: false,
                    rebake_enabled: true,
                    rebake_rounds: 1,
                    rebake_secs: 45.0,
                    spec_alpha_enabled: true,
                    spec_alpha_width: 5,
                    spec_alpha_cands: 2,
                    spec_alpha_rounds: 3,
                    spec_alpha_entries: 1_024,
                },
                // #joint-refresh: not set by this fixture -> dark defaults.
                joint_refresh: JointRefreshRuntimePolicy::default(),
                // #root-certify-once: not set by this fixture -> dark defaults.
                root_certify_once: RootCertifyOnceRuntimePolicy::default(),
                head_groups: super::HeadGroupPolicy::new(5.0, 2, 4),
                classwise: false,
                domain_alpha: DomainAlphaRuntimePolicy {
                    enabled: true,
                    iters: 8,
                    eta: 1.0,
                    min_depth: 0,
                    stride: 1,
                    gap: f64::INFINITY,
                },
                epoch: EpochRuntimePolicy {
                    depth: 3,
                    max_attempts: 5,
                    tier0_exact: 8,
                    tier0_universe: 128,
                    retain_per_layer: 256,
                    parent_topk: 16,
                    debug_trace: true,
                },
                beta_per_column: true,
                beta_columns: 32,
                column_retire: true,
                parallel_prebuild: false,
                // #wave-beta-bab: not fed by this fixture -> `legacy`.
                wave: WavePolicy::default(),
            }
        );
    }

    /// #bi-window / #joint-refresh DARK BY DEFAULT: with no key written the
    /// frozen route resolves the crop off and zero refresh rounds, so the lane
    /// is byte-identical; with the typed keys fed (names taken from the
    /// declarations, never spelled as literals) every value resolves FROM the
    /// frozen LeverSet — a non-default value is the only proof of that.
    #[test]
    fn bi_window_and_joint_refresh_are_dark_by_default_and_resolve_when_fed() {
        use ny_levers::decls::margin_row as d;

        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert!(!dark.backward_interm.window, "#bi-window must ship dark");
        assert_eq!(
            dark.joint_refresh,
            JointRefreshRuntimePolicy::default(),
            "#joint-refresh must ship dark"
        );
        assert_eq!(
            dark.joint_refresh.rounds, 0,
            "rounds 0 is the disarmed state"
        );
        assert!(!BackwardIntermRuntimePolicy::default().window);
        assert!(
            !BackwardIntermRuntimePolicy::from_config(None).window,
            "an explicit BiCfg (rebake round) never carries the crop"
        );

        let (window, rounds, secs) = (
            d::MARGIN_ROW_BI_WINDOW.name,
            d::MARGIN_ROW_JOINT_REFRESH.name,
            d::MARGIN_ROW_JOINT_REFRESH_SECS.name,
        );
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            if name == window {
                Some("1".to_owned())
            } else if name == rounds {
                Some(" 40 ".to_owned())
            } else if name == secs {
                Some(" 12.5 ".to_owned())
            } else {
                None
            }
        });
        let armed = MarginRowRuntimePolicy::from_levers(&fed).unwrap();
        assert!(armed.backward_interm.window);
        assert_eq!(
            armed.joint_refresh,
            JointRefreshRuntimePolicy {
                // Clamped at the reader to the hard ceiling.
                rounds: JointRefreshRuntimePolicy::MAX_ROUNDS,
                secs: 12.5,
            }
        );
    }

    /// #root-certify-once DARK BY DEFAULT: with no key written the frozen route
    /// resolves the phase disarmed (so the lane entry returns before touching
    /// anything) at its declaration defaults; with the typed keys fed (names
    /// taken from the declarations, never spelled as literals) every value
    /// resolves FROM the frozen LeverSet — a non-default value is the only proof
    /// of that — and the reader clamps hold.
    #[test]
    fn root_certify_once_is_dark_by_default_and_resolves_when_fed() {
        use ny_levers::decls::margin_row as d;

        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert_eq!(
            dark.root_certify_once,
            RootCertifyOnceRuntimePolicy::default(),
            "#root-certify-once must ship dark at its declaration defaults"
        );
        assert!(
            !dark.root_certify_once.enabled,
            "disarmed is the shipped state"
        );
        assert_eq!(
            (
                dark.root_certify_once.iters,
                dark.root_certify_once.refresh,
                dark.root_certify_once.secs
            ),
            (20, 0, 20.0)
        );
        assert_eq!(
            MarginRowRuntimePolicy::default().root_certify_once,
            RootCertifyOnceRuntimePolicy::default()
        );

        let (armed_name, iters_name, refresh_name, secs_name) = (
            d::MARGIN_ROW_ROOT_CERTIFY_ONCE.name,
            d::MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS.name,
            d::MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH.name,
            d::MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS.name,
        );
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            if name == armed_name {
                Some("1".to_owned())
            } else if name == iters_name {
                Some(" 500 ".to_owned())
            } else if name == refresh_name {
                Some(" 40 ".to_owned())
            } else if name == secs_name {
                Some(" 12.5 ".to_owned())
            } else {
                None
            }
        });
        assert_eq!(
            MarginRowRuntimePolicy::from_levers(&fed)
                .unwrap()
                .root_certify_once,
            RootCertifyOnceRuntimePolicy {
                enabled: true,
                // Both counts clamped at the reader to their hard ceilings.
                iters: RootCertifyOnceRuntimePolicy::MAX_ITERS,
                refresh: RootCertifyOnceRuntimePolicy::MAX_REFRESH,
                secs: 12.5,
                // Not fed: the declaration default.
                search: RootSearch::Ascent,
            }
        );
        // #bbe-joint-root: the search selector resolves FROM the frozen
        // LeverSet (name taken from the declaration, never a literal), the
        // default is `ascent`, and nothing else moves when it is fed.
        assert_eq!(dark.root_certify_once.search, RootSearch::Ascent);
        let search_name = d::MARGIN_ROW_ROOT_SEARCH.name;
        let joint = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            (name == search_name).then(|| RootSearch::JointF32.token().to_owned())
        });
        assert_eq!(
            RootCertifyOnceRuntimePolicy::from_levers(&joint).unwrap(),
            RootCertifyOnceRuntimePolicy {
                search: RootSearch::JointF32,
                ..RootCertifyOnceRuntimePolicy::default()
            }
        );
        let zero_iters = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            (name == iters_name).then(|| "0".to_owned())
        });
        assert_eq!(
            RootCertifyOnceRuntimePolicy::from_levers(&zero_iters)
                .unwrap()
                .iters,
            1,
            "an explicit zero iteration count still runs one iteration when armed"
        );
    }

    /// #root-box-source (ENGINE_PROGRAM M5b) DARK BY DEFAULT: the declaration
    /// default is `tableau`, the frozen route with no key written, the
    /// compatibility route, the policy default and every explicit tableau
    /// oracle entry resolve `Tableau` (the root build never enters
    /// `root_box.rs`); with the typed key fed (its name taken from the
    /// declaration, never spelled as a literal) `patch_backward` resolves FROM
    /// the frozen LeverSet into the ROOT PRECISION policy — the one every root
    /// walk (initial, rebake, epoch) receives — and nothing else moves.
    #[test]
    fn root_box_source_is_dark_by_default_and_resolves_when_fed() {
        use ny_levers::decls::margin_row as d;

        assert!(matches!(
            d::MARGIN_ROW_ROOT_BOX_SOURCE.default,
            ny_levers::DefaultSpec::Text("tableau")
        ));
        let ny_levers::LeverKind::Enum(members) = d::MARGIN_ROW_ROOT_BOX_SOURCE.kind else {
            panic!("margin_row.root_box_source must be declared as an Enum");
        };
        assert_eq!(members, &RootBoxSource::TOKENS[..]);
        for token in RootBoxSource::TOKENS {
            assert_eq!(
                RootBoxSource::from_token(token).map(RootBoxSource::token),
                Some(token)
            );
        }
        assert_eq!(RootBoxSource::from_token("patch"), None);
        assert_eq!(RootBoxSource::default(), RootBoxSource::Tableau);
        assert_eq!(
            RootPrecisionRuntimePolicy::default().box_source,
            RootBoxSource::Tableau
        );

        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert_eq!(dark.root_precision.box_source, RootBoxSource::Tableau);
        assert_eq!(
            RootPrecisionRuntimePolicy::from_layered_process().box_source,
            RootBoxSource::Tableau,
            "the unwritten compatibility cell resolves the declaration default"
        );

        let name = d::MARGIN_ROW_ROOT_BOX_SOURCE.name;
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |n| {
            (n == name).then(|| RootBoxSource::PatchBackward.token().to_owned())
        });
        let armed = MarginRowRuntimePolicy::from_levers(&fed).unwrap();
        assert_eq!(
            armed.root_precision,
            RootPrecisionRuntimePolicy {
                box_source: RootBoxSource::PatchBackward,
                ..dark.root_precision
            }
        );
        assert_eq!(armed.backward_interm, dark.backward_interm);
        assert_eq!(armed.root_certify_once, dark.root_certify_once);
    }

    /// #wave-beta-bab ONE AUTHORITY (ENGINE_PROGRAM M6): the scored lane's tree
    /// engine resolves FROM the frozen, receipt-owning LeverSet — the same set
    /// the flight receipt serialises — never from the bridge-written governed
    /// cells. Dark, the frozen route and the policy default are `legacy` with
    /// the declaration knobs; fed (names taken from the declarations, never
    /// spelled as literals), all three keys arrive in `MarginRowRuntimePolicy::
    /// wave`, clamped by the same reader clamps as the compatibility route, and
    /// nothing else in the policy moves.
    #[test]
    fn wave_policy_rides_the_frozen_lever_set_not_the_cells() {
        use ny_levers::decls::margin_row as d;

        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert_eq!(dark.wave, WavePolicy::default());
        assert_eq!(
            MarginRowRuntimePolicy::default().wave,
            WavePolicy::default()
        );
        assert!(!dark.wave.armed());

        let (mode, batch, threads) = (
            d::MARGIN_ROW_TREE_MODE.name,
            d::MARGIN_ROW_TREE_WAVE_BATCH.name,
            d::MARGIN_ROW_TREE_WAVE_CERT_THREADS.name,
        );
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |n| {
            if n == mode {
                Some(TreeMode::Wave.token().to_owned())
            } else if n == batch {
                Some("512".to_owned())
            } else if n == threads {
                Some("6".to_owned())
            } else {
                None
            }
        });
        let armed = MarginRowRuntimePolicy::from_levers(&fed).unwrap();
        assert_eq!(
            armed.wave,
            WavePolicy {
                mode: TreeMode::Wave,
                batch: 512,
                cert_threads: 6,
            }
        );
        assert!(armed.wave.armed());
        assert_eq!(
            MarginRowRuntimePolicy {
                wave: dark.wave,
                ..armed
            },
            dark,
            "only the wave policy may move"
        );

        // The reader clamps are the compatibility route's: an out-of-range
        // batch arrives clamped, never raw.
        let wide = ny_levers::LeverSet::freeze_with(ny_levers::all(), |n| {
            if n == mode {
                Some(TreeMode::Wave.token().to_owned())
            } else if n == batch {
                Some(usize::MAX.to_string())
            } else {
                None
            }
        });
        let clamped = MarginRowRuntimePolicy::from_levers(&wide).unwrap().wave;
        assert_eq!(
            clamped,
            WavePolicy {
                mode: TreeMode::Wave,
                batch: usize::MAX,
                cert_threads: 0,
            }
            .clamped()
        );
    }

    /// #bi-gpu DARK BY DEFAULT: with no key written the frozen route, the
    /// compatibility route, the policy default and an explicit (rebake) config
    /// all resolve the device route OFF, so the phase builds no route and is
    /// byte-identical; with the typed key fed (its name taken from the
    /// declaration, never spelled as a literal) the flag resolves FROM the
    /// frozen LeverSet and nothing else moves.
    #[test]
    fn bi_gpu_is_dark_by_default_and_resolves_when_fed() {
        use ny_levers::decls::margin_row as d;

        assert!(matches!(
            d::MARGIN_ROW_BI_GPU.default,
            ny_levers::DefaultSpec::Bool(false)
        ));
        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert!(!dark.backward_interm.gpu, "#bi-gpu must ship dark");
        assert!(!BackwardIntermRuntimePolicy::default().gpu);
        assert!(
            !BackwardIntermRuntimePolicy::from_config(None).gpu,
            "an explicit BiCfg (rebake round) never carries the device route"
        );
        assert!(
            !BackwardIntermRuntimePolicy::from_config(
                BackwardIntermRuntimePolicy {
                    enabled: true,
                    gpu: true,
                    ..BackwardIntermRuntimePolicy::default()
                }
                .config(super::RoundMode::Outward)
            )
            .gpu,
            "the device flag must not leak through BiCfg into a rebake round"
        );
        assert!(
            !crate::preset_gates::margin_row_bi_gpu_enabled(),
            "the compatibility cell must resolve dark with no preset and no env"
        );

        let name = d::MARGIN_ROW_BI_GPU.name;
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |key| {
            (key == name).then(|| "1".to_owned())
        });
        let armed = MarginRowRuntimePolicy::from_levers(&fed).unwrap();
        assert!(
            armed.backward_interm.gpu,
            "the fed key must reach the policy"
        );
        assert_eq!(
            BackwardIntermRuntimePolicy {
                gpu: false,
                ..armed.backward_interm
            },
            dark.backward_interm,
            "#bi-gpu must move nothing but its own flag"
        );
    }

    /// #bi-patch DARK BY DEFAULT: with no key written the frozen route, the
    /// compatibility route, the policy default and an explicit (rebake) config
    /// all resolve patch mode OFF (and the phase's `cfg()` is `None`, so no
    /// plan is ever built); with the three typed keys fed (names taken from
    /// the declarations, never spelled as literals) every value resolves FROM
    /// the frozen LeverSet — a non-default value is the only proof of that —
    /// the ratio clamp and the `mb >= 1` floor hold, and nothing else moves.
    #[test]
    fn bi_patch_is_dark_by_default_and_resolves_when_fed() {
        use ny_levers::decls::margin_row as d;

        assert!(matches!(
            d::MARGIN_ROW_BI_PATCH.default,
            ny_levers::DefaultSpec::Bool(false)
        ));
        let dark = MarginRowRuntimePolicy::from_levers(&ny_levers::LeverSet::freeze_with(
            ny_levers::all(),
            |_| None,
        ))
        .unwrap();
        assert_eq!(
            dark.backward_interm.patch,
            super::BiPatchRuntimePolicy::default(),
            "#bi-patch must ship dark at its declaration defaults"
        );
        assert!(!dark.backward_interm.patch.enabled);
        assert!(dark.backward_interm.patch.cfg().is_none());
        assert_eq!(
            (
                dark.backward_interm.patch.ratio,
                dark.backward_interm.patch.mb
            ),
            (super::patch::DEFAULT_RATIO, super::patch::DEFAULT_MB)
        );
        assert!(!BackwardIntermRuntimePolicy::default().patch.enabled);
        assert!(
            !BackwardIntermRuntimePolicy::from_config(None).patch.enabled,
            "an explicit BiCfg (rebake round) never carries patch mode"
        );
        assert!(
            !BackwardIntermRuntimePolicy::from_config(
                BackwardIntermRuntimePolicy {
                    enabled: true,
                    patch: super::BiPatchRuntimePolicy {
                        enabled: true,
                        ..super::BiPatchRuntimePolicy::default()
                    },
                    ..BackwardIntermRuntimePolicy::default()
                }
                .config(super::RoundMode::Outward)
            )
            .patch
            .enabled,
            "patch mode must not leak through BiCfg into a rebake round"
        );
        assert!(
            !crate::preset_gates::margin_row_bi_patch_enabled(),
            "the compatibility cell must resolve dark with no preset and no env"
        );

        let (armed, ratio, mb) = (
            d::MARGIN_ROW_BI_PATCH.name,
            d::MARGIN_ROW_BI_PATCH_RATIO.name,
            d::MARGIN_ROW_BI_PATCH_MB.name,
        );
        let fed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |key| {
            if key == armed {
                Some("1".to_owned())
            } else if key == ratio {
                Some(" 0.25 ".to_owned())
            } else if key == mb {
                Some(" 64 ".to_owned())
            } else {
                None
            }
        });
        let policy = MarginRowRuntimePolicy::from_levers(&fed).unwrap();
        assert_eq!(
            policy.backward_interm.patch,
            super::BiPatchRuntimePolicy {
                enabled: true,
                ratio: 0.25,
                mb: 64,
            },
            "the fed keys must reach the policy"
        );
        let cfg = policy
            .backward_interm
            .patch
            .cfg()
            .expect("an armed policy yields a phase config");
        assert_eq!((cfg.ratio, cfg.mb), (0.25, 64));
        assert_eq!(
            BackwardIntermRuntimePolicy {
                patch: super::BiPatchRuntimePolicy::default(),
                ..policy.backward_interm
            },
            dark.backward_interm,
            "#bi-patch must move nothing but its own policy"
        );
        // An out-of-range ratio is REJECTED by the typed parser (declaration
        // default), never silently clamped into the treatment.
        let rejected = ny_levers::LeverSet::freeze_with(ny_levers::all(), |key| {
            (key == ratio).then(|| "7.5".to_owned())
        });
        assert_eq!(
            MarginRowRuntimePolicy::from_levers(&rejected)
                .unwrap()
                .backward_interm
                .patch
                .ratio,
            super::patch::DEFAULT_RATIO
        );
    }

    /// #mr-errlane: the frozen route resolves every token of the typed key
    /// (and nothing else) into the root-precision policy every engine copies.
    /// The name is taken from the declaration, never spelled as a literal.
    #[test]
    fn frozen_err_envelope_resolves_into_root_precision_policy() {
        let name = ny_levers::decls::margin_row::MARGIN_ROW_ERR_ENVELOPE.name;
        for (raw, want) in [
            (None, ErrLane::PerCol),
            (Some("off"), ErrLane::PerCol),
            (Some("envelope"), ErrLane::Envelope),
            (Some("per_channel"), ErrLane::PerChannel),
            (Some("audit"), ErrLane::Audit),
            // Inadmissible tokens resolve to the declaration default.
            (Some("1"), ErrLane::PerCol),
            (Some(" envelope"), ErrLane::PerCol),
            (Some("row"), ErrLane::PerCol),
        ] {
            let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |candidate| {
                (candidate == name)
                    .then(|| raw.map(str::to_owned))
                    .flatten()
            });
            assert_eq!(
                RootPrecisionRuntimePolicy::from_levers(&levers)
                    .unwrap()
                    .err_lane,
                want,
                "{raw:?}"
            );
        }
        for token in ErrLane::TOKENS {
            let lane = ErrLane::from_token(token).expect("canonical token parses");
            assert_eq!(lane.token(), token, "token round-trip");
        }
    }

    #[test]
    fn frozen_root_alpha_seconds_keep_the_historical_positive_filter() {
        for raw in ["0", " -0 ", "-1", "NaN", "malformed"] {
            let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
                (name == "NY_MARGIN_ROW_ALPHA_SECS").then(|| raw.to_owned())
            });
            assert_eq!(
                RootAlphaRuntimePolicy::from_levers(&levers)
                    .unwrap()
                    .alpha_secs,
                20.0,
                "{raw:?} must preserve the original positive-seconds fallback"
            );
        }
    }

    #[test]
    fn frozen_domain_alpha_keeps_historical_clamps_and_zero_eta_fallback() {
        let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| match name {
            "NY_MARGIN_ROW_DOMAIN_ALPHA" => Some("1".to_owned()),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS" => Some("0".to_owned()),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA" => Some("0".to_owned()),
            _ => None,
        });
        assert_eq!(
            DomainAlphaRuntimePolicy::from_levers(&levers).unwrap(),
            DomainAlphaRuntimePolicy {
                enabled: true,
                iters: 1,
                eta: 0.25,
                min_depth: 0,
                stride: 1,
                gap: f64::INFINITY,
            }
        );
    }
}

fn k_preset(cell: &std::sync::atomic::AtomicUsize) -> Option<usize> {
    match cell.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        k => Some(k),
    }
}

/// End-to-end lane: compile, build certified root gates, run the BaB.
/// `t` = true class, `adv` = adversarial classes of the robustness
/// disjunction. Only ever returns `Unsat` or `Unknown` (fail-closed).
pub fn run_margin_row_lane(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
) -> MarginRowOutcome {
    lane_impl(spec, lo, hi, t, adv, deadline, max_expansions)
}

/// Scored entry point: run with the immutable policy derived from the same
/// frozen [`ny_levers::LeverSet`] that was serialized into the run receipt.
#[allow(clippy::too_many_arguments)]
pub fn run_margin_row_lane_with_policy(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    policy: MarginRowRuntimePolicy,
) -> MarginRowOutcome {
    lane_impl_with_policy(spec, lo, hi, t, adv, deadline, max_expansions, &policy)
}

/// Quarantined compatibility entry for the unproven external-head experiment.
///
/// `external_head` is unconditionally discarded. In particular,
/// `NY_MARGIN_ROW_HEAD_INJECT` has no authority: no environment value can move
/// a bound or verdict through this entry point. This function is a hard
/// semantic alias of [`run_margin_row_lane`] until a typed, provenance-checked
/// `CertifiedHeadBox` exists. See `docs/MARGIN_ROW_ROOT_JOINT_COUPLING.md`.
pub fn run_margin_row_lane_with_head(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    external_head: Option<(Vec<f64>, Vec<f64>)>,
) -> MarginRowOutcome {
    drop(external_head);
    run_margin_row_lane(spec, lo, hi, t, adv, deadline, max_expansions)
}

#[allow(clippy::too_many_arguments)]
fn lane_impl(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
) -> MarginRowOutcome {
    lane_impl_with_policy(
        spec,
        lo,
        hi,
        t,
        adv,
        deadline,
        max_expansions,
        &MarginRowRuntimePolicy::from_process_environment(),
    )
}

#[allow(clippy::too_many_arguments)]
/// #publish-deadline: seconds of the caller's wrap-up reserve that may be spent
/// publishing an ALREADY-COMPLETE proof.
///
/// MUST stay strictly below the 3 s the caller reserves, or the tail it exists
/// to protect (result write, witness gating) is consumed and a verdict can be
/// lost to exactly the overrun this is meant to prevent
/// (see memory `post-verdict-work-can-destroy-the-verdict`).
///
/// `0.0` ships INERT: `publish_deadline` stays `None` and falls back to the
/// search deadline, byte-identical to the historical behaviour.
pub(crate) const PUBLISH_GRACE_SECS: f64 = 0.0;

///
/// #router-r2 CR-R: when the ACTING router armed the one deadline authority
/// with a publish grace (`ny_core::deadline_authority`, `0 < g < 3 s` and
/// below the join margin, enforced when it is armed), that grace is used, and
/// every collector of this lane waits the same `g` (the inline lane's external
/// cap and the concurrent lane's final join). Unarmed — every run without
/// `router.mode: on` — this is exactly the constant path above.
pub(crate) fn publish_grace() -> Option<std::time::Duration> {
    if let Some(grace) = ny_core::deadline_authority::global().publish_grace() {
        return Some(grace).filter(|g| *g < std::time::Duration::from_secs(3));
    }
    (PUBLISH_GRACE_SECS > 0.0 && PUBLISH_GRACE_SECS < 3.0)
        .then(|| std::time::Duration::from_secs_f64(PUBLISH_GRACE_SECS))
}

/// #mr-census wrapper: every lane invocation prints exactly ONE `[mr-census]`
/// line, on every return path of [`lane_impl_with_policy_body`] and during a
/// panic unwind (the line is emitted by the guard's `Drop`). The body is
/// unchanged; this wrapper only brackets it.
#[allow(clippy::too_many_arguments)]
fn lane_impl_with_policy(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    runtime_policy: &MarginRowRuntimePolicy,
) -> MarginRowOutcome {
    // #attack-open-disjuncts: bracket the invocation so a consumer waiting
    // for the root can tell "still building" from "returned without one".
    // Inert (no state change) unless the publication was armed.
    let _root_open_lane = root_open::LaneGuard::enter();
    let mut lane_census = census::LaneCensus::begin();
    let out = lane_impl_with_policy_body(
        spec,
        lo,
        hi,
        t,
        adv,
        deadline,
        max_expansions,
        runtime_policy,
    );
    // #mr-close-telemetry: publish the returned lane's exact proof-integrity
    // tally BEFORE the census guard drops, so its `[mr-close]` stop receipt
    // prints the final state rather than the last live one. Telemetry only.
    close_integrity::publish_outcome(&out);
    lane_census.finish(&out);
    out
}

/// Resolve the frame configuration once, before the optional direct proof or
/// any mutable root owner. In particular, publication grace and the legacy
/// scheduling cells cannot differ between the early and ordinary authorizers.
#[allow(clippy::too_many_arguments)]
fn frozen_bab_config(
    runtime_policy: &MarginRowRuntimePolicy,
    root: &RootGates,
    max_expansions: usize,
    deadline: Option<Instant>,
    root_build_secs: f64,
    retain_cfg: Option<root::RetainCfg>,
    epoch_parent_generations_left: usize,
    root_alpha_execution: &RootAlphaExecutionEvidence,
) -> BabConfig {
    // y-pack cache capacity (trunk-sets). Bit-identical to the serial oracle
    // (a hit returns the SAME deterministic pack a miss would rebuild), so a
    // larger cache only trades memory for fewer recomputations — no bound
    // moves. Overridable for A/B measurement (`NY_MARGIN_ROW_LRU`).
    let lru_cap = crate::preset_gates::margin_row_lru_cap();
    // Parallel best-first frontier width. The legacy banked exact-timeout wall
    // sweep used all available Rayon workers and the serial-vs-parallel gates
    // prove every per-domain bound bit-identical; make that measured recipe the
    // margin-row-only default. `NY_MARGIN_ROW_PARALLEL=0` is the serial kill
    // switch, `=1` explicitly selects all workers, and `=N` pins a width.
    let frontier_available = rayon::current_num_threads().max(1);
    let frontier_contextual = crate::preset_gates::margin_row_frontier_lower(frontier_available);
    let frontier = crate::preset_gates::margin_row_frontier(frontier_available);
    // Default-OFF cross-domain score-row stack (`#domain-stack`).
    //
    // PROMOTED to the typed preset key `margin_row.domain_stack`. It was a raw
    // `std::env::var` read, which — with `run_instance.sh` exporting no `NY_*`
    // — meant the mechanism could not be armed on a scored binary at all, so
    // its wall-clock effect could never be measured through the route that
    // competes. It ships DARK (default-off, armed by no yaml): promote first,
    // measure second, arm third.
    //
    // Schedule only. Reductions are column-local, so a stacked score is
    // bit-identical to the independent one; this changes what the lane can
    // finish inside its budget, never a published bound.
    let domain_stack = crate::preset_gates::margin_row_domain_stack_enabled();
    // #margin-row-branch-width: EXPERIMENTAL candidate-budget override.
    //
    // ny scores k_head + k_trunk candidates EXACTLY in one batched pass per
    // expansion (default 8 + 8 = 16). Measured on the lane that actually proves
    // cifar100 rows: `score_candidates(batched)` is 40.0% of it at 554 ms/call.
    // alpha-beta-CROWN instead ranks 7 candidates with a CHEAP kfsb score and
    // spends exact work only on the winner, so its expansions cost milliseconds.
    //
    // Open question: idx_8600 runs 729 expansions and still fails, against 189
    // for a row that PROVES. Are 729 well-chosen expansions worse than several
    // thousand cheaper ones? These knobs exist to answer that.
    //
    // The complete configured and lower-authority width policy was frozen at
    // lane entry. Candidate selection and nested trees receive plain values;
    // no scored execution reopens environment or preset cells.
    let width = runtime_policy.width;
    BabConfig {
        max_expansions,
        deadline,
        head_groups: runtime_policy.head_groups,
        // #publish-deadline: a proof that is ALREADY COMPLETE may be published
        // slightly past the SEARCH deadline, because the remaining work is a
        // struct clone rather than more searching.
        //
        // The caller sets this lane's `deadline` to `scored_deadline - 3s`
        // (`ny-cli/.../margin_row_bab.rs:2681`) to reserve a tail for its own
        // wrap-up. `PUBLISH_GRACE_SECS` gives back a BOUNDED slice of that
        // reserve for publication only, and must stay STRICTLY BELOW 3 s so the
        // caller's tail is never consumed. `0.0` => `None` => byte-identical.
        publish_deadline: publish_grace().map(|g| deadline.map_or_else(Instant::now, |d| d + g)),
        lru_cap,
        frontier,
        frontier_contextual,
        k_head: width.k_head,
        contextual_k_head: width.contextual_k_head,
        k_trunk: width.k_trunk,
        contextual_k_trunk: width.contextual_k_trunk,
        effective_k_head: width.k_head,
        effective_contextual_k_head: width.contextual_k_head,
        effective_k_trunk: width.k_trunk,
        effective_contextual_k_trunk: width.contextual_k_trunk,
        adaptive: width.adaptive,
        contextual_adaptive: width.contextual_adaptive,
        domain_stack,
        domain_alpha_policy: runtime_policy.domain_alpha,
        retained: None,
        epoch_policy: runtime_policy.epoch,
        epoch_trigger_depth: runtime_policy.epoch.depth,
        epoch_attempt_cap: runtime_policy.epoch.max_attempts,
        epoch_parent_generations_left,
        root_build_secs,
        retain_cfg,
        col_retire: runtime_policy.column_retire,
        beta_per_column: runtime_policy.beta_per_column,
        beta_columns: runtime_policy.beta_columns,
        parallel_prebuild: runtime_policy.parallel_prebuild,
        root_precision_policy: root.root_precision_policy,
        backward_interm_policy: runtime_policy.backward_interm,
        root_precision_execution: root.root_precision_execution,
        backward_interm_execution: root.backward_interm_execution,
        root_alpha_execution: *root_alpha_execution,
        // #wave-beta-bab (ENGINE_PROGRAM M6): `margin_row.tree_mode` and its
        // two knobs, frozen at lane entry with the rest of the runtime policy
        // (scored runs: from the receipt-owning LeverSet, so the tree the lane
        // runs is the tree the flight receipt names; one authority, no cell
        // re-read). Dark (`legacy`, the declaration default, armed by no
        // shipped yaml) the tree dispatch is the established one, bit for bit.
        wave: runtime_policy.wave,
        ..BabConfig::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn lane_impl_with_policy_body(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    runtime_policy: &MarginRowRuntimePolicy,
) -> MarginRowOutcome {
    let lane_start = Instant::now();
    let mut runtime_policy = *runtime_policy;
    let requested_root_source = runtime_policy.root_precision.box_source;
    let source_conflicts = root_box::policy_conflicts(&runtime_policy);
    runtime_policy.root_precision.box_source = root_box::select_source(
        spec,
        runtime_policy.root_precision.box_source,
        source_conflicts,
    );
    let wide_patch = !spec.root_tableau_within_cap()
        && runtime_policy.root_precision.box_source == RootBoxSource::PatchBackward;
    if wide_patch {
        eprintln!("[root-source-admission] requested={} effective={} working_set_bytes={} tableau_entries={:?} cap={}",
            requested_root_source.token(), runtime_policy.root_precision.box_source.token(),
            root_box::wide_working_set_bytes(), spec.root_tableau_entries(), root_box::ROOT_SOURCE_BYTES_CAP);
    }
    if wide_patch && !root_box::wide_consumers_supported(&runtime_policy) {
        return MarginRowOutcome::Unknown {
            reason:
                "wide patch source: unpriced root/device/wave/beta/frozen-tier consumers; complete-engine fallback"
                    .into(),
            stats: None,
        };
    }
    if prof::enabled() {
        prof::reset();
    }
    // #mr-errlane R9: the frozen error-lane policy announces itself once per
    // lane, armed or dark, before any engine exists. Arrival AT THE ENGINE is
    // the stop receipt's `env_passes=` / `pc_passes=`, not this line.
    errlane::announce(runtime_policy.root_precision.err_lane);
    // #mr-gpu-eval R9: the conv-offload route announces itself once per lane,
    // armed or dark. Arrival AT THE ENGINE is the stop receipt's `gpu=`.
    gpu_conv::announce();
    let net = match TwinNet::compile_with_root_source(
        spec,
        runtime_policy.root_precision.box_source,
        source_conflicts,
    ) {
        Ok(n) => n,
        Err(e) => {
            return MarginRowOutcome::Unknown {
                reason: format!("compile: {e}"),
                stats: None,
            }
        }
    };
    // Producer provenance is enabled before any root/source arithmetic and
    // held for this invocation only. Raw rows and split roots cannot acquire
    // a full-box head-group binding. The dark path performs no source digest.
    let _head_group_source =
        head_group_source::HeadGroupSourceGuard::enter(runtime_policy.head_groups.is_some());
    // Tier-0 (#epoch-bab): the rank-1 candidate ranker scores a WIDE trunk
    // universe in O(nf·n_in) each and sends only the best `k` to the exact
    // Tier-1 pass. DEFAULT OFF; `NY_EPOCH_TIER0=k` enables it.
    //
    // Ranker-only by construction: candidate ORDERING cannot move a verdict
    // — every pushed bound and every closure still flows through the
    // unchanged certified Outward pass, so the worst a bad ranking can do
    // is waste budget.
    //
    // Held OFF by the completed same-binary A/B on the cifar100 pyrat-easy
    // tier (docs/EPOCH_BAB_STATE_2026-07-18.md). Established: expansion
    // counts IDENTICAL wherever both lanes close (2, 13, 29, 23, 31 — the
    // verified tree is reproduced exactly), `score_candidates` 124 → 66
    // ms/call, and every shared closure is reached faster. NOT established:
    // a net closure gain — the tier count is 6 = 6 (Tier-0 gains prop_1588,
    // loses prop_4605; wider ranking explores a different tree, better on
    // some instances and worse on others). Since scoring is 10 points per
    // solve against a small time bonus, closure count is the load-bearing
    // number and it is a wash, so this stays gated until an A/B over the
    // full wall set — not a 13-instance tier — shows a net win.
    //
    // Declared #epoch-bab levers since the ratchet conversion. Their frozen
    // policy resolver preserves the historical parser, which is
    // unchanged and deliberately UNTRIMMED: `LeverKind::U64` is
    // `parse::<u64>()` with no `trim()`, which is what these readers always
    // did, so `" 4"` remains a rejection that lands on the declaration default
    // before the lane starts.
    let epoch_policy = runtime_policy.epoch;
    let tier0_exact = epoch_policy.tier0_exact;
    // Tier-2 (#epoch-bab): `NY_EPOCH_DEPTH=d` re-linearizes a domain's
    // subtree once it carries `d` trunk splits (0/unset = off);
    // `NY_EPOCH_ATTEMPTS` caps Unknown-ending rebuilds per run.
    // The scored route consumes the entry-time receipt-owned policy; legacy
    // environment/preset layering is confined to the compatibility constructor.
    let epoch_depth = epoch_policy.depth;
    let epoch_max_attempts = epoch_policy.max_attempts;
    // #epoch-parent: dark incremental refresh. A positive value is the max
    // number of downstream preactivation columns re-bounded per layer from
    // the already-tight parent frame. Unset/0 preserves the full-tableau epoch
    // path exactly; no VNN-COMP preset arms this pending sealed measurement.
    let epoch_parent_topk = epoch_policy.parent_topk;
    let epoch_parent_generations_left =
        epoch_parent_generation_budget(epoch_parent_topk, epoch_depth, epoch_max_attempts);
    if epoch_parent_generations_left > 0 {
        eprintln!(
            "[epoch-parent] ARMED: topk={} cadence={} new trunk splits attempts={} \
             generations={}",
            epoch_parent_topk,
            epoch_depth,
            epoch_max_attempts,
            bab::EPOCH_PARENT_GENERATION_CAP
        );
    } else if epoch_parent_topk > 0 {
        let reason = if epoch_depth == 0 {
            "epoch_depth=0"
        } else {
            "epoch_attempts=0"
        };
        eprintln!(
            "[epoch-parent] INERT: topk={} requested but {}; no parent refresh can run",
            epoch_parent_topk, reason
        );
    }
    // The incremental arm neither consumes nor can safely inherit Tier-0 rows
    // (its refreshed unstable index space differs). Do not buy that retention
    // unless Tier-0 itself is requested or the historical full epoch needs it.
    let retain_cfg =
        epoch_retention_requested(tier0_exact, epoch_depth, epoch_parent_topk).then(|| {
            root::RetainCfg {
                per_layer: epoch_policy.retain_per_layer,
                ..root::RetainCfg::default()
            }
        });
    let build_t0 = Instant::now();
    let (mut root, mut retained) = {
        let _t = prof::Timer::start(prof::Phase::RootGate);
        match RootGates::build_retaining_with_policy(
            &net,
            lo,
            hi,
            RoundMode::Outward,
            deadline,
            retain_cfg.as_ref(),
            &[],
            runtime_policy.root_precision,
            runtime_policy.backward_interm,
            // The ROOT build has no prior alpha to carry — `gates_from_box` IS
            // the origin here. Only a Tier-2 epoch rebuild seeds (#epoch-carry).
            None,
        ) {
            Ok(r) => r,
            Err(e) => {
                return MarginRowOutcome::Unknown {
                    reason: format!("root gates: {e}"),
                    stats: None,
                }
            }
        }
    };
    let root_build_secs = build_t0.elapsed().as_secs_f64();
    // #alpha-opt (`NY_MARGIN_ROW_ALPHA_OPT=1`, default OFF — byte-identical
    // when unset): optimize the root trunk alpha where the root is still owned
    // mutably, BEFORE any consumer (probe, y-pack, tree) reads the gates. The
    // committed gates then produce the published root bound through the
    // unchanged `root_eval`; the retained Tier-0 rows and `clip_rows` stay
    // valid because the forward tableau and every `(l, u)` are frozen — only
    // `alpha` (unstable neurons) and its Lipschitz coupling `ms` move.
    let root_alpha_eligible = alpha_opt::execution_eligible(&root, adv);
    // Earliest supported full-root entry: the forward boxes are qualified,
    // while standalone alpha, rebake, joint refresh and certify-once have not
    // yet produced identity rows or evaluated the root. WidePatch retains its
    // existing separate admission/replay path and is explicitly ineligible.
    let mut cfg = frozen_bab_config(
        &runtime_policy,
        &root,
        max_expansions,
        deadline,
        root_build_secs,
        retain_cfg,
        epoch_parent_generations_left,
        &RootAlphaExecutionEvidence::skipped_by_direct_cover(
            runtime_policy.root_alpha,
            root_alpha_eligible,
        ),
    );
    let mut direct = if wide_patch {
        None
    } else {
        let engine = engine::BackwardEngine::new(&net, &root);
        bab::direct_root_attempt(&engine, t, adv, &cfg)
    };
    {
        let engine = engine::BackwardEngine::new(&net, &root);
        if let Some(out) =
            bab::direct_root_outcome(&engine, t, adv, &cfg, &[], direct.as_mut(), lane_start)
        {
            eprintln!(
                "[root-owners] skipped=direct_complete_cover authorized={} alpha_enabled={} alpha_guard={} rebake_enabled={} joint_refresh_rounds={} root_certify_once={}",
                matches!(&out, MarginRowOutcome::Unsat(..)),
                runtime_policy.root_alpha.alpha_enabled,
                runtime_policy.root_alpha.alpha_guard,
                runtime_policy.root_alpha.rebake_enabled,
                runtime_policy.joint_refresh.rounds,
                runtime_policy.root_certify_once.enabled,
            );
            if prof::enabled() {
                eprint!("{}", prof::dump());
            }
            return out;
        }
    }
    let alpha_report = alpha_opt::maybe_optimize_root_alpha(
        &net,
        &mut root,
        t,
        adv,
        deadline,
        runtime_policy.root_alpha,
    );
    // #rebake (`NY_MARGIN_ROW_REBAKE=1`, default OFF — byte-identical when
    // unset): the joint forward re-bake. Iterates alpha ascent -> forward
    // rebuild with the optimized alpha baked into `apply_gates` -> interm
    // re-run, with round-level monotone accept on the certified `root_eval`
    // bound. On accept the WHOLE (root, retained) pair is replaced by the
    // rebuilt one, so the Tier-0 rows and `clip_rows` always describe the
    // tableau actually published. Runs while the root is still owned mutably,
    // before any consumer reads the gates.
    let rebake_report = rebake::maybe_rebake_root(
        &net,
        &mut root,
        &mut retained,
        retain_cfg.as_ref(),
        t,
        adv,
        deadline,
        root_build_secs,
        runtime_policy.root_alpha,
        runtime_policy.backward_interm,
    );
    // #joint-refresh (`margin_row.joint_refresh`, default 0 — byte-identical
    // when unset: the entry returns before touching anything and prints
    // nothing). The backward-only half of alpha-CROWN's joint loop: per round,
    // re-derive every trunk intermediate box from the CURRENT gates (the
    // committed alpha included) with the lane's own certified engine,
    // shrink-only, then a short ascent whose frozen head frame is rebuilt from
    // the refreshed gates, accepted only on a strict improvement of the
    // unchanged certified `root_eval` bound. Runs while the root is still owned
    // mutably, after both ascent owners, before `#spec-alpha` and every
    // consumer. The known baseline is the certified bound the previous owner
    // already computed for EXACTLY these gates (a wrong value could only change
    // which internally certified build is kept, never a bound).
    let joint_known_baseline = match (rebake_report.as_ref(), alpha_report.as_ref()) {
        (Some(rebake), _) => Some(rebake.best),
        (None, Some(alpha)) => {
            if alpha.accepted {
                alpha.published_after
            } else {
                alpha.published_before
            }
        }
        (None, None) => None,
    };
    let joint_refresh_report = rebake::maybe_joint_refresh_root(
        &net,
        &mut root,
        retained.is_some(),
        t,
        adv,
        deadline,
        joint_known_baseline,
        runtime_policy.joint_refresh,
        runtime_policy.root_alpha,
        runtime_policy.backward_interm,
    );
    // #root-certify-once (`margin_row.root_certify_once`, default false —
    // byte-identical when unset: the entry returns before touching anything
    // and prints nothing). Phase-0.5 root alpha, SEARCH UNTRUSTED / CERTIFY
    // ONCE: the root alpha ascent (optionally alternating with an untrusted
    // intermediate refresh) runs on a Parity SCRATCH copy of the root, then ONE
    // certified `root_eval` of the chosen alpha (plus, after a joint phase, one
    // certified refresh of its boxes) is published only if it strictly beats
    // the certified incumbent — the root every owner above left, i.e. today's
    // root. Runs while the root is still owned mutably, after every ascent
    // owner, before `#spec-alpha` and every consumer.
    //
    // The known incumbent is the certified bound the previous owner computed
    // for EXACTLY these gates: `#joint-refresh`'s `best` when it ran far
    // enough to score the root (a refusal before scoring leaves `best` NaN and
    // the root untouched, so the earlier owner's value still describes it).
    let certify_once_known_incumbent = match joint_refresh_report.as_ref() {
        Some(report) if report.best.is_finite() => Some(report.best),
        _ => joint_known_baseline,
    };
    let _root_certify_once_report = root_certify_once::maybe_certify_once_root(
        &net,
        &mut root,
        retained.is_some(),
        t,
        adv,
        deadline,
        certify_once_known_incumbent,
        runtime_policy.root_certify_once,
        runtime_policy.root_alpha,
        runtime_policy.backward_interm,
    );
    let root_alpha_execution = RootAlphaExecutionEvidence::completed(
        runtime_policy.root_alpha,
        root_alpha_eligible,
        alpha_report.as_ref(),
        rebake_report.as_ref(),
    );
    // #spec-alpha: install the per-START-NODE alpha policy on the root, on the
    // still-mutably-owned root, AFTER both ascent owners are done with it.
    //
    // THE ORDERING IS LOAD-BEARING, not stylistic. `#rebake` re-runs the
    // FORWARD walk with the optimized alpha baked into `apply_gates`, and that
    // walk charges ONE slope pair per neuron (`gate_gerr_out(alpha[j], s[j],
    // ..)`), so a per-start-node alpha has nowhere to go in it TODAY.
    // Installing the policy here means `rebake::certified_root_bound` and every
    // `#alpha-opt` published-frame guard evaluation observe `None`, so the two
    // mechanisms are provably disjoint rather than disjoint by discipline.
    //
    // CORRECTION (#gate-gerr-max-nonuniform): this comment used to say the
    // forward contraction REQUIRES alpha uniform over the coefficient index and
    // that "a per-start-node alpha cannot be baked forward". That was a
    // statement about the derivation's convenient FORM, not about the bound.
    // `gate_gerr_out_nonuniform` carries the generalized charge
    // `max_i max(a_ji, s_ji)`, a valid ERROR carry for slopes that vary over
    // the coefficient index, and it still contracts. That settles the error
    // functional only. The coefficient index is an INPUT coordinate
    // (`naug = n_in + 1`), not a start node, and whether the relaxation stays a
    // valid enclosure under a non-uniform lower slope is OPEN (see
    // `gate_gerr_out_nonuniform`). Do NOT arm a non-uniform
    // alpha on the forward walk; the disjointness above is a property of the
    // CURRENT forward walk and must stay one until that question is answered.
    root.spec_alpha_policy =
        spec_alpha::SpecAlphaPolicy::from_policy(runtime_policy.root_alpha, deadline);
    let root = root; // freeze: no mutation past this point
                     // READ-ONLY diagnostic (#hyperplane-probe): default-off behind
                     // `NY_HYPERPLANE_PROBE`. Computes and logs the trunk->head Jacobian
                     // spectrum, then returns. Touches no bound, gate, or verdict; when the env
                     // var is unset this closure is never entered, so behavior is byte-identical.
    if hyperplane_probe::enabled() {
        hyperplane_probe::run(&net, &root, t, adv);
    }
    // #frozen-dual-bab (V1 stage B): `margin_row.frozen_dual_bab`, declaration
    // default false and armed by no shipped yaml, no longer runs the stage-A
    // state probe here. It arms the certified search TIER inside the tree
    // driver (`MarginRowBab::run_inner` / `run_classwise_inner`, outer frame
    // only), which reuses the tree's own root evaluation and hands every class
    // it cannot close back to the tree unchanged. Dark, that is one latched
    // bool test and nothing else runs.
    // Mutable root owners finished. The same entry-time configuration and
    // optional allowance are handed on; only their actual execution report
    // and the retained rows are attached here.
    cfg.root_alpha_execution = root_alpha_execution;
    cfg.retained = retained.map(std::sync::Arc::new);
    // Default-OFF bounded experiment: replace the joint multi-margin tree with
    // a conjunction of independent full-box class proofs. Unset/0 preserves
    // the established joint call exactly.
    let classwise = runtime_policy.classwise;
    if wide_patch {
        let census = match root_box::source_census(spec) {
            Some(census) => census,
            None => {
                return MarginRowOutcome::Unknown {
                    reason: "wide patch source: consumer census overflow".into(),
                    stats: None,
                }
            }
        };
        if let Err(error) = root_box::wide_memory_admission(census.root_peak_bytes) {
            return MarginRowOutcome::Unknown {
                reason: format!("wide patch root consumer admission: {error}"),
                stats: None,
            };
        }
        let engine = engine::BackwardEngine::new(&net, &root);
        let evaluated = match bab::root_eval_store(&engine, &net, t, adv) {
            Ok(evaluated) => evaluated,
            Err(error) => {
                return MarginRowOutcome::Unknown {
                    reason: format!("wide patch root replay: {error}"),
                    stats: None,
                }
            }
        };
        let nf = evaluated.tree_classes.len();
        let consumers =
            root_box::consumer_census(census, &runtime_policy, nf, cfg.frontier, cfg.lru_cap);
        let columns = consumers.map(|c| c.columns);
        let peak = consumers.map(|c| c.peak_bytes);
        eprintln!("[root-source-admission] source=patch_backward tableau_entries={} root_peak_bytes={} tree_columns={:?} workers={} tree_peak_bytes={:?} cap={} unresolved={nf}",
            census.tableau_entries, census.root_peak_bytes, columns, cfg.frontier.max(1), peak, root_box::ROOT_SOURCE_BYTES_CAP);
        if nf > 0 && peak.is_none_or(|n| n > root_box::ROOT_SOURCE_BYTES_CAP) {
            return MarginRowOutcome::Unknown { reason: "wide patch source: candidate/cache consumer budget refused; complete-engine fallback (tile_score_candidates required)".into(), stats: None };
        }
        if nf > 0 {
            if let Err(error) = root_box::wide_memory_admission(
                peak.expect("consumer cap guard admitted an exact peak"),
            ) {
                return MarginRowOutcome::Unknown {
                    reason: format!("wide patch tree consumer admission: {error}"),
                    stats: None,
                };
            }
        }
    }
    // #router-r3 O-STOP: the tree is the sink of every early stop above it
    // (the alpha ascent's `rule.governor` yield, the interm candidates'
    // `rule.ia_yield` credit). It starts with the lane's own deadline — the
    // one the R2 cell recorded, never moved — so whatever an earlier phase
    // freed is inside its window; the ledger says so (`realized=`). Prints
    // and records nothing unless the acting optimizer armed a stop rule.
    let _ = marginal::hand_on_to("margin-row.tree", deadline);
    let out = if classwise {
        MarginRowBab::run_classwise_with_direct(&net, &root, t, adv, cfg, direct)
    } else {
        MarginRowBab::run_with_direct(&net, &root, t, adv, cfg, direct)
    };
    if prof::enabled() {
        eprint!("{}", prof::dump());
    }
    out
}

/// Parse the existing classwise request without reading process state. CLI
/// dispatch can freeze this request before any competing proof engine starts.
pub fn margin_row_classwise_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1"
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("on")
            || value.eq_ignore_ascii_case("yes")
    })
}

/// Historical exact-token parser for `NY_MARGIN_ROW_DOMAIN_STACK`.
///
/// No longer on the production path: the gate is now the typed preset key
/// `margin_row.domain_stack`, whose `bool_gate!` cell keeps environment
/// precedence and applies the registry's own exact-`"1"`/`"0"` arming rule.
/// Retained under `cfg(test)` so the parse contract this promotion inherited
/// stays pinned.
#[cfg(test)]
fn margin_row_domain_stack_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

pub(crate) fn margin_row_frontier_from_env(value: Option<&str>, available: usize) -> usize {
    let available = available.max(1);
    match value.map(str::trim) {
        None => available,
        Some("") | Some("0") => 1,
        Some(v)
            if v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no") =>
        {
            1
        }
        Some("1") => available,
        Some(v)
            if v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("on")
                || v.eq_ignore_ascii_case("yes") =>
        {
            available
        }
        Some(v) => v.parse::<usize>().unwrap_or(1).max(1),
    }
}
