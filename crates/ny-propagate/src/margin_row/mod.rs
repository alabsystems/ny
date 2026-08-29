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
pub mod bounds;
/// #clip-and-verify: split-derived halfspaces + the closed-form dual that
/// re-concretizes over `box INTERSECT halfspaces`. See the module docs for the
/// frontier-explosion measurement that motivates it.
pub mod clip;
/// #closure-adaptive: the runtime policy that picks this lane's closure
/// mechanisms from the child-closure fraction the lane measures about ITSELF.
pub mod closure;
/// #margin-row-domain-alpha: default-OFF PER-DOMAIN alpha. The lower-line
/// slope is re-optimized against each BaB domain's own gates (clip tightening
/// included) instead of being frozen at the root value, and each child
/// warm-starts from its parent's accepted alpha. Tightness-only — any
/// `alpha in [0, 1]` is a valid lower line pointwise — with the certified pass
/// as the only scorer. See the module docs for the soundness obligations and
/// the clip-pairing measurement contract.
pub(crate) mod domain_alpha;
pub mod engine;
pub(crate) mod gpu_backward;
pub(crate) mod gpu_seam;
pub mod hyperplane_probe;
pub mod net;
pub mod prof;
/// #rebake: frozen-policy gated (typed preset or exact legacy
/// `NY_MARGIN_ROW_REBAKE=1`, declaration default OFF) joint forward re-bake —
/// iterate alpha ascent -> forward tableau rebuild with the
/// optimized alpha baked into `apply_gates` -> backward-interm re-run, with
/// round-level monotone accept on the certified `root_eval` bound. See the
/// module docs for the soundness argument and the one re-derived widening.
pub mod rebake;
pub mod root;
pub mod rounding;
pub mod spec; // #margin-row-gpu-eft: dark skeleton, docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md

#[cfg(test)]
mod tests;

use std::time::Instant;

pub use alpha_opt::RootAlphaExecutionEvidence;
pub use bab::{
    BabConfig, BabStats, ClassBabStats, EpochExecutionEvidence, EpochFrameRoute,
    FrontierExecutionEvidence, FrontierRoute, MarginRowBab, MarginRowOutcome,
    PolicyExecutionEvidence, WidthExecutionEvidence,
};
pub use backward_interm::BackwardIntermExecutionEvidence;
/// #margin-row-gpu-eft: register the injected device executor for the
/// certified GPU backward lane (ny-cli calls this once at startup; the
/// closure is only invoked on the first ARMED admission).
pub use gpu_backward::device::set_margin_row_lane_gpu_factory;
/// #binary-identity: the implementation-generation marker `ny --build-id`
/// compiles into the binary's canonical identity, so the R8 provenance check
/// is a field comparison rather than a remembered `strings | grep`.
pub use gpu_backward::PROVENANCE_MARKER as GPU_BACKWARD_PROVENANCE_MARKER;
pub use domain_alpha::Preset as DomainAlphaPreset;
pub use net::TwinNet;
pub use root::RootGates;
pub use root::RootPrecisionExecutionEvidence;
pub use rounding::RoundMode;
pub use spec::{TwinOpSpec, TwinSpec};

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
                .ok_or_else(|| {
                    format!("{} is not a platform-sized unsigned integer", decl.name)
                })
        };
        let min_depth = unsigned(&d::MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH)?;
        let stride = unsigned(&d::MARGIN_ROW_DOMAIN_ALPHA_STRIDE)?;
        // The declaration's `0.0` default (and any non-positive value) means
        // "no window"; `normalized` maps those onto `f64::INFINITY`.
        let gap = value(&d::MARGIN_ROW_DOMAIN_ALPHA_GAP)?
            .as_f64()
            .ok_or_else(|| {
                format!("{} is not a number", d::MARGIN_ROW_DOMAIN_ALPHA_GAP.name)
            })?;
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
        }
    }
}

/// Immutable entry-time policy for root-tableau compute and storage precision.
///
/// The two values must travel together because f32 storage is meaningful only
/// when f32 compute is effective. Keeping them in the receipt-owned runtime
/// policy also prevents rebakes and nested epoch roots from observing a later
/// process-global mutation than the initial root did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RootPrecisionRuntimePolicy {
    pub root_f32: bool,
    pub tableau_f32: bool,
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
        Ok(Self {
            root_f32: resolved_bool(&d::MARGIN_ROW_ROOT_F32)?,
            tableau_f32: resolved_bool(&d::MARGIN_ROW_TABLEAU_F32)?,
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
        Self {
            root_f32,
            tableau_f32: crate::preset_gates::margin_row_tableau_f32_enabled(),
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
        Ok(Self {
            enabled,
            secs: if secs == 0.0 { 0.0 } else { secs },
            chunk,
            chunk_secs: if chunk_secs == 0.0 { 0.0 } else { chunk_secs },
            topk,
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
        }
    }

    pub(crate) fn config(self, mode: RoundMode) -> Option<backward_interm::BiCfg> {
        (self.enabled && mode.outward()).then_some(backward_interm::BiCfg {
            secs: self.secs,
            chunk: self.chunk,
            chunk_secs: self.chunk_secs,
            topk: self.topk,
        })
    }

    pub(crate) fn from_config(config: Option<backward_interm::BiCfg>) -> Self {
        config.map_or_else(Self::default, |config| Self {
            enabled: true,
            secs: config.secs,
            chunk: config.chunk.clamp(1, 4096),
            chunk_secs: config.chunk_secs,
            topk: config.topk.max(1),
        })
    }
}

impl Default for MarginRowRuntimePolicy {
    fn default() -> Self {
        Self {
            width: MarginRowWidthRuntimePolicy::default(),
            root_precision: RootPrecisionRuntimePolicy::default(),
            backward_interm: BackwardIntermRuntimePolicy::default(),
            root_alpha: RootAlphaRuntimePolicy::default(),
            domain_alpha: DomainAlphaRuntimePolicy::default(),
            epoch: EpochRuntimePolicy::default(),
            beta_per_column: false,
            beta_columns: 8,
            column_retire: false,
            parallel_prebuild: false,
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
        Ok(Self {
            width: MarginRowWidthRuntimePolicy::from_levers(levers)?,
            root_precision: RootPrecisionRuntimePolicy::from_levers(levers)?,
            backward_interm: BackwardIntermRuntimePolicy::from_levers(levers)?,
            root_alpha: RootAlphaRuntimePolicy::from_levers(levers)?,
            domain_alpha: DomainAlphaRuntimePolicy::from_levers(levers)?,
            epoch: EpochRuntimePolicy::from_levers(levers)?,
            beta_per_column,
            beta_columns,
            column_retire,
            parallel_prebuild,
        })
    }

    fn from_process_environment() -> Self {
        let levers = ny_levers::LeverSet::freeze(ny_levers::all());
        let mut policy = Self::from_levers(&levers)
            .expect("the complete workspace LeverSet must contain the margin-row policy");
        policy.width = MarginRowWidthRuntimePolicy::from_layered_process();
        policy.root_precision = RootPrecisionRuntimePolicy::from_layered_process();
        policy.backward_interm = BackwardIntermRuntimePolicy::from_layered_process();
        policy.root_alpha = RootAlphaRuntimePolicy::from_layered_process();
        policy.domain_alpha = DomainAlphaRuntimePolicy::from_layered_process();
        policy.epoch = EpochRuntimePolicy::from_layered_process();
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

fn epoch_retention_requested(
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
        DomainAlphaRuntimePolicy, EpochRuntimePolicy, MarginRowRuntimePolicy,
        MarginRowWidthRuntimePolicy, RootAlphaRuntimePolicy, RootPrecisionRuntimePolicy,
    };

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
            _ => None,
        });
        assert_eq!(
            MarginRowRuntimePolicy::from_levers(&levers).unwrap(),
            MarginRowRuntimePolicy {
                width: MarginRowWidthRuntimePolicy::default(),
                root_precision: RootPrecisionRuntimePolicy {
                    root_f32: true,
                    tableau_f32: true,
                },
                backward_interm: BackwardIntermRuntimePolicy {
                    enabled: true,
                    secs: 40.0,
                    chunk: 1,
                    chunk_secs: 2.5,
                    topk: 1,
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
                },
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
            }
        );
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
    lane_impl_with_policy(spec, lo, hi, t, adv, deadline, max_expansions, policy)
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
        MarginRowRuntimePolicy::from_process_environment(),
    )
}

#[allow(clippy::too_many_arguments)]
fn lane_impl_with_policy(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_expansions: usize,
    runtime_policy: MarginRowRuntimePolicy,
) -> MarginRowOutcome {
    if prof::enabled() {
        prof::reset();
    }
    let net = match TwinNet::compile(spec) {
        Ok(n) => n,
        Err(e) => {
            return MarginRowOutcome::Unknown {
                reason: format!("compile: {e}"),
                stats: None,
            }
        }
    };
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
    let root_precision_policy = root.root_precision_policy;
    let root_precision_execution = root.root_precision_execution;
    let backward_interm_execution = root.backward_interm_execution;
    // #alpha-opt (`NY_MARGIN_ROW_ALPHA_OPT=1`, default OFF — byte-identical
    // when unset): optimize the root trunk alpha where the root is still owned
    // mutably, BEFORE any consumer (probe, y-pack, tree) reads the gates. The
    // committed gates then produce the published root bound through the
    // unchanged `root_eval`; the retained Tier-0 rows and `clip_rows` stay
    // valid because the forward tableau and every `(l, u)` are frozen — only
    // `alpha` (unstable neurons) and its Lipschitz coupling `ms` move.
    let root_alpha_eligible = alpha_opt::execution_eligible(&root, adv);
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
    let root_alpha_execution = RootAlphaExecutionEvidence::completed(
        runtime_policy.root_alpha,
        root_alpha_eligible,
        alpha_report.as_ref(),
        rebake_report.as_ref(),
    );
    let root = root; // freeze: no mutation past this point
                     // READ-ONLY diagnostic (#hyperplane-probe): default-off behind
                     // `NY_HYPERPLANE_PROBE`. Computes and logs the trunk->head Jacobian
                     // spectrum, then returns. Touches no bound, gate, or verdict; when the env
                     // var is unset this closure is never entered, so behavior is byte-identical.
    if hyperplane_probe::enabled() {
        hyperplane_probe::run(&net, &root, t, adv);
    }
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
    // Default-OFF cross-domain score-row stack. Only exact `=1` arms this
    // canary; unset, malformed, and friendly truthy spellings all retain the
    // established independent `score_candidates` calls.
    let domain_stack = margin_row_domain_stack_from_env(
        std::env::var("NY_MARGIN_ROW_DOMAIN_STACK").ok().as_deref(),
    );
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
    let cfg = BabConfig {
        max_expansions,
        deadline,
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
        retained: retained.map(std::sync::Arc::new),
        epoch_policy,
        epoch_trigger_depth: epoch_policy.depth,
        epoch_attempt_cap: epoch_policy.max_attempts,
        epoch_parent_generations_left,
        root_build_secs,
        retain_cfg,
        col_retire: runtime_policy.column_retire,
        beta_per_column: runtime_policy.beta_per_column,
        beta_columns: runtime_policy.beta_columns,
        parallel_prebuild: runtime_policy.parallel_prebuild,
        root_precision_policy,
        root_precision_execution,
        backward_interm_execution,
        root_alpha_execution,
        ..BabConfig::default()
    };
    // Default-OFF bounded experiment: replace the joint multi-margin tree with
    // a conjunction of independent full-box class proofs. Unset/0 preserves
    // the established joint call exactly.
    let classwise =
        margin_row_classwise_from_env(std::env::var("NY_MARGIN_ROW_CLASSWISE").ok().as_deref());
    let out = if classwise {
        MarginRowBab::run_classwise(&net, &root, t, adv, cfg)
    } else {
        MarginRowBab::run(&net, &root, t, adv, cfg)
    };
    if prof::enabled() {
        eprint!("{}", prof::dump());
    }
    out
}

fn margin_row_classwise_from_env(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        value == "1"
            || value.eq_ignore_ascii_case("true")
            || value.eq_ignore_ascii_case("on")
            || value.eq_ignore_ascii_case("yes")
    })
}

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
