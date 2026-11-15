// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The lever SEARCH SPACE: which knobs an automated search may move, what values
//! it may give them, and which combinations are provably inert.
//!
//! WHY THIS IS NOT `crate::all()`. The registry is a *declaration* of levers that
//! happen to be governed; it is not a statement about what is worth searching, and
//! it is not complete. Three facts make a naive sweep over `all()` waste almost
//! its entire budget:
//!
//! 1. **Most points are inert.** The four root-phase levers have FIVE distinct
//!    behaviours, not sixteen: an earlier phase that fires OWNS the slot and the
//!    later ones never run. Twelve declared levers do nothing at all unless a
//!    companion is also set, and five of those companions are not declared here.
//! 2. **Some levers must never be moved automatically.** [`Class::Unsafe`] marks
//!    the ones that change what is being proved, replace a certified error bound,
//!    or hard-fail engine construction.
//! 3. **A measurement lever is not a treatment.** `MoatRisk::Low` means "cannot
//!    touch a bound" — it does NOT mean free. Extra clocks, formatting and stderr
//!    perturb a deadline-sensitive run, so telemetry axes are instruments and must
//!    never appear in a timing arm.
//!
//! THE PARSER CONTRACT, which a search violates by accident. `resolve_raw` arms a
//! Bool on the EXACT byte string `"1"` and disarms on exact `"0"`; `"true"`,
//! `"01"`, `" 1"` and `""` are REJECTIONS that resolve to the declaration default
//! and are recorded in `Resolved::rejected_raw`. A harness that emits `"true"`
//! therefore measures the baseline and reports it as a treatment. [`Domain::Bool`]
//! only ever emits `"0"` / `"1"`, and callers are expected to assert
//! `rejected_raw` is empty on every completed run.
//!
//! # The 2026-08-23 reachability batch (#lever-reachability)
//!
//! `tests/reachability_ratchet.rs` measured the debt this module had been
//! silently accumulating: **62 of the 75 axes that [`Class::VerdictAffecting`]
//! says can change an answer were [`Deliver::EnvOnly`]**, and at the time
//! `vnncomp_scripts/run_instance.sh` exported only one `NY_*` variable. So 83%
//! of everything the registry called verdict-affecting was dead code in the
//! scored configuration. That is not a bug in any single lever; it is a missing
//! step in the workflow — build behind an env gate, A/B it, bank it, **promote
//! it**, ship — whose fourth step has no forcing function.
//!
//! This batch took 62 to 30 in three honest ways, and it is worth separating
//! them, because only the first is work:
//!
//! * **21 PROMOTED.** A typed preset field, a reader, a bridge line
//!   (`commands::vnncomp::run_and_translate` -> `preset_gates::apply`), a
//!   setter/reader pair over an [`crate::gate`] cell, and this axis. Every one
//!   is default OFF and set by no shipped yaml, so the batch is byte-identical
//!   on every scored run: promotion makes a mechanism DELIVERABLE, it does not
//!   arm it.
//! * **4 CORRECTED.** `NY_BNN_SIGN_SPACE`, `NY_BNN_STE_PGD`,
//!   `NY_UPFRONT_ATTACK_PROBE_FIRST` and `NY_MARGIN_ROW_K_ADAPT` already had a
//!   typed key, a reader and a bridge line. The axis was simply WRONG about
//!   them, which meant the ratchet was counting a debt that had been paid — and
//!   a gate that miscounts in the safe direction still teaches the wrong lesson
//!   about which mechanisms are dark.
//! * **7 RECLASSIFIED** into `INSTRUMENT_ONLY`: the six `NY_STAR_DARK_*`
//!   measurement controls and `NY_MARGIN_ROW_GPU_SHADOW`. Each reason is
//!   recorded at the name itself rather than here, because a reclassification
//!   is only as good as the evidence attached to it. The rule applied was: if
//!   it can move a PUBLISHED BOUND it is a promotion, full stop — stealing
//!   wall clock is not moving a bound, and an instrument that steals wall clock
//!   is exactly what this list already existed to hold.
//!
//! # The 2026-08-24 completion
//!
//! The deferred `margin_row` and `ny-gpu` tranches have now landed, followed by
//! every remaining [`Class::SafeToSearch`] axis. All current searchable axes
//! have typed preset delivery; `tests/reachability_ratchet.rs` pins both the
//! verdict-affecting and the full searchable counts at zero env-only debt.
//! Unsafe, test-only and instrumentation names remain environment-only on
//! purpose and are not search dimensions.
//!
//! Two legacy parsers are deliberately NOT modelled as Bool, because for them the
//! usual tokens mean the opposite of what they look like:
//! `NY_MIP_TRACE` is presence-armed — `"0"` and even `""` ARM it — and
//! `NY_CONV_PATCHES_DEBUG` arms on any non-empty value that is not exactly `"0"`,
//! so `" 0"` (with a space) ARMS it. Both are telemetry, hence excluded anyway,
//! but they are listed here so nobody re-adds them as ordinary Bools.

use std::collections::BTreeMap;

/// What values a search may give an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Exactly `"0"` or `"1"`. No other spelling survives the parser.
    Bool,
    /// A fixed set of admissible tokens, emitted verbatim.
    Enum(&'static [&'static str]),
    /// Integer grid, emitted with `to_string()`.
    Grid(&'static [u64]),
}

/// How much damage a wrong value can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Never moved by an automated search. See each axis's `why` field.
    Unsafe,
    /// May change a published bound. Searchable, but every surviving candidate
    /// must clear the moat gate before promotion.
    VerdictAffecting,
    /// Cannot change a verdict's soundness. Still costs wall clock.
    SafeToSearch,
}

/// Whether a value this search picks can actually reach a scored run.
///
/// This is the single most expensive thing to get wrong.
/// `vnncomp_scripts/run_instance.sh` now exports no `NY_*` variables; every
/// lever runs at its typed preset/contextual/default layer unless the harness
/// explicitly injects an override. An axis that is [`Deliver::EnvOnly`] can be searched
/// and can be measured, but its result is worth zero until a typed preset key
/// exists — the failure mode `crates/ny-cli/tests/measured_gate_delivery.rs`
/// was written to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deliver {
    /// Reaches a scored run through this preset key.
    PresetKey(&'static str),
    /// Environment only: measurable, but NOT shippable as-is.
    EnvOnly,
}

/// One searchable dimension.
#[derive(Debug, Clone, Copy)]
pub struct Axis {
    pub name: &'static str,
    pub domain: Domain,
    pub class: Class,
    pub deliver: Deliver,
    /// Why this axis is classified as it is. Load-bearing for review.
    pub why: &'static str,
}

/// What an ABSENT override means at the live reader.
///
/// [`DefaultSpec`](crate::DefaultSpec) is a parser/receipt default, not always
/// an execution baseline. In particular, `DefaultSpec::Unset` may mean an
/// exact runtime value, a model/device-dependent policy, or the absence of an
/// otherwise optional mechanism. A search that guesses here spends a complete
/// instance budget on a value that may be byte-identical to absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsenceBaseline {
    /// The declaration's typed default is the execution baseline.
    DeclarationDefault,
    /// Absence is semantically different from every explicit domain token.
    Distinct,
    /// Absence is byte-identical to this exact domain token.
    Equivalent(&'static str),
    /// The effective value depends on model, device, or another runtime fact,
    /// so pre-flight cannot identify and drop a baseline-equivalent token.
    /// Explicit domain tokens remain valid treatments and must stay searchable.
    Contextual,
}

/// Non-default absence contracts for searchable axes.
///
/// This is data rather than prose so both the compiled CLI schema and the
/// developer-only source parser can consume the same contract. Any searchable
/// `DefaultSpec::Unset` declaration must appear here; a test below enforces it.
const ABSENCE_BASELINES: &[(&str, AbsenceBaseline)] = &[
    (
        "NY_ROOT_COMP_GPU_INTERM_ROWS",
        AbsenceBaseline::Equivalent("32"),
    ),
    (
        "NY_ROOT_COMP_GPU_INTERM_SECS",
        AbsenceBaseline::Equivalent("20"),
    ),
    ("NY_CONV_IBP_BLOCKED", AbsenceBaseline::Equivalent("1")),
    ("NY_MARGIN_ROW_PARALLEL", AbsenceBaseline::Contextual),
    ("NY_ALPHA_ZERO_YIELD_FRAC", AbsenceBaseline::Distinct),
    ("NY_BAB_RESERVE_FRAC", AbsenceBaseline::Equivalent("0.0")),
    ("NY_PER_NODE_FLOOR_SECS", AbsenceBaseline::Equivalent("2")),
    // Absence is an adaptive function of each live remaining budget, not an
    // unknown fixed number. Every explicit cap therefore changes the policy
    // even when one row happens to land on the same numeric ceiling.
    ("NY_PER_NODE_CAP_SECS", AbsenceBaseline::Distinct),
    ("NY_MARGIN_ROW_K_ADAPT", AbsenceBaseline::Equivalent("0")),
    (
        "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        AbsenceBaseline::Equivalent("0"),
    ),
    (
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        AbsenceBaseline::Contextual,
    ),
    ("NY_SWEEP_CLASS_ROWS", AbsenceBaseline::Distinct),
    ("NY_SWEEP_CLASS_MIB", AbsenceBaseline::Contextual),
    ("NY_BNN_SIGN_SPACE_TRUST_REGION", AbsenceBaseline::Distinct),
    ("NY_CGAN_STACKED_BUDGET_MB", AbsenceBaseline::Contextual),
    ("NY_POLLABLE_FAER", AbsenceBaseline::Equivalent("0")),
    // The declaration's false value is a source-sensitive sentinel: absent
    // selects the schedule/property-derived automatic policy, while explicit
    // false disables it. No single explicit token is baseline-equivalent.
    ("NY_UPFRONT_ATTACK", AbsenceBaseline::Contextual),
    ("NY_PGD_VJP_K", AbsenceBaseline::Contextual),
    ("NY_PGD_VJP_STEPS", AbsenceBaseline::Contextual),
];

/// Effective absence contract for one axis name.
#[must_use]
pub fn absence_baseline(name: &str) -> AbsenceBaseline {
    ABSENCE_BASELINES
        .iter()
        .find_map(|(candidate, baseline)| (*candidate == name).then_some(*baseline))
        .unwrap_or(AbsenceBaseline::DeclarationDefault)
}

/// Ordered preset keys that determine an axis's effective baseline.
///
/// The delivery key is always first. Most axes have no runtime preset
/// inheritance, so [`preset_baseline_keys`] supplies that direct key as the
/// singleton default. Only axes whose live reader falls back through another
/// preset field belong in this override table. Search must use the first key
/// present in the immutable preset snapshot; otherwise it can mistake a live
/// inherited value for the declaration default and invert baseline/treatment.
const PRESET_BASELINE_KEYS: &[(&str, &[&str])] = &[
    (
        "NY_MO_KFSB_K",
        &[
            "bab.branching.kfsb_multi_candidates",
            "bab.branching.candidates",
        ],
    ),
    (
        "NY_MO_KFSB_REDUCE",
        &[
            "bab.branching.kfsb_multi_reduceop",
            "bab.branching.reduceop",
        ],
    ),
];

/// Ordered preset keys that determine the effective baseline for one axis.
///
/// Preset-delivered axes return a non-empty list beginning with their direct
/// [`Deliver::PresetKey`]. Environment-only and unknown names return an empty
/// slice because no preset baseline can govern them.
#[must_use]
pub fn preset_baseline_keys(name: &str) -> &'static [&'static str] {
    if let Some(keys) = PRESET_BASELINE_KEYS
        .iter()
        .find_map(|(candidate, keys)| (*candidate == name).then_some(*keys))
    {
        return keys;
    }

    let Some(axis) = AXES.iter().find(|axis| axis.name == name) else {
        return &[];
    };
    match &axis.deliver {
        Deliver::PresetKey(key) => std::slice::from_ref(key),
        Deliver::EnvOnly => &[],
    }
}

/// Category-name prefixes for mechanisms whose own implementation is scoped
/// to one VNN-COMP family.
///
/// Empty means that category names alone cannot rule the axis out (the reader
/// may still have a model/backend predicate). A non-empty list is a hard
/// applicability contract: measuring the axis on any other category is a
/// full-budget baseline replay, not negative evidence about the mechanism.
const CATEGORY_PREFIXES: &[(&str, &[&str])] = &[
    ("NY_CGAN_RECLAIM", &["cgan"]),
    ("NY_CGAN_POLISH", &["cgan"]),
    ("NY_CGAN_STACKED_BACKWARD", &["cgan"]),
    ("NY_CGAN_STACKED_BUDGET_MB", &["cgan"]),
    (
        "NY_SCORED_WALL_CLAMP",
        &["isomorphic_acasxu", "monotonic_acasxu"],
    ),
    ("NY_NN4SYS_CTX_GROUP", &["nn4sys"]),
    ("NY_NN4SYS_1D_COMPLETE_COVER", &["nn4sys"]),
    ("NY_NN4SYS_1D_COVER_DEPTH", &["nn4sys"]),
    ("NY_NN4SYS_1D_COVER_SLICE", &["nn4sys"]),
    ("NY_SAFENLP_MIP_CYCLE", &["safenlp"]),
    ("NY_BNN_SIGN_SPACE", &["traffic_signs_recognition"]),
    (
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        &["traffic_signs_recognition"],
    ),
    (
        "NY_BNN_SIGN_SPACE_TRUST_REGION",
        &["traffic_signs_recognition"],
    ),
    ("NY_BNN_STE_PGD", &["traffic_signs_recognition"]),
    ("NY_BNN_ATTACK_FRAC", &["traffic_signs_recognition"]),
    ("NY_BNN_ATTACK_CAP", &["traffic_signs_recognition"]),
    ("NY_POSTBAB_RESERVE_SECS", &["acasxu"]),
    ("NY_BAB_RESNET_WIDE", &["cifar100", "tinyimagenet"]),
    ("NY_BAB_RESNET_WIDE_SUBGROUP", &["cifar100", "tinyimagenet"]),
    ("NY_RESNET_ERR_CONCRETIZE", &["cifar100", "tinyimagenet"]),
    (
        "NY_RESNET_ERR_CONCRETIZE_FINE",
        &["cifar100", "tinyimagenet"],
    ),
];

/// Hard VNN-COMP category prefixes for one axis, or an empty slice when the
/// category name cannot decide applicability.
#[must_use]
pub fn category_prefixes(name: &str) -> &'static [&'static str] {
    CATEGORY_PREFIXES
        .iter()
        .find_map(|(candidate, prefixes)| (*candidate == name).then_some(*prefixes))
        .unwrap_or(&[])
}

/// A condition another axis must satisfy for this one to do anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// The named axis must be exactly `"1"`.
    Armed(&'static str),
    /// At least one named axis must be exactly `"1"`.
    ///
    /// This is one OR requirement, not several [`Self::Armed`] edges (which
    /// would be ANDed). The order is policy: dependency repair uses the first
    /// reachable arm when none is already active.
    AnyArmed(&'static [&'static str]),
    /// The named axis must be present and NOT `"0"`.
    NonZero(&'static str),
    /// The named axis must parse to a value strictly greater than the bound.
    GreaterThan(&'static str, u64),
    /// The named axis must be absent or `"0"` — it OWNS the slot when armed.
    NotArmed(&'static str),
    /// The named axis must be absent or differ from this exact effective token.
    ///
    /// This preserves contextual absence while rejecting an explicitly
    /// suppressing sibling arm. It is used when `0` is a kill switch but an
    /// absent master selects a live automatic policy. The sentinel must survive
    /// the named lever's parser, but need not itself be searchable; the named
    /// axis's domain must contain at least one different repair token.
    NotEquals(&'static str, &'static str),
    /// The named axis must be absent from the treatment.
    ///
    /// Unlike a Boolean requirement, neither explicit token can satisfy this:
    /// the child belongs only to the parent's contextual automatic state.
    Absent(&'static str),
    /// The named axis must be present with exactly this domain token.
    ///
    /// Unlike [`Self::NotArmed`], absence does not satisfy this condition. This
    /// matters for default-on selectors: omitting `NY_MARGIN_ROW_BETA_POLYAK`
    /// selects Polyak at runtime, so an ETA experiment must explicitly carry
    /// `POLYAK=0` rather than relying on an absent sample entry.
    Equals(&'static str, &'static str),
}

/// `child` does nothing unless `requires` holds.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub child: &'static str,
    pub requires: Requirement,
    /// The `&&` / `?` / early-return that makes this true, for auditing.
    pub site: &'static str,
}

/// Why a sample was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inert {
    /// The axis that would have had no effect.
    pub axis: String,
    /// The unmet requirement, rendered.
    pub because: String,
}

impl std::fmt::Display for Inert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` is inert: {}", self.axis, self.because)
    }
}

/// UNSAFE axes, excluded permanently. Listed rather than omitted so that a future
/// reader can see they were considered and rejected on purpose.
const UNSAFE_AXES: &[Axis] = &[
    Axis {
        name: crate::decls::kernel_inputs::CONVTRANSPOSE_SOUND_F64_GPU.name,
        domain: Domain::Enum(&["0", "1"]),
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "Existing raw kernel input with different finite and legacy defaults; source/error/deadline/backend authority must remain unchanged. Manual rollback/conformance only; not an automated search dimension",
    },
    Axis {
        name: crate::decls::kernel_inputs::CONV_SKIP_DEAD_F32.name,
        domain: Domain::Enum(&["0", "1"]),
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "Inherited literal-zero rollback over certified f64 recomputation; declaring the input preserves its not-zero predicate and default. No new search or typed preset promotion",
    },
    Axis {
        name: crate::decls::kernel_inputs::FORWARD_LINEAR_MACS_PER_SEC.name,
        domain: Domain::Grid(&[]),
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "Manual positive-u128 cost calibration override; no automatic tuning domain. Both existing readers retain env/probe/cache/fallback precedence and their original parser",
    },
    Axis {
        name: "NY_ROOT_CERTIFIED_CUT",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "Owner-directed default-on authored raw-chain loading and one-query certified root-cut attempt; literal0 rollback. Actual source/deadline/memory/work guards required; no current category performance qualification or automated search permission",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLASSWISE",
        domain: Domain::Enum(&["0", "1"]),
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "legacy classwise research schedule changes proof scheduling and suppresses \
              parallel frontiers; declaration preserves its broader truthy Text predicate. \
              Default off, no typed promotion or automated search qualification",
    },
    Axis {
        name: "NY_SWEEP_CROSSING_TARGETS_ONLY",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact-\"1\" research switch for the comprehensive root sweep's per-chunk row \
              budget (ccb88c16b): drops zero-crossing targets and gives each survivor its own \
              window cursor. Landed declared but UNCATEGORISED, which left \
              every_declared_lever_is_accounted_for red on origin/main. Withheld from search \
              rather than promoted: its own declaration is moat High and says the forfeited \
              stable rows are an UNQUANTIFIED loss, with no armed-vs-unarmed comparison yet. \
              Absence is its declaration default (dark, byte-for-byte today's allocation). \
              Promote it to a typed preset key only once that comparison exists",
    },
    Axis {
        name: "NY_NO_CELL_ENUM",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact-\"1\" kill switch that WITHDRAWS the cell-enumeration authority at all \
              three of its sites. Excluded rather than searched because one arm strictly \
              dominates: absence leaves the authority engaged, and arming it can only remove \
              a verdict path — try_cell_enumeration returns None so the ordinary pipeline \
              runs unchanged, and f64_point_definitely_safe returns None, i.e. NO CLAIM. It \
              can lose a proof cell enumeration would have produced and cannot mint one, so \
              there is nothing for a search to find. Declared 2026-09-02 (#no-cell-enum) to \
              retire the raw-read debt of its three byte-identical direct-literal readers; \
              it stays a manual research/kill switch",
    },
    Axis {
        name: "NY_PATCHES_GPU",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "legacy truthy-grammar master for the Conv2d Patches compose engine-GEMM seam whose \
              ABSENCE defers to fast_f32_gemm::is_installed() — a contextual default no emitted \
              token can restore, so a search that touches it cannot return to baseline. Declared \
              2026-09-01 (#patches-engine-wiring) to retire its literal-frontier debt; the seam's \
              scored-route levers are the typed bab.patches_engine_refusal_containment / \
              bab.fast_f32_prewarm pair, and this stays a manual research/kill switch",
    },
    Axis {
        name: "NY_NO_FORWARD_LINEAR_REF",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact-\"1\" kill switch that withdraws the default-ON conv-DAG forward-linear \
              reference source from every consumer at once (Step-1 intermediates, alpha \
              references, the spec-margin rebuild, the attack-phase cache warmer). Arming it \
              is a whole-route rollback onto incomparable collectors, not a treatment a \
              search should sample. Declared 2026-09-30 (#fl-alpha-composition) to retire \
              its literal-frontier debt; it stays a manual research/kill switch",
    },
    Axis {
        name: "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact-\"1\" kill switch for the default-ON ConvTranspose2d/BatchNorm image \
              forward-linear surface, whose engaged arm is the cGAN root enclosure recorded \
              ~1000x tighter than the collectors it replaced. Arming it only rolls that \
              surface back. Declared 2026-09-30 (#fl-alpha-composition) to retire its \
              literal-frontier debt; it stays a manual research/kill switch",
    },
    Axis {
        name: "NY_FORWARD_LINEAR_SEQ_CONV_REF",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "default-dark #cora-fwdlin-ref research gate admitting plain-Conv2d SEQUENTIAL \
              chains to the forward-linear alpha reference attempt. It has no typed preset \
              key and no armed-vs-unarmed comparison, so it is withheld from search rather \
              than counted as undelivered verdict-affecting debt. Declared 2026-09-30 \
              (#fl-alpha-composition) to retire its literal-frontier debt",
    },
    Axis {
        name: "NY_FORWARD_LINEAR_F32",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("model.forward_linear_f32_seam"),
        why: "legacy env route of the #fl-f32-seam-key f32 value-GEMM seam, which REPLACES \
              the composition's certified f64 GEMM error bound with the looser Higham f32 \
              one: sound but PRECISION-NEGATIVE (0 gains, 1 regression on cifar100 per \
              docs/CIFAR100_BOUND_PARITY_ORACLE.md) with a standing do-not-arm record. \
              The CLI layers this variable over the key exactly as the flight receipt does \
              (resolve_forward_linear_f32_seam), so a sampled env value WOULD override a \
              preset's own choice, one more reason it is never searched. \
              Arm it per preset on measured evidence through model.forward_linear_f32_seam, \
              never by search. Declared 2026-09-30 (#fl-alpha-composition) to retire its \
              literal-frontier debt",
    },
    Axis {
        name: "NY_PATCHES_DEADLINE_FLAT_BIAS",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "finite-deadline Patches kernel experiment whose checked flat-row \
              reduction changes target completion jointly with row shape and deadline. \
              It remains default-dark and manually measurable until typed delivery and \
              an engagement receipt distinguish real flat-row work from inert samples",
    },
    Axis {
        name: "NY_CROWN_HONEST_PROVENANCE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "default-on quality-publication guard. Exact 0 restores unconditional Crown/tight \
              provenance for an almost-entirely non-finite CROWN result; a search must never \
              improve its score by disabling provenance honesty",
    },
    Axis {
        name: "NY_CROWN_OBJ_CHUNK",
        domain: Domain::Grid(&[0, 32, 64, 256, 512, 1024]),
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "coupled objective-frontier resource knob: width changes prefix replay, \
              deadline polling, retained results, and memory admission. The production \
              driver already derives a deadline-safe width; automated search must wait \
              for typed delivery and a closed memory/time receipt proving the requested \
              width actually ran",
    },
    Axis {
        name: "NY_NO_CHUNK_ABORT",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact 1 disables projected early abort and can burn the whole per-node \
              deadline after measured throughput already shows the target cannot finish. \
              Its effect is coupled to chunk width, growth, salvage, and budget, so it is \
              a controlled rollback rather than an independent search coordinate",
    },
    Axis {
        name: "NY_NO_CHUNK_GROW",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact 1 restores fixed-width full-prefix replay. It can multiply work and \
              changes deadline completion jointly with the initial width and abort policy; \
              automated search is withheld until the coupled driver has an execution \
              receipt rather than treating this rollback as an isolated Boolean",
    },
    Axis {
        name: "NY_NO_CHUNK_WAVE_PAR",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "exact 0 disables the production aggregate-memory guard and re-enables \
              objective-chunk waves whose workers have no shared byte reservation or \
              complete per-worker receipt. On the 28,800-row cGAN target the legacy \
              row quotient can admit 32 independently full-budget backward frontiers. \
              This arm is retained for controlled measurement only and must not be \
              optimized until execution owns a closed aggregate-memory contract",
    },
    Axis {
        name: "NY_CUDA_GEMM_TRANSPORT",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "names which CUDA GEMM transport the engine uses. The right answer \
              is a property of the MACHINE — driver, cuBLAS presence, arch — and \
              compute_backend already derives it. A search that moved this would \
              be measuring the host it happens to run on and reporting it as a \
              tuning result, and on a host missing the chosen transport every \
              arm degrades identically, which reads as `no effect`",
    },
    Axis {
        name: "NY_EFT_ERR",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "replaces the a-priori gamma_n certified-error bound with an \
              EFT-MEASURED one, i.e. swaps a proof for an observation, and does it \
              through raw ungoverned reads outside the registry chokepoint",
    },
    Axis {
        name: "NY_CUDA_DISCRETE_MODE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "ny-cuda applies its own stricter parser and FAILS ENGINE \
              CONSTRUCTION on any token that is not exactly 0 or 1; a search that \
              emits anything else does not measure a treatment, it kills the run",
    },
    Axis {
        name: "NY_CONST_MATMUL_INTERVAL",
        domain: Domain::Bool,
        class: Class::Unsafe,
        // #interval-carrier-quarantine: narrowed from
        // Deliver::PresetKey("solver.const_matmul_interval") to EnvOnly on
        // 2026-08-24. A preset key is reachable from a SCORED run; an
        // environment variable is not. This lever is Class::Unsafe /
        // MoatRisk::High on a BOUND path, and while it was preset-armable it
        // was demonstrably capable of certifying a false property -- the
        // reachable-path defect it exposed (97909a8e1) published
        // [-1.88e-32, +1.88e-32] as Sound on a four-node graph whose true
        // value is 0.7. Nothing that can emit a false proof should be one
        // config line away from the competition harness.
        deliver: Deliver::EnvOnly,
        why: "arms an interval-valued carrier for frozen constants a bit-exact \
              fold cannot represent — a BOUND path, on a family where ny has \
              already published false `unsat` verdicts once. Its containment \
              argument is per-op and pinned against exact rational ground \
              truth, but its live failure mode is a propagation path that has \
              not been taught the interval, which is indistinguishable in its \
              output from the unsound point fold it exists to avoid. That is \
              not a treatment an automated sweep should be allowed to arm on \
              its own: every armed run needs the differential containment \
              check and a cross-check of every `unsat` it produces",
    },
    Axis {
        name: "NY_JOINT_BILINEAR_QUAD",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("solver.joint_bilinear_quad"),
        why: "arms a JOINT relaxation of a fully-reduced elementwise product — a \
              BOUND path, on the same family where ny has already published \
              false `unsat` verdicts once. Its construction is shrink-only \
              (the result is published only through an intersection against \
              the caller's own CROWN bounds) and its curvature term charges \
              back the exact eigen-reconstruction residual, so no accuracy \
              assumption is load-bearing. But a sweep that arms it on its own \
              would be scoring proofs on a family that has no independent \
              checker on this host: every `unsat` it enables needs an \
              independently constructed reference on the same domain first",
    },
    Axis {
        name: "NY_MARGIN_ROW_EPOCH_PARENT_TOPK",
        domain: Domain::Grid(&[8, 16, 32, 64]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.epoch_parent_topk"),
        why: "arms a new authoritative parent-frame epoch BOUND path whose \
              idx_1298 cost/closure hypothesis is not yet measured. The typed \
              route is deliberately deliverable, but a valid treatment must \
              hand-author the coupled `margin_row.epoch_depth`, \
              `epoch_attempts`, and `epoch_parent_topk` keys; top-K alone is \
              inert. Class::Unsafe therefore keeps this axis out of automated \
              search until a sealed A/B establishes engagement, independent \
              soundness receipts, wall cost, and a useful cadence",
    },
    // The reserve allocator is one coupled policy, not four independent
    // coordinates. These fields remain preset-deliverable so the scored
    // receipt can freeze the exact schedule that ran, but automated search is
    // intentionally withheld until a row-local terminal can prove that each
    // coordinate changed an actual verifier/lane allocation.
    Axis {
        name: "NY_MARGIN_ROW_RESERVE_SECS",
        domain: Domain::Grid(&[0, 15, 30, 45, 60, 82]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.reserve_secs"),
        why: "coupled lane/verifier allocation: the same configured seconds can be held, \
              adaptively released, capped, or structurally refused. Historical +23/+67 \
              bookkeeping does not seal this reserve route, and searching it beside the \
              concurrent lane can double-tax the verifier",
    },
    Axis {
        name: "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.adaptive_reserve"),
        why: "changes both verifier allocation and sparse-root-CROWN authority. Preserved \
              adaptive reserve aliases the fixed route, released reserve can coexist safely \
              with concurrency, and held reserve plus concurrency is an unmeasured double tax",
    },
    Axis {
        name: "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        domain: Domain::Enum(&["0.15", "0.25", "0.35", "0.40", "0.50"]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.reserve_max_frac"),
        why: "skipped whenever reserve is zero or adaptively released and inert whenever its \
              floored ceiling does not bind. No shipped preset reaches a binding case and no \
              sealed verdict A/B exists",
    },
    Axis {
        name: "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
        domain: Domain::Enum(&["0.20", "0.25", "0.40", "0.50"]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.release_frac"),
        why: "meaningful only with positive structurally admitted reserve and adaptive policy, \
              and only when treatment/control straddle the exact budget threshold. The widened \
              structural route still lacks its requested GB10 A/B",
    },
    // Resource ownership and terminal publication are part of the same frozen
    // scheduler policy, but they are not independent optimization coordinates.
    // Keep their typed routes reproducible while withholding automated search
    // until a closed resource receipt proves the allocation that actually ran.
    Axis {
        name: "NY_MARGIN_ROW_EARLY_EXIT",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.early_exit"),
        why: "default-on production publication policy, not a performance coordinate. Its off \
              arm only waits after an independently certified lane proof and currently has no \
              closed counterfactual terminal proving what would have published early; an \
              automated search must not reward process-exit or receipt-bookkeeping artifacts",
    },
    Axis {
        name: "NY_VERIFIER_THREADS",
        domain: Domain::Grid(&[0, 1, 2, 4, 8, 16]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.verifier_threads"),
        why: "process-global Rayon initialization is host- and order-dependent: a configured cap \
              can be refused after earlier Rayon use, so configuration alone is not an applied \
              resource treatment. Search waits for a receipt binding host cores, requested cap, \
              actual global width and the engaged/refused route",
    },
    Axis {
        name: "NY_MARGIN_ROW_CONCURRENT_THREADS",
        domain: Domain::Grid(&[0, 1, 2, 4, 8, 16]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("margin_row.concurrent_threads"),
        why: "coupled to verifier allocation and host cores: Unset derives only from observed \
              host parallelism minus actual global width, explicit zero forces the shared \
              global pool, and a positive request requires an exact private pool. Missing or \
              exhausted remainder and private-pool build failure refuse the lane; search waits \
              for a closed receipt proving the effective width and private/global/refusal route",
    },
    Axis {
        name: "NY_BRANCH_KFSB_CHILDSIM",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "legacy compatibility switch with COMPOUND behavior, not an alias \
              for NY_MO_KFSB: only while the canonical environment master is \
              absent it both arms the wave-batched child-simulation selector \
              and, when NY_MO_KFSB_K is absent, changes the candidate count to \
              two. Searching it as a second master spelling would silently \
              confound lane admission with candidate-budget tuning; automated \
              search must use the canonical bab.branching.kfsb_multi delivery \
              together with the separately governed NY_MO_KFSB_K axis",
    },
    Axis {
        name: "NY_STRIP_TERMINAL_SOFTMAX",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "rewrites the network before verification, i.e. changes the property \
              being proved; a 'win' here is not a win",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_LAYERS",
        domain: Domain::Enum(&["all"]),
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_layers"),
        why: "free-form graph-local ReLU names are deliverable for reproducible instance research but do not define a portable global treatment. The sole listed token is the automatic-selection control; automated search must not synthesize node names or graduate an unmatched selector as negative evidence",
    },
    // RETIRED 2026-09-29 (ENGINE_PROGRAM R1 kill line + R2). Both were promoted
    // `margin_row.*` keys; the typed keys are now refused at preset validation
    // and the lanes refuse an environment request, so neither is a treatment a
    // search may set.
    Axis {
        // #mr-gpu-eval: the certified pass's conv backward products on cuBLAS.
        name: "NY_MARGIN_ROW_GPU_CONV",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "RETIRED by measurement: per-op conv offload of the certified margin-row pass \
              (38,552 convs at 3.74 ms each) turned cifar100 c2050 unsat 88.5 s into a timeout \
              and c1590 unsat 78 s into unknown on the yaml route with zero NY_* (W7B), which \
              is program rule R1's kill line; its device products also enter certified values, \
              which R2 forbids until the M9 preconditions hold. The typed key \
              margin_row.gpu_conv is refused at preset validation and the lane never arms the \
              route (`gpu_conv::armed` is false; a request prints `[mr-gpu-conv] refused \
              reason=retired(...)`). Kept declared so an old environment gets a loud refusal",
    },
    Axis {
        // #bi-gpu: backward-interm / joint-refresh prefix pairs on cuBLAS.
        name: "NY_MARGIN_ROW_BI_GPU",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "RETIRED under R2 (the device never certifies): its cuBLAS Dgemm products are the \
              CERTIFIED intermediate boxes, covered only by a-priori charges, with no \
              `-fmad=false` build, SASS-only authority, live-state KAT or A10G moat (M9 \
              preconditions), and its gated tests ran emulated engines only; it is also the \
              per-op offload shape R1 retired with margin_row.gpu_conv. The typed key \
              margin_row.bi_gpu is refused at preset validation and an armed phase is refused \
              at runtime (`[bi-gpu] refused reason=r2_device_never_certifies`, the dark CPU \
              pairs bit for bit). Kept declared so an old environment gets a loud refusal",
    },
    Axis {
        // #router-r2 (ROUTER_PROGRAM R2). The seven acting-rule ablation
        // switches, appended LAST so no existing literal site's structural
        // anchor moves.
        name: "NY_ROUTER_RULE_ADMISSION",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.admission"),
        why: "ablation switch of the acting router rule CR-A: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-A \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_CONTRACTS",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.contracts"),
        why: "ablation switch of the acting router rule CR-C: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-C \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_CAP_AUTHORITY",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.cap_authority"),
        why: "ablation switch of the acting router rule CR-1: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-1 \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_TERMINAL",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.terminal"),
        why: "ablation switch of the acting router rule CR-T: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-T \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_HAND_ON",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.hand_on"),
        why: "ablation switch of the acting router rule CR-H: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-H \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_PUBLISH_GRACE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.publish_grace"),
        why: "ablation switch of the acting router rule CR-R: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-R \
              acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_DERIVED_CAPS",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.derived_caps"),
        why: "ablation switch of the acting router rule CR-K: it can move a verdict only \
              under router.mode: on, which search never selects (NY_ROUTER_MODE's domain is \
              off|shadow), and a rule joins or leaves the shipped acting set by a compiled \
              move after its moat, never by a searched value. Typed key so a measurement \
              ablates it with zero NY_*; engagement evidence is `[router] act rule=CR-K \
              acts=`",
    },
    // #router-r3 (ROUTER_PROGRAM R3). The four O-STOP optimizer-rule ablation
    // switches, appended LAST so no existing literal site's structural anchor
    // moves.
    Axis {
        name: "NY_OPTIMIZER_RULE_GOVERNOR",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.governor"),
        why: "ablation switch of the acting optimizer rule rule.governor: it can move a verdict only \
              under optimizer.mode: on, which search never selects (the optimizer mode's \
              searched domain is off|shadow), and a rule joins or leaves the shipped acting \
              set by a compiled move after its moat, never by a searched value. Typed key so \
              a measurement ablates it with zero environment; engagement evidence is \
              `[optimizer] act rule=rule.governor acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_IA_YIELD",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.ia_yield"),
        why: "ablation switch of the acting optimizer rule rule.ia_yield: it can move a verdict only \
              under optimizer.mode: on, which search never selects (the optimizer mode's \
              searched domain is off|shadow), and a rule joins or leaves the shipped acting \
              set by a compiled move after its moat, never by a searched value. Typed key so \
              a measurement ablates it with zero environment; engagement evidence is \
              `[optimizer] act rule=rule.ia_yield acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_ATTACK_DEFER",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.attack_defer"),
        why: "ablation switch of the acting optimizer rule rule.attack_defer: it can move a verdict only \
              under optimizer.mode: on, which search never selects (the optimizer mode's \
              searched domain is off|shadow), and a rule joins or leaves the shipped acting \
              set by a compiled move after its moat, never by a searched value. Typed key so \
              a measurement ablates it with zero environment; engagement evidence is \
              `[optimizer] act rule=rule.attack_defer acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_ROOT_ADMISSION",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.root_admission"),
        why: "ablation switch of the acting optimizer rule rule.root_admission: it can move a verdict only \
              under optimizer.mode: on, which search never selects (the optimizer mode's \
              searched domain is off|shadow), and a rule joins or leaves the shipped acting \
              set by a compiled move after its moat, never by a searched value. Typed key so \
              a measurement ablates it with zero environment; engagement evidence is \
              `[optimizer] act rule=rule.root_admission acts=`",
    },
    // #router-r4 (ROUTER_PROGRAM R4). Acting set 3's two ablation switches,
    // appended LAST so no existing literal site's structural anchor moves.
    Axis {
        name: "NY_ROUTER_RULE_LEASE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.lease"),
        why: "ablation switch of the acting router rule CR-L (the memory lease): it can move a \
              verdict only under router.mode: on, which search never selects (the router \
              mode's searched domain is off|shadow), and a rule joins or leaves the shipped \
              acting set by a compiled move after its moat, never by a searched value. Typed \
              key so a measurement ablates it with zero environment; engagement evidence is \
              `[router] act rule=CR-L acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_FAR_HAND_ON",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.far_hand_on"),
        why: "ablation switch of the acting router rule CR-F (the FAR hand-on): it can move a \
              verdict only under router.mode: on, which search never selects (the router \
              mode's searched domain is off|shadow), and a rule joins or leaves the shipped \
              acting set by a compiled move after its moat, never by a searched value. Typed \
              key so a measurement ablates it with zero environment; engagement evidence is \
              `[router] act rule=CR-F acts=`",
    },
    // #router-r5 (ROUTER_PROGRAM R5). Acting set 4's two ablation switches,
    // appended LAST so no existing literal site's structural anchor moves.
    Axis {
        name: "NY_ROUTER_RULE_ROUTE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.route"),
        why: "ablation switch of the acting router rule CR-RS (capabilities routed only at \
              SR.seal): it can move a verdict only under router.mode: on, which search never \
              selects (the router mode's searched domain is off|shadow), and a rule joins or \
              leaves the shipped acting set by a compiled move after its moat, never by a \
              searched value. Typed key so a measurement ablates it with zero environment; \
              engagement evidence is `[router] act rule=CR-RS acts=`",
    },
    Axis {
        name: "NY_ROUTER_RULE_GAP_CONTRACT",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("router.rules.gap_contract"),
        why: "ablation switch of the acting router rule CR-G (gap-order contracts inside \
              multi-objective lanes): it can move a verdict only under router.mode: on, which \
              search never selects (the router mode's searched domain is off|shadow), and a \
              rule joins or leaves the shipped acting set by a compiled move after its moat, \
              never by a searched value. Typed key so a measurement ablates it with zero \
              environment; engagement evidence is `[router] act rule=CR-G acts=`",
    },
    // #router-r6 (ROUTER_PROGRAM R6). The four O-TRIAL optimizer-rule
    // ablation switches, appended LAST so no existing literal site's
    // structural anchor moves.
    Axis {
        name: "NY_OPTIMIZER_RULE_TRIAL_WIDTH",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.trial_width"),
        why: "ablation switch of the acting optimizer trial rule rule.trial_width: it can move a \
              verdict only under optimizer.mode: on, which search never selects (the \
              optimizer mode's searched domain is off|shadow), and a rule joins or leaves \
              the shipped acting set by a compiled move after its moat, never by a searched \
              value. Typed key so a measurement ablates it with zero environment; \
              engagement evidence is `[optimizer] act rule=rule.trial_width acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_TRIAL_LR_BACKOFF",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.trial_lr_backoff"),
        why: "ablation switch of the acting optimizer trial rule rule.trial_lr_backoff: it can move a \
              verdict only under optimizer.mode: on, which search never selects (the \
              optimizer mode's searched domain is off|shadow), and a rule joins or leaves \
              the shipped acting set by a compiled move after its moat, never by a searched \
              value. Typed key so a measurement ablates it with zero environment; \
              engagement evidence is `[optimizer] act rule=rule.trial_lr_backoff acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_TRIAL_SPLIT_RESTART",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.trial_split_restart"),
        why: "ablation switch of the acting optimizer trial rule rule.trial_split_restart: it can move a \
              verdict only under optimizer.mode: on, which search never selects (the \
              optimizer mode's searched domain is off|shadow), and a rule joins or leaves \
              the shipped acting set by a compiled move after its moat, never by a searched \
              value. Typed key so a measurement ablates it with zero environment; \
              engagement evidence is `[optimizer] act rule=rule.trial_split_restart acts=`",
    },
    Axis {
        name: "NY_OPTIMIZER_RULE_TRIAL_ATTACK_HALVING",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.rules.trial_attack_halving"),
        why: "ablation switch of the acting optimizer trial rule rule.trial_attack_halving: it can move a \
              verdict only under optimizer.mode: on, which search never selects (the \
              optimizer mode's searched domain is off|shadow), and a rule joins or leaves \
              the shipped acting set by a compiled move after its moat, never by a searched \
              value. Typed key so a measurement ablates it with zero environment; \
              engagement evidence is `[optimizer] act rule=rule.trial_attack_halving acts=`",
    },
    // #router-r7 (ROUTER_PROGRAM R7). The offline-prior switch, appended LAST
    // so no existing literal site's structural anchor moves.
    Axis {
        name: "NY_OPTIMIZER_PRIORS",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::PresetKey("optimizer.priors"),
        why: "lets the router read the compiled offline cost-model prior for planning \
              (first rungs, CR-C contracts; CR-A keeps an unfit priced slot on today's \
              deadline under router.mode: on). Withheld from search like the rule switches: \
              a prior joins the shipped plan by a compiled default move after its moat and \
              its held-out gates, never by a searched value. Typed key so a measurement arms \
              it with zero environment; engagement evidence is `[router] priors v=`",
    },
];

/// Telemetry axes. Instruments, never treatment arms.
///
/// They cannot touch a bound, but they are NOT free: each adds clocks, formatting
/// and stderr to a deadline-sensitive run. `NY_ITER0_PARITY_TRACE` emits one line
/// per node per backward walk and must never be on in a timing arm.
const INSTRUMENT_ONLY: &[&str] = &[
    "NY_PHASE_TELEMETRY",
    "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
    "NY_BETA_GPU_PROBE",
    "NY_SEG_PROBE",
    "NY_WIDE_PROBE",
    "NY_MARGIN_ROW_PROFILE",
    "NY_MARGIN_ROW_GPU_PROF",
    // #bab-diagnostics. Both are print-only and cannot move a bound, but both
    // add clocks and stderr to deadline-sensitive runs, so they are instruments
    // and must never appear in a timing arm. NOTE the parsers DISAGREE:
    // NY_ACASXU_PROF is PRESENCE-armed (so "0" arms it), NY_BAB_CHILD_TRACE is
    // exact-"1" — which is exactly why they are declared rather than guessed.
    "NY_ACASXU_PROF",
    "NY_BAB_CHILD_TRACE",
    // #epoch-bab's trace, and a THIRD parser in the same family: it is
    // PRESENCE-armed through `var(..).is_ok()`, so "0" and "" arm it like
    // NY_ACASXU_PROF — but unlike a `var_os` presence gate, a non-UTF-8 value
    // leaves it DARK. Print-only: one line per epoch attempt and one per
    // outcome. It is the instrument that made the NY_EPOCH_DEPTH engagement
    // observable at all (a second `tree_loop entry` plus a stop reason naming
    // the epoch), which is exactly why it must never be on in a timing arm.
    "NY_EPOCH_DEBUG",
    // Configures an instrument rather than being one: it selects WHICH call
    // sites the (already excluded) M2 shadow observes. It cannot arm the
    // shadow, cannot touch verdict authority, and cannot move a bound —
    // shadow results are compared and DISCARDED — so like NY_MARGIN_ROW_GPU_PROF
    // above it is never a treatment arm; its whole effect is on the device
    // wall clock the shadow lever already spends, and its purpose is to
    // spend LESS of it (one site per measurement cycle instead of four).
    "NY_MARGIN_ROW_GPU_SHADOW_SITES",
    // RECLASSIFIED 2026-08-23 (#lever-reachability triage). The parenthesis in
    // the note above — "the (already excluded) M2 shadow" — was FALSE: the
    // shadow itself sat in `AXES` as `VerdictAffecting`, so the registry said
    // the configurator was an instrument and the thing it configures was a
    // treatment. Only one of those can be right.
    //
    // The shadow is the instrument. It computes the certified GPU backward,
    // COMPARES it against the CPU pass, and DISCARDS it: the bounds are the
    // unchanged CPU pass BY CONSTRUCTION, which is the M2 parity protocol, not
    // an implementation detail that could drift. Its own axis note already said
    // so and then added "never arm on a scored run", which is the definition of
    // an instrument in this file. What it does cost is device time inside the
    // verdict window — exactly the "NOT free" hazard this list exists to name.
    //
    // Promoting it instead would have been dishonest in the other direction: a
    // typed preset key is a delivery route, and there is nothing here to
    // deliver.
    "NY_MARGIN_ROW_GPU_SHADOW",
    // RECLASSIFIED 2026-08-23 (#lever-reachability triage), all six.
    //
    // The exact-star lane is a MEASUREMENT probe with no verdict authority, and
    // the source says so at every level: `run_dark_star_probe` returns `()`;
    // its module header states "nothing here can EMIT or ALTER a competition
    // verdict"; `emittable_verdict_today` maps EVERY search outcome to
    // `"unknown"` and is documented as the single place a future authority
    // would have to change visibly; the two authority obligations it would need
    // (`#star-witness` SAT extraction, the UNSAT roundoff enclosure over the
    // split tree) are undischarged. The whole result reaches the operator as
    // one `NY_STAR_DARK_V1` stderr line.
    //
    // So these six cannot move a published bound, which is precisely this
    // list's admission rule — and they are the canonical case of its second
    // sentence: the probe spends its budget ON THE CALLER'S THREAD, and an
    // `unsat` row has been MEASURED turning into a `timeout` with it armed.
    // Instruments are not free; that is why they are excluded from search
    // rather than merely annotated.
    //
    // The honest alternative was `UNSAFE_AXES`, and it was rejected: `Unsafe`
    // is for levers that change WHAT IS PROVED, replace a certified error
    // bound, or kill engine construction. A probe that can only cost wall clock
    // is an instrument. If `star_verify` ever discharges its two obligations
    // and `emittable_verdict_today` starts returning a real verdict, these
    // belong back in `AXES` with a typed preset key — and that function is
    // where the change would be visible.
    "NY_STAR_DARK_SECONDS",
    "NY_STAR_DARK_MAX_STARS",
    "NY_STAR_DARK_MAX_DEPTH",
    "NY_STAR_DARK_DUAL_ITERS",
    "NY_STAR_DARK_INPUT_SPLIT",
    "NY_STAR_DARK_EXACT_BELOW",
    "NY_INPUT_SPLIT_PROBE",
    "NY_ENVELOPE_RESCALE_PROBE",
    "NY_ENVELOPE_XSTAR_PROBE",
    "NY_ITER0_PARITY_TRACE",
    "NY_PATCHES_CARRIER_TRACE",
    "NY_GPU_MEM_TRACE",
    "NY_MIP_TRACE",
    "NY_CONV_PATCHES_DEBUG",
    "NY_BNN_SIGN_SPACE_TRACE",
    "NY_DUMP_NODE_BOUNDS",
    // #graph-free-candidates (#smart-turn-x86-harvest), 2026-09-26. A side
    // file, not a treatment: it records the candidates the graph-free lane
    // offers to its UNCHANGED acceptance gates (and a bounded sample of the
    // surrogate incumbents) for offline replay on a checker-parity x86 host,
    // and nothing it writes is read back by ny. It cannot move a bound or a
    // verdict — this list's admission rule — but it is not free: each record
    // is ~6.7 MB of encoding and I/O inside the lane's deadline, which is
    // exactly the "instruments are NOT free" hazard, so it must never ride in
    // a timing arm. It has a typed preset key (`attack.graph_free_candidate_sidefile`)
    // only because run_instance.sh exports no NY_*, so a zero-NY_* diagnostic
    // run needs a yaml route; the flight receipt projects that key through
    // `graph_free_sidefile::preset_config_value`, not through an axis.
    "NY_GRAPH_FREE_CANDIDATE_SIDEFILE",
];

/// Test-only axes, excluded: they exist to drive fixtures, not the scored path.
const TEST_ONLY: &[&str] = &[
    "NY_HULL_RESOURCE_ORIGINAL_REQUEST",
    "NY_GROUPED_ORIGINAL_PROBE_REQUEST",
    "NY_VIT_BENCH",
    "NY_FULL_MEASUREMENTS",
    "NY_BENCH_ROOT",
    "NY_BENCH_ROOT_2026",
    "NY_FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL",
];

/// The searchable axes.
const AXES: &[Axis] = &[
    // --- root intermediate-tightening family (mutually exclusive, see EDGES) ---
    Axis {
        name: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm"),
        why: "atomic all-target sound-GPU root sweep; shrink-only, but it changes \
              published intermediate bounds",
    },
    Axis {
        name: "NY_INTERM_ROW_CHUNKS",
        domain: Domain::Grid(&[1, 4, 16, 64]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm_chunks"),
        why: "disjoint row windows accumulated by the comprehensive sweep; 1 is \
              byte-identical to the historical single sweep",
    },
    Axis {
        name: "NY_ROOT_PHASE_RESIDENT_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_phase_resident_crown"),
        why: "OWNS the root slot when armed, suppressing the comprehensive and \
              wide-demanded phases entirely",
    },
    Axis {
        name: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_wide_demanded_interm_crown"),
        why: "one-target wide demanded intermediate CROWN; reachable only when \
              both earlier root phases are unarmed",
    },
    Axis {
        name: "NY_ROOT_CPU_PARALLEL_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_cpu_parallel_interm_crown"),
        why: "replaces the dense-head tightener outright rather than adding to it",
    },
    Axis {
        name: "NY_ROOT_COMP_GPU_INTERM_ROWS",
        domain: Domain::Grid(&[16, 32, 64, 128]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm_rows"),
        why: "per-target row ceiling for the comprehensive sweep; the device class \
              policy may cap it below the requested value",
    },
    Axis {
        name: "NY_ROOT_COMP_GPU_INTERM_SECS",
        domain: Domain::Grid(&[10, 20, 40]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm_secs"),
        why: "local authority slice for the comprehensive sweep; capped in turn by \
              half the remaining global budget, so large values saturate",
    },
    // --- margin-row GPU seam ---
    Axis {
        name: "NY_MARGIN_ROW_GPU",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.gpu"),
        why: "routes the margin-row lane through the GPU seam; LATCHED in a \
              OnceLock, so it can only be varied across processes",
    },
    Axis {
        name: "NY_MARGIN_ROW_GPU_BATCH",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.gpu_batch"),
        why: "batches the margin-row GPU seam; inert unless the seam itself is on",
    },
    // --- margin-row lane allocation ---
    Axis {
        name: "NY_MARGIN_ROW_CONCURRENT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.concurrent"),
        why: "starts the certified CPU margin-row lane beside the GPU verifier and is the \
              concurrency coordinate of the retained TinyImageNet recipe. Its historical \
              conversion is confounded with root-f32, so search exists to collect an isolated \
              result rather than inherit that delta. Credit requires a receipt-owned frozen \
              schedule, a real stats-owning worker overlapping the verifier, and zero held \
              margin-row reserve; requested-but-suppressed and double-tax rows are void",
    },
    // --- margin-row tree throughput (measured defaults, no YAML arming) ---
    Axis {
        name: "NY_MARGIN_ROW_LRU",
        domain: Domain::Grid(&[16, 32, 64, 128]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.lru"),
        why: "deterministic y-pack cache capacity. Hits return exactly the pack a miss \
              rebuilds, so completed bound values stay fixed; memory/recomputation and therefore \
              deadline reachability move. 64 is the measured joint-tree shipped default (26% -> \
              86% hit rate and 2.49x expansion throughput). The later default-dark classwise \
              schedule consumes nonzero values but floors legacy env 0 to its mandatory root \
              entry; classwise-contaminated evidence does not extend the retained provenance",
    },
    Axis {
        name: "NY_MARGIN_ROW_PARALLEL",
        // `1` is deliberately absent: in the legacy grammar it means ALL
        // available workers, not a width of one. Omission is that all-worker
        // control; `0` is the explicit serial treatment.
        domain: Domain::Enum(&["0", "2", "4", "8", "16"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.parallel"),
        why: "parallel best-first frontier width on the established joint tree. Certified \
              per-domain bounds are bit-identical, but exploration order/deadline granularity \
              move. Omission is the measured all-worker shipped control; explicit 0 is serial and \
              2/4/8/16 are canonical pinned widths. The later default-dark classwise experiment \
              forcibly uses frontier 1 for exact global expansion accounting, so an ambient \
              classwise run makes this treatment inert/contaminated and must be voided",
    },
    Axis {
        name: "NY_MARGIN_ROW_K_HEAD",
        domain: Domain::Grid(&[2, 4, 8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.k_head"),
        why: "head-neuron shortlist width for the exact branch scorer. Every \
              candidate bound is certified, but the width changes which domains \
              enter the tree and which proof is reached before the deadline; 8 is \
              the declaration control and CIFAR100 currently ships the measured 16",
    },
    Axis {
        name: "NY_MARGIN_ROW_K_TRUNK",
        domain: Domain::Grid(&[2, 4, 8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.k_trunk"),
        why: "trunk-neuron shortlist width for the same exact branch scorer. It \
              selects a distinct candidate population and therefore changes the \
              certified tree trajectory and deadline reachability; 8 is the \
              declaration control and CIFAR100 currently ships the measured 16",
    },
    // --- deadline-carrying kernel selection (value-neutral) ---
    Axis {
        name: "NY_CONV_IBP_BLOCKED",
        domain: Domain::Enum(&["0", "seq", "1"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.conv_ibp_blocked"),
        why: "picks the KERNEL for the finite-deadline certified f64 Conv2d IBP \
              forward, not its arithmetic: all three arms fold every output over \
              the same live taps in the same order and publish through the same \
              gamma_{K+2} certificate and directed casts, so the box is \
              byte-identical and only wall clock and poll granularity move \
              (MoatRisk::None). The shipped blocked arm remains the default; the \
              typed route now also delivers `0` (the scalar oracle) and `seq` \
              (single-threaded blocked), so scored measurements can separate \
              blocking from threads and fall back byte-for-byte",
    },
    // --- wide BaB lane ---
    Axis {
        name: "NY_BAB_RESNET_WIDE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.bab_resnet_wide"),
        why: "domain-stacked wide resnet BaB lane",
    },
    Axis {
        name: "NY_BAB_RESNET_WIDE_SUBGROUP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.bab_resnet_wide_subgroup"),
        why: "subgroup path within the wide lane; inert unless the wide lane is on",
    },
    // --- alpha ---
    Axis {
        name: "NY_ALPHA_ENVELOPE_GRAD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // Shipped as a typed preset key 2026-08-18. Was EnvOnly, which meant a
        // search could never propose it for a scored run however well it scored.
        deliver: Deliver::PresetKey("bab.alpha_crown.alpha_envelope_grad"),
        why: "envelope gradient for the alpha ascent; changes which alpha is \
              reached, hence the bound. MEASURED to move the root census where \
              every other alpha lever left it bit-identical",
    },
    Axis {
        name: "NY_ALPHA_ZERO_YIELD_FRAC",
        // `LeverKind::F64Open { min: 0.0, max: 0.9 }` is an OPEN interval, so
        // tokens must parse as f64 STRICTLY inside it. This axis previously
        // offered the integer grid [0, 25, 50]; every one of those is rejected
        // by the parser and silently resolves to the declaration default, so a
        // search would have spent a full instance budget measuring the baseline
        // and labelled it a treatment — the exact failure this module exists to
        // prevent. Pinned by `every_domain_token_survives_the_real_parser`.
        domain: Domain::Enum(&["0.1", "0.25", "0.5"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.alpha_crown.alpha_zero_yield_frac"),
        why: "the ONLY axis in the registry carrying Provenance::Measured — and \
              its recorded delta is zero row conversions",
    },
    // #vit-margin-alpha (2026-09-24). Both PROMOTED dark: a typed key armed by
    // no shipped yaml, so this is DELIVERABLE, not armed. The vit measurement
    // arms them only in a scratch configs dir.
    Axis {
        name: "NY_ROOT_ALPHA_MARGIN_GRADIENT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // Was an undeclared raw env read (`propagate_dag/mod.rs`), i.e. dead in
        // competition; now resolved env-over-config from this key.
        deliver: Deliver::PresetKey("bab.alpha_crown.root_alpha_margin_gradient"),
        why: "margin C-row root alpha loss; on bilinear-state roots it runs the \
              certified C-row margin ascent and publishes a certify-once margin \
              bound the root intersects, so it can move a published bound",
    },
    Axis {
        name: "NY_ROOT_ALPHA_MATMUL",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.alpha_crown.matmul_alpha"),
        why: "alpha on the attention McCormick planes inside the margin ascent; \
              changes the certified margin bound the root merges",
    },
    // --- misc engine ---
    Axis {
        name: "NY_TRUE_GRAD_GPU_REPLAY",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.true_grad_gpu_replay"),
        why: "GPU replay of the true-gradient lane",
    },
    Axis {
        name: "NY_INPUT_SPLIT_NESTED_DEADLINE",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.branching.input_split.nested_deadline"),
        why: "nested deadline discipline for input split; scheduling only",
    },
    Axis {
        name: "NY_PATCHES_FINITE_EXPIRY",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.patches_finite_expiry"),
        why: "finite expiry on the patches path; refusal timing, not bound math",
    },
    Axis {
        name: "NY_NO_WALK_RECORD_ADMISSION",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.no_walk_record_admission"),
        why: "admission bookkeeping; LATCHED in a OnceLock",
    },
    // --- CROWN-IBP per-node budget policy (#cgan-bn11-budget) ---
    //
    // These preset fields and their operator aliases predate the governed
    // search space. The cap alias was explicitly introduced so the historical
    // 12-second ceiling could be measured instead of argued about, but omission
    // here meant the optimizer still could not move it. Both controls only
    // choose sound CROWN completion versus the certified IBP fallback; they can
    // nevertheless change which bound/verdict is reached inside the deadline.
    Axis {
        name: "NY_PER_NODE_FLOOR_SECS",
        domain: Domain::Enum(&["0.5", "1", "2", "5", "10"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.crown_ibp_per_node_floor_secs"),
        why: "minimum worthwhile target slice. Absence is the 2-second shipped control; lower values admit more short attempts while higher values preserve the collector window for targets whose weighted share is large enough to finish",
    },
    Axis {
        name: "NY_PER_NODE_CAP_SECS",
        domain: Domain::Grid(&[12, 25, 50, 100, 150, 300, 600]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.crown_ibp_per_node_cap_secs"),
        why: "fixed per-target cap replacing the contextual adaptive policy. The grid spans the historical clamp, budget-share scale, shipped cgan base cap, and the dimension-scaled safety ceiling; truncation falls back to sound IBP but can erase every useful CROWN tightening in the collected map",
    },
    // --- root/BaB window arbitration (#bab-floor; see decls/bab_budget.rs) ---
    //
    // These are the knobs a hand-run found non-monotonic: on cifar100_2024
    // idx_8600 the reserve gives a 97/99 root census at 0.10 and 0/99 at 0.20.
    // A share that only shortens phases should not collapse a census between
    // two neighbouring points, so the region between them is exactly what a
    // grid search has to resolve rather than a human bisecting it by hand.
    Axis {
        name: "NY_BAB_RESERVE_FRAC",
        // 0.0 is the disarmed control and belongs IN the grid: it is the
        // baseline arm the other four are scored against, and it is the only
        // setting that also disarms the two children below.
        domain: Domain::Enum(&["0.0", "0.10", "0.15", "0.20", "0.30"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_bab_reserve_frac"),
        why: "subtracts BaB's share before any root phase sizes itself; every \
              phase stays shrink-only and fails closed, but which verdicts are \
              REACHED inside the window changes, and the decl carries MoatRisk::High",
    },
    Axis {
        name: "NY_ROOT_SPEC_FRAC",
        domain: Domain::Enum(&["0.10", "0.15", "0.25"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_spec_frac"),
        why: "the root objective pass's share; starves the pass that PRODUCES \
              the bounds BaB consumes, so it cannot be scored independently of \
              the reserve above it",
    },
    Axis {
        name: "NY_ROOT_ALPHA_FRAC",
        domain: Domain::Enum(&["0.20", "0.30", "0.45"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_alpha_frac"),
        why: "converts the bootstrap's fixed 40 s wall into a share; that wall \
              is 51% of the BaB slice at 100 s and 4% at 1200 s, so it is the \
              one claimant whose cost does NOT scale with the window",
    },
    // --- accounted for by every_declared_lever_is_accounted_for ---
    // CORRECTED 2026-08-24 (#alpha-opt-adam). These six were EnvOnly, which
    // is precisely why there are zero `[alpha-opt]` lines in all 14
    // scored-configuration logs of
    // `docs/ABC_ROOT_GAP_ON_WINNABLE_ROWS_2026-08-24.md`:
    // `vnncomp_scripts/run_instance.sh` exports no `NY_*`, so the
    // whole root ascent was unreachable in competition — the fourth
    // BUILT / CORRECT / UNREACHABLE mechanism after NY_EPOCH_DEPTH,
    // NY_CGAN_RECLAIM and NY_MARGIN_ROW_CLIP_ROWS. The typed keys, their
    // readers and their bridge line landed together with the redesign.
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_OPT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_opt"),
        why: "alpha ascent on the margin-row lane; changes the published row bound",
    },
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_ITERS",
        domain: Domain::Grid(&[4, 8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_iters"),
        why: "ascent steps for that lane; more steps tighten but spend the row's window",
    },
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_SECS",
        domain: Domain::Grid(&[10, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_secs"),
        why: "wall cap on the margin-row ascent; the cap and the iteration count bind at different instance sizes, so neither prices the other",
    },
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_LR",
        domain: Domain::Enum(&["0", "0.05", "0.1", "0.2"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_lr"),
        why: "Adam step for the root ascent; `0` selects the legacy \
              normalized-supergradient step, which is the control arm, and the \
              measured defect was step SIZE (8 of 8 iterations accepted, eta \
              never halved) rather than direction",
    },
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_COLS",
        domain: Domain::Grid(&[1, 4, 8, 16]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_cols"),
        why: "worst-column steering width; the objective is a min over 3..37 \
              surviving classes and the legacy gradient saw only the argmin, so \
              neurons with no support there could never move. `1` is the control",
    },
    Axis {
        name: "NY_MARGIN_ROW_ALPHA_GUARD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.alpha_guard"),
        why: "published-frame revert guard; makes the published root bound \
              max(binary, optimized) by selection, so the ascent cannot lose a \
              banked proof to a frozen-frame gain that fails to transfer",
    },
    Axis {
        name: "NY_MARGIN_ROW_SPEC_ALPHA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.spec_alpha"),
        why: "#spec-alpha: the per-START-NODE (spec) alpha axis at the root \
              direct pass. abc optimises alpha[relu][start_node] and gains a \
              median +0.163 at the root; ny holds one shared alpha per \
              (layer, neuron). Folds by element-wise max, so it raises the \
              published root or costs only wall",
    },
    Axis {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_WIDTH",
        domain: Domain::Grid(&[1, 2, 3, 4, 8]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.spec_alpha_width"),
        why: "how many hardest failing classes get candidates; candidates are \
              pass WIDTH not pass COUNT, so this prices columns rather than \
              passes. `1` is the control",
    },
    Axis {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_CANDS",
        domain: Domain::Grid(&[1, 2, 3, 5]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.spec_alpha_cands"),
        why: "candidates per class from the five-entry closed-form ladder, \
              ordered most-distinct-first so a truncated dose keeps the \
              informative candidates",
    },
    Axis {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS",
        domain: Domain::Grid(&[1, 2, 3]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.spec_alpha_rounds"),
        why: "gradient rounds after the ladder; `1` (enumerate only) is the \
              control and each further round is one more certified pass, which \
              is what prices the mechanism against the row's wall",
    },
    Axis {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES",
        domain: Domain::Grid(&[0, 4096, 65536, 1048576]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.spec_alpha_entries"),
        why: "hard cap on committed exception entries; the margin-row lane \
              consults no memory governor and competition runs with no cgroup, \
              so this is one of the only two bounds that exist. `0` refuses \
              loudly and is the negative control",
    },
    Axis {
        name: "NY_MARGIN_ROW_K_ADAPT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // CORRECTED 2026-08-23. The typed key, its reader and its bridge line
        // already existed and were fully wired; this axis simply said
        // otherwise, which is a reachability ratchet counting a debt that
        // had already been paid.
        deliver: Deliver::PresetKey("margin_row.k_adaptive"),
        why: "adapts K per row instead of holding the preset value",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.clip"),
        why: "clips the margin row against the certified envelope",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP_HEAD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.clip_head"),
        why: "per-domain HEAD-split halfspaces re-concretize the OTHER head neurons before the \
              head gates are derived (alpha-beta-CROWN clip_interm_domain, head layer -- the \
              dominant term on the traced rows). Needs no clip_rows retention: the certified \
              head lines come from the y-pack the eval already holds",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP_INTERM",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // Promoted from EnvOnly: `run_instance.sh` exports exactly one `NY_*`,
        // so while this was env-only the whole Clip-and-Verify intermediate
        // half was unreachable on a scored run however well it measured.
        deliver: Deliver::PresetKey("margin_row.clip_interm"),
        why: "extends that clip to intermediate bounds; refinement of the clip above",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP_STATEFUL",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.clip_stateful"),
        why: "accumulates the clip's intermediate tightening per domain and inherits it \
              parent -> child instead of recomputing from the root frame and discarding it. \
              This is the alpha-beta-CROWN 2024->2025 cifar100 diff (add clip_interm_domain, \
              REMOVE interm_transfer:False) and it is the reason ny's stateless port measured \
              a FLAT dose-response. Only searchable jointly with NY_MARGIN_ROW_CLIP_INTERM: \
              armed alone it is inert",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP_ROWS",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.clip_rows"),
        why: "retains the certified clip halfspace rows required by both clip consumers; retention alone is value-neutral, but the declaration records Low moat risk because those rows are load-bearing for bound-moving paths",
        // Promoted from EnvOnly: this is THE THIRD GATE. Root-build retention is
        // the prerequisite of every Clip-and-Verify consumer, so while it was
        // env-only the other two clip axes could not fire on a scored run at all.
    },
    Axis {
        name: "NY_MARGIN_ROW_CLIP_TOPK",
        domain: Domain::Grid(&[5, 20, 50]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.clip_topk"),
        why: "how many rows the clip considers; a budget, not a bound change",
    },
    Axis {
        name: "NY_RESNET_ERR_CONCRETIZE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.resnet_err_concretize"),
        why: "moves WHERE the sound GPU-resident ResNet walk discharges its accumulated \
              coefficient error -- per segment against the frontier magnitudes, instead of \
              carried forward as coefficient radii. It is on the CERTIFIED lane and feeds \
              the main-bound auto-fallback, so it changes which bound is published; not a \
              plumbing switch despite looking like one",
    },
    Axis {
        name: "NY_RESNET_ERR_CONCRETIZE_FINE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.resnet_err_concretize_fine"),
        why: "per-RELU rather than per-segment error concretization on the same certified \
              walk; a refinement of the axis above and inert unless that one is armed",
    },
    // --- #closure-adaptive: the runtime escalation policy ---
    //
    // All five are VerdictAffecting: they decide which tightening mechanisms
    // the margin-row lane runs, and therefore which bound it publishes. None is
    // Unsafe — every tier is sound on its own (beta by weak duality, both clip
    // tiers shrink-only) and the certified pass stays the only scorer, so a
    // wrong value here costs proofs and cannot manufacture one.
    //
    // THE BRIDGE HAS LANDED (2026-08-23). `commands::vnncomp` now calls
    // `set_closure_adaptive_preset` from `closure_adaptive_from_preset`,
    // alongside the k-adaptive / bi_secs / backward_interm readers, so the
    // typed key is the real delivery route and this is `PresetKey`.
    //
    // The previous note here — that the bridge "has not landed, so as of this
    // writing the policy CANNOT fire in a scored run" — was accurate and load
    // bearing: the ladder was built, unit-tested and MEASURED entirely through
    // its env half while being dead code on every scored run. That is the same
    // failure as `NY_EPOCH_DEPTH`, `NY_CGAN_RECLAIM` (bridged 2026-08-23, see
    // its axis below) and `NY_MARGIN_ROW_CLIP_ROWS`. Recording that this enum
    // CAUGHT it, which is what it is for.
    //
    // Still default OFF and no preset sets the key, so arming remains a
    // separate, measured decision.
    // #branch-reduceop. Searchable: it is a real A/B axis with a measured
    // anti-parity behind it (abc's TUNED cifar100 and tinyimagenet configs both
    // set `reduceop: max`, overriding its own global `min` default, on exactly
    // the two benchmarks ny fails). VerdictAffecting because a different tree
    // reaches a different set of domains inside the budget — it can lose a
    // banked row. It is NOT UnsafeAxes: the bound still comes from the
    // certified pass, so it cannot manufacture a proof.
    // --- #epoch-bab: the Tier-2 rebuild and the Tier-0 ranker beside it ---
    //
    // The CADENCE PAIR is VerdictAffecting and, since `a1c80f6a3`, delivered.
    // `try_epoch` rebuilds every intermediate with the trunk splits baked in as
    // absolute neuron indices and PUBLISHES the rebuilt frame, so it moves a
    // bound by construction. Measured once, cifar100 idx_4169 at the official
    // 100 s: the rebuild was 0.0186 WORSE than its own parent until
    // `inherit_parent_intervals` intersected the parent's certified intervals
    // into it, after which it is 0.0546 BETTER, 3/3
    // (docs/PROGRAM_2026-08-22_CLOSURE_FIRST.md §0d). Closure movement 0.000,
    // so this is searchable, not qualified.
    Axis {
        name: "NY_EPOCH_DEPTH",
        domain: Domain::Grid(&[1, 2, 3, 4]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.epoch_depth"),
        why: "trunk-split cadence at which a domain's subtree is re-linearized with the \
              splits BAKED IN — the only path in ny that recomputes intermediates under \
              the split constraints the way alpha-beta-CROWN does, and it publishes the \
              rebuilt frame. Grid starts at 1 because 0 is the dark arm and the measured \
              control at 8 never fired inside 93 s",
    },
    Axis {
        name: "NY_EPOCH_ATTEMPTS",
        domain: Domain::Grid(&[1, 2, 4]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.epoch_attempts"),
        why: "how many non-closing Tier-2 rebuilds one run may pay for. Each is a full \
              root-build against the same official deadline, so this prices the cadence \
              above rather than being independent of it; its historical default is 2, not \
              0, so absence is not the dark arm",
    },
    // The Tier-0 TRIPLE is SafeToSearch, and the reason is worth stating
    // because the neighbouring NY_MARGIN_ROW_BRANCH_REDUCE_MAX is a ranker too
    // and sits in VerdictAffecting.
    //
    // Both are ranker-only; the difference is what the two enums are for. This
    // file's `VerdictAffecting` means "may change a PUBLISHED BOUND", and
    // `SafeToSearch` means "cannot change a verdict's soundness, still costs
    // wall clock". Tier-0 cannot change a published bound: the retained rows
    // are f32 and read by exactly one consumer, the nearest-mode
    // `bounds::trunk_variant` ranker (`root.rs`, `RetainedLayer`: "No Outward
    // (verdict-grade) pass ever reads them"); retention itself is pure capture,
    // copied out of `mi`/`di` AFTER `concretize_box` without touching
    // `l`/`u`/`alpha`/`s`/`c`/`ms`/`unst`; and `tier0_candidates` reorders a
    // candidate list that `score_candidates` (Outward, certified) then scores
    // unchanged.
    //
    // What it CAN do is reach a different set of domains inside the budget, and
    // that has been measured: on the 13-instance pyrat-easy tier the closure
    // count is 6 = 6, Tier-0 gaining prop_1588 and losing prop_4605
    // (docs/EPOCH_BAB_STATE_2026-07-18.md). So a sweep may move these three,
    // must price them on closure count rather than on bound, and must not read
    // a win here as a bound improvement. If a future reader decides that
    // "reaches different domains" should itself be VerdictAffecting, then
    // NY_MARGIN_ROW_BRANCH_REDUCE_MAX, NY_MARGIN_ROW_BETA_ETA/ITERS and these
    // three all move together. The three below now have typed preset routes;
    // no shipped preset arms them.
    Axis {
        name: "NY_EPOCH_TIER0",
        domain: Domain::Grid(&[4, 8, 16]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.epoch_tier0"),
        why: "arms Tier-0 rank-1 candidate ranking and sets how many candidates reach the \
              exact Tier-1 pass; measured 124 -> 66 ms/call on the July 2026 binary with \
              expansion counts IDENTICAL wherever both lanes close the same tree, and NO \
              net closure gain (6 = 6), which is why it is dark",
    },
    Axis {
        name: "NY_EPOCH_UNIVERSE",
        domain: Domain::Grid(&[32, 64, 128]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.epoch_universe"),
        why: "how many trunk candidates the Tier-0 ranker scores before truncating to the \
              width above; a width knob under an arming gate, inert while NY_EPOCH_TIER0 \
              is 0, and its 64 default means absence is not the off arm",
    },
    Axis {
        name: "NY_EPOCH_RETAIN",
        domain: Domain::Grid(&[64, 128, 256]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.epoch_retain"),
        why: "RetainCfg::per_layer — retained f32 sandwich rows per trunk ReLU, the \
              ranker's input width. Whether retention runs at all is a DISJUNCTION \
              (NY_EPOCH_TIER0 > 0 OR an armed cadence with the incremental refresh off) \
              but with a cadence and no Tier-0 the rows are retained and never read, so \
              searchable treatment delivery requires the Tier-0 arm",
    },
    Axis {
        name: "NY_MARGIN_ROW_BRANCH_REDUCE_MAX",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.branch_reduce_max"),
        why: "rank a branching candidate by its BETTER child (BaBSR s_SR, De Palma eq. 22) \
              instead of its WORSE child (FSB balanced, eq. 23) — abc's tuned cifar100 and \
              tinyimagenet configs both choose the former and ny hardcoded the latter",
    },
    Axis {
        name: "NY_MARGIN_ROW_PARENT_LIFT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.parent_lift"),
        why: "fold the branching scores the expansion already paid for into the child bound \
              floor: each scored candidate splits the parent into two children that COVER \
              it, so min(l_i, r_i) lower-bounds the parent and max_i of that is free \
              evidence ny currently discards for all but the candidate it splits on",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.closure_adaptive"),
        why: "arms the closure-driven runtime escalation: the lane measures its own \
              child-closure fraction and climbs base / beta / beta_deep / \
              beta_deep+epoch when its children stop closing, instead of running whatever \
              a preset guessed",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLOSURE_THRESH",
        // A search may only move this ACROSS abc's measured bands, not inside
        // them: 0.34 is abc's loss ceiling, 0.50 its win floor, 0.40 the shipped
        // midpoint of the empty band between them. A finer grid would be
        // fitting noise, because there is no evidence between 0.34 and 0.50.
        domain: Domain::Enum(&["0.25", "0.34", "0.40", "0.50"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.closure_thresh"),
        why: "the closure fraction that triggers escalation; moving it changes which rows \
              get the extra mechanisms and therefore which bound is published",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLOSURE_WARMUP",
        domain: Domain::Grid(&[8, 16, 24, 48]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.closure_warmup"),
        why: "expansions before the first decision; too short and a row that has not yet \
              discharged its first batch is misread as stuck, too long and the budget is \
              gone before the escalation can pay",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLOSURE_WINDOW",
        domain: Domain::Grid(&[8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.closure_window"),
        why: "expansions per re-evaluation; sets how fast the ladder can be climbed and how \
              much noise one window carries",
    },
    Axis {
        name: "NY_MARGIN_ROW_CLOSURE_MAX_TIER",
        domain: Domain::Grid(&[0, 1, 2, 3]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.closure_max_tier"),
        why: "caps the runtime ladder (0 base, 1 beta, 2 beta_deep, 3 \
              beta_deep+epoch); the axis that isolates ONE mechanism, without which a null \
              result cannot be attributed",
    },
    Axis {
        name: "NY_ROOT_OBJECTIVE_DIRECTED_ROWS",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_objective_directed_rows"),
        why: "directs the root objective pass at the rows that actually gate the verdict rather than sweeping all of them",
    },
    Axis {
        name: "NY_CPU_COMPUTE_ENGINE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.cpu_compute_engine"),
        why: "retains or suppresses the optional CPU compute-engine presence handle; this can change which sound verifier work reaches the deadline, while an independently qualified WGPU proof device is always retained",
    },
    Axis {
        name: "NY_PATCHES_ENGINE_REFUSAL_CONTAINMENT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED DARK 2026-08-31 (#patches-engine-refusal-containment,
        // reports/measured-2026/cct_bound_gap_2026-08-31). Default OFF and set
        // by no category-resolvable yaml, so this is a no-op today: it makes
        // the containment DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.patches_engine_refusal_containment"),
        why: "treats a POLICY-refusing GEMM engine at the Conv2d Patches compose seam as an absent accelerator so the walk keeps its byte-identical CPU scatter in patches instead of dying into a dense retry that blows the memory budget and leaves IBP-grade intermediates",
    },
    Axis {
        name: "NY_FAST_F32_PREWARM",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED DARK 2026-08-31 (#fast-f32-prewarm-all-hosts, same report).
        // Scheduling-only (engine materialization, no bound math), but it
        // changes which certified work completes before the deadline.
        deliver: Deliver::PresetKey("bab.fast_f32_prewarm"),
        why: "materializes the registered fast-f32 GEMM factory on the vnncomp route for every build, so finite-deadline consumers (which fail closed on a cold slot) can reach a real GEMM instead of the quarantined proof adapter",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha"),
        why: "master arm shared by the input-split and graph root-map transactions, finite-deadline GPU admission and CLI engine selection; one governed answer prevents a half-enabled scored treatment",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_deadline_ascent"),
        why: "allows optimizer iterations under finite authority only when the audited bounded joint-adjoint and sound-fold capability is present; otherwise the route retains its zero-iteration certified base fold",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
        domain: Domain::Grid(&[8, 20, 30, 60]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_secs"),
        why: "sizes the root-joint wall request and scheduler price; it decides how many target folds can publish before the root map freezes and how much budget remains for BaB",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        domain: Domain::Grid(&[1, 12, 50, 100]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_iters"),
        why: "caps bounded joint-alpha optimizer steps after finite-deadline ascent is admitted; changing it changes the certified shrink reached before publication",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        domain: Domain::Enum(&["0.01", "0.05", "0.1", "0.2"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_lr"),
        why: "sets the joint-alpha Adam trajectory and therefore which sound intermediate bound is reached within the fixed iteration/deadline authority",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
        domain: Domain::Grid(&[1, 4, 8]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_max_sel"),
        why: "caps crossing rows seeded per target; the finite-deadline CUDA route audits at most eight rows, and execution receipts credit a treatment only when raw crossing demand exceeded the configured cap and the selected width was actually reached",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_frozen_stop"),
        why: "admits a certified suffix that stops at a finite frozen interior box, making otherwise unreachable cGAN targets executable; receipts require at least one accepted prep whose stop_node proves the arm acted",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_pure_chain"),
        why: "admits a certified suffix with no residual block; receipts credit it only for an accepted all-Chain prep where frozen-stop was not an independent admitting authority",
    },
    Axis {
        name: "NY_BN_GPU_EXTRACT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.bn_gpu_extract"),
        why: "admits the certified BatchNorm-to-diagonal-Conv extraction required by the measured cGAN root-joint recipe; receipts require an accepted target whose extraction actually traversed BatchNorm",
    },
    Axis {
        name: "NY_ROOT_JOINT_MIN_REMAINING_SECS",
        domain: Domain::Grid(&[0, 120, 240, 360]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_joint_min_remaining_secs"),
        why: "graph-route admission floor for the measured root-joint tax; it decides whether the transaction publishes at all while preserving the conservative 240-second shipped default",
    },
    Axis {
        name: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        domain: Domain::Grid(&[64, 256, 1024, 2048, 32768]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.root_joint_interm_alpha_max_dim"),
        why: "dimension ceiling above which the joint intermediate alpha pass declines; the ceiling decides whether the pass runs at all on a wide graph",
    },
    Axis {
        name: "NY_SWEEP_CLASS_ROWS",
        domain: Domain::Grid(&[64, 256, 1024]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.sweep_class_rows"),
        why: "row count that classifies a sweep as large; picks the execution class",
    },
    Axis {
        name: "NY_SWEEP_CLASS_MIB",
        domain: Domain::Grid(&[64, 256, 1024, 2048, 4096, 8192]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.sweep_class_mib"),
        why: "the memory half of the same classification; rows and MiB can disagree, and which one binds is the thing to measure",
    },
    Axis {
        name: "NY_MO_GPU_CHUNK_DEADLINE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.mo_gpu_chunk_deadline"),
        why: "gives each multi-objective GPU chunk its own deadline instead of one deadline for the batch",
    },
    Axis {
        name: "NY_MO_ALPHA_HEURISTIC",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.mo_alpha_heuristic"),
        why: "replaces stale ancestor graph alpha with the sound current-box \
              heuristic after split clamps. The measured deep-child lower \
              bound moved, but production stays dark until a sealed row-level \
              search establishes conversion and non-regression",
    },
    Axis {
        name: "NY_MO_BETA_SPSA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.mo_beta_spsa"),
        why: "selects the measured two-sided beta gradient or the analytical \
              rollback. The analytical form differentiates the wrong quantity; \
              SPSA moved the probe bound, but its two extra sound passes per \
              iteration can change which domains finish before the deadline",
    },
    Axis {
        name: "NY_MO_ALPHA_STEP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.mo_alpha_step"),
        why: "replaces every fourth beta-SPSA optimizer round with an isolated \
              current-box delta-alpha round on relu-split children. The fixed \
              schedule preserves twelve of fifteen beta rounds and cannot be \
              tuned independently from the shared iteration budget",
    },
    Axis {
        name: "NY_MO_KFSB",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.kfsb_multi"),
        why: "canonical master for wave-batched multi-objective kFSB child \
              simulation. It changes the committed split and therefore which \
              certified domains are reached before the deadline; the existing \
              typed branching key makes both search arms deliverable without \
              relying on process-environment injection",
    },
    Axis {
        name: "NY_MO_KFSB_CERT_REUSE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.kfsb_cert_reuse"),
        why: "proof-carrying reuse can close rows, children, or an untouched \
              parent and skip later child evaluation. Typed delivery makes the \
              configured CIFAR100 baseline and both override arms part of the \
              frozen flight receipt; scored search remains delivery-only until \
              execution evidence proves accepted reuse or a binding refusal",
    },
    Axis {
        name: "NY_MO_KFSB_K",
        domain: Domain::Grid(&[0, 2, 5, 7, 8, 10]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.kfsb_multi_candidates"),
        why: "candidate budget for wave-batched kFSB simulation. Zero declines \
              the lane; the positive grid covers the measured legacy width, \
              shipped FSB/kFSB widths, and the historical solver default. It \
              changes both simulation cost and the split portfolio",
    },
    Axis {
        name: "NY_MO_KFSB_REDUCE",
        domain: Domain::Enum(&["min", "max"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.kfsb_multi_reduceop"),
        why: "exact environment domain for reducing the two simulated child \
              scores. Min and Max can choose different committed splits; typed \
              presets retain their broader existing Mean option, but automated \
              environment search preserves the live min|max parser",
    },
    Axis {
        name: "NY_KFSB_SIM_SHARE",
        domain: Domain::Enum(&["0.0", "0.15", "0.35", "0.60"]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.kfsb_sim_share"),
        why: "share of the kFSB branching budget spent on simulation; branching \
              order decides which subproblems are ever opened. Zero is the \
              canonical global-deadline control; 1.0 is excluded from automated \
              search because it resolves to that same deadline",
    },
    // --- batched exact-VJP graph PGD policy ---
    Axis {
        name: "NY_PGD_EXACT_BATCHED",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_exact_batched"),
        why: "default-on conjunctive routing between generic restart-batched SPSA and the exact joint-gradient path; both candidates and the scored wall spent reaching them can change",
    },
    Axis {
        name: "NY_PGD_FAST_GRAD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_fast_grad"),
        why: "default-on sequential point-Jacobian attempt before certified CROWN and SPSA fallbacks; gradient cost and quality change which attack trajectories can produce a candidate before the scored deadline",
    },
    Axis {
        name: "NY_PGD_VJP_BATCH",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_vjp_batch"),
        why: "shared default-on admission for the conjunctive and disjunctive wide exact-VJP drivers; exact 0 selects their sequential fallback",
    },
    Axis {
        name: "NY_PGD_VJP_K",
        domain: Domain::Grid(&[8, 16, 32, 52, 64, 96, 128]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_vjp_k"),
        why: "restart width trades per-step throughput, wave depth and basin diversity; the consumer may halve it on an adapter capacity refusal, so execution evidence must prove the effective width",
    },
    Axis {
        name: "NY_PGD_LEGACY_LANES",
        domain: Domain::Grid(&[0, 8, 16, 20]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_legacy_lanes"),
        why: "replaces trailing first-wave ODI lanes with the old uniform seed replay; the measured positive values recover one basin while losing more ODI basins",
    },
    Axis {
        name: "NY_PGD_ODI_STEPS",
        domain: Domain::Grid(&[0, 4, 8, 16]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_odi_steps"),
        why: "number of wide VJP passes spent reaching output-diverse basins before conjunctive joint-margin ascent",
    },
    Axis {
        name: "NY_PGD_ODI_ETA",
        domain: Domain::Enum(&["0.25", "0.5", "1", "1.5", "2"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_odi_eta"),
        why: "finite positive f32 step scale for ODI; it is inert when ODI_STEPS is zero and otherwise changes every initialized trajectory",
    },
    Axis {
        name: "NY_PGD_VJP_STEPS",
        domain: Domain::Grid(&[100, 250, 500, 800, 1000]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_vjp_steps"),
        why: "optional per-wave joint-margin cap trading deep trajectories for more ODI waves; absence contextually inherits the category's typed PGD step count",
    },
    Axis {
        name: "NY_PGD_LANE_WINDOW",
        domain: Domain::Grid(&[0, 50, 100, 200]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_lane_window"),
        why: "rate-projection recycling can discard or replace a lane before it reaches its counterexample basin; zero is the measured incumbent",
    },
    Axis {
        name: "NY_PGD_EXPLOIT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_exploit"),
        why: "default-on endgame respawning changes which near-boundary lineage receives the final exact-VJP steps",
    },
    Axis {
        name: "NY_PGD_HINGE_TAU",
        domain: Domain::Enum(&["0", "0.0001", "0.001", "0.01"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_hinge_tau"),
        why: "odd-lane tau-hinge threshold changes which conjunct rows contribute to the exact gradient; zero is the measured incumbent",
    },
    Axis {
        name: "NY_PGD_DISJ_ODI",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_disj_odi"),
        why: "#disj-odi: arms the ODI phase on every disjunctive exact-VJP wave's dynamic lanes, so ODI_STEPS/ODI_ETA reach the disjunctive driver; it moves those lanes' start points and spends ODI VJP passes of the attack slice",
    },
    Axis {
        name: "NY_PGD_DISJ_FOCUS_LANES",
        domain: Domain::Grid(&[0, 16, 32, 48, 64]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.pgd_disj_focus_lanes"),
        why: "#disj-odi: dangerous-class focus lanes appended after the disjunctive restart waves (ODI init, HINGE_TAU hinge, cosine momentum-sign steps); they spend the rest of the attack slice on the two closest clauses",
    },
    // --- VNN-COMP scored-entry upfront falsification policy ---
    //
    // These four values are resolved from one entry-time LeverSet and the same
    // object produces the flight receipt. The master has three effective
    // states: absence is contextual auto, exact 0 disables, exact 1 forces.
    // Caps can saturate against the live remaining wall, so a banked search
    // must use the execution receipt rather than inferring engagement from the
    // requested token alone.
    Axis {
        name: "NY_UPFRONT_ATTACK",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.upfront_attack"),
        why: "exact force/kill override above contextual automatic disjunction and typed relational admission. Absence is a third, runtime-dependent baseline rather than false: 0 removes the lane, 1 forces it on structurally supported rows, and both choices reallocate scored wall against branch-and-bound",
    },
    Axis {
        name: "NY_UPFRONT_ATTACK_CAP",
        domain: Domain::Grid(&[0, 1, 2, 4, 8, 12, 16, 25]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.upfront_attack_cap_secs"),
        why: "ordinary force/typed-relational whole-second cap. Zero is the parser-valid direct disable; the positive grid includes the measured 8-second default, the automatic scale, and the historical 25-second regression point. The effective window is min(fractional live wall, cap), so execution evidence must distinguish a binding cap from a saturated no-op",
    },
    Axis {
        name: "NY_UPFRONT_ATTACK_AUTO_CAP",
        domain: Domain::Grid(&[0, 1, 2, 4, 6, 8, 12]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.upfront_attack_auto_cap_secs"),
        why: "extra clamp used only by contextual automatic multi-clause-disjunction admission. Zero directly declines that automatic lane; positive values protect proof time on shared robustness categories. Forced and typed-relational routes intentionally bypass it, so arms must be banked only where the flight route records automatic admission",
    },
    Axis {
        name: "NY_UPFRONT_ATTACK_FRAC",
        domain: Domain::Enum(&["0.02", "0.04", "0.08", "0.12", "0.2", "0.5", "1"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.upfront_attack_fraction"),
        why: "fraction of live scored wall offered before the ordinary cap. It composes with both caps and therefore can bind or saturate by row budget; changing it reallocates time between trusted-gated falsification and certified branch-and-bound",
    },
    Axis {
        name: "NY_BNN_ATTACK_FRAC",
        domain: Domain::Enum(&["0.25", "0.4", "0.5", "0.6", "0.75", "0.9", "1"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.bnn_attack_fraction"),
        why: "Sign-network share of live scored wall offered to the surrogate-gradient upfront lane. It is applied only after ordinary admission and base-budget gating, then capped by NY_BNN_ATTACK_CAP and raised to the ordinary budget; exact execution evidence must prove that this fraction, rather than either cap or the ordinary floor, determined a row's effective budget",
    },
    Axis {
        name: "NY_BNN_ATTACK_CAP",
        domain: Domain::Grid(&[60, 120, 150, 180, 210, 240, 300, 460]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.bnn_attack_cap_secs"),
        why: "Sign-network whole-second cap for the surrogate-gradient upfront lane. The grid covers smaller allocation points, the 150-second measured win scale, the retained 240-second default, and the measured-negative 460-second hard-tail probe; exact execution evidence rejects values hidden by the live-wall fraction or ordinary floor",
    },
    Axis {
        name: "NY_POSTBAB_RESERVE_SECS",
        domain: Domain::Grid(&[15, 30, 60, 100]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.phase_budget.post_bab_reserve_secs"),
        why: "fixed whole-second transfer from the internal verifier to the outer post-BaB trusted-ORT falsifier. Absence/zero is the current historical baseline rather than an advertised treatment; 100 is the ACAS Xu hypothesis that produced host-local SAT candidates, with smaller allocations pricing how much proof time that result actually needs. Search is restricted to ACAS Xu until another family has both outcome evidence and typed engine-deadline provenance. Configured search also excludes exact relational entry bypasses and any category preset that explicitly disables the outer wrapper before spending a run. A request can saturate at the ten-second verifier floor or lose/tie another deadline authority, and the lane can fail to start real work, so only an unsaturated versioned execution receipt can establish engagement",
    },
    Axis {
        name: "NY_POSTBAB_RESTARTS",
        domain: Domain::Grid(&[2, 4, 8, 16]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.post_bab_restarts"),
        why: "diversified-restart count for the post-BaB stage-0 alternation after it self-terminates with attack window remaining. Absence/zero is the historical convergent single-descent baseline; the 2026-08-31 soundnessbench window probe measured that stack reaching bit-identical fixed points at 11 s and 49.6 s windows, so restart DIVERSITY (different basins) rather than window length is the axis under measurement, and the reference tool runs 250 restarts on that family. Engagement evidence is the lane's own per-restart basin/margin lines plus its terminal restarts-used line; a run without them is VOID, not negative. Every restart candidate still passes the unchanged trusted-ORT acceptance and the caller's unchanged gate_sat_with_trusted_oracle",
    },
    Axis {
        name: "NY_POSTBAB_ATTACK",
        // The legacy process input is deliberately a one-way rollback: only
        // exact `0` owns the env layer. Search measures the kill against the
        // enabled absence/preset baseline; a future force treatment must use
        // the typed preset key instead of pretending env `1` overrides it.
        domain: Domain::Enum(&["0"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.phase_budget.vnncomp_post_bab_attack"),
        why: "independent wrapper admission for the trusted-ORT-gated post-BaB falsifier. False can transfer otherwise reserved scored wall back to proof only when the typed engine fraction is explicitly zero; true/default retains the SAT-recovery lane. The execution receipt must distinguish a typed opt-out from the environment-only kill because only the former may own the proof tail",
    },
    Axis {
        name: "NY_FALSIFY_PORTFOLIO",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.falsify_portfolio"),
        why: "runs the falsification portfolio; produces SAT verdicts, which is the half of the score BaB cannot reach",
    },
    Axis {
        name: "NY_FALSIFY_PORTFOLIO_SECONDS",
        domain: Domain::Grid(&[0, 5, 15, 30]),
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.falsify_portfolio_secs"),
        why: "wall budget for that portfolio, taken from the same instance window the sound lanes spend",
    },
    Axis {
        name: "NY_FLAT_SURROGATE_MAX_BLOCKS",
        domain: Domain::Grid(&[0, 512, 2048, 4096]),
        class: Class::VerdictAffecting,
        // PROMOTED AT BIRTH (#flat-ladder-depth, 2026-08-31): the typed key is
        // the primary route and the env name exists only for A/B recipes.
        // Default 0 and set by no shipped yaml, so this is a no-op today: it
        // makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.flat_surrogate_max_blocks"),
        why: "caps the active Hadamard block count in the flat-objective surrogate ladder, switching it to selective deep refinement; changes which zeroth-order probes are spent and therefore which counterexamples are findable within the budget",
    },
    Axis {
        name: "NY_FLAT_SURROGATE_TRANSFER",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED AT BIRTH (#flat-surrogate-transfer, 2026-08-31): the typed
        // key is the primary route and the env name exists only for A/B
        // recipes. Default false and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.flat_surrogate_transfer"),
        why: "runs the surrogate->real transfer stage after the surrogate margin crosses zero, walking the real output's quantizer staircase from the surrogate optimum; changes which zeroth-order probes are spent and therefore which counterexamples are findable within the budget",
    },
    Axis {
        name: "NY_ATTACK_PRE_SOFTMAX_OBJECTIVE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.pre_softmax_objective"),
        why: "attacks the pre-softmax objective instead of the post-softmax one; changes which counterexamples are findable",
    },
    Axis {
        name: "NY_BNN_SIGN_SPACE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // CORRECTED 2026-08-23. The typed key, its reader and its bridge line
        // already existed and were fully wired; this axis simply said
        // otherwise, which is a reachability ratchet counting a debt that
        // had already been paid.
        deliver: Deliver::PresetKey("attack.bnn_sign_space"),
        why: "sign-space encoding for binarised networks; the arming gate for the two refinements below",
    },
    Axis {
        name: "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("attack.bnn_sign_space_minimal_move"),
        why: "minimal-move variant of that encoding; MoatRisk::Low in the decl",
    },
    Axis {
        name: "NY_BNN_SIGN_SPACE_TRUST_REGION",
        domain: Domain::Enum(&["box", "tight", "linf"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("attack.bnn_sign_space_trust_region"),
        why: "trust-region shape for the sign-space search",
    },
    Axis {
        name: "NY_LANE_BUDGET_FRAME",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.lane_budget_frame"),
        why: "sizes every attack-slice lane once, up front, from each lane's own unchanged rule, so a \
              lane that declines cannot have its unspent seconds absorbed by a downstream lane that \
              re-derives its window from the live remaining budget; it changes how much budget each \
              falsification lane gets, so it changes which counterexamples are reachable within it",
    },
    Axis {
        name: "NY_LANE_BUDGET_ALLOCATOR",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.lane_budget_allocator"),
        why: "commits every attack-slice lane's cap jointly and up front by solving a multiple-choice \
              knapsack, and can take a structurally blind lane to zero seconds; it changes which lane \
              runs under what cap, so it changes which counterexamples are reachable within the budget",
    },
    Axis {
        name: "NY_UPFRONT_ATTACK_PROBE_FIRST",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // CORRECTED 2026-08-23. The typed key, its reader and its bridge line
        // already existed and were fully wired; this axis simply said
        // otherwise, which is a reachability ratchet counting a debt that
        // had already been paid.
        deliver: Deliver::PresetKey("attack.upfront_attack_probe_first"),
        why: "consults the exact-gradient upfront falsification lane once at the TOP of the attack \
              slice under a 5 s cap instead of only behind the two BNN lanes; it changes which lane \
              runs when, and takes a bounded number of seconds from the lanes behind it, so it \
              changes which counterexamples are reachable within the budget",
    },
    Axis {
        name: "NY_SCORED_WALL_CLAMP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("solver.scored_wall_clamp"),
        why: "separates the branch-and-bound STOP from the effort HORIZON its timeout shapes, so a \
              lane that deliberately inflates its deadline to keep a non-monotonic search converging \
              can no longer buy that trajectory with wall-clock past the scored budget; armed, a row \
              that would have decided after the wall publishes `timeout` instead, which is a verdict \
              change on the authoritative route",
    },
    Axis {
        name: "NY_LANE_VALUE_SCHEDULER",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("attack.lane_value_scheduler"),
        why: "reallocates a stalled lane's remaining seconds to a later lane; changes which lane runs under what cap, so it changes which counterexamples are reachable within the budget",
    },
    Axis {
        name: "NY_BNN_STE_PGD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // CORRECTED 2026-08-23. The typed key, its reader and its bridge line
        // already existed and were fully wired; this axis simply said
        // otherwise, which is a reachability ratchet counting a debt that
        // had already been paid.
        deliver: Deliver::PresetKey("attack.bnn_ste_pgd"),
        why: "straight-through-estimator PGD for binarised nets; a falsification path",
    },
    Axis {
        name: "NY_CLIP_HOST_MEAN_LA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.clip_host_mean_la"),
        why: "host-side mean look-ahead clip",
    },
    Axis {
        name: "NY_CLIP_INTERM_CERTIFIED",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.clip_interm_certified"),
        why: "clips intermediate bounds against the certified envelope",
    },
    Axis {
        name: "NY_GRAPH_MIP_LEAF_SAT",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.graph_mip_leaf_sat"),
        why: "hands BaB leaves to the MIP backend for a SAT decision instead of leaving them unknown",
    },
    Axis {
        name: "NY_MIP_LEAF_UNDECIDED_STREAK_CAP",
        domain: Domain::Grid(&[0, 1, 2, 3]),
        class: Class::VerdictAffecting,
        // PROMOTED AT BIRTH (#metaroom-mip-governor, 2026-09-11): the typed key
        // is the primary route and the env name exists only for A/B recipes.
        // Default 0 (no cap) and set by no shipped yaml, so this is a no-op
        // today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.mip.leaf_undecided_streak_cap"),
        why: "stops the Graph-MIP leaf oracle after N consecutive undecided consults and returns the \
              remaining seconds to branch-and-bound; on metaroom spec_idx_28 the oracle spent 60.4s of \
              a 210s budget for six undecided replies, so the cap changes how much search the row gets",
    },
    Axis {
        name: "NY_MIP_ESCALATE_SKIP_CERTIFIED",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED AT BIRTH (#metaroom-mip-governor, 2026-09-11): same shape as
        // the sibling above. Default false and set by no shipped yaml.
        deliver: Deliver::PresetKey("bab.mip.escalation_skips_certified_clauses"),
        why: "seeds the disjunctive MIP escalation's pending set from the uncertified clauses only, so the \
              one open clause gets the whole slice instead of 1/19th of it; changes which clause is \
              actually solved within the escalation budget",
    },
    // --- margin-row backward / beta / cgan (4227a10bc, declared after the fact) ---
    Axis {
        name: "NY_MARGIN_ROW_BACKWARD_INTERM",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.backward_interm"),
        why: "backward intermediate refinement; the one axis here already SHIPPED on the scored cifar100 and \
              tinyimagenet presets, on a measured row conversion, so its arms are baseline-vs-shipped rather \
              than dark-vs-dark",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_SECS",
        // 40 is deliberately NOT in the grid even though cifar100 now ships it:
        // the grid is what an automated SEARCH may emit, and a search that
        // lands on the shipped value learns nothing. Keep the shipped point
        // outside the search points.
        // The official twin-wall rows have a 100 s budget and this phase is
        // capped to 40% of remaining wall time. A requested 60 can therefore
        // never execute exactly (and the T1 receipt correctly refuses it).
        // The retained current-binary runbook reviews 20 vs 30. Keeping an
        // unreviewed 5 here also made every SECS x CHUNK_SECS pair-cover abort:
        // its 0.5 s target cap makes both reviewed 1/2 s chunk targets inert.
        domain: Domain::Grid(&[20, 30]),
        class: Class::VerdictAffecting,
        // Promoted from EnvOnly by b4b77868d: `margin_row.bi_secs` is a typed
        // preset key, so a winning value can actually reach a scored run.
        // Env still wins over the preset in both directions at the reader.
        deliver: Deliver::PresetKey("margin_row.bi_secs"),
        why: "wall budget for the backward-intermediate refinement; it fails closed on expiry, \
              so a short budget publishes looser bounds rather than wrong ones. MEASURED \
              2026-08-22: the phase is TRUNCATED at the 20s default on 10/10 cifar100 \
              resnet_LARGE rows (which need 24-35s) and 0/9 resnet_MEDIUM (6.6-8.4s), so the \
              right value is model-shape dependent, not global — cifar100 ships 40, and \
              tinyimagenet deliberately does NOT. Its pre-tableau-f32 higher-dose arm solved \
              idx_5185 once but validated 0/3 afterward (1/4 total), so it did not bank; the \
              later tableau-f32 speedup invalidated the old wall/tree rationale and requires \
              a clean current-binary remeasurement rather than a transferred setting",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_CHUNK",
        domain: Domain::Grid(&[64, 256, 1024]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_chunk"),
        why: "batching grain over independent columns, but the clock is checked only between \
              passes: changing the grain can change which completed columns reach the \
              published root before the deadline, so the declaration correctly classifies \
              it as MoatRisk::High and promotion requires scored evidence",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_CHUNK_SECS",
        // 0 is the SHIPPED DEFAULT, not a kill switch: it is the historical
        // fixed-grain chunking, byte-identical to every banked run.
        // 5 cannot execute as requested at the 2025 twin-wall categories'
        // 100 s budget: the adaptive target is capped to 10% of the BI budget,
        // itself capped to 40% of remaining wall time (< 4 s). Keep only
        // values for which a scored receipt can prove the exact request ran.
        domain: Domain::Grid(&[0, 1, 2]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_chunk_secs"),
        why: "target wall per backward prefix pass; armed, the width is MEASURED per layer \
              rather than assumed. The earlier idx_9611 claim is exploratory only: its logs \
              were not retained, its binary carried unrelated dirty changes, and its cited \
              commit did not contain the experiment. Default 0 preserves fixed-grain history; \
              a non-zero value remains unmeasured until the versioned execution receipt proves \
              adaptive widths changed completed work on a clean sealed binary. Sizing is \
              tightness-only, but it changes where the wall-clock cut lands and therefore \
              remains verdict-affecting rather than safe-to-promote",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_TOPK",
        domain: Domain::Grid(&[256, 1024, 4096]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_topk"),
        why: "how many neurons get refined; unlike the chunk size this changes WHICH bounds move",
    },
    Axis {
        // #interm-alpha. Without this axis the typed key existed but the frozen
        // LeverSet had no config layer for it, so the scored path resolved the
        // declaration default 0 while `set_margin_row_interm_alpha_preset` set a
        // compat cell the scored path never reads — the 2026-08-27 preset-bridge
        // proof run showed `[backward-interm] armed` and NO `[interm-alpha]`
        // line with `interm_alpha: 2` in the loaded yaml. Delivery is the axis
        // entry, not the cell.
        name: "NY_MARGIN_ROW_INTERM_ALPHA",
        domain: Domain::Grid(&[0, 1, 2, 3, 4, 5]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.interm_alpha"),
        why: "per-START-NODE prefix alpha candidates in the backward-intermediate phase; \
              each candidate box is intersected shrink-only into the published box, so a \
              wrong dose costs wall (charged as ia_secs against the BI budget), never \
              soundness — but the wall it costs can lose a marginal proof, which keeps it \
              verdict-affecting",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.beta"),
        why: "beta ascent over split constraints; the direct arming gate for the refinements below, and the margin-row lane's analogue of what makes beta-CROWN beat alpha-CROWN",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_ETA",
        domain: Domain::Enum(&["0.25", "0.5", "1.0"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.beta_eta"),
        why: "dual ascent step; any iterate is a valid dual point, so this costs tightness, not soundness",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_ITERS",
        domain: Domain::Grid(&[1, 2, 4]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.beta_iters"),
        why: "ascent trials per domain; each is a full certified pass, so this is a direct per-domain cost multiplier",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_PERCOL",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.beta_per_column"),
        why: "give each selected failing output column independent split-dual state instead \
              of scoring one shared vector against only the worst column; execution credit \
              requires a live per-column ascent, not configuration presence",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_COLS",
        domain: Domain::Grid(&[1, 4, 8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.beta_columns"),
        why: "maximum live per-column beta width. The runtime clamps to 1..=32 and the \
              execution receipt records whether the requested width actually bound a \
              candidate set relative to the historical declaration default 8",
    },
    Axis {
        name: "NY_MARGIN_ROW_COL_RETIRE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.column_retire"),
        why: "inherit certificate-closed output columns and omit them from descendant \
              authoritative passes; evidence records new certificates, while search \
              credit requires an inherited narrowing event",
    },
    Axis {
        name: "NY_MARGIN_ROW_PARALLEL_PREBUILD",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.parallel_prebuild"),
        why: "prebuild the deterministic y-pack for each domain in an executable \
              multi-domain frontier batch; bound values are unchanged, while scheduling, \
              duplicate work and deadline reachability can move",
    },
    // PROMOTED ON LANDING (2026-08-24). #margin-row-domain-alpha shipped WITH
    // its typed keys and these axes. Scored runs now resolve the complete
    // family from the receipt-owning frozen LeverSet and carry plain policy
    // values through every tree. Default OFF and set by no shipped yaml, so
    // promotion is not arming.
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha"),
        why: "per-domain alpha re-optimization with parent warm start; the missing CONSUMER for \
              clip's tightened intermediate box (abc ships the two as a compounding pair) and \
              the only candidate whose deficit grows with DEPTH, which is the shape the \
              bit-identical depth-indexed gating curve needs",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
        domain: Domain::Grid(&[1, 2, 4]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha_iters"),
        why: "ascent trials per domain; each is a full certified pass, so this is a direct \
              per-domain cost multiplier as well as a choice of accepted alpha",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA",
        domain: Domain::Enum(&["0.125", "0.25", "0.5"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha_eta"),
        why: "initial normalized ascent step in [0,1] slope units; any iterate is a valid \
              relaxation, so this costs tightness rather than soundness, but it decides which \
              proposals the certified scorer accepts",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH",
        domain: Domain::Grid(&[0, 2, 4]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha_min_depth"),
        why: "SCHEDULE: skip the ascent above this split depth. The cost is uniform in depth \
              and the measured gain GROWS with depth (+0.0017 at depth 1 to +0.0137 at depth \
              12), so the floor buys the deep half of the curve without paying the shallow \
              half — against the countervailing risk that a shallow accepted alpha is what \
              every descendant inherits. 0 is the unscheduled baseline arm",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_STRIDE",
        domain: Domain::Grid(&[1, 2, 3]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha_stride"),
        why: "SCHEDULE: pay the ascent only where depth % stride == 0 and let the FREE warm \
              start carry the accepted alpha across the unpaid levels. Distinguishes 'the gain \
              is carried by inheritance' (stride divides cost, not gain) from 'each node's own \
              trials move it' (stride divides both). 1 is the unscheduled baseline arm",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_GAP",
        domain: Domain::Enum(&["0", "0.02", "0.05"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_alpha_gap"),
        why: "SCHEDULE: pay the ascent only within this much of the strict b > 0 closure \
              threshold. cifar100 fails by a COUNTING problem (frontier 18 -> 415 while the \
              bound improves at a proving row's rate), so passes spent on a domain at -0.30 \
              close nothing while the same passes at -0.004 can retire the domain and its \
              whole subtree. 0 reads as unlimited: the unscheduled baseline arm",
    },
    Axis {
        name: "NY_MARGIN_ROW_REBAKE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.rebake"),
        why: "joint forward re-bake: alpha-optimized gates re-baked into the forward tableau, \
              interm re-run, round-monotone accept on the certified root bound — the remaining \
              CPU root-quality capability now that one-shot interm is measured saturated",
    },
    Axis {
        name: "NY_MARGIN_ROW_REBAKE_ROUNDS",
        domain: Domain::Grid(&[1, 2, 3]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.rebake_rounds"),
        why: "re-bake round cap; each round is one alpha ascent + one forward rebuild + one \
              certified eval, so this multiplies root-phase cost against the instance deadline",
    },
    Axis {
        name: "NY_MARGIN_ROW_REBAKE_SECS",
        domain: Domain::Grid(&[15, 30, 60]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.rebake_secs"),
        why: "wall budget for the re-bake loop; expiry fails closed to the previously accepted \
              certified build, so a short budget publishes an earlier root rather than a wrong one",
    },
    Axis {
        // #bi-window (cct plan P5 step 1).
        name: "NY_MARGIN_ROW_BI_WINDOW",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_window"),
        why: "receptive-field-cropped backward-interm prefix passes; each cropped pair is a \
              certified pass of the unchanged engine on a net that computes exactly the seeded \
              pre-activations (never looser than the dense pair), so it cannot unsound a box, \
              but it changes how many boxes the phase can afford inside its wall budget and \
              therefore the published root",
    },
    Axis {
        // #bi-patch (per-column receptive-field prefix passes).
        name: "NY_MARGIN_ROW_BI_PATCH",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_patch"),
        why: "backward-interm and joint-refresh prefix pass pairs as per-column receptive-field \
              walks; each patch pair is the dense pair restricted to its support (never looser, \
              certified on its own), so it cannot unsound a box, but it changes how many boxes \
              the phase can afford inside its wall budget and therefore the published root",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_PATCH_RATIO",
        // Every token must survive the F64ClosedTrimmed { 0.05, 1.0 } parser,
        // pinned by `every_domain_token_survives_the_real_parser`.
        domain: Domain::Enum(&["0.25", "0.5", "1.0"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_patch_ratio"),
        why: "the patch-to-dense crossover of the patch walk; every setting is the same \
              certified pass over a support containing the true one, but it moves the phase's \
              wall and therefore how many boxes it tightens",
    },
    Axis {
        name: "NY_MARGIN_ROW_BI_PATCH_MB",
        domain: Domain::Grid(&[128, 256, 512]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.bi_patch_mb"),
        why: "per-pass working-set cap of the patch walk (batching only: the same columns, \
              grouped differently); it moves the phase's wall and memory and therefore how \
              many boxes it tightens",
    },
    Axis {
        // #joint-refresh (cct plan P5 step 2).
        name: "NY_MARGIN_ROW_JOINT_REFRESH",
        domain: Domain::Grid(&[0, 2, 4]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.joint_refresh"),
        why: "rounds of the backward-only joint intermediate refresh (re-derive every trunk box \
              from the current gates shrink-only, re-steer alpha on the refreshed head frame, \
              round-monotone accept on the certified root bound); an accepted round moves the \
              published root bound and spends root wall the tree needs",
    },
    Axis {
        name: "NY_MARGIN_ROW_JOINT_REFRESH_SECS",
        domain: Domain::Grid(&[15, 30, 60]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.joint_refresh_secs"),
        why: "wall budget for the joint refresh; a round the remainder cannot pay is refused \
              and expiry fails closed to the previously accepted certified root",
    },
    Axis {
        name: "NY_MARGIN_ROW_HEAD_GROUPS",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.head_groups"),
        why: "complete two- or four-neuron phase covers can improve published outer-root \
              class bounds through source-owned outward weak-dual replay; the fixed pass \
              spends at most five seconds within the original deadline. User-directed \
              default true, explicitly enabled by SafeNLP; false disables the whole \
              route. This default is not evidence of an added solved query",
    },
    Axis {
        name: "NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.head_directional_cuts"),
        why: "Source-certified directional cuts share one root relaxation and the \
              existing bounded deadline. Default true; explicit false disables only \
              this stage, while parent head_groups false disables the entire route. \
              Invalid replay or expiry keeps the incumbent; benefit remains unmeasured",
    },
    Axis {
        name: "NY_MARGIN_ROW_HEAD_PHASE_COVER",
        domain: Domain::Grid(&[0, 1, 2, 3, 4, 5, 6, 7]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.head_phase_cover"),
        why: "opt-in complete original-input Source phase cover can strengthen the root \
              only after every closed leaf is locally replayed; at most seven heads, \
              within the existing head-group and original instance deadlines. Zero and \
              absence are off, head_groups must also be armed, and no shipped yaml \
              sets this key; performance qualification remains pending",
    },
    Axis {
        // #root-certify-once (VIBECHECK_PLAN Phase-0.5 root alpha,
        // search-untrusted / certify-once).
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_certify_once"),
        why: "runs the root alpha ascent UNTRUSTED on a Parity scratch root and publishes ONE \
              certified evaluation of the chosen alpha only if it strictly beats the certified \
              incumbent; an accepted candidate moves the published root bound, and the phase \
              spends root wall the tree needs. Engagement evidence is the \
              `[root-certify-once] armed=1 iters=` terminal line; an armed run without it is VOID",
    },
    Axis {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS",
        domain: Domain::Grid(&[10, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_certify_once_iters"),
        why: "untrusted spec-phase iterations; they decide which alpha reaches the single \
              certified evaluation and how much of the phase's wall the search spends",
    },
    Axis {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH",
        domain: Domain::Grid(&[0, 5, 10]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_certify_once_refresh"),
        why: "untrusted joint-intermediate rounds (abc's per-iteration intermediate refresh on \
              the scratch); a round that tightens adds one certified refresh of the chosen \
              alpha's boxes, which can become the published root, and costs root wall",
    },
    Axis {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS",
        domain: Domain::Grid(&[10, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_certify_once_secs"),
        why: "wall budget for the certify-once phase; the certified evaluation is reserved \
              first and a budget that cannot pay a search step refuses, leaving the certified \
              incumbent root standing",
    },
    Axis {
        // #bbe-joint-root (ENGINE_PROGRAM M5a).
        name: "NY_MARGIN_ROW_ROOT_SEARCH",
        domain: Domain::Enum(&["ascent", "joint_f32"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_search"),
        why: "selects the UNTRUSTED search inside #root-certify-once: today's ascent \
              (bit-identical) or the injected ny-bbe joint-alpha root search (CROWN init, spec \
              prune, per-start-node alpha with in-loop refresh, early exits). Its slopes reach \
              the certified root only through the existing ingest gate and ONE certified \
              evaluation, but an accepted candidate moves the published root bound and the \
              search spends root wall. Engagement evidence is the \
              `[bbe] phase=joint_root iters= search_s= certify_s= root_lb=` line; an armed \
              `joint_f32` run without it is VOID",
    },
    Axis {
        name: "NY_MARGIN_ROW_ROOT_F32",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_f32"),
        why: "f32 COMPUTE for the bandwidth-bound root tableau convolutions; every \
              rounding perturbation is charged into the certified g_err concretize slack, \
              so the treatment can lose a proof but cannot manufacture one",
    },
    Axis {
        name: "NY_MARGIN_ROW_TABLEAU_F32",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.tableau_f32"),
        why: "f32 STORAGE for the bandwidth-bound root tableau walk (#tableau-f32): halves \
              streamed bytes and deletes the per-layer cast pass; every store is exact, \
              directed-up, or charged into the certified g_err concretize slack, so it can \
              only LOOSEN a published bound (costing a proof, never manufacturing one). \
              Inert — refused loudly — unless margin_row.root_f32 arms the f32 compute lane",
    },
    Axis {
        // #mr-ibp-seed (FRONTIER_PLAN 7.F2).
        name: "NY_MARGIN_ROW_IBP_SEED",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.ibp_seed"),
        why: "intersects every trunk box of the root tableau walk shrink-only with its certified \
              IBP box (after the forward-only bi_cal capture, before backward-interm); on \
              IBP-trained nets it moves the published root bound by whole units, and an \
              under-charged radius would be a false-unsat generator. Residual trunks decline \
              byte-identically. Engagement evidence is the `[mr-ibp-seed] layer=` line per \
              trunk relu; an armed run without it is VOID",
    },
    Axis {
        name: "NY_MARGIN_ROW_DOMAIN_STACK",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.domain_stack"),
        why: "One wide certified scoring pass across the whole frontier batch instead of one \
              pass per domain (#domain-stack). The certified pass is linear above R~64 columns \
              on a 45-60 ms fixed cost but the lane runs it at R = nf = 3-11, ~13x the marginal \
              cost, and scoring is 83% of the lane — so this moves the prover onto the linear \
              part of its own cost curve, and is the affordability precondition for a \
              per-domain joint alpha+beta re-optimisation. Column-local by construction: a \
              stacked score is BIT-IDENTICAL to the independent one (pinned by \
              domain_stacked_candidate_scores_match_independent_frontier_bit_exact), so it \
              selects a schedule, not a relaxation, and cannot move a published bound. \
              Verdict-affecting because it changes what the lane can finish inside its budget",
    },
    Axis {
        name: "NY_MARGIN_ROW_GPU_BACKWARD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.gpu_backward"),
        why: "certified error-free-transformation GPU backward; refuses rather than degrading on device failure, so an arm that finds the lane dead measures the CPU path",
    },
    Axis {
        name: "NY_CGAN_RECLAIM",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // THE BRIDGE HAS LANDED (2026-08-23). `commands::vnncomp` now calls
        // `set_cgan_reclaim_preset` from `preset::cgan_reclaim_from_preset`,
        // so the typed key is the real delivery route. Until that commit this
        // was `EnvOnly` and therefore DEAD CODE on every scored run — the note
        // at the top of this file names it as one of the mechanisms that
        // failure class caught. Still default OFF and no preset sets the key,
        // so arming remains a separate, measured decision.
        deliver: Deliver::PresetKey("bab.cgan_reclaim"),
        why: "cgan budget reclaim: cross-mode collection-cache serve + crown-partial \
              expiry retention + sequential CROWN-IBP expiry salvage + ConvTranspose2d \
              mid-step expiry salvage + forward-fallback publication past expiry, one \
              gate so the set cannot be measured split \
              (docs/CGAN_BUDGET_RECLAIM_2026-08-20.md)",
    },
    Axis {
        name: "NY_CGAN_POLISH",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.cgan_polish"),
        why: "near-miss survivor alpha polish on the grouped-disjunctive eager lane: \
              children inside the knife-edge band get one certified warm-alpha \
              re-bound with real SPSA steps (the preset's 1-iteration eager pass \
              performs zero optimizer steps), floor-merged so bounds only tighten",
    },
    Axis {
        name: "NY_CGAN_STACKED_BACKWARD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.cgan_stacked_backward"),
        why: "one stacked backward walk in place of per-target walks in the cgan collector",
    },
    Axis {
        name: "NY_SAFENLP_MIP_CYCLE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.safenlp_mip_cycle"),
        why: "SafeNLP full-cycle master arm: conditional MIP handoff (historical BaB \
              floor kept) + AY margin reframe on the marked unsafe row + serial \
              one-model solve, one gate so the composed cycle cannot be measured \
              split (docs/SAFENLP_CYCLE_2026-08-20.md)",
    },
    Axis {
        name: "NY_CGAN_STACKED_BUDGET_MB",
        domain: Domain::Grid(&[512, 4096, 32768]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.cgan_stacked_budget_mb"),
        why: "memory admission for that walk; decides only whether it is ATTEMPTED, since the fallback computes the same bounds",
    },
    Axis {
        name: "NY_CROWN_PARTIAL_FINITE_EXPIRY",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2. Default OFF and set
        // by no shipped yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.crown_partial_finite_expiry"),
        why: "switches two per-node partial CROWN sites from deadline PRESENCE to \
              deadline EXPIRY. Worth searching precisely because the audit predicts \
              a null unless the WHOLE root-cause-D set switches together — a zero \
              here is evidence about the split, not about the fix",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_LAMBDA",
        domain: Domain::Enum(&["0.5", "1.0", "2.0"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("margin_row.beta_lambda"),
        why: "Polyak relaxation factor; also the recovery ceiling the per-node \
              lambda memory grows back toward, so it sets both the step size and \
              how fast an accepted step is rewarded",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_POLYAK",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.beta_polyak"),
        why: "gap-targeted step rule vs the legacy sign step. Armed by default \
              inside the beta lane, so here the 0 arm is the TREATMENT and the \
              default is the incumbent -- the reverse of the rest of this family",
    },
    Axis {
        name: "NY_MARGIN_ROW_BETA_HEADS",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.beta_heads"),
        why: "head-split beta terms, armed by default inside the lane; the 0 arm \
              is trunk-only and exists to PRICE the head terms, not to revert them",
    },
    // --- ratchet conversions of committed raw reads (2026-08-20) ---
    Axis {
        name: "NY_NN4SYS_CTX_GROUP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch (see the module note
        // above). Default OFF and set by no shipped yaml, so this is a
        // no-op today: it makes the mechanism DELIVERABLE, not armed.
        deliver: Deliver::PresetKey("bab.nn4sys_ctx_group"),
        why: "context-grouped disjunctive refinement covers on nn4sys: one shared \
              tree over the hull of bit-identical clause boxes instead of one tree \
              per box. Fail-closed cover/disjointness proofs, but it changes WHICH \
              boxes get bounded, so coverage — hence verdicts — can move",
    },
    // --- #nn4sys-1d-cover (2026-09-11) ------------------------------------
    // Promoted TOGETHER. The master alone is reachable and INERT: the cover's
    // pass is all-or-nothing, so the reader's DEFAULT_DEPTH of 8 decides
    // whether an armed run publishes anything at all, and an unfenced cover
    // can take the whole screen slice and return `None`.
    Axis {
        name: "NY_NN4SYS_1D_COMPLETE_COVER",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-09-11. Default OFF. Armed by exactly ONE shipped yaml,
        // `configs/vnncomp25/nn4sys_1dcover_canary.yaml`, whose basename no
        // category can resolve to (`preset_basename_candidates` yields only
        // the lowercased category and its year-stripped base), so this is a
        // no-op on every scored run — it makes the mechanism DELIVERABLE and
        // MEASURABLE through the real route, not armed.
        deliver: Deliver::PresetKey("bab.nn4sys_1d_cover"),
        why: "exact one-axis f64 complete cover as a prepass in front of the \
              box-refinement clause screen. The census says 97.2-97.8% of non-dual \
              and 75.8-79.1% of DUAL nn4sys clauses are one-axis candidates, so this \
              is the only mechanism on the front that reduces the root box COUNT \
              rather than the per-box constant. All-or-nothing and fail-closed, but \
              it publishes per-clause discharges, so verdicts move",
    },
    Axis {
        name: "NY_NN4SYS_1D_COVER_DEPTH",
        // The reader clamps to HARD_MAX_DEPTH = 24 and requires
        // centered_start_depth <= depth, so every token here is admissible at
        // the read site as well as at the parser.
        domain: Domain::Grid(&[8, 12, 16, 20, 24]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.nn4sys_1d_cover_depth"),
        why: "per-axis bisection depth of the cover. NOT a refinement of the master \
              but a condition on it: the pass discards everything on one undecided \
              leaf, so too small a depth delivers a correct, reachable, INERT \
              mechanism — the alpha_iteration:1 / cols=8->cols=1 shape",
    },
    Axis {
        name: "NY_NN4SYS_1D_COVER_SLICE",
        // Every token must survive the F64ClosedTrimmed { 0.02, 0.5 } parser,
        // pinned by `every_domain_token_survives_the_real_parser`.
        domain: Domain::Enum(&["0.05", "0.10", "0.25", "0.50"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.nn4sys_1d_cover_slice"),
        why: "fraction of the caller's REMAINING deadline the cover may consume, and \
              the selector for the bounded partial mode. A budget only — but this \
              benchmark's own banked sat_escape_branch A/B took 12 dual rows to 1 by \
              re-slicing the same budget, so it is VerdictAffecting",
    },
    Axis {
        name: "NY_POLLABLE_FAER",
        // NOTE: the declaration is Text (legacy truthy parser: trimmed non-empty,
        // not "0", not case-insensitive "false"), so BOTH Bool tokens survive the
        // registry parser and mean what they look like at the read site.
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("bab.pollable_faer"),
        why: "pollable chunked-faer host GEMM for deadline-carrying batched Linear \
              CROWN; bound math is bit-identical (pinned parity test), so this is \
              throughput/coverage only. LATCHED in a OnceLock, so it can only be \
              varied across processes",
    },
    Axis {
        name: "NY_SEQ_WEIGHTED_BUDGET",
        // NOTE: OPT-OUT on a legacy truthy parser (`!= "0"`), declared Text with
        // effective default "1" so receipts/search reflect that the ARMED arm
        // ships (engaged by absence) while present raw tokens stay exact.
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        // PROMOTED 2026-08-23, #lever-reachability batch 2 (see the module note
        // above). Default OFF (or, where the declaration ships armed, a typed
        // kill switch that changes nothing while absent) and set by no shipped
        // yaml, so this is a no-op today.
        deliver: Deliver::PresetKey("bab.seq_weighted_budget"),
        why: "cost-weighted vs count-based per-candidate grants for the sequential \
              CROWN-IBP collection budget. A candidate that loses its grant degrades \
              to IBP and that degraded intermediate is published into every \
              downstream bound, so the choice moves bounds — hence verdicts — in \
              both directions",
    },
    // --- #dense-sparse-seed (2026-09-24) -----------------------------------
    // ARMED at 0.1 by configs/vnncomp26/challenging_certified_training_2026.yaml
    // (and its patches-wiring canary, which must differ from it by exactly two
    // other keys). Not category-scoped in the implementation: any sequential
    // CROWN-IBP collection with a Dense-carrier ReLU-successor target reads it.
    Axis {
        name: "NY_CROWN_PARTIAL_DENSE_MIN_STABLE",
        // Every token must survive the F64ClosedTrimmed { 0.0, 1.0 } parser,
        // pinned by `every_domain_token_survives_the_real_parser`. "0.9" is the
        // historical literal (the declaration default); "0.1" is
        // alpha-beta-CROWN's `minimum_sparsity: 0.9` in ny's convention.
        domain: Domain::Enum(&["0.1", "0.25", "0.5", "0.9"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.crown_partial_dense_min_stable"),
        why: "minimum IBP-stable fraction at which a Dense-seeded CROWN-IBP partial \
              walk with a ReLU successor seeds only its IBP-unstable rows. Untracked \
              stable neurons publish the certified IBP over the tightened prior \
              instead of initial-IBP-intersect-CROWN; neither contains the other, so \
              their magnitudes, and every later bound chained from them, move in \
              BOTH directions. The successor ReLU's stable set can only grow (to \
              rounding). The walk's row count falls with the unstable fraction, and \
              that decides whether the collection finishes inside the budget at all",
    },
    // --- #lsnc-attack-twin (2026-09-24) -------------------------------------
    // Post-BaB exact-gradient PGD on a point-folded ATTACK twin. Not category
    // scoped in the implementation: it engages on any net whose real graph is
    // outside the point-VJP fragment and whose constant frontier ONNX Runtime
    // can fold; lsnc_relu is the motivating family.
    Axis {
        name: "NY_ATTACK_POINT_FOLDED_TWIN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("attack.point_folded_twin"),
        why: "admits the post-BaB twin lane, which can convert unknown to sat (every \
              candidate through the unchanged trusted-ORT gate) and takes the post-BaB \
              tail the ordinary lanes would otherwise own, so it changes which \
              counterexamples are reachable within the budget. Engagement evidence is the \
              lane's own `[lsnc-attack-twin] armed folded=` line plus its terminal `done` \
              line; a run without them is VOID, not negative",
    },
    // --- #lsnc-lane (2026-09-24) --------------------------------------------
    // The #lowdim-prover lane's LANE-PRIVATE interval twin: a second,
    // interval-folded load of a model whose exact load left OpaqueSkip nodes,
    // readable only by the lowdim lane. Not category scoped in the
    // implementation; lsnc_relu is the motivating family. The global carrier
    // NY_CONST_MATMUL_INTERVAL stays in UNSAFE_AXES, EnvOnly.
    Axis {
        name: "NY_LOWDIM_INTERVAL_TWIN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_interval_twin"),
        why: "admits the lowdim lane on a model the exact loader leaves unboundable, so \
              the lane can convert unknown to unsat (and hand a candidate point to the \
              trusted-ORT confirmer) on rows it otherwise declines; a bound path on a \
              family with a false-unsat history. Engagement evidence is the \
              `[lowdim-twin] armed=1 ... attached=1` line and the lane's own \
              `[lowdim] armed=1 entry=... interval_consts=K` with K > 0; a run \
              without them is VOID, not negative",
    },
    // #lsnc-lane stage 2 (2026-09-25): the lowdim lane's JOINT QUADRATIC bound
    // at ReduceSum(Mul) sites. Not category scoped in the implementation (a
    // program without such a site is unaffected); lsnc_relu's lane twin is the
    // only lowered program that has one.
    Axis {
        name: "NY_LOWDIM_JOINT_QUADRATIC",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_joint_quadratic"),
        why: "replaces per-product McCormick at the lowdim lane's ReduceSum(Mul) sites \
              with one quadratic over the input box (CROWN operand certificates, exact \
              vertex maximisation, eigen residual charged), which the numpy gate \
              measured as 2e5-1.4e6 instead of ~5.6e7 domains per lsnc_relu unsat \
              row — a new bound path on a family with a false-unsat history. \
              Engagement evidence is the grouped entry's `[lowdim] joint sites=2 \
              armed=1 root dv_ub=.. v_lb=.. site[..]=[..] planes=1` line (planes=1 at \
              every site); `armed=1` with planes=0 is VOID, not negative",
    },
    // #lsnc-lane stage 3 (2026-09-25): the lowdim lane's SEARCH step. Not
    // category scoped in the implementation; armed only in the lsnc
    // measurement configs.
    Axis {
        name: "NY_LOWDIM_SPLIT",
        domain: Domain::Enum(&["widest", "gradient", "sb", "sbwr"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_split"),
        why: "chooses the coordinate the lowdim lane bisects: the numpy gate measured the \
              aspect-guarded sbwr rule at 2.0e5-1.4e6 domains per lsnc_relu unsat row \
              against 3.1e5-3.7e6 for plain sb with the same stack, so the tree shape \
              decides whether a deadline-bounded row converts. Engagement evidence is \
              `[lowdim-split] armed=1 value=..` and the lane's `[lowdim] lane-config \
              split=..` line; a run whose lane-config shows the default is VOID",
    },
    Axis {
        name: "NY_LOWDIM_CLIP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_clip"),
        why: "relaxed Clip-and-Verify of each lowdim box by its own passes' planes, \
              outward rounded: a new way to CLOSE a box on a family whose false unsat \
              came from clipping with reused planes (807bf511). Engagement evidence is \
              the per-clause `[lowdim] clause=.. clip_closed=.. clip_shrunk=..` counters; \
              armed with both zero is VOID, not negative",
    },
    Axis {
        name: "NY_LOWDIM_PARALLEL",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_parallel"),
        why: "decides each lowdim domain inside its rayon task instead of serially after \
              the bound: outcome-identical by construction, but lsnc_relu is \
              throughput-bound (1e5+ domains/s needed in 25 s), so it moves the \
              deadline-bounded verdict. Engagement evidence is `[lowdim] lane-config \
              parallel=full threads=T` with T > 1",
    },
    // #mr-errlane (margin-row throughput plan B3, 2026-09-26). `audit` is a
    // declared token but deliberately NOT a search value: it publishes the
    // `off` lane and doubles pass cost to MEASURE the envelopes' looseness.
    Axis {
        name: "NY_MARGIN_ROW_ERR_ENVELOPE",
        domain: Domain::Enum(&["off", "envelope", "per_channel"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.err_envelope"),
        why: "carries the margin-row certified coefficient-error lane as ONE dominating \
              envelope per row (`envelope`, passes with R >= 16) or per channel \
              (`per_channel`, every pass) instead of an n x R matrix. The error lane is \
              exactly half of every certified pass (B1 probe, all three nets), so this \
              moves wide passes toward half cost; dominance is monotone induction and \
              the coefficient lane is untouched, so it only LOOSENS the certified \
              penalty — costing proofs, never manufacturing one. Engagement evidence \
              is `[mr-errlane] done ... env_passes=` (or `pc_passes=`) > 0; an armed run \
              with zero is VOID",
    },
    // #frozen-dual-state (2026-09-27): V1 stage A of the frozen dual-ascent BaB;
    // #frozen-dual-bab: stage B makes it arm the certified search tier.
    Axis {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_bab"),
        why: "arms the certified frozen alpha-zonotope dual BaB as a tier between the \
              margin-row root and its tree: per open class, stage A's certified state \
              (binary64, explicit gamma charges, certified fold errors, outward rounding), \
              the static box+halfspace order, and a level-synchronous BFS whose nodes are \
              pruned only on a certified node bound > 0 at clamped dual points. A class \
              closes only when every leaf is certified and the level ledger closes, and it \
              then leaves the tree's conjunction — hence verdict-affecting; the tier's wall \
              also comes out of the tree's window. Engagement evidence is the \
              `[frozen-dual] armed=1 objectives=` summary; an armed run without it is VOID. \
              User-directed CIFAR100/TinyImageNet presets arm its bounded automatic tier",
    },
    Axis {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP",
        domain: Domain::Grid(&[250_000, 2_000_000, 8_000_000]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_bab_node_cap"),
        why: "nodes one objective's frozen-dual search may evaluate before it is handed \
              back open; a cap can never close a class, but it decides which classes the \
              tier closes inside the budget and how much wall it spends before the tree",
    },
    Axis {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS",
        // 0 selects bounded automatic mode (4 s / 5%, 8192 nodes per objective,
        // 256 MiB state); dense estimates >=64 MiB select one objective,
        // otherwise at most two. Positive requests retain manual mode.
        domain: Domain::Grid(&[0, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_bab_secs"),
        why: "wall the frozen-dual tier may take from the tree's window; it decides how \
              many classes the tier can close and how much budget the tree keeps for the \
              rest, so it moves verdicts in both directions on the lane that decides \
              cifar100 and tinyimagenet",
    },
    // #breakpoint-dual: bounded category-default refinement of the frozen-dual tier.
    Axis {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS",
        domain: Domain::Grid(&[0, 4, 8, 16, 32]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_refine_steps"),
        why: "optional untrusted breakpoint dual refinement inside the frozen-dual tier; \
              every candidate is scored by the unchanged certified node evaluator and \
              retained only on strict certified improvement, while the cover ledger is \
              unchanged. The work changes pruning and spends the tier's wall; generic \
              default 0 keeps the historical evaluator, while the user-directed shipped \
              CIFAR100 and TinyImageNet presets request 4 steps in bounded automatic mode",
    },
    Axis {
        // #fd-shared-state (ENGINE_PROGRAM M3).
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE",
        domain: Domain::Enum(&["per_class", "shared"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_bab_state"),
        why: "how the frozen-dual tier builds its per-class states: one full build per \
              class inside its share (per_class, bit-identical) or ONE shared trunk of rows \
              built before the shares plus a one-column margin tail per class (shared). \
              Shared slopes change every class's state and root (the shared-lambda root \
              loss) and move the build out of the class shares, so which classes the tier \
              builds, searches and closes moves in both directions. Engagement evidence is \
              the `[frozen-dual] trunk state=shared lambda=` line; an armed `shared` run \
              without it is VOID",
    },
    // #wide-fd-search (ENGINE_PROGRAM M4, 2026-09-29).
    Axis {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH",
        domain: Domain::Enum(&["f64", "cpu_f32", "cuda", "auto"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.frozen_dual_bab_search"),
        why: "selects which search scores the frozen-dual tier's nodes: the tier's own f64 \
              search (bit-identical) or an injected UNTRUSTED backend (ny-bbe CPU f32 SIMD, \
              ny-cuda) whose proposed dual points the tier certifies in f64 with its own \
              evaluator, splitting every node no proposal certifies. Every prune stays \
              certified, but the search decides which nodes prune inside the budget, hence \
              which classes close. Engagement evidence is the `[bbe] armed=1 backend= \
              probe=ok` line and `search=cpu_f32 claimed=` on the class lines; an armed \
              non-`f64` run without them is VOID",
    },
    // #wave-beta-bab (ENGINE_PROGRAM M6, 2026-09-29).
    //
    // `wave_cuda` (M6's device half) searches on the device, bit for bit the
    // CPU f32 search; a refused device falls through to it. A run the device
    // searched ALONE reads exactly `backend=cuda ` (trailing space) on the
    // `[bbe] phase=wave` line; `backend=cuda->cpu_f32` starts with the same
    // bytes and was switched to the CPU search by a device fault (it counts as
    // a CPU run for engagement). This note is a COMMENT on
    // purpose: `source_frontier.rs` fingerprints every lever-name literal by
    // the 12 tokens on each side, string contents included, and the `why`
    // string below sits 8 tokens before the next axis's
    // `NY_MARGIN_ROW_TREE_WAVE_BATCH` literal, so editing its text
    // re-anchors that registered frontier row. Comments are not tokens.
    Axis {
        name: "NY_MARGIN_ROW_TREE_MODE",
        domain: Domain::Enum(&["legacy", "wave", "wave_cuda"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.tree_mode"),
        why: "selects the margin-row outer frame's tree engine: today's best-first tree \
              (bit-identical) or waves of injected UNTRUSTED output-only beta-CROWN search \
              whose claims a private pool certifies with the lane's own Outward evaluator \
              (frozen -> clip -> y-pack ladder), splitting every domain it does not claim \
              and every claim it cannot certify. Every close stays certified and passes the \
              path-based cover gate, but the engine replaces the whole search trajectory on \
              the lane that decides cifar100 and tinyimagenet. Engagement evidence is the \
              `[bbe] phase=wave armed=1 backend=` line; an armed `wave` run without it is \
              VOID",
    },
    Axis {
        name: "NY_MARGIN_ROW_TREE_WAVE_BATCH",
        domain: Domain::Grid(&[64, 256, 1024]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.tree_wave_batch"),
        why: "domains per wave of the wave tree: the width decides which domains are \
              searched, claimed and certified inside the budget, so it moves verdicts in \
              both directions although every close stays certified",
    },
    Axis {
        name: "NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS",
        // 0 is the SHIPPED DEFAULT (half of the lane's rayon threads).
        domain: Domain::Grid(&[0, 4, 10]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.tree_wave_cert_threads"),
        why: "threads of the wave tree's private certify pool; the tree is independent of \
              it, but the certify wall per wave is not, so it decides how many claims \
              certify inside the budget",
    },
    // #lowdim-bundle (2026-09-27): four dark keys of the #lowdim-prover lane.
    // Not category scoped in the implementation; the motivating rows are
    // dist_shift_2023 index2734 (varying dims + Sigmoid), acasxu 3_3 prop_2
    // (boundary alpha), lsnc_relu row 58 (leaf attack) and the nn4sys
    // mscn_2048d_dual residual clauses (shared worklist).
    Axis {
        name: "NY_LOWDIM_VARYING_DIMS",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_varying_dims"),
        why: "admits the lowdim lane on a box wider than its 16-dimension cap whose other \
              coordinates are pinned, and lowers Sigmoid to rigorous planes: a new bound \
              path that can convert unknown to unsat on rows the lane declined \
              (dist_shift_2023's 8 varying of 792 inputs). Engagement evidence is every \
              lane entry's `[lowdim-varying] armed=1 status=folded d_free=..`; an armed \
              run on a pinned box without status=folded is VOID, not negative",
    },
    Axis {
        name: "NY_LOWDIM_ALPHA_ITERS",
        domain: Domain::Grid(&[0, 5, 10, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_alpha_iters"),
        why: "optimises the lowdim lane's lower ReLU slopes on open leaves within 10 of \
              their threshold (Vibecheck's acasxu wide band took 3_3 prop_2 from 1.88M \
              leaves/unknown to 431k/verified), so it closes leaves the historical bound \
              leaves open and moves the deadline-bounded verdict. Engagement evidence is \
              the per-run `[lowdim-alpha] done iters=.. domains=..` with domains > 0; \
              armed with domains=0 is VOID, not negative",
    },
    Axis {
        name: "NY_LOWDIM_LEAF_ATTACK_EVERY",
        domain: Domain::Grid(&[0, 3, 10, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_leaf_attack_every"),
        why: "runs PGD on the lowdim lane's worst open leaves every N waves; a hit leaves \
              the lane as a candidate that the entry's trusted forward and the unchanged \
              trusted-oracle gate decide, so it can turn unknown into sat (lsnc_relu row \
              58) and ends the lane for a clause whose candidate is rejected. Engagement \
              evidence is the per-run `[lowdim-leaf-attack] done rounds=.. evals=..`; \
              armed with rounds=0 and skipped=0 never reached a due wave",
    },
    Axis {
        name: "NY_LOWDIM_SHARED_WORKLIST",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_shared_worklist"),
        why: "proves the lowdim fold's families on one round-robin worklist without the \
              per-clause 200k-domain cap instead of family by family, which changes which \
              nn4sys dual clauses a deadline-bounded run closes. Engagement evidence is \
              the fold's `[lowdim-shared] done families=.. waves=..`; an armed fold run \
              without it is VOID, not negative",
    },
    Axis {
        // #root-box-source (ENGINE_PROGRAM M5b). Appended LAST so no existing
        // literal site's structural anchor moves.
        name: "NY_MARGIN_ROW_ROOT_BOX_SOURCE",
        domain: Domain::Enum(&["tableau", "patch_backward"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("margin_row.root_box_source"),
        why: "selects how every root walk (initial, rebake round, epoch rebuild) builds its \
              trunk boxes: the forward tableau (bit-identical) or a certified IBP reference \
              plus a certified patch-backward pass over the reference-unstable neurons, \
              intersected shrink-only. It replaces the source of every root box and moves the \
              published root bound and the root wall. Engagement evidence is the \
              `[root-box] source=patch_backward ... certify_s=` line; an armed run without it \
              (or with `source=tableau declined=`) is VOID",
    },
    // #router-r1 (ROUTER_PROGRAM R1). Appended LAST so no existing literal
    // site's structural anchor moves. The domain omits `on`: automated search
    // never picks the acting mode, which is a compiled-default move after a
    // both-year moat (§2.5), never a searched value.
    //
    // #router-r2: the `why` below predates R2 and is kept byte-for-byte (it
    // sits inside the next literal's structural-anchor window). Since R2 `on`
    // ACTS through acting set 1 (lane deadlines, the publish grace, rule-owned
    // caps), so the parenthetical "`on` is effectively shadow" no longer holds
    // for `on`; it still holds for the searched domain, which omits `on`, so
    // this axis still cannot change a verdict. The declaration's own doc and
    // its MoatRisk (High since R2) carry the acting mode.
    Axis {
        name: "NY_ROUTER_MODE",
        domain: Domain::Enum(&["off", "shadow"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("router.mode"),
        why: "the verdict-free runtime router in shadow: it extracts static facts, logs a \
              pure hashed plan, and evaluates its checkpoints retrospectively AFTER the \
              verdict is published. It emits no value and moves no cap (`on` is effectively \
              shadow while ACTING_RULES is empty), so it cannot change a verdict; it costs \
              wall clock only (kill line 0.5% on 20 s rows). Engagement evidence is the \
              `[router] v1 mode=shadow` and `[router] done faults=` pair; a shadow run \
              without both is VOID",
    },
    Axis {
        name: "NY_OPTIMIZER_MODE",
        domain: Domain::Enum(&["off", "shadow"]),
        class: Class::SafeToSearch,
        deliver: Deliver::PresetKey("optimizer.mode"),
        why: "the runtime optimizer in shadow: after publication it logs what each O-STOP rule \
              would have done (`[optimizer] rule= would=`); it changes no iterate, cap or \
              clamp (`on` is effectively shadow while ACTING_OPTIMIZER_RULES is empty), so it \
              cannot change a verdict and costs wall clock only",
    },
    Axis {
        // #m8-input-split-batch (ENGINE_PROGRAM M8a). Appended LAST so no
        // existing literal site's structural anchor moves.
        name: "NY_LOWDIM_BATCH_ENGINE",
        domain: Domain::Enum(&["legacy", "bbe"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.lowdim_batch_engine"),
        why: "chooses the lowdim lane's wave bounder: its own f64 forms on every domain \
              (bit-identical) or an injected untrusted forward-linear search whose claims the \
              lane re-bounds in f64 and whose other leaves it splits, which changes the tree \
              and the throughput a deadline-bounded verdict depends on. Engagement evidence \
              is `[bbe] phase=input_split lane=lowdim scheduler=.. fallback=none ... \
              m8-input-split-batch/v1` with calls > 1 and certified > 0; an armed run whose \
              line shows another fallback is VOID, not negative",
    },
    Axis {
        // #m8-input-split-batch (ENGINE_PROGRAM M8a).
        name: "NY_INPUT_SPLIT_ENGINE",
        domain: Domain::Enum(&["legacy", "bbe"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.branching.input_split.engine"),
        why: "chooses the lowdim lane's multi-clause scheduler: one tree per clause, or one \
              disjunction tree per shared box (a leaf serves every clause still open on it) \
              and the fold's shared worklist, which changes which clauses a deadline-bounded \
              run proves. Engagement evidence is the grouped entry's `[lowdim-disjunction] \
              done clauses=..` line or the fold's `[lowdim-shared] done`; an armed run \
              without one is VOID, not negative",
    },
    // #vit-v3-bankable: the multi-objective cover certification gate.
    Axis {
        name: "NY_MO_COVER_CERTIFY",
        domain: Domain::Enum(&["off", "ledger", "replay"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.mo_cover_certify"),
        why: "downgrade-only certification of the disjunctive graph multi-objective \
              BaB: `ledger` withdraws a Verified whose per-row leaves do not cover the \
              root (and lets a proved one publish from a token), `replay` also withdraws \
              one whose witness leaves do not re-certify without reused lane state; it \
              never creates or tightens a verdict, but it can turn a published unsat \
              into unknown, so every candidate must clear the moat gate",
    },
];

/// Prerequisite and exclusion edges.
///
/// Every one of these corresponds to a literal `&&`, `?` or early return in
/// the engine. A sample that violates one is not a cheap experiment — it is a
/// 100 s measurement of the baseline, mislabelled as a treatment.
///
const EDGES: &[Edge] = &[
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "sound_gpu_gate.rs:root_joint_deadline_lane_requested — both governed default-dark gates must be armed before the finite-deadline capability is requested",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "input_split/root_bounds.rs and multi_objective/root_phases.rs — the wall request is read only inside the master transaction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "root_bounds.rs/root_phases.rs — iteration count is read only inside the master transaction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT"),
        site: "interm_refine.rs:root_joint_effective_iters — every live root route carries finite authority, so the default base-fold-only policy forces zero optimizer iterations",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "root_bounds.rs/root_phases.rs — the Adam step is read only inside the master transaction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT"),
        site: "interm_refine.rs:root_joint_effective_iters — with ascent disabled the effective iteration count is zero and learning rate is inert",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "interm_refine.rs:root_joint_tighten_options — the row ceiling is consumed only by the armed root-joint target loop",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "interm_refine.rs:prep_resnet_domain_ext — frozen stop changes only the armed root-joint extraction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "interm_refine.rs:prep_resnet_domain_ext — pure-chain admission changes only the armed root-joint extraction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
        requires: Requirement::NotArmed("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP"),
        site: "interm_refine.rs:effective_allow_pure_chain — frozen-stop independently implies pure-chain admission, so arming both cannot attribute engagement to the dedicated pure-chain knob",
    },
    Edge {
        child: "NY_BN_GPU_EXTRACT",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "interm_refine.rs:root_joint_tighten_options — BatchNorm extraction is currently consumed only by the armed root-joint lane",
    },
    Edge {
        child: "NY_ROOT_JOINT_MIN_REMAINING_SECS",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "multi_objective/root_phases.rs:root_joint_interm_alpha — the admission floor is consulted only inside the armed graph-route transaction",
    },
    Edge {
        child: "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        requires: Requirement::Armed("NY_ROOT_JOINT_INTERM_ALPHA"),
        site: "root_bounds.rs/root_phases.rs — both target selectors are reached only inside the master transaction",
    },
    Edge {
        child: "NY_PGD_VJP_K",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched{,_disj}.rs — width is consulted only after batched exact-VJP admission",
    },
    Edge {
        child: "NY_PGD_LEGACY_LANES",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — the sequential fallback has no restart lanes",
    },
    Edge {
        child: "NY_PGD_LEGACY_LANES",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive exact-VJP policy is unreachable when generic batched routing is selected",
    },
    Edge {
        child: "NY_PGD_ODI_STEPS",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — ODI rows exist only inside the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_ODI_STEPS",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive ODI is unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_ODI_ETA",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — ODI eta is consumed only inside the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_ODI_ETA",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive ODI is unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_ODI_ETA",
        requires: Requirement::GreaterThan("NY_PGD_ODI_STEPS", 0),
        site: "graph_pgd_vjp_batched.rs — eta sizes only ODI steps, and the loop is skipped at zero",
    },
    Edge {
        child: "NY_PGD_VJP_STEPS",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — wave depth is consumed only inside the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_VJP_STEPS",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive exact-VJP waves are unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_LANE_WINDOW",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — recycling state exists only inside the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_LANE_WINDOW",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive lane recycling is unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_EXPLOIT",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — exploit recycling exists only inside the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_EXPLOIT",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive exploit recycling is unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_HINGE_TAU",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched.rs — tau contributes only to rows built by the admitted wide driver",
    },
    Edge {
        child: "NY_PGD_HINGE_TAU",
        requires: Requirement::Armed("NY_PGD_EXACT_BATCHED"),
        site: "graph_pgd.rs — conjunctive tau-hinge rows are unreachable under generic batched routing",
    },
    Edge {
        child: "NY_PGD_DISJ_ODI",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched_disj.rs — the ODI phase runs only inside the admitted wide disjunctive driver",
    },
    Edge {
        child: "NY_PGD_DISJ_ODI",
        requires: Requirement::NotEquals("NY_PGD_ODI_STEPS", "0"),
        site: "graph_pgd_vjp_disj_odi.rs:run_odi_phase — zero ODI steps skips the phase, so the arm would re-measure the baseline",
    },
    Edge {
        child: "NY_PGD_DISJ_FOCUS_LANES",
        requires: Requirement::Armed("NY_PGD_VJP_BATCH"),
        site: "graph_pgd_vjp_batched_disj.rs — the focus wave runs only inside the admitted wide disjunctive driver",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_CAP",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — an explicit master kill returns before budget sizing; absence remains contextual and may execute",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_CAP",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/vnncomp.rs:CommittedAttackSlice::from_today — lossy knapsack caps deliberately emit generic evidence, so numeric treatments cannot be banked under that authority",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_AUTO_CAP",
        requires: Requirement::Absent("NY_UPFRONT_ATTACK"),
        site: "commands/vnncomp.rs:try_upfront_falsify — the automatic cap applies only to contextual AutoDisjunction; either explicit master token selects a different route",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_AUTO_CAP",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK_CAP", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — a zero ordinary wall cap fails the private base minimum before automatic-disjunction clamping",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_AUTO_CAP",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/vnncomp.rs:CommittedAttackSlice::from_today — lossy knapsack caps deliberately emit generic evidence, so numeric treatments cannot be banked under that authority",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_FRAC",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — an explicit master kill returns before fractional budget sizing; absence remains contextual and may execute",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_FRAC",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK_CAP", "0"),
        site: "commands/vnncomp.rs:upfront_attack_base_budget — a zero wall cap is selected before the minimum-budget return, so changing the fraction cannot affect execution",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_FRAC",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/vnncomp.rs:CommittedAttackSlice::from_today — lossy knapsack caps deliberately emit generic evidence, so numeric treatments cannot be banked under that authority",
    },
    Edge {
        child: "NY_UPFRONT_ATTACK_PROBE_FIRST",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — the probe consultation is immediately declined by an explicit master kill; absence retains contextual admission",
    },
    Edge {
        child: "NY_BNN_ATTACK_CAP",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — an explicit master kill returns before Sign-network budget sizing",
    },
    Edge {
        child: "NY_BNN_ATTACK_CAP",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK_CAP", "0"),
        site: "commands/vnncomp.rs:upfront_attack_base_budget — the ordinary base must pass the minimum gate before the Sign-network policy is consulted",
    },
    Edge {
        child: "NY_BNN_ATTACK_CAP",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/vnncomp.rs:CommittedAttackSlice::from_today — the knapsack floors the upfront cap to whole seconds and deliberately emits generic evidence, so a BNN numeric treatment cannot be banked under that authority",
    },
    Edge {
        child: "NY_BNN_ATTACK_FRAC",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK", "0"),
        site: "commands/vnncomp.rs:try_upfront_falsify — an explicit master kill returns before Sign-network budget sizing",
    },
    Edge {
        child: "NY_BNN_ATTACK_FRAC",
        requires: Requirement::NotEquals("NY_UPFRONT_ATTACK_CAP", "0"),
        site: "commands/vnncomp.rs:upfront_attack_base_budget — the ordinary base must pass the minimum gate before the Sign-network policy is consulted",
    },
    Edge {
        child: "NY_BNN_ATTACK_FRAC",
        requires: Requirement::NotEquals("NY_BNN_ATTACK_CAP", "0"),
        site: "commands/vnncomp.rs:upfront_effective_budget_decision — a zero Sign-network cap suppresses every fractional candidate before the ordinary max",
    },
    Edge {
        child: "NY_BNN_ATTACK_FRAC",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/vnncomp.rs:CommittedAttackSlice::from_today — the knapsack floors the upfront cap to whole seconds and deliberately emits generic evidence, so a BNN numeric treatment cannot be banked under that authority",
    },
    Edge {
        child: "NY_MARGIN_ROW_ALPHA_ITERS",
        requires: Requirement::Armed("NY_MARGIN_ROW_ALPHA_OPT"),
        site: "margin_row/alpha_opt.rs:maybe_optimize_root_alpha — the frozen \
               iteration count is consumed only after the optimizer arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_ALPHA_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_ALPHA_OPT"),
        site: "margin_row/alpha_opt.rs:maybe_optimize_root_alpha — the frozen wall \
               request is consumed only after the optimizer arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_ALPHA_LR",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_ALPHA_OPT",
            "NY_MARGIN_ROW_REBAKE",
            "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE",
        ]),
        site: "margin_row/alpha_opt.rs:maybe_optimize_root_alpha, \
               margin_row/rebake.rs:rebake_root and \
               margin_row/root_certify_once.rs:certify_once_root — every root-ascent owner \
               (the untrusted certify-once ascent included) consumes the shared Adam step",
    },
    Edge {
        child: "NY_MARGIN_ROW_ALPHA_COLS",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_ALPHA_OPT",
            "NY_MARGIN_ROW_REBAKE",
            "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE",
        ]),
        site: "margin_row/alpha_opt.rs:maybe_optimize_root_alpha, \
               margin_row/rebake.rs:rebake_root and \
               margin_row/root_certify_once.rs:certify_once_root — every root-ascent owner \
               (the untrusted certify-once ascent included) consumes the shared steering width",
    },
    Edge {
        child: "NY_MARGIN_ROW_SPEC_ALPHA_WIDTH",
        requires: Requirement::Armed("NY_MARGIN_ROW_SPEC_ALPHA"),
        site: "margin_row/spec_alpha.rs:refine — the hardest-class width is \
               consumed only after the axis arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_SPEC_ALPHA_CANDS",
        requires: Requirement::Armed("NY_MARGIN_ROW_SPEC_ALPHA"),
        site: "margin_row/spec_alpha.rs:refine — the ladder dose is consumed \
               only after the axis arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS",
        requires: Requirement::Armed("NY_MARGIN_ROW_SPEC_ALPHA"),
        site: "margin_row/spec_alpha.rs:refine — the gradient round count is \
               consumed only after the axis arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES",
        requires: Requirement::Armed("NY_MARGIN_ROW_SPEC_ALPHA"),
        site: "margin_row/spec_alpha.rs:refine — the committed-exception cap is \
               consumed only after the axis arm passes",
    },
    Edge {
        child: "NY_MARGIN_ROW_ALPHA_GUARD",
        requires: Requirement::Armed("NY_MARGIN_ROW_ALPHA_OPT"),
        site: "margin_row/alpha_opt.rs:maybe_optimize_root_alpha — the frozen \
               published-frame guard is consumed only after the optimizer arm passes",
    },
    Edge {
        child: "NY_FALSIFY_PORTFOLIO_SECONDS",
        requires: Requirement::Armed("NY_FALSIFY_PORTFOLIO"),
        site: "commands/vnncomp/falsify_portfolio.rs:portfolio_wall_cap — consulted only \
               after the portfolio lane is armed",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLOSURE_THRESH",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
        site: "margin_row/closure.rs:armed — tuning is consulted only by the armed policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLOSURE_WARMUP",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
        site: "margin_row/closure.rs:armed — tuning is consulted only by the armed policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLOSURE_WINDOW",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
        site: "margin_row/closure.rs:armed — tuning is consulted only by the armed policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLOSURE_MAX_TIER",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
        site: "margin_row/closure.rs:armed — tuning is consulted only by the armed policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLIP",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLIP_ROWS"),
        site: "margin_row/bab.rs:clip_tighten_pack — returns when no clip rows were retained",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLIP_INTERM",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLIP_ROWS"),
        site: "margin_row/engine.rs:clip_tighten_gates — every layer without retained clip \
               rows is skipped",
    },
    Edge {
        child: "NY_MARGIN_ROW_CLIP_TOPK",
        requires: Requirement::Armed("NY_MARGIN_ROW_CLIP_INTERM"),
        site: "margin_row/engine.rs:clip_tighten_gates — top-K is read only inside the \
               intermediate-clip path",
    },
    Edge {
        child: "NY_LANE_BUDGET_ALLOCATOR",
        requires: Requirement::NotArmed("NY_LANE_VALUE_SCHEDULER"),
        site: "commands/lane_allocation.rs:allocator_admitted — the scheduler owns the BNN \
               lane caps when armed",
    },
    Edge {
        child: "NY_LANE_BUDGET_FRAME",
        requires: Requirement::NotArmed("NY_LANE_BUDGET_ALLOCATOR"),
        site: "commands/lane_allocation.rs:frame_admitted — the allocator is the stronger \
               cap authority",
    },
    Edge {
        child: "NY_LANE_BUDGET_FRAME",
        requires: Requirement::NotArmed("NY_LANE_VALUE_SCHEDULER"),
        site: "commands/lane_allocation.rs:frame_admitted — the scheduler owns the BNN lane caps",
    },
    // Direct beta is first deliberately: it is repair's deterministic choice
    // only when neither route is already armed.
    Edge {
        child: "NY_MARGIN_ROW_BETA_LAMBDA",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/beta.rs:enabled — beta ascent is live through either the direct \
               beta arm or closure-adaptive tier >= beta",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_LAMBDA",
        requires: Requirement::Armed("NY_MARGIN_ROW_BETA_POLYAK"),
        site: "margin_row/bab.rs:beta_ascent — lambda sizes only the Polyak arm; the legacy sign-step ignores it",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_POLYAK",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/beta.rs:enabled — selector is consumed by either beta arming path",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_HEADS",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/beta.rs:enabled — head terms are consumed by either beta arming path",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers — the arming gate \
               and all four BI_* knobs are frozen together before the scored lane starts",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_CHUNK",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers — same frozen policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_CHUNK_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers — same frozen policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_CHUNK_SECS",
        requires: Requirement::GreaterThan("NY_MARGIN_ROW_BI_SECS", 19),
        site: "margin_row/backward_interm.rs:BiState::new — the adaptive target is capped to \
               10% of the effective BI budget; secs >19 keeps every reviewed 0/1/2-second \
               treatment exact, while lower joint doses are rejected before measurement",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_TOPK",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers — same frozen policy",
    },
    // #epoch-bab. Retention can be allocated by the full-epoch cadence too,
    // but without Tier-0 those rows are never consumed. Search therefore
    // requires Tier-0 for both ranker sizing knobs and refuses a vacuous
    // cadence-only retention treatment.
    Edge {
        child: "NY_EPOCH_UNIVERSE",
        requires: Requirement::GreaterThan("NY_EPOCH_TIER0", 0),
        site: "margin_row/bab.rs:select_candidates — tier0_candidates is the only \
               reader of tier0_universe, and it is reached only when tier0_exact > 0",
    },
    Edge {
        child: "NY_EPOCH_RETAIN",
        requires: Requirement::GreaterThan("NY_EPOCH_TIER0", 0),
        site: "margin_row/bab.rs:tier0_candidates — retained rows allocated by a cadence \
               with tier0_exact == 0 are never consumed by the ranker",
    },
    Edge {
        child: "NY_EPOCH_ATTEMPTS",
        requires: Requirement::GreaterThan("NY_EPOCH_DEPTH", 0),
        site: "margin_row/bab.rs:try_epoch — returns immediately on \
               `self.cfg.epoch_depth == 0`, so the attempt cap is never consulted",
    },
    Edge {
        child: "NY_MARGIN_ROW_TABLEAU_F32",
        requires: Requirement::Armed("NY_MARGIN_ROW_ROOT_F32"),
        site: "margin_row/root.rs:build_retaining_inner — f32 tableau storage is \
               refused unless the f32 compute lane is effective",
    },
    Edge {
        child: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP",
        requires: Requirement::Armed("NY_MARGIN_ROW_FROZEN_DUAL_BAB"),
        site: "margin_row/bab.rs:run_inner — the tier (and its TierPolicy::from_gates read) \
               runs only when margin_row::zono_state::armed() latches true",
    },
    Edge {
        child: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_FROZEN_DUAL_BAB"),
        site: "margin_row/bab.rs:run_inner — the tier (and its TierPolicy::from_gates read) \
               runs only when margin_row::zono_state::armed() latches true",
    },
    Edge {
        child: "NY_MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS",
        requires: Requirement::Armed("NY_MARGIN_ROW_FROZEN_DUAL_BAB"),
        site: "margin_row/bab.rs:run_inner — refinement is read only by the armed \
               frozen-dual tier's TierPolicy::from_gates and zero steps keeps its evaluator",
    },
    Edge {
        child: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE",
        requires: Requirement::Armed("NY_MARGIN_ROW_FROZEN_DUAL_BAB"),
        site: "margin_row/bab.rs:run_inner — the tier (and its TierPolicy::from_gates read) \
               runs only when margin_row::zono_state::armed() latches true",
    },
    Edge {
        child: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH",
        requires: Requirement::Armed("NY_MARGIN_ROW_FROZEN_DUAL_BAB"),
        site: "margin_row/bab.rs:run_inner — the tier (and its TierPolicy::from_gates read of \
               the search selector) runs only when margin_row::zono_state::armed() latches \
               true",
    },
    Edge {
        child: "NY_MARGIN_ROW_TREE_WAVE_BATCH",
        requires: Requirement::Equals("NY_MARGIN_ROW_TREE_MODE", "wave"),
        site: "margin_row/bab.rs:run_inner — the wave engine (and its width) runs only when \
               BabConfig::wave is armed on the outer frame",
    },
    Edge {
        child: "NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS",
        requires: Requirement::Equals("NY_MARGIN_ROW_TREE_MODE", "wave"),
        site: "margin_row/bab.rs:run_inner — the wave engine (and its certify pool) runs only \
               when BabConfig::wave is armed on the outer frame",
    },
    Edge {
        child: "NY_MARGIN_ROW_REBAKE_ROUNDS",
        requires: Requirement::Armed("NY_MARGIN_ROW_REBAKE"),
        site: "margin_row/rebake.rs:maybe_rebake_root — the frozen round cap is \
               consumed only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_REBAKE_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_REBAKE"),
        site: "margin_row/rebake.rs:maybe_rebake_root — the frozen wall request \
               is consumed only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_WINDOW",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/backward_interm.rs:BiState::tighten_layer and \
               refresh_root_intermediates — the crop flag is read only by the phase (which \
               returns before any pass when disarmed) and by #joint-refresh, which refuses \
               unless the phase is armed",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_PATCH",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/backward_interm.rs:BiState::new and rebake.rs:joint_refresh_root — \
               the patch config reaches the phase only when the phase is armed, and \
               #joint-refresh refuses unless the phase is armed",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_PATCH_RATIO",
        requires: Requirement::Armed("NY_MARGIN_ROW_BI_PATCH"),
        site: "margin_row/mod.rs:BiPatchRuntimePolicy::cfg — the crossover ratio is carried \
               into a phase config only when patch mode is armed",
    },
    Edge {
        child: "NY_MARGIN_ROW_BI_PATCH_MB",
        requires: Requirement::Armed("NY_MARGIN_ROW_BI_PATCH"),
        site: "margin_row/mod.rs:BiPatchRuntimePolicy::cfg — the working-set cap is carried \
               into a phase config only when patch mode is armed",
    },
    Edge {
        child: "NY_MARGIN_ROW_JOINT_REFRESH",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/rebake.rs:joint_refresh_root — refuses loudly unless the \
               backward-interm phase is armed (the refresh re-runs its frozen policy)",
    },
    Edge {
        child: "NY_MARGIN_ROW_JOINT_REFRESH_SECS",
        requires: Requirement::GreaterThan("NY_MARGIN_ROW_JOINT_REFRESH", 0),
        site: "margin_row/rebake.rs:maybe_joint_refresh_root — the frozen wall request is \
               consumed only after a nonzero round request passes the entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS",
        requires: Requirement::Armed("NY_MARGIN_ROW_ROOT_CERTIFY_ONCE"),
        site: "margin_row/root_certify_once.rs:maybe_certify_once_root — the frozen policy is \
               consumed only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH",
        requires: Requirement::Armed("NY_MARGIN_ROW_ROOT_CERTIFY_ONCE"),
        site: "margin_row/root_certify_once.rs:maybe_certify_once_root — the frozen policy is \
               consumed only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH",
        requires: Requirement::Armed("NY_MARGIN_ROW_BACKWARD_INTERM"),
        site: "margin_row/root_certify_once.rs:certify_once_root — the joint rounds are \
               refused (spec phase only) unless the backward-interm phase is armed; they \
               re-run its frozen topk/chunk/window policy",
    },
    Edge {
        child: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS",
        requires: Requirement::Armed("NY_MARGIN_ROW_ROOT_CERTIFY_ONCE"),
        site: "margin_row/root_certify_once.rs:maybe_certify_once_root — the frozen wall \
               request is consumed only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_ROOT_SEARCH",
        requires: Requirement::Armed("NY_MARGIN_ROW_ROOT_CERTIFY_ONCE"),
        site: "margin_row/root_certify_once.rs:maybe_certify_once_root — the search selector \
               is consumed (and the backend resolved) only inside the armed entry",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_ETA",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/beta.rs:enabled — legacy eta is live through either beta arming path",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_ETA",
        requires: Requirement::Equals("NY_MARGIN_ROW_BETA_POLYAK", "0"),
        site: "margin_row/bab.rs:beta_ascent — eta sizes only the legacy sign-step selected by POLYAK=0",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_ITERS",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/beta.rs:enabled — iteration count is consumed by either beta arming path",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_PERCOL",
        requires: Requirement::AnyArmed(&[
            "NY_MARGIN_ROW_BETA",
            "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        ]),
        site: "margin_row/bab.rs:eval — per-column state exists only inside a live beta ascent",
    },
    Edge {
        child: "NY_MARGIN_ROW_BETA_COLS",
        requires: Requirement::Armed("NY_MARGIN_ROW_BETA_PERCOL"),
        site: "margin_row/bab.rs:eval — the width truncates only the per-column candidate set",
    },
    Edge {
        child: "NY_MARGIN_ROW_PARALLEL_PREBUILD",
        requires: Requirement::NotEquals("NY_MARGIN_ROW_PARALLEL", "0"),
        site: "margin_row/bab.rs:run_loop — serial frontier decisions never form a multi-domain prebuild batch",
    },
    Edge {
        child: "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
        requires: Requirement::Armed("NY_MARGIN_ROW_DOMAIN_ALPHA"),
        site: "margin_row/bab.rs:eval_with_pack — the frozen cap controls certified \
               trials only inside the armed per-domain ascent",
    },
    Edge {
        child: "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA",
        requires: Requirement::Armed("NY_MARGIN_ROW_DOMAIN_ALPHA"),
        site: "margin_row/bab.rs:eval_with_pack — the frozen step controls proposals \
               only inside the armed per-domain ascent",
    },
    Edge {
        child: "NY_MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH",
        requires: Requirement::Armed("NY_MARGIN_ROW_DOMAIN_ALPHA"),
        site: "margin_row/domain_alpha.rs:schedule_admits_with — the schedule is \
               consulted only inside the armed block in bab.rs::eval_with_pack",
    },
    Edge {
        child: "NY_MARGIN_ROW_DOMAIN_ALPHA_STRIDE",
        requires: Requirement::Armed("NY_MARGIN_ROW_DOMAIN_ALPHA"),
        site: "margin_row/domain_alpha.rs:schedule_admits_with — same arming gate",
    },
    Edge {
        child: "NY_MARGIN_ROW_DOMAIN_ALPHA_GAP",
        requires: Requirement::Armed("NY_MARGIN_ROW_DOMAIN_ALPHA"),
        site: "margin_row/domain_alpha.rs:schedule_admits_with — same arming gate",
    },
    Edge {
        child: "NY_CGAN_STACKED_BUDGET_MB",
        requires: Requirement::Armed("NY_CGAN_STACKED_BACKWARD"),
        site: "graph_alpha/bounds/cgan_stacked.rs:stacked_budget_bytes — the budget is \
               consulted only when the stacked walk is armed",
    },
    Edge {
        child: "NY_ROOT_SPEC_FRAC",
        requires: Requirement::NonZero("NY_BAB_RESERVE_FRAC"),
        site: "multi_objective/root.rs — the spec share is read only under an \
               armed reserve; unarmed it is dead and the arm measures baseline",
    },
    Edge {
        child: "NY_ROOT_ALPHA_FRAC",
        requires: Requirement::NonZero("NY_BAB_RESERVE_FRAC"),
        site: "multi_objective/root.rs — same arming gate as the spec share",
    },
    Edge {
        child: "NY_MARGIN_ROW_GPU_BATCH",
        requires: Requirement::Armed("NY_MARGIN_ROW_GPU"),
        site: "margin_row/gpu_seam/batch.rs — batch seam checks the seam gate first",
    },
    Edge {
        child: "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        requires: Requirement::Armed("NY_BNN_SIGN_SPACE"),
        site: "commands/bnn_falsification_policy.rs:BnnFalsificationPolicy::from_levers + commands/beta_crown/sign_space_falsify.rs:segment_move — frozen at entry and lowered only inside the armed sign-space lane",
    },
    Edge {
        child: "NY_BNN_SIGN_SPACE_TRUST_REGION",
        requires: Requirement::Armed("NY_BNN_SIGN_SPACE"),
        site: "commands/bnn_falsification_policy.rs:BnnFalsificationPolicy::from_levers + commands/beta_crown/sign_space_falsify.rs:core_trust_region — frozen at entry and lowered only inside the armed sign-space lane",
    },
    Edge {
        child: "NY_BAB_RESNET_WIDE_SUBGROUP",
        requires: Requirement::NonZero("NY_BAB_RESNET_WIDE"),
        site: "decls/wide_lane.rs — subgroup is a refinement of the wide lane",
    },
    Edge {
        child: "NY_INTERM_ROW_CHUNKS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — chunks are consumed only by the \
               comprehensive sweep policy",
    },
    Edge {
        child: "NY_MO_ALPHA_STEP",
        requires: Requirement::Armed("NY_MO_BETA_SPSA"),
        site: "multi_objective/objectives.rs: the alpha perturbation and update \
               live inside the SPSA gradient branch",
    },
    Edge {
        child: "NY_MO_KFSB_CERT_REUSE",
        requires: Requirement::Armed("NY_MO_KFSB"),
        site: "multi_objective/batched/kfsb_multi.rs: \
               certificate authority is consulted only inside an admitted \
               wave-batched kFSB selector",
    },
    Edge {
        child: "NY_MO_KFSB_K",
        requires: Requirement::Armed("NY_MO_KFSB"),
        site: "multi_objective/batched/kfsb_multi.rs:kfsb_multi_wave_enabled — \
               under the search baseline's default-off depth-two policy, the \
               candidate budget cannot affect a wave until the canonical \
               selector is armed",
    },
    Edge {
        child: "NY_MO_KFSB_REDUCE",
        requires: Requirement::Armed("NY_MO_KFSB"),
        site: "multi_objective/batched/kfsb_multi.rs:kfsb_multi_wave_enabled — \
               the reduction policy ranks child simulations produced only by \
               the canonical selector in an automated search",
    },
    Edge {
        child: "NY_KFSB_SIM_SHARE",
        requires: Requirement::Armed("NY_MO_KFSB"),
        site: "multi_objective/batched/kfsb_multi.rs:select_graph_branch_kfsb_multi_batched — \
               the simulation deadline share is consumed only by an admitted \
               wave-batched kFSB selector",
    },
    Edge {
        child: "NY_ROOT_COMP_GPU_INTERM_ROWS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — row ceiling belongs to the comprehensive \
               policy",
    },
    Edge {
        child: "NY_ROOT_COMP_GPU_INTERM_SECS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — slice belongs to the comprehensive policy",
    },
    // The root-phase ownership lattice: an earlier phase that fires OWNS the slot.
    Edge {
        child: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_PHASE_RESIDENT_CROWN"),
        site: "multi_objective/root.rs phase_resident_or_comprehensive — resident \
               wins and the comprehensive closure is never called",
    },
    Edge {
        child: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_PHASE_RESIDENT_CROWN"),
        site: "multi_objective/root.rs — resident owns the slot",
    },
    Edge {
        child: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs comprehensive_gpu_or_legacy_wide — a \
               comprehensive Some(0) still owns the slot, so wide never runs",
    },
    Edge {
        child: "NY_ROOT_ALPHA_MATMUL",
        requires: Requirement::Armed("NY_ROOT_ALPHA_MARGIN_GRADIENT"),
        site: "propagate_dag/margin_alpha.rs prepare_margin_alpha_lane — the \
               interpolation parameters exist only inside the certified C-row \
               margin ascent, which only the gradient key admits",
    },
];

/// The searchable axes, excluding unsafe, telemetry and test-only names.
#[must_use]
pub fn axes() -> &'static [Axis] {
    AXES
}

/// Axes that are permanently excluded, with the reason recorded.
#[must_use]
pub fn unsafe_axes() -> &'static [Axis] {
    UNSAFE_AXES
}

/// Names that are instruments rather than treatments.
#[must_use]
pub fn instrument_only() -> &'static [&'static str] {
    INSTRUMENT_ONLY
}

/// Names excluded because they only drive tests.
#[must_use]
pub fn test_only() -> &'static [&'static str] {
    TEST_ONLY
}

/// The interaction lattice.
#[must_use]
pub fn edges() -> &'static [Edge] {
    EDGES
}

#[must_use]
fn lookup(sample: &BTreeMap<&str, String>, name: &str) -> Option<String> {
    sample.get(name).cloned()
}

fn armed(sample: &BTreeMap<&str, String>, name: &str) -> bool {
    lookup(sample, name).as_deref() == Some("1")
}

fn non_zero(sample: &BTreeMap<&str, String>, name: &str) -> bool {
    match lookup(sample, name) {
        // NUMERIC, not textual. This predicate was written when its only users
        // were integer levers, where the disarming token is literally "0". A
        // fraction disarms with "0.0", and a string compare calls that ARMED —
        // so the lattice stopped suppressing children of a disarmed parent and
        // every such sample became a full-budget measurement of the baseline
        // wearing a treatment's label. That is the exact failure the EDGES doc
        // comment above describes, produced by the checker meant to prevent it.
        //
        // Falls back to the textual rule for a non-numeric token, so enum and
        // text parents keep the behaviour they had.
        Some(value) => {
            let trimmed = value.trim();
            trimmed
                .parse::<f64>()
                .map_or_else(|_| trimmed != "0", |parsed| parsed != 0.0)
        }
        None => false,
    }
}

fn greater_than(sample: &BTreeMap<&str, String>, name: &str, bound: u64) -> bool {
    lookup(sample, name)
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|value| value > bound)
}

fn equals(sample: &BTreeMap<&str, String>, name: &str, token: &str) -> bool {
    lookup(sample, name).as_deref() == Some(token)
}

/// Whether two admissible raw spellings resolve to the same typed lever value.
///
/// `NotEquals` describes an effective kill-switch value, not a byte spelling:
/// a numeric alias such as `00` must not satisfy a `!= 0` prerequisite.
fn equals_effective(sample: &BTreeMap<&str, String>, name: &str, token: &str) -> bool {
    let Some(raw_candidate) = lookup(sample, name) else {
        return false;
    };
    let Some(decl) = crate::all().get(name) else {
        return raw_candidate == token;
    };
    let Ok(candidate) = crate::read_over_config_with(decl, |_| Some(raw_candidate.clone()), None)
    else {
        return raw_candidate == token;
    };
    let Ok(sentinel) = crate::read_over_config_with(decl, |_| Some(token.to_string()), None) else {
        return raw_candidate == token;
    };
    candidate.source == crate::Source::LegacyEnv
        && sentinel.source == crate::Source::LegacyEnv
        && candidate.value == sentinel.value
}

fn value_is_in_domain(domain: Domain, value: &str) -> bool {
    match domain {
        Domain::Bool => matches!(value, "0" | "1"),
        Domain::Enum(values) => values.contains(&value),
        Domain::Grid(values) => values
            .iter()
            .any(|candidate| candidate.to_string() == value),
    }
}

fn rendered_domain(domain: Domain) -> String {
    match domain {
        Domain::Bool => "[\"0\", \"1\"]".to_string(),
        Domain::Enum(values) => format!("{values:?}"),
        Domain::Grid(values) => format!("{values:?}"),
    }
}

/// Turn a sample into the environment assignment to apply, or explain why the
/// sample is inert.
///
/// A sample only mentions the axes it wants to move; an absent axis means "leave
/// at the compiled default". `expand` refuses when the sample sets an axis whose
/// prerequisite is unmet, because running it would spend a full instance budget
/// re-measuring the baseline. That refusal is the whole point of this module.
///
/// # Errors
/// Returns [`Inert`] naming the first axis whose requirement is unmet, and the
/// requirement that failed.
pub fn expand(sample: &BTreeMap<&str, String>) -> Result<Vec<(String, String)>, Inert> {
    for (name, value) in sample {
        if UNSAFE_AXES.iter().any(|axis| axis.name == *name) {
            return Err(Inert {
                axis: (*name).to_string(),
                because: "axis is Class::Unsafe and must never be set by a search".to_string(),
            });
        }
        let Some(axis) = AXES.iter().find(|axis| axis.name == *name) else {
            return Err(Inert {
                axis: (*name).to_string(),
                because: "name is not a searchable axis in ny_levers::space".to_string(),
            });
        };
        if !value_is_in_domain(axis.domain, value) {
            return Err(Inert {
                axis: (*name).to_string(),
                because: format!(
                    "value {value:?} is outside the exact search domain {}",
                    rendered_domain(axis.domain)
                ),
            });
        }
        for edge in EDGES {
            if edge.child != *name {
                continue;
            }
            let (ok, rendered) = match edge.requires {
                Requirement::Armed(parent) => (
                    armed(sample, parent),
                    format!("requires `{parent}` = \"1\" ({})", edge.site),
                ),
                Requirement::AnyArmed(parents) => (
                    parents.iter().any(|parent| armed(sample, parent)),
                    format!(
                        "requires any of {} = \"1\" ({})",
                        parents
                            .iter()
                            .map(|parent| format!("`{parent}`"))
                            .collect::<Vec<_>>()
                            .join(" or "),
                        edge.site
                    ),
                ),
                Requirement::NonZero(parent) => (
                    non_zero(sample, parent),
                    format!("requires `{parent}` present and not \"0\" ({})", edge.site),
                ),
                Requirement::GreaterThan(parent, bound) => (
                    greater_than(sample, parent, bound),
                    format!("requires `{parent}` > {bound} ({})", edge.site),
                ),
                Requirement::NotArmed(parent) => (
                    !armed(sample, parent),
                    format!("suppressed while `{parent}` = \"1\" ({})", edge.site),
                ),
                Requirement::NotEquals(parent, token) => (
                    !equals_effective(sample, parent, token),
                    format!("suppressed while `{parent}` = {token:?} ({})", edge.site),
                ),
                Requirement::Absent(parent) => (
                    lookup(sample, parent).is_none(),
                    format!("requires `{parent}` to remain absent ({})", edge.site),
                ),
                Requirement::Equals(parent, token) => (
                    equals(sample, parent, token),
                    format!("requires `{parent}` = {token:?} ({})", edge.site),
                ),
            };
            if !ok {
                return Err(Inert {
                    axis: (*name).to_string(),
                    because: rendered,
                });
            }
        }
    }
    Ok(sample
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect())
}

#[cfg(test)]
mod tests {

    /// A fraction's disarming token is "0.0", and it must disarm.
    ///
    /// The regression this pins: `non_zero` compared strings, so "0.0" read as
    /// ARMED and `NY_ROOT_SPEC_FRAC` survived alongside a reserve of 0.0. A
    /// live 46-arm grid spent its whole opening slice on samples of that shape
    /// — full instance budgets measuring the baseline, reported as treatments.
    #[test]
    fn a_zero_valued_fraction_disarms_its_children() {
        for token in ["0", "0.0", "0.00", " 0.0 ", "0e0"] {
            let sample = BTreeMap::from([
                ("NY_BAB_RESERVE_FRAC", token.to_string()),
                ("NY_ROOT_SPEC_FRAC", "0.15".to_string()),
            ]);
            assert!(
                !non_zero(&sample, "NY_BAB_RESERVE_FRAC"),
                "NY_BAB_RESERVE_FRAC={token} must be numerically disarmed"
            );
        }
        let valid_control = BTreeMap::from([
            ("NY_BAB_RESERVE_FRAC", "0.0".to_string()),
            ("NY_ROOT_SPEC_FRAC", "0.15".to_string()),
        ]);
        assert!(
            expand(&valid_control).is_err(),
            "the exact in-domain zero control must leave the spec share inert"
        );
    }

    #[test]
    fn not_equals_compares_effective_numeric_values_not_raw_spellings() {
        let sample = BTreeMap::from([("NY_BNN_ATTACK_CAP", "00".to_string())]);
        assert!(equals_effective(&sample, "NY_BNN_ATTACK_CAP", "0"));
        assert!(!equals_effective(&sample, "NY_BNN_ATTACK_CAP", "1"));
    }

    /// ... and a genuinely armed fraction still admits them.
    #[test]
    fn a_nonzero_fraction_arms_its_children() {
        let sample = BTreeMap::from([
            ("NY_BAB_RESERVE_FRAC", "0.10".to_string()),
            ("NY_ROOT_SPEC_FRAC", "0.15".to_string()),
        ]);
        assert!(
            expand(&sample).is_ok(),
            "an armed reserve must admit its shares"
        );
    }
    use std::collections::BTreeSet;

    /// Every declared lever must be ACCOUNTED FOR by this module.
    ///
    /// This is the guard for a failure that already happened. `NY_BAB_RESERVE_FRAC`
    /// and its two children were declared, wired, documented and then tuned BY
    /// HAND across four values, because nothing connected declaring a lever to
    /// making it searchable. The search platform could not have found the
    /// operating point it exists to find: the axis simply was not there.
    ///
    /// Absence is the bug this catches. A lever may be searchable (`AXES`), or
    /// deliberately withheld (`UNSAFE_AXES`), or excluded as instrumentation or
    /// test scaffolding — but it may not be MISSING, because a missing lever
    /// looks exactly like a covered one from the outside. Adding a name to an
    /// exclusion list is a fine way to satisfy this test; that is a recorded
    /// decision, which is the whole point. Saying nothing is not.
    #[test]
    fn every_declared_lever_is_accounted_for() {
        let mut covered: BTreeSet<&str> = BTreeSet::new();
        covered.extend(AXES.iter().map(|a| a.name));
        covered.extend(UNSAFE_AXES.iter().map(|a| a.name));
        covered.extend(INSTRUMENT_ONLY.iter().copied());
        covered.extend(TEST_ONLY.iter().copied());

        let missing: Vec<&str> = crate::all()
            .all()
            .iter()
            .map(|d| d.name)
            .filter(|n| !covered.contains(n))
            .collect();

        assert!(
            missing.is_empty(),
            "{} declared lever(s) are in no search-space category — not searchable, \
             not excluded, just absent. Each must join AXES, UNSAFE_AXES, \
             INSTRUMENT_ONLY or TEST_ONLY:\n  {}",
            missing.len(),
            missing.join("\n  "),
        );
    }

    /// Nothing may be listed here that is not actually declared.
    ///
    /// The companion to the test above: a stale name in an exclusion list is a
    /// silent hole, because it makes the accounting look complete while
    /// covering a lever that no longer exists.
    #[test]
    fn the_space_names_only_declared_levers() {
        let declared: BTreeSet<&str> = crate::all().all().iter().map(|d| d.name).collect();
        let mut listed: Vec<&str> = Vec::new();
        listed.extend(AXES.iter().map(|a| a.name));
        listed.extend(UNSAFE_AXES.iter().map(|a| a.name));
        listed.extend(INSTRUMENT_ONLY.iter().copied());
        listed.extend(TEST_ONLY.iter().copied());

        let phantom: Vec<&str> = listed
            .into_iter()
            .filter(|n| !declared.contains(n))
            .collect();

        assert!(
            phantom.is_empty(),
            "search space names undeclared levers: {phantom:?}"
        );
    }
    use super::*;

    fn sample(pairs: &[(&'static str, &str)]) -> BTreeMap<&'static str, String> {
        pairs
            .iter()
            .map(|(name, value)| (*name, (*value).to_string()))
            .collect()
    }

    #[test]
    fn every_searchable_unset_default_has_an_explicit_absence_contract() {
        let declared = crate::all();
        for axis in AXES {
            let decl = declared.get(axis.name).expect("axis declaration");
            if matches!(decl.default, crate::DefaultSpec::Unset) {
                let baseline = absence_baseline(axis.name);
                assert_ne!(
                    baseline,
                    AbsenceBaseline::DeclarationDefault,
                    "{} is DefaultSpec::Unset, so a search needs to know whether absence is an \
                     explicit token, distinct, or contextual",
                    axis.name
                );
                if let AbsenceBaseline::Equivalent(token) = baseline {
                    assert!(
                        value_is_in_domain(axis.domain, token),
                        "{} says absence == {token:?}, but that token is outside its search domain",
                        axis.name
                    );
                }
            }
        }

        let axis_names = AXES.iter().map(|axis| axis.name).collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for (name, _) in ABSENCE_BASELINES {
            assert!(
                axis_names.contains(name),
                "absence metadata names non-axis {name}"
            );
            assert!(seen.insert(*name), "duplicate absence metadata for {name}");
        }
    }

    #[test]
    fn known_absent_no_ops_and_dynamic_policies_are_machine_readable() {
        assert_eq!(
            absence_baseline("NY_BAB_RESERVE_FRAC"),
            AbsenceBaseline::Equivalent("0.0")
        );
        assert_eq!(
            absence_baseline("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
            AbsenceBaseline::Equivalent("0")
        );
        assert_eq!(
            absence_baseline("NY_ROOT_COMP_GPU_INTERM_SECS"),
            AbsenceBaseline::Equivalent("20")
        );
        assert_eq!(
            absence_baseline("NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM"),
            AbsenceBaseline::Contextual
        );
        assert_eq!(
            absence_baseline("NY_SWEEP_CLASS_ROWS"),
            AbsenceBaseline::Distinct
        );
        assert_eq!(
            absence_baseline("NY_SWEEP_CLASS_MIB"),
            AbsenceBaseline::Contextual
        );
        assert_eq!(
            absence_baseline("NY_MARGIN_ROW_PARALLEL"),
            AbsenceBaseline::Contextual
        );
        assert_eq!(
            absence_baseline("NY_UPFRONT_ATTACK"),
            AbsenceBaseline::Contextual
        );
        assert_eq!(
            absence_baseline("NY_PGD_VJP_K"),
            AbsenceBaseline::Contextual
        );
        assert_eq!(
            absence_baseline("NY_PGD_VJP_STEPS"),
            AbsenceBaseline::Contextual
        );
    }

    #[test]
    fn preset_baseline_key_chains_are_ordered_complete_and_unique() {
        let axis_names = AXES.iter().map(|axis| axis.name).collect::<BTreeSet<_>>();
        let mut override_names = BTreeSet::new();
        for (name, keys) in PRESET_BASELINE_KEYS {
            assert!(
                axis_names.contains(name),
                "preset baseline metadata names non-axis {name}"
            );
            assert!(
                override_names.insert(*name),
                "duplicate preset baseline metadata for {name}"
            );
            assert!(
                !keys.is_empty(),
                "{name} has an empty preset baseline chain"
            );
        }

        for axis in AXES {
            let keys = preset_baseline_keys(axis.name);
            match axis.deliver {
                Deliver::PresetKey(direct) => {
                    assert!(!keys.is_empty(), "{} has no preset baseline key", axis.name);
                    assert_eq!(
                        keys[0], direct,
                        "{} must begin with its direct delivery key",
                        axis.name
                    );
                    if !override_names.contains(axis.name) {
                        assert_eq!(
                            keys,
                            &[direct],
                            "{} has no reviewed fallback and must use only its direct key",
                            axis.name
                        );
                    }
                }
                Deliver::EnvOnly => assert!(
                    keys.is_empty(),
                    "environment-only axis {} cannot have preset baseline keys",
                    axis.name
                ),
            }

            let mut seen = BTreeSet::new();
            for key in keys {
                assert!(!key.is_empty(), "{} has an empty preset key", axis.name);
                assert!(
                    seen.insert(*key),
                    "{} repeats preset baseline key {key}",
                    axis.name
                );
            }
        }

        assert_eq!(
            preset_baseline_keys("NY_MO_KFSB_K"),
            &[
                "bab.branching.kfsb_multi_candidates",
                "bab.branching.candidates",
            ]
        );
        assert_eq!(
            preset_baseline_keys("NY_MO_KFSB_REDUCE"),
            &[
                "bab.branching.kfsb_multi_reduceop",
                "bab.branching.reduceop",
            ]
        );
        assert_eq!(preset_baseline_keys("NOT_A_LEVER"), &[] as &[&str]);
    }

    #[test]
    fn exact_vjp_axes_have_typed_delivery_and_safe_route_dependencies() {
        let expected = [
            ("NY_PGD_EXACT_BATCHED", "attack.pgd_exact_batched"),
            ("NY_PGD_FAST_GRAD", "attack.pgd_fast_grad"),
            ("NY_PGD_VJP_BATCH", "attack.pgd_vjp_batch"),
            ("NY_PGD_VJP_K", "attack.pgd_vjp_k"),
            ("NY_PGD_LEGACY_LANES", "attack.pgd_legacy_lanes"),
            ("NY_PGD_ODI_STEPS", "attack.pgd_odi_steps"),
            ("NY_PGD_ODI_ETA", "attack.pgd_odi_eta"),
            ("NY_PGD_VJP_STEPS", "attack.pgd_vjp_steps"),
            ("NY_PGD_LANE_WINDOW", "attack.pgd_lane_window"),
            ("NY_PGD_EXPLOIT", "attack.pgd_exploit"),
            ("NY_PGD_HINGE_TAU", "attack.pgd_hinge_tau"),
            ("NY_PGD_DISJ_ODI", "attack.pgd_disj_odi"),
            ("NY_PGD_DISJ_FOCUS_LANES", "attack.pgd_disj_focus_lanes"),
        ];
        for (name, key) in expected {
            let axis = AXES.iter().find(|axis| axis.name == name).unwrap();
            assert_eq!(axis.class, Class::VerdictAffecting, "{name}");
            assert_eq!(axis.deliver, Deliver::PresetKey(key), "{name}");
        }
        assert!(
            EDGES.iter().all(|edge| edge.child != "NY_PGD_FAST_GRAD"),
            "the sequential fast-gradient attempt remains contextually reachable through CPU and wide-route fallbacks, so it must not have a static dependency edge"
        );

        let killed_shared = expand(&sample(&[
            ("NY_PGD_VJP_BATCH", "0"),
            ("NY_PGD_VJP_K", "52"),
        ]))
        .expect_err("a killed shared driver makes width inert");
        assert_eq!(killed_shared.axis, "NY_PGD_VJP_K");

        expand(&sample(&[
            ("NY_PGD_VJP_BATCH", "1"),
            ("NY_PGD_EXACT_BATCHED", "0"),
            ("NY_PGD_VJP_K", "52"),
        ]))
        .expect("shared width remains live on the disjunctive route when conjunctive exact is off");

        let killed_conjunctive = expand(&sample(&[
            ("NY_PGD_VJP_BATCH", "1"),
            ("NY_PGD_EXACT_BATCHED", "0"),
            ("NY_PGD_ODI_STEPS", "8"),
        ]))
        .expect_err("conjunctive-only controls require exact conjunctive routing");
        assert_eq!(killed_conjunctive.axis, "NY_PGD_ODI_STEPS");

        let zero_odi = expand(&sample(&[
            ("NY_PGD_VJP_BATCH", "1"),
            ("NY_PGD_EXACT_BATCHED", "1"),
            ("NY_PGD_ODI_STEPS", "0"),
            ("NY_PGD_ODI_ETA", "0.5"),
        ]))
        .expect_err("ODI eta is inert when no ODI step executes");
        assert_eq!(zero_odi.axis, "NY_PGD_ODI_ETA");

        expand(&sample(&[
            ("NY_PGD_VJP_BATCH", "1"),
            ("NY_PGD_EXACT_BATCHED", "1"),
            ("NY_PGD_ODI_STEPS", "8"),
            ("NY_PGD_ODI_ETA", "0.5"),
        ]))
        .expect("fully armed ODI configuration must be measurable");
    }

    #[test]
    fn upfront_budget_controls_preserve_contextual_absence_but_refuse_a_killed_lane() {
        for (child, value) in [
            ("NY_UPFRONT_ATTACK_CAP", "8"),
            ("NY_UPFRONT_ATTACK_AUTO_CAP", "4"),
            ("NY_UPFRONT_ATTACK_FRAC", "0.08"),
            ("NY_BNN_ATTACK_CAP", "150"),
            ("NY_BNN_ATTACK_FRAC", "0.75"),
            ("NY_UPFRONT_ATTACK_PROBE_FIRST", "1"),
        ] {
            expand(&sample(&[(child, value)])).unwrap_or_else(|error| {
                panic!("contextual master absence must leave {child} measurable: {error}")
            });
            let killed = expand(&sample(&[("NY_UPFRONT_ATTACK", "0"), (child, value)]))
                .expect_err("an explicit master kill makes every upfront child inert");
            assert_eq!(killed.axis, child);
            assert!(killed.because.contains("NY_UPFRONT_ATTACK"));
            let forced = expand(&sample(&[("NY_UPFRONT_ATTACK", "1"), (child, value)]));
            if child == "NY_UPFRONT_ATTACK_AUTO_CAP" {
                let inert = forced.expect_err("forced admission bypasses the automatic cap");
                assert_eq!(inert.axis, child);
            } else {
                forced.unwrap_or_else(|error| {
                    panic!("forced master must leave {child} live: {error}")
                });
            }
        }

        for child in [
            "NY_UPFRONT_ATTACK_AUTO_CAP",
            "NY_UPFRONT_ATTACK_FRAC",
            "NY_BNN_ATTACK_CAP",
            "NY_BNN_ATTACK_FRAC",
        ] {
            let zero_cap = expand(&sample(&[
                ("NY_UPFRONT_ATTACK_CAP", "0"),
                (
                    child,
                    match child {
                        "NY_UPFRONT_ATTACK_AUTO_CAP" => "6",
                        "NY_UPFRONT_ATTACK_FRAC" => "0.5",
                        "NY_BNN_ATTACK_CAP" => "150",
                        "NY_BNN_ATTACK_FRAC" => "0.75",
                        _ => unreachable!(),
                    },
                ),
            ]))
            .expect_err("a zero ordinary cap makes later upfront clamps inert");
            assert_eq!(zero_cap.axis, child);
            assert!(zero_cap.because.contains("NY_UPFRONT_ATTACK_CAP"));
        }

        let zero_bnn_cap = expand(&sample(&[
            ("NY_BNN_ATTACK_CAP", "0"),
            ("NY_BNN_ATTACK_FRAC", "0.75"),
        ]))
        .expect_err("the inert zero BNN cap is deliberately outside the search domain");
        assert_eq!(zero_bnn_cap.axis, "NY_BNN_ATTACK_CAP");
        assert!(zero_bnn_cap
            .because
            .contains("outside the exact search domain"));
    }

    #[test]
    fn category_scoped_mechanisms_publish_non_vacuous_applicability() {
        let axis_names = AXES.iter().map(|axis| axis.name).collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for (name, prefixes) in CATEGORY_PREFIXES {
            assert!(
                axis_names.contains(name),
                "category metadata names non-axis {name}"
            );
            assert!(seen.insert(*name), "duplicate category metadata for {name}");
            assert!(
                !prefixes.is_empty(),
                "{name} has an empty applicability record"
            );
            assert!(
                prefixes.iter().all(|prefix| !prefix.is_empty()),
                "{name} has an empty category prefix"
            );
        }

        assert_eq!(category_prefixes("NY_CGAN_POLISH"), &["cgan"]);
        assert_eq!(
            category_prefixes("NY_BAB_RESNET_WIDE"),
            &["cifar100", "tinyimagenet"]
        );
        assert_eq!(
            category_prefixes("NY_BNN_ATTACK_FRAC"),
            &["traffic_signs_recognition"]
        );
        assert_eq!(
            category_prefixes("NY_BNN_ATTACK_CAP"),
            &["traffic_signs_recognition"]
        );
        assert_eq!(category_prefixes("NY_MO_ALPHA_HEURISTIC"), &[] as &[&str]);
    }

    #[test]
    fn every_axis_name_is_declared_in_the_registry() {
        // The space may EXCLUDE declared levers, but it must never invent one:
        // a typo'd axis is a search dimension that silently does nothing.
        let declared: Vec<&str> = crate::all().all().iter().map(|decl| decl.name).collect();
        for axis in AXES.iter().chain(UNSAFE_AXES) {
            assert!(
                declared.contains(&axis.name),
                "axis `{}` is not a declared lever",
                axis.name
            );
        }
    }

    #[test]
    fn no_axis_is_both_searchable_and_excluded() {
        for axis in AXES {
            assert!(
                !INSTRUMENT_ONLY.contains(&axis.name),
                "`{}` is searchable and instrument-only",
                axis.name
            );
            assert!(
                !TEST_ONLY.contains(&axis.name),
                "`{}` is searchable and test-only",
                axis.name
            );
            assert!(
                axis.class != Class::Unsafe,
                "`{}` is Unsafe but listed as searchable",
                axis.name
            );
        }
    }

    #[test]
    fn every_edge_refers_to_a_known_axis() {
        let known: Vec<&str> = AXES.iter().map(|axis| axis.name).collect();
        for edge in EDGES {
            assert!(known.contains(&edge.child), "edge child `{}`", edge.child);
            let parents: Vec<&str> = match edge.requires {
                Requirement::Armed(parent)
                | Requirement::NonZero(parent)
                | Requirement::NotArmed(parent)
                | Requirement::Absent(parent) => vec![parent],
                Requirement::AnyArmed(parents) => {
                    assert!(
                        !parents.is_empty(),
                        "edge child `{}` has an empty OR",
                        edge.child
                    );
                    let unique = parents.iter().copied().collect::<BTreeSet<_>>();
                    assert_eq!(
                        unique.len(),
                        parents.len(),
                        "edge child `{}` repeats an OR parent",
                        edge.child
                    );
                    parents.to_vec()
                }
                Requirement::GreaterThan(parent, _) => vec![parent],
                Requirement::Equals(parent, _) | Requirement::NotEquals(parent, _) => vec![parent],
            };
            for parent in parents {
                assert!(known.contains(&parent), "edge parent `{parent}`");
            }
            if let Requirement::Equals(parent, token) = edge.requires {
                let axis = AXES
                    .iter()
                    .find(|axis| axis.name == parent)
                    .expect("known edge parent");
                assert!(
                    value_is_in_domain(axis.domain, token),
                    "edge requires `{parent}` = {token:?}, outside its domain {}",
                    rendered_domain(axis.domain)
                );
            }
            if let Requirement::NotEquals(parent, token) = edge.requires {
                let axis = AXES
                    .iter()
                    .find(|axis| axis.name == parent)
                    .expect("known edge parent");
                let decl = crate::all().get(parent).expect("edge parent declaration");
                let parsed = crate::read_over_config_with(decl, |_| Some(token.to_string()), None)
                    .expect("edge sentinel must be admissible to the declaration");
                assert_eq!(
                    parsed.source,
                    crate::Source::LegacyEnv,
                    "edge sentinel `{parent}` = {token:?} must describe a real effective value"
                );
                let domain_tokens = match axis.domain {
                    Domain::Bool => vec!["0".to_string(), "1".to_string()],
                    Domain::Enum(values) => values.iter().map(ToString::to_string).collect(),
                    Domain::Grid(values) => values.iter().map(ToString::to_string).collect(),
                };
                let has_repair = domain_tokens.into_iter().any(|candidate| {
                    crate::read_over_config_with(decl, |_| Some(candidate.clone()), None).is_ok_and(
                        |repair| {
                            repair.source == crate::Source::LegacyEnv
                                && repair.value != parsed.value
                        },
                    )
                });
                assert!(
                    has_repair,
                    "edge has no parser-distinct searchable repair for `{parent}` != {token:?}"
                );
            }
        }
    }

    #[test]
    fn unsafe_axes_are_refused() {
        let err = expand(&sample(&[("NY_STRIP_TERMINAL_SOFTMAX", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_STRIP_TERMINAL_SOFTMAX");
        assert!(err.because.contains("Unsafe"), "{}", err.because);
    }

    #[test]
    fn scheduler_resource_controls_are_typed_but_unsafe() {
        for (name, domain, delivery) in [
            (
                "NY_MARGIN_ROW_EARLY_EXIT",
                Domain::Bool,
                "margin_row.early_exit",
            ),
            (
                "NY_VERIFIER_THREADS",
                Domain::Grid(&[0, 1, 2, 4, 8, 16]),
                "margin_row.verifier_threads",
            ),
            (
                "NY_MARGIN_ROW_CONCURRENT_THREADS",
                Domain::Grid(&[0, 1, 2, 4, 8, 16]),
                "margin_row.concurrent_threads",
            ),
        ] {
            let axis = UNSAFE_AXES
                .iter()
                .find(|axis| axis.name == name)
                .unwrap_or_else(|| panic!("missing scheduler resource axis {name}"));
            assert_eq!(axis.class, Class::Unsafe, "{name}");
            assert_eq!(axis.domain, domain, "{name}");
            assert_eq!(axis.deliver, Deliver::PresetKey(delivery), "{name}");
            let token = match domain {
                Domain::Bool => "0",
                Domain::Grid(_) => "1",
                Domain::Enum(_) => unreachable!(),
            };
            let error = expand(&sample(&[(name, token)])).unwrap_err();
            assert_eq!(error.axis, name);
            assert!(error.because.contains("Unsafe"), "{error}");
        }
    }

    #[test]
    fn canonical_kfsb_is_deliverable_and_legacy_compound_switch_is_unsafe() {
        let canonical = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_KFSB")
            .expect("canonical kFSB search axis");
        assert_eq!(canonical.domain, Domain::Bool);
        assert_eq!(canonical.class, Class::VerdictAffecting);
        assert_eq!(
            canonical.deliver,
            Deliver::PresetKey("bab.branching.kfsb_multi")
        );

        let candidates = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_KFSB_K")
            .expect("governed candidate-count axis");
        assert_eq!(candidates.domain, Domain::Grid(&[0, 2, 5, 7, 8, 10]));
        assert_eq!(
            candidates.deliver,
            Deliver::PresetKey("bab.branching.kfsb_multi_candidates")
        );
        let cert_reuse = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_KFSB_CERT_REUSE")
            .expect("governed certificate-reuse axis");
        assert_eq!(cert_reuse.domain, Domain::Bool);
        assert_eq!(cert_reuse.class, Class::VerdictAffecting);
        assert_eq!(
            cert_reuse.deliver,
            Deliver::PresetKey("bab.branching.kfsb_cert_reuse")
        );
        let reduce = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_KFSB_REDUCE")
            .expect("governed reduce-policy axis");
        assert_eq!(reduce.domain, Domain::Enum(&["min", "max"]));
        assert_eq!(
            reduce.deliver,
            Deliver::PresetKey("bab.branching.kfsb_multi_reduceop")
        );
        let sim_share = AXES
            .iter()
            .find(|axis| axis.name == "NY_KFSB_SIM_SHARE")
            .expect("governed simulation-share axis");
        assert_eq!(
            sim_share.domain,
            Domain::Enum(&["0.0", "0.15", "0.35", "0.60"])
        );

        for (child, value) in [
            ("NY_MO_KFSB_CERT_REUSE", "1"),
            ("NY_MO_KFSB_K", "7"),
            ("NY_MO_KFSB_REDUCE", "max"),
            ("NY_KFSB_SIM_SHARE", "0.35"),
        ] {
            let err = expand(&sample(&[(child, value)])).unwrap_err();
            assert_eq!(err.axis, child);
            assert!(err.because.contains("NY_MO_KFSB"), "{err}");
            assert!(
                expand(&sample(&[("NY_MO_KFSB", "1"), (child, value)])).is_ok(),
                "{child} must become applicable under the canonical master"
            );
        }
        assert!(expand(&sample(&[
            ("NY_MO_KFSB", "1"),
            ("NY_KFSB_SIM_SHARE", "0.0"),
        ]))
        .is_ok());
        let duplicate = expand(&sample(&[
            ("NY_MO_KFSB", "1"),
            ("NY_KFSB_SIM_SHARE", "1.0"),
        ]))
        .unwrap_err();
        assert_eq!(duplicate.axis, "NY_KFSB_SIM_SHARE");
        assert!(duplicate.because.contains("outside"), "{duplicate}");

        let legacy = UNSAFE_AXES
            .iter()
            .find(|axis| axis.name == "NY_BRANCH_KFSB_CHILDSIM")
            .expect("legacy compound compatibility axis");
        assert!(legacy.why.contains("COMPOUND"));
        let err = expand(&sample(&[("NY_BRANCH_KFSB_CHILDSIM", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_BRANCH_KFSB_CHILDSIM");
        assert!(err.because.contains("Unsafe"), "{err}");
    }

    #[test]
    fn measured_multi_objective_alpha_refresh_is_dark_but_preset_deliverable() {
        let axis = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_ALPHA_HEURISTIC")
            .expect("multi-objective alpha heuristic search axis");
        assert_eq!(axis.domain, Domain::Bool);
        assert_eq!(axis.class, Class::VerdictAffecting);
        assert_eq!(axis.deliver, Deliver::PresetKey("bab.mo_alpha_heuristic"));
        assert_eq!(
            expand(&sample(&[("NY_MO_ALPHA_HEURISTIC", "1")])).unwrap(),
            vec![("NY_MO_ALPHA_HEURISTIC".to_string(), "1".to_string())]
        );
    }

    #[test]
    fn multi_objective_alpha_step_dependencies_refuse_inert_samples() {
        let beta = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_BETA_SPSA")
            .expect("multi-objective beta SPSA search axis");
        assert_eq!(beta.domain, Domain::Bool);
        assert_eq!(beta.class, Class::VerdictAffecting);
        assert_eq!(beta.deliver, Deliver::PresetKey("bab.mo_beta_spsa"));

        let alpha = AXES
            .iter()
            .find(|axis| axis.name == "NY_MO_ALPHA_STEP")
            .expect("multi-objective alpha-step search axis");
        assert_eq!(alpha.domain, Domain::Bool);
        assert_eq!(alpha.class, Class::VerdictAffecting);
        assert_eq!(alpha.deliver, Deliver::PresetKey("bab.mo_alpha_step"));

        let missing_spsa = expand(&sample(&[("NY_MO_ALPHA_STEP", "1")])).unwrap_err();
        assert_eq!(missing_spsa.axis, "NY_MO_ALPHA_STEP");
        assert!(missing_spsa.because.contains("NY_MO_BETA_SPSA"));

        let disabled_spsa = expand(&sample(&[
            ("NY_MO_BETA_SPSA", "0"),
            ("NY_MO_ALPHA_STEP", "1"),
        ]))
        .unwrap_err();
        assert!(disabled_spsa.because.contains("NY_MO_BETA_SPSA"));

        assert!(
            expand(&sample(&[
                ("NY_MO_BETA_SPSA", "1"),
                ("NY_MO_ALPHA_STEP", "1"),
            ]))
            .is_ok(),
            "the alpha step is live under beta SPSA"
        );

        assert!(
            expand(&sample(&[
                ("NY_MO_BETA_SPSA", "1"),
                ("NY_MO_ALPHA_HEURISTIC", "1"),
                ("NY_MO_ALPHA_STEP", "1"),
            ]))
            .is_ok(),
            "relu-split delta alpha composes with the structural current-box heuristic"
        );
    }

    #[test]
    fn unknown_axes_and_values_outside_the_exact_domain_are_refused() {
        let err = expand(&sample(&[("NY_NOT_A_REAL_LEVER", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_NOT_A_REAL_LEVER");
        assert!(err.because.contains("not a searchable axis"), "{err}");

        for bad in ["true", "01", " 1", ""] {
            let err = expand(&sample(&[("NY_FALSIFY_PORTFOLIO", bad)])).unwrap_err();
            assert!(err.because.contains("exact search domain"), "{err}");
        }
        for (name, bad) in [
            ("NY_SWEEP_CLASS_ROWS", "65"),
            ("NY_SWEEP_CLASS_ROWS", "064"),
            ("NY_MO_KFSB_K", "3"),
            ("NY_MO_KFSB_REDUCE", "mean"),
            ("NY_KFSB_SIM_SHARE", "0.6"),
        ] {
            let err = expand(&sample(&[(name, bad)])).unwrap_err();
            assert_eq!(err.axis, name);
            assert!(err.because.contains("exact search domain"), "{err}");
        }
    }

    #[test]
    fn chunks_without_the_comprehensive_sweep_are_refused() {
        // This is the concrete waste the module exists to prevent: chunks are read
        // only by the comprehensive policy, so this sample would spend a full
        // instance budget measuring the baseline.
        let err = expand(&sample(&[("NY_INTERM_ROW_CHUNKS", "64")])).unwrap_err();
        assert_eq!(err.axis, "NY_INTERM_ROW_CHUNKS");
        assert!(
            err.because
                .contains("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
            "{}",
            err.because
        );
    }

    #[test]
    fn chunks_with_the_comprehensive_sweep_are_accepted() {
        let expanded = expand(&sample(&[
            ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1"),
            ("NY_INTERM_ROW_CHUNKS", "64"),
        ]))
        .expect("armed sample must expand");
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn phase_resident_suppresses_the_comprehensive_sweep() {
        let err = expand(&sample(&[
            ("NY_ROOT_PHASE_RESIDENT_CROWN", "1"),
            ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1"),
        ]))
        .unwrap_err();
        assert_eq!(err.axis, "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN");
        assert!(err.because.contains("suppressed"), "{}", err.because);
    }

    #[test]
    fn wide_demanded_needs_both_earlier_phases_unarmed() {
        for blocker in [
            "NY_ROOT_PHASE_RESIDENT_CROWN",
            "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        ] {
            let err = expand(&sample(&[
                (blocker, "1"),
                ("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN", "1"),
            ]))
            .unwrap_err();
            assert!(
                err.axis == "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN" || err.axis == blocker,
                "unexpected refusal {err}"
            );
        }
        expand(&sample(&[("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN", "1")]))
            .expect("reachable at (0,0,1)");
    }

    #[test]
    fn margin_row_batch_requires_the_seam() {
        let err = expand(&sample(&[("NY_MARGIN_ROW_GPU_BATCH", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_MARGIN_ROW_GPU_BATCH");
        expand(&sample(&[
            ("NY_MARGIN_ROW_GPU", "1"),
            ("NY_MARGIN_ROW_GPU_BATCH", "1"),
        ]))
        .expect("armed seam must accept the batch axis");
    }

    #[test]
    fn epoch_ranker_axes_are_preset_deliverable_and_refuse_orphan_widths() {
        for (name, path) in [
            ("NY_EPOCH_TIER0", "margin_row.epoch_tier0"),
            ("NY_EPOCH_UNIVERSE", "margin_row.epoch_universe"),
            ("NY_EPOCH_RETAIN", "margin_row.epoch_retain"),
        ] {
            let axis = AXES
                .iter()
                .find(|axis| axis.name == name)
                .expect("epoch ranker axis");
            assert_eq!(axis.class, Class::SafeToSearch);
            assert_eq!(axis.deliver, Deliver::PresetKey(path));
        }

        for (child, value) in [("NY_EPOCH_UNIVERSE", "128"), ("NY_EPOCH_RETAIN", "256")] {
            let err = expand(&sample(&[(child, value)])).unwrap_err();
            assert_eq!(err.axis, child);
            assert!(err.because.contains("NY_EPOCH_TIER0"));
        }

        expand(&sample(&[
            ("NY_EPOCH_TIER0", "8"),
            ("NY_EPOCH_UNIVERSE", "128"),
            ("NY_EPOCH_RETAIN", "256"),
        ]))
        .expect("Tier-0 makes both ranker sizing controls live");
    }

    #[test]
    fn margin_row_tree_throughput_axes_are_independent_typed_treatments() {
        for (name, path, expected_domain) in [
            (
                "NY_MARGIN_ROW_LRU",
                "margin_row.lru",
                Domain::Grid(&[16, 32, 64, 128]),
            ),
            (
                "NY_MARGIN_ROW_PARALLEL",
                "margin_row.parallel",
                Domain::Enum(&["0", "2", "4", "8", "16"]),
            ),
        ] {
            let axis = AXES
                .iter()
                .find(|axis| axis.name == name)
                .expect("margin-row throughput axis");
            assert_eq!(axis.class, Class::SafeToSearch);
            assert_eq!(axis.deliver, Deliver::PresetKey(path));
            assert_eq!(axis.domain, expected_domain);
        }

        expand(&sample(&[("NY_MARGIN_ROW_LRU", "16")])).expect("LRU is live without another arm");
        expand(&sample(&[("NY_MARGIN_ROW_PARALLEL", "0")]))
            .expect("serial frontier is live without another arm");
    }

    #[test]
    fn margin_row_branch_width_axes_are_independent_typed_treatments() {
        for (name, path) in [
            ("NY_MARGIN_ROW_K_HEAD", "margin_row.k_head"),
            ("NY_MARGIN_ROW_K_TRUNK", "margin_row.k_trunk"),
        ] {
            let axis = AXES
                .iter()
                .find(|axis| axis.name == name)
                .expect("margin-row branch-width axis");
            assert_eq!(axis.class, Class::VerdictAffecting);
            assert_eq!(axis.deliver, Deliver::PresetKey(path));
            assert_eq!(axis.domain, Domain::Grid(&[2, 4, 8, 16, 32]));
            expand(&sample(&[(name, "16")]))
                .unwrap_or_else(|error| panic!("{name} must be independently live: {error}"));
        }
    }

    #[test]
    fn a_child_control_arm_still_needs_its_parent() {
        // An explicit zero is still a labelled arm. Without the parent it spends
        // a full row measuring the baseline, so it is inert just like the on arm.
        assert!(expand(&sample(&[("NY_MARGIN_ROW_GPU_BATCH", "0")])).is_err());
        expand(&sample(&[
            ("NY_MARGIN_ROW_GPU", "1"),
            ("NY_MARGIN_ROW_GPU_BATCH", "0"),
        ]))
        .expect("the unbatched control is meaningful while the GPU seam is armed");
    }

    #[test]
    fn newly_recorded_master_child_relationships_refuse_vacuous_arms() {
        for (child, value) in [
            ("NY_MARGIN_ROW_ALPHA_ITERS", "8"),
            ("NY_MARGIN_ROW_ALPHA_SECS", "20"),
            ("NY_MARGIN_ROW_ALPHA_LR", "0.05"),
            ("NY_MARGIN_ROW_ALPHA_COLS", "4"),
            ("NY_FALSIFY_PORTFOLIO_SECONDS", "15"),
            ("NY_MARGIN_ROW_CLOSURE_THRESH", "0.40"),
            ("NY_MARGIN_ROW_CLOSURE_WARMUP", "24"),
            ("NY_MARGIN_ROW_CLOSURE_WINDOW", "16"),
            ("NY_MARGIN_ROW_CLOSURE_MAX_TIER", "0"),
            ("NY_MARGIN_ROW_CLIP", "1"),
            ("NY_MARGIN_ROW_CLIP_INTERM", "1"),
            ("NY_MARGIN_ROW_CLIP_TOPK", "20"),
            ("NY_BNN_SIGN_SPACE_MINIMAL_MOVE", "1"),
            ("NY_BNN_SIGN_SPACE_TRUST_REGION", "linf"),
        ] {
            let err = expand(&sample(&[(child, value)])).unwrap_err();
            assert_eq!(err.axis, child, "unexpected refusal for {child}: {err}");
        }

        expand(&sample(&[
            ("NY_MARGIN_ROW_ALPHA_OPT", "1"),
            ("NY_MARGIN_ROW_ALPHA_ITERS", "8"),
            ("NY_MARGIN_ROW_ALPHA_SECS", "20"),
            ("NY_MARGIN_ROW_ALPHA_LR", "0.05"),
            ("NY_MARGIN_ROW_ALPHA_COLS", "4"),
        ]))
        .expect("armed alpha optimizer must admit its tuning");
        expand(&sample(&[
            ("NY_MARGIN_ROW_REBAKE", "1"),
            ("NY_MARGIN_ROW_ALPHA_LR", "0.05"),
            ("NY_MARGIN_ROW_ALPHA_COLS", "4"),
        ]))
        .expect("armed re-bake optimizer must admit its shared alpha steering controls");
        expand(&sample(&[
            ("NY_FALSIFY_PORTFOLIO", "1"),
            ("NY_FALSIFY_PORTFOLIO_SECONDS", "15"),
        ]))
        .expect("armed portfolio must admit its wall cap");
        expand(&sample(&[
            ("NY_MARGIN_ROW_CLOSURE_ADAPTIVE", "1"),
            ("NY_MARGIN_ROW_CLOSURE_THRESH", "0.40"),
            ("NY_MARGIN_ROW_CLOSURE_WARMUP", "24"),
            ("NY_MARGIN_ROW_CLOSURE_WINDOW", "16"),
            ("NY_MARGIN_ROW_CLOSURE_MAX_TIER", "3"),
        ]))
        .expect("armed closure policy must admit its tuning");
        expand(&sample(&[
            ("NY_MARGIN_ROW_CLIP_ROWS", "1"),
            ("NY_MARGIN_ROW_CLIP", "1"),
            ("NY_MARGIN_ROW_CLIP_INTERM", "1"),
            ("NY_MARGIN_ROW_CLIP_TOPK", "20"),
        ]))
        .expect("retained rows and intermediate clip must admit both clip consumers");
        expand(&sample(&[
            ("NY_BNN_SIGN_SPACE", "1"),
            ("NY_BNN_SIGN_SPACE_MINIMAL_MOVE", "1"),
            ("NY_BNN_SIGN_SPACE_TRUST_REGION", "linf"),
        ]))
        .expect("the armed sign-space lane must admit both refinements");
    }

    #[test]
    fn beta_step_parameters_cannot_cross_selector_arms() {
        let legacy = sample(&[
            ("NY_MARGIN_ROW_BETA", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "0"),
            ("NY_MARGIN_ROW_BETA_ETA", "0.5"),
        ]);
        expand(&legacy).expect("eta is live on the legacy sign-step arm");

        let polyak = sample(&[
            ("NY_MARGIN_ROW_BETA", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "1"),
            ("NY_MARGIN_ROW_BETA_LAMBDA", "1.0"),
        ]);
        expand(&polyak).expect("lambda is live on the Polyak arm");

        let inert_eta = sample(&[
            ("NY_MARGIN_ROW_BETA", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "1"),
            ("NY_MARGIN_ROW_BETA_ETA", "0.5"),
        ]);
        assert!(expand(&inert_eta).is_err(), "Polyak ignores eta");

        let inert_lambda = sample(&[
            ("NY_MARGIN_ROW_BETA", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "0"),
            ("NY_MARGIN_ROW_BETA_LAMBDA", "1.0"),
        ]);
        assert!(
            expand(&inert_lambda).is_err(),
            "the sign-step ignores lambda"
        );

        let closure_legacy = sample(&[
            ("NY_MARGIN_ROW_CLOSURE_ADAPTIVE", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "0"),
            ("NY_MARGIN_ROW_BETA_ETA", "0.5"),
        ]);
        expand(&closure_legacy).expect("closure tier beta must make the exact legacy eta arm live");

        let closure_polyak = sample(&[
            ("NY_MARGIN_ROW_CLOSURE_ADAPTIVE", "1"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "1"),
            ("NY_MARGIN_ROW_BETA_LAMBDA", "1.0"),
        ]);
        expand(&closure_polyak).expect("closure tier beta must make the Polyak lambda arm live");
    }

    #[test]
    fn every_beta_subcontrol_accepts_the_closure_arming_path() {
        for (child, value) in [
            ("NY_MARGIN_ROW_BETA_ITERS", "2"),
            ("NY_MARGIN_ROW_BETA_POLYAK", "0"),
            ("NY_MARGIN_ROW_BETA_HEADS", "0"),
        ] {
            let err = expand(&sample(&[(child, value)])).unwrap_err();
            assert_eq!(err.axis, child);
            assert!(
                err.because.contains("NY_MARGIN_ROW_BETA")
                    && err.because.contains("NY_MARGIN_ROW_CLOSURE_ADAPTIVE"),
                "{err}"
            );
            expand(&sample(&[
                ("NY_MARGIN_ROW_CLOSURE_ADAPTIVE", "1"),
                (child, value),
            ]))
            .unwrap_or_else(|error| panic!("closure should make {child} live: {error}"));
        }
    }

    #[test]
    fn lane_cap_ownership_refuses_the_suppressed_authority() {
        for pairs in [
            &[
                ("NY_LANE_BUDGET_ALLOCATOR", "1"),
                ("NY_LANE_VALUE_SCHEDULER", "1"),
            ][..],
            &[
                ("NY_LANE_BUDGET_FRAME", "1"),
                ("NY_LANE_BUDGET_ALLOCATOR", "1"),
            ][..],
            &[
                ("NY_LANE_BUDGET_FRAME", "1"),
                ("NY_LANE_VALUE_SCHEDULER", "1"),
            ][..],
        ] {
            assert!(expand(&sample(pairs)).is_err(), "must refuse {pairs:?}");
        }

        expand(&sample(&[("NY_LANE_VALUE_SCHEDULER", "1")]))
            .expect("the winning scheduler authority remains a valid arm");
    }

    /// EVERY token of EVERY axis must survive the REAL parser.
    ///
    /// This replaces a version that inspected only `Domain::Enum` and was
    /// therefore vacuous for every axis in this file (all were Bool/Grid) — a
    /// guard that passed while `NY_ALPHA_ZERO_YIELD_FRAC` shipped an integer
    /// grid against an `F64Open { 0.0, 0.9 }` declaration, i.e. three tokens the
    /// parser rejects outright.
    ///
    /// A rejected token does not fail loudly: it resolves to the declaration
    /// default and lands in `rejected_raw`, so the run measures the BASELINE and
    /// a search reports it as a treatment. Inspecting token shape by hand cannot
    /// catch that; only the parser can. So ask it, through the same `read_with`
    /// path production uses.
    #[test]
    fn every_domain_token_survives_the_real_parser() {
        for axis in AXES {
            let decl = crate::all()
                .get(axis.name)
                .unwrap_or_else(|| panic!("axis `{}` is not declared", axis.name));
            for token in domain_tokens(axis.domain) {
                let resolved = crate::read_with(decl, |_| Some(token.clone()));
                assert!(
                    resolved.rejected_raw.is_none(),
                    "axis `{}` offers token {token:?}, which the parser REJECTS. \
                     A rejected token silently resolves to the declaration \
                     default, so this value would measure the baseline and be \
                     reported as a treatment.",
                    axis.name
                );
                assert_eq!(
                    resolved.source,
                    crate::Source::LegacyEnv,
                    "axis `{}` token {token:?} did not take effect",
                    axis.name
                );
            }
        }
    }

    /// The guard above is only meaningful if it can fail. Prove that it does,
    /// using the exact tokens this file used to ship for that axis.
    #[test]
    fn the_parser_guard_is_not_vacuous() {
        let decl = crate::all()
            .get("NY_ALPHA_ZERO_YIELD_FRAC")
            .expect("declared");
        for bad in ["0", "25", "50"] {
            let resolved = crate::read_with(decl, |_| Some(bad.to_string()));
            assert!(
                resolved.rejected_raw.is_some(),
                "expected the parser to reject {bad:?} for an F64Open(0.0, 0.9) \
                 lever; if this passes, the guard above proves nothing"
            );
        }
    }

    fn domain_tokens(domain: Domain) -> Vec<String> {
        match domain {
            Domain::Bool => vec!["0".to_string(), "1".to_string()],
            Domain::Enum(values) => values.iter().map(|v| (*v).to_string()).collect(),
            Domain::Grid(values) => values.iter().map(u64::to_string).collect(),
        }
    }
}
