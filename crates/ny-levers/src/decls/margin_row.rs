// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The margin-row (#twinwall) lane's governed runtime and experimental
//! programs: its measured y-pack cache / parallel-frontier throughput pair,
//! `#margin-row-beta`, `#backward-interm`, the incremental parent-frame epoch,
//! `#rebake`, and the certified GPU backward (`#margin-row-gpu-eft`).
//!
//! The measured [`MARGIN_ROW_LRU`] / [`MARGIN_ROW_PARALLEL`] pair is the
//! module's deliberate exception: both ship at their July 2026 throughput
//! defaults on the established joint-tree schedule. LRU changes only
//! recomputation and deadline reachability; PARALLEL changes exploration order
//! and deadline granularity. Both are Low. The later default-dark classwise
//! schedule has narrower runtime contracts documented at each declaration.
//! Declaration defaults and effective scored presets are deliberately distinct.
//! Most experimental families below remain dark in both layers, while the
//! measured `#backward-interm` gate is default-false here and explicitly armed
//! by the CIFAR100 and TinyImageNet presets. Apart from the documented
//! discard-only/ranking probes, these controls change what the lane's ONE
//! certified concretize is handed. The margin-row lane publishes `Unsat`;
//! anything that moves a bound it publishes is [`MoatRisk::High`] here even
//! where a structural argument says the bound stays sound.
//!
//! * [`MARGIN_ROW_BETA`] (+ [`MARGIN_ROW_BETA_ETA`], [`MARGIN_ROW_BETA_ITERS`])
//!   attach one `beta_j >= 0` Lagrangian term per split so a child domain's
//!   bound can actually improve on its parent's. Weak duality makes each term
//!   valid for ANY `beta >= 0`, and the unchanged certified pass is still the
//!   scorer that accepts or rejects a proposal — but the accepted proposal
//!   changes the PUBLISHED bound, and each trial costs one more certified pass
//!   against the same deadline.
//! * [`MARGIN_ROW_DOMAIN_ALPHA`] (+ [`MARGIN_ROW_DOMAIN_ALPHA_ITERS`],
//!   [`MARGIN_ROW_DOMAIN_ALPHA_ETA`], and the SCHEDULE
//!   [`MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH`] / [`MARGIN_ROW_DOMAIN_ALPHA_STRIDE`]
//!   / [`MARGIN_ROW_DOMAIN_ALPHA_GAP`], which decide WHICH domains pay for the
//!   ascent rather than what it computes) re-optimize the lower-line slope PER
//!   DOMAIN — against that domain's own gates, clip tightening included —
//!   instead of freezing the root's value for the whole tree, and warm-start
//!   each child from its parent's accepted alpha. Any `alpha in [0, 1]` is a
//!   valid lower line pointwise on all of R, so this is tightness-only; but
//!   the accepted value changes the published bound and each ascent trial is
//!   one more certified pass against the deadline.
//! * [`MARGIN_ROW_BACKWARD_INTERM`] (+ [`MARGIN_ROW_BI_SECS`],
//!   [`MARGIN_ROW_BI_CHUNK`], [`MARGIN_ROW_BI_CHUNK_SECS`],
//!   [`MARGIN_ROW_BI_TOPK`]) recompute each trunk
//!   ReLU's input box with the lane's own certified backward engine and
//!   shrink-only intersect it with the forward tableau box. Intersecting two
//!   valid enclosures is sound by construction, and skipping any layer, chunk
//!   or neuron is always sound — but `backward_interm.rs` documents ONE
//!   ORDERING TRAP (`LayerGates::clip_rows` slack must be calibrated against
//!   the FORWARD-only bounds, or every Clip-and-Verify halfspace built from the
//!   line cuts into the true subdomain and becomes a false-`unsat` generator).
//!   A lever that arms a phase with a named false-`unsat` failure mode is High.
//! * [`MARGIN_ROW_EPOCH_PARENT_TOPK`] selects the width of the default-dark
//!   incremental epoch refresh. It reuses the parent's certified intervals,
//!   bakes exact split gates, and re-bounds selected downstream columns with
//!   the same prefix-backward engine. It is bounded to four nested generations
//!   and has a typed, default-silent preset route so canonical scoring can
//!   measure it; no shipped preset arms it, and its cost/solve hypotheses
//!   remain unmeasured.
//! * [`EPOCH_DEPTH`] (+ [`EPOCH_ATTEMPTS`]) are the ORIGINAL `#epoch-bab`
//!   Tier-2 slot the incremental refresh above is a variant of: at `d` trunk
//!   splits, rebuild every intermediate with the splits BAKED IN as absolute
//!   neuron indices and run a nested subtree. This is the only path in ny that
//!   does what alpha-beta-CROWN does on every pass, it publishes the rebuilt
//!   frame, and it is therefore High. [`EPOCH_TIER0`], [`EPOCH_UNIVERSE`] and
//!   [`EPOCH_RETAIN`] are the Tier-0 RANKER beside it and are the module's
//!   other non-probe Lows: they decide which candidates are offered to the
//!   unchanged certified pass, never what it computes. [`EPOCH_DEBUG`] is the
//!   family's trace and is presence-armed, unlike every exact-`"1"` gate here.
//! * [`MARGIN_ROW_REBAKE`] (+ [`MARGIN_ROW_REBAKE_ROUNDS`],
//!   [`MARGIN_ROW_REBAKE_SECS`]) iterate alpha ascent -> forward tableau
//!   rebuild with the optimized alpha baked in -> `#backward-interm` re-run,
//!   with round-level monotone accept on the certified `root_eval` bound —
//!   the joint mechanism behind alpha-CROWN's root. Every candidate build is
//!   certified by the unchanged build construction, but the accepted one
//!   changes the published root bound and the loop spends root-window wall
//!   clock, so the whole family is High.
//! * [`MARGIN_ROW_GPU_BACKWARD`] gates a lane that is designed to be
//!   GPU-AUTHORITATIVE, not a shadow. Its device transaction is delivered but
//!   M3-gated: an armed run passes the on-device M1 self-check and then
//!   refuses `Refusal::ShadowOnly` (verdict authority waits for the design's
//!   M3 same-binary A/B; the trusted code constant `M3_AUTHORITY_GRANTED`
//!   stays false until then). [`MARGIN_ROW_GPU_SHADOW`] is the M2 protocol:
//!   compute the device pass, compare against the CPU pass, DISCARD. The
//!   declarations are classified for what the gates admit, exactly as
//!   `NY_EFT_ERR`'s staged CPU arm is.
//!
//! [`MARGIN_ROW_LRU`] and [`MARGIN_ROW_PARALLEL`] have shipped at their measured
//! defaults since July 2026; [`MARGIN_ROW_BACKWARD_INTERM`] has its own later,
//! category-specific scored evidence below. No environment override acquires
//! shipped authority merely from a structural argument. Each non-default value
//! needs retained scored-row evidence, and a typed preset keeps its own
//! evidence/authority rather than inheriting it from an environment override.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const BETA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-beta",
};

const BACKWARD_INTERM: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-backward-interm",
};

const DOMAIN_ALPHA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-domain-alpha",
};

const EPOCH_PARENT: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-epoch-parent",
};

/// The ORIGINAL `#epoch-bab` family: Tier-0 rank-1 candidate ranking and the
/// Tier-2 baked-split tableau rebuild. Distinct from [`EPOCH_PARENT`], which
/// owns the later INCREMENTAL parent-frame variant of the same Tier-2 slot.
const EPOCH_BAB: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-epoch-bab",
};

const GPU_EFT: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-gpu-eft",
};

const BRANCHING: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-branching",
};

const LANE_RUNTIME: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-bab-runtime",
};

const NY_CLI_PRESET: Scope = Scope {
    package: "ny-cli",
    subsystem: "preset-delivery",
};

const SCHEDULE: Scope = Scope {
    package: "ny-cli",
    subsystem: "margin-row-schedule",
};

/// The outer verifier/margin-row schedule is resolved once into an immutable
/// policy. Scored callers consume the receipt-owning [`crate::LeverSet`]; the
/// layered constructor is retained only for compatibility callers whose typed
/// `MarginRowPreset` has not yet entered that frozen snapshot.
const SCHEDULE_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the validated typed margin_row scheduler fields into the same frozen \
               LeverSet that owns the scored flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:\
               margin_row_schedule_preset_config_value",
    },
    ReaderSite {
        scope: SCHEDULE,
        role: "derive and freeze all eight scheduler controls from the same LeverSet that owns \
               the scored flight receipt",
        site: "crates/ny-cli/src/commands/margin_row_bab.rs:\
               MarginRowScheduleRuntimePolicy::from_levers",
    },
    ReaderSite {
        scope: SCHEDULE,
        role: "layer each historical environment grammar over its exact typed margin_row \
               preset field for non-scored compatibility callers",
        site: "crates/ny-cli/src/commands/margin_row_bab.rs:\
               MarginRowScheduleRuntimePolicy::from_layered_process",
    },
    ReaderSite {
        scope: SCHEDULE,
        role: "consume the immutable reserve, adaptive-release and budget-cap policy while \
               choosing the authoritative verifier/margin-row time split",
        site: "crates/ny-cli/src/commands/margin_row_bab.rs:\
               margin_row_reserve_decision_with_policy / \
               MarginRowReserveDecision::capped_to_internal_budget_with_policy",
    },
    ReaderSite {
        scope: SCHEDULE,
        role: "consume the immutable concurrency, lane-width and early-publication policy when \
               arming and spawning the authoritative companion lane",
        site: "crates/ny-cli/src/commands/margin_row_bab.rs:\
               concurrent_lane_armed_with_policy / spawn_concurrent_lane_with_schedule_policy",
    },
    ReaderSite {
        scope: SCHEDULE,
        role: "apply the immutable verifier thread cap to the process-global Rayon pool on the \
               main thread before any verifier work starts",
        site: "crates/ny-cli/src/commands/vnncomp.rs:handle_vnncomp_command",
    },
];

/// The five searchable `#epoch-bab` integers share one scored authority: the
/// generic preset projection populates the receipt-owning frozen snapshot,
/// then the margin-row entry policy consumes that same snapshot once. The
/// final site below is exclusively for non-scored compatibility callers.
const EPOCH_BAB_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed epoch field through the generic schema lookup into the \
               frozen VNN-COMP lever snapshot",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: EPOCH_BAB,
        role: "derive and freeze the complete epoch policy from the same LeverSet that owns \
               the flight receipt",
        site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers",
    },
    ReaderSite {
        scope: EPOCH_BAB,
        role: "resolve the historical process/preset layers once at entry for non-scored \
               compatibility callers",
        site: "crates/ny-propagate/src/margin_row/mod.rs:EpochRuntimePolicy::from_layered_process",
    },
];

/// The unsafe incremental-parent width uses the same immutable delivery path
/// without becoming a promoted/search-safe axis. Its final site is likewise a
/// non-scored compatibility boundary, not a second scored authority.
const EPOCH_PARENT_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed unsafe parent-refresh width through the generic schema \
               lookup into the frozen VNN-COMP lever snapshot",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: EPOCH_PARENT,
        role: "derive and freeze the incremental parent-refresh width with the complete \
               epoch policy from the receipt-owning LeverSet",
        site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers",
    },
    ReaderSite {
        scope: EPOCH_PARENT,
        role: "resolve the historical process/preset layers once at entry for non-scored \
               compatibility callers",
        site: "crates/ny-propagate/src/margin_row/mod.rs:EpochRuntimePolicy::from_layered_process",
    },
];

/// The complete per-domain alpha family has one scored authority: generic
/// preset projection into the receipt-owning frozen snapshot, followed by one
/// immutable runtime-policy derivation. The legacy globals remain solely for
/// non-scored compatibility callers and are resolved once at lane entry.
const DOMAIN_ALPHA_POLICY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed per-domain alpha field through the generic schema lookup \
               into the frozen VNN-COMP lever snapshot",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: DOMAIN_ALPHA,
        role: "derive and freeze the complete per-domain alpha policy from the same LeverSet \
               that owns the flight receipt",
        site: "crates/ny-propagate/src/margin_row/mod.rs:DomainAlphaRuntimePolicy::from_levers",
    },
    ReaderSite {
        scope: DOMAIN_ALPHA,
        role: "resolve the historical process/preset cells once at entry for non-scored \
               compatibility callers",
        site: "crates/ny-propagate/src/margin_row/mod.rs:DomainAlphaRuntimePolicy::from_layered_process",
    },
];

const LANE_RUNTIME_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "deliver margin_row.lru / margin_row.parallel to the propagation preset cells \
               before any lane runs",
        site: "crates/ny-cli/src/preset_gates.rs:apply",
    },
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed margin-row value into the early flight receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: LANE_RUNTIME,
        role: "layer the legacy environment token over typed preset delivery and the \
               declaration default",
        site: "crates/ny-propagate/src/preset_gates.rs:margin_row_lru_cap / \
               margin_row_frontier",
    },
    ReaderSite {
        scope: LANE_RUNTIME,
        role: "size the y-pack cache and parallel best-first frontier used by the \
               authoritative margin-row lane",
        site: "crates/ny-propagate/src/margin_row/mod.rs:lane_impl",
    },
];

/// The branch-width pair has one scored authority: generic typed-preset
/// projection into the receipt-owning frozen set, followed by immutable
/// runtime-policy derivation and consumption. The process/environment resolver
/// remains only as the explicit non-scored compatibility entry.
const BRANCH_WIDTH_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: NY_CLI_PRESET,
        role: "project the typed branch width through the generic schema lookup into the \
               frozen VNN-COMP lever snapshot",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: BRANCHING,
        role: "derive and freeze both configured widths and their exact lower-authority \
               counterfactuals from the same LeverSet that owns the flight receipt",
        site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowWidthRuntimePolicy::from_levers",
    },
    ReaderSite {
        scope: BRANCHING,
        role: "resolve legacy environment over typed compatibility cells once for \
               non-scored callers that do not supply a frozen LeverSet",
        site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowWidthRuntimePolicy::from_layered_process",
    },
    ReaderSite {
        scope: BRANCHING,
        role: "consume the immutable configured and lower-authority widths when building \
               the authoritative ordinary branch scorer and its execution evidence",
        site: "crates/ny-propagate/src/margin_row/mod.rs:run_margin_row_lane_with_policy / \
               crates/ny-propagate/src/margin_row/bab.rs:MarginRowBab::head_prerank / \
               MarginRowBab::trunk_shortlist",
    },
];

const REBAKE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-rebake",
};

/// The root forward-tableau precision family: f32 COMPUTE and its dependent
/// f32 STORAGE lane.
const ROOT_F32: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-root-f32",
};

/// The `#tableau-f32` f32 STORAGE lane of the root forward-tableau walk.
/// Distinct from the [`ROOT_F32`] COMPUTE lane it requires.
const TABLEAU_F32: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-tableau-f32",
};

/// The `#domain-stack` cross-domain candidate-scoring lane: one wide certified
/// backward pass over the whole frontier batch instead of one pass per domain.
const DOMAIN_STACK: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-domain-stack",
};

/// The `#mr-errlane` envelope certified error lane (throughput plan B3).
const ERR_ENVELOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-err-envelope",
};

/// The `#mr-gpu-eval` conv-backward offload of the certified pass onto the
/// process-global sound f64 GEMM engine.
const GPU_CONV: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-gpu-conv",
};

/// The `#frozen-dual-state` certified frozen alpha-zonotope query state (V1
/// stage A) and the `#frozen-dual-bab` search tier over it (V1 stage B).
const FROZEN_DUAL: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-frozen-dual",
};

/// The `#joint-refresh` backward-only joint intermediate refresh (cct plan P5
/// step 2): per round, re-derive the root's intermediate boxes from its
/// current gates, re-steer the alpha on the refreshed frame, accept on the
/// certified root bound.
const JOINT_REFRESH: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-joint-refresh",
};

/// The `#root-certify-once` Phase-0.5 root alpha: the root ascent run UNTRUSTED
/// on a Parity scratch root (optionally with an untrusted per-round
/// intermediate refresh), then ONE certified evaluation of the chosen alpha,
/// kept only if it strictly beats the certified incumbent.
const ROOT_CERTIFY_ONCE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-root-certify-once",
};

/// Complete small head-phase covers on the source-owned outer root frame.
const HEAD_GROUPS: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-head-groups",
};

/// The `#mr-ibp-seed` certified interval (IBP) seed of the root tableau walk:
/// every trunk box intersected shrink-only with its certified IBP box.
const IBP_SEED: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-ibp-seed",
};

/// The `#wave-beta-bab` (ENGINE_PROGRAM M6) wave tree engine: batched
/// untrusted beta-CROWN search over waves of sub-domains, a streaming certify
/// queue on a private pool, the path-based cover gate.
const WAVE_TREE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-wave-tree",
};

/// The `#root-box-source` (ENGINE_PROGRAM M5b) selector of the root's trunk-box
/// construction: the forward tableau, or a certified IBP reference plus a
/// certified patch-backward pass over the reference-unstable neurons.
const ROOT_BOX_SOURCE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-root-box-source",
};

/// All four `#backward-interm` tuning knobs are resolved with their arming gate
/// from the same receipt-owning [`crate::LeverSet`] before the scored lane starts.
/// Root builds and rebakes consume that immutable policy and never reopen the
/// process environment.
const CLIP: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-clip",
};

/// All three `#backward-interm` tuning knobs are resolved by the same function,
/// per BUILD rather than through a process-wide latch, so a sealed A/B or a
/// unit test can flip arms without restarting the process.
const BI_TUNING_READER: ReaderSite = ReaderSite {
    scope: BACKWARD_INTERM,
    role: "size the backward-intermediate phase's budget, pass width and per-layer selection",
    site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
};

const BETA_LAMBDA_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "size the Polyak step as a fraction of the direct-path gap",
    site: "crates/ny-propagate/src/margin_row/beta.rs:lambda",
}];

const BETA_POLYAK_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "opt OUT of Polyak step sizing, falling back to the fixed eta step",
    site: "crates/ny-propagate/src/margin_row/beta.rs:polyak_enabled",
}];

const BETA_HEADS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "opt OUT of head-split beta terms",
    site: "crates/ny-propagate/src/margin_row/beta.rs:heads_enabled",
}];

declare_levers! {
    registry MARGIN_ROW_LEVERS;

    /// `NY_MARGIN_ROW_CLASSWISE` — legacy classwise research schedule.
    pub MARGIN_ROW_CLASSWISE = LeverDecl {
        name: "NY_MARGIN_ROW_CLASSWISE",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Chooses the existing classwise research schedule instead of the joint \
              schedule. Text preserves its exact historical predicate: after trim, \
              case-insensitive true/on/yes or exact1 arms; every other value, \
              absence and non-Unicode input are off. CLI reads the receipt-owning \
              frozen LeverSet; compatibility entry freezes its environment once. \
              No typed preset promotion or default change is made here.",
        provenance: Provenance::Unmeasured { why_ok: "legacy research schedule remains default off and excluded from automated search" },
        owner: Scope { package: "ny-propagate", subsystem: "margin-row-classwise" },
        readers: &[
            ReaderSite { scope: Scope { package: "ny-cli", subsystem: "prepared-root-route" }, role: "consume the identical frozen classwise bit for prepared-route admission", site: "crates/ny-cli/src/commands/vnncomp.rs:run_and_translate" },
            ReaderSite { scope: Scope { package: "ny-propagate", subsystem: "margin-row-classwise" }, role: "freeze the historical text predicate into the runtime policy shared by scored and compatibility entries", site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers" },
            ReaderSite { scope: Scope { package: "ny-propagate", subsystem: "margin-row-classwise" }, role: "consume the frozen schedule without reopening environment state", site: "crates/ny-propagate/src/margin_row/mod.rs:lane_impl_with_policy_body" },
        ],
    };

    /// `NY_MARGIN_ROW_LRU` — deterministic y-pack cache capacity.
    pub MARGIN_ROW_LRU = LeverDecl {
        name: "NY_MARGIN_ROW_LRU",
        // Historical reader was bare `parse::<usize>()`: no trimming, and 0
        // is a valid capacity rather than a malformed token. U64 is the only
        // untrimmed registry integer; the 32-bit receipt caveat is documented.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(64),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::Low,
        doc: "\
Capacity, in distinct trunk split-sets, of the established joint margin-row BaB \
tree's y-pack LRU. Default 64. The parser is the historical untrimmed \
`parse::<usize>()` on \
the supported 64-bit competition target: `0` explicitly disables retention, \
whitespace and malformed values restore 64, and a present malformed \
environment value suppresses any typed `margin_row.lru` treatment. A non-UTF-8 \
environment value likewise suppresses the typed layer and produces 64, matching \
the old `std::env::var(...).ok()` reader's effective answer while preserving \
the governed presence rule. The registry uses its untrimmed U64 kind; on a \
32-bit target an above-`usize` value is therefore accepted in the receipt but \
refused by the runtime conversion and still executes the historical 64 fallback.

The later default-dark `NY_MARGIN_ROW_CLASSWISE` experiment still consumes this \
capacity, but floors `0` to one entry because every independent class tree must \
start with its immutable root pack. Nonzero typed values remain live there. The \
retained July evidence predates classwise and establishes only the joint-tree \
default; a classwise-contaminated A/B cannot extend that provenance.

VALUE-NEUTRAL BY CONSTRUCTION. A hit returns the same deterministic y-pack a \
miss rebuilds; capacity changes memory and recomputation only, never the value \
of a completed bound, a tree choice, or a reduction order. It is Low rather \
than None because this authoritative lane is deadline-bounded: the measured \
2.49x throughput change can decide whether a sound verdict is reached before \
the wall or the row returns unknown. The July profile found only 47 distinct \
trunk sets on the representative timeout, so 64 captured essentially the whole \
working set. Typed delivery exists to re-measure that cache footprint on current \
VNN-COMP walls; no shipped yaml sets it.",
        provenance: Provenance::Measured {
            commit: "5315287659c1612af5bef8e2f9aa2697dd8dc85c",
            date: "2026-07-17",
            artifact: "commit 531528765 (`perf(margin-row): ~3x BaB throughput`), \
                       profile for cifar100 timeout 4921",
            delta: "LRU 6 -> 64 raised hit rate 26% -> 86%, cut y-refresh \
                    63 s -> 29 s, and raised 100 s throughput 136 -> 339 \
                    expansions (2.49x), with deterministic-pack parity",
        },
        owner: LANE_RUNTIME,
        readers: LANE_RUNTIME_READERS,
    };

    /// `NY_MARGIN_ROW_PARALLEL` — parallel best-first frontier width.
    pub MARGIN_ROW_PARALLEL = LeverDecl {
        name: "NY_MARGIN_ROW_PARALLEL",
        // Free-form Text is load-bearing: the legacy reader accepts a trimmed,
        // case-insensitive truthy/falsy vocabulary plus every usize. The typed
        // preset surface is deliberately narrower and is validated separately.
        kind: LeverKind::Text,
        // Exact legacy token `1` means all available Rayon workers. It is NOT
        // offered by the typed search domain, where omission is the all-worker
        // control and explicit treatments are 0,2,4,8,16.
        default: DefaultSpec::Text("1"),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::Low,
        doc: "\
Parallel best-first frontier width for the established joint margin-row BaB \
tree. Absence, exact \
`1`, or trimmed case-insensitive `true|on|yes` selects all available Rayon \
workers. Empty, `0`, `false|off|no`, or any malformed token selects one worker. \
Any other `usize` N selects `max(N, 1)` with NO clamp to the available worker \
count. Those are the historical semantics, including whitespace trimming and \
case handling.

The legacy environment remains free-form Text so old recipes keep working. \
The typed `margin_row.parallel` route is intentionally canonical and accepts \
only `0|2|4|8|16`: typed `1` is rejected because legacy `1` means ALL workers, \
not a one-worker treatment. Omission is the typed all-worker control. A present \
malformed environment token suppresses the typed treatment and selects one \
worker. A present non-UTF-8 value also suppresses the typed layer, but is passed \
to the legacy parser as absent and therefore selects all workers, exactly the \
answer produced by the former `std::env::var(...).ok()` reader.

Low, not None: every domain bound is bit-identical to serial and every pushed \
bound still comes from the certified pass, but batching changes exploration \
order and deadline granularity, so it can change which sound verdict is reached \
inside a fixed wall. The shipped all-worker default was adopted from the \
banked exact-timeout sweep; typed delivery permits explicit width measurement \
without exporting an environment variable. No shipped yaml sets it.

The later default-dark `NY_MARGIN_ROW_CLASSWISE` experiment intentionally forces \
each per-class tree to frontier 1 so its single global expansion budget remains \
exact. A run contaminated by a truthy ambient classwise gate therefore makes \
every PARALLEL treatment inert and must be voided; the retained parallel evidence \
predates classwise and applies only to the joint schedule.",
        provenance: Provenance::Measured {
            commit: "7b004fbaaf2e57e2c8367e7e12e0357e8c11d1f9",
            date: "2026-07-18",
            artifact: "commits 531528765 (`perf(margin-row): ~3x BaB throughput`) \
                       and 7b004fbaa (`vnncomp: harden 2026 scoring, evidence, \
                       and witnesses`)",
            delta: "all-worker frontier added 1.19x over cached serial (2.97x \
                    combined); prop_idx_1588 moved from timeout at 318 \
                    expansions to UNSAT at 420 expansions / 95.8 s; serial \
                    parity and three GT-sat moat rows held",
        },
        owner: LANE_RUNTIME,
        readers: LANE_RUNTIME_READERS,
    };

    /// `NY_MARGIN_ROW_RESERVE_SECS` — fixed verifier/margin-row time reserve.
    pub MARGIN_ROW_RESERVE_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_RESERVE_SECS",
        kind: LeverKind::U64Override,
        default: DefaultSpec::U64(45),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds withheld from the internal verifier for the authoritative margin-row \
lane, before adaptive release or the optional budget-fraction ceiling. The \
historical parser is an UNTRIMMED `parse::<u64>()`: every value including zero \
is admissible, while whitespace, minus signs, overflow, malformed Unicode and \
non-Unicode input are ignored and leave typed `margin_row.reserve_secs` (then \
the declaration default 45) authoritative. Zero is the explicit full-release \
treatment.

This is High because zero also makes the adaptive route NotApplicable; a \
nonzero reserve can therefore be a prerequisite for the production route that \
arms scored sparse root CROWN and moves a published bound. The retained \
historical score notes do not isolate this value from adaptive policy, lane \
concurrency, category mix, or wall tier, so it is declared without a search \
axis until a sealed same-binary A/B attributes its net conversion effect.",
        provenance: Provenance::Unmeasured {
            why_ok: "the existing production default is preserved exactly, but historical \
                     score bookkeeping does not isolate the reserve from the rest of the \
                     schedule",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_ADAPTIVE_RESERVE` — adaptive reserve-route selector.
    pub MARGIN_ROW_ADAPTIVE_RESERVE = LeverDecl {
        name: "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the adaptive reserve route above typed `margin_row.adaptive_reserve`. \
Exact Unicode `1` arms and exact Unicode `0` disarms. Every other present \
Unicode token, including whitespace and empty input, and every present \
non-Unicode value rejects to the declaration default false and SUPPRESSES a \
typed true. That fail-closed asymmetry is the historical `var_os` contract, \
not ordinary override fallthrough.

The gate is High because an armed release decision is the sole production \
authority for scored sparse root CROWN; changing it can move a published bound. \
The declaration stays out of the generic search space because the shipped \
category treatment has not been separated from reserve size, release threshold \
and concurrent-lane timing by retained paired evidence.",
        provenance: Provenance::Unmeasured {
            why_ok: "the default remains dark and every malformed explicit token fails closed; \
                     promotion requires schedule-isolated scored evidence",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_RESERVE_MAX_FRAC` — optional reserve ceiling.
    pub MARGIN_ROW_RESERVE_MAX_FRAC = LeverDecl {
        name: "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        kind: LeverKind::F64OpenUnicodeOwned { min: 0.0, max: 1.0 },
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Optional ceiling on reserved seconds as a fraction of the INTERNAL verifier \
budget. Unicode input is parsed without trimming as finite `f64` strictly in \
`(0, 1)`. Any present but inadmissible Unicode token owns the override and \
resolves to Unset, deliberately suppressing typed \
`margin_row.reserve_max_frac`; this keeps `0` an exact kill switch for a typed \
ceiling. A non-Unicode value is invisible to the old `std::env::var` reader and \
therefore falls through to the typed `f32` value (widened exactly to `f64`) or \
the Unset default.

Low: the ceiling changes only how wall time is divided between two independently \
sound lanes, and it explicitly cannot change the adaptive route or arm sparse \
root CROWN. It can change whether a proof finishes before the deadline, so it is \
not None. No search axis is admitted before conversion evidence measures the \
ceiling jointly with the fixed reserve on VNN-COMP walls.",
        provenance: Provenance::Unmeasured {
            why_ok: "the optional ceiling is absent by default and only removes margin-row wall \
                     time; its conversion effect remains unmeasured",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC` — adaptive release threshold.
    pub MARGIN_ROW_ADAPTIVE_RELEASE_FRAC = LeverDecl {
        name: "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
        kind: LeverKind::F64OpenOverride { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.40),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Adaptive route threshold: release the fixed reserve when it consumes at least \
this fraction of the INTERNAL budget. Environment input is an untrimmed finite \
`f64` strictly in `(0, 1)`. Invalid Unicode and non-Unicode values are ignored \
and preserve typed `margin_row.release_frac` (an `f32` widened exactly to \
`f64`), then the declaration default 0.40. Endpoints, whitespace, NaN and \
infinities are therefore no-override tokens, not kill switches.

This is High because crossing the threshold selects the adaptive released route, \
which is the authority that arms scored sparse root CROWN. The 0.40 production \
default encodes budget arithmetic observed at two wall tiers, but its net effect \
has not been isolated from reserve size or category treatment; it remains \
unsearchable pending retained paired evidence.",
        provenance: Provenance::Unmeasured {
            why_ok: "the historical 0.40 answer and every fallback path are preserved; the \
                     threshold has no isolated scored attribution",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_CONCURRENT` — run the margin-row lane concurrently.
    pub MARGIN_ROW_CONCURRENT = LeverDecl {
        name: "NY_MARGIN_ROW_CONCURRENT",
        kind: LeverKind::TrimmedExactOne,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Runs the independently sound margin-row lane concurrently with the internal \
verifier. Every Unicode environment value is a complete Boolean override after \
trimming: exactly `1` is true and every other token (including `0`, `true`, \
garbage and empty input) is false and suppresses typed \
`margin_row.concurrent`. A present non-Unicode value falls through to the typed \
Boolean or the false default, matching the former `std::env::var` reader.

Low: concurrency changes allocation pressure, completion order and deadline \
reachability, but either published verdict still comes from an unchanged \
authoritative verifier lane. It cannot change a completed bound. Automated \
search is admitted only to collect new evidence through the context-bound \
schedule execution predicate: the treatment must spawn real stats-owning work, \
overlap a live verifier, and hold zero reserve seconds. Launch-only, structurally \
refused, CPU-device and double-tax rows receive no credit. The retained +20 \
CIFAR result in 7534933d changed this gate together with root-f32, while the \
older concurrent-only run predates the corrected result join. Neither isolates \
this coordinate, so neither may be quoted as its causal delta.",
        provenance: Provenance::Unmeasured {
            why_ok: "the default remains dark and the new allocation/overlap receipt can \
                     authenticate a future reserve-free concurrent-only A/B; retained history \
                     either confounds root precision or predates the corrected result join",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_EARLY_EXIT` — publish a concurrent-lane proof immediately.
    pub MARGIN_ROW_EARLY_EXIT = LeverDecl {
        name: "NY_MARGIN_ROW_EARLY_EXIT",
        kind: LeverKind::TrimmedExactOne,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::Low,
        doc: "\
Publishes the concurrent margin-row lane's certified `unsat` as soon as the \
schedule allocation receipt is durable, instead of waiting for the internal \
verifier to consume its remaining budget. Every Unicode environment token owns \
the layer after trimming: exactly `1` arms; every other token, including `0`, \
garbage and empty input, disarms and suppresses typed `margin_row.early_exit`. A \
non-Unicode value falls through to the typed Boolean or the true default, exactly \
matching the historical `std::env::var` reader.

Low: this changes publication timing only after an independently certified \
fail-closed lane has proved the row. It cannot change a completed bound, and the \
single-writer publication protocol prevents the lane and main thread from \
overwriting one another. The default stays ON because the measured proving row \
`idx_7704` closed at 10.4 seconds while the process otherwise ran to 89.7 seconds. \
The typed key is a production rollback, not an automated search coordinate.",
        provenance: Provenance::Measured {
            commit: "2ec01b4aabd81da41a9ce7144dc3a6c3f88a27b3",
            date: "2026-08-22",
            artifact: "docs/PROGRAM_2026-08-22_CLOSURE_FIRST.md, Front B",
            delta: "cifar100 idx_7704: concurrent lane certified UNSAT at 10.4 s but the \
                    process did not publish until 89.7 s, wasting about 79 s",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_VERIFIER_THREADS` — process-global internal-verifier Rayon cap.
    pub VERIFIER_THREADS = LeverDecl {
        name: "NY_VERIFIER_THREADS",
        kind: LeverKind::UsizeTrimmedUnicodeOwned,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Optional platform-sized thread cap for the internal alpha/beta verifier's \
process-global Rayon pool. Unicode input is trimmed and parsed as `usize`. Every \
Unicode token owns the environment layer: a positive value requests that cap, \
while explicit zero remains source-visible in the receipt and the consuming \
policy normalizes it to no cap. Malformed input and overflow suppress typed \
`margin_row.verifier_threads` and resolve to the Unset default. A non-Unicode \
value is invisible to the historical `std::env::var` reader and therefore falls \
through to typed config or the Unset default.

Low: a thread count changes resource allocation and deadline reachability, never \
a completed bound. The observation that the internal verifier produced zero of \
33 UNSAT results motivates this seam but is not an isolated cap A/B. It remains \
default-Unset and Class::Unsafe until a versioned execution receipt proves the \
requested cap was actually installed rather than refused by an initialized \
global pool.",
        provenance: Provenance::Unmeasured {
            why_ok: "Unset preserves the historical shared global Rayon pool; no retained \
                     same-binary A/B isolates a verifier thread cap",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_CONCURRENT_THREADS` — companion-lane Rayon width.
    pub MARGIN_ROW_CONCURRENT_THREADS = LeverDecl {
        name: "NY_MARGIN_ROW_CONCURRENT_THREADS",
        kind: LeverKind::UsizeOverride { min: 0 },
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Optional platform-sized Rayon width for the concurrent margin-row lane. The \
historical environment parser is an UNTRIMMED `parse::<usize>()`: every value \
including zero is admissible, while whitespace, malformed input, overflow and \
non-Unicode values are ignored and preserve typed \
`margin_row.concurrent_threads` or the Unset default.

Unset and zero are deliberately distinct. Without a positive verifier request, \
Unset uses the shared global pool. With one, Unset derives a private width only \
as host parallelism minus the OBSERVED actual global-pool width; missing \
observations or no positive remainder refuse the lane. It never derives from the \
requested verifier cap. Explicit zero forces the shared global pool; a positive \
value requests a private pool of exactly that width, and build failure refuses \
the lane instead of falling back globally. Low: the choice changes contention \
and deadline reachability, never a completed bound. It remains unmeasured and \
Class::Unsafe until execution evidence binds host cores, actual global width, the \
requested lane width, and private-pool success or refusal.",
        provenance: Provenance::Unmeasured {
            why_ok: "Unset preserves the historical dynamic/global-pool policy, while retained \
                     conversion evidence does not isolate a concurrent-lane width",
        },
        owner: SCHEDULE,
        readers: SCHEDULE_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_K_HEAD` — exact head-neuron branch shortlist width.
    pub MARGIN_ROW_K_HEAD = LeverDecl {
        name: "NY_MARGIN_ROW_K_HEAD",
        kind: LeverKind::UsizeTrimmedOverride { min: 1 },
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Number of head-layer branch candidates scored exactly per margin-row expansion, \
default 8. The historical environment parser trims surrounding whitespace, \
parses platform-sized `usize`, and accepts only positive values. A valid \
environment value wins over typed `margin_row.k_head`; malformed, zero, \
overflowing, and non-Unicode values are recorded as ignored and preserve the \
typed preset or declaration default instead of suppressing it. The environment \
floor does not reject typed preset `0`: that historical no-preset sentinel is \
accepted into the frozen record and normalized by the runtime policy to the \
built-in width 8.

This width changes WHICH head domains enter the certified scorer and therefore \
the proof trajectory and deadline reachability, while every published bound \
still comes from the unchanged certified scorer. It is Low for the same \
ranker/schedule-only reason as `NY_MARGIN_ROW_BRANCH_REDUCE_MAX`; the governed \
search space remains conservatively VerdictAffecting because the trajectory can \
change which sound verdict is reached before the deadline. The \
CIFAR100 preset ships 16 from the retained k8/k16/k32 row conversion and moat; \
other categories retain the declaration default unless their own evidence sets \
the typed key. Scored execution consumes the receipt-owned frozen runtime policy.",
        provenance: Provenance::Measured {
            commit: "b0f30eb93fda31ef15938246da609d8d4e581bfb",
            date: "2026-08-19",
            artifact: "configs/vnncomp25/cifar100_2024.yaml #margin-row-branch-width \
                       evidence; docs/BIG3_CORE_CAMPAIGN_2026-08-19.md",
            delta: "on CIFAR100 resnet_medium idx_8600, k8 timed out after 728 \
                    expansions while k16 proved UNSAT in 186 and k32 in 173; \
                    the k16 moat retained idx_6659 and idx_8762 UNSAT 3/3 with \
                    SAT rows unchanged",
        },
        owner: BRANCHING,
        readers: BRANCH_WIDTH_READERS,
    };

    /// `NY_MARGIN_ROW_K_TRUNK` — exact trunk-neuron branch shortlist width.
    pub MARGIN_ROW_K_TRUNK = LeverDecl {
        name: "NY_MARGIN_ROW_K_TRUNK",
        kind: LeverKind::UsizeTrimmedOverride { min: 1 },
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Number of trunk-layer branch candidates scored exactly per margin-row expansion, \
default 8. Parser and precedence are deliberately identical to \
`NY_MARGIN_ROW_K_HEAD`: trim, parse `usize`, require a positive value; valid \
environment input overrides typed `margin_row.k_trunk`, while malformed, zero, \
overflowing, and non-Unicode input is recorded as ignored and leaves the lower \
typed/default layer authoritative. Typed preset `0` remains an admissible \
historical no-preset sentinel and the runtime policy normalizes it to width 8.

Head and trunk candidates share one exact batched scoring pass, but this width \
chooses a distinct candidate population and can therefore change the certified \
tree explored before the deadline. It is Low moat because it only schedules \
which identities reach the unchanged certified scorer, while the search axis \
stays conservatively VerdictAffecting for fixed-wall outcomes. CIFAR100 \
ships 16 as the jointly measured branch-width treatment; absence elsewhere \
retains 8. The scored lane derives and consumes the pair from the same frozen \
LeverSet that owns its receipt; the compatibility resolver is explicitly \
non-scored.",
        provenance: Provenance::Measured {
            commit: "b0f30eb93fda31ef15938246da609d8d4e581bfb",
            date: "2026-08-19",
            artifact: "configs/vnncomp25/cifar100_2024.yaml #margin-row-branch-width \
                       evidence; docs/BIG3_CORE_CAMPAIGN_2026-08-19.md",
            delta: "on CIFAR100 resnet_medium idx_8600, k8 timed out after 728 \
                    expansions while k16 proved UNSAT in 186 and k32 in 173; \
                    the k16 moat retained idx_6659 and idx_8762 UNSAT 3/3 with \
                    SAT rows unchanged",
        },
        owner: BRANCHING,
        readers: BRANCH_WIDTH_READERS,
    };

    /// `NY_MARGIN_ROW_BETA_LAMBDA` — Polyak relaxation factor.
    pub MARGIN_ROW_BETA_LAMBDA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_LAMBDA",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(1.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
The fraction of the direct-path gap a Polyak step is sized to close. Parser is \
`trim().parse::<f64>()`, finite, with the `> 0.0` half of the legacy filter \
left AT THE READER so an explicit `0` still resolves and stays distinguishable \
from absence in the receipt; both land on the 1.0 default, so this is a \
receipt-fidelity choice and not a behaviour change.

Inert unless `NY_MARGIN_ROW_BETA` is armed, which ships dark. MoatRisk::High \
because when the beta lane IS armed this scales every ascent step and so \
changes which certified bound the authoritative margin-row lane publishes.",
        provenance: Provenance::Unmeasured {
            why_ok: "the beta program is recorded BUILT-but-unmeasured; this tunes a \
                     lane that is itself dark, so no shipped path reads it",
        },
        owner: BETA,
        readers: BETA_LAMBDA_READERS,
    };

    /// `NY_MARGIN_ROW_BETA_POLYAK` — opt OUT of Polyak step sizing.
    pub MARGIN_ROW_BETA_POLYAK = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_POLYAK",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
OPT-OUT, and the polarity is the point: the reader tests `!= Some(\"0\")`, so \
absence and every token other than exact `0` leave Polyak sizing ENGAGED. The \
declaration records that shipped arm as `DefaultSpec::Bool(true)`.

`DefaultSpec::Bool(true)` here does NOT mean this lever arms anything on a \
scored run. It is a sub-switch of `NY_MARGIN_ROW_BETA`, which ships dark, so \
the whole beta lane — and therefore this choice — is inert by default. That is \
why the group test admits an armed default for it and for \
`NY_MARGIN_ROW_BETA_HEADS` while forbidding one anywhere else in this module.",
        provenance: Provenance::Unmeasured {
            why_ok: "sub-switch of a dark master gate; unreachable on any shipped path",
        },
        owner: BETA,
        readers: BETA_POLYAK_READERS,
    };

    /// `NY_MARGIN_ROW_BETA_HEADS` — opt OUT of head-split beta terms.
    pub MARGIN_ROW_BETA_HEADS = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_HEADS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
OPT-OUT with the same shape and the same reasoning as \
`NY_MARGIN_ROW_BETA_POLYAK`: exact `0` disables head-split beta terms, \
everything else leaves them engaged, and the lane is inert while \
`NY_MARGIN_ROW_BETA` ships dark.",
        provenance: Provenance::Unmeasured {
            why_ok: "sub-switch of a dark master gate; unreachable on any shipped path",
        },
        owner: BETA,
        readers: BETA_HEADS_READERS,
    };


    /// `NY_MARGIN_ROW_BETA` — the lane's β-CROWN split Lagrangians.
    pub MARGIN_ROW_BETA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#margin-row-beta`: one `beta_j >= 0` Lagrangian term per branch-and-bound \
split in the margin-row lane's backward pass, plus the head-split terms carried \
on the margin seed. Exact `\"1\"` arms and exact `\"0\"` disarms; every other byte \
string (`\"true\"`, `\" 1\"`, `\"01\"`, `\"\"`, non-Unicode) is a RECORDED REJECTION \
that resolves to this declaration's `false` default. An armed run announces \
itself once on stderr (`[beta] armed`) even if no domain ever carries a split, \
so an INERT arming is detectable in one log line — that telemetry is why the \
decision is latched in a `OnceLock`.

WHY IT EXISTS: without it the lane's splits only piece-fix a neuron's gate, so \
the split constraint never enters the bound as a dual term and children barely \
improve on parents (measured frontier explosion: idx_8600 goes 18 -> 415 open \
domains at depth 30 while idx_6659 drains and proves).

WHY High. The soundness argument is strong — weak duality makes \
`min_region f >= min_region [f - sum_j beta_j s_j z_j]` hold for ANY \
`beta >= 0`, the engine realizes the terms as coefficient shifts BEFORE the one \
unchanged certified concretize, and a proposal is accepted only because that \
same certified pass reported a better bound. But High is not a soundness \
score: armed, this changes which certified bound the lane PUBLISHES for every \
domain that carries a split, and it spends extra certified passes against the \
instance deadline, so it can convert a row in either direction. Disarmed the \
`DomainGates::beta` map stays empty, the seed is the untouched base margin seed, \
and the engine's application site is never entered — bit-identical passes.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and bit-identical disarmed; the campaign ledger \
                     records this build as BUILT, unmeasured, and no armed-vs-unarmed \
                     scored-row A/B is retained, so it cannot leave Bucket::Debug",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "arm the lane's β-CROWN split Lagrangians (latched once, with engagement telemetry)",
            site: "crates/ny-propagate/src/margin_row/beta.rs:enabled",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_ETA` — step size for the LEGACY sign-step arm.
    pub MARGIN_ROW_BETA_ETA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_ETA",
        // Trimmed, finite, unbounded above: the legacy reader is
        // `trim().parse::<f64>()` with NO upper filter, so `max` is `f64::MAX`
        // rather than an invented ceiling. The lower end is CLOSED at zero on
        // purpose — see the doc: the `> 0.0` half of the filter stays at the
        // reader so an explicit `0` is still distinguishable from absence.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.5),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Relative β ascent step size, in units of the split neuron's incoming \
`|coefficient|`, default 0.5. Surrounding whitespace is trimmed before \
`parse::<f64>()`; the reader then keeps the legacy `is_finite() && > 0.0` \
filter, so an explicit `0`, a negative value, `nan`, `inf` and any malformed \
token all leave 0.5. Latched once per process.

IT ONLY APPLIES TO ONE ARM. The shipped step rule is the gap-targeted Polyak \
step `t = lambda*(0 - b_direct)/sum g^2`, which IGNORES eta entirely; this knob \
is read only when `NY_MARGIN_ROW_BETA_POLYAK=0` selects the legacy sign-only \
step `±eta*|v_k|`. It is therefore inert twice over on a shipped run: β must be \
armed AND the Polyak rule must be disarmed.

High rather than Low because when it does apply it changes which β proposals \
the certified scorer accepts, and therefore which certified bound the \
authoritative margin-row lane publishes — a different value here is a different \
proof attempt, not a different amount of logging.",
        provenance: Provenance::Unmeasured {
            why_ok: "doubly inert on the shipped configuration (β is dark, and the \
                     Polyak rule that ignores eta is the default step rule when it is not); \
                     no retained sweep qualifies any other step size",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "step size for the legacy sign-only β step (the NY_MARGIN_ROW_BETA_POLYAK=0 A/B arm)",
            site: "crates/ny-propagate/src/margin_row/beta.rs:eta",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_ITERS` — β ascent trials per domain evaluation.
    pub MARGIN_ROW_BETA_ITERS = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
β ascent trials per expanded domain, default 1. Whitespace is trimmed before \
`parse::<usize>()`; absent, malformed, negative and overflowing input all leave \
1, and the reader then CLAMPS the resolved value to `1..=8`, so an in-range \
number outside that window is pulled to the nearest bound rather than rejected. \
Latched once per process.

EACH TRIAL IS ONE MORE CERTIFIED PASS on every expanded domain that carries a \
split. That is the whole cost model of `#margin-row-beta`, which is why this \
is High and not Low: raising it both changes which β is accepted (hence the \
published bound) and multiplies the lane's per-expansion cost against a fixed \
instance deadline, so it can lose rows the shipped setting would have proved. \
Inert unless NY_MARGIN_ROW_BETA is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     the runbook proposes sweeping 3 as an experiment, which is a plan, \
                     not a measurement",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "ascent trials per domain evaluation for the lane's β-CROWN",
            site: "crates/ny-propagate/src/margin_row/beta.rs:iters",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_PERCOL` — per-failing-column β state.
    pub MARGIN_ROW_BETA_PERCOL = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_PERCOL",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects the per-column β-CROWN program: each of the worst failing output \
columns carries and ascends its own split multipliers instead of sharing one \
vector scored against only the worst column. Exact `1` arms and exact `0` \
disarms; every other spelling is rejected to the false declaration default. \
The typed preset key is `margin_row.beta_per_column`, and a present admissible \
legacy environment value retains precedence in both directions.

Inert unless β ascent is live through either `NY_MARGIN_ROW_BETA=1` or \
`NY_MARGIN_ROW_CLOSURE_ADAPTIVE=1`. When armed on either route this changes \
the certified proposals accepted by the authoritative margin-row lane and can \
change both bound progress and cost, so it remains High and default-dark. The \
scored reader consumes the immutable entry-time LeverSet through \
`MarginRowRuntimePolicy`; compatibility/research entry points resolve once at \
lane entry rather than through the former process-wide OnceLock.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one child of the already-dark beta program; no shipped \
                     preset arms it and no retained official A/B qualifies a default",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "select per-column beta state from the scored run's immutable runtime policy",
            site: "crates/ny-propagate/src/margin_row/bab.rs:eval",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_COLS` — maximum per-column β width.
    pub MARGIN_ROW_BETA_COLS = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_COLS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Maximum number of worst failing output columns that receive independent β \
state, default 8. The historical parser trims and parses `usize`; malformed, \
negative, non-UTF-8 and overflowing values restore 8, then the consuming \
policy clamps the result to `[1, 32]`. The typed preset key is \
`margin_row.beta_columns`.

Inert unless `NY_MARGIN_ROW_BETA_PERCOL=1` and β ascent is live through either \
`NY_MARGIN_ROW_BETA=1` or `NY_MARGIN_ROW_CLOSURE_ADAPTIVE=1`. Where live, \
changing the width changes which independently certified column proposals are \
attempted and the work spent before the deadline, so it \
is High and unmeasured. Runtime execution evidence records whether this width \
actually bound a live candidate set; configuration presence alone earns no \
search credit.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark sizing child of the dark per-column beta program; the shipped \
                     default is preserved and no retained official width sweep exists",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "truncate the live failing-column beta candidate set from the immutable policy",
            site: "crates/ny-propagate/src/margin_row/bab.rs:eval",
        }],
    };

    /// `NY_MARGIN_ROW_COL_RETIRE` — inherit certified closed columns.
    pub MARGIN_ROW_COL_RETIRE = LeverDecl {
        name: "NY_MARGIN_ROW_COL_RETIRE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Retires a margin column after an ancestor has certified it closed, so \
descendants recompute only the surviving columns. Exact `1` arms; exact `0` \
and rejected spellings retain the false declaration default. The typed preset \
key is `margin_row.column_retire`.

The retirement invariant is certificate-backed and inherited monotonically, \
but it changes the width handed to later authoritative passes and therefore \
the proof trajectory and deadline reachability. It stays High, default-dark, \
and unmeasured. The scored path carries the decision in \
`MarginRowRuntimePolicy`; execution evidence records new certificates but \
grants search credit only after an inherited retired set narrows a live pass.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one optimization with certificate-backed fail-closed \
                     semantics; no shipped preset arms it and no official A/B qualifies it",
        },
        owner: LANE_RUNTIME,
        readers: &[ReaderSite {
            scope: LANE_RUNTIME,
            role: "narrow descendant margin batches using inherited certified closures",
            site: "crates/ny-propagate/src/margin_row/bab.rs:eval",
        }],
    };

    /// `NY_MARGIN_ROW_PARALLEL_PREBUILD` — prebuild y-packs for a frontier batch.
    pub MARGIN_ROW_PARALLEL_PREBUILD = LeverDecl {
        name: "NY_MARGIN_ROW_PARALLEL_PREBUILD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Prebuilds the deterministic y-pack for every domain selected into a parallel \
frontier batch before their bound evaluations start. Exact `1` arms; exact \
`0` and rejected spellings retain the false declaration default. The typed \
preset key is `margin_row.parallel_prebuild`.

A prebuilt pack is byte-for-byte the pack the domain would lazily build, so \
this changes scheduling and duplicate work only. It is Low rather than None \
because wall-clock and parallel contention can move a deadline verdict. It is \
inert unless a live frontier decision selects more than one domain; execution \
evidence records actual prebuilt batches and refuses configuration-only credit.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark schedule-only optimization; no shipped preset arms it \
                     and no retained official A/B qualifies a shipped default",
        },
        owner: LANE_RUNTIME,
        readers: &[ReaderSite {
            scope: LANE_RUNTIME,
            role: "prebuild deterministic y-packs for an executable multi-domain frontier batch",
            site: "crates/ny-propagate/src/margin_row/bab.rs:run_loop",
        }],
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA` — frozen-policy PER-DOMAIN alpha.
    pub MARGIN_ROW_DOMAIN_ALPHA = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#margin-row-domain-alpha`: the lane's lower-line slope `alpha` is \
RE-OPTIMIZED per BaB domain — against that domain's own gates, including any \
Clip-and-Verify tightening — instead of being frozen at the root value \
`alpha_opt` produced before the tree started. A child WARM-STARTS from its \
parent's accepted alpha (alpha-beta-CROWN's `domain_updater.py:123-128`), so the \
tightening compounds down the tree. Exact `\"1\"` arms and exact `\"0\"` disarms, \
IN BOTH DIRECTIONS over the typed preset; a PRESENT near-miss token (`\"true\"`, \
`\" 1\"`, `\"\"`) is a recorded rejection that suppresses the preset and lands on \
this declaration's `false` default; absence falls through to the preset, then to \
`false`. On scored VNN-COMP runs, generic preset projection places this \
resolution in the receipt-owning frozen LeverSet and \
`DomainAlphaRuntimePolicy::from_levers` derives the complete family once. The \
lane, classwise routes and nested epoch trees carry only those plain frozen \
values; no scored path rereads a preset or writes/reopens a process-global \
cell. Historical globals remain a non-scored compatibility boundary resolved \
once at lane entry. The scored route publishes its frozen values and causal \
work in the terminal domain-alpha execution receipt.

WHY IT EXISTS. ny optimizes alpha ONCE, at the root, and freezes it; every \
child clones the root vectors and overrides only the split neuron's index, so \
at depth 30 every remaining unstable neuron is still relaxed with a slope tuned \
for the whole ROOT box — a deficit that GROWS WITH DEPTH by construction. That \
is the shape the measured wall needs: the frontier-worst bound at a given depth \
is bit-identical across base, beta, beta*8, stateless clip, stateful clip and \
both branching reductions, gaining ~0.0096-0.0120 per level against a gap of \
0.155-0.203. It is also the MISSING CONSUMER for clip: abc ships clip and \
per-domain alpha as a compounding pair (tighter box -> better triangle -> \
re-optimized slope -> deeper next cut), and ny had only the first half, which \
is why a measured 1.5-2.7e-2 stateful-clip shrink moved the gating curve by \
exactly zero. Measure the two as a 2x2, never one at a time.

WHY High. Tightness-only by construction — `alpha*y <= relu(y)` holds \
pointwise for ANY `alpha in [0, 1]` (validity of the lower line is \
box-independent, which is also why inheriting an ancestor's value is safe), \
the upper chord `(s, c)` is never touched, the error-contraction constant \
`ms = max(alpha, s)` is re-derived at the single write site, exact lines are \
excluded by an `0 < s < 1` admission test plus explicit split exclusion, and \
the unchanged certified pass is the only scorer with monotone keep-best. But \
High is not a soundness score: armed, this changes the certified bound the lane \
PUBLISHES at every domain with a movable neuron, and each ascent trial is one \
more certified pass against the instance deadline, so it can convert a row in \
either direction. Disarmed, no slot is collected, no gate is written, the \
gradient capture stays off, and the lane's passes are bit-identical.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default in BOTH layers (the declaration default is false and \
                     no shipped preset sets the key); disarmed the lane is byte-identical, \
                     and the pre-registered 2x2 (off / clip / alpha / both) on idx_1298 and \
                     idx_4597 has not been run, so it cannot leave Bucket::Debug",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS` — per-domain alpha ascent trials.
    pub MARGIN_ROW_DOMAIN_ALPHA_ITERS = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(2),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Per-domain alpha ascent trials per expanded domain, default 2. Whitespace is \
trimmed before `parse::<usize>()`; absent, malformed, negative and overflowing \
input all leave 2, and the reader then CLAMPS the resolved value to `1..=8`, so \
an in-range number outside that window is pulled to the nearest bound rather \
than rejected. The typed preset key is `margin_row.domain_alpha_iters` and the \
environment wins over it in both directions. The scored route freezes this \
value with the gate and eta in one immutable runtime policy; only non-scored \
compatibility callers consult the historical process/preset cells at entry.

EACH TRIAL IS ONE MORE CERTIFIED PASS on every expanded domain with a movable \
neuron — the whole cost model. The measured column-scaling fit \
(`pass_ms ~= 60 + 2.4*R` on resnet_medium; `~45 + 10.4*R` on resnet_large) puts \
a lane pass at the measured 3-11 failing columns near 90-130 ms, so the default \
2 trials cost ~0.2-0.3 s per domain. abc runs 10 Adam steps, but it does not \
pay a certified pass per step; here the affordable shape is few trials plus the \
FREE warm start (applying a per-domain gate override measures a 0.98-1.12x \
pass-cost ratio at every width), which lets the slope keep improving down the \
tree without being re-paid at each node. High for the same two reasons as its \
parent: it changes which alpha is accepted, hence the published bound, and it \
multiplies per-expansion cost against a fixed deadline. Inert unless \
NY_MARGIN_ROW_DOMAIN_ALPHA is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     no retained sweep qualifies any other trial count",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_ETA` — per-domain alpha initial step size.
    pub MARGIN_ROW_DOMAIN_ALPHA_ETA = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA",
        // Trimmed, finite, CLOSED at zero: the `> 0.0` half of the filter stays
        // at the reader so an explicit `0` is still distinguishable from
        // absence at the chokepoint (the MARGIN_ROW_BETA_ETA convention). The
        // reader also clamps to `[ETA_MIN, ETA_MAX] = [0.01, 1.0]`, the legal
        // travel of a `[0, 1]` slope.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.25),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Initial NORMALIZED step size for the per-domain alpha ascent, default 0.25. The \
step is `alpha' = clamp(alpha + eta*g/gmax, 0, 1)`, so this is literally how far \
the largest-gradient neuron moves through the legal `[0, 1]` slope range in one \
trial; 0.25 is a quarter of that range, the value `alpha_opt`'s root search \
uses. Accepts grow it by 1.5x (capped at 1.0), rejects halve it, and the ascent \
stops below 0.01. Surrounding whitespace is trimmed before `parse::<f64>()`; the \
reader then keeps an `is_finite() && > 0.0` filter, so an explicit `0`, a \
negative value, `nan`, `inf` and any malformed token all leave 0.25. The typed \
preset key is `margin_row.domain_alpha_eta`; the environment wins over it in \
both directions. The scored route freezes this value with the gate and trial \
cap in one immutable runtime policy; only non-scored compatibility callers \
consult the historical process/preset cells at entry.

High because when it applies it changes which alpha proposals the certified \
scorer accepts, and therefore which certified bound the authoritative \
margin-row lane publishes — a different value here is a different proof \
attempt. Inert unless NY_MARGIN_ROW_DOMAIN_ALPHA is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     no retained sweep qualifies any other step size",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH` — per-domain alpha depth floor.
    pub MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
SCHEDULE: skip the per-domain alpha ascent at domains SHALLOWER than this split \
depth (`trunk + heads`), default 0 = every depth pays. Whitespace is trimmed \
before `parse::<usize>()`; absent, malformed, negative and overflowing input \
all leave 0, and the reader caps the resolved value at 64. The typed preset key \
is `margin_row.domain_alpha_min_depth` and the environment wins over it in both \
directions. Latched once per process.

WHY A SCHEDULE EXISTS. The ascent costs `iters` certified passes at EVERY \
admitted domain, uniformly in depth, while the measured gain on the \
depth-indexed gating curve GROWS with depth (+0.0017 at depth 1 to +0.0137 at \
depth 12 on idx_4597; +0.0111 to +0.0495 on idx_1298). A depth floor buys the \
deep part of that curve without paying the shallow part. Skipping is affordable \
only because the WARM START is free (0.98-1.12x pass-cost ratio at every \
width): an unpaid domain still inherits whatever tightening its ancestors \
bought. The countervailing risk is the reason this is a measurement and not an \
optimization: a SHALLOW accepted alpha is inherited by every descendant, so \
declining it may cost more of the curve than the ~14% of expansions it saves.

Skipping is TIGHTNESS-only and can never be a soundness event — an unpaid \
domain publishes the bound the disarmed lane would publish, from the same \
certified pass. High for the same reason as its parent: it changes which \
certified bound the authoritative margin-row lane publishes, and how much wall \
clock the lane spends doing it. Inert unless NY_MARGIN_ROW_DOMAIN_ALPHA is \
armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     the schedule sweep that would qualify a non-zero floor is the \
                     measurement this knob exists to make possible",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_STRIDE` — per-domain alpha depth stride.
    pub MARGIN_ROW_DOMAIN_ALPHA_STRIDE = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_STRIDE",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
SCHEDULE: pay the per-domain alpha ascent only where `depth % stride == 0`, \
default 1 = every depth pays. Whitespace is trimmed before `parse::<usize>()`; \
absent, malformed, negative and overflowing input all leave 1, and the reader \
CLAMPS the resolved value to `1..=16`. The typed preset key is \
`margin_row.domain_alpha_stride` and the environment wins over it in both \
directions. Latched once per process.

THIS IS THE AMORTIZATION ARM, and it is a real question rather than a tuning \
grid. A child WARM-STARTS from its parent's accepted alpha for free, so the \
tightening is carried down the tree without being re-paid; the sampled \
telemetry shows each node's own two trials adding only ~1.7e-3 while the \
CUMULATIVE curve moves +0.0495 by depth 6, which says most of the gain is \
carried by inheritance. If that reading is right, `stride = k` divides the cost \
by `k` and keeps most of the curve. If instead each node's own trials are what \
move it, stride divides the gain too. Those two outcomes are distinguishable in \
one sweep, which is why the knob exists.

Tightness-only: an unpaid domain publishes the bound the disarmed lane would \
publish. High for the same reason as its parent — it changes the published \
bound and the lane's wall clock. Inert unless NY_MARGIN_ROW_DOMAIN_ALPHA is \
armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     no retained sweep qualifies any stride",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_DOMAIN_ALPHA_GAP` — per-domain alpha closure window.
    pub MARGIN_ROW_DOMAIN_ALPHA_GAP = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_ALPHA_GAP",
        // Trimmed, finite, CLOSED at zero: an explicit `0` still resolves and
        // stays distinguishable from absence in the receipt, and the reader's
        // `> 0.0` filter then maps both onto "no window" (the MARGIN_ROW_BETA_ETA
        // and MARGIN_ROW_DOMAIN_ALPHA_ETA convention).
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
SCHEDULE: pay the per-domain alpha ascent only at domains within this much of \
CLOSURE, i.e. only where the pre-ascent bound satisfies `b >= -gap`. Default 0, \
which the reader reads as UNLIMITED (no window), so a silent preset is \
byte-identical to the unscheduled arm the mechanism was measured with; a \
negative value, `nan`, `inf` and any malformed token land there too. \
Surrounding whitespace is trimmed before `parse::<f64>()`. The typed preset key \
is `margin_row.domain_alpha_gap`; the environment wins over it in both \
directions. Latched once per process.

WHY THIS IS THE SHARPEST OF THE THREE SCHEDULE KNOBS. cifar100's failing rows \
fail by a COUNTING problem, not a bound-quality one: the open frontier goes \
18 -> 41 -> 156 -> 297 -> 415 while the bound improves at the same rate as a \
proving row's. What converts a row is therefore not a better frontier-WORST but \
MORE DOMAINS CLOSING — and the domains that CAN be made to close are the ones \
already near the strict `b > 0` threshold. Two certified passes spent on a \
domain at -0.30 buy ~+0.002 and close nothing; the same two spent at -0.004 can \
close it outright and delete it, and its whole subtree, from the frontier. This \
gate spends the passes where they can retire a domain.

Tightness-only: an unadmitted domain publishes the bound the disarmed lane \
would publish. High for the same reason as its parent — it changes the \
published bound and the lane's wall clock. Inert unless \
NY_MARGIN_ROW_DOMAIN_ALPHA is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     no retained sweep qualifies any window",
        },
        owner: DOMAIN_ALPHA,
        readers: DOMAIN_ALPHA_POLICY_READERS,
    };

    /// `NY_MARGIN_ROW_BACKWARD_INTERM` — backward-computed root intermediates,
    /// layered OVER the typed preset.
    pub MARGIN_ROW_BACKWARD_INTERM = LeverDecl {
        name: "NY_MARGIN_ROW_BACKWARD_INTERM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#backward-interm`: each trunk ReLU's INPUT box is recomputed with the \
lane's own certified backward engine (identity rows seeded at that layer's \
pre-activation, run through the already-frozen prefix gates, concretized over \
the root box — alpha-CROWN intermediate bounds in the lane's arithmetic) and \
INTERSECTED shrink-only with the forward (M, D) tableau box before \
`gates_from_box` derives `(alpha, s, c, ms)`. It runs DURING the forward build, \
layer by layer, so each layer's pass consumes gates that earlier layers have \
already tightened and the effect COMPOUNDS down the trunk. Root build only \
(`splits.is_empty()`), never inside Tier-2 epoch rebuilds, and never in the \
bit-parity `RoundMode` — the parity oracle against `core.py` must not move.

THIS IS AN OVERRIDE, NOT THE ONLY WAY IN. The typed preset key \
`margin_row.backward_interm` is what can arm this on a SCORED run, because \
`vnncomp_scripts/run_instance.sh` exports no `NY_*` variables, so an \
env-only lever cannot fire in competition however well it measures. Layering is \
`read_over_config`'s, which is precisely the three-way rule the raw reader \
already implemented: exact `\"1\"` arms and exact `\"0\"` disarms, IN BOTH \
DIRECTIONS over the preset; a PRESENT near-miss token (`\"true\"`, `\" 1\"`, `\"\"`) \
is a recorded rejection that suppresses the preset and lands on this \
declaration's `false` default, so a typo is a kill switch rather than a silent \
promotion; absence falls through to the preset, then to `false`.

WHY High, given the intersection is shrink-only. Two valid enclosures intersect \
to a valid enclosure, a crossed intersection is treated as a defect signal \
rather than an infeasibility certificate (this is the ROOT box; acting on \
\"empty\" would be a false-UNSAT), and skipping any layer, chunk or neuron \
simply leaves the forward box standing. The risk is the documented ORDERING \
TRAP: `LayerGates::clip_rows` slack calibration must recover the LINE's own \
certified slack, and against a TIGHTENED published `l` that difference \
under-recovers (clamping to 0), which would make every Clip-and-Verify \
halfspace built from that line cut into the true subdomain — a false-`unsat` \
generator. `root.rs` calibrates against the forward-only bounds captured before \
this phase runs, and that ordering is the only thing standing between an armed \
run and an unsound cut. It also publishes different intermediate bounds and \
spends a budget slice, either of which can move a verdict.",
        provenance: Provenance::Measured {
            commit: "4227a10bc",
            date: "2026-08-19",
            artifact: "docs/BIG3_CORE_CAMPAIGN_2026-08-19.md, CYCLES 1-3 + SHIP; \
                       category rationale in configs/vnncomp25/cifar100_2024.yaml and \
                       configs/vnncomp25/tinyimagenet_2024.yaml",
            delta: "measured and shipped on both scored categories: CIFAR100 deep roots \
                    tightened by about 0.18-0.20 and proving rows became 5-10x cheaper; \
                    the same-session TinyImageNet k16 context converted idx_4330 timeout \
                    -> UNSAT 3/3, and the category ship gate recorded zero regressions. \
                    Later history identifies the k16 context, so it is not a claim that \
                    idx_4330's banked wall represents today's shipped configuration",
        },
        owner: BACKWARD_INTERM,
        readers: &[ReaderSite {
            scope: BACKWARD_INTERM,
            role: "arm the backward-intermediate phase over the typed `margin_row.backward_interm` preset",
            site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
        }],
    };

    /// `NY_MARGIN_ROW_EPOCH_PARENT_TOPK` — default-off incremental epoch
    /// refresh width.
    pub MARGIN_ROW_EPOCH_PARENT_TOPK = LeverDecl {
        name: "NY_MARGIN_ROW_EPOCH_PARENT_TOPK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the incremental parent-frame variant of margin-row Tier-2 epochs when set \
to a positive integer, or when the typed \
`margin_row.epoch_parent_topk` preset key is positive. The value is the maximum \
number of widest downstream unstable preactivations re-bounded per ReLU layer. \
An explicit `0` and absence both keep the established full-tableau epoch \
implementation unchanged; malformed, negative and overflowing environment \
values are rejected to the same dark zero default and suppress the preset. A \
positive `margin_row.epoch_depth` (or legacy `NY_EPOCH_DEPTH`) must independently \
select a cadence or the reader reports the request INERT and no refresh can run. \
No shipped preset supplies any epoch key. On a scored run, the generic \
`preset_delivered_config_value` schema projection includes this unsafe axis in \
the frozen LeverSet, and `MarginRowRuntimePolicy::from_levers` consumes that \
same receipt-owned value once with the other epoch controls. The lane and every \
nested tree then carry the immutable policy without reopening the preset or \
process environment. This delivery plumbing does NOT promote the axis: \
`MARGIN_ROW_EPOCH_PARENT_TOPK` remains outside the safe search space.

The incremental path clones the already-certified parent intervals, bakes the \
domain's split ReLUs as exact active/inactive lines, re-bounds only the selected \
downstream columns with the existing certified prefix-backward engine, and \
intersects shrink-only before re-deriving gates. It may recur only through a \
fixed four-generation budget carried down the nested tree, so the call stack is \
explicitly bounded. The intended benefit is to reduce the full-tableau rebuild \
cost observed on CIFAR100 idx_1298; the magnitude and solve impact remain \
unmeasured and are not shipped claims.

MoatRisk::High because an armed refresh changes published intermediate boxes \
and hence authoritative margin bounds, while its extra prefix passes consume \
the same official deadline as BaB. Soundness is structural (subset-valid exact \
split lines, certified prefix passes, shrink-only intersection), but transfer \
and wall-time value require a sealed scored-row A/B.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark default, no shipped preset arm, fixed nesting cap, and the retained \
                     CIFAR100 idx_1298 measurement is explicitly still required",
        },
        owner: EPOCH_PARENT,
        readers: EPOCH_PARENT_POLICY_READERS,
    };

    /// `NY_EPOCH_DEPTH` — the Tier-2 epoch CADENCE, layered over the typed
    /// `margin_row.epoch_depth` preset.
    pub EPOCH_DEPTH = LeverDecl {
        name: "NY_EPOCH_DEPTH",
        // NOT `UsizeTrimmed`. The historical reader is
        // `raw.parse::<usize>().unwrap_or(0)` with NO trim, so `" 4"` has
        // always been a REJECTION that falls back to the default. Declaring it
        // trimmed would silently start accepting a token the lever never
        // accepted, which is a behaviour change dressed as a migration.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Number of TRUNK SPLITS a domain must carry before the margin-row lane \
re-linearizes its subtree (`bab.rs:try_epoch`). `0` (the shipped default) and \
absence keep the epoch dark; `d > 0` fires one rebuild per qualifying domain, \
capped by `NY_EPOCH_ATTEMPTS`.

WHY THIS ONE MATTERS MORE THAN ITS SIZE SUGGESTS. `try_epoch` calls \
`RootGates::build_retaining(net, lo, hi, mode, deadline, retain_cfg, &abs)` with \
the trunk splits baked in as ABSOLUTE neuron indices. That is a full rebuild of \
every intermediate UNDER the split constraints — alpha-beta-CROWN's actual \
mechanism, and the only code path in ny that has it. Every other child bound in \
the lane comes from `domain_gates_split_only`, a purely local gate edit.

WHAT WAS MEASURED, and it is not what anyone expected \
(`docs/PROGRAM_2026-08-22_CLOSURE_FIRST.md` §0d; cifar100 `idx_4169`, official \
100 s, same binary, under `scripts/ny_box.sh measure`, all arms \
`quiesced=yes`). Arm A baseline: one tree loop, root −0.371373. Arm B \
`NY_EPOCH_DEPTH=4`: TWO tree loops and a stop reason naming the epoch attempt \
(engagement confirmed by two unfakeable observables), second loop entering at \
**−0.389998** — WORSE than its own parent, on a domain with four neurons FIXED, \
i.e. a strict subset of the box whose bound cannot legitimately be looser. Arm \
C `NY_EPOCH_DEPTH=8` is the control: the tree never accumulates eight trunk \
splits inside 93 s, so the epoch never fires and arm C reproduces arm A's stop \
reason exactly. Cause: `build_retaining` is the ROOT build, while the \
`#backward-interm` tightening is applied separately during lane setup and was \
never re-run for the nested tree, so the rebuild discarded 584 tightened \
columns bought with 7.11 s in exchange for 4 fixed neurons. \
`inherit_parent_intervals` then intersected the parent's certified intervals \
into the rebuilt frame (shrink-only, subset-valid) and flipped the sign 3/3: \
−0.371373 → **−0.316766**, +0.0546 better than the parent against 0.0186 worse \
before, a +0.0732 swing, and the rebuild is now monotone by construction. \
Closure stayed 0.000, so this is a BOUND result, not yet a conversion.

AND IT HAD NEVER RUN IN COMPETITION. Until `a1c80f6a3` this was an undeclared \
raw `env::var` with a 0 default and no preset key, and \
`vnncomp_scripts/run_instance.sh` exports no `NY_*` variables. The current \
scored route is receipt-owned end to end: generic \
`commands/vnncomp.rs:preset_delivered_config_value` projects \
`margin_row.epoch_depth` and the other searchable epoch fields from the one \
immutable preset snapshot into the frozen LeverSet, then \
`MarginRowRuntimePolicy::from_levers` derives the complete epoch policy from \
that same set. Outer, classwise and nested trees receive only those frozen \
plain values; no scored path performs a second preset read or a late raw \
environment read. No shipped yaml sets the key, so both layers remain dark.

LAYERING is the frozen snapshot builder's `read_over_config` contract: an \
admissible environment integer wins over the projected preset value; a PRESENT \
malformed or non-UTF-8 value suppresses the preset and lands on this \
declaration's `0`, so a typo is a kill switch and never a silent promotion; \
only true absence falls through to the preset, then to `0`. The receipt and \
runtime therefore cannot select different depth values.

High because an armed cadence publishes DIFFERENT intermediate boxes and \
therefore different margin bounds, and because each rebuild spends the same \
official per-instance deadline the tree search needs.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark in BOTH layers — the declaration default is 0 and no shipped \
                     preset sets `margin_row.epoch_depth` — so an unset run is \
                     byte-identical to its history. The one banked A/B (idx_4169, \
                     2026-08-22) is a BOUND measurement on a single row that produced \
                     0.000 closure movement, which qualifies nothing",
        },
        owner: EPOCH_BAB,
        readers: EPOCH_BAB_POLICY_READERS,
    };

    /// `NY_EPOCH_ATTEMPTS` — cap on Unknown-ending Tier-2 rebuilds per run,
    /// layered over the typed `margin_row.epoch_attempts` preset.
    pub EPOCH_ATTEMPTS = LeverDecl {
        name: "NY_EPOCH_ATTEMPTS",
        // Same untrimmed `parse::<usize>()` contract as NY_EPOCH_DEPTH.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(2),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
How many Tier-2 epoch rebuilds that end WITHOUT closing their subtree the lane \
will pay for in one run. Default 2 — note this is the one member of the family \
whose historical default is NOT zero, so absence does not mean `off` here; the \
family is held dark by `NY_EPOCH_DEPTH`/`margin_row.epoch_depth` being 0, and \
this number only decides how expensive an ARMED cadence is allowed to get.

An explicit `0` is a real value and a second kill switch: with attempts at 0 no \
rebuild runs however low the cadence is set, and `lane_impl` reports the \
coupled `#epoch-parent` request INERT with `epoch_attempts=0` as the named \
reason. Same layering and same untrimmed integer parser as `NY_EPOCH_DEPTH`: \
admissible env wins, a present malformed value suppresses the typed \
`margin_row.epoch_attempts` preset and restores this declaration's 2, absence \
falls through to the preset. For scored runs the generic preset schema \
projection records that resolution in the frozen LeverSet, and \
`MarginRowRuntimePolicy::from_levers` consumes the receipt-owned attempt cap \
once for every outer, classwise and nested epoch path.

High for the same reason as the cadence: every attempt this permits is a full \
`build_retaining` rebuild that publishes its own intermediate boxes and spends \
the official per-instance deadline.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while `NY_EPOCH_DEPTH` and `margin_row.epoch_depth` are both 0, \
                     which is the shipped state; the nonzero default is the historical \
                     value preserved byte for byte by this migration, not a new arm",
        },
        owner: EPOCH_BAB,
        readers: EPOCH_BAB_POLICY_READERS,
    };

    /// `NY_EPOCH_TIER0` — the Tier-0 rank-1 candidate ranker's shortlist width.
    pub EPOCH_TIER0 = LeverDecl {
        name: "NY_EPOCH_TIER0",
        // Untrimmed `parse::<usize>()`, exactly as the historical reader.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Arms Tier-0 candidate ranking and sets `k`, the number of candidates that reach \
the exact Tier-1 pass. `0` (default) and absence keep the legacy shortlist \
protocol (`head_prerank` + `trunk_shortlist`). A positive `k` scores a WIDE \
universe cheaply with the rank-1 ranker — O(nf·n_in) per candidate against \
retained f32 tableau rows — and forwards only the best `k`.

RANKER-ONLY BY CONSTRUCTION, and that is why this is Low rather than High. The \
retained rows are f32 and are read by exactly one consumer, the nearest-mode \
`bounds::trunk_variant` ranker (`root.rs`, `RetainedLayer` doc: \"No Outward \
(verdict-grade) pass ever reads them\"); `bab.rs:tier0_candidates` reorders a \
candidate list and nothing else. Every pushed bound and every closure still \
comes from the unchanged certified Outward `score_candidates`. So a bad ranking \
reaches a DIFFERENT SET OF DOMAINS inside the budget — a points risk — and \
cannot manufacture a proof. Same Low/High line as \
`NY_MARGIN_ROW_BRANCH_REDUCE_MAX`: High means `can move a published bound`, not \
`can change the answer`.

MEASURED 2026-07-18, ON A BINARY THAT NO LONGER EXISTS \
(`docs/EPOCH_BAB_STATE_2026-07-18.md`): exact scoring 124 -> 66 ms/call, \
expansion counts IDENTICAL wherever both lanes close the same tree (2, 13, 29, \
23, 31 — the verified tree is reproduced exactly), and moat intact across 14 \
cifar GT-sat rows with Tier-0 on. What was NOT established is a net closure \
gain: on the 13-instance pyrat-easy tier the count is 6 = 6 (Tier-0 gains \
`prop_1588`, loses `prop_4605`). Since a solve is worth ten points against a \
small time bonus, closure is the load-bearing number and it was a wash, so the \
lever was left dark pending an A/B over the full wall set. That measurement \
predates `#backward-interm`, `#rebake` and `inherit_parent_intervals`, so it \
does not qualify anything on today's sound path.

SIDE EFFECT WORTH KNOWING: a positive `k` is also one of the two conditions \
that turn ROOT-BUILD RETENTION on (`NY_EPOCH_RETAIN` sizes it), so arming this \
adds root build time and memory before any ranking happens. The typed scored \
route is `margin_row.epoch_tier0`: the generic schema projection layers an \
admissible environment value over that field in the frozen LeverSet, and \
`MarginRowRuntimePolicy::from_levers` carries the same receipt-owned shortlist \
width into every ranker call.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and set by no shipped preset. The 2026-07-18 A/B \
                     is a throughput result on a July binary and \
                     explicitly did NOT establish a closure gain, so it qualifies no \
                     default on the current sound path",
        },
        owner: EPOCH_BAB,
        readers: EPOCH_BAB_POLICY_READERS,
    };

    /// `NY_EPOCH_UNIVERSE` — how many trunk candidates the Tier-0 ranker scores.
    pub EPOCH_UNIVERSE = LeverDecl {
        name: "NY_EPOCH_UNIVERSE",
        // Untrimmed `parse::<usize>()`, exactly as the historical reader.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(64),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
The trunk-candidate universe the Tier-0 ranker scores before truncating to the \
best `NY_EPOCH_TIER0` overall (`bab.rs:tier0_candidates`, \
`dyn_scored.truncate(self.cfg.tier0_universe)`). Default 64.

NOTE THE NONZERO DEFAULT: absence does not mean `off`. This number is frozen \
unconditionally in the entry policy and is simply never consumed while \
`NY_EPOCH_TIER0` is 0, which is the shipped state. Setting it alone therefore \
changes nothing — it is a width knob under an arming gate, not a gate.

Low for the same reason as its arming gate: the wider universe only changes \
WHICH candidates are offered to the unchanged certified Outward pass. It cannot \
move a published bound, but it is not free — ranking cost grows with the \
universe, and the lane runs at roughly one expansion per second on the rows \
that matter. The typed scored route is `margin_row.epoch_universe`; the \
generic schema projection records any higher-precedence environment override \
in the frozen LeverSet, and `MarginRowRuntimePolicy::from_levers` supplies that \
same receipt-owned universe to the ranker.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while `NY_EPOCH_TIER0` is 0, which is the shipped state; the \
                     default 64 is the historical value preserved byte for byte by this \
                     migration",
        },
        owner: EPOCH_BAB,
        readers: EPOCH_BAB_POLICY_READERS,
    };

    /// `NY_EPOCH_RETAIN` — retained pre-activation rows per trunk ReLU layer.
    pub EPOCH_RETAIN = LeverDecl {
        name: "NY_EPOCH_RETAIN",
        // Untrimmed `parse::<usize>()`, exactly as the historical reader.
        kind: LeverKind::U64,
        default: DefaultSpec::U64(128),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
`RetainCfg::per_layer`: how many unstable neurons per trunk ReLU layer keep \
their pre-activation sandwich rows `M ± D` after the root build, chosen by the \
relaxation-slack score `c * (u - l)`. Default 128, matching \
`RetainCfg::default()`. The companion `budget_bytes` cap (256 MiB) is NOT \
exposed and stops retention from growing without bound; later layers are \
dropped first-come once it is exceeded.

WHETHER RETENTION HAPPENS AT ALL is decided elsewhere, by a DISJUNCTION this \
declaration cannot model as a single search-space edge: `lane_impl` builds a \
`RetainCfg` when `NY_EPOCH_TIER0 > 0` OR when `epoch_depth > 0` and the \
incremental `#epoch-parent` refresh is off. In the second case the rows are \
retained and then never read, because the nested epoch config only consumes \
them when `tier0_exact > 0` — so with a cadence but no Tier-0 this knob buys \
root build time and memory and nothing else.

Low because retention is PURE CAPTURE. `root.rs:build_walk` copies rows out of \
`mi`/`di` after `concretize_box` has already produced `l`/`u`; it does not \
touch `l`, `u`, `alpha`, `s`, `c`, `ms` or `unst`, and the rows are f32 and \
ranker-only. A run with retention on and a run with it off publish the same \
bounds; they differ in wall clock and resident memory. The typed scored route \
is `margin_row.epoch_retain`: generic preset projection records it and any \
higher-precedence environment override in the frozen LeverSet, while \
`MarginRowRuntimePolicy::from_levers` supplies that same receipt-owned row \
budget to root construction and nested rankers.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert unless a Tier-0 width or an epoch cadence is armed, and both \
                     are dark on every shipped configuration; the default 128 is \
                     `RetainCfg::default()`'s own value, preserved by this migration",
        },
        owner: EPOCH_BAB,
        readers: EPOCH_BAB_POLICY_READERS,
    };

    /// `NY_EPOCH_DEBUG` — PRESENCE-armed Tier-2 epoch trace.
    pub EPOCH_DEBUG = LeverDecl {
        // The historical grammar is PRESENCE-armed through
        // `env::var(..).is_ok()`, so it is declared Text/Unset for the same
        // reason `NY_ACASXU_PROF` is: a `Bool` declaration would put `false`
        // in the receipt for `NY_EPOCH_DEBUG=0` while the run was ARMED. It is
        // deliberately NOT `LeverKind::Presence` either — `read_presence`
        // uses `var_os`, while the compatibility grammar uses `var`, so a
        // non-UTF-8 value must remain DARK. Text/Unset plus the frozen
        // receipt's UTF-8-presence bit reproduces that contract byte for byte.
        name: "NY_EPOCH_DEBUG",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
PRESENCE-armed trace for the Tier-2 epoch: one line per attempt (depth, trunk \
and head counts, the outer bound it is trying to beat) and one per outcome \
(CLOSED with the nested root bound, expansions and max depth, or the reason it \
stopped). `NY_EPOCH_DEBUG=0` and even `NY_EPOCH_DEBUG=` ARM it, because the \
historical grammar is `is_ok()` on a present value — the same spelling as \
`NY_ACASXU_PROF` and the opposite of the exact-`\"1\"` gates elsewhere in this \
module. The `#epoch-bab` family retains THREE input grammars at the frozen \
snapshot boundary — bare untrimmed integer (TIER0/UNIVERSE/RETAIN), untrimmed \
integer layered over a typed preset (DEPTH/ATTEMPTS), and this presence gate. \
Scored runs resolve the trace bit from the receipt-owning LeverSet through \
`EpochRuntimePolicy::from_levers`; compatibility callers resolve the process \
once through `EpochRuntimePolicy::from_layered_process`. `try_epoch` consumes \
only `cfg.epoch_policy.debug_trace` and never reopens the environment.

This is the instrument that made the `NY_EPOCH_DEPTH` measurement above \
readable: without it the second `tree_loop entry` and the epoch stop reason are \
the only evidence the arm engaged at all. It is print-only and cannot move a \
bound, but it is NOT free — it adds stderr inside a deadline-sensitive lane, so \
it must never appear in a timing arm.",
        provenance: Provenance::Unmeasured {
            why_ok: "print-only diagnostic, unset by default, and doubly inert because the \
                     epoch it traces is itself dark; no bound depends on it",
        },
        owner: EPOCH_BAB,
        readers: &[
            ReaderSite {
                scope: EPOCH_BAB,
                role: "freeze the UTF-8 presence gate from the receipt-owning LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:EpochRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: EPOCH_BAB,
                role: "freeze the historical process-presence grammar once for compatibility callers",
                site: "crates/ny-propagate/src/margin_row/mod.rs:EpochRuntimePolicy::from_layered_process",
            },
            ReaderSite {
                scope: EPOCH_BAB,
                role: "consume cfg.epoch_policy.debug_trace for per-attempt and per-outcome tracing without a late environment read",
                site: "crates/ny-propagate/src/margin_row/bab.rs:try_epoch",
            },
        ],
    };

    /// `NY_MARGIN_ROW_CLIP_STATEFUL` — accumulate the Clip-and-Verify
    /// intermediate tightening per domain and inherit it parent -> child.
    pub MARGIN_ROW_CLIP_STATEFUL = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP_STATEFUL",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Makes the margin-row Clip-and-Verify intermediate tightening STATEFUL: the \
tightened `[l, u]` intervals are stored per domain as a sparse \
`(layer, neuron, lo, hi)` delta, inherited by children, intersected shrink-only \
with whatever the child derives, and the per-layer top-k is RESCORED against the \
current intervals rather than the root ones. Inert unless \
`NY_MARGIN_ROW_CLIP_INTERM` / `margin_row.clip_interm` is also armed.

WHY. alpha-beta-CROWN's 2024 -> 2025 cifar100 config diff is exactly two lines: \
it ADDS `bab.clip_n_verify.clip_interm_domain` and it REMOVES \
`bab.interm_transfer: False`. The second line is the mechanism. \
`interm_transfer: False` means \"do not carry per-domain intermediate bounds \
down the tree\", which is only valid while they never change; clipping makes \
them change, so abc had to start transferring them, and it writes each round's \
result back over the parent's stored interval \
(`state/intermediate_bounds.py:589-593`). ny's port recomputed from the ROOT \
frame every call and discarded the result, which is why its measured \
dose-response was FLAT — 2.1x more tightening gave an identical bound and an \
identical 429-domain frontier. What accumulation buys is COVERAGE, not a deeper \
per-neuron cut: the retained rows are frozen root lines, so re-solving one \
neuron's dual returns the same number, but the union over a 13-30 level ancestor \
chain touches 13-30x the neurons, and rescoring the top-k off the current \
intervals is what stops every level re-picking the same 20.

WHY High, given every step is shrink-only. Two valid enclosures intersect to a \
valid enclosure and the induction bottoms out at an empty root set, so a stored \
interval always encloses its neuron's true range over the domain's region. The \
risk this carries is that a stored interval OUTLIVES the call that derived it: \
if inheritance ever ran across anything other than parent -> child, a bound \
valid on one region would be applied to another. The mechanism therefore fails \
closed at every seam — `clip::frame_ok` gates the whole path, a merge whose \
intersection would CROSS is REFUSED (kept at the parent value and counted) \
instead of being read as an emptiness certificate the way the stateless arm \
reads it, and a full accumulator simply stops accepting new neurons. It also \
publishes different intermediate bounds, which moves the bound the lane \
publishes.

THE ORDERING TRAP IS NOT RE-OPENED, and the reason is structural: accumulated \
intervals never feed back into row construction. `halfspaces_for_splits` and the \
dual objectives read `LayerGates::clip_rows` verbatim from the frozen root \
record, and the accumulated numbers enter only AFTER the dual is solved — at the \
top-k ranking and at the final max/min. See \
`margin_row/clip.rs:tighten_intermediate_acc`.

Env wins over the typed preset in BOTH directions, exact `1`/`0`; a present \
near-miss token is a recorded rejection that suppresses the preset and lands on \
this declaration's `false` default.",
        provenance: Provenance::Measured {
            commit: "00e667dc6",
            date: "2026-08-29",
            artifact: "docs/CLIP_STATEFUL_AB_2026-08-29.md",
            delta: "NEGATIVE, and interpretable rather than void. 39 runs, official 100s \
                    budget, pinned binary re-verified by sha256 after the last run. On \
                    idx_4169 the STATEFUL arm is BIT-IDENTICAL to STATELESS on every metric \
                    (exp 291, frontier worst -0.18901, open 292) while gen>=2 proves 8 real \
                    parent->child inheritance events — it ran, compounded, and changed \
                    nothing. 2 of 3 idx_4597 stateful reps likewise match stateless exactly. \
                    The number that explains it: `carried` is 18-24 neurons against a \
                    stateless sweep already touching 3.4k-4.1k, so cumulative coverage rises \
                    only 3928->4112 (+4.7%) and buys zero. Contributing cap: [clip-acc] depth \
                    is trunk_splits.len(), which on idx_1298 never exceeds 1 while trace \
                    depth reaches 6 — the lane splits mostly in the HEAD, the one axis this \
                    accumulator does not accumulate over, which is why abc's \
                    clip_interm_domain does not transfer. Separately REAL: arming clip at all \
                    (off->stateless) moves the bound +0.0018 on both responsive rows. Zero \
                    conversions; all 10 sentinel runs unsat, moat holds. Contradict this by \
                    showing carried growing by orders of magnitude, not by re-running the \
                    same dial",
        },
        owner: CLIP,
        readers: &[ReaderSite {
            scope: CLIP,
            role: "accumulate and inherit the Clip-and-Verify intermediate tightening, over the typed `margin_row.clip_stateful` preset",
            site: "crates/ny-propagate/src/margin_row/engine.rs:clip_stateful_enabled",
        }],
    };

    /// `NY_MARGIN_ROW_BRANCH_REDUCE_MAX` — the BaBSR/FSB reduceop.
    pub MARGIN_ROW_BRANCH_REDUCE_MAX = LeverDecl {
        name: "NY_MARGIN_ROW_BRANCH_REDUCE_MAX",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Selects how a branching candidate's TWO child bounds reduce to one ranking \
score. `false` (default, ny's historical behaviour) = `min`, De Palma eq. (23), \
the FSB \"balanced\" score: rank a candidate by its WORSE child, preferring \
splits that lift both sides. `true` = `max`, De Palma eq. (22), the original \
BaBSR s_SR score: rank by the BETTER child, preferring splits that make one side \
decisively easier even if the other stagnates.

MEASURED ANTI-PARITY, which is the whole reason this exists. `min` is \
alpha-beta-CROWN's GLOBAL DEFAULT, but its TUNED \
`exp_configs/vnncomp25/cifar100.yaml` AND `tinyimagenet.yaml` both set \
`branching: {reduceop: max}`, deliberately overriding that default on EXACTLY \
the two benchmarks ny fails. ny hardcoded `min` and the in-tree comment claimed \
\"np.lexsort parity\" — parity with abc's default, not with its tuned config for \
these networks.

WHY Low AND NOT High. This is RANKER-ONLY: it selects WHICH candidate index to \
split on at `bab.rs`'s pick loop. Every child bound is still produced by the \
same certified pass, `closed()` is untouched, and the monotone floor \
`cb = raw.max(b)` still applies, so a bad choice costs TREE SHAPE and can never \
manufacture a proof. It is not `None` because a different tree reaches different \
domains within the budget, so it can lose a banked row — which is a points risk, \
not a soundness one.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default in both layers (declaration default false, no preset \
                     sets `margin_row.branch_reduce_max`), so the shipped lane is \
                     byte-identical; the A/B against the two confirmed closure-loss rows \
                     plus banked-unsat moat rows is pre-registered and unrun",
        },
        owner: BRANCHING,
        readers: &[ReaderSite {
            scope: BRANCHING,
            role: "select max-vs-min reduction over a candidate's two children, over the typed `margin_row.branch_reduce_max` preset",
            site: "crates/ny-propagate/src/margin_row/mod.rs:branch_reduce_max",
        }],
    };

    /// `NY_MARGIN_ROW_PARENT_LIFT` — recover discarded candidate evidence.
    pub MARGIN_ROW_PARENT_LIFT = LeverDecl {
        name: "NY_MARGIN_ROW_PARENT_LIFT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Folds the branching scores the expansion ALREADY paid for into the child bound \
floor. `false` (default) discards them, ny's historical behaviour.

THE ARGUMENT. `score_candidates` runs a certified backward pass over every \
shortlisted candidate, yielding `ch[i] = (l_i, r_i)`: sound lower bounds on the \
two children candidate `i` would create. The expansion splits on ONE candidate \
and throws the other ~31 pairs away. Each discarded pair still certifies \
something about the PARENT: candidate `i` partitions `D` into `D_i^+ ∪ D_i^-` \
(a ReLU sign split covers both signs; a head split covers `y_i >= 0` and \
`y_i <= 0`), so every `x ∈ D` lies in some child and \
`margin(x) >= min(l_i, r_i)`. That holds per candidate independently, so \
`lift = max_i min(l_i, r_i)` lower-bounds the parent, for free.

WHY High AND NOT Low. Unlike `NY_MARGIN_ROW_BRANCH_REDUCE_MAX`, which is \
ranker-only, this MOVES A PUBLISHED BOUND — it raises the floor a child \
inherits, and can therefore close a domain that would otherwise stay open. \
`High` is the honest bucket for that, and the reason the arming A/B is gated on \
banked-unsat moat rows rather than frontier movement alone.

WHY IT IS NONETHELESS SOUND — AND WHAT IT DOES TRUST. `l_i` and `r_i` are the same \
`concretize_lower` outputs the expansion already relies on to CLOSE a child via \
`closed(mode, cb)` on `cb = raw.max(b)`; were they not sound lower bounds, ny \
would already be unsound at that existing test. `closed()` is untouched and \
closure stays strict, the lift is applied through the SAME monotone floor \
(`cb = raw.max(b).max(lift)`) rather than by weakening any predicate, and an \
empty/infeasible child can only make `min(l_i, r_i)` smaller than the truth, so \
the reduction degrades conservatively instead of over-claiming.

IT IS NOT TRUST-FREE, AND THIS IS THE PART TO GUARD. Closing a child needs only \
that child's own bound; concluding anything about the PARENT additionally needs \
the candidate's two children to COVER it. That premise is not used anywhere else \
in the lane, so nothing else tests it. It holds for both candidate kinds today \
(trunk = ReLU sign dichotomy, head = `y_i >= 0` vs `y_i <= 0`), and \
`bab.rs::CANDIDATES_ARE_A_COVER` records it. A future non-dichotomy candidate \
would make this a false-`unsat` generator with every existing test still green, \
which is why the bucket is High.",
        provenance: Provenance::Measured {
            commit: "81d30f6b4cdf558cf8642a016fca2e6f593cd159",
            date: "2026-08-30",
            artifact: "commit 81d30f6b4, official 100 s, cifar100_2024, 171 rows, \
                       engagement 124/124 pool rows; armed by \
                       `configs/vnncomp25/cifar100_2024.yaml` (`margin_row.parent_lift: true`)",
            delta: "converts idx_8739_sidx_1540 (timeout -> unsat), a row the armed pair \
                    alone does NOT convert; moat 47/47 unsat with ZERO breaks; wall cost \
                    ZERO (mean ia_secs 18.08 armed vs 18.15 unarmed); FEWER expansions on \
                    3 of 5 dose-response rows (386->346, 146->125, 324->304). The payoff \
                    is DOMAIN ELIMINATION, not tightening: per expansion it gains \
                    +0.05-0.09 on the EXPANDED domain but only ~+0.0007 reaches the \
                    frontier, which is the min over all OPEN domains. \
                    STALENESS NOTE (2026-09-01): this block previously read \
                    `Unmeasured { why_ok: \"... no preset sets `margin_row.parent_lift`, \
                    so the shipped lane is byte-identical ... pre-registered and unrun\" }`, \
                    which 81d30f6b4 falsified the moment it armed the cifar100 yaml. A \
                    MoatRisk::High lever read as dark for two days while live on a scored \
                    benchmark. NOT armed on tinyimagenet_2024 as of this edit.",
        },
        owner: BRANCHING,
        readers: &[ReaderSite {
            scope: BRANCHING,
            role: "fold `max_i min(l_i, r_i)` over the scored candidates into the child bound floor, over the typed `margin_row.parent_lift` preset",
            site: "crates/ny-propagate/src/margin_row/mod.rs:parent_lift_enabled",
        }],
    };

    /// `NY_MARGIN_ROW_BI_SECS` — wall-clock budget for that phase.
    pub MARGIN_ROW_BI_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_BI_SECS",
        // `trim().parse::<f64>()` filtered on `is_finite() && >= 0.0`, with no
        // upper filter — a CLOSED lower end (zero is a real setting: budget
        // exhausted before the first chunk) and `f64::MAX` above.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(20.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds the backward-intermediate phase may spend, default 20. Whitespace is \
trimmed before `parse::<f64>()`; non-finite, negative and malformed values all \
leave 20. ZERO IS ADMISSIBLE and means what it says — the budget is exhausted \
before the first chunk, so the phase does nothing — which is why the lower \
bound is closed rather than a `> 0` filter that would silently restore 20.

The value is a REQUEST, not an authority: it is further capped to 40% of the \
remaining deadline at construction (the same starvation guard `alpha_opt` \
uses), and the clock is checked before every chunk, so the tree search cannot \
be starved however large a number is supplied.

High because more budget means more chunks, more chunks means more trunk \
neurons get a TIGHTER published intermediate box, and the tightened box is what \
`gates_from_box` re-certifies the gates on. Stopping early is always sound, so \
the risk is not truncation — it is that this knob decides how much of the \
armed phase's bound-moving behaviour actually happens, and it takes wall clock \
from branch-and-bound to do it. Inert unless #backward-interm is armed.",
        provenance: Provenance::Measured {
            commit: "b4b77868d",
            date: "2026-08-22",
            artifact: "configs/vnncomp25/cifar100_2024.yaml measured rationale; \
                       docs/BI_DOSE_RESPONSE_5185_2026-08-24.md for the separate \
                       TinyImageNet non-promotion",
            delta: "CIFAR100 resnet_large 20 -> 40 converted idx_9883 unknown 3/3 -> \
                    UNSAT 3/3, shrank retained trees, and preserved the 7/7 moat; 40 \
                    ships only there. TinyImageNet's later higher-dose arm was \
                    row-dependent and validated 0/3 after one solve, so it remains 20",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_BI_CHUNK` — columns per prefix pass (memory grain).
    pub MARGIN_ROW_BI_CHUNK = LeverDecl {
        name: "NY_MARGIN_ROW_BI_CHUNK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(256),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Columns per backward prefix pass, default 256 — the memory grain of the \
backward-intermediate phase, whose working set is `O(max_tensor * chunk)`. \
Whitespace is trimmed before `parse::<usize>()`; absent, malformed, negative \
and overflowing input all leave 256, and the reader then CLAMPS the resolved \
value to `1..=4096`, so `0` and 99999 are pulled to the nearest bound rather \
than rejected.

Chunking is a BATCHING decision, not a mathematical one: each chunk runs one \
Lower and one Upper prefix pass over its own disjoint columns and commits a \
shrink-only intersect, so the union of what a large chunk and many small chunks \
tighten is the same set of neurons. What it really trades is memory against \
per-chunk deadline granularity — a larger chunk checks the clock less often and \
so is likelier to be cut off mid-layer, leaving different neurons tightened. \
High for that reason: a different value produces a different set of published \
intermediate bounds on the authoritative lane. Inert unless #backward-interm \
is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "the measured parent phase ships at the inherited default 256, but no \
                     retained A/B qualifies an alternate grain; execution evidence must \
                     prove the requested chunk actually bound completed passes",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_BI_CHUNK_SECS` — target wall-seconds for ONE prefix pass.
    pub MARGIN_ROW_BI_CHUNK_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_BI_CHUNK_SECS",
        // Same shape as `NY_MARGIN_ROW_BI_SECS`: trimmed f64, finite, CLOSED at
        // zero because zero is a real setting — it is the shipped default, and
        // it IS the historical fixed-grain chunking exactly.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Wall-seconds ONE backward prefix pass should target; default 0 = sizer OFF, \
the historical fixed 256-column grain plus its `last_chunk_secs` predictor, \
byte-identical to every banked run. Armed (non-zero), the pass width is \
derived from this and the layer's own measured seconds/column, with \
`NY_MARGIN_ROW_BI_CHUNK` as the memory ceiling. Whitespace is trimmed before \
`parse::<f64>()`; non-finite, negative and malformed values all leave 0. The \
value is a REQUEST: it is further capped to 10% of the effective budget, so a \
small `NY_MARGIN_ROW_BI_SECS` cannot be spent one chunk at a time. Typed \
preset key `margin_row.bi_chunk_secs`; env wins over it in both directions.

WHY IT SHIPS DARK. No retained clean-binary, receipt-closed scored A/B qualifies \
a non-zero target. A 2026-08-24 exploratory run on idx_9611 suggested that \
probe+sized passes repeated affine per-pass overhead while producing the same \
tree, but its logs were not retained in-tree, its binary contained unrelated \
dirty changes, and the declaration previously cited unrelated commit \
`1e8449bc6` (a Cora result). That observation is design context, not promotion \
authority. The optimizer must keep this axis unmeasured until the versioned \
execution receipt proves adaptive widths changed completed live work on a \
sealed clean binary.

WHY IT EXISTS AT ALL. `out_of_time()` is polled BETWEEN passes and never \
inside one, so a pass that has started always completes and the phase \
overshoots by exactly one pass; a measured width bounds that overshoot once \
the phase is ever charged its own time into the deep trunk layers.

DO NOT quote the `[backward-interm] done: ... secs=` field as the overshoot. \
That clock starts immediately before the forward tableau op walk, not at the \
phase, so it is op-walk wall rather than full-build wall or phase compute; \
the 195/195 tinyimagenet logs that read \"armed 20.0s, done median 42.2s, \
exhausted=true\" were read that way and produced a false 45%-of-lane story. \
MEASURED DIRECTLY 2026-08-24 on idx_5451 at the official 100s budget, from the \
per-layer lines: the phase's own compute is 2.83s of a 48.9s walk interval, one \
256-column pass is 0.64s, and `exhausted=true` means the FORWARD OP WALK spent \
the 20s clock and cut the phase off at trunk layer 2 of ~10. `work_secs` on \
the done line now reports phase compute so the two cannot be confused.

High because when armed it decides where the wall-clock cut lands, and \
therefore WHICH trunk neurons get a tightened published box on the \
authoritative lane — though every partition commits the same shrink-only \
intersections, and stopping early leaves the forward box standing, so no \
value here can loosen a certified bound or invent a verdict. Inert unless \
#backward-interm is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 preserves the historical fixed grain; the only exploratory \
                     non-zero observation used an unretained dirty-binary artifact and \
                     cannot satisfy scored provenance, so no alternate value is qualified",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_BI_TOPK` — unstable neurons re-derived per layer.
    pub MARGIN_ROW_BI_TOPK = LeverDecl {
        name: "NY_MARGIN_ROW_BI_TOPK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1024),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
How many unstable neurons the backward-intermediate phase re-derives per layer, \
widest first, default 1024. Whitespace is trimmed before `parse::<usize>()`; \
absent, malformed, negative and overflowing input all leave 1024, and the \
reader then applies `.max(1)`, so an explicit `0` becomes 1 rather than \
disabling the phase — disarming is `NY_MARGIN_ROW_BACKWARD_INTERM=0`, not a \
zero here.

This is the SELECTION knob: it decides WHICH pre-activations get a tightened, \
published box, and because the phase runs layer by layer and later layers \
consume earlier layers' tightened gates, the choice compounds down the trunk. \
High for exactly that reason — it changes the published intermediate bounds on \
the authoritative margin-row route (and spends the phase's budget doing it), \
even though every individual selection is sound and skipping any neuron leaves \
its forward box standing. Inert unless #backward-interm is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "the measured parent phase ships at the inherited default 1024, but no \
                     retained A/B qualifies an alternate selection width; execution \
                     evidence must prove top-k omitted candidates on completed live work",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_INTERM_ALPHA` — per-START-NODE prefix alpha candidates.
    pub MARGIN_ROW_INTERM_ALPHA = LeverDecl {
        name: "NY_MARGIN_ROW_INTERM_ALPHA",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#interm-alpha. How many PER-START-NODE lower-slope candidates the \
backward-intermediate phase tries on each start node beyond the one shared \
heuristic alpha, default 0 (byte-identical to the pre-#interm-alpha phase). \
Clamped to the policy count; 1 = the chord-parallel line `alpha = s`, 2 adds \
the flipped area-heuristic vertex, and 3-5 add the interpolations `0.5`, \
`(s+1)/2` and `s/2` that the two vertices and the chord leave unexplored. \
Every candidate is a slope in [0,1], so each is a VALID relaxation and the \
shrink-only intersection makes a higher dose monotonically tighter; the cost is \
WALL (`ia_secs`, polled against the BI budget), never soundness.

THE AXIS. alpha-beta-CROWN holds a SEPARATE optimizable alpha set per \
(ReLU layer, downstream START NODE) — `alpha[relu][start_node]`, shape \
`[2, spec, batch, *relu_shape]` (verified against \
`references/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/relu.py:155` and \
`backward_bound.py:874-944`) — so every intermediate layer's bound is computed \
under slopes chosen FOR THAT LAYER. ny reuses the ONE shared (layer, neuron) \
heuristic alpha for every start node and both lanes, and this is the only \
route in the margin-row lane that lets a start node choose its own. The \
one-shot phase is separately MEASURED SATURATED on budget (bit-identical roots \
at 2x secs and 4x topk, `BIG3_CORE_CAMPAIGN_2026-08-19.md`), so more of the \
same dose is known to buy nothing — a different relaxation is the only \
remaining move on this phase.

High by this registry's own Low/High line — High is `can move a PUBLISHED \
bound`, not `can change the answer` — because the tightened boxes reach the \
published root, exactly like its BI siblings. The asymmetry is still worth \
stating. Every candidate box is produced \
by the unchanged certified engine under a valid relaxation (`alpha*z <= relu(z)` \
holds pointwise for ANY `alpha in [0, 1]`; only unstable neurons are touched; \
the upper chord `(s, c)` is a function of `(l, u)` alone and is not moved; the \
error-carry constant `ms = max(alpha, s)` is re-derived from the candidate) and \
is INTERSECTED shrink-only into the box the phase would otherwise have \
published. The published box is therefore a SUBSET of the pre-#interm-alpha \
one, `gates_from_box` + `repair_upper_lines` re-certify on it as always, and a \
useless candidate contributes exactly nothing. Unlike its parent \
`NY_MARGIN_ROW_BACKWARD_INTERM` this lever cannot introduce the clip_rows \
ordering trap either: that calibration is taken against the FORWARD-only \
bounds captured before the whole phase runs, which this lever does not move. \
The live risk is WALL — each candidate is one more Lower+Upper prefix pass \
pair against the phase's budget — which is why it is charged separately as \
`ia_secs` and polled through `out_of_time` before every candidate.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 is byte-identical to the shipped phase; the axis has \
                     never been built on this lane, so no retained A/B qualifies a dose",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_GPU_BACKWARD` — the certified GPU backward for the lane.
    pub MARGIN_ROW_GPU_BACKWARD = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_BACKWARD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the margin-row lane's certified GPU backward (`#margin-row-gpu-eft`): an \
admitted certified-outward pass would be computed by the EFT / double-single \
device kernels instead of the CPU backward walk, and concretized by the lane's \
own unchanged f64 concretize. Exact `\"1\"` arms; every other value — including \
`\"0\"`, `\"true\"` and non-Unicode bytes — leaves it dark, and the reason the \
declaration is a `Bool` rather than a presence gate is that `=0` must \
mean OFF here, as `Refusal::Disabled` states. (Renamed from \
`NY_MARGIN_ROW_GPU_EFT` in the implementation session, per the session brief; \
the OLD seam's `NY_MARGIN_ROW_GPU` is a different lever and stays untouched.)

READ AS A LATCHED RAW STRING, deliberately. `env_raw` latches the STRING once \
in a `OnceLock` and `armed_from_raw` derives the DECISION per call — the same \
discipline as `gpu_seam::env_raw` — because `armed_from_raw` is a pure, \
unit-tested predicate that IS the spec of the arming rule and must stay in the \
production path rather than being restated by the chokepoint.

STAGED, NOT DEAD. Both the on-device M1 self-check AND the L-lane device \
transaction are delivered (implementation session 2: head gemm + gate \
transform + bias reduce + conv gather + residual merges as ONE drain-safe \
submit, host directed-rounding egress, floor + probe guards), but verdict \
authority is M3-GATED: after a passing qualification every armed pass refuses \
`Refusal::ShadowOnly` until the trusted code constant `M3_AUTHORITY_GRANTED` \
is flipped by the M3 A/B session. `run_transaction` — the only `PassOut` \
producer — requires a `VerdictAuthority` token, so a prematurely-wired call \
site cannot reach verdict math; the transaction runs today only under \
`NY_MARGIN_ROW_GPU_SHADOW=1` (compute, compare, DISCARD).

High is classified for what the gate ADMITS, exactly as `NY_EFT_ERR`'s staged \
CPU arm is: this lane is designed to be AUTHORITATIVE rather than a shadow \
(removing the CPU backward walk is the cost it exists to remove), so once the \
transaction lands, arming it decides whether a published margin-row bound came \
from the device. Its guards — the certified-error floor, the realization probe, \
the NaN/Inf firewall, the lane-local channel-death latch — all fail closed to \
the exact CPU pass, and none of them lowers what a wrong value here could cost.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and structurally unreachable past the authority gate \
                     in this delivery, so the OFF arm is the only arm that computes \
                     anything; it cannot leave Bucket::Debug without a measured A/B on \
                     the delivered transaction",
        },
        owner: GPU_EFT,
        readers: &[ReaderSite {
            scope: GPU_EFT,
            role: "latch the raw gate string for the pure `armed_from_raw` arming predicate",
            site: "crates/ny-propagate/src/margin_row/gpu_backward/mod.rs:env_raw",
        }],
    };

    /// `NY_MARGIN_ROW_GPU_SHADOW` — M2 shadow mode for the GPU backward.
    pub MARGIN_ROW_GPU_SHADOW = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_SHADOW",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the M2 SHADOW protocol of the certified GPU backward \
(`#margin-row-gpu-eft`, design section 7 M2): after every CPU margin-row \
backward pass, the L-lane device transaction computes the SAME pass on the \
lane's dedicated device, asserts (a) enclosure overlap against the CPU pass \
(`|a_gpu - a_cpu| <= e_gpu + e_cpu` elementwise, same for bias), (b) the \
realization probe on the GPU pass, (c) prints the per-row certified radius \
telemetry — and then DISCARDS the device result. No verdict is ever computed \
from it; the verdicts of a shadow run are bit-identical to an unshadowed run \
by construction (the CPU pass is returned either way). Exact `\"1\"` arms.

Any shadow TRIP (enclosure, probe, or error-floor) latches the lane's channel \
dead for the process and is printed unconditionally — the pre-registered M2 \
kill criterion; the run log must also be grepped for the sweep post-submit \
kill line per design section 7.

High like the rest of the group, though for a different mechanism: it cannot \
change a published BOUND (results are discarded and the CPU pass is returned \
unchanged, so shadowed bounds are bit-identical by construction), but it \
spends device wall clock INSIDE the verdict window of the authoritative lane \
— enough added latency turns a provable row into a timeout — and its trips \
latch the lane channel. It must never be armed on a scored run; it exists for \
the dedicated M2 measurement slot only.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; the shadow arm returns the unchanged CPU pass by \
                     construction, so only wall clock is at stake until the M2 \
                     measurement session reads its telemetry",
        },
        owner: GPU_EFT,
        readers: &[ReaderSite {
            scope: GPU_EFT,
            role: "latch the raw shadow gate string for the pure arming predicate",
            site: "crates/ny-propagate/src/margin_row/gpu_backward/mod.rs:shadow_env_raw",
        }],
    };

    /// `NY_MARGIN_ROW_GPU_SHADOW_SITES` — WHICH sites the M2 shadow observes.
    pub MARGIN_ROW_GPU_SHADOW_SITES = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_SHADOW_SITES",
        kind: LeverKind::Text,
        default: DefaultSpec::Text("root"),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Selects WHICH call sites the M2 shadow of the certified GPU backward observes \
(`#margin-row-gpu-eft`). Inert unless `NY_MARGIN_ROW_GPU_SHADOW=1` is armed: \
this lever never arms the shadow, it only narrows it.

THE MAPPING, exactly (`gpu_backward::shadow_sites_from` is the unit-tested \
spec and the only admissible reading):

* UNSET (the default) => `root` ALONE — the root direct pass, once per \
  instance. That is the impl-v7 shape the whole M2 protocol was designed and \
  banked on. impl-v8 wired three more sites (B4 score, B5 eval, B6 y-refresh) \
  and the shadow lever armed ALL FOUR at once: ~2 device passes per BaB \
  expansion plus 2 per y-pack LRU miss, against ONE pass per instance before. \
  Unset therefore RESTORES the measured shape rather than inheriting v8's.
* `root` | `y_refresh` | `eval` | `score` => that site alone. This is what \
  makes the design's own `one lever per cycle` rule executable: measure \
  SITE_EVAL alone, then SITE_SCORE alone, each against the same plain arm.
* a comma list (`eval,score`) => the UNION of the named sites.
* `all` => all four (impl-v8's implicit behaviour, kept reachable so the v8 \
  measurements stay reproducible).
* `none` => NO site shadows: the control arm that prices the shadow LEVER \
  itself (env read, banner, nothing else) against a plain run.
* anything else => REJECTED: no site shadows and the lane prints \
  `[gpu-backward] shadow site selector REJECTED ...` once. Deliberately NOT a \
  fallback to the default — a typo that silently measured `root` while the \
  operator believed they were measuring `eval` is the `reads as armed, \
  measured something else` failure this project keeps paying for. Zero shadow \
  lines plus one loud banner cannot be misread.

Tokens are split on commas, trimmed, matched EXACTLY and lowercase (`EVAL` is \
a typo, not an alias — the same chokepoint discipline that refuses `true` for \
a Bool); empty tokens are skipped, so a trailing comma is not an error. The \
spelling is the same one the `site=` telemetry prints, so a log line always \
names a value this lever accepts.

Low, not High: it cannot arm anything, cannot touch verdict authority (that \
needs `NY_MARGIN_ROW_GPU_BACKWARD=1` plus a trusted per-site code constant), \
and cannot change a published bound (shadow results are compared and \
DISCARDED, so a shadowed run's verdicts are bit-identical to an unshadowed \
one by construction). Its whole effect is on DEVICE WALL CLOCK inside the \
verdict window — and its purpose is to REDUCE that, from four sites to one.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert unless the (dark, Debug-bucket) shadow lever is armed, and its \
                     default restores the impl-v7 root-only shadow shape that the banked M2 \
                     measurements were taken on — so the default arm is the measured one",
        },
        owner: GPU_EFT,
        readers: &[ReaderSite {
            scope: GPU_EFT,
            role: "latch the raw site-selector string for the pure `shadow_sites_from` spec",
            site: "crates/ny-propagate/src/margin_row/gpu_backward/mod.rs:shadow_sites_env_raw",
        }],
    };

    /// `NY_MARGIN_ROW_GPU_PROF` — per-step profiling for the GPU backward's
    /// M2 shadow passes.
    pub MARGIN_ROW_GPU_PROF = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_PROF",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Arms PER-STEP device profiling for the certified GPU backward's M2 SHADOW \
passes only (`#margin-row-gpu-eft`, the B2 perf observable): a profiled \
program runs ONE STEP PER SUBMIT, each submit unconditionally drained to idle \
before the next (the R2 drain-safe shape holds for every submit), and the \
per-step wall times are printed as `[gpu-backward] prof ...` lines. Exact \
`\"1\"` arms; inert unless `NY_MARGIN_ROW_GPU_SHADOW=1` is also armed, because \
the profile flag is plumbed ONLY through the shadow entry — the (M3-gated) \
verdict path passes `profile=false` structurally, so a profiled program can \
never compute anything verdict-bearing. Multiplying submits changes device \
wall clock, never values: the same dispatches run over the same arena in the \
same order, with the inter-step barriers the encoder split preserves.

Low, not High: it cannot change a published bound (shadow results are \
discarded by construction) and unlike the shadow gate itself it adds wall \
clock only to passes the shadow lever already spends wall clock on. It must \
still never be armed on a scored run (it multiplies the shadow's device \
latency severalfold); it exists for the B1-B3 perf measurement slots only.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default; only reachable through the discard-only M2 shadow \
                     entry, so values are never consumed — wall clock in a measurement \
                     slot is the only stake",
        },
        owner: GPU_EFT,
        readers: &[ReaderSite {
            scope: GPU_EFT,
            role: "latch the raw profiling gate string for the pure arming predicate",
            site: "crates/ny-propagate/src/margin_row/gpu_backward/mod.rs:prof_env_raw",
        }],
    };

    /// `NY_MARGIN_ROW_REBAKE` — the joint forward re-bake (#rebake).
    pub MARGIN_ROW_REBAKE = LeverDecl {
        name: "NY_MARGIN_ROW_REBAKE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#rebake`: the margin-row root's joint forward re-bake — per round, one \
alpha ascent (`#alpha-opt`'s optimizer) on the current build, then a REBUILD of \
the forward tableau with the optimized alpha baked into `apply_gates` (the \
alpha-CROWN root mechanism this lane was missing: tighter boxes -> narrower \
triangles -> better alpha -> tighter boxes), then a re-run of the \
`#backward-interm` phase (halved budget) over the rebuilt prefix gates, \
accepted at the ROUND level only if the certified `root_eval` bound strictly \
improves. Exact `1` arms; absence and every other value leave it dark and the \
lane byte-identical (the entry returns before touching anything). Armed, it \
announces itself (`[rebake] armed`) even when every round is refused.

WHY High. Every candidate build is internally certified by the unchanged build \
path for ANY clamped seed (acceptance merely chooses among valid builds), but \
the accepted build changes which certified bound the authoritative margin-row \
lane PUBLISHES, and the loop spends root-window wall clock (each round is one \
ascent + one forward build + one certified evaluation). The one genuinely new \
soundness surface — `apply_gates`' outward widening under a FRACTIONAL lower \
product — carries its own re-derived magnitude-scaled branch with the \
derivation at the site (`root.rs::apply_gates`) and a discriminating grid \
test; the historical binary-alpha line is preserved bit-identically.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and byte-identical when unset; built in the \
                     2026-08-20 rebake session, runbook \
                     docs/REBAKE_JOINT_ROOT_2026-08-20.md, no scored-row A/B retained yet",
        },
        owner: REBAKE,
        readers: &[
            ReaderSite {
                scope: REBAKE,
                role: "freeze the complete root-alpha/rebake policy from the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: REBAKE,
                role: "resolve the legacy layered process only for compatibility callers",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_REBAKE_ROUNDS` — re-bake rounds per instance.
    pub MARGIN_ROW_REBAKE_ROUNDS = LeverDecl {
        name: "NY_MARGIN_ROW_REBAKE_ROUNDS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(2),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Re-bake rounds, default 2. Whitespace is trimmed before `parse::<usize>()`; \
absent, malformed and overflowing input all leave 2, and the reader CLAMPS the \
resolved value to `1..=8`, so an explicit `0` becomes 1 rather than disabling \
the loop — disarming is `NY_MARGIN_ROW_REBAKE=0`, not a zero here. The loop \
stops early on its own at an alpha fixed point or a rejected round (the step \
is deterministic, so a rejected round would only repeat), so this is a CAP, \
not a demand.

High because each round is one alpha ascent + one forward rebuild + one \
certified `root_eval` against the instance deadline, and the accepted round \
changes the published root bound. Inert unless NY_MARGIN_ROW_REBAKE is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state",
        },
        owner: REBAKE,
        readers: &[
            ReaderSite {
                scope: REBAKE,
                role: "freeze the re-bake round cap with the shared alpha controls",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: REBAKE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_REBAKE_SECS` — wall-clock slice for the loop.
    pub MARGIN_ROW_REBAKE_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_REBAKE_SECS",
        // Same shape as `NY_MARGIN_ROW_BI_SECS`: trimmed f64, finite, CLOSED
        // at zero (an explicit 0 is a real setting — every round is refused as
        // unaffordable — and must stay distinguishable from absence).
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(30.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds the joint forward re-bake may spend, default 30. Whitespace is trimmed \
before `parse::<f64>()`; non-finite, negative and malformed values all leave \
30. The value is a REQUEST, not an authority: it is further capped to 40% of \
the remaining deadline (the same starvation guard `alpha_opt` and \
`#backward-interm` use), a round the remainder cannot afford is refused \
LOUDLY before it starts (build + eval + minimum ascent slice priced from \
measured costs), and every abort path leaves the previously accepted build \
standing — expiry fails closed to an earlier certified root, never to a wrong \
one.

High because the budget decides how many rounds actually run, each accepted \
round changes the published root bound, and the wall clock comes out of the \
same window the tree search needs. Inert unless NY_MARGIN_ROW_REBAKE is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state",
        },
        owner: REBAKE,
        readers: &[
            ReaderSite {
                scope: REBAKE,
                role: "freeze the re-bake wall request with the shared alpha controls",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: REBAKE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_F32` — f32 COMPUTE for root-tableau convolutions.
    pub MARGIN_ROW_ROOT_F32 = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_F32",
        // Preserve the historical public grammar exactly. The search domain is
        // deliberately narrower (canonical 0/1), but old research recipes using
        // trimmed `true` or `on` must keep selecting the same execution lane.
        kind: LeverKind::TruthyTrimmed,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Runs both bandwidth-bound root-tableau convolution lanes (M and D) in f32, \
widens their accumulators exactly into the f64 tableau, and carries a certified \
per-neuron `g_err` envelope through every subsequent operation. The envelope \
is subtracted from lower endpoints and added to upper endpoints at concretize, \
so f32 compute can only LOOSEN a published box: it may cost a proof and cannot \
manufacture one. It is effective only in Outward mode.

The environment grammar is intentionally historical: after trimming, `1`, \
`true`, and `on` arm; every other Unicode token is an admissible kill switch. \
A non-Unicode value behaves as absent and falls through to the typed \
`margin_row.root_f32` key. Scored runs resolve this once from the receipt-owning \
LeverSet and carry the immutable answer through initial roots, rebakes, and \
nested epoch roots; no build reopens the process environment.

High because this changes the outward tableau that feeds authoritative Unsat \
bounds. Soundness is pinned by the 44,262,336-check cross-layer rounding oracle, \
23,424/23,424 f32-contains-f64 box checks, and the full margin-row enclosure \
suite. The current search domain emits only canonical `0`/`1` treatments even \
though the compatibility parser remains broader.",
        provenance: Provenance::Measured {
            commit: "1d27d4eb9b457a0061d7dab5391bff08cd591b71",
            date: "2026-08-03",
            artifact: "commit 1d27d4eb9 and the measured rationale in \
                       configs/vnncomp25/tinyimagenet_2024.yaml; CIFAR100 pair \
                       revalidation in commit 7534933d0",
            delta: "TinyImageNet prop_idx_5478 at the official 100 s budget: \
                    concurrent + f64 root timed out, while concurrent + f32 root \
                    proved UNSAT (58.1 s lane, 90.4 s wall). The original root \
                    implementation measurement cut prop_idx_7473's root from \
                    74.0 s to 58.7 s and closed UNSAT in 71.7 s",
        },
        owner: ROOT_F32,
        readers: &[
            ReaderSite {
                scope: ROOT_F32,
                role: "freeze f32 root-convolution compute policy from env over typed preset \
                       and carry it through every root build",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_F32,
                role: "resolve the same env-over-preset grammar once for compatibility callers \
                       that do not enter through a scored frozen LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_TABLEAU_F32` — f32 STORAGE for the root tableau walk
    /// (`#tableau-f32`).
    pub MARGIN_ROW_TABLEAU_F32 = LeverDecl {
        name: "NY_MARGIN_ROW_TABLEAU_F32",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Stores the M and D lanes of the root forward tableau as `Array2<f32>` \
end-to-end across the walk (`#tableau-f32`, doc block in `root.rs`), deleting \
the per-layer f64->f32 cast pass and halving the streamed bytes of the \
bandwidth-bound conv walk (39.93 s of the 41.35 s tinyimagenet root build, \
measured 2026-08-24 under NY_MARGIN_ROW_ROOT_TIMING). Requires the f32 \
COMPUTE lane (`margin_row.root_f32`) and Outward mode; requested without them \
it REFUSES loudly (`[tableau-f32] INERT`) and runs f64 storage, so a mis-armed \
arm is readable in its own log. Every store is class (a) EXACT (conv outputs \
publish the identical f32 accumulator), class (b) DIRECTED-UP (radius-role \
stores can only grow), or class (c) NEAREST with the perturbation folded into \
the same-site D widening AND charged into the per-neuron `g_err` concretize \
slack, so the published box can only LOOSEN versus the f64 path \
(per-site oracle `tableau_f32_store_per_site_containment_oracle`, 31,845 \
exact checks; end-to-end `tableau_f32_store_dj_encloses_true_margin`).

NOTE the exact-token contract: promotion to this registry narrowed the \
historical ad-hoc parser (`1`/`true`/`on`) to the repo arming rule — exact \
`\"1\"` arms, exact `\"0\"` disarms, anything else present is a kill switch \
that suppresses the preset and resolves dark.

High because the margin-row lane publishes `Unsat` from bounds this lever \
changes: a looser box costs proofs (never manufactures one), and the deleted \
cast passes change root-build wall clock against the same deadline the tree \
search spends.",
        provenance: Provenance::Measured {
            commit: "d7075c54f4e8d52c3a7f4163fe9d36fac0f6ff92",
            date: "2026-08-25",
            artifact: "commit d7075c54f and the retained category rationale in \
                       configs/vnncomp25/tinyimagenet_2024.yaml",
            delta: "TinyImageNet forward conv 41.4 -> 24.6 s and root build \
                    46.9 -> 29.1 s; idx_5451 moved from 1/3 to 3/3 UNSAT and \
                    idx_9611 from 1/3 to 3/3 UNSAT under official 100 s walls. \
                    Four current marginal proofs and three sat sentinels held \
                    3/3; the broader stale-bank moat is explicitly not claimed",
        },
        owner: TABLEAU_F32,
        readers: &[
            ReaderSite {
                scope: TABLEAU_F32,
                role: "freeze f32 M/D storage policy with root compute policy and carry it \
                       through every root build",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: TABLEAU_F32,
                role: "resolve the governed tableau preset gate once for compatibility callers \
                       that do not enter through a scored frozen LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_DOMAIN_STACK` — one wide certified scoring pass across the
    /// whole frontier batch instead of one pass per domain (`#domain-stack`).
    pub MARGIN_ROW_DOMAIN_STACK = LeverDecl {
        name: "NY_MARGIN_ROW_DOMAIN_STACK",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Scores the candidate children of EVERY domain in a frontier batch in ONE \
certified backward pass (`score_candidates_domain_stacked`, bab.rs) instead of \
one independent `score_candidates` call per domain, by concatenating each \
domain's seed columns into a single wide seed and slicing the result back per \
domain.

WHY IT EXISTS. The certified pass is LINEAR in columns above R~64 (2.4 ms/col \
medium, 10.4 ms/col large) on a 45-60 ms fixed cost, but the lane runs it at \
R = nf = 3-11 columns, i.e. 29-34 ms/col — roughly 13x the marginal cost — and \
`eval_node` + `score_candidates` are 83% of the lane at ~1.15 s/expansion \
(docs/MARGIN_ROW_COLUMN_SCALING_2026-08-23.md). Stacking moves the lane onto \
the linear part of its own cost curve. It is also the affordability \
PRECONDITION for a per-domain joint alpha+beta re-optimisation, which is \
alpha-beta-CROWN's core BaB mechanism and is priced per certified trial today.

SOUNDNESS. Column-local: each domain's reductions read only its own column \
slice, so a stacked score is bit-identical to the independent one. Pinned by \
`domain_stacked_candidate_scores_match_independent_frontier_bit_exact` (whose \
in-pass oracle compares EVERY child score's bits before any candidate is \
selected) and `beta_domain_stacked_matches_independent_passes`. The lever \
selects a schedule, not a relaxation: it cannot move a published bound.

Since #mr-stack (throughput plan B5) the armed lane also runs the batch's MAIN \
eval passes as ONE stacked pass (the B1 probe: 0.15-0.20x the CPU of K=8 \
separate passes, bit-identical); each domain's block is bit-identical to its \
own pass (`stacked_main_eval_bit_identical_to_per_domain`, \
`expand_batch_trajectory_hash_unchanged`). β / domain-alpha trials stay per \
domain; a `beta_pc` domain falls back to its own pass. Receipt: `[mr-stack]` \
at stop, `stk_*` in `[mr-census]`.

High because it changes the shape of the pass that scores every branch \
decision on the lane that publishes `Unsat` for cifar100 and tinyimagenet, and \
because a mis-sliced stack would attribute one domain's score to another — \
which is what the bit-exact differential exists to catch.

Promoted DARK (default-off, armed by no yaml) so the A/B can be run through \
the scored preset route at all: with `run_instance.sh` exporting no `NY_*`, an \
env-only mechanism cannot be measured on a scored binary. Promote first, \
measure second, arm third.",
        provenance: Provenance::Unmeasured {
            why_ok: "Default-off and armed by no shipped yaml, so no scored run \
                     changes behaviour on promotion. The bit-exactness of the \
                     stacked scores against independent passes is already pinned \
                     by two differential tests; what is unmeasured is the \
                     WALL-CLOCK effect on real rows at official budgets, which \
                     is exactly what promoting the key makes runnable.",
        },
        owner: DOMAIN_STACK,
        readers: &[ReaderSite {
            scope: DOMAIN_STACK,
            role: "freeze the cross-domain scoring schedule once at lane entry and carry it \
                   into every expansion through BabConfig",
            site: "crates/ny-propagate/src/margin_row/mod.rs:lane_impl_with_policy",
        }],
    };

    /// `NY_MARGIN_ROW_ERR_ENVELOPE` — the certified coefficient-error lane's
    /// representation (`#mr-errlane`, margin-row throughput plan B3).
    pub MARGIN_ROW_ERR_ENVELOPE = LeverDecl {
        name: "NY_MARGIN_ROW_ERR_ENVELOPE",
        kind: LeverKind::Enum(&["off", "envelope", "per_channel", "audit"]),
        default: DefaultSpec::Text("off"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Selects how the margin-row backward engine carries its CERTIFIED \
coefficient-error lane `E` (the `n x R` matrix bounding the rounding of every \
backward coefficient in Outward mode). Measured by the B1 probe \
(`col-scaling-v2`, resnet_medium / resnet_large / tinyimagenet) as exactly half \
of every pass at every width: `outward_over_parity` 2.00-2.04 at 1 thread.

* `off` (default): today's per-column lane, byte-identical.
* `envelope`: on passes with `R >= 16` columns (score, y-refresh, \
backward-interm, tinyimagenet's wide evals), carry ONE envelope value per row \
that dominates `E[j, c]` for every column `c`, through the same per-op \
formulas with `|L|` replaced by its column maximum. The conv step becomes the \
unchanged `|W|` backward kernel at `R = 1`. Narrow passes keep the per-column \
lane.
* `per_channel`: one envelope value per CHANNEL on every pass; the conv step \
becomes `O(ci·co·taps)` (per-tap channel sums, max over positions), so the \
error lane is nearly free even at `R = 1`. Looseness unmeasured.
* `audit`: publish the per-column lane (every published bound is the `off` \
bound; only wall clock differs) and SHADOW \
both envelopes on the same passes, recording penalty differences, bound \
loosening and lost positive bounds in the `[mr-errlane] done` receipt. A \
measurement arm (about two extra passes per pass), never a treatment.

SOUNDNESS. Every per-op update of `E` is monotone non-decreasing in (`E`, \
`|L|`) with column-independent non-negative coefficients (ReLU `ms`, \
exception `max(|a2|,|s2|)`, beta `1`, affine `|s|`, conv `|W|`, merge `1`), so \
the envelope dominates the per-column lane by induction — bit-for-bit, since \
each fl op is monotone (`eel_dominates_percol_every_op`). The coefficient lane \
`L` and bias `b` are untouched, so only the certified penalty can grow: a \
looser `E` can cost a proof and cannot manufacture one. The per-channel conv \
sums every listed tap against the channel maximum (dominating each position's \
own sum) and certifies the nested sum with `gamma(2·k_bwd + max_taps + 16)`.

High because it changes the certified penalty of every wide pass on the lane \
that publishes `Unsat` for cifar100 and tinyimagenet, and because a \
dominance bug here would be a false-`unsat` generator with every \
tightness-only test still green — which is what the per-op dominance oracle \
and the exact-rational falsifier exist to catch.",
        provenance: Provenance::Unmeasured {
            why_ok: "Default `off` and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion. The per-column lane is the unchanged code path \
                     (every envelope branch is guarded by an envelope that `off` never \
                     builds). What is unmeasured is (a) the looseness of both envelopes on \
                     real rows (the `audit` arm's gate: max penalty increase <= 1e-6 and < \
                     1% of the smallest closing margin on the root + ~50 domains of 4 gap \
                     rows) and (b) the wall-clock effect at official budgets (plan §4 A/B).",
        },
        owner: ERR_ENVELOPE,
        readers: &[
            ReaderSite {
                scope: ERR_ENVELOPE,
                role: "freeze the error-lane representation with the root-precision policy \
                       from the receipt-owning LeverSet; every root build, rebake, epoch root \
                       and backward-interm scratch root carries it and every BackwardEngine \
                       copies it at construction",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ERR_ENVELOPE,
                role: "resolve the same env-over-preset token once for compatibility callers \
                       that do not enter through a scored frozen LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_GPU_CONV` — the certified pass's conv backward products on
    /// the process-global sound f64 GEMM engine (`#mr-gpu-eval`).
    pub MARGIN_ROW_GPU_CONV = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_CONV",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
RETIRED 2026-09-29 (ENGINE_PROGRAM R1 kill line + R2): the lane never arms this \
route in any build (`gpu_conv::armed` is false); a request prints \
`[mr-gpu-conv] refused reason=retired(...)`, the typed key `margin_row.gpu_conv` \
is refused at preset validation, and the axis is Unsafe / EnvOnly. Measured on \
the yaml route with zero NY_* (W7B, `w7b/w7b_v2.log`): 38,552 per-op convs at \
3.74 ms each; cifar100 c2050 unsat 88.5 s -> timeout, c1590 unsat 78 s -> \
unknown. What it was:

Routes the margin-row certified pass's CONV BACKWARD products — the signed \
coefficient product `W^T L`, the per-column error lane's `|W|^T comb` and the \
row envelope's `R = 1` `|W|` product — to the process-global sound f64 GEMM \
engine (`sound_f64_gemm`, cuBLAS `Dgemm` on a CUDA build) as ONE full-contraction \
GEMM per spatial tile (`margin_row/gpu_conv.rs`). Every pass of the lane goes \
through that one conv arm: the tree's main eval (and the stacked main eval \
`domain_stack` batches), candidate scoring, the y-refresh, the root eval, the \
backward-interm prefix passes and the alpha ascent.

WHY IT EXISTS. Per conv op each product costs `k_bwd = taps*co` (144-2304) \
multiply-adds per output entry while every other op of the pass is O(1) per \
entry, and the pass measures linear in trunk work x columns (2.4 ms/col \
resnet_medium, 10.4 ms/col resnet_large, 20 cores; \
docs/MARGIN_ROW_COLUMN_SCALING_2026-08-23.md). The GB10 runs `Dgemm` at \
0.41 TF/s, roughly 15x the CPU kernel's effective rate, and the GPU otherwise \
sits idle on this lane.

SOUNDNESS. f64, not f32: the device consumes the STORED f64 operands \
bit-for-bit (no cast, no split) and the engine contract is IEEE binary64 \
round-to-nearest in ANY summation order (FMA / split-K included; dispatch \
splits output axes only). Higham's `gamma_n * sum|terms|` is order-independent \
and the gather's absent taps are EXACT zeros (no rounding), so the lane's \
UNCHANGED charges — `g = gamma(k_bwd+2) + rel + gamma*rel` inside \
`comb = E + g(|L| + E)` for the signed product and \
`certify_up(., gamma(k_bwd+8))` for the non-negative one — cover the device \
result exactly as they cover the CPU kernel's own order. Beta, exceptions and \
gates stay host-side in the ReLU arm (the moved op is linear), so beta domains \
are supported by construction. Fail-closed: transposed convs, duplicate taps, \
non-finite operands, a missing or non-CUDA engine, any engine error or \
per-call budget expiry, and any wrong-length or non-finite output run the \
unchanged CPU kernel, which overwrites the whole result.

High because it changes the arithmetic of every certified pass on the lane \
that publishes `Unsat` for cifar100 and tinyimagenet (published bounds move at \
the ulp level of a reordered f64 sum), and because a gather or indexing bug \
would be a false-`unsat` generator with every dark test green — which is what \
the exact-rational containment test, the routed-pass falsifier and the \
Lagrangian (beta) oracle exist to catch. Engagement evidence: \
`[mr-gpu-conv] done ... gpu=` > 0; an armed lane with `gpu=0` prints \
`verdict=VOID`.",
        provenance: Provenance::Unmeasured {
            why_ok: "RETIRED: inert in every build (the lane never arms the route; a \
                     request only prints the refusal), so no run can change behaviour \
                     through it. The yaml-route measurement that retired it (W7B: lost \
                     cifar100 c2050 and c1590, ENGINE_PROGRAM R1 kill line) is recorded \
                     in the doc; the dark code stays exercised only through test-injected \
                     routes.",
        },
        owner: GPU_CONV,
        readers: &[ReaderSite {
            scope: GPU_CONV,
            role: "latch the RETIRED switch's request once per process, for the lane-start \
                   refusal line only; it never arms the route",
            site: "crates/ny-propagate/src/margin_row/gpu_conv.rs:requested",
        }],
    };

    /// `NY_MARGIN_ROW_FROZEN_DUAL_BAB` — the certified frozen alpha-zonotope
    /// dual BaB tier (`#frozen-dual-bab`, V1 stage B on stage A's
    /// `#frozen-dual-state` query state).
    pub MARGIN_ROW_FROZEN_DUAL_BAB = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the frozen alpha-zonotope dual-ascent BaB (VIBECHECK_PLAN.md 6.V1) as a \
TIER between the margin-row lane's certified root and its best-first tree, on \
the outer frame, for admitted classes the root leaves open. Per class it builds \
stage A's certified QUERY STATE (`margin_row/zono_state.rs`: the \
generator-space linearisation of the margin and of every unstable \
pre-activation over the lane's frozen root boxes and committed alpha, \
direction-adaptive slopes, binary64 with explicit Higham gamma charges, \
certified BatchNorm-fold model errors and outward rounding), sorts the \
unstable rows once by the box + single-halfspace score, and runs a \
level-synchronous BFS (`margin_row/zono_bab.rs`): each node is evaluated by the \
certified node bound at four clamped dual points (warm start, slack-clamped, \
the box-corner floor, the clamped far probe), pruned ONLY on a certified value \
> 0, and passes a log-bucket dual step or a strictly better certified breakpoint \
candidate to its children as their warm start. \
A class is published closed only when every leaf is certified and the level \
ledger closes; a deadline, node cap, memory refusal or exhausted order leaves \
it open and hands it to the tree unchanged. Nothing here emits `sat`.

The shipped CIFAR100 and TinyImageNet presets arm this tier with four breakpoint \
refinement steps. The generic declaration remains false. Budget: \
`margin_row.frozen_dual_bab_secs` defaults to 0, the automatic mode: at most \
4 seconds and 5% of remaining lane time, 8192 nodes per objective and \
256 MiB of state. Dense row estimates at least 64 MiB select one easiest open \
objective; smaller states select at most two. A positive seconds request uses the \
manual tier, capped to three quarters of remaining time, with the configured \
`margin_row.frozen_dual_bab_node_cap`. Observable: one `[frozen-dual] class=` \
line per selected class and the `[frozen-dual] armed=1 objectives= closed= nodes= \
depth_max= secs=` summary on every armed run that reaches the tree driver; an \
armed run without the summary is VOID.

High because a closed class drops out of the tree's conjunction and the lane \
publishes `Unsat`: a wrong remainder, an indexing bug in the reverse build or \
an under-charged node bound is a false-`unsat` generator, and the tier's wall \
comes out of the tree's window. The exact-rational containment tests, the \
double-double real-ONNX containment pin, the exact-rational re-verification of \
the search's own prunes, the gamma mutation pin, the planted-violation and \
sat-class negative controls exist to catch it.",
        provenance: Provenance::Unmeasured {
            why_ok: "Generic default false; the shipped CIFAR100 and TinyImageNet presets \
                     enable the bounded automatic tier at the user's explicit request. \
                     Existing certifier and complete-tree cover checks retain verdict \
                     authority. The new default's official-budget benefit, full banked \
                     SAT controls and both-year regression comparison remain unmeasured; \
                     no current competition A/B or moat is claimed.",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "latch the governed env-over-preset gate once per process; the tree driver \
                   runs the certified frozen-dual tier on its outer-frame root when armed",
            site: "crates/ny-propagate/src/margin_row/zono_state.rs:armed",
        }],
    };

    /// `NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP` — nodes one objective's frozen
    /// dual search may evaluate.
    pub MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(2_000_000),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#frozen-dual-bab. Nodes (root excluded) one objective's certified BFS may \
evaluate, configured default 2,000,000 (P1x: cifar100 idx_3494 closes at 55k nodes and \
idx_9585 at 1.13M under the gate arm). Whitespace is trimmed before \
`parse::<usize>()`; malformed input leaves the default and the reader clamps \
to `1..=200,000,000`. Automatic mode additionally caps it at 8192 nodes per \
objective; a positive seconds request uses the configured cap. The search stops \
OPEN before a level that would cross \
the cap, so the cap can only hand a class back to the tree, never close one. \
High because it moves how much of the tree's window the tier may spend and \
which classes reach the tree. Inert unless NY_MARGIN_ROW_FROZEN_DUAL_BAB is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "The configured cap cannot certify a prune. The user-directed category \
                     defaults use the automatic 8192-node ceiling; explicit positive \
                     seconds requests retain the configured cap. Competition effects \
                     of the new default remain unmeasured.",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "resolve the per-objective node cap when the armed tier starts",
            site: "crates/ny-propagate/src/margin_row/zono_bab.rs:TierPolicy::from_gates",
        }],
    };

    /// `NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS` — wall request of the frozen dual
    /// tier.
    pub MARGIN_ROW_FROZEN_DUAL_BAB_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS",
        // Same shape as `NY_MARGIN_ROW_BI_CHUNK_SECS`: trimmed f64, finite,
        // CLOSED at zero because zero selects the bounded automatic tier.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#frozen-dual-bab. Seconds the tier may spend over open classes. Default 0 \
selects automatic mode: at most 4 seconds and 5% of remaining lane time, \
8192 nodes per objective and 256 MiB of state. Dense row estimates at least \
64 MiB select one easiest open objective; smaller states select at most two. A positive \
value is a REQUEST, capped to three quarters of that time so the tree always \
keeps a quarter. Whitespace is trimmed before `parse::<f64>()`; non-finite, \
negative and malformed values leave 0. The wall is split fairly across the \
open classes, easiest (largest root bound) first; a class whose share runs out \
is handed to the tree open. High because the tier's wall comes out of the \
tree's window on the lane that decides cifar100 and tinyimagenet. Inert unless \
NY_MARGIN_ROW_FROZEN_DUAL_BAB is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "The user-directed CIFAR100 and TinyImageNet defaults use bounded \
                     automatic mode. A budget can only leave unfinished objectives open; \
                     every closure still requires the existing certifier and complete \
                     tree cover. Competition effects remain unmeasured.",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "resolve the tier's wall request when the armed tier starts",
            site: "crates/ny-propagate/src/margin_row/zono_bab.rs:TierPolicy::from_gates",
        }],
    };

    /// `NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE` — how the frozen dual tier builds
    /// its per-class states (`#fd-shared-state`, ENGINE_PROGRAM M3).
    pub MARGIN_ROW_FROZEN_DUAL_BAB_STATE = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_STATE",
        kind: LeverKind::Enum(&["per_class", "shared"]),
        default: DefaultSpec::Text("per_class"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#fd-shared-state (ENGINE_PROGRAM M3). How the #frozen-dual-bab tier builds the \
certified state of each open class. Inert unless NY_MARGIN_ROW_FROZEN_DUAL_BAB \
is armed.

* `per_class` (default): today's tier bit for bit. Each class builds its own \
state (one margin pass that chooses every slope direction-adaptively, then one \
row pass per unstable layer chunk) inside its own fair share of the tier.
* `shared`: the row passes read only the slopes of the layers upstream of their \
row, so every class's rows are identical once those slopes are shared. Before \
any fair share is computed, under the tier deadline, the tier prices two slope \
candidates with one-column margin passes only (the joint margin sum_j (y_t - \
y_j)'s direction-adaptive slopes, and the hardest open class's own slopes), \
keeps the one with the most classes within a root loss of 0.01 (then the \
smaller summed loss) and builds every unstable row ONCE under it (the trunk). \
Each class then runs a one-column margin tail that reads every trunk slope and \
re-chooses only the tail layers no row reads (the head ReLU), copies the rows \
and searches exactly as before. On tiny t4863 six per-class builds took 11.41 s \
of a 27.2 s tier; on t8853 all 15 builds were refused because each ran under a \
~1.3 s share. Observable: `[frozen-dual] trunk state=shared lambda= \
candidates= select_ms= trunk_ms= passes= shared_root_loss_max= within_tol= \
refused=` plus `state= trunk_ms= classes_built= shared_root_loss_max= lambda= \
fd-shared-state/v1` on the summary line; an armed `shared` run without the \
trunk line is VOID.

SOUNDNESS. Any slope in [0, 1] certifies (mu is certified from the certified \
box), so the candidate choice costs only tightness (the shared-lambda root \
loss). The assembly refuses a class tail whose slope or mu differs in any bit \
from the trunk's on a layer a row may read, and a trunk whose content digest \
changed since it was built. With the anchor candidate the anchor class's \
shared state is its per-class state bit for bit. High because a closed class \
leaves the tree's conjunction and the mode moves how much of the tier's wall \
each class's search gets.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `per_class` (bit-identical, pinned against a frozen copy of the \
                     pre-M3 builder) and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; the shared states' exact-rational and \
                     double-double containment, the anchor bit-identity, the tail/trunk \
                     slope-consistency and digest refusals, garbage-slope fault injection \
                     and the tier's sat-class / planted-violation negative controls are \
                     pinned by unit tests. Unmeasured: the shared-lambda root loss, classes \
                     built and conversions on t8853 / t4863 / FDB6 and 10 E1-26 tiny rows \
                     (ctl vs armed x3), the sat negative controls with the tier engaged and \
                     a both-year moat",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "resolve how the armed tier builds its per-class states",
            site: "crates/ny-propagate/src/margin_row/zono_bab.rs:TierPolicy::from_gates",
        }],
    };

    /// `NY_MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH` — which search scores the
    /// frozen-dual tier's nodes (`#wide-fd-search`, ENGINE_PROGRAM M4).
    pub MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH",
        kind: LeverKind::Enum(&["f64", "cpu_f32", "cuda", "auto"]),
        default: DefaultSpec::Text("f64"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#wide-fd-search (ENGINE_PROGRAM M4). Selects which search SCORES the \
frozen-dual tier's nodes; the certificate is the same for every value. Inert \
unless NY_MARGIN_ROW_FROZEN_DUAL_BAB is armed.

* `f64` (default): the tier's own f64 search (#fd-block-eval) bit for bit; no \
backend is consulted.
* `cpu_f32`: the injected `ny_core::search::SearchBackend` registered by ny-cli \
(ny-bbe's CPU f32 SIMD level search) scores every node of each level chunk at \
its four dual points (warm, slack-clamped, floor, far; far-first) and proposes \
the points and the children's warm start. A node whose search value exceeds \
-tau (tau = 4 gamma_f32(D) S) is CERTIFIED on the lane's CPU pool by the \
tier's own f64 block evaluator (bit-identical to eval_node) at the winning \
point, then at the remaining points, and pruned only on a certified value > 0; \
a node no point certifies is REFUTED and split, never pruned. The level ledger \
counts certified prunes only.
* `cuda`: the same protocol on ny-cuda's device search: kernels that reproduce \
the CPU f32 search bit for bit (known-answer-tested against it), in a CUDA \
context of their own, lazily opened and self-tested on the first upload and \
fail-soft (no driver, no device or a failed self-test refuses, and the class \
runs `f64`).
* `auto`: `cuda` when it uploads and passes its probe, else `cpu_f32`, else \
`f64`.

Per class the backend gets an f32 copy of the problem under a memory lease \
(the class's substituted problem, whichever way `margin_row.frozen_dual_bab_state` \
built its state: `per_class`, or over the `shared` trunk after the trunk's \
class, layout and content-digest bindings accepted the class's tail) \
and a lane-start known-answer probe (warm values on 16 deterministic nodes \
against the certified evaluator); a refusal, panic or probe failure runs the \
class on `f64`, and a mid-search refusal scores the rest of the class on \
`f64`. Observables: `[bbe] armed=1 backend= probe= lease_mb= search= classes=` \
once per armed tier, and `search= probe= claimed= refuted= fallback=` on every \
`[frozen-dual] class=` line (`search= claimed= refuted= fd-wide-search/v1` on \
the summary); an armed non-`f64` run without the `[bbe]` line is VOID.

SOUNDNESS. The backend is untrusted end to end: it never sees or changes the \
tree (the level's nodes and sides are the tier's own), and its multipliers \
enter the certified evaluator only through the ingest gate (finite, >= 0; \
anything else becomes 0) — every non-negative dual point is valid by weak \
duality, so a wrong proposal can only fail to certify. High because it changes \
which nodes the tier prunes inside its budget, hence which classes close and \
how much of the tree's window the tier spends.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `f64` (the M1 path, bit-identical: the backend is never \
                     consulted) and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; fault injection (garbage backend values never \
                     create a close), the exact-rational re-verification of the wide path's \
                     own prunes, the honest-backend tree comparison and thread-count \
                     independence are pinned by unit tests; the certified nodes/s (S1), the \
                     refuted/claimed rate and the conversions A/B are what promoting the \
                     key makes runnable",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "resolve the tier's search selector when the armed tier starts",
            site: "crates/ny-propagate/src/margin_row/zono_bab.rs:TierPolicy::from_gates",
        }],
    };

    /// `NY_MARGIN_ROW_TREE_MODE` — which tree engine the margin-row lane's
    /// outer frame runs (`#wave-beta-bab`, ENGINE_PROGRAM M6).
    pub MARGIN_ROW_TREE_MODE = LeverDecl {
        name: "NY_MARGIN_ROW_TREE_MODE",
        kind: LeverKind::Enum(&["legacy", "wave", "wave_cuda"]),
        default: DefaultSpec::Text("legacy"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#wave-beta-bab (ENGINE_PROGRAM M6). Selects the margin-row tree engine of the \
OUTER frame; nested epoch frames and the classwise schedule always run the \
legacy tree.

* `legacy` (default): today's best-first tree (serial or parallel frontier) \
bit for bit; the wave engine is never entered.
* `wave`: every open tree class is a root domain in ONE LIFO host pool. A wave \
pops up to NY_MARGIN_ROW_TREE_WAVE_BATCH domains and hands them to the \
injected `ny_core::search::SearchBackend` registered by ny-cli (ny-bbe's CPU \
f32 search): output-only beta-CROWN with the certified root boxes FROZEN and \
the splits PINNED, Adam on each domain's slopes and split multipliers, \
keep-best, then BaBSR intercept branching. A domain whose search bound clears \
-tau + delta (tau = 4 gamma_f32(L) S; delta rises while the certify queue \
refutes more than 20% of its claims) is CLAIMED; every other domain is split \
(several levels at once while the pool is thinner than a wave). Claims are \
certified on a PRIVATE rayon pool (NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS), \
concurrently with the next wave's search, by the lane's own Outward domain \
evaluation with the hints clamped at ingest, climbing the frozen (root y-pack) \
-> clip (split half-spaces) -> y-pack (the domain's own) ladder; a claim no \
rung certifies is REFUTED and split, never closed. A leaf closes only on the \
strict certified predicate or a certified emptiness slack; the verdict needs \
every class's path-based cover proof and the frame's authorization. A missing \
backend, a refused upload or a pool failure runs the legacy tree.
* `wave_cuda`: `wave` with the search on the DEVICE (ny-cuda's \
CudaSearchBackend, registered by ny-cli built with the `cuda` feature): the \
committed kernels `wave_init/_backward/_concretize/_forward/_adam/_babsr` \
reproduce the CPU f32 search bit for bit, in the backend's own CUDA context, \
self-tested on the first upload, admitted against the same lease. A device \
that is absent, unavailable, fails its self-test or refuses the lease falls \
through to the CPU f32 search (`backend=cpu_f32 device_refused=<why>`), and \
only then to the legacy tree. THE DEVICE NEVER CERTIFIES: the certify queue, \
the ladder and the cover gate are `wave`'s, unchanged.

Observable: one `[bbe] phase=wave armed=1 backend= B= cert_threads= universe= \
... claims= closed= refuted= rung_frozen= rung_clip= rung_ypack= tau= delta= \
stop= mode= device_refused= device_switch= wave-beta-bab/v1` line per armed outer \
frame (with \
`refused=` and `fallback=legacy` on a refusal); an armed run without it is \
VOID, and an armed `wave_cuda` run is a DEVICE run only with `backend=cuda`.

SOUNDNESS. The backend is untrusted end to end and reads a plain-data copy of \
the lane IR; its multipliers enter the certified pass only through the beta \
installers (finite and > 0, else dropped: every beta >= 0 is a valid weak-\
duality multiplier), its slopes only through the domain-alpha writer and the \
head gates (clamped into [0, 1], ms re-derived), its branch candidates only \
after validation against the domain's own literals, and its bounds only decide \
WHICH domains are certified. High because it replaces the lane's search \
trajectory on the lane that decides cifar100 and tinyimagenet.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `legacy` (bit-identical: the wave engine is never entered) and \
                     armed by no shipped yaml, so no scored run changes behaviour on \
                     promotion; fault injection (garbage search values never create a close), \
                     sampled containment of every wave leaf, the claim-as-close and clip-side \
                     mutation pins, the cover-gate leaf-drop pin, Err-as-close refusal and \
                     thread-count independence are pinned by unit tests; the certified \
                     closes/s, the refuted/claimed rate and the conversions A/B are what \
                     promoting the key makes runnable",
        },
        owner: WAVE_TREE,
        readers: &[ReaderSite {
            scope: WAVE_TREE,
            role: "freeze the tree engine with its two knobs into the scored lane's runtime \
                   policy (MarginRowRuntimePolicy::wave) from the same frozen LeverSet the \
                   flight receipt serialises (compatibility callers without a frozen set: \
                   WavePolicy::from_gates)",
            site: "crates/ny-propagate/src/margin_row/bab/wave.rs:WavePolicy::from_levers",
        }],
    };

    /// `NY_MARGIN_ROW_TREE_WAVE_BATCH` — domains per wave of the wave tree.
    pub MARGIN_ROW_TREE_WAVE_BATCH = LeverDecl {
        name: "NY_MARGIN_ROW_TREE_WAVE_BATCH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(256),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#wave-beta-bab. Domains one wave pops from the host pool and searches, default \
256 (abc's cifar BaB batch order). Whitespace is trimmed before \
`parse::<usize>()`; malformed input leaves the default and the reader clamps \
to `1..=16384`. At the host-pool cap the engine halves it (deepen instead of \
widening). Schedule only: every close stays certified, but the width decides \
which domains are certified inside the budget. High because it moves the \
search trajectory. Inert unless NY_MARGIN_ROW_TREE_MODE is `wave` or \
`wave_cuda`.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent selector is `legacy`, which is the shipped state",
        },
        owner: WAVE_TREE,
        readers: &[ReaderSite {
            scope: WAVE_TREE,
            role: "freeze the wave width with the tree engine into the scored lane's runtime \
                   policy (compatibility: WavePolicy::from_gates)",
            site: "crates/ny-propagate/src/margin_row/bab/wave.rs:WavePolicy::from_levers",
        }],
    };

    /// `NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS` — threads of the wave tree's
    /// private certify pool.
    pub MARGIN_ROW_TREE_WAVE_CERT_THREADS = LeverDecl {
        name: "NY_MARGIN_ROW_TREE_WAVE_CERT_THREADS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#wave-beta-bab. Threads of the PRIVATE rayon pool that certifies the wave \
tree's claims while the next wave searches, default 0 = half of the lane's \
rayon threads (the search takes the rest). Whitespace is trimmed before \
`parse::<usize>()`; malformed input leaves the default and the reader clamps \
to `..=1024`. The certify queue is double-buffered and merged in claim order, \
so the tree does not depend on this value; the wall per wave does. High \
because it moves how much of the budget certification takes. Inert unless \
NY_MARGIN_ROW_TREE_MODE is `wave` or `wave_cuda`.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent selector is `legacy`, which is the shipped state",
        },
        owner: WAVE_TREE,
        readers: &[ReaderSite {
            scope: WAVE_TREE,
            role: "freeze the certify-pool width with the tree engine into the scored lane's \
                   runtime policy (compatibility: WavePolicy::from_gates)",
            site: "crates/ny-propagate/src/margin_row/bab/wave.rs:WavePolicy::from_levers",
        }],
    };

    /// Optional breakpoint dual refinement within the existing frozen-dual tier.
    pub MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS = LeverDecl {
        name: "NY_MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#breakpoint-dual. Optional refinement steps per node in the existing frozen-dual \
BaB tier. Generic default 0 preserves its historical evaluator; the shipped \
CIFAR100 and TinyImageNet presets request 4 steps. The consuming policy \
clamps to `0..=32`. The untrusted search proposes projected multi-cut and \
coordinate steps at reduced-cost sign breakpoints. Every candidate still passes \
the existing `FrozenDualProblem::eval_node` certified bound, and only a strict \
certified improvement is retained. The tree cover and level ledger are unchanged; \
no hint can close a node. Whitespace is trimmed before `parse::<usize>()`; \
malformed input restores 0. Inert unless NY_MARGIN_ROW_FROZEN_DUAL_BAB is armed. \
High because the refinement changes pruning and consumes the tier's wall budget \
on the lane that publishes cifar100 and tinyimagenet unsat verdicts.",
        provenance: Provenance::Unmeasured {
            why_ok: "Generic default 0; the shipped CIFAR100 and TinyImageNet presets \
                     request 4 steps at the user's explicit request within the bounded \
                     automatic tier. All proposed dual points require the existing \
                     certifier and strict improvement. Official-budget conversions and \
                     the full both-year regression comparison remain unmeasured.",
        },
        owner: FROZEN_DUAL,
        readers: &[ReaderSite {
            scope: FROZEN_DUAL,
            role: "resolve the bounded refinement work when the armed frozen-dual tier starts",
            site: "crates/ny-propagate/src/margin_row/zono_bab.rs:TierPolicy::from_gates",
        }],
    };

    /// `NY_MARGIN_ROW_BI_WINDOW` — receptive-field-cropped backward-interm
    /// prefix passes (`#bi-window`, cct plan P5 step 1).
    pub MARGIN_ROW_BI_WINDOW = LeverDecl {
        name: "NY_MARGIN_ROW_BI_WINDOW",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#bi-window. Runs every `#backward-interm` prefix pass pair (and every \
`#joint-refresh` pair, which re-runs that phase) on a RECEPTIVE-FIELD CROP of \
the prefix network instead of the dense full-tensor walk. The phase orders its \
selected neurons tile-major, and for each chunk (one spatial tile of the seeded \
layer) builds the prefix ops restricted to that tile's cone — one rectangle per \
tensor, each conv recompiled over its rectangle with the crop's own padding — \
plus the matching slice of the root gates and the input box, and runs the \
UNCHANGED certified engine on it. Default false: the crop code is never \
entered and every pass is the historical dense pass, byte-identical.

WHY. A backward-interm column's coefficients are exactly zero outside its \
receptive field, but the dense pass walks every row of every prefix tensor: \
measured on cct narrow CNN7 the per-column cost tracks the dense prefix MAC \
count (2.2 / 3.9 / 6.7 / 7.3 ms per Lower+Upper column at relu_b..relu_e), so \
the phase covers ~30% of the unstable neurons and never reaches trunk layer 4 \
inside the 120 s tier (scratchpad/vbs/diag6/cct.md §2a).

SOUNDNESS. The cropped net computes EXACTLY the seeded pre-activations as a \
function of the cropped input (a crop-coordinate conv tap is either a real \
input inside the rectangle or a position that was zero padding in the full \
conv), so bounds over the cropped box equal bounds over the full box; the \
engine instantiates every certified envelope at the crop's own sizes, which are \
the true accumulation lengths of the nonzero terms. A cropped pair is never \
looser than the dense pair (same coefficients and bias up to the sign of zero, \
error lane dominated entrywise). Anything the crop cannot express (a residual \
Add, a transposed conv, a multi-consumer tensor) falls back to the dense pass. \
High because it changes which boxes the phase can afford to tighten inside its \
wall budget, and therefore the published root bound.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; the dense/crop differential and sampled \
                     containment are pinned by unit tests; the wall-clock effect on cct \
                     rows at official budgets is what promoting the key makes runnable",
        },
        owner: BACKWARD_INTERM,
        readers: &[
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "freeze the crop flag with the backward-intermediate policy from the \
                       scored LeverSet; the phase and #joint-refresh read it from there",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_BI_GPU` — the backward-interm prefix pass pairs on the
    /// certified sound-f64 GPU conv route (`#bi-gpu`, cct WAVE6 P3 /
    /// Vibecheck study_1 I3).
    pub MARGIN_ROW_BI_GPU = LeverDecl {
        name: "NY_MARGIN_ROW_BI_GPU",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
RETIRED 2026-09-29 (ENGINE_PROGRAM R2, the device never certifies): an armed \
phase is REFUSED at runtime (`[bi-gpu] refused reason=r2_device_never_certifies`, \
the dark CPU pairs bit for bit), the typed key `margin_row.bi_gpu` is refused at \
preset validation, and the axis is Unsafe / EnvOnly. Its cuBLAS products were \
the CERTIFIED intermediate boxes with none of the M9 preconditions (fmad-free \
kernels, SASS-only authority, live-state KAT, A10G moat), and its gated tests \
ran emulated engines only. What it was:

#bi-gpu. Runs every `#backward-interm` prefix pass pair — the baseline dense \
pair, the `#bi-window` cropped pair, every `#interm-alpha` candidate pair — and \
every `#joint-refresh` pair on an engine whose conv backward products go to the \
process-global SOUND f64 GEMM engine (cuBLAS `Dgemm` on a CUDA build) through a \
SCOPED `#mr-gpu-eval` route with its own counters (`margin_row/bi_gpu.rs`). \
Independent of `margin_row.gpu_conv`. Default false: no route is built and \
every pair is the historical CPU pair, byte-identical.

WHY. The root-phase intermediate refinement is the phase cct never finishes: \
on the 120 s narrow-CNN7 rows it spends ~38-40 s of work, covers ~30% of the \
unstable neurons and never reaches trunk layer 4, at a per-column cost that \
tracks the dense prefix MAC count (scratchpad/vbs/diag6/cct.md 2a) — and \
every one of those MACs is a conv backward product. WAVE6 cct P3 and \
Vibecheck study_1 I3 name it as the enabler for a joint refresh at abc's \
cadence (~45 s -> seconds per full refresh).

SOUNDNESS. The moved op, its admission and its per-op fallback are \
`#mr-gpu-eval`'s: the device consumes the stored f64 operands bit-for-bit, \
the engine contract is binary64 round-to-nearest in ANY summation order, and \
the pass's UNCHANGED order-independent charges (`gamma(k_bwd+2)` inside \
`comb`, `certify_up(., gamma(k_bwd+8))`) cover the device sum as they cover \
the CPU kernel's. Gates, the error carry from the exact identity seed and \
the outward concretize are unchanged host code, so each routed pair is a \
certified enclosure the phase intersects shrink-only as before. Every refused \
product (no/non-CUDA engine, deadline, backoff, non-finite operand or output, \
wrong length, transposed conv) runs the unchanged CPU kernel, bit-identical \
for that op; a pair is RSS-admitted (half the process headroom must cover \
512 MiB) or runs wholly on the CPU. High because it changes the arithmetic of \
the passes that set the root intermediate boxes (published bounds move at the \
ulp level of a reordered f64 sum) and the wall the phase spends, and a gather \
bug would be a false-`unsat` generator — which the exact-rational containment \
tests exist to catch. Engagement evidence: `[bi-gpu] done armed=1 pairs= \
gpu=` > 0; armed with `gpu=0` prints `verdict=VOID`.",
        provenance: Provenance::Unmeasured {
            why_ok: "RETIRED: inert in every production build (an armed phase is refused \
                     and runs BackwardEngine::new exactly as dark); exact-rational \
                     containment of routed pairs, CPU parity and bit-identical refusal \
                     fallback stay pinned by unit tests under a test-only device admission, \
                     as the reference for an M9 device certifier",
        },
        owner: BACKWARD_INTERM,
        readers: &[
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "freeze the (retired) device-route flag with the backward-intermediate \
                       policy from the scored LeverSet; the phase and #joint-refresh read it \
                       from there and REFUSE it (R2)",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_BI_PATCH` — per-column receptive-field (patch-mode)
    /// backward-interm prefix passes (`#bi-patch`).
    pub MARGIN_ROW_BI_PATCH = LeverDecl {
        name: "NY_MARGIN_ROW_BI_PATCH",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#bi-patch. Runs every `#backward-interm` prefix pass pair (the baseline pair \
and every `#interm-alpha` candidate pair) and every `#joint-refresh` pair as a \
PER-COLUMN RECEPTIVE-FIELD walk (`margin_row/patch.rs`): each trunk-seeded \
column carries its coefficients over its own receptive field on every prefix \
tensor — a virtual per-column frame, one shared geometry per chunk, only the \
anchors differ — instead of a dense row over the whole tensor. Convs run as \
virtual convs over the frames (same weights, bias and fold charges) through \
the unchanged conv kernel and route; relu / channel-affine read their gates \
at each row's absolute neuron; out-of-image rows are masked to exact zero; \
residual Adds embed into union frames. Walking backward, the first tensor \
whose frame exceeds the crossover ratio (and every tensor below it) runs as \
the dense tail. Lower and Upper are fused into one walk. Takes precedence \
over `#bi-window` on every layer it can plan; anything it cannot plan falls \
back to the crop or the dense pair. Default false: the patch code is never \
entered and every pass is the historical pass, byte-identical.

WHY. On cct narrow CNN7 the per-column signed product of a prefix pass costs \
0.052 / 0.449 / 2.76 / 8.07 M MACs in patch mode against 39.5 / 58.4 / 96.1 / \
133.9 M dense at relu_b..relu_e — ~37x under dense and ~3.3x under the \
measured crop at full coverage on idx448 — the substrate a joint refresh at \
abc's cadence needs (design: #bi-patch, geometry script pm_cost.py).

SOUNDNESS. The patch pass is the dense pass restricted to its support: a \
column's coefficient is exactly zero outside its frame (conv receptive field, \
elementwise relu/affine, Add union), every kept coefficient and bias is the \
dense f64 value up to the sign of zero, every Higham envelope is instantiated \
at the DENSE sizes, and the patch's error charges are an ordered subset of the \
dense charges through monotone updates, so its error lane is dominated \
entrywise and its bounds are never looser than the dense pair's — and it is a \
certified pass in its own right. Beta / per-column beta / infeasible gate \
overrides, transposed convs, unequal-stride unions and planner tripwires \
refuse and fall back. Pinned by the patch_tests suite (dense dominance, \
exact-rational containment at sampled points including point boxes, frame \
coverage, crossover, fused pair, refusals, cost model). High because it \
changes which boxes the phase can afford inside its wall budget, and \
therefore the published root bound. Engagement evidence: `[bi-patch] armed` \
then `[bi-patch] done: pairs=` > 0; armed with pairs=0 is VOID.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion: the dark phase never builds a plan and runs \
                     the historical passes; dense dominance, exact-rational containment \
                     and the refusal fallbacks are pinned by unit tests; the cct wall A/B \
                     at official budgets and the yaml-route arrival proof are what \
                     promoting the key makes runnable",
        },
        owner: BACKWARD_INTERM,
        readers: &[
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "freeze the patch-mode flag with the backward-intermediate policy from \
                       the scored LeverSet; the phase and #joint-refresh read it from there",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_BI_PATCH_RATIO` — the `#bi-patch` crossover ratio.
    pub MARGIN_ROW_BI_PATCH_RATIO = LeverDecl {
        name: "NY_MARGIN_ROW_BI_PATCH_RATIO",
        kind: LeverKind::F64ClosedTrimmed { min: 0.05, max: 1.0 },
        default: DefaultSpec::F64(0.5),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
The `#bi-patch` crossover ratio rho, default 0.5, admitted in the closed range \
[0.05, 1.0] after trimming (anything else resolves to 0.5). Walking a prefix \
backward, the first tensor whose per-column frame holds more than rho x its \
dense row count — and every tensor below it — runs as the DENSE TAIL (full \
frames, the real ops; the boundary is an exact scatter). A plan whose seed \
tensor itself would convert is refused (no gain) and the layer falls back. \
Execution strategy only: every setting is the same certified pass restricted \
to a support that contains the true one, so the ratio moves wall and bounds \
at the ulp level of the dropped exact-zero charges, never soundness. High \
because it changes the wall the phase spends. Inert unless patch mode is \
armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while patch mode is disarmed, which is the shipped state",
        },
        owner: BACKWARD_INTERM,
        readers: &[
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "freeze the crossover ratio with the backward-intermediate policy from \
                       the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_BI_PATCH_MB` — the `#bi-patch` per-pass working set.
    pub MARGIN_ROW_BI_PATCH_MB = LeverDecl {
        name: "NY_MARGIN_ROW_BI_PATCH_MB",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(256),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
The `#bi-patch` per-pass working set in MiB, default 256 (whitespace trimmed; \
the reader floors it at 1). A patch chunk's width is additionally capped so \
one fused Lower+Upper walk's live frame states plus conv temporaries fit \
this budget (priced from the plan's own frame sizes); the phase's configured \
chunk ceiling and adaptive width still apply. Batching only: the same \
columns are bounded, grouped differently. High because it changes the wall \
the phase spends and its memory on a unified-memory host. Inert unless patch \
mode is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while patch mode is disarmed, which is the shipped state",
        },
        owner: BACKWARD_INTERM,
        readers: &[
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "freeze the working-set cap with the backward-intermediate policy from \
                       the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: BACKWARD_INTERM,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:BackwardIntermRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_JOINT_REFRESH` — joint intermediate refresh rounds
    /// (`#joint-refresh`, cct plan P5 step 2). `0` disarms.
    pub MARGIN_ROW_JOINT_REFRESH = LeverDecl {
        name: "NY_MARGIN_ROW_JOINT_REFRESH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#joint-refresh. Rounds of the backward-only JOINT intermediate refresh, run \
after `#alpha-opt` / `#rebake` and before the tree. Default 0 disarms it: the \
entry returns before touching anything and prints nothing, so the lane is \
byte-identical. Whitespace is trimmed before `parse::<usize>()`; the reader \
clamps to 16. Per round: (1) re-derive every trunk intermediate box from the \
root's CURRENT gates (the committed alpha included) with the lane's own \
certified backward engine — the `#backward-interm` phase re-run over the \
finished root, `bi_topk` widest unstable neurons per layer, compounding down \
the trunk — intersected SHRINK-ONLY, touched gates re-certified \
(`gates_from_box` + `repair_upper_lines`, `ms = max(alpha, s)`); (2) score \
the candidate with the unchanged certified `root_eval`; (3) a 4-iteration \
alpha ascent whose frozen head frame (y-box, head gates, m1/m2v) is rebuilt \
FROM THE REFRESHED GATES, kept only on a strict certified improvement; \
(4) accept the round only if the certified root bound strictly improves. This \
is the backward-only half of alpha-CROWN's joint loop (every intermediate box \
recomputed from the current alphas, `interm_bounds=None`), without `#rebake`'s \
forward tableau rebuild, which measured unaffordable (4/4 refusals).

WHY. The cct ship-pin census: the frozen-frame ascent saturates by iteration \
~12 (idx448 -0.2957 -> -0.2943 over iters 12..20) or is inert with 2 of 3 \
classes pinned by the frozen head frame (idx1031, 0/20 accepted), while the \
root ascent pass costs only 0.385-0.47 s — the frame, not the pass, binds.

SOUNDNESS. Every candidate is internally certified by construction (certified \
enclosures intersected shrink-only; a CROSSED intersection refuses the whole \
round; chords re-certified on the tightened box; any alpha in [0,1]); the loop \
only CHOOSES among such roots on the certified number, every abort leaves the \
previous root standing, and the published bound is recomputed by `root_eval`. \
Refused unless `#backward-interm` is armed (it reuses that frozen policy). High \
because an accepted round changes the published root bound and spends root \
wall the tree search needs.",
        provenance: Provenance::Unmeasured {
            why_ok: "default 0 and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; shrink-only containment, never-looser and \
                     never-verifies-a-false-bound are pinned by unit tests; the cct A/B at \
                     official budgets is what promoting the key makes runnable",
        },
        owner: JOINT_REFRESH,
        readers: &[
            ReaderSite {
                scope: JOINT_REFRESH,
                role: "freeze the round cap from the scored LeverSet before the lane starts",
                site: "crates/ny-propagate/src/margin_row/mod.rs:JointRefreshRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: JOINT_REFRESH,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:JointRefreshRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_JOINT_REFRESH_SECS` — wall request for the joint refresh.
    pub MARGIN_ROW_JOINT_REFRESH_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_JOINT_REFRESH_SECS",
        // Same shape as `NY_MARGIN_ROW_REBAKE_SECS`: trimmed f64, finite,
        // CLOSED at zero (an explicit 0 refuses every round as unaffordable).
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(30.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds the joint intermediate refresh may spend, default 30. Whitespace is \
trimmed before `parse::<f64>()`; non-finite, negative and malformed values all \
leave 30. A REQUEST, not an authority: further capped to 40% of the remaining \
deadline; each round reserves two certified evaluations and one ascent slice \
(priced from measured costs) before its refresh starts and is refused LOUDLY \
when the remainder cannot pay; the refresh itself is cut at its slice (a \
partial refresh is sound). Every abort leaves the previously accepted root \
standing. High because the budget decides how many rounds run and the wall \
comes out of the tree search's window. Inert unless NY_MARGIN_ROW_JOINT_REFRESH \
requests a round.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is 0, which is the shipped state",
        },
        owner: JOINT_REFRESH,
        readers: &[
            ReaderSite {
                scope: JOINT_REFRESH,
                role: "freeze the wall request with the round cap from the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:JointRefreshRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: JOINT_REFRESH,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:JointRefreshRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_HEAD_GROUPS` — bounded Source-owned root refinement.
    pub MARGIN_ROW_HEAD_GROUPS = LeverDecl {
        name: "NY_MARGIN_ROW_HEAD_GROUPS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Enables source-owned joint head relaxations once on the outer margin-row root. \
Each selected two- or four-neuron group closes every phase; floating-point \
proposals acquire authority only through outward replay against the owned \
root sandwiches, authored final-head coefficient differences, and charged \
fold errors. The fixed policy allows at most two groups per remaining margin, \
four proposal rounds, and five seconds bounded by the original lane deadline. \
The directional-cut stage has a separate boolean and complete phase cover is \
independently opt-in. Each accepted class bound is combined \
with its incumbent by max, and the existing complete-class receipt and verdict \
gate remain the only proof authority. High because an improved root bound \
changes what the lane may prove and the pass spends the tree's wall budget. \
Default true and explicitly enabled by the shipped SafeNLP preset. Explicit \
false disables the entire group route; malformed legacy values return the \
true declaration default. No competition gain is implied by this default.",
        provenance: Provenance::OwnerDirected {
            directive: "The user explicitly requests combined Source-certified directional \
                        cuts DEFAULT ON, including the parent head-group production route",
            limitations: "No measured performance benefit is established. Only Source-certified \
                          replay may raise a bound within the original deadline. Existing \
                          phase-cover development measurements added no solved queries; \
                          the new directional stage needs its own clean qualification",
        },
        owner: HEAD_GROUPS,
        readers: &[
            ReaderSite {
                scope: NY_CLI_PRESET,
                role: "project the typed boolean into the scored receipt-owning LeverSet",
                site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "freeze the arming boolean from the scored LeverSet into the fixed bounded policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "resolve the compatibility environment and typed preset once at lane entry",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_process_environment",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "run one complete head-phase pass on the owned outer root after memo reuse",
                site: "crates/ny-propagate/src/margin_row/bab.rs:run_inner / run_classwise_inner",
            },
        ],
    };

    /// `NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS` — Source-certified coupled root cuts.
    pub MARGIN_ROW_HEAD_DIRECTIONAL_CUTS = LeverDecl {
        name: "NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Combines locally replayed Source directional cuts in one root relaxation, \
inside the existing head-group slice and original instance deadline. Default \
true; explicit false disables only this stage. The parent head_groups false \
disables the whole route. Invalid advice, failed replay or expiry preserves \
the incumbent. Proposals carry no proof authority and accepted bounds still \
pass the existing complete-class verdict gate. Malformed legacy boolean \
values return the true declaration default. No runtime or points claim.",
        provenance: Provenance::OwnerDirected {
            directive: "The user explicitly requests combined Source-certified directional \
                        cuts DEFAULT ON, including the parent head-group production route",
            limitations: "No measured performance benefit is established. Source-owned \
                          certification and original-deadline fallback remain required, \
                          with independent qualification before benefit claims",
        },
        owner: HEAD_GROUPS,
        readers: &[
            ReaderSite {
                scope: NY_CLI_PRESET,
                role: "project the typed boolean into the scored receipt-owning LeverSet",
                site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "freeze the directional stage choice inside the parent head-group policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "resolve compatibility layering once at lane entry",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_process_environment",
            },
        ],
    };

    /// `NY_MARGIN_ROW_HEAD_PHASE_COVER` — optional bounded Source phase cover.
    pub MARGIN_ROW_HEAD_PHASE_COVER = LeverDecl {
        name: "NY_MARGIN_ROW_HEAD_PHASE_COVER",
        kind: LeverKind::Usize { min: 0 },
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Optional complete phase cover on the authenticated original-input Source head. \
Zero is off; explicit counts 1..=7 select at most seven heads and 128 closed \
leaves, within the existing head-group slice and original instance deadline. \
Requires the separately frozen head_groups boolean. Scored runtime policy \
rejects counts above seven; compatibility callers refuse that optional cover. \
Every leaf bound must be locally replayed against the same authored Source, \
and a partial, invalid or expired cover cannot replace the incumbent root. \
The global declaration stays zero. The owner-authorized SafeNLP2024 category \
sets seven; other callers retain their explicit count or the off default. \
Category arming changes no proof or deadline authority and claims no matched \
performance, broad nonregression, points or compiler proof.",
        provenance: Provenance::Unmeasured {
            why_ok: "global default zero; the owner-authorized SafeNLP2024 category \
                     explicitly selects seven with independently qualified Source \
                     replay and original-clock refusal. Category policy authority \
                     is not measured performance evidence; no matched benefit or \
                     broad nonregression claim is established here",
        },
        owner: HEAD_GROUPS,
        readers: &[
            ReaderSite {
                scope: NY_CLI_PRESET,
                role: "project the typed count into the receipt-owning frozen LeverSet",
                site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "validate and freeze the optional complete Source phase-cover policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: HEAD_GROUPS,
                role: "resolve compatibility layering once and refuse unbounded cover requests",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowRuntimePolicy::from_process_environment",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_CERTIFY_ONCE` — Phase-0.5 root alpha, SEARCH
    /// UNTRUSTED / CERTIFY ONCE (`#root-certify-once`).
    pub MARGIN_ROW_ROOT_CERTIFY_ONCE = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#root-certify-once. Arms the Phase-0.5 root alpha on the margin-row lane, run \
after `#alpha-opt` / `#rebake` / `#joint-refresh` and before `#spec-alpha` and \
the tree. The root alpha ascent (the lane's own `search_alpha`: frozen-frame \
objective, supergradient, Adam, head alpha, monotone accept) runs UNTRUSTED on \
a Parity scratch copy of the root — plain f64 round-to-nearest, no |W| error \
lane — optionally preceded by joint rounds that re-derive the scratch's \
intermediate boxes from its current gates (the untrusted twin of the \
`#joint-refresh` refresh); then the chosen alpha is installed into the \
certified root through an ingest gate (finite, clamped to [0, 1], only on \
neurons unstable in the certified boxes, ms = max(alpha, s) re-derived) and \
scored by ONE certified Outward `root_eval` (after a joint phase also ONE \
certified refresh of its boxes and one more evaluation). The root is replaced \
only by a candidate whose CERTIFIED bound strictly beats the certified \
incumbent; otherwise it is restored bit-for-bit. Default false: the entry \
returns before touching anything and prints nothing, so the lane is \
byte-identical.

WHY. The certified ascent spends one Outward pass per candidate purely to \
RANK alphas (27-29 s of a 100 s tinyimagenet row for 9-20 iterations, \
vibecheck study 3 idea 2); abc and Vibecheck run the same free parameter \
uncertified. Any alpha in [0, 1] is a valid lower line, so only the published \
alpha needs a certified pass.

SOUNDNESS. Untrusted numbers only rank: the scratch is never read by a \
verdict path, the untrusted ascent and refresh refuse an Outward root, and \
the published bound is the unchanged certified root_eval of exactly the kept \
gates. High because an accepted candidate changes the published root bound \
and the phase spends root wall the tree search needs.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; containment (sampled and exact-rational), \
                     never-worse-than-the-incumbent, never-verifies-a-false-bound and the \
                     published-root-is-the-certified-one contract are pinned by unit tests; \
                     the cifar100 / tinyimagenet / cct A/B at official budgets is what \
                     promoting the key makes runnable",
        },
        owner: ROOT_CERTIFY_ONCE,
        readers: &[
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "freeze the arming gate from the scored LeverSet before the lane starts",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS` — untrusted spec-phase ascent
    /// iterations of `#root-certify-once`.
    pub MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(20),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Untrusted spec-phase iterations of `#root-certify-once` (the abc / Vibecheck \
Phase-0.5 recipe's 20), run on the Parity scratch with its intermediate boxes \
frozen. Default 20. Whitespace is trimmed before `parse::<usize>()`; the \
reader clamps to 1..=128. The phase's wall request is the binding cap in \
practice. High because the iteration count decides which alpha is offered to \
the single certified evaluation. Inert unless NY_MARGIN_ROW_ROOT_CERTIFY_ONCE \
is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent gate is off, which is the shipped state",
        },
        owner: ROOT_CERTIFY_ONCE,
        readers: &[
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "freeze the untrusted iteration count with the arming gate",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH` — untrusted joint-intermediate
    /// rounds of `#root-certify-once`.
    pub MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_REFRESH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Untrusted joint-intermediate rounds of `#root-certify-once` before its spec \
phase (abc's per-iteration intermediate recomputation; the Phase-0.5 recipe \
runs 10). Each round re-derives every trunk box of the Parity SCRATCH from its \
current gates (the `#backward-interm` policy's bi_topk / bi_chunk / bi_window, \
shrink-only, uncertified) and takes one untrusted ascent step on the refreshed \
frame. When a round tightened anything, the certification also runs ONE \
certified refresh of the chosen alpha's boxes (the `#joint-refresh` refresh \
verbatim) and keeps it only if its certified bound strictly beats both the \
incumbent and the alpha-only candidate. Default 0 (spec phase only). \
Whitespace is trimmed before `parse::<usize>()`; the reader clamps to 16. \
Refused (spec phase only) unless `#backward-interm` is armed. High because \
it decides which boxes and alpha reach the certified evaluation and spends \
root wall. Inert unless NY_MARGIN_ROW_ROOT_CERTIFY_ONCE is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent gate is off, which is the shipped state",
        },
        owner: ROOT_CERTIFY_ONCE,
        readers: &[
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "freeze the untrusted joint round count with the arming gate",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS` — wall request of
    /// `#root-certify-once`.
    pub MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_CERTIFY_ONCE_SECS",
        // Same shape as `NY_MARGIN_ROW_JOINT_REFRESH_SECS`: trimmed f64,
        // finite, CLOSED at zero (an explicit 0 refuses the phase as
        // unaffordable).
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(20.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds `#root-certify-once` may spend, default 20. Whitespace is trimmed \
before `parse::<f64>()`; non-finite, negative and malformed values all leave \
20. A REQUEST, not an authority: further capped to 40% of the remaining \
deadline; the certified evaluation (and, with joint rounds, the certified \
refresh slice) is reserved before the untrusted search may spend anything, and \
the phase refuses LOUDLY when the remainder cannot pay a search step. Every \
refusal leaves the incumbent root standing. High because the budget decides \
how far the untrusted search gets and the wall comes out of the tree search's \
window. Inert unless NY_MARGIN_ROW_ROOT_CERTIFY_ONCE is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent gate is off, which is the shipped state",
        },
        owner: ROOT_CERTIFY_ONCE,
        readers: &[
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "freeze the wall request with the arming gate",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_IBP_SEED` — certified IBP boxes intersected shrink-only
    /// with the root tableau's trunk boxes (`#mr-ibp-seed`).
    pub MARGIN_ROW_IBP_SEED = LeverDecl {
        name: "NY_MARGIN_ROW_IBP_SEED",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#mr-ibp-seed. Carries one CERTIFIED interval (IBP) box per tensor through the \
margin-row root tableau walk, from the declared input box: conv and channel \
affine as centre `fl(W mid + b)` plus radius `|W| rad` with every rounding \
charge (Higham gamma for the centre dot product, the bias add, gradual \
underflow) AND every fold charge (the BN fold's relative weight / scale error \
and absolute bias / shift error), rounded outward; ReLU from the layer's FINAL \
published box. At every trunk ReLU, AFTER the forward-only `bi_cal` capture \
and BEFORE `#backward-interm`, the tableau box is intersected SHRINK-ONLY with \
the IBP box; a crossing keeps the tableau box and is counted (a defect signal, \
never an emptiness certificate). Default false: the interval state is never \
built, the walk is byte-identical and prints nothing.

WHY. On IBP-trained nets (the challenging_certified_training_2026 eps8 CNN7s) \
interval boxes dominate linear-relaxation boxes: CROWN back-substitution \
tightened 0 of 4,378 IBP-unstable neurons on idx8072, and a 0.1 s CROWN-IBP \
proves idx5323 at the root (+0.086) where the lane root sat at -4.818 with 8 \
classes open, because the lane never intersected its boxes with IBP.

SCOPE. Chain trunks only (phase 1): a trunk with a residual Add declines with \
`declined=residual` and intersects nothing, so the cifar100 / tinyimagenet \
resnets stay byte-identical even with the key armed. Parity mode declines.

SOUNDNESS. The intersection of two certified enclosures, shrink-only, with the \
`clip_rows` calibration kept on the forward-only lines; every charge is a \
non-negative magnitude added to a radius, so none depends on an input's sign \
(the class of leak #root-rounding / #fold-ball closed in the tableau). High \
because an UNDER-charged radius would publish an over-tight box — a false-unsat \
generator uncatchable by field truth on NOBODY rows — which the exact \
dyadic-rational containment tests (with a dropped-fold-charge mutation pin) and \
the real-ONNX cct gates exist to catch. Observable on every armed build: \
`[mr-ibp-seed] layer=<k> neurons= tightened= stabilized= crossed= declined=` \
per trunk ReLU and a `[mr-ibp-seed] done` line; an armed run without them is \
VOID.",
        provenance: Provenance::Unmeasured {
            why_ok: "default false and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; exact-rational containment (a CNN7-shaped chain \
                     with BN folds, asymmetric boxes, active relus, and a dropped-fold-charge \
                     mutation pin), shrink-only / crossing refusal, the bi_cal ordering \
                     mutation, dark byte-identity and the residual decline are pinned by unit \
                     tests; the cct GAP / NOBODY / negative-control / moat measurement is what \
                     promoting the key makes runnable",
        },
        owner: IBP_SEED,
        readers: &[
            ReaderSite {
                scope: IBP_SEED,
                role: "freeze the seed flag with the root-precision policy every root walk \
                       (initial build, rebake round, epoch rebuild) receives",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: IBP_SEED,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_SEARCH` — which UNTRUSTED search proposes the root
    /// slopes inside `#root-certify-once` (`#bbe-joint-root`, ENGINE_PROGRAM
    /// M5a).
    pub MARGIN_ROW_ROOT_SEARCH = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_SEARCH",
        kind: LeverKind::Enum(&["ascent", "joint_f32"]),
        default: DefaultSpec::Text("ascent"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#bbe-joint-root (ENGINE_PROGRAM M5a). Selects the UNTRUSTED search half of \
`#root-certify-once`; the certify half is identical for both values. Inert \
unless NY_MARGIN_ROW_ROOT_CERTIFY_ONCE is armed.

* `ascent` (default): today's phase bit for bit — the Parity-scratch ascent \
(plus its optional untrusted refresh rounds); no search backend is consulted.
* `joint_f32`: abc's joint-alpha ROOT recipe run by the injected \
`ny_core::search::SearchBackend` (ny-bbe's CPU f32 backend, registered by \
ny-cli): CROWN init against the certified boxes as reference, spec prune, \
per-start-node alpha with an in-loop refresh of the unstable rows (per row, \
shared per start node when it does not fit the memory cap), Adam with lr 0.25 \
and decay 0.98, early exits (all classes above zero, patience 10, the \
iteration cap `margin_row.root_certify_once_iters`, the phase's search \
window). Its final start node's slopes go through the existing ingest gate \
(finite, clamped to [0, 1], only on neurons unstable in the CERTIFIED boxes, \
ms = max(alpha, s) re-derived) into ONE certified Outward root_eval, kept only \
if it strictly beats the certified incumbent; with \
`margin_row.root_certify_once_refresh > 0` and `#backward-interm` armed, also \
ONE certified refresh of that root's boxes. Observable on every armed return \
path: `[bbe] phase=joint_root iters= search_s= certify_s= root_lb= ...` \
(`root_lb` is the search's own f32 bound, the program's S3 metric, never \
published); an armed run without it is VOID.

SOUNDNESS. The search is untrusted end to end (f32, no rounding charged) and \
reads a plain-data COPY of the lane IR; only slopes leave it, only through the \
ingest gate, and any slope in [0, 1] is a valid lower line. A backend refusal, \
panic or missing registration leaves the incumbent bit-for-bit. High because \
an accepted candidate changes the published root bound and the search spends \
the phase's root wall.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `ascent` (bit-identical) and armed by no shipped yaml, so no \
                     scored run changes behaviour on promotion; the fault-injected ingest \
                     (garbage hints never loosen the root and never certify past sampled or \
                     exact-rational truth), ascent bit-identity with the backend never \
                     consulted, and the search's f32-vs-f64 reference tracking are pinned by \
                     unit tests; S3 (search root within 0.01 / 2% of abc's on E1/E2 rows) \
                     and the conversions A/B are what promoting the key makes runnable",
        },
        owner: ROOT_CERTIFY_ONCE,
        readers: &[
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "freeze the search selector with the arming gate from the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_CERTIFY_ONCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootCertifyOnceRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ROOT_BOX_SOURCE` — which certified construction supplies
    /// the margin-row root's trunk boxes (`#root-box-source`, ENGINE_PROGRAM
    /// M5b).
    pub MARGIN_ROW_ROOT_BOX_SOURCE = LeverDecl {
        name: "NY_MARGIN_ROW_ROOT_BOX_SOURCE",
        kind: LeverKind::Enum(&["tableau", "patch_backward"]),
        default: DefaultSpec::Text("tableau"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#root-box-source (ENGINE_PROGRAM M5b). Selects how every margin-row root walk \
(the initial build, each #rebake round, each Tier-2 epoch rebuild) builds its \
trunk boxes.

* `tableau` (default): the dense forward DeepPoly (M, D) tableau, bit for bit; \
the root build compares one enum and never enters the patch-backward module, \
and prints nothing new.
* `patch_backward`: a CERTIFIED interval (IBP) reference box per tensor \
(#mr-ibp-seed's certified builders, restarted at every ReLU from the layer's \
final box, residual trunks included), then, for the neurons UNSTABLE in that \
reference only, one certified fused Lower+Upper backward prefix pass through \
the frozen gates of the earlier layers: #bi-patch's per-column \
receptive-field walk, the unchanged dense prefix pair wherever the plan \
refuses; CPU only. The published box is the shrink-only intersection of the \
two certified enclosures; the gates are derived from it exactly as the \
tableau walk derives them. A crossing (a defect signal), a non-finite \
reference or Parity mode DECLINES to the unchanged tableau build. Under it, \
clip_rows and Tier-0 retained rows are not produced (their consumers skip), \
#backward-interm and #mr-ibp-seed are subsumed, root_f32 / tableau_f32 are \
inert. Observable on every armed Outward build: `[root-box] layer=` per trunk \
ReLU and a terminal `[root-box] source=<patch_backward|tableau> ... \
certify_s=` line (`declined=` on the tableau fallback); an armed run without \
it is VOID.

SOUNDNESS. Every published box is the intersection of two certified \
enclosures (sign-safe radius charges; the engine's certified error lane and \
outward concretize), the unstable set is read from the published box, and the \
only untrusted input (the #rebake alpha seed) is admitted exactly as the \
tableau admits it. High because it replaces the source of every root box, \
the new composition is new TCB, and the published root bound moves.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `tableau` (bit-identical, the patch-backward module never \
                     entered) and armed by no shipped yaml, so no scored run changes \
                     behaviour on promotion; exact-rational containment of the \
                     patch-backward boxes (asymmetric boxes, active relus, the cancelling \
                     stable nets that hid the historical root leak, fold charges, \
                     residual trunks), E_patch <= E_dense on the built prefix, the \
                     parity / unclamped-reference / shifted-bound mutation pins, the \
                     crossing decline, garbage alpha seeds and thread-count independence \
                     are pinned by unit tests; the tiny / cifar root-wall measurement \
                     (kill line: certify_s > 0.5 x the tableau build), the ctl-vs-armed \
                     x3 and the both-year moat are what promoting the key makes runnable",
        },
        owner: ROOT_BOX_SOURCE,
        readers: &[
            ReaderSite {
                scope: ROOT_BOX_SOURCE,
                role: "freeze the box source with the root-precision policy every root walk \
                       (initial build, rebake round, epoch rebuild) receives",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: ROOT_BOX_SOURCE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootPrecisionRuntimePolicy::from_layered_process",
            },
        ],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, read_with, LeverValue, Source};

    fn resolve(decl: &'static LeverDecl, raw: Option<&str>) -> (LeverValue, Source) {
        let owned = raw.map(str::to_owned);
        let resolved = read_with(decl, move |_| owned);
        (resolved.value, resolved.source)
    }

    #[test]
    fn head_root_booleans_keep_true_defaults_and_layer_ownership() {
        for decl in [&MARGIN_ROW_HEAD_GROUPS, &MARGIN_ROW_HEAD_DIRECTIONAL_CUTS] {
            let layered = |raw: Option<&str>, preset: Option<bool>| {
                let owned = raw.map(str::to_owned);
                read_over_config_with(decl, move |_| owned, preset.map(LeverValue::Bool))
                    .expect("typed booleans are admissible")
            };

            let default = layered(None, None);
            assert_eq!(default.value, LeverValue::Bool(true));
            assert_eq!(default.source, Source::Default);
            for requested in [false, true] {
                let configured = layered(None, Some(requested));
                assert_eq!(configured.value, LeverValue::Bool(requested));
                assert_eq!(configured.source, Source::Config);
            }
            for (token, requested) in [("0", false), ("1", true)] {
                let overridden = layered(Some(token), Some(!requested));
                assert_eq!(overridden.value, LeverValue::Bool(requested));
                assert_eq!(overridden.source, Source::LegacyEnv);
            }
            for malformed in ["true", "false", " 1", "1 ", "", "on"] {
                let rejected = layered(Some(malformed), Some(false));
                assert_eq!(rejected.value, LeverValue::Bool(true));
                assert_eq!(rejected.source, Source::LegacyEnvRejected);
                assert_eq!(rejected.rejected_raw.as_deref(), Some(malformed));
            }
            assert!(read_over_config_with(decl, |_| None, Some(LeverValue::U64(1)),).is_err());
        }
    }

    #[test]
    fn head_phase_cover_count_keeps_off_default_and_layer_ownership() {
        let layered = |raw: Option<&str>, config: Option<u64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_HEAD_PHASE_COVER,
                move |_| owned,
                config.map(LeverValue::U64),
            )
            .unwrap()
        };
        let dark = layered(None, None);
        assert_eq!(dark.value, LeverValue::U64(0));
        assert_eq!(dark.source, Source::Default);
        for count in 0..=7 {
            let configured = layered(None, Some(count));
            assert_eq!(configured.value, LeverValue::U64(count));
            assert_eq!(configured.source, Source::Config);
        }
        for (token, count) in [("0", 0), ("1", 1), ("7", 7)] {
            let overridden = layered(Some(token), Some(3));
            assert_eq!(overridden.value, LeverValue::U64(count));
            assert_eq!(overridden.source, Source::LegacyEnv);
        }
        for malformed in ["-1", "true", " 7", "7 ", "", "18446744073709551616"] {
            let rejected = layered(Some(malformed), Some(7));
            assert_eq!(rejected.value, LeverValue::U64(0));
            assert_eq!(rejected.source, Source::LegacyEnvRejected);
        }
        assert!(read_over_config_with(
            &MARGIN_ROW_HEAD_PHASE_COVER,
            |_| None,
            Some(LeverValue::Bool(true)),
        )
        .is_err());
    }

    #[test]
    fn schedule_declarations_pin_defaults_kinds_and_immutable_reader_boundary() {
        let declarations = [
            (
                &MARGIN_ROW_RESERVE_SECS,
                LeverKind::U64Override,
                DefaultSpec::U64(45),
                Bucket::Debug,
                MoatRisk::High,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_ADAPTIVE_RESERVE,
                LeverKind::Bool,
                DefaultSpec::Bool(false),
                Bucket::Debug,
                MoatRisk::High,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_RESERVE_MAX_FRAC,
                LeverKind::F64OpenUnicodeOwned { min: 0.0, max: 1.0 },
                DefaultSpec::Unset,
                Bucket::Debug,
                MoatRisk::Low,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_ADAPTIVE_RELEASE_FRAC,
                LeverKind::F64OpenOverride { min: 0.0, max: 1.0 },
                DefaultSpec::F64(0.40),
                Bucket::Debug,
                MoatRisk::High,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_CONCURRENT,
                LeverKind::TrimmedExactOne,
                DefaultSpec::Bool(false),
                Bucket::Debug,
                MoatRisk::Low,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_EARLY_EXIT,
                LeverKind::TrimmedExactOne,
                DefaultSpec::Bool(true),
                Bucket::DefaultOn,
                MoatRisk::Low,
                "measured",
            ),
            (
                &VERIFIER_THREADS,
                LeverKind::UsizeTrimmedUnicodeOwned,
                DefaultSpec::Unset,
                Bucket::Debug,
                MoatRisk::Low,
                "unmeasured",
            ),
            (
                &MARGIN_ROW_CONCURRENT_THREADS,
                LeverKind::UsizeOverride { min: 0 },
                DefaultSpec::Unset,
                Bucket::Debug,
                MoatRisk::Low,
                "unmeasured",
            ),
        ];

        for (decl, kind, default, bucket, moat, provenance) in declarations {
            assert_eq!(decl.kind, kind, "{}", decl.name);
            assert_eq!(decl.default, default, "{}", decl.name);
            assert_eq!(decl.bucket, bucket, "{}", decl.name);
            assert_eq!(decl.moat, moat, "{}", decl.name);
            assert_eq!(decl.owner, SCHEDULE, "{}", decl.name);
            assert_eq!(decl.readers, SCHEDULE_POLICY_READERS, "{}", decl.name);
            assert_eq!(decl.provenance.tag(), provenance, "{}", decl.name);
        }

        let sites = SCHEDULE_POLICY_READERS
            .iter()
            .map(|reader| reader.site)
            .collect::<Vec<_>>();
        assert!(sites.iter().any(|site| site.contains("from_levers")));
        assert!(sites
            .iter()
            .any(|site| site.contains("margin_row_schedule_preset_config_value")));
        assert!(sites
            .iter()
            .any(|site| site.contains("from_layered_process")));
        assert!(sites
            .iter()
            .any(|site| site.contains("margin_row_reserve_decision_with_policy")));
        assert!(sites
            .iter()
            .any(|site| site.contains("spawn_concurrent_lane_with_schedule_policy")));
        assert!(sites
            .iter()
            .any(|site| site.contains("handle_vnncomp_command")));
    }

    #[test]
    fn scheduler_resource_parsers_preserve_unicode_ownership_and_explicit_zero() {
        let layered = |decl: &'static LeverDecl, raw: Option<&str>, config: Option<LeverValue>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(decl, move |_| owned, config)
                .expect("typed scheduler resource values are admissible")
        };

        let default_early = layered(&MARGIN_ROW_EARLY_EXIT, None, None);
        assert_eq!(default_early.value, LeverValue::Bool(true));
        assert_eq!(default_early.source, Source::Default);
        let typed_early = layered(&MARGIN_ROW_EARLY_EXIT, None, Some(LeverValue::Bool(false)));
        assert_eq!(typed_early.value, LeverValue::Bool(false));
        assert_eq!(typed_early.source, Source::Config);
        let padded_early = layered(
            &MARGIN_ROW_EARLY_EXIT,
            Some(" 1 "),
            Some(LeverValue::Bool(false)),
        );
        assert_eq!(padded_early.value, LeverValue::Bool(true));
        assert_eq!(padded_early.source, Source::LegacyEnv);
        let typo_early = layered(
            &MARGIN_ROW_EARLY_EXIT,
            Some("true"),
            Some(LeverValue::Bool(true)),
        );
        assert_eq!(typo_early.value, LeverValue::Bool(false));
        assert_eq!(typo_early.source, Source::LegacyEnv);

        let absent_verifier = layered(&VERIFIER_THREADS, None, None);
        assert_eq!(absent_verifier.value, LeverValue::Unset);
        assert_eq!(absent_verifier.source, Source::Default);
        let zero_verifier = layered(&VERIFIER_THREADS, Some(" 0 "), Some(LeverValue::U64(7)));
        assert_eq!(zero_verifier.value, LeverValue::U64(0));
        assert_eq!(zero_verifier.source, Source::LegacyEnv);
        let malformed_verifier = layered(
            &VERIFIER_THREADS,
            Some("malformed"),
            Some(LeverValue::U64(7)),
        );
        assert_eq!(malformed_verifier.value, LeverValue::Unset);
        assert_eq!(malformed_verifier.source, Source::LegacyEnvRejected);

        let absent_lane = layered(&MARGIN_ROW_CONCURRENT_THREADS, None, None);
        assert_eq!(absent_lane.value, LeverValue::Unset);
        assert_eq!(absent_lane.source, Source::Default);
        let zero_lane = layered(
            &MARGIN_ROW_CONCURRENT_THREADS,
            Some("0"),
            Some(LeverValue::U64(7)),
        );
        assert_eq!(zero_lane.value, LeverValue::U64(0));
        assert_eq!(zero_lane.source, Source::LegacyEnv);
        let whitespace_lane = layered(
            &MARGIN_ROW_CONCURRENT_THREADS,
            Some(" 0 "),
            Some(LeverValue::U64(7)),
        );
        assert_eq!(whitespace_lane.value, LeverValue::U64(7));
        assert_eq!(whitespace_lane.source, Source::Config);
        assert_eq!(whitespace_lane.rejected_raw, None);
        assert_eq!(whitespace_lane.ignored_raw.as_deref(), Some(" 0 "));
    }

    #[test]
    fn adaptive_reserve_exact_bool_rejects_near_misses_over_typed_true() {
        let layered = |raw: Option<&str>, typed: bool| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_ADAPTIVE_RESERVE,
                move |_| owned,
                Some(LeverValue::Bool(typed)),
            )
            .expect("adaptive reserve typed config is Boolean")
        };

        assert_eq!(layered(None, true).source, Source::Config);
        assert_eq!(layered(Some("1"), false).value, LeverValue::Bool(true));
        assert_eq!(layered(Some("0"), true).value, LeverValue::Bool(false));
        for raw in ["true", " 1", "1 ", "", "garbage"] {
            let rejected = layered(Some(raw), true);
            assert_eq!(rejected.value, LeverValue::Bool(false), "{raw:?}");
            assert_eq!(rejected.source, Source::LegacyEnvRejected, "{raw:?}");
            assert_eq!(rejected.rejected_raw.as_deref(), Some(raw), "{raw:?}");
        }
    }

    /// Both exact-`"1"` gates keep the repo's arming rule byte for byte, and
    /// `"0"` stays an ADMISSIBLE disarm rather than a rejection — which is what
    /// `NY_MARGIN_ROW_GPU_BACKWARD`'s `Refusal::Disabled` and the
    /// `#backward-interm` env-wins-both-directions rule both depend on.
    #[test]
    fn exact_one_gates_preserve_the_arming_rule() {
        for decl in [
            &MARGIN_ROW_BETA,
            &MARGIN_ROW_GPU_BACKWARD,
            &MARGIN_ROW_BACKWARD_INTERM,
        ] {
            assert_eq!(
                resolve(decl, Some("1")),
                (LeverValue::Bool(true), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some("0")),
                (LeverValue::Bool(false), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, None),
                (LeverValue::Bool(false), Source::Default),
                "{}",
                decl.name
            );
            for reject in ["true", "TRUE", "yes", " 1", "01", "", "2"] {
                let (value, source) = resolve(decl, Some(reject));
                assert_eq!(value, LeverValue::Bool(false), "{} {reject:?}", decl.name);
                assert_eq!(
                    source,
                    Source::LegacyEnvRejected,
                    "{} {reject:?}",
                    decl.name
                );
            }
        }
    }

    /// The three-way `Some("1") / Some(_) / None` reader is exactly
    /// `read_over_config`'s layering, with the typed
    /// `margin_row.backward_interm` preset as the config layer.
    #[test]
    fn backward_interm_layers_env_over_preset_over_dark() {
        let layered = |raw: Option<&str>, preset: bool| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_BACKWARD_INTERM,
                move |_| owned,
                Some(LeverValue::Bool(preset)),
            )
            .expect("Bool config is admissible for a Bool declaration")
        };

        // Absent: the preset decides, in both directions.
        let from_preset = layered(None, true);
        assert!(from_preset.value.as_bool());
        assert_eq!(from_preset.source, Source::Config);
        assert!(!layered(None, false).value.as_bool());

        // Present and admissible: env wins over the preset, both ways.
        assert!(layered(Some("1"), false).value.as_bool());
        let killed = layered(Some("0"), true);
        assert!(
            !killed.value.as_bool(),
            "an explicit 0 must disarm the preset"
        );
        assert_eq!(killed.source, Source::LegacyEnv);

        // Present and inadmissible: `Some(_) => return None` in the legacy
        // reader. A near-miss token suppresses the preset instead of riding it.
        for reject in ["true", " 1", "01", "", "2"] {
            let r = layered(Some(reject), true);
            assert!(!r.value.as_bool(), "{reject:?} must not arm the phase");
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
        }
    }

    /// #margin-row-domain-alpha resolves through the same three-way layering as
    /// `#backward-interm`, with the typed `margin_row.domain_alpha` preset as
    /// the config layer. This is the half the reachability ratchet cares about:
    /// without it the mechanism cannot fire on a scored run at all.
    #[test]
    fn domain_alpha_layers_env_over_preset_over_dark() {
        let layered = |raw: Option<&str>, preset: bool| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_DOMAIN_ALPHA,
                move |_| owned,
                Some(LeverValue::Bool(preset)),
            )
            .expect("Bool config is admissible for a Bool declaration")
        };

        // Absent: the preset decides, in both directions.
        let from_preset = layered(None, true);
        assert!(from_preset.value.as_bool());
        assert_eq!(from_preset.source, Source::Config);
        assert!(!layered(None, false).value.as_bool());

        // Present and admissible: env wins over the preset, both ways.
        assert!(layered(Some("1"), false).value.as_bool());
        let killed = layered(Some("0"), true);
        assert!(
            !killed.value.as_bool(),
            "an explicit 0 must disarm the preset"
        );
        assert_eq!(killed.source, Source::LegacyEnv);

        // Present and inadmissible: a near-miss token SUPPRESSES the preset and
        // lands on the dark default — a typo is a kill switch, never a silent
        // promotion.
        for reject in ["true", " 1", "01", "", "2"] {
            let r = layered(Some(reject), true);
            assert!(!r.value.as_bool(), "{reject:?} must not arm the mechanism");
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
        }

        // Both ascent knobs layer the same way over their own typed keys.
        let iters = |raw: Option<&str>, preset: Option<u64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_DOMAIN_ALPHA_ITERS,
                move |_| owned,
                preset.map(LeverValue::U64),
            )
            .expect("u64 config is admissible for UsizeTrimmed")
        };
        assert_eq!(
            iters(None, None).value,
            LeverValue::U64(2),
            "shipped default"
        );
        assert_eq!(iters(None, Some(4)).value, LeverValue::U64(4));
        assert_eq!(iters(Some(" 1 "), Some(4)).value, LeverValue::U64(1));
        assert_eq!(
            iters(Some("bad"), Some(4)).source,
            Source::LegacyEnvRejected
        );

        let eta = |raw: Option<&str>, preset: Option<f64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_DOMAIN_ALPHA_ETA,
                move |_| owned,
                preset.map(LeverValue::F64),
            )
            .expect("f64 config is admissible for F64ClosedTrimmed")
        };
        assert_eq!(
            eta(None, None).value,
            LeverValue::F64(0.25),
            "shipped default"
        );
        assert_eq!(eta(None, Some(0.125)).value, LeverValue::F64(0.125));
        assert_eq!(eta(Some(" 0.5 "), Some(0.125)).value, LeverValue::F64(0.5));
        for reject in ["-1", "nan", "inf", "lots"] {
            assert_eq!(
                eta(Some(reject), Some(0.125)).source,
                Source::LegacyEnvRejected,
                "{reject:?}"
            );
        }

        // THE SCHEDULE LAYERS THE SAME WAY, and its defaults are the arm the
        // 2x2 measured: floor 0, stride 1, window 0 (= unlimited at the
        // reader). A schedule that silently defaulted to anything else would
        // make every future comparison against that 2x2 invalid.
        let sched_u64 = |decl: &'static LeverDecl, raw: Option<&str>, preset: Option<u64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(decl, move |_| owned, preset.map(LeverValue::U64))
                .expect("u64 config is admissible for UsizeTrimmed")
        };
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH, None, None).value,
            LeverValue::U64(0),
            "the shipped depth floor must admit every depth"
        );
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH, None, Some(4)).value,
            LeverValue::U64(4)
        );
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_MIN_DEPTH, Some(" 2 "), Some(4)).value,
            LeverValue::U64(2)
        );
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_STRIDE, None, None).value,
            LeverValue::U64(1),
            "the shipped stride must pay at every depth"
        );
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_STRIDE, None, Some(3)).value,
            LeverValue::U64(3)
        );
        assert_eq!(
            sched_u64(&MARGIN_ROW_DOMAIN_ALPHA_STRIDE, Some("bad"), Some(3)).source,
            Source::LegacyEnvRejected
        );

        let sched_gap = |raw: Option<&str>, preset: Option<f64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_DOMAIN_ALPHA_GAP,
                move |_| owned,
                preset.map(LeverValue::F64),
            )
            .expect("f64 config is admissible for F64ClosedTrimmed")
        };
        assert_eq!(
            sched_gap(None, None).value,
            LeverValue::F64(0.0),
            "the shipped window is 0, which the reader reads as UNLIMITED"
        );
        assert_eq!(sched_gap(None, Some(0.05)).value, LeverValue::F64(0.05));
        assert_eq!(
            sched_gap(Some(" 0.02 "), Some(0.05)).value,
            LeverValue::F64(0.02)
        );
        for reject in ["-1", "nan", "inf", "wide"] {
            assert_eq!(
                sched_gap(Some(reject), Some(0.05)).source,
                Source::LegacyEnvRejected
            );
        }
    }

    #[test]
    fn lru_declaration_preserves_the_untrimmed_integer_parser() {
        assert_eq!(
            resolve(&MARGIN_ROW_LRU, None),
            (LeverValue::U64(64), Source::Default)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_LRU, Some("16")),
            (LeverValue::U64(16), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_LRU, Some("0")),
            (LeverValue::U64(0), Source::LegacyEnv),
            "zero disables cache retention and is not malformed"
        );
        for reject in [" 16", "16 ", "", "-1", "lots"] {
            assert_eq!(
                resolve(&MARGIN_ROW_LRU, Some(reject)),
                (LeverValue::U64(64), Source::LegacyEnvRejected),
                "{reject:?}"
            );
        }
    }

    #[test]
    fn parallel_declaration_keeps_the_legacy_token_for_the_runtime_parser() {
        assert_eq!(
            resolve(&MARGIN_ROW_PARALLEL, None),
            (LeverValue::Text("1".to_owned()), Source::Default)
        );
        for token in ["", "0", "1", "TRUE", " 8 ", "malformed"] {
            assert_eq!(
                resolve(&MARGIN_ROW_PARALLEL, Some(token)),
                (LeverValue::Text(token.to_owned()), Source::LegacyEnv),
                "Text declaration must preserve {token:?} for the exact runtime grammar"
            );
        }
    }

    /// The trimming integer parsers, and the reader-side clamps that are
    /// deliberately NOT folded into the declarations.
    #[test]
    fn trimmed_integer_knobs_keep_their_defaults_and_leave_clamping_to_readers() {
        for (decl, default) in [
            (&MARGIN_ROW_BETA_ITERS, 1),
            (&MARGIN_ROW_BI_CHUNK, 256),
            (&MARGIN_ROW_BI_TOPK, 1024),
            (&MARGIN_ROW_EPOCH_PARENT_TOPK, 0),
            (&MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP, 2_000_000),
            (&MARGIN_ROW_FROZEN_DUAL_REFINE_STEPS, 0),
        ] {
            assert_eq!(
                resolve(decl, None),
                (LeverValue::U64(default), Source::Default),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some(" 12 ")),
                (LeverValue::U64(12), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some("bad")).1,
                Source::LegacyEnvRejected,
                "{}",
                decl.name
            );
            // An explicit 0 RESOLVES. Existing positive-only tuning readers
            // clamp it to 1; the epoch-parent master intentionally keeps zero
            // as OFF. In both cases it must remain distinguishable from an
            // absent/default value in the receipt.
            assert_eq!(
                resolve(decl, Some("0")),
                (LeverValue::U64(0), Source::LegacyEnv),
                "{}",
                decl.name
            );
        }
    }

    #[test]
    fn epoch_parent_topk_layers_env_over_typed_preset_over_dark() {
        let layered = |raw: Option<&str>, preset: Option<u64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_EPOCH_PARENT_TOPK,
                move |_| owned,
                preset.map(LeverValue::U64),
            )
            .expect("u64 config is admissible for UsizeTrimmed")
        };

        let from_preset = layered(None, Some(16));
        assert_eq!(from_preset.value, LeverValue::U64(16));
        assert_eq!(from_preset.source, Source::Config);
        assert_eq!(layered(None, None).value, LeverValue::U64(0));

        let env_wins = layered(Some("32"), Some(16));
        assert_eq!(env_wins.value, LeverValue::U64(32));
        assert_eq!(env_wins.source, Source::LegacyEnv);
        let killed = layered(Some("0"), Some(16));
        assert_eq!(killed.value, LeverValue::U64(0));
        assert_eq!(killed.source, Source::LegacyEnv);

        let rejected = layered(Some("bad"), Some(16));
        assert_eq!(rejected.value, LeverValue::U64(0));
        assert_eq!(rejected.source, Source::LegacyEnvRejected);
    }

    /// The `#epoch-bab` integers are UNTRIMMED, and that is load-bearing.
    ///
    /// Every one of the five historical input contracts used
    /// `var(..).ok().and_then(|v| v.parse().ok()).unwrap_or(D)` —
    /// `parse::<usize>()` with no `trim()`. The generic frozen-snapshot parser
    /// and compatibility `epoch_over_preset` path preserve that grammar.
    /// Declaring them `UsizeTrimmed`, which is what the rest of this module
    /// uses, would silently start accepting `" 4"`: a token the lever has
    /// always REJECTED. That is a behaviour change wearing a migration's
    /// clothes, so it is pinned here rather than left to review.
    #[test]
    fn epoch_integer_knobs_are_untrimmed_and_keep_their_historical_defaults() {
        for (decl, default) in [
            (&EPOCH_DEPTH, 0),
            (&EPOCH_ATTEMPTS, 2),
            (&EPOCH_TIER0, 0),
            (&EPOCH_UNIVERSE, 64),
            (&EPOCH_RETAIN, 128),
        ] {
            assert_eq!(
                resolve(decl, None),
                (LeverValue::U64(default), Source::Default),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some("12")),
                (LeverValue::U64(12), Source::LegacyEnv),
                "{}",
                decl.name
            );
            // An explicit 0 RESOLVES and stays distinguishable from absence:
            // for the cadence and the Tier-0 width it is the dark arm, and for
            // NY_EPOCH_ATTEMPTS it is a second kill switch.
            assert_eq!(
                resolve(decl, Some("0")),
                (LeverValue::U64(0), Source::LegacyEnv),
                "{}",
                decl.name
            );
            for reject in [" 12", "12 ", "bad", "", "-1", "1.5"] {
                let (value, source) = resolve(decl, Some(reject));
                assert_eq!(
                    value,
                    LeverValue::U64(default),
                    "{} {reject:?} must land on the historical default",
                    decl.name
                );
                assert_eq!(
                    source,
                    Source::LegacyEnvRejected,
                    "{} {reject:?}",
                    decl.name
                );
            }
        }
    }

    /// Every promoted epoch integer's frozen snapshot layers env over the
    /// generic typed-preset projection over the historical default. The
    /// compatibility `epoch_over_preset` path implements the same exact
    /// three-way rule; this declaration-level test pins the shared contract.
    #[test]
    fn epoch_integers_layer_env_over_typed_preset_over_history() {
        for (decl, default) in [
            (&EPOCH_DEPTH, 0_u64),
            (&EPOCH_ATTEMPTS, 2),
            (&EPOCH_TIER0, 0),
            (&EPOCH_UNIVERSE, 64),
            (&EPOCH_RETAIN, 128),
        ] {
            let layered = |raw: Option<&str>, preset: Option<u64>| {
                let owned = raw.map(str::to_owned);
                read_over_config_with(decl, move |_| owned, preset.map(LeverValue::U64))
                    .expect("u64 config is admissible for a U64 declaration")
            };

            // True absence falls through to the preset, then to history.
            let from_preset = layered(None, Some(3));
            assert_eq!(from_preset.value, LeverValue::U64(3), "{}", decl.name);
            assert_eq!(from_preset.source, Source::Config, "{}", decl.name);
            assert_eq!(
                layered(None, None).value,
                LeverValue::U64(default),
                "{}",
                decl.name
            );

            // A present admissible value wins over the preset, in BOTH
            // directions — including the explicit 0 that disarms it.
            let env_wins = layered(Some("7"), Some(3));
            assert_eq!(env_wins.value, LeverValue::U64(7), "{}", decl.name);
            assert_eq!(env_wins.source, Source::LegacyEnv, "{}", decl.name);
            let killed = layered(Some("0"), Some(3));
            assert_eq!(killed.value, LeverValue::U64(0), "{}", decl.name);
            assert_eq!(killed.source, Source::LegacyEnv, "{}", decl.name);

            // A present malformed value suppresses the preset and restores the
            // historical default. A typo is a kill switch, never a promotion.
            let rejected = layered(Some("malformed"), Some(3));
            assert_eq!(rejected.value, LeverValue::U64(default), "{}", decl.name);
            assert_eq!(rejected.source, Source::LegacyEnvRejected, "{}", decl.name);
        }
    }

    /// `NY_EPOCH_DEBUG` is PRESENCE-armed through `var(..).is_ok()`, so `"0"`
    /// and `""` ARM it. Declared `Text`/`Unset` for the `NY_ACASXU_PROF`
    /// reason: a `Bool` declaration would record `false` in the receipt for a
    /// run the reader treated as armed, and a receipt that contradicts the run
    /// is worse than no receipt.
    #[test]
    fn epoch_debug_is_presence_armed_not_an_exact_one_gate() {
        assert_eq!(
            resolve(&EPOCH_DEBUG, None),
            (LeverValue::Unset, Source::Default)
        );
        for armed in ["1", "0", "", "true", "no"] {
            let (value, source) = resolve(&EPOCH_DEBUG, Some(armed));
            assert_eq!(
                value,
                LeverValue::Text(armed.to_owned()),
                "{armed:?} is PRESENT and therefore arms the trace"
            );
            assert_eq!(source, Source::LegacyEnv, "{armed:?}");
        }
    }

    /// `trim().parse::<f64>()` with `is_finite()` and a CLOSED lower bound.
    #[test]
    fn trimmed_float_knobs_preserve_their_filters() {
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, None),
            (LeverValue::F64(20.0), Source::Default)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, Some(" 40 ")),
            (LeverValue::F64(40.0), Source::LegacyEnv)
        );
        // Zero is a real budget, not a rejection: the phase simply runs no chunk.
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, Some("0")),
            (LeverValue::F64(0.0), Source::LegacyEnv)
        );
        for reject in ["-1", "nan", "inf", "lots"] {
            let (value, source) = resolve(&MARGIN_ROW_BI_SECS, Some(reject));
            assert_eq!(value, LeverValue::F64(20.0), "{reject:?}");
            assert_eq!(source, Source::LegacyEnvRejected, "{reject:?}");
        }

        // eta keeps its `> 0.0` half AT THE READER, so `0` resolves here.
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, None),
            (LeverValue::F64(0.5), Source::Default)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some(" 2.0 ")),
            (LeverValue::F64(2.0), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some("0")),
            (LeverValue::F64(0.0), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some("-0.5")).0,
            LeverValue::F64(0.5)
        );
    }

    /// The root compute gate predates the exact-`1` registry convention. Its
    /// broader grammar is part of the public research interface, while typed
    /// preset delivery remains a Boolean fallback that env can kill in both
    /// directions.
    #[test]
    fn root_f32_truthy_parser_and_layering_preserve_the_legacy_contract() {
        for raw in ["1", "true", "on", " 1 ", "\ttrue\n"] {
            assert_eq!(
                resolve(&MARGIN_ROW_ROOT_F32, Some(raw)),
                (LeverValue::Bool(true), Source::LegacyEnv),
                "{raw:?}"
            );
        }
        for raw in ["0", "false", "off", "bad", "", "01", "TRUE"] {
            assert_eq!(
                resolve(&MARGIN_ROW_ROOT_F32, Some(raw)),
                (LeverValue::Bool(false), Source::LegacyEnv),
                "{raw:?}"
            );
        }

        let configured =
            read_over_config_with(&MARGIN_ROW_ROOT_F32, |_| None, Some(LeverValue::Bool(true)))
                .unwrap();
        assert_eq!(configured.value, LeverValue::Bool(true));
        assert_eq!(configured.source, Source::Config);

        let killed = read_over_config_with(
            &MARGIN_ROW_ROOT_F32,
            |_| Some("not-on".into()),
            Some(LeverValue::Bool(true)),
        )
        .unwrap();
        assert_eq!(killed.value, LeverValue::Bool(false));
        assert_eq!(killed.source, Source::LegacyEnv);
    }

    /// Only the measured July throughput pair and the measured early-publication
    /// policy may ship as `DefaultOn`. Other declarations remain `Debug`; that
    /// bucket does not erase measurements or imply that every typed scored
    /// preset inherits the declaration default.
    #[test]
    fn production_defaults_distinguish_measurement_from_explicit_user_direction() {
        for decl in MARGIN_ROW_LEVERS.decls() {
            let is_production_default = matches!(
                decl.name,
                "NY_MARGIN_ROW_LRU" | "NY_MARGIN_ROW_PARALLEL" | "NY_MARGIN_ROW_EARLY_EXIT"
            );
            let is_owner_directed = matches!(
                decl.name,
                "NY_MARGIN_ROW_HEAD_GROUPS" | "NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS"
            );
            assert_eq!(
                decl.bucket,
                if is_production_default || is_owner_directed {
                    Bucket::DefaultOn
                } else {
                    Bucket::Debug
                },
                "{}",
                decl.name
            );
            // Documented exceptions, both structurally discard-only and both
            // arguing MoatRisk::Low in their own doc:
            // * the GPU-shadow PROFILER is plumbed ONLY through
            //   `shadow_observe` (the verdict path passes `profile=false`);
            // * the GPU-shadow SITE SELECTOR only narrows which sites that
            //   same discard-only shadow observes — it cannot arm the shadow,
            //   cannot reach verdict authority (which needs
            //   NY_MARGIN_ROW_GPU_BACKWARD plus a trusted code constant), and
            //   its DEFAULT strictly reduces v8's device traffic (four sites
            //   -> root alone).
            // Neither can move a published bound.
            //
            // NY_MARGIN_ROW_BRANCH_REDUCE_MAX joins them for a DIFFERENT
            // reason worth spelling out, because it is the first non-probe
            // exception: it is RANKER-ONLY. It selects which candidate index
            // the pick loop splits on. Every child bound is still produced by
            // the same certified pass, `closed()` is untouched, and the
            // monotone floor `cb = raw.max(b)` still applies, so a bad choice
            // reaches a different set of domains inside the budget — a POINTS
            // risk — and cannot manufacture a proof. That is exactly the
            // Low/High line: High is "can move a published bound", not "can
            // change the answer".
            //
            // The three Tier-0 knobs (#epoch-bab) join for exactly the
            // NY_MARGIN_ROW_BRANCH_REDUCE_MAX reason, one step earlier in the
            // same pipeline: NY_EPOCH_TIER0 decides which candidates are
            // OFFERED to `score_candidates`, NY_EPOCH_UNIVERSE how many are
            // ranked, and NY_EPOCH_RETAIN how many f32 rows the ranker gets.
            // The retained rows are ranker-only by construction (`RetainedLayer`:
            // "No Outward (verdict-grade) pass ever reads them") and retention
            // is pure capture in `root.rs:build_walk` — it copies out of
            // `mi`/`di` AFTER `concretize_box` and touches none of
            // `l`/`u`/`alpha`/`s`/`c`/`ms`/`unst`. Arming them spends wall
            // clock and reaches a different set of domains; it cannot move a
            // published bound.
            //
            // NY_EPOCH_DEBUG is the ordinary case: print-only.
            //
            // NY_EPOCH_DEPTH and NY_EPOCH_ATTEMPTS are NOT here. They rebuild
            // every intermediate under the split constraints and the rebuilt
            // frame IS published (measured: -0.371373 -> -0.316766 on
            // cifar100 idx_4169). Those are High.
            // Everything else stays High.
            let is_low_risk = matches!(
                decl.name,
                "NY_MARGIN_ROW_LRU"
                    | "NY_MARGIN_ROW_PARALLEL"
                    | "NY_MARGIN_ROW_GPU_PROF"
                    | "NY_MARGIN_ROW_GPU_SHADOW_SITES"
                    | "NY_MARGIN_ROW_K_HEAD"
                    | "NY_MARGIN_ROW_K_TRUNK"
                    | "NY_MARGIN_ROW_BRANCH_REDUCE_MAX"
                    | "NY_MARGIN_ROW_PARALLEL_PREBUILD"
                    | "NY_MARGIN_ROW_RESERVE_MAX_FRAC"
                    | "NY_MARGIN_ROW_CONCURRENT"
                    | "NY_MARGIN_ROW_EARLY_EXIT"
                    | "NY_VERIFIER_THREADS"
                    | "NY_MARGIN_ROW_CONCURRENT_THREADS"
                    | "NY_EPOCH_TIER0"
                    | "NY_EPOCH_UNIVERSE"
                    | "NY_EPOCH_RETAIN"
                    | "NY_EPOCH_DEBUG"
            );
            assert_eq!(
                decl.moat,
                if is_low_risk {
                    MoatRisk::Low
                } else {
                    MoatRisk::High
                },
                "{}",
                decl.name
            );
            // Debug-bucket declarations may carry retained scored evidence:
            // the branch-width pair, BI master/dose and root-precision pair
            // are category-scoped shipped choices. The concurrent scheduler's
            // retained result also changed root precision and is deliberately
            // not on this list until an isolated current-binary A/B exists.
            // Debug is a search/governance bucket, not a synonym for Unmeasured.
            let is_measured_debug = matches!(
                decl.name,
                "NY_MARGIN_ROW_K_HEAD"
                    | "NY_MARGIN_ROW_K_TRUNK"
                    | "NY_MARGIN_ROW_BACKWARD_INTERM"
                    | "NY_MARGIN_ROW_BI_SECS"
                    | "NY_MARGIN_ROW_ROOT_F32"
                    | "NY_MARGIN_ROW_TABLEAU_F32"
                    // Measured 2026-08-29 (docs/CLIP_STATEFUL_AB_2026-08-29.md):
                    // 39 runs, NEGATIVE — bit-identical to the stateless arm on
                    // idx_4169 while gen>=2 proved 8 real inheritance events. It
                    // stays Debug because a measured-negative lever is still a
                    // search axis; it is no longer Unmeasured.
                    | "NY_MARGIN_ROW_CLIP_STATEFUL"
                    // Measured on TWO categories, with OPPOSITE outcomes, which is
                    // why it stays Debug rather than becoming a production default:
                    //  * cifar100 (81d30f6b4, 2026-08-30): ARMED by
                    //    `configs/vnncomp25/cifar100_2024.yaml`. 171 rows, engagement
                    //    124/124, converts idx_8739_sidx_1540 (timeout -> unsat),
                    //    moat 47/47 unsat ZERO breaks, wall cost ZERO.
                    //  * tinyimagenet (2026-09-01): NEGATIVE, and NOT armed. Engaged
                    //    via the yaml route with zero `NY_*` set (17 `[branch-parent-lift]`
                    //    events over 2 rows, `preset=tinyimagenet_2024.yaml`), so this is
                    //    a real negative and not a VOID: mean gain 0.0439 against a worst
                    //    domain floor of -0.5135, i.e. a 12x SHORTFALL, and `closes=0` on
                    //    every event. The payoff is DOMAIN ELIMINATION, so a lift that
                    //    cannot bridge the remaining gap eliminates nothing.
                    | "NY_MARGIN_ROW_PARENT_LIFT"
            );
            if is_production_default || is_measured_debug {
                assert!(
                    matches!(decl.provenance, Provenance::Measured { .. }),
                    "{}",
                    decl.name
                );
            } else if is_owner_directed {
                let Provenance::OwnerDirected {
                    directive,
                    limitations,
                } = decl.provenance
                else {
                    panic!(
                        "{}: explicit direction must not masquerade as measured evidence",
                        decl.name
                    );
                };
                assert_ne!(directive.trim(), "");
                assert_ne!(limitations.trim(), "");
            } else {
                assert!(
                    matches!(decl.provenance, Provenance::Unmeasured { .. }),
                    "{}",
                    decl.name
                );
            }
            // The two OPT-OUTS ship `Bool(true)` because their reader tests
            // `!= Some("0")`; they are sub-switches of the default-dark
            // `NY_MARGIN_ROW_BETA`, so their defaults arm nothing on a scored
            // run. Early publication is the separate measured production
            // default pinned above.
            let is_beta_opt_out = matches!(
                decl.name,
                "NY_MARGIN_ROW_BETA_POLYAK" | "NY_MARGIN_ROW_BETA_HEADS"
            );
            assert!(
                is_beta_opt_out
                    || is_owner_directed
                    || decl.name == "NY_MARGIN_ROW_EARLY_EXIT"
                    || !matches!(decl.default, DefaultSpec::Bool(true)),
                "{}: no gate in this group ships armed",
                decl.name
            );
        }
    }
}
