// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified f32 GPU lane backward — EFT/double-single (#margin-row-gpu-eft).
//!
//! State (implementation session 2, v4): the M1 on-device self-check AND the
//! L-lane device transaction ([`transaction`]) are REAL — but verdict
//! authority is M3-GATED: [`run_pass_certified`] refuses
//! [`Refusal::ShadowOnly`] even when armed (after a passing qualification),
//! and the transaction runs only under the M2 SHADOW protocol
//! (`NY_MARGIN_ROW_GPU_SHADOW=1`: compute on device, compare against the CPU
//! pass, DISCARD — [`shadow_compare`]). This module therefore still cannot
//! touch a bound or a verdict. Full design, failure diagnosis of the
//! predecessor seam, and the moat protocol:
//! `docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md`.
//!
//! # Why this exists when `gpu_seam` already does (the non-repeat contract)
//!
//! The existing seam (`NY_MARGIN_ROW_GPU=1`, `gpu_seam.rs`) is measured
//! VERDICT-BREAKING: it turns banked-unsat `idx_6659` into `unknown`
//! regardless of sweep budget. Three diagnosed defects, each a hard design
//! requirement here (design section 2/3):
//!
//! 1. **Precision (the budget-independent killer)**: the device publishes an
//!    A-PRIORI f32 Higham radius (`gamma_k * sum|A||W|`), measured
//!    0.088–0.126 absolute vs ~1e-5 actual on the cifar100 fold — above the
//!    lane's own closing margins (~1e-2..1e-1), so every child bound loosens
//!    below zero and the tree stops closing at ANY budget. ⇒ R1: certified
//!    error is A-POSTERIORI (measured EFT residual) wherever the accumulation
//!    permits, and the value path is double-single (f64-comparable). Sound
//!    both ways; only this direction keeps the proofs.
//! 2. **Shared-fate device state**: the seam installs the caller's deadline
//!    on the PROCESS-GLOBAL backend (`gpu_seam.rs` dispatch,
//!    `GpuCrownBackendDeadlineScope::set`) while the internal verifier's
//!    sweep has fallible deadline checks AFTER queue submission
//!    (`intermediate_sweep.rs`), whose early return permanently poisons the
//!    device memory ledger ("exited after submission without a final drain").
//!    ⇒ R2/R3: submit→drain is one unconditional critical section; deadlines
//!    are host-side only; failure marks a LANE-LOCAL latch
//!    ([`mark_channel_dead`]) and nothing else.
//! 3. **Serialized-device stalls**: refusal paths must not block on the GPU
//!    mutex the sweep holds for seconds. ⇒ R4: admission is non-blocking;
//!    a busy device is [`Refusal::Busy`] and the caller runs the CPU pass.
//!
//! # Fail-closed rules
//!
//! * DARK unless its governed environment/preset resolution is armed (default
//!   OFF; an environment override still requires exact `"1"`;
//!   renamed from `NY_MARGIN_ROW_GPU_EFT` per the implementation-session
//!   brief — the OLD seam's `NY_MARGIN_ROW_GPU` is a different lever and
//!   stays untouched). Typed delivery exists for controlled scored A/Bs, but
//!   no shipped preset arms it and verdict authority remains post-M3.
//! * Any `Err` from any entry means: run the untouched CPU pass, bit-for-bit
//!   (the `engine::run_seamed` contract). No refusal repairs anything.
//! * Verdict authority is gated on the moat protocol (design section 7) AND
//!   enforced by the type system at runtime (adversarial-review item 2 — the
//!   M1 bit gate is BLOCKING, not advisory): [`run_transaction`], the only
//!   `PassOut` producer in this module, requires an
//!   [`authority::VerdictAuthority`], whose sole constructor consumes an
//!   [`authority::DeviceParityProof`] — mintable only by the once-per-process
//!   ON-DEVICE bit-parity self-check (which also requires the resolved
//!   DenormPreserve channel ON, item 3). On top of that: M2 shadow enclosure
//!   on the moat rows x3, M3 same-binary A/B, and the ported error-floor and
//!   realization-probe guards at every admitted pass.
//!
//! # Engagement telemetry (rule R9)
//!
//! Arming prints `[gpu-backward] armed ...` once per process; the M1
//! self-check prints its parity state + device_ms once. A measurement without
//! the armed line is not a measurement of this lever. The banner embeds
//! [`PROVENANCE_MARKER`] for the R8 binary-provenance check
//! (`strings <binary> | grep margin-row-gpu-eft`).
//!
//! PER SITE (v9, FIX D4): every distinct `(site, refusal reason)` pair prints
//! once and is COUNTED — the reason is [`Refusal::label`], so an
//! [`Refusal::Unmappable`] keeps its own string instead of collapsing into one
//! tag; each site's OWN first shadow observation always prints a `site=` line
//! (then every 32nd); and [`site_census_line`] renders the whole per-site
//! census (ok / trip / refusal counts) every 64 shadow observations, on every
//! TRIP, and at channel death. The M2 shadow banner names the SELECTED sites
//! (`NY_MARGIN_ROW_GPU_SHADOW_SITES`, [`shadow_sites_from`]; unset = `root`
//! alone, the impl-v7 shape) — a measurement with no `site=<the site you
//! selected>` line is not a measurement of that site.

// The lane is wired into `engine::run_seamed` (armed runs get the once-per-
// process M1 self-check + engagement telemetry; the verdict path then refuses
// `ShadowOnly` into the unchanged CPU walk, and the M2 shadow entry runs the
// real transaction compare-and-discard). A few items stay staged for the M3
// authority session — drop this allow when the constant flips.
#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Instant;

use super::engine::{BackwardEngine, Collect, DomainGates, Exceptions, LaneDir, PassOut, Seed};
use super::rounding::next_up;

use authority::VerdictAuthority;

pub(crate) mod authority;
pub(crate) mod cache;
pub(crate) mod device;
pub(crate) mod ds;
pub(crate) mod probe;
pub(crate) mod transaction;

#[cfg(test)]
mod tests;

/// R8 binary-provenance marker: lands in `.rodata` via the armed banner.
/// Bump the suffix on every landing that changes behavior.
/// v2: adversarial-review resolution — authority chain (M1 made blocking),
/// denorm-preserve admission, gate-transform intercept pair.
/// v3: implementation session — the once-per-process M1 self-check is REAL
/// (ds-primitive + gate-transform probe lanes dispatched on the lane's
/// dedicated device, bit-compared against the host twin); gate renamed to
/// `NY_MARGIN_ROW_GPU_BACKWARD`; telemetry prefix `[gpu-backward]`.
/// v4: implementation session 2 — the L-lane device transaction is REAL
/// (head gemm + gate transform + bias reduce + conv gather + residual merges,
/// one submit, unconditional drain, host directed-rounding egress, floor +
/// probe guards), reachable ONLY through the M2 shadow protocol
/// (`NY_MARGIN_ROW_GPU_SHADOW=1`: compute, compare against the CPU pass,
/// DISCARD); the armed verdict path refuses `ShadowOnly` until the M3 A/B.
/// v5: error-charge tightening (GPU session 3) — weights/biases ride as
/// exact-split ds PAIRS of the engine's f64 parameters (the u32 downcast
/// ball, the measured dominant radius term, is gone), the a-priori dot
/// envelope moves to documented per-tap bounds (`gamma_ds_dot`, 10u^2/tap),
/// the bias-reduce double fold charge is removed, and the first pass prints a
/// once-per-process charge profile. Fixture radii 3.3e-6 -> 5.2e-12.
/// v6: verify-fix (session-3 adversarial verify) — the exact-residual
/// split-floor channel: the DS_SPLIT_REL relative split ball is
/// MEASURED-FALSE in the f32-subnormal parameter band (component rounding is
/// absolute there), so every split site (head/conv weights, per-entry
/// biases, gate quads) now also charges the EXACT per-element residual
/// excess host-side, carried through tracked operator norms into absolute
/// `split_e`/`split_eb` egress floors (0.0 for every normal-band parameter;
/// counterexample-pinned). Doc traps fixed with it: ds_mul's `5u^2` citation
/// (wrong algorithm — implemented DWTimesDW2 is `6u^2`), the `Sterbenz`
/// split-exactness wording (representability/29-low-bits + the subnormal
/// exception).
/// v7: perf session (B1+B2+B3, semantics-neutral) — B1 plan/weight RESIDENCY
/// cache + executor arena pool (`cache::PlanStatic`/`PlanCache`: ds-pair
/// planes split + uploaded once per RootGates instance, charges computed
/// once by the same helpers — host charges BIT-identical, pinned; per-dom
/// gate admission re-run on every hit; fix A3: the realization probe's
/// midpoint forward cached in the PlanStatic); B2 tree bias-reduce
/// (`bias_reduce_wg`: 256-lane workgroup per row, serial per-lane blocks +
/// fixed binary tree; `gamma_ds_dot(n+2)` kept — dominates any summation
/// order, argument at `push_reduce`; host twin mirrors the TREE op-for-op);
/// B3 conv weight staging (workgroup-memory tiles of the shared `ci` kernel
/// slice, per-element arithmetic ORDER unchanged — memory access only, so
/// device-vs-host bit identity is preserved and re-pinned). Plus
/// `NY_MARGIN_ROW_GPU_PROF=1` (one step per submit, shadow-only) and the
/// `build_ms=/upload_ms=/dispatch_ms=/plan_cache=` shadow telemetry.
/// v8: B4+B5+B6 seams (throughput session) — the certified lane is wired
/// UNDER the margin-row prover's two hot calls and under the y-pack refresh,
/// each as its OWN site with its OWN authority bit, every new site defaulting
/// to SHADOW: B4 `run_pass_certified_exc` (batched-children scoring,
/// bab.rs:4193) with a per-relu gate EXCEPTION table
/// (`gate_transform_exc` + `transaction::exception_records`) and FIX A1 —
/// the override magnitudes are folded into the layer's host split-floor
/// stats, counterexample-pinned; B5 `run_pass_certified_collect`
/// (`eval_node`, bab.rs:2828) admitting ONLY
/// `Collect{unst_abs:true, rows:None, unst_rows:false}` with the new
/// heuristic-only `collect_abs` kernel; B6 the y-pack refresh
/// (`engine::y_rows_seamed`) as two certified transactions BEFORE the dark
/// old seam. Plus FIX A6: the per-site shadow compares the FULL CONSUMED
/// surface (`consumed_surface_overlap` — every concretized column, which the
/// aggregate `b_eval` and the col-retire per-column directs are both
/// pass-independent functions of), and the B1 plan cache handoff is real
/// (`LaneCtx::plan_cache`, owned by the BaB struct).
/// v9: HARDENING session — the seven defects the v8 adversarial verify
/// raised, all of which block or corrupt the shadow measurement phase (none
/// soundness-blocking):
/// D5a the PER-SITE SHADOW SELECTOR (`NY_MARGIN_ROW_GPU_SHADOW_SITES`, a
/// declared lever): UNSET now means ROOT ONLY — the one-pass-per-instance
/// impl-v7 shape the M2 protocol was designed for — and the phase names the
/// site it is measuring (`eval`, then `score`), instead of v8's one lever
/// arming all four sites at ~2 device passes per expansion;
/// D1 the exception run-length merge key compares SOURCE f64 quads on both
/// sides (v8 compared source bits against the RECONSTRUCTED ds-pair value,
/// equal only for f32-exact quads, so every FRACTIONAL override silently
/// stopped compressing and crossed `MAX_EXC_RECORDS` into a CPU fallback —
/// an armed-reading inert seam);
/// D4 PER-SITE R9 accounting (per-site ok/trip counters, per-site refusal
/// counts keyed on the `Unmappable` STRING, and a census line) — v8's single
/// process-wide counter could starve a low-frequency site of `site=` lines
/// and its first `Unmappable` masked all the others;
/// D2 the per-pass ARENA BYTE BUDGET (`MAX_PASS_ARENA_BYTES`, refuse -> CPU),
/// the design's risk-3 mitigation, previously unimplemented;
/// D3 the resident-plane registry is RELEASABLE and the planes own their ids
/// (v8 gave every `MarginRowBab` its own plan cache, turning an append-only
/// registry into a per-BaB-instance growth term);
/// D6 the duplicate-exception check is per `(neuron, row)` ELEMENT, not per
/// row (v8 refused a legal multi-neuron row pair);
/// D7 the bit-cast-SUBNORMAL exception header's dependency on the
/// DenormPreserve admission is documented at both ends and pinned.
///
/// `pub` because `ny --build-id` compiles it into the binary's canonical
/// identity (#binary-identity): the R8 provenance check was a manual
/// `strings | grep` for exactly this marker, and a manual check is what kept
/// failing.
pub const PROVENANCE_MARKER: &str = "margin-row-gpu-eft-impl-v9";

/// M3 authority switch (design §7): a TRUSTED CODE CONSTANT, deliberately not
/// an env var (the `margin_row_bab_enabled` discipline — no untrusted process
/// may flip verdict authority). It stays `false` until the same-binary M3 A/B
/// of design §7 passes on the moat rows x3; flipping it is that future
/// session's one-line, human-reviewed change. While `false`, an armed
/// non-shadow pass refuses [`Refusal::ShadowOnly`] AFTER the M1 device
/// qualification — the transaction is reachable only via the shadow entry.
///
/// SCOPE (v8): this constant governs the ROOT site alone
/// ([`SITE_ROOT`]) — the site the M2 shadow x3 and the M3 A/B were run on.
/// The B4/B5/B6 sites each carry their OWN bit in [`SITE_AUTHORITY_GRANTED`]
/// and are NOT swept along by an M3 flip (pinned by
/// `m3_flip_does_not_authorize_the_new_sites`).
/// REVERTED TO FALSE 2026-08-22 by adversarial audit, BEFORE it ever ran armed.
///
/// It was flipped true on the M2 x15 evidence, and the flip is NOT believed
/// unsound (the root site's error enters concretize as additive slack and
/// `closed()` is strict `b > 0.0`, so a looser device bound costs proofs and
/// cannot manufacture one). It was reverted because the evidence does not
/// support it and reverting costs ZERO points:
/// * the M2 protocol has no power to detect the failure it exists to detect —
///   all 3 GPU-exercised rows are banked UNSAT, where a wrongly-TIGHT device
///   bound still yields the correct verdict, and the 6 sat sentinels never
///   entered the lane at all (0 gpu-backward lines; they are falsified upfront);
/// * the authority configuration has NEVER been executed: the M3 A/B run on
///   2026-08-22 reported gpu_lines=0 on 9/9 runs because its harness never
///   exported NY_MARGIN_ROW_GPU_BACKWARD — it measured the dark CPU lane and
///   would have read as a pass (wrong-lane trap #11);
/// * the original telemetry lied under authority: it printed "ShadowOnly"
///   unconditionally, while `shadow_compare_site` returns before its banner at
///   an Authorized site. [`authority_sites_label`] now derives the armed banner
///   from the same effective mask as [`site_mode`], so a future reviewed flip
///   is visible rather than being misreported. Keep that observable intact;
/// * the typed delivery route still cannot grant verdict authority: no shipped
///   preset arms it, and the compile-time authority bit below remains false, so
///   nothing was gained by leaving it true.
///
/// RE-FLIP REQUIRES: an armed run that actually emits the lane's telemetry, a
/// protocol containing rows where a wrong bound is DETECTABLE (near-threshold
/// and/or sat rows that enter the lane), and honest observables.
const M3_AUTHORITY_GRANTED: bool = false;

// ---------------------------------------------------------------------------
// PER-SITE VERDICT AUTHORITY (design §3, staged flips — one lever per cycle)
// ---------------------------------------------------------------------------

/// The root direct pass (`bab.rs:920` -> `engine::run_seamed`), once per
/// instance. Authorized by [`M3_AUTHORITY_GRANTED`], not by the bitmask.
pub(crate) const SITE_ROOT: u32 = 1 << 0;
/// B6: the 2 x n_y y-pack refresh (`engine::y_rows_seamed`, bab.rs:616).
pub(crate) const SITE_Y_REFRESH: u32 = 1 << 1;
/// B5: the `eval_node` main certified pass (bab.rs:2828), ~600 ms measured.
pub(crate) const SITE_EVAL: u32 = 1 << 2;
/// B4: the batched-children scoring pass (bab.rs:4193), ~554 ms measured.
pub(crate) const SITE_SCORE: u32 = 1 << 3;
/// Every declared site (mask sanity + telemetry).
pub(crate) const SITE_ALL: u32 = SITE_ROOT | SITE_Y_REFRESH | SITE_EVAL | SITE_SCORE;

/// Per-site verdict authority for the NEW seams — a TRUSTED CODE CONSTANT,
/// same discipline as [`M3_AUTHORITY_GRANTED`] (no env var may flip verdict
/// authority; `ny_levers` cannot fire in competition anyway).
///
/// `0` = EVERY new site is SHADOW: it computes the device pass at its own
/// call site, enclosure-compares it against the CPU pass it shadows AT THAT
/// EXACT CALL SITE, and DISCARDS. Each site's own x3 moat then earns its own
/// one-line, human-reviewed bit — never a shared flip, because the sites
/// differ in what consumes them (the design's "do not let B4 ride any other
/// step's flip"):
///
/// * [`SITE_SCORE`] closes children without re-proving (`cb = raw.max(b)`,
///   bab.rs:1612, consumed by `closed(mode, b)` at bab.rs:1555) and is the
///   only site with a new EXCEPTION path (FIX A1);
/// * [`SITE_EVAL`] feeds BOTH the aggregate node bound and the col-retire
///   per-column directs (FIX A6);
/// * [`SITE_Y_REFRESH`] feeds every downstream via-y composition.
const SITE_AUTHORITY_GRANTED: u32 = 0;

/// What a site is allowed to do on this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SiteMode {
    /// The lane never runs here (gate disarmed, or armed with no shadow).
    Off,
    /// Compute, enclosure-compare against the CPU pass at this call site,
    /// DISCARD. The caller returns the CPU pass bit-for-bit.
    Shadow,
    /// The device pass may BE the pass (verdict-bearing).
    Authorized,
}

/// The effective authority mask: the bitmask for the new sites, with
/// [`SITE_ROOT`] driven by [`M3_AUTHORITY_GRANTED`] alone. Keeping root out
/// of the bitmask is what makes the two flips independent in BOTH
/// directions — an M3 flip cannot authorize eval/score, and an eval/score
/// bit cannot authorize root.
pub(crate) const fn effective_site_mask(granted: u32, m3: bool) -> u32 {
    let base = granted & SITE_ALL & !SITE_ROOT;
    if m3 {
        base | SITE_ROOT
    } else {
        base
    }
}

/// The PURE spec of the per-site gate, unit-tested without a device.
///
/// ARMING SEMANTICS:
///
/// * `NY_MARGIN_ROW_GPU_SHADOW=1` arms the shadow LEVER, exactly as
///   `shadow_compare` did before v8 (it checked `shadow_enabled()` and
///   nothing else, and `shadow_observe` ran the full M1 -> transaction chain
///   from there);
/// * WHICH sites that lever shadows is the separate, declared per-site
///   selector [`shadow_sites`] (`NY_MARGIN_ROW_GPU_SHADOW_SITES`,
///   FIX D5a) — UNSET means `root` alone, i.e. exactly the ONE pass per
///   instance impl-v7 shadowed and the shape the M2 protocol was designed
///   for. impl-v8 shipped the shadow lever arming all FOUR sites at once
///   (~2 device passes per expansion plus 2 per y-pack LRU miss), which is
///   what makes "one lever per cycle" unmeasurable; the selector is how the
///   measurement phase gets SITE_EVAL alone, then SITE_SCORE alone;
/// * VERDICT authority additionally requires the lane's own gate
///   `NY_MARGIN_ROW_GPU_BACKWARD=1` AND this site's bit — so no untrusted
///   process can promote a site by setting one env var. The selector is a
///   SHADOW-only knob: it cannot grant, withhold or alter verdict authority
///   (pinned by `shadow_site_selector_cannot_touch_verdict_authority`);
/// * an authorized site does NOT also shadow (it already ran the device pass
///   through `run_pass_certified_site`; shadowing it would double the device
///   cost and compare a pass against itself).
pub(crate) fn site_mode_from(
    armed: bool,
    shadow: bool,
    shadow_sites: u32,
    mask: u32,
    site: u32,
) -> SiteMode {
    if armed && mask & site != 0 {
        return SiteMode::Authorized;
    }
    if shadow && shadow_sites & site != 0 {
        SiteMode::Shadow
    } else {
        SiteMode::Off
    }
}

/// This build's mode for `site`.
pub(crate) fn site_mode(site: u32) -> SiteMode {
    site_mode_from(
        enabled(),
        shadow_enabled(),
        shadow_sites(),
        effective_site_mask(SITE_AUTHORITY_GRANTED, M3_AUTHORITY_GRANTED),
        site,
    )
}

/// Ledger slot for a site: the four declared sites plus one catch-all so an
/// unrecognized bit can still be counted instead of being folded into a real
/// site's census (D4).
const SITE_SLOTS: usize = 5;

/// `site` -> its [`SITE_SLOTS`] index (total; unknown bits land in the
/// catch-all slot).
fn site_slot(site: u32) -> usize {
    match site {
        SITE_ROOT => 0,
        SITE_Y_REFRESH => 1,
        SITE_EVAL => 2,
        SITE_SCORE => 3,
        _ => 4,
    }
}

/// Stable short tag per ledger slot (never verdict input). This is also the
/// token vocabulary of [`MARGIN_ROW_GPU_SHADOW_SITES`][decl] — one spelling,
/// used by the selector, the telemetry and the census alike, so a log line and
/// the lever that produced it can never disagree.
///
/// [decl]: ny_levers::decls::margin_row::MARGIN_ROW_GPU_SHADOW_SITES
const SLOT_TAGS: [&str; SITE_SLOTS] = ["root", "y_refresh", "eval", "score", "unknown"];

/// Stable short tag for telemetry (never verdict input).
fn site_tag(site: u32) -> &'static str {
    SLOT_TAGS[site_slot(site)]
}

/// The declared sites, paired with their selector tokens.
const SITE_TOKENS: [(&str, u32); 4] = [
    ("root", SITE_ROOT),
    ("y_refresh", SITE_Y_REFRESH),
    ("eval", SITE_EVAL),
    ("score", SITE_SCORE),
];

/// Everything a seamed entry needs beyond the plain L-lane walk.
///
/// Deliberately a value type carried into `run_transaction`: the site's
/// SHAPE and its AUTHORITY are separate axes, so a shadow observation runs
/// the exact program its authorized twin would.
#[derive(Default, Clone, Copy)]
pub(crate) struct PassShape<'e> {
    /// B4 (verdict-bearing): per-(row, neuron) gate overrides.
    pub(crate) exc: Option<&'e Exceptions>,
    /// B5 (heuristic-only): emit the per-unstable |coefficient| collector.
    pub(crate) collect_unst_abs: bool,
}

/// Pre-registered M2/M3 KILL criterion (adversarial-review item 1): the
/// literal substring of the internal verifier sweep's post-submit poison log
/// (`ny-gpu/src/wgpu_device/ops/intermediate_sweep.rs:277`). The sweep checks
/// its OWN `request.deadline` AFTER queue submission (`intermediate_sweep.rs:845`,
/// `:865`; armed at `:803` via `CallLocalCrownDeadlineScope` — NOT the backend
/// lease the old seam installed), so ANY added GPU contention — including this
/// lane's dedicated-queue traffic — can push it past that deadline between
/// submit and drain, and one trip poisons the device memory ledger to
/// `usize::MAX` for the rest of the process. R3 alone therefore does NOT
/// remove the trigger. Until the implementation session applies R2 to the
/// sweep itself (hoist those checks before submission or drain
/// unconditionally — a REQUIRED precondition for M3, design section 9), the
/// moat harness MUST grep every M2 shadow / M3 A/B run log for this constant:
/// any hit = KILL, the measured section-2.2 failure recurring. `tests.rs`
/// pins it against the sweep source so a reworded log line cannot silently
/// blind the criterion.
pub(crate) const SWEEP_POST_SUBMIT_KILL_LINE: &str =
    "exited after submission without a final drain";

/// EFT/double-single primitives (fma TwoProduct, fma-barrier TwoSum, ds ops).
/// Concatenated before every consumer kernel at pipeline build. Host twin:
/// [`ds`] — the M1 moat gate is bit-identity between the two.
pub(crate) const DS_PRIMITIVES_WGSL: &str = include_str!("kernels/ds_primitives.wgsl");

/// ReLU gate backward transform (design section 5, kernel 2).
pub(crate) const GATE_TRANSFORM_WGSL: &str = include_str!("kernels/gate_transform.wgsl");

/// Transposed-conv gather backward (design section 5, kernel 3).
pub(crate) const CONV_BACKWARD_WGSL: &str = include_str!("kernels/conv_backward.wgsl");

/// Head-gemm backward — the margin seed fold (design section 5, kernel 4).
pub(crate) const HEAD_GEMM_WGSL: &str = include_str!("kernels/head_gemm.wgsl");

/// Per-row bias reduction into the running accumulators (design section 5,
/// the bias tree-reduce kernel; serial-per-row v1, deterministic).
pub(crate) const REDUCE_WGSL: &str = include_str!("kernels/bias_reduce.wgsl");

/// Residual-merge elementwise ds add (the `merge_into` twin).
pub(crate) const MERGE_ADD_WGSL: &str = include_str!("kernels/merge_add.wgsl");

/// B5: the per-unstable |coefficient| collector (HEURISTIC-ONLY — the kernel
/// header states why it carries no certified charge).
pub(crate) const COLLECT_ABS_WGSL: &str = include_str!("kernels/collect_abs.wgsl");

/// M1 parity probe kernel (design section 7 M1): the fixed ds-primitive
/// composition dispatched over the adversarial lane table by
/// [`probe::self_check_streams`]. Concatenated after [`DS_PRIMITIVES_WGSL`].
pub(crate) const DS_PROBE_WGSL: &str = include_str!("kernels/ds_probe.wgsl");

/// Why the certified GPU backward declined this pass. Every variant means the
/// caller MUST run the exact CPU backward; none is recoverable in-lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// `NY_MARGIN_ROW_GPU_BACKWARD` is not exactly `"1"`.
    Disabled,
    /// A staged capability was reached that has no implementation on this
    /// path (kept for API stability; the L-lane transaction itself is REAL
    /// as of v4 and refuses [`Refusal::ShadowOnly`] on the verdict path).
    Unimplemented,
    /// The transaction exists and the M1 qualification passed, but verdict
    /// authority is M3-gated ([`M3_AUTHORITY_GRANTED`] is `false`): per the
    /// moat protocol (design section 7) the device pass may only run under
    /// the M2 SHADOW entry — computed, compared against the CPU pass,
    /// DISCARDED. The caller runs the exact CPU pass.
    ShadowOnly,
    /// A prior device failure latched this lane's channel dead for the rest
    /// of the process ([`mark_channel_dead`]). Lane-local by design (R3):
    /// nothing shared is poisoned, and there is deliberately no reset.
    ChannelDead,
    /// The device (or its exclusive token) is held elsewhere. Non-blocking by
    /// design (R4): the caller runs the CPU pass instead of waiting.
    Busy,
    /// The pass is not in the certified-outward mode.
    NotOutward,
    /// Zero rows, or a seed width disagreement.
    Rows,
    /// The op list / gates / seed cannot be expressed at the kernel boundary.
    Unmappable(&'static str),
    /// The lane's deadline expired before submission, or the result returned
    /// after it (checked HOST-SIDE only — R3: never installed on a backend).
    Deadline,
    /// The device returned an error. Also latches [`mark_channel_dead`].
    Device,
    /// The returned payload failed the structural/finite check.
    Payload,
    /// Certified-error floor guard trip (model-ball floor; ported from
    /// `gpu_seam::error_floor_ok` per design section 7).
    ErrorFloor,
    /// Realization-probe guard trip: a published bound excluded a realized
    /// value of the functional — an unsoundness signal, never rate-limited.
    Probe,
    /// The NaN/Inf firewall rejected the converted pass.
    NonFinite,
}

impl Refusal {
    /// Stable label for the per-site refusal ledger and the once-per
    /// (site, reason) diagnostic. NOT verdict input.
    ///
    /// FIX D4 (hardening session, impl-v9): this REPLACES the old numeric
    /// `tag()`, which collapsed every [`Refusal::Unmappable`] to one tag —
    /// so the FIRST `Unmappable` seen anywhere in the process permanently
    /// masked every other one, including all four B4/B5 reasons
    /// (`"collection shape outside the B5 seam"`, `"too many exception
    /// records"`, `"pass exceeds the 65535x256 thread cap"`, `"two
    /// exceptions on one (neuron, row) element"`) and the new D2 budget
    /// refusal. Keying the ledger on the `Unmappable` STRING is what lets a
    /// shadow run prove WHY a site refused.
    fn label(self) -> &'static str {
        match self {
            Refusal::Disabled => "disabled",
            Refusal::Unimplemented => "unimplemented",
            Refusal::ChannelDead => "channel-dead",
            Refusal::Busy => "busy",
            Refusal::NotOutward => "not-outward",
            Refusal::Rows => "rows",
            Refusal::Unmappable(what) => what,
            Refusal::Deadline => "deadline",
            Refusal::Device => "device",
            Refusal::Payload => "payload",
            Refusal::ErrorFloor => "error-floor",
            Refusal::Probe => "probe",
            Refusal::NonFinite => "non-finite",
            Refusal::ShadowOnly => "shadow-only",
        }
    }
}

/// Pure legacy-environment arming predicate. The governed authority selector
/// now has a typed preset route; the shadow/profile research siblings still
/// consume this exact parser in production.
#[inline]
fn armed_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Is the certified GPU backward armed? The resolved environment/preset
/// decision remains process-wide and first-read latched, matching the
/// historical raw-string `OnceLock`'s observable behavior.
#[inline]
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(crate::preset_gates::margin_row_gpu_backward_enabled)
}

/// Lane-local channel-death latch (design R3).
///
/// A device failure marks THIS lane dead for the rest of the process and
/// touches nothing else — expressly NOT the sweep's memory ledger, whose
/// `usize::MAX` poisoning is one of the defects this module exists to not
/// repeat. Fail-closed: there is no reset, and every subsequent admission
/// refuses with [`Refusal::ChannelDead`].
fn channel_latch() -> &'static OnceLock<&'static str> {
    static DEAD: OnceLock<&'static str> = OnceLock::new();
    &DEAD
}

/// Latch the lane's channel dead. Idempotent; the FIRST reason wins.
pub(crate) fn mark_channel_dead(reason: &'static str) {
    if channel_latch().set(reason).is_ok() {
        // Always printed: channel death is a soundness-adjacent signal, not a
        // profiling note. Once per process by construction (OnceLock).
        eprintln!("[gpu-backward] channel dead: {reason}");
        // D4: channel death is the lane's terminal event — the closest thing
        // this module has to an end-of-run hook — so dump the per-site census
        // with it (suppressed when nothing has been observed, e.g. an M1
        // qualification failure before any pass).
        if let Some(census) = site_census_line() {
            eprintln!("{census}");
        }
    }
}

/// The latched death reason, if any.
pub(crate) fn channel_dead() -> Option<&'static str> {
    channel_latch().get().copied()
}

/// Print the armed banner once per process (engagement telemetry, rule R9).
fn note_armed_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let authority = authority_sites_label(SITE_AUTHORITY_GRANTED, M3_AUTHORITY_GRANTED);
        eprintln!(
            "[gpu-backward] armed ({PROVENANCE_MARKER}): M1 self-check runs on first \
             admission; verdict_authority_sites={authority}; every ungranted verdict site \
             returns shadow-only, and shadow execution separately requires \
             NY_MARGIN_ROW_GPU_SHADOW=1 (design section 7)"
        );
    });
}

// ---------------------------------------------------------------------------
// PER-SITE R9 ACCOUNTING (FIX D4): the shadow phase must be able to PROVE
// which site engaged, how often, and why it refused.
// ---------------------------------------------------------------------------

/// One site's engagement census. Counters are per SITE (not per process):
/// impl-v8 kept ONE process-wide OK counter printed at `n == 0 || n % 32 == 0`,
/// so a low-frequency site (y_refresh) could shadow hundreds of times and
/// print zero `site=` lines while a hot site consumed the whole cadence.
struct SiteLedger {
    /// Successful shadow observations (enclosure-consistent, discarded).
    oks: std::sync::atomic::AtomicU64,
    /// Pre-registered M2 kills observed at this site.
    trips: std::sync::atomic::AtomicU64,
    /// Refusals by [`Refusal::label`] — the `Unmappable` string included, so
    /// the four B4/B5 reasons and the D2 budget stop masking each other.
    refusals: std::sync::Mutex<std::collections::BTreeMap<&'static str, u64>>,
}

impl SiteLedger {
    const fn new() -> Self {
        Self {
            oks: std::sync::atomic::AtomicU64::new(0),
            trips: std::sync::atomic::AtomicU64::new(0),
            refusals: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }
}

static SITE_LEDGERS: [SiteLedger; SITE_SLOTS] = [
    SiteLedger::new(),
    SiteLedger::new(),
    SiteLedger::new(),
    SiteLedger::new(),
    SiteLedger::new(),
];

/// Report each DISTINCT (site, reason) pair once per process, and COUNT every
/// occurrence (the `gpu_seam::note_refusal` lesson — a refusal counter with no
/// reason attached costs a measurement cycle — extended per site by D4).
fn note_refusal(site: u32, refusal: Refusal) {
    let label = refusal.label();
    let ledger = &SITE_LEDGERS[site_slot(site)];
    let first = {
        let mut seen = ledger
            .refusals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let count = seen.entry(label).or_insert(0);
        *count += 1;
        *count == 1
    };
    if first {
        eprintln!("[gpu-backward] refused site={}: {label}", site_tag(site));
    }
}

/// The per-site census as one line, or `None` when nothing was observed at
/// any site (so an idle process prints nothing).
///
/// There is deliberately no process-exit hook in this module (statics never
/// drop, and the lane must not install one), so the census is emitted on a
/// cadence: every [`CENSUS_EVERY`] shadow observations, on every TRIP, and at
/// channel death. It is a pure formatter so a test can pin its shape.
pub(crate) fn site_census_line() -> Option<String> {
    use std::sync::atomic::Ordering;
    let mut parts: Vec<String> = Vec::new();
    let mut any = false;
    for slot in 0..SITE_SLOTS {
        let ledger = &SITE_LEDGERS[slot];
        let oks = ledger.oks.load(Ordering::Relaxed);
        let trips = ledger.trips.load(Ordering::Relaxed);
        let refusals = ledger
            .refusals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if oks == 0 && trips == 0 && refusals.is_empty() {
            continue;
        }
        any = true;
        let reasons = refusals
            .iter()
            .map(|(label, count)| format!("{label}={count}"))
            .collect::<Vec<_>>()
            .join(",");
        parts.push(format!(
            "{}[ok={oks} trip={trips} refused={{{reasons}}}]",
            SLOT_TAGS[slot]
        ));
    }
    any.then(|| format!("[gpu-backward] site census: {}", parts.join(" ")))
}

/// Cadence of the cumulative per-site census line (shadow observations).
const CENSUS_EVERY: u64 = 64;

/// Everything the certified pass needs beyond the CPU lane's arguments.
///
/// Deliberately NOT `gpu_seam::SeamCtx`: the deadline here is a HOST-SIDE
/// value consulted only between transactions (before submit / after drain,
/// design section 6) and is never installed on any backend object — the type
/// distinction keeps that contract from silently eroding into the
/// `GpuCrownBackendDeadlineScope` pattern that poisoned the shared device.
#[derive(Default, Clone, Copy)]
pub(crate) struct LaneCtx<'y> {
    /// Per-head-neuron magnitude bound `max(|ly_j|, |uy_j|)` over the y-box
    /// the seed was built against (the `seed_penalty` contract, unchanged
    /// from `gpu_seam`).
    pub(crate) y_abs: Option<&'y [f64]>,
    /// The caller's real deadline. Host-side checks only (R3).
    pub(crate) deadline: Option<Instant>,
    /// B1 PLAN-CACHE HANDOFF (the seam stub of impl-v7 made real): the
    /// residency cache, OWNED BY THE CALLER that owns the `RootGates` the
    /// statics are built from — `MarginRowBab::lane_cache` for the hot
    /// sites. Ownership is what excludes staleness structurally (the cache
    /// dies with the driver that owns the engine); `PlanCache::probe` still
    /// re-validates a fail-closed identity fingerprint on every hit, and
    /// every hit re-runs the per-dom gate admission. `None` = rebuild the
    /// statics per pass (bit-for-bit the pre-B1 behavior, and what the
    /// not-yet-wired sites do).
    pub(crate) plan_cache: Option<&'y cache::PlanCache>,
}

/// One certified pass, GPU-authoritative on `Ok`.
///
/// CONTRACT (identical shape to `engine::run_seamed`): on `Ok` the returned
/// [`PassOut`] is a drop-in replacement for
/// `BackwardEngine::run(seed, dom, dir, None, false)` — the caller
/// concretizes it with the lane's own unchanged f64 `concretize_*`. On ANY
/// `Err` the caller MUST run that exact CPU call; the refusal carries no
/// partial result and grants no authority.
///
/// v4 state: admission + the on-device M1 qualification are real; the
/// verdict path then refuses [`Refusal::ShadowOnly`] until the M3 A/B flips
/// [`M3_AUTHORITY_GRANTED`] (design sections 6–7). The transaction itself
/// runs under the M2 shadow entry ([`shadow_compare`]).
pub(crate) fn run_pass_certified(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    run_pass_certified_site(SITE_ROOT, eng, seed, dom, dir, PassShape::default(), ctx)
}

/// B4 — the batched-children scoring pass (`bab.rs:4193`), with the per-row
/// gate EXCEPTION table.
///
/// Same contract as [`run_pass_certified`]; the exceptions are FULLY
/// CERTIFIED, not heuristic: this pass's output is banked as the child's
/// inherited bound (`cb = raw.max(b)`, bab.rs:1612) and can close the child
/// at pop (bab.rs:1555) without re-proving. See FIX A1 in
/// `transaction::Builder::apply_activation` for the host charge the override
/// magnitudes carry, and `transaction::exception_records` for the two
/// derivation obligations (intercept sign select; the `E*mi` kink term).
pub(crate) fn run_pass_certified_exc(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    exc: &Exceptions,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    run_pass_certified_site(
        SITE_SCORE,
        eng,
        seed,
        dom,
        dir,
        PassShape {
            exc: Some(exc),
            collect_unst_abs: false,
        },
        ctx,
    )
}

/// B5 — the `eval_node` main pass (`bab.rs:2828`) with the heuristic-only
/// `unst_abs` collection.
///
/// ADMISSION: only `Collect{unst_abs:true, rows:None, unst_rows:false}` —
/// anything else refuses so the caller runs the exact CPU pass. The tier-0
/// row capture (`Collect::rows`, needs `tier0_exact > 0`, default 0) and the
/// #alpha-opt `unst_rows` capture (beta-armed) therefore stays on CPU,
/// where they are today.
pub(crate) fn run_pass_certified_collect(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    collect: Collect<'_>,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    // The gate first, THEN the shape: a DISARMED process must report
    // `Disabled`, not a shape refusal it never actually evaluated (v8 checked
    // the shape first — cosmetic, since the caller only reaches here on an
    // admitting shape, but the per-site ledger now prints refusal reasons and
    // a dark run must not accrue B5 shape reasons it never met).
    if !enabled() {
        return Err(Refusal::Disabled);
    }
    if !collect_shape_admits(collect) {
        let refusal = Refusal::Unmappable("collection shape outside the B5 seam");
        note_refusal(SITE_EVAL, refusal);
        return Err(refusal);
    }
    run_pass_certified_site(
        SITE_EVAL,
        eng,
        seed,
        dom,
        dir,
        PassShape {
            exc: None,
            collect_unst_abs: true,
        },
        ctx,
    )
}

/// B6 — one lane of the 2 x n_y y-pack refresh (`engine::y_rows_seamed`).
pub(crate) fn run_pass_certified_y(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    run_pass_certified_site(
        SITE_Y_REFRESH,
        eng,
        seed,
        dom,
        dir,
        PassShape::default(),
        ctx,
    )
}

/// The PURE B5 admission predicate (unit-tested without an engine).
pub(crate) fn collect_shape_admits(collect: Collect<'_>) -> bool {
    collect.unst_abs && collect.rows.is_none() && !collect.unst_rows
}

/// The one armed producer path every site routes through: env latch ->
/// engagement telemetry -> admission -> [`device_authority`] ->
/// [`run_transaction`]. No site adds an authority path or a constructor.
fn run_pass_certified_site(
    site: u32,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    if !enabled() {
        return Err(Refusal::Disabled);
    }
    note_armed_once();
    let result = run_pass_armed(site, eng, seed, dom, dir, shape, ctx);
    if let Err(refusal) = result.as_ref() {
        note_refusal(site, *refusal);
    }
    result
}

/// The armed path with the env latch already decided, so tests and the
/// shadow mode can drive it without touching the process-wide `OnceLock`.
fn run_pass_armed(
    site: u32,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    if channel_dead().is_some() {
        return Err(Refusal::ChannelDead);
    }
    admission_shape(
        eng.root.mode.outward(),
        seed.s.ncols(),
        seed.s.nrows(),
        eng.net.n_y,
    )?;
    let auth = device_authority()?;
    // The moat protocol (design §7) per SITE: the transaction is real, but a
    // site's verdict authority waits for that site's own x3 moat + one-line
    // flip. Until then an armed pass refuses AFTER qualification —
    // engagement telemetry shows the true state — and the device runs only
    // under this site's shadow entry.
    if site_mode(site) != SiteMode::Authorized {
        return Err(Refusal::ShadowOnly);
    }
    // B1 handoff (the impl-v7 seam stub, now real): the VERDICT path takes
    // the CALLER-OWNED cache. `profile` is structurally `false` here —
    // NY_MARGIN_ROW_GPU_PROF is shadow-only.
    run_transaction(
        auth,
        eng,
        seed,
        dom,
        dir,
        shape,
        ctx,
        ctx.plan_cache,
        false,
        &mut transaction::ExecStats::default(),
    )
}

/// The authority gate (design section 6 "admit"): once per process, cached
/// like `ny_core::eft::eft_available`, dispatch the adversarial probe lanes
/// (the ds-primitive composition of `kernels/ds_probe.wgsl` PLUS the
/// production gate-transform kernel) through `ds_primitives.wgsl` on the
/// live device and let [`authority::DeviceParityProof::qualify`] bit-compare
/// the readback against the [`ds`] twin (also checking the resolved
/// denorm-preserve channel). Only that proof can mint the
/// [`VerdictAuthority`] every transaction requires — the M1 gate is BLOCKING
/// at runtime on the running device/driver, not a one-time qualification
/// session (adversarial-review item 2).
fn device_authority() -> Result<&'static VerdictAuthority, Refusal> {
    static AUTHORITY: OnceLock<Option<VerdictAuthority>> = OnceLock::new();
    match AUTHORITY.get_or_init(run_device_self_check) {
        Some(auth) => Ok(auth),
        None => Err(match channel_dead() {
            // A failed self-check latched the channel; report that, forever.
            Some(_) => Refusal::ChannelDead,
            None => Refusal::Unimplemented,
        }),
    }
}

/// The REAL once-per-process M1 self-check (design section 6 "admit" /
/// section 7 M1): acquire the lane's injected device executor, dispatch the
/// ds-primitive probe lanes AND the production gate-transform kernel over the
/// adversarial fixture ([`probe::self_check_streams`], both drain-safe
/// transactions on the lane's DEDICATED device), and let
/// [`authority::DeviceParityProof::qualify`] bit-compare the readback against
/// the host ds twin — also requiring the RESOLVED denorm-preserve channel ON.
/// Every failure latches the lane's channel DEAD for the process (fail-closed,
/// observable, no retry) and every outcome prints `[gpu-backward]` telemetry
/// (device_ms + parity state, rule R6/R9).
///
/// Runs inside the `device_authority` `OnceLock` initializer, so it executes
/// at most once and no transaction can race it (transactions require the
/// authority this mints).
fn run_device_self_check() -> Option<VerdictAuthority> {
    let gpu = match device::lane_gpu() {
        Ok(gpu) => gpu,
        Err(reason) => {
            mark_channel_dead(reason);
            return None;
        }
    };
    let streams = match probe::self_check_streams(gpu.as_ref()) {
        Ok(streams) => streams,
        Err(reason) => {
            mark_channel_dead(reason);
            return None;
        }
    };
    match authority::DeviceParityProof::qualify(
        gpu.denorm_preserve_enabled(),
        &streams.host,
        &streams.device,
    ) {
        Ok(proof) => {
            eprintln!(
                "[gpu-backward] M1 parity PASS: {} lanes bit-identical, denorm-preserve ON, \
                 self-check device_ms={:.1} ({})",
                proof.lanes(),
                streams.device_ms,
                gpu.adapter_summary()
            );
            Some(VerdictAuthority::grant(proof))
        }
        Err(refusal) => {
            eprintln!(
                "[gpu-backward] M1 parity REFUSED: {refusal:?}, self-check device_ms={:.1} ({})",
                streams.device_ms,
                gpu.adapter_summary()
            );
            // A bit mismatch already latched inside `qualify`; the other
            // refusals (denorm-preserve resolved OFF, under-covered stream)
            // are equally permanent for this process — latch them too so
            // every subsequent admission refuses `ChannelDead` in O(ns).
            if channel_dead().is_none() {
                mark_channel_dead("M1 qualification refused (see the parity line above)");
            }
            None
        }
    }
}

/// The drain-safe transaction (design section 6): the ONLY `PassOut` producer
/// in this module, unreachable without a [`VerdictAuthority`] — which only
/// the on-device parity self-check can mint (and DOES mint on this GB10 —
/// 6144-lane bit parity, 2026-08-19). REAL as of v4: acquires the lane's
/// non-blocking exclusivity token (R4), then runs the whole pass as ONE
/// program submit with an unconditional drain ([`transaction::execute`]:
/// plan reuse from `gpu_seam`, head gemm + gates + bias reduces + convs +
/// merges, host directed-rounding egress, error-floor + realization-probe
/// guards). Callers: the (M3-gated) armed verdict path and the M2 shadow.
#[allow(clippy::too_many_arguments)]
fn run_transaction(
    authority: &VerdictAuthority,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &LaneCtx<'_>,
    plan_cache: Option<&cache::PlanCache>,
    profile: bool,
    stats: &mut transaction::ExecStats,
) -> Result<PassOut, Refusal> {
    // Authority implies the device resolved (the self-check ran on it), so a
    // failure here is a torn-down process state: fail closed as ChannelDead.
    let gpu = device::lane_gpu().map_err(|_| Refusal::ChannelDead)?;
    let _token = device::try_acquire_device().ok_or(Refusal::Busy)?;
    transaction::execute(
        authority, gpu, eng, seed, dom, dir, shape, ctx, plan_cache, profile, stats,
    )
}

// ---------------------------------------------------------------------------
// M2 SHADOW MODE (design section 7 M2): compute on device, compare against
// the CPU pass, DISCARD. No verdict authority — the caller returns the CPU
// pass unchanged whether the shadow ran, refused, or tripped.
// ---------------------------------------------------------------------------

/// Latched raw `NY_MARGIN_ROW_GPU_SHADOW` string: latch the STRING once, then
/// derive the DECISION per call through the unit-tested [`armed_from_raw`]
/// predicate.
fn shadow_env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::margin_row::MARGIN_ROW_GPU_SHADOW))
        .as_deref()
}

/// Is the M2 shadow armed?
#[inline]
pub(crate) fn shadow_enabled() -> bool {
    armed_from_raw(shadow_env_raw())
}

// ---------------------------------------------------------------------------
// PER-SITE SHADOW SELECTOR (FIX D5a — the shadow phase's "one lever per
// cycle" instrument)
// ---------------------------------------------------------------------------

/// The PURE spec of `NY_MARGIN_ROW_GPU_SHADOW_SITES`, unit-tested without an
/// environment. Returns the mask of sites the M2 shadow may observe; the
/// shadow LEVER (`NY_MARGIN_ROW_GPU_SHADOW=1`) still has to be armed for any
/// of them to run.
///
/// MAPPING (exact, and the only admissible one):
///
/// | value                       | shadowed sites                        |
/// |-----------------------------|---------------------------------------|
/// | unset                       | `root` — the impl-v7 behaviour        |
/// | `""` / whitespace only      | `root` (an empty value is not a name) |
/// | `root` \| `y_refresh` \| `eval` \| `score` | just that site        |
/// | comma list, e.g. `eval,score` | the UNION of the named sites        |
/// | `all`                       | all four (impl-v8's implicit default) |
/// | `none`                      | NO site — the shadow-lever-only control arm |
/// | anything else               | `Err(token)` -> NO site shadows, loudly |
///
/// Tokens are split on `,`, individually trimmed, and empty tokens are
/// skipped (a trailing comma is not an error). Matching is EXACT and
/// lowercase — the `ny_levers` chokepoint discipline that refuses
/// `"true"`/`"yes"` for a Bool applies here too, so `EVAL` is a typo, not an
/// alias. The token spelling is [`SLOT_TAGS`], the same spelling the
/// telemetry prints, so a `site=` line always names a value this selector
/// accepts.
///
/// AN UNRECOGNIZED TOKEN SHADOWS NOTHING (and says so once, loudly) rather
/// than falling back to the default. Falling back would let a typo'd
/// `NY_MARGIN_ROW_GPU_SHADOW_SITES=evla` produce a full, plausible-looking
/// run whose site was `root` — the exact "reads as armed, measured something
/// else" class this campaign has paid for nine times. Zero shadow lines plus
/// one refusal banner is unmistakable.
pub(crate) fn shadow_sites_from(raw: Option<&str>) -> Result<u32, &str> {
    let Some(text) = raw else {
        return Ok(SITE_ROOT);
    };
    if text.trim().is_empty() {
        return Ok(SITE_ROOT);
    }
    let mut mask = 0u32;
    for token in text.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if token == "all" {
            mask |= SITE_ALL;
            continue;
        }
        if token == "none" {
            continue;
        }
        match SITE_TOKENS.iter().find(|(name, _)| *name == token) {
            Some((_, bit)) => mask |= *bit,
            None => return Err(token),
        }
    }
    Ok(mask)
}

/// Latched raw `NY_MARGIN_ROW_GPU_SHADOW_SITES` string (same discipline as
/// [`env_raw`]: latch the STRING once, derive the DECISION per call through
/// the unit-tested [`shadow_sites_from`] spec).
fn shadow_sites_env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| {
        ny_levers::read_raw(&ny_levers::decls::margin_row::MARGIN_ROW_GPU_SHADOW_SITES)
    })
    .as_deref()
}

/// This process's shadow-site mask. A rejected value shadows NOTHING and
/// prints once (see [`shadow_sites_from`]).
pub(crate) fn shadow_sites() -> u32 {
    match shadow_sites_from(shadow_sites_env_raw()) {
        Ok(mask) => mask,
        Err(token) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[gpu-backward] shadow site selector REJECTED: \
                     NY_MARGIN_ROW_GPU_SHADOW_SITES contains the unknown token {token:?} — \
                     NO site will shadow. Accepted tokens: root, y_refresh, eval, score, \
                     all, none (comma-separated); unset means root"
                );
            });
            0
        }
    }
}

/// Render the shadow-site mask as its own selector vocabulary (telemetry).
fn shadow_sites_label(mask: u32) -> String {
    if mask == 0 {
        return "none".to_string();
    }
    SITE_TOKENS
        .iter()
        .filter(|(_, bit)| mask & bit != 0)
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(",")
}

/// Render the compile-time authority configuration through the same mask
/// composition used by [`site_mode`]. Telemetry only, but deliberately pure
/// so a future one-line authority flip cannot retain a stale banner.
fn authority_sites_label(granted: u32, m3: bool) -> String {
    shadow_sites_label(effective_site_mask(granted, m3))
}

/// Latched raw `NY_MARGIN_ROW_GPU_PROF` string. Profile mode is SHADOW-ONLY:
/// the flag is plumbed
/// exclusively through the shadow entry, so the (M3-gated) verdict path is
/// structurally `profile=false`.
fn prof_env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::margin_row::MARGIN_ROW_GPU_PROF))
        .as_deref()
}

/// Is the per-step profile mode armed (effective only with the shadow)?
#[inline]
pub(crate) fn prof_enabled() -> bool {
    armed_from_raw(prof_env_raw())
}

/// The SHADOW entry's B1 plan-residency cache. SHADOW-SCOPED by design:
/// the shadow path is discard-only (results are compared and dropped;
/// verdicts are bit-identical to an unshadowed run by construction), so the
/// worst a stale hit can do is a false SHADOW TRIP — fail-closed, the
/// channel latches dead. The VERDICT path takes NO cache until the B4+
/// wiring gives it a caller-owned one (see `run_pass_armed`'s seam stub);
/// a process-scoped slot is deliberately FORBIDDEN there. Every probe
/// re-validates the engine fingerprint and every hit re-runs the per-dom
/// gate admission (`cache::PlanCache::probe`, `cache::check_gate_admission`).
fn shadow_plan_cache() -> &'static cache::PlanCache {
    static CACHE: OnceLock<cache::PlanCache> = OnceLock::new();
    CACHE.get_or_init(cache::PlanCache::default)
}

/// What one shadow pass observed (returned for tests; production reads the
/// telemetry lines).
#[derive(Debug)]
pub(crate) enum ShadowOutcome {
    /// Device pass computed AND enclosure-consistent with the CPU pass.
    Ok {
        /// Median over rows of the device pass's concretized certified
        /// radius `sum_i e[i,r]*xabs[i] + eb[r]` — the R1/M2 telemetry.
        median_radius: f64,
        /// Same statistic for the CPU pass (context for the R1 ratio).
        cpu_median_radius: f64,
        /// Wall time of the device side (plan + program + drain + guards).
        device_ms: f64,
        /// Whether assertion (a) ran: coefficient/bias overlap is only a
        /// valid invariant when both passes consumed the IDENTICAL seed —
        /// an f32-inexact (or error-carrying) seed's residual is charged
        /// into the device `eb` at FUNCTIONAL level (`seed_penalty`), not
        /// into the coefficient lanes, so the elementwise comparison is not
        /// defined there. The realization probe (assertion (b)) still ran.
        coeff_overlap_checked: bool,
        /// B1 perf telemetry (build/upload/dispatch/plan-cache/per-step).
        ///
        /// Boxed: `ExecStats` is the bulk of this variant, and `Refused`/`Trip`
        /// would otherwise pay for it in every slot. Layout only — the same
        /// telemetry, read through the same field names.
        stats: Box<transaction::ExecStats>,
    },
    /// The device pass refused (CPU pass stands; nothing was compared).
    Refused(Refusal),
    /// A pre-registered M2 KILL: enclosure/probe/floor inconsistency. The
    /// lane channel is latched dead by the time this is returned.
    Trip(&'static str),
}

/// M2 assertion (a): the two enclosures must OVERLAP —
/// `|a_gpu - a_cpu| <= e_gpu + e_cpu` elementwise (both intervals contain the
/// same exact-real coefficient, so disjointness proves one of them wrong),
/// and the same for bias against `eb_gpu + eb_cpu`. One ulp of comparison
/// headroom (`next_up`) so the check's own f64 adds cannot false-trip.
fn enclosure_overlap(cpu: &PassOut, gpu: &PassOut) -> Result<(), &'static str> {
    if gpu.a.raw_dim() != cpu.a.raw_dim() || gpu.b.len() != cpu.b.len() {
        return Err("shape");
    }
    let zeros;
    let e_cpu = match cpu.e.as_ref() {
        Some(e) => e,
        None => {
            zeros = ndarray::Array2::<f64>::zeros(cpu.a.raw_dim());
            &zeros
        }
    };
    let Some(e_gpu) = gpu.e.as_ref() else {
        return Err("device pass without an error lane");
    };
    let (ac, ag) = (
        cpu.a.as_slice().expect("standard layout"),
        gpu.a.as_slice().expect("standard layout"),
    );
    let (ec, eg) = (
        e_cpu.as_slice().expect("standard layout"),
        e_gpu.as_slice().expect("standard layout"),
    );
    for i in 0..ac.len() {
        if (ag[i] - ac[i]).abs() > next_up(eg[i] + ec[i]) {
            return Err("coefficient enclosure");
        }
    }
    for r in 0..cpu.b.len() {
        if (gpu.b[r] - cpu.b[r]).abs() > next_up(gpu.eb[r] + cpu.eb[r]) {
            return Err("bias enclosure");
        }
    }
    Ok(())
}

/// Per-row certified radius `sum_i e[i,r]*xabs[i] + eb[r]` — the exact
/// uncertainty of [`nominal_rows`], because `xabs[i] = |mid_i| + rad_i`
/// (`root.rs:856`).
fn radius_rows(pass: &PassOut, xabs: &[f64]) -> Vec<f64> {
    let rows = pass.a.ncols();
    let n_in = pass.a.nrows();
    let mut radii = vec![0.0f64; rows];
    if let Some(e) = pass.e.as_ref() {
        let es = e.as_slice().expect("standard layout");
        for i in 0..n_in.min(xabs.len()) {
            let row = &es[i * rows..(i + 1) * rows];
            for (r, ev) in row.iter().enumerate() {
                radii[r] += ev * xabs[i];
            }
        }
    }
    for (r, radius) in radii.iter_mut().enumerate() {
        *radius += pass.eb[r];
    }
    radii
}

/// The UNPENALIZED lower concretization per pass row —
/// `sum_i a[i,r]*mid_i + b[r] - sum_i |a[i,r]|*rad_i` — i.e.
/// `engine::concretize` with the certified slack left OUT, so the value is
/// the pass's NOMINAL estimate of the exact functional and
/// [`radius_rows`] is its enclosure radius.
fn nominal_rows(pass: &PassOut, mid: &[f64], rad: &[f64]) -> Vec<f64> {
    let rows = pass.a.ncols();
    let n_in = pass.a.nrows();
    let asl = pass.a.as_slice().expect("standard layout");
    let mut v = vec![0.0f64; rows];
    for i in 0..n_in.min(mid.len()).min(rad.len()) {
        let arow = &asl[i * rows..(i + 1) * rows];
        for (r, av) in arow.iter().enumerate() {
            v[r] += av * mid[i] - av.abs() * rad[i];
        }
    }
    for (r, out) in v.iter_mut().enumerate() {
        *out += pass.b[r];
    }
    v
}

/// FIX A6 — the per-site shadow's CONSUMED-SURFACE assertion.
///
/// `enclosure_overlap` compares the coefficient/bias planes elementwise, but
/// it only APPLIES when the seed is f32-exact and error-free
/// ([`seed_is_device_exact`]); the eval and score sites are seeded with the
/// MARGIN seed, which carries certified error, so on those sites it is
/// skipped and the plane comparison alone would leave the shadow with only
/// the realization probe. This assertion covers what those sites actually
/// CONSUME, at the consumer's own concretization:
///
/// * `eval_node` (bab.rs:2838-2872) reads `per_class_direct` =
///   `concretize_lower(pass)[r] + cst[r] - cst_err[r]` (bounds.rs:253-269)
///   for EVERY column, keeps a per-column copy for the col-retire decision
///   (`per_track`, bab.rs:2853 / :3341 — the second verdict-bearing
///   consumer the design did not name), and reduces the columns to the
///   aggregate `b` by `min_c max(direct_c, m1_c, m2v_c)`;
/// * `score_candidates` (bab.rs:4200-4238) reads the same
///   `concretize_lower(pass)` per candidate row.
///
/// `cst`, `cst_err`, `m1`, `m2v` and the min/max reduction are all
/// PASS-INDEPENDENT, so comparing every row of the concretized surface
/// dominates both the per-column directs AND the aggregate: two passes whose
/// per-row surfaces overlap cannot disagree on any function of them by more
/// than the same amount.
///
/// The comparison itself is the enclosure one: both passes bound the SAME
/// exact functional, so `|nominal_gpu - nominal_cpu|` must not exceed
/// `radius_gpu + radius_cpu` (one ulp of headroom so the check's own f64 adds
/// cannot false-trip). Disjointness proves one of them wrong.
fn consumed_surface_overlap(
    eng: &BackwardEngine<'_>,
    cpu: &PassOut,
    gpu: &PassOut,
) -> Result<(), &'static str> {
    if gpu.a.raw_dim() != cpu.a.raw_dim() || gpu.b.len() != cpu.b.len() {
        return Err("surface shape");
    }
    let root = eng.root;
    let (mid, rad, xabs) = (
        root.mid.as_slice(),
        root.rad.as_slice(),
        root.xabs.as_slice(),
    );
    let nc = nominal_rows(cpu, mid, rad);
    let ng = nominal_rows(gpu, mid, rad);
    let rc = radius_rows(cpu, xabs);
    let rg = radius_rows(gpu, xabs);
    for r in 0..nc.len() {
        if !(nc[r].is_finite() && ng[r].is_finite()) {
            return Err("non-finite consumed surface");
        }
        if (ng[r] - nc[r]).abs() > next_up(rg[r] + rc[r]) {
            return Err("consumed surface enclosure");
        }
    }
    Ok(())
}

/// Median over rows of the concretized certified radius
/// `sum_i e[i,r]*xabs[i] + eb[r]` (M2 telemetry (c); NOT verdict input).
fn radius_median(pass: &PassOut, xabs: &[f64]) -> f64 {
    let rows = pass.a.ncols();
    let n_in = pass.a.nrows();
    let mut radii = vec![0.0f64; rows];
    if let Some(e) = pass.e.as_ref() {
        let es = e.as_slice().expect("standard layout");
        for i in 0..n_in.min(xabs.len()) {
            let row = &es[i * rows..(i + 1) * rows];
            for (r, ev) in row.iter().enumerate() {
                radii[r] += ev * xabs[i];
            }
        }
    }
    for (r, radius) in radii.iter_mut().enumerate() {
        *radius += pass.eb[r];
    }
    radii.sort_by(f64::total_cmp);
    if radii.is_empty() {
        return f64::NAN;
    }
    radii[radii.len() / 2]
}

/// One M2 shadow observation: run the FULL producer chain (admission ->
/// authority -> transaction incl. floor/probe guards), compare, DISCARD.
/// Any enclosure trip latches the lane channel dead (fail-closed: an
/// inconsistent channel must not keep shadowing, and the latch is the
/// in-process form of the pre-registered M2 kill).
#[allow(clippy::too_many_arguments)]
pub(crate) fn shadow_observe(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &LaneCtx<'_>,
    cpu: &PassOut,
) -> ShadowOutcome {
    if channel_dead().is_some() {
        return ShadowOutcome::Refused(Refusal::ChannelDead);
    }
    if let Err(refusal) = admission_shape(
        eng.root.mode.outward(),
        seed.s.ncols(),
        seed.s.nrows(),
        eng.net.n_y,
    ) {
        return ShadowOutcome::Refused(refusal);
    }
    let auth = match device_authority() {
        Ok(auth) => auth,
        Err(refusal) => return ShadowOutcome::Refused(refusal),
    };
    let started = Instant::now();
    let mut stats = transaction::ExecStats::default();
    // The caller-owned cache when the site carries one (so a shadow run
    // exercises exactly the residency path its authorized twin would); the
    // process-scoped shadow slot otherwise. Both are discard-only here.
    let cache_for_shadow = ctx.plan_cache.unwrap_or_else(|| shadow_plan_cache());
    let gpu_pass = match run_transaction(
        auth,
        eng,
        seed,
        dom,
        dir,
        shape,
        ctx,
        Some(cache_for_shadow),
        prof_enabled(),
        &mut stats,
    ) {
        Ok(pass) => pass,
        // Guard trips ARE M2 kills (assertion (b) is the probe; the floor is
        // the model-ball sanity of the same protocol) — latch, do not tune.
        Err(refusal @ (Refusal::Probe | Refusal::ErrorFloor)) => {
            mark_channel_dead(match refusal {
                Refusal::Probe => "M2 shadow realization-probe trip",
                _ => "M2 shadow error-floor trip",
            });
            return ShadowOutcome::Trip(match refusal {
                Refusal::Probe => "realization probe",
                _ => "error floor",
            });
        }
        Err(refusal) => return ShadowOutcome::Refused(refusal),
    };
    let device_ms = started.elapsed().as_secs_f64() * 1e3;
    let coeff_overlap_checked = seed_is_device_exact(seed);
    if coeff_overlap_checked {
        if let Err(plane) = enclosure_overlap(cpu, &gpu_pass) {
            mark_channel_dead("M2 shadow enclosure trip");
            return ShadowOutcome::Trip(plane);
        }
    }
    // FIX A6: the FULL consumed surface, on EVERY site and every seed kind
    // (this is the only assertion that applies to the error-carrying margin
    // seeds the eval/score sites use).
    if let Err(what) = consumed_surface_overlap(eng, cpu, &gpu_pass) {
        mark_channel_dead("M2 shadow consumed-surface trip");
        return ShadowOutcome::Trip(what);
    }
    let xabs = eng.root.xabs.as_slice();
    ShadowOutcome::Ok {
        median_radius: radius_median(&gpu_pass, xabs),
        cpu_median_radius: radius_median(cpu, xabs),
        device_ms,
        coeff_overlap_checked,
        stats: Box::new(stats),
    }
    // gpu_pass drops here: DISCARDED by construction.
}

/// Does the device see EXACTLY the seed the CPU pass consumed? True iff every
/// value is f32-exact and the seed carries no certified error — then the
/// coefficient lanes of the two passes bound the same exact recursion and
/// assertion (a) applies. Production margin/identity seeds are of this kind.
fn seed_is_device_exact(seed: &Seed) -> bool {
    let exact_vals = seed.s.iter().all(|&v| {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        hi.is_finite() && ny_core::f32_to_f64_exact(hi) == v
    });
    exact_vals && seed.e.as_ref().is_none_or(|e| e.iter().all(|&v| v == 0.0))
}

/// Engine-facing shadow entry: telemetry wrapper around [`shadow_observe`].
/// Called AFTER the CPU pass; never changes anything the caller returns.
pub(crate) fn shadow_compare(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
    cpu: &PassOut,
) {
    shadow_compare_site(
        SITE_ROOT,
        eng,
        seed,
        dom,
        dir,
        PassShape::default(),
        ctx,
        cpu,
    );
}

/// The per-site shadow (design §3.3: "`shadow_compare` generalized to take
/// the site tag"). Runs ONLY when this site's mode is [`SiteMode::Shadow`] —
/// an OFF site never touches the device and an AUTHORIZED site already got
/// the device pass through [`run_pass_certified_site`], so shadowing it again
/// would double the device cost for nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shadow_compare_site(
    site: u32,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &LaneCtx<'_>,
    cpu: &PassOut,
) {
    if site_mode(site) != SiteMode::Shadow {
        return;
    }
    static ARMED: std::sync::Once = std::sync::Once::new();
    ARMED.call_once(|| {
        // Deliberately does NOT quote SWEEP_POST_SUBMIT_KILL_LINE verbatim:
        // the M2/M3 protocol greps every run log for that literal, and a
        // banner containing it is a guaranteed false KILL hit.
        eprintln!(
            "[gpu-backward] SHADOW armed ({PROVENANCE_MARKER}) sites={}: device passes are \
             computed, compared against the CPU pass, and DISCARDED (M2; grep run logs for the \
             sweep post-submit poison line, mod.rs SWEEP_POST_SUBMIT_KILL_LINE)",
            shadow_sites_label(shadow_sites())
        );
    });
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Total shadow observations across all sites — the census cadence only.
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    let tag = site_tag(site);
    let ledger = &SITE_LEDGERS[site_slot(site)];
    match shadow_observe(eng, seed, dom, dir, shape, ctx, cpu) {
        ShadowOutcome::Ok {
            median_radius,
            cpu_median_radius,
            device_ms,
            coeff_overlap_checked,
            stats,
        } => {
            // D4: PER-SITE counter. A site's own first observation always
            // prints, so a low-frequency site can never be starved of
            // `site=` lines by a hot one.
            let n = ledger.oks.fetch_add(1, Ordering::Relaxed);
            let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
            // First pass at this site and every 32nd afterwards: enough to
            // read the R1 radius statistic off a run log without flooding
            // it. The B1 perf fields and the D2/D3 availability pair ride
            // the same line (R9: a perf run without them is not a
            // measurement of the B1 lever).
            if n == 0 || n.is_multiple_of(32) {
                let plan_cache = match stats.plan_cache_hit {
                    Some(true) => "hit",
                    Some(false) => "miss",
                    None => "none",
                };
                eprintln!(
                    "[gpu-backward] shadow ok #{n} site={tag}: rows={} \
                     median_radius={median_radius:.3e} (cpu {cpu_median_radius:.3e}) \
                     device_ms={device_ms:.1} overlap_checked={} build_ms={:.1} \
                     upload_ms={:.1} dispatch_ms={:.1} plan_cache={plan_cache} \
                     arena_mb={:.1} resident_mb={:.1}",
                    seed.s.ncols(),
                    coeff_overlap_checked,
                    stats.build_ms,
                    stats.upload_ms,
                    stats.dispatch_ms,
                    mib(stats.arena_bytes),
                    mib(stats.resident_bytes),
                );
            }
            if total.is_multiple_of(CENSUS_EVERY) {
                if let Some(census) = site_census_line() {
                    eprintln!("{census}");
                }
            }
            // Profile mode: per-step device timings, aggregated by label
            // (NY_MARGIN_ROW_GPU_PROF=1; shadow-only by construction).
            if let Some(per_step) = stats.per_step_ms {
                let mut agg: Vec<(&'static str, f64, u32)> = Vec::new();
                for (label, ms) in per_step {
                    match agg.iter_mut().find(|(l, _, _)| *l == label) {
                        Some((_, total, count)) => {
                            *total += ms;
                            *count += 1;
                        }
                        None => agg.push((label, ms, 1)),
                    }
                }
                let line = agg
                    .iter()
                    .map(|(l, total, count)| format!("{l}={total:.2}ms/{count}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!("[gpu-backward] prof #{n} site={tag}: {line}");
            }
        }
        ShadowOutcome::Refused(refusal) => note_refusal(site, refusal),
        // Always printed, never rate-limited: this is the M2 kill criterion.
        ShadowOutcome::Trip(what) => {
            ledger.trips.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[gpu-backward] SHADOW TRIP site={tag} ({what}): device/CPU enclosures \
                 inconsistent — M2 kill criterion, channel latched dead (design section 7)"
            );
            if let Some(census) = site_census_line() {
                eprintln!("{census}");
            }
        }
    }
}

/// Bytes -> MiB for the telemetry lines (never verdict input).
#[allow(clippy::cast_precision_loss)]
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Pure admission predicate (unit-tested without an engine): certified-
/// outward mode only, and the seed must be a non-empty `(n_y, R)` matrix.
fn admission_shape(
    outward: bool,
    rows: usize,
    seed_rows: usize,
    n_y: usize,
) -> Result<(), Refusal> {
    if !outward {
        return Err(Refusal::NotOutward);
    }
    if rows == 0 || n_y == 0 || seed_rows != n_y {
        return Err(Refusal::Rows);
    }
    Ok(())
}
