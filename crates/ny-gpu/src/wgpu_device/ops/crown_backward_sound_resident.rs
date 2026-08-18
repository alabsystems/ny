// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound GPU-RESIDENT CROWN backward (task #15, the win keystone).
//!
//! Keeps the coefficient pair `(lower_a, upper_a)` AND its certified error
//! `(lower_err, upper_err)` on GPU buffers across the whole backward loop —
//! eliminating the per-layer host round-trip that makes `crown_backward_sound_host`
//! slow — and only downloads the FINAL coefficients for the sound concretize.
//! Numerically it must match the proven host composition
//! (`crown_backward_sound_host`); that is the soundness reference.
//!
//! Built incrementally: R1 = single Linear layer (no bias). Activation, bias,
//! multi-layer and Conv2d follow, each gated behind a Metal soundness test
//! against the host reference.
//!
//! # Charged-authority guard coverage (#flush-charge Lane A, audited 2026-08-13)
//!
//! Under CHARGED-flush authority (`QualifiedWithFlushCharge`,
//! `ops/sound_authority.rs`) every route that can reach GPU arithmetic must
//! either (a) pass [`charged_walk_guard`] and the audited charge sites, or
//! (b) be structurally unreachable / verdict-dead, with a PIN that fails the
//! moment that stops being true. One row per route; "pin" names the test(s)
//! that break if the row's claim drifts. Routes GA1/GA2 are the only two
//! chokepoints that carry charged arithmetic; everything else must reduce to
//! them or be refused/dead.
//!
//! | # | Route (entry points) | Guard / immunity | Pin |
//! |---|---|---|---|
//! | GA1 | THE walk body `crown_backward_sound_resident_coeff_seeded_err_gather` — every bounds/coefficients entry funnels its fold arithmetic here: `crown_backward_sound_resident{,_seeded}`, `..._coeff_seeded{,_err}`, both certified-coeffs egresses, `..._residual`, `backward_branch_fine`/`backward_branch_cut_fold`, every resnet branch sub-walk (flat, batched wide, beta/grad/vjp inners) | GUARDED: authority recheck (typed refusal when neither authority holds) → `charged_walk_guard(layers, policy, eft_requested)` → armed charge sites (`charged_bias_slack_or`, `charged_act_bias_slack_or`, `daz_cover_armed` GEMM flush cover) — ALL before the walk's single `run_gpu_checked` GPU section | `charged_walk_guard_tests::*`; `charged_route_coverage_tests::charged_route_funnels_and_guard_ordering_are_pinned` |
//! | GA2 | Concretize funnel `concretize_sound_gpu_batched` (crown_concretize_sound.rs; reached via `concretize_resident_coeff{,_batched}`, the resnet concretize, and the host driver) | GUARDED: charged consult (EFT arm refused, subnormal input-box endpoints refused, `charged_concretize_slack` widening) + armed #u4 C1 row-word consult, all pre-dispatch | `flush_charge_oracle` §F oracles; `taint_consult_is_fail_closed_on_every_arm`; ordering row in `charged_route_funnels_and_guard_ordering_are_pinned` |
//! | GA3 | #seg-resident device stream (`NY_SEG_RESIDENT=1` seed/keep + `seg_merge_dispatch` f32 error lanes) | STRUCTURALLY VERDICT-DEAD under BOTH authorities: worded ⇒ typed refusal at the sub-walk entry (no word channel across device-resident segment boundaries); unworded ⇒ `taint_rows: None` (the merge shell and `download_resident_coeff`) and the armed C1 consult / coeffs firewall refuses absent words. The un-audited on-device merge error lanes can therefore never feed a verdict | `taint_resnet_seg_resident_stream_refuses` (worded); `unworded_frontier_is_verdict_dead_at_the_concretize_funnel` (unworded); `taint_word_gate_is_armed_per_the_2026_08_11_review` (C1 const) |
//! | GA4 | Host f64 seams: the residual skip merge (`crown_backward_sound_resident_residual` loop), `merge_streams`, `add_skip_stream`, `concretize_error_into_bias` | IMMUNE: host IEEE-754 f64 with gradual underflow (ladder rung 4 attests the host reference); a flushing GPU adapter cannot touch this arithmetic, and its outputs re-enter GA1/GA2 | `charged_route_coverage_tests::host_merge_seams_preserve_subnormal_error_mass` (fails if this arithmetic is moved onto a flushing device or starts dropping subnormal error mass) |
//! | GA5 | `crown_backward_sound_host` (host driver dispatching the raw diagnostic GEMM) | REFUSED under charged authority: typed refusal at entry — its Higham/γ charges are not audited against the flush model. It also has no production caller (`#[allow(dead_code)]`, tests only) | host-driver row in `charged_route_funnels_and_guard_ordering_are_pinned` |
//! | GA6 | Fast unsound path (`crown_backward_gpu`, `crown_backward_gpu_seeded`) | NO verdict authority in ANY mode (diagnostics tier, TAINT_GUARD_AUDIT.md §3); the routing seam consumes only the `*_sound` entries behind `provides_sound_gpu_crown` | `sound_authority::gpu_tests::ordinary_device_is_unconditionally_unarmed`; ny-propagate sound_gpu_gate routing tests |
//! | GA7 | Raw `GemmEngine` surface (ops/gemm.rs) | Quarantined typed `Err` on every raw op; the ONLY authority-bearing accessor is `as_gpu_crown_backward`, gated on the same two cached predicates as `provides_sound_gpu_crown` | `ordinary_device_is_unconditionally_unarmed`; `explicit_constructor_is_typed_and_fail_closed` |
//! | GA8 | Cut-fold kernels (`cut_fold_resident`) | UNREACHABLE: ny-core `resident_cut_fold_proof_authority_enabled()` is compile-time `false`, so `active_resident_cut_fold()` is always `None` and the cut-fold branch never arms | ny-core `resident_cut_fold_registry_cannot_acquire_proof_authority` |
//! | GA9 | MaxPool2d / dual-alpha kernels | UNREACHABLE in the sound walk: `resident_fold_plan` rejects the layer kinds, and `charged_walk_guard` refuses them independently (selection/comparison under DAZ is un-audited) | `charged_walk_guard_admits_clean_layers_and_refuses_each_channel`; the `resident_preflight_*` rejection tests |
//! | GA10 | EFT channel (min-combine, concretize EFT arm) | REFUSED: `eft_forbidden` at the walk guard AND the concretize consult; `eft_primitives_cached()` is additionally false on a flushing adapter by the #u2b entailment | guard EFT arm in `charged_walk_guard_admits_clean_layers_and_refuses_each_channel`; `report_ladder_and_pin_conjunction` (#u2b) |
//! | GA11 | `FlValueGemmDevice` (fl_value_gemm.rs) | SEPARATE, never-charged device: consults no charged policy; its single-kernel magnitude refusal (audit G10) is exact by construction and unchanged by charged mode | fl_value_gemm refusal tests |
//! | GA12 | Steering channels (joint-alpha, alpha-gradient, point-VJP, attack/gradient steering) | NO bound state published (gradients/captures only steer optimization; any α ≥ relaxation-valid / β ≥ 0 stays sound); bounds still funnel through GA1+GA2 | — (soundness does not depend on these values) |

use std::sync::Arc;

use ny_core::dd::next_up_f64;
use ny_core::{
    f32_to_f64_exact, f64_to_f32_down, CertifiedWeightError, GpuCrownLayer, GpuCrownSeed,
    GpuResnetSegment, NyError, Result,
};

use super::super::WgpuDevice;
use super::gemm::select_gemm_dispatch;
use super::intermediate_sweep_carrier::{DeviceSweepCarrier, SweepCarrierLayout};
use super::resident_weights::WeightForm;
use crate::wgpu_device::params::{ConvCol2imParams, ConvReshapeParams, GemmParams};
// `gamma_k_f32`, `combine_slack_f32`, `up_f32` now live in the shared sound-consts
// home so CROWN, the sound concretize, and the sound IBP forward share ONE copy
// (docs/SOUND_GPU_IBP_PLAN.md §2.1). `down_f32` stays local (only the CROWN bias
// fold needs the downward round).
use crate::wgpu_device::sound_consts::{combine_slack_f32, eft_r_slack_f32, gamma_k_f32, up_f32};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AbsParams {
    n: u32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    n: u32,
    /// SOUNDNESS slack on the f32 error GEMM products `s_prod = fl(|A|@|W|)` and
    /// `prop = fl(err@|W|)`. Both are f32-accumulated, so each UNDER-reports its
    /// exact value by up to a factor `γ_k`; multiplying by `slack ≥ 1/(1−γ_k)`
    /// (host-computed, rounded UP) recovers an outward bound. Replaces the old
    /// fixed `SLACK=1.000001`, which only covered the combine's own ~4 ULPs and
    /// silently under-counted the GEMM contraction error for wide k (false proof).
    slack: f32,
    gamma_k: f32,
    additive: f32,
    /// Contraction length `k` (the `A·W` reduction), for the §0 weight-amplified
    /// operand-flush over-bound `flushacc = 1 + k + ‖a_i‖₁ + max_j‖w_j‖₁`.
    k: u32,
    /// Output columns (so `i / out_cols` selects the spec row for `row_abs_a`).
    out_cols: u32,
    /// Scalar host over-bound `≥ max_j‖w_j‖₁` (the `|W|` max column L1).
    w_l1_max: f32,
    _pad: u32,
}

/// #eft-err: params for `CROWN_EFT_MIN_COMBINE_SHADER`. Same 32-byte layout
/// discipline as [`CombineParams`]; `r_slack` replaces `gamma_k` (the EFT
/// channel charges the MEASURED residual, not the a-priori worst case).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EftCombineParams {
    n: u32,
    /// Outward cover for the twin's f32-accumulated residual sum (`≥
    /// 1/(1−γ_{2k+2})` with min-combine op headroom; `eft_r_slack_f32`).
    r_slack: f32,
    /// The SAME `combine_slack_f32(k)` the Higham combine uses — applied to the
    /// propagated `prop = fl(err@|W|)` term, which the EFT channel keeps.
    slack: f32,
    additive: f32,
    k: u32,
    out_cols: u32,
    w_l1_max: f32,
    _pad: u32,
}

/// #seg-resident: params for `RESIDENT_SEG_MERGE_SHADER` (16-byte uniform).
/// `stride` = total dispatched threads (grid-stride loop; see the shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SegMergeParams {
    n: u32,
    slack: f32,
    stride: u32,
    _p1: u32,
}

/// #eft-err process gate: `NY_EFT_ERR=1` (dark, default OFF ⇒ the EFT passes
/// are never dispatched ⇒ byte-identical). The per-adapter capability half of
/// the gate (`verify_eft_primitives`) is checked at the dispatch site.
/// Deliberately NOT OnceLock-cached: read once per FOLD (not per layer), so
/// the differential A/B tests can flip it under `with_env_edits`.
///
/// NOTE the flag BUNDLES two claims beyond residual MEASUREMENT, both now
/// discharged by dedicated oracles (sound_authority.rs ledger, 2026-08-10):
/// * U5, the a-priori Lipschitz propagation swap (`|sel|` / `max(|ls|,|us|)`
///   instead of `|ls|+|us|`, coefficients and intercepts) — validated against
///   the exact worst-realization sup in `u5_activation_lipschitz::{
///   act_eft_err_encloses_worst_realization,
///   act_intercept_bias_eft_err_encloses_worst_realization}`;
/// * U6, concretize is NOT value-neutral in EFT mode (by design; both modes
///   enclose, refusal is bit-identical) — validated in
///   `crown_concretize_sound::tests::
///   u6_concretize_eft_vs_legacy_enclose_and_fail_closed_identity`.
fn eft_err_env_enabled() -> bool {
    std::env::var("NY_EFT_ERR").ok().as_deref() == Some("1")
}

/// Governed live read for the resident segment-composition diagnostics.
///
/// This deliberately remains uncached: scoped diagnostic tests toggle the
/// probe within one process, and every historical reader sampled it live.
fn seg_probe_armed() -> bool {
    ny_levers::read(&ny_levers::decls::telemetry::SEG_PROBE)
        .value
        .as_bool()
}

/// #seg-resident: device-side twin of [`ResidentCoeff`] — the coefficient
/// stream (4 coeff lanes + 4 bias lanes) held in GPU buffers between resnet
/// segments, eliminating the per-segment download → CPU merge → re-upload
/// round-trip (measured: 2810 per-segment fold calls at ~8.6 ms fixed cost
/// each in a 70 s cifar100 BaB run). `wgpu::Buffer` is a cheap ref-counted
/// handle; clones share the underlying storage.
#[derive(Clone)]
pub(crate) struct ResidentCoeffBufs {
    pub(crate) la: wgpu::Buffer,
    pub(crate) ua: wgpu::Buffer,
    pub(crate) le: wgpu::Buffer,
    pub(crate) ue: wgpu::Buffer,
    pub(crate) blo: wgpu::Buffer,
    pub(crate) buo: wgpu::Buffer,
    pub(crate) ble: wgpu::Buffer,
    pub(crate) bue: wgpu::Buffer,
    pub(crate) dim: usize,
    pub(crate) num_specs: usize,
}

/// #seg-resident: the seed-in / keep-out slot state the resnet orchestrator
/// arms around a per-segment fold call. The fold consumes `seed`
/// (encoder-copies it into its ping-0/bias buffers instead of the host-slice
/// upload; `zero_bias_seed` clears the bias lanes — the ResidualProj P
/// branch), and when `keep` is set it SKIPS the final readback and deposits
/// handle-clones of its result buffers in `out`.
#[derive(Default)]
pub(crate) struct ResidentIoState {
    pub(crate) seed: Option<ResidentCoeffBufs>,
    pub(crate) zero_bias_seed: bool,
    pub(crate) keep: bool,
    pub(crate) out: Option<ResidentCoeffBufs>,
    /// Authoritative intermediate-sweep seed/keep transport. Unlike the
    /// legacy segment stream above, this owns all four word twins and the
    /// sticky row accumulator, so it may cross a resident fold without
    /// laundering C1 taint state.
    sweep_seed: Option<DeviceSweepCarrier>,
    sweep_keep: bool,
    sweep_out: Option<DeviceSweepCarrier>,
}

thread_local! {
    /// #seg-resident: THREAD-LOCAL by design, NOT a device field — the
    /// per-domain BaB fan-out (`NY_BAB_RESNET_PARALLEL=1`, default-off) runs
    /// concurrent Rayon workers that each perform their own
    /// resnet gather; a shared slot would let worker A's fold consume worker
    /// B's armed seed (same network ⇒ same dims ⇒ the shape check passes ⇒
    /// WRONG frontier ⇒ false-VERIFIED risk). Arm and consume always happen on
    /// the same thread (the gather calls the fold synchronously).
    ///
    /// This isolation is load-bearing whenever the fan-out is armed, so it must
    /// survive any future attempt to enable it by default. Do not hoist this into
    /// `WgpuDevice`.
    static RESIDENT_IO: std::cell::RefCell<ResidentIoState> =
        std::cell::RefCell::new(ResidentIoState::default());
}

/// Panic/error-safe ownership of the worded resident seed/keep TLS seam.
/// The fold synchronously consumes the seed and deposits exactly one output;
/// every other exit clears the slot so a later unrelated walk cannot inherit
/// stale device buffers.
struct SweepResidentIoGuard {
    armed: bool,
}

impl SweepResidentIoGuard {
    fn arm(seed: DeviceSweepCarrier) -> Result<Self> {
        RESIDENT_IO.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.seed.is_some()
                || slot.zero_bias_seed
                || slot.keep
                || slot.out.is_some()
                || slot.sweep_seed.is_some()
                || slot.sweep_keep
                || slot.sweep_out.is_some()
            {
                return Err(NyError::InternalError(
                    "nested resident seed/keep carrier scope".into(),
                ));
            }
            slot.sweep_seed = Some(seed);
            slot.sweep_keep = true;
            Ok(Self { armed: true })
        })
    }

    fn take_output(mut self) -> Result<DeviceSweepCarrier> {
        let output = RESIDENT_IO.with(|slot| {
            let mut slot = slot.borrow_mut();
            slot.sweep_seed = None;
            slot.sweep_keep = false;
            slot.sweep_out.take()
        });
        self.armed = false;
        output.ok_or_else(|| {
            NyError::InternalError("worded resident fold produced no sweep carrier".into())
        })
    }
}

impl Drop for SweepResidentIoGuard {
    fn drop(&mut self) {
        if self.armed {
            RESIDENT_IO.with(|slot| {
                let mut slot = slot.borrow_mut();
                slot.sweep_seed = None;
                slot.sweep_keep = false;
                slot.sweep_out = None;
            });
        }
    }
}

/// #seg-resident process gate (dark, `NY_SEG_RESIDENT=1`, default OFF ⇒ the
/// per-segment download/merge/re-upload path, byte-identical).
fn seg_resident_enabled() -> bool {
    std::env::var("NY_SEG_RESIDENT").ok().as_deref() == Some("1")
}

/// #seg-resident: outward slack for the on-device merge's f32 evaluation of
/// the CPU merge's f64 error expression `err_a + err_b + |s|·u` (3 RN ops ⇒
/// under-report ≤ γ₃ ≈ 1.8e-7 rel; the multiply adds one more). 5e-7 > (1+u)⁴−1.
const SEG_MERGE_SLACK: f32 = 1.000_000_5;

/// #fold-coalesce process gate (dark, `NY_FOLD_COALESCE=1`, default OFF ⇒
/// byte-identical per-layer submits): collect every layer's command buffer and
/// submit the WHOLE per-chain fold in ONE `queue.submit`, eliminating the
/// per-layer submit/bubble boundaries (the fold idles the GPU 40–60% between
/// them). Numerically byte-identical by construction: the same passes encode
/// in the same order; only the submission granularity changes. Per-layer
/// uniform/slope values stay correct because their uploads become
/// encoder-ordered copies from [`FoldStagingArena`] instead of
/// `queue.write_buffer` (which is submission-ordered and would collapse every
/// layer's write to the last value under a single submit).
pub(super) fn fold_coalesce_enabled() -> bool {
    std::env::var("NY_FOLD_COALESCE").ok().as_deref() == Some("1")
}

/// #u4 process gate: carry the out-of-band `u32` taint words through the MAIN
/// resident walk by dispatching the taint-twin shaders
/// (TAINT_GUARD_AUDIT.md §4). Read ONCE per walk entry (beside the walk's
/// other env reads, `eft_err_env_enabled` / `NY_CONV_ERR_ROWMAX`), deliberately
/// NOT OnceLock-cached so differential A/B tests can flip it under a scoped env
/// guard.
///
/// Gate ON changes NO value bits on Linear/Activation/Conv chains (the twins are
/// drift-pinned bit-identical to the base kernels); the only value-visible
/// difference is a strictly WIDENING refusal: the EFT min-combine consult can
/// skip a tightening (audit C2).
/// #u4 gate state from the environment. ARMED BY DEFAULT (2026-08-11 UTC arming
/// review): `None` = AUTO — the worded walk runs whenever the taint twins are
/// available on this device (measured tax 1.09x after the on-device row-OR,
/// `taint_gate_overhead_report`). `Some(false)` (`NY_GPU_TAINT_WORDS=0`) is
/// the explicit opt-out; `Some(true)` (`=1`) demands words and turns
/// twin-unavailability into a typed refusal instead of a silent un-worded
/// walk.
fn gpu_taint_words_env() -> Option<bool> {
    match std::env::var("NY_GPU_TAINT_WORDS").ok().as_deref() {
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}

impl WgpuDevice {
    /// Resolve the #u4 gate for this device: explicit env wins; AUTO arms
    /// exactly when the twins can be built (storage-buffer limit >= 11). Use
    /// the same cheap capability predicate as `resident_backward_pipelines`
    /// rather than constructing that large cache here: unsupported worded
    /// routes must be able to refuse during preflight without allocating it.
    pub(super) fn taint_words_armed(&self) -> bool {
        match gpu_taint_words_env() {
            Some(v) => v,
            None => self.device.limits().max_storage_buffers_per_shader_stage >= 11,
        }
    }
}

/// Arming boundary for the worded resident route. Conv2d is admitted because
/// reshape, both GEMM schedules, and col2im have exact-value word twins.
/// Keeping this predicate explicit makes a future Conv sub-route opt in to the
/// same transport contract rather than silently relying on boundary reseeding.
const fn taint_walk_conv_route_admitted(_taint_on: bool, _has_conv: bool) -> bool {
    true
}

/// #u4 G13 seeding rule (TAINT_GUARD_AUDIT.md §2c): a seed coefficient at
/// `|a| >= CROWN_COEFF_MAX` is the CPU-side transport sentinel shipped to the
/// GPU as if it were a legitimate value — it enters the walk PRE-TAINTED.
/// `CROWN_COEFF_MAX == FALLBACK_BOUND` is pinned by
/// `sentinel_taint_selfcheck::cpu_tests::sentinel_matches_core`, so this is the
/// same threshold the GEMM twin self-seeds at (G7). NaN/Inf count too — they
/// are the other "magnitude unknown" markers.
fn taint_seed_word(value: f32) -> u32 {
    u32::from(!value.is_finite() || value.abs() >= ny_core::CROWN_COEFF_MAX)
}

/// #u4: OR an `[rows × cols]` word buffer down to its per-spec-row
/// accumulator (`rows[i / cols] |= word[i]`). The unconditional form of the
/// fail-closed no-twin transport: never drops a word (annihilation conjuncts,
/// where sound, are applied by the specialized companions instead —
/// `bias_fold_taint` / `intercept_fold_taint` in crown_backward_sound_host.rs).
///
/// TEST-REFERENCE STATUS (2026-08-10): the walk no longer calls this — every
/// transport row-OR now runs ON-DEVICE (`TAINT_ROW_OR_SHADER`, use_partner=0
/// is this exact rule, with the word VALUE OR'd) into `TaintWalkState::
/// rows_dev`, read back once at walk end. Kept as the committed CPU statement
/// of the row-OR semantics the shader mirrors.
#[allow(dead_code)]
fn or_taint_words_into_rows(words: &[u32], cols: usize, rows: &mut [u32]) {
    let cols = cols.max(1);
    for (i, &word) in words.iter().enumerate() {
        if word != 0 {
            if let Some(slot) = rows.get_mut(i / cols) {
                *slot |= word;
            }
        }
    }
}

/// #u4 uniform of [`super::super::shaders_taint::TAINT_ROW_OR_SHADER`] (the
/// on-device word→row transport). `use_partner`: 0 = unconditional, 1 =
/// per-COLUMN partner (`partner[i % cols]`), 2 = per-ELEMENT partner
/// (`partner[i]`) — see the shader doc for the annihilation contract.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TaintRowOrParams {
    rows: u32,
    cols: u32,
    use_partner: u32,
    _pad: u32,
}

/// #u4 walk state (allocated only when the worded route is armed, either AUTO
/// or explicit `NY_GPU_TAINT_WORDS=1`): the out-of-band `u32` word channel
/// riding beside the resident value buffers. The four
/// ping-pong pairs mirror `la`/`ua`/`le`/`ue` exactly (same element counts,
/// rotated with the same `ping`); `ws`/`wprop` mirror the `s_scratch`/
/// `prop_scratch` reductions; `w_rowabs_*` mirror the §0 row-L1 output per
/// side; `w_conv_reshaped`/`w_conv_gemm` mirror Conv's internal value
/// scratches and are reused in encoder order for A, S, and prop; `zw` is the
/// all-zero word buffer bound as `taint_b` for every weight
/// operand — host weights are exact data, NEVER tainted (audit §2, G7 row:
/// taint is born only at saturation or shipped in a seed). `rows_dev` is the
/// ON-DEVICE per-spec-row accumulator (`u32 [num_specs]`, zero-init by wgpu)
/// every transport `TAINT_ROW_OR_SHADER` dispatch atomicOrs into — the walk
/// performs ZERO mid-walk word readbacks and reads `rows_dev` ONCE at walk
/// end. `rows` is the small host-side companion holding only the WALK-BOUNDARY
/// contributions (the G13 seed-BIAS row words, folded at walk entry where the
/// seed biases are host data anyway); the final readback ORs `rows_dev` into
/// it.
struct TaintWalkState {
    wla: [wgpu::Buffer; 2],
    wua: [wgpu::Buffer; 2],
    wle: [wgpu::Buffer; 2],
    wue: [wgpu::Buffer; 2],
    ws: wgpu::Buffer,
    wprop: wgpu::Buffer,
    w_rowabs_lo: wgpu::Buffer,
    w_rowabs_hi: wgpu::Buffer,
    w_conv_reshaped: wgpu::Buffer,
    w_conv_gemm: wgpu::Buffer,
    zw: wgpu::Buffer,
    rows_dev: wgpu::Buffer,
    rows: Vec<u32>,
}

/// #fold-coalesce: bump-allocated, mapped-at-creation staging arena for the
/// per-layer uniform/slope/β/bias uploads of one fold call. Each `upload`
/// writes the bytes into the arena and encodes a `copy_buffer_to_buffer` into
/// the destination INSIDE the layer's own encoder — so under a single
/// submission each layer's passes still read that layer's values (copies and
/// passes execute in encode order). The arena MUST be unmapped (`finish`)
/// before the collected submission.
struct FoldStagingArena {
    buf: wgpu::Buffer,
    cap: u64,
    cursor: u64,
}

impl FoldStagingArena {
    fn new(device: &wgpu::Device, cap: u64) -> Self {
        let cap = cap.max(8);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("res_fold_staging"),
            size: cap,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        Self {
            buf,
            cap,
            cursor: 0,
        }
    }

    /// Stage `data` and encode its copy into `dst[0..len]`. Errors (arena
    /// overflow = a sizing bug) abort the fold, which the callers translate
    /// into the proven sound CPU fallback — fail-closed, never a wrong bound.
    fn upload(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        data: &[u8],
    ) -> Result<()> {
        let len = data.len() as u64;
        if len == 0 {
            return Ok(());
        }
        debug_assert_eq!(len % 4, 0, "fold uploads are f32/u32 arrays");
        if self.cursor + len > self.cap {
            return Err(NyError::InternalError(format!(
                "resident fold staging arena overflow: cursor {} + {} > cap {} (sizing bug)",
                self.cursor, len, self.cap
            )));
        }
        self.buf
            .slice(self.cursor..self.cursor + len)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        encoder.copy_buffer_to_buffer(&self.buf, self.cursor, dst, 0, len);
        // Keep the next mapped sub-range 8-aligned (wgpu MAP alignment).
        self.cursor = (self.cursor + len + 7) & !7;
        Ok(())
    }

    /// Unmap for submission. The returned buffer must stay alive until the
    /// collected command buffers are submitted.
    fn finish(self) -> wgpu::Buffer {
        self.buf.unmap();
        self.buf
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasParams {
    num_specs: u32,
    k: u32,
    gamma_k: f32,
    additive: f32,
    /// §0 amplified-flush combine slack (≥ 1). The bias reduction `Σ a·bias`
    /// drops `|bias|·FLT_MIN` when a subnormal `a` flushes under Metal FTZ (and
    /// the `γ_k·Σ|a·bias|` error term reads the same flushed `a` as 0), so the
    /// on-device `flushacc·slack·F32_MIN_NORMAL` term certifies it back.
    slack: f32,
    /// #eft-err (former padding): 1 ⇒ measured residual charge (·`eft_r_slack`)
    /// replaces the a-priori `γ_k·Σ|a·bias|`. 0 ⇒ byte-identical legacy.
    eft_mode: u32,
    eft_r_slack: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActParams {
    num_specs: u32,
    num_neurons: u32,
    is_upper: u32,
    additive: f32,
    /// #batched-bab: per-domain spec-row count. `num_specs_per_dom == num_specs`
    /// (single domain) → the shader's domain index is always 0 → byte-identical.
    num_specs_per_dom: u32,
    /// #eft-err (former padding): 1 ⇒ measured gap residuals + Lipschitz
    /// propagation in the activation shader. 0 ⇒ byte-identical legacy.
    eft_mode: u32,
    _p: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActBiasParams {
    num_specs: u32,
    num_neurons: u32,
    is_upper: u32,
    gamma_k: f32,
    additive: f32,
    /// §0 amplified-flush combine slack (≥ 1); see [`BiasParams::slack`]. Here the
    /// reduction `Σ a·sel_int` drops `|intercept|·FLT_MIN` on a flushed subnormal `a`.
    /// #flush-charge §E: under charged authority this uniform is widened by
    /// `FlushChargePolicy::act_bias_slack_factor` (the double-DAZ demand of the
    /// value + propagated channels; oracle
    /// `charged_act_bias_factor_covers_the_double_daz_demand`), identity when
    /// unarmed.
    slack: f32,
    /// #batched-bab: per-domain spec-row count (`== num_specs` single-domain →
    /// domain index 0 → byte-identical). Reuses a former padding slot.
    num_specs_per_dom: u32,
    /// #eft-err (former padding): 1 ⇒ measured residuals + Lipschitz intercept
    /// propagation; `gamma_k` then carries `eft_r_slack` (the γ term is unused
    /// in that mode). 0 ⇒ byte-identical legacy.
    eft_mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GradAlphaParams {
    num_specs: u32,
    num_neurons: u32,
    /// Rows per domain block for the wide/batched lane (#w4 wide α+β ascent):
    /// the shader reduces each domain's `num_specs_per_dom` rows into its own
    /// `n_domains*num_neurons` grad block. 0 = legacy single-domain (reduce all
    /// rows as one domain — byte-identical to the pre-widening kernel).
    num_specs_per_dom: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::wgpu_device) struct StridedGatherParams {
    num_specs: u32,
    num_neurons: u32,
    num_indices: u32,
    _p1: u32,
}

/// Preserve the legacy four-byte-copy implementation for genuinely small β
/// gathers. Besides avoiding a compute dispatch for a handful of values, this
/// keeps the established small-gather caller path byte-for-byte untouched.
const LEGACY_BETA_GATHER_MAX_COPIES: usize = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvErrParams {
    num_specs: u32,
    out_dim: u32,
    new_dim: u32,
    _p0: u32,
    gamma: f32,
    kernel_l1: f32,
    _p1: u32,
    _p2: u32,
}

/// Generic 4×u32 uniform for the on-device joint α-gradient elementwise shaders
/// (`JOINT_*`): interpretation depends on the shader (`(num_specs, dim, flag, _)`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JointU4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// Conv geometry uniform for the on-device joint α-gradient conv shaders
/// (`JOINT_CONV_T_FWD` forward transpose, `JOINT_CONV_ADJ` adjoint plain conv).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JointConvParams {
    num_specs: u32,
    oc: u32,
    ic: u32,
    oh: u32,
    ow: u32,
    ih: u32,
    iw: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
}

/// A frozen per-ReLU forward checkpoint for the on-device joint α-gradient adjoint:
/// the resident PRE-transform lower coefficient `A_preᵏ` (num_specs × nn) — the only
/// intermediate stored (design doc §"Intermediates to store"). σ/τ are recomputed in
/// the adjoint from `sign(A_preᵏ)` + the layer's slopes/intercepts.
struct JointReluCap {
    a_pre: wgpu::Buffer,
    nn: usize,
}

/// The downloaded coefficient frontier after a (seeded) resident backward, BEFORE
/// concretization — so callers can compose it (e.g. add a residual skip stream)
/// and concretize later. All over the final coefficient dim `dim`; bias is split
/// into center + certified error per side.
pub(crate) struct ResidentCoeff {
    pub lower_a: Vec<f32>,
    pub upper_a: Vec<f32>,
    pub lower_err: Vec<f32>,
    pub upper_err: Vec<f32>,
    pub lower_b: Vec<f32>,
    pub upper_b: Vec<f32>,
    pub lower_b_err: Vec<f32>,
    pub upper_b_err: Vec<f32>,
    pub dim: usize,
    /// Per-ReLU analytic alpha gradients captured during the backward, one entry
    /// per `Activation` layer in this chain (backward order), empty unless the
    /// caller requested capture via `relu_pre_lower` (the gradient-capable warmup
    /// path). Each is `grad[i] = pre_lower[i]·Σ_j max(A_lower[j,i],0)` over that
    /// ReLU's pre-transform lower coefficient. Non-soundness-critical (gradients
    /// only steer alpha; any alpha is a sound relaxation).
    pub relu_grads: Vec<Vec<f32>>,
    /// Per-ReLU gathered LOWER A-coefficient values at caller-requested (split)
    /// neuron columns, one entry per `Activation` layer in this chain (backward
    /// order), empty unless the caller requested capture via `beta_gather_idx`
    /// (#w4-split-tightening). Entry `r` is row-major `num_specs × idx_r.len()`
    /// read from the PRE-transform lower coefficient (the same capture point as
    /// the CPU `a_at_relu`). Non-soundness-critical (values only steer β; any
    /// β ≥ 0 is a valid Lagrangian dual).
    pub beta_gather: Vec<Vec<f32>>,
    /// #u4: per-SPEC-ROW OR of every out-of-band taint word that fed this
    /// frontier — the final coefficient/error word buffers of the taint-twin
    /// chain plus every admitted fail-closed no-twin transport (bias/intercept
    /// folds and the row-L1 flush term). Conv walks are not admitted until
    /// their internal operations have real twins. `Some(rows)`
    /// (len `num_specs`, nonzero = tainted) only when the walk — or a
    /// composition of walks: the resnet segment path ORs its sub-walks' rows
    /// across every merge/skip-add/re-seed seam, see
    /// `resnet_seeded_compose_coeff` — ran with the gate ON; `None` when the
    /// gate is off OR the frontier came through a path that genuinely cannot
    /// carry words (seg-resident device streams, which REFUSE under the gate
    /// at walk entry). `concretize_resident_coeff_batched` forwards this slice to
    /// the armed C1 consult of `concretize_sound_gpu_batched`; `None`, a wrong
    /// row count, or any nonzero word is the fail-closed refusal value
    /// (TAINT_GUARD_AUDIT.md §4 C1).
    pub taint_rows: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResidentFoldPlan {
    pub(super) num_specs_u32: u32,
    pub(super) num_specs_per_dom_u32: u32,
    pub(super) n_domains: usize,
    pub(super) seed_elems: usize,
    pub(super) final_dim: usize,
    pub(super) max_dim: usize,
    pub(super) max_gemm_out: usize,
    pub(super) a_elems: usize,
    pub(super) slope_dim: usize,
    pub(super) max_wg: usize,
}

fn resident_checked_product(parts: &[usize], label: &str) -> Result<usize> {
    parts.iter().try_fold(1usize, |product, &part| {
        product.checked_mul(part).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "crown_backward_sound_resident: {label} overflows usize"
            ))
        })
    })
}

fn resident_checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: {label}={value} exceeds u32"
        ))
    })
}

// ---------------------------------------------------------------------------
// #cert-err — charging a caller-declared `CertifiedWeightError` into the walk
// ---------------------------------------------------------------------------
//
// The walk's per-layer coefficient step ships `A_new = fl(A @ W)` with the
// certified radius the AW-error combine writes:
//
//     err_new = round_up( (gamma·S + P)·slack + flush ),
//         S = fl(|A| @ |W|),   P = fl(err @ |W|)
//
// That is an enclosure of `A* @ W` — the exact predecessor coefficient folded
// through the SUPPLIED weight. When the caller declares
// `weight_rel_err = w` (`|W* − W| <= w·|W|` elementwise), the quantity that must
// be enclosed is `A* @ W*`, and (see `CertifiedWeightError::charged_gamma` for
// the full derivation)
//
//     |A*@W* − fl(A@W)|  <=  ( (gamma + w)·|A| + (1 + w)·err ) @ |W|.
//
// Two host-only uniform substitutions realise exactly that, with NO shader
// change (so the kernels stay drift-pinned and the zero-`cert_err` walk stays
// byte-identical):
//
//   1. `gamma_k := g = gamma + w + gamma·w`  — `g >= gamma + w`, so `g·S`
//      dominates the `(gamma + w)·|A| @ |W|` term (the `gamma·w` cross term
//      absorbs the rounding of forming `g` itself);
//   2. `slack := slack·(1 + w)`  — the combine multiplies the WHOLE
//      `(g·S + P)` sum by `slack`, so scaling `slack` by `(1 + w)` yields
//      `>= g·S·slack + (1+w)·P·slack`, which dominates the bound above term by
//      term (`slack >= 1`). The base `slack` already carries `1/(1-gamma_k)`
//      for the two GEMMs' undercount plus `(1+u)^4` for the combine's own four
//      f32 ops, and multiplying it scales that coverage with the charge.
//
// BOTH substitutions are the IDENTITY at `w = 0`: `charged_gamma(gamma)` returns
// `gamma`'s exact bits, and `up_f32(slack · 1.0)` returns `slack`'s exact bits.
// That is what makes the default (exact-weight) walk byte-identical — pinned by
// `zero_cert_err_charges_are_byte_identical`.
//
// The bias side is charged by a SECOND dispatch of the unmodified bias kernel
// (see `cert_bias_charge_slack`), not by a uniform substitution.

/// The `CertifiedWeightError` a layer declares (`Default` = exact for every
/// variant that cannot carry one).
fn layer_cert_err(layer: &GpuCrownLayer) -> CertifiedWeightError {
    match layer {
        GpuCrownLayer::Linear { cert_err, .. } | GpuCrownLayer::Conv2d { cert_err, .. } => {
            *cert_err
        }
        _ => CertifiedWeightError::default(),
    }
}

/// Whether ANY layer declares a nonzero absolute bias error, i.e. whether this
/// walk must allocate and dispatch the extra bias-error charge below. `false`
/// (the default for every existing caller) allocates nothing and dispatches
/// nothing, so the walk is byte-identical to the pre-`cert_err` build.
fn cert_bias_charge_required(layers: &[GpuCrownLayer]) -> bool {
    layers
        .iter()
        .any(|layer| layer_cert_err(layer).bias_abs_err != 0.0)
}

/// #flush-charge: any nonzero value below the smallest normal f32.
fn slice_has_subnormal(values: &[f32]) -> bool {
    values
        .iter()
        .any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE)
}

/// #flush-charge: walk-entry admission guard for a CHARGED-flush device
/// (`FlushChargePolicy`, `ops/sound_authority.rs`). Every refusal here is a
/// typed fail-closed error, never a silent downgrade:
///
/// * the EFT compensated channel is FORBIDDEN (`eft_primitives_cached()` is
///   already false on a flushing adapter; this pins the intent against a
///   future gate edit);
/// * only Linear / Activation / Conv2d layers are admitted — every other kind
///   (maxpool selection logic, dual-alpha routing, ...) carries un-audited
///   comparison/selection behavior under DAZ;
/// * `cert_err` layers are refused (the certified weight-error charge is not
///   audited against the flush model);
/// * subnormal BIAS entries are refused — the bias combine's `err·|b|` channel
///   loses `err·2^-126` when `b` is DAZ-zeroed and no uniform scales with err;
/// * subnormal SLOPES and INTERCEPTS are refused — the elementwise `μ·|a|`
///   amplification channel and the rung-5 `b != 0` annihilation consult under
///   a DAZ compare (`shaders_taint`) are both closed by this one predicate.
///   For INTERCEPTS the refusal is PERMANENT: the §E oracle proves the
///   subnormal-intercept channel unchargeable — the loss scales with the
///   runtime radius `err`, which no intercept-bias uniform carries
///   (`flush_charge_oracle::charged_act_bias_cannot_cover_subnormal_intercepts`).
///   Nonzero NORMAL intercepts are ADMITTED: their double-DAZ demand is paid
///   by the charged `ActBiasParams.slack` widening
///   (`FlushChargePolicy::act_bias_slack_factor`, oracle
///   `charged_act_bias_factor_covers_the_double_daz_demand`).
///
/// Unreachable on every non-charged device (the caller only invokes it when
/// `charged_flush_authority_cached()` returned a policy).
fn charged_walk_guard(
    layers: &[GpuCrownLayer],
    policy: &super::sound_authority::FlushChargePolicy,
    eft_requested: bool,
) -> Result<()> {
    if policy.eft_forbidden && eft_requested {
        return Err(NyError::UnsupportedOp(
            "#flush-charge: NY_EFT_ERR=1 is refused under charged-flush \
             authority — the compensated channel measures the residuals this \
             adapter flushes (fail-closed)"
                .into(),
        ));
    }
    for (index, layer) in layers.iter().enumerate() {
        if !layer_cert_err(layer).is_exact() {
            return Err(NyError::UnsupportedOp(format!(
                "#flush-charge: layer {index} declares a CertifiedWeightError, \
                 which is not audited against the flush-charge model — \
                 refusing (fail-closed)"
            )));
        }
        match layer {
            GpuCrownLayer::Linear { bias, .. } => {
                if policy.refuse_subnormal_bias {
                    if let Some(b) = bias {
                        if slice_has_subnormal(b) {
                            return Err(NyError::UnsupportedOp(format!(
                                "#flush-charge: layer {index} has a SUBNORMAL \
                                 bias entry; on a DAZ adapter its loss in the \
                                 bias combine is bounded only by err·2^-126, \
                                 which no uniform carries — refusing"
                            )));
                        }
                    }
                }
            }
            GpuCrownLayer::Conv2d { bias_expanded, .. } => {
                if policy.refuse_subnormal_bias {
                    if let Some(b) = bias_expanded {
                        if slice_has_subnormal(b) {
                            return Err(NyError::UnsupportedOp(format!(
                                "#flush-charge: layer {index} has a SUBNORMAL \
                                 expanded conv bias entry — refusing \
                                 (see the bias-combine flush audit)"
                            )));
                        }
                    }
                }
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                ..
            } => {
                if policy.refuse_subnormal_slopes
                    && (slice_has_subnormal(lower_slope)
                        || slice_has_subnormal(upper_slope)
                        || slice_has_subnormal(lower_intercept)
                        || slice_has_subnormal(upper_intercept))
                {
                    return Err(NyError::UnsupportedOp(format!(
                        "#flush-charge: layer {index} has a SUBNORMAL \
                         activation slope/intercept; the DAZ amplification and \
                         taint-annihilation channels for it are refused, not \
                         charged — refusing (fail-closed)"
                    )));
                }
                // #flush-charge §E (landed 2026-08-13): nonzero NORMAL
                // intercepts are ADMITTED under the charge — the double-DAZ
                // shape (a runtime coefficient and a runtime err, neither
                // host-refusable, against a NORMAL intercept) is paid by the
                // widened `ActBiasParams.slack`
                // (`FlushChargePolicy::act_bias_slack_factor`, oracle
                // `charged_act_bias_factor_covers_the_double_daz_demand`).
                // SUBNORMAL intercepts stay refused by the predicate above:
                // that channel's loss scales with the runtime radius `err`,
                // which no intercept-bias uniform carries — proven
                // unchargeable by
                // `charged_act_bias_cannot_cover_subnormal_intercepts`.
            }
            _ => {
                return Err(NyError::UnsupportedOp(format!(
                    "#flush-charge: layer {index} is not an admitted \
                     charged-mode layer kind (Linear/Activation/Conv2d only); \
                     its selection/comparison behavior under DAZ is un-audited \
                     — refusing (fail-closed)"
                )));
            }
        }
    }
    Ok(())
}

/// #flush-charge: widen a bias-combine `slack` uniform under an armed charge
/// policy; the identity everywhere else (byte-identical dark path).
fn charged_bias_slack_or(
    policy: Option<&super::sound_authority::FlushChargePolicy>,
    slack: f32,
) -> Result<f32> {
    match policy {
        Some(p) => {
            crate::wgpu_device::sound_consts::charged_bias_slack(slack, p.bias_combine_factor)
        }
        None => Ok(slack),
    }
}

/// #flush-charge §E: widen the activation intercept-bias `slack` uniform
/// (`ActBiasParams.slack`) under an armed charge policy; the identity
/// everywhere else (byte-identical dark path — pinned by
/// `charged_act_bias_slack_is_identity_when_unarmed_and_outward_when_armed`).
fn charged_act_bias_slack_or(
    policy: Option<&super::sound_authority::FlushChargePolicy>,
    slack: f32,
) -> Result<f32> {
    match policy {
        Some(p) => {
            crate::wgpu_device::sound_consts::charged_act_bias_slack(slack, p.act_bias_slack_factor)
        }
        None => Ok(slack),
    }
}

/// The combine `slack` charged by a layer's declared relative weight error
/// (substitution 2 above): `slack · (1 + weight_rel_err)`.
///
/// It is the `(1 + w)` on the PROPAGATED error term — NOT `(1 + g)`. The `gamma`
/// part of `g` is already carried by `gamma_k·S`; multiplying it in here as well
/// would (harmlessly but pointlessly) widen every exact-weight walk and destroy
/// the byte-identity pin.
///
/// `base` and `w` are f32, so `base·(1+w)` is computed exactly in f64 and only
/// the f32 narrowing rounds — UPWARD, keeping the factor `>= base·(1+w)`. At
/// `w = 0` the product is exactly `base`, which round-trips to `base`'s own
/// bits. An invalid declaration or a non-finite product is a refusal, never a
/// saturated (finite, under-charging) substitute.
fn cert_charged_slack(base: f32, cert_err: CertifiedWeightError, index: usize) -> Result<f32> {
    if !base.is_finite() || base < 0.0 || !cert_err.is_valid() {
        return Err(NyError::UnsupportedOp(format!(
            "#cert-err: layer {index} has no finite charged combine slack \
             (base={base:e}, weight_rel_err={:e}) — refusing (fail-closed)",
            cert_err.weight_rel_err
        )));
    }
    // Review defect 1 (sibling of ny-core's charged_gamma): `1 + w` needs up
    // to 47 significand bits and the product up to ~71, so the f64 multiply
    // rounds to NEAREST and can land below `base·(1+w)`; if it lands on the
    // f32 grid, `up_f32` does not bump and the shipped slack is too small.
    // For `w < 2^-53` the `1.0 + w` itself collapses to 1.0 and the charge
    // vanishes silently. Exact weights return `base` untouched (byte-identity
    // contract); otherwise every step rounds OUTWARD.
    let charged = if cert_err.is_exact() {
        base
    } else {
        let factor = next_up_f64(1.0 + f64::from(cert_err.weight_rel_err));
        up_f32(next_up_f64(f64::from(base) * factor))
    };
    if !charged.is_finite() {
        return Err(NyError::UnsupportedOp(format!(
            "#cert-err: layer {index} charged combine slack overflows f32 \
             (base={base:e}, weight_rel_err={:e}) — refusing (fail-closed)",
            cert_err.weight_rel_err
        )));
    }
    Ok(charged)
}

/// The `slack` for the EXTRA bias-error dispatch that charges `bias_abs_err`.
///
/// # What that dispatch computes and why it is exactly the missing term
///
/// The layer's bias fold ships `b_new = b_old + fl(Σ_j a_j·bias_j)` with the
/// radius `round_up(round_up(gamma_k·Σ|a_j·bias_j| + Σ err_j·|bias_j|)·slack)`.
/// That encloses the fold of the SUPPLIED bias. For the exact bias `b*` with
/// `|b*_j − bias_j| <= d` (`d = bias_abs_err`) the missing term is
///
/// ```text
/// |Σ a*_j·b*_j − Σ a_j·bias_j|  −  (already charged)
///     <=  d · ( Σ|a_j| + Σ err_j ).
/// ```
///
/// Re-dispatching the SAME kernel with `bias := [d; k]` and `gamma_k := 1`
/// computes `round_up(round_up(1·Σ|a_j·d| + Σ err_j·d)·slack)`, i.e. exactly
/// `d·(Σ|a_j| + Σ err_j)` recovered outward — and accumulates it into the same
/// `bias_err_out`, which the kernel updates with `+=`. Its `bias_out` binding is
/// pointed at a throwaway sink so the CENTER bias is untouched.
///
/// # Why the slack must be widened here
///
/// In the ordinary dispatch `gamma_k·Σ|a·bias|` is a SECOND-order correction, so
/// `combine_slack_f32(k)`'s recovery of the reduction's own undercount is ample.
/// Here the same reduction carries the FIRST-order term, and it absorbs `k`
/// product roundings plus `k−1` tree adds on each of the two lanes. Charging
/// `combine_slack_f32(2k + 4) >= 1/(1 − gamma_{2k+4})` dominates every one of
/// those roundings with room to spare; the factor is `1 + O(k·u)` so the cost is
/// nil, and an over-long reduction fails closed inside `combine_slack_f32`.
fn cert_bias_charge_slack(k: usize) -> Result<f32> {
    let terms = k
        .checked_mul(2)
        .and_then(|t| t.checked_add(4))
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "#cert-err: bias charge reduction length 2*{k}+4 overflows — refusing"
            ))
        })?;
    combine_slack_f32(terms)
}

/// Operands for one [`WgpuDevice::cert_bias_charge_pass`] dispatch pair
/// (lower + upper side). Grouped into a struct only to keep the call under the
/// argument-count lint.
struct CertBiasChargeArgs<'a> {
    /// The layer's declaration; only `bias_abs_err` is read here.
    cert_err: CertifiedWeightError,
    /// Reduction length `k` of the bias fold (`out_features`, or the conv's
    /// expanded `in_d`).
    reduction: usize,
    num_specs: usize,
    /// Index of the layer in the walk, for refusal messages only.
    layer_index: usize,
    /// The dedicated `BiasParams` uniform; `None` when the walk allocated no
    /// charge buffers (which is a refusal if a charge is actually due).
    params: Option<&'a wgpu::Buffer>,
    /// The constant `[bias_abs_err; k]` operand buffer.
    operand: Option<&'a wgpu::Buffer>,
    /// Throwaway centre-bias output; nothing reads it.
    sink: Option<&'a wgpu::Buffer>,
    /// `[lower, upper]` incoming coefficient buffers.
    a: [&'a wgpu::Buffer; 2],
    /// `[lower, upper]` incoming coefficient-error buffers.
    a_err: [&'a wgpu::Buffer; 2],
    /// `[lower, upper]` bias-error accumulators to charge into.
    bias_err_out: [&'a wgpu::Buffer; 2],
}

fn resident_f32_bytes(elements: usize, label: &str) -> Result<u64> {
    elements
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "crown_backward_sound_resident: {label} byte count overflows"
            ))
        })
}

fn resident_check_gemm_dispatch(
    m: usize,
    k: usize,
    n: usize,
    max_wg: usize,
    label: &str,
    batched_domains: bool,
) -> Result<()> {
    let dispatch = select_gemm_dispatch(
        resident_checked_u32(m, &format!("{label} m"))?,
        resident_checked_u32(k, &format!("{label} k"))?,
        resident_checked_u32(n, &format!("{label} n"))?,
    );
    if dispatch.wg_x as usize > max_wg || dispatch.wg_y as usize > max_wg {
        if batched_domains {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: (dispatch.wg_x as usize).max(dispatch.wg_y as usize),
                capacity: max_wg,
                unit: "workgroups",
                site: "resident GEMM preflight",
            });
        }
        return Err(NyError::UnsupportedOp(format!(
            "crown_backward_sound_resident: {label} dispatch ({}, {}) exceeds \
             max_compute_workgroups_per_dimension {max_wg}",
            dispatch.wg_x, dispatch.wg_y
        )));
    }
    Ok(())
}

fn resident_conv_output_extent(
    input: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    axis: &str,
    layer_index: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: conv layer {layer_index} has zero {axis} stride"
        )));
    }
    let double_pad = pad.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: conv layer {layer_index} padded {axis} overflows"
        ))
    })?;
    let padded = input.checked_add(double_pad).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: conv layer {layer_index} padded {axis} overflows"
        ))
    })?;
    let available = padded.checked_sub(kernel).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: conv layer {layer_index} {axis} kernel {kernel} \
             exceeds padded input {padded}"
        ))
    })?;
    available
        .checked_div(stride)
        .and_then(|steps| steps.checked_add(1))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "crown_backward_sound_resident: conv layer {layer_index} output {axis} overflows"
            ))
        })
}

/// Validate every dimension that is later multiplied, cast into a WGSL `u32`,
/// used as a storage binding, or used to size a dispatch. This runs before the
/// first allocation/submission, so malformed metadata cannot wrap into a small
/// apparently-valid shader uniform.
pub(super) fn resident_fold_plan(
    layers: &[GpuCrownLayer],
    num_specs: usize,
    num_specs_per_dom: usize,
    output_dim: usize,
    max_compute_workgroups_per_dimension: u32,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<ResidentFoldPlan> {
    if num_specs == 0 {
        return Err(NyError::InvalidSpec(
            "crown_backward_sound_resident: num_specs must be nonzero".into(),
        ));
    }
    if output_dim == 0 {
        return Err(NyError::InvalidSpec(
            "crown_backward_sound_resident: output_dim must be nonzero".into(),
        ));
    }
    if num_specs_per_dom == 0
        || num_specs_per_dom > num_specs
        || !num_specs.is_multiple_of(num_specs_per_dom)
    {
        return Err(NyError::InvalidSpec(format!(
            "crown_backward_sound_resident: num_specs_per_dom={num_specs_per_dom} \
             must be nonzero and divide num_specs={num_specs}"
        )));
    }

    let num_specs_u32 = resident_checked_u32(num_specs, "num_specs")?;
    let num_specs_per_dom_u32 = resident_checked_u32(num_specs_per_dom, "num_specs_per_dom")?;
    let n_domains = num_specs / num_specs_per_dom;
    let seed_elems =
        resident_checked_product(&[num_specs, output_dim], "seed coefficient elements")?;
    resident_checked_u32(seed_elems, "seed coefficient elements")?;

    if max_compute_workgroups_per_dimension == 0 {
        return Err(NyError::UnsupportedOp(
            "crown_backward_sound_resident: device reports zero \
             max_compute_workgroups_per_dimension"
                .into(),
        ));
    }
    let max_wg = max_compute_workgroups_per_dimension as usize;
    let mut cur = output_dim;
    let mut max_dim = output_dim;
    let mut max_gemm_out = 1usize;
    let mut max_storage_elems = seed_elems.max(num_specs).max(output_dim);

    resident_checked_u32(output_dim, "output_dim")?;

    for (layer_index, layer) in layers.iter().enumerate() {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                if *out_features == 0 || *in_features == 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: linear layer {layer_index} has a \
                         zero dimension ({out_features}x{in_features})"
                    )));
                }
                if *out_features != cur {
                    return Err(NyError::shape_mismatch(vec![cur], vec![*out_features]));
                }
                resident_checked_u32(*out_features, "linear out_features")?;
                resident_checked_u32(*in_features, "linear in_features")?;
                let weight_elems = resident_checked_product(
                    &[*out_features, *in_features],
                    "linear weight elements",
                )?;
                if weight.len() != weight_elems {
                    return Err(NyError::shape_mismatch(
                        vec![*out_features, *in_features],
                        vec![weight.len()],
                    ));
                }
                if let Some(values) = bias {
                    if values.len() != *out_features {
                        return Err(NyError::shape_mismatch(
                            vec![*out_features],
                            vec![values.len()],
                        ));
                    }
                }
                let incoming =
                    resident_checked_product(&[num_specs, *out_features], "linear input rows")?;
                let outgoing =
                    resident_checked_product(&[num_specs, *in_features], "linear output rows")?;
                resident_checked_u32(incoming, "linear input elements")?;
                resident_checked_u32(outgoing, "linear output elements")?;
                resident_check_gemm_dispatch(
                    num_specs,
                    *out_features,
                    *in_features,
                    max_wg,
                    "linear GEMM",
                    n_domains > 1,
                )?;
                resident_check_gemm_dispatch(
                    num_specs,
                    *out_features,
                    1,
                    max_wg,
                    "linear row-L1 GEMM",
                    n_domains > 1,
                )?;
                max_dim = max_dim.max(*in_features);
                max_storage_elems = max_storage_elems
                    .max(weight_elems)
                    .max(incoming)
                    .max(outgoing);
                cur = *in_features;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                if *num_neurons == 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: activation layer {layer_index} has \
                         zero neurons"
                    )));
                }
                if *num_neurons != cur {
                    return Err(NyError::shape_mismatch(vec![cur], vec![*num_neurons]));
                }
                resident_checked_u32(*num_neurons, "activation num_neurons")?;
                let state_elems = resident_checked_product(
                    &[n_domains, *num_neurons],
                    "activation domain-state elements",
                )?;
                for (name, actual) in [
                    ("lower_slope", lower_slope.len()),
                    ("upper_slope", upper_slope.len()),
                    ("lower_intercept", lower_intercept.len()),
                    ("upper_intercept", upper_intercept.len()),
                ] {
                    if actual != state_elems {
                        return Err(NyError::InvalidSpec(format!(
                            "crown_backward_sound_resident: activation layer {layer_index} \
                             {name}.len()={actual} != n_domains*num_neurons={state_elems}"
                        )));
                    }
                }
                let coefficient_elems = resident_checked_product(
                    &[num_specs, *num_neurons],
                    "activation coefficient elements",
                )?;
                resident_checked_u32(coefficient_elems, "activation coefficient elements")?;
                max_storage_elems = max_storage_elems.max(state_elems).max(coefficient_elems);
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
                ..
            } => {
                if [
                    *out_channels,
                    *in_channels,
                    *kernel_h,
                    *kernel_w,
                    *stride_h,
                    *stride_w,
                    *out_h,
                    *out_w,
                    *in_h,
                    *in_w,
                ]
                .contains(&0)
                {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: conv layer {layer_index} has a \
                         zero channel/kernel/stride/spatial dimension"
                    )));
                }
                for (name, value) in [
                    ("out_channels", *out_channels),
                    ("in_channels", *in_channels),
                    ("kernel_h", *kernel_h),
                    ("kernel_w", *kernel_w),
                    ("stride_h", *stride_h),
                    ("stride_w", *stride_w),
                    ("pad_h", *pad_h),
                    ("pad_w", *pad_w),
                    ("out_h", *out_h),
                    ("out_w", *out_w),
                    ("in_h", *in_h),
                    ("in_w", *in_w),
                ] {
                    resident_checked_u32(value, &format!("conv {name}"))?;
                }
                let expected_out_h = resident_conv_output_extent(
                    *in_h,
                    *kernel_h,
                    *stride_h,
                    *pad_h,
                    "height",
                    layer_index,
                )?;
                let expected_out_w = resident_conv_output_extent(
                    *in_w,
                    *kernel_w,
                    *stride_w,
                    *pad_w,
                    "width",
                    layer_index,
                )?;
                if (*out_h, *out_w) != (expected_out_h, expected_out_w) {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: conv layer {layer_index} output \
                         geometry ({out_h},{out_w}) != expected \
                         ({expected_out_h},{expected_out_w}) for input ({in_h},{in_w}), \
                         kernel ({kernel_h},{kernel_w}), stride ({stride_h},{stride_w}), \
                         padding ({pad_h},{pad_w})"
                    )));
                }
                let spatial = resident_checked_product(&[*out_h, *out_w], "conv output spatial")?;
                let in_d = resident_checked_product(
                    &[*out_channels, spatial],
                    "conv entering coefficient dimension",
                )?;
                let out_d = resident_checked_product(
                    &[*in_channels, *in_h, *in_w],
                    "conv exiting coefficient dimension",
                )?;
                if in_d != cur {
                    return Err(NyError::shape_mismatch(vec![cur], vec![in_d]));
                }
                let kernel_cols = resident_checked_product(
                    &[*in_channels, *kernel_h, *kernel_w],
                    "conv kernel columns",
                )?;
                let weight_elems = resident_checked_product(
                    &[*out_channels, kernel_cols],
                    "conv weight elements",
                )?;
                if weight_col.len() != weight_elems {
                    return Err(NyError::shape_mismatch(
                        vec![*out_channels, kernel_cols],
                        vec![weight_col.len()],
                    ));
                }
                if let Some(values) = bias_expanded {
                    if values.len() != in_d {
                        return Err(NyError::shape_mismatch(vec![in_d], vec![values.len()]));
                    }
                }
                let gemm_rows = resident_checked_product(&[num_specs, spatial], "conv GEMM rows")?;
                let gemm_out =
                    resident_checked_product(&[gemm_rows, kernel_cols], "conv GEMM output")?;
                let incoming = resident_checked_product(&[num_specs, in_d], "conv input elements")?;
                let outgoing =
                    resident_checked_product(&[num_specs, out_d], "conv output elements")?;
                for (name, value) in [
                    ("conv entering dimension", in_d),
                    ("conv exiting dimension", out_d),
                    ("conv spatial", spatial),
                    ("conv kernel columns", kernel_cols),
                    ("conv GEMM rows", gemm_rows),
                    ("conv GEMM output elements", gemm_out),
                    ("conv input elements", incoming),
                    ("conv output elements", outgoing),
                ] {
                    resident_checked_u32(value, name)?;
                }
                resident_check_gemm_dispatch(
                    gemm_rows,
                    *out_channels,
                    kernel_cols,
                    max_wg,
                    "conv GEMM",
                    n_domains > 1,
                )?;
                resident_check_gemm_dispatch(
                    num_specs,
                    in_d,
                    1,
                    max_wg,
                    "conv row-L1 GEMM",
                    n_domains > 1,
                )?;
                max_dim = max_dim.max(out_d);
                max_gemm_out = max_gemm_out.max(gemm_out);
                max_storage_elems = max_storage_elems
                    .max(weight_elems)
                    .max(incoming)
                    .max(outgoing)
                    .max(gemm_out);
                cur = out_d;
            }
            _ => {
                return Err(NyError::UnsupportedOp(
                    "crown_backward_sound_resident R4: Linear/Activation/Conv2d only".into(),
                ));
            }
        }
    }

    let final_dim = cur;
    let a_elems =
        resident_checked_product(&[num_specs, max_dim], "coefficient workspace elements")?;
    let slope_dim =
        resident_checked_product(&[n_domains, max_dim], "activation workspace elements")?;
    resident_checked_u32(a_elems, "coefficient workspace elements")?;
    resident_checked_u32(slope_dim, "activation workspace elements")?;
    max_storage_elems = max_storage_elems
        .max(a_elems)
        .max(slope_dim)
        .max(max_gemm_out);

    let worst_1d = num_specs.max(a_elems.div_ceil(256));
    if worst_1d > max_wg {
        if n_domains > 1 {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: worst_1d,
                capacity: max_wg,
                unit: "workgroups",
                site: "resident 1-D dispatch preflight",
            });
        }
        return Err(NyError::UnsupportedOp(format!(
            "crown_backward_sound_resident: 1-D dispatch {worst_1d} exceeds \
             max_compute_workgroups_per_dimension {max_wg} (num_specs={num_specs}, \
             width={max_dim}) — sub-chunk the batch"
        )));
    }

    let storage_bytes = resident_f32_bytes(max_storage_elems.max(1), "largest storage buffer")?;
    if storage_bytes > max_buffer_size || storage_bytes > max_storage_buffer_binding_size {
        if n_domains > 1 {
            let capacity = max_buffer_size.min(max_storage_buffer_binding_size);
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: usize::try_from(storage_bytes).unwrap_or(usize::MAX),
                capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
                unit: "bytes",
                site: "resident storage-binding preflight",
            });
        }
        return Err(NyError::UnsupportedOp(format!(
            "crown_backward_sound_resident: largest storage binding needs {storage_bytes} \
             bytes, but max_buffer_size={max_buffer_size} and \
             max_storage_buffer_binding_size={max_storage_buffer_binding_size}"
        )));
    }

    Ok(ResidentFoldPlan {
        num_specs_u32,
        num_specs_per_dom_u32,
        n_domains,
        seed_elems,
        final_dim,
        max_dim,
        max_gemm_out,
        a_elems,
        slope_dim,
        max_wg,
    })
}

/// Conservative sum of every per-layer upload that can remain queued before
/// the resident walk's final readback drains the device.
pub(super) fn resident_fold_staging_capacity(
    layers: &[GpuCrownLayer],
    n_domains: usize,
) -> Result<u64> {
    let mut capacity = 4096u64;
    for layer in layers {
        let upload_elems = match layer {
            GpuCrownLayer::Activation { num_neurons, .. } => resident_checked_product(
                &[6, n_domains, *num_neurons],
                "fold staging activation elements",
            )?,
            GpuCrownLayer::Linear {
                bias,
                out_features,
                cert_err,
                ..
            } => bias
                .as_ref()
                .map_or(0, |values| values.len())
                .checked_add(if cert_err.bias_abs_err != 0.0 {
                    (*out_features).max(1)
                } else {
                    0
                })
                .ok_or_else(|| {
                    NyError::InvalidSpec(
                        "crown_backward_sound_resident: linear fold staging overflows".into(),
                    )
                })?,
            GpuCrownLayer::Conv2d {
                bias_expanded,
                out_channels,
                out_h,
                out_w,
                cert_err,
                ..
            } => {
                let cert_bias_elems = if cert_err.bias_abs_err != 0.0 {
                    resident_checked_product(
                        &[*out_channels, *out_h, *out_w],
                        "conv certified-bias staging elements",
                    )?
                    .max(1)
                } else {
                    0
                };
                bias_expanded
                    .as_ref()
                    .map_or(0, |values| values.len())
                    .checked_add(cert_bias_elems)
                    .ok_or_else(|| {
                        NyError::InvalidSpec(
                            "crown_backward_sound_resident: conv fold staging overflows".into(),
                        )
                    })?
            }
            _ => 0,
        };
        let upload_bytes = resident_f32_bytes(upload_elems, "fold staging layer upload")?;
        capacity = capacity
            .checked_add(1024)
            .and_then(|value| value.checked_add(upload_bytes))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "crown_backward_sound_resident: fold staging capacity overflows".into(),
                )
            })?;
    }
    Ok(capacity)
}

fn joint_segment_preflight(
    segments: &[GpuResnetSegment],
    num_specs: usize,
    output_dim: usize,
    max_compute_workgroups_per_dimension: u32,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<usize> {
    let chain = |layers: &[GpuCrownLayer], start_dim: usize| {
        resident_fold_plan(
            layers,
            num_specs,
            num_specs,
            start_dim,
            max_compute_workgroups_per_dimension,
            max_buffer_size,
            max_storage_buffer_binding_size,
        )
        .map(|plan| plan.final_dim)
    };

    let mut dim = output_dim;
    for segment in segments {
        dim = match segment {
            GpuResnetSegment::Chain(layers) => chain(layers, dim)?,
            GpuResnetSegment::Residual(branch) => {
                let branch_dim = chain(branch, dim)?;
                if branch_dim != dim {
                    return Err(NyError::shape_mismatch(vec![dim], vec![branch_dim]));
                }
                dim
            }
            GpuResnetSegment::ResidualProj(branch, projection) => {
                let branch_dim = chain(branch, dim)?;
                let projection_dim = chain(projection, dim)?;
                if branch_dim != projection_dim {
                    return Err(NyError::shape_mismatch(
                        vec![branch_dim],
                        vec![projection_dim],
                    ));
                }
                branch_dim
            }
        };
    }
    Ok(dim)
}

/// Validate every resident fold in a multi-domain coefficient request before
/// the first GPU submission. A later segment can require a larger workspace
/// than the first; discovering that capacity failure inside the composition
/// loop would make narrowing retry re-issue already-completed GPU work.
fn batched_coefficient_segment_preflight(
    segments: &[GpuResnetSegment],
    num_specs: usize,
    num_specs_per_dom: usize,
    output_dim: usize,
    max_compute_workgroups_per_dimension: u32,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<usize> {
    let chain = |layers: &[GpuCrownLayer], start_dim: usize| {
        resident_fold_plan(
            layers,
            num_specs,
            num_specs_per_dom,
            start_dim,
            max_compute_workgroups_per_dimension,
            max_buffer_size,
            max_storage_buffer_binding_size,
        )
        .map(|plan| plan.final_dim)
    };

    let mut dim = output_dim;
    for segment in segments {
        dim = match segment {
            GpuResnetSegment::Chain(layers) => chain(layers, dim)?,
            GpuResnetSegment::Residual(branch) => {
                let branch_dim = chain(branch, dim)?;
                if branch_dim != dim {
                    return Err(NyError::shape_mismatch(vec![dim], vec![branch_dim]));
                }
                dim
            }
            GpuResnetSegment::ResidualProj(branch, projection) => {
                let branch_dim = chain(branch, dim)?;
                let projection_dim = chain(projection, dim)?;
                if branch_dim != projection_dim {
                    return Err(NyError::shape_mismatch(
                        vec![branch_dim],
                        vec![projection_dim],
                    ));
                }
                branch_dim
            }
        };
    }
    Ok(dim)
}

#[cfg(test)]
mod charged_walk_guard_tests {
    use ny_core::{CertifiedWeightError, GpuCrownLayer};

    use super::super::sound_authority::FlushChargePolicy;
    use super::{charged_act_bias_slack_or, charged_bias_slack_or, charged_walk_guard};

    const SUBNORMAL: f32 = 1.0e-45;

    fn linear(bias: Option<Vec<f32>>, cert_err: CertifiedWeightError) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: vec![1.0f32; 4].into(),
            bias: bias.map(Into::into),
            out_features: 2,
            in_features: 2,
            cert_err,
        }
    }

    fn activation(lower_slope: Vec<f32>) -> GpuCrownLayer {
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope: vec![1.0; 2],
            lower_intercept: vec![0.0; 2],
            upper_intercept: vec![0.0; 2],
            num_neurons: 2,
        }
    }

    #[test]
    fn charged_walk_guard_admits_clean_layers_and_refuses_each_channel() {
        let policy = FlushChargePolicy::production();
        let clean = [
            linear(Some(vec![1.0, 0.0]), CertifiedWeightError::default()),
            activation(vec![0.5, 1.0]),
        ];
        assert!(charged_walk_guard(&clean, &policy, false).is_ok());

        // The EFT compensated channel is forbidden outright.
        assert!(charged_walk_guard(&clean, &policy, true).is_err());
        // A subnormal bias entry refuses (the err·|b| channel has no cover).
        let sub_bias = [linear(
            Some(vec![1.0, SUBNORMAL]),
            CertifiedWeightError::default(),
        )];
        assert!(charged_walk_guard(&sub_bias, &policy, false).is_err());
        // A subnormal slope refuses (amplification + rung-5 annihilation).
        let sub_slope = [activation(vec![SUBNORMAL, 1.0])];
        assert!(charged_walk_guard(&sub_slope, &policy, false).is_err());
        // #flush-charge §E (landed): a nonzero NORMAL intercept is ADMITTED —
        // its double-DAZ demand is paid by the charged `ActBiasParams.slack`
        // widening (`charged_act_bias_factor_covers_the_double_daz_demand`).
        let nonzero_intercept = [GpuCrownLayer::Activation {
            lower_slope: vec![0.5, 1.0],
            upper_slope: vec![1.0; 2],
            lower_intercept: vec![0.0; 2],
            upper_intercept: vec![0.5, 0.0],
            num_neurons: 2,
        }];
        assert!(charged_walk_guard(&nonzero_intercept, &policy, false).is_ok());
        // A SUBNORMAL intercept stays refused PERMANENTLY: the §E oracle
        // proves that channel unchargeable (the loss scales with the runtime
        // radius `err`, which no intercept-bias uniform carries —
        // `charged_act_bias_cannot_cover_subnormal_intercepts`).
        let subnormal_intercept = [GpuCrownLayer::Activation {
            lower_slope: vec![0.5, 1.0],
            upper_slope: vec![1.0; 2],
            lower_intercept: vec![0.0; 2],
            upper_intercept: vec![SUBNORMAL, 0.0],
            num_neurons: 2,
        }];
        assert!(charged_walk_guard(&subnormal_intercept, &policy, false).is_err());
        // A declared CertifiedWeightError is un-audited under the flush model.
        let cert = [linear(
            None,
            CertifiedWeightError {
                weight_rel_err: 1.0e-6,
                bias_abs_err: 0.0,
            },
        )];
        assert!(charged_walk_guard(&cert, &policy, false).is_err());
        // A non-admitted layer kind refuses.
        let dual_alpha = [GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: vec![1.0; 2],
            cross_slope: vec![0.5; 2],
            upper_neg_slope: vec![1.0; 2],
            cross_intercept: vec![0.0; 2],
            num_neurons: 2,
        }];
        assert!(charged_walk_guard(&dual_alpha, &policy, false).is_err());
    }

    #[test]
    fn charged_bias_slack_is_identity_when_unarmed_and_outward_when_armed() {
        let policy = FlushChargePolicy::production();
        for slack in [1.0f32, 1.000_001, 2.5] {
            assert_eq!(
                charged_bias_slack_or(None, slack).unwrap().to_bits(),
                slack.to_bits(),
                "the dark path must be byte-identical"
            );
            let widened = charged_bias_slack_or(Some(&policy), slack).unwrap();
            assert!(
                f64::from(widened) >= f64::from(slack) * f64::from(policy.bias_combine_factor),
                "armed widening must be outward"
            );
        }
    }

    /// #flush-charge §E: the act-bias widening is the identity on every
    /// uncharged device (byte-identity pin — production charged authority is
    /// compile-time closed, so every shipped `ActBiasParams.slack` byte is
    /// unchanged) and outward by the audited factor when armed.
    #[test]
    fn charged_act_bias_slack_is_identity_when_unarmed_and_outward_when_armed() {
        let policy = FlushChargePolicy::production();
        for slack in [1.0f32, 1.000_001, 2.5] {
            assert_eq!(
                charged_act_bias_slack_or(None, slack).unwrap().to_bits(),
                slack.to_bits(),
                "the dark path must be byte-identical"
            );
            let widened = charged_act_bias_slack_or(Some(&policy), slack).unwrap();
            assert!(
                f64::from(widened) >= f64::from(slack) * f64::from(policy.act_bias_slack_factor),
                "armed widening must be outward"
            );
        }
    }
}

/// #flush-charge Lane A: pins for the module-doc "Charged-authority guard
/// coverage" table (routes GA1–GA5). Every row claiming "guarded" or
/// "structurally unreachable / immune" gets a test here (or is named where its
/// pin already lives) so a future edit cannot silently open an uncharged route.
#[cfg(test)]
mod charged_route_coverage_tests {
    use super::{add_skip_stream, merge_streams, ResidentCoeff, WgpuDevice};

    /// A minimal clean 1×1 frontier for the host-seam arithmetic pins.
    fn tiny_frontier(err: f32, taint_rows: Option<Vec<u32>>) -> ResidentCoeff {
        ResidentCoeff {
            lower_a: vec![0.5],
            upper_a: vec![0.75],
            lower_err: vec![err],
            upper_err: vec![err],
            lower_b: vec![0.0],
            upper_b: vec![0.0],
            lower_b_err: vec![err],
            upper_b_err: vec![err],
            dim: 1,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            taint_rows,
        }
    }

    /// GA1/GA2/GA5 SOURCE PINS. Charged coverage lives at exactly two
    /// arithmetic chokepoints plus one explicit host-driver refusal; this test
    /// fails if any of the following drifts:
    ///
    /// * GA1 — the walk body has exactly ONE production `charged_walk_guard`
    ///   call site, ordered AFTER the authority recheck and BEFORE the walk's
    ///   deadline-aware checked GPU section, so no dispatch can precede the guard;
    /// * GA2 — the concretize funnel consults the charged policy (EFT +
    ///   subnormal-input refusals + widened slack) and the armed C1 word
    ///   consult BEFORE its own GPU section
    ///   (`run_gpu_checked_with_crown_deadline("concretize_sound_gpu"`);
    /// * GA5 — the host driver refuses charged authority at entry, BEFORE
    ///   constructing its raw diagnostic GEMM adapter.
    #[test]
    fn charged_route_funnels_and_guard_ordering_are_pinned() {
        // ---- GA1: the resident walk ----------------------------------------
        // WALK includes THIS test's own source, so every searched literal is
        // split with concat! (the escaped-quote GPU-section marker is
        // self-immune: its raw source text carries backslashes).
        const WALK: &str = include_str!("crown_backward_sound_resident.rs");
        let guard_call = concat!("charged_walk_guard(layers, ", "policy, eft_requested)?");
        assert_eq!(
            WALK.matches(guard_call).count(),
            1,
            "the walk must have exactly ONE production charged_walk_guard call \
             site (a second one means a second, separately-audited walk; zero \
             means the charged admission guard was dropped)"
        );
        let authority_recheck = concat!(
            "WGPU verdict authority closed while ",
            "materializing resident shaders"
        );
        let walk_gpu_section =
            "run_gpu_checked_with_crown_deadline(\"crown_backward_sound_resident\"";
        let at_recheck = WALK
            .find(authority_recheck)
            .expect("the walk's authority recheck refusal is gone");
        let at_guard = WALK
            .find(guard_call)
            .expect("guard call found above by count");
        let at_gpu = WALK
            .find(walk_gpu_section)
            .expect("the walk's GPU section marker is gone");
        assert!(
            at_recheck < at_guard && at_guard < at_gpu,
            "GA1 ordering broken: authority recheck ({at_recheck}) -> \
             charged_walk_guard ({at_guard}) -> GPU section ({at_gpu}) must be \
             strictly ordered so no dispatch precedes the charged admission"
        );

        // ---- GA2: the concretize funnel ------------------------------------
        const CONC: &str = include_str!("crown_concretize_sound.rs");
        let conc_consult = "self.charged_flush_authority_cached()";
        let conc_c1 = "consult_spec_row_taint(taint, num_specs)?";
        let conc_gpu_section = "run_gpu_checked_with_crown_deadline(\"concretize_sound_gpu\"";
        let at_conc_consult = CONC
            .find(conc_consult)
            .expect("the concretize charged consult is gone");
        let at_conc_c1 = CONC.find(conc_c1).expect("the armed C1 consult is gone");
        let at_conc_gpu = CONC
            .find(conc_gpu_section)
            .expect("the concretize GPU section marker is gone");
        assert!(
            at_conc_c1 < at_conc_gpu && at_conc_consult < at_conc_gpu,
            "GA2 ordering broken: the C1 consult ({at_conc_c1}) and the \
             charged consult ({at_conc_consult}) must both precede the \
             concretize GPU section ({at_conc_gpu})"
        );
        for refusal in [
            "the EFT concretize arm is refused under",
            "the input box contains a SUBNORMAL",
            "charged_concretize_slack(",
        ] {
            assert!(
                CONC.contains(refusal),
                "GA2 charged arm missing from the concretize funnel: {refusal}"
            );
        }

        // ---- GA5: the host driver ------------------------------------------
        // The refusal string is line-wrapped in the source, so anchor on its
        // two halves in order rather than the exact concatenation.
        const HOST: &str = include_str!("crown_backward_sound_host.rs");
        let at_host_refusal = HOST
            .find("is not audited")
            .and_then(|i| HOST[i..].find("charged flush model").map(|j| i + j))
            .expect("the host driver's charged refusal is gone");
        let at_host_gemm = HOST
            .find("WgpuDiagnosticGemm::new(self)")
            .expect("the host driver's diagnostic GEMM adapter is gone");
        assert!(
            at_host_refusal < at_host_gemm,
            "GA5 ordering broken: the charged refusal ({at_host_refusal}) must \
             precede the raw diagnostic GEMM adapter ({at_host_gemm})"
        );
    }

    /// GA4 BEHAVIORAL PIN: the host-side merge seams (`merge_streams`,
    /// `add_skip_stream`) and the error→bias fold
    /// (`concretize_error_into_bias`) preserve SUBNORMAL error mass exactly as
    /// IEEE-754 host f64 arithmetic must. This is what makes them immune to
    /// the charged flush model: they never run on the adapter. If a future
    /// edit moves any of them onto a DAZ/FTZ device (or truncates through a
    /// flushing f32 path), the subnormal contributions below flush to zero and
    /// this test fails.
    #[test]
    fn host_merge_seams_preserve_subnormal_error_mass() {
        // Smallest positive f32 subnormal: the exact value a DAZ path zeroes.
        let sub = f32::from_bits(1);
        assert!(sub > 0.0 && sub < f32::MIN_POSITIVE);

        // merge_streams: both streams carry a subnormal err; the merged err
        // must keep at least their (outward-rounded) sum — never zero.
        let merged = merge_streams(tiny_frontier(sub, None), &tiny_frontier(sub, None));
        assert!(
            merged.lower_err[0] >= 2.0 * sub && merged.upper_err[0] >= 2.0 * sub,
            "merge_streams dropped subnormal error mass: {:e}",
            merged.lower_err[0]
        );
        assert!(
            merged.lower_b_err[0] >= 2.0 * sub,
            "merge_streams dropped subnormal bias-error mass"
        );

        // add_skip_stream: a subnormal err on the skip stream must survive
        // into the summed stream.
        let skipped = add_skip_stream(tiny_frontier(0.0, None), &tiny_frontier(sub, None));
        assert!(
            skipped.lower_err[0] >= sub && skipped.upper_err[0] >= sub,
            "add_skip_stream dropped the skip stream's subnormal error"
        );

        // concretize_error_into_bias: a subnormal coefficient err against a
        // normal node bound must fold a strictly positive bias-error charge
        // (f64 host arithmetic: subnormal-f32 × normal-f32 never underflows).
        let mut coeff = tiny_frontier(sub, None);
        WgpuDevice::concretize_error_into_bias(&mut coeff, 1, 1, &[1.0]);
        assert_eq!(
            coeff.lower_err[0], 0.0,
            "the fold must reset the coefficient error"
        );
        assert!(
            coeff.lower_b_err[0] >= 2.0 * sub && coeff.upper_b_err[0] >= 2.0 * sub,
            "concretize_error_into_bias dropped subnormal error mass \
             (bias err {:e}); has this fold moved onto a flushing device?",
            coeff.lower_b_err[0]
        );
    }
}

#[cfg(test)]
mod resident_preflight_tests {
    use std::sync::Arc;

    use ny_core::{GpuCrownLayer, GpuResnetSegment};

    use super::{
        batched_coefficient_segment_preflight, resident_fold_plan, taint_walk_conv_route_admitted,
    };
    use crate::wgpu_device::shaders::{
        CONV_COL2IM_TAINT_SHADER, CONV_RESHAPE_TAINT_SHADER, GEMM_F32_SMALL_K_TAINT_SHADER,
    };

    const MAX_WG: u32 = 65_535;
    const MAX_BYTES: u64 = u64::MAX;

    #[test]
    fn batched_coeff_preflight_types_a_later_segment_capacity_refusal() {
        let linear = |out_features: usize, in_features: usize| GpuCrownLayer::Linear {
            weight: vec![0.0; out_features * in_features].into(),
            bias: None,
            out_features,
            in_features,
            cert_err: ny_core::CertifiedWeightError::default(),
        };
        let segments = vec![
            GpuResnetSegment::Chain(vec![linear(1, 8)]),
            GpuResnetSegment::Chain(vec![linear(8, 100)]),
        ];
        let error = batched_coefficient_segment_preflight(&segments, 2, 1, 1, MAX_WG, 128, 128)
            .expect_err("the second segment exceeds the storage-binding cap");
        assert!(error.is_gpu_batch_capacity_exceeded(), "{error}");
    }

    #[test]
    fn worded_conv_route_is_admitted_with_internal_transport() {
        assert!(taint_walk_conv_route_admitted(false, false));
        assert!(taint_walk_conv_route_admitted(false, true));
        assert!(taint_walk_conv_route_admitted(true, false));
        assert!(taint_walk_conv_route_admitted(true, true));
    }

    #[test]
    fn conv_taint_twins_pin_exact_value_statements_and_word_bindings() {
        assert!(CONV_RESHAPE_TAINT_SHADER.contains("dst[idx] = src[src_idx];"));
        assert!(CONV_RESHAPE_TAINT_SHADER.contains("taint_dst[idx] = taint_src[src_idx];"));
        assert!(CONV_COL2IM_TAINT_SHADER.contains("sum = sum + gemm_out[src];"));
        assert!(CONV_COL2IM_TAINT_SHADER.contains("taint = taint | taint_gemm[src];"));
        assert!(CONV_COL2IM_TAINT_SHADER.contains("sum != sum || abs(sum) >= FALLBACK_BOUND"));
        assert!(GEMM_F32_SMALL_K_TAINT_SHADER.contains("const ROWS_PER_THREAD: u32 = 4u;"));
        assert!(GEMM_F32_SMALL_K_TAINT_SHADER.contains("sum = sum + av * bv;"));
        assert!(GEMM_F32_SMALL_K_TAINT_SHADER.contains("out[row * params.n + col] = guarded;"));
    }

    #[test]
    fn conv_reshape_word_mapping_is_asymmetric_and_identical_to_value_mapping() {
        let (specs, channels, spatial) = (2usize, 3usize, 2usize);
        let src: Vec<u32> = (0..specs * channels * spatial)
            .map(|i| 100 + i as u32)
            .collect();
        let mut dst = vec![0u32; src.len()];
        for (idx, slot) in dst.iter_mut().enumerate() {
            let flat_row = idx / channels;
            let channel = idx % channels;
            let spec = flat_row / spatial;
            let pos = flat_row % spatial;
            let src_idx = spec * channels * spatial + channel * spatial + pos;
            *slot = src[src_idx];
        }
        assert_eq!(
            dst,
            vec![100, 102, 104, 101, 103, 105, 106, 108, 110, 107, 109, 111]
        );
    }

    #[test]
    fn conv_col2im_word_survives_internal_saturation_then_cancellation() {
        // OC=IC=1, KH=1, KW=2, OW=2, IW=3. For input x=1 the
        // gather visits (row=1,col=0) then (row=0,col=1). Those two
        // GEMM outputs cancel in the value channel, while either word must
        // survive. This is the exact hole a boundary-only scan missed.
        let gemm = [0.0f32, 1.0e10, -1.0e10, 0.0];
        let words = [0u32, 1, 1, 0];
        let mut sum = 0.0f32;
        let mut word = 0u32;
        for src in [2usize, 1usize] {
            sum += gemm[src];
            word |= words[src];
            if !sum.is_finite() || sum.abs() >= ny_core::CROWN_COEFF_MAX {
                word |= 1;
            }
        }
        assert_eq!(sum.to_bits(), 0.0f32.to_bits());
        assert_eq!(
            word, 1,
            "cancelled value must retain its internal Conv word"
        );
    }

    fn activation(state_len: usize, neurons: usize) -> GpuCrownLayer {
        GpuCrownLayer::Activation {
            lower_slope: vec![1.0; state_len],
            upper_slope: vec![1.0; state_len],
            lower_intercept: vec![0.0; state_len],
            upper_intercept: vec![0.0; state_len],
            num_neurons: neurons,
        }
    }

    fn conv(
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> GpuCrownLayer {
        GpuCrownLayer::Conv2d {
            weight_col: Arc::from(vec![1.0; kernel_h.saturating_mul(kernel_w)]),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err: Default::default(),
        }
    }

    #[test]
    fn resident_preflight_requires_a_real_domain_partition() {
        for (num_specs, per_domain) in [(0, 0), (4, 0), (4, 3), (4, 5)] {
            assert!(
                resident_fold_plan(&[], num_specs, per_domain, 1, MAX_WG, MAX_BYTES, MAX_BYTES,)
                    .is_err(),
                "partition ({num_specs}, {per_domain}) must fail closed"
            );
        }

        let layer = activation(2, 1);
        let plan = resident_fold_plan(&[layer], 2, 1, 1, MAX_WG, MAX_BYTES, MAX_BYTES)
            .expect("two one-row domains are valid");
        assert_eq!(plan.n_domains, 2);
        assert_eq!(plan.a_elems, 2);
        assert_eq!(plan.slope_dim, 2);
    }

    #[test]
    fn resident_preflight_rejects_wrapping_or_truncated_dimensions() {
        if usize::BITS > 32 {
            let too_wide = (u32::MAX as usize) + 1;
            let layer = activation(0, too_wide);
            assert!(
                resident_fold_plan(&[layer], 1, 1, too_wide, MAX_WG, MAX_BYTES, MAX_BYTES,)
                    .is_err()
            );
        }

        let overflowing_conv = GpuCrownLayer::Conv2d {
            weight_col: Arc::from([]),
            bias_expanded: None,
            out_channels: u32::MAX as usize,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: u32::MAX as usize,
            out_w: 2,
            in_h: 1,
            in_w: 1,
            cert_err: Default::default(),
        };
        assert!(
            resident_fold_plan(&[overflowing_conv], 1, 1, 1, MAX_WG, MAX_BYTES, MAX_BYTES,)
                .is_err()
        );
    }

    #[test]
    fn resident_preflight_enforces_state_buffer_and_dispatch_limits() {
        assert!(
            resident_fold_plan(&[activation(1, 1)], 1, 1, 1, 0, MAX_BYTES, MAX_BYTES).is_err(),
            "a zero device workgroup limit must fail closed"
        );
        assert!(
            resident_fold_plan(&[activation(1, 1)], 2, 2, 1, MAX_WG, 4, 4).is_err(),
            "two f32 coefficient elements do not fit a four-byte binding"
        );
        assert!(
            resident_fold_plan(&[activation(1, 1)], 257, 257, 1, 256, MAX_BYTES, MAX_BYTES,)
                .is_err(),
            "257 one-row bias workgroups exceed a 256-workgroup device limit"
        );
        assert!(
            resident_fold_plan(&[activation(0, 1)], 1, 1, 1, MAX_WG, MAX_BYTES, MAX_BYTES,)
                .is_err(),
            "short activation state must not leave stale GPU buffer contents"
        );
    }

    #[test]
    fn resident_preflight_validates_conv_geometry_exactly() {
        resident_fold_plan(
            &[conv(5, 7, 3, 3, 2, 2, 1, 1, 3, 4)],
            1,
            1,
            12,
            MAX_WG,
            MAX_BYTES,
            MAX_BYTES,
        )
        .expect("checked convolution geometry is valid");

        for malformed in [
            conv(5, 7, 3, 3, 2, 2, 1, 1, 2, 4),
            conv(5, 7, 3, 3, 2, 2, 1, 1, 3, 3),
            conv(2, 7, 5, 3, 1, 1, 0, 0, 1, 5),
        ] {
            assert!(
                resident_fold_plan(&[malformed], 1, 1, 12, MAX_WG, MAX_BYTES, MAX_BYTES,).is_err(),
                "malformed convolution geometry must fail closed"
            );
        }

        if usize::BITS > 32 {
            let overflowing_pad = (usize::MAX / 2) + 1;
            assert!(
                resident_fold_plan(
                    &[conv(1, 1, 1, 1, 1, 1, overflowing_pad, 0, 1, 1)],
                    1,
                    1,
                    1,
                    MAX_WG,
                    MAX_BYTES,
                    MAX_BYTES,
                )
                .is_err(),
                "padding arithmetic must be checked"
            );
        }
    }
}

/// A resnet decomposed into backward-order segments for the resident backward.
#[allow(dead_code)]
pub(crate) enum ResnetSegment<'a> {
    /// A plain sequential sub-chain of layers.
    Chain(&'a [GpuCrownLayer]),
    /// An identity-skip residual block `out = F(z) + z`; the slice is `F`'s sub-chain
    /// (which must map the block dim back to itself).
    Residual(&'a [GpuCrownLayer]),
    /// A PROJECTION residual block `out = F(z) + P(z)` (e.g. a 1×1-conv skip at a
    /// stage transition): `(F_branch, P_branch)`. Both branches map the block input
    /// dim to the block output dim. `A_in = backward_F(A) + backward_P(A)`.
    ResidualProj(&'a [GpuCrownLayer], &'a [GpuCrownLayer]),
}

/// THE #cert-coeffs FIREWALL: turn a resident coefficient frontier into a
/// publishable [`ny_core::CertifiedCoeffs`], or refuse.
///
/// There is exactly ONE copy of these screens, shared by BOTH coefficient
/// egresses — the flat chain
/// ([`WgpuDevice::crown_backward_sound_resident_certified_coeffs`]) and the
/// resnet segment composition
/// ([`WgpuDevice::crown_backward_gpu_resnet_sound_certified_coeffs`]) — so the
/// segment path can never drift into a weaker contract than the flat one.
/// Pinned by `both_coefficient_egresses_share_one_firewall`.
///
/// In order:
///
/// 1. `#u4` C1 per-spec-row taint consult (absent / wrong-length / nonzero
///    words ⇒ typed refusal, never a publication);
/// 2. NON-FINITE screen — a NaN/inf coefficient, bias or radius is not an
///    enclosure and the coefficient consumer has no downstream preflight;
/// 3. `FALLBACK_BOUND` SATURATION-SENTINEL screen — the BOUNDS entry reaches
///    `concretize_sound_gpu_batched`, whose host preflight proves in f64 that
///    the outward affine radius is enclosed by `FALLBACK_BOUND`; the value
///    GEMM's `nan_safe_clamp` writes exactly ±1e10, which is FINITE and sails
///    past screen 2. Without this the egress would be strictly WEAKER than the
///    entry whose affine-radius proof it claims parity with;
/// 4. NEGATIVE-RADIUS screen — the error arrays are RADII; a negative entry is
///    a tightening, so it is refused rather than clamped.
///
/// Every refusal is fail-closed: the frontier is discarded, never repaired.
pub(crate) fn certified_coeffs_from_resident(
    c: ResidentCoeff,
    num_specs: usize,
) -> Result<ny_core::CertifiedCoeffs> {
    if super::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD {
        super::crown_concretize_sound::consult_spec_row_taint(c.taint_rows.as_deref(), num_specs)?;
    }
    // Fail-closed firewall: a non-finite coefficient, bias, or radius is
    // never an enclosure, and the coefficient consumer has no downstream
    // `FALLBACK_BOUND` preflight of its own to catch it.
    for (label, values) in [
        ("lower_a", &c.lower_a),
        ("upper_a", &c.upper_a),
        ("lower_a_err", &c.lower_err),
        ("upper_a_err", &c.upper_err),
        ("lower_b", &c.lower_b),
        ("upper_b", &c.upper_b),
        ("lower_b_err", &c.lower_b_err),
        ("upper_b_err", &c.upper_b_err),
    ] {
        if let Some(bad) = values.iter().copied().find(|v| !v.is_finite()) {
            return Err(NyError::NumericalInstability(format!(
                "#cert-coeffs: {label} carries a non-finite entry ({bad:e}) — \
                 refusing to publish the frontier (fail-closed)"
            )));
        }
    }
    // Review defect 2: the BOUNDS entry reaches `concretize_sound_gpu_batched`,
    // whose host preflight PROVES in f64 that the outward affine radius is
    // enclosed by FALLBACK_BOUND — the documented catcher for the resident
    // value GEMM's saturation sentinel (`nan_safe_clamp` writes exactly
    // ±1e10, which is FINITE and sails past the non-finite screen above).
    // The coefficient egress has no downstream preflight at all, so
    // publishing without this check made it strictly WEAKER than the entry
    // whose contract parity it claims. Refuse any saturated magnitude.
    for (label, values) in [
        ("lower_a", &c.lower_a),
        ("upper_a", &c.upper_a),
        ("lower_a_err", &c.lower_err),
        ("upper_a_err", &c.upper_err),
        ("lower_b", &c.lower_b),
        ("upper_b", &c.upper_b),
        ("lower_b_err", &c.lower_b_err),
        ("upper_b_err", &c.upper_b_err),
    ] {
        if let Some(bad) = values
            .iter()
            .copied()
            .find(|v| v.abs() >= ny_core::FALLBACK_BOUND)
        {
            return Err(NyError::NumericalInstability(format!(
                "#cert-coeffs: {label} carries a saturation-sentinel \
                 magnitude ({bad:e} >= FALLBACK_BOUND) — the value GEMM \
                 clamped, so this frontier is not an enclosure; refusing \
                 (fail-closed, matching the bounds entry's affine-radius \
                 proof)"
            )));
        }
    }
    // The error arrays are RADII: negative entries would be a tightening.
    for (label, values) in [
        ("lower_a_err", &c.lower_err),
        ("upper_a_err", &c.upper_err),
        ("lower_b_err", &c.lower_b_err),
        ("upper_b_err", &c.upper_b_err),
    ] {
        if values.iter().copied().any(|v| v < 0.0) {
            return Err(NyError::NumericalInstability(format!(
                "#cert-coeffs: {label} carries a NEGATIVE radius — refusing \
                 to publish the frontier (fail-closed)"
            )));
        }
    }
    Ok(ny_core::CertifiedCoeffs {
        lower_a: c.lower_a,
        upper_a: c.upper_a,
        lower_a_err: c.lower_err,
        upper_a_err: c.upper_err,
        lower_b: c.lower_b,
        upper_b: c.upper_b,
        lower_b_err: c.lower_b_err,
        upper_b_err: c.upper_b_err,
        num_specs,
        dim: c.dim,
    })
}

/// THE #margin-row-gpu-batch SLOT MAP: split ONE wide, domain-major
/// [`ny_core::CertifiedCoeffs`] into `n_domains` per-domain payloads.
///
/// `resnet_seeded_compose_coeff` stacks the batch DOMAIN-MAJOR — row `s`
/// belongs to domain `s / num_specs_per_dom`, which is precisely the
/// `dom = row/num_specs_per_dom` rule its shaders use to index each domain's
/// own Activation block and input box. So domain `d` owns the CONTIGUOUS row
/// range `[d*nsp, (d+1)*nsp)` and this split is a pure reshape of that layout.
///
/// # Why this is written the way it is
///
/// A slot/permutation error here is the killer defect of the whole batched
/// lane: it would publish, for domain A, a bound computed for domain B's gates.
/// So:
///
/// * the total row count is checked against `n_domains * nsp` BEFORE any
///   slicing, and every array's length is checked against the row count, so a
///   short or long payload REFUSES instead of silently shifting the mapping;
/// * the walk is a single ordered `chunks_exact` zip with NO free index
///   arithmetic — there is nowhere to type a wrong subscript;
/// * the coefficient and bias lanes are advanced by the SAME iterator step, so
///   they cannot drift relative to one another.
///
/// Every returned payload reports `num_specs == nsp` (the per-domain count) and
/// the unchanged `dim`: the wide stack is an implementation detail and must not
/// leak into what the lane concretizes.
pub(crate) fn split_batched_certified_coeffs(
    wide: &ny_core::CertifiedCoeffs,
    num_specs_per_dom: usize,
    n_domains: usize,
) -> Result<Vec<ny_core::CertifiedCoeffs>> {
    let dim = wide.dim;
    let expected_rows = num_specs_per_dom
        .checked_mul(n_domains)
        .ok_or_else(|| NyError::InvalidSpec("#cert-coeffs batch: row count overflow".into()))?;
    if num_specs_per_dom == 0 || n_domains == 0 || dim == 0 {
        return Err(NyError::InvalidSpec(
            "#cert-coeffs batch: zero-sized split requested".into(),
        ));
    }
    if wide.num_specs != expected_rows {
        return Err(NyError::shape_mismatch(
            vec![expected_rows],
            vec![wide.num_specs],
        ));
    }
    let a_len = expected_rows
        .checked_mul(dim)
        .ok_or_else(|| NyError::InvalidSpec("#cert-coeffs batch: element count overflow".into()))?;
    for (label, len) in [
        ("lower_a", wide.lower_a.len()),
        ("upper_a", wide.upper_a.len()),
        ("lower_a_err", wide.lower_a_err.len()),
        ("upper_a_err", wide.upper_a_err.len()),
    ] {
        if len != a_len {
            return Err(NyError::InvalidSpec(format!(
                "#cert-coeffs batch: {label} has {len} elements, expected {a_len} — refusing to \
                 split (a mis-sized payload would re-associate domains)"
            )));
        }
    }
    for (label, len) in [
        ("lower_b", wide.lower_b.len()),
        ("upper_b", wide.upper_b.len()),
        ("lower_b_err", wide.lower_b_err.len()),
        ("upper_b_err", wide.upper_b_err.len()),
    ] {
        if len != expected_rows {
            return Err(NyError::InvalidSpec(format!(
                "#cert-coeffs batch: {label} has {len} rows, expected {expected_rows} — refusing \
                 to split (a mis-sized payload would re-associate domains)"
            )));
        }
    }
    let block = num_specs_per_dom * dim;
    let out: Vec<ny_core::CertifiedCoeffs> = wide
        .lower_a
        .chunks_exact(block)
        .zip(wide.upper_a.chunks_exact(block))
        .zip(wide.lower_a_err.chunks_exact(block))
        .zip(wide.upper_a_err.chunks_exact(block))
        .zip(wide.lower_b.chunks_exact(num_specs_per_dom))
        .zip(wide.upper_b.chunks_exact(num_specs_per_dom))
        .zip(wide.lower_b_err.chunks_exact(num_specs_per_dom))
        .zip(wide.upper_b_err.chunks_exact(num_specs_per_dom))
        .map(
            |(((((((la, ua), lae), uae), lb), ub), lbe), ube)| ny_core::CertifiedCoeffs {
                lower_a: la.to_vec(),
                upper_a: ua.to_vec(),
                lower_a_err: lae.to_vec(),
                upper_a_err: uae.to_vec(),
                lower_b: lb.to_vec(),
                upper_b: ub.to_vec(),
                lower_b_err: lbe.to_vec(),
                upper_b_err: ube.to_vec(),
                num_specs: num_specs_per_dom,
                dim,
            },
        )
        .collect();
    // Defensive: the zip above is length-driven, so a silent short-circuit
    // would hand back FEWER domains than asked for and the caller would index
    // the wrong ones. Refuse instead.
    if out.len() != n_domains {
        return Err(NyError::InvalidSpec(format!(
            "#cert-coeffs batch: split produced {} domains, expected {n_domains}",
            out.len()
        )));
    }
    Ok(out)
}

/// HOST-ONLY pins for the batched slot map. Pure host arithmetic on a payload
/// the caller supplies, so these are falsifiable on every build.
#[cfg(test)]
mod cert_coeffs_batch_split_tests {
    use super::*;

    /// A wide payload whose every entry ENCODES its own (row, coordinate), so a
    /// permutation or an off-by-one block is visible in the values themselves.
    #[allow(clippy::cast_precision_loss)]
    fn wide(n_domains: usize, nsp: usize, dim: usize) -> ny_core::CertifiedCoeffs {
        let rows = n_domains * nsp;
        let a: Vec<f32> = (0..rows * dim).map(|k| k as f32).collect();
        let b: Vec<f32> = (0..rows).map(|s| 1000.0 + s as f32).collect();
        ny_core::CertifiedCoeffs {
            lower_a: a.clone(),
            upper_a: a.iter().map(|v| v + 0.5).collect(),
            lower_a_err: vec![1e-6; rows * dim],
            upper_a_err: vec![2e-6; rows * dim],
            lower_b: b.clone(),
            upper_b: b.iter().map(|v| v + 0.5).collect(),
            lower_b_err: vec![1e-7; rows],
            upper_b_err: vec![2e-7; rows],
            num_specs: rows,
            dim,
        }
    }

    /// THE SLOT-MAPPING PIN. Domain `d` must receive rows `[d*nsp, (d+1)*nsp)`
    /// of the domain-major wide payload — not `d`'s rows shifted, interleaved,
    /// or another domain's. The fixture encodes the global row index in every
    /// value, so this asserts the mapping, not merely the shape.
    #[test]
    fn batched_split_is_domain_major_and_contiguous() {
        let (n_domains, nsp, dim) = (3usize, 2usize, 4usize);
        let w = wide(n_domains, nsp, dim);
        let parts = split_batched_certified_coeffs(&w, nsp, n_domains).expect("split");
        assert_eq!(parts.len(), n_domains);
        for (d, part) in parts.iter().enumerate() {
            assert_eq!(part.num_specs, nsp, "per-domain row count must be nsp");
            assert_eq!(part.dim, dim);
            for r in 0..nsp {
                let global = d * nsp + r;
                #[allow(clippy::cast_precision_loss)]
                let want_b = 1000.0 + global as f32;
                assert_eq!(part.lower_b[r], want_b, "domain {d} row {r} bias slot");
                for j in 0..dim {
                    #[allow(clippy::cast_precision_loss)]
                    let want_a = (global * dim + j) as f32;
                    assert_eq!(
                        part.lower_a[r * dim + j],
                        want_a,
                        "domain {d} row {r} coord {j} coefficient slot"
                    );
                }
            }
        }
        // An INTERLEAVED (row-major-by-spec) reading is a different answer:
        // this is the permutation the pin exists to exclude.
        assert_ne!(
            parts[1].lower_b[0], w.lower_b[1],
            "domain 1 must own row nsp, not row 1 — an interleaved map would pass here"
        );
    }

    /// Every mis-sized payload REFUSES; none is repaired by truncation, which
    /// would shift the mapping by whole domains.
    #[test]
    fn batched_split_refuses_every_mis_sized_payload() {
        let (n_domains, nsp, dim) = (2usize, 3usize, 2usize);
        assert!(split_batched_certified_coeffs(&wide(n_domains, nsp, dim), nsp, n_domains).is_ok());

        // Wrong declared row count.
        let mut w = wide(n_domains, nsp, dim);
        w.num_specs += 1;
        assert!(split_batched_certified_coeffs(&w, nsp, n_domains).is_err());

        // Truncated coefficient lane.
        let mut w = wide(n_domains, nsp, dim);
        w.lower_a.pop();
        assert!(split_batched_certified_coeffs(&w, nsp, n_domains).is_err());

        // Truncated bias lane.
        let mut w = wide(n_domains, nsp, dim);
        w.upper_b_err.pop();
        assert!(split_batched_certified_coeffs(&w, nsp, n_domains).is_err());

        // Asking for a different partition of the SAME payload must refuse
        // rather than silently re-cut the domains.
        let w = wide(n_domains, nsp, dim);
        assert!(split_batched_certified_coeffs(&w, nsp, n_domains + 1).is_err());
        assert!(split_batched_certified_coeffs(&w, nsp + 1, n_domains).is_err());
        assert!(split_batched_certified_coeffs(&w, 0, n_domains).is_err());
        assert!(split_batched_certified_coeffs(&w, nsp, 0).is_err());
    }

    /// A single-domain batch must reproduce the payload unchanged: the batched
    /// entry degenerates to the single-domain egress.
    #[test]
    fn batched_split_of_one_domain_is_the_identity() {
        let w = wide(1, 5, 3);
        let parts = split_batched_certified_coeffs(&w, 5, 1).expect("split");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].lower_a, w.lower_a);
        assert_eq!(parts[0].upper_a, w.upper_a);
        assert_eq!(parts[0].lower_a_err, w.lower_a_err);
        assert_eq!(parts[0].upper_a_err, w.upper_a_err);
        assert_eq!(parts[0].lower_b, w.lower_b);
        assert_eq!(parts[0].upper_b, w.upper_b);
        // Review defect D2: the BIAS-ERROR lanes were the only ones no host or
        // device test touched. A lower_b_err <-> upper_b_err swap in the zip
        // would hand the lower lane the SMALLER radius in convert_and_check
        // (eb = next_up(bev + pen_seed[r])) — a bound tighter than certified,
        // i.e. a false VERIFIED — and every other assertion here would still
        // pass. The fixture's 1e-7 / 2e-7 values are distinguishable.
        assert_eq!(parts[0].lower_b_err, w.lower_b_err);
        assert_eq!(parts[0].upper_b_err, w.upper_b_err);
        assert_eq!(parts[0].num_specs, w.num_specs);
        assert_eq!(parts[0].dim, w.dim);
    }
}

/// HOST-ONLY pins for the shared #cert-coeffs firewall.
///
/// Deliberately NOT under `feature = "gpu-tests"`: the screens are pure host
/// arithmetic on a frontier the caller supplies, so they must be falsifiable on
/// every build, not only on a conformant GPU host.
#[cfg(test)]
mod cert_coeffs_firewall_tests {
    use super::*;

    /// A clean frontier for the firewall pins: `num_specs x dim`, zero taint.
    fn clean_frontier(num_specs: usize, dim: usize) -> ResidentCoeff {
        ResidentCoeff {
            lower_a: vec![0.5; num_specs * dim],
            upper_a: vec![0.75; num_specs * dim],
            lower_err: vec![1e-6; num_specs * dim],
            upper_err: vec![1e-6; num_specs * dim],
            lower_b: vec![-0.25; num_specs],
            upper_b: vec![0.25; num_specs],
            lower_b_err: vec![1e-7; num_specs],
            upper_b_err: vec![1e-7; num_specs],
            dim,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            taint_rows: Some(vec![0u32; num_specs]),
        }
    }

    /// The fail-closed firewall, pinned on the SHARED helper both egresses call
    /// — so the segment path is covered by construction, not by a second copy of
    /// the screens (`both_coefficient_egresses_share_one_firewall`).
    ///
    /// Every screen is a REFUSAL, never a repair: a non-finite entry, a value
    /// GEMM saturation sentinel (`|v| >= FALLBACK_BOUND`, finite and therefore
    /// invisible to the non-finite screen), a NEGATIVE radius (a tightening),
    /// and the `#u4` C1 taint consult.
    #[test]
    fn certified_coeffs_firewall_refuses_every_poisoned_frontier() {
        let (rows, dim) = (2usize, 3usize);
        // Baseline: a clean frontier publishes unchanged.
        let ok = certified_coeffs_from_resident(clean_frontier(rows, dim), rows)
            .expect("a clean frontier must publish");
        assert_eq!((ok.num_specs, ok.dim), (rows, dim));

        // (1) NON-FINITE, on every lane in turn.
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for lane in 0..8usize {
                let mut c = clean_frontier(rows, dim);
                match lane {
                    0 => c.lower_a[1] = poison,
                    1 => c.upper_a[1] = poison,
                    2 => c.lower_err[1] = poison,
                    3 => c.upper_err[1] = poison,
                    4 => c.lower_b[0] = poison,
                    5 => c.upper_b[0] = poison,
                    6 => c.lower_b_err[0] = poison,
                    _ => c.upper_b_err[0] = poison,
                }
                assert!(
                    certified_coeffs_from_resident(c, rows).is_err(),
                    "lane {lane} published a non-finite ({poison}) frontier"
                );
            }
        }

        // (2) SATURATION SENTINEL: finite, so screen (1) cannot see it. This is
        // the value GEMM's `nan_safe_clamp` output and the documented catcher
        // the bounds entry gets from its concretize preflight.
        for sign in [1.0f32, -1.0] {
            let mut c = clean_frontier(rows, dim);
            c.upper_a[2] = sign * ny_core::FALLBACK_BOUND;
            assert!(
                c.upper_a[2].is_finite(),
                "the sentinel must be FINITE for this pin to mean anything"
            );
            assert!(
                certified_coeffs_from_resident(c, rows).is_err(),
                "a {sign}x FALLBACK_BOUND saturation sentinel was published"
            );
        }
        // Just under the sentinel still publishes (the screen is not a blanket
        // magnitude cap that would make the egress useless).
        let mut near = clean_frontier(rows, dim);
        near.upper_a[2] = ny_core::FALLBACK_BOUND * 0.5;
        assert!(certified_coeffs_from_resident(near, rows).is_ok());

        // (3) NEGATIVE RADIUS — a signed correction masquerading as a radius
        // would TIGHTEN the consumer's concretization.
        for lane in 0..4usize {
            let mut c = clean_frontier(rows, dim);
            match lane {
                0 => c.lower_err[0] = -1e-9,
                1 => c.upper_err[0] = -1e-9,
                2 => c.lower_b_err[0] = -1e-9,
                _ => c.upper_b_err[0] = -1e-9,
            }
            assert!(
                certified_coeffs_from_resident(c, rows).is_err(),
                "radius lane {lane} published a NEGATIVE radius"
            );
        }

        // (4) The #u4 C1 taint consult: absent, wrong-length, or nonzero words.
        if crate::wgpu_device::ops::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD {
            let mut absent = clean_frontier(rows, dim);
            absent.taint_rows = None;
            assert!(certified_coeffs_from_resident(absent, rows).is_err());
            let mut short = clean_frontier(rows, dim);
            short.taint_rows = Some(vec![0u32; rows - 1]);
            assert!(certified_coeffs_from_resident(short, rows).is_err());
            let mut tainted = clean_frontier(rows, dim);
            tainted.taint_rows = Some(vec![0u32, 1u32]);
            assert!(certified_coeffs_from_resident(tainted, rows).is_err());
        }
    }

    /// There must be exactly ONE firewall. The flat egress and the segment
    /// egress publish the same type with the same authority, so a second copy of
    /// the screens is a drift hazard: the review defect that made the FIRST
    /// egress weaker than its bounds entry (the missing saturation screen) is
    /// exactly the kind of divergence a duplicate re-introduces.
    ///
    /// Pinned structurally: `ny_core::CertifiedCoeffs` is PUBLISHED in exactly
    /// one place in this module — inside `certified_coeffs_from_resident` — and
    /// every egress entry reaches it.
    ///
    /// The flat egress calls the firewall directly; the resnet single-domain and
    /// BATCHED egresses both go through
    /// `resnet_certified_coeffs_unconcretized`, which is the one transcription
    /// of their box-independent composition and ends in the same firewall. That
    /// indirection is checked here rather than assumed, because a batched path
    /// that grew its own copy could silently become weaker than the
    /// single-domain one.
    #[test]
    fn both_coefficient_egresses_share_one_firewall() {
        const SRC: &str = include_str!("crown_backward_sound_resident.rs");
        // Split so this assertion's own text is not a match.
        let ctor = concat!("Ok(ny_core::Certified", "Coeffs {");
        assert_eq!(
            SRC.matches(ctor).count(),
            1,
            "CertifiedCoeffs is published outside the shared firewall"
        );
        // Direct callers of the firewall: the flat egress and the ONE shared
        // resnet un-concretized helper. Nothing else.
        for entry in [
            "crown_backward_sound_resident_certified_coeffs",
            "resnet_certified_coeffs_unconcretized",
        ] {
            // Anchor on the DEFINITION, not the first textual match: the
            // first occurrence of `fn {entry}(` in this file is this test's
            // own search-string literal, so the scan was reading unrelated
            // code and failing on a correct implementation.
            let at = ["\n    pub(crate) fn ", "\n    pub fn ", "\n    fn "]
                .iter()
                .find_map(|vis| SRC.find(&format!("{vis}{entry}(")))
                .unwrap_or_else(|| panic!("{entry} definition not found"));
            // A fixed window from the definition, deliberately simple: the
            // earlier "stop at the next item" heuristics kept mis-bounding
            // (a nested block closing at method indent; a doc comment inside
            // the body) and failed this pin on CORRECT code twice. Both
            // egresses are well under this size.
            let body = &SRC[at..];
            // Stop at the NEXT item (its doc comment counts as the boundary):
            // a fixed window spilled into the shared two-pass helper, whose
            // doc legitimately names the recovery rule these entries must not
            // re-implement.
            let end = [
                "\n    /// ",
                "\n    pub(crate) fn ",
                "\n    pub fn ",
                "\n    fn ",
            ]
            .iter()
            .filter_map(|marker| body[1..].find(marker).map(|i| i + 1))
            .min()
            .unwrap_or(body.len());
            assert!(
                body[..end].contains("certified_coeffs_from_resident("),
                "{entry} does not publish through the shared firewall"
            );
        }
    }

    /// THE BATCHED CONTRACT CANNOT DRIFT: the multi-domain coefficient egress
    /// must reuse the single-domain un-concretized helper, not a second copy.
    /// The helper is the authority boundary that ensures neither entry can fold
    /// coefficient radii against a domain box or abs-max table before publishing
    /// [`ny_core::CertifiedCoeffs`].
    #[test]
    fn batched_certified_coeffs_share_the_single_domain_unconcretized_walk() {
        const SRC: &str = include_str!("crown_backward_sound_resident.rs");
        for entry in [
            "crown_backward_gpu_resnet_sound_certified_coeffs",
            "crown_backward_gpu_resnet_sound_batched_certified_coeffs",
        ] {
            // Anchor on the DEFINITION with its visibility+indent: the first
            // textual `fn {entry}(` in this file is this test's own search
            // string, so the scan read unrelated code. Then take a fixed
            // window rather than guessing where the body ends (a nested block
            // closing at method indent, or a doc comment inside the body, each
            // mis-bounded it and failed this pin on CORRECT code).
            let at = ["\n    pub(crate) fn ", "\n    pub fn ", "\n    fn "]
                .iter()
                .find_map(|vis| SRC.find(&format!("{vis}{entry}(")))
                .unwrap_or_else(|| panic!("{entry} definition not found"));
            let body = &SRC[at..];
            let end = body.len().min(6000);
            assert!(
                body[..end].contains("resnet_certified_coeffs_unconcretized("),
                "{entry} does not go through the shared un-concretized composition"
            );
        }
        // The batched egress must also SPLIT through the pinned slot map rather
        // than slicing rows itself.
        let at = SRC
            .find("fn crown_backward_gpu_resnet_sound_batched_certified_coeffs(")
            .expect("batched egress not found");
        let body = &SRC[at..];
        let end = body.find("\n    }\n").unwrap_or(body.len());
        assert!(
            body[..end].contains("split_batched_certified_coeffs("),
            "the batched egress does not use the pinned domain-major slot map"
        );
    }
}

/// Merge two coefficient streams `cf` and `other` summing BOTH the coefficient and
/// the bias (with the two errors + certified f32-add terms). Used for projection
/// residuals, where each branch carries its own bias. `other` must be seeded with
/// ZERO bias so the incoming bias is counted once (it is already in `cf`).
fn merge_streams(mut cf: ResidentCoeff, other: &ResidentCoeff) -> ResidentCoeff {
    for i in 0..cf.lower_a.len() {
        let sl = f32_to_f64_exact(cf.lower_a[i]) + f32_to_f64_exact(other.lower_a[i]);
        let fl_l = sl as f32;
        let lower_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.lower_err[i]),
            f32_to_f64_exact(other.lower_err[i]),
        );
        let lower_gap = (f32_to_f64_exact(fl_l) - sl).abs();
        cf.lower_err[i] = up_f32(add_nonnegative_f64_up(lower_err, lower_gap));
        cf.lower_a[i] = fl_l;
        let su = f32_to_f64_exact(cf.upper_a[i]) + f32_to_f64_exact(other.upper_a[i]);
        let fl_u = su as f32;
        let upper_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.upper_err[i]),
            f32_to_f64_exact(other.upper_err[i]),
        );
        let upper_gap = (f32_to_f64_exact(fl_u) - su).abs();
        cf.upper_err[i] = up_f32(add_nonnegative_f64_up(upper_err, upper_gap));
        cf.upper_a[i] = fl_u;
    }
    for s in 0..cf.lower_b.len() {
        let sl = f32_to_f64_exact(cf.lower_b[s]) + f32_to_f64_exact(other.lower_b[s]);
        let fl_l = sl as f32;
        let lower_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.lower_b_err[s]),
            f32_to_f64_exact(other.lower_b_err[s]),
        );
        let lower_gap = (f32_to_f64_exact(fl_l) - sl).abs();
        cf.lower_b_err[s] = up_f32(add_nonnegative_f64_up(lower_err, lower_gap));
        cf.lower_b[s] = fl_l;
        let su = f32_to_f64_exact(cf.upper_b[s]) + f32_to_f64_exact(other.upper_b[s]);
        let fl_u = su as f32;
        let upper_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.upper_b_err[s]),
            f32_to_f64_exact(other.upper_b_err[s]),
        );
        let upper_gap = (f32_to_f64_exact(fl_u) - su).abs();
        cf.upper_b_err[s] = up_f32(add_nonnegative_f64_up(upper_err, upper_gap));
        cf.upper_b[s] = fl_u;
    }
    cf.taint_rows = merge_taint_rows(cf.taint_rows.take(), other.taint_rows.as_deref());
    cf
}

/// #u4: merge two streams' per-spec-row taint words. Both `Some` ⇒ element-wise
/// OR (a row tainted in EITHER stream is tainted in the sum — the merge is an
/// ADD, so every input's word survives with a trivially-nonzero partner).
/// Either side `None` ⇒ `None`: a stream that carried no words cannot vouch for
/// its rows, and once the C1 consult arms, `None` is the value that REFUSES
/// verdict use — strictly fail-closed, never a silent "clean".
fn merge_taint_rows(mine: Option<Vec<u32>>, other: Option<&[u32]>) -> Option<Vec<u32>> {
    match (mine, other) {
        (Some(mut a), Some(b)) if a.len() == b.len() => {
            for (dst, &src) in a.iter_mut().zip(b) {
                *dst |= src;
            }
            Some(a)
        }
        _ => None,
    }
}

/// Add the identity-skip coefficient stream `skip` into the branch result `cf`:
/// `A_in = A_F + A_skip`, with the two streams' errors summed plus a certified
/// f32-add rounding term `u·|sum|`. The bias is the branch's (the identity skip
/// contributes no bias). Both must be over the same dim.
fn add_skip_stream(mut cf: ResidentCoeff, skip: &ResidentCoeff) -> ResidentCoeff {
    let n = cf.lower_a.len();
    for i in 0..n {
        let sl = f32_to_f64_exact(cf.lower_a[i]) + f32_to_f64_exact(skip.lower_a[i]);
        let fl_l = sl as f32;
        let lower_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.lower_err[i]),
            f32_to_f64_exact(skip.lower_err[i]),
        );
        let lower_gap = (f32_to_f64_exact(fl_l) - sl).abs();
        cf.lower_err[i] = up_f32(add_nonnegative_f64_up(lower_err, lower_gap));
        cf.lower_a[i] = fl_l;
        let su = f32_to_f64_exact(cf.upper_a[i]) + f32_to_f64_exact(skip.upper_a[i]);
        let fl_u = su as f32;
        let upper_err = add_nonnegative_f64_up(
            f32_to_f64_exact(cf.upper_err[i]),
            f32_to_f64_exact(skip.upper_err[i]),
        );
        let upper_gap = (f32_to_f64_exact(fl_u) - su).abs();
        cf.upper_err[i] = up_f32(add_nonnegative_f64_up(upper_err, upper_gap));
        cf.upper_a[i] = fl_u;
    }
    // #u4: identical fail-closed rule as `merge_streams` (the skip add is also
    // an ADD — words OR; a word-less side poisons to `None`).
    cf.taint_rows = merge_taint_rows(cf.taint_rows.take(), skip.taint_rows.as_deref());
    cf
}

/// Round an `f64` DOWN to `f32` (outward toward −∞) for the final bias fold. The
/// UP counterpart (`up_f32`) and the error-sizing helpers (`gamma_k_f32`,
/// `combine_slack_f32`) now live in `crate::wgpu_device::sound_consts`.
fn down_f32(x: f64) -> f32 {
    f64_to_f32_down(x)
}

fn add_nonnegative_f64_up(lhs: f64, rhs: f64) -> f64 {
    if !lhs.is_finite() || !rhs.is_finite() || lhs < 0.0 || rhs < 0.0 {
        return f64::INFINITY;
    }
    if rhs == 0.0 {
        return lhs;
    }
    if lhs == 0.0 {
        return rhs;
    }
    next_up_f64(lhs + rhs)
}

/// Certified Cut-CROWN stem fold (`NY_MULTINEURON_STEM`, `sound_round=true`):
/// add `add` to a LOWER-side coefficient column, folding the f32 rounding gap
/// OUTWARD into the certified per-column error so the final concretization can
/// only widen (`concretize_error_into_bias` consumes `lower_err` via `up_f32`
/// and subtracts it). `a[idx]` keeps the nearest-f32 sum; the discrepancy
/// `|nearest − exact|` joins `err[idx]`. Sound for the lower objective: the
/// realized bound never exceeds the exact linear form. Mirrors the CPU-lane
/// `LinearBounds::add_to_lower_column` discipline (`linear.rs`).
fn fold_add_lower_coeff_outward(a: &mut [f32], err: &mut [f32], idx: usize, add: f32) {
    let exact = f32_to_f64_exact(a[idx]) + f32_to_f64_exact(add);
    let nearest = exact as f32;
    a[idx] = nearest;
    let gap = (f32_to_f64_exact(nearest) - exact).abs();
    err[idx] = up_f32(f32_to_f64_exact(err[idx]) + gap);
}

/// Certified Cut-CROWN stem fold: add `add` to a spec row's LOWER bias, rounding
/// the result DOWN (outward toward −∞ — the lower bias adds directly to the
/// lower bound, so rounding down is conservative) and widening `b_err` by the
/// non-negative rounding gap. The final bound is `down_f32(b − b_err)`, so both
/// a smaller `b` and a larger `b_err` can only lower it. Sound over-approx.
fn fold_add_lower_bias_outward(b: &mut f32, b_err: &mut f32, add: f32) {
    let exact = f32_to_f64_exact(*b) + f32_to_f64_exact(add);
    let rounded = down_f32(exact);
    *b = rounded;
    let gap = (exact - f32_to_f64_exact(rounded)).max(0.0);
    *b_err = up_f32(f32_to_f64_exact(*b_err) + gap);
}

/// Validate a resident cut fold against the exact `Activation` it would modify.
///
/// This check is deliberately all-or-nothing: post-activation coefficients, the
/// bias shift, and pre-activation coefficients are the three pieces of ONE
/// Lagrangian constraint. Applying only a valid-looking subset changes the
/// constraint and is not sound in general. Callers must run this before splitting
/// the branch or mutating any coefficient/bias channel.
fn resident_cut_fold_valid_for_activation(
    fold: &super::cut_fold_resident::ResidentCutFold,
    num_neurons: usize,
) -> bool {
    fold.bias_shift.is_finite()
        && fold.coeffs.iter().all(|&(idx, coeff)| {
            usize::try_from(idx).is_ok_and(|idx| idx < num_neurons) && coeff.is_finite()
        })
        && fold.pre_coeffs.iter().all(|&(idx, coeff)| {
            usize::try_from(idx).is_ok_and(|idx| idx < num_neurons) && coeff.is_finite()
        })
}

impl WgpuDevice {
    /// Fold one maximal unary DAG run from an owned, fully worded device
    /// carrier into a new owned carrier. The internal resident scratch remains
    /// unchanged; this seam only replaces host seed uploads and final readback
    /// with encoder-ordered device copies. The DAG driver applies its bounded
    /// completion fence after any destination merge, so this fold's scratch and
    /// the unit-local transform buffers share one explicitly charged lifetime.
    pub(super) fn crown_backward_sound_resident_sweep_carrier(
        &self,
        layers: &[GpuCrownLayer],
        seed: DeviceSweepCarrier,
    ) -> Result<DeviceSweepCarrier> {
        if !self.sound_gpu_authority_cached() {
            return Err(NyError::UnsupportedOp(
                "worded DAG resident carrier requires full-IEEE authority".into(),
            ));
        }
        if !self.taint_words_armed() {
            return Err(NyError::UnsupportedOp(
                "worded DAG resident carrier requires the taint-word route".into(),
            ));
        }
        let rows = seed.layout.rows;
        let output_dim = seed.layout.dim;
        let limits = self.device.limits();
        let fold = resident_fold_plan(
            layers,
            rows,
            rows,
            output_dim,
            limits.max_compute_workgroups_per_dimension,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        // Host-seeded callers count these two direct queue writes in their
        // transaction seed receipt. The worded device-seed route skips that
        // host seed, so count only the still-live ones vector and beta-zero
        // initialization here.
        let setup_bytes = fold
            .max_dim
            .max(1)
            .checked_add(fold.slope_dim)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                NyError::InvalidSpec("worded DAG resident setup transfer overflow".into())
            })?;
        super::intermediate_sweep::note_host_to_device(setup_bytes);
        let scope = SweepResidentIoGuard::arm(seed)?;
        let empty_f32: &[f32] = &[];
        let _empty_result = self.crown_backward_sound_resident_coeff_seeded_err(
            layers,
            empty_f32,
            empty_f32,
            empty_f32,
            empty_f32,
            empty_f32,
            empty_f32,
            empty_f32,
            empty_f32,
            rows,
            output_dim,
            &[],
            &[],
        )?;
        let output = scope.take_output()?;
        output.validate_owned_sizes()?;
        if !self.sound_gpu_authority_cached() {
            return Err(NyError::UnsupportedOp(
                "full-IEEE authority was lost after a worded DAG resident fold".into(),
            ));
        }
        Ok(output)
    }

    /// Sound resident CROWN backward over Linear layers (R1: single Linear, no
    /// bias). Returns `(lower, upper)` per spec row, matching
    /// `crown_backward_sound_host` but with the layer GEMMs kept on-device.
    /// Driven by `crown_backward_gpu_sound` (the `GpuCrownBackward` trait method).
    pub(crate) fn crown_backward_sound_resident(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Non-seeded entry: the spec C is exact and symmetric, bias 0.
        let zb = vec![0.0f32; num_specs];
        self.crown_backward_sound_resident_seeded(
            layers,
            spec,
            spec,
            &zb,
            &zb,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
        )
    }

    /// Seeded sound resident backward: start from an asymmetric frontier
    /// (`lower_a`/`upper_a` coefficients + `lower_b`/`upper_b` bias), as the
    /// graph alpha-CROWN suffix path does. The frontier coefficient is treated as
    /// EXACT (incoming error 0) and only the suffix's own f32 rounding is tracked
    /// — sound, and matching the CPU sound suffix path, which carries no coefficient
    /// error frontier on `LinearBounds`. (Composing a valid linear bound with sound
    /// suffix relaxations + tracked propagation rounding stays sound.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_seeded(
        &self,
        layers: &[GpuCrownLayer],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.crown_backward_sound_resident_coeff_seeded(
            layers, lower_a, upper_a, lower_b, upper_b, num_specs, output_dim,
        )?;
        self.concretize_resident_coeff(&c, num_specs, input_lower, input_upper)
    }

    /// Sound-concretize a (possibly composed) [`ResidentCoeff`] frontier: fold the
    /// bias error outward into the bias, then run the certified GPU concretize.
    pub(crate) fn concretize_resident_coeff(
        &self,
        c: &ResidentCoeff,
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // #batched-bab: single domain (per-domain count == total spec count).
        self.concretize_resident_coeff_batched(c, num_specs, num_specs, input_lower, input_upper)
    }

    /// #batched-bab: domain-block form of [`concretize_resident_coeff`]. `num_specs`
    /// is the TOTAL stacked-row count `N = n_domains * num_specs_per_dom`; the input
    /// box `input_lower`/`input_upper` is `n_domains * c.dim` wide (each domain block
    /// concretizes against its OWN box, HOLE 3). `num_specs_per_dom == num_specs`
    /// (single domain) → byte-identical to [`concretize_resident_coeff`].
    pub(crate) fn concretize_resident_coeff_batched(
        &self,
        c: &ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let bias_lower: Vec<f32> = (0..num_specs)
            .map(|s| down_f32(f32_to_f64_exact(c.lower_b[s]) - f32_to_f64_exact(c.lower_b_err[s])))
            .collect();
        let bias_upper: Vec<f32> = (0..num_specs)
            .map(|s| up_f32(f32_to_f64_exact(c.upper_b[s]) + f32_to_f64_exact(c.upper_b_err[s])))
            .collect();
        self.concretize_sound_gpu_batched(
            num_specs,
            num_specs_per_dom,
            c.dim,
            &c.lower_a,
            &c.upper_a,
            &c.lower_err,
            &c.upper_err,
            input_lower,
            input_upper,
            &bias_lower,
            &bias_upper,
            // #u4 C1 (TAINT_GUARD_AUDIT.md §4): the per-spec-row word slice the
            // MAIN resident walk — or the resnet segment composition, which
            // ORs its sub-walks' rows across every seam
            // (`resnet_seeded_compose_coeff`) — pre-OR'd under
            // the worded route (AUTO or explicit NY_GPU_TAINT_WORDS=1; final
            // coefficient/error words + every fail-closed no-twin transport).
            // The bias-err → bias outward fold above is per-spec-row and so
            // word-invariant. Explicit gate-off carries `None`, as do genuinely
            // unwordable segment-resident device streams; the armed C1 consult
            // treats either case as a typed fail-closed refusal.
            c.taint_rows.as_deref(),
        )
    }

    /// COEFFICIENT EGRESS (#cert-coeffs): the seeded sound resident walk's
    /// certified affine frontier, published across the `GpuCrownBackward`
    /// boundary as [`ny_core::CertifiedCoeffs`] INSTEAD of being concretized.
    ///
    /// This exists because the lane that actually proves the deep cifar100 rows
    /// consumes COEFFICIENTS (it folds them onward itself). Every pre-existing
    /// certified GPU entry concretizes on device, which makes it structurally
    /// unseamable there; whole-pass-and-concretize is the only shape it can
    /// express, and that is not the shape the lane needs.
    ///
    /// # Authority
    ///
    /// Publishing coefficients is strictly MORE authority than publishing the
    /// bound derived from them (the caller can concretize them over any box), so
    /// this runs the identical `#u4` C1 per-spec-row taint consult the
    /// concretize entry runs before it lets a row decide anything: absent,
    /// wrong-length, or nonzero words are a typed refusal, never a publication.
    /// The trait-level gate (`provides_sound_gpu_crown`) is enforced by the
    /// caller in `ops/crown_backward.rs`.
    pub(crate) fn crown_backward_sound_resident_certified_coeffs(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
    ) -> Result<ny_core::CertifiedCoeffs> {
        let num_specs = seed.num_specs;
        let c = self.crown_backward_sound_resident_coeff_seeded(
            layers,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            num_specs,
            seed.current_dim,
        )?;
        // ONE copy of the taint consult + fail-closed screens, shared with the
        // SEGMENT egress (`crown_backward_gpu_resnet_sound_certified_coeffs`).
        certified_coeffs_from_resident(c, num_specs)
    }

    /// SEGMENT COEFFICIENT EGRESS (#cert-coeffs-resnet): the resnet segment
    /// composition's COMPOSED certified frontier, published across the
    /// `GpuCrownBackward` boundary as [`ny_core::CertifiedCoeffs`] INSTEAD of
    /// being concretized.
    ///
    /// This is the residual twin of
    /// [`Self::crown_backward_sound_resident_certified_coeffs`]. It exists
    /// because EVERY cifar100/tinyimagenet net the margin-row lane must
    /// accelerate is a resnet: the flat egress refuses the moment the lane's op
    /// walk meets a residual `Add`, so the lane's seam measured
    /// `gpu_seam_ok=0 / gpu_seam_refused=2` — the seam never ran.
    ///
    /// # What is published
    ///
    /// The single `ResidentCoeff` returned by
    /// [`Self::resnet_seeded_compose_coeff`] AFTER the whole segment loop — i.e.
    /// the frontier COMPOSED ACROSS every segment (each `Chain` folded, each
    /// `Residual` merged with `add_skip_stream`, and each `ResidualProj` merged
    /// with `merge_streams`), at `coeff.dim == ` the network input width. It is
    /// never a single segment's intermediate frontier: the loop replaces
    /// `coeff` on every merge and only the final value is returned.
    ///
    /// # Coefficient authority
    ///
    /// A [`ny_core::CertifiedCoeffs`] radius bounds the exact coefficient itself,
    /// independently of any input box. Therefore this entry deliberately passes
    /// EMPTY per-segment and per-ReLU abs tables to the composition and disables
    /// both forced folds. Moving `err_a[j] * abs(z[j])` into bias and zeroing
    /// `err_a[j]` is sound for a functional on that supplied domain, but it is
    /// not a coefficient-wise enclosure and must never cross this API boundary.
    /// Environment gates cannot override the empty-table prerequisite.
    ///
    /// The one un-concretized frontier goes through the SAME firewall as the
    /// flat egress ([`certified_coeffs_from_resident`]): taint consult,
    /// non-finite screen, `FALLBACK_BOUND` saturation-sentinel screen, and
    /// negative-radius screen. A refusal propagates and the caller runs its CPU
    /// path; there is no domain-concretized recovery pass.
    ///
    /// # Authority
    ///
    /// Same as the flat egress: publishing coefficients is strictly more
    /// authority than publishing a bound derived from them, so the trait-level
    /// `provides_sound_gpu_crown` gate is enforced by the caller in
    /// `ops/crown_backward.rs`, and the `#u4` C1 per-spec-row taint consult runs
    /// inside the shared firewall.
    pub(crate) fn crown_backward_gpu_resnet_sound_certified_coeffs(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
    ) -> Result<ny_core::CertifiedCoeffs> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_coeffs: empty segment list".into(),
            ));
        }
        // Same borrowed translation the bounds entry performs.
        let internal: Vec<ResnetSegment> = segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let num_specs = seed.num_specs;
        // #batched-bab: single-domain caller (per-dom == total), matching
        // `crown_backward_sound_resident_resnet_seeded`.
        self.resnet_certified_coeffs_unconcretized(&internal, seed, num_specs, num_specs)
    }

    /// The SHARED un-concretized composition + firewall behind BOTH resnet
    /// coefficient egresses — the single-domain one above and the multi-domain
    /// [`Self::crown_backward_gpu_resnet_sound_batched_certified_coeffs`].
    ///
    /// `num_specs` is the TOTAL stacked-row count `N`; `num_specs_per_dom` is
    /// the shared per-domain spec-row count (`N / n_domains`). With one domain
    /// they are equal and this is byte-identical to what the single-domain
    /// egress did before the batched sibling existed.
    ///
    /// Keeping ONE copy is deliberate: empty abs tables plus disabled forced
    /// folds preserve true coefficient errors, and the fail-closed
    /// [`certified_coeffs_from_resident`] firewall screens the result. A second
    /// transcription is a place for the batched path to silently weaken. Pinned
    /// by `batched_certified_coeffs_share_the_single_domain_unconcretized_walk`.
    fn resnet_certified_coeffs_unconcretized(
        &self,
        internal: &[ResnetSegment],
        seed: &GpuCrownSeed,
        num_specs: usize,
        num_specs_per_dom: usize,
    ) -> Result<ny_core::CertifiedCoeffs> {
        let (coeff, _grads, _gathers) = self.resnet_seeded_compose_coeff(
            internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            num_specs,
            num_specs_per_dom,
            seed.current_dim,
            &[],
            &[],
            &[],
            &[],
            false,
            &[],
            false,
        )?;
        certified_coeffs_from_resident(coeff, num_specs)
    }

    /// BATCHED SEGMENT COEFFICIENT EGRESS (#margin-row-gpu-batch): fold `N`
    /// domains' backward in ONE wide resident pass and publish `N` COMPOSED
    /// certified frontiers, one per domain, un-concretized.
    ///
    /// The caller (`ops/crown_backward.rs`) has already run the authority gate,
    /// the HOLE-7 homogeneity gate, the HOLE-8 unbatchable-layer gate and the
    /// device dispatch-limit check, and has stacked the per-domain relaxation
    /// blocks onto ONE shared skeleton (`stack_wide_segments`) with the shared
    /// spec seed tiled `n_domains` times. This function is the resident half:
    /// it runs the SAME un-concretized composition + firewall as the single-domain
    /// egress over the `N = n_domains * num_specs_per_dom` stacked rows, then
    /// SPLITS the wide frontier back into per-domain payloads.
    ///
    /// # The slot mapping (the killer defect)
    ///
    /// `resnet_seeded_compose_coeff` lays the wide frontier out DOMAIN-MAJOR:
    /// row `s` belongs to domain `s / num_specs_per_dom`, which is exactly the
    /// layout its own `dom = row/num_specs_per_dom` indexing of the per-domain
    /// Activation blocks assumes. The coefficient path supplies no domain box
    /// or abs-max data to this composition. Domain `d` owns the CONTIGUOUS row
    /// range `[d*nsp, (d+1)*nsp)`, so the split is a pure reshape. The split
    /// below is written as a single ordered `chunks_exact` walk with no free
    /// index, and the row count is checked against `n_domains * nsp` BEFORE any
    /// slicing, so a short or long payload refuses instead of associating one
    /// domain's coefficients with another domain's relaxation.
    ///
    /// # Fail-closed
    ///
    /// The firewall runs ONCE over the WHOLE wide payload. A poisoned row
    /// anywhere refuses the ENTIRE batch — never a partial answer, never a
    /// per-domain rescue. The caller then runs its one-at-a-time path.
    pub(crate) fn crown_backward_gpu_resnet_sound_batched_certified_coeffs(
        &self,
        wide_segments: &[GpuResnetSegment],
        wide_seed: &GpuCrownSeed,
        num_specs_per_dom: usize,
        n_domains: usize,
    ) -> Result<Vec<ny_core::CertifiedCoeffs>> {
        if wide_segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_batched_coeffs: empty segment list".into(),
            ));
        }
        if n_domains == 0 || num_specs_per_dom == 0 {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_batched_coeffs: empty batch".into(),
            ));
        }
        let expected = num_specs_per_dom
            .checked_mul(n_domains)
            .ok_or_else(|| NyError::InvalidSpec("batched coeffs: row count overflow".into()))?;
        if wide_seed.num_specs != expected {
            return Err(NyError::shape_mismatch(
                vec![expected],
                vec![wide_seed.num_specs],
            ));
        }
        // Capacity MUST be known before segment one submits. Only the typed
        // capacity variant may drive the caller's narrowing ladder; every
        // validation/firewall/deadline/device error remains terminal.
        let limits = self.device.limits();
        let final_dim = batched_coefficient_segment_preflight(
            wide_segments,
            expected,
            num_specs_per_dom,
            wide_seed.current_dim,
            limits.max_compute_workgroups_per_dimension,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        if final_dim == 0 {
            return Err(NyError::InvalidSpec(
                "batched coeffs: preflight produced a zero-width frontier".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = wide_segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let wide = self.resnet_certified_coeffs_unconcretized(
            &internal,
            wide_seed,
            expected,
            num_specs_per_dom,
        )?;
        split_batched_certified_coeffs(&wide, num_specs_per_dom, n_domains)
    }

    /// Run the (seeded) resident backward and return the raw coefficient frontier
    /// (no concretize) — the composable form used by the residual path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_coeff_seeded(
        &self,
        layers: &[GpuCrownLayer],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        output_dim: usize,
    ) -> Result<ResidentCoeff> {
        let limits = self.device.limits();
        let plan = resident_fold_plan(
            layers,
            num_specs,
            num_specs,
            output_dim,
            limits.max_compute_workgroups_per_dimension,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        let za = vec![0.0f32; plan.seed_elems];
        let zb2 = vec![0.0f32; num_specs];
        self.crown_backward_sound_resident_coeff_seeded_err(
            layers,
            lower_a,
            upper_a,
            &za,
            &za,
            lower_b,
            upper_b,
            &zb2,
            &zb2,
            num_specs,
            output_dim,
            &[],
            &[],
        )
    }

    /// Like [`crown_backward_sound_resident_coeff_seeded`] but the seed carries an
    /// INCOMING coefficient/bias error (`*_a_err`, `*_b_err`) — required when
    /// composing segments (e.g. stacked residual blocks): the previous segment's
    /// error must propagate through this one, not be dropped to 0. Seeds `le[0]` /
    /// bias-error buffers from these instead of zero.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_coeff_seeded_err(
        &self,
        layers: &[GpuCrownLayer],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_a_err: &[f32],
        upper_a_err: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        lower_b_err: &[f32],
        upper_b_err: &[f32],
        num_specs: usize,
        output_dim: usize,
        relu_pre_lower: &[&[f32]],
        beta_signed: &[&[f32]],
    ) -> Result<ResidentCoeff> {
        self.crown_backward_sound_resident_coeff_seeded_err_gather(
            layers,
            lower_a,
            upper_a,
            lower_a_err,
            upper_a_err,
            lower_b,
            upper_b,
            lower_b_err,
            upper_b_err,
            num_specs,
            num_specs, // #batched-bab: single-domain caller (per-dom == total).
            output_dim,
            relu_pre_lower,
            beta_signed,
            &[],
        )
    }

    /// Gather-capable form of [`crown_backward_sound_resident_coeff_seeded_err`]
    /// (#w4-split-tightening): identical bound computation, plus an optional
    /// per-`Activation` A-value GATHER channel for the analytic β gradient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_coeff_seeded_err_gather(
        &self,
        layers: &[GpuCrownLayer],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_a_err: &[f32],
        upper_a_err: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        lower_b_err: &[f32],
        upper_b_err: &[f32],
        num_specs: usize,
        // #batched-bab: per-domain spec-row count. `num_specs` is the TOTAL stacked-row
        // count `N = n_domains * num_specs_per_dom`; per-domain Activation state (slopes/
        // intercepts/β) is stacked in `n_domains` blocks and the resident shaders index
        // it at `dom*num_neurons`, `dom = row/num_specs_per_dom` (HOLES 1/2). With
        // `num_specs_per_dom == num_specs` (single domain) `n_domains == 1`, every `dom`
        // is 0 and every stacked buffer is one block wide → byte-identical to the
        // pre-batch verdict path.
        num_specs_per_dom: usize,
        output_dim: usize,
        // Gradient-capable warmup (#unsat-keystone): per-`Activation`-layer (backward
        // order) masked pre-activation lower bounds. Empty ⇒ no capture (the verdict
        // path), making the bound computation byte-for-byte unchanged. When provided,
        // each ReLU's analytic alpha gradient is captured into `ResidentCoeff.relu_grads`.
        relu_pre_lower: &[&[f32]],
        // Beta-capable per-domain backward (#unsat-keystone step 4): per-`Activation`-layer
        // (backward order) signed beta `β·sign` per neuron (0 for non-split neurons).
        // Empty ⇒ no beta (byte-for-byte unchanged). Folds the β-CROWN split-constraint
        // dual into the POST-slope coefficient (lower −=, upper +=); sound for any β≥0.
        beta_signed: &[&[f32]],
        // Beta-GRADIENT gather (#w4-split-tightening): per-`Activation`-layer (backward
        // order) neuron column indices whose PRE-transform LOWER coefficient values are
        // read back (`ResidentCoeff.beta_gather`, row-major num_specs × idx.len()) — the
        // CPU `a_at_relu` capture point for `∂lb/∂β = −sign·A_lower[row, k]`. Empty ⇒ no
        // capture (byte-for-byte unchanged: the gather only COPIES from the coefficient
        // buffer, never writes bound state).
        beta_gather_idx: &[&[u32]],
    ) -> Result<ResidentCoeff> {
        // #seg-resident: take any armed device-seed/keep state FIRST — with a
        // device seed the host slices are unused placeholders, so the host
        // shape checks below are skipped (the device seed carries its own
        // dim/num_specs, validated at the copy site). TAKEN (reset) so a stale
        // state can never leak into an unrelated later fold call.
        let (dev_seed, dev_zero_bias, dev_keep, sweep_seed, sweep_keep) = RESIDENT_IO.with(|io| {
            let mut io = io.borrow_mut();
            (
                io.seed.take(),
                std::mem::take(&mut io.zero_bias_seed),
                std::mem::take(&mut io.keep),
                io.sweep_seed.take(),
                std::mem::take(&mut io.sweep_keep),
            )
        });
        let device_limits = self.device.limits();
        let plan = resident_fold_plan(
            layers,
            num_specs,
            num_specs_per_dom,
            output_dim,
            device_limits.max_compute_workgroups_per_dimension,
            device_limits.max_buffer_size,
            device_limits.max_storage_buffer_binding_size,
        )?;
        let ResidentFoldPlan {
            num_specs_u32,
            num_specs_per_dom_u32,
            n_domains,
            seed_elems,
            final_dim,
            max_dim,
            max_gemm_out,
            a_elems,
            slope_dim,
            max_wg,
        } = plan;

        if dev_seed.is_some() && sweep_seed.is_some() {
            return Err(NyError::InternalError(
                "resident fold received both legacy and worded device seeds".into(),
            ));
        }
        if dev_keep && sweep_keep {
            return Err(NyError::InternalError(
                "resident fold received both legacy and worded keep requests".into(),
            ));
        }
        if let Some(seed) = sweep_seed.as_ref() {
            if seed.layout.dim != output_dim || seed.layout.rows != num_specs {
                return Err(NyError::shape_mismatch(
                    vec![num_specs, output_dim],
                    vec![seed.layout.rows, seed.layout.dim],
                ));
            }
            seed.validate_owned_sizes()?;
        } else if let Some(seed) = dev_seed.as_ref() {
            if seed.dim != output_dim || seed.num_specs != num_specs {
                return Err(NyError::shape_mismatch(
                    vec![num_specs, output_dim],
                    vec![seed.num_specs, seed.dim],
                ));
            }
            let coefficient_bytes = resident_f32_bytes(seed_elems, "device seed coefficient")?;
            let bias_bytes = resident_f32_bytes(num_specs, "device seed bias")?;
            for (name, actual, required) in [
                ("la", seed.la.size(), coefficient_bytes),
                ("ua", seed.ua.size(), coefficient_bytes),
                ("le", seed.le.size(), coefficient_bytes),
                ("ue", seed.ue.size(), coefficient_bytes),
                ("blo", seed.blo.size(), bias_bytes),
                ("buo", seed.buo.size(), bias_bytes),
                ("ble", seed.ble.size(), bias_bytes),
                ("bue", seed.bue.size(), bias_bytes),
            ] {
                if actual < required {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: device seed buffer {name} has \
                         {actual} bytes, needs {required}"
                    )));
                }
            }
        } else {
            for (name, actual, expected) in [
                ("lower_a", lower_a.len(), seed_elems),
                ("upper_a", upper_a.len(), seed_elems),
                ("lower_a_err", lower_a_err.len(), seed_elems),
                ("upper_a_err", upper_a_err.len(), seed_elems),
                ("lower_b", lower_b.len(), num_specs),
                ("upper_b", upper_b.len(), num_specs),
                ("lower_b_err", lower_b_err.len(), num_specs),
                ("upper_b_err", upper_b_err.len(), num_specs),
            ] {
                if actual != expected {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: {name}.len()={actual} != {expected}"
                    )));
                }
            }
        }
        if (dev_keep || sweep_keep) && (!relu_pre_lower.is_empty() || !beta_gather_idx.is_empty()) {
            return Err(NyError::UnsupportedOp(
                "seg-resident keep mode with capture channels armed".into(),
            ));
        }

        let activation_dims: Vec<usize> = layers
            .iter()
            .filter_map(|layer| match layer {
                GpuCrownLayer::Activation { num_neurons, .. } => Some(*num_neurons),
                _ => None,
            })
            .collect();
        for (channel, values) in [
            ("relu_pre_lower", relu_pre_lower),
            ("beta_signed", beta_signed),
        ] {
            if values.len() > activation_dims.len() {
                return Err(NyError::InvalidSpec(format!(
                    "crown_backward_sound_resident: {channel} has {} entries for {} \
                     activation layers",
                    values.len(),
                    activation_dims.len()
                )));
            }
            for (activation, (values, &neurons)) in values.iter().zip(&activation_dims).enumerate()
            {
                let expected = resident_checked_product(
                    &[n_domains, neurons],
                    "capture domain-state elements",
                )?;
                if values.len() != expected {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_backward_sound_resident: {channel}[{activation}].len()={} \
                         != n_domains*num_neurons={expected}",
                        values.len()
                    )));
                }
            }
        }
        if beta_gather_idx.len() > activation_dims.len() {
            return Err(NyError::InvalidSpec(format!(
                "crown_backward_sound_resident: beta_gather_idx has {} entries for {} \
                 activation layers",
                beta_gather_idx.len(),
                activation_dims.len()
            )));
        }
        for (activation, indices) in beta_gather_idx.iter().enumerate() {
            resident_checked_u32(indices.len(), "gather index count")?;
            let gather_elems =
                resident_checked_product(&[num_specs, indices.len()], "gather output elements")?;
            resident_checked_u32(gather_elems, "gather output elements")?;
            let gather_bytes = resident_f32_bytes(gather_elems.max(1), "gather output")?;
            if gather_bytes > device_limits.max_buffer_size {
                return Err(NyError::UnsupportedOp(format!(
                    "crown_backward_sound_resident: beta_gather_idx[{activation}] needs \
                     {gather_bytes} bytes, max_buffer_size={}",
                    device_limits.max_buffer_size
                )));
            }
            if gather_elems > LEGACY_BETA_GATHER_MAX_COPIES {
                let index_bytes = resident_f32_bytes(indices.len().max(1), "gather index storage")?;
                let max_storage = device_limits.max_storage_buffer_binding_size;
                if gather_bytes > max_storage || index_bytes > max_storage {
                    return Err(NyError::UnsupportedOp(format!(
                        "crown_backward_sound_resident: beta gather storage exceeds \
                         max_storage_buffer_binding_size={max_storage}"
                    )));
                }
                let dispatch = gather_elems.div_ceil(256);
                if dispatch > max_wg {
                    return Err(NyError::UnsupportedOp(format!(
                        "crown_backward_sound_resident: beta gather dispatch {dispatch} \
                         exceeds max_compute_workgroups_per_dimension {max_wg}"
                    )));
                }
            }
        }

        // Resolve all verdict-sensitive arithmetic certificates before the first
        // GPU allocation. A later layer cannot partially run and then discover
        // that its reduction has no finite Higham/recovery bound.
        //
        // #u4: resolve AUTO / explicit opt-in / explicit opt-out ONCE per walk
        // entry; see `gpu_taint_words_env` for the contract.
        let taint_on = self.taint_words_armed();
        if taint_on && (dev_seed.is_some() || dev_keep) {
            // #u4 fail-closed: the #seg-resident device streams (seed-in /
            // keep-out) carry NO word channel — half-wiring them would launder
            // every word at the segment boundary. Refuse loudly; the resnet
            // orchestrator's un-worded path still works with the gate off.
            return Err(NyError::UnsupportedOp(
                "the worded resident route (AUTO or NY_GPU_TAINT_WORDS=1) is \
                 not wired for seg-resident device \
                 seed/keep streams (no word channel across device-resident \
                 segment boundaries yet) — refusing (fail-closed)"
                    .into(),
            ));
        }
        if (sweep_seed.is_some() || sweep_keep) && !taint_on {
            return Err(NyError::UnsupportedOp(
                "worded intermediate-sweep carrier requires the armed taint-word route".into(),
            ));
        }
        let has_conv = layers
            .iter()
            .any(|layer| matches!(layer, GpuCrownLayer::Conv2d { .. }));
        debug_assert!(taint_walk_conv_route_admitted(taint_on, has_conv));

        // #rung3-denorm-uniformity: resident pipelines are lazy. Build every
        // resident/EFT module BEFORE accepting the cached adapter probes so a
        // requested DenormPreserve passthrough failure can poison the cache
        // read. Otherwise the probe could pass first and a later production
        // module could silently fall back to plain WGSL in this same walk.
        let eft_requested = eft_err_env_enabled();
        // Every authoritative resident walk consumes this lazy pipeline set,
        // including its taint twins even when EFT tightening is disabled. Build
        // it before re-reading authority so a passthrough failure permanently
        // closes the exact device before any verdict-bearing dispatch.
        let _ = self.resident_backward_pipelines();
        if !self.sound_gpu_authority_cached() && self.charged_flush_authority_cached().is_none() {
            return Err(NyError::UnsupportedConfiguration(
                "WGPU verdict authority closed while materializing resident shaders; \
                 refusing the GPU walk (fail-closed)"
                    .to_string(),
            ));
        }
        let eft_on = eft_requested && self.eft_primitives_cached();
        // #flush-charge: charged-flush authority guard + arming. On a charged
        // device (rung 3 refused, PURE-FLUSH class, typed constructor) every
        // un-audited or un-chargeable channel is refused at walk entry and the
        // audited DAZ covers are armed. `None` on every other device — then
        // `charged_walk_guard` never runs and `daz_cover_armed` reduces to the
        // dark env gate, so every uniform below is byte-identical.
        let charged_policy = self.charged_flush_authority_cached().copied();
        if let Some(policy) = charged_policy.as_ref() {
            charged_walk_guard(layers, policy, eft_requested)?;
        }
        let daz_cover_armed = crate::wgpu_device::sound_consts::daz_flush_cover_v2_enabled()
            || charged_policy.is_some();
        let conv_err_rowmax = std::env::var("NY_CONV_ERR_ROWMAX").ok().as_deref() == Some("1");
        if taint_on && has_conv && conv_err_rowmax {
            // The legacy diagnostic broadcasts a row-max error without an
            // elementwise word output. The default per-entry Conv path is fully
            // worded; keep this opt-in comparison mode fail-closed rather than
            // pretending its coarser error kernel transports words.
            return Err(NyError::UnsupportedOp(
                "NY_CONV_ERR_ROWMAX=1 is a legacy unworded Conv diagnostic and \
                 cannot run with the armed taint-word authority; unset it or set \
                 NY_GPU_TAINT_WORDS=0"
                    .into(),
            ));
        }
        if taint_on && self.resident_backward_pipelines().gemm_taint.is_none() {
            // #u4 fail-closed: the twins were not built because the granted
            // storage-buffer limit cannot host the 11-binding activation twin
            // (e.g. NY_GPU_BIG_BINDINGS=0 ⇒ wgpu default 8). Refuse the worded
            // walk; the caller falls back to the un-worded/CPU path.
            return Err(NyError::UnsupportedOp(format!(
                "NY_GPU_TAINT_WORDS=1 needs 11 storage buffers per stage; this \
                 device granted {} — taint twins unavailable, refusing the \
                 worded walk (fail-closed)",
                self.device.limits().max_storage_buffers_per_shader_stage
            )));
        }
        // ---- #cert-err preflight (fail-closed) ----------------------------
        // Prove, BEFORE any dispatch, that every declared `CertifiedWeightError`
        // is usable and that this walk is in a mode that actually charges it.
        // Layers declaring the exact-weight default skip the whole block, so
        // the pre-`cert_err` behaviour is untouched.
        let cert_bias_charge = cert_bias_charge_required(layers);
        for (index, layer) in layers.iter().enumerate() {
            let cert_err = layer_cert_err(layer);
            if cert_err.is_exact() {
                continue;
            }
            if !cert_err.is_valid() {
                return Err(NyError::UnsupportedOp(format!(
                    "#cert-err: layer {index} declares an unusable \
                     CertifiedWeightError (weight_rel_err={:e}, \
                     bias_abs_err={:e}); both must be finite and >= 0 — \
                     refusing (fail-closed)",
                    cert_err.weight_rel_err, cert_err.bias_abs_err
                )));
            }
            if conv_err_rowmax {
                // The legacy row-max conv error multiplies by ‖W‖₁ instead of
                // running the per-entry combine this charge is derived for.
                return Err(NyError::UnsupportedOp(
                    "#cert-err: NY_CONV_ERR_ROWMAX=1 selects the legacy row-max \
                     conv error, which is not the per-entry combine the \
                     certified weight-error charge is derived for; unset it — \
                     refusing (fail-closed)"
                        .into(),
                ));
            }
            if eft_on {
                // The EFT min-combine MEASURES the rounding of `A@W` for the
                // SUPPLIED weights. Its a-posteriori bound therefore contains no
                // `weight_rel_err` term at all, and `min(higham_charged, eft)`
                // would hand back exactly the charge we just added.
                return Err(NyError::UnsupportedOp(
                    "#cert-err: the EFT residual channel (NY_EFT_ERR=1) bounds \
                     only the rounding of A@W for the SUPPLIED weights, so \
                     min(higham, eft) would erase the certified weight-error \
                     charge; refusing the combination (fail-closed)"
                        .into(),
                ));
            }
            match layer {
                GpuCrownLayer::Linear { out_features, .. } => {
                    let g = cert_err.charged_gamma(gamma_k_f32(*out_features)?);
                    if !g.is_finite() {
                        return Err(NyError::UnsupportedOp(format!(
                            "#cert-err: layer {index} has no finite charged gamma                              — refusing (fail-closed)"
                        )));
                    }
                    cert_charged_slack(combine_slack_f32(*out_features)?, cert_err, index)?;
                }
                GpuCrownLayer::Conv2d {
                    out_channels,
                    kernel_h,
                    kernel_w,
                    ..
                } => {
                    let conv_reduction = resident_checked_product(
                        &[*out_channels, *kernel_h, *kernel_w],
                        "conv reduction length",
                    )?;
                    let g = cert_err.charged_gamma(gamma_k_f32(conv_reduction)?);
                    if !g.is_finite() {
                        return Err(NyError::UnsupportedOp(format!(
                            "#cert-err: layer {index} has no finite charged gamma                              — refusing (fail-closed)"
                        )));
                    }
                    cert_charged_slack(combine_slack_f32(conv_reduction)?, cert_err, index)?;
                }
                _ => {}
            }
        }
        for layer in layers {
            match layer {
                GpuCrownLayer::Activation { num_neurons, .. } => {
                    gamma_k_f32(*num_neurons)?;
                    combine_slack_f32(*num_neurons)?;
                    if eft_on {
                        eft_r_slack_f32(*num_neurons)?;
                    }
                }
                GpuCrownLayer::Linear { out_features, .. } => {
                    gamma_k_f32(*out_features)?;
                    combine_slack_f32(*out_features)?;
                    if eft_on {
                        eft_r_slack_f32(*out_features)?;
                    }
                }
                GpuCrownLayer::Conv2d {
                    bias_expanded,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                    ..
                } => {
                    let conv_reduction = resident_checked_product(
                        &[*out_channels, *kernel_h, *kernel_w],
                        "conv reduction length",
                    )?;
                    gamma_k_f32(conv_reduction)?;
                    if !conv_err_rowmax {
                        combine_slack_f32(conv_reduction)?;
                        if eft_on {
                            eft_r_slack_f32(conv_reduction)?;
                        }
                    }
                    if bias_expanded.is_some() {
                        let bias_reduction = resident_checked_product(
                            &[*out_channels, *out_h, *out_w],
                            "conv bias reduction length",
                        )?;
                        gamma_k_f32(bias_reduction)?;
                        combine_slack_f32(bias_reduction)?;
                        if eft_on {
                            eft_r_slack_f32(bias_reduction)?;
                        }
                    }
                }
                _ => unreachable!("resident_fold_plan rejected unsupported layers"),
            }
        }

        let coalesce = fold_coalesce_enabled();
        if cert_bias_charge && coalesce {
            // #cert-err fail-closed: the bias-error charge writes its constant
            // `[d; k]` operand and its own uniform with plain queue writes, which
            // are submission-ordered. Under NY_FOLD_COALESCE=1 the whole fold is
            // one deferred submission whose uploads must be encoder-ordered arena
            // copies, and the arena is sized from the pre-`cert_err` per-layer
            // budget. Refuse rather than risk an ordering/sizing hazard.
            return Err(NyError::UnsupportedOp(
                "#cert-err: the certified bias-error charge is incompatible with \
                 NY_FOLD_COALESCE=1 (its operand uploads are submission-ordered, \
                 not arena copies); unset it — refusing (fail-closed)"
                    .into(),
            ));
        }
        if taint_on && coalesce {
            // #u4 fail-closed: kept even though the transports are now fully
            // on-device (encoder-ordered dispatches, no mid-walk readbacks —
            // in principle single-submission-safe). The per-dispatch taint
            // uniforms are mapped-at-creation (submission-order independent),
            // but the worded walk has only ever been validated on the
            // per-layer-submit path; lifting this refusal needs its own
            // differential session, not a drive-by.
            return Err(NyError::UnsupportedOp(
                "the worded resident route (AUTO or NY_GPU_TAINT_WORDS=1) is \
                 incompatible with NY_FOLD_COALESCE=1 \
                 (the worded walk is validated only on the per-layer-submit \
                 path) — refusing (fail-closed)"
                    .into(),
            ));
        }
        // #u4: the all-zero `taint_b` word buffer must span every weight-shaped
        // GEMM operand the twins bind (Linear weights and the row-L1 `ones`
        // vector and every Linear/Conv weight operand).
        let taint_zw_elems = if taint_on {
            let mut span = max_dim.max(1);
            for layer in layers {
                match layer {
                    GpuCrownLayer::Linear {
                        out_features,
                        in_features,
                        ..
                    } => {
                        span = span.max(resident_checked_product(
                            &[*out_features, *in_features],
                            "taint zero-word span",
                        )?);
                    }
                    GpuCrownLayer::Conv2d { weight_col, .. } => {
                        span = span.max(weight_col.len());
                    }
                    GpuCrownLayer::Activation { .. } => {}
                    _ => {}
                }
            }
            span
        } else {
            0
        };
        let fold_staging_cap = if coalesce {
            let capacity = resident_fold_staging_capacity(layers, n_domains)?;
            if capacity > device_limits.max_buffer_size {
                return Err(NyError::UnsupportedOp(format!(
                    "crown_backward_sound_resident: fold staging needs {capacity} bytes, \
                     max_buffer_size={}",
                    device_limits.max_buffer_size
                )));
            }
            capacity
        } else {
            0
        };
        // (#lever1 weight residency) The former shared `max_w`-sized weight
        // scratch (`res_w`/`res_abs_w`, re-written per layer per call) is gone:
        // each Linear/Conv2d layer now binds its own GPU-resident buffer from
        // `resident_weight_buf` (Arc-identity keyed, keep-alive guarded,
        // uploaded once per model). Same shader bindings, same bytes.

        // Resident dispatch + single download, under the GPU-serialize lock.
        // The sound concretize runs OUTSIDE this closure: it re-locks the same
        // (non-reentrant) gpu_serialize mutex, so calling it here would deadlock.
        #[allow(clippy::type_complexity)]
        let (fla, fua, fle, fue, fblo, fbuo, fble, fbue, f_relu_grads, f_beta_gather, f_taint_rows): (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<Vec<f32>>,
            Vec<Vec<f32>>,
            // #u4: Some(per-spec-row words) iff taint_on; None ⇒ gate off.
            Option<Vec<u32>>,
        ) = self.run_gpu_checked_with_crown_deadline("crown_backward_sound_resident", || {
            // #NY_WIDE_PROBE: per-resident-call phase breakdown so STEP-1 profiling can
            // attribute the wide-node chunk overhead (setup / CPU weight-prep / gpu
            // submit / readback). Inert unless the probe env is set.
            let __probe = std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1");
            let __t_start = std::time::Instant::now();
            let mut __cpu_wprep = std::time::Duration::ZERO;
            // Cached, build-once resident pipelines (pure compiled shaders, no
            // numerical data) — reused across every segment/sub-chain instead of
            // recompiled per call. Bit-for-bit identical bounds; only removes
            // redundant shader/pipeline compilation from the hot path.
            let res_pipes = self.resident_backward_pipelines();
            let abs_pipe = &res_pipes.abs;
            let combine_pipe = &res_pipes.combine;
            let bias_pipe = &res_pipes.bias;
            let act_pipe = &res_pipes.act;
            let act_bias_pipe = &res_pipes.act_bias;
            // Conv pipelines (cached on the device — see the
            // `ResidentBackwardPipelines` field docs for the per-call-creation
            // driver-crash history).
            let conv_pipes = has_conv.then_some((
                &res_pipes.conv_reshape,
                &res_pipes.conv_col2im,
                &res_pipes.conv_err,
            ));

            let storage = |label: &str, n: usize| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (n.max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let zeros = vec![0.0f32; a_elems.max(num_specs)];

            // Ping-pong coefficient + error buffers (resident across the whole loop).
            let la = [storage("res_la0", a_elems), storage("res_la1", a_elems)];
            let ua = [storage("res_ua0", a_elems), storage("res_ua1", a_elems)];
            let le = [storage("res_le0", a_elems), storage("res_le1", a_elems)];
            let ue = [storage("res_ue0", a_elems), storage("res_ue1", a_elems)];
            // Running bias buffers (seeded below).
            let blo = storage("res_blo", num_specs);
            let buo = storage("res_buo", num_specs);
            let ble = storage("res_ble", num_specs);
            let bue = storage("res_bue", num_specs);

            // #seg-resident: the armed device seed/keep state was TAKEN (reset)
            // at fn entry (before the host shape checks) so a stale state can
            // never leak into an unrelated later fold call.
            if let Some(sd) = &sweep_seed {
                // Authoritative worded sweep seed: copy all value/error lanes
                // on-device. Its word twins and row accumulator are copied
                // after the taint scratch set has been allocated below.
                let seed_bytes = resident_f32_bytes(seed_elems, "worded sweep device seed")?;
                let bias_bytes = resident_f32_bytes(num_specs, "worded sweep bias seed")?;
                let mut se = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("res_sweep_seed_values"),
                    });
                se.copy_buffer_to_buffer(
                    &sd.matrix.lower_center,
                    0,
                    &la[0],
                    0,
                    seed_bytes,
                );
                se.copy_buffer_to_buffer(
                    &sd.matrix.upper_center,
                    0,
                    &ua[0],
                    0,
                    seed_bytes,
                );
                se.copy_buffer_to_buffer(
                    &sd.matrix.lower_radius,
                    0,
                    &le[0],
                    0,
                    seed_bytes,
                );
                se.copy_buffer_to_buffer(
                    &sd.matrix.upper_radius,
                    0,
                    &ue[0],
                    0,
                    seed_bytes,
                );
                if seed_elems < a_elems {
                    se.clear_buffer(&le[0], seed_bytes, None);
                    se.clear_buffer(&ue[0], seed_bytes, None);
                }
                for (source, destination) in [
                    (&sd.row.lower_bias, &blo),
                    (&sd.row.upper_bias, &buo),
                    (&sd.row.lower_bias_radius, &ble),
                    (&sd.row.upper_bias_radius, &bue),
                ] {
                    se.copy_buffer_to_buffer(source, 0, destination, 0, bias_bytes);
                }
                self.submit_ticked(se.finish());
            } else if let Some(sd) = &dev_seed {
                // Device-resident seed: encoder-ordered buffer copies replace the
                // host-slice uploads. Dimensions were checked before allocation.
                let seed_bytes = resident_f32_bytes(seed_elems, "device seed")?;
                let bias_bytes = resident_f32_bytes(num_specs, "device bias seed")?;
                let mut se = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("res_seed_dev"),
                    });
                se.copy_buffer_to_buffer(&sd.la, 0, &la[0], 0, seed_bytes);
                se.copy_buffer_to_buffer(&sd.ua, 0, &ua[0], 0, seed_bytes);
                se.copy_buffer_to_buffer(&sd.le, 0, &le[0], 0, seed_bytes);
                se.copy_buffer_to_buffer(&sd.ue, 0, &ue[0], 0, seed_bytes);
                // Zero the err lanes' unused tail (mirrors the host path).
                if seed_elems < a_elems {
                    se.clear_buffer(&le[0], seed_bytes, None);
                    se.clear_buffer(&ue[0], seed_bytes, None);
                }
                if dev_zero_bias {
                    // ResidualProj P branch: coefficient stream only, zero bias.
                    se.clear_buffer(&blo, 0, None);
                    se.clear_buffer(&buo, 0, None);
                    se.clear_buffer(&ble, 0, None);
                    se.clear_buffer(&bue, 0, None);
                } else {
                    se.copy_buffer_to_buffer(&sd.blo, 0, &blo, 0, bias_bytes);
                    se.copy_buffer_to_buffer(&sd.buo, 0, &buo, 0, bias_bytes);
                    se.copy_buffer_to_buffer(&sd.ble, 0, &ble, 0, bias_bytes);
                    se.copy_buffer_to_buffer(&sd.bue, 0, &bue, 0, bias_bytes);
                }
                self.queue.submit(Some(se.finish()));
            } else {
                // Seed the frontier coefficients (treated as EXACT → incoming error 0).
                self.queue
                    .write_buffer(&la[0], 0, bytemuck::cast_slice(lower_a));
                self.queue
                    .write_buffer(&ua[0], 0, bytemuck::cast_slice(upper_a));
                // Incoming coefficient error (0 for a fresh spec/frontier; nonzero when
                // composing a previous segment's output). The seed error fills the head
                // [0, seed_len); zero ONLY the unused tail [seed_len, a_elems) so the whole
                // buffer is freshly written this segment. (Previously the full a_elems was
                // zeroed and then the head re-written with the seed — the head zeroing was
                // pure redundant CPU→GPU transfer; final buffer contents are byte-identical.)
                self.queue
                    .write_buffer(&le[0], 0, bytemuck::cast_slice(lower_a_err));
                self.queue
                    .write_buffer(&ue[0], 0, bytemuck::cast_slice(upper_a_err));
                let zero_tail = |buf: &wgpu::Buffer, head: usize| {
                    if head < a_elems {
                        self.queue.write_buffer(
                            buf,
                            (head * size_of::<f32>()) as u64,
                            bytemuck::cast_slice(&zeros[..a_elems - head]),
                        );
                    }
                };
                zero_tail(&le[0], lower_a_err.len());
                zero_tail(&ue[0], upper_a_err.len());

                // Running bias: seeded from the frontier bias, error starts 0.
                self.queue
                    .write_buffer(&blo, 0, bytemuck::cast_slice(lower_b));
                self.queue
                    .write_buffer(&buo, 0, bytemuck::cast_slice(upper_b));
                self.queue
                    .write_buffer(&ble, 0, bytemuck::cast_slice(lower_b_err));
                self.queue
                    .write_buffer(&bue, 0, bytemuck::cast_slice(upper_b_err));
            }

            let abs_a = storage("res_abs_a", a_elems);
            let s_scratch = storage("res_s", a_elems);
            let prop_scratch = storage("res_prop", a_elems);
            // §0 weight-amplified DAZ floor (#gpu-metal-daz): `ones` (all-1 vector, the
            // GEMM operand that row-reduces `|A|` to the per-spec `‖a_i‖₁`) and the
            // `row_abs_a` result. Both reused per layer; `ones` filled once.
            let ones_buf = storage("res_ones", max_dim.max(1));
            let row_abs_a = storage("res_row_abs_a", num_specs.max(1));
            self.queue.write_buffer(
                &ones_buf,
                0,
                bytemuck::cast_slice(&vec![1.0f32; max_dim.max(1)]),
            );
            let bias_buf = storage("res_bias", max_dim);
            // #cert-err: the constant `[bias_abs_err; k]` operand and the
            // throwaway centre-bias sink for the extra bias-error dispatch.
            // Allocated ONLY when some layer declares a nonzero `bias_abs_err`,
            // so the default (exact-weight) walk allocates nothing new.
            let cert_bias_buf = cert_bias_charge.then(|| storage("res_cert_bias", max_dim));
            let cert_bias_sink =
                cert_bias_charge.then(|| storage("res_cert_bias_sink", num_specs.max(1)));
            // Activation slope/intercept buffers (reused per activation layer).
            // #batched-bab: `slope_dim = n_domains*max_dim` — one block per domain, so a
            // wide row reads its OWN domain's relaxation (single domain → max_dim, same).
            let ls_buf = storage("res_ls", slope_dim);
            let us_buf = storage("res_us", slope_dim);
            let lint_buf = storage("res_lint", slope_dim);
            let uint_buf = storage("res_uint", slope_dim);

            // #u4 (gate ON only — gate off allocates NOTHING here): the word
            // channel. u32 words are the same width as the f32 values, so each
            // buffer mirrors its value twin's element count exactly. Seeding:
            // coefficient AND composed coefficient-error words from the host
            // seed under the G13 rule (`taint_seed_word`); sentinel-magnitude
            // bias/bias-error seeds condemn their spec rows below. The `[1]`
            // slots and scratches need no init — every twin dispatch fully
            // overwrites its word output before anything reads it. `zw` stays
            // all-zero for the walk lifetime (wgpu zero-initializes; nothing
            // ever writes it).
            let mut taint: Option<TaintWalkState> = if taint_on {
                let u32_storage = |label: &str, n: usize| -> wgpu::Buffer {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (n.max(1) * size_of::<u32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                };
                let tb = TaintWalkState {
                    wla: [u32_storage("res_w_la0", a_elems), u32_storage("res_w_la1", a_elems)],
                    wua: [u32_storage("res_w_ua0", a_elems), u32_storage("res_w_ua1", a_elems)],
                    wle: [u32_storage("res_w_le0", a_elems), u32_storage("res_w_le1", a_elems)],
                    wue: [u32_storage("res_w_ue0", a_elems), u32_storage("res_w_ue1", a_elems)],
                    ws: u32_storage("res_w_s", a_elems),
                    wprop: u32_storage("res_w_prop", a_elems),
                    w_rowabs_lo: u32_storage("res_w_rowabs_lo", num_specs.max(1)),
                    w_rowabs_hi: u32_storage("res_w_rowabs_hi", num_specs.max(1)),
                    // Conv permutation/GEMM scratch words mirror the two value
                    // scratches and are overwritten for coefficient, S, and
                    // propagated-error subchains in encoder order.
                    w_conv_reshaped: u32_storage("res_w_conv_reshaped", a_elems),
                    w_conv_gemm: u32_storage("res_w_conv_gemm", max_gemm_out),
                    zw: u32_storage("res_w_zero", taint_zw_elems),
                    // The on-device row accumulator: zero-init (wgpu
                    // zero-initializes storage buffers), monotone (atomicOr is
                    // its only writer), read back ONCE at walk end.
                    rows_dev: u32_storage("res_w_rows", num_specs.max(1)),
                    rows: vec![0u32; num_specs],
                };
                let mut tb = tb;
                if let Some(sd) = &sweep_seed {
                    // The sweep carrier already owns the exact word state.
                    // Copy its active head and sticky row accumulator; newly
                    // allocated tails are WebGPU-zeroed and never contribute
                    // before a layer overwrites them.
                    let coefficient_bytes =
                        resident_f32_bytes(seed_elems, "worded sweep word seed")?;
                    let row_bytes = resident_f32_bytes(num_specs, "worded sweep row seed")?;
                    let mut encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("res_sweep_seed_words"),
                        },
                    );
                    for (source, destination) in [
                        (&sd.matrix.lower_center_word, &tb.wla[0]),
                        (&sd.matrix.upper_center_word, &tb.wua[0]),
                        (&sd.matrix.lower_radius_word, &tb.wle[0]),
                        (&sd.matrix.upper_radius_word, &tb.wue[0]),
                    ] {
                        encoder.copy_buffer_to_buffer(
                            source,
                            0,
                            destination,
                            0,
                            coefficient_bytes,
                        );
                    }
                    encoder.copy_buffer_to_buffer(
                        &sd.row.taint_rows,
                        0,
                        &tb.rows_dev,
                        0,
                        row_bytes,
                    );
                    self.submit_ticked(encoder.finish());
                } else {
                    // Host-seed route. One reusable scratch vec, head = seed
                    // words, tail = 0 (the seed head is `seed_elems`, the
                    // buffer `a_elems`; the tail mirrors the value buffers'
                    // zeroed tail).
                    let mut word_scratch = vec![0u32; a_elems];
                    for (dst, &v) in word_scratch.iter_mut().zip(lower_a.iter()) {
                        *dst = taint_seed_word(v);
                    }
                    self.queue
                        .write_buffer(&tb.wla[0], 0, bytemuck::cast_slice(&word_scratch));
                    word_scratch.fill(0);
                    for (dst, &v) in word_scratch.iter_mut().zip(upper_a.iter()) {
                        *dst = taint_seed_word(v);
                    }
                    self.queue
                        .write_buffer(&tb.wua[0], 0, bytemuck::cast_slice(&word_scratch));
                    // Error words: G13 over the COMPOSED seed errors too — a
                    // `1e30`/`1e10` marker shipped in an error seed must enter
                    // worded, exactly like a coefficient marker.
                    word_scratch.fill(0);
                    for (dst, &v) in word_scratch.iter_mut().zip(lower_a_err.iter()) {
                        *dst = taint_seed_word(v);
                    }
                    self.queue
                        .write_buffer(&tb.wle[0], 0, bytemuck::cast_slice(&word_scratch));
                    word_scratch.fill(0);
                    for (dst, &v) in word_scratch.iter_mut().zip(upper_a_err.iter()) {
                        *dst = taint_seed_word(v);
                    }
                    self.queue
                        .write_buffer(&tb.wue[0], 0, bytemuck::cast_slice(&word_scratch));
                    // Seed biases fold straight into the per-spec-row host
                    // companion because they are host data on this route.
                    for seed in [lower_b, upper_b, lower_b_err, upper_b_err] {
                        for (row, &v) in seed.iter().enumerate() {
                            if row < tb.rows.len() && taint_seed_word(v) != 0 {
                                tb.rows[row] |= 1;
                            }
                        }
                    }
                }
                Some(tb)
            } else {
                None
            };

            let uniform = |label: &str, bytes: usize| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let gp_buf = uniform("res_gp", size_of::<GemmParams>());
            // Separate GemmParams for the n=1 row-L1 reduction `|A|@ones → row_abs_a`
            // (a distinct uniform so it coexists with the n=out_cols main GEMM in one
            // encoder). #gpu-metal-daz.
            let gp1_buf = uniform("res_gp1", size_of::<GemmParams>());
            let cp_buf = uniform("res_cp", size_of::<CombineParams>());
            let ap_buf = uniform("res_ap", size_of::<AbsParams>());
            // #eft-err: the dark compensated-error channel. Both gate halves are
            // resolved ONCE per fold: the env request AND the per-adapter
            // primitive self-check (fail-closed — a refusing adapter keeps the
            // Higham charge byte-identically). Scratches are allocated only
            // when armed, so gate-off adds zero buffers and zero dispatches.
            // NOTE: the CACHED read (never initializes) — this fold runs inside
            // the GPU-checked section; first-initializing the probe here would
            // self-deadlock. The cache is populated eagerly at device creation
            // when NY_EFT_ERR=1 (device.rs); uninitialized ⇒ Higham unchanged.
            let (eft_v_buf, eft_r_buf, eft_cp_buf) = if eft_on {
                (
                    Some(storage("res_eft_v", a_elems)),
                    Some(storage("res_eft_r", a_elems)),
                    Some(uniform("res_eft_cp", size_of::<EftCombineParams>())),
                )
            } else {
                (None, None, None)
            };
            // Conv-arm EFT scratches (GEMM-shaped, pre-col2im): the twin GEMM's
            // value/residual streams that the col2im twin then gathers.
            let (eft_vg_buf, eft_rg_buf) = if eft_on {
                (
                    Some(storage("res_eft_vg", max_gemm_out)),
                    Some(storage("res_eft_rg", max_gemm_out)),
                )
            } else {
                (None, None)
            };
            // #fold-coalesce (dark, NY_FOLD_COALESCE=1, default OFF ⇒ per-layer
            // submits, byte-identical): collect every layer's command buffer and
            // submit the whole chain in ONE queue.submit. Per-layer values stay
            // correct because their uploads go through the staging arena as
            // encoder-ordered copies (see FoldStagingArena). Sizing: exact
            // per-layer upload bytes + generous per-layer uniform slack.
            let mut fold_cmds: Vec<wgpu::CommandBuffer> = Vec::new();
            let mut arena = if coalesce {
                Some(FoldStagingArena::new(&self.device, fold_staging_cap))
            } else {
                None
            };
            let bp_buf = uniform("res_bp", size_of::<BiasParams>());
            // #cert-err: a SECOND `BiasParams` uniform so the bias-error charge
            // (gamma_k = 1, widened slack) coexists with the ordinary bias fold
            // inside one encoder. `None` unless a layer declares a bias error.
            let cert_bp_buf = cert_bias_charge.then(|| uniform("res_cert_bp", size_of::<BiasParams>()));
            // Separate lower/upper uniforms: within one submit, queue.write_buffer
            // is ordered BEFORE all encoder passes, so reusing one buffer for both
            // sides would make every pass see only the last-written (upper) value.
            let actp_lo = uniform("res_actp_lo", size_of::<ActParams>());
            let actp_hi = uniform("res_actp_hi", size_of::<ActParams>());
            let actbp_lo = uniform("res_actbp_lo", size_of::<ActBiasParams>());
            let actbp_hi = uniform("res_actbp_hi", size_of::<ActBiasParams>());
            // Conv scratch + uniforms (S·OH·OW·OC reshaped ≤ a_elems; GEMM out sized).
            let conv_reshaped = storage("res_conv_reshaped", a_elems);
            let conv_gemm = storage("res_conv_gemm", max_gemm_out);
            let crp_buf = uniform("res_crp", size_of::<ConvReshapeParams>());
            let ccp_buf = uniform("res_ccp", size_of::<ConvCol2imParams>());
            let cep_buf = uniform("res_cep", size_of::<ConvErrParams>());

            // --- the resident layer loop (per-layer submit; A/err/bias buffers
            // persist on-device across submits, so there is NO per-layer download) ---
            // Gradient capture state (#unsat-keystone). When `relu_pre_lower` is
            // non-empty, at each Activation layer we dispatch the alpha-gradient
            // kernel on the PRE-transform lower coefficient (la[ping]); this is
            // purely additive (writes only its own grad buffers, never the bound
            // buffers) so the verdict path with empty `relu_pre_lower` is unchanged.
            let grad_pipe = (!relu_pre_lower.is_empty()).then_some(&res_pipes.alpha_grad);
            // #w4 wide α+β ascent: `slope_dim`-wide so the wide lane can stage each
            // domain's stacked pre-activation block (dom*nn + i); single domain
            // (`slope_dim == max_dim`) is byte-identical.
            let grad_pl_buf = storage("res_grad_pl", slope_dim);
            let grad_params = uniform("res_grad_params", size_of::<GradAlphaParams>());
            let mut grad_bufs: Vec<(wgpu::Buffer, usize)> = Vec::new();
            let mut act_capture_idx = 0usize;
            let __t_setup = __t_start.elapsed();
            let __t_loop_start = std::time::Instant::now();

            // Beta-capable per-domain state (#unsat-keystone step 4). `beta_buf` holds the
            // current Activation's per-neuron signed beta (β·sign); all-zero ⇒ inert (the
            // CROWN_ACTIVATION_RESIDENT_SHADER adds it post-slope). Zero-initialized once;
            // only rewritten per-layer when `beta_signed` is provided, so the no-beta
            // verdict path keeps it all-zero and is byte-for-byte unchanged.
            // #batched-bab: `slope_dim`-wide so the shader reads β at `dom*num_neurons`.
            let beta_buf = storage("res_beta", slope_dim);
            self.queue
                .write_buffer(&beta_buf, 0, bytemuck::cast_slice(&vec![0.0f32; slope_dim]));
            let mut act_beta_idx = 0usize;

            // Beta-GRADIENT gather state (#w4-split-tightening). When `beta_gather_idx`
            // is non-empty, at each Activation layer the requested PRE-transform lower
            // coefficient entries (la[ping], the CPU `a_at_relu` capture point) are
            // staged via per-element buffer copies into a MAP_READ buffer — a pure
            // read of the coefficient stream, so the bound computation is
            // byte-for-byte unchanged. `None` entries keep fold-order alignment for
            // ReLUs with an empty index list.
            let mut gather_bufs: Vec<Option<(wgpu::Buffer, usize)>> = Vec::new();
            let mut act_gather_idx = 0usize;

            // Conv coefficient-error mode (#w4-conv-err-per-entry): PER-ENTRY by
            // default — the certified conv-transpose error is computed with the SAME
            // reshape→GEMM→col2im pipeline on (|A|,|W|) and (err,|W|) and combined via
            // the audited AW-error combine (`slack·(γ_k·S + prop) + additive`, per
            // entry), exactly mirroring the Linear layers. The legacy row-max
            // broadcast (`γ·rowmax|A|·‖W‖₁ + rowmax|err|·‖W‖₁` written to EVERY
            // output entry) over-counts by (a) the full-kernel L1 vs the ~OC·KH·KW
            // receptive column and (b) a dim× factor at every discharge — the
            // measured ~25× root-bound gap vs the certified forward pass on deep
            // conv resnets (#w4). Opt out with NY_CONV_ERR_ROWMAX=1 for A/B.
            let mut ping = 0usize;
            // `li` is the layer's index in `layers`, used only to name the layer
            // in `#cert-err` refusal messages.
            for (li, layer) in layers.iter().enumerate() {
                // Cooperative cancellation (#w4-refresh-deadline): a deep resnet
                // walk is a long sequence of per-layer submits; between layers is
                // a safe stop point. Callers treat DeadlineExceeded as a sound
                // CPU/reference fallback. Unset deadline ⇒ no-op (pre-existing
                // behavior).
                if self.crown_backward_deadline_expired() {
                    return Err(NyError::DeadlineExceeded(
                        "GPU sound resident CROWN backward deadline exceeded between layers".into(),
                    ));
                }
                // ---- Activation layer (elementwise; dim unchanged) ----
                if let GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    lower_intercept,
                    upper_intercept,
                    num_neurons,
                } = layer
                {
                    let nn = *num_neurons;
                    let g = gamma_k_f32(nn)?;
                    let slack = combine_slack_f32(nn)?;
                    let eft_slack = if eft_on { eft_r_slack_f32(nn)? } else { 0.0 };
                    // FTZ-SAFE additive underflow floors (#gpu-metal): Metal MSL
                    // flushes subnormals to zero, so the old `8·ETA` / `8n·ETA`
                    // (ETA = 2^-149, subnormal) floors would vanish on Apple GPUs →
                    // error under-added. `ftz_safe_underflow_floor` returns a
                    // NORMAL-range floor (≥ FLT_MIN 2^-126) that survives FTZ and
                    // dominates the old subnormal floor (strictly widening, so still
                    // sound on Vulkan and a strict improvement on Metal).
                    // `add_e` (elementwise, coeff ≤ 1) is fully FTZ-sound. `add_b`
                    // (the abs-sum REDUCTION) is the weight-INDEPENDENT BASE of the
                    // flush floor; the intercept-bias shader now ALSO adds the on-device
                    // WEIGHT-AMPLIFIED term `flushacc·slack·F32_MIN_NORMAL` (fed via
                    // `ActBiasParams::slack`), since a subnormal coeff flushed then
                    // scaled by a large intercept loses up to |sel|·FLT_MIN. This
                    // completes the Metal FTZ fix for the reduction path.
                    // See docs/SOUND_GPU_IBP_PLAN.md §0.
                    let add_e = super::super::sound_consts::rung3_flush_safe_additive(1)?; // elementwise: complete
                    // Reduction: the admitted fma-residual flush happens before
                    // the EFT recovery multiply, so scale its base floor by the
                    // same outward r_slack. Legacy mode keeps the unscaled base.
                    let nn_u32 = u32::try_from(nn).unwrap_or(u32::MAX);
                    let add_b = if eft_on {
                        super::super::sound_consts::rung3_flush_safe_additive_scaled(
                            nn_u32, eft_slack,
                        )?
                    } else {
                        super::super::sound_consts::rung3_flush_safe_additive(nn_u32)?
                    };
                    // #fold-coalesce: the encoder exists BEFORE the uploads so
                    // they can be arena-copies ordered ahead of this layer's
                    // passes (legacy mode keeps write_buffer semantics).
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("res_act"),
                            });
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &ls_buf,
                        bytemuck::cast_slice(lower_slope),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &us_buf,
                        bytemuck::cast_slice(upper_slope),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &lint_buf,
                        bytemuck::cast_slice(lower_intercept),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &uint_buf,
                        bytemuck::cast_slice(upper_intercept),
                    )?;
                    // Beta fold (#unsat-keystone step 4): write this Activation's per-neuron
                    // signed beta into beta_buf. #batched-bab: `beta_signed[act]` is the
                    // per-domain-STACKED `[n_domains*nn]` slice (single domain → nn), laid
                    // out as `n_domains` contiguous blocks of `nn` so the shader reads
                    // `beta[dom*nn + i]`. Only when beta_signed is provided — else beta_buf
                    // stays the zero-init (inert, byte-identical).
                    if !beta_signed.is_empty() {
                        let mut beta_layer = vec![0.0f32; n_domains * nn];
                        if let Some(bs) = beta_signed.get(act_beta_idx) {
                            for (d, &s) in beta_layer.iter_mut().zip(bs.iter()) {
                                *d = s;
                            }
                        }
                        self.fold_upload(
                            arena.as_mut(),
                            &mut encoder,
                            &beta_buf,
                            bytemuck::cast_slice(&beta_layer),
                        )?;
                        act_beta_idx += 1;
                    }
                    let elem_wg = ((num_specs * nn) as u32).div_ceil(256);

                    // #flush-charge §E: under charged authority the intercept-bias
                    // reduction's slack is widened by the audited act-bias factor
                    // (double-DAZ demand, oracle
                    // `charged_act_bias_factor_covers_the_double_daz_demand`);
                    // identity (byte-identical) on every uncharged device.
                    let act_bias_slack = charged_act_bias_slack_or(charged_policy.as_ref(), slack)?;
                    // Write the four lower/upper uniforms ONCE each (distinct buffers).
                    let mk_actbp = |is_up: u32| ActBiasParams {
                        num_specs: num_specs_u32,
                        num_neurons: nn as u32,
                        is_upper: is_up,
                        // #eft-err: in EFT mode the γ field carries r_slack (the
                        // shader's γ term is unused there).
                        gamma_k: if eft_on { eft_slack } else { g },
                        additive: add_b,
                        slack: act_bias_slack,
                        num_specs_per_dom: num_specs_per_dom_u32,
                        eft_mode: u32::from(eft_on),
                    };
                    let mk_actp = |is_up: u32| ActParams {
                        num_specs: num_specs_u32,
                        num_neurons: nn as u32,
                        is_upper: is_up,
                        additive: add_e,
                        // #batched-bab: dom = row/num_specs_per_dom; single domain → 0.
                        num_specs_per_dom: num_specs_per_dom_u32,
                        eft_mode: u32::from(eft_on),
                        _p: [0; 2],
                    };
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &actbp_lo,
                        bytemuck::bytes_of(&mk_actbp(0)),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &actbp_hi,
                        bytemuck::bytes_of(&mk_actbp(1)),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &actp_lo,
                        bytemuck::bytes_of(&mk_actp(0)),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &actp_hi,
                        bytemuck::bytes_of(&mk_actp(1)),
                    )?;

                    // intercept -> running bias (reads PRE-transform coefficient).
                    self.pass_simple(
                        &mut encoder,
                        act_bias_pipe,
                        &actbp_lo,
                        &[&la[ping], &le[ping], &lint_buf, &uint_buf, &blo, &ble],
                        num_specs_u32,
                    );
                    self.pass_simple(
                        &mut encoder,
                        act_bias_pipe,
                        &actbp_hi,
                        &[&ua[ping], &ue[ping], &lint_buf, &uint_buf, &buo, &bue],
                        num_specs_u32,
                    );
                    // coefficient + error (elementwise, lower then upper); beta_buf (binding 7)
                    // folds the β-CROWN dual post-slope (shader: lower −=, upper += beta_signed).
                    // #u4 gate ON: the SAME dispatch through the activation
                    // taint twin (values bit-identical, drift-pinned) with the
                    // word pair rotated alongside the value pair.
                    if let Some(tb) = taint.as_ref() {
                        self.pass_simple(
                            &mut encoder,
                            res_pipes.act_taint.as_ref().expect("#u4: availability checked at walk entry"),
                            &actp_lo,
                            &[
                                &la[ping],
                                &le[ping],
                                &ls_buf,
                                &us_buf,
                                &la[1 - ping],
                                &le[1 - ping],
                                &beta_buf,
                                &tb.wla[ping],
                                &tb.wle[ping],
                                &tb.wla[1 - ping],
                                &tb.wle[1 - ping],
                            ],
                            elem_wg,
                        );
                        self.pass_simple(
                            &mut encoder,
                            res_pipes.act_taint.as_ref().expect("#u4: availability checked at walk entry"),
                            &actp_hi,
                            &[
                                &ua[ping],
                                &ue[ping],
                                &ls_buf,
                                &us_buf,
                                &ua[1 - ping],
                                &ue[1 - ping],
                                &beta_buf,
                                &tb.wua[ping],
                                &tb.wue[ping],
                                &tb.wua[1 - ping],
                                &tb.wue[1 - ping],
                            ],
                            elem_wg,
                        );
                    } else {
                        self.pass_simple(
                            &mut encoder,
                            act_pipe,
                            &actp_lo,
                            &[
                                &la[ping],
                                &le[ping],
                                &ls_buf,
                                &us_buf,
                                &la[1 - ping],
                                &le[1 - ping],
                                &beta_buf,
                            ],
                            elem_wg,
                        );
                        self.pass_simple(
                            &mut encoder,
                            act_pipe,
                            &actp_hi,
                            &[
                                &ua[ping],
                                &ue[ping],
                                &ls_buf,
                                &us_buf,
                                &ua[1 - ping],
                                &ue[1 - ping],
                                &beta_buf,
                            ],
                            elem_wg,
                        );
                    }
                    // Per-ReLU alpha gradient from the PRE-transform lower coefficient
                    // la[ping] (read-only here; the transform writes la[1-ping]).
                    if let Some(gp) = grad_pipe {
                        if act_capture_idx < relu_pre_lower.len() {
                            // #w4 wide α+β ascent: the wide lane stages each domain's
                            // pre-activation block stacked (`n_domains*nn`, dom*nn + i)
                            // and the shader reduces each domain's own `num_specs_per_dom`
                            // row block into its own grad block — never blended across
                            // domains. Single domain (`n_domains == 1`, nsp == 0 or ==
                            // num_specs) is byte-identical to the pre-widening capture.
                            let pl = relu_pre_lower[act_capture_idx];
                            let grad_dim = (n_domains * nn).min(pl.len());
                            self.fold_upload(
                                arena.as_mut(),
                                &mut encoder,
                                &grad_pl_buf,
                                bytemuck::cast_slice(&pl[..grad_dim]),
                            )?;
                            self.fold_upload(
                                arena.as_mut(),
                                &mut encoder,
                                &grad_params,
                                bytemuck::bytes_of(&GradAlphaParams {
                                    num_specs: num_specs_u32,
                                    num_neurons: nn as u32,
                                    num_specs_per_dom: num_specs_per_dom_u32,
                                    _p1: 0,
                                }),
                            )?;
                            let gbuf = storage("res_grad_out", grad_dim);
                            self.pass_simple(
                                &mut encoder,
                                gp,
                                &grad_params,
                                &[&la[ping], &grad_pl_buf, &gbuf],
                                (grad_dim as u32).div_ceil(256),
                            );
                            grad_bufs.push((gbuf, grad_dim));
                            act_capture_idx += 1;
                        }
                    }
                    // Beta-gradient / Complete Clip A-value gather
                    // (#w4-split-tightening): stage the requested la[ping] entries
                    // (PRE-transform lower coefficient — this layer's passes only
                    // WRITE la[1-ping], so la[ping] is stable within this encoder).
                    //
                    // A small β gather retains the historical per-element byte-copy
                    // path exactly. Dense Complete Clip requests can contain hundreds
                    // of columns across thousands of wide rows; encoding one command
                    // per value is catastrophic, so those use one strided compute
                    // dispatch plus one contiguous readback copy per ReLU.
                    if act_gather_idx < beta_gather_idx.len() {
                        let idxs = beta_gather_idx[act_gather_idx];
                        if idxs.is_empty() {
                            gather_bufs.push(None);
                        } else {
                            let n_idx = idxs.len();
                            let gather_elems = num_specs.checked_mul(n_idx).ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "resident gather element-count overflow".into(),
                                )
                            })?;
                            let gather_bytes = gather_elems
                                .checked_mul(size_of::<f32>())
                                .and_then(|n| u64::try_from(n).ok())
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "resident gather byte-count overflow".into(),
                                    )
                                })?;
                            if gather_bytes > self.device.limits().max_buffer_size {
                                return Err(NyError::UnsupportedOp(format!(
                                    "resident gather buffer needs {gather_bytes} bytes, device \
                                     max_buffer_size is {}",
                                    self.device.limits().max_buffer_size
                                )));
                            }
                            let gbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("res_beta_gather"),
                                size: gather_bytes.max(size_of::<f32>() as u64),
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            if gather_elems <= LEGACY_BETA_GATHER_MAX_COPIES {
                                for s in 0..num_specs {
                                    for (i, &idx) in idxs.iter().enumerate() {
                                        let idx = idx as usize;
                                        if idx >= nn {
                                            continue; // out-of-range stays 0 (zero-init buffer)
                                        }
                                        encoder.copy_buffer_to_buffer(
                                            &la[ping],
                                            ((s * nn + idx) * size_of::<f32>()) as u64,
                                            &gbuf,
                                            ((s * n_idx + i) * size_of::<f32>()) as u64,
                                            size_of::<f32>() as u64,
                                        );
                                    }
                                }
                            } else {
                                let num_specs_u32 = u32::try_from(num_specs).map_err(|_| {
                                    NyError::InvalidSpec(
                                        "resident gather num_specs exceeds u32".into(),
                                    )
                                })?;
                                let nn_u32 = u32::try_from(nn).map_err(|_| {
                                    NyError::InvalidSpec(
                                        "resident gather num_neurons exceeds u32".into(),
                                    )
                                })?;
                                let n_idx_u32 = u32::try_from(n_idx).map_err(|_| {
                                    NyError::InvalidSpec(
                                        "resident gather index count exceeds u32".into(),
                                    )
                                })?;
                                let _gather_elems_u32 =
                                    u32::try_from(gather_elems).map_err(|_| {
                                        NyError::InvalidSpec(
                                            "resident gather element count exceeds u32".into(),
                                        )
                                    })?;
                                let idx_bytes = n_idx
                                    .checked_mul(size_of::<u32>())
                                    .and_then(|n| u64::try_from(n).ok())
                                    .ok_or_else(|| {
                                        NyError::InvalidSpec(
                                            "resident gather index byte-count overflow".into(),
                                        )
                                    })?;
                                let max_storage =
                                    self.device.limits().max_storage_buffer_binding_size;
                                if gather_bytes > max_storage || idx_bytes > max_storage {
                                    return Err(NyError::UnsupportedOp(format!(
                                        "resident gather storage binding exceeds device limit \
                                         {max_storage} bytes (output={gather_bytes}, \
                                         indices={idx_bytes})"
                                    )));
                                }
                                let dispatch = gather_elems.checked_add(255).ok_or_else(|| {
                                    NyError::InvalidSpec("resident gather dispatch overflow".into())
                                })? / 256;
                                if dispatch > max_wg {
                                    return Err(NyError::UnsupportedOp(format!(
                                        "resident gather dispatch {dispatch} exceeds \
                                         max_compute_workgroups_per_dimension {max_wg}"
                                    )));
                                }
                                let dispatch_u32 = u32::try_from(dispatch).map_err(|_| {
                                    NyError::InvalidSpec(
                                        "resident gather dispatch exceeds u32".into(),
                                    )
                                })?;
                                let idx_buf = storage("res_beta_gather_idx", n_idx);
                                self.queue
                                    .write_buffer(&idx_buf, 0, bytemuck::cast_slice(idxs));
                                let params = uniform(
                                    "res_beta_gather_params",
                                    size_of::<StridedGatherParams>(),
                                );
                                self.queue.write_buffer(
                                    &params,
                                    0,
                                    bytemuck::bytes_of(&StridedGatherParams {
                                        num_specs: num_specs_u32,
                                        num_neurons: nn_u32,
                                        num_indices: n_idx_u32,
                                        _p1: 0,
                                    }),
                                );
                                let dense = storage("res_beta_gather_dense", gather_elems);
                                self.pass_simple(
                                    &mut encoder,
                                    self.resident_strided_gather_pipeline(),
                                    &params,
                                    &[&la[ping], &idx_buf, &dense],
                                    dispatch_u32,
                                );
                                encoder.copy_buffer_to_buffer(&dense, 0, &gbuf, 0, gather_bytes);
                                if __probe {
                                    eprintln!(
                                        "[wide-gather] mode=strided rows={num_specs} \
                                         columns={n_idx} values={gather_elems} commands=2"
                                    );
                                }
                            }
                            gather_bufs.push(Some((gbuf, gather_elems)));
                        }
                        act_gather_idx += 1;
                    }
                    // #u4 fail-closed transport, ON-DEVICE — the intercept→bias
                    // fold (CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER) has NO
                    // taint twin: its per-spec bias contribution consumes the
                    // PRE-transform coefficient AND error, so their words must
                    // reach the row accumulator (audit §4 C1, intercept fold).
                    // Encoded HERE, no host readback: the `[ping]` word buffers
                    // are stable within this encoder (this layer's twins write
                    // only `[1 - ping]`), and `lint_buf`/`uint_buf` were
                    // uploaded above for the value fold. Single-domain walks
                    // express the committed `li != 0 || ui != 0` annihilation
                    // conjunct as TWO per-column-partner row-OR dispatches per
                    // word buffer (a word survives iff EITHER intercept
                    // dispatch keeps it ≡ the disjunction — the CPU reference
                    // is `intercept_fold_taint`, crown_backward_sound_host.rs,
                    // now test-reference only); batched-domain walks keep the
                    // unconditional row-OR fallback (strictly more
                    // conservative — refusal-only risk; per-domain conjunct is
                    // a TODO).
                    // (The α-grad / β-gather captures also read la[ping] but
                    // are non-soundness-critical steering data — no words.)
                    if let Some(tb) = taint.as_ref() {
                        for wb in [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]] {
                            if n_domains == 1 {
                                self.taint_row_or_dispatch(
                                    &mut encoder,
                                    wb,
                                    Some((&lint_buf, false)),
                                    num_specs,
                                    nn,
                                    &tb.rows_dev,
                                );
                                self.taint_row_or_dispatch(
                                    &mut encoder,
                                    wb,
                                    Some((&uint_buf, false)),
                                    num_specs,
                                    nn,
                                    &tb.rows_dev,
                                );
                            } else {
                                self.taint_row_or_dispatch(
                                    &mut encoder,
                                    wb,
                                    None,
                                    num_specs,
                                    nn,
                                    &tb.rows_dev,
                                );
                            }
                        }
                    }
                    if coalesce {
                        fold_cmds.push(encoder.finish());
                    } else {
                        self.submit_ticked(encoder.finish());
                    }
                    ping = 1 - ping;
                    self.apply_intermediate_sweep_boundary(
                        li + 1,
                        nn,
                        num_specs,
                        &la[ping],
                        &ua[ping],
                        &le[ping],
                        &ue[ping],
                        &blo,
                        &buo,
                        &ble,
                        &bue,
                        taint.as_mut(),
                    )?;
                    continue;
                }

                // ---- Conv2d layer (reshape → GEMM → col2im + over-bound error) ----
                if let GpuCrownLayer::Conv2d {
                    weight_col,
                    bias_expanded,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    cert_err,
                } = layer
                {
                    let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
                    let (oh, ow, ih, iw) = (*out_h, *out_w, *in_h, *in_w);
                    let in_d = oc
                        .checked_mul(oh)
                        .and_then(|value| value.checked_mul(ow))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "resident CROWN conv input dimension overflow".into(),
                            )
                        })?; // coeff entering
                    let out_d = ic
                        .checked_mul(ih)
                        .and_then(|value| value.checked_mul(iw))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "resident CROWN conv output dimension overflow".into(),
                            )
                        })?; // coeff exiting
                    let spatial = oh.checked_mul(ow).ok_or_else(|| {
                        NyError::InvalidSpec("resident CROWN conv spatial overflow".into())
                    })?;
                    let kernel_cols = ic
                        .checked_mul(kh)
                        .and_then(|value| value.checked_mul(kw))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "resident CROWN conv kernel columns overflow".into(),
                            )
                        })?;
                    let m = num_specs.checked_mul(spatial).ok_or_else(|| {
                        NyError::InvalidSpec("resident CROWN conv GEMM rows overflow".into())
                    })?;
                    let (k, n) = (oc, kernel_cols);
                    let conv_reduction = oc
                        .checked_mul(kh)
                        .and_then(|value| value.checked_mul(kw))
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "resident CROWN conv reduction length overflow".into(),
                            )
                        })?;
                    // #cert-err: identical substitution to the Linear arm — the
                    // conv per-entry error runs the SAME `slack·(gamma·S + P)`
                    // combine, so charging `g = gamma + w_rel + gamma·w_rel` in
                    // `gamma_k` and the matching `(1 + w_rel)` in `slack` dominates
                    // `((gamma+w_rel)·|A| + (1+w_rel)·err) ⊛ |W|`. `conv_err_rowmax` (the legacy
                    // broadcast) is refused with a nonzero `cert_err` in the
                    // walk preflight, so `conv_slack == 0.0` never carries a
                    // charge.
                    let g_conv_exact = gamma_k_f32(conv_reduction)?;
                    let g_conv = cert_err.charged_gamma(g_conv_exact);
                    let conv_slack = if conv_err_rowmax {
                        0.0
                    } else {
                        cert_charged_slack(combine_slack_f32(conv_reduction)?, *cert_err, li)?
                    };
                    let conv_eft_slack = if !conv_err_rowmax && eft_on {
                        eft_r_slack_f32(conv_reduction)?
                    } else {
                        0.0
                    };
                    let (rp, cp, ep) = conv_pipes.expect("conv pipes present");
                    // #fold-coalesce: encoder BEFORE the uploads (arena copies
                    // must be encoder-ordered ahead of this layer's passes).
                    let mut enc =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("res_conv"),
                            });

                    // #lever1 weight residency: the constant conv weight (and its
                    // derived |W|) are GPU-resident — uploaded once per model,
                    // Arc-identity keyed with a keep-alive guard (see
                    // ops/resident_weights.rs) — instead of re-uploaded (and |W|
                    // re-computed on CPU) per domain per call. Identical bytes on
                    // the identical read-only bindings.
                    let w_buf = self.resident_weight_buf(weight_col, WeightForm::Raw)?;
                    let abs_w_buf = if conv_err_rowmax {
                        None
                    } else {
                        Some(self.resident_weight_buf(weight_col, WeightForm::Abs)?)
                    };
                    if conv_err_rowmax {
                        // SOUNDNESS: the row-max conv error multiplier is the weight L1
                        // norm. An f32-summed L1 ROUNDS DOWN (round-to-nearest reduction)
                        // and would under-report ‖W‖₁ → the certified conv-coeff error
                        // under-counts → false proof. Accumulate in f64 (f32→f64 widen +
                        // |·| are exact, only the f64 sum rounds) and round the f32 cast
                        // OUTWARD (up). Mirrors the proven conv fix (becc501).
                        let kl1_f64: f64 = weight_col
                            .iter()
                            .map(|&value| f32_to_f64_exact(value).abs())
                            .sum();
                        let kl1: f32 = up_f32(kl1_f64);
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &cep_buf,
                            bytemuck::bytes_of(&ConvErrParams {
                                num_specs: num_specs_u32,
                                out_dim: in_d as u32,
                                new_dim: out_d as u32,
                                _p0: 0,
                                gamma: g_conv,
                                kernel_l1: kl1,
                                _p1: 0,
                                _p2: 0,
                            }),
                        )?;
                    } else {
                        // Per-entry certified error (#w4-conv-err-per-entry): the error
                        // is propagated through the SAME conv-transpose structure as the
                        // coefficient — S = |A|⊛|W| and prop = err⊛|W| per entry — then
                        // combined as `slack·(γ_{OC·KH·KW}·S + prop) + additive`.
                        // SOUND: per output entry the conv-transpose accumulates ≤
                        // OC·KH·KW products (GEMM contraction over OC, col2im gather of
                        // ≤ KH·KW partials), so Higham's order-independent bound gives
                        // |fl(A⊛W) − A⊛W| ≤ γ_{OC·KH·KW}·(|A|⊛|W|)_exact; the incoming
                        // per-entry error is amplified by at most (err⊛|W|)_exact. Both
                        // RHS terms are themselves f32-computed (UNDER-reporting by ≤ a
                        // γ_k factor), so the combine's `slack ≥ 1/(1−γ_k)` recovers an
                        // outward bound and `additive` floors the FTZ underflow — the
                        // exact scheme already audited for the Linear layers.
                        // (|W| itself is the resident `abs_w_buf` above.)
                        // §0 DAZ floor for conv: `‖w_j‖₁` over the oc·kh·kw receptive
                        // taps ≤ the TOTAL weight L1 `‖W_col‖₁,₁` — a scalar OUTWARD
                        // over-bound (#gpu-metal-daz). `n=1` row-L1 reduction of the
                        // incoming coeff `|A|[num_specs × in_d] @ ones` gives `‖a_i‖₁`.
                        // Compute from `weight_col` with a bit-exact lift so DAZ
                        // cannot erase a subnormal weight.
                        // #daz-flush-cover-v2 (dark, default OFF ⇒ byte-identical):
                        // the shipped `flushacc` carries `‖w_j‖₁` once but a DAZ
                        // adapter has TWO `μ‖w‖₁` operand-flush channels per output
                        // (the coefficient GEMM and the propagated-error GEMM), and a
                        // third `μ‖err_i‖₁` channel with no term at all. See
                        // `sound_consts::daz_flush_cover_w_l1`.
                        crate::wgpu_device::sound_consts::refuse_subnormal_weight_under_daz_cover(
                            weight_col,
                            "conv-transpose weight_col",
                            daz_cover_armed,
                        )?;
                        let w_l1_max_conv: f32 =
                            crate::wgpu_device::sound_consts::daz_flush_cover_w_l1(
                                up_f32(
                                    weight_col
                                        .iter()
                                        .map(|&value| f32_to_f64_exact(value).abs())
                                        .sum(),
                                ),
                                daz_cover_armed,
                            )?;
                        let conv_reduction_u32 =
                            u32::try_from(conv_reduction).unwrap_or(u32::MAX);
                        let conv_additive = if eft_on {
                            super::super::sound_consts::rung3_flush_safe_additive_scaled(
                                conv_reduction_u32,
                                conv_eft_slack,
                            )?
                        } else {
                            super::super::sound_consts::rung3_flush_safe_additive(
                                conv_reduction_u32,
                            )?
                        };
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &gp1_buf,
                            bytemuck::bytes_of(&GemmParams {
                                m: num_specs_u32,
                                k: in_d as u32,
                                n: 1,
                                _padding: 0,
                            }),
                        )?;
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &cp_buf,
                            bytemuck::bytes_of(&CombineParams {
                                n: (num_specs * out_d) as u32,
                                slack: conv_slack,
                                gamma_k: g_conv,
                                additive: conv_additive,
                                k: conv_reduction as u32,
                                out_cols: out_d as u32,
                                w_l1_max: w_l1_max_conv,
                                _pad: 0,
                            }),
                        )?;
                        // #eft-err conv params: same flush/prop fields as the conv
                        // combine; r_slack covers the twin's residual accumulation
                        // over the FULL oc·kh·kw contraction (GEMM + col2im adds).
                        if let Some(eft_cp) = eft_cp_buf.as_ref() {
                            self.fold_upload(
                                arena.as_mut(),
                                &mut enc,
                                eft_cp,
                                bytemuck::bytes_of(&EftCombineParams {
                                    n: (num_specs * out_d) as u32,
                                    r_slack: conv_eft_slack,
                                    slack: conv_slack,
                                    additive: conv_additive,
                                    k: conv_reduction as u32,
                                    out_cols: out_d as u32,
                                    w_l1_max: w_l1_max_conv,
                                    _pad: 0,
                                }),
                            )?;
                        }
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &ap_buf,
                            bytemuck::bytes_of(&AbsParams {
                                n: (num_specs * in_d) as u32,
                                _p: [0; 3],
                            }),
                        )?;
                    }
                    if let Some(b) = bias_expanded {
                        let bias_gamma = gamma_k_f32(in_d)?;
                        // #flush-charge: identity unless charged authority is
                        // armed (then the oracle-derived bias factor widens it).
                        let bias_slack =
                            charged_bias_slack_or(charged_policy.as_ref(), combine_slack_f32(in_d)?)?;
                        let bias_eft_slack = if eft_on { eft_r_slack_f32(in_d)? } else { 0.0 };
                        let bias_reduction_u32 = u32::try_from(in_d).unwrap_or(u32::MAX);
                        let bias_additive = if eft_on {
                            super::super::sound_consts::rung3_flush_safe_additive_scaled(
                                bias_reduction_u32,
                                bias_eft_slack,
                            )?
                        } else {
                            super::super::sound_consts::rung3_flush_safe_additive(
                                bias_reduction_u32,
                            )?
                        };
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &bias_buf,
                            bytemuck::cast_slice(b),
                        )?;
                        self.fold_upload(
                            arena.as_mut(),
                            &mut enc,
                            &bp_buf,
                            bytemuck::bytes_of(&BiasParams {
                                num_specs: num_specs_u32,
                                k: in_d as u32,
                                gamma_k: bias_gamma,
                                additive: bias_additive,
                                slack: bias_slack,
                                eft_mode: u32::from(eft_on),
                                eft_r_slack: bias_eft_slack,
                                _p: 0,
                            }),
                        )?;
                    }
                    self.fold_upload(
                        arena.as_mut(),
                        &mut enc,
                        &crp_buf,
                        bytemuck::bytes_of(&ConvReshapeParams {
                            num_specs: num_specs_u32,
                            out_channels: oc as u32,
                            spatial: spatial as u32,
                            _padding: 0,
                        }),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut enc,
                        &gp_buf,
                        bytemuck::bytes_of(&GemmParams {
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            _padding: 0,
                        }),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut enc,
                        &ccp_buf,
                        bytemuck::bytes_of(&ConvCol2imParams {
                            num_specs: num_specs_u32,
                            flat_input_dim: out_d as u32,
                            out_h: oh as u32,
                            out_w: ow as u32,
                            in_channels: ic as u32,
                            in_h: ih as u32,
                            in_w: iw as u32,
                            kernel_h: kh as u32,
                            kernel_w: kw as u32,
                            stride_h: *stride_h as u32,
                            stride_w: *stride_w as u32,
                            pad_h: *pad_h as u32,
                            pad_w: *pad_w as u32,
                            kernel_cols: kernel_cols as u32,
                            _padding2: [0; 2],
                        }),
                    )?;
                    let disp = select_gemm_dispatch(m as u32, k as u32, n as u32);
                    let gemm_pipe = if disp.use_small_k {
                        &self.gemm_f32_small_k_pipeline
                    } else {
                        &self.gemm_f32_pipeline
                    };
                    // n=1 dispatch for the §0 DAZ row-L1 reduction `|A|@ones → row_abs_a`
                    // (incoming coeff `[num_specs × in_d]`).
                    let disp1 = select_gemm_dispatch(num_specs_u32, in_d as u32, 1);
                    let gemm_pipe1 = if disp1.use_small_k {
                        &self.gemm_f32_small_k_pipeline
                    } else {
                        &self.gemm_f32_pipeline
                    };
                    let reshape_wg = ((num_specs * spatial * oc) as u32).div_ceil(256);
                    let col2im_wg = ((num_specs * out_d) as u32).div_ceil(256);

                    if bias_expanded.is_some() {
                        self.pass_simple(
                            &mut enc,
                            bias_pipe,
                            &bp_buf,
                            &[&la[ping], &le[ping], &bias_buf, &blo, &ble],
                            num_specs_u32,
                        );
                        self.pass_simple(
                            &mut enc,
                            bias_pipe,
                            &bp_buf,
                            &[&ua[ping], &ue[ping], &bias_buf, &buo, &bue],
                            num_specs_u32,
                        );
                        // The bias fold consumes the incoming coefficient and
                        // error streams. Preserve their words at row granularity
                        // under the same exact-zero bias annihilation rule as
                        // Linear, before any Conv twin overwrites the next ping.
                        if let Some(tb) = taint.as_ref() {
                            for wb in
                                [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]]
                            {
                                self.taint_row_or_dispatch(
                                    &mut enc,
                                    wb,
                                    Some((&bias_buf, false)),
                                    num_specs,
                                    in_d,
                                    &tb.rows_dev,
                                );
                            }
                        }
                    }
                    // #cert-err bias charge — same construction as the Linear
                    // arm; the conv bias fold reduces over the EXPANDED bias
                    // (`in_d = oc·oh·ow`), so that is the reduction length.
                    self.cert_bias_charge_pass(
                        &mut enc,
                        bias_pipe,
                        CertBiasChargeArgs {
                            cert_err: *cert_err,
                            reduction: in_d,
                            num_specs,
                            layer_index: li,
                            params: cert_bp_buf.as_ref(),
                            operand: cert_bias_buf.as_ref(),
                            sink: cert_bias_sink.as_ref(),
                            a: [&la[ping], &ua[ping]],
                            a_err: [&le[ping], &ue[ping]],
                            bias_err_out: [&ble, &bue],
                        },
                    )?;
                    if let (Some(tb), true) = (taint.as_ref(), cert_err.bias_abs_err != 0.0) {
                        // #u4: unconditional row-OR — the charge dispatch reads
                        // the coefficient/error streams with no word twin and its
                        // constant operand is nonzero, so no annihilation applies.
                        for wb in [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]] {
                            self.taint_row_or_dispatch(
                                &mut enc,
                                wb,
                                None,
                                num_specs,
                                in_d,
                                &tb.rows_dev,
                            );
                        }
                    }
                    // #eft-err conv fit: the tiled twin's 16×16 grid must respect
                    // the device dispatch limit; past it, BOTH conv EFT blocks
                    // are skipped together (fail-closed to Higham — never a
                    // stale-buffer min-combine without its twin GEMM).
                    let conv_eft_fits = (n as u32).div_ceil(16) as usize <= max_wg
                        && (m as u32).div_ceil(16) as usize <= max_wg;
                    // Per side: coeff (reshape → GEMM → col2im), then — per-entry mode —
                    // the certified error through the SAME structure: S = |A|⊛|W| (abs of
                    // the already-reshaped coeff, so the reshape is not repeated), prop =
                    // err⊛|W|, combined per entry into the post-transform error buffer.
                    for &(side, src_a, src_e, dst_a, dst_e) in &[
                        (0usize, &la[ping], &le[ping], &la[1 - ping], &le[1 - ping]),
                        (1usize, &ua[ping], &ue[ping], &ua[1 - ping], &ue[1 - ping]),
                    ] {
                        let word_side = taint.as_ref().map(|tb| {
                            if side == 0 {
                                (
                                    &tb.wla[ping],
                                    &tb.wle[ping],
                                    &tb.wla[1 - ping],
                                    &tb.wle[1 - ping],
                                    &tb.w_rowabs_lo,
                                )
                            } else {
                                (
                                    &tb.wua[ping],
                                    &tb.wue[ping],
                                    &tb.wua[1 - ping],
                                    &tb.wue[1 - ping],
                                    &tb.w_rowabs_hi,
                                )
                            }
                        });
                        if let (Some(tb), Some((src_wa, _, dst_wa, _, _))) =
                            (taint.as_ref(), word_side)
                        {
                            let gemm_taint_pipe = if disp.use_small_k {
                                res_pipes.gemm_small_k_taint.as_ref().expect(
                                    "#u4: small-K twin availability checked at walk entry",
                                )
                            } else {
                                res_pipes.gemm_taint.as_ref().expect(
                                    "#u4: tiled twin availability checked at walk entry",
                                )
                            };
                            self.pass_simple(
                                &mut enc,
                                res_pipes.conv_reshape_taint.as_ref().expect(
                                    "#u4: Conv reshape twin availability checked at walk entry",
                                ),
                                &crp_buf,
                                &[src_a, &conv_reshaped, src_wa, &tb.w_conv_reshaped],
                                reshape_wg,
                            );
                            self.pass_simple_2d(
                                &mut enc,
                                gemm_taint_pipe,
                                &gp_buf,
                                &[
                                    &conv_reshaped,
                                    &w_buf,
                                    &conv_gemm,
                                    &tb.w_conv_reshaped,
                                    &tb.zw,
                                    &tb.w_conv_gemm,
                                ],
                                disp.wg_x,
                                disp.wg_y,
                            );
                            self.pass_simple(
                                &mut enc,
                                res_pipes.conv_col2im_taint.as_ref().expect(
                                    "#u4: Conv col2im twin availability checked at walk entry",
                                ),
                                &ccp_buf,
                                &[&conv_gemm, dst_a, &tb.w_conv_gemm, dst_wa],
                                col2im_wg,
                            );
                        } else {
                            self.pass_simple(
                                &mut enc,
                                rp,
                                &crp_buf,
                                &[src_a, &conv_reshaped],
                                reshape_wg,
                            );
                            self.pass_gemm(
                                &mut enc,
                                gemm_pipe,
                                &gp_buf,
                                &conv_reshaped,
                                &w_buf,
                                &conv_gemm,
                                disp.wg_x,
                                disp.wg_y,
                            );
                            self.pass_simple(
                                &mut enc,
                                cp,
                                &ccp_buf,
                                &[&conv_gemm, dst_a],
                                col2im_wg,
                            );
                        }
                        // #eft-err conv twin GEMM: recompute the conv GEMM with the
                        // barrier-fma sequence + exact residuals while
                        // `conv_reshaped` still holds the reshaped VALUE coeff (the
                        // error path overwrites it below). Per-entry mode only (the
                        // rowmax legacy path has no per-entry prop stream to keep).
                        // The word gate does not change the EFT value/residual
                        // calculation. Its later min-combine switches to the C2
                        // consult twin and reads the fully col2im-transported
                        // S/prop words.
                        if !conv_err_rowmax && conv_eft_fits {
                            if let (Some(evg), Some(erg)) =
                                (eft_vg_buf.as_ref(), eft_rg_buf.as_ref())
                            {
                                let pipes = self.resident_backward_pipelines();
                                self.pass_simple_2d(
                                    &mut enc,
                                    &pipes.eft_twin,
                                    &gp_buf,
                                    &[&conv_reshaped, &w_buf, evg, erg],
                                    (n as u32).div_ceil(16),
                                    (m as u32).div_ceil(16),
                                );
                            }
                        }
                        if !conv_err_rowmax {
                            let abs_w = abs_w_buf.as_deref().expect("per-entry conv error mode");
                            // §0 DAZ: ‖a_spec‖₁ of the INCOMING coeff, reduced BEFORE the
                            // error path overwrites `abs_a` with |reshaped| (#gpu-metal-daz).
                            self.pass_simple(
                                &mut enc,
                                abs_pipe,
                                &ap_buf,
                                &[src_a, &abs_a],
                                reshape_wg,
                            );
                            if let Some(tb) = taint.as_ref() {
                                let (src_wa, _, _, _, w_rowabs) =
                                    word_side.expect("word side iff taint is armed");
                                let row_taint_pipe = if disp1.use_small_k {
                                    res_pipes.gemm_small_k_taint.as_ref().expect(
                                        "#u4: small-K twin availability checked at walk entry",
                                    )
                                } else {
                                    res_pipes.gemm_taint.as_ref().expect(
                                        "#u4: tiled twin availability checked at walk entry",
                                    )
                                };
                                self.pass_simple_2d(
                                    &mut enc,
                                    row_taint_pipe,
                                    &gp1_buf,
                                    &[&abs_a, &ones_buf, &row_abs_a, src_wa, &tb.zw, w_rowabs],
                                    disp1.wg_x,
                                    disp1.wg_y,
                                );
                            } else {
                                self.pass_gemm(
                                    &mut enc,
                                    gemm_pipe1,
                                    &gp1_buf,
                                    &abs_a,
                                    &ones_buf,
                                    &row_abs_a,
                                    disp1.wg_x,
                                    disp1.wg_y,
                                );
                            }
                            self.pass_simple(
                                &mut enc,
                                abs_pipe,
                                &ap_buf,
                                &[&conv_reshaped, &abs_a],
                                reshape_wg,
                            );
                            if let Some(tb) = taint.as_ref() {
                                let (_, src_we, _, dst_we, _) =
                                    word_side.expect("word side iff taint is armed");
                                let gemm_taint_pipe = if disp.use_small_k {
                                    res_pipes.gemm_small_k_taint.as_ref().expect(
                                        "#u4: small-K twin availability checked at walk entry",
                                    )
                                } else {
                                    res_pipes.gemm_taint.as_ref().expect(
                                        "#u4: tiled twin availability checked at walk entry",
                                    )
                                };
                                let reshape_taint = res_pipes.conv_reshape_taint.as_ref().expect(
                                    "#u4: Conv reshape twin availability checked at walk entry",
                                );
                                let col2im_taint = res_pipes.conv_col2im_taint.as_ref().expect(
                                    "#u4: Conv col2im twin availability checked at walk entry",
                                );
                                // S = col2im(|reshape(A)| @ |W|). Abs preserves
                                // the coefficient word exactly.
                                self.pass_simple_2d(
                                    &mut enc,
                                    gemm_taint_pipe,
                                    &gp_buf,
                                    &[
                                        &abs_a,
                                        abs_w,
                                        &conv_gemm,
                                        &tb.w_conv_reshaped,
                                        &tb.zw,
                                        &tb.w_conv_gemm,
                                    ],
                                    disp.wg_x,
                                    disp.wg_y,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    col2im_taint,
                                    &ccp_buf,
                                    &[&conv_gemm, &s_scratch, &tb.w_conv_gemm, &tb.ws],
                                    col2im_wg,
                                );
                                // prop = col2im(reshape(err) @ |W|).
                                self.pass_simple(
                                    &mut enc,
                                    reshape_taint,
                                    &crp_buf,
                                    &[src_e, &conv_reshaped, src_we, &tb.w_conv_reshaped],
                                    reshape_wg,
                                );
                                self.pass_simple_2d(
                                    &mut enc,
                                    gemm_taint_pipe,
                                    &gp_buf,
                                    &[
                                        &conv_reshaped,
                                        abs_w,
                                        &conv_gemm,
                                        &tb.w_conv_reshaped,
                                        &tb.zw,
                                        &tb.w_conv_gemm,
                                    ],
                                    disp.wg_x,
                                    disp.wg_y,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    col2im_taint,
                                    &ccp_buf,
                                    &[&conv_gemm, &prop_scratch, &tb.w_conv_gemm, &tb.wprop],
                                    col2im_wg,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    res_pipes.combine_taint.as_ref().expect(
                                        "#u4: combine twin availability checked at walk entry",
                                    ),
                                    &cp_buf,
                                    &[
                                        &s_scratch,
                                        &prop_scratch,
                                        dst_e,
                                        &row_abs_a,
                                        &tb.ws,
                                        &tb.wprop,
                                        dst_we,
                                    ],
                                    col2im_wg,
                                );
                            } else {
                                self.pass_gemm(
                                    &mut enc,
                                    gemm_pipe,
                                    &gp_buf,
                                    &abs_a,
                                    abs_w,
                                    &conv_gemm,
                                    disp.wg_x,
                                    disp.wg_y,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    cp,
                                    &ccp_buf,
                                    &[&conv_gemm, &s_scratch],
                                    col2im_wg,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    rp,
                                    &crp_buf,
                                    &[src_e, &conv_reshaped],
                                    reshape_wg,
                                );
                                self.pass_gemm(
                                    &mut enc,
                                    gemm_pipe,
                                    &gp_buf,
                                    &conv_reshaped,
                                    abs_w,
                                    &conv_gemm,
                                    disp.wg_x,
                                    disp.wg_y,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    cp,
                                    &ccp_buf,
                                    &[&conv_gemm, &prop_scratch],
                                    col2im_wg,
                                );
                                self.pass_simple(
                                    &mut enc,
                                    combine_pipe,
                                    &cp_buf,
                                    &[&s_scratch, &prop_scratch, dst_e, &row_abs_a],
                                    col2im_wg,
                                );
                            }
                            // #eft-err conv: gather the twin (value, residual)
                            // streams through col2im, then min-tighten the conv
                            // combine's per-entry error with the measured bound.
                            // Same fits-guard as the twin GEMM above — the two
                            // blocks fire together or not at all. Under #u4 the
                            // C2 twin consults the post-col2im S/prop words.
                            if let (true, Some(evg), Some(erg), Some(ev), Some(er), Some(ecp)) = (
                                conv_eft_fits,
                                eft_vg_buf.as_ref(),
                                eft_rg_buf.as_ref(),
                                eft_v_buf.as_ref(),
                                eft_r_buf.as_ref(),
                                eft_cp_buf.as_ref(),
                            ) {
                                let pipes = self.resident_backward_pipelines();
                                self.pass_simple(
                                    &mut enc,
                                    &pipes.eft_col2im,
                                    &ccp_buf,
                                    &[evg, erg, ev, er],
                                    col2im_wg,
                                );
                                if let Some(tb) = taint.as_ref() {
                                    self.pass_simple(
                                        &mut enc,
                                        pipes.eft_min_combine_taint.as_ref().expect(
                                            "#u4: C2 twin availability checked at walk entry",
                                        ),
                                        ecp,
                                        &[
                                            ev,
                                            er,
                                            dst_a,
                                            &prop_scratch,
                                            dst_e,
                                            &row_abs_a,
                                            &s_scratch,
                                            &tb.ws,
                                            &tb.wprop,
                                        ],
                                        col2im_wg,
                                    );
                                } else {
                                    self.pass_simple(
                                        &mut enc,
                                        &pipes.eft_min_combine,
                                        ecp,
                                        &[
                                            ev,
                                            er,
                                            dst_a,
                                            &prop_scratch,
                                            dst_e,
                                            &row_abs_a,
                                            &s_scratch,
                                        ],
                                        col2im_wg,
                                    );
                                }
                            }
                        }
                    }
                    if conv_err_rowmax {
                        // Legacy row-max broadcast (reads PRE-transform coeff/err).
                        self.pass_simple(
                            &mut enc,
                            ep,
                            &cep_buf,
                            &[&la[ping], &le[ping], &le[1 - ping]],
                            num_specs_u32,
                        );
                        self.pass_simple(
                            &mut enc,
                            ep,
                            &cep_buf,
                            &[&ua[ping], &ue[ping], &ue[1 - ping]],
                            num_specs_u32,
                        );
                    }
                    if let Some(tb) = taint.as_ref() {
                        // The combine has no row-L1 word input. Preserve a
                        // saturation in that DAZ-cover reduction directly in
                        // the per-spec receipt, after both sides wrote it.
                        self.taint_row_or_dispatch(
                            &mut enc,
                            &tb.w_rowabs_lo,
                            None,
                            num_specs,
                            1,
                            &tb.rows_dev,
                        );
                        self.taint_row_or_dispatch(
                            &mut enc,
                            &tb.w_rowabs_hi,
                            None,
                            num_specs,
                            1,
                            &tb.rows_dev,
                        );
                    }
                    if coalesce {
                        fold_cmds.push(enc.finish());
                    } else {
                        self.submit_ticked(enc.finish());
                    }
                    ping = 1 - ping;
                    self.apply_intermediate_sweep_boundary(
                        li + 1,
                        out_d,
                        num_specs,
                        &la[ping],
                        &ua[ping],
                        &le[ping],
                        &ue[ping],
                        &blo,
                        &buo,
                        &ble,
                        &bue,
                        taint.as_mut(),
                    )?;
                    continue;
                }

                // ---- Linear layer ----
                let GpuCrownLayer::Linear {
                    weight,
                    bias,
                    out_features,
                    in_features,
                    cert_err,
                } = layer
                else {
                    unreachable!("validated above");
                };
                let (of, if_) = (*out_features, *in_features);
                // #cert-err: `g` becomes `gamma + w_rel + gamma·w_rel` and the
                // combine `slack` picks up the matching `(1 + w_rel)` factor;
                // together they dominate `((gamma+w_rel)·|A| + (1+w_rel)·err) @ |W|`,
                // the same composition the CPU margin-row lane uses. `cert_err`
                // all-zero ⇒ `charged_gamma` returns `gamma`'s bits and the slack
                // factor is exactly 1, so both uniforms are bit-identical to the
                // pre-`cert_err` build.
                let g_exact = gamma_k_f32(of)?;
                let slack_exact = combine_slack_f32(of)?;
                let g = cert_err.charged_gamma(g_exact);
                let slack = cert_charged_slack(slack_exact, *cert_err, li)?;
                let eft_slack = if eft_on { eft_r_slack_f32(of)? } else { 0.0 };
                let reduction_u32 = u32::try_from(of).unwrap_or(u32::MAX);
                let additive = if eft_on {
                    super::super::sound_consts::rung3_flush_safe_additive_scaled(
                        reduction_u32,
                        eft_slack,
                    )?
                } else {
                    super::super::sound_consts::rung3_flush_safe_additive(reduction_u32)?
                }; // FTZ+rung3-fma-safe

                let __t_wp = std::time::Instant::now();
                // #lever1 weight residency: constant W and |W| are GPU-resident
                // (uploaded once per model, Arc-identity keyed + keep-alive; see
                // ops/resident_weights.rs) instead of re-uploaded — with |W|
                // re-computed on CPU — per domain per call. Identical bytes on
                // the identical read-only bindings.
                let w_buf = self.resident_weight_buf(weight, WeightForm::Raw)?;
                let abs_w_buf = self.resident_weight_buf(weight, WeightForm::Abs)?;
                // §0 weight-amplified DAZ floor: max_j‖w_j‖₁ over the `of × if_` weight
                // (each output column j sums `of` weight rows). A scalar OUTWARD bound
                // on every column's L1 (#gpu-metal-daz). Summing `weight[..].abs()` is
                // bit-identical to summing the old CPU `absw` vector (|·| is exact).
                // #daz-flush-cover-v2 (dark, default OFF ⇒ byte-identical): see
                // `sound_consts::daz_flush_cover_w_l1` for the derivation of why
                // `‖w_j‖₁` must be carried more than once on a DAZ adapter, and why
                // a subnormal weight has to fail closed instead.
                crate::wgpu_device::sound_consts::refuse_subnormal_weight_under_daz_cover(
                    weight,
                    "linear weight",
                    daz_cover_armed,
                )?;
                let mut w_l1_max = 0.0f32;
                for c in 0..if_ {
                    let s = (0..of)
                        .map(|r| f32_to_f64_exact(weight[r * if_ + c]).abs())
                        .sum();
                    w_l1_max = w_l1_max.max(up_f32(s));
                }
                let w_l1_max = crate::wgpu_device::sound_consts::daz_flush_cover_w_l1(
                    w_l1_max,
                    daz_cover_armed,
                )?;
                __cpu_wprep += __t_wp.elapsed();
                // #fold-coalesce: encoder BEFORE the uploads (arena copies must
                // be encoder-ordered ahead of this layer's passes).
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("res_layer"),
                        });
                self.fold_upload(
                    arena.as_mut(),
                    &mut encoder,
                    &gp_buf,
                    bytemuck::bytes_of(&GemmParams {
                        m: num_specs_u32,
                        k: of as u32,
                        n: if_ as u32,
                        _padding: 0,
                    }),
                )?;
                // n=1 row-L1 reduction params: |A|[num_specs × of] @ ones[of × 1].
                self.fold_upload(
                    arena.as_mut(),
                    &mut encoder,
                    &gp1_buf,
                    bytemuck::bytes_of(&GemmParams {
                        m: num_specs_u32,
                        k: of as u32,
                        n: 1,
                        _padding: 0,
                    }),
                )?;
                self.fold_upload(
                    arena.as_mut(),
                    &mut encoder,
                    &cp_buf,
                    bytemuck::bytes_of(&CombineParams {
                        n: (num_specs * if_) as u32,
                        slack,
                        gamma_k: g,
                        additive,
                        k: of as u32,
                        out_cols: if_ as u32,
                        w_l1_max,
                        _pad: 0,
                    }),
                )?;
                if let Some(eft_cp) = eft_cp_buf.as_ref() {
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        eft_cp,
                        bytemuck::bytes_of(&EftCombineParams {
                            n: (num_specs * if_) as u32,
                            r_slack: eft_slack,
                            slack,
                            additive,
                            k: of as u32,
                            out_cols: if_ as u32,
                            w_l1_max,
                            _pad: 0,
                        }),
                    )?;
                }
                self.fold_upload(
                    arena.as_mut(),
                    &mut encoder,
                    &ap_buf,
                    bytemuck::bytes_of(&AbsParams {
                        n: (num_specs * of) as u32,
                        _p: [0; 3],
                    }),
                )?;
                if let Some(b) = bias {
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &bias_buf,
                        bytemuck::cast_slice(b),
                    )?;
                    self.fold_upload(
                        arena.as_mut(),
                        &mut encoder,
                        &bp_buf,
                        bytemuck::bytes_of(&BiasParams {
                            num_specs: num_specs_u32,
                            k: of as u32,
                            gamma_k: g,
                            additive,
                            // #flush-charge: identity unless charged authority
                            // is armed (oracle-derived bias factor widening).
                            slack: charged_bias_slack_or(charged_policy.as_ref(), slack)?,
                            eft_mode: u32::from(eft_on),
                            eft_r_slack: eft_slack,
                            _p: 0,
                        }),
                    )?;
                }

                let disp = select_gemm_dispatch(num_specs_u32, of as u32, if_ as u32);
                let gemm_pipe = if disp.use_small_k {
                    &self.gemm_f32_small_k_pipeline
                } else {
                    &self.gemm_f32_pipeline
                };
                // n=1 dispatch for the row-L1 reduction `|A|@ones → row_abs_a`.
                let disp1 = select_gemm_dispatch(num_specs_u32, of as u32, 1);
                let gemm_pipe1 = if disp1.use_small_k {
                    &self.gemm_f32_small_k_pipeline
                } else {
                    &self.gemm_f32_pipeline
                };
                // #u4 gate ON: match the base scheduler exactly. Both tiled
                // and large-M/small-K schedules have exact-value word twins,
                // so no shape-specific refusal or reduction-order drift remains.
                let taint_dims = taint.as_ref().map(|_| {
                    (
                        (disp.wg_x, disp.wg_y),
                        (disp1.wg_x, disp1.wg_y),
                    )
                });
                let taint_gemm_pipe = taint.as_ref().map(|_| {
                    if disp.use_small_k {
                        res_pipes.gemm_small_k_taint.as_ref().expect(
                            "#u4: small-K twin availability checked at walk entry",
                        )
                    } else {
                        res_pipes.gemm_taint.as_ref().expect(
                            "#u4: tiled twin availability checked at walk entry",
                        )
                    }
                });
                let taint_row_gemm_pipe = taint.as_ref().map(|_| {
                    if disp1.use_small_k {
                        res_pipes.gemm_small_k_taint.as_ref().expect(
                            "#u4: small-K twin availability checked at walk entry",
                        )
                    } else {
                        res_pipes.gemm_taint.as_ref().expect(
                            "#u4: tiled twin availability checked at walk entry",
                        )
                    }
                });
                let abs_wg = ((num_specs * of) as u32).div_ceil(256);
                let mn = ((num_specs * if_) as u32).div_ceil(256);

                // Bias contribution reads the PRE-GEMM coefficient (host ordering).
                if bias.is_some() {
                    self.pass_simple(
                        &mut encoder,
                        bias_pipe,
                        &bp_buf,
                        &[&la[ping], &le[ping], &bias_buf, &blo, &ble],
                        num_specs_u32,
                    );
                    self.pass_simple(
                        &mut encoder,
                        bias_pipe,
                        &bp_buf,
                        &[&ua[ping], &ue[ping], &bias_buf, &buo, &bue],
                        num_specs_u32,
                    );
                }
                // #cert-err bias charge: `d·(Σ|a_j| + Σ err_j)` via a second
                // dispatch of the SAME kernel with the constant `[d; of]` operand
                // and `gamma_k = 1`; its centre-bias output goes to the sink so
                // only `ble`/`bue` (which the kernel accumulates into) change.
                // Runs even when `bias` is None: the declared error says the
                // EXACT fold has a bias this close to the supplied (absent, i.e.
                // zero) one, and that discrepancy still folds into the bound.
                self.cert_bias_charge_pass(
                    &mut encoder,
                    bias_pipe,
                    CertBiasChargeArgs {
                        cert_err: *cert_err,
                        reduction: of,
                        num_specs,
                        layer_index: li,
                        params: cert_bp_buf.as_ref(),
                        operand: cert_bias_buf.as_ref(),
                        sink: cert_bias_sink.as_ref(),
                        a: [&la[ping], &ua[ping]],
                        a_err: [&le[ping], &ue[ping]],
                        bias_err_out: [&ble, &bue],
                    },
                )?;
                if let (Some(tb), true) = (taint.as_ref(), cert_err.bias_abs_err != 0.0) {
                    // #u4: the charge dispatch consumes the coefficient AND error
                    // streams with no word twin. Transport their words to the row
                    // accumulator UNCONDITIONALLY (no annihilation exception —
                    // the constant operand `d` is nonzero by construction here).
                    for wb in [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]] {
                        self.taint_row_or_dispatch(&mut encoder, wb, None, num_specs, of, &tb.rows_dev);
                    }
                }
                // A_new = A @ W.
                // #u4 gate ON: the CROWN-path linear GEMM through the taint
                // twin — value bits identical (drift-pinned), coefficient words
                // rotated `[ping] → [1 - ping]`; the weight word is the all-zero
                // `zw` (host weights are exact data, never tainted — G7).
                if let (Some(tb), Some(((twx, twy), _)), Some(tg)) =
                    (taint.as_ref(), taint_dims, taint_gemm_pipe)
                {
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[
                            &la[ping],
                            &w_buf,
                            &la[1 - ping],
                            &tb.wla[ping],
                            &tb.zw,
                            &tb.wla[1 - ping],
                        ],
                        twx,
                        twy,
                    );
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[
                            &ua[ping],
                            &w_buf,
                            &ua[1 - ping],
                            &tb.wua[ping],
                            &tb.zw,
                            &tb.wua[1 - ping],
                        ],
                        twx,
                        twy,
                    );
                } else {
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &la[ping],
                        &w_buf,
                        &la[1 - ping],
                        disp.wg_x,
                        disp.wg_y,
                    );
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &ua[ping],
                        &w_buf,
                        &ua[1 - ping],
                        disp.wg_x,
                        disp.wg_y,
                    );
                }
                // err_new = combine(γ_k·|A|@|W|, err@|W|). `row_abs_a = |A|@ones` (the
                // §0 DAZ per-spec ‖a_i‖₁) is reduced from the SAME |A| the combine
                // reads, in-order before its combine consumes it (#gpu-metal-daz).
                self.pass_simple(
                    &mut encoder,
                    abs_pipe,
                    &ap_buf,
                    &[&la[ping], &abs_a],
                    abs_wg,
                );
                // #u4 gate ON (lower err chain): the row-L1 + S + prop GEMMs go
                // through the taint twin (`word(|A|) == word(A)` — abs is
                // magnitude-preserving, so the S/row-L1 twins bind wla[ping]
                // directly) and the combine goes through its twin, ORing
                // word(S)|word(P) plus its own degrade seeds into wle[1-ping].
                if let (
                    Some(tb),
                    Some(((twx, twy), (twx1, twy1))),
                    Some(tg),
                    Some(tg1),
                ) = (
                    taint.as_ref(),
                    taint_dims,
                    taint_gemm_pipe,
                    taint_row_gemm_pipe,
                )
                {
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[&abs_a, &abs_w_buf, &s_scratch, &tb.wla[ping], &tb.zw, &tb.ws],
                        twx,
                        twy,
                    );
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[
                            &le[ping],
                            &abs_w_buf,
                            &prop_scratch,
                            &tb.wle[ping],
                            &tb.zw,
                            &tb.wprop,
                        ],
                        twx,
                        twy,
                    );
                    // §0 row-L1 word: the combine twin has no `row_abs_a` word
                    // binding, so `‖a_i‖₁`'s own saturation word rides to the
                    // row accumulator directly (on-device row-OR at the end of
                    // this layer's encoder).
                    self.pass_simple_2d(
                        &mut encoder,
                        tg1,
                        &gp1_buf,
                        &[
                            &abs_a,
                            &ones_buf,
                            &row_abs_a,
                            &tb.wla[ping],
                            &tb.zw,
                            &tb.w_rowabs_lo,
                        ],
                        twx1,
                        twy1,
                    );
                    self.pass_simple(
                        &mut encoder,
                        res_pipes.combine_taint.as_ref().expect("#u4: availability checked at walk entry"),
                        &cp_buf,
                        &[
                            &s_scratch,
                            &prop_scratch,
                            &le[1 - ping],
                            &row_abs_a,
                            &tb.ws,
                            &tb.wprop,
                            &tb.wle[1 - ping],
                        ],
                        mn,
                    );
                } else {
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &abs_a,
                        &abs_w_buf,
                        &s_scratch,
                        disp.wg_x,
                        disp.wg_y,
                    );
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &le[ping],
                        &abs_w_buf,
                        &prop_scratch,
                        disp.wg_x,
                        disp.wg_y,
                    );
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe1,
                        &gp1_buf,
                        &abs_a,
                        &ones_buf,
                        &row_abs_a,
                        disp1.wg_x,
                        disp1.wg_y,
                    );
                    self.pass_simple(
                        &mut encoder,
                        combine_pipe,
                        &cp_buf,
                        &[&s_scratch, &prop_scratch, &le[1 - ping], &row_abs_a],
                        mn,
                    );
                }
                // #eft-err (LOWER side): recompute A@W with the deterministic
                // barrier-fma twin (value + exact residual sum), then tighten the
                // just-written Higham error via min. Sequenced BEFORE the upper
                // side reuses prop_scratch/row_abs_a. Gate off ⇒ no dispatches.
                // Tiled-twin grid (16×16); y over rows. Fail-closed past the
                // device dispatch limit: skip the tightening, keep Higham.
                let eft_wg_x = (if_ as u32).div_ceil(16);
                let eft_wg_y = num_specs_u32.div_ceil(16);
                let eft_fits = eft_wg_x as usize <= max_wg && eft_wg_y as usize <= max_wg;
                if let (true, Some(ev), Some(er), Some(ecp)) = (
                    eft_fits,
                    eft_v_buf.as_ref(),
                    eft_r_buf.as_ref(),
                    eft_cp_buf.as_ref(),
                ) {
                    let pipes = self.resident_backward_pipelines();
                    self.pass_simple_2d(
                        &mut encoder,
                        &pipes.eft_twin,
                        &gp_buf,
                        &[&la[ping], &w_buf, ev, er],
                        eft_wg_x,
                        eft_wg_y,
                    );
                    // #u4 gate ON: the min-combine CONSULT twin (audit C2) —
                    // words in (S/P words of this side's GEMM twins); a set
                    // word refuses the tightening IN-SHADER, keeping the
                    // Higham charge (strictly widening, never tighter).
                    if let Some(tb) = taint.as_ref() {
                        self.pass_simple(
                            &mut encoder,
                            res_pipes.eft_min_combine_taint.as_ref().expect("#u4: availability checked at walk entry"),
                            ecp,
                            &[
                                ev,
                                er,
                                &la[1 - ping],
                                &prop_scratch,
                                &le[1 - ping],
                                &row_abs_a,
                                &s_scratch,
                                &tb.ws,
                                &tb.wprop,
                            ],
                            mn,
                        );
                    } else {
                        self.pass_simple(
                            &mut encoder,
                            &pipes.eft_min_combine,
                            ecp,
                            &[
                                ev,
                                er,
                                &la[1 - ping],
                                &prop_scratch,
                                &le[1 - ping],
                                &row_abs_a,
                                &s_scratch,
                            ],
                            mn,
                        );
                    }
                }
                self.pass_simple(
                    &mut encoder,
                    abs_pipe,
                    &ap_buf,
                    &[&ua[ping], &abs_a],
                    abs_wg,
                );
                // #u4 gate ON (upper err chain): mirror of the lower chain with
                // wua/wue and the upper row-L1 word buffer.
                if let (
                    Some(tb),
                    Some(((twx, twy), (twx1, twy1))),
                    Some(tg),
                    Some(tg1),
                ) = (
                    taint.as_ref(),
                    taint_dims,
                    taint_gemm_pipe,
                    taint_row_gemm_pipe,
                )
                {
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[&abs_a, &abs_w_buf, &s_scratch, &tb.wua[ping], &tb.zw, &tb.ws],
                        twx,
                        twy,
                    );
                    self.pass_simple_2d(
                        &mut encoder,
                        tg,
                        &gp_buf,
                        &[
                            &ue[ping],
                            &abs_w_buf,
                            &prop_scratch,
                            &tb.wue[ping],
                            &tb.zw,
                            &tb.wprop,
                        ],
                        twx,
                        twy,
                    );
                    self.pass_simple_2d(
                        &mut encoder,
                        tg1,
                        &gp1_buf,
                        &[
                            &abs_a,
                            &ones_buf,
                            &row_abs_a,
                            &tb.wua[ping],
                            &tb.zw,
                            &tb.w_rowabs_hi,
                        ],
                        twx1,
                        twy1,
                    );
                    self.pass_simple(
                        &mut encoder,
                        res_pipes.combine_taint.as_ref().expect("#u4: availability checked at walk entry"),
                        &cp_buf,
                        &[
                            &s_scratch,
                            &prop_scratch,
                            &ue[1 - ping],
                            &row_abs_a,
                            &tb.ws,
                            &tb.wprop,
                            &tb.wue[1 - ping],
                        ],
                        mn,
                    );
                } else {
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &abs_a,
                        &abs_w_buf,
                        &s_scratch,
                        disp.wg_x,
                        disp.wg_y,
                    );
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe,
                        &gp_buf,
                        &ue[ping],
                        &abs_w_buf,
                        &prop_scratch,
                        disp.wg_x,
                        disp.wg_y,
                    );
                    self.pass_gemm(
                        &mut encoder,
                        gemm_pipe1,
                        &gp1_buf,
                        &abs_a,
                        &ones_buf,
                        &row_abs_a,
                        disp1.wg_x,
                        disp1.wg_y,
                    );
                    self.pass_simple(
                        &mut encoder,
                        combine_pipe,
                        &cp_buf,
                        &[&s_scratch, &prop_scratch, &ue[1 - ping], &row_abs_a],
                        mn,
                    );
                }
                // #eft-err (UPPER side): same twin + min tightening on ue.
                if let (true, Some(ev), Some(er), Some(ecp)) = (
                    eft_fits,
                    eft_v_buf.as_ref(),
                    eft_r_buf.as_ref(),
                    eft_cp_buf.as_ref(),
                ) {
                    let pipes = self.resident_backward_pipelines();
                    self.pass_simple_2d(
                        &mut encoder,
                        &pipes.eft_twin,
                        &gp_buf,
                        &[&ua[ping], &w_buf, ev, er],
                        eft_wg_x,
                        eft_wg_y,
                    );
                    // #u4 gate ON: upper-side C2 consult (see the lower block).
                    if let Some(tb) = taint.as_ref() {
                        self.pass_simple(
                            &mut encoder,
                            res_pipes.eft_min_combine_taint.as_ref().expect("#u4: availability checked at walk entry"),
                            ecp,
                            &[
                                ev,
                                er,
                                &ua[1 - ping],
                                &prop_scratch,
                                &ue[1 - ping],
                                &row_abs_a,
                                &s_scratch,
                                &tb.ws,
                                &tb.wprop,
                            ],
                            mn,
                        );
                    } else {
                        self.pass_simple(
                            &mut encoder,
                            &pipes.eft_min_combine,
                            ecp,
                            &[
                                ev,
                                er,
                                &ua[1 - ping],
                                &prop_scratch,
                                &ue[1 - ping],
                                &row_abs_a,
                                &s_scratch,
                            ],
                            mn,
                        );
                    }
                }

                if let Some(tb) = taint.as_ref() {
                    // #u4 fail-closed transport, ON-DEVICE — the Linear bias
                    // fold (CROWN_BIAS_ERR_ACCUMULATE_SHADER) has NO taint
                    // twin: it consumes the PRE-GEMM coefficient AND error per
                    // spec row, so their words row-OR into `rows_dev` under
                    // the committed `bias[k] != 0` annihilation conjunct
                    // (audit §4 C1, bias_fold_f64 analogue) — expressed as a
                    // per-COLUMN partner read of `bias_buf` (uploaded above
                    // for the value fold; only its first `of` entries are
                    // indexed). The CPU reference is `bias_fold_taint`
                    // (crown_backward_sound_host.rs, now test-reference only).
                    // `[ping]` words are stable within this encoder (this
                    // layer's twins write only `[1 - ping]` and the
                    // scratches).
                    if bias.is_some() {
                        for wb in [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]] {
                            self.taint_row_or_dispatch(
                                &mut encoder,
                                wb,
                                Some((&bias_buf, false)),
                                num_specs,
                                of,
                                &tb.rows_dev,
                            );
                        }
                    }
                    // #u4 fail-closed transport, ON-DEVICE — the combine twin
                    // carries no `row_abs_a` word binding, so the §0 row-L1
                    // reduction's own saturation word (an under-covering flush
                    // term) ORs straight into its spec row — encoded AFTER
                    // both sides' twin GEMMs wrote `w_rowabs_lo`/`w_rowabs_hi`
                    // above. NOTE (tightness, not soundness): this per-LAYER
                    // row-OR condemns the spec row even when a later dead-ReLU
                    // would have annihilated the per-element word —
                    // refusal-only conservatism, fine while dark.
                    self.taint_row_or_dispatch(
                        &mut encoder,
                        &tb.w_rowabs_lo,
                        None,
                        num_specs,
                        1,
                        &tb.rows_dev,
                    );
                    self.taint_row_or_dispatch(
                        &mut encoder,
                        &tb.w_rowabs_hi,
                        None,
                        num_specs,
                        1,
                        &tb.rows_dev,
                    );
                }
                if coalesce {
                    fold_cmds.push(encoder.finish());
                } else {
                    self.submit_ticked(encoder.finish());
                }
                ping = 1 - ping;
                self.apply_intermediate_sweep_boundary(
                    li + 1,
                    if_,
                    num_specs,
                    &la[ping],
                    &ua[ping],
                    &le[ping],
                    &ue[ping],
                    &blo,
                    &buo,
                    &ble,
                    &bue,
                    taint.as_mut(),
                )?;
            }

            // #fold-coalesce: ONE submission for the whole chain. The arena must
            // be unmapped first (a mapped buffer in a submission is a validation
            // error) and stay alive until after the submit.
            let _arena_keepalive = arena.take().map(FoldStagingArena::finish);
            if coalesce && !fold_cmds.is_empty() {
                self.queue.submit(fold_cmds);
            }

            let __t_loop = __t_loop_start.elapsed();

            // Authoritative intermediate-sweep keep-out. Copy the active
            // value/error frontier, all four word twins, and the sticky row
            // accumulator into one exact-sized, independently owned carrier.
            // No word is folded to host and no coefficient crosses PCIe.
            if sweep_keep {
                if !grad_bufs.is_empty() || !gather_bufs.is_empty() {
                    return Err(NyError::UnsupportedOp(
                        "worded sweep keep mode with capture channels armed".into(),
                    ));
                }
                let tb = taint.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "worded sweep keep mode lost the taint-word route".into(),
                    )
                })?;
                let limits = self.device.limits();
                let layout = SweepCarrierLayout::new(num_specs, final_dim)?
                    .validate_device_limits(
                        limits.max_buffer_size,
                        limits.max_storage_buffer_binding_size,
                    )?;
                let out = DeviceSweepCarrier::allocate_zero_initialized(
                    &self.device,
                    layout,
                    "res_sweep_out",
                )?;
                let mut encoder = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("res_sweep_keep"),
                    },
                );
                for (source, destination) in [
                    (&la[ping], &out.matrix.lower_center),
                    (&ua[ping], &out.matrix.upper_center),
                    (&le[ping], &out.matrix.lower_radius),
                    (&ue[ping], &out.matrix.upper_radius),
                    (&tb.wla[ping], &out.matrix.lower_center_word),
                    (&tb.wua[ping], &out.matrix.upper_center_word),
                    (&tb.wle[ping], &out.matrix.lower_radius_word),
                    (&tb.wue[ping], &out.matrix.upper_radius_word),
                ] {
                    encoder.copy_buffer_to_buffer(
                        source,
                        0,
                        destination,
                        0,
                        layout.matrix_bytes,
                    );
                }
                for (source, destination) in [
                    (&blo, &out.row.lower_bias),
                    (&buo, &out.row.upper_bias),
                    (&ble, &out.row.lower_bias_radius),
                    (&bue, &out.row.upper_bias_radius),
                    (&tb.rows_dev, &out.row.taint_rows),
                ] {
                    encoder.copy_buffer_to_buffer(
                        source,
                        0,
                        destination,
                        0,
                        layout.row_bytes,
                    );
                }
                self.submit_ticked(encoder.finish());
                RESIDENT_IO.with(|io| io.borrow_mut().sweep_out = Some(out));
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    None,
                ));
            }

            // #seg-resident keep-out: deposit handle-clones of the final stream
            // and SKIP the readback entirely — the caller (the resnet segment
            // orchestrator) consumes the buffers on-device. The returned CPU
            // vectors are intentionally EMPTY (contract: a keep-mode caller
            // never reads them). Only legal without capture channels.
            if dev_keep {
                if !grad_bufs.is_empty() || !gather_bufs.is_empty() {
                    return Err(NyError::UnsupportedOp(
                        "seg-resident keep mode with capture channels armed".into(),
                    ));
                }
                let out = ResidentCoeffBufs {
                    la: la[ping].clone(),
                    ua: ua[ping].clone(),
                    le: le[ping].clone(),
                    ue: ue[ping].clone(),
                    blo,
                    buo,
                    ble,
                    bue,
                    dim: final_dim,
                    num_specs,
                };
                RESIDENT_IO.with(|io| io.borrow_mut().out = Some(out));
                // Match the checked-closure's tuple shape with EMPTY host data;
                // the outer ResidentCoeff construction flows through unchanged.
                // (#u4: taint_on + keep-mode was refused at fn entry ⇒ None.)
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    None,
                ));
            }

            let __t_dl_start = std::time::Instant::now();
            // Stage the FINAL coefficients + bias into MAP_READ buffers — ONE
            // download for the whole backward (the per-layer round-trip is gone).
            let out_elems = num_specs * final_dim;
            let stage = |label: &str, n: usize| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (n.max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let cbytes = (out_elems * size_of::<f32>()) as u64;
            let bbytes = (num_specs * size_of::<f32>()) as u64;
            let st_la = stage("st_la", out_elems);
            let st_ua = stage("st_ua", out_elems);
            let st_le = stage("st_le", out_elems);
            let st_ue = stage("st_ue", out_elems);
            let st_blo = stage("st_blo", num_specs);
            let st_buo = stage("st_buo", num_specs);
            let st_ble = stage("st_ble", num_specs);
            let st_bue = stage("st_bue", num_specs);
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("res_dl"),
                });
            enc.copy_buffer_to_buffer(&la[ping], 0, &st_la, 0, cbytes);
            enc.copy_buffer_to_buffer(&ua[ping], 0, &st_ua, 0, cbytes);
            enc.copy_buffer_to_buffer(&le[ping], 0, &st_le, 0, cbytes);
            enc.copy_buffer_to_buffer(&ue[ping], 0, &st_ue, 0, cbytes);
            enc.copy_buffer_to_buffer(&blo, 0, &st_blo, 0, bbytes);
            enc.copy_buffer_to_buffer(&buo, 0, &st_buo, 0, bbytes);
            enc.copy_buffer_to_buffer(&ble, 0, &st_ble, 0, bbytes);
            enc.copy_buffer_to_buffer(&bue, 0, &st_bue, 0, bbytes);
            // #u4 final word fold, ON-DEVICE (audit §4 C1 "plumbed from"):
            // row-OR the FINAL coefficient/error word buffers into `rows_dev`
            // (unconditional — the frontier ships as-is, nothing multiplies
            // it here), then stage the per-spec-row accumulator for the
            // walk's ONE word readback below — the only word transport that
            // ever touches the host.
            let taint_rows_stage = if let Some(tb) = taint.as_ref() {
                for wb in [&tb.wla[ping], &tb.wua[ping], &tb.wle[ping], &tb.wue[ping]] {
                    self.taint_row_or_dispatch(
                        &mut enc,
                        wb,
                        None,
                        num_specs,
                        final_dim,
                        &tb.rows_dev,
                    );
                }
                let st = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("st_taint_rows"),
                    size: (num_specs.max(1) * size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(
                    &tb.rows_dev,
                    0,
                    &st,
                    0,
                    (num_specs * size_of::<u32>()) as u64,
                );
                Some(st)
            } else {
                None
            };
            self.queue.submit(Some(enc.finish()));
            super::intermediate_sweep::note_submits(1);

            // Download per-ReLU alpha gradients (small; empty unless capturing).
            let mut relu_grads: Vec<Vec<f32>> = Vec::with_capacity(grad_bufs.len());
            for (gb, n) in &grad_bufs {
                let stg = stage("st_grad", *n);
                let mut ge = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("res_grad_dl"),
                    });
                ge.copy_buffer_to_buffer(gb, 0, &stg, 0, (*n * size_of::<f32>()) as u64);
                self.queue.submit(Some(ge.finish()));
                relu_grads.push(Self::read_buffer(&self.device, &stg, *n)?);
            }

            // Download per-ReLU gathered A-values (tiny; empty unless capturing).
            // The gather buffers are already MAP_READ (filled by in-encoder copies),
            // so they map directly without a second staging hop.
            let mut beta_gather_v: Vec<Vec<f32>> = Vec::with_capacity(gather_bufs.len());
            for slot in &gather_bufs {
                match slot {
                    Some((gb, n)) => beta_gather_v.push(Self::read_buffer(&self.device, gb, *n)?),
                    None => beta_gather_v.push(Vec::new()),
                }
            }

            // All 8 staging buffers were filled by the SINGLE `res_dl` submit
            // above, so they are all ready after one poll. Map them together with
            // ONE blocking `device.poll(Wait)` instead of 8 sequential polls
            // (one per `read_buffer`). Bit-identical: each returned vec is the same
            // `get_mapped_range()[..count].to_vec()` of the same staging buffer.
            let mut dl = Self::read_buffers_batched(
                &self.device,
                &[
                    (&st_la, out_elems),
                    (&st_ua, out_elems),
                    (&st_le, out_elems),
                    (&st_ue, out_elems),
                    (&st_blo, num_specs),
                    (&st_buo, num_specs),
                    (&st_ble, num_specs),
                    (&st_bue, num_specs),
                ],
            )?;
            if __probe {
                let __t_dl = __t_dl_start.elapsed();
                eprintln!(
                    "[resident] num_specs={num_specs} max_dim={max_dim} a_elems={a_elems} \
                     ({a_mib:.0}MiB) | setup={setup:.3}s cpu_wprep={wp:.3}s loop={lp:.3}s \
                     readback={dl:.3}s total={tot:.3}s",
                    a_mib = (a_elems * 4) as f64 / (1024.0 * 1024.0),
                    setup = __t_setup.as_secs_f64(),
                    wp = __cpu_wprep.as_secs_f64(),
                    lp = __t_loop.as_secs_f64() - __cpu_wprep.as_secs_f64(),
                    dl = __t_dl.as_secs_f64(),
                    tot = __t_start.elapsed().as_secs_f64(),
                );
            }
            let fbue_v = dl.pop().expect("8 readbacks");
            let fble_v = dl.pop().expect("8 readbacks");
            let fbuo_v = dl.pop().expect("8 readbacks");
            let fblo_v = dl.pop().expect("8 readbacks");
            let fue_v = dl.pop().expect("8 readbacks");
            let fle_v = dl.pop().expect("8 readbacks");
            let fua_v = dl.pop().expect("8 readbacks");
            let fla_v = dl.pop().expect("8 readbacks");
            // #u4: the walk's single word readback. `rows_dev` (staged into
            // the `res_dl` submit above, AFTER the on-device final fold)
            // already holds EVERY admitted transport — the intercept/bias
            // conjunct folds, the row-L1 words and the final
            // coefficient/error fold. OR it into the host accumulator
            // that carried the walk-boundary G13 seed-BIAS rows, then hand
            // the rows out for `concretize_resident_coeff_batched` to pass to
            // the consult.
            let taint_rows_out = if let Some(mut tb) = taint.take() {
                let rows_stage = taint_rows_stage
                    .as_ref()
                    .expect("#u4: gate on staged the row words in the res_dl submit");
                let dev_rows = Self::read_u32_buffer(&self.device, rows_stage, num_specs)?;
                for (slot, w) in tb.rows.iter_mut().zip(dev_rows) {
                    *slot |= w;
                }
                Some(tb.rows)
            } else {
                None
            };
            Ok((
                fla_v,
                fua_v,
                fle_v,
                fue_v,
                fblo_v,
                fbuo_v,
                fble_v,
                fbue_v,
                relu_grads,
                beta_gather_v,
                taint_rows_out,
            ))
        })?;

        Ok(ResidentCoeff {
            lower_a: fla,
            upper_a: fua,
            lower_err: fle,
            upper_err: fue,
            lower_b: fblo,
            upper_b: fbuo,
            lower_b_err: fble,
            upper_b_err: fbue,
            dim: final_dim,
            relu_grads: f_relu_grads,
            beta_gather: f_beta_gather,
            // #u4: Some(per-spec-row words) iff the AUTO/explicit worded route
            // ran the twin chain above; None ⇒ byte-identical opt-out walk.
            taint_rows: f_taint_rows,
        })
    }

    /// Sound backward through ONE residual block `out = F(z) + z` (identity skip),
    /// where `F` is the `branch` sub-chain and `z` is the block input (= block
    /// output dim `block_dim`). Forks the incoming coefficient:
    /// `A_in = backward_F(A) + A`. The branch backward is the proven resident path;
    /// the identity-skip stream is the seed `A` itself (exact), added to the branch
    /// coefficient with a certified f32-add rounding term `u·|sum|` folded into the
    /// error. The bias is the branch's (the identity skip contributes none). This is
    /// the core residual operation; stacked/projection blocks and suffix-extraction
    /// integration build on it.
    // Verified in isolation; wired into the resnet suffix path in the next step.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn crown_backward_sound_resident_residual(
        &self,
        branch: &[GpuCrownLayer],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        block_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut cf = self.crown_backward_sound_resident_coeff_seeded(
            branch, lower_a, upper_a, lower_b, upper_b, num_specs, block_dim,
        )?;
        if cf.dim != block_dim {
            // Identity skip requires F: block_dim → block_dim.
            return Err(NyError::shape_mismatch(vec![block_dim], vec![cf.dim]));
        }
        let n = resident_checked_product(&[num_specs, block_dim], "residual coefficient elements")?;
        if lower_a.len() != n || upper_a.len() != n {
            return Err(NyError::shape_mismatch(
                vec![num_specs, block_dim],
                vec![lower_a.len()],
            ));
        }
        for i in 0..n {
            let sum_l = f32_to_f64_exact(cf.lower_a[i]) + f32_to_f64_exact(lower_a[i]);
            let fl_l = sum_l as f32;
            let lower_gap = (f32_to_f64_exact(fl_l) - sum_l).abs();
            cf.lower_err[i] = up_f32(add_nonnegative_f64_up(
                f32_to_f64_exact(cf.lower_err[i]),
                lower_gap,
            ));
            cf.lower_a[i] = fl_l;
            let sum_u = f32_to_f64_exact(cf.upper_a[i]) + f32_to_f64_exact(upper_a[i]);
            let fl_u = sum_u as f32;
            let upper_gap = (f32_to_f64_exact(fl_u) - sum_u).abs();
            cf.upper_err[i] = up_f32(add_nonnegative_f64_up(
                f32_to_f64_exact(cf.upper_err[i]),
                upper_gap,
            ));
            cf.upper_a[i] = fl_u;
        }
        // #u4 WIRED: the branch walk's `taint_rows` already cover BOTH streams
        // of the host merge above. The identity-skip stream is the very seed
        // (`lower_a`/`upper_a`) the branch walk G13-worded at its entry, and
        // the walk ORs those words (plus every transport) into its per-spec
        // rows at exit — so a sentinel-magnitude skip coefficient is already
        // in `cf.taint_rows`, double-covered exactly like the segment-loop
        // seam (G13 per-coefficient + row OR; both only ADD words ⇒ sound, not
        // a launder). The f32-add rounding gaps folded into `*_err` are
        // computed non-negative magnitudes, not sentinels — nothing to word;
        // a host add that overflows to ±inf is caught by the concretize
        // preflight's bit tests (G5), independent of the word channel.
        self.concretize_resident_coeff(&cf, num_specs, input_lower, input_upper)
    }

    /// FINER error-concretization (#unsat-keystone, the deep-resnet error fix):
    /// run ONE branch's backward but split it at each `Activation` (ReLU) boundary,
    /// concretizing the accumulated coefficient ERROR into the (non-amplifying)
    /// scalar bias error against THAT node's abs-max bound — and reset the
    /// coefficient error — between sub-chains. This caps the `|W|`-amplification of
    /// the certified f32 error at every ReLU instead of only at the (coarse) segment
    /// boundary, so the L1 error cannot compound across the deep suffix.
    ///
    /// `node_abs` are the per-Activation pre-node abs-max bounds (`max(|l|,|u|)` per
    /// dim) in the SAME order the branch's Activations are consumed (output→input).
    /// Each entry must match that ReLU's pre-transform coefficient dim. A missing /
    /// mismatched entry simply skips that concretization point (sound — the error is
    /// still carried, just not capped there). The seed coefficient/bias/error and the
    /// per-Activation `relu_pre_lower`/`beta_signed` slices are threaded through
    /// unchanged, so the result is the SAME backward, only with the error periodically
    /// folded into the bias error. SOUND: `|err_a[j]|·max(|z_l[j]|,|z_u[j]|)` over-
    /// approximates coefficient-j's error contribution to the bound (`fab[j] ≥ |x[j]|`,
    /// error terms are non-negative magnitudes), exactly like the per-segment gate.
    #[allow(clippy::too_many_arguments)]
    fn backward_branch_fine(
        &self,
        branch: &[GpuCrownLayer],
        seed: &ResidentCoeff,
        num_specs: usize,
        // #batched-bab: per-domain spec-row count (== num_specs single domain). Threads
        // to the sub-chain backward (per-domain slopes) and the per-node error fold
        // (concretize_error_into_bias, HOLE 4) so each domain block folds against ITS
        // OWN node_abs.
        num_specs_per_dom: usize,
        relu_pre_lower: &[&[f32]],
        beta_signed: &[&[f32]],
        beta_gather_idx: &[&[u32]],
        node_abs: &[&[f32]],
    ) -> Result<ResidentCoeff> {
        // Split the branch (backward order: output→input) into sub-chains delimited
        // by Activation layers: [.. up to & including ReLU_0], [.. up to & including
        // ReLU_1], .., [tail with no ReLU]. Each Activation's PRE-node abs-max bound is
        // node_abs[k] (same order). We run each sub-chain via the proven resident
        // backward (carrying the incoming error), then concretize the error against that
        // ReLU's node bound before the next sub-chain.
        let mut splits: Vec<&[GpuCrownLayer]> = Vec::new();
        let mut start = 0usize;
        for (i, l) in branch.iter().enumerate() {
            if matches!(l, GpuCrownLayer::Activation { .. }) {
                splits.push(&branch[start..=i]);
                start = i + 1;
            }
        }
        if start < branch.len() {
            splits.push(&branch[start..]);
        }
        if splits.is_empty() {
            splits.push(branch);
        }

        let mut coeff = ResidentCoeff {
            lower_a: seed.lower_a.clone(),
            upper_a: seed.upper_a.clone(),
            lower_err: seed.lower_err.clone(),
            upper_err: seed.upper_err.clone(),
            lower_b: seed.lower_b.clone(),
            upper_b: seed.upper_b.clone(),
            lower_b_err: seed.lower_b_err.clone(),
            upper_b_err: seed.upper_b_err.clone(),
            dim: seed.dim,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            // #u4: the fine-split walk starts from the seed's row words — the
            // per-ReLU sub-chains below re-seed from VALUES only, so laundered
            // history must ride the rows (see the seam OR in the loop).
            taint_rows: seed.taint_rows.clone(),
        };
        let mut all_grads: Vec<Vec<f32>> = Vec::new();
        let mut all_gathers: Vec<Vec<f32>> = Vec::new();
        let mut act_idx = 0usize; // index into relu_pre_lower / beta_signed / node_abs
        for sub in &splits {
            let sub_acts = sub
                .iter()
                .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
                .count();
            let pre_slice: Vec<&[f32]> = if relu_pre_lower.is_empty() {
                Vec::new()
            } else {
                let end = (act_idx + sub_acts).min(relu_pre_lower.len());
                relu_pre_lower[act_idx.min(end)..end].to_vec()
            };
            let beta_slice: Vec<&[f32]> = if beta_signed.is_empty() {
                Vec::new()
            } else {
                let end = (act_idx + sub_acts).min(beta_signed.len());
                beta_signed[act_idx.min(end)..end].to_vec()
            };
            let gather_slice: Vec<&[u32]> = if beta_gather_idx.is_empty() {
                Vec::new()
            } else {
                let end = (act_idx + sub_acts).min(beta_gather_idx.len());
                beta_gather_idx[act_idx.min(end)..end].to_vec()
            };
            let mut cf = self.crown_backward_sound_resident_coeff_seeded_err_gather(
                sub,
                &coeff.lower_a,
                &coeff.upper_a,
                &coeff.lower_err,
                &coeff.upper_err,
                &coeff.lower_b,
                &coeff.upper_b,
                &coeff.lower_b_err,
                &coeff.upper_b_err,
                num_specs,
                num_specs_per_dom,
                coeff.dim,
                &pre_slice,
                &beta_slice,
                &gather_slice,
            )?;
            all_grads.append(&mut cf.relu_grads);
            all_gathers.append(&mut cf.beta_gather);
            // #u4 SEAM OR (identical rule to the segment loop): the sub-chain
            // walk G13-covered `coeff`'s VALUES; its laundered row history
            // survives only via this OR. Either side `None` ⇒ `None` (whole-
            // result poisoning, never partial). Gate OFF: `None|None = None`.
            cf.taint_rows = merge_taint_rows(cf.taint_rows.take(), coeff.taint_rows.as_deref());
            coeff = cf;
            // Concretize the error against THIS ReLU's pre-node abs-max bound. The
            // sub-chain ending in a ReLU has its frontier = that ReLU's pre-node; the
            // final (no-ReLU) tail's frontier is the segment input, which the caller
            // concretizes against frontier_abs, so we don't double-fold it here.
            let ends_in_relu = sub
                .last()
                .map(|l| matches!(l, GpuCrownLayer::Activation { .. }))
                .unwrap_or(false);
            if ends_in_relu {
                if let Some(fab) = node_abs.get(act_idx) {
                    // #batched-bab HOLE 4: `fab` is the per-domain-STACKED node abs-max
                    // (`n_domains*coeff.dim`, single domain → coeff.dim); each row folds
                    // against ITS OWN domain block (`dom = s/num_specs_per_dom`).
                    // #u4 ROW-INVARIANT: err→bias-err movement within the same
                    // spec row; the err's taint already sits in the row word
                    // (walk-exit `le`/`ue` OR + G13 seed-err wording), so no
                    // per-coefficient companion call is needed at row
                    // granularity — see the segment-loop analogue.
                    Self::concretize_error_into_bias(&mut coeff, num_specs, num_specs_per_dom, fab);
                }
            }
            act_idx += sub_acts;
        }
        coeff.relu_grads = all_grads;
        coeff.beta_gather = all_gathers;
        Ok(coeff)
    }

    /// Run one branch PART (a contiguous backward-order sub-slice) from a
    /// [`ResidentCoeff`] seed through the fine or plain resident backward —
    /// the shared runner for the C2 cut-fold branch split. An empty part is a
    /// no-op (the seed passes through untouched, no GPU round-trip). The
    /// per-Activation channel slices must already be cut to THIS part's
    /// Activations.
    #[allow(clippy::too_many_arguments)]
    fn backward_branch_part(
        &self,
        part: &[GpuCrownLayer],
        seed: &ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        pre_slice: &[&[f32]],
        beta_slice: &[&[f32]],
        gather_slice: &[&[u32]],
        node_slice: &[&[f32]],
        concretize_fine: bool,
    ) -> Result<ResidentCoeff> {
        if part.is_empty() {
            return Ok(ResidentCoeff {
                lower_a: seed.lower_a.clone(),
                upper_a: seed.upper_a.clone(),
                lower_err: seed.lower_err.clone(),
                upper_err: seed.upper_err.clone(),
                lower_b: seed.lower_b.clone(),
                upper_b: seed.upper_b.clone(),
                lower_b_err: seed.lower_b_err.clone(),
                upper_b_err: seed.upper_b_err.clone(),
                dim: seed.dim,
                relu_grads: Vec::new(),
                beta_gather: Vec::new(),
                // #u4: an empty part is an exact passthrough (no ops run), so
                // the seed's row words pass through verbatim.
                taint_rows: seed.taint_rows.clone(),
            });
        }
        if concretize_fine {
            self.backward_branch_fine(
                part,
                seed,
                num_specs,
                num_specs_per_dom,
                pre_slice,
                beta_slice,
                gather_slice,
                node_slice,
            )
        } else {
            let mut cf = self.crown_backward_sound_resident_coeff_seeded_err_gather(
                part,
                &seed.lower_a,
                &seed.upper_a,
                &seed.lower_err,
                &seed.upper_err,
                &seed.lower_b,
                &seed.upper_b,
                &seed.lower_b_err,
                &seed.upper_b_err,
                num_specs,
                num_specs_per_dom,
                seed.dim,
                pre_slice,
                beta_slice,
                gather_slice,
            )?;
            // #u4 SEAM OR (identical rule to the segment loop): the walk
            // G13-covered the seed's VALUES (and the cut-fold host adds
            // between parts are exact finite validated data — never a taint
            // source; any saturation they produce is re-G13'd right here by
            // the following part's entry seeding); the seed's laundered row
            // history survives only via this OR. Either side `None` ⇒ `None`.
            cf.taint_rows = merge_taint_rows(cf.taint_rows.take(), seed.taint_rows.as_deref());
            Ok(cf)
        }
    }

    /// Retired Cut-CROWN C2 raw fold kernel.
    ///
    /// The proof-bearing caller can no longer reach this function because
    /// `active_resident_cut_fold()` is hard-quarantined before environment or
    /// registry state is read. It remains here for non-authoritative arithmetic
    /// tests and for a future provenance-bound replacement.
    ///
    /// The branch is split at that Activation on the HOST — the resident
    /// backward already round-trips the coefficient frontier between segments
    /// and (in fine mode) between per-ReLU sub-chains, and the split is
    /// bit-transparent: every per-layer GPU op depends only on the current f32
    /// buffer contents, which a download/re-upload preserves exactly. Between
    /// the two parts the frontier coefficient is over the target ReLU's
    /// POST-activation (= PRE-transform for its relaxation), which is exactly
    /// where `λ·cc` must be added on the LOWER side (`λ·cc_i` multiplies
    /// `relu(ẑ_i)` itself, unlike the post-transform `beta_signed`); the
    /// `−Σ λ_j·B_j` constant joins the lower bias at the same point. The upper
    /// side is untouched (a `+λ·g` fold is only valid for lower bounds).
    ///
    /// SOUND for any λ ≥ 0 with valid cut bounds B (Lean
    /// `cuts_fold_lower_bound`). Before any branch split or mutation, the complete
    /// post/bias/pre entry is validated for finite values and target-activation
    /// indices. Any malformed entry refuses the WHOLE fold and runs the untouched
    /// branch — a partial Lagrangian is never applied.
    /// Rounding is experiment-grade (see `cut_fold_resident` module docs).
    #[allow(clippy::too_many_arguments)]
    fn backward_branch_cut_fold(
        &self,
        branch: &[GpuCrownLayer],
        seed: &ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        pre_slice: &[&[f32]],
        beta_slice: &[&[f32]],
        gather_slice: &[&[u32]],
        node_slice: &[&[f32]],
        concretize_fine: bool,
        local_act_idx: usize,
        fold: &super::cut_fold_resident::ResidentCutFold,
    ) -> Result<ResidentCoeff> {
        // Locate the local_act_idx-th Activation layer within the branch.
        let pos = branch
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, GpuCrownLayer::Activation { .. }))
            .nth(local_act_idx)
            .map(|(i, _)| i);
        let Some(pos) = pos else {
            debug_assert!(
                false,
                "resident cut fold: local activation index {local_act_idx} out of range"
            );
            return self.backward_branch_part(
                branch,
                seed,
                num_specs,
                num_specs_per_dom,
                pre_slice,
                beta_slice,
                gather_slice,
                node_slice,
                concretize_fine,
            );
        };
        let target_num_neurons = match &branch[pos] {
            GpuCrownLayer::Activation { num_neurons, .. } => *num_neurons,
            _ => unreachable!("resident cut-fold target was selected as an Activation"),
        };
        // Validate ALL three pieces before even splitting the branch. In
        // particular, a bad post entry must not leave the pre channel live, and a
        // bad pre entry must not be discovered after post+bias already mutated the
        // lower objective.
        if !resident_cut_fold_valid_for_activation(fold, target_num_neurons) {
            return self.backward_branch_part(
                branch,
                seed,
                num_specs,
                num_specs_per_dom,
                pre_slice,
                beta_slice,
                gather_slice,
                node_slice,
                concretize_fine,
            );
        }
        let (part1, part2) = branch.split_at(pos);
        // Per-Activation channel slices for each part (part1 holds the first
        // `local_act_idx` Activations; the target Activation starts part2).
        // Empty channels stay empty for both parts (the "not captured" state).
        fn split_chan<'x, T: ?Sized>(s: &[&'x T], k: usize) -> (Vec<&'x T>, Vec<&'x T>) {
            if s.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let k = k.min(s.len());
                (s[..k].to_vec(), s[k..].to_vec())
            }
        }
        let (pre1, pre2) = split_chan(pre_slice, local_act_idx);
        let (beta1, beta2) = split_chan(beta_slice, local_act_idx);
        let (gat1, gat2) = split_chan(gather_slice, local_act_idx);
        let (node1, node2) = split_chan(node_slice, local_act_idx);

        let mut c1 = self.backward_branch_part(
            part1,
            seed,
            num_specs,
            num_specs_per_dom,
            &pre1,
            &beta1,
            &gat1,
            &node1,
            concretize_fine,
        )?;
        // The target Activation metadata and the realized incoming frontier must
        // agree before any fold term is applied. A structural mismatch degrades to
        // the untouched branch, never to a partially folded objective.
        if c1.dim != target_num_neurons {
            return self.backward_branch_part(
                branch,
                seed,
                num_specs,
                num_specs_per_dom,
                pre_slice,
                beta_slice,
                gather_slice,
                node_slice,
                concretize_fine,
            );
        }

        // C2b capture (dark `NY_CUT_FOLD_CAPTURE` gate): copy the incoming
        // LOWER-side frontier coefficients over the target ReLU's
        // post-activation BEFORE the fold add — the objective-signed group
        // selection reads these `A` rows host-side. Read-only: bounds are
        // untouched.
        if super::cut_fold_resident::resident_cut_fold_capture_enabled() {
            super::cut_fold_resident::store_resident_cut_fold_capture(
                super::cut_fold_resident::ResidentCutFoldCapture {
                    num_specs,
                    dim: c1.dim,
                    lower_a: c1.lower_a.clone(),
                },
            );
        }

        // The POST-activation fold: `+λ·cc` on the ReLU-OUTPUT frontier +
        // `−Σλ·B` on the lower bias, every spec row, BEFORE the target
        // Activation transform. `sound_round` selects the production
        // outward-rounded fold (stem lever) vs the legacy plain-f32 add
        // (byte-identical to the `NY_CUT_FOLD_RESIDENT` experiment path).
        let d = c1.dim;
        for s in 0..num_specs {
            let base = s * d;
            if fold.sound_round {
                for &(i, c) in &fold.coeffs {
                    fold_add_lower_coeff_outward(
                        &mut c1.lower_a,
                        &mut c1.lower_err,
                        base + i as usize,
                        c,
                    );
                }
                fold_add_lower_bias_outward(
                    &mut c1.lower_b[s],
                    &mut c1.lower_b_err[s],
                    fold.bias_shift,
                );
            } else {
                for &(i, c) in &fold.coeffs {
                    c1.lower_a[base + i as usize] += c;
                }
                c1.lower_b[s] += fold.bias_shift;
            }
        }
        super::cut_fold_resident::note_resident_cut_fold_applied();

        // PRE-activation fold: `+β·a_i` on the ReLU-INPUT frontier (POST the
        // target Activation transform). When `pre_coeffs` is empty we run
        // `part2` as ONE part — byte-identical to the legacy fold site (no
        // extra sub-split). Otherwise split `part2` = [target Activation] +
        // rest: transform through the Activation to reach the ReLU-input
        // frontier, add `+β·a_i` (same outward discipline), then continue to
        // the network input.
        let mut c2 = if fold.pre_coeffs.is_empty() {
            self.backward_branch_part(
                part2,
                &c1,
                num_specs,
                num_specs_per_dom,
                &pre2,
                &beta2,
                &gat2,
                &node2,
                concretize_fine,
            )?
        } else {
            // `part2[0]` is the target Activation (`pos` located it); split it off.
            let (act_part, rest_part) = part2.split_at(1);
            let (apre, rpre) = split_chan(&pre2, 1);
            let (abeta, rbeta) = split_chan(&beta2, 1);
            let (agat, rgat) = split_chan(&gat2, 1);
            let (anode, rnode) = split_chan(&node2, 1);
            let mut c1p = self.backward_branch_part(
                act_part,
                &c1,
                num_specs,
                num_specs_per_dom,
                &apre,
                &abeta,
                &agat,
                &anode,
                concretize_fine,
            )?;
            let dp = c1p.dim;
            debug_assert_eq!(dp, target_num_neurons);
            for s in 0..num_specs {
                let base = s * dp;
                for &(i, c) in &fold.pre_coeffs {
                    if fold.sound_round {
                        fold_add_lower_coeff_outward(
                            &mut c1p.lower_a,
                            &mut c1p.lower_err,
                            base + i as usize,
                            c,
                        );
                    } else {
                        c1p.lower_a[base + i as usize] += c;
                    }
                }
            }
            let mut cr = self.backward_branch_part(
                rest_part,
                &c1p,
                num_specs,
                num_specs_per_dom,
                &rpre,
                &rbeta,
                &rgat,
                &rnode,
                concretize_fine,
            )?;
            // Stitch the sub-split capture channels: act_part then rest.
            let mut g = std::mem::take(&mut c1p.relu_grads);
            g.append(&mut cr.relu_grads);
            cr.relu_grads = g;
            let mut gg = std::mem::take(&mut c1p.beta_gather);
            gg.append(&mut cr.beta_gather);
            cr.beta_gather = gg;
            cr
        };
        // Stitch the capture channels back into branch order (part1 then part2).
        let mut grads = std::mem::take(&mut c1.relu_grads);
        grads.append(&mut c2.relu_grads);
        c2.relu_grads = grads;
        let mut gathers = std::mem::take(&mut c1.beta_gather);
        gathers.append(&mut c2.beta_gather);
        c2.beta_gather = gathers;
        Ok(c2)
    }

    /// Carrier-driven resident Cut-CROWN branch split (the observation-only
    /// SHADOW; `ops/cut_shadow_resident.rs` is the driver and audit home).
    ///
    /// Mirrors [`Self::backward_branch_cut_fold`]'s geometry — part1 → POST
    /// channels → [target Activation] → PRE channels + per-row bias → rest,
    /// matching the CUDA resident cut fold's application order — but consumes a
    /// call-local per-row [`super::cut_shadow_resident::CutApplySnapshot`]
    /// (built from one validated `ResidentLowerCutCarrier`) instead of the
    /// hard-quarantined registry entry, applies every channel on the charged
    /// device through the audited cut-apply kernel (source error + resident
    /// mutation rounding + flush cover charged into the LOWER error lanes), and
    /// REFUSES (typed) on any structural mismatch instead of degrading to the
    /// untouched branch: a silently unapplied cut would let the shadow driver
    /// mislabel `shadow == baseline` as a measured Δ=0.
    ///
    /// #u4 taint note (same argument as the legacy cut-fold seam): the apply
    /// kernel writes finite validated values that the next part's G13 entry
    /// seeding re-words; error-lane widening is refusal-only mass and cannot
    /// launder a row word.
    #[allow(clippy::too_many_arguments)]
    fn backward_branch_carrier_cut(
        &self,
        branch: &[GpuCrownLayer],
        seed: &ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        pre_slice: &[&[f32]],
        beta_slice: &[&[f32]],
        gather_slice: &[&[u32]],
        node_slice: &[&[f32]],
        concretize_fine: bool,
        local_act_idx: usize,
        snapshot: &super::cut_shadow_resident::CutApplySnapshot,
    ) -> Result<ResidentCoeff> {
        use super::cut_shadow_resident::CutChannelKind;
        // The shadow driver is a serial single-domain entry; a wide fold would
        // apply one row's channel to another domain's block.
        if num_specs_per_dom != num_specs || snapshot.rows().len() != num_specs {
            return Err(NyError::SoundnessRefusal(
                "wgpu resident cut shadow: carrier rows do not match the serial fold".into(),
            ));
        }
        let pos = branch
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, GpuCrownLayer::Activation { .. }))
            .nth(local_act_idx)
            .map(|(i, _)| i)
            .ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "wgpu resident cut shadow: target activation index is outside its branch"
                        .into(),
                )
            })?;
        let target_num_neurons = match &branch[pos] {
            GpuCrownLayer::Activation { num_neurons, .. } => *num_neurons,
            _ => unreachable!("resident cut-shadow target was selected as an Activation"),
        };
        if target_num_neurons != snapshot.target_width() {
            return Err(NyError::SoundnessRefusal(
                "wgpu resident cut shadow: target width does not match the resident activation"
                    .into(),
            ));
        }
        let (part1, part2) = branch.split_at(pos);
        fn split_chan<'x, T: ?Sized>(s: &[&'x T], k: usize) -> (Vec<&'x T>, Vec<&'x T>) {
            if s.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let k = k.min(s.len());
                (s[..k].to_vec(), s[k..].to_vec())
            }
        }
        let (pre1, pre2) = split_chan(pre_slice, local_act_idx);
        let (beta1, beta2) = split_chan(beta_slice, local_act_idx);
        let (gat1, gat2) = split_chan(gather_slice, local_act_idx);
        let (node1, node2) = split_chan(node_slice, local_act_idx);

        let mut c1 = self.backward_branch_part(
            part1,
            seed,
            num_specs,
            num_specs_per_dom,
            &pre1,
            &beta1,
            &gat1,
            &node1,
            concretize_fine,
        )?;
        if c1.dim != target_num_neurons {
            return Err(NyError::SoundnessRefusal(
                "wgpu resident cut shadow: realized frontier does not reach the target width"
                    .into(),
            ));
        }
        if self.crown_backward_deadline_expired() {
            return Err(NyError::DeadlineExceeded(
                "wgpu resident cut shadow: deadline expired before the post-channel apply".into(),
            ));
        }
        // POST channels participate in sign selection, intercept, and slope
        // composition, so they must land BEFORE the target relaxation.
        self.resident_cut_apply_lower_pair_columns(
            &mut c1.lower_a,
            &mut c1.lower_err,
            target_num_neurons,
            num_specs,
            snapshot,
            CutChannelKind::Post,
        )?;

        // `part2[0]` is the target Activation (`pos` located it); run it alone
        // so the PRE channels and the per-row bias land on the ReLU-INPUT
        // frontier, bypassing the relaxation — exactly the CUDA fold order.
        let (act_part, rest_part) = part2.split_at(1);
        let (apre, rpre) = split_chan(&pre2, 1);
        let (abeta, rbeta) = split_chan(&beta2, 1);
        let (agat, rgat) = split_chan(&gat2, 1);
        let (anode, rnode) = split_chan(&node2, 1);
        let mut c1p = self.backward_branch_part(
            act_part,
            &c1,
            num_specs,
            num_specs_per_dom,
            &apre,
            &abeta,
            &agat,
            &anode,
            concretize_fine,
        )?;
        if c1p.dim != target_num_neurons {
            return Err(NyError::SoundnessRefusal(
                "wgpu resident cut shadow: target relaxation changed the frontier width".into(),
            ));
        }
        if self.crown_backward_deadline_expired() {
            return Err(NyError::DeadlineExceeded(
                "wgpu resident cut shadow: deadline expired before the pre-channel apply".into(),
            ));
        }
        self.resident_cut_apply_lower_pair_columns(
            &mut c1p.lower_a,
            &mut c1p.lower_err,
            target_num_neurons,
            num_specs,
            snapshot,
            CutChannelKind::Pre,
        )?;
        // The bias belongs to the same Lagrangian row and is charged once here.
        self.resident_cut_apply_lower_bias(&mut c1p.lower_b, &mut c1p.lower_b_err, snapshot)?;

        let mut cr = self.backward_branch_part(
            rest_part,
            &c1p,
            num_specs,
            num_specs_per_dom,
            &rpre,
            &rbeta,
            &rgat,
            &rnode,
            concretize_fine,
        )?;
        // Stitch capture channels back into branch order: part1, act, rest.
        let mut grads = std::mem::take(&mut c1.relu_grads);
        grads.append(&mut c1p.relu_grads);
        grads.append(&mut cr.relu_grads);
        cr.relu_grads = grads;
        let mut gathers = std::mem::take(&mut c1.beta_gather);
        gathers.append(&mut c1p.beta_gather);
        gathers.append(&mut cr.beta_gather);
        cr.beta_gather = gathers;
        Ok(cr)
    }

    /// Fold the accumulated per-coefficient error `(lower_err,upper_err)` into the
    /// scalar bias error `(lower_b_err,upper_b_err)` against the node abs-max bound
    /// `fab` (`fab[j] = max(|z_l[j]|,|z_u[j]|) ≥ |z[j]|`), then RESET the coefficient
    /// error to 0. This is the per-node analogue of the per-segment fold in the resnet
    /// loop. SOUND over-approximation (non-negative magnitudes × an upper bound on
    /// `|z[j]|`, certified-add rounded up). No-op if `fab` doesn't match the dim.
    ///
    /// #batched-bab HOLE 4: with `n_domains = num_specs/num_specs_per_dom > 1`, `fab` is
    /// the per-domain-STACKED node abs-max (`n_domains*d`, `d = coeff.dim`), laid out as
    /// `n_domains` contiguous blocks of `d`. Each spec row `s` folds against ITS OWN
    /// domain block `dom = s/num_specs_per_dom` at `fab[dom*d + j]`. Sharing one domain's
    /// (possibly smaller) abs-max across another domain's rows would UNDER-count the
    /// error ⇒ a tighter bound ⇒ a false VERIFIED. Single domain (`num_specs_per_dom ==
    /// num_specs`) ⇒ `dom == 0`, `fab.len() == d` ⇒ byte-identical.
    fn concretize_error_into_bias(
        coeff: &mut ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        fab: &[f32],
    ) {
        let d = coeff.dim;
        let Some(n_domains) = num_specs
            .checked_div(num_specs_per_dom)
            .filter(|_| num_specs.is_multiple_of(num_specs_per_dom))
        else {
            return;
        };
        let Some(expected_fab) = d.checked_mul(n_domains) else {
            return;
        };
        let Some(coeff_elems) = num_specs.checked_mul(d) else {
            return;
        };
        if fab.len() != expected_fab
            || coeff.lower_err.len() != coeff_elems
            || coeff.upper_err.len() != coeff_elems
            || coeff.lower_b_err.len() != num_specs
            || coeff.upper_b_err.len() != num_specs
            || fab.iter().any(|value| {
                let bits = value.to_bits();
                bits & 0x7f80_0000 == 0x7f80_0000
                    || (bits & 0x8000_0000 != 0 && bits & 0x7fff_ffff != 0)
            })
        {
            return;
        }
        for s in 0..num_specs {
            let dom = s / num_specs_per_dom;
            let fbase = dom * d;
            let row = s * d;
            let mut le = 0.0f64;
            let mut ue = 0.0f64;
            for j in 0..d {
                let b = f32_to_f64_exact(fab[fbase + j]);
                let lower_term = f32_to_f64_exact(coeff.lower_err[row + j]) * b;
                let upper_term = f32_to_f64_exact(coeff.upper_err[row + j]) * b;
                le = add_nonnegative_f64_up(le, lower_term);
                ue = add_nonnegative_f64_up(ue, upper_term);
                coeff.lower_err[row + j] = 0.0;
                coeff.upper_err[row + j] = 0.0;
            }
            coeff.lower_b_err[s] = up_f32(add_nonnegative_f64_up(
                f32_to_f64_exact(coeff.lower_b_err[s]),
                le,
            ));
            coeff.upper_b_err[s] = up_f32(add_nonnegative_f64_up(
                f32_to_f64_exact(coeff.upper_b_err[s]),
                ue,
            ));
        }
    }

    /// #u4 taint companion of [`Self::concretize_error_into_bias`]
    /// (TAINT_GUARD_AUDIT.md §4 C1, "plumbed from"): OR the per-coefficient
    /// ERR-taint words into the per-spec BIAS-taint words, one channel (lower
    /// or upper) per call, to be invoked BEFORE the real fold zeroes the
    /// per-coefficient err it mirrors.
    ///
    /// Canon rule: `taint_out = OR over inputs of (taint_in AND its
    /// multiplicative partner != 0)`. Each err element's multiplicative partner
    /// here is `fab[dom*d + j]` (the per-domain node abs-max the real fold
    /// charges the err against), so `fab == 0.0` — either sign of zero —
    /// annihilates: a zero abs-max is a PROVEN exactly-zero pre-activation
    /// (`|z[j]| ≤ 0`), the one case where dropping the taint is sound
    /// (`R·0 == 0` for every finite real the sentinel stands for). The fold's
    /// own saturation term is absent by construction: it accumulates in f64
    /// (which never clamps to the finite sentinel) and any non-finite escape is
    /// refused by the concretize host preflight bit tests
    /// (crown_concretize_sound.rs, guard G5).
    ///
    /// Validation and indexing mirror the real fold EXACTLY (same
    /// `n_domains` partition and `fab` shape checks, same fab
    /// non-finite/negative bit tests, same `fab[dom*d + j]` per-domain block
    /// addressing — HOLE 4): whenever the real fold no-ops, this companion
    /// no-ops too, leaving the err-taint words paired with the (unzeroed)
    /// per-coefficient err they describe — nothing is lost. After the real fold
    /// zeroes the err channel, its taint words are stale-conservative at worst
    /// (a kept word on an exact 0.0 can only cause refusal, never a tighter
    /// bound).
    ///
    /// The ordinary resident walk and ResNet composition now carry row words
    /// without invoking this older CPU companion. It remains as a focused
    /// reference for the fold semantics; armed C1 receives the composed rows.
    #[allow(dead_code)]
    fn concretize_error_taint_into_bias(
        err_taint: &[u32],
        fab: &[f32],
        per_spec_bias_taint: &mut [u32],
        num_specs: usize,
        num_specs_per_dom: usize,
        dim: usize,
    ) {
        let d = dim;
        let Some(n_domains) = num_specs
            .checked_div(num_specs_per_dom)
            .filter(|_| num_specs.is_multiple_of(num_specs_per_dom))
        else {
            return;
        };
        let Some(expected_fab) = d.checked_mul(n_domains) else {
            return;
        };
        let Some(coeff_elems) = num_specs.checked_mul(d) else {
            return;
        };
        if fab.len() != expected_fab
            || err_taint.len() != coeff_elems
            || per_spec_bias_taint.len() != num_specs
            || fab.iter().any(|value| {
                let bits = value.to_bits();
                bits & 0x7f80_0000 == 0x7f80_0000
                    || (bits & 0x8000_0000 != 0 && bits & 0x7fff_ffff != 0)
            })
        {
            return;
        }
        for s in 0..num_specs {
            let dom = s / num_specs_per_dom;
            let fbase = dom * d;
            let row = s * d;
            let mut word = 0u32;
            for j in 0..d {
                // Annihilation conjunct. `-0.0 != 0.0` is false, so a negative
                // zero abs-max annihilates too (|z[j]| ≤ 0 ⇒ z[j] exactly 0).
                if fab[fbase + j] != 0.0 {
                    word |= err_taint[row + j];
                }
            }
            per_spec_bias_taint[s] |= word;
        }
    }

    /// #u4: a tiny one-shot UNIFORM buffer written via `mapped_at_creation` —
    /// deliberately NOT `queue.write_buffer` (submission-ordered: many
    /// transport dispatches share one per-layer encoder/submit, and reusing a
    /// written uniform would collapse every dispatch's params to the last
    /// value). Each on-device taint dispatch owns its params buffer; the bind
    /// group keeps it alive past this scope.
    fn taint_uniform(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
        buf.unmap();
        super::intermediate_sweep::note_host_to_device(bytes.len());
        buf
    }

    /// #u4 (gate-ON only): encode ONE `TAINT_ROW_OR_SHADER` dispatch that ORs
    /// a `[rows × cols]` word buffer down to the per-spec-row device
    /// accumulator `rows_out` (atomicOr — monotone, so transport order is
    /// free). This replaced every mid-walk `taint_read_words` + host-fold
    /// round trip (the measured 2.3–3.1× gate-ON tax).
    ///
    /// `partner`:
    /// * `None` — unconditional row-OR (the fail-closed no-twin form; the
    ///   shader's partner binding is filled with `words` itself, never read);
    /// * `Some((buf, false))` — per-COLUMN annihilation partner `buf[k]`,
    ///   `len ≥ cols` (the `bias[k] != 0` / intercept conjuncts);
    /// * `Some((buf, true))` — per-ELEMENT partner `buf[i]`, same shape as
    ///   `words` (no walk site yet; the shader supports it).
    ///
    /// The extra bias-error dispatch that charges a layer's declared
    /// `bias_abs_err` (`d`). No-op when `d == 0`, which is every pre-`cert_err`
    /// caller — so the default walk issues no dispatch and stays byte-identical.
    ///
    /// See [`cert_bias_charge_slack`] for the derivation: re-running the
    /// UNMODIFIED bias kernel with the constant operand `[d; k]` and
    /// `gamma_k = 1` computes `d·(Σ|a_j| + Σ err_j)` recovered outward, and the
    /// kernel accumulates it (`+=`) into the very `bias_err_out` the ordinary
    /// fold just wrote. The centre-bias output is aimed at `sink`, a buffer
    /// nothing else reads, so `blo`/`buo` never move.
    fn cert_bias_charge_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bias_pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
        args: CertBiasChargeArgs<'_>,
    ) -> Result<()> {
        let d = args.cert_err.bias_abs_err;
        if d == 0.0 {
            return Ok(());
        }
        let (Some(params), Some(operand), Some(sink)) = (args.params, args.operand, args.sink)
        else {
            return Err(NyError::UnsupportedOp(format!(
                "#cert-err: layer {} declares bias_abs_err={d:e} but this walk \
                 allocated no charge buffers — refusing (fail-closed)",
                args.layer_index
            )));
        };
        if !d.is_finite() || d < 0.0 {
            return Err(NyError::UnsupportedOp(format!(
                "#cert-err: layer {} declares a non-finite/negative \
                 bias_abs_err={d:e} — refusing (fail-closed)",
                args.layer_index
            )));
        }
        let k = args.reduction;
        let k_u32 = resident_checked_u32(k, "cert bias charge reduction")?;
        let num_specs_u32 = resident_checked_u32(args.num_specs, "cert bias charge spec rows")?;
        // #flush-charge belt-and-braces: charged authority refuses cert_err
        // layers at walk entry, so this is unreachable there — but if that
        // guard ever regressed, the bias-combine widening still applies.
        let slack = charged_bias_slack_or(
            self.charged_flush_authority_cached(),
            cert_bias_charge_slack(k)?,
        )?;
        let additive = crate::wgpu_device::sound_consts::rung3_flush_safe_additive(k_u32)?;
        self.queue
            .write_buffer(operand, 0, bytemuck::cast_slice(&vec![d; k.max(1)]));
        self.queue.write_buffer(
            params,
            0,
            bytemuck::bytes_of(&BiasParams {
                num_specs: num_specs_u32,
                k: k_u32,
                // `gamma_k = 1` turns the kernel's `gamma_k·Σ|a·bias|` rounding
                // correction into the FULL first-order `d·Σ|a|` charge.
                gamma_k: 1.0,
                additive,
                slack,
                // `eft_mode = 0` is mandatory, not a default: the EFT lane
                // REPLACES the `gamma_k·Σ|a·bias|` term with a measured residual
                // of the value reduction, which would delete this charge
                // entirely. The walk preflight already refuses `cert_err` under
                // `NY_EFT_ERR`; this is the second, local guarantee.
                eft_mode: 0,
                eft_r_slack: 0.0,
                _p: 0,
            }),
        );
        super::intermediate_sweep::note_host_to_device(
            k.max(1)
                .saturating_mul(size_of::<f32>())
                .saturating_add(size_of::<BiasParams>()),
        );
        for side in 0..2 {
            self.pass_simple(
                encoder,
                bias_pipe,
                params,
                &[
                    args.a[side],
                    args.a_err[side],
                    operand,
                    sink,
                    args.bias_err_out[side],
                ],
                num_specs_u32,
            );
        }
        Ok(())
    }

    fn taint_row_or_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        words: &wgpu::Buffer,
        partner: Option<(&wgpu::Buffer, bool)>,
        rows: usize,
        cols: usize,
        rows_out: &wgpu::Buffer,
    ) {
        let cols = cols.max(1);
        let (mode, pbuf) = match partner {
            None => (0u32, words),
            Some((buf, false)) => (1u32, buf),
            Some((buf, true)) => (2u32, buf),
        };
        let params = self.taint_uniform(
            "res_taint_row_or_p",
            bytemuck::bytes_of(&TaintRowOrParams {
                rows: rows as u32,
                cols: cols as u32,
                use_partner: mode,
                _pad: 0,
            }),
        );
        let pipes = self.resident_backward_pipelines();
        self.pass_simple(
            encoder,
            &pipes.taint_row_or,
            &params,
            &[words, pbuf, rows_out],
            ((rows * cols) as u32).div_ceil(256),
        );
    }

    /// Sound resident backward over a RESNET decomposed into backward-order
    /// `segments` (plain chains + identity-skip residual blocks). Folds the
    /// coefficient frontier through each segment, carrying its certified error so
    /// stacked blocks compose soundly; at a residual block the coefficient forks
    /// (branch backward + identity skip) and merges via `add_skip_stream`. Each
    /// segment's internal layers run GPU-resident; only the coefficient crosses
    /// segment boundaries. This is the resnet form the cifar100/tinyimagenet
    /// suffix path needs.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn crown_backward_sound_resident_resnet(
        &self,
        segments: &[ResnetSegment],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Non-seeded entry: the spec C is exact and symmetric, bias 0.
        let zb = vec![0.0f32; num_specs];
        let (lo, hi, _grads) = self.crown_backward_sound_resident_resnet_seeded(
            segments,
            spec,
            spec,
            &zb,
            &zb,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            &[],
            &[],
            &[],
            false,
            &[],
            false,
        )?;
        Ok((lo, hi))
    }

    /// Seeded form of [`crown_backward_sound_resident_resnet`]: fold an ASYMMETRIC
    /// frontier (`lower_a`/`upper_a` coefficients + `lower_b`/`upper_b` bias) through
    /// the resnet segments, as the graph alpha-CROWN suffix path does. The frontier
    /// is treated as EXACT (incoming error 0), matching the CPU sound suffix path and
    /// the unary [`crown_backward_sound_resident_seeded`]; only the suffix's own f32
    /// rounding is tracked with directed/over-bounded error, so the result is a sound
    /// enclosure. This is what the cifar100/tinyimagenet resnet verdict suffix uses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_resnet_seeded(
        &self,
        segments: &[ResnetSegment],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        relu_pre_lower: &[&[f32]],
        beta_signed: &[&[f32]],
        frontier_abs: &[&[f32]],
        force_concretize: bool,
        node_abs: &[&[f32]],
        force_fine: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
        let (lo, hi, grads, _gathers) = self.crown_backward_sound_resident_resnet_seeded_gather(
            segments,
            lower_a,
            upper_a,
            lower_b,
            upper_b,
            num_specs,
            num_specs, // #batched-bab: single-domain caller (per-dom == total).
            output_dim,
            input_lower,
            input_upper,
            relu_pre_lower,
            beta_signed,
            &[],
            frontier_abs,
            force_concretize,
            node_abs,
            force_fine,
            None,
            None,
        )?;
        Ok((lo, hi, grads))
    }

    /// Gather-capable form of [`crown_backward_sound_resident_resnet_seeded`]
    /// (#w4-split-tightening): identical bound computation, plus the per-ReLU
    /// A-value GATHER channel for the analytic β gradient, returned 4th in fold
    /// order (aligned with `beta_signed` / `relu_pre_lower` indexing).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_resident_resnet_seeded_gather(
        &self,
        segments: &[ResnetSegment],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        // #batched-bab: per-domain spec-row count. `num_specs` is the TOTAL stacked-row
        // count `N = n_domains * num_specs_per_dom`. Per-domain state (Activation slopes/
        // intercepts/β in the segments, the frontier_abs/node_abs fab tables, the input
        // box) is stacked in `n_domains` blocks; each row folds against ITS OWN block
        // (`dom = row/num_specs_per_dom`). Single domain (`== num_specs`) → byte-identical.
        num_specs_per_dom: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        // Gradient-capable warmup: per-ReLU masked pre-activation lower bounds, flat
        // in fold order (each branch consumes its Activations in order, F before P for
        // a projection block). Empty ⇒ no capture (verdict path). Captured gradients
        // are accumulated ACROSS segments (the per-segment `coeff` is replaced on each
        // merge, so its `relu_grads` would otherwise be lost) and returned 3rd.
        relu_pre_lower: &[&[f32]],
        // Beta-capable per-domain (#unsat-keystone step 4): per-ReLU signed beta (β·sign),
        // flat in the SAME fold order as `relu_pre_lower`. Empty ⇒ no beta. Sliced per
        // branch and folded into each segment's post-slope coefficient.
        beta_signed: &[&[f32]],
        // Beta-GRADIENT gather (#w4-split-tightening): per-ReLU neuron column indices
        // whose pre-transform lower A-values are read back, flat in the SAME fold order
        // as `beta_signed`. Empty ⇒ no capture (byte-for-byte unchanged bounds).
        beta_gather_idx: &[&[u32]],
        // #unsat-keystone error-concretization: per-segment frontier (input-side) node
        // abs-max bounds (max(|l|,|u|) per dim), SAME order as `segments`. When non-empty
        // AND NY_RESNET_ERR_CONCRETIZE=1, after each segment the accumulated coefficient
        // ERROR is concretized against the frontier bounds into the (scalar) bias error
        // and the coefficient error is reset — capping the L1 error blow-up through the
        // deep resnet (the certified f32 error otherwise grows ~|W| per layer with no
        // cancellation while the coefficient cancels). SOUND: |err_a[j]|·max(|z_l[j]|,
        // |z_u[j]|) over-approximates coefficient-j's error contribution to the bound.
        // Empty ⇒ byte-identical to the pre-concretization path (verdict default).
        frontier_abs: &[&[f32]],
        // When true, force the frontier_abs error-concretization ON regardless of the
        // NY_RESNET_ERR_CONCRETIZE env gate. Used by the main-bound auto-fallback: when the
        // un-concretized bound came back non-finite (the L1 blow-up overflowed f32), the
        // caller re-runs with this set to recover a finite, sound, capped bound. Default
        // false preserves the env-gated behaviour for every other caller.
        force_concretize: bool,
        // #unsat-keystone FINER error-concretization: per-Activation pre-node abs-max
        // bounds (max(|l|,|u|) per dim) in FOLD order (same order as relu_pre_lower /
        // beta_signed — each branch consumes its ReLUs output→input, F before P). When
        // non-empty AND (force_fine OR NY_RESNET_ERR_CONCRETIZE_FINE=1), each branch's
        // backward is split at every ReLU and the accumulated coefficient error is
        // concretized against that ReLU's node bound (then reset) — capping the |W|-
        // amplification of the certified f32 error at EVERY ReLU instead of only at the
        // (coarse) per-segment boundary. SOUND (over-approximates, like the segment gate);
        // empty ⇒ byte-identical to the per-segment / pre-concretization path.
        node_abs: &[&[f32]],
        force_fine: bool,
        // #batched-vjp: write-only side channel — when Some, receives the FOLDED
        // input-level LOWER coefficient rows (num_specs x input_dim, row-major)
        // right before concretization. For a mask-slope (point-VJP) fold these
        // rows ARE the exact per-row gradients d(spec_row . output)/d(input).
        // None (every existing caller) => byte-for-byte unchanged.
        input_coeff_out: Option<&mut Vec<f32>>,
        // #clip-interm-resnet-batched write-only side channel: when Some, receives the
        // FULL downloaded coefficient frontier (all 8 vecs + dim) captured at the SAME
        // point as `input_coeff_out`, right before concretization. Used only by the dark
        // clip lane's coeff-capture wide entry; None (every other caller) => no capture,
        // byte-for-byte unchanged. `num_specs_per_dom` is set by the caller.
        coeff_full_out: Option<&mut ny_core::GpuResidentCoeffBatched>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let (coeff, all_grads, all_gathers) = self.resnet_seeded_compose_coeff(
            segments,
            lower_a,
            upper_a,
            lower_b,
            upper_b,
            num_specs,
            num_specs_per_dom,
            output_dim,
            relu_pre_lower,
            beta_signed,
            beta_gather_idx,
            frontier_abs,
            force_concretize,
            node_abs,
            force_fine,
        )?;
        if let Some(out) = input_coeff_out {
            out.clear();
            out.extend_from_slice(&coeff.lower_a);
        }
        if let Some(out) = coeff_full_out {
            out.lower_a = coeff.lower_a.clone();
            out.upper_a = coeff.upper_a.clone();
            out.lower_err = coeff.lower_err.clone();
            out.upper_err = coeff.upper_err.clone();
            out.lower_b = coeff.lower_b.clone();
            out.upper_b = coeff.upper_b.clone();
            out.lower_b_err = coeff.lower_b_err.clone();
            out.upper_b_err = coeff.upper_b_err.clone();
            out.dim = coeff.dim;
            out.num_specs = num_specs;
            out.num_specs_per_dom = num_specs_per_dom;
        }
        // #u4 (previously: forced `taint_rows = None` here): the segment
        // composition now carries the word channel end-to-end (see
        // `resnet_seeded_compose_coeff`), so `coeff.taint_rows` is handed to
        // the C1 consult as-is. Gate ON on admitted host per-segment Linear /
        // Activation/Conv paths ⇒ `Some(rows)`; gate OFF ⇒ `None`, which armed
        // C1 refuses. Segment-resident device streams and coalesced folds
        // likewise refuse before reaching here until their own word seams arm.
        // A `None` at this boundary therefore still means "no words carried"
        // — the honest fail-closed value at the armed consult.
        let (lo, hi) = self.concretize_resident_coeff_batched(
            &coeff,
            num_specs,
            num_specs_per_dom,
            input_lower,
            input_upper,
        )?;
        Ok((lo, hi, all_grads, all_gathers))
    }

    /// The resnet segment-composition loop of
    /// [`Self::crown_backward_sound_resident_resnet_seeded_gather`], returning
    /// the COMPOSED coefficient frontier (pre-concretize) plus the accumulated
    /// per-ReLU gradient/gather channels. Split out so the #u4 word channel is
    /// observable/testable at the exact frontier the C1 consult will read:
    /// `ResidentCoeff::taint_rows` here is the per-spec-row OR over every
    /// sub-walk, skip add, projection merge, re-seed seam and host fold of the
    /// whole segment walk (`None` iff the gate is off or a seam genuinely
    /// carried no words — fail-closed at the consult, never a partial set).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resnet_seeded_compose_coeff(
        &self,
        segments: &[ResnetSegment],
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        num_specs: usize,
        num_specs_per_dom: usize,
        output_dim: usize,
        relu_pre_lower: &[&[f32]],
        beta_signed: &[&[f32]],
        beta_gather_idx: &[&[u32]],
        frontier_abs: &[&[f32]],
        force_concretize: bool,
        node_abs: &[&[f32]],
        force_fine: bool,
    ) -> Result<(ResidentCoeff, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let concretize_err = !frontier_abs.is_empty()
            && (force_concretize
                || std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1"));
        let concretize_fine = !node_abs.is_empty()
            && (force_fine
                || std::env::var("NY_RESNET_ERR_CONCRETIZE_FINE")
                    .ok()
                    .as_deref()
                    == Some("1"));
        if seg_probe_armed() {
            eprintln!(
                "[conc-gate] concretize_err={concretize_err} concretize_fine={concretize_fine} \
                 frontier_abs.len()={} node_abs.len()={} seg.len()={}",
                frontier_abs.len(),
                node_abs.len(),
                segments.len()
            );
        }
        let n0 =
            resident_checked_product(&[num_specs, output_dim], "resnet seed coefficient elements")?;
        resident_checked_u32(n0, "resnet seed coefficient elements")?;
        if lower_a.len() != n0 || upper_a.len() != n0 {
            return Err(NyError::shape_mismatch(
                vec![num_specs, output_dim],
                vec![lower_a.len()],
            ));
        }
        if lower_b.len() != num_specs || upper_b.len() != num_specs {
            return Err(NyError::shape_mismatch(
                vec![num_specs],
                vec![lower_b.len()],
            ));
        }
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_sound_resident_resnet: empty segment list".into(),
            ));
        }
        // #u4 (AUTO or explicit NY_GPU_TAINT_WORDS=1; explicit opt-out ⇒
        // `None`, byte-identical): the composition's OWN G13 seeding of the
        // ENTRY frontier, at spec-row
        // granularity (the same granularity the walk's seed-BIAS wording uses).
        // A sentinel-magnitude coefficient/bias shipped into the resnet path
        // condemns its row before any sub-walk runs. The first sub-walk G13-
        // seeds the same host values again per-coefficient — that double-cover
        // is sound, not a launder: both mechanisms only ever ADD words (OR),
        // neither can clear one, so the composed set is always a superset of
        // the true taint at row granularity. Seed errors enter as exact 0.0
        // and need no wording.
        let taint_on = self.taint_words_armed();
        let seed_taint_rows: Option<Vec<u32>> = if taint_on {
            let mut rows = vec![0u32; num_specs];
            for (s, row) in rows.iter_mut().enumerate() {
                let base = s * output_dim;
                let mut word = 0u32;
                for j in 0..output_dim {
                    word |= taint_seed_word(lower_a[base + j]);
                    word |= taint_seed_word(upper_a[base + j]);
                }
                word |= taint_seed_word(lower_b[s]);
                word |= taint_seed_word(upper_b[s]);
                *row = word;
            }
            Some(rows)
        } else {
            None
        };
        let mut coeff = ResidentCoeff {
            lower_a: lower_a.to_vec(),
            upper_a: upper_a.to_vec(),
            lower_err: vec![0.0; n0],
            upper_err: vec![0.0; n0],
            lower_b: lower_b.to_vec(),
            upper_b: upper_b.to_vec(),
            lower_b_err: vec![0.0; num_specs],
            upper_b_err: vec![0.0; num_specs],
            dim: output_dim,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            taint_rows: seed_taint_rows,
        };
        // Captured per-ReLU gradients + gathered A-values, accumulated across
        // segments in fold order (the per-segment `coeff` is replaced on each
        // merge, so we drain each branch's channels here before that happens).
        let mut all_grads: Vec<Vec<f32>> = Vec::new();
        let mut all_gathers: Vec<Vec<f32>> = Vec::new();
        let mut grad_idx = 0usize;
        let n_act = |b: &[GpuCrownLayer]| {
            b.iter()
                .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
                .count()
        };
        let slice_for = |start: usize, count: usize| -> Vec<&[f32]> {
            if relu_pre_lower.is_empty() {
                return Vec::new();
            }
            let end = (start + count).min(relu_pre_lower.len());
            relu_pre_lower[start.min(end)..end].to_vec()
        };
        // Same fold-order indexing as `relu_pre_lower` (one entry per Activation, F before P).
        let beta_slice_for = |start: usize, count: usize| -> Vec<&[f32]> {
            if beta_signed.is_empty() {
                return Vec::new();
            }
            let end = (start + count).min(beta_signed.len());
            beta_signed[start.min(end)..end].to_vec()
        };
        // Same fold-order indexing for the beta-gradient A-value gather channel.
        let gather_slice_for = |start: usize, count: usize| -> Vec<&[u32]> {
            if beta_gather_idx.is_empty() {
                return Vec::new();
            }
            let end = (start + count).min(beta_gather_idx.len());
            beta_gather_idx[start.min(end)..end].to_vec()
        };
        // Per-Activation pre-node abs-max bounds, same fold-order indexing (for the
        // finer per-ReLU error concretization). Empty unless fine concretization is on.
        let node_slice_for = |start: usize, count: usize| -> Vec<&[f32]> {
            if node_abs.is_empty() {
                return Vec::new();
            }
            let end = (start + count).min(node_abs.len());
            node_abs[start.min(end)..end].to_vec()
        };
        // Certified Cut-CROWN C2 (dark `NY_CUT_FOLD_RESIDENT` gate): the fold
        // targets the LAST Activation in fold order — the network's FIRST ReLU
        // (innermost segment), whose post-activation the L1 cuts constrain.
        // `None` (gate off / no entry / no Activations) ⇒ the dispatch below is
        // byte-identical to today.
        //
        // #mn-head-resident research retarget: the target derivation below is
        // retained, but `head_resident_retarget_enabled()` is production-authority
        // quarantined. An environment variable therefore cannot move a registered
        // stem fold to index 0. Re-authorization requires a checker-backed facet
        // certificate with f32 reduction error and bound target identity.
        let cut_fold = super::cut_fold_resident::active_resident_cut_fold();
        let cut_fold_target: Option<usize> = cut_fold.as_ref().and_then(|_| {
            let total: usize = segments
                .iter()
                .map(|s| match s {
                    ResnetSegment::Chain(b) | ResnetSegment::Residual(b) => n_act(b),
                    ResnetSegment::ResidualProj(f, p) => n_act(f) + n_act(p),
                })
                .sum();
            if super::cut_fold_resident::head_resident_retarget_enabled() {
                // HEAD = fold-order index 0 (only when there is ≥1 Activation).
                (total >= 1).then_some(0usize)
            } else {
                // STEM = last Activation in fold order.
                total.checked_sub(1)
            }
        });
        // Carrier-driven resident Cut-CROWN SHADOW hook (observation-only;
        // `ops/cut_shadow_resident.rs`). Armed strictly around one synchronous
        // shadow fold by the cut-shadow driver on this thread; `None` on every
        // production walk ⇒ byte-identical. Unlike the quarantined registry
        // fold above, an armed hook that cannot be applied exactly once is a
        // typed refusal, never a silent untouched walk.
        let cut_shadow_target = super::cut_shadow_resident::armed_cut_shadow_target();
        if let Some(target) = cut_shadow_target {
            let total: usize = segments
                .iter()
                .map(|s| match s {
                    ResnetSegment::Chain(b) | ResnetSegment::Residual(b) => n_act(b),
                    ResnetSegment::ResidualProj(f, p) => n_act(f) + n_act(p),
                })
                .sum();
            if target >= total {
                return Err(NyError::SoundnessRefusal(
                    "wgpu resident cut shadow: target activation is outside the fold".into(),
                ));
            }
        }
        let mut cut_shadow_applied = false;
        // #seg-resident (dark `NY_SEG_RESIDENT=1`): keep the coefficient stream
        // ON DEVICE across segments — the per-segment download → CPU merge →
        // re-upload round-trip (measured ~8.6 ms fixed cost × 2810 calls in a
        // 70 s BaB run) collapses to ONE download after the loop. Eligible only
        // on the plain fold path: no fine/segment error concretization (CPU
        // per-segment ops on the frontier), no cut fold, no α-gradient or
        // β-gather capture channels (keep-mode readback skips them; β values
        // themselves are fine — they ride the per-layer activation passes).
        // First segment must be a Chain so the device stream exists before any
        // skip merge. OFF ⇒ byte-identical legacy path.
        let seg_resident = seg_resident_enabled()
            && !concretize_fine
            && !concretize_err
            && cut_fold.is_none()
            && cut_shadow_target.is_none()
            && relu_pre_lower.is_empty()
            && beta_gather_idx.is_empty()
            && matches!(segments.first(), Some(ResnetSegment::Chain(_)));
        if seg_resident_enabled() && seg_probe_armed() {
            eprintln!(
                "[seg-resident] eligible={seg_resident} fine={concretize_fine} \
                 err={concretize_err} cut={} pre={} gather={} first_chain={} nseg={}",
                cut_fold.is_some(),
                relu_pre_lower.len(),
                beta_gather_idx.len(),
                matches!(segments.first(), Some(ResnetSegment::Chain(_))),
                segments.len()
            );
        }
        let mut coeff_dev: Option<ResidentCoeffBufs> = None;
        for (seg_idx, seg) in segments.iter().enumerate() {
            // The "F" branch (or the plain chain) always backward-propagates the
            // FULL incoming frontier (coefficient + bias + their errors).
            let branch = match seg {
                ResnetSegment::Chain(layers) => layers,
                ResnetSegment::Residual(branch) => branch,
                ResnetSegment::ResidualProj(f_branch, _) => f_branch,
            };
            let fb_count = n_act(branch);
            let fb_pre = slice_for(grad_idx, fb_count);
            let fb_beta = beta_slice_for(grad_idx, fb_count);
            let fb_gather = gather_slice_for(grad_idx, fb_count);
            // C2 cut fold: does the target Activation live in THIS branch?
            let fb_fold = cut_fold.as_ref().and_then(|f| {
                cut_fold_target
                    .filter(|&t| t >= grad_idx && t < grad_idx + fb_count)
                    .map(|t| (t - grad_idx, f))
            });
            // Carrier-driven cut SHADOW: same branch-window rule as the legacy
            // fold, dispatched first (the registry fold is hard-quarantined and
            // can never coexist with an armed hook in practice).
            let fb_cut_shadow = cut_shadow_target
                .filter(|&t| t >= grad_idx && t < grad_idx + fb_count)
                .map(|t| t - grad_idx);
            let mut cf = if let Some(local_act) = fb_cut_shadow {
                let fb_node = node_slice_for(grad_idx, fb_count);
                let snapshot =
                    super::cut_shadow_resident::armed_cut_shadow_snapshot().ok_or_else(|| {
                        NyError::InternalError(
                            "wgpu resident cut shadow: armed hook lost its snapshot".into(),
                        )
                    })?;
                let out = self.backward_branch_carrier_cut(
                    branch,
                    &coeff,
                    num_specs,
                    num_specs_per_dom,
                    &fb_pre,
                    &fb_beta,
                    &fb_gather,
                    &fb_node,
                    concretize_fine,
                    local_act,
                    &snapshot,
                )?;
                cut_shadow_applied = true;
                super::cut_shadow_resident::note_cut_shadow_walk_applied();
                out
            } else if let Some((local_act, fold)) = fb_fold {
                let fb_node = node_slice_for(grad_idx, fb_count);
                self.backward_branch_cut_fold(
                    branch,
                    &coeff,
                    num_specs,
                    num_specs_per_dom,
                    &fb_pre,
                    &fb_beta,
                    &fb_gather,
                    &fb_node,
                    concretize_fine,
                    local_act,
                    fold,
                )?
            } else if concretize_fine {
                let fb_node = node_slice_for(grad_idx, fb_count);
                self.backward_branch_fine(
                    branch,
                    &coeff,
                    num_specs,
                    num_specs_per_dom,
                    &fb_pre,
                    &fb_beta,
                    &fb_gather,
                    &fb_node,
                )?
            } else {
                if seg_resident {
                    // Arm the fold's slot: seed from the device stream (None on
                    // the first segment ⇒ the legacy host-slice upload) and keep
                    // the result on device (skip the readback, deposit handles).
                    RESIDENT_IO.with(|io| {
                        let mut io = io.borrow_mut();
                        io.seed = coeff_dev.clone();
                        io.zero_bias_seed = false;
                        io.keep = true;
                        io.out = None;
                    });
                }
                self.crown_backward_sound_resident_coeff_seeded_err_gather(
                    branch,
                    &coeff.lower_a,
                    &coeff.upper_a,
                    &coeff.lower_err,
                    &coeff.upper_err,
                    &coeff.lower_b,
                    &coeff.upper_b,
                    &coeff.lower_b_err,
                    &coeff.upper_b_err,
                    num_specs,
                    num_specs_per_dom,
                    coeff.dim,
                    &fb_pre,
                    &fb_beta,
                    &fb_gather,
                )?
            };
            grad_idx += fb_count;
            all_grads.append(&mut cf.relu_grads);
            all_gathers.append(&mut cf.beta_gather);
            // #seg-resident: the on-device analogue of the CPU match below. The
            // fold deposited its result handles; skip merges run as seg_merge
            // dispatches (value lanes bit-identical to the CPU merge, error
            // lanes ≥ — see `seg_merge_dispatch`). `coeff` becomes an empty
            // shell carrying only `dim`; the device stream is authoritative
            // until the ONE post-loop download.
            if seg_resident {
                let f_out = RESIDENT_IO
                    .with(|io| io.borrow_mut().out.take())
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "seg-resident: fold deposited no result handles".into(),
                        )
                    })?;
                let merged = match seg {
                    ResnetSegment::Chain(_) => f_out,
                    ResnetSegment::Residual(_) => {
                        let prev = coeff_dev.take().ok_or_else(|| {
                            NyError::InternalError(
                                "seg-resident: skip merge with no device frontier".into(),
                            )
                        })?;
                        if f_out.dim != prev.dim {
                            return Err(NyError::shape_mismatch(vec![prev.dim], vec![f_out.dim]));
                        }
                        let n = num_specs * f_out.dim;
                        self.seg_merge_dispatch(&[
                            (&f_out.la, &f_out.le, &prev.la, &prev.le, n),
                            (&f_out.ua, &f_out.ue, &prev.ua, &prev.ue, n),
                        ])?;
                        f_out
                    }
                    ResnetSegment::ResidualProj(_, p_branch) => {
                        let prev = coeff_dev.take().ok_or_else(|| {
                            NyError::InternalError(
                                "seg-resident: proj merge with no device frontier".into(),
                            )
                        })?;
                        // P branch: seed = the SAME pre-segment frontier (its
                        // buffers are only READ by the seed copy), zero bias so
                        // the incoming bias is counted once (in F's stream).
                        RESIDENT_IO.with(|io| {
                            let mut io = io.borrow_mut();
                            io.seed = Some(prev.clone());
                            io.zero_bias_seed = true;
                            io.keep = true;
                            io.out = None;
                        });
                        let pb_count = n_act(p_branch);
                        let pb_beta = beta_slice_for(grad_idx, pb_count);
                        // Host slices are unused placeholders under a device
                        // seed (the fold skips the host checks + upload).
                        let mut cp = self.crown_backward_sound_resident_coeff_seeded_err_gather(
                            p_branch,
                            &[],
                            &[],
                            &[],
                            &[],
                            &[],
                            &[],
                            &[],
                            &[],
                            num_specs,
                            num_specs_per_dom,
                            prev.dim,
                            &[],
                            &pb_beta,
                            &[],
                        )?;
                        grad_idx += pb_count;
                        all_grads.append(&mut cp.relu_grads);
                        all_gathers.append(&mut cp.beta_gather);
                        let p_out = RESIDENT_IO
                            .with(|io| io.borrow_mut().out.take())
                            .ok_or_else(|| {
                                NyError::InternalError(
                                    "seg-resident: P fold deposited no result handles".into(),
                                )
                            })?;
                        if f_out.dim != p_out.dim {
                            return Err(NyError::shape_mismatch(vec![f_out.dim], vec![p_out.dim]));
                        }
                        let n = num_specs * f_out.dim;
                        self.seg_merge_dispatch(&[
                            (&f_out.la, &f_out.le, &p_out.la, &p_out.le, n),
                            (&f_out.ua, &f_out.ue, &p_out.ua, &p_out.ue, n),
                            (&f_out.blo, &f_out.ble, &p_out.blo, &p_out.ble, num_specs),
                            (&f_out.buo, &f_out.bue, &p_out.buo, &p_out.bue, num_specs),
                        ])?;
                        f_out
                    }
                };
                let dim = merged.dim;
                coeff_dev = Some(merged);
                coeff = ResidentCoeff {
                    lower_a: Vec::new(),
                    upper_a: Vec::new(),
                    lower_err: Vec::new(),
                    upper_err: Vec::new(),
                    lower_b: Vec::new(),
                    upper_b: Vec::new(),
                    lower_b_err: Vec::new(),
                    upper_b_err: Vec::new(),
                    dim,
                    relu_grads: Vec::new(),
                    beta_gather: Vec::new(),
                    // #u4 GENUINELY UNWIRABLE (seg-resident device stream
                    // placeholder): the frontier lives on-device with no word
                    // buffers. Unreachable under the gate — taint_on +
                    // seed/keep streams is a typed refusal at the sub-walk
                    // entry (pinned by
                    // `taint_resnet_seg_resident_stream_refuses`) — so this
                    // `None` only ever flows on the gate-off path, where it is
                    // today's exact value.
                    taint_rows: None,
                };
                continue;
            }
            // #u4 SEAM OR (re-seed carriage): the next frontier `cf`/the merge
            // below was seeded from THIS `coeff`'s VALUES — the sub-walk's G13
            // entry seeding re-words anything still at sentinel magnitude, but
            // a word whose value was already LAUNDERED below the magnitude
            // threshold in an earlier segment survives ONLY in `coeff`'s
            // per-spec rows. OR them into the outgoing frontier (rows are
            // spec-stable across segments). Together with G13 this DOUBLE-
            // COVERS the seam, which is sound rather than a launder: G13 is
            // exact (per-coefficient, with dead-partner annihilation) for
            // still-visible sentinels, the row OR is conservative (refusal-
            // only) for laundered history, and both operations can only ADD
            // words — no path through the seam can clear one. Fail-closed:
            // either side `None` poisons the whole result to `None`
            // (`merge_taint_rows`), never a partial `Some`. Gate OFF: every
            // side is `None` ⇒ `None`, byte-identical.
            let prev_taint_rows = coeff.taint_rows.clone();
            coeff = match seg {
                ResnetSegment::Chain(_) => cf,
                ResnetSegment::Residual(_) => {
                    if cf.dim != coeff.dim {
                        return Err(NyError::shape_mismatch(vec![coeff.dim], vec![cf.dim]));
                    }
                    add_skip_stream(cf, &coeff)
                }
                ResnetSegment::ResidualProj(_, p_branch) => {
                    // Second branch P carries ONLY the coefficient/its error (bias
                    // seeded to 0 so the incoming bias is counted once, in `cf`).
                    let zb = vec![0.0f32; num_specs];
                    let pb_count = n_act(p_branch);
                    let pb_pre = slice_for(grad_idx, pb_count);
                    let pb_beta = beta_slice_for(grad_idx, pb_count);
                    let pb_gather = gather_slice_for(grad_idx, pb_count);
                    // C2 cut fold: target Activation in the P branch (only possible
                    // when the LAST segment is a ResidualProj — F precedes P in fold
                    // order, so the last fold index lands in P).
                    let pb_fold = cut_fold.as_ref().and_then(|f| {
                        cut_fold_target
                            .filter(|&t| t >= grad_idx && t < grad_idx + pb_count)
                            .map(|t| (t - grad_idx, f))
                    });
                    // Carrier-driven cut SHADOW target inside the P branch:
                    // supported through the same zero-bias P seed (the bias
                    // channel applied inside P survives `merge_streams`, which
                    // adds the two branch streams — the F stream carries the
                    // incoming bias once).
                    let pb_cut_shadow = cut_shadow_target
                        .filter(|&t| t >= grad_idx && t < grad_idx + pb_count)
                        .map(|t| t - grad_idx);
                    // P branch carries ONLY the coefficient/its error (zero bias).
                    let p_seed = (concretize_fine || pb_fold.is_some() || pb_cut_shadow.is_some())
                        .then(|| ResidentCoeff {
                            lower_a: coeff.lower_a.clone(),
                            upper_a: coeff.upper_a.clone(),
                            lower_err: coeff.lower_err.clone(),
                            upper_err: coeff.upper_err.clone(),
                            lower_b: zb.clone(),
                            upper_b: zb.clone(),
                            lower_b_err: zb.clone(),
                            upper_b_err: zb.clone(),
                            dim: coeff.dim,
                            relu_grads: Vec::new(),
                            beta_gather: Vec::new(),
                            // #u4: the P branch consumes the SAME incoming frontier
                            // (coefficients + errors; bias zeroed — an exact 0.0
                            // contributes no word), so it inherits the frontier's
                            // row words. Without this the fine/cut-fold P walk
                            // would start from `None` and poison the projection
                            // merge (fail-closed but needlessly refusing the whole
                            // common path).
                            taint_rows: coeff.taint_rows.clone(),
                        });
                    let mut cp = if let Some(local_act) = pb_cut_shadow {
                        let pb_node = node_slice_for(grad_idx, pb_count);
                        let snapshot = super::cut_shadow_resident::armed_cut_shadow_snapshot()
                            .ok_or_else(|| {
                                NyError::InternalError(
                                    "wgpu resident cut shadow: armed hook lost its snapshot".into(),
                                )
                            })?;
                        let out = self.backward_branch_carrier_cut(
                            p_branch,
                            p_seed
                                .as_ref()
                                .expect("p_seed built when the shadow is set"),
                            num_specs,
                            num_specs_per_dom,
                            &pb_pre,
                            &pb_beta,
                            &pb_gather,
                            &pb_node,
                            concretize_fine,
                            local_act,
                            &snapshot,
                        )?;
                        cut_shadow_applied = true;
                        super::cut_shadow_resident::note_cut_shadow_walk_applied();
                        out
                    } else if let Some((local_act, fold)) = pb_fold {
                        let pb_node = node_slice_for(grad_idx, pb_count);
                        self.backward_branch_cut_fold(
                            p_branch,
                            p_seed.as_ref().expect("p_seed built when fold is set"),
                            num_specs,
                            num_specs_per_dom,
                            &pb_pre,
                            &pb_beta,
                            &pb_gather,
                            &pb_node,
                            concretize_fine,
                            local_act,
                            fold,
                        )?
                    } else if concretize_fine {
                        let pb_node = node_slice_for(grad_idx, pb_count);
                        self.backward_branch_fine(
                            p_branch,
                            p_seed.as_ref().expect("p_seed built when fine is set"),
                            num_specs,
                            num_specs_per_dom,
                            &pb_pre,
                            &pb_beta,
                            &pb_gather,
                            &pb_node,
                        )?
                    } else {
                        self.crown_backward_sound_resident_coeff_seeded_err_gather(
                            p_branch,
                            &coeff.lower_a,
                            &coeff.upper_a,
                            &coeff.lower_err,
                            &coeff.upper_err,
                            &zb,
                            &zb,
                            &zb,
                            &zb,
                            num_specs,
                            num_specs_per_dom,
                            coeff.dim,
                            &pb_pre,
                            &pb_beta,
                            &pb_gather,
                        )?
                    };
                    grad_idx += pb_count;
                    all_grads.append(&mut cp.relu_grads);
                    all_gathers.append(&mut cp.beta_gather);
                    if cf.dim != cp.dim {
                        return Err(NyError::shape_mismatch(vec![cf.dim], vec![cp.dim]));
                    }
                    merge_streams(cf, &cp)
                }
            };
            // (#u4 seam OR, see the comment above `prev_taint_rows`.) For the
            // Residual arm `add_skip_stream` already OR'd the same rows —
            // ORing them again is idempotent, keeping this seam uniform.
            coeff.taint_rows =
                merge_taint_rows(coeff.taint_rows.take(), prev_taint_rows.as_deref());
            // #unsat-keystone: concretize the accumulated coefficient error against the
            // frontier node bounds → fold into the (scalar, non-amplifying) bias error,
            // then reset the coefficient error. Caps the per-segment L1 error blow-up.
            // SOUND: each coefficient j's error contributes at most |err_a[j]|·max(|z_l|,
            // |z_u|) to the bound; folding that into the bias error and zeroing err_a is a
            // valid over-approximation (mirrors per-node CPU concretization).
            if concretize_err {
                if let Some(fab) = frontier_abs.get(seg_idx) {
                    // #u4 ROW-INVARIANT host fold: the per-coefficient taint
                    // companion (`concretize_error_taint_into_bias`) is NOT
                    // needed at this altitude — the composition carries words
                    // only per SPEC ROW, and this fold moves err mass into the
                    // bias err of the SAME spec row. The err's taint is already
                    // in the row word (each sub-walk ORs its final `le`/`ue`
                    // words into its rows at exit; seed errs are G13-worded at
                    // walk entry), so the fold cannot launder it; skipping the
                    // `fab == 0` annihilation is conservative-only (a row stays
                    // condemned that a per-coefficient channel might clear).
                    Self::concretize_error_into_bias(&mut coeff, num_specs, num_specs_per_dom, fab);
                }
            }
            if seg_probe_armed() {
                let cmax = coeff
                    .lower_a
                    .iter()
                    .chain(coeff.upper_a.iter())
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                let emax = coeff
                    .lower_err
                    .iter()
                    .chain(coeff.upper_err.iter())
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                let bemax = coeff
                    .lower_b_err
                    .iter()
                    .chain(coeff.upper_b_err.iter())
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                eprintln!(
                    "[seg] coeff_Linf={cmax:.4e} err_Linf={emax:.4e} bias_err={bemax:.4e} dim={}",
                    coeff.dim
                );
            }
        }
        // The armed cut-shadow hook must have been applied exactly once (the
        // disjoint, increasing branch windows make a double application
        // unrepresentable; a miss means the target/fold accounting disagreed).
        // Mirrors the CUDA "target activation was not encountered exactly
        // once" refusal — never a silent untouched walk.
        if cut_shadow_target.is_some() && !cut_shadow_applied {
            return Err(NyError::SoundnessRefusal(
                "wgpu resident cut shadow: target activation was not encountered exactly once"
                    .into(),
            ));
        }
        // #seg-resident: the ONE download for the whole backward — every
        // downstream consumer (input-coeff capture, full-coeff capture, the
        // final concretization) then flows through the unchanged host path.
        // (#u4: unreachable under the gate — taint_on + device seed/keep
        // streams is a typed refusal at the sub-walk entry, so `coeff_dev` is
        // only ever `Some` with the gate off, where `taint_rows == None` is
        // the correct value.)
        if let Some(bufs) = coeff_dev.take() {
            coeff = self.download_resident_coeff(&bufs)?;
        }
        Ok((coeff, all_grads, all_gathers))
    }

    /// #seg-resident: dispatch the on-device stream merge for each `(a, err_a,
    /// b, err_b, n)` lane pair — `a += b` (f32 RN add of two f32s IS the
    /// correctly-rounded f64 sum, so the value lane is bit-identical to the CPU
    /// merge), `err_a = up(((err_a + err_b) + |s|·u) · SEG_MERGE_SLACK)` (the
    /// f32 evaluation of the CPU's exact-f64 error expression, slacked outward
    /// so device err ≥ CPU err always — soundness can only widen). All pairs
    /// encode into ONE submit.
    fn seg_merge_dispatch(
        &self,
        pairs: &[(
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            usize,
        )],
    ) -> Result<()> {
        self.run_gpu_checked("seg_merge", || {
            let pipes = self.resident_backward_pipelines();
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("seg_merge"),
                });
            // Intentionally never read: keeps the param buffers alive until submit.
            #[allow(clippy::collection_is_never_read)]
            let mut _params_keepalive: Vec<wgpu::Buffer> = Vec::with_capacity(pairs.len());
            for &(a, ea, b, eb, n) in pairs {
                let wg = (super::gpu_checked_u32(n, "seg_merge n")?)
                    .div_ceil(256)
                    .min(32768);
                let pbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("seg_merge_params"),
                    size: size_of::<SegMergeParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(
                    &pbuf,
                    0,
                    bytemuck::bytes_of(&SegMergeParams {
                        n: n as u32,
                        slack: SEG_MERGE_SLACK,
                        stride: wg * 256,
                        _p1: 0,
                    }),
                );
                self.pass_simple(&mut enc, &pipes.seg_merge, &pbuf, &[a, ea, b, eb], wg);
                _params_keepalive.push(pbuf);
            }
            self.queue.submit(Some(enc.finish()));
            Ok(())
        })
    }

    /// #seg-resident: download the device-resident coefficient stream back to a
    /// host [`ResidentCoeff`] — the ONE download for the whole resnet backward
    /// (replacing the per-segment round-trip). Same staging + batched-map idiom
    /// as the fold's own readback tail.
    fn download_resident_coeff(&self, bufs: &ResidentCoeffBufs) -> Result<ResidentCoeff> {
        let num_specs = bufs.num_specs;
        let dim = bufs.dim;
        let out_elems = num_specs * dim;
        self.run_gpu_checked("seg_resident_download", || {
            let stage = |label: &str, n: usize| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (n.max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let cbytes = (out_elems * size_of::<f32>()) as u64;
            let bbytes = (num_specs * size_of::<f32>()) as u64;
            let st_la = stage("segres_la", out_elems);
            let st_ua = stage("segres_ua", out_elems);
            let st_le = stage("segres_le", out_elems);
            let st_ue = stage("segres_ue", out_elems);
            let st_blo = stage("segres_blo", num_specs);
            let st_buo = stage("segres_buo", num_specs);
            let st_ble = stage("segres_ble", num_specs);
            let st_bue = stage("segres_bue", num_specs);
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("segres_dl"),
                });
            enc.copy_buffer_to_buffer(&bufs.la, 0, &st_la, 0, cbytes);
            enc.copy_buffer_to_buffer(&bufs.ua, 0, &st_ua, 0, cbytes);
            enc.copy_buffer_to_buffer(&bufs.le, 0, &st_le, 0, cbytes);
            enc.copy_buffer_to_buffer(&bufs.ue, 0, &st_ue, 0, cbytes);
            enc.copy_buffer_to_buffer(&bufs.blo, 0, &st_blo, 0, bbytes);
            enc.copy_buffer_to_buffer(&bufs.buo, 0, &st_buo, 0, bbytes);
            enc.copy_buffer_to_buffer(&bufs.ble, 0, &st_ble, 0, bbytes);
            enc.copy_buffer_to_buffer(&bufs.bue, 0, &st_bue, 0, bbytes);
            self.queue.submit(Some(enc.finish()));
            let mut dl = Self::read_buffers_batched(
                &self.device,
                &[
                    (&st_la, out_elems),
                    (&st_ua, out_elems),
                    (&st_le, out_elems),
                    (&st_ue, out_elems),
                    (&st_blo, num_specs),
                    (&st_buo, num_specs),
                    (&st_ble, num_specs),
                    (&st_bue, num_specs),
                ],
            )?;
            let upper_b_err = dl.pop().expect("8 readbacks");
            let lower_b_err = dl.pop().expect("8 readbacks");
            let upper_b = dl.pop().expect("8 readbacks");
            let lower_b = dl.pop().expect("8 readbacks");
            let upper_err = dl.pop().expect("8 readbacks");
            let lower_err = dl.pop().expect("8 readbacks");
            let upper_a = dl.pop().expect("8 readbacks");
            let lower_a = dl.pop().expect("8 readbacks");
            Ok(ResidentCoeff {
                lower_a,
                upper_a,
                lower_err,
                upper_err,
                lower_b,
                upper_b,
                lower_b_err,
                upper_b_err,
                dim,
                relu_grads: Vec::new(),
                beta_gather: Vec::new(),
                // #u4 GENUINELY UNWIRABLE: seg-resident device streams carry
                // no word buffers, and taint_on + device streams is a typed
                // refusal at the walk entry (pinned by
                // `taint_resnet_seg_resident_stream_refuses`) — this download
                // only ever runs gate-off, where `None` is today's exact value.
                taint_rows: None,
            })
        })
    }

    /// Trait-boundary entry for the resnet sound resident backward: run from a
    /// [`ny_core::GpuResnetSegment`] decomposition (owned layer vecs, backward order)
    /// plus a [`GpuCrownSeed`] frontier. Translates the owned segments into the
    /// internal borrowed [`ResnetSegment`] form and delegates to the seeded fold.
    /// Driven by the `GpuCrownBackward::crown_backward_gpu_resnet_sound` trait method.
    pub(crate) fn crown_backward_gpu_resnet_sound_inner(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[&[f32]],
        // #unsat-keystone FINER concretization: per-Activation pre-node abs-max bounds
        // in fold order (empty ⇒ off; the verdict default). When provided AND
        // NY_RESNET_ERR_CONCRETIZE_FINE=1, the per-ReLU error concretization fires.
        node_abs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        // First pass: env-gated concretization (force=false → off by default, on if
        // NY_RESNET_ERR_CONCRETIZE=1; frontier_abs is threaded so the env path still works).
        let env_concretize = std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1");
        let (lo, hi, _grads) = self.crown_backward_sound_resident_resnet_seeded(
            &internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
            &[],
            &[],
            frontier_abs,
            false,
            node_abs,
            false,
        )?;
        // #unsat-keystone auto-fallback: if the un-concretized bound is useless for a verdict —
        // re-run with the error-concretization FORCED and return the element-wise TIGHTER of the
        // two. SOUND: both bounds are valid over-approximations, so their intersection (max of
        // lowers, min of uppers) still contains the true output — it can only tighten, never a
        // false proof. NO-REGRESSION: coordinates the cheap bound already bounded well keep their
        // value (max/min picks them); only the exploded coordinates are replaced by the capped
        // concretized ones, and f32::max/min take the finite concretized value where the cheap one
        // is ±inf/NaN. Cost is 1× on healthy bounds (the threshold never fires); the extra pass is
        // paid only when the cheap bound already failed. Skipped when the env path concretized.
        //
        // EXPLOSION DETECTION: the un-concretized certified f32 error L1-explodes through a deep
        // resnet (~|W| per layer, no cancellation) and the sound concretize CLAMPS the resulting
        // overflow OUTWARD to the ±FALLBACK_BOUND (1e10) sentinel (see CROWN_CONCRETIZE_SOUND_SHADER:
        // non-finite / |a|≥FALLBACK_BOUND ⇒ ±FALLBACK_BOUND). So the explosion surfaces NOT as a raw
        // 1e30 but as an endpoint pinned at ±FALLBACK_BOUND (or, if it slipped under the clamp, a
        // finite-but-astronomically-wide value). We therefore trigger on `!is_finite()` OR
        // `|v| ≥ FALLBACK_BOUND` — capturing the clamp sentinel that a healthy verdict-scale bound
        // never legitimately reaches (1e10 is the overflow-repair floor, not a real activation
        // magnitude). This is what makes deep cifar100/tinyimagenet resnets recover a finite,
        // error-free bound AUTOMATICALLY instead of returning the useless clamped 1e10.
        //
        // PREFER FINE: when `node_abs` is non-empty we force the per-ReLU FINE concretization
        // (`force_fine=true`) instead of only the per-segment fold. Fine folds the accumulated
        // coefficient error into the bias against EVERY ReLU's pre-activation abs-max bound (and
        // resets it), so it caps the |W|-amplification at each ReLU rather than only at the coarse
        // segment boundary — strictly ≥ as tight as the per-segment path (measured ~460× tighter
        // on the deep resnet). Setting force_fine also forces force_concretize ON inside the fold
        // (the per-segment fold still runs as a secondary cap), so this is "fine PLUS segment".
        // Empty `node_abs` ⇒ the recovery falls back to the per-segment `frontier_abs` path exactly
        // as before. Either way the un-concretized first pass — and thus the verdict default for a
        // NON-exploding net (whose endpoints stay well under 1e10) — is byte-for-byte unchanged.
        if !env_concretize
            && Self::resnet_wants_concretized_merge(frontier_abs.is_empty(), &lo, &hi)
        {
            return self.resnet_seeded_fallback_merge(
                &internal,
                seed,
                input_lower,
                input_upper,
                &[],
                frontier_abs,
                node_abs,
                &lo,
                &hi,
            );
        }
        Ok((lo, hi))
    }

    /// Explosion detector shared by the resnet auto-fallbacks (main bound, warmup
    /// grad, BaB beta): the un-concretized certified f32 error L1-explodes through a
    /// deep resnet, surfacing either as a non-finite endpoint or as one at/above the
    /// ±FALLBACK_BOUND (1e10) clamp sentinel — a magnitude a healthy verdict-scale
    /// bound never legitimately reaches.
    fn resnet_bound_exploded(lo: &[f32], hi: &[f32]) -> bool {
        lo.iter()
            .chain(hi.iter())
            .any(|v| !v.is_finite() || v.abs() >= crate::FALLBACK_BOUND)
    }

    /// Decide whether to run the error-concretized second pass + element-wise
    /// tighter merge. Default (#w4-conv-err-per-entry): ALWAYS when the caller
    /// supplied frontier/node abs bounds — the merge is a sound intersection of two
    /// valid enclosures, so it can only tighten, and on deep conv resnets the
    /// concretized pass is the verdict-relevant bound even when the carried-error
    /// pass comes back finite-but-loose (an explosion-only trigger misses exactly
    /// that regime). Cost: one extra resident backward (~sub-second) — paid only on
    /// the resnet path, which needs it. `NY_RESNET_ERR_MERGE=0` restores the legacy
    /// explosion-only trigger for A/B.
    fn resnet_wants_concretized_merge(frontier_abs_empty: bool, lo: &[f32], hi: &[f32]) -> bool {
        if frontier_abs_empty {
            return false;
        }
        if std::env::var("NY_RESNET_ERR_MERGE").ok().as_deref() == Some("0") {
            return Self::resnet_bound_exploded(lo, hi);
        }
        true
    }

    /// Shared explosion auto-fallback re-run for the three resnet trait entries
    /// (main bound / warmup grad / BaB beta): re-run the seeded fold with the
    /// error-concretization FORCED (fine per-ReLU when `node_abs` is available)
    /// and return the element-wise TIGHTER merge with the exploded first-pass
    /// bound. `dual_signed` threads the per-domain β·sign duals (empty for the
    /// non-beta entries — both passes must fold the SAME duals for the merge to
    /// be an intersection of comparable enclosures). Gradients of the re-run are
    /// discarded (callers keep the first pass's).
    #[allow(clippy::too_many_arguments)]
    fn resnet_seeded_fallback_merge(
        &self,
        internal: &[ResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        dual_signed: &[&[f32]],
        frontier_abs: &[&[f32]],
        node_abs: &[&[f32]],
        lo: &[f32],
        hi: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let force_fine = !node_abs.is_empty();
        match self.crown_backward_sound_resident_resnet_seeded(
            internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
            &[],
            dual_signed,
            frontier_abs,
            true,
            node_abs,
            force_fine,
        ) {
            Ok((clo, chi, _grads)) => Ok(Self::merge_tighter_sound(lo, hi, &clo, &chi)),
            // FAIL-OPEN on a HEALTHY first pass (#w4-conv-err-per-entry): under the
            // always-merge policy the second pass also runs when the first-pass bound
            // is already verdict-usable, so a cooperative-deadline expiry (or any GPU
            // error) mid-second-pass must not discard it — return the sound first
            // pass. An EXPLODED first pass is useless, so there the error propagates
            // (the caller's CPU/reference fallback takes over), matching the legacy
            // explosion-only behaviour.
            Err(e) => {
                if Self::resnet_bound_exploded(lo, hi) {
                    Err(e)
                } else {
                    Ok((lo.to_vec(), hi.to_vec()))
                }
            }
        }
    }

    /// Element-wise TIGHTER merge of two valid over-approximations (max of lowers,
    /// min of uppers). SOUND: both inputs enclose the true range, so their
    /// intersection still does — it can only tighten, never produce a false proof.
    /// `f32::max`/`min` take the finite concretized value where the cheap one is
    /// ±inf/NaN. Shared by the three resnet auto-fallback sites.
    fn merge_tighter_sound(
        lo: &[f32],
        hi: &[f32],
        clo: &[f32],
        chi: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let lo2: Vec<f32> = lo.iter().zip(clo.iter()).map(|(&u, &c)| u.max(c)).collect();
        let hi2: Vec<f32> = hi.iter().zip(chi.iter()).map(|(&u, &c)| u.min(c)).collect();
        (lo2, hi2)
    }

    /// Gradient-capturing variant of [`crown_backward_gpu_resnet_sound_inner`]:
    /// returns the SAME sound bounds plus each ReLU's analytic alpha gradient (fold
    /// order), for the GPU-resident warmup alpha optimization. `relu_pre_lower` are
    /// the masked per-ReLU pre-activation lower bounds in fold order.
    pub(crate) fn crown_backward_gpu_resnet_sound_grad_inner(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        relu_pre_lower: &[&[f32]],
        frontier_abs: &[&[f32]],
        node_abs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_grad: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let env_concretize = std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1");
        let (lo, hi, grads) = self.crown_backward_sound_resident_resnet_seeded(
            &internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
            relu_pre_lower,
            &[],
            frontier_abs,
            false,
            node_abs,
            false,
        )?;
        // #w4-gpu-dag-backward: SAME explosion auto-fallback as the main-bound inner
        // (see `crown_backward_gpu_resnet_sound_inner`). Without it the WARMUP bound
        // on a deep resnet came back as a useless finite-but-astronomical value
        // (measured -6.85e32 on cifar100 resnet-medium), so the alpha loop optimized
        // against garbage. The gradients returned are the FIRST pass's (identical
        // relaxation structure; gradients only steer alpha, any alpha ∈ [0,1] is
        // sound). The merged bound is the element-wise intersection of two valid
        // enclosures — sound by the same argument as the main-bound fallback.
        if !env_concretize
            && Self::resnet_wants_concretized_merge(frontier_abs.is_empty(), &lo, &hi)
        {
            let (lo2, hi2) = self.resnet_seeded_fallback_merge(
                &internal,
                seed,
                input_lower,
                input_upper,
                &[],
                frontier_abs,
                node_abs,
                &lo,
                &hi,
            )?;
            return Ok((lo2, hi2, grads));
        }
        Ok((lo, hi, grads))
    }

    /// Beta-capable variant of [`crown_backward_gpu_resnet_sound_inner`] (#unsat-keystone
    /// step 4): folds the per-domain β-CROWN split-constraint dual into the bound. `beta_signed`
    /// is the per-ReLU `β·sign` in fold order (0 for non-split neurons). Returns the sound
    /// (β≥0 ⇒ valid dual) concretized bounds. No gradients (bounds-only, like the non-grad inner).
    pub(crate) fn crown_backward_gpu_resnet_sound_beta_inner(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[&[f32]],
        frontier_abs: &[&[f32]],
        node_abs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let env_concretize = std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1");
        let (lo, hi, _grads) = self.crown_backward_sound_resident_resnet_seeded(
            &internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
            &[],
            beta_signed,
            frontier_abs,
            false,
            node_abs,
            false,
        )?;
        // #w4-gpu-dag-backward: SAME explosion auto-fallback as the main-bound inner
        // (see `crown_backward_gpu_resnet_sound_inner`), for the BaB per-domain beta
        // bound. Both passes fold the SAME per-domain β·sign duals (β ≥ 0 ⇒ each is a
        // valid Lagrangian-dual bound), so the element-wise tighter merge of the two
        // enclosures is sound. Without this, every per-domain bound on a deep resnet
        // explodes to the clamp sentinel and BaB cannot prune a single domain.
        if !env_concretize
            && Self::resnet_wants_concretized_merge(frontier_abs.is_empty(), &lo, &hi)
        {
            return self.resnet_seeded_fallback_merge(
                &internal,
                seed,
                input_lower,
                input_upper,
                beta_signed,
                frontier_abs,
                node_abs,
                &lo,
                &hi,
            );
        }
        Ok((lo, hi))
    }

    /// Beta-GRADIENT variant of [`crown_backward_gpu_resnet_sound_beta_inner`]
    /// (#w4-split-tightening): same sound β-folded bounds (incl. the always-merge
    /// concretized second pass), plus each requested ReLU's pre-transform LOWER
    /// A-values gathered at the requested (split) neuron columns — the analytic
    /// β-gradient inputs (`∂lb_row/∂β_k = −sign_k·A_lower[row, k]`, the CPU
    /// `compute_gradients_for_spec_row` rule). Gathers come from the FIRST pass;
    /// the merge pass re-runs without capture (the coefficient stream is identical
    /// — error concretization only touches the err/bias-err channels — so this
    /// loses nothing). Gathered values are non-soundness-critical.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_gpu_resnet_sound_beta_grad_inner(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[&[f32]],
        beta_gather_idx: &[&[u32]],
        frontier_abs: &[&[f32]],
        node_abs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_grad: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let env_concretize = std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1");
        let (lo, hi, _grads, gathers) = self.crown_backward_sound_resident_resnet_seeded_gather(
            &internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.num_specs, // #batched-bab: single-domain caller (per-dom == total).
            seed.current_dim,
            input_lower,
            input_upper,
            &[],
            beta_signed,
            beta_gather_idx,
            frontier_abs,
            false,
            node_abs,
            false,
            None,
            None,
        )?;
        // SAME explosion/always-merge auto-fallback as the beta inner (both passes
        // fold the SAME β·sign duals ⇒ each is a valid Lagrangian-dual bound; the
        // element-wise tighter merge of the two enclosures is sound). Gathers stay
        // the first pass's.
        if !env_concretize
            && Self::resnet_wants_concretized_merge(frontier_abs.is_empty(), &lo, &hi)
        {
            let (lo2, hi2) = self.resnet_seeded_fallback_merge(
                &internal,
                seed,
                input_lower,
                input_upper,
                beta_signed,
                frontier_abs,
                node_abs,
                &lo,
                &hi,
            )?;
            return Ok((lo2, hi2, gathers));
        }
        Ok((lo, hi, gathers))
    }

    /// #batched-bab: the WIDE resident β-CROWN backward — the single GPU pass that
    /// runs ALL `n_domains` BaB subdomains over `N = seed.num_specs` stacked rows
    /// (`num_specs_per_dom` rows per domain), replacing the reference stacker's serial
    /// per-domain loop. Every per-domain input is domain-block-STACKED:
    /// - `wide_segments`: shared skeleton with each `Activation`'s slopes/intercepts
    ///   concatenated into `n_domains` blocks of `num_neurons` (HOLES 1/2).
    /// - `seed`: the shared spec seed TILED `n_domains` times (`num_specs = N`).
    /// - `input_lower/upper`: `n_domains * input_dim` (HOLE 3).
    /// - `beta_signed`/`node_abs`: `n_domains * num_neurons` per Activation (fold order);
    ///   `frontier_abs`: `n_domains * seg_dim` per segment (HOLE 4).
    ///
    /// Row `s`'s domain is `s / num_specs_per_dom`; the resident shaders + the two host
    /// error folds read that domain's block, so no cross-domain state leaks. This is
    /// EXACTLY the per-domain `crown_backward_gpu_resnet_sound_beta_inner` computation
    /// applied to every block at once (the two-sided differential oracle verifies the
    /// wide bound matches the serial per-domain bound within f32-reorder tol). Any
    /// mis-index would fold one domain's rows against another's relaxation/box/abs-max
    /// ⇒ a tighter (WRONG) bound ⇒ caught by the oracle before this path is trusted.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_gpu_resnet_sound_beta_wide_inner(
        &self,
        wide_segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        num_specs_per_dom: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[&[f32]],
        frontier_abs: &[&[f32]],
        node_abs: &[&[f32]],
        // #batched-bab part A (wide β-opt): per-ReLU UNION gather-column list (fold order),
        // whose PRE-transform LOWER A-values are read back for the analytic β gradient.
        // The gather is captured from the FIRST pass; the merge/concretize re-run leaves
        // the `la` coefficient stream byte-identical (force_concretize only mutates the
        // err/bias-err channels), so the first-pass gather stays valid even when the merge
        // produces the returned enclosure. Empty ⇒ bounds byte-for-byte unchanged (the
        // bound-only callers pass `&[]`). Returns `gathers[r]` = N × |union_cols[r]|
        // row-major: `gathers[r][s*U_r+i] = A_lower[wide-row s, union_cols[r][i]]`.
        wide_beta_gather_idx: &[&[u32]],
        // #w4 wide α+β ascent: per-ReLU (fold order) DOMAIN-STACKED pre-activation
        // lower bounds (`n_domains*nn_r`, stable neurons masked to 0). Non-empty ⇒
        // the domain-blocked alpha-gradient capture runs on the FIRST pass (same
        // first-pass-only rationale as the β gather above) and `alpha_grads[r]` =
        // `n_domains*nn_r` with domain d's block at `d*nn_r`. Empty ⇒ no capture,
        // bounds byte-for-byte unchanged.
        wide_relu_pre_lower: &[&[f32]],
        // #clip-interm-resnet-batched: when Some, the FULL coefficient frontier is
        // captured from a FORCE-FINE pass (so per-coefficient error is folded into the
        // bias error) for the batched clip. Requesting it GUARANTEES a force-fine
        // concretize pass runs, and the returned bounds are the tighter merge of the
        // first pass and that force-fine pass. None ⇒ byte-for-byte unchanged.
        coeff_full_out: Option<&mut ny_core::GpuResidentCoeffBatched>,
        // `true` is the historical clip contract above. `false` captures the
        // first pass for Hydra trajectory banking: its certified coefficient
        // errors remain live and are discharged by the consumer, avoiding a
        // second backward solely for capture.
        force_fine_coeff: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if wide_segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_wide: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = wide_segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let env_concretize = std::env::var("NY_RESNET_ERR_CONCRETIZE").ok().as_deref() == Some("1");
        let want_coeff = coeff_full_out.is_some();
        let mut coeff_full_out = coeff_full_out;
        let first_pass_coeff = if force_fine_coeff {
            None
        } else {
            coeff_full_out.take()
        };
        // First pass: env-gated concretization (default off), num_specs = N wide rows.
        // BOTH capture channels (β A-gather + domain-blocked α gradients) are captured
        // HERE (the first pass); the merge pass below does NOT re-request them (its
        // coefficient stream is byte-identical).
        let (lo, hi, alpha_grads, gathers) = self
            .crown_backward_sound_resident_resnet_seeded_gather(
                &internal,
                &seed.lower_a,
                &seed.upper_a,
                &seed.lower_b,
                &seed.upper_b,
                seed.num_specs,
                num_specs_per_dom,
                seed.current_dim,
                input_lower,
                input_upper,
                wide_relu_pre_lower,
                beta_signed,
                wide_beta_gather_idx,
                frontier_abs,
                false,
                node_abs,
                false,
                None,
                first_pass_coeff,
            )?;
        // Same explosion / always-merge auto-fallback as the per-domain beta inner, but
        // wide: re-run with error-concretization FORCED (fine per-ReLU when node_abs is
        // present) and return the element-wise TIGHTER merge. Both passes fold the SAME
        // per-domain β·sign duals over the SAME domain blocks ⇒ the intersection is a
        // valid enclosure of every domain's true output (sound; only tightens). The
        // first-pass `gathers` are returned unchanged (the merge pass's `la` is identical).
        //
        // #clip-interm-resnet-batched: when the clip requests the coeff frontier
        // (`want_coeff`), FORCE the force-fine pass (regardless of the merge heuristic)
        // and capture the coeff FROM it — the force-fine pass has already concretized the
        // per-coefficient error into the (scalar) bias error, so the captured rows are the
        // usable, near-error-free enclosure the clip needs. The returned bounds are still
        // the tighter merge. A coeff-pass Err returns Err (caller drops the clip for this
        // batch — sound, keeps frozen intermediates).
        let force_merge = (want_coeff && force_fine_coeff)
            || (!env_concretize
                && Self::resnet_wants_concretized_merge(frontier_abs.is_empty(), &lo, &hi));
        if force_merge {
            let force_fine = !node_abs.is_empty();
            return match self.crown_backward_sound_resident_resnet_seeded_gather(
                &internal,
                &seed.lower_a,
                &seed.upper_a,
                &seed.lower_b,
                &seed.upper_b,
                seed.num_specs,
                num_specs_per_dom,
                seed.current_dim,
                input_lower,
                input_upper,
                &[],
                beta_signed,
                &[],
                frontier_abs,
                true,
                node_abs,
                force_fine,
                None,
                coeff_full_out,
            ) {
                Ok((clo, chi, _, _)) => {
                    let (mlo, mhi) = Self::merge_tighter_sound(&lo, &hi, &clo, &chi);
                    Ok((mlo, mhi, alpha_grads, gathers))
                }
                Err(e) => {
                    if want_coeff || Self::resnet_bound_exploded(&lo, &hi) {
                        Err(e)
                    } else {
                        Ok((lo, hi, alpha_grads, gathers))
                    }
                }
            };
        }
        Ok((lo, hi, alpha_grads, gathers))
    }

    /// #batched-vjp: the LEAN wide point-VJP fold — ONE resident backward pass over
    /// `N = n_domains * num_specs_per_dom` stacked rows that returns the FOLDED
    /// input-level LOWER coefficient rows (`N × input_dim`, row-major) via the
    /// `input_coeff_out` side channel of the seeded fold. For a mask-slope fold
    /// (each domain's `Activation` slopes == that restart's 0/1 ReLU mask, zero
    /// intercepts, `lower_slope == upper_slope`) these rows ARE the exact per-row
    /// point gradients `d(spec_row · f(x)) / d(input)`.
    ///
    /// Sibling of [`Self::crown_backward_gpu_resnet_sound_beta_wide_inner`] that
    /// SKIPS the merge/concretized second pass entirely: the concretized bounds are
    /// unused by the VJP caller (attack-only), so one pass suffices — no
    /// `frontier_abs` / `node_abs` / β / gather channels. NOT wrapped in
    /// `run_gpu_checked` for the same non-reentrant-lock reason as the other resnet
    /// entries (each inner op takes the lock itself); any GPU fault propagates as
    /// `Err` and the caller falls back to the sequential exact gradient.
    pub(crate) fn crown_backward_gpu_point_vjp_wide_inner(
        &self,
        wide_segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        num_specs_per_dom: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<Vec<f32>> {
        if wide_segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_point_vjp_wide: empty segment list".into(),
            ));
        }
        let internal: Vec<ResnetSegment> = wide_segments
            .iter()
            .map(|s| match s {
                GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                }
            })
            .collect();
        let mut input_coeff = Vec::new();
        // Single pass, all optional channels empty. The bounds (lo/hi) are computed
        // against the caller's (dummy) box and DISCARDED — only the pre-concretize
        // folded lower coefficient matters here.
        let (_lo, _hi, _grads, _gathers) = self
            .crown_backward_sound_resident_resnet_seeded_gather(
                &internal,
                &seed.lower_a,
                &seed.upper_a,
                &seed.lower_b,
                &seed.upper_b,
                seed.num_specs,
                num_specs_per_dom,
                seed.current_dim,
                input_lower,
                input_upper,
                &[],
                &[],
                &[],
                &[],
                false,
                &[],
                false,
                Some(&mut input_coeff),
                None,
            )?;
        Ok(input_coeff)
    }

    /// GPU per-ReLU analytic alpha gradient (cifar100/tinyimagenet unsat keystone,
    /// step 1 of the gradient-capable GPU-resident alpha-CROWN warmup):
    /// `grad[i] = pre_lower[i] · Σ_j max(a_lower[j,i], 0)`, with `a_lower` the
    /// `num_specs × num_neurons` (row-major) lower coefficient entering an unstable
    /// ReLU and `pre_lower[i]` its pre-activation lower bound (caller folds the
    /// unstable mask in — 0 for stable neurons). Numerically matches the CPU
    /// `compute_graph_chain_rule_gradients`; computing it on-device avoids the
    /// per-iteration dense-coefficient round-trip that makes the resnet warmup
    /// overrun the budget (BaB then never runs — measured: 0 domains at ≤400 s).
    // Production migrated to the fused capture (`relu_pre_lower` channel) and the joint
    // adjoint; this standalone entry remains the CPU-formula differential oracle target
    // (gpu-tests `crown_alpha_gradient_resident_matches_cpu_formula`).
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    pub(crate) fn crown_alpha_gradient_resident(
        &self,
        a_lower: &[f32],
        pre_lower: &[f32],
        num_specs: usize,
        num_neurons: usize,
    ) -> Result<Vec<f32>> {
        let coefficient_elems = resident_checked_product(
            &[num_specs, num_neurons],
            "standalone alpha-gradient coefficient elements",
        )?;
        let num_specs_u32 = resident_checked_u32(num_specs, "standalone alpha-gradient specs")?;
        let num_neurons_u32 =
            resident_checked_u32(num_neurons, "standalone alpha-gradient neurons")?;
        resident_checked_u32(
            coefficient_elems,
            "standalone alpha-gradient coefficient elements",
        )?;
        if a_lower.len() != coefficient_elems {
            return Err(NyError::shape_mismatch(
                vec![num_specs, num_neurons],
                vec![a_lower.len()],
            ));
        }
        if pre_lower.len() != num_neurons {
            return Err(NyError::shape_mismatch(
                vec![num_neurons],
                vec![pre_lower.len()],
            ));
        }
        if num_neurons == 0 {
            return Ok(Vec::new());
        }
        let limits = self.device.limits();
        let coefficient_bytes =
            resident_f32_bytes(coefficient_elems.max(1), "standalone alpha-gradient input")?;
        let neuron_bytes =
            resident_f32_bytes(num_neurons.max(1), "standalone alpha-gradient output")?;
        let storage_limit = limits.max_storage_buffer_binding_size;
        if coefficient_bytes > limits.max_buffer_size
            || coefficient_bytes > storage_limit
            || neuron_bytes > limits.max_buffer_size
            || neuron_bytes > storage_limit
        {
            return Err(NyError::UnsupportedOp(
                "standalone alpha-gradient buffers exceed device limits".into(),
            ));
        }
        let workgroups = num_neurons_u32.div_ceil(256);
        if limits.max_compute_workgroups_per_dimension == 0
            || workgroups > limits.max_compute_workgroups_per_dimension
        {
            return Err(NyError::UnsupportedOp(format!(
                "standalone alpha-gradient dispatch {workgroups} exceeds device limit {}",
                limits.max_compute_workgroups_per_dimension
            )));
        }
        self.run_gpu_checked("crown_alpha_gradient_resident", || {
            let storage = |label: &str, n: usize| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (n.max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let a_buf = storage("grad_a", a_lower.len());
            let pl_buf = storage("grad_pl", num_neurons);
            let g_buf = storage("grad_out", num_neurons);
            self.queue
                .write_buffer(&a_buf, 0, bytemuck::cast_slice(a_lower));
            self.queue
                .write_buffer(&pl_buf, 0, bytemuck::cast_slice(pre_lower));
            let params = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grad_params"),
                size: size_of::<GradAlphaParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(
                &params,
                0,
                bytemuck::bytes_of(&GradAlphaParams {
                    num_specs: num_specs_u32,
                    num_neurons: num_neurons_u32,
                    // 0 = single-domain full reduction (legacy standalone entry).
                    num_specs_per_dom: 0,
                    _p1: 0,
                }),
            );
            let pipe = &self.resident_backward_pipelines().alpha_grad;
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("grad_enc"),
                });
            self.pass_simple(
                &mut enc,
                pipe,
                &params,
                &[&a_buf, &pl_buf, &g_buf],
                workgroups,
            );
            let st = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grad_stage"),
                size: (num_neurons * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&g_buf, 0, &st, 0, (num_neurons * size_of::<f32>()) as u64);
            self.queue.submit(Some(enc.finish()));
            Self::read_buffer(&self.device, &st, num_neurons)
        })
    }

    /// ON-DEVICE TRUE joint α-gradient (design doc §3): the coefficient-channel
    /// forward fold + hand-derived reverse-mode adjoint of `ny_core::joint_alpha_grad`
    /// (the FD-proven CPU oracle), run entirely on device so the correct joint
    /// gradient no longer pays the per-domain CPU re-fold (task #39). Returns
    /// `∂(lower_bound)/∂α` per ReLU in FOLD order (one `Vec<f32>` of length
    /// `num_neurons` per `Activation`), matching the CPU oracle's semantics and order
    /// exactly.
    ///
    /// Single-domain (`num_specs` = this domain's spec rows; all rows reduced into
    /// one gradient), matching the per-domain CPU call in `gpu_beta_optimize_wide`.
    /// `seed_lower_a` is the shared spec seed (num_specs × output_dim, row-major);
    /// `input_lower/upper` this domain's input box; the per-domain α is baked into
    /// the `Activation` layers' `lower_slope`.
    ///
    /// **NON-soundness-critical.** The gradient only proposes the next α∈[0,1]; the
    /// verdict bound is always the sound fold (design doc §4). So this drops the
    /// certified-error channel (safe) and, like the CPU oracle, tracks only the
    /// lower coefficient (no bias accumulator — the adjoint needs neither).
    ///
    /// Knobs: `NY_WIDE_ALPHA_NOBIAS=1` drops the adjoint bias channel (the ~0.7×
    /// degradation A/B). `NY_WIDE_ALPHA_ADJ_DEPTH=D` caps the number of ReLUs (from
    /// the INPUT side, where joint ≠ local matters most) harvested with the true
    /// adjoint; deeper output-side ReLUs get gradient 0 (α frozen this iteration) —
    /// a sound compute/memory truncation. Unset = full joint (all ReLUs).
    ///
    /// Returns `Err(UnsupportedOp)` on a topology the wide α path gates out
    /// (dual-alpha / maxpool) so the caller falls back to the CPU oracle (still the
    /// correct gradient) or the local rule — never unsound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_joint_alpha_gradient_resident(
        &self,
        segments: &[GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.crown_joint_alpha_gradient_resident_impl(
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_joint_alpha_gradient_resident_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        deadline: std::time::Instant,
    ) -> Result<Vec<Vec<f32>>> {
        self.crown_joint_alpha_gradient_resident_impl(
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            Some(deadline),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn crown_joint_alpha_gradient_resident_impl(
        &self,
        segments: &[GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<Vec<f32>>> {
        if num_specs == 0 || output_dim == 0 {
            return Err(NyError::InvalidSpec("joint grad: empty spec/output".into()));
        }
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "joint grad: empty segment list".into(),
            ));
        }
        let seed_elems =
            resident_checked_product(&[num_specs, output_dim], "joint-gradient seed elements")?;
        resident_checked_u32(seed_elems, "joint-gradient seed elements")?;
        if seed_lower_a.len() != seed_elems {
            return Err(NyError::shape_mismatch(
                vec![num_specs, output_dim],
                vec![seed_lower_a.len()],
            ));
        }
        let input_dim = input_lower.len();
        if input_dim == 0 || input_upper.len() != input_dim {
            return Err(NyError::shape_mismatch(
                vec![input_dim],
                vec![input_upper.len()],
            ));
        }
        let limits = self.device.limits();
        let final_dim = joint_segment_preflight(
            segments,
            num_specs,
            output_dim,
            limits.max_compute_workgroups_per_dimension,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        if final_dim != input_dim {
            return Err(NyError::shape_mismatch(vec![input_dim], vec![final_dim]));
        }
        let bias_channel = std::env::var("NY_WIDE_ALPHA_NOBIAS").ok().as_deref() != Some("1");
        let adj_depth = std::env::var("NY_WIDE_ALPHA_ADJ_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        Self::poll_joint_alpha_deadline(deadline)?;
        let run = || {
            Self::poll_joint_alpha_deadline(deadline)?;
            let jp = self.joint_adjoint_pipelines();

            // ---- forward fold (lower coefficient only; capture per-ReLU A_preᵏ) ----
            let a0 = self.joint_data_buf(seed_lower_a);
            let mut relu_caps: Vec<JointReluCap> = Vec::new();
            let mut a = a0;
            let mut dim = output_dim;
            for seg in segments {
                let (na, nd) = Self::run_joint_alpha_deadline_unit(deadline, || {
                    self.joint_fwd_segment(jp, seg, a, num_specs, dim, &mut relu_caps, deadline)
                })?;
                a = na;
                dim = nd;
            }
            if dim != input_dim {
                return Err(NyError::shape_mismatch(vec![input_dim], vec![dim]));
            }
            let a_input = a; // folded input-level coefficient A⁰

            // ---- seed the adjoint at the input box: ξ (design doc §2 terminal) ----
            let in_lo_buf = self.joint_data_buf(input_lower);
            let in_hi_buf = self.joint_data_buf(input_upper);
            let abar0 = self.joint_buf(num_specs * input_dim);
            {
                let params = self.joint_uniform(&JointU4 {
                    a: num_specs as u32,
                    b: input_dim as u32,
                    c: 0,
                    d: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_xi"),
                    });
                self.pass_simple(
                    &mut enc,
                    &jp.xi_seed,
                    &params,
                    &[&a_input, &in_lo_buf, &in_hi_buf, &abar0],
                    ((num_specs * input_dim) as u32).div_ceil(256),
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
            }

            // ---- adjoint pass (input→output), harvesting each ReLU's gradient ----
            let grad_bufs: Vec<(wgpu::Buffer, usize)> = relu_caps
                .iter()
                .map(|c| (self.joint_buf(c.nn), c.nn))
                .collect();
            let mut cursor = relu_caps.len();
            let mut harvested = 0usize;
            let _ = self.joint_adj_segments(
                jp,
                segments,
                abar0,
                num_specs,
                input_dim,
                &relu_caps,
                &mut cursor,
                &grad_bufs,
                bias_channel,
                adj_depth,
                &mut harvested,
                deadline,
            )?;
            if cursor != 0 {
                return Err(NyError::InvalidSpec(
                    "joint grad: ReLU record count mismatch".into(),
                ));
            }

            // ---- download the per-ReLU gradients (fold order) ----
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(grad_bufs.len());
            for (gb, n) in &grad_bufs {
                Self::poll_joint_alpha_deadline(deadline)?;
                let st = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("joint_grad_stage"),
                    size: ((*n).max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_grad_dl"),
                    });
                enc.copy_buffer_to_buffer(gb, 0, &st, 0, (*n * size_of::<f32>()) as u64);
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                Self::poll_joint_alpha_deadline(deadline)?;
                let values = Self::read_buffer(&self.device, &st, *n)?;
                Self::poll_joint_alpha_deadline(deadline)?;
                out.push(values);
            }
            Self::poll_joint_alpha_deadline(deadline)?;
            Ok(out)
        };
        match deadline {
            Some(value) => self.run_gpu_checked_with_deadline(
                "crown_joint_alpha_gradient_resident",
                value,
                run,
            ),
            None => self.run_gpu_checked("crown_joint_alpha_gradient_resident", run),
        }
    }

    #[inline]
    fn poll_joint_alpha_deadline(deadline: Option<std::time::Instant>) -> Result<()> {
        if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
            Err(NyError::DeadlineExceeded(
                "WGPU joint alpha adjoint exceeded its call-local deadline".into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Execute one scheduler unit with polls on both sides.  The post-unit poll
    /// is load-bearing: if a submitted/readback unit consumes the remaining
    /// budget, the next scripted unit (and therefore the whole tail) is never
    /// entered.
    fn run_joint_alpha_deadline_unit<T>(
        deadline: Option<std::time::Instant>,
        work: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        Self::poll_joint_alpha_deadline(deadline)?;
        let value = work()?;
        Self::poll_joint_alpha_deadline(deadline)?;
        Ok(value)
    }

    /// Fresh resident storage buffer of `n` f32 (zero-initialized by wgpu).
    fn joint_buf(&self, n: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_coeff"),
            size: (n.max(1) * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Storage buffer pre-filled with `data`.
    fn joint_data_buf(&self, data: &[f32]) -> wgpu::Buffer {
        let buf = self.joint_buf(data.len());
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    /// Uniform buffer holding one Pod value.
    fn joint_uniform<T: bytemuck::Pod>(&self, val: &T) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_uniform"),
            size: (size_of::<T>().max(16)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(val));
        buf
    }

    /// Forward fold one segment (coefficient channel only, design doc §1).
    fn joint_fwd_segment(
        &self,
        jp: &super::super::JointAdjointPipelines,
        seg: &GpuResnetSegment,
        a: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &mut Vec<JointReluCap>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        Self::poll_joint_alpha_deadline(deadline)?;
        match seg {
            GpuResnetSegment::Chain(layers) => {
                self.joint_fwd_chain(jp, layers, a, num_specs, dim, relu_caps, deadline)
            }
            GpuResnetSegment::Residual(f) => {
                // out = F(z) + z; skip = identity. A_in = A_skip + A_F.
                let a_skip = a.clone();
                let (a_f, dim_f) =
                    self.joint_fwd_chain(jp, f, a, num_specs, dim, relu_caps, deadline)?;
                if dim_f != dim {
                    return Err(NyError::shape_mismatch(vec![dim], vec![dim_f]));
                }
                let merge_elems = resident_checked_product(
                    &[num_specs, dim],
                    "joint forward residual merge elements",
                )?;
                let merged = self.joint_add(jp, &a_skip, &a_f, merge_elems, deadline)?;
                Ok((merged, dim))
            }
            GpuResnetSegment::ResidualProj(f, p) => {
                // out = F(z) + P(z). Fold F THEN P (matches CPU relu-cap order).
                let a_p_in = a.clone();
                let (a_f, dim_f) =
                    self.joint_fwd_chain(jp, f, a, num_specs, dim, relu_caps, deadline)?;
                let (a_p, dim_p) =
                    self.joint_fwd_chain(jp, p, a_p_in, num_specs, dim, relu_caps, deadline)?;
                if dim_f != dim_p {
                    return Err(NyError::shape_mismatch(vec![dim_f], vec![dim_p]));
                }
                let merge_elems = resident_checked_product(
                    &[num_specs, dim_f],
                    "joint forward projection merge elements",
                )?;
                let merged = self.joint_add(jp, &a_f, &a_p, merge_elems, deadline)?;
                Ok((merged, dim_f))
            }
        }
    }

    fn joint_fwd_chain(
        &self,
        jp: &super::super::JointAdjointPipelines,
        layers: &[GpuCrownLayer],
        a: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &mut Vec<JointReluCap>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        let mut cur = a;
        let mut cur_dim = dim;
        for layer in layers {
            Self::poll_joint_alpha_deadline(deadline)?;
            let (na, nd) =
                self.joint_fwd_layer(jp, layer, cur, num_specs, cur_dim, relu_caps, deadline)?;
            cur = na;
            cur_dim = nd;
        }
        Ok((cur, cur_dim))
    }

    fn joint_fwd_layer(
        &self,
        jp: &super::super::JointAdjointPipelines,
        layer: &GpuCrownLayer,
        a: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &mut Vec<JointReluCap>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        Self::poll_joint_alpha_deadline(deadline)?;
        match layer {
            GpuCrownLayer::Linear {
                weight,
                out_features,
                in_features,
                ..
            } => {
                let (of, if_) = (*out_features, *in_features);
                if of != dim || weight.len() != of * if_ {
                    return Err(NyError::shape_mismatch(vec![of, if_], vec![weight.len()]));
                }
                // A' = A @ W (A: num_specs×of, W: of×if_).
                // #lever1 weight residency: constant W is GPU-resident (uploaded
                // once, Arc-identity keyed + keep-alive; ops/resident_weights.rs).
                let w_buf = self.resident_weight_buf(weight, WeightForm::Raw)?;
                let out = self.joint_buf(num_specs * if_);
                let disp = select_gemm_dispatch(num_specs as u32, of as u32, if_ as u32);
                let pipe = if disp.use_small_k {
                    &self.gemm_f32_small_k_pipeline
                } else {
                    &self.gemm_f32_pipeline
                };
                let params = self.joint_uniform(&GemmParams {
                    m: num_specs as u32,
                    k: of as u32,
                    n: if_ as u32,
                    _padding: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_fwd_lin"),
                    });
                self.pass_gemm(
                    &mut enc, pipe, &params, &a, &w_buf, &out, disp.wg_x, disp.wg_y,
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                Ok((out, if_))
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
                ..
            } => {
                let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
                let (oh, ow, ih, iw) = (*out_h, *out_w, *in_h, *in_w);
                let out_d = oc * oh * ow;
                let in_d = ic * ih * iw;
                if out_d != dim || weight_col.len() != oc * ic * kh * kw {
                    return Err(NyError::shape_mismatch(vec![out_d], vec![dim]));
                }
                // #lever1 weight residency: resident constant conv weight.
                let w_buf = self.resident_weight_buf(weight_col, WeightForm::Raw)?;
                let out = self.joint_buf(num_specs * in_d);
                let params = self.joint_uniform(&JointConvParams {
                    num_specs: num_specs as u32,
                    oc: oc as u32,
                    ic: ic as u32,
                    oh: oh as u32,
                    ow: ow as u32,
                    ih: ih as u32,
                    iw: iw as u32,
                    kh: kh as u32,
                    kw: kw as u32,
                    sh: *stride_h as u32,
                    sw: *stride_w as u32,
                    ph: *pad_h as u32,
                    pw: *pad_w as u32,
                    has_bias: 0,
                    _p0: 0,
                    _p1: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_fwd_conv"),
                    });
                self.pass_simple(
                    &mut enc,
                    &jp.conv_t_fwd,
                    &params,
                    &[&a, &*w_buf, &out],
                    ((num_specs * in_d) as u32).div_ceil(256),
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                Ok((out, in_d))
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                num_neurons,
                ..
            } => {
                let nn = *num_neurons;
                if nn != dim || lower_slope.len() != nn || upper_slope.len() != nn {
                    return Err(NyError::shape_mismatch(vec![nn], vec![dim]));
                }
                let ls_buf = self.joint_data_buf(lower_slope);
                let us_buf = self.joint_data_buf(upper_slope);
                let out = self.joint_buf(num_specs * nn);
                let params = self.joint_uniform(&JointU4 {
                    a: num_specs as u32,
                    b: nn as u32,
                    c: 0,
                    d: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_fwd_relu"),
                    });
                // A' = A·σ ; the INCOMING `a` is the captured A_preᵏ (kept resident).
                self.pass_simple(
                    &mut enc,
                    &jp.relu_fwd,
                    &params,
                    &[&a, &ls_buf, &us_buf, &out],
                    ((num_specs * nn) as u32).div_ceil(256),
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                relu_caps.push(JointReluCap { a_pre: a, nn });
                Ok((out, nn))
            }
            // Gated out upstream by the wide α path — fall back to the CPU oracle.
            GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => Err(
                NyError::UnsupportedOp("joint grad: dual-alpha/maxpool".into()),
            ),
        }
    }

    /// Elementwise `out = x + y` over `n` elements (residual merge / fan-out sum).
    fn joint_add(
        &self,
        jp: &super::super::JointAdjointPipelines,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<wgpu::Buffer> {
        Self::poll_joint_alpha_deadline(deadline)?;
        let n_u32 = resident_checked_u32(n, "joint add elements")?;
        let out = self.joint_buf(n);
        let params = self.joint_uniform(&JointU4 {
            a: n_u32,
            b: 0,
            c: 0,
            d: 0,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("joint_add"),
            });
        self.pass_simple(
            &mut enc,
            &jp.add,
            &params,
            &[x, y, &out],
            n_u32.div_ceil(256),
        );
        Self::poll_joint_alpha_deadline(deadline)?;
        self.queue.submit(Some(enc.finish()));
        Ok(out)
    }

    /// Adjoint over segments walked in REVERSE (input→output, design doc §2).
    #[allow(clippy::too_many_arguments)]
    fn joint_adj_segments(
        &self,
        jp: &super::super::JointAdjointPipelines,
        segments: &[GpuResnetSegment],
        mut abar: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &[JointReluCap],
        cursor: &mut usize,
        grads: &[(wgpu::Buffer, usize)],
        bias_channel: bool,
        adj_depth: Option<usize>,
        harvested: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        let mut cur_dim = dim;
        for seg in segments.iter().rev() {
            Self::poll_joint_alpha_deadline(deadline)?;
            let (na, nd) = self.joint_adj_segment(
                jp,
                seg,
                abar,
                num_specs,
                cur_dim,
                relu_caps,
                cursor,
                grads,
                bias_channel,
                adj_depth,
                harvested,
                deadline,
            )?;
            abar = na;
            cur_dim = nd;
        }
        Ok((abar, cur_dim))
    }

    #[allow(clippy::too_many_arguments)]
    fn joint_adj_segment(
        &self,
        jp: &super::super::JointAdjointPipelines,
        seg: &GpuResnetSegment,
        abar: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &[JointReluCap],
        cursor: &mut usize,
        grads: &[(wgpu::Buffer, usize)],
        bias_channel: bool,
        adj_depth: Option<usize>,
        harvested: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        Self::poll_joint_alpha_deadline(deadline)?;
        match seg {
            GpuResnetSegment::Chain(layers) => self.joint_adj_chain(
                jp,
                layers,
                abar,
                num_specs,
                dim,
                relu_caps,
                cursor,
                grads,
                bias_channel,
                adj_depth,
                harvested,
                deadline,
            ),
            GpuResnetSegment::Residual(f) => {
                // Ā_out = Ā_in + adjoint_F(Ā_in) (skip fan-out).
                let abar_f_in = abar.clone();
                let (abar_f, dim_f) = self.joint_adj_chain(
                    jp,
                    f,
                    abar_f_in,
                    num_specs,
                    dim,
                    relu_caps,
                    cursor,
                    grads,
                    bias_channel,
                    adj_depth,
                    harvested,
                    deadline,
                )?;
                if dim_f != dim {
                    return Err(NyError::shape_mismatch(vec![dim], vec![dim_f]));
                }
                let merge_elems = resident_checked_product(
                    &[num_specs, dim],
                    "joint adjoint residual merge elements",
                )?;
                let out = self.joint_add(jp, &abar, &abar_f, merge_elems, deadline)?;
                Ok((out, dim))
            }
            GpuResnetSegment::ResidualProj(f, p) => {
                // Ā_out = adjoint_F(Ā_in) + adjoint_P(Ā_in). Consume P's records
                // BEFORE F's (reverse of the forward F-then-P fold order).
                let (abar_p, dim_p) = self.joint_adj_chain(
                    jp,
                    p,
                    abar.clone(),
                    num_specs,
                    dim,
                    relu_caps,
                    cursor,
                    grads,
                    bias_channel,
                    adj_depth,
                    harvested,
                    deadline,
                )?;
                let (abar_f, dim_f) = self.joint_adj_chain(
                    jp,
                    f,
                    abar,
                    num_specs,
                    dim,
                    relu_caps,
                    cursor,
                    grads,
                    bias_channel,
                    adj_depth,
                    harvested,
                    deadline,
                )?;
                if dim_f != dim_p {
                    return Err(NyError::shape_mismatch(vec![dim_f], vec![dim_p]));
                }
                let merge_elems = resident_checked_product(
                    &[num_specs, dim_f],
                    "joint adjoint projection merge elements",
                )?;
                let out = self.joint_add(jp, &abar_f, &abar_p, merge_elems, deadline)?;
                Ok((out, dim_f))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn joint_adj_chain(
        &self,
        jp: &super::super::JointAdjointPipelines,
        layers: &[GpuCrownLayer],
        mut abar: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &[JointReluCap],
        cursor: &mut usize,
        grads: &[(wgpu::Buffer, usize)],
        bias_channel: bool,
        adj_depth: Option<usize>,
        harvested: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        let mut cur_dim = dim;
        for layer in layers.iter().rev() {
            Self::poll_joint_alpha_deadline(deadline)?;
            let (na, nd) = self.joint_adj_layer(
                jp,
                layer,
                abar,
                num_specs,
                cur_dim,
                relu_caps,
                cursor,
                grads,
                bias_channel,
                adj_depth,
                harvested,
                deadline,
            )?;
            abar = na;
            cur_dim = nd;
        }
        Ok((abar, cur_dim))
    }

    #[allow(clippy::too_many_arguments)]
    fn joint_adj_layer(
        &self,
        jp: &super::super::JointAdjointPipelines,
        layer: &GpuCrownLayer,
        abar: wgpu::Buffer,
        num_specs: usize,
        dim: usize,
        relu_caps: &[JointReluCap],
        cursor: &mut usize,
        grads: &[(wgpu::Buffer, usize)],
        bias_channel: bool,
        adj_depth: Option<usize>,
        harvested: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<(wgpu::Buffer, usize)> {
        Self::poll_joint_alpha_deadline(deadline)?;
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                // Ā_out[s,i] = Σ_j Ā_in[s,j]·W[i,j] + bias[i]  (Ā_in dim = in_features).
                let dof = *out_features;
                let din = *in_features;
                if din != dim || weight.len() != dof * din {
                    return Err(NyError::shape_mismatch(vec![din], vec![dim]));
                }
                // Wᵀ: (din × dof), wt[j*dof+i] = weight[i*din+j].
                // #lever1 weight residency: the transpose is a pure permutation of
                // the constant weight, so it is derived + uploaded ONCE (Arc-identity
                // keyed with the dims in the key; ops/resident_weights.rs replicates
                // this exact layout) instead of CPU-transposed per call.
                let wt_buf =
                    self.resident_weight_buf(weight, WeightForm::Transposed { dof, din })?;
                let tmp = self.joint_buf(num_specs * dof);
                let disp = select_gemm_dispatch(num_specs as u32, din as u32, dof as u32);
                let pipe = if disp.use_small_k {
                    &self.gemm_f32_small_k_pipeline
                } else {
                    &self.gemm_f32_pipeline
                };
                let gparams = self.joint_uniform(&GemmParams {
                    m: num_specs as u32,
                    k: din as u32,
                    n: dof as u32,
                    _padding: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_adj_lin"),
                    });
                self.pass_gemm(
                    &mut enc, pipe, &gparams, &abar, &wt_buf, &tmp, disp.wg_x, disp.wg_y,
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                // + bias[i] (the bias channel) when present and enabled.
                match (bias_channel, bias) {
                    (true, Some(b)) => {
                        // #lever1: constant bias Arc — resident under Raw.
                        let b_buf = self.resident_weight_buf(b, WeightForm::Raw)?;
                        let out = self.joint_buf(num_specs * dof);
                        let params = self.joint_uniform(&JointU4 {
                            a: num_specs as u32,
                            b: dof as u32,
                            c: 0,
                            d: 0,
                        });
                        let mut e2 =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("joint_adj_lin_bias"),
                                });
                        self.pass_simple(
                            &mut e2,
                            &jp.rowvec_add,
                            &params,
                            &[&tmp, &*b_buf, &out],
                            ((num_specs * dof) as u32).div_ceil(256),
                        );
                        Self::poll_joint_alpha_deadline(deadline)?;
                        self.queue.submit(Some(e2.finish()));
                        Ok((out, dof))
                    }
                    _ => Ok((tmp, dof)),
                }
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                in_h,
                in_w,
                ..
            } => {
                // Ā_in dim = ic*ih*iw (incoming abar); Ā_out dim = oc*oh*ow.
                let (oc, ic, kh, kw) = (*out_channels, *in_channels, *kernel_h, *kernel_w);
                let (oh, ow, ih, iw) = (*out_h, *out_w, *in_h, *in_w);
                let in_d = ic * ih * iw;
                let out_d = oc * oh * ow;
                if in_d != dim || weight_col.len() != oc * ic * kh * kw {
                    return Err(NyError::shape_mismatch(vec![in_d], vec![dim]));
                }
                // #lever1 weight residency: resident constant conv weight + bias.
                let w_buf = self.resident_weight_buf(weight_col, WeightForm::Raw)?;
                let has_bias = bias_channel && bias_expanded.is_some();
                let b_buf = match (has_bias, bias_expanded) {
                    (true, Some(be)) => self.resident_weight_buf(be, WeightForm::Raw)?,
                    _ => Arc::new(self.joint_buf(out_d)), // inert (has_bias=0 ⇒ unread)
                };
                let out = self.joint_buf(num_specs * out_d);
                let params = self.joint_uniform(&JointConvParams {
                    num_specs: num_specs as u32,
                    oc: oc as u32,
                    ic: ic as u32,
                    oh: oh as u32,
                    ow: ow as u32,
                    ih: ih as u32,
                    iw: iw as u32,
                    kh: kh as u32,
                    kw: kw as u32,
                    sh: *stride_h as u32,
                    sw: *stride_w as u32,
                    ph: *pad_h as u32,
                    pw: *pad_w as u32,
                    has_bias: has_bias as u32,
                    _p0: 0,
                    _p1: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_adj_conv"),
                    });
                self.pass_simple(
                    &mut enc,
                    &jp.conv_adj,
                    &params,
                    &[&abar, &*w_buf, &*b_buf, &out],
                    ((num_specs * out_d) as u32).div_ceil(256),
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                Ok((out, out_d))
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                let nn = *num_neurons;
                if nn != dim || *cursor == 0 {
                    return Err(NyError::shape_mismatch(vec![nn], vec![dim]));
                }
                *cursor -= 1;
                let rec = &relu_caps[*cursor];
                if rec.nn != nn {
                    return Err(NyError::shape_mismatch(vec![rec.nn], vec![nn]));
                }
                // Harvest grad[i] = Σ_s Ā_out[s,i]·max(A_preᵏ[s,i],0), depth-capped.
                let do_harvest = match adj_depth {
                    Some(d) => *harvested < d,
                    None => true,
                };
                if do_harvest {
                    let hp = self.joint_uniform(&JointU4 {
                        a: num_specs as u32,
                        b: nn as u32,
                        c: 0,
                        d: 0,
                    });
                    let mut enc =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("joint_harvest"),
                            });
                    self.pass_simple(
                        &mut enc,
                        &jp.relu_harvest,
                        &hp,
                        &[&abar, &rec.a_pre, &grads[*cursor].0],
                        (nn as u32).div_ceil(256),
                    );
                    Self::poll_joint_alpha_deadline(deadline)?;
                    self.queue.submit(Some(enc.finish()));
                    *harvested += 1;
                }
                // Propagate Ā_in[s,i] = Ā_out[s,i]·σ + τ (τ = the bias channel).
                let ls = self.joint_data_buf(lower_slope);
                let us = self.joint_data_buf(upper_slope);
                let li = self.joint_data_buf(lower_intercept);
                let ui = self.joint_data_buf(upper_intercept);
                let out = self.joint_buf(num_specs * nn);
                let pp = self.joint_uniform(&JointU4 {
                    a: num_specs as u32,
                    b: nn as u32,
                    c: bias_channel as u32,
                    d: 0,
                });
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("joint_prop"),
                    });
                self.pass_simple(
                    &mut enc,
                    &jp.relu_prop,
                    &pp,
                    &[&abar, &rec.a_pre, &ls, &us, &li, &ui, &out],
                    ((num_specs * nn) as u32).div_ceil(256),
                );
                Self::poll_joint_alpha_deadline(deadline)?;
                self.queue.submit(Some(enc.finish()));
                Ok((out, nn))
            }
            GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => Err(
                NyError::UnsupportedOp("joint grad: dual-alpha/maxpool".into()),
            ),
        }
    }

    /// Borrow the ON-DEVICE joint α-gradient adjoint pipelines, compiling them once
    /// on first use (under the `gpu_serialize` lock) and caching them on the device.
    pub(in crate::wgpu_device) fn joint_adjoint_pipelines(
        &self,
    ) -> &super::super::JointAdjointPipelines {
        self.joint_adjoint_pipelines.get_or_init(|| {
            use super::super::shaders as sh;
            super::super::JointAdjointPipelines {
                xi_seed: self.create_simple_pipeline(
                    sh::JOINT_XI_SEED_SHADER,
                    "joint_xi_seed",
                    &[false, false, false, true],
                ),
                relu_fwd: self.create_simple_pipeline(
                    sh::JOINT_RELU_FWD_SHADER,
                    "joint_relu_fwd",
                    &[false, false, false, true],
                ),
                conv_t_fwd: self.create_simple_pipeline(
                    sh::JOINT_CONV_T_FWD_SHADER,
                    "joint_conv_t_fwd",
                    &[false, false, true],
                ),
                add: self.create_simple_pipeline(
                    sh::JOINT_ADD_SHADER,
                    "joint_add",
                    &[false, false, true],
                ),
                rowvec_add: self.create_simple_pipeline(
                    sh::JOINT_ROWVEC_ADD_SHADER,
                    "joint_rowvec_add",
                    &[false, false, true],
                ),
                relu_harvest: self.create_simple_pipeline(
                    sh::JOINT_RELU_HARVEST_SHADER,
                    "joint_relu_harvest",
                    &[false, false, true],
                ),
                relu_prop: self.create_simple_pipeline(
                    sh::JOINT_RELU_PROP_SHADER,
                    "joint_relu_prop",
                    &[false, false, false, false, false, false, true],
                ),
                conv_adj: self.create_simple_pipeline(
                    sh::JOINT_CONV_ADJ_SHADER,
                    "joint_conv_adj",
                    &[false, false, false, true],
                ),
            }
        })
    }

    /// Reset rows injected at one intermediate boundary on the active resident
    /// frontier. Queue writes are ordered after the just-submitted layer and
    /// before the next submit/readback. Every arithmetic/error/taint lane is
    /// overwritten, so work accumulated while a row was dormant is erased.
    #[allow(clippy::too_many_arguments)]
    fn apply_intermediate_sweep_boundary(
        &self,
        boundary: usize,
        dim: usize,
        num_specs: usize,
        la: &wgpu::Buffer,
        ua: &wgpu::Buffer,
        le: &wgpu::Buffer,
        ue: &wgpu::Buffer,
        blo: &wgpu::Buffer,
        buo: &wgpu::Buffer,
        ble: &wgpu::Buffer,
        bue: &wgpu::Buffer,
        taint: Option<&mut TaintWalkState>,
    ) -> Result<()> {
        let Some(scheduled) = super::intermediate_sweep::take_boundary(boundary, dim)? else {
            return Ok(());
        };
        if scheduled.resets.is_empty() {
            return Ok(());
        }
        let taint = taint.ok_or_else(|| {
            NyError::UnsupportedOp(
                "WGPU intermediate sweep requires the authoritative word-taint resident route"
                    .into(),
            )
        })?;
        let row_bytes = dim.checked_mul(size_of::<f32>()).ok_or_else(|| {
            NyError::InvalidSpec("WGPU intermediate sweep reset row byte overflow".into())
        })?;
        let mut identity = vec![0.0f32; dim];
        let zeros_f32 = vec![0.0f32; dim];
        let zeros_u32 = vec![0u32; dim];
        let zero_f32 = [0.0f32];
        let zero_u32 = [0u32];
        for (index, reset) in scheduled.resets.iter().enumerate() {
            if index.is_multiple_of(256) && self.crown_backward_deadline_expired() {
                return Err(NyError::DeadlineExceeded(
                    "WGPU intermediate sweep deadline exceeded while injecting rows".into(),
                ));
            }
            if reset.carrier_row >= num_specs || reset.coordinate >= dim {
                return Err(NyError::InternalError(format!(
                    "WGPU intermediate sweep reset ({}, {}) outside carrier ({num_specs}, {dim})",
                    reset.carrier_row, reset.coordinate
                )));
            }
            identity[reset.coordinate] = 1.0;
            let coeff_offset = reset
                .carrier_row
                .checked_mul(row_bytes)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or_else(|| {
                    NyError::InvalidSpec(
                        "WGPU intermediate sweep coefficient offset overflow".into(),
                    )
                })?;
            let bias_offset = reset
                .carrier_row
                .checked_mul(size_of::<f32>())
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or_else(|| {
                    NyError::InvalidSpec("WGPU intermediate sweep bias offset overflow".into())
                })?;
            self.queue
                .write_buffer(la, coeff_offset, bytemuck::cast_slice(&identity));
            self.queue
                .write_buffer(ua, coeff_offset, bytemuck::cast_slice(&identity));
            for buffer in [le, ue] {
                self.queue
                    .write_buffer(buffer, coeff_offset, bytemuck::cast_slice(&zeros_f32));
            }
            for buffer in [blo, buo, ble, bue] {
                self.queue
                    .write_buffer(buffer, bias_offset, bytemuck::cast_slice(&zero_f32));
            }
            // `ping` and the number of completed unary layers have identical
            // parity, so boundary parity selects the active word frontier.
            for buffer in [
                &taint.wla[boundary % 2],
                &taint.wua[boundary % 2],
                &taint.wle[boundary % 2],
                &taint.wue[boundary % 2],
            ] {
                self.queue
                    .write_buffer(buffer, coeff_offset, bytemuck::cast_slice(&zeros_u32));
            }
            self.queue.write_buffer(
                &taint.rows_dev,
                bias_offset,
                bytemuck::cast_slice(&zero_u32),
            );
            taint.rows[reset.carrier_row] = 0;
            identity[reset.coordinate] = 0.0;
            let bytes = dim
                .checked_mul(8)
                .and_then(|value| value.checked_add(5))
                .and_then(|value| value.checked_mul(size_of::<f32>()))
                .ok_or_else(|| {
                    NyError::InvalidSpec("WGPU intermediate sweep transfer byte overflow".into())
                })?;
            super::intermediate_sweep::note_host_to_device(bytes);
        }
        Ok(())
    }

    /// Borrow the always-built resident-backward pipelines, compiling them once on
    /// first use and caching them on the device for every later segment/sub-chain.
    /// These are pure compiled shader programs (no numerical data), so reusing them
    /// is bit-for-bit identical to building them fresh — it only removes redundant
    /// shader-module + pipeline compilation from the deep-resnet hot path. Built
    /// under the `gpu_serialize` lock (held by the calling `run_gpu_checked`), so the
    /// one-time initialization is single-threaded.
    pub(in crate::wgpu_device) fn resident_backward_pipelines(
        &self,
    ) -> &super::super::ResidentBackwardPipelines {
        // #u4: the widest twin (activation) needs 11 storage bindings; the
        // GRANTED limit is 8 when NY_GPU_BIG_BINDINGS=0 (and on adapters that
        // cap below 11). See the field docs on the struct.
        let taint_twins_supported = self.device.limits().max_storage_buffers_per_shader_stage >= 11;
        self.resident_pipelines
            .get_or_init(|| super::super::ResidentBackwardPipelines {
                abs: self.create_simple_pipeline(
                    super::super::shaders::ABS_COPY_SHADER,
                    "abs_copy",
                    &[false, true],
                ),
                combine: self.create_simple_pipeline(
                    super::super::shaders::CROWN_AW_ERROR_COMBINE_SHADER,
                    "aw_err_combine",
                    &[false, false, true, false],
                ),
                bias: self.create_simple_pipeline(
                    super::super::shaders::CROWN_BIAS_ERR_ACCUMULATE_SHADER,
                    "bias_err_acc",
                    &[false, false, false, true, true],
                ),
                act: self.create_simple_pipeline(
                    super::super::shaders::CROWN_ACTIVATION_RESIDENT_SHADER,
                    "act_resident",
                    &[false, false, false, false, true, true, false],
                ),
                act_bias: self.create_simple_pipeline(
                    super::super::shaders::CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER,
                    "act_intercept_bias",
                    &[false, false, false, false, true, true],
                ),
                eft_twin: self.create_simple_pipeline(
                    super::super::shaders::GEMM_F32_EFT_TWIN_SHADER,
                    "eft_twin_gemm",
                    &[false, false, true, true],
                ),
                eft_min_combine: self.create_simple_pipeline(
                    super::super::shaders::CROWN_EFT_MIN_COMBINE_SHADER,
                    "eft_min_combine",
                    // binding 7 (read) = s_prod, for the sentinel-stickiness guard.
                    &[false, false, false, false, true, false, false],
                ),
                eft_col2im: self.create_simple_pipeline(
                    super::super::shaders::CONV_COL2IM_EFT_TWIN_SHADER,
                    "eft_col2im_twin",
                    &[false, false, true, true],
                ),
                seg_merge: self.create_simple_pipeline(
                    super::super::shaders::RESIDENT_SEG_MERGE_SHADER,
                    "seg_merge",
                    &[true, true, false, false],
                ),
                conv_reshape: self.create_simple_pipeline(
                    super::super::shaders::CONV_RESHAPE_SHADER,
                    "conv_reshape",
                    &[false, true],
                ),
                conv_col2im: self.create_simple_pipeline(
                    super::super::shaders::CONV_COL2IM_SHADER,
                    "conv_col2im",
                    &[false, true],
                ),
                conv_err: self.create_simple_pipeline(
                    super::super::shaders::CROWN_CONV_ERROR_ROWMAX_SHADER,
                    "conv_err",
                    &[false, false, true],
                ),
                alpha_grad: self.create_simple_pipeline(
                    super::super::shaders::CROWN_ALPHA_GRADIENT_SHADER,
                    "crown_alpha_grad_capture",
                    &[false, false, true],
                ),
                // #u4 taint twins (AUTO or explicit NY_GPU_TAINT_WORDS=1):
                // rw-flag arrays copied verbatim from the probe authors — the GEMM/activation
                // twins from `sentinel_taint_selfcheck::dual_chain_run`, the
                // combine twin from `taint_chain.rs`, the min-combine consult
                // twin from `eft_min_combine_taint_probe.rs`. Built ONLY when
                // the granted limit can host the widest twin (act = 11 storage
                // bindings): a BGL validation error here would poison the
                // shared OnceCell cache and kill every gate-OFF walk too.
                gemm_taint: taint_twins_supported.then(|| {
                    // GEMM twin: a, b, out(rw), taint_a, taint_b, taint_out(rw).
                    self.create_simple_pipeline(
                        super::super::shaders::GEMM_F32_TAINT_SHADER,
                        "gemm_f32_taint",
                        &[false, false, true, false, false, true],
                    )
                }),
                gemm_small_k_taint: taint_twins_supported.then(|| {
                    // Exact-value twin of the large-M small-K schedule.
                    self.create_simple_pipeline(
                        super::super::shaders::GEMM_F32_SMALL_K_TAINT_SHADER,
                        "gemm_f32_small_k_taint",
                        &[false, false, true, false, false, true],
                    )
                }),
                conv_reshape_taint: taint_twins_supported.then(|| {
                    // src, dst(rw), source words, destination words(rw).
                    self.create_simple_pipeline(
                        super::super::shaders::CONV_RESHAPE_TAINT_SHADER,
                        "conv_reshape_taint",
                        &[false, true, false, true],
                    )
                }),
                conv_col2im_taint: taint_twins_supported.then(|| {
                    // GEMM values, dst(rw), GEMM words, destination words(rw).
                    self.create_simple_pipeline(
                        super::super::shaders::CONV_COL2IM_TAINT_SHADER,
                        "conv_col2im_taint",
                        &[false, true, false, true],
                    )
                }),
                act_taint: taint_twins_supported.then(|| {
                    // a_in, err_in, ls, us, a_out(rw), err_out(rw), beta,
                    // ta_in, te_in, ta_out(rw), te_out(rw) — 11 storage.
                    self.create_simple_pipeline(
                        super::super::shaders::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
                        "act_resident_taint",
                        &[
                            false, false, false, false, true, true, false, false, false, true, true,
                        ],
                    )
                }),
                combine_taint: taint_twins_supported.then(|| {
                    // s_prod, prop, err_out(rw), row_abs_a, taint_sprod_in,
                    // taint_prop_in, taint_e_out(rw).
                    self.create_simple_pipeline(
                        super::super::shaders::CROWN_AW_ERROR_COMBINE_TAINT_SHADER,
                        "aw_err_combine_taint",
                        &[false, false, true, false, false, false, true],
                    )
                }),
                eft_min_combine_taint: taint_twins_supported.then(|| {
                    // Audit C2: v_twin, r_in, value, prop, err_out(rw),
                    // row_abs_a, s_prod, taint_s, taint_p — no taint output
                    // binding (its refusal is in-shader).
                    self.create_simple_pipeline(
                        super::super::shaders::CROWN_EFT_MIN_COMBINE_TAINT_SHADER,
                        "eft_min_combine_taint",
                        &[false, false, false, false, true, false, false, false, false],
                    )
                }),
                // #u4 on-device word transport: 3 storage bindings, so built
                // unconditionally (see the field docs) and dispatched only
                // under the gate. The source-level G13 reseed stays dormant and
                // unbuilt: internal Conv births are captured at the exact op.
                // `rows_out` is `array<atomic<u32>>` in WGSL; atomics need
                // `read_write`, hence the single `true`.
                taint_row_or: self.create_simple_pipeline(
                    // words, partner, rows_out(rw, atomic).
                    super::super::shaders::TAINT_ROW_OR_SHADER,
                    "taint_row_or",
                    &[false, false, true],
                ),
            })
    }

    /// Lazily build the dense strided-gather kernel only when a caller actually
    /// requests more than [`LEGACY_BETA_GATHER_MAX_COPIES`] values. Bound-only
    /// CROWN and the established small β-gather lane never compile or dispatch it.
    pub(in crate::wgpu_device) fn resident_strided_gather_pipeline(
        &self,
    ) -> &(wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        self.resident_gather_pipeline.get_or_init(|| {
            self.create_simple_pipeline(
                super::super::shaders::CROWN_STRIDED_GATHER_SHADER,
                "crown_strided_gather",
                &[false, false, true],
            )
        })
    }

    /// Create a compute pipeline from WGSL with binding 0 = uniform params and
    /// bindings 1.. = storage (`rw[i]` true ⇒ read_write, false ⇒ read).
    ///
    /// `pub(super)` so the sound IBP forward driver (`ops/ibp_forward_sound.rs`)
    /// builds its sound pipelines through the same battle-tested helper.
    pub(super) fn create_simple_pipeline(
        &self,
        src: &str,
        label: &str,
        rw: &[bool],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = crate::wgpu_device::shader_loading::create_compute_module(
            &self.device,
            self.denorm_preserve_enabled(),
            label,
            src,
        );
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, &is_rw) in rw.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: !is_rw },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        (pipeline, layout)
    }

    /// Dispatch a `create_simple_pipeline` shader: binding 0 = params, 1.. =
    /// the given storage buffers, in its own compute pass (barrier vs neighbors).
    pub(in crate::wgpu_device) fn pass_simple(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
        params: &wgpu::Buffer,
        storage: &[&wgpu::Buffer],
        workgroups_x: u32,
    ) {
        super::intermediate_sweep::note_dispatches(1);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }];
        for (i, b) in storage.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_simple_bg"),
            layout: &pipe.1,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("res_simple_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.0);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups_x.max(1), 1, 1);
    }

    /// #fold-coalesce upload seam: arena-copy when coalescing (encoder-ordered,
    /// correct under a single submission), `queue.write_buffer` otherwise
    /// (submission-ordered — correct because the legacy path submits per layer).
    fn fold_upload(
        &self,
        arena: Option<&mut FoldStagingArena>,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        data: &[u8],
    ) -> Result<()> {
        super::intermediate_sweep::note_host_to_device(data.len());
        match arena {
            Some(a) => a.upload(encoder, dst, data),
            None => {
                self.queue.write_buffer(dst, 0, data);
                Ok(())
            }
        }
    }

    /// Like [`Self::pass_simple`] but with a 2-D workgroup grid (for tiled
    /// GEMM-shaped `create_simple_pipeline` shaders, e.g. the #eft-err twin).
    pub(in crate::wgpu_device) fn pass_simple_2d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
        params: &wgpu::Buffer,
        storage: &[&wgpu::Buffer],
        workgroups_x: u32,
        workgroups_y: u32,
    ) {
        super::intermediate_sweep::note_dispatches(1);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }];
        for (i, b) in storage.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_simple2d_bg"),
            layout: &pipe.1,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("res_simple2d_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.0);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups_x.max(1), workgroups_y.max(1), 1);
    }

    /// Dispatch the shared GEMM pipeline `out = a @ b` on the given buffers
    /// (binding 0 = GemmParams, 1 = a, 2 = b, 3 = out), in its own compute pass.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::wgpu_device) fn pass_gemm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipe: &wgpu::ComputePipeline,
        params: &wgpu::Buffer,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        wg_x: u32,
        wg_y: u32,
    ) {
        super::intermediate_sweep::note_dispatches(1);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_gemm_bg"),
            layout: &self.gemm_f32_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("res_gemm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_x.max(1), wg_y.max(1), 1);
    }
}

// CPU-only unit tests for the Certified Cut-CROWN stem-fold outward-rounding
// helpers (no GPU device required). Proves INV-C: a `sound_round=true` fold add
// widens the certified error OUTWARD and the realized (concretized) lower form
// never exceeds the exact f64 linear form.
#[cfg(test)]
mod stem_fold_rounding_tests {
    use super::{
        fold_add_lower_bias_outward, fold_add_lower_coeff_outward, merge_streams,
        resident_cut_fold_valid_for_activation, ResidentCoeff,
    };
    use ny_core::{f32_to_f64_exact, resident_cut_fold::ResidentCutFold};

    fn fold(
        coeffs: Vec<(u32, f32)>,
        bias_shift: f32,
        pre_coeffs: Vec<(u32, f32)>,
    ) -> ResidentCutFold {
        ResidentCutFold {
            coeffs,
            bias_shift,
            pre_coeffs,
            sound_round: true,
        }
    }

    #[test]
    fn resident_fold_validation_accepts_complete_finite_entry() {
        let entry = fold(vec![(0, 0.25), (2, -0.5)], -1.5, vec![(1, 0.75)]);
        assert!(resident_cut_fold_valid_for_activation(&entry, 3));
    }

    #[test]
    fn resident_fold_validation_rejects_post_index_out_of_bounds() {
        let entry = fold(vec![(3, 0.25)], -1.5, vec![(1, 0.75)]);
        assert!(!resident_cut_fold_valid_for_activation(&entry, 3));
    }

    #[test]
    fn resident_fold_validation_rejects_pre_index_out_of_bounds() {
        let entry = fold(vec![(0, 0.25)], -1.5, vec![(3, 0.75)]);
        assert!(!resident_cut_fold_valid_for_activation(&entry, 3));
    }

    #[test]
    fn resident_fold_validation_rejects_mixed_valid_and_invalid_entries() {
        let entry = fold(vec![(0, 0.25), (3, -0.5)], -1.5, vec![(1, 0.75)]);
        assert!(!resident_cut_fold_valid_for_activation(&entry, 3));
    }

    #[test]
    fn resident_fold_validation_rejects_every_nonfinite_channel() {
        let nonfinite_entries = [
            fold(vec![(0, f32::NAN)], -1.5, vec![(1, 0.75)]),
            fold(vec![(0, f32::INFINITY)], -1.5, vec![(1, 0.75)]),
            fold(vec![(0, 0.25)], f32::NEG_INFINITY, vec![(1, 0.75)]),
            fold(vec![(0, 0.25)], -1.5, vec![(1, f32::NAN)]),
        ];
        for entry in &nonfinite_entries {
            assert!(!resident_cut_fold_valid_for_activation(entry, 3));
        }
    }

    #[test]
    fn coeff_fold_widens_error_and_stays_below_exact() {
        // A value whose f64 sum is NOT representable exactly in f32 (so the
        // nearest-round has a non-zero gap that MUST be folded into err).
        let mut a = [1.0f32 / 3.0];
        let mut err = [0.0f32];
        let add = 1.0f32 / 7.0;
        let exact = f64::from(a[0]) + f64::from(add);
        fold_add_lower_coeff_outward(&mut a, &mut err, 0, add);
        // The error grew outward (INV-C) by at least the rounding gap.
        let gap = (f64::from(a[0]) - exact).abs();
        assert!(
            err[0] as f64 >= gap,
            "err must absorb the rounding gap outward"
        );
        // The nearest sum ± the certified err brackets the exact value: the
        // conservative lower use `a - err` never exceeds `exact`.
        assert!(f64::from(a[0]) - f64::from(err[0]) <= exact + 1e-12);
    }

    #[test]
    fn coeff_fold_zero_gap_leaves_error_unchanged() {
        // Exactly representable add ⇒ zero gap ⇒ err unchanged (no needless slack).
        let mut a = [0.5f32];
        let mut err = [0.0f32];
        fold_add_lower_coeff_outward(&mut a, &mut err, 0, 0.25);
        assert_eq!(a[0], 0.75);
        assert_eq!(err[0], 0.0);
    }

    #[test]
    fn bias_fold_rounds_down_and_widens_error() {
        let mut b = 1.0f32 / 3.0;
        let mut b_err = 0.0f32;
        let add = 1.0f32 / 7.0;
        let exact = f64::from(b) + f64::from(add);
        fold_add_lower_bias_outward(&mut b, &mut b_err, add);
        // Rounded DOWN (outward for a lower bias): b <= exact.
        assert!(
            f64::from(b) <= exact,
            "lower bias must round down (outward)"
        );
        assert!(b_err >= 0.0);
        // Final concretization form `b - b_err` never exceeds the exact bias.
        assert!(f64::from(b) - f64::from(b_err) <= exact);
    }

    #[test]
    fn stream_merge_charges_a_subnormal_publication_gap() {
        let tiny = f32::from_bits(1);
        let make = || ResidentCoeff {
            lower_a: vec![tiny],
            upper_a: vec![tiny],
            lower_err: vec![0.0],
            upper_err: vec![0.0],
            lower_b: vec![tiny],
            upper_b: vec![tiny],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim: 1,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            taint_rows: None,
        };
        let exact = 2.0 * f32_to_f64_exact(tiny);
        let merged = merge_streams(make(), &make());
        for (center, error) in [
            (merged.lower_a[0], merged.lower_err[0]),
            (merged.upper_a[0], merged.upper_err[0]),
            (merged.lower_b[0], merged.lower_b_err[0]),
            (merged.upper_b[0], merged.upper_b_err[0]),
        ] {
            let center = f32_to_f64_exact(center);
            let error = f32_to_f64_exact(error);
            assert!(center - error <= exact && exact <= center + error);
        }
    }
}

#[cfg(test)]
mod joint_deadline_scheduler_tests {
    use super::WgpuDevice;
    use ny_core::{NyError, Result};
    use std::time::{Duration, Instant};

    #[test]
    fn deadline_crossed_inside_scripted_unit_leaves_tail_unexecuted() {
        let deadline = Instant::now() + Duration::from_millis(10);
        let mut executed = Vec::new();
        let result: Result<()> = (|| {
            for unit in 0..4 {
                WgpuDevice::run_joint_alpha_deadline_unit(Some(deadline), || {
                    executed.push(unit);
                    if unit == 0 {
                        std::thread::sleep(Duration::from_millis(30));
                    }
                    Ok(())
                })?;
            }
            Ok(())
        })();

        assert!(
            matches!(result, Err(NyError::DeadlineExceeded(_))),
            "the scripted scheduler must return the cooperative deadline error"
        );
        assert_eq!(
            executed,
            vec![0],
            "units 1..3 are the unexecuted joint-adjoint tail"
        );
    }
}

// CPU-only unit tests for the #u4 concretize-error taint companion (no GPU
// device required): annihilation on an exactly-zero fab partner, OR
// accumulation into a pre-set bias word, the HOLE-4 per-domain fab block
// addressing cross-checked against the REAL value fold, and the mirrored
// no-op refusals.
/// #cert-err — the host-side arithmetic pins for the certified weight-error
/// charge. These need no GPU: they pin the exact uniform substitutions the walk
/// performs, which is where the whole soundness argument lives.
#[cfg(test)]
mod cert_err_charge_tests {
    use super::{
        cert_bias_charge_required, cert_bias_charge_slack, cert_charged_slack, combine_slack_f32,
        gamma_k_f32,
    };
    use ny_core::{CertifiedWeightError, GpuCrownLayer};
    use std::sync::Arc;

    const U: f64 = 5.960_464_477_539_063e-8; // 2^-24, the binary32 unit roundoff

    fn linear(cert_err: CertifiedWeightError) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(vec![1.0f32, 0.0, 0.0, 1.0]),
            bias: Some(Arc::from(vec![0.0f32, 0.0])),
            out_features: 2,
            in_features: 2,
            cert_err,
        }
    }

    fn conv(cert_err: CertifiedWeightError) -> GpuCrownLayer {
        GpuCrownLayer::Conv2d {
            weight_col: Arc::from(vec![1.0f32]),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 1,
            in_h: 1,
            in_w: 1,
            cert_err,
        }
    }

    /// THE NON-BREAKING PIN. Every pre-`cert_err` caller constructs the default
    /// (all-zero) declaration, and this asserts that such a walk substitutes
    /// BIT-IDENTICAL uniforms and allocates/dispatches no charge at all — i.e.
    /// the walk is byte-identical to the build before `cert_err` existed.
    #[test]
    fn zero_cert_err_charges_are_byte_identical() {
        let exact = CertifiedWeightError::default();
        assert_eq!(exact, CertifiedWeightError::EXACT);
        assert!(exact.is_exact() && exact.is_valid());

        for k in [1usize, 2, 7, 64, 512, 4096, 100_000] {
            let gamma = gamma_k_f32(k).expect("finite gamma");
            let slack = combine_slack_f32(k).expect("finite slack");
            let charged_gamma = exact.charged_gamma(gamma);
            assert_eq!(
                charged_gamma.to_bits(),
                gamma.to_bits(),
                "k={k}: zero cert_err must leave gamma_k bit-identical"
            );
            let charged_slack = cert_charged_slack(slack, exact, 0).expect("finite charged slack");
            assert_eq!(
                charged_slack.to_bits(),
                slack.to_bits(),
                "k={k}: zero cert_err must leave the combine slack bit-identical"
            );
        }

        // `-0.0` is still exact, and no charge buffers are demanded.
        let neg_zero = CertifiedWeightError {
            weight_rel_err: -0.0,
            bias_abs_err: -0.0,
        };
        assert!(neg_zero.is_exact());
        assert!(!cert_bias_charge_required(&[
            linear(CertifiedWeightError::default()),
            conv(neg_zero),
        ]));
    }

    /// A declared error must move the charge STRICTLY OUTWARD, and by at least
    /// the mathematically required `gamma + w` (the derivation in
    /// `CertifiedWeightError::charged_gamma`).
    #[test]
    fn nonzero_cert_err_charge_is_strictly_outward() {
        for k in [1usize, 64, 4096] {
            let gamma = gamma_k_f32(k).expect("finite gamma");
            let slack = combine_slack_f32(k).expect("finite slack");
            for w in [1e-7f32, 1e-5, 1e-3, 0.25] {
                let cert_err = CertifiedWeightError {
                    weight_rel_err: w,
                    bias_abs_err: 0.0,
                };
                let g = cert_err.charged_gamma(gamma);
                assert!(g > gamma, "k={k} w={w}: charge must widen gamma");
                assert!(
                    f64::from(g) >= f64::from(gamma) + f64::from(w),
                    "k={k} w={w}: charged gamma {g:e} must dominate gamma + w"
                );
                let charged = cert_charged_slack(slack, cert_err, 0).expect("finite");
                assert!(
                    f64::from(charged) >= f64::from(slack) * (1.0 + f64::from(w)),
                    "k={k} w={w}: charged slack must dominate slack*(1+w) — the \
                     (1+w) factor is what covers the PROPAGATED error term"
                );
                assert!(charged > slack, "k={k} w={w}: charge must widen the slack");
            }
        }
    }

    /// A meaningless declaration must saturate to `+inf` (a refusal) rather
    /// than wrap to some finite, under-charging factor.
    #[test]
    fn invalid_cert_err_saturates_and_refuses() {
        let gamma = gamma_k_f32(64).expect("finite gamma");
        for bad in [
            CertifiedWeightError {
                weight_rel_err: f32::NAN,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: f32::INFINITY,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: -1e-6,
                bias_abs_err: 0.0,
            },
            CertifiedWeightError {
                weight_rel_err: 0.0,
                bias_abs_err: f32::NAN,
            },
        ] {
            assert!(!bad.is_valid(), "{bad:?} must be rejected as a declaration");
            assert_eq!(
                bad.charged_gamma(gamma),
                f32::INFINITY,
                "{bad:?} must saturate outward"
            );
            assert!(
                cert_charged_slack(1.0, bad, 0).is_err(),
                "{bad:?} must refuse a charged slack"
            );
        }
        assert!(
            CertifiedWeightError::default()
                .charged_gamma(f32::NAN)
                .is_infinite(),
            "a non-finite base gamma must saturate too"
        );
    }

    /// The extra bias dispatch runs with `gamma_k = 1`, so its `slack` is the
    /// ONLY thing recovering the two f32 reductions' undercount. Pin that it
    /// dominates the `1/(1 - gamma_{k+1})` the derivation demands.
    #[test]
    fn cert_bias_charge_slack_dominates_the_reduction_undercount() {
        for k in [1usize, 2, 33, 1024, 65_536] {
            let slack = f64::from(cert_bias_charge_slack(k).expect("finite bias charge slack"));
            let terms = (k + 1) as f64;
            let gamma_next = terms * U / (1.0 - terms * U);
            let required = 1.0 / (1.0 - gamma_next);
            assert!(
                slack >= required,
                "k={k}: bias charge slack {slack:.17e} must dominate {required:.17e}"
            );
            assert!(slack >= 1.0, "k={k}: slack must never shrink a radius");
        }
        // Reduction lengths whose gamma has no finite recovery fail closed.
        assert!(cert_bias_charge_slack(usize::MAX).is_err());
    }

    /// `cert_bias_charge_required` is the allocation trigger: it must fire on a
    /// nonzero `bias_abs_err` in EITHER variant, and only then.
    #[test]
    fn bias_charge_trigger_tracks_the_declaration() {
        let with_bias_err = CertifiedWeightError {
            weight_rel_err: 0.0,
            bias_abs_err: 1e-6,
        };
        let weight_only = CertifiedWeightError {
            weight_rel_err: 1e-6,
            bias_abs_err: 0.0,
        };
        assert!(cert_bias_charge_required(&[linear(with_bias_err)]));
        assert!(cert_bias_charge_required(&[conv(with_bias_err)]));
        assert!(!cert_bias_charge_required(&[
            linear(weight_only),
            conv(weight_only)
        ]));
        assert!(!cert_bias_charge_required(&[]));
    }
}

#[cfg(test)]
mod concretize_error_taint_tests {
    use super::{ResidentCoeff, WgpuDevice};

    /// `fab == 0.0` (either sign of zero) is a proven exactly-zero
    /// pre-activation: the err-taint word must annihilate instead of reaching
    /// the bias word (canon: `R·0 == 0`).
    #[test]
    fn exactly_zero_fab_annihilates_err_taint() {
        let err_taint = vec![0xdead_beef_u32, 0x1, 0x2];
        let fab = vec![0.0f32, -0.0, 0.0];
        let mut bias_taint = vec![0u32];
        WgpuDevice::concretize_error_taint_into_bias(&err_taint, &fab, &mut bias_taint, 1, 1, 3);
        assert_eq!(bias_taint, vec![0], "exact-zero partners must annihilate");
    }

    /// A nonzero fab partner carries the word, and the fold ORs INTO the
    /// existing bias word (words are provenance bits — accumulated, never
    /// overwritten).
    #[test]
    fn nonzero_fab_ors_words_and_accumulates() {
        let err_taint = vec![0x4u32, 0x0, 0x10];
        let fab = vec![1.5f32, 2.0, 0.25];
        let mut bias_taint = vec![0x1u32];
        WgpuDevice::concretize_error_taint_into_bias(&err_taint, &fab, &mut bias_taint, 1, 1, 3);
        assert_eq!(bias_taint, vec![0x1 | 0x4 | 0x10]);
    }

    /// HOLE-4 indexing, cross-checked against the REAL fold on the same
    /// inputs: 2 spec rows in 2 one-row domains, `d = 3`, `fab` = 2 stacked
    /// per-domain blocks. With every err element `1.0` and a unique word per
    /// element, the value fold's per-spec bias err (≈ Σ of ITS domain's
    /// nonzero fab entries) pins exactly which partners contributed — the
    /// companion's word must be the OR of exactly those elements' bits.
    /// Sharing domain 0's block across domain 1's rows (the under-count HOLE 4
    /// forbids) would show up here as bit 3 leaking into spec row 1.
    #[test]
    fn per_domain_fab_block_addressing_matches_real_fold() {
        let (num_specs, per_dom, d) = (2usize, 1usize, 3usize);
        // dom 0 block: [1.0, 0.0, 2.0] — dom 1 block: [0.0, 4.0, 0.5].
        let fab = vec![1.0f32, 0.0, 2.0, 0.0, 4.0, 0.5];
        let err_taint: Vec<u32> = (0..num_specs * d).map(|i| 1u32 << i).collect();

        // Real fold on the same shape: err = 1.0 everywhere, so each spec row's
        // bias err is (up to outward rounding) the sum of ITS fab block.
        let mut coeff = ResidentCoeff {
            lower_a: vec![0.0; num_specs * d],
            upper_a: vec![0.0; num_specs * d],
            lower_err: vec![1.0; num_specs * d],
            upper_err: vec![1.0; num_specs * d],
            lower_b: vec![0.0; num_specs],
            upper_b: vec![0.0; num_specs],
            lower_b_err: vec![0.0; num_specs],
            upper_b_err: vec![0.0; num_specs],
            taint_rows: None,
            dim: d,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
        };
        WgpuDevice::concretize_error_into_bias(&mut coeff, num_specs, per_dom, &fab);
        // Spec 0 folded against dom 0 (1.0 + 0.0 + 2.0), spec 1 against dom 1
        // (0.0 + 4.0 + 0.5) — proving which partner each element multiplied.
        assert!((f64::from(coeff.lower_b_err[0]) - 3.0).abs() < 1e-5);
        assert!((f64::from(coeff.lower_b_err[1]) - 4.5).abs() < 1e-5);
        assert!(coeff.lower_err.iter().all(|&e| e == 0.0), "err reset to 0");

        let mut bias_taint = vec![0x100u32, 0x200];
        WgpuDevice::concretize_error_taint_into_bias(
            &err_taint,
            &fab,
            &mut bias_taint,
            num_specs,
            per_dom,
            d,
        );
        // Spec 0: elements 0 (fab 1.0) and 2 (fab 2.0) survive; element 1
        // annihilates (fab 0.0). Spec 1: elements 4 and 5 survive; element 3
        // annihilates against DOMAIN 1's fab[3] = 0.0 — bit 3 present would
        // mean the companion consulted domain 0's (nonzero) block.
        assert_eq!(bias_taint[0], 0x100 | (1 << 0) | (1 << 2));
        assert_eq!(bias_taint[1], 0x200 | (1 << 4) | (1 << 5));
    }

    /// The companion mirrors the real fold's no-op refusals exactly: malformed
    /// fab (NaN / +inf / negative), shape mismatch, and a degenerate domain
    /// partition all leave the bias words untouched (in those cases the real
    /// fold leaves the per-coefficient err unzeroed, so the err-taint words
    /// stay live beside it — nothing is laundered).
    #[test]
    fn mirrors_real_fold_no_op_refusals() {
        let err_taint = vec![0xfu32, 0xf, 0xf];
        let cases: [(&[f32], usize, usize, usize); 6] = [
            (&[f32::NAN, 1.0, 1.0], 1, 1, 3),      // NaN fab
            (&[f32::INFINITY, 1.0, 1.0], 1, 1, 3), // +inf fab
            (&[-1.0, 1.0, 1.0], 1, 1, 3),          // negative fab
            (&[1.0, 1.0], 1, 1, 3),                // fab length mismatch
            (&[1.0, 1.0, 1.0], 1, 0, 3),           // zero-row domains
            // fab fits 2 domains × d=3, but err_taint (len 3) ≠ num_specs·d = 6.
            (&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2, 1, 3),
        ];
        for (fab, num_specs, per_dom, d) in cases {
            let mut bias_taint = vec![0u32; num_specs];
            WgpuDevice::concretize_error_taint_into_bias(
                &err_taint,
                fab,
                &mut bias_taint,
                num_specs,
                per_dom,
                d,
            );
            assert!(
                bias_taint.iter().all(|&w| w == 0),
                "refusal case ({fab:?}, {num_specs}, {per_dom}, {d}) must no-op"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::wgpu_device::test_support::{
        gpu_test_serial_guard, require_device, require_verdict_device,
    };
    // Blessed env-mutation choke point (clippy env wall): all env writes in
    // these tests are ScopedEnvVar guards, serialized by gpu_test_serial_guard.
    use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuResnetBatchedDomainRef};
    use ny_test_utils::env::ScopedEnvVar;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    /// Concretize a raw, explicitly unworded coefficient frontier for arithmetic
    /// tests only. This is not a verdict seam: it uses directed host arithmetic
    /// and requires the frontier to carry no C1 receipt. Production callers must
    /// use `concretize_resident_coeff`, whose armed consult rejects this frontier.
    fn concretize_unworded_test_frontier(
        coeff: &ResidentCoeff,
        num_specs: usize,
        num_specs_per_dom: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        assert!(
            coeff.taint_rows.is_none(),
            "unworded arithmetic helper must never consume a C1 receipt"
        );
        assert!(num_specs_per_dom > 0);
        assert!(num_specs.is_multiple_of(num_specs_per_dom));
        let n_domains = num_specs / num_specs_per_dom;
        assert_eq!(input_lower.len(), n_domains * coeff.dim);
        assert_eq!(input_upper.len(), n_domains * coeff.dim);
        for field in [
            &coeff.lower_a,
            &coeff.upper_a,
            &coeff.lower_err,
            &coeff.upper_err,
        ] {
            assert_eq!(field.len(), num_specs * coeff.dim);
        }
        for field in [
            &coeff.lower_b,
            &coeff.upper_b,
            &coeff.lower_b_err,
            &coeff.upper_b_err,
        ] {
            assert_eq!(field.len(), num_specs);
        }

        let next_down = |x: f64| -next_up_f64(-x);
        let mut lower = Vec::with_capacity(num_specs);
        let mut upper = Vec::with_capacity(num_specs);
        for s in 0..num_specs {
            let dom = s / num_specs_per_dom;
            let xbase = dom * coeff.dim;
            let abase = s * coeff.dim;
            let mut lo = next_down(
                f32_to_f64_exact(coeff.lower_b[s]) - f32_to_f64_exact(coeff.lower_b_err[s]),
            );
            let mut hi = next_up_f64(
                f32_to_f64_exact(coeff.upper_b[s]) + f32_to_f64_exact(coeff.upper_b_err[s]),
            );
            for j in 0..coeff.dim {
                let xl = f32_to_f64_exact(input_lower[xbase + j]);
                let xu = f32_to_f64_exact(input_upper[xbase + j]);
                let la = f32_to_f64_exact(coeff.lower_a[abase + j]);
                let le = f32_to_f64_exact(coeff.lower_err[abase + j]);
                let ua = f32_to_f64_exact(coeff.upper_a[abase + j]);
                let ue = f32_to_f64_exact(coeff.upper_err[abase + j]);
                let lprod = [
                    (la - le) * xl,
                    (la - le) * xu,
                    (la + le) * xl,
                    (la + le) * xu,
                ]
                .into_iter()
                .fold(f64::INFINITY, f64::min);
                let uprod = [
                    (ua - ue) * xl,
                    (ua - ue) * xu,
                    (ua + ue) * xl,
                    (ua + ue) * xu,
                ]
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
                lo = next_down(lo + lprod);
                hi = next_up_f64(hi + uprod);
            }
            lower.push(down_f32(lo));
            upper.push(up_f32(hi));
        }
        (lower, upper)
    }

    #[allow(clippy::too_many_arguments)]
    fn unworded_resident_test_bounds(
        device: &WgpuDevice,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let zero_bias = vec![0.0; num_specs];
        let coeff = device.crown_backward_sound_resident_coeff_seeded(
            layers, spec, spec, &zero_bias, &zero_bias, num_specs, output_dim,
        )?;
        Ok(concretize_unworded_test_frontier(
            &coeff,
            num_specs,
            num_specs,
            input_lower,
            input_upper,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn unworded_resident_test_chunked(
        device: &WgpuDevice,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        chunk: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<ny_core::GpuCrownResult> {
        let mut lower_bounds = Vec::with_capacity(num_specs);
        let mut upper_bounds = Vec::with_capacity(num_specs);
        let mut start = 0;
        while start < num_specs {
            let end = (start + chunk.max(1)).min(num_specs);
            let (lo, hi) = unworded_resident_test_bounds(
                device,
                layers,
                &spec[start * output_dim..end * output_dim],
                end - start,
                output_dim,
                input_lower,
                input_upper,
            )?;
            lower_bounds.extend(lo);
            upper_bounds.extend(hi);
            start = end;
        }
        Ok(ny_core::GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn unworded_resnet_test_bounds(
        device: &WgpuDevice,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<ny_core::GpuCrownResult> {
        let internal: Vec<ResnetSegment<'_>> = segments
            .iter()
            .map(|segment| match segment {
                GpuResnetSegment::Chain(layers) => ResnetSegment::Chain(layers),
                GpuResnetSegment::Residual(layers) => ResnetSegment::Residual(layers),
                GpuResnetSegment::ResidualProj(branch, projection) => {
                    ResnetSegment::ResidualProj(branch, projection)
                }
            })
            .collect();
        let beta_refs: Vec<&[f32]> = beta_signed.iter().map(Vec::as_slice).collect();
        let frontier_refs: Vec<&[f32]> = frontier_abs.iter().map(Vec::as_slice).collect();
        let node_refs: Vec<&[f32]> = node_abs.iter().map(Vec::as_slice).collect();
        let (coeff, _grads, _gathers) = device.resnet_seeded_compose_coeff(
            &internal,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.num_specs,
            seed.current_dim,
            &[],
            &beta_refs,
            &[],
            &frontier_refs,
            false,
            &node_refs,
            false,
        )?;
        let (lower_bounds, upper_bounds) = concretize_unworded_test_frontier(
            &coeff,
            seed.num_specs,
            seed.num_specs,
            input_lower,
            input_upper,
        );
        Ok(ny_core::GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    #[test]
    fn worded_conv_runtime_returns_clean_receipt() {
        let _g = gpu_test_serial_guard();
        let _words = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let _rowmax = ScopedEnvVar::unset("NY_CONV_ERR_ROWMAX");
        let device = require_verdict_device();
        let layers = [GpuCrownLayer::Conv2d {
            weight_col: vec![1.0].into(),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 1,
            in_h: 1,
            in_w: 1,
            cert_err: Default::default(),
        }];
        let coeff = device
            .crown_backward_sound_resident_coeff_seeded(
                &layers,
                &[1.0],
                &[1.0],
                &[0.0],
                &[0.0],
                1,
                1,
            )
            .expect("the internally worded Conv route must be admitted");
        assert_eq!(coeff.lower_a[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(coeff.upper_a[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(coeff.taint_rows, Some(vec![0]));
    }

    #[test]
    fn worded_conv_gate_preserves_asymmetric_value_bits() {
        let _g = gpu_test_serial_guard();
        let _eft = ScopedEnvVar::unset("NY_EFT_ERR");
        let _rowmax = ScopedEnvVar::unset("NY_CONV_ERR_ROWMAX");
        let device = require_verdict_device();
        let layers = [GpuCrownLayer::Conv2d {
            weight_col: vec![0.5, -0.25, 0.75, 0.125].into(),
            bias_expanded: None,
            out_channels: 2,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 2,
            in_h: 1,
            in_w: 3,
            cert_err: Default::default(),
        }];
        let lower = [0.25, -0.5, 0.75, -1.0, -0.125, 0.375, -0.625, 0.875];
        let upper = [0.5, -0.25, 1.0, -0.75, 0.125, 0.625, -0.375, 1.125];
        let zeros = [0.0f32; 2];
        let run = |gate: &str| {
            let _words = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", gate);
            device
                .crown_backward_sound_resident_coeff_seeded(
                    &layers, &lower, &upper, &zeros, &zeros, 2, 4,
                )
                .expect("asymmetric Conv word route")
        };
        let off = run("0");
        let on = run("1");
        for (lhs, rhs) in [
            (&off.lower_a, &on.lower_a),
            (&off.upper_a, &on.upper_a),
            (&off.lower_err, &on.lower_err),
            (&off.upper_err, &on.upper_err),
            (&off.lower_b, &on.lower_b),
            (&off.upper_b, &on.upper_b),
            (&off.lower_b_err, &on.lower_b_err),
            (&off.upper_b_err, &on.upper_b_err),
        ] {
            assert_eq!(
                lhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(off.taint_rows, None);
        assert_eq!(on.taint_rows, Some(vec![0, 0]));
    }

    #[test]
    fn worded_conv_catches_gemm_saturation_cancelled_by_col2im() {
        let _g = gpu_test_serial_guard();
        let _words = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let _eft = ScopedEnvVar::unset("NY_EFT_ERR");
        let _rowmax = ScopedEnvVar::unset("NY_CONV_ERR_ROWMAX");
        let device = require_verdict_device();
        let layers = [GpuCrownLayer::Conv2d {
            weight_col: vec![20.0, 20.0].into(),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 2,
            in_h: 1,
            in_w: 3,
            cert_err: Default::default(),
        }];
        // Neither seed is tainted. Each Conv GEMM output saturates, and the
        // two contributions at input position 1 then cancel to exactly zero.
        let coeff = device
            .crown_backward_sound_resident_coeff_seeded(
                &layers,
                &[1.0e9, -1.0e9],
                &[1.0e9, -1.0e9],
                &[0.0],
                &[0.0],
                1,
                2,
            )
            .expect("internally saturated Conv must complete with a tainted receipt");
        assert_eq!(coeff.lower_a[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(coeff.upper_a[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(coeff.taint_rows, Some(vec![1]));
    }

    #[test]
    fn small_k_taint_twin_matches_base_bits_and_carries_words() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let (m, k, n) = (3u32, 3u32, 2u32);
        let params = GemmParams {
            m,
            k,
            n,
            _padding: 0,
        };
        let a = [1.25f32, -2.0, 0.5, -0.75, 4.0, 2.5, 3.0, -1.0, 0.25];
        let b = [0.5f32, -1.0, 2.0, 0.25, -0.5, 3.0];
        let ta = [0u32, 1, 0, 0, 0, 0, 0, 0, 0];
        let tb = [0u32; 6];
        let init = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage,
                })
        };
        let pbuf = init(
            "small_k_taint_params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let abuf = init(
            "small_k_taint_a",
            bytemuck::cast_slice(&a),
            wgpu::BufferUsages::STORAGE,
        );
        let bbuf = init(
            "small_k_taint_b",
            bytemuck::cast_slice(&b),
            wgpu::BufferUsages::STORAGE,
        );
        let tabuf = init(
            "small_k_taint_wa",
            bytemuck::cast_slice(&ta),
            wgpu::BufferUsages::STORAGE,
        );
        let tbbuf = init(
            "small_k_taint_wb",
            bytemuck::cast_slice(&tb),
            wgpu::BufferUsages::STORAGE,
        );
        let scratch = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(m * n) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let base = scratch("small_k_base_out");
        let twin = scratch("small_k_twin_out");
        let words = scratch("small_k_twin_words");
        let stage = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(m * n) * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let base_stage = stage("small_k_base_stage");
        let twin_stage = stage("small_k_twin_stage");
        let word_stage = stage("small_k_word_stage");
        let mut enc = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("small_k_taint_test"),
            });
        device.pass_gemm(
            &mut enc,
            &device.gemm_f32_small_k_pipeline,
            &pbuf,
            &abuf,
            &bbuf,
            &base,
            1,
            1,
        );
        let pipe = device
            .resident_backward_pipelines()
            .gemm_small_k_taint
            .as_ref()
            .expect("gpu-tests require the word twin binding limit");
        device.pass_simple_2d(
            &mut enc,
            pipe,
            &pbuf,
            &[&abuf, &bbuf, &twin, &tabuf, &tbbuf, &words],
            1,
            1,
        );
        let bytes = u64::from(m * n) * 4;
        enc.copy_buffer_to_buffer(&base, 0, &base_stage, 0, bytes);
        enc.copy_buffer_to_buffer(&twin, 0, &twin_stage, 0, bytes);
        enc.copy_buffer_to_buffer(&words, 0, &word_stage, 0, bytes);
        device.queue.submit(Some(enc.finish()));
        let base_values = WgpuDevice::read_buffer(&device.device, &base_stage, (m * n) as usize)
            .expect("read base small-K values");
        let twin_values = WgpuDevice::read_buffer(&device.device, &twin_stage, (m * n) as usize)
            .expect("read twin small-K values");
        let twin_words = WgpuDevice::read_u32_buffer(&device.device, &word_stage, (m * n) as usize)
            .expect("read small-K words");
        assert_eq!(
            base_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            twin_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(&twin_words[..2], &[1, 1]);
        assert!(twin_words[2..].iter().all(|&word| word == 0));
    }

    /// Census-shape regression for the word channel: enabling it must preserve
    /// every value bit and return an explicit clean receipt. Performance data
    /// belongs in Criterion, not an ignored unit test.
    #[test]
    fn taint_gate_census_shape_preserves_values() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let mut state: u64 = 0x5EED_CAFE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // cifar100-head-ish: 100 spec rows over a width-512 stack, 6 blocks.
        let (w, specs, blocks) = (512usize, 100usize, 6usize);
        let mut layers: Vec<GpuCrownLayer> = Vec::new();
        for _ in 0..blocks {
            let wt: Vec<f32> = (0..w * w).map(|_| rng() * (1.0 / 22.6)).collect();
            let b: Vec<f32> = (0..w).map(|_| rng() * 0.1).collect();
            let ls: Vec<f32> = (0..w).map(|_| 0.25 + rng().abs() * 0.5).collect();
            let li: Vec<f32> = (0..w).map(|_| rng() * 0.05).collect();
            let ui: Vec<f32> = (0..w)
                .map(|i| li[i].abs() + 0.02 + rng().abs() * 0.05)
                .collect();
            layers.push(GpuCrownLayer::Linear {
                weight: Arc::from(wt.into_boxed_slice()),
                bias: Some(Arc::from(b.into_boxed_slice())),
                out_features: w,
                in_features: w,
                cert_err: Default::default(),
            });
            layers.push(GpuCrownLayer::Activation {
                lower_slope: ls.clone(),
                upper_slope: ls,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: w,
            });
        }
        let mut spec = vec![0.0f32; specs * w];
        for i in 0..specs {
            spec[i * w + i] = 1.0;
        }
        let zb = vec![0.0f32; specs];
        let run = || {
            device
                .crown_backward_sound_resident_coeff_seeded(
                    &layers, &spec, &spec, &zb, &zb, specs, w,
                )
                .expect("resident walk")
        };
        let off = {
            let _off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
            run()
        };
        let on = {
            let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
            run()
        };
        assert_eq!(off.lower_a.len(), on.lower_a.len());
        assert!(off
            .lower_a
            .iter()
            .zip(on.lower_a.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(
            on.taint_rows
                .as_deref()
                .is_some_and(|rows| rows.iter().all(|&word| word == 0)),
            "clean census walk must return an all-clean word receipt"
        );
    }

    /// #u4 differential oracle: `NY_GPU_TAINT_WORDS` OFF vs ON is VALUE
    /// bit-identical on a Linear→Activation→Linear walk (the taint twins
    /// recompute the base arithmetic byte-for-byte — the walk-scale
    /// counterpart of the probe-scale `random_wide_twin_drift_pin`; the
    /// on-device transports write only `rows_dev` and the word buffers, never
    /// a value buffer), the gate-off frontier carries NO words
    /// (`taint_rows == None`, exactly today's behavior), and the gate-on
    /// frontier's row words are ALL ZERO for clean inputs (the twins must
    /// never invent taint).
    ///
    /// Env discipline: env vars are process-global and the GPU tests are
    /// serialized by `gpu_test_serial_guard` — every ScopedEnvVar is created
    /// INSIDE the guard's scope and dropped (restoring the var) before it.
    #[test]
    fn taint_walk_gate_off_is_bit_identical() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // Determinism: the EFT lane must stay dark for both runs (the sibling
        // C2 test arms the per-device EFT cache process-wide).
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let mut state: u64 = 0x00D4_71A7;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (din, h, dout) = (7usize, 9usize, 4usize);
        let w1: Vec<f32> = (0..h * din).map(|_| rng() * 0.7).collect();
        let b1: Vec<f32> = (0..h).map(|_| rng() * 0.4).collect();
        let w2: Vec<f32> = (0..dout * h).map(|_| rng() * 0.7).collect();
        let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.4).collect();
        // Any valid relaxation works: this oracle compares the walk to itself.
        let ls: Vec<f32> = (0..h).map(|_| 0.25 + rng().abs() * 0.5).collect();
        let li: Vec<f32> = (0..h).map(|_| rng() * 0.1).collect();
        let ui: Vec<f32> = (0..h)
            .map(|i| li[i].abs() + 0.05 + rng().abs() * 0.1)
            .collect();
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(w2.into_boxed_slice()),
                bias: Some(Arc::from(b2.into_boxed_slice())),
                out_features: dout,
                in_features: h,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: ls.clone(),
                upper_slope: ls,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: h,
            },
            GpuCrownLayer::Linear {
                weight: Arc::from(w1.into_boxed_slice()),
                bias: Some(Arc::from(b1.into_boxed_slice())),
                out_features: h,
                in_features: din,
                cert_err: Default::default(),
            },
        ];
        let mut spec = vec![0.0f32; dout * dout];
        for i in 0..dout {
            spec[i * dout + i] = 1.0;
        }
        let zb = vec![0.0f32; dout];
        let run = || {
            device
                .crown_backward_sound_resident_coeff_seeded(
                    &layers, &spec, &spec, &zb, &zb, dout, dout,
                )
                .expect("resident walk")
        };
        let base = {
            let _off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
            run()
        };
        let gated = {
            let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
            run()
        };

        for (name, b, g) in [
            ("lower_a", &base.lower_a, &gated.lower_a),
            ("upper_a", &base.upper_a, &gated.upper_a),
            ("lower_err", &base.lower_err, &gated.lower_err),
            ("upper_err", &base.upper_err, &gated.upper_err),
            ("lower_b", &base.lower_b, &gated.lower_b),
            ("upper_b", &base.upper_b, &gated.upper_b),
            ("lower_b_err", &base.lower_b_err, &gated.lower_b_err),
            ("upper_b_err", &base.upper_b_err, &gated.upper_b_err),
        ] {
            assert_eq!(b.len(), g.len(), "{name}: length drift");
            for (i, (x, y)) in b.iter().zip(g.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name}[{i}]: gate off ({x}) vs on ({y}) diverged — the taint \
                     twins are not bit-identical to the base kernels on this adapter"
                );
            }
        }
        assert!(
            base.taint_rows.is_none(),
            "gate off must carry NO words (byte-identical to today)"
        );
        let rows = gated.taint_rows.expect("gate on must carry row words");
        assert_eq!(rows.len(), dout);
        assert!(
            rows.iter().all(|&w| w == 0),
            "clean inputs must produce all-zero row words, got {rows:?}"
        );
    }

    /// #u4 G13 seeding + carriage: a seed coefficient at exactly
    /// `CROWN_COEFF_MAX` (the CPU transport sentinel, == FALLBACK_BOUND)
    /// enters PRE-TAINTED and its word survives a Linear layer with NONZERO
    /// weights — arriving nonzero in the final per-spec-row accumulator even
    /// though the downscaled VALUE (1e10·0.5 = 5e9) passes every magnitude
    /// guard. The clean spec row stays zero (words are never invented).
    #[test]
    fn taint_walk_seeds_and_carries() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let layers = vec![GpuCrownLayer::Linear {
            // 2×2, every weight nonzero ⇒ no annihilation path; magnitudes
            // scale the sentinel DOWN so only the word can carry the taint.
            weight: Arc::from(vec![0.5f32, 0.25, -0.75, 1.0].into_boxed_slice()),
            bias: Some(Arc::from(vec![0.1f32, -0.2].into_boxed_slice())),
            out_features: 2,
            in_features: 2,
            cert_err: Default::default(),
        }];
        // Spec row 0 carries the sentinel at (0,0); row 1 is clean.
        let lower_a = vec![ny_core::CROWN_COEFF_MAX, 0.0, 1.0, 1.0];
        let upper_a = lower_a.clone();
        let zb = vec![0.0f32; 2];
        let c = device
            .crown_backward_sound_resident_coeff_seeded(&layers, &lower_a, &upper_a, &zb, &zb, 2, 2)
            .expect("resident walk");
        let rows = c.taint_rows.expect("gate on must carry row words");
        assert_eq!(rows.len(), 2);
        assert_ne!(
            rows[0],
            0,
            "the pre-tainted seed's word must arrive at spec row 0 (values: \
             lower_a={:?})",
            &c.lower_a[..2]
        );
        assert_eq!(
            rows[1], 0,
            "the clean spec row must stay word-free (annihilation/invention \
             check), got {rows:?}"
        );
        // The laundered VALUE is small and finite — invisible to every
        // magnitude guard — which is exactly why the word channel must exist.
        assert!(c.lower_a[0].abs() < ny_core::FALLBACK_BOUND);
    }

    /// #u4 on-device bias-fold conjunct pin (`TAINT_ROW_OR_SHADER`,
    /// per-COLUMN partner): a seed ERROR word whose ONLY route to the row
    /// accumulator is the Linear bias transport is DROPPED when its
    /// multiplicative partner `bias[k] == 0.0` and CARRIED when
    /// `bias[k] != 0` — matching the CPU reference `bias_fold_taint` exactly.
    ///
    /// Route isolation: the word rides `lower_a_err` (1e30 degrade marker ⇒
    /// G13-worded at walk entry) at (row 0, k 0), and weight ROW 0 is exactly
    /// zero, so (a) the `err@|W|` GEMM twin annihilates it per tap (partner
    /// `|W|[0,j] == 0`), keeping the outgoing words — and hence the final
    /// fold — clean, and (b) the §0 row-L1 word channel reads only the
    /// COEFFICIENT words, which are clean. That leaves the bias transport as
    /// the word's single path to `rows_dev`, pinning the on-device conjunct
    /// through the whole walk.
    #[test]
    fn taint_walk_bias_conjunct_annihilates_on_device() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let run = |bias0: f32| {
            let layers = vec![GpuCrownLayer::Linear {
                // Weight ROW 0 exactly zero: the k=0 err word cannot ride the
                // GEMM twins into the outgoing words (per-tap annihilation).
                weight: Arc::from(vec![0.0f32, 0.0, 0.5, -0.5].into_boxed_slice()),
                bias: Some(Arc::from(vec![bias0, 0.2f32].into_boxed_slice())),
                out_features: 2,
                in_features: 2,
                cert_err: Default::default(),
            }];
            // Clean coefficients; the ERROR seed carries the 1e30 degrade
            // marker at (row 0, k 0).
            let lower_a = vec![1.0f32, 0.0, 0.0, 1.0];
            let upper_a = lower_a.clone();
            let lower_a_err = vec![1e30f32, 0.0, 0.0, 0.0];
            let zero_err = vec![0.0f32; 4];
            let zb = vec![0.0f32; 2];
            device
                .crown_backward_sound_resident_coeff_seeded_err(
                    &layers,
                    &lower_a,
                    &upper_a,
                    &lower_a_err,
                    &zero_err,
                    &zb,
                    &zb,
                    &zb,
                    &zb,
                    2,
                    2,
                    &[],
                    &[],
                )
                .expect("resident walk")
                .taint_rows
                .expect("gate on must carry row words")
        };
        let annihilated = run(0.0);
        assert_eq!(
            annihilated,
            vec![0, 0],
            "bias[0] == 0.0 must annihilate the k=0 err word ON-DEVICE (no \
             other transport reaches its row), got {annihilated:?}"
        );
        let carried = run(0.1);
        assert_ne!(
            carried[0], 0,
            "bias[0] != 0 must carry the k=0 err word to spec row 0, got \
             {carried:?}"
        );
        assert_eq!(
            carried[1], 0,
            "the clean spec row must stay word-free, got {carried:?}"
        );
    }

    /// #u4 audit C2 through the WALK: on the lane-2 shape (sentinel × tiny
    /// weight ⇒ every magnitude innocent) the EFT min-combine CONSULT twin
    /// refuses the tightening on the worded element — the walk's error output
    /// is bit-identical to the un-tightened Higham charge.  When the composed
    /// EFT authorization passes, the base (gate-off) min-combine demonstrably
    /// TIGHTENS that same element (the laundering hole this consult closes).
    /// When any EFT precondition refuses the channel, this oracle instead pins
    /// the required fail-closed Higham fallback.
    #[test]
    fn taint_walk_eft_word_refuses_min_combine_tightening() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let eft_authorized = device.verify_eft_primitives();
        // Separation shape: k = 64 taps but only TWO are nonzero in the seed
        // rows, so Higham's a-priori γ_64·s_prod over-counts the ACTUAL
        // rounding (≈ 1 add) by ~16x — the base min-combine tightening is
        // reliably STRICT wherever it is permitted.
        let (of, if_) = (64usize, 2usize);
        // W (64×2): column 0 carries the 1e-20 launder tap at row 0 and a
        // 1e-3 tap at row 1 (rest exactly 0 — exact-zero products add no
        // rounding); column 1 is all ones (drives the in-band saturation arm
        // on the sentinel row, exercised but not asserted here).
        let mut w = vec![0.0f32; of * if_];
        w[0] = 1e-20;
        w[if_] = 1e-3;
        for r in 0..of {
            w[r * if_ + 1] = 1.0;
        }
        let layers = vec![GpuCrownLayer::Linear {
            weight: Arc::from(w.into_boxed_slice()),
            bias: None,
            out_features: of,
            in_features: if_,
            cert_err: Default::default(),
        }];
        // Spec row 0: the sentinel at (0,0) plus one clean tap at (0,1).
        // Spec row 1: clean, one tap at (1,1).
        let mut lower_a = vec![0.0f32; 2 * of];
        lower_a[0] = ny_core::CROWN_COEFF_MAX;
        lower_a[1] = 1.0;
        lower_a[of + 1] = 1.0;
        let upper_a = lower_a.clone();
        let zb = vec![0.0f32; 2];
        let run = || {
            device
                .crown_backward_sound_resident_coeff_seeded(
                    &layers, &lower_a, &upper_a, &zb, &zb, 2, of,
                )
                .expect("resident walk")
        };
        // (a) Higham reference: gate ON, EFT OFF (no tightening dispatched).
        let higham = {
            let _t = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
            let _e = ScopedEnvVar::unset("NY_EFT_ERR");
            run()
        };
        // (b) EFT requested, gate OFF: when the composed authorization passes,
        // the base min-combine sees only innocent magnitudes
        // (s_prod(0,0) ≈ 1e-3 « FALLBACK_BOUND) and TIGHTENS; otherwise
        // it falls back to Higham.
        let eft_base = {
            let _t = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
            let _e = ScopedEnvVar::set("NY_EFT_ERR", "1");
            run()
        };
        // (c) EFT requested, gate ON: when authorized, the consult twin reads
        // the carried word and REFUSES; a globally refused authorization also
        // falls back to the un-tightened Higham charge.
        let eft_taint = {
            let _t = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
            let _e = ScopedEnvVar::set("NY_EFT_ERR", "1");
            run()
        };
        // Element (0,0): spec row 0, output column 0 — the laundered tap.
        let idx = 0usize;
        if eft_authorized {
            assert!(
                eft_base.lower_err[idx] < higham.lower_err[idx],
                "precondition: the base EFT min-combine must strictly tighten the \
                 laundered element ({} !< {}) — if this fails the shape no longer \
                 separates the channels, pick taps further apart",
                eft_base.lower_err[idx],
                higham.lower_err[idx],
            );
        } else {
            assert_eq!(
                eft_base.lower_err[idx].to_bits(),
                higham.lower_err[idx].to_bits(),
                "a refused composed EFT authorization must disable the unworded \
                 EFT tightening and preserve the Higham charge"
            );
        }
        assert_eq!(
            eft_taint.lower_err[idx].to_bits(),
            higham.lower_err[idx].to_bits(),
            "the carried word must REFUSE the tightening: gate-on error {} != \
             un-tightened Higham {}",
            eft_taint.lower_err[idx],
            higham.lower_err[idx],
        );
        // The worded row is condemned in the accumulator; the clean row's
        // tightening stays permitted (never wider than Higham).
        let rows = eft_taint.taint_rows.as_ref().expect("gate on words");
        assert_ne!(rows[0], 0, "the laundered row must be worded");
        assert_eq!(rows[1], 0, "the clean row must stay word-free");
        // Post-layer errors are [num_specs × if_]: spec row 1, column 0.
        let clean_idx = if_;
        assert!(
            eft_taint.lower_err[clean_idx] <= higham.lower_err[clean_idx],
            "clean-row tightening must never widen"
        );
    }

    /// Shared fixture for the #u4 resnet-composition tests: a small resnet-
    /// shaped segment walk in backward order (cribbed from the stacked-resnet
    /// and beta fixtures) — `[Chain(Linear out), Residual([Linear, ReLU])]`,
    /// every weight/slope nonzero so no annihilation path exists. Returns the
    /// owned layer vecs; callers borrow them into `ResnetSegment`s.
    fn taint_resnet_fixture(d: usize, dout: usize) -> (Vec<GpuCrownLayer>, Vec<GpuCrownLayer>) {
        let mut state: u64 = 0x00D4_C0FE_E5E7;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // Nonzero-everywhere weights: they scale the seed sentinel DOWN below
        // FALLBACK_BOUND (launder) while keeping every multiplicative partner
        // nonzero (no annihilation). The Chain (first-consumed) weights are
        // ~0.02–0.07 so a 1e10 seed sentinel drops to ~e8 territory and the
        // later d-term GEMM sums (≤ d·0.7·|A|) can never climb back to the
        // 1e10 clamp — the value channel stays strictly sub-threshold through
        // the WHOLE composition, so only the word channel can carry the taint.
        let mut nz = |n: usize, lo: f32, span: f32| -> Vec<f32> {
            (0..n)
                .map(|_| (lo + rng().abs() * span) * if rng() > 0.0 { 1.0 } else { -1.0 })
                .collect()
        };
        let ow = nz(dout * d, 0.02, 0.05);
        let ob = nz(dout, 0.02, 0.05);
        let rw = nz(d * d, 0.2, 0.5);
        let rb = nz(d, 0.2, 0.5);
        let ls: Vec<f32> = (0..d).map(|_| 0.25 + rng().abs() * 0.5).collect();
        let li: Vec<f32> = (0..d).map(|_| rng() * 0.1).collect();
        let ui: Vec<f32> = (0..d)
            .map(|i| li[i].abs() + 0.05 + rng().abs() * 0.1)
            .collect();
        let out_chain = vec![GpuCrownLayer::Linear {
            weight: Arc::from(ow.into_boxed_slice()),
            bias: Some(Arc::from(ob.into_boxed_slice())),
            out_features: dout,
            in_features: d,
            cert_err: Default::default(),
        }];
        let res_branch = vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(rw.into_boxed_slice()),
                bias: Some(Arc::from(rb.into_boxed_slice())),
                out_features: d,
                in_features: d,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: ls.clone(),
                upper_slope: ls,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: d,
            },
        ];
        (out_chain, res_branch)
    }

    /// #u4 resnet composition, differential oracle: on a resnet-shaped segment
    /// walk (Chain + identity Residual with a ReLU), `NY_GPU_TAINT_WORDS` OFF
    /// vs ON is VALUE bit-identical on every channel of the COMPOSED frontier,
    /// gate off carries no words (`taint_rows == None`, today's behavior), and
    /// gate ON carries `Some(all-zero)` rows for clean inputs — Some, NOT
    /// None: the segment composition must produce a real word set, otherwise
    /// an ARMED C1 consult would refuse every resnet row to the CPU path.
    #[test]
    fn taint_resnet_compose_gate_off_identical_and_clean_rows() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _seg_off = ScopedEnvVar::unset("NY_SEG_RESIDENT");
        let (d, dout) = (6usize, 4usize);
        let (out_chain, res_branch) = taint_resnet_fixture(d, dout);
        let segments = [
            ResnetSegment::Chain(&out_chain),
            ResnetSegment::Residual(&res_branch),
        ];
        let mut spec = vec![0.0f32; dout * dout];
        for i in 0..dout {
            spec[i * dout + i] = 1.0;
        }
        let zb = vec![0.0f32; dout];
        let run = || {
            device
                .resnet_seeded_compose_coeff(
                    &segments,
                    &spec,
                    &spec,
                    &zb,
                    &zb,
                    dout,
                    dout,
                    dout,
                    &[],
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    false,
                )
                .expect("resnet compose")
                .0
        };
        let base = {
            let _off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
            run()
        };
        let gated = {
            let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
            run()
        };
        for (name, b, g) in [
            ("lower_a", &base.lower_a, &gated.lower_a),
            ("upper_a", &base.upper_a, &gated.upper_a),
            ("lower_err", &base.lower_err, &gated.lower_err),
            ("upper_err", &base.upper_err, &gated.upper_err),
            ("lower_b", &base.lower_b, &gated.lower_b),
            ("upper_b", &base.upper_b, &gated.upper_b),
            ("lower_b_err", &base.lower_b_err, &gated.lower_b_err),
            ("upper_b_err", &base.upper_b_err, &gated.upper_b_err),
        ] {
            assert_eq!(b.len(), g.len(), "{name}: length drift");
            for (i, (x, y)) in b.iter().zip(g.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name}[{i}]: gate off ({x}) vs on ({y}) diverged through the \
                     segment composition"
                );
            }
        }
        assert!(
            base.taint_rows.is_none(),
            "gate off must carry NO words (byte-identical to today)"
        );
        let rows = gated
            .taint_rows
            .expect("gate on must carry Some(rows) through the resnet composition — None here means a seam dropped the word set");
        assert_eq!(rows.len(), dout);
        assert!(
            rows.iter().all(|&w| w == 0),
            "clean inputs must produce all-zero row words, got {rows:?}"
        );
    }

    /// #u4 resnet composition, sentinel carriage: a `CROWN_COEFF_MAX` seed
    /// coefficient in spec row 0 is G13-worded at entry, LAUNDERED below every
    /// magnitude guard by the first (Chain) segment's sub-1.0 weights — so the
    /// Residual segment's own G13 re-seed sees only innocent values — and its
    /// row word still arrives set through the FULL composition (Chain sub-walk
    /// → seam OR → Residual branch sub-walk + skip add). Other rows stay zero
    /// (words are never invented; the seam OR is per-row exact).
    #[test]
    fn taint_resnet_compose_sentinel_row_survives_composition() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _seg_off = ScopedEnvVar::unset("NY_SEG_RESIDENT");
        let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let (d, dout) = (6usize, 4usize);
        let (out_chain, res_branch) = taint_resnet_fixture(d, dout);
        let segments = [
            ResnetSegment::Chain(&out_chain),
            ResnetSegment::Residual(&res_branch),
        ];
        let mut lower_a = vec![0.0f32; dout * dout];
        for i in 0..dout {
            lower_a[i * dout + i] = 1.0;
        }
        // Spec row 0 ships the CPU transport sentinel; rows 1.. stay clean.
        lower_a[0] = ny_core::CROWN_COEFF_MAX;
        let upper_a = lower_a.clone();
        let zb = vec![0.0f32; dout];
        let (coeff, _grads, _gathers) = device
            .resnet_seeded_compose_coeff(
                &segments,
                &lower_a,
                &upper_a,
                &zb,
                &zb,
                dout,
                dout,
                dout,
                &[],
                &[],
                &[],
                &[],
                false,
                &[],
                false,
            )
            .expect("resnet compose");
        let rows = coeff
            .taint_rows
            .expect("gate on must carry Some(rows) through the resnet composition");
        assert_eq!(rows.len(), dout);
        assert_ne!(
            rows[0], 0,
            "the sentinel seed's word must survive the full segment \
             composition into spec row 0"
        );
        for (s, &w) in rows.iter().enumerate().skip(1) {
            assert_eq!(w, 0, "clean spec row {s} must stay word-free, got {rows:?}");
        }
        // The VALUE channel is fully laundered (finite, sub-threshold) by the
        // sub-1.0 weights/slopes — only the word channel still knows. This is
        // the property that makes the wiring load-bearing: the Residual
        // segment's G13 re-seed alone could NOT have re-worded row 0.
        assert!(
            coeff
                .lower_a
                .iter()
                .chain(coeff.upper_a.iter())
                .all(|v| v.is_finite() && v.abs() < ny_core::FALLBACK_BOUND),
            "fixture drift: the sentinel was expected to launder below \
             FALLBACK_BOUND through the composition"
        );
    }

    /// #u4 genuinely-unwirable configuration, pinned: seg-resident device
    /// seed/keep streams carry no word buffers, so under the gate the walk
    /// entry REFUSES (typed), and the resnet composition surfaces that error
    /// instead of silently running un-worded (fail-closed — the gate-off
    /// seg-resident path is untouched).
    #[test]
    fn taint_resnet_seg_resident_stream_refuses() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _seg_on = ScopedEnvVar::set("NY_SEG_RESIDENT", "1");
        let _on = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "1");
        let (d, dout) = (6usize, 4usize);
        let (out_chain, res_branch) = taint_resnet_fixture(d, dout);
        // First segment Chain + no fine/cut/capture channels ⇒ seg-resident
        // eligible, so the first sub-walk runs in keep mode.
        let segments = [
            ResnetSegment::Chain(&out_chain),
            ResnetSegment::Residual(&res_branch),
        ];
        let mut spec = vec![0.0f32; dout * dout];
        for i in 0..dout {
            spec[i * dout + i] = 1.0;
        }
        let zb = vec![0.0f32; dout];
        // (No `expect_err`: `ResidentCoeff` deliberately has no Debug impl.)
        let err = match device.resnet_seeded_compose_coeff(
            &segments,
            &spec,
            &spec,
            &zb,
            &zb,
            dout,
            dout,
            dout,
            &[],
            &[],
            &[],
            &[],
            false,
            &[],
            false,
        ) {
            Ok(_) => {
                panic!("taint_on + seg-resident device streams must refuse (typed, fail-closed)")
            }
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("seg-resident device"),
            "expected the typed seg-resident word-channel refusal, got: {msg}"
        );
    }

    /// #flush-charge Lane A, route GA3 (module-doc guard-coverage table): a
    /// frontier that carries NO row words — the ONLY kind a seg-resident
    /// device stream can produce (its merge shell and `download_resident_coeff`
    /// both set `taint_rows: None`) — is VERDICT-DEAD at the concretize
    /// funnel: the armed C1 consult refuses it BEFORE any dispatch, on every
    /// adapter, under every authority mode. This is the pin that keeps the
    /// seg-resident stream's un-audited on-device merge error lanes
    /// (`seg_merge_dispatch`) outside both the uncharged and the charged
    /// verdict surface without a walk guard of their own.
    #[test]
    fn unworded_frontier_is_verdict_dead_at_the_concretize_funnel() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        // Part 1 (deterministic on every adapter, refusal is pre-dispatch):
        // an unworded frontier is refused by the armed C1 consult.
        let c = ResidentCoeff {
            lower_a: vec![0.5, -0.25],
            upper_a: vec![0.75, 0.25],
            lower_err: vec![0.0; 2],
            upper_err: vec![0.0; 2],
            lower_b: vec![0.0],
            upper_b: vec![0.0],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim: 2,
            relu_grads: Vec::new(),
            beta_gather: Vec::new(),
            taint_rows: None,
        };
        let msg = device
            .concretize_resident_coeff(&c, 1, &[-1.0, -1.0], &[1.0, 1.0])
            .expect_err("an unworded frontier must never concretize")
            .to_string();
        assert!(
            msg.contains("taint words absent"),
            "expected the armed C1 absent-words refusal, got: {msg}"
        );

        // Part 2: the seg-resident stream itself, under the explicit un-worded
        // opt-out (so the sub-walk's worded refusal cannot fire first). On a
        // non-authoritative adapter the sub-walk refuses outright (also
        // verdict-dead, by an earlier gate); on an authoritative one the
        // composed frontier must carry NO words and die at the same funnel.
        let _eft_off = ScopedEnvVar::unset("NY_EFT_ERR");
        let _coalesce_off = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _seg_on = ScopedEnvVar::set("NY_SEG_RESIDENT", "1");
        let _words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let (d, dout) = (6usize, 4usize);
        let (out_chain, res_branch) = taint_resnet_fixture(d, dout);
        let segments = [
            ResnetSegment::Chain(&out_chain),
            ResnetSegment::Residual(&res_branch),
        ];
        let mut spec = vec![0.0f32; dout * dout];
        for i in 0..dout {
            spec[i * dout + i] = 1.0;
        }
        let zb = vec![0.0f32; dout];
        match device.resnet_seeded_compose_coeff(
            &segments,
            &spec,
            &spec,
            &zb,
            &zb,
            dout,
            dout,
            dout,
            &[],
            &[],
            &[],
            &[],
            false,
            &[],
            false,
        ) {
            Ok((coeff, _grads, _gathers)) => {
                assert!(
                    coeff.taint_rows.is_none(),
                    "a seg-resident stream grew row words without a word \
                     channel across device-resident segment boundaries"
                );
                let xl = vec![-1.0f32; coeff.dim];
                let xu = vec![1.0f32; coeff.dim];
                let msg = device
                    .concretize_resident_coeff(&coeff, dout, &xl, &xu)
                    .expect_err("the un-worded seg-resident frontier must die at C1")
                    .to_string();
                assert!(msg.contains("taint words absent"), "got: {msg}");
            }
            Err(e) => {
                assert!(
                    !e.to_string().is_empty(),
                    "seg-resident refusal must be a typed error"
                );
            }
        }
    }

    /// #NY_GPU_BATCHED_COLLECT differential oracle: spec-row chunking is EXACT.
    ///
    /// The sound-resident backward run in `chunk`-row batches
    /// (`crown_backward_sound_chunked`) must return bounds ELEMENT-WISE IDENTICAL to
    /// the single unchunked dispatch (`crown_backward_gpu_sound` with the gate off).
    /// CROWN backward has no cross-spec-row reduction, so partitioning the rows can
    /// only reproduce each row's own value — never tighten it. This is the soundness
    /// precondition for routing the wide-TLL collection through the chunked path:
    /// chunked bounds enclose exactly what the proven single-dispatch sound bound
    /// encloses (which itself encloses the CPU f64+γ·S bound).
    #[test]
    fn spec_row_chunk_is_exact_vs_unchunked() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();

        let (out_dim, mid, in_dim) = (8usize, 32usize, 4usize);
        let mut state: u64 = 0x00C0_FFEE_1234_5678;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // Backward order (output-to-input): Linear(out_dim←mid), ReLU(mid), Linear(mid←in_dim).
        let w0: Arc<[f32]> = (0..out_dim * mid)
            .map(|_| rng() * 0.3)
            .collect::<Vec<_>>()
            .into();
        let w1: Arc<[f32]> = (0..mid * in_dim)
            .map(|_| rng() * 0.3)
            .collect::<Vec<_>>()
            .into();
        let b0: Arc<[f32]> = (0..out_dim).map(|_| rng() * 0.2).collect::<Vec<_>>().into();
        let b1: Arc<[f32]> = (0..mid).map(|_| rng() * 0.2).collect::<Vec<_>>().into();
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: w0,
                bias: Some(b0),
                out_features: out_dim,
                in_features: mid,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: (0..mid).map(|_| 0.4 + 0.2 * rng()).collect(),
                upper_slope: (0..mid).map(|_| 0.8 + 0.1 * rng()).collect(),
                lower_intercept: vec![0.0; mid],
                upper_intercept: (0..mid).map(|_| 0.1 + 0.05 * rng()).collect(),
                num_neurons: mid,
            },
            GpuCrownLayer::Linear {
                weight: w1,
                bias: Some(b1),
                out_features: mid,
                in_features: in_dim,
                cert_err: Default::default(),
            },
        ];
        // Identity spec: one row per output neuron.
        let mut spec = vec![0.0f32; out_dim * out_dim];
        for i in 0..out_dim {
            spec[i * out_dim + i] = 1.0;
        }
        let in_lo: Vec<f32> = (0..in_dim).map(|j| -1.0 - 0.05 * j as f32).collect();
        let in_hi: Vec<f32> = (0..in_dim).map(|j| 1.0 + 0.05 * j as f32).collect();

        // Reference: the single unchunked dispatch. The KILL SWITCH (`=0`) pins this to
        // the genuinely unchunked path rather than relying on the fit-preserving first
        // attempt, so the reference cannot silently become a chunked result if this
        // shape ever grows past a device limit.
        let _collect_off = ScopedEnvVar::set("NY_GPU_BATCHED_COLLECT", "0");
        let reference = device
            .crown_backward_gpu_sound(&layers, &spec, out_dim, &in_lo, &in_hi)
            .expect("unchunked sound backward");

        assert_eq!(reference.lower_bounds.len(), out_dim);
        // CHUNK-SIZE-INVARIANCE: every partition of the spec rows — 1 row/chunk
        // (maximal fragmentation), 3 (the original oracle), 5, and the whole batch as
        // one chunk — must reproduce the unchunked bounds bit-for-bit. This is the
        // soundness precondition my chunk-sizing correction (`sound_spec_row_chunk`
        // /256 fix, 11→3 chunks on the 6272 node) relies on: the CROWN backward has no
        // cross-spec-row reduction, so the ROW COUNT per dispatch is irrelevant to any
        // row's value — only WHICH rows are present. Asserting it across chunk sizes
        // proves the correction cannot perturb a single bit of any verdict-feeding
        // bound, whatever chunk `sound_spec_row_chunk` picks for this adapter.
        for chunk in [1usize, 3, 5, out_dim] {
            let chunked = device
                .crown_backward_sound_chunked(
                    &layers, &spec, out_dim, out_dim, chunk, &in_lo, &in_hi,
                )
                .expect("chunked sound backward");
            assert_eq!(chunked.lower_bounds.len(), out_dim);
            for s in 0..out_dim {
                assert!(
                    reference.lower_bounds[s].is_finite() && reference.upper_bounds[s].is_finite(),
                    "reference bound non-finite at {s}"
                );
                assert!(
                    reference.lower_bounds[s] <= reference.upper_bounds[s],
                    "reference lo>hi at {s}"
                );
                // EXACT: same kernel, same rows, only partitioned.
                assert_eq!(
                    chunked.lower_bounds[s].to_bits(),
                    reference.lower_bounds[s].to_bits(),
                    "chunk={chunk} lower differs from unchunked at spec {s}: {} vs {}",
                    chunked.lower_bounds[s],
                    reference.lower_bounds[s]
                );
                assert_eq!(
                    chunked.upper_bounds[s].to_bits(),
                    reference.upper_bounds[s].to_bits(),
                    "chunk={chunk} upper differs from unchunked at spec {s}: {} vs {}",
                    chunked.upper_bounds[s],
                    reference.upper_bounds[s]
                );
            }
        }
    }

    /// #wg-limit-subchunk TAIL ORACLE on a CONV chain — the gap the Linear-only oracle
    /// above left open, and the precondition for making the chunking DEFAULT-ON.
    ///
    /// Why a separate test: the production tripping shape (relusplitter/cifar_biasfield)
    /// is a Conv2d chain, and a conv layer takes a completely different resident route
    /// than Linear — an im2col reshape whose GEMM `m` is row-count-dependent (so
    /// `select_gemm_dispatch`'s `use_small_k` branch is a *function of the chunk size*),
    /// plus a col2im scatter and a separate `ceil(rows·oc·oh·ow / 256)` elementwise
    /// dispatch width distinct from the `ic·ih·iw` one. If chunking were going to drop
    /// or mis-accumulate a row, conv is where it would happen.
    ///
    /// `num_specs = 7` is PRIME, so EVERY chunk size in `[1,2,3,4,5,6,7]` leaves a
    /// different short tail (7 = 3+3+**1** = 4+**3** = 5+**2** = 6+**1**), i.e. this
    /// asserts the off-by-one tail case the Linear oracle only hit for chunk∈{3,5}. The
    /// spec rows are DENSE random (not the identity used above): an identity row reads a
    /// single output column, which would mask a row-indexing error that a dense row
    /// exposes.
    #[test]
    fn spec_row_chunk_is_exact_vs_unchunked_conv_chain() {
        let _g = gpu_test_serial_guard();
        // This is a raw coefficient/chunking oracle, not a verdict-path test.
        // Disable words so it isolates row slicing from receipt composition;
        // dedicated tests exercise the production worded Conv route.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();

        let mut state: u64 = 0x00C0_FFEE_C047_0001;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // Backward order (output→input), 3x3 stride-1 pad-1 so H/W are preserved:
        //   Conv2d(oc=4 ← ic=3) : coeff 4*6*6=144 → 3*6*6=108
        //   Activation(108)
        //   Conv2d(oc=3 ← ic=2) : coeff 108 → 2*6*6=72   (= input dim)
        let (hw, k) = (6usize, 3usize);
        let (oc0, ic0, oc1, ic1) = (4usize, 3usize, 3usize, 2usize);
        let out_dim = oc0 * hw * hw; // 144 = spec width
        let mid = ic0 * hw * hw; // 108
        let in_dim = ic1 * hw * hw; // 72
        let num_specs = 7usize; // PRIME → every chunk size exercises a short tail

        let wcol0: Arc<[f32]> = (0..oc0 * ic0 * k * k)
            .map(|_| rng() * 0.25)
            .collect::<Vec<_>>()
            .into();
        let wcol1: Arc<[f32]> = (0..oc1 * ic1 * k * k)
            .map(|_| rng() * 0.25)
            .collect::<Vec<_>>()
            .into();
        let b0: Arc<[f32]> = (0..out_dim).map(|_| rng() * 0.2).collect::<Vec<_>>().into();
        let b1: Arc<[f32]> = (0..mid).map(|_| rng() * 0.2).collect::<Vec<_>>().into();
        let conv = |weight_col: Arc<[f32]>,
                    bias_expanded: Option<Arc<[f32]>>,
                    out_channels: usize,
                    in_channels: usize| GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h: k,
            kernel_w: k,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            out_h: hw,
            out_w: hw,
            in_h: hw,
            in_w: hw,
            cert_err: Default::default(),
        };
        let layers = vec![
            conv(Arc::clone(&wcol0), Some(Arc::clone(&b0)), oc0, ic0),
            GpuCrownLayer::Activation {
                lower_slope: (0..mid).map(|_| 0.35 + 0.25 * rng()).collect(),
                upper_slope: (0..mid).map(|_| 0.75 + 0.15 * rng()).collect(),
                lower_intercept: vec![0.0; mid],
                upper_intercept: (0..mid).map(|_| 0.1 + 0.05 * rng()).collect(),
                num_neurons: mid,
            },
            conv(Arc::clone(&wcol1), Some(Arc::clone(&b1)), oc1, ic1),
        ];

        // DENSE random spec rows (num_specs × out_dim), not identity.
        let spec: Vec<f32> = (0..num_specs * out_dim).map(|_| rng() * 0.5).collect();
        let in_lo: Vec<f32> = (0..in_dim).map(|j| -1.0 - 0.01 * (j % 7) as f32).collect();
        let in_hi: Vec<f32> = (0..in_dim).map(|j| 1.0 + 0.01 * (j % 5) as f32).collect();

        // Raw, explicitly unworded reference: one resident coefficient dispatch.
        // The separate runtime-preflight test pins that this is not a verdict route.
        let reference = unworded_resident_test_chunked(
            &device, &layers, &spec, num_specs, out_dim, num_specs, &in_lo, &in_hi,
        )
        .expect("unchunked raw conv backward");
        assert_eq!(reference.lower_bounds.len(), num_specs);
        assert_eq!(reference.upper_bounds.len(), num_specs);

        for chunk in 1..=num_specs {
            let chunked = unworded_resident_test_chunked(
                &device, &layers, &spec, num_specs, out_dim, chunk, &in_lo, &in_hi,
            )
            .expect("chunked raw conv backward");
            // ROW COUNT: no dropped/duplicated tail row.
            assert_eq!(
                chunked.lower_bounds.len(),
                num_specs,
                "chunk={chunk} returned {} lower bounds for {num_specs} spec rows",
                chunked.lower_bounds.len()
            );
            assert_eq!(chunked.upper_bounds.len(), num_specs, "chunk={chunk} upper");
            for s in 0..num_specs {
                assert!(
                    reference.lower_bounds[s].is_finite() && reference.upper_bounds[s].is_finite(),
                    "reference bound non-finite at {s}"
                );
                assert!(
                    reference.lower_bounds[s] <= reference.upper_bounds[s],
                    "reference lo>hi at {s}"
                );
                // EXACT and IN ORDER: row s of every partition is row s of the whole.
                assert_eq!(
                    chunked.lower_bounds[s].to_bits(),
                    reference.lower_bounds[s].to_bits(),
                    "conv chunk={chunk} lower differs at spec {s}: {} vs {}",
                    chunked.lower_bounds[s],
                    reference.lower_bounds[s]
                );
                assert_eq!(
                    chunked.upper_bounds[s].to_bits(),
                    reference.upper_bounds[s].to_bits(),
                    "conv chunk={chunk} upper differs at spec {s}: {} vs {}",
                    chunked.upper_bounds[s],
                    reference.upper_bounds[s]
                );
            }
        }
    }

    /// #wg-limit-subchunk DEFAULT LOCK: with NO env var set, `sound_spec_row_chunk` must
    /// SPLIT an over-limit row batch (auto-detected from this adapter's real
    /// `device.limits()`), and `NY_GPU_BATCHED_COLLECT=0` must restore never-chunk.
    ///
    /// This is the regression that would catch a silent revert of the default. It also
    /// pins the two no-op invariants that make default-ON safe: a shape that FITS this
    /// adapter yields `chunk == num_specs` (the caller's `chunk < num_specs` guard then
    /// takes the unchunked path, byte-identical to pre-fix), and the split chunk is
    /// small enough that `worst_1d = max(rows, ceil(rows·W/256))` clears the adapter's
    /// own cap — the exact predicate `crown_backward_sound_resident` fails closed on.
    #[test]
    fn sound_spec_row_chunk_defaults_to_auto_detected_split() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let max_wg = device
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1) as usize;

        // The production tripping shape: relusplitter/cifar_biasfield's widest conv
        // node, 8192 spec rows × W=16384 ⇒ worst_1d = 524288 (8.0× the 65535 cap).
        let (num_specs, width) = (8192usize, 16384usize);
        let act = |n: usize| GpuCrownLayer::Activation {
            lower_slope: vec![0.5; n],
            upper_slope: vec![0.9; n],
            lower_intercept: vec![0.0; n],
            upper_intercept: vec![0.1; n],
            num_neurons: n,
        };
        let wide = vec![act(width)];
        // Sanity: this shape really is over THIS adapter's limit (else the test is
        // vacuous on some future adapter and should be re-sized, not silently passed).
        let worst_1d = num_specs.max(num_specs * width / 256);
        assert!(
            worst_1d > max_wg,
            "test shape no longer exceeds this adapter's cap \
             (worst_1d={worst_1d}, max_compute_workgroups_per_dimension={max_wg}) — re-size it"
        );

        // DEFAULT (no env var): must split.
        let chunk = {
            let _unset = ScopedEnvVar::unset("NY_GPU_BATCHED_COLLECT");
            device
                .sound_spec_row_chunk(&wide, num_specs)
                .expect("default must size a chunk, not decline")
        };
        assert!(
            chunk >= 1 && chunk < num_specs,
            "default must SPLIT an over-limit batch: chunk={chunk} of {num_specs}"
        );
        let chunk_worst_1d = chunk.max(chunk * width / 256);
        assert!(
            chunk_worst_1d <= max_wg,
            "chunk={chunk} still dispatches {chunk_worst_1d} > cap {max_wg}"
        );

        // KILL SWITCH: `=0` ⇒ never chunk (pre-fix hard-fail → CPU sound fallback).
        {
            let _off = ScopedEnvVar::set("NY_GPU_BATCHED_COLLECT", "0");
            assert_eq!(
                device.sound_spec_row_chunk(&wide, num_specs),
                None,
                "NY_GPU_BATCHED_COLLECT=0 must disable chunking"
            );
        }
        // Legacy opt-in value keeps working (was the only enabling value pre-flip).
        {
            let _on = ScopedEnvVar::set("NY_GPU_BATCHED_COLLECT", "1");
            let c = device
                .sound_spec_row_chunk(&wide, num_specs)
                .expect("=1 must still chunk");
            assert_eq!(c, chunk, "=1 must agree with the default");
        }

        // BLAST-RADIUS FENCE: an over-limit batch whose layers include something the
        // resident fold rejects outright ("R4": not Linear/Activation/Conv2d) must
        // DECLINE, not spin one futile sub-call per chunk — chunking cannot fix an R4
        // rejection. Keeps the default flip inert on MaxPool2d / dual-alpha graphs.
        {
            let _unset = ScopedEnvVar::unset("NY_GPU_BATCHED_COLLECT");
            let unchunkable = vec![
                act(width),
                GpuCrownLayer::MaxPool2d {
                    routing: vec![0],
                    ibp_lower: vec![0.0],
                    ibp_upper: vec![1.0],
                    input_dim: width,
                    output_dim: width,
                },
            ];
            assert_eq!(
                device.sound_spec_row_chunk(&unchunkable, num_specs),
                None,
                "an R4-rejected layer list must not be chunked"
            );
        }

        // NO-OP INVARIANT: a narrow shape that fits ⇒ chunk == num_specs ⇒ the caller's
        // `chunk < num_specs` guard declines ⇒ byte-identical to the pre-fix path.
        let narrow = vec![act(64)];
        let _unset = ScopedEnvVar::unset("NY_GPU_BATCHED_COLLECT");
        assert_eq!(
            device.sound_spec_row_chunk(&narrow, 10),
            Some(10),
            "a fitting shape must not be split"
        );
        assert_eq!(
            device.sound_spec_row_chunk(&narrow, 0),
            None,
            "zero rows must decline"
        );
    }

    /// INC2 (the TRUE joint α-gradient, `docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md`):
    /// the production `ny_core::joint_alpha_grad` adjoint must match central finite
    /// differences of the ACTUAL sound serial GPU bound
    /// (`crown_backward_gpu_resnet_sound_beta`) w.r.t. the per-ReLU lower slope α, on
    /// a small conv resnet (Conv chain + identity residual, 2 ReLUs). Two proofs:
    ///   (1) the joint fold's own lower bound tracks the GPU serial bound (so the
    ///       frozen signs the adjoint uses ARE the GPU's), and
    ///   (2) the joint gradient matches FD of the GPU bound (relative L2 + cosine),
    ///       while DROPPING the bias channel visibly diverges (the ≈0.7× degradation
    ///       the design doc §2 predicts and the FD validators encode).
    #[test]
    fn joint_alpha_gradient_matches_gpu_serial_bound_fd() {
        use ny_core::joint_alpha_grad::{
            joint_alpha_gradient, joint_lower_bound_debug, JointGradConfig,
        };
        let _g = gpu_test_serial_guard();
        // This finite-difference oracle intentionally isolates Conv arithmetic
        // from the verdict receipt; dedicated worded Conv tests cover authority.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        // Conv is same-padding (k=3,pad=1 → out=hw). Block dim d = c·hw·hw.
        let (c, hw, k) = (2usize, 3usize, 3usize);
        let d = c * hw * hw; // 18
        let num_specs = 2usize;
        let mut state: u64 = 0x10E5_7A11_C0DE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // Small, well-conditioned weights → the certified-error channel (omitted by
        // the adjoint) stays negligible, so FD of the sound bound ≈ FD of the
        // coefficient-channel bound the adjoint differentiates.
        let conv_w: Arc<[f32]> = (0..c * c * k * k)
            .map(|_| rng() * 0.3)
            .collect::<Vec<_>>()
            .into();
        let lin_w: Arc<[f32]> = (0..d * d).map(|_| rng() * 0.15).collect::<Vec<_>>().into();
        // Nonzero layer biases → the bias channel (`+ c` in the adjoint) genuinely
        // steers the gradient, so dropping it degrades ≈0.7× as the design doc §2 /
        // the Python validators show (with bias:None it only shows the smaller
        // ReLU-intercept contribution).
        let conv_b: Arc<[f32]> = (0..d).map(|_| rng() * 0.4).collect::<Vec<_>>().into();
        let lin_b: Arc<[f32]> = (0..d).map(|_| rng() * 0.4).collect::<Vec<_>>().into();
        let seed_a: Vec<f32> = (0..num_specs * d).map(|_| rng()).collect();
        let seed_b = vec![0.0f32; num_specs];
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.clone().into(),
            lower_b: seed_b.clone().into(),
            upper_b: seed_b.clone().into(),
            num_specs,
            current_dim: d,
        };
        // Per-neuron α in (0.2,0.8), distinct per ReLU. Fixed chord upper relaxation
        // (α-independent); lower_intercept ≡ 0 (a real ReLU lower relaxation).
        let alpha0: Vec<f32> = (0..d).map(|_| 0.5 + 0.3 * rng()).collect();
        let alpha1: Vec<f32> = (0..d).map(|_| 0.5 + 0.3 * rng()).collect();
        let upper0: Vec<f32> = (0..d).map(|_| 0.55 + 0.1 * rng()).collect();
        let upper1: Vec<f32> = (0..d).map(|_| 0.60 + 0.1 * rng()).collect();
        let uint0: Vec<f32> = (0..d).map(|_| 0.20 + 0.05 * rng()).collect();
        let uint1: Vec<f32> = (0..d).map(|_| 0.15 + 0.05 * rng()).collect();
        let in_lo: Vec<f32> = (0..d).map(|j| -1.0 - 0.03 * j as f32).collect();
        let in_hi: Vec<f32> = (0..d).map(|j| 1.0 + 0.03 * j as f32).collect();

        // Build the segments with a given α for relu0/relu1 (used for perturbation).
        let build = |a0: &[f32], a1: &[f32]| -> Vec<GpuResnetSegment> {
            let conv = GpuCrownLayer::Conv2d {
                weight_col: conv_w.clone(),
                bias_expanded: Some(conv_b.clone()),
                out_channels: c,
                in_channels: c,
                kernel_h: k,
                kernel_w: k,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                out_h: hw,
                out_w: hw,
                in_h: hw,
                in_w: hw,
                cert_err: Default::default(),
            };
            let act = |a: &[f32], up: &[f32], ui: &[f32]| GpuCrownLayer::Activation {
                lower_slope: a.to_vec(),
                upper_slope: up.to_vec(),
                lower_intercept: vec![0.0; d],
                upper_intercept: ui.to_vec(),
                num_neurons: d,
            };
            let lin = GpuCrownLayer::Linear {
                weight: lin_w.clone(),
                bias: Some(lin_b.clone()),
                out_features: d,
                in_features: d,
                cert_err: Default::default(),
            };
            vec![
                GpuResnetSegment::Chain(vec![conv, act(a0, &upper0, &uint0)]),
                GpuResnetSegment::Residual(vec![lin, act(a1, &upper1, &uint1)]),
            ]
        };
        let gpu_bound = |segs: &[GpuResnetSegment]| -> Vec<f32> {
            unworded_resnet_test_bounds(&device, segs, &seed, &in_lo, &in_hi, &[], &[], &[])
                .expect("serial sound beta bound")
                .lower_bounds
        };

        let segs = build(&alpha0, &alpha1);

        // (1) the joint fold's own lower bound tracks the sound GPU serial bound.
        let gpu_lo = gpu_bound(&segs);
        let fold_lo =
            joint_lower_bound_debug(&segs, &seed_a, &seed_b, num_specs, d, &in_lo, &in_hi).unwrap();
        for s in 0..num_specs {
            let tol = 5e-2 * (1.0 + gpu_lo[s].abs());
            assert!(
                (gpu_lo[s] - fold_lo[s]).abs() <= tol,
                "fold bound {} vs GPU serial bound {} (spec {s}) — the joint fold does \
                 not track the GPU bound; frozen signs would be wrong",
                fold_lo[s],
                gpu_lo[s]
            );
        }

        // (2) the joint gradient vs central FD of the GPU serial bound.
        let g_joint = joint_alpha_gradient(
            &segs,
            &seed_a,
            &seed_b,
            num_specs,
            d,
            &in_lo,
            &in_hi,
            JointGradConfig::default(),
        )
        .expect("joint gradient");
        let g_nobias = joint_alpha_gradient(
            &segs,
            &seed_a,
            &seed_b,
            num_specs,
            d,
            &in_lo,
            &in_hi,
            JointGradConfig {
                bias_channel: false,
            },
        )
        .expect("no-bias gradient");
        assert_eq!(g_joint.len(), 2);

        let eps = 2e-3f32;
        let sum_specs = |v: &[f32]| -> f32 { v.iter().sum() };
        let mut g_fd: Vec<Vec<f32>> = vec![vec![0.0f32; d], vec![0.0f32; d]];
        for relu in 0..2usize {
            for n in 0..d {
                let mut a0p = alpha0.clone();
                let mut a1p = alpha1.clone();
                let mut a0m = alpha0.clone();
                let mut a1m = alpha1.clone();
                if relu == 0 {
                    a0p[n] += eps;
                    a0m[n] -= eps;
                } else {
                    a1p[n] += eps;
                    a1m[n] -= eps;
                }
                let bp = sum_specs(&gpu_bound(&build(&a0p, &a1p)));
                let bm = sum_specs(&gpu_bound(&build(&a0m, &a1m)));
                g_fd[relu][n] = (bp - bm) / (2.0 * eps);
            }
        }

        // Robust aggregate metrics (a couple of near-sign-flip neurons could spike a
        // single relative error; L2 + cosine over the full field are the honest test).
        let rel_l2 = |g: &[Vec<f32>]| -> f32 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for r in 0..2 {
                for n in 0..d {
                    let diff = (g[r][n] - g_fd[r][n]) as f64;
                    num += diff * diff;
                    den += (g_fd[r][n] as f64) * (g_fd[r][n] as f64);
                }
            }
            (num / den.max(1e-30)).sqrt() as f32
        };
        let cosine = |g: &[Vec<f32>]| -> f32 {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for r in 0..2 {
                for n in 0..d {
                    dot += (g[r][n] as f64) * (g_fd[r][n] as f64);
                    na += (g[r][n] as f64).powi(2);
                    nb += (g_fd[r][n] as f64).powi(2);
                }
            }
            (dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
        };
        let joint_l2 = rel_l2(&g_joint);
        let joint_cos = cosine(&g_joint);
        let nobias_l2 = rel_l2(&g_nobias);
        eprintln!(
            "[joint-fd] JOINT rel_l2={joint_l2:.4e} cos={joint_cos:.6}  NO-BIAS rel_l2={nobias_l2:.4e}"
        );
        assert!(
            joint_l2 < 5e-2,
            "joint adjoint vs GPU-bound FD relative-L2 {joint_l2} too large"
        );
        assert!(
            joint_cos > 0.999,
            "joint adjoint vs FD cosine {joint_cos} too low"
        );
        assert!(
            nobias_l2 > 0.1,
            "dropping the bias channel must visibly diverge from FD (got rel_l2 {nobias_l2}); \
             the bias channel is not actually contributing"
        );

        // (3) ON-DEVICE adjoint (task #39): the GPU joint α-gradient must match the
        // PROVEN-CORRECT CPU oracle `joint_alpha_gradient` (which (2) just tied to FD
        // of the sound GPU bound), at MULTIPLE random α, with the bias channel present.
        // A per-neuron rel-L2 < 1e-3 confirms the on-device forward fold + reverse
        // adjoint (conv-transpose fwd, plain-conv adjoint, GEMM/GEMMᵀ, ReLU
        // harvest/propagate, ξ seed, residual fan-out) reproduces the CPU semantics.
        let rel_l2_pair = |g: &[Vec<f32>], h: &[Vec<f32>]| -> f32 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (gr, hr) in g.iter().zip(h.iter()) {
                for (gv, hv) in gr.iter().zip(hr.iter()) {
                    let diff = (*gv - *hv) as f64;
                    num += diff * diff;
                    den += (*hv as f64) * (*hv as f64);
                }
            }
            (num / den.max(1e-30)).sqrt() as f32
        };
        let mut rng2 = {
            let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
            move || {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            }
        };
        for trial in 0..3usize {
            let (a0t, a1t) = if trial == 0 {
                (alpha0.clone(), alpha1.clone())
            } else {
                (
                    (0..d).map(|_| 0.5 + 0.3 * rng2()).collect::<Vec<f32>>(),
                    (0..d).map(|_| 0.5 + 0.3 * rng2()).collect::<Vec<f32>>(),
                )
            };
            let segs_t = build(&a0t, &a1t);
            let g_cpu = joint_alpha_gradient(
                &segs_t,
                &seed_a,
                &seed_b,
                num_specs,
                d,
                &in_lo,
                &in_hi,
                JointGradConfig::default(),
            )
            .expect("cpu joint gradient");
            let g_gpu = device
                .crown_joint_alpha_gradient_resident(&segs_t, &seed_a, num_specs, d, &in_lo, &in_hi)
                .expect("gpu joint gradient");
            assert_eq!(g_gpu.len(), g_cpu.len(), "gpu/cpu relu count");
            let gpu_cpu_l2 = rel_l2_pair(&g_gpu, &g_cpu);
            // Cross-check the GPU adjoint also tracks FD of the sound bound directly.
            let gpu_fd_l2 = rel_l2(&g_gpu);
            eprintln!(
                "[joint-gpu-adj] trial={trial} GPU-vs-CPU rel_l2={gpu_cpu_l2:.4e}  GPU-vs-FD rel_l2={gpu_fd_l2:.4e}"
            );
            assert!(
                gpu_cpu_l2 < 1e-3,
                "GPU on-device adjoint vs CPU oracle rel-L2 {gpu_cpu_l2} too large (trial {trial})"
            );
        }

        // Bias channel present on device: dropping it (NY_WIDE_ALPHA_NOBIAS) must
        // visibly diverge from the CPU full adjoint — the on-device `+τ`/`+bias`
        // channel is load-bearing (the ~0.7× degradation, design doc §2).
        let g_gpu_nobias = {
            let _nobias = ScopedEnvVar::set("NY_WIDE_ALPHA_NOBIAS", "1");
            device
                .crown_joint_alpha_gradient_resident(&segs, &seed_a, num_specs, d, &in_lo, &in_hi)
                .expect("gpu joint gradient (no bias)")
        };
        let gpu_nobias_l2 = rel_l2_pair(&g_gpu_nobias, &g_joint);
        eprintln!("[joint-gpu-adj] NO-BIAS GPU-vs-CPU rel_l2={gpu_nobias_l2:.4e}");
        assert!(
            gpu_nobias_l2 > 0.1,
            "dropping the on-device bias channel must visibly diverge (got {gpu_nobias_l2})"
        );
    }

    fn matmul(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
            .collect()
    }

    /// A malformed resident fold is one indivisible Lagrangian entry. No valid
    /// post/bias/pre subset may leak into the objective when any sibling term is
    /// out of range or non-finite.
    #[test]
    fn crown_resident_cut_fold_malformed_entries_are_atomic_noops() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        use crate::wgpu_device::{
            clear_resident_cut_fold, reset_resident_cut_fold_applied_count,
            resident_cut_fold_applied_count, set_resident_cut_fold, ResidentCutFold,
        };

        // Same exact-value geometry as the valid-path parity test below. The
        // target Activation has width 3, so index 3 is deliberately invalid.
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![-1.0f32, -1.0, -1.0].into_boxed_slice()),
                bias: None,
                out_features: 1,
                in_features: 3,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.0; 3],
                upper_slope: vec![0.5; 3],
                lower_intercept: vec![0.0; 3],
                upper_intercept: vec![0.5, 1.5, 1.5],
                num_neurons: 3,
            },
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![1.0f32, 0.0, -1.0, 2.0, -1.0, -2.0].into_boxed_slice()),
                bias: None,
                out_features: 3,
                in_features: 2,
                cert_err: Default::default(),
            },
        ];
        let segments = [ResnetSegment::Chain(&layers)];
        let (seed_a, seed_b) = (vec![1.0f32], vec![0.0f32]);
        let (xl, xu) = (vec![-1.0f32, -1.0], vec![1.0f32, 1.0]);
        let run = |dev: &WgpuDevice| -> (f32, f32) {
            let (lo, hi, _grads) = dev
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed_a,
                    &seed_a,
                    &seed_b,
                    &seed_b,
                    1,
                    1,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    false,
                )
                .expect("resident malformed-fold backward");
            (lo[0], hi[0])
        };
        let entry = |coeffs, bias_shift, pre_coeffs| ResidentCutFold {
            coeffs,
            bias_shift,
            pre_coeffs,
            sound_round: true,
        };

        let _fold_off = ScopedEnvVar::unset("NY_CUT_FOLD_RESIDENT");
        clear_resident_cut_fold();
        let baseline = run(&device);
        assert!(
            (f64::from(baseline.0) + 4.0).abs() < 1e-4,
            "fixture baseline must be -4, got {}",
            baseline.0
        );

        let _fold_on = ScopedEnvVar::set("NY_CUT_FOLD_RESIDENT", "1");
        let malformed = [
            (
                "post-oob with valid pre",
                entry(vec![(3, 0.25)], -0.75, vec![(0, 0.5)]),
            ),
            (
                "pre-oob after valid post+bias",
                entry(vec![(0, 0.25)], -0.75, vec![(3, 0.5)]),
            ),
            (
                "mixed valid+invalid post",
                entry(vec![(0, 0.25), (3, -0.5)], -0.75, vec![(1, 0.5)]),
            ),
            (
                "mixed valid+invalid pre",
                entry(vec![(0, 0.25)], -0.75, vec![(1, 0.5), (3, -0.5)]),
            ),
            (
                "nonfinite post coefficient",
                entry(vec![(0, f32::INFINITY)], -0.75, vec![(1, 0.5)]),
            ),
            (
                "nonfinite bias metadata",
                entry(vec![(0, 0.25)], f32::NAN, vec![(1, 0.5)]),
            ),
            (
                "nonfinite pre coefficient",
                entry(vec![(0, 0.25)], -0.75, vec![(1, f32::NEG_INFINITY)]),
            ),
        ];
        for (name, malformed_entry) in malformed {
            set_resident_cut_fold(malformed_entry);
            reset_resident_cut_fold_applied_count();
            let got = run(&device);
            assert_eq!(
                got.0.to_bits(),
                baseline.0.to_bits(),
                "{name}: lower bound must be the bit-identical untouched result"
            );
            assert_eq!(
                got.1.to_bits(),
                baseline.1.to_bits(),
                "{name}: upper bound must be the bit-identical untouched result"
            );
            assert_eq!(
                resident_cut_fold_applied_count(),
                0,
                "{name}: rejected entry must not count as applied"
            );
        }
        clear_resident_cut_fold();
    }

    /// Resident cut-fold proof-authority quarantine on the exact k=3 geometry.
    ///
    /// The raw registry request historically tightened −4 to −4+λ here. Public
    /// env/registry state must now leave the ordinary resident verifier exactly
    /// unchanged and record zero applications.
    #[test]
    fn crown_resident_cut_fold_k3_geometry_is_quarantined() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        use crate::wgpu_device::{
            clear_resident_cut_fold, reset_resident_cut_fold_applied_count,
            resident_cut_fold_applied_count, set_resident_cut_fold, ResidentCutFold,
        };

        // Backward order: f = head(relu(pre(x))). Pre-activation boxes over
        // x ∈ [−1,1]²: z1 ∈ [−1,1], z2/z3 ∈ [−3,3] — all unstable; upper
        // chord slope u/(u−l) = 0.5, intercept −u·l/(u−l) = {0.5, 1.5, 1.5};
        // lower slope 0 (never selected while the folded coeff stays ≤ 0).
        let layers = vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![-1.0f32, -1.0, -1.0].into_boxed_slice()),
                bias: None,
                out_features: 1,
                in_features: 3,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.0; 3],
                upper_slope: vec![0.5; 3],
                lower_intercept: vec![0.0; 3],
                upper_intercept: vec![0.5, 1.5, 1.5],
                num_neurons: 3,
            },
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![1.0f32, 0.0, -1.0, 2.0, -1.0, -2.0].into_boxed_slice()),
                bias: None,
                out_features: 3,
                in_features: 2,
                cert_err: Default::default(),
            },
        ];
        let segments = [ResnetSegment::Chain(&layers)];
        let (seed_a, seed_b) = (vec![1.0f32], vec![0.0f32]);
        let (xl, xu) = (vec![-1.0f32, -1.0], vec![1.0f32, 1.0]);
        let run = |dev: &WgpuDevice| -> (f32, f32) {
            let (lo, hi, _grads) = dev
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed_a,
                    &seed_a,
                    &seed_b,
                    &seed_b,
                    1,
                    1,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    false,
                )
                .expect("resident k=3 backward");
            (lo[0], hi[0])
        };

        let _fold_off = ScopedEnvVar::unset("NY_CUT_FOLD_RESIDENT");
        clear_resident_cut_fold();
        let (base_lo, base_hi) = run(&device);
        assert!(
            (f64::from(base_lo) + 4.0).abs() < 1e-4,
            "plain resident CROWN on the k=3 geometry must be −4, got {base_lo}"
        );

        let _fold_on = ScopedEnvVar::set("NY_CUT_FOLD_RESIDENT", "1");
        for lambda in [0.25f32, 0.5, 1.0] {
            set_resident_cut_fold(ResidentCutFold {
                coeffs: vec![(0, lambda), (1, lambda), (2, lambda)],
                bias_shift: -3.0 * lambda,
                ..Default::default()
            });
            reset_resident_cut_fold_applied_count();
            let (lo, hi) = run(&device);
            assert_eq!(
                resident_cut_fold_applied_count(),
                0,
                "λ={lambda}: resident verifier must not consume public cut-fold state"
            );
            assert_eq!(
                lo.to_bits(),
                base_lo.to_bits(),
                "λ={lambda}: lower bound must be bit-identical to baseline"
            );
            assert_eq!(
                hi.to_bits(),
                base_hi.to_bits(),
                "λ={lambda}: upper bound must be bit-identical to baseline"
            );
        }
        clear_resident_cut_fold();
        // Env guards restore the pre-test state on drop.
    }

    // =======================================================================
    /// The on-device per-ReLU alpha gradient must match the CPU analytic formula
    /// `compute_graph_chain_rule_gradients`: grad[i] = pre_lower[i]·Σ_j max(A[j,i],0)
    /// (the lower-relaxation derivative for unstable ReLUs). Step 1 of the
    /// gradient-capable GPU-resident warmup (cifar100/tinyimagenet unsat keystone).
    #[test]
    fn crown_alpha_gradient_resident_matches_cpu_formula() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let mut state: u64 = 0xA1FA_C0DE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for &(num_specs, num_neurons) in &[(1usize, 5usize), (8, 16), (100, 257), (37, 1024)] {
            let a_lower: Vec<f32> = (0..num_specs * num_neurons).map(|_| rng() * 3.0).collect();
            // pre_lower: negative for "unstable" neurons (l<0), 0 for stable (mask folded in).
            let pre_lower: Vec<f32> = (0..num_neurons)
                .map(|i| if i % 4 == 0 { 0.0 } else { rng() * 0.5 - 0.6 })
                .collect();

            let expected: Vec<f32> = (0..num_neurons)
                .map(|i| {
                    let s: f32 = (0..num_specs)
                        .map(|j| a_lower[j * num_neurons + i].max(0.0))
                        .sum();
                    pre_lower[i] * s
                })
                .collect();

            let got = device
                .crown_alpha_gradient_resident(&a_lower, &pre_lower, num_specs, num_neurons)
                .expect("alpha gradient");
            assert_eq!(got.len(), num_neurons);
            for i in 0..num_neurons {
                let tol = 1e-3 * (1.0 + expected[i].abs());
                assert!(
                    (got[i] - expected[i]).abs() <= tol,
                    "grad[{i}]: gpu={} cpu={} (specs={num_specs}, neurons={num_neurons})",
                    got[i],
                    expected[i]
                );
            }
        }
    }

    /// R-grad step 2: the resident backward, when given `relu_pre_lower`, captures
    /// each ReLU's analytic alpha gradient from the PRE-transform lower coefficient.
    /// Backward-order chain Linear2(O×H) → ReLU(H) → Linear1(H×I) with an identity
    /// seed (num_specs=O) gives a_at_relu = I_O @ W2 = W2, so the captured gradient
    /// must equal pre_lower[k]·Σ_j max(W2[j,k],0). Confirms the capture is wired into
    /// the real backward (not just the standalone primitive) and is byte-additive.
    #[test]
    fn crown_resident_backward_captures_relu_alpha_gradients() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (o, h, i) = (3usize, 5usize, 4usize);
        let mut state: u64 = 0xBEEF_F00D;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let w2: Vec<f32> = (0..o * h).map(|_| rng() * 2.0).collect();
        let w1: Vec<f32> = (0..h * i).map(|_| rng()).collect();
        let pre_lower: Vec<f32> = (0..h)
            .map(|k| if k % 3 == 0 { 0.0 } else { rng() * 0.5 - 0.7 })
            .collect();

        let layers = vec![
            GpuCrownLayer::Linear {
                weight: w2.clone().into(),
                bias: None,
                out_features: o,
                in_features: h,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.5; h],
                upper_slope: vec![0.7; h],
                lower_intercept: vec![0.0; h],
                upper_intercept: vec![0.1; h],
                num_neurons: h,
            },
            GpuCrownLayer::Linear {
                weight: w1.into(),
                bias: None,
                out_features: h,
                in_features: i,
                cert_err: Default::default(),
            },
        ];
        let mut seed = vec![0.0f32; o * o];
        for r in 0..o {
            seed[r * o + r] = 1.0;
        }
        let zero_a = vec![0.0f32; o * o];
        let zb = vec![0.0f32; o];

        let cf = device
            .crown_backward_sound_resident_coeff_seeded_err(
                &layers,
                &seed,
                &seed,
                &zero_a,
                &zero_a,
                &zb,
                &zb,
                &zb,
                &zb,
                o,
                o,
                &[&pre_lower[..]],
                &[],
            )
            .expect("resident backward with gradient capture");

        assert_eq!(cf.relu_grads.len(), 1, "exactly one ReLU captured");
        let got = &cf.relu_grads[0];
        assert_eq!(got.len(), h);
        for k in 0..h {
            let s: f32 = (0..o).map(|j| w2[j * h + k].max(0.0)).sum();
            let expected = pre_lower[k] * s;
            let tol = 1e-3 * (1.0 + expected.abs());
            assert!(
                (got[k] - expected).abs() <= tol,
                "captured grad[{k}]: gpu={} expected={}",
                got[k],
                expected
            );
        }

        // The verdict path (empty relu_pre_lower) must capture nothing.
        let cf_none = device
            .crown_backward_sound_resident_coeff_seeded_err(
                &layers,
                &seed,
                &seed,
                &zero_a,
                &zero_a,
                &zb,
                &zb,
                &zb,
                &zb,
                o,
                o,
                &[],
                &[],
            )
            .expect("resident backward no capture");
        assert!(
            cf_none.relu_grads.is_empty(),
            "no capture when not requested"
        );
    }

    /// R-grad step 3a: the resnet FOLD (`crown_backward_sound_resident_resnet_seeded`)
    /// threads gradient capture across segments and accumulates them in fold order.
    /// Two Chain segments: seg0 = [Linear_A] (no ReLU), seg1 = [Linear_C, ReLU, Linear_D].
    /// With an identity seed the coefficient entering the ReLU is W_A·W_C, so the one
    /// captured gradient must equal pre_lower[c]·Σ_a max((W_A·W_C)[a,c], 0). Confirms
    /// the per-branch slicing, cross-segment accumulation, and 3-tuple return.
    #[test]
    fn crown_resnet_seeded_fold_captures_gradients_across_segments() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (o, h, k, i) = (2usize, 3usize, 4usize, 5usize);
        let mut state: u64 = 0xD00D_5EED;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let wa: Vec<f32> = (0..o * h).map(|_| rng() * 1.5).collect(); // O×H
        let wc: Vec<f32> = (0..h * k).map(|_| rng() * 1.5).collect(); // H×K
        let wd: Vec<f32> = (0..k * i).map(|_| rng()).collect(); // K×I
        let pre_lower: Vec<f32> = (0..k)
            .map(|n| if n == 1 { 0.0 } else { rng() * 0.5 - 0.6 })
            .collect();

        let seg0_layers = vec![GpuCrownLayer::Linear {
            weight: wa.clone().into(),
            bias: None,
            out_features: o,
            in_features: h,
            cert_err: Default::default(),
        }];
        let seg1_layers = vec![
            GpuCrownLayer::Linear {
                weight: wc.clone().into(),
                bias: None,
                out_features: h,
                in_features: k,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.5; k],
                upper_slope: vec![0.6; k],
                lower_intercept: vec![0.0; k],
                upper_intercept: vec![0.0; k],
                num_neurons: k,
            },
            GpuCrownLayer::Linear {
                weight: wd.into(),
                bias: None,
                out_features: k,
                in_features: i,
                cert_err: Default::default(),
            },
        ];
        let segments = vec![
            ResnetSegment::Chain(&seg0_layers),
            ResnetSegment::Chain(&seg1_layers),
        ];
        let mut seed = vec![0.0f32; o * o];
        for r in 0..o {
            seed[r * o + r] = 1.0;
        }
        let zb = vec![0.0f32; o];
        let in_lo = vec![-1.0f32; i];
        let in_hi = vec![1.0f32; i];

        let (_lo, _hi, grads) = device
            .crown_backward_sound_resident_resnet_seeded(
                &segments,
                &seed,
                &seed,
                &zb,
                &zb,
                o,
                o,
                &in_lo,
                &in_hi,
                &[&pre_lower[..]],
                &[],
                &[],
                false,
                &[],
                false,
            )
            .expect("resnet fold with gradient capture");

        assert_eq!(
            grads.len(),
            1,
            "exactly one ReLU captured across the two segments"
        );
        // a_at_relu = W_A · W_C  (O×K).
        let mut m = vec![0.0f32; o * k];
        for a in 0..o {
            for c in 0..k {
                let mut s = 0.0f32;
                for b in 0..h {
                    s += wa[a * h + b] * wc[b * k + c];
                }
                m[a * k + c] = s;
            }
        }
        for c in 0..k {
            let pos: f32 = (0..o).map(|a| m[a * k + c].max(0.0)).sum();
            let expected = pre_lower[c] * pos;
            let tol = 1e-2 * (1.0 + expected.abs());
            assert!(
                (grads[0][c] - expected).abs() <= tol,
                "fold grad[{c}]: gpu={} expected={}",
                grads[0][c],
                expected
            );
        }
    }

    /// R-beta-4 (acceptance gate): the GPU beta term matches the CPU β-CROWN formula
    /// (`apply_constrained_relu_beta_contribution`) to ULP. A single Activation with an
    /// identity seed gives, per output o: lower coeff la[o,i] = (o==i ? lower_slope[o] : 0)
    /// − signed_beta[i]; upper ua[o,i] = (o==i ? upper_slope[o] : 0) + signed_beta[i]; bias =
    /// the intercepts (beta does NOT touch bias). Concretizing over the pre-activation box
    /// [xl,xu] gives a closed form we check the GPU sound bound against (sound ⇒ GPU lower ≤
    /// exact, upper ≥ exact, within ULP). Also a β=0 control = the no-beta bound, byte-exact.
    #[test]
    fn crown_resnet_beta_matches_cpu_formula() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let nn = 3usize;
        let lower_slope = vec![0.5f32, 0.6, 0.4];
        let upper_slope = vec![0.7f32, 0.8, 0.5];
        let lower_intercept = vec![0.0f32, 0.1, 0.0];
        let upper_intercept = vec![0.2f32, 0.0, 0.1];
        let xl = vec![-1.0f32, -2.0, -0.5];
        let xu = vec![1.0f32, 2.0, 0.5];
        // neuron 0 ACTIVE β=0.5 (signed +0.5); neuron 1 none; neuron 2 INACTIVE β=0.3 (signed −0.3).
        let signed_beta = vec![0.5f32, 0.0, -0.3];

        let act = GpuCrownLayer::Activation {
            lower_slope: lower_slope.clone(),
            upper_slope: upper_slope.clone(),
            lower_intercept: lower_intercept.clone(),
            upper_intercept: upper_intercept.clone(),
            num_neurons: nn,
        };
        let seg_layers = vec![act];
        let segments = vec![ResnetSegment::Chain(&seg_layers)];
        // identity seed (num_specs = output_dim = nn).
        let mut seed = vec![0.0f32; nn * nn];
        for r in 0..nn {
            seed[r * nn + r] = 1.0;
        }
        let zb = vec![0.0f32; nn];

        // Closed-form expected bound given a per-neuron signed_beta vector.
        let expected = |sb: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let mut lo = vec![0.0f32; nn];
            let mut hi = vec![0.0f32; nn];
            for o in 0..nn {
                let mut l = lower_intercept[o];
                let mut u = upper_intercept[o];
                for i in 0..nn {
                    let la = (if o == i { lower_slope[o] } else { 0.0 }) - sb[i];
                    let ua = (if o == i { upper_slope[o] } else { 0.0 }) + sb[i];
                    l += if la >= 0.0 { la * xl[i] } else { la * xu[i] };
                    u += if ua >= 0.0 { ua * xu[i] } else { ua * xl[i] };
                }
                lo[o] = l;
                hi[o] = u;
            }
            (lo, hi)
        };

        // --- β control: empty beta must equal the closed form with signed_beta = 0 ---
        let (lo0, hi0, _g0) = device
            .crown_backward_sound_resident_resnet_seeded(
                &segments,
                &seed,
                &seed,
                &zb,
                &zb,
                nn,
                nn,
                &xl,
                &xu,
                &[],
                &[],
                &[],
                false,
                &[],
                false,
            )
            .expect("resnet beta=0 control");
        let (elo0, ehi0) = expected(&vec![0.0f32; nn]);
        for o in 0..nn {
            let tol = 1e-4 * (1.0 + elo0[o].abs().max(ehi0[o].abs()));
            assert!(
                (lo0[o] - elo0[o]).abs() <= tol,
                "β=0 lower[{o}]: gpu={} exp={}",
                lo0[o],
                elo0[o]
            );
            assert!(
                (hi0[o] - ehi0[o]).abs() <= tol,
                "β=0 upper[{o}]: gpu={} exp={}",
                hi0[o],
                ehi0[o]
            );
            assert!(lo0[o] <= elo0[o] + tol, "β=0 lower must be sound (≤ exact)");
            assert!(hi0[o] >= ehi0[o] - tol, "β=0 upper must be sound (≥ exact)");
        }

        // --- β applied: must match the CPU β-CROWN closed form to ULP, and stay sound ---
        let (lob, hib, _gb) = device
            .crown_backward_sound_resident_resnet_seeded(
                &segments,
                &seed,
                &seed,
                &zb,
                &zb,
                nn,
                nn,
                &xl,
                &xu,
                &[],
                &[&signed_beta[..]],
                &[],
                false,
                &[],
                false,
            )
            .expect("resnet beta applied");
        let (elob, ehib) = expected(&signed_beta);
        for o in 0..nn {
            let tol = 1e-4 * (1.0 + elob[o].abs().max(ehib[o].abs()));
            assert!(
                (lob[o] - elob[o]).abs() <= tol,
                "β lower[{o}]: gpu={} exp={} (signed_beta folded post-slope?)",
                lob[o],
                elob[o]
            );
            assert!(
                (hib[o] - ehib[o]).abs() <= tol,
                "β upper[{o}]: gpu={} exp={}",
                hib[o],
                ehib[o]
            );
            assert!(lob[o] <= elob[o] + tol, "β lower must be sound (≤ exact)");
            assert!(hib[o] >= ehib[o] - tol, "β upper must be sound (≥ exact)");
        }
        // The beta term must actually CHANGE the bound (guards against a silent no-op).
        let changed =
            (0..nn).any(|o| (lob[o] - lo0[o]).abs() > 1e-5 || (hib[o] - hi0[o]).abs() > 1e-5);
        assert!(changed, "beta must change the bound vs the β=0 control");
    }

    /// R1: resident single bias-free Linear encloses the host reference + samples.
    #[test]
    fn crown_backward_sound_resident_single_linear_matches_host() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x12EE_5151;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &(din, dout) in &[(4usize, 3usize), (16, 8), (33, 5)] {
            let w: Vec<f32> = (0..dout * din).map(|_| rng() * 0.8).collect();
            let layers = vec![GpuCrownLayer::Linear {
                weight: Arc::from(w.clone().into_boxed_slice()),
                bias: None,
                out_features: dout,
                in_features: din,
                cert_err: Default::default(),
            }];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.25).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.25).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident(&layers, &spec, dout, dout, &xl, &xu)
                .expect("resident");
            let (hlo, hhi) = device
                .crown_backward_sound_host(&layers, &spec, dout, dout, &xl, &xu)
                .expect("host");

            for k in 0..dout {
                assert!(
                    f64::from(rlo[k]) <= f64::from(hlo[k]) + 1e-4,
                    "lower not enclosing"
                );
                assert!(
                    f64::from(rhi[k]) >= f64::from(hhi[k]) - 1e-4,
                    "upper not enclosing"
                );
            }
            for t in 0..150 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 29 + i * 11) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let y = matmul(&w, &x, dout, din);
                for o in 0..dout {
                    assert!(rlo[o] <= y[o] + 1e-3 && y[o] <= rhi[o] + 1e-3, "UNSOUND r1");
                }
            }
        }
    }

    /// R1-DAZ (#gpu-metal-daz): a SUBNORMAL objective coefficient × a LARGE weight.
    /// On a Metal/DAZ adapter the subnormal operand flushes to 0 *before* the multiply,
    /// so the point coefficient `a·w` and `s = |A|@|W|` lose the whole (normal-range)
    /// product; only the weight-amplified `flushacc·slack·F32_MIN_NORMAL` term certifies
    /// that lost mass back. The GPU-resident bound must still enclose the EXACT objective
    /// (and the CPU host). Without the flushacc term this collapses to ~0 and the upper
    /// bound drops below the true ~2^-110 output → UNSOUND; with it, the bound stays
    /// outward. On a preserving path this is a widening-only non-regression; the
    /// live gradual-underflow probe, not Vulkan/NVIDIA naming, establishes that fact.
    #[test]
    fn crown_backward_sound_resident_daz_subnormal_coeff_stays_outward() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // (subnormal coeff a, large weight w): the exact product a·w is a NORMAL f32
        // that a flush-to-zero GPU drops to 0. obj(x) = a·w·x over x ∈ [0.5, 1.5].
        let cases: &[(f32, f32)] = &[
            (2.0f32.powi(-130), 2.0f32.powi(20)),  // a·w = 2^-110
            (2.0f32.powi(-135), 2.0f32.powi(30)),  // 2^-105
            (f32::from_bits(1), 2.0f32.powi(100)), // 2^-149 · 2^100 = 2^-49
        ];
        for &(a_sub, w_large) in cases {
            let layers = vec![GpuCrownLayer::Linear {
                weight: Arc::from(vec![w_large].into_boxed_slice()),
                bias: None,
                out_features: 1,
                in_features: 1,
                cert_err: Default::default(),
            }];
            let spec = vec![a_sub]; // 1×1 objective coefficient (subnormal)
            let (xl, xu) = (vec![0.5f32], vec![1.5f32]);
            let (rlo, rhi) = device
                .crown_backward_sound_resident(&layers, &spec, 1, 1, &xl, &xu)
                .expect("resident");
            let (hlo, hhi) = device
                .crown_backward_sound_host(&layers, &spec, 1, 1, &xl, &xu)
                .expect("host");
            // Exact objective over x ∈ [0.5, 1.5]: obj(x) = a_sub·w_large·x (f64-exact,
            // f32·f32 ⊂ f64), monotone increasing (coeff > 0), so extrema at the ends.
            let coeff = f64::from(a_sub) * f64::from(w_large);
            let (ylo, yhi) = (coeff * 0.5, coeff * 1.5);
            assert!(
                f64::from(rlo[0]) <= ylo && f64::from(rhi[0]) >= yhi,
                "DAZ UNSOUND: a={a_sub:e} w={w_large:e} exact obj [{ylo:e}, {yhi:e}] \
                 not enclosed by GPU [{}, {}]",
                rlo[0],
                rhi[0]
            );
            // The independent host oracle must enclose the same exact range.
            // The two certified implementations use different error accounting,
            // so neither sound interval is required to enclose the other.
            assert!(
                f64::from(hlo[0]) <= ylo && f64::from(hhi[0]) >= yhi,
                "DAZ HOST UNSOUND: a={a_sub:e} w={w_large:e} exact obj [{ylo:e}, {yhi:e}] \
                 not enclosed by host [{}, {}]",
                hlo[0],
                hhi[0]
            );
        }
    }

    /// R2: resident MULTI-LAYER Linear + bias (the ping-pong residency loop).
    /// Net: x → W1·x+b1 (h) → W2·(...)+b2 (dout). Backward layers are output→input:
    /// [W2-layer, W1-layer]. Must enclose the host reference AND sampled outputs.
    #[test]
    fn crown_backward_sound_resident_multilayer_bias_matches_host() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x77AB_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &(din, h, dout) in &[(5usize, 7usize, 4usize), (12, 9, 6), (20, 16, 3)] {
            let w1: Vec<f32> = (0..h * din).map(|_| rng() * 0.7).collect(); // (h × din)
            let b1: Vec<f32> = (0..h).map(|_| rng() * 0.5).collect();
            let w2: Vec<f32> = (0..dout * h).map(|_| rng() * 0.7).collect(); // (dout × h)
            let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.5).collect();
            let layers = vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(w2.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b2.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: h,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Linear {
                    weight: Arc::from(w1.clone().into_boxed_slice()),
                    bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                    out_features: h,
                    in_features: din,
                    cert_err: Default::default(),
                },
            ];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident(&layers, &spec, dout, dout, &xl, &xu)
                .expect("resident");
            let (hlo, hhi) = device
                .crown_backward_sound_host(&layers, &spec, dout, dout, &xl, &xu)
                .expect("host");

            for k in 0..dout {
                assert!(
                    f64::from(rlo[k]) <= f64::from(hlo[k]) + 2e-4,
                    "({din},{h},{dout}) k{k}: resident lower {} not <= host {}",
                    rlo[k],
                    hlo[k]
                );
                assert!(
                    f64::from(rhi[k]) >= f64::from(hhi[k]) - 2e-4,
                    "({din},{h},{dout}) k{k}: resident upper {} not >= host {}",
                    rhi[k],
                    hhi[k]
                );
                assert!(rlo[k] <= rhi[k]);
            }
            for t in 0..200 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 31 + i * 7) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut y1 = matmul(&w1, &x, h, din);
                for j in 0..h {
                    y1[j] += b1[j];
                }
                let mut y2 = matmul(&w2, &y1, dout, h);
                for j in 0..dout {
                    y2[j] += b2[j];
                }
                for o in 0..dout {
                    assert!(
                        rlo[o] <= y2[o] + 2e-3 && y2[o] <= rhi[o] + 2e-3,
                        "UNSOUND r2: out[{o}]={} not in [{}, {}]",
                        y2[o],
                        rlo[o],
                        rhi[o]
                    );
                }
            }
        }
    }

    /// R3: resident Linear→Activation→Linear. (a) identity activation makes the
    /// net affine — resident must enclose the affine forward AND the host;
    /// (b) an arbitrary valid relaxation — resident must enclose the host.
    #[test]
    fn crown_backward_sound_resident_activation_matches_host() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x5AC7_9001;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &(din, h, dout) in &[(5usize, 6usize, 4usize), (10, 8, 5)] {
            let w1: Vec<f32> = (0..h * din).map(|_| rng() * 0.7).collect();
            let b1: Vec<f32> = (0..h).map(|_| rng() * 0.4).collect();
            let w2: Vec<f32> = (0..dout * h).map(|_| rng() * 0.7).collect();
            let b2: Vec<f32> = (0..dout).map(|_| rng() * 0.4).collect();
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.15).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.15).collect();

            let mk = |ls: Vec<f32>, us: Vec<f32>, li: Vec<f32>, ui: Vec<f32>| {
                vec![
                    GpuCrownLayer::Linear {
                        weight: Arc::from(w2.clone().into_boxed_slice()),
                        bias: Some(Arc::from(b2.clone().into_boxed_slice())),
                        out_features: dout,
                        in_features: h,
                        cert_err: Default::default(),
                    },
                    GpuCrownLayer::Activation {
                        lower_slope: ls,
                        upper_slope: us,
                        lower_intercept: li,
                        upper_intercept: ui,
                        num_neurons: h,
                    },
                    GpuCrownLayer::Linear {
                        weight: Arc::from(w1.clone().into_boxed_slice()),
                        bias: Some(Arc::from(b1.clone().into_boxed_slice())),
                        out_features: h,
                        in_features: din,
                        cert_err: Default::default(),
                    },
                ]
            };

            // (a) identity activation -> affine net.
            let id = mk(vec![1.0; h], vec![1.0; h], vec![0.0; h], vec![0.0; h]);
            let (rlo, rhi) = device
                .crown_backward_sound_resident(&id, &spec, dout, dout, &xl, &xu)
                .expect("res id");
            let (hlo, hhi) = device
                .crown_backward_sound_host(&id, &spec, dout, dout, &xl, &xu)
                .expect("host id");
            for k in 0..dout {
                assert!(
                    f64::from(rlo[k]) <= f64::from(hlo[k]) + 3e-4,
                    "id lower enclose"
                );
                assert!(
                    f64::from(rhi[k]) >= f64::from(hhi[k]) - 3e-4,
                    "id upper enclose"
                );
            }
            for t in 0..200 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 23 + i * 5) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut y1 = matmul(&w1, &x, h, din);
                for j in 0..h {
                    y1[j] += b1[j];
                }
                let mut y2 = matmul(&w2, &y1, dout, h);
                for j in 0..dout {
                    y2[j] += b2[j];
                }
                for o in 0..dout {
                    assert!(
                        rlo[o] <= y2[o] + 3e-3 && y2[o] <= rhi[o] + 3e-3,
                        "UNSOUND r3 id"
                    );
                }
            }

            // (b) a REAL ReLU relaxation (asymmetric ls≠us) with a CONCRETE
            // soundness check: the resident bounds must enclose ReLU(W1·x+b1)→W2.
            // Pre-activation bounds via IBP through W1 over the input box.
            let mut pl = vec![0.0f32; h];
            let mut pu = vec![0.0f32; h];
            for i in 0..h {
                let mut lo = b1[i];
                let mut hi = b1[i];
                for j in 0..din {
                    let w = w1[i * din + j];
                    if w >= 0.0 {
                        lo += w * xl[j];
                        hi += w * xu[j];
                    } else {
                        lo += w * xu[j];
                        hi += w * xl[j];
                    }
                }
                pl[i] = lo;
                pu[i] = hi;
            }
            // CROWN ReLU relaxation: stable → exact; unstable → lower y≥0,
            // upper y ≤ (u/(u−l))·x + (−u·l/(u−l)).
            let (mut ls, mut us, mut li, mut ui) = (
                vec![0.0f32; h],
                vec![0.0f32; h],
                vec![0.0f32; h],
                vec![0.0f32; h],
            );
            for i in 0..h {
                let (l, u) = (pl[i], pu[i]);
                if l >= 0.0 {
                    ls[i] = 1.0;
                    us[i] = 1.0;
                } else if u <= 0.0 {
                    // all zero (inactive)
                } else {
                    let slope = u / (u - l);
                    ls[i] = 0.0;
                    us[i] = slope;
                    li[i] = 0.0;
                    ui[i] = -u * l / (u - l); // = slope·(−l) ≥ 0
                }
            }
            let rx = mk(ls, us, li, ui);
            let (rlo, rhi) = device
                .crown_backward_sound_resident(&rx, &spec, dout, dout, &xl, &xu)
                .expect("res rx");
            for k in 0..dout {
                assert!(rlo[k].is_finite() && rhi[k].is_finite() && rlo[k] <= rhi[k]);
            }
            // Concrete soundness: enclose the true ReLU network output.
            for t in 0..300 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 19 + i * 3) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut y1 = matmul(&w1, &x, h, din);
                for j in 0..h {
                    y1[j] = (y1[j] + b1[j]).max(0.0); // ReLU
                }
                let mut y2 = matmul(&w2, &y1, dout, h);
                for j in 0..dout {
                    y2[j] += b2[j];
                }
                for o in 0..dout {
                    assert!(
                        rlo[o] <= y2[o] + 3e-3 && y2[o] <= rhi[o] + 3e-3,
                        "UNSOUND r3 relu: out[{o}]={} not in [{}, {}]",
                        y2[o],
                        rlo[o],
                        rhi[o]
                    );
                }
            }
        }
    }

    /// R4 diagnostic: the explicitly unworded resident single-Conv coefficient
    /// walk must enclose sampled forward outputs. Dedicated tests above pin the
    /// worded route and gate-on/off identity separately.
    #[test]
    fn crown_backward_sound_resident_single_conv_raw_is_sound() {
        let _g = gpu_test_serial_guard();
        // Raw Conv arithmetic/enclosure oracle; explicitly disable receipts so
        // the helper can inspect the pre-concretize value/error channels.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        let mut state: u64 = 0xC0AB_2026;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let (ic, oc, kh, kw, ih, iw) = (1usize, 2usize, 2usize, 2usize, 3usize, 3usize);
        let (oh, ow) = (ih - kh + 1, iw - kw + 1);
        let out_dim = oc * oh * ow; // 8
        let in_dim = ic * ih * iw; // 9
        for _ in 0..4 {
            let weight_col: Vec<f32> = (0..oc * ic * kh * kw).map(|_| rng() * 0.8).collect();
            let layers = vec![GpuCrownLayer::Conv2d {
                weight_col: Arc::from(weight_col.clone().into_boxed_slice()),
                bias_expanded: None,
                out_channels: oc,
                in_channels: ic,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                out_h: oh,
                out_w: ow,
                in_h: ih,
                in_w: iw,
                cert_err: Default::default(),
            }];
            let mut spec = vec![0.0f32; out_dim * out_dim];
            for i in 0..out_dim {
                spec[i * out_dim + i] = 1.0;
            }
            let xc: Vec<f32> = (0..in_dim).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let (rlo, rhi) =
                unworded_resident_test_bounds(&device, &layers, &spec, out_dim, out_dim, &xl, &xu)
                    .expect("res conv");
            for k in 0..out_dim {
                assert!(rlo[k] <= rhi[k]);
            }
            // conv forward: out[oc,oh,ow] = Σ_{kh,kw} W[oc,kh*KW+kw]·x[(oh+kh)*IW+(ow+kw)]
            for t in 0..200 {
                let x: Vec<f32> = (0..in_dim)
                    .map(|i| xl[i] + (((t * 17 + i * 9) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                for c in 0..oc {
                    for yy in 0..oh {
                        for xx in 0..ow {
                            let mut sm = 0.0f32;
                            for a in 0..kh {
                                for b in 0..kw {
                                    sm += weight_col[c * (ic * kh * kw) + a * kw + b]
                                        * x[(yy + a) * iw + (xx + b)];
                                }
                            }
                            let o = c * oh * ow + yy * ow + xx;
                            assert!(
                                rlo[o] <= sm + 3e-3 && sm <= rhi[o] + 3e-3,
                                "UNSOUND r4 conv: out[{o}]={sm} not in [{}, {}]",
                                rlo[o],
                                rhi[o]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Seeded path: an asymmetric frontier (lower_a≠upper_a, lower_b≠upper_b)
    /// composed with an affine Linear suffix. The seeded bounds must ENCLOSE the
    /// exact composed linear functions L_lo(x)=lower_a·(W·x+b)+lower_b and
    /// U_hi(x)=upper_a·(W·x+b)+upper_b — validating that the seed coefficient AND
    /// bias are incorporated soundly.
    #[test]
    fn crown_backward_sound_resident_seeded_frontier_is_sound() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x5EED_0001;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &(num_specs, cdim, din) in &[(3usize, 5usize, 4usize), (6, 8, 7)] {
            // Affine suffix: one Linear (out_features=cdim, in_features=din).
            let w: Vec<f32> = (0..cdim * din).map(|_| rng() * 0.6).collect();
            let bsuf: Vec<f32> = (0..cdim).map(|_| rng() * 0.4).collect();
            let layers = vec![GpuCrownLayer::Linear {
                weight: Arc::from(w.clone().into_boxed_slice()),
                bias: Some(Arc::from(bsuf.clone().into_boxed_slice())),
                out_features: cdim,
                in_features: din,
                cert_err: Default::default(),
            }];
            // Asymmetric frontier (num_specs × cdim) + bias.
            let lower_a: Vec<f32> = (0..num_specs * cdim).map(|_| rng() * 0.8).collect();
            let upper_a: Vec<f32> = (0..num_specs * cdim).map(|_| rng() * 0.8).collect();
            let lower_b: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();
            let upper_b: Vec<f32> = (0..num_specs).map(|_| rng() * 0.3).collect();

            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident_seeded(
                    &layers, &lower_a, &upper_a, &lower_b, &upper_b, num_specs, cdim, &xl, &xu,
                )
                .expect("seeded resident");

            for t in 0..200 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 27 + i * 5) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                // z = W·x + b  (the suffix output, cdim-dim)
                let z: Vec<f32> = (0..cdim)
                    .map(|k| (0..din).map(|j| w[k * din + j] * x[j]).sum::<f32>() + bsuf[k])
                    .collect();
                for s in 0..num_specs {
                    let l_lo: f32 =
                        (0..cdim).map(|k| lower_a[s * cdim + k] * z[k]).sum::<f32>() + lower_b[s];
                    let u_hi: f32 =
                        (0..cdim).map(|k| upper_a[s * cdim + k] * z[k]).sum::<f32>() + upper_b[s];
                    assert!(
                        rlo[s] <= l_lo + 3e-3,
                        "UNSOUND seeded lower: spec{s} rlo={} > L_lo={l_lo}",
                        rlo[s]
                    );
                    assert!(
                        rhi[s] >= u_hi - 3e-3,
                        "UNSOUND seeded upper: spec{s} rhi={} < U_hi={u_hi}",
                        rhi[s]
                    );
                }
            }
        }
    }

    /// Residual block out = F(x) + x (identity skip). With an affine branch the
    /// composition is exact: out = (W+I)·x + b for a single-Linear F. The residual
    /// backward must enclose the sampled out — validating the fork + branch backward
    /// + certified skip-add (the core residual operation for resnets).
    #[test]
    fn crown_backward_sound_resident_residual_block_is_sound() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x4E51_DEAD;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &d in &[4usize, 9, 16] {
            // F = single Linear (D→D); identity skip; spec = identity (D×D).
            let w: Vec<f32> = (0..d * d).map(|_| rng() * 0.5).collect();
            let b: Vec<f32> = (0..d).map(|_| rng() * 0.3).collect();
            let branch = vec![GpuCrownLayer::Linear {
                weight: Arc::from(w.clone().into_boxed_slice()),
                bias: Some(Arc::from(b.clone().into_boxed_slice())),
                out_features: d,
                in_features: d,
                cert_err: Default::default(),
            }];
            let mut seed = vec![0.0f32; d * d];
            for i in 0..d {
                seed[i * d + i] = 1.0;
            }
            let zb = vec![0.0f32; d];
            let xc: Vec<f32> = (0..d).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.2).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.2).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident_residual(
                    &branch, &seed, &seed, &zb, &zb, d, d, &xl, &xu,
                )
                .expect("residual block");

            for t in 0..200 {
                let x: Vec<f32> = (0..d)
                    .map(|i| xl[i] + (((t * 31 + i * 7) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                // out = F(x) + x = (W·x + b) + x
                for o in 0..d {
                    let fx: f32 = (0..d).map(|j| w[o * d + j] * x[j]).sum::<f32>() + b[o];
                    let out = fx + x[o];
                    assert!(
                        rlo[o] <= out + 3e-3 && out <= rhi[o] + 3e-3,
                        "UNSOUND residual: out[{o}]={out} not in [{}, {}]",
                        rlo[o],
                        rhi[o]
                    );
                }
            }
        }
    }

    /// STACKED resnet: out = Linear(block2(block1(x))), each block out = F(z)+z
    /// (identity skip, affine F). Validates segment composition WITH error carried
    /// between blocks (seeding err=0 between segments would be unsound). Backward
    /// segments: [Chain(Linear_out), Residual(F2), Residual(F1)].
    #[test]
    fn crown_backward_sound_resident_stacked_resnet_is_sound() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x57AC_C0DE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let (d, dout) = (6usize, 4usize);
        for _ in 0..3 {
            let f1w: Vec<f32> = (0..d * d).map(|_| rng() * 0.4).collect();
            let f1b: Vec<f32> = (0..d).map(|_| rng() * 0.2).collect();
            let f2w: Vec<f32> = (0..d * d).map(|_| rng() * 0.4).collect();
            let f2b: Vec<f32> = (0..d).map(|_| rng() * 0.2).collect();
            let ow: Vec<f32> = (0..dout * d).map(|_| rng() * 0.5).collect();
            let ob: Vec<f32> = (0..dout).map(|_| rng() * 0.3).collect();

            let lin = |w: &[f32], b: &[f32], o: usize, i: usize| GpuCrownLayer::Linear {
                weight: Arc::from(w.to_vec().into_boxed_slice()),
                bias: Some(Arc::from(b.to_vec().into_boxed_slice())),
                out_features: o,
                in_features: i,
                cert_err: Default::default(),
            };
            let out_chain = vec![lin(&ow, &ob, dout, d)];
            let f2_branch = vec![lin(&f2w, &f2b, d, d)];
            let f1_branch = vec![lin(&f1w, &f1b, d, d)];
            let segments = vec![
                ResnetSegment::Chain(&out_chain),
                ResnetSegment::Residual(&f2_branch),
                ResnetSegment::Residual(&f1_branch),
            ];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..d).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.15).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.15).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident_resnet(&segments, &spec, dout, dout, &xl, &xu)
                .expect("stacked resnet");

            let mm = |w: &[f32], x: &[f32], r: usize, c: usize| -> Vec<f32> {
                (0..r)
                    .map(|i| (0..c).map(|j| w[i * c + j] * x[j]).sum())
                    .collect()
            };
            for t in 0..200 {
                let x: Vec<f32> = (0..d)
                    .map(|i| xl[i] + (((t * 23 + i * 5) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut z1 = mm(&f1w, &x, d, d);
                for i in 0..d {
                    z1[i] += f1b[i] + x[i];
                }
                let mut z2 = mm(&f2w, &z1, d, d);
                for i in 0..d {
                    z2[i] += f2b[i] + z1[i];
                }
                let mut out = mm(&ow, &z2, dout, d);
                for k in 0..dout {
                    out[k] += ob[k];
                }
                for k in 0..dout {
                    assert!(
                        rlo[k] <= out[k] + 4e-3 && out[k] <= rhi[k] + 4e-3,
                        "UNSOUND stacked resnet: out[{k}]={} not in [{}, {}]",
                        out[k],
                        rlo[k],
                        rhi[k]
                    );
                }
            }
        }
    }

    /// #unsat-keystone validation: on a DEEP affine resnet the certified f32
    /// coefficient error grows ~|W| per residual block (the L1 blow-up ca23d58
    /// diagnosed on cifar100/tinyimagenet). The per-segment `frontier_abs`
    /// error-concretization (gate on) folds that growing coefficient error into the
    /// non-amplifying scalar bias error at each segment boundary, capping the blow-up.
    /// Asserts the concretized bound is (a) SOUND — contains every sampled concrete
    /// output, so a too-small `frontier_abs` would fail — and (b) no looser than the
    /// un-concretized bound on the deep stack (the keystone's purpose).
    #[test]
    fn crown_backward_resnet_err_concretize_caps_soundly() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x4357_0117;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let d = 6usize;
        let depth = 14usize; // deep enough for the un-concretized error to blow up
        let ws: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..d * d).map(|_| rng() * 0.5).collect())
            .collect();
        let bs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..d).map(|_| rng() * 0.2).collect())
            .collect();
        let xc: Vec<f32> = (0..d).map(|_| rng()).collect();
        let xl: Vec<f32> = xc.iter().map(|&c| c - 0.1).collect();
        let xu: Vec<f32> = xc.iter().map(|&c| c + 0.1).collect();

        // Forward IBP bounds at each boundary z_0=x .. z_depth (residual affine
        // z_{k+1} = (W_k + I)·z_k + b_k) → the frontier abs-max bounds.
        let mut z_lo = vec![xl.clone()];
        let mut z_hi = vec![xu.clone()];
        for k in 0..depth {
            let lo_prev = z_lo.last().unwrap().clone();
            let hi_prev = z_hi.last().unwrap().clone();
            let mut nlo = vec![0.0f32; d];
            let mut nhi = vec![0.0f32; d];
            for i in 0..d {
                let mut l = bs[k][i];
                let mut h = bs[k][i];
                for j in 0..d {
                    let coef = ws[k][i * d + j] + if i == j { 1.0 } else { 0.0 };
                    if coef >= 0.0 {
                        l += coef * lo_prev[j];
                        h += coef * hi_prev[j];
                    } else {
                        l += coef * hi_prev[j];
                        h += coef * lo_prev[j];
                    }
                }
                nlo[i] = l;
                nhi[i] = h;
            }
            z_lo.push(nlo);
            z_hi.push(nhi);
        }

        // Backward segments: identity output Chain, then Residual(F_{depth-1})..Residual(F_0).
        let lin = |w: &[f32], b: &[f32]| GpuCrownLayer::Linear {
            weight: Arc::from(w.to_vec().into_boxed_slice()),
            bias: Some(Arc::from(b.to_vec().into_boxed_slice())),
            out_features: d,
            in_features: d,
            cert_err: Default::default(),
        };
        let mut id_w = vec![0.0f32; d * d];
        for i in 0..d {
            id_w[i * d + i] = 1.0;
        }
        let out_chain = vec![lin(&id_w, &vec![0.0f32; d])];
        let branches: Vec<Vec<GpuCrownLayer>> = (0..depth)
            .rev()
            .map(|k| vec![lin(&ws[k], &bs[k])])
            .collect();
        let mut segments: Vec<ResnetSegment> = vec![ResnetSegment::Chain(&out_chain)];
        for br in &branches {
            segments.push(ResnetSegment::Residual(br.as_slice()));
        }

        // frontier_abs in backward-segment order: [z_depth, z_{depth-1}, .., z_0].
        let absmax = |lo: &[f32], hi: &[f32]| -> Vec<f32> {
            (0..d).map(|i| lo[i].abs().max(hi[i].abs())).collect()
        };
        let mut frontier: Vec<Vec<f32>> = vec![absmax(&z_lo[depth], &z_hi[depth])];
        for k in (0..depth).rev() {
            frontier.push(absmax(&z_lo[k], &z_hi[k]));
        }
        assert_eq!(frontier.len(), segments.len());
        let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();

        let mut seed = vec![0.0f32; d * d];
        for i in 0..d {
            seed[i * d + i] = 1.0;
        }
        let zb = vec![0.0f32; d];

        // Gate OFF (empty frontier_abs ⇒ no concretization, the verdict default).
        let (lo_off, hi_off, _) = device
            .crown_backward_sound_resident_resnet_seeded(
                &segments,
                &seed,
                &seed,
                &zb,
                &zb,
                d,
                d,
                &xl,
                &xu,
                &[],
                &[],
                &[],
                false,
                &[],
                false,
            )
            .expect("gate off");

        // Gate ON (frontier_abs populated + env), env restored on drop.
        let (lo_on, hi_on) = {
            let _guard = ScopedEnvVar::set("NY_RESNET_ERR_CONCRETIZE", "1");
            let (lo, hi, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed,
                    &seed,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &frontier_refs,
                    false,
                    &[],
                    false,
                )
                .expect("gate on");
            (lo, hi)
        };

        let width = |lo: &[f32], hi: &[f32]| -> f32 { (0..d).map(|i| hi[i] - lo[i]).sum() };
        let (w_off, w_on) = (width(&lo_off, &hi_off), width(&lo_on, &hi_on));
        eprintln!("[err-concretize] depth={depth} width_off={w_off} width_on={w_on}");

        // (a) SOUNDNESS: the concretized bound must contain every concrete output.
        for t in 0..300 {
            let x: Vec<f32> = (0..d)
                .map(|i| xl[i] + (((t * 23 + i * 5) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                .collect();
            let mut z = x.clone();
            for k in 0..depth {
                let mut nz = vec![0.0f32; d];
                for i in 0..d {
                    let mut s = bs[k][i] + z[i];
                    for j in 0..d {
                        s += ws[k][i * d + j] * z[j];
                    }
                    nz[i] = s;
                }
                z = nz;
            }
            for o in 0..d {
                assert!(
                    lo_on[o] <= z[o] + 5e-3 && z[o] <= hi_on[o] + 5e-3,
                    "UNSOUND concretized resnet: out[{o}]={} not in [{}, {}]",
                    z[o],
                    lo_on[o],
                    hi_on[o]
                );
            }
        }

        // (b) CAPPING: the un-concretized certified error blows up through the deep
        // stack; the concretization must stay finite and no looser than gate-off.
        assert!(w_on.is_finite(), "concretized width not finite: {w_on}");
        assert!(
            w_on <= w_off + 1e-3,
            "concretization did not cap the blow-up: width_on={w_on} width_off={w_off}"
        );
    }

    /// PROJECTION residual block: out = F(x) + P(x) (both affine, D_in→D_out), then
    /// Linear. Validates the two-branch fork/merge (merge_streams adds BOTH coeff and
    /// bias, with the incoming bias counted once). Backward: [Chain(Linear_out),
    /// ResidualProj(F, P)].
    #[test]
    fn crown_backward_sound_resident_projection_block_is_sound() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0x9803_1CE5;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let (din, dmid, dout) = (5usize, 4usize, 3usize);
        for _ in 0..3 {
            let fw: Vec<f32> = (0..dmid * din).map(|_| rng() * 0.5).collect();
            let fb: Vec<f32> = (0..dmid).map(|_| rng() * 0.3).collect();
            let pw: Vec<f32> = (0..dmid * din).map(|_| rng() * 0.5).collect();
            let pb: Vec<f32> = (0..dmid).map(|_| rng() * 0.3).collect();
            let ow: Vec<f32> = (0..dout * dmid).map(|_| rng() * 0.5).collect();
            let ob: Vec<f32> = (0..dout).map(|_| rng() * 0.3).collect();

            let lin = |w: &[f32], b: &[f32], o: usize, i: usize| GpuCrownLayer::Linear {
                weight: Arc::from(w.to_vec().into_boxed_slice()),
                bias: Some(Arc::from(b.to_vec().into_boxed_slice())),
                out_features: o,
                in_features: i,
                cert_err: Default::default(),
            };
            let out_chain = vec![lin(&ow, &ob, dout, dmid)];
            let f_branch = vec![lin(&fw, &fb, dmid, din)];
            let p_branch = vec![lin(&pw, &pb, dmid, din)];
            let segments = vec![
                ResnetSegment::Chain(&out_chain),
                ResnetSegment::ResidualProj(&f_branch, &p_branch),
            ];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }
            let xc: Vec<f32> = (0..din).map(|_| rng()).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.15).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.15).collect();

            let (rlo, rhi) = device
                .crown_backward_sound_resident_resnet(&segments, &spec, dout, dout, &xl, &xu)
                .expect("projection block");

            let mm = |w: &[f32], x: &[f32], r: usize, c: usize| -> Vec<f32> {
                (0..r)
                    .map(|i| (0..c).map(|j| w[i * c + j] * x[j]).sum())
                    .collect()
            };
            for t in 0..200 {
                let x: Vec<f32> = (0..din)
                    .map(|i| xl[i] + (((t * 29 + i * 3) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut fx = mm(&fw, &x, dmid, din);
                let px = mm(&pw, &x, dmid, din);
                for i in 0..dmid {
                    fx[i] += fb[i] + px[i] + pb[i]; // out = F(x) + P(x)
                }
                let mut out = mm(&ow, &fx, dout, dmid);
                for k in 0..dout {
                    out[k] += ob[k];
                }
                for k in 0..dout {
                    assert!(
                        rlo[k] <= out[k] + 4e-3 && out[k] <= rhi[k] + 4e-3,
                        "UNSOUND projection: out[{k}]={} not in [{}, {}]",
                        out[k],
                        rlo[k],
                        rhi[k]
                    );
                }
            }
        }
    }

    /// R4 composition: Conv → ReLU → Linear (cifar100's architecture shape).
    /// Resident must enclose the host reference AND the true conv-relu-linear
    /// forward. Backward layers (output→input): [Linear, Activation, Conv2d].
    #[test]
    fn crown_backward_sound_resident_conv_relu_linear_is_sound() {
        let _g = gpu_test_serial_guard();
        // Raw Conv arithmetic/enclosure oracle; production AUTO is covered by
        // the worded Conv receipt and equivalence tests above.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        let mut state: u64 = 0xC04E_70F0;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // Conv: IC=2, OC=3, K=2×2, IH=IW=4 → OH=OW=3 ; flatten=OC·OH·OW=27 ; Linear 27→dout.
        let (ic, oc, kh, kw, ih, iw) = (2usize, 3usize, 2usize, 2usize, 4usize, 4usize);
        let (oh, ow) = (ih - kh + 1, iw - kw + 1);
        let conv_out = oc * oh * ow; // 27
        let in_dim = ic * ih * iw; // 32
        let dout = 4usize;

        for _ in 0..3 {
            let weight_col: Vec<f32> = (0..oc * ic * kh * kw).map(|_| rng() * 0.6).collect();
            let wlin: Vec<f32> = (0..dout * conv_out).map(|_| rng() * 0.4).collect();
            let blin: Vec<f32> = (0..dout).map(|_| rng() * 0.3).collect();

            // Pre-activation (post-conv) bounds via IBP over the input box.
            let xc: Vec<f32> = (0..in_dim).map(|_| rng() * 0.5).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - 0.1).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + 0.1).collect();
            let conv_fwd = |x: &[f32]| -> Vec<f32> {
                let mut out = vec![0.0f32; conv_out];
                for c in 0..oc {
                    for yy in 0..oh {
                        for xx in 0..ow {
                            let mut sm = 0.0f32;
                            for chan in 0..ic {
                                for a in 0..kh {
                                    for b in 0..kw {
                                        sm += weight_col
                                            [c * (ic * kh * kw) + chan * kh * kw + a * kw + b]
                                            * x[chan * ih * iw + (yy + a) * iw + (xx + b)];
                                    }
                                }
                            }
                            out[c * oh * ow + yy * ow + xx] = sm;
                        }
                    }
                }
                out
            };
            // IBP post-conv bounds (conv is linear): min/max over the box per output.
            let (mut pl, mut pu) = (vec![0.0f32; conv_out], vec![0.0f32; conv_out]);
            for o in 0..conv_out {
                // recompute the linear map row for output o by probing unit inputs.
                let mut lo = 0.0f32;
                let mut hi = 0.0f32;
                // coefficient of input j on output o:
                for j in 0..in_dim {
                    let mut e = vec![0.0f32; in_dim];
                    e[j] = 1.0;
                    let w = conv_fwd(&e)[o];
                    if w >= 0.0 {
                        lo += w * xl[j];
                        hi += w * xu[j];
                    } else {
                        lo += w * xu[j];
                        hi += w * xl[j];
                    }
                }
                pl[o] = lo;
                pu[o] = hi;
            }
            let (mut ls, mut us, li, mut ui) = (
                vec![0.0f32; conv_out],
                vec![0.0f32; conv_out],
                vec![0.0f32; conv_out],
                vec![0.0f32; conv_out],
            );
            for i in 0..conv_out {
                let (l, u) = (pl[i], pu[i]);
                if l >= 0.0 {
                    ls[i] = 1.0;
                    us[i] = 1.0;
                } else if u <= 0.0 {
                } else {
                    us[i] = u / (u - l);
                    ui[i] = -u * l / (u - l);
                }
            }

            let layers = vec![
                GpuCrownLayer::Linear {
                    weight: Arc::from(wlin.clone().into_boxed_slice()),
                    bias: Some(Arc::from(blin.clone().into_boxed_slice())),
                    out_features: dout,
                    in_features: conv_out,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Activation {
                    lower_slope: ls,
                    upper_slope: us,
                    lower_intercept: li,
                    upper_intercept: ui,
                    num_neurons: conv_out,
                },
                GpuCrownLayer::Conv2d {
                    weight_col: Arc::from(weight_col.clone().into_boxed_slice()),
                    bias_expanded: None,
                    out_channels: oc,
                    in_channels: ic,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                    out_h: oh,
                    out_w: ow,
                    in_h: ih,
                    in_w: iw,
                    cert_err: Default::default(),
                },
            ];
            let mut spec = vec![0.0f32; dout * dout];
            for i in 0..dout {
                spec[i * dout + i] = 1.0;
            }

            let (rlo, rhi) =
                unworded_resident_test_bounds(&device, &layers, &spec, dout, dout, &xl, &xu)
                    .expect("res conv-relu-linear");
            for k in 0..dout {
                assert!(rlo[k].is_finite() && rhi[k].is_finite() && rlo[k] <= rhi[k]);
            }
            // Concrete soundness vs true Conv→ReLU→Linear forward.
            for t in 0..300 {
                let x: Vec<f32> = (0..in_dim)
                    .map(|i| xl[i] + (((t * 13 + i * 7) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut hpost = conv_fwd(&x);
                for v in hpost.iter_mut() {
                    *v = v.max(0.0);
                }
                let mut y = matmul(&wlin, &hpost, dout, conv_out);
                for j in 0..dout {
                    y[j] += blin[j];
                }
                for o in 0..dout {
                    assert!(
                        rlo[o] <= y[o] + 4e-3 && y[o] <= rhi[o] + 4e-3,
                        "UNSOUND r4 conv-relu-linear: out[{o}]={} not in [{}, {}]",
                        y[o],
                        rlo[o],
                        rhi[o]
                    );
                }
            }
        }
    }

    /// #w4-conv-err-per-entry: the per-entry certified conv error (default) vs the
    /// legacy row-max·‖W‖₁ broadcast (NY_CONV_ERR_ROWMAX=1) on a DEEP
    /// `(Conv→ReLU)×depth → Conv` chain — the cifar100 shape where the broadcast's
    /// full-kernel L1 amplification compounds per conv layer while the per-entry
    /// error tracks the receptive-field column sums. Asserts, per spec row:
    ///   (1) BOTH modes are SOUND (350-sample MC containment of the true forward),
    ///   (2) the per-entry bound is never looser than the row-max bound (small
    ///       slack-scale tolerance), and
    ///   (3) at depth the per-entry bound is DECISIVELY tighter (the fold fix).
    #[test]
    fn crown_backward_conv_err_per_entry_tighter_than_rowmax_and_sound() {
        let _g = gpu_test_serial_guard();
        // Compare the two Conv error algorithms directly. The legacy row-max
        // diagnostic is intentionally unworded, so disable receipts for this A/B.
        let _taint_words_off = ScopedEnvVar::set("NY_GPU_TAINT_WORDS", "0");
        let device = require_verdict_device();
        let mut state: u64 = 0xC0DE_C0EF;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // Constant-dim conv stack: C=3 channels, 6×6 spatial, 3×3 kernel, pad 1.
        let (c, hw, k) = (3usize, 6usize, 3usize);
        let dim = c * hw * hw; // 108, constant through the chain
        let depth = 6usize; // conv layers; ReLU between consecutive convs
        let num_specs = 4usize;

        let wcols: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..c * c * k * k).map(|_| rng() * 0.8).collect())
            .collect();
        let xc: Vec<f32> = (0..dim).map(|_| rng() * 0.5).collect();
        let xl: Vec<f32> = xc.iter().map(|&v| v - 0.1).collect();
        let xu: Vec<f32> = xc.iter().map(|&v| v + 0.1).collect();

        // True forward of one conv layer (pad 1, stride 1) in the (C,H,W) layout.
        let conv_fwd = |w: &[f32], x: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; dim];
            for co in 0..c {
                for yy in 0..hw {
                    for xx in 0..hw {
                        let mut sm = 0.0f32;
                        for ci in 0..c {
                            for a in 0..k {
                                for b in 0..k {
                                    let (sy, sx) = (yy + a, xx + b);
                                    if sy >= 1 && sy <= hw && sx >= 1 && sx <= hw {
                                        sm += w[co * (c * k * k) + ci * k * k + a * k + b]
                                            * x[ci * hw * hw + (sy - 1) * hw + (sx - 1)];
                                    }
                                }
                            }
                        }
                        out[co * hw * hw + yy * hw + xx] = sm;
                    }
                }
            }
            out
        };

        // Forward interval propagation (per-layer coefficient probing: conv is
        // linear, so unit-vector probes give exact per-layer IBP) to obtain each
        // interior ReLU's pre-activation bounds for the relaxation slopes.
        let mut cur_l = xl.clone();
        let mut cur_u = xu.clone();
        let mut relaxations: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for w in wcols.iter().take(depth - 1) {
            let (mut pl, mut pu) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            for j in 0..dim {
                let mut e = vec![0.0f32; dim];
                e[j] = 1.0;
                let col = conv_fwd(w, &e);
                for (o, &cw) in col.iter().enumerate() {
                    if cw >= 0.0 {
                        pl[o] += cw * cur_l[j];
                        pu[o] += cw * cur_u[j];
                    } else {
                        pl[o] += cw * cur_u[j];
                        pu[o] += cw * cur_l[j];
                    }
                }
            }
            let (mut ls, mut us, lint, mut uint) = (
                vec![0.0f32; dim],
                vec![0.0f32; dim],
                vec![0.0f32; dim],
                vec![0.0f32; dim],
            );
            for i in 0..dim {
                let (l, u) = (pl[i], pu[i]);
                if l >= 0.0 {
                    ls[i] = 1.0;
                    us[i] = 1.0;
                } else if u > 0.0 {
                    us[i] = u / (u - l);
                    uint[i] = -u * l / (u - l);
                }
            }
            relaxations.push((ls, us, lint, uint));
            cur_l = pl.iter().map(|&v| v.max(0.0)).collect();
            cur_u = pu.iter().map(|&v| v.max(0.0)).collect();
        }

        // Backward layer list (output→input): conv_{depth-1}, relu_{depth-2}, ...,
        // relu_0, conv_0.
        let conv_layer = |w: &Vec<f32>| GpuCrownLayer::Conv2d {
            weight_col: Arc::from(w.clone().into_boxed_slice()),
            bias_expanded: None,
            out_channels: c,
            in_channels: c,
            kernel_h: k,
            kernel_w: k,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            out_h: hw,
            out_w: hw,
            in_h: hw,
            in_w: hw,
            cert_err: Default::default(),
        };
        let mut layers: Vec<GpuCrownLayer> = Vec::new();
        for li in (0..depth).rev() {
            layers.push(conv_layer(&wcols[li]));
            if li > 0 {
                let (ls, us, lint, uint) = relaxations[li - 1].clone();
                layers.push(GpuCrownLayer::Activation {
                    lower_slope: ls,
                    upper_slope: us,
                    lower_intercept: lint,
                    upper_intercept: uint,
                    num_neurons: dim,
                });
            }
        }

        let spec: Vec<f32> = (0..num_specs * dim).map(|_| rng()).collect();

        // Per-entry (default) vs legacy row-max (env), same layers/spec/box.
        let (lo_pe, hi_pe) =
            unworded_resident_test_bounds(&device, &layers, &spec, num_specs, dim, &xl, &xu)
                .expect("per-entry conv err backward");
        let (lo_rm, hi_rm) = {
            let _guard = ScopedEnvVar::set("NY_CONV_ERR_ROWMAX", "1");
            unworded_resident_test_bounds(&device, &layers, &spec, num_specs, dim, &xl, &xu)
                .expect("row-max conv err backward")
        };

        let width = |lo: &[f32], hi: &[f32]| -> f64 {
            (0..num_specs)
                .map(|s| f64::from(hi[s]) - f64::from(lo[s]))
                .sum()
        };
        let (w_pe, w_rm) = (width(&lo_pe, &hi_pe), width(&lo_rm, &hi_rm));
        eprintln!("[conv-err] depth={depth} width_per_entry={w_pe:.4e} width_rowmax={w_rm:.4e}");

        // (1) SOUNDNESS: both modes contain the true (Conv→ReLU)*→Conv forward.
        for t in 0..350 {
            let x: Vec<f32> = (0..dim)
                .map(|i| xl[i] + (((t * 31 + i * 11) % 101) as f32 / 100.0) * (xu[i] - xl[i]))
                .collect();
            let mut h = x;
            for (li, w) in wcols.iter().enumerate() {
                h = conv_fwd(w, &h);
                if li + 1 < depth {
                    for v in h.iter_mut() {
                        *v = v.max(0.0);
                    }
                }
            }
            for s in 0..num_specs {
                let y: f32 = (0..dim).map(|j| spec[s * dim + j] * h[j]).sum();
                let tol = 1e-3 * (1.0 + y.abs());
                assert!(
                    lo_pe[s] <= y + tol && y <= hi_pe[s] + tol,
                    "UNSOUND per-entry: spec{s} y={y} not in [{}, {}]",
                    lo_pe[s],
                    hi_pe[s]
                );
                assert!(
                    lo_rm[s] <= y + tol && y <= hi_rm[s] + tol,
                    "UNSOUND row-max: spec{s} y={y} not in [{}, {}]",
                    lo_rm[s],
                    hi_rm[s]
                );
            }
        }

        // (2) NEVER LOOSER: per entry ≤ row-max per spec row (slack-scale tolerance —
        // the per-entry combine multiplies by `slack ≥ 1/(1−γ_k)`, the row-max path
        // does not, so exact ties can differ by ~γ_k relative).
        for s in 0..num_specs {
            let tol = 1e-3 * (1.0 + f64::from(hi_rm[s]) - f64::from(lo_rm[s])).abs();
            assert!(
                f64::from(lo_pe[s]) >= f64::from(lo_rm[s]) - tol,
                "per-entry LOWER looser than row-max at spec{s}: {} < {}",
                lo_pe[s],
                lo_rm[s]
            );
            assert!(
                f64::from(hi_pe[s]) <= f64::from(hi_rm[s]) + tol,
                "per-entry UPPER looser than row-max at spec{s}: {} > {}",
                hi_pe[s],
                hi_rm[s]
            );
        }

        // (3) DECISIVELY TIGHTER at depth: the row-max broadcast compounds the
        // full-kernel L1 per conv (×~‖W‖₁ each layer); per-entry tracks the
        // receptive columns. 4× total-width margin is far below the expected gap.
        assert!(
            w_pe * 4.0 <= w_rm,
            "per-entry conv error not decisively tighter at depth {depth}: \
             width_per_entry={w_pe:.4e} width_rowmax={w_rm:.4e}"
        );
    }

    /// #unsat-keystone DEEP-ReLU-resnet measurement (the cifar100/tinyimagenet error
    /// explosion + the finer per-ReLU concretization fix). Builds ONE residual block whose
    /// branch is a DEEP `(Linear→ReLU)×depth → Linear_final` chain (`out = F(x) + x`,
    /// depth=14 interior ReLUs) — so the certified f32 ERROR accumulates MONOLITHICALLY
    /// across all 14 ReLUs WITHIN the single segment (`err` propagates as `|W|·err`, L1, no
    /// cancellation ⇒ grows ~|W| per layer while the signed coefficient cancels and stays
    /// bounded — exactly the cifar100 `err_Linf → 1e19` blow-up from commit ca23d58). All
    /// ReLUs are kept strictly active (large +bias ⇒ slope 1, intercept 0) so the FORWARD
    /// stays bounded and the bound WIDTH tracks the certified-error term, isolating the fix.
    ///
    /// Measures the per-segment `err_Linf` (via NY_SEG_PROBE, visible in --nocapture) and the
    /// final certified WIDTH in THREE modes:
    ///   OFF      : no concretization (the diagnosed explosion baseline)
    ///   SEGMENT  : the existing per-SEGMENT `frontier_abs` gate (one fold at the block input)
    ///   FINE     : the new per-ReLU `node_abs` gate (folds at every interior ReLU)
    ///
    /// The per-SEGMENT gate folds only ONCE (at the block-input boundary), so it cannot cap
    /// the WITHIN-segment accumulation across the 14 interior ReLUs; FINE folds at each one.
    /// Win condition: FINE keeps `err_Linf` ≈ 0 at every segment (vs the OFF blow-up), the
    /// width stays finite and ≥2× tighter than OFF, no looser than SEGMENT, and every
    /// sampled concrete output is enclosed (soundness preserved).
    #[test]
    fn crown_backward_deep_relu_resnet_fine_concretize_caps_explosion() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let mut state: u64 = 0xDEEB_3110;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let d = 16usize;
        // ONE residual block whose BRANCH is a DEEP (Linear→ReLU)×L chain. This is the
        // decisive structure: the certified ERROR accumulates across ALL L interior ReLUs
        // WITHIN the single segment, so the existing per-SEGMENT concretization (which
        // folds only at the block-input boundary, ONCE) cannot cap the within-segment
        // L1-blow-up, while the new per-ReLU FINE concretization folds at every interior
        // ReLU. (cifar100's deep suffix has the same shape: long chains of conv+ReLU where
        // the monolithic error accumulates between the coarse segment boundaries.)
        let depth = 14usize; // ≥10 interior ReLUs in the branch (cifar100-scale depth)
                             //
                             // KEY test-design choice (to ISOLATE the diagnosed certified-error mechanism):
                             // the CROWN backward's certified f32 ERROR propagates as |W|·err (L1 norm, NO
                             // cancellation) so it grows ~|W| per layer, while the COEFFICIENT propagates as
                             // W·coeff (signed, CANCELS) and stays bounded. The cifar100 blow-up (err_Linf →
                             // 1e19, bound = sane_coeff·input − exploded_error = useless) is THIS error term.
                             // We keep every ReLU strictly ACTIVE (large positive Linear bias ⇒ pre-activation
                             // > 0 ⇒ slope 1, intercept 0 — NO relaxation looseness) and contractive weights so
                             // the FORWARD stays bounded; the |W|-accumulated error is then the dominant term,
                             // exactly the cifar100 regime, so the per-ReLU concretization win is unambiguous.
                             // Contractive (spectral ≪ 1) + large positive bias ⇒ the chain's forward stays near
                             // the bias level (~+8, strictly POSITIVE) so EVERY ReLU is provably active (slope 1,
                             // intercept 0 — no relaxation looseness). The certified error still accumulates as
                             // |W| (L1) across all 14 ReLUs — the term the per-ReLU concretization caps.
        let ws: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..d * d).map(|_| rng() * 0.18).collect())
            .collect();
        let bs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..d).map(|_| 8.0 + rng() * 0.1).collect())
            .collect();
        // Final Linear maps the branch output back to the block dim (identity-ish, small).
        let w_final: Vec<f32> = (0..d * d).map(|_| rng() * 0.05).collect();
        let b_final: Vec<f32> = (0..d).map(|_| rng() * 0.05).collect();

        // Small input box ⇒ forward range stays bounded (coeff·input small); the error,
        // which compounds via |W| regardless, is the term that explodes.
        let xc: Vec<f32> = (0..d).map(|_| rng() * 0.3).collect();
        let xl: Vec<f32> = xc.iter().map(|&c| c - 0.02).collect();
        let xu: Vec<f32> = xc.iter().map(|&c| c + 0.02).collect();

        let mm = |w: &[f32], lo: &[f32], hi: &[f32]| -> (Vec<f32>, Vec<f32>) {
            // IBP over the linear map w (d×d): per output, split by coefficient sign.
            let mut nlo = vec![0.0f32; d];
            let mut nhi = vec![0.0f32; d];
            for i in 0..d {
                let (mut l, mut h) = (0.0f32, 0.0f32);
                for j in 0..d {
                    let c = w[i * d + j];
                    if c >= 0.0 {
                        l += c * lo[j];
                        h += c * hi[j];
                    } else {
                        l += c * hi[j];
                        h += c * lo[j];
                    }
                }
                nlo[i] = l;
                nhi[i] = h;
            }
            (nlo, nhi)
        };
        let absmax = |lo: &[f32], hi: &[f32]| -> Vec<f32> {
            (0..lo.len())
                .map(|i| lo[i].abs().max(hi[i].abs()))
                .collect()
        };

        // Forward IBP through the BRANCH chain (Linear_k → ReLU)×depth → Linear_final.
        // y_0 = x (block input); y_{k+1} = ReLU(W_k·y_k + b_k); branch_out = W_final·y_depth.
        // The block output is branch_out + x (identity skip).
        let mut y_lo = vec![xl.clone()];
        let mut y_hi = vec![xu.clone()];
        let mut relu_pre_lo: Vec<Vec<f32>> = Vec::new(); // forward order, one per interior ReLU
        let mut relu_pre_hi: Vec<Vec<f32>> = Vec::new();
        for k in 0..depth {
            let yl = y_lo.last().unwrap().clone();
            let yh = y_hi.last().unwrap().clone();
            let (mut p_lo, mut p_hi) = mm(&ws[k], &yl, &yh);
            for i in 0..d {
                p_lo[i] += bs[k][i];
                p_hi[i] += bs[k][i];
            }
            relu_pre_lo.push(p_lo.clone());
            relu_pre_hi.push(p_hi.clone());
            y_lo.push(p_lo.iter().map(|&v| v.max(0.0)).collect());
            y_hi.push(p_hi.iter().map(|&v| v.max(0.0)).collect());
        }
        // branch_out bounds (post W_final), then block output = branch_out + x.
        let (bf_lo, bf_hi) = mm(&w_final, y_lo.last().unwrap(), y_hi.last().unwrap());

        let lin = |w: &[f32], b: &[f32]| GpuCrownLayer::Linear {
            weight: Arc::from(w.to_vec().into_boxed_slice()),
            bias: Some(Arc::from(b.to_vec().into_boxed_slice())),
            out_features: d,
            in_features: d,
            cert_err: Default::default(),
        };
        let mut id_w = vec![0.0f32; d * d];
        for i in 0..d {
            id_w[i * d + i] = 1.0;
        }
        let out_chain = vec![lin(&id_w, &vec![0.0f32; d])];

        // The residual branch (BACKWARD order, output→input): Linear_final, then
        // (ReLU_k, Linear_k) for k = depth-1 .. 0. ReLU slopes from forward pre-acts.
        let relu_layer = |k: usize| -> GpuCrownLayer {
            let (pl, pu) = (&relu_pre_lo[k], &relu_pre_hi[k]);
            let (mut ls, mut us, li, mut ui) = (
                vec![0.0f32; d],
                vec![0.0f32; d],
                vec![0.0f32; d],
                vec![0.0f32; d],
            );
            for i in 0..d {
                let (l, u) = (pl[i], pu[i]);
                if l >= 0.0 {
                    ls[i] = 1.0;
                    us[i] = 1.0;
                } else if u <= 0.0 {
                    // inactive: all zero (already)
                } else {
                    us[i] = u / (u - l);
                    ui[i] = -u * l / (u - l);
                }
            }
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: li,
                upper_intercept: ui,
                num_neurons: d,
            }
        };
        let mut branch: Vec<GpuCrownLayer> = vec![lin(&w_final, &b_final)];
        for k in (0..depth).rev() {
            branch.push(relu_layer(k));
            branch.push(lin(&ws[k], &bs[k]));
        }
        let branches = [branch];
        let segments: Vec<ResnetSegment> = vec![
            ResnetSegment::Chain(&out_chain),
            ResnetSegment::Residual(branches[0].as_slice()),
        ];

        // frontier_abs (per-segment input-side bound), backward-segment order:
        // [block_output (=Chain frontier), block_input x (=Residual frontier)].
        let blk_out_lo: Vec<f32> = (0..d).map(|i| bf_lo[i] + xl[i]).collect();
        let blk_out_hi: Vec<f32> = (0..d).map(|i| bf_hi[i] + xu[i]).collect();
        let frontier: Vec<Vec<f32>> = vec![absmax(&blk_out_lo, &blk_out_hi), absmax(&xl, &xu)];
        assert_eq!(frontier.len(), segments.len());
        let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();

        // node_abs: per-Activation pre-node abs-max bound in FOLD order — the order the
        // branch consumes its ReLUs (backward: ReLU_{depth-1} first .. ReLU_0 last).
        let mut node_abs: Vec<Vec<f32>> = Vec::new();
        for k in (0..depth).rev() {
            node_abs.push(absmax(&relu_pre_lo[k], &relu_pre_hi[k]));
        }
        let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();

        let mut seed = vec![0.0f32; d * d];
        for i in 0..d {
            seed[i * d + i] = 1.0;
        }
        let zb = vec![0.0f32; d];

        // Diagnostic: the forward IBP bound magnitude (drives the ReLU-intercept bias) and
        // the input-side frontier — so the report can separate the CERTIFIED-ERROR explosion
        // (what the fix caps) from the RELAXATION looseness (exploding IBP → intercepts). A
        // bounded ibp_out with the ReLUs all-active (pre-act > 0) means the width tracks the
        // certified-error term, isolating the fix's effect.
        let ibp_out_max = absmax(&blk_out_lo, &blk_out_hi)
            .iter()
            .fold(0.0f32, |m, &v| m.max(v));
        let relu_pre_min = relu_pre_lo
            .iter()
            .flatten()
            .fold(f32::INFINITY, |m, &v| m.min(v));
        let relu_pre_max = relu_pre_lo
            .iter()
            .zip(relu_pre_hi.iter())
            .flat_map(|(l, h)| absmax(l, h))
            .fold(0.0f32, |m, v| m.max(v));
        eprintln!(
            "[deep-relu-ibp] ibp_out_absmax={ibp_out_max:.4e} relu_pre_absmax={relu_pre_max:.4e} \
             relu_pre_min={relu_pre_min:.4e} (>0 ⇒ all ReLUs active, no intercept looseness)"
        );

        // Helper: run a mode + capture per-segment err_Linf via NY_SEG_PROBE-equivalent —
        // but since the probe only eprintln's, we instead directly call the seeded fold and
        // read the final width. For the per-segment err_Linf BEFORE/AFTER measurement we run
        // with NY_SEG_PROBE so the [seg] lines appear in --nocapture output.
        let width = |lo: &[f32], hi: &[f32]| -> f32 { (0..d).map(|i| hi[i] - lo[i]).sum() };

        // MODE OFF (no concretization — the diagnosed explosion baseline).
        let (lo_off, hi_off) = {
            let _g = ScopedEnvVar::set("NY_SEG_PROBE", "1");
            eprintln!("=== [deep-relu] MODE=OFF (no concretization) ===");
            let (lo, hi, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed,
                    &seed,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    false,
                )
                .expect("mode off");
            (lo, hi)
        };

        // MODE SEGMENT (existing per-segment frontier_abs gate, forced ON).
        let (lo_seg, hi_seg) = {
            let _g = ScopedEnvVar::set("NY_SEG_PROBE", "1");
            eprintln!("=== [deep-relu] MODE=SEGMENT (per-segment frontier_abs) ===");
            let (lo, hi, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed,
                    &seed,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &frontier_refs,
                    true, // force_concretize (per-segment)
                    &[],
                    false,
                )
                .expect("mode segment");
            (lo, hi)
        };

        // MODE FINE (new per-ReLU node_abs gate, forced ON) — also keeps per-segment ON for
        // the segment-boundary fold (the two compose: interior-ReLU + segment-boundary).
        let (lo_fine, hi_fine) = {
            let _g = ScopedEnvVar::set("NY_SEG_PROBE", "1");
            eprintln!("=== [deep-relu] MODE=FINE (per-ReLU node_abs) ===");
            let (lo, hi, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &segments,
                    &seed,
                    &seed,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &frontier_refs,
                    true, // per-segment fold ON
                    &node_refs,
                    true, // force_fine (per-ReLU) ON
                )
                .expect("mode fine");
            (lo, hi)
        };

        let (w_off, w_seg, w_fine) = (
            width(&lo_off, &hi_off),
            width(&lo_seg, &hi_seg),
            width(&lo_fine, &hi_fine),
        );
        eprintln!(
            "[deep-relu-result] depth={depth} width_off={w_off:.4e} width_seg={w_seg:.4e} \
             width_fine={w_fine:.4e}"
        );

        // (a) SOUNDNESS: the FINE bound must enclose every sampled concrete output.
        // Forward: block_out = W_final·((Linear_k→ReLU)×depth applied to x) + x.
        for t in 0..400 {
            let x: Vec<f32> = (0..d)
                .map(|i| xl[i] + (((t * 17 + i * 11) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                .collect();
            let mut y = x.clone();
            for k in 0..depth {
                let mut p = vec![0.0f32; d];
                for i in 0..d {
                    let mut s = bs[k][i];
                    for j in 0..d {
                        s += ws[k][i * d + j] * y[j];
                    }
                    p[i] = s.max(0.0);
                }
                y = p;
            }
            let mut z = vec![0.0f32; d];
            for i in 0..d {
                let mut s = b_final[i] + x[i]; // + identity skip
                for j in 0..d {
                    s += w_final[i * d + j] * y[j];
                }
                z[i] = s;
            }
            for o in 0..d {
                assert!(
                    lo_fine[o] <= z[o] + 5e-3 && z[o] <= hi_fine[o] + 5e-3,
                    "UNSOUND fine concretized deep resnet: out[{o}]={} not in [{}, {}]",
                    z[o],
                    lo_fine[o],
                    hi_fine[o]
                );
                // The per-segment mode must also be sound.
                assert!(
                    lo_seg[o] <= z[o] + 5e-3 && z[o] <= hi_seg[o] + 5e-3,
                    "UNSOUND segment concretized deep resnet: out[{o}]={} not in [{}, {}]",
                    z[o],
                    lo_seg[o],
                    hi_seg[o]
                );
            }
        }

        // (b) CAPPING: FINE must stay finite and be no looser than OFF. Because the deep
        // branch accumulates the certified error across ALL `depth` interior ReLUs WITHIN
        // the single residual segment, the per-SEGMENT gate (one fold at the block input)
        // cannot cap the within-segment growth, whereas FINE folds at every interior ReLU
        // — so FINE must be at least as tight as SEGMENT, and dramatically tighter than OFF.
        assert!(w_fine.is_finite(), "fine width not finite: {w_fine}");
        assert!(
            w_fine <= w_seg * (1.0 + 1e-4) + 1e-3,
            "fine should be no looser than per-segment: width_fine={w_fine} width_seg={w_seg}"
        );
        assert!(
            !w_off.is_finite() || w_fine <= w_off,
            "fine must not be looser than off: width_fine={w_fine} width_off={w_off}"
        );
        // The OFF baseline must actually exercise the explosion (else the test is vacuous):
        // FINE must be at least 2× tighter than OFF on this deep ReLU branch.
        //
        // #eft-err: under NY_EFT_ERR=1 there is NO explosion left to cap — the
        // Lipschitz activation propagation (|sel| instead of |ls|+|us|) stops the
        // per-ReLU error doubling entirely (measured: width_off 2.198 vs fine
        // 2.192 on this very branch), so the legacy-relative pin is vacuous and
        // skipped. The EFT mode's soundness is pinned by its own oracles.
        if w_off.is_finite() && !eft_err_env_enabled() {
            assert!(
                w_fine * 2.0 <= w_off,
                "FINE did not substantially cap the explosion: width_fine={w_fine} width_off={w_off}"
            );
        }
    }

    /// #unsat-keystone DEPLOYMENT proof: the AUTO path — the trait-boundary entry the
    /// production caller uses (`crown_backward_gpu_resnet_sound_inner`, behind
    /// `GpuCrownBackward::crown_backward_gpu_resnet_sound`) — now THREADS `node_abs` and, on a
    /// deep ReLU resnet whose un-concretized certified error explodes into the ±FALLBACK_BOUND
    /// clamp, AUTOMATICALLY (no env var, no force flag) detects the explosion and re-runs with the
    /// per-ReLU FINE concretization, returning the SOUND element-wise intersection — recovering a
    /// finite, non-garbage bound. Three claims, each on the exact production entry point:
    ///   1. THREADING + DETECTION + INTERSECTION + SOUNDNESS (deep "clamp" net): the AUTO bound is
    ///      finite, sound (encloses sampled outputs), and ≤ the un-concretized OFF bound elementwise
    ///      (the intersection can only tighten) — proving `node_abs` reaches the fallback and the
    ///      fallback fired.
    ///   2. FINE REACHABILITY (the cifar100-shape "moderate" net, the regime where the forward —
    ///      hence each ReLU's node_abs — stays bounded while the OFF error compounds): the fine
    ///      concretization the fallback now invokes caps OFF≈70k → ~100 (≫2× tighter), proving the
    ///      fallback's force_fine path produces the keystone win.
    ///   3. NON-EXPLODING CONTROL: a shallow net whose OFF bound is already finite returns the SAME
    ///      bound with or without `node_abs` (the threshold never fires ⇒ no fine pass ⇒ the verdict
    ///      default path is byte-for-byte unchanged — no always-on per-ReLU cost).
    #[test]
    fn crown_backward_resnet_auto_fallback_uses_fine_no_env() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();

        // Shared helpers.
        let absmax = |lo: &[f32], hi: &[f32]| -> Vec<f32> {
            (0..lo.len())
                .map(|i| lo[i].abs().max(hi[i].abs()))
                .collect()
        };

        // ============================================================================
        // Build one residual block `out = F(x) + x` whose branch F is a deep
        // (Linear_k → ReLU)×depth → Linear_final chain, with active ReLUs (large +bias).
        // Returns the owned `GpuResnetSegment`s, the seed, frontier_abs, node_abs (fold
        // order), the input box, and the per-ReLU forward pre-acts (for sampling).
        // ============================================================================
        #[allow(clippy::type_complexity)]
        let build = |seed0: u64,
                     d: usize,
                     depth: usize,
                     wscale: f32,
                     const_mag: bool,
                     bias: f32,
                     boxh: f32|
         -> (
            Vec<GpuResnetSegment>,
            GpuCrownSeed,
            Vec<f32>,      // seed_a (identity)
            Vec<Vec<f32>>, // frontier_abs
            Vec<Vec<f32>>, // node_abs (fold order)
            Vec<f32>,      // xl
            Vec<f32>,      // xu
            Vec<Vec<f32>>, // ws
            Vec<Vec<f32>>, // bs
            Vec<f32>,      // w_final
            Vec<f32>,      // b_final
        ) {
            let mut state = seed0;
            let mut rng = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            };
            let ws: Vec<Vec<f32>> = (0..depth)
                .map(|_| {
                    (0..d * d)
                        .map(|_| {
                            if const_mag {
                                if rng() >= 0.0 {
                                    wscale
                                } else {
                                    -wscale
                                }
                            } else {
                                rng() * wscale
                            }
                        })
                        .collect()
                })
                .collect();
            let bs: Vec<Vec<f32>> = (0..depth)
                .map(|_| (0..d).map(|_| bias + rng() * 0.1).collect())
                .collect();
            let w_final: Vec<f32> = (0..d * d).map(|_| rng() * 0.05).collect();
            let b_final: Vec<f32> = (0..d).map(|_| rng() * 0.05).collect();
            let xc: Vec<f32> = (0..d).map(|_| rng() * 0.3).collect();
            let xl: Vec<f32> = xc.iter().map(|&c| c - boxh).collect();
            let xu: Vec<f32> = xc.iter().map(|&c| c + boxh).collect();

            // IBP through one linear map (d×d), per output split by coefficient sign.
            let mm = |w: &[f32], lo: &[f32], hi: &[f32]| -> (Vec<f32>, Vec<f32>) {
                let mut nlo = vec![0.0f32; d];
                let mut nhi = vec![0.0f32; d];
                for i in 0..d {
                    let (mut l, mut h) = (0.0f32, 0.0f32);
                    for j in 0..d {
                        let c = w[i * d + j];
                        if c >= 0.0 {
                            l += c * lo[j];
                            h += c * hi[j];
                        } else {
                            l += c * hi[j];
                            h += c * lo[j];
                        }
                    }
                    nlo[i] = l;
                    nhi[i] = h;
                }
                (nlo, nhi)
            };
            // Forward IBP through the branch; collect per-ReLU pre-activation bounds.
            let mut y_lo = vec![xl.clone()];
            let mut y_hi = vec![xu.clone()];
            let mut relu_pre_lo: Vec<Vec<f32>> = Vec::new();
            let mut relu_pre_hi: Vec<Vec<f32>> = Vec::new();
            for k in 0..depth {
                let yl = y_lo.last().unwrap().clone();
                let yh = y_hi.last().unwrap().clone();
                let (mut p_lo, mut p_hi) = mm(&ws[k], &yl, &yh);
                for i in 0..d {
                    p_lo[i] += bs[k][i];
                    p_hi[i] += bs[k][i];
                }
                relu_pre_lo.push(p_lo.clone());
                relu_pre_hi.push(p_hi.clone());
                y_lo.push(p_lo.iter().map(|&v| v.max(0.0)).collect());
                y_hi.push(p_hi.iter().map(|&v| v.max(0.0)).collect());
            }
            let (bf_lo, bf_hi) = mm(&w_final, y_lo.last().unwrap(), y_hi.last().unwrap());

            let lin = |w: &[f32], b: &[f32]| GpuCrownLayer::Linear {
                weight: Arc::from(w.to_vec().into_boxed_slice()),
                bias: Some(Arc::from(b.to_vec().into_boxed_slice())),
                out_features: d,
                in_features: d,
                cert_err: Default::default(),
            };
            let mut id_w = vec![0.0f32; d * d];
            for i in 0..d {
                id_w[i * d + i] = 1.0;
            }
            let relu_layer = |k: usize| -> GpuCrownLayer {
                let (pl, pu) = (&relu_pre_lo[k], &relu_pre_hi[k]);
                let (mut ls, mut us, li, mut ui) = (
                    vec![0.0f32; d],
                    vec![0.0f32; d],
                    vec![0.0f32; d],
                    vec![0.0f32; d],
                );
                for i in 0..d {
                    let (l, u) = (pl[i], pu[i]);
                    if l >= 0.0 {
                        ls[i] = 1.0;
                        us[i] = 1.0;
                    } else if u <= 0.0 {
                    } else {
                        us[i] = u / (u - l);
                        ui[i] = -u * l / (u - l);
                    }
                }
                GpuCrownLayer::Activation {
                    lower_slope: ls,
                    upper_slope: us,
                    lower_intercept: li,
                    upper_intercept: ui,
                    num_neurons: d,
                }
            };
            // BACKWARD-order branch: Linear_final, then (ReLU_k, Linear_k) for k=depth-1..0.
            let mut branch: Vec<GpuCrownLayer> = vec![lin(&w_final, &b_final)];
            for k in (0..depth).rev() {
                branch.push(relu_layer(k));
                branch.push(lin(&ws[k], &bs[k]));
            }
            let out_chain = vec![lin(&id_w, &vec![0.0f32; d])];
            let segments = vec![
                GpuResnetSegment::Chain(out_chain),
                GpuResnetSegment::Residual(branch),
            ];

            // frontier_abs (per-segment): [block_output, block_input x].
            let blk_out_lo: Vec<f32> = (0..d).map(|i| bf_lo[i] + xl[i]).collect();
            let blk_out_hi: Vec<f32> = (0..d).map(|i| bf_hi[i] + xu[i]).collect();
            let frontier: Vec<Vec<f32>> = vec![absmax(&blk_out_lo, &blk_out_hi), absmax(&xl, &xu)];
            // node_abs (per-ReLU, FOLD order = backward: ReLU_{depth-1}..ReLU_0).
            let mut node_abs: Vec<Vec<f32>> = Vec::new();
            for k in (0..depth).rev() {
                node_abs.push(absmax(&relu_pre_lo[k], &relu_pre_hi[k]));
            }

            let mut seed_a = vec![0.0f32; d * d];
            for i in 0..d {
                seed_a[i * d + i] = 1.0;
            }
            let seed = GpuCrownSeed {
                lower_a: seed_a.clone().into(),
                upper_a: seed_a.clone().into(),
                lower_b: vec![0.0f32; d].into(),
                upper_b: vec![0.0f32; d].into(),
                num_specs: d,
                current_dim: d,
            };
            (
                segments, seed, seed_a, frontier, node_abs, xl, xu, ws, bs, w_final, b_final,
            )
        };

        // Sample the concrete branch forward + identity skip, assert `lo ≤ z ≤ hi`.
        let assert_sound = |d: usize,
                            depth: usize,
                            ws: &[Vec<f32>],
                            bs: &[Vec<f32>],
                            w_final: &[f32],
                            b_final: &[f32],
                            xl: &[f32],
                            xu: &[f32],
                            lo: &[f32],
                            hi: &[f32],
                            tag: &str| {
            for t in 0..300 {
                let x: Vec<f32> = (0..d)
                    .map(|i| xl[i] + (((t * 17 + i * 11) % 100) as f32 / 99.0) * (xu[i] - xl[i]))
                    .collect();
                let mut y = x.clone();
                for k in 0..depth {
                    let mut p = vec![0.0f32; d];
                    for i in 0..d {
                        let mut s = bs[k][i];
                        for j in 0..d {
                            s += ws[k][i * d + j] * y[j];
                        }
                        p[i] = s.max(0.0);
                    }
                    y = p;
                }
                let mut z = vec![0.0f32; d];
                for i in 0..d {
                    let mut s = b_final[i] + x[i];
                    for j in 0..d {
                        s += w_final[i * d + j] * y[j];
                    }
                    z[i] = s;
                }
                for o in 0..d {
                    if !z[o].is_finite() {
                        continue; // forward itself overflowed for this sample; skip.
                    }
                    let tol = 5e-3 * (1.0 + z[o].abs());
                    assert!(
                        f64::from(lo[o]) <= f64::from(z[o]) + f64::from(tol)
                            && f64::from(z[o]) <= f64::from(hi[o]) + f64::from(tol),
                        "UNSOUND {tag}: out[{o}]={} not in [{}, {}]",
                        z[o],
                        lo[o],
                        hi[o]
                    );
                }
            }
        };
        let width = |lo: &[f32], hi: &[f32]| -> f32 {
            lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum()
        };

        // ----------------------------------------------------------------------------
        // CLAIM 1 — deep OVERSIZE net (constant-magnitude ±wscale): its exact
        // outward affine radius itself exceeds FALLBACK_BOUND.  Since the
        // sentinel is finite rather than mathematical infinity, neither the
        // cheap nor AUTO path may publish a clamped interval; both must refuse
        // before dispatch.  The moderate fixture below separately exercises
        // the valid fine-concretization recovery regime.
        // ----------------------------------------------------------------------------
        {
            let d = 16usize;
            let depth = 52usize;
            let (segments, seed, _seed_a, frontier, node_abs, xl, xu, ..) =
                build(0xC1A_3110, d, depth, 0.22, true, 0.0, 0.05);
            let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();
            let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();

            let off_error = device
                .crown_backward_gpu_resnet_sound_inner(&segments, &seed, &xl, &xu, &[], &[])
                .expect_err("oversize OFF path must refuse the finite fallback sentinel");
            let auto_error = device
                .crown_backward_gpu_resnet_sound_inner(
                    &segments,
                    &seed,
                    &xl,
                    &xu,
                    &frontier_refs,
                    &node_refs,
                )
                .expect_err("oversize AUTO path must refuse the finite fallback sentinel");
            for (route, error) in [("OFF", off_error), ("AUTO", auto_error)] {
                let message = error.to_string();
                assert!(
                    matches!(error, NyError::InvalidSpec(_))
                        && message.contains("outward affine radius")
                        && message.contains("FALLBACK_BOUND"),
                    "oversize {route} path returned the wrong refusal: {message}"
                );
            }
        }

        // ----------------------------------------------------------------------------
        // CLAIM 2 — cifar100-shape MODERATE net (contractive forward ⇒ each ReLU's node_abs
        // stays bounded ~O(10) while the OFF certified error compounds to ~7e4 width). The
        // FINE concretization the fallback now invokes (force_fine) must cap it ≫2× tighter —
        // proving the fallback's fine path is the keystone fix. Driven through the same seeded
        // fold the inner's fallback calls, with force_fine=true (what the AUTO inner sets when
        // node_abs is non-empty).
        // ----------------------------------------------------------------------------
        {
            let d = 16usize;
            let depth = 80usize;
            let (segments, _seed, seed_a, frontier, node_abs, xl, xu, ws, bs, w_final, b_final) =
                build(0xDEEB_3110, d, depth, 0.18, false, 8.0, 0.02);
            let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();
            let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
            let zb = vec![0.0f32; d];
            let internal: Vec<ResnetSegment> = segments
                .iter()
                .map(|s| match s {
                    GpuResnetSegment::Chain(l) => ResnetSegment::Chain(l.as_slice()),
                    GpuResnetSegment::Residual(l) => ResnetSegment::Residual(l.as_slice()),
                    GpuResnetSegment::ResidualProj(f, p) => {
                        ResnetSegment::ResidualProj(f.as_slice(), p.as_slice())
                    }
                })
                .collect();
            // OFF (no concretization).
            let (lo_off, hi_off, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &internal,
                    &seed_a,
                    &seed_a,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    false,
                )
                .expect("moderate OFF");
            // FINE — exactly what the AUTO inner invokes in the fallback when node_abs is set
            // (force_concretize=true, node_abs threaded, force_fine=true).
            let (lo_fine, hi_fine, _) = device
                .crown_backward_sound_resident_resnet_seeded(
                    &internal,
                    &seed_a,
                    &seed_a,
                    &zb,
                    &zb,
                    d,
                    d,
                    &xl,
                    &xu,
                    &[],
                    &[],
                    &frontier_refs,
                    true,
                    &node_refs,
                    true,
                )
                .expect("moderate FINE");
            let w_off = width(&lo_off, &hi_off);
            let w_fine = width(&lo_fine, &hi_fine);
            eprintln!("[auto/moderate] width_off={w_off:.3e} width_fine={w_fine:.3e}");
            assert!(w_fine.is_finite(), "FINE width not finite: {w_fine}");
            assert!(
                w_off.is_finite() && w_off > 1e3,
                "moderate OFF not in the useless-wide regime (test vacuous): width_off={w_off}"
            );
            assert!(
                w_fine * 2.0 <= w_off,
                "FINE did not substantially cap the OFF explosion: width_fine={w_fine} width_off={w_off}"
            );
            assert_sound(
                d,
                depth,
                &ws,
                &bs,
                &w_final,
                &b_final,
                &xl,
                &xu,
                &lo_fine,
                &hi_fine,
                "FINE moderate",
            );
        }

        // ----------------------------------------------------------------------------
        // CLAIM 3 — NON-EXPLODING CONTROL (#w4-conv-err-per-entry policy): with abs
        // bounds provided the concretized pass + element-wise tighter merge now ALWAYS
        // runs, so the result must be element-wise AT LEAST AS TIGHT as the plain pass
        // (never looser — the merge is a sound intersection). Under
        // NY_RESNET_ERR_MERGE=0 (legacy explosion-only trigger) a healthy net must
        // return the plain bound BYTE-IDENTICAL — the old default-path invariant.
        // ----------------------------------------------------------------------------
        {
            let d = 16usize;
            let depth = 1usize;
            let (segments, seed, _seed_a, frontier, node_abs, xl, xu, _ws, _bs, _wf, _bf) =
                build(0x5A1E_0001, d, depth, 0.18, false, 8.0, 0.02);
            let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();
            let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
            let (lo_plain, hi_plain) = device
                .crown_backward_gpu_resnet_sound_inner(&segments, &seed, &xl, &xu, &[], &[])
                .expect("control plain");
            // Only assert if the plain bound is genuinely healthy (well under the clamp).
            if lo_plain
                .iter()
                .chain(hi_plain.iter())
                .all(|v| v.is_finite() && v.abs() < crate::FALLBACK_BOUND)
            {
                let (lo_na, hi_na) = device
                    .crown_backward_gpu_resnet_sound_inner(
                        &segments,
                        &seed,
                        &xl,
                        &xu,
                        &frontier_refs,
                        &node_refs,
                    )
                    .expect("control with node_abs");
                for o in 0..d {
                    assert!(
                        lo_na[o] >= lo_plain[o] && hi_na[o] <= hi_plain[o],
                        "merge-always made a non-exploding control LOOSER at {o}: \
                         plain=[{}, {}] node_abs=[{}, {}]",
                        lo_plain[o],
                        hi_plain[o],
                        lo_na[o],
                        hi_na[o]
                    );
                }
                // Legacy trigger (NY_RESNET_ERR_MERGE=0): healthy ⇒ no second pass ⇒
                // byte-identical to plain.
                let (lo_legacy, hi_legacy) = {
                    let _guard = ScopedEnvVar::set("NY_RESNET_ERR_MERGE", "0");
                    device
                        .crown_backward_gpu_resnet_sound_inner(
                            &segments,
                            &seed,
                            &xl,
                            &xu,
                            &frontier_refs,
                            &node_refs,
                        )
                        .expect("control legacy trigger")
                };
                for o in 0..d {
                    assert!(
                        lo_legacy[o] == lo_plain[o] && hi_legacy[o] == hi_plain[o],
                        "legacy trigger (NY_RESNET_ERR_MERGE=0) changed a healthy bound at {o}: \
                         plain=[{}, {}] legacy=[{}, {}]",
                        lo_plain[o],
                        hi_plain[o],
                        lo_legacy[o],
                        hi_legacy[o]
                    );
                }
            }
        }

        // ----------------------------------------------------------------------------
        // CLAIM 4 — GRAD variant (#w4-gpu-dag-backward): the alpha-warmup entry
        // must enforce the SAME finite-sentinel preflight as the plain entry.  The
        // deep oversize fixture's exact outward radius exceeds FALLBACK_BOUND, so
        // neither OFF nor AUTO may publish a clamped bound or gradients.
        // ----------------------------------------------------------------------------
        {
            let d = 16usize;
            let depth = 52usize;
            let (segments, seed, _seed_a, frontier, node_abs, xl, xu, ..) =
                build(0xC1A_3110, d, depth, 0.22, true, 0.0, 0.05);
            let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();
            let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
            // Masked pre-activation lower stand-ins for gradient capture (values only
            // scale the steering gradients; no soundness role).
            let pre_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();

            let off_error = device
                .crown_backward_gpu_resnet_sound_grad_inner(
                    &segments,
                    &seed,
                    &xl,
                    &xu,
                    &pre_refs,
                    &[],
                    &[],
                )
                .expect_err("oversize grad OFF path must refuse the finite fallback sentinel");
            let auto_error = device
                .crown_backward_gpu_resnet_sound_grad_inner(
                    &segments,
                    &seed,
                    &xl,
                    &xu,
                    &pre_refs,
                    &frontier_refs,
                    &node_refs,
                )
                .expect_err("oversize grad AUTO path must refuse the finite fallback sentinel");
            for (route, error) in [("OFF", off_error), ("AUTO", auto_error)] {
                let message = error.to_string();
                assert!(
                    matches!(error, NyError::InvalidSpec(_))
                        && message.contains("outward affine radius")
                        && message.contains("FALLBACK_BOUND"),
                    "oversize grad {route} path returned the wrong refusal: {message}"
                );
            }
        }

        // ----------------------------------------------------------------------------
        // CLAIM 5 — BETA variant (#w4-gpu-dag-backward): the BaB per-domain entry
        // must likewise refuse the oversize fixture for both OFF and AUTO.  A zero
        // dual does not make a finite FALLBACK_BOUND sentinel infinite.
        // ----------------------------------------------------------------------------
        {
            let d = 16usize;
            let depth = 52usize;
            let (segments, seed, _seed_a, frontier, node_abs, xl, xu, ..) =
                build(0xC1A_3110, d, depth, 0.22, true, 0.0, 0.05);
            let frontier_refs: Vec<&[f32]> = frontier.iter().map(|v| v.as_slice()).collect();
            let node_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
            let zeros: Vec<Vec<f32>> = (0..depth).map(|_| vec![0.0f32; d]).collect();
            let beta_refs: Vec<&[f32]> = zeros.iter().map(|v| v.as_slice()).collect();

            let off_error = device
                .crown_backward_gpu_resnet_sound_beta_inner(
                    &segments,
                    &seed,
                    &xl,
                    &xu,
                    &beta_refs,
                    &[],
                    &[],
                )
                .expect_err("oversize beta OFF path must refuse the finite fallback sentinel");
            let auto_error = device
                .crown_backward_gpu_resnet_sound_beta_inner(
                    &segments,
                    &seed,
                    &xl,
                    &xu,
                    &beta_refs,
                    &frontier_refs,
                    &node_refs,
                )
                .expect_err("oversize beta AUTO path must refuse the finite fallback sentinel");
            for (route, error) in [("OFF", off_error), ("AUTO", auto_error)] {
                let message = error.to_string();
                assert!(
                    matches!(error, NyError::InvalidSpec(_))
                        && message.contains("outward affine radius")
                        && message.contains("FALLBACK_BOUND"),
                    "oversize beta {route} path returned the wrong refusal: {message}"
                );
            }
        }
    }

    /// REPRO (SOUNDNESS, two GPU bugs): the resident AW-error combine and the conv
    /// L1 multiplier under-counted the certified coefficient error on WIDE layers.
    ///
    /// (B) `CROWN_AW_ERROR_COMBINE_SHADER` reads `s_prod = fl(|A|@|W|)` and
    /// `prop = fl(err@|W|)` — both f32-accumulated over the length-`of` contraction,
    /// so each can UNDER-report its EXACT value by up to a factor `γ_of`
    /// (catastrophic when a large partial sum ABSORBS the later small terms) — then
    /// scaled by a FIXED `SLACK = 1.000001`. For any `of ≥ 64`,
    /// `1/(1−γ_of) > 1.000001`, so the fixed slack could NOT recover an outward
    /// bound: `err_out` then UNDER-counts the true coefficient error ⇒ a concretized
    /// bound tighter than the true reachable value = FALSE PROOF.
    ///
    /// The fix scales by a host `slack = combine_slack_f32(of) ≥ 1/(1−γ_of)` (with
    /// combine-ULP headroom) and rounds the result UP. This test reproduces the EXACT
    /// element-wise combine the shader runs (`(γ_k·s_prod + prop)·slack [+round_up]`)
    /// on a DETERMINISTIC worst-case under-reported f32 product (a `[2²⁴, 1, 1, …]`
    /// dot whose f32 sum absorbs all trailing ones), comparing the certified result
    /// to the EXACT f64 propagated error with ZERO tolerance. It FAILS with the old
    /// fixed `1.000001` (+no round_up) and PASSES with `combine_slack_f32` (+round_up)
    /// — independent of any GPU reduction order. A second leg runs the real GPU path
    /// for a sound-and-not-loose regression.
    #[test]
    fn crown_backward_sound_resident_aw_combine_slack_covers_f32_gemm_undercount() {
        // ---- LEG 1: deterministic worst-case combine math (toggles with the fix) ----
        // Mirror the shader (CROWN_AW_ERROR_COMBINE_SHADER): the per-element error is
        //   err_out = round_up_pos((γ_k·s_prod + prop)·slack + additive)
        // with s_prod = fl(|A|@|W|), prop = fl(err@|W|) the f32 GEMM products.
        let round_up_pos = |x: f32| -> f32 {
            if x <= 0.0 {
                0.0
            } else {
                f32::from_bits(x.to_bits() + 1)
            }
        };
        for &k in &[64usize, 256, 512, 2048] {
            // Worst-case f32 dot `err@|W|`: a big leading term that absorbs the rest.
            // err = [2²⁴, 1, 1, …, 1] (k entries), |W| = 1 ⇒ products = err.
            // f32 sequential sum = 2²⁴ (every trailing +1 rounds away); exact = 2²⁴+(k−1).
            let mut err = vec![1.0f32; k];
            err[0] = (1u32 << 24) as f32;
            let prop_f32: f32 = err.iter().fold(0.0f32, |a, &v| a + v); // under-reports
            let prop_exact: f64 = err.iter().map(|&v| f64::from(v)).sum();
            assert!(
                f64::from(prop_f32) < prop_exact,
                "k={k}: setup failed to under-report (prop_f32={prop_f32} exact={prop_exact})"
            );
            // Tiny coefficient ⇒ γ_k·s_prod negligible; isolates the prop under-report.
            let s_prod = 5e-4f32;
            let g = gamma_k_f32(k).expect("test reduction has finite Higham gamma");
            let additive = crate::wgpu_device::sound_consts::rung3_flush_safe_additive(
                u32::try_from(k).unwrap_or(u32::MAX),
            )
            .expect("test reduction has a representable rung-3 point count"); // FTZ+rung3-fma-safe

            // OLD (buggy) combine: fixed slack 1.000001, NO round_up — UNSOUND here.
            let old_cert = (g * s_prod + prop_f32) * 1.000001f32 + additive;
            assert!(
                f64::from(old_cert) < prop_exact,
                "k={k}: the OLD fixed-slack combine was expected to UNDER-count here \
                 (old_cert={old_cert} should be < exact {prop_exact}) — repro no longer valid"
            );

            // NEW (fixed) combine: k-scaled slack + round_up_pos — must be OUTWARD.
            let slack = combine_slack_f32(k).expect("test reduction has finite recovery slack");
            let new_cert = round_up_pos((g * s_prod + prop_f32) * slack + additive);
            assert!(
                f64::from(new_cert) >= prop_exact,
                "UNSOUND AW-combine (k={k}): certified {new_cert} < exact propagated error \
                 {prop_exact} (prop_f32={prop_f32}, under-report={}, slack={slack})",
                prop_exact - f64::from(prop_f32)
            );
        }

        // ---- LEG 2: real GPU resident path stays sound (and not absurdly loose) ----
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let num_specs = 3usize;
        let of = 512usize;
        let if_ = 4usize;
        let mut state: u64 = 0x5A0C_BEEF;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        // |W| = 1 so `prop = err@|W|` is a pure reduction over `err`; a large leading
        // err entry then absorbs the trailing ones in the f32 GEMM (regardless of the
        // exact reduction order, the magnitude gap forces low-bit loss), driving the
        // on-device `prop` below its exact value — the worst case the slack must cover.
        let weight: Vec<f32> = vec![1.0f32; of * if_];
        let seed_a: Vec<f32> = (0..num_specs * of).map(|_| 1e-6 + rng() * 1e-6).collect();
        let mut in_err: Vec<f32> = vec![1.0f32; num_specs * of];
        for s in 0..num_specs {
            in_err[s * of] = (1u32 << 23) as f32; // big leading term per row
        }
        let zb = vec![0.0f32; num_specs];
        let layers = vec![GpuCrownLayer::Linear {
            weight: Arc::from(weight.clone().into_boxed_slice()),
            bias: None,
            out_features: of,
            in_features: if_,
            cert_err: Default::default(),
        }];
        let c = device
            .crown_backward_sound_resident_coeff_seeded_err(
                &layers,
                &seed_a,
                &seed_a,
                &in_err,
                &in_err,
                &zb,
                &zb,
                &zb,
                &zb,
                num_specs,
                of,
                &[],
                &[],
            )
            .expect("resident coeff seeded err");
        let mut worst_ratio = 0.0f64;
        for s in 0..num_specs {
            for j in 0..if_ {
                let mut coeff_exact = 0.0f64;
                let mut prop_exact = 0.0f64;
                for l in 0..of {
                    let w = f64::from(weight[l * if_ + j]);
                    coeff_exact += f64::from(seed_a[s * of + l]) * w;
                    prop_exact += f64::from(in_err[s * of + l]) * w.abs();
                }
                let idx = s * if_ + j;
                let stored = f64::from(c.lower_a[idx]);
                let true_err = (stored - coeff_exact).abs() + prop_exact;
                let cert = f64::from(c.lower_err[idx]);
                assert!(
                    cert >= true_err,
                    "UNSOUND GPU AW-combine (of={of}) [{s},{j}]: certified {cert:.6e} < true \
                     {true_err:.6e}"
                );
                if true_err > 0.0 {
                    worst_ratio = worst_ratio.max(cert / true_err);
                }
            }
        }
        assert!(
            worst_ratio < 100.0,
            "AW-combine certificate is implausibly loose: {worst_ratio}x"
        );
    }

    /// REPRO (SOUNDNESS): the resident Conv2d error multiplier `kernel_l1` was
    /// f32-SUMMED (`weight_col.iter().map(|v| v.abs()).sum::<f32>()`), which ROUNDS
    /// DOWN on a wide kernel and UNDER-reports ‖W‖₁ → the certified conv-coeff error
    /// (`γ·rowmax|a|·kl1 + rowmax|err|·kl1`) under-counts ⇒ a tighter-than-true
    /// bound = FALSE PROOF. The fix accumulates ‖W‖₁ in f64 and rounds the f32 cast
    /// UP (`up_f32(Σ|f64::from(v)|)`).
    ///
    /// This unit check builds a wide same-sign kernel and asserts (with ZERO
    /// tolerance) that the certified multiplier `up_f32(f64-L1)` is a valid OUTWARD
    /// bound on the exact f64 ‖W‖₁, while the OLD f32-summed value is NOT — i.e. it
    /// strictly under-reports. Mirrors the proven conv fix (becc501).
    #[test]
    fn crown_backward_sound_resident_conv_kernel_l1_is_outward_bound() {
        // Wide, same-sign, near-1 kernel: the f32 accumulator drops low bits and
        // sums to STRICTLY LESS than the exact f64 L1.
        let n = 8192usize;
        let weight_col: Vec<f32> = (0..n).map(|i| 1.0f32 + (i as f32) * 1e-7).collect();

        let exact_l1: f64 = weight_col.iter().map(|v| f64::from(*v).abs()).sum();
        let old_f32_sum: f32 = weight_col.iter().map(|v| v.abs()).sum();
        // The NEW certified multiplier (matches the production code's `kl1`).
        let new_kl1: f32 = up_f32(weight_col.iter().map(|v| f64::from(*v).abs()).sum());

        // The bug: the old f32 sum strictly UNDER-reports the true L1.
        assert!(
            f64::from(old_f32_sum) < exact_l1,
            "test setup did not trigger f32 L1 under-report: f32_sum={old_f32_sum} >= exact={exact_l1}"
        );
        // The fix: the new multiplier is a sound OUTWARD (>=) bound, ZERO tolerance.
        assert!(
            f64::from(new_kl1) >= exact_l1,
            "UNSOUND conv kernel_l1: certified {new_kl1} < exact ‖W‖₁ {exact_l1}"
        );
        // And it would have FAILED with the old f32-summed multiplier.
        assert!(
            f64::from(old_f32_sum) < exact_l1 && f64::from(new_kl1) >= exact_l1,
            "repro must distinguish old (under) from new (outward)"
        );
    }

    /// THE UN-GATE SOUNDNESS GATE for the per-node IBP CROWN-partial backward
    /// (#vnncomp-gpu-crown-soundness, un-gating site #5).
    ///
    /// The per-node IBP partial path now dispatches the verdict-relevant
    /// INTERMEDIATE CROWN bound to `GpuCrownBackward::crown_backward_gpu_sound`
    /// (the exact trait method called from `try_gpu_crown_partial_backward` when
    /// `use_sound` is set) instead of the proven-sound CPU loop. This test proves
    /// that method's bound is a SOUND ENCLOSURE of BOTH:
    ///   (a) the proven-sound CPU host backward (`crown_backward_sound_host`,
    ///       which composes A·W in f64 and adds the certified γ_n·S term), and
    ///   (b) a Monte-Carlo sample of TRUE network outputs.
    ///
    /// over random Linear+ReLU nets of VARIED depth/width with ADVERSARIAL
    /// coefficient signs + heavy cancellation (weights centered on 0 spanning ±,
    /// so the signed A·W composition cancels while the certified |A|·|W| error term
    /// does not — the exact regime where an f32 round-to-nearest GEMM without γ_n·S
    /// widening would under-report and produce a false proof).
    ///
    /// ZERO violations over every spec of every case is the gate. We assert in the
    /// soundness direction with NO favorable slack: `gpu_lower <= cpu_lower` and
    /// `gpu_upper >= cpu_upper` (a tiny outward epsilon only, never inward), and
    /// `gpu_lower <= y <= gpu_upper` for the true outputs.
    #[test]
    fn crown_backward_gpu_sound_encloses_cpu_sound_and_samples_adversarial() {
        use ny_core::GpuCrownBackward;
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // The trait object the production un-gated path actually calls.
        let gpu: &dyn GpuCrownBackward = &*device;

        let mut state: u64 = 0xADBE_5160_F00D_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        // Varied depth (#hidden layers) × width, exercising 1..=4 Linear+ReLU
        // stages. Each shape is run several times with fresh adversarial weights.
        let shapes: &[(usize, &[usize], usize)] = &[
            (4, &[6], 3),            // 1 hidden
            (5, &[8, 6], 4),         // 2 hidden
            (6, &[10, 8, 7], 5),     // 3 hidden
            (8, &[12, 10, 9, 6], 4), // 4 hidden
            (3, &[16, 16], 3),       // wide, heavy cancellation
        ];

        let mut total_specs = 0usize;
        let mut total_samples = 0usize;
        for &(din, hidden, dout) in shapes {
            for _trial in 0..8 {
                // Build dims: din -> hidden[0] -> ... -> dout.
                let mut dims = vec![din];
                dims.extend_from_slice(hidden);
                dims.push(dout);

                // Adversarial weights/biases: centered on 0, spanning ±, scaled so
                // forward activations land near 0 (maximally-unstable ReLUs and the
                // most signed cancellation in the A·W composition).
                let mut weights: Vec<Vec<f32>> = Vec::new();
                let mut biases: Vec<Vec<f32>> = Vec::new();
                for w in 0..dims.len() - 1 {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    // Row-major (no × ni). Symmetric ± with a few large-magnitude
                    // pairs that nearly cancel.
                    let wt: Vec<f32> = (0..no * ni).map(|_| rng() * 1.3).collect();
                    let bs: Vec<f32> = (0..no).map(|_| rng() * 0.15).collect();
                    weights.push(wt);
                    biases.push(bs);
                }

                // Input box centered near 0 (drives the post-Linear pre-activations
                // toward the unstable regime).
                let xc: Vec<f32> = (0..din).map(|_| rng() * 0.4).collect();
                let xl: Vec<f32> = xc.iter().map(|&c| c - 0.3).collect();
                let xu: Vec<f32> = xc.iter().map(|&c| c + 0.3).collect();

                // Forward IBP to get per-stage pre-activation bounds (for ReLU
                // relaxation slopes). Conv-free, so this is interval matmul + bias.
                let relu = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| x.max(0.0)).collect() };
                let mut cur_l = xl.clone();
                let mut cur_u = xu.clone();
                // pre_l/pre_u[stage] are the pre-activation bounds feeding ReLU stage.
                let mut pre_l: Vec<Vec<f32>> = Vec::new();
                let mut pre_u: Vec<Vec<f32>> = Vec::new();
                for w in 0..dims.len() - 1 {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    let wt = &weights[w];
                    let bs = &biases[w];
                    let mut nl = vec![0.0f32; no];
                    let mut nu = vec![0.0f32; no];
                    for o in 0..no {
                        let mut lo = bs[o];
                        let mut hi = bs[o];
                        for j in 0..ni {
                            let coeff = wt[o * ni + j];
                            if coeff >= 0.0 {
                                lo += coeff * cur_l[j];
                                hi += coeff * cur_u[j];
                            } else {
                                lo += coeff * cur_u[j];
                                hi += coeff * cur_l[j];
                            }
                        }
                        nl[o] = lo;
                        nu[o] = hi;
                    }
                    // ReLU applied after every Linear EXCEPT the final one.
                    if w < dims.len() - 2 {
                        pre_l.push(nl.clone());
                        pre_u.push(nu.clone());
                        cur_l = relu(&nl);
                        cur_u = relu(&nu);
                    }
                }

                // Build backward-order layers (output -> input):
                // [Linear_last, ReLU_{k-1}, Linear_{k-1}, ..., ReLU_0, Linear_0].
                let mut layers: Vec<GpuCrownLayer> = Vec::new();
                let n_lin = dims.len() - 1;
                for w in (0..n_lin).rev() {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    layers.push(GpuCrownLayer::Linear {
                        weight: Arc::from(weights[w].clone().into_boxed_slice()),
                        bias: Some(Arc::from(biases[w].clone().into_boxed_slice())),
                        out_features: no,
                        in_features: ni,
                        cert_err: Default::default(),
                    });
                    // The ReLU BEFORE this Linear (stage index w-1 in pre_l/pre_u).
                    if w > 0 {
                        let stage = w - 1;
                        let l = &pre_l[stage];
                        let u = &pre_u[stage];
                        let nn = l.len();
                        let mut ls = vec![0.0f32; nn];
                        let mut us = vec![0.0f32; nn];
                        let li = vec![0.0f32; nn];
                        let mut ui = vec![0.0f32; nn];
                        for i in 0..nn {
                            let (lo, hi) = (l[i], u[i]);
                            if lo >= 0.0 {
                                // Stable active: identity.
                                ls[i] = 1.0;
                                us[i] = 1.0;
                            } else if hi <= 0.0 {
                                // Stable inactive: zero (slopes/intercepts all 0).
                            } else {
                                // Unstable: lower slope (adversarial alpha in [0,1])
                                // and the standard chord upper relaxation. Any alpha is
                                // a sound lower relaxation; pick a non-trivial one to
                                // stress the sign routing.
                                let alpha = 0.5 + 0.49 * rng(); // in (0.005, 0.995)
                                ls[i] = alpha.clamp(0.0, 1.0);
                                us[i] = hi / (hi - lo);
                                ui[i] = -hi * lo / (hi - lo);
                            }
                        }
                        layers.push(GpuCrownLayer::Activation {
                            lower_slope: ls,
                            upper_slope: us,
                            lower_intercept: li,
                            upper_intercept: ui,
                            num_neurons: nn,
                        });
                    }
                }

                // Identity spec (one row per output neuron) — exactly what both the
                // sequential and IBP-partial GPU paths build.
                let mut spec = vec![0.0f32; dout * dout];
                for i in 0..dout {
                    spec[i * dout + i] = 1.0;
                }

                // (a) The production trait method called by the un-gated partial path.
                let sound = gpu
                    .crown_backward_gpu_sound(&layers, &spec, dout, &xl, &xu)
                    .expect("sound GPU CROWN backward");
                // (b) The proven-sound CPU host backward (the soundness reference).
                let (hlo, hhi) = device
                    .crown_backward_sound_host(&layers, &spec, dout, dout, &xl, &xu)
                    .expect("host sound backward");

                // ENCLOSURE vs the CPU sound bound — 0 violations, soundness
                // direction only (gpu must be at least as WIDE). The small epsilon is
                // OUTWARD-only headroom for the f32/f64 dtype gap; we never permit the
                // GPU bound to sit INSIDE the CPU bound by more than this.
                //
                // #eft-err: under NY_EFT_ERR=1 this direction is INTENTIONALLY
                // violated — the compensated channel legitimately lands the GPU
                // bound INSIDE the CPU's a-priori-charged bound (that is the whole
                // point). The EFT mode's soundness is pinned by its OWN oracle
                // (`eft_err_channel_ab_tightens_and_stays_sound`: exact-f64
                // reference + true-sample enclosure); the direction asserts here
                // pin the HIGHAM-channel contract, so they are skipped when the
                // gate is on. The true-sample enclosure below always runs.
                let higham_direction = !eft_err_env_enabled();
                const ENC_EPS: f32 = 2e-4;
                for k in 0..dout {
                    let (glo, ghi) = (sound.lower_bounds[k], sound.upper_bounds[k]);
                    assert!(
                        glo.is_finite() && ghi.is_finite() && glo <= ghi,
                        "non-finite/inverted GPU sound bound [{glo}, {ghi}] at spec {k}"
                    );
                    assert!(
                        !higham_direction || glo <= hlo[k] + ENC_EPS,
                        "ENCLOSURE VIOLATION (lower): gpu_lower {glo} > cpu_sound_lower {} \
                         at spec {k} (dims {dims:?}) — GPU bound is INSIDE the proven CPU bound",
                        hlo[k]
                    );
                    assert!(
                        !higham_direction || ghi >= hhi[k] - ENC_EPS,
                        "ENCLOSURE VIOLATION (upper): gpu_upper {ghi} < cpu_sound_upper {} \
                         at spec {k} (dims {dims:?}) — GPU bound is INSIDE the proven CPU bound",
                        hhi[k]
                    );
                    total_specs += 1;
                }

                // Monte-Carlo enclosure of TRUE outputs — a violation here is a real
                // false proof, so ZERO favorable slack on the bound (only f32-forward
                // noise headroom). Many deterministic + pseudo-random samples.
                for t in 0..400 {
                    let x: Vec<f32> = (0..din)
                        .map(|i| {
                            let frac = (((t * 37 + i * 13) % 101) as f32) / 100.0;
                            xl[i] + frac * (xu[i] - xl[i])
                        })
                        .collect();
                    // True forward: (Linear -> ReLU) repeated, final Linear no ReLU.
                    let mut v = x.clone();
                    for w in 0..n_lin {
                        let (ni, no) = (dims[w], dims[w + 1]);
                        let mut nv = matmul(&weights[w], &v, no, ni);
                        for o in 0..no {
                            nv[o] += biases[w][o];
                        }
                        if w < n_lin - 1 {
                            for o in 0..no {
                                nv[o] = nv[o].max(0.0);
                            }
                        }
                        v = nv;
                    }
                    for o in 0..dout {
                        assert!(
                            sound.lower_bounds[o] <= v[o] + 3e-3
                                && v[o] <= sound.upper_bounds[o] + 3e-3,
                            "UNSOUND: true output[{o}]={} not in GPU sound bound [{}, {}] \
                             (dims {dims:?}, sample {t})",
                            v[o],
                            sound.lower_bounds[o],
                            sound.upper_bounds[o]
                        );
                        total_samples += 1;
                    }
                }
            }
        }
        assert!(
            total_specs >= 100 && total_samples >= 20_000,
            "coverage too thin: {total_specs} specs, {total_samples} samples"
        );
    }

    /// #eft-err DIFFERENTIAL ORACLE (increment 2/3 validation): run the SAME
    /// adversarial folds with the EFT channel OFF and ON and assert, per spec:
    ///   1. the ON bounds are at least as TIGHT (the min-combine can only
    ///      shrink the certified error; RN concretization is monotone in it);
    ///   2. the ON bounds still ENCLOSE every true sampled output (0 favorable
    ///      slack — a violation is a false proof);
    ///   3. the ON bounds still enclose the proven CPU-sound host bound (the
    ///      EFT-measured f32 error ~1e-5 stays above the host's f64-class
    ///      channel, so the historical oracle direction is preserved);
    ///   4. the channel actually FIRES: on these cancellation-heavy folds the
    ///      Higham charge is orders above the actual error, so a measurable
    ///      fraction of specs must tighten strictly.
    ///
    /// `#u2b` — claim 4 is CONDITIONAL on the adapter, claims 1-3 are not.
    /// `verify_eft_primitives()` now entails the rung-3 subnormal policy, so on
    /// an adapter that violates that policy the channel is REFUSED and cannot
    /// fire. Asserting that it fires would demand tightening without its full
    /// preconditions. So when the gate refuses, this test
    /// flips to the complementary oracle that IS checkable there and is just as
    /// load-bearing: the refusal must be BYTE-IDENTICAL — `NY_EFT_ERR=1` must
    /// produce bit-for-bit the same bounds as `NY_EFT_ERR` unset. That is the
    /// fail-closed contract the whole channel design rests on, and before the
    /// #u2b composition it was FALSE on this box (measured: 72/72 specs
    /// tightened on an adapter with `verify_gradual_underflow() == false`).
    #[test]
    fn eft_err_channel_ab_tightens_and_stays_sound() {
        use ny_core::GpuCrownBackward;
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let gpu: &dyn GpuCrownBackward = &*device;
        // Hardware fact, not an assertion. Drives which of claim 4 / the
        // byte-identity claim is checked below.
        let eft_authorized = device.verify_eft_primitives();
        println!(
            "[eft-ab] adapter={} backend={:?} verify_eft_primitives={eft_authorized} \
             (verify_gradual_underflow={})",
            device.adapter_info.name,
            device.adapter_info.backend,
            device.verify_gradual_underflow(),
        );
        let mut n_bit_identical = 0usize;

        let mut state: u64 = 0x5EED_EF71_2026_0723;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let shapes: &[(usize, &[usize], usize)] = &[
            (5, &[8, 6], 4),
            (6, &[10, 8, 7], 5),
            (3, &[16, 16], 3), // wide, heavy cancellation
            (8, &[24, 20, 12], 6),
        ];

        let mut n_specs = 0usize;
        let mut n_tightened = 0usize;
        let mut width_off_sum = 0.0f64;
        let mut width_on_sum = 0.0f64;
        for &(din, hidden, dout) in shapes {
            for _trial in 0..4 {
                let mut dims = vec![din];
                dims.extend_from_slice(hidden);
                dims.push(dout);
                let mut weights: Vec<Vec<f32>> = Vec::new();
                let mut biases: Vec<Vec<f32>> = Vec::new();
                for w in 0..dims.len() - 1 {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    weights.push((0..no * ni).map(|_| rng() * 1.3).collect());
                    biases.push((0..no).map(|_| rng() * 0.15).collect());
                }
                let xc: Vec<f32> = (0..din).map(|_| rng() * 0.4).collect();
                let xl: Vec<f32> = xc.iter().map(|&c| c - 0.3).collect();
                let xu: Vec<f32> = xc.iter().map(|&c| c + 0.3).collect();

                // Forward IBP for ReLU relaxation bounds (same as the enclosure test).
                let relu = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| x.max(0.0)).collect() };
                let mut cur_l = xl.clone();
                let mut cur_u = xu.clone();
                let mut pre_l: Vec<Vec<f32>> = Vec::new();
                let mut pre_u: Vec<Vec<f32>> = Vec::new();
                for w in 0..dims.len() - 1 {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    let wt = &weights[w];
                    let bs = &biases[w];
                    let mut nl = vec![0.0f32; no];
                    let mut nu = vec![0.0f32; no];
                    for o in 0..no {
                        let mut lo = bs[o];
                        let mut hi = bs[o];
                        for j in 0..ni {
                            let c = wt[o * ni + j];
                            if c >= 0.0 {
                                lo += c * cur_l[j];
                                hi += c * cur_u[j];
                            } else {
                                lo += c * cur_u[j];
                                hi += c * cur_l[j];
                            }
                        }
                        nl[o] = lo;
                        nu[o] = hi;
                    }
                    if w < dims.len() - 2 {
                        pre_l.push(nl.clone());
                        pre_u.push(nu.clone());
                        cur_l = relu(&nl);
                        cur_u = relu(&nu);
                    }
                }
                let mut layers: Vec<GpuCrownLayer> = Vec::new();
                let n_lin = dims.len() - 1;
                for w in (0..n_lin).rev() {
                    let (ni, no) = (dims[w], dims[w + 1]);
                    layers.push(GpuCrownLayer::Linear {
                        weight: Arc::from(weights[w].clone().into_boxed_slice()),
                        bias: Some(Arc::from(biases[w].clone().into_boxed_slice())),
                        out_features: no,
                        in_features: ni,
                        cert_err: Default::default(),
                    });
                    if w > 0 {
                        let stage = w - 1;
                        let (l, u) = (&pre_l[stage], &pre_u[stage]);
                        let nn = l.len();
                        let mut ls = vec![0.0f32; nn];
                        let mut us = vec![0.0f32; nn];
                        let li = vec![0.0f32; nn];
                        let mut ui = vec![0.0f32; nn];
                        for i in 0..nn {
                            let (lo, hi) = (l[i], u[i]);
                            if lo >= 0.0 {
                                ls[i] = 1.0;
                                us[i] = 1.0;
                            } else if hi > 0.0 {
                                let alpha = 0.5 + 0.49 * rng();
                                ls[i] = alpha.clamp(0.0, 1.0);
                                us[i] = hi / (hi - lo);
                                ui[i] = -hi * lo / (hi - lo);
                            }
                        }
                        layers.push(GpuCrownLayer::Activation {
                            lower_slope: ls,
                            upper_slope: us,
                            lower_intercept: li,
                            upper_intercept: ui,
                            num_neurons: nn,
                        });
                    }
                }
                let mut spec = vec![0.0f32; dout * dout];
                for i in 0..dout {
                    spec[i * dout + i] = 1.0;
                }

                // A/B: gate OFF then ON (env flipped under the serialized guard;
                // the off-arm explicitly UNSETS so an outer NY_EFT_ERR=1 —
                // e.g. a whole-suite EFT battery — cannot collapse off==on).
                let off = ny_test_utils::env::with_env_edits(|env| {
                    env.remove("NY_EFT_ERR");
                    gpu.crown_backward_gpu_sound(&layers, &spec, dout, &xl, &xu)
                        .expect("sound backward, EFT off")
                });
                let on = ny_test_utils::env::with_env_edits(|env| {
                    env.set("NY_EFT_ERR", "1");
                    gpu.crown_backward_gpu_sound(&layers, &spec, dout, &xl, &xu)
                        .expect("sound backward, EFT on")
                });

                // EXACT f64 reference of the SAME CROWN relaxation (identical
                // backward walk in f64; its own rounding is ~1e-13-class —
                // negligible against the asserted tolerances). This is the
                // decisive soundness reference: the EFT-tightened bound may be
                // TIGHTER than the CPU host's CHARGED bound (that was the whole
                // point), but it must never cross the exact relaxation bound.
                let (exact_lo, exact_hi) = {
                    let n0 = dout;
                    let mut al: Vec<Vec<f64>> = (0..n0)
                        .map(|i| (0..n0).map(|j| f64::from(spec[i * n0 + j])).collect())
                        .collect();
                    let mut au: Vec<Vec<f64>> = al.clone();
                    let mut bl = vec![0.0f64; n0];
                    let mut bu = vec![0.0f64; n0];
                    for layer in &layers {
                        match layer {
                            GpuCrownLayer::Linear {
                                weight,
                                bias,
                                out_features,
                                in_features,
                                ..
                            } => {
                                let (of, if_) = (*out_features, *in_features);
                                if let Some(bs) = bias {
                                    for i in 0..n0 {
                                        for o in 0..of {
                                            bl[i] += al[i][o] * f64::from(bs[o]);
                                            bu[i] += au[i][o] * f64::from(bs[o]);
                                        }
                                    }
                                }
                                let mm = |a: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
                                    a.iter()
                                        .map(|row| {
                                            (0..if_)
                                                .map(|j| {
                                                    (0..of)
                                                        .map(|o| {
                                                            row[o] * f64::from(weight[o * if_ + j])
                                                        })
                                                        .sum()
                                                })
                                                .collect()
                                        })
                                        .collect()
                                };
                                al = mm(&al);
                                au = mm(&au);
                            }
                            GpuCrownLayer::Activation {
                                lower_slope,
                                upper_slope,
                                lower_intercept,
                                upper_intercept,
                                num_neurons,
                            } => {
                                for i in 0..n0 {
                                    for j in 0..*num_neurons {
                                        let (ls, us) =
                                            (f64::from(lower_slope[j]), f64::from(upper_slope[j]));
                                        let (li, ui) = (
                                            f64::from(lower_intercept[j]),
                                            f64::from(upper_intercept[j]),
                                        );
                                        let c = al[i][j];
                                        if c >= 0.0 {
                                            al[i][j] = c * ls;
                                            bl[i] += c * li;
                                        } else {
                                            al[i][j] = c * us;
                                            bl[i] += c * ui;
                                        }
                                        let c = au[i][j];
                                        if c >= 0.0 {
                                            au[i][j] = c * us;
                                            bu[i] += c * ui;
                                        } else {
                                            au[i][j] = c * ls;
                                            bu[i] += c * li;
                                        }
                                    }
                                }
                            }
                            _ => panic!("unexpected layer kind in the A/B fold"),
                        }
                    }
                    let lo: Vec<f64> = (0..n0)
                        .map(|i| {
                            bl[i]
                                + al[i]
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &c)| (c * f64::from(xl[j])).min(c * f64::from(xu[j])))
                                    .sum::<f64>()
                        })
                        .collect();
                    let hi: Vec<f64> = (0..n0)
                        .map(|i| {
                            bu[i]
                                + au[i]
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &c)| (c * f64::from(xl[j])).max(c * f64::from(xu[j])))
                                    .sum::<f64>()
                        })
                        .collect();
                    (lo, hi)
                };

                for k in 0..dout {
                    let (lo_off, hi_off) = (off.lower_bounds[k], off.upper_bounds[k]);
                    let (lo_on, hi_on) = (on.lower_bounds[k], on.upper_bounds[k]);
                    assert!(lo_on.is_finite() && hi_on.is_finite() && lo_on <= hi_on);
                    // (1) Monotone: the EFT min can only tighten. Zero slack.
                    assert!(
                        lo_on >= lo_off && hi_on <= hi_off,
                        "EFT channel LOOSENED a bound: off=[{lo_off},{hi_off}] on=[{lo_on},{hi_on}] spec {k}"
                    );
                    // (3) THE soundness law: never cross the EXACT relaxation
                    // bound. Tiny slack covers only the f64 reference's own
                    // rounding (~1e-13-class) — effectively zero at f32 scale.
                    assert!(
                        f64::from(lo_on) <= exact_lo[k] + 1e-6
                            && f64::from(hi_on) >= exact_hi[k] - 1e-6,
                        "EFT bound CROSSES the exact relaxation bound: \
                         on=[{lo_on},{hi_on}] exact=[{},{}] spec {k}",
                        exact_lo[k],
                        exact_hi[k]
                    );
                    if lo_on > lo_off || hi_on < hi_off {
                        n_tightened += 1;
                    }
                    // #u2b: when the gate REFUSES, the fallback must be exact —
                    // same bits, not merely "no worse". Counted always, asserted
                    // below only in the refused case.
                    if lo_on.to_bits() == lo_off.to_bits() && hi_on.to_bits() == hi_off.to_bits() {
                        n_bit_identical += 1;
                    }
                    width_off_sum += f64::from(hi_off) - f64::from(lo_off);
                    width_on_sum += f64::from(hi_on) - f64::from(lo_on);
                    n_specs += 1;
                }

                // (2) True-output enclosure for the ON bounds, zero favorable slack.
                for t in 0..300 {
                    let x: Vec<f32> = (0..din)
                        .map(|i| {
                            let frac = (((t * 37 + i * 13) % 101) as f32) / 100.0;
                            xl[i] + frac * (xu[i] - xl[i])
                        })
                        .collect();
                    let mut v = x.clone();
                    for w in 0..n_lin {
                        let (ni, no) = (dims[w], dims[w + 1]);
                        let mut nv = matmul(&weights[w], &v, no, ni);
                        for o in 0..no {
                            nv[o] += biases[w][o];
                        }
                        if w < n_lin - 1 {
                            for o in 0..no {
                                nv[o] = nv[o].max(0.0);
                            }
                        }
                        v = nv;
                    }
                    for o in 0..dout {
                        assert!(
                            on.lower_bounds[o] <= v[o] + 3e-3 && v[o] <= on.upper_bounds[o] + 3e-3,
                            "UNSOUND with EFT on: true output[{o}]={} not in [{}, {}]",
                            v[o],
                            on.lower_bounds[o],
                            on.upper_bounds[o]
                        );
                    }
                }
            }
        }
        assert!(n_specs > 0, "the A/B fold produced no specs to compare");
        if eft_authorized {
            // (4) The channel must actually fire on cancellation-heavy folds.
            assert!(
                n_tightened * 2 >= n_specs,
                "EFT channel barely fired: {n_tightened}/{n_specs} specs tightened"
            );
        } else {
            // (4′) #u2b BYTE-IDENTICAL REFUSAL. The gate refused, so
            // `NY_EFT_ERR=1` must be
            // a complete no-op on the published bounds — every spec bit-for-bit
            // equal to the gate-off arm. Anything else means some part of the
            // compensated path is still reachable behind a refused gate.
            assert_eq!(
                n_bit_identical,
                n_specs,
                "EFT gate REFUSED (verify_eft_primitives=false) but NY_EFT_ERR=1 \
                 still changed {} / {n_specs} bounds — the refusal is not \
                 byte-identical, so the compensated channel is partially \
                 reachable behind a closed gate",
                n_specs - n_bit_identical
            );
            assert_eq!(
                n_tightened, 0,
                "EFT gate REFUSED but the channel still tightened {n_tightened} \
                 specs — a refused gate must not narrow any bound"
            );
        }
        println!(
            "[eft-ab] authorized={eft_authorized} specs={n_specs} tightened={n_tightened} \
             bit_identical={n_bit_identical} mean_width off={:.6e} on={:.6e} (ratio {:.3})",
            width_off_sum / n_specs as f64,
            width_on_sum / n_specs as f64,
            width_off_sum / width_on_sum.max(1e-300),
        );
    }

    /// #batched-bab INCREMENT 1 differential oracle: the reference-stacker batched
    /// entry must return, for every domain-block, EXACTLY (bit-for-bit) what the
    /// serial per-domain `crown_backward_gpu_resnet_sound_beta` returns — across
    /// DELIBERATELY DISTINCT per-domain relaxation slopes / β / input boxes, so an
    /// off-by-one-block mis-index (the HOLE-1/2/3 hazard the wide kernel must
    /// avoid) would deviate `>>` tol. Plus a contamination probe (mutating one
    /// domain leaves the others byte-unchanged) and the homogeneity gate (a
    /// heterogeneous skeleton aborts to `Err` → serial fallback). This harness is
    /// reused verbatim by increment 2 (switching CHECK A to two-sided `|batched −
    /// serial| ≤ tol` once the wide kernel reorders each row's own contraction).
    #[test]
    fn crown_batched_reference_stacker_matches_serial_per_domain() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (o, h, i) = (3usize, 6usize, 4usize);
        let num_specs = 2usize;
        let mut state: u64 = 0x5EED_B0B5;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // SHARED weights (same Arc across every domain → Arc::ptr_eq holds).
        let w2: Arc<[f32]> = (0..o * h).map(|_| rng() * 1.5).collect::<Vec<_>>().into();
        let w1: Arc<[f32]> = (0..h * i).map(|_| rng()).collect::<Vec<_>>().into();
        let seed_a: Vec<f32> = (0..num_specs * o).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: o,
        };

        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
        }
        // Per-domain DISTINCT relaxation/β/box; SHARED weight Arcs.
        let build = |d: usize, w2: &Arc<[f32]>, w1: &Arc<[f32]>| -> Dom {
            let df = d as f32;
            Dom {
                segments: vec![GpuResnetSegment::Chain(vec![
                    GpuCrownLayer::Linear {
                        weight: w2.clone(),
                        bias: None,
                        out_features: o,
                        in_features: h,
                        cert_err: Default::default(),
                    },
                    GpuCrownLayer::Activation {
                        lower_slope: vec![0.30 + 0.13 * df; h],
                        upper_slope: vec![0.62 + 0.11 * df; h],
                        lower_intercept: vec![0.02 * df; h],
                        upper_intercept: vec![0.10 + 0.03 * df; h],
                        num_neurons: h,
                    },
                    GpuCrownLayer::Linear {
                        weight: w1.clone(),
                        bias: None,
                        out_features: h,
                        in_features: i,
                        cert_err: Default::default(),
                    },
                ])],
                in_lo: (0..i).map(|k| -1.0 - 0.2 * df - 0.05 * k as f32).collect(),
                in_hi: (0..i).map(|k| 1.0 + 0.2 * df + 0.05 * k as f32).collect(),
                beta: vec![vec![0.05 * df; h]],
                fa: vec![],
                na: vec![],
            }
        };
        let doms: Vec<Dom> = (0..3).map(|d| build(d, &w2, &w1)).collect();
        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dd| GpuResnetBatchedDomainRef {
                segments: &dd.segments,
                input_lower: &dd.in_lo,
                input_upper: &dd.in_hi,
                beta_signed: &dd.beta,
                frontier_abs: &dd.fa,
                node_abs: &dd.na,
            })
            .collect();

        // CHECK A (bit-exact): batched[d] == serial per-domain, for every DISTINCT domain.
        let batched = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
            .expect("batched reference stacker");
        assert_eq!(batched.len(), doms.len());
        for (d, dd) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_beta(
                    &dd.segments,
                    &seed,
                    &dd.in_lo,
                    &dd.in_hi,
                    &dd.beta,
                    &dd.fa,
                    &dd.na,
                )
                .expect("serial per-domain");
            assert_eq!(
                batched[d].lower_bounds, serial.lower_bounds,
                "domain {d} lower mismatch (partition/unpack/mis-index bug)"
            );
            assert_eq!(
                batched[d].upper_bounds, serial.upper_bounds,
                "domain {d} upper mismatch (partition/unpack/mis-index bug)"
            );
        }

        // CONTAM: mutating ONLY domain 1 leaves domains 0 and 2 byte-unchanged.
        let mut doms2: Vec<Dom> = (0..3).map(|d| build(d, &w2, &w1)).collect();
        if let GpuResnetSegment::Chain(ls) = &mut doms2[1].segments[0] {
            if let GpuCrownLayer::Activation { lower_slope, .. } = &mut ls[1] {
                for s in lower_slope.iter_mut() {
                    *s += 0.4;
                }
            }
        }
        let refs2: Vec<GpuResnetBatchedDomainRef> = doms2
            .iter()
            .map(|dd| GpuResnetBatchedDomainRef {
                segments: &dd.segments,
                input_lower: &dd.in_lo,
                input_upper: &dd.in_hi,
                beta_signed: &dd.beta,
                frontier_abs: &dd.fa,
                node_abs: &dd.na,
            })
            .collect();
        let batched2 = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs2, &seed)
            .expect("batched after domain-1 mutation");
        assert_eq!(
            batched2[0].lower_bounds, batched[0].lower_bounds,
            "domain 0 contaminated by domain 1's mutation"
        );
        assert_eq!(
            batched2[2].lower_bounds, batched[2].lower_bounds,
            "domain 2 contaminated by domain 1's mutation"
        );
        assert_ne!(
            batched2[1].lower_bounds, batched[1].lower_bounds,
            "domain 1 mutation had no effect (fixture bug — not exercising the path)"
        );

        // HETERO: a domain with a different skeleton aborts the WHOLE batch to Err
        // (homogeneity gate) so the caller falls back to the serial path.
        let mut het = build(0, &w2, &w1);
        if let GpuResnetSegment::Chain(ls) = &mut het.segments[0] {
            ls.push(GpuCrownLayer::Activation {
                lower_slope: vec![0.5; i],
                upper_slope: vec![0.5; i],
                lower_intercept: vec![0.0; i],
                upper_intercept: vec![0.0; i],
                num_neurons: i,
            });
        }
        let het_refs = vec![
            GpuResnetBatchedDomainRef {
                segments: &doms[0].segments,
                input_lower: &doms[0].in_lo,
                input_upper: &doms[0].in_hi,
                beta_signed: &doms[0].beta,
                frontier_abs: &doms[0].fa,
                node_abs: &doms[0].na,
            },
            GpuResnetBatchedDomainRef {
                segments: &het.segments,
                input_lower: &het.in_lo,
                input_upper: &het.in_hi,
                beta_signed: &het.beta,
                frontier_abs: &het.fa,
                node_abs: &het.na,
            },
        ];
        assert!(
            device
                .crown_backward_gpu_resnet_sound_beta_batched(&het_refs, &seed)
                .is_err(),
            "heterogeneous batch must abort to Err so the caller uses the serial path"
        );
    }

    /// #batched-bab ARMING test (2026-08-11): on a synthetic 2-domain
    /// RESNET-ish stack (Chain[Linear,Activation] + Residual[Linear], shared
    /// weight `Arc`s, per-domain DISTINCT relaxation/β/box) the batched trait
    /// entry must (i) TAKE the wide one-pass sound lane — asserted via the
    /// PRODUCTION-ARMED `wide_resnet_batched_taken_count()` counter, not probe
    /// stderr — and (ii) produce per-domain bounds that ENCLOSE the serial
    /// per-domain sound reference (never tighter beyond the f32 GEMM-reorder
    /// tolerance: a tighter wide bound is exactly the false-VERIFY hazard the
    /// refold guard exists for) while matching it two-sided within the same
    /// tolerance (the in-tree differential-oracle contract).
    ///
    /// Also pins the newly-armed deadline-bounded capability surface on the
    /// REAL device: honest K=8 capacity, working 2-row bounded call that is
    /// byte-identical to the unbounded sound entry, and a refusal (never a
    /// late publication) once the deadline has passed.
    #[test]
    fn crown_batched_wide_sound_lane_taken_and_encloses_serial_reference() {
        let _g = gpu_test_serial_guard();
        // The wide lane is DEFAULT-ON; make the arming explicit + scoped so a
        // stray environment cannot silently turn this into the stacker path.
        let _wide_on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE", "1");
        let device = require_verdict_device();

        // Capability surface (the pre-fix decline point was max_rows == 0).
        assert!(device.provides_sound_gpu_crown());
        assert!(device.honors_crown_backward_deadline());
        assert!(device.provides_deadline_bounded_single_row_resnet_sound());
        assert_eq!(
            device.deadline_bounded_resnet_sound_max_rows(),
            ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
            "honest bounded-rows capacity must be the full audited K=8 contract"
        );

        let (o, h) = (3usize, 5usize);
        let num_specs = 2usize;
        let mut state: u64 = 0x71DE_A51D_5EED;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // SHARED weights (same Arc across domains — the homogeneity gate needs
        // Arc::ptr_eq); per-domain DISTINCT Activation relaxation, box, and β.
        let w_out: Arc<[f32]> = (0..o * h).map(|_| rng()).collect::<Vec<_>>().into();
        let w_f: Arc<[f32]> = (0..h * h).map(|_| rng() * 0.4).collect::<Vec<_>>().into();
        let seed_a: Vec<f32> = (0..num_specs * o).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: o,
        };
        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
        }
        let build = |d: usize| -> Dom {
            let df = d as f32;
            Dom {
                segments: vec![
                    // FOLD order (output→input): output Chain, then the
                    // residual block (identity skip, affine F) — resnet-ish.
                    GpuResnetSegment::Chain(vec![
                        GpuCrownLayer::Linear {
                            weight: w_out.clone(),
                            bias: None,
                            out_features: o,
                            in_features: h,
                            cert_err: Default::default(),
                        },
                        GpuCrownLayer::Activation {
                            lower_slope: vec![0.25 + 0.17 * df; h],
                            upper_slope: vec![0.60 + 0.12 * df; h],
                            lower_intercept: vec![0.03 * df; h],
                            upper_intercept: vec![0.08 + 0.04 * df; h],
                            num_neurons: h,
                        },
                    ]),
                    GpuResnetSegment::Residual(vec![GpuCrownLayer::Linear {
                        weight: w_f.clone(),
                        bias: None,
                        out_features: h,
                        in_features: h,
                        cert_err: Default::default(),
                    }]),
                ],
                in_lo: (0..h).map(|k| -0.8 - 0.15 * df - 0.03 * k as f32).collect(),
                in_hi: (0..h).map(|k| 0.8 + 0.15 * df + 0.03 * k as f32).collect(),
                beta: vec![vec![0.04 * df; h]],
            }
        };
        let doms: Vec<Dom> = (0..2).map(build).collect();
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dd| GpuResnetBatchedDomainRef {
                segments: &dd.segments,
                input_lower: &dd.in_lo,
                input_upper: &dd.in_hi,
                beta_signed: &dd.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect();

        // (i) the wide sound lane is TAKEN (production counter, not stderr).
        let taken_before = super::super::crown_backward::wide_resnet_batched_taken_count();
        let batched = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
            .expect("batched resnet-ish 2-domain call");
        assert_eq!(batched.len(), doms.len());
        assert!(
            super::super::crown_backward::wide_resnet_batched_taken_count() > taken_before,
            "the ONE-pass wide sound lane must be TAKEN for a 2-domain homogeneous \
             resnet-ish batch (counter unchanged ⇒ it fell to the reference stacker)"
        );

        // (ii) per-domain bounds ENCLOSE the serial sound reference (and match
        // it two-sided within the oracle's f32 GEMM-reorder tolerance).
        for (d, dd) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_beta(
                    &dd.segments,
                    &seed,
                    &dd.in_lo,
                    &dd.in_hi,
                    &dd.beta,
                    &empty,
                    &empty,
                )
                .expect("serial per-domain sound reference");
            for r in 0..num_specs {
                let (wl, wu) = (batched[d].lower_bounds[r], batched[d].upper_bounds[r]);
                let (sl, su) = (serial.lower_bounds[r], serial.upper_bounds[r]);
                let tol = 1e-3 * (1.0 + sl.abs().max(su.abs()));
                // ENCLOSURE (soundness-critical direction): the wide bound must
                // never be TIGHTER than the serial sound reference beyond tol.
                assert!(
                    wl <= sl + tol,
                    "domain {d} row {r}: wide lower {wl} tighter than serial {sl} (+{tol})"
                );
                assert!(
                    wu >= su - tol,
                    "domain {d} row {r}: wide upper {wu} tighter than serial {su} (-{tol})"
                );
                // PARITY (two-sided differential oracle contract).
                assert!(
                    (wl - sl).abs() <= tol && (wu - su).abs() <= tol,
                    "domain {d} row {r}: wide [{wl},{wu}] vs serial [{sl},{su}] exceeds tol {tol}"
                );
            }
        }

        // Deadline-bounded 2-row entry: byte-identical to the unbounded sound
        // entry under a generous deadline (same inner resident path)...
        let dd = &doms[0];
        let generous = std::time::Instant::now() + std::time::Duration::from_mins(1);
        let bounded = device
            .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                &dd.segments,
                &seed,
                &dd.in_lo,
                &dd.in_hi,
                &empty,
                &empty,
                generous,
            )
            .expect("bounded 2-row deadline entry");
        let plain = device
            .crown_backward_gpu_resnet_sound(
                &dd.segments,
                &seed,
                &dd.in_lo,
                &dd.in_hi,
                &empty,
                &empty,
            )
            .expect("unbounded sound entry");
        assert_eq!(bounded.lower_bounds, plain.lower_bounds);
        assert_eq!(bounded.upper_bounds, plain.upper_bounds);
        // ...and a REFUSAL (never a late publication) once the deadline passed.
        // A deadline captured NOW is already unmet by the entry's pre-check
        // (`Instant::now() >= deadline`), with no Instant-underflow risk.
        let expired = std::time::Instant::now();
        assert!(
            device
                .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                    &dd.segments,
                    &seed,
                    &dd.in_lo,
                    &dd.in_hi,
                    &empty,
                    &empty,
                    expired,
                )
                .is_err(),
            "an expired deadline must refuse, not publish"
        );
    }

    /// #batched-bab HOLE-7 SUB-GROUPING device oracle (the coverage increment).
    ///
    /// A HETEROGENEOUS wave `[A, A, B, B]` — two skeletons that differ in SEGMENT
    /// COUNT, per-domain distinct relaxation/box/β — is exactly what today's
    /// homogeneity gate throws away wholesale. This pins all three behaviours:
    ///
    /// 1. GATE OFF (the default, and the scored configuration): the entry still
    ///    returns `Err` and the production publication counter does NOT move, so
    ///    landing the lane changes nothing until it is deliberately armed.
    /// 2. GATE ON: the entry PUBLISHES — the production
    ///    `wide_resnet_batched_taken_count()` rises (one wide pass per homogeneous
    ///    run), which is the coverage the campaign is buying — and returns one
    ///    result per domain in the caller's domain order.
    /// 3. Every published per-domain bound ENCLOSES that domain's SERIAL sound
    ///    reference (never tighter beyond the f32 GEMM-reorder tolerance — a
    ///    tighter sub-grouped bound is precisely the false-VERIFY hazard) and
    ///    matches it two-sided: the same differential-oracle contract the
    ///    homogeneous wide lane is held to.
    #[test]
    fn crown_batched_wide_subgroup_publishes_and_encloses_serial_reference() {
        // Review defect 1 (mirror): the unit pin in crown_backward/tests.rs
        // writes the same NY_BAB_RESNET_WIDE_SUBGROUP under lock_env() only.
        // Take BOTH guards here too, in the same order, so neither suite can
        // observe the other's value mid-assertion.
        let _env_guard = ny_test_utils::env::lock_env();
        let _g = gpu_test_serial_guard();
        let _wide_on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE", "1");
        let device = require_verdict_device();

        let (o, h) = (3usize, 5usize);
        let num_specs = 2usize;
        let mut state: u64 = 0x5B67_0FF5_1234;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // SHARED weights so homogeneity holds WITHIN each run (Arc::ptr_eq).
        let w_out: Arc<[f32]> = (0..o * h).map(|_| rng()).collect::<Vec<_>>().into();
        let w_f: Arc<[f32]> = (0..h * h).map(|_| rng() * 0.4).collect::<Vec<_>>().into();
        let seed_a: Vec<f32> = (0..num_specs * o).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: o,
        };
        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
        }
        // `with_residual == false` drops the Residual segment, so skeleton A and
        // skeleton B differ STRUCTURALLY (segment count) — `resnet_skeleton_matches`
        // refuses across the A/B boundary while holding inside each run. Both map
        // input dim h → output dim o, so one shared seed and box length serve both.
        let build = |d: usize, with_residual: bool| -> Dom {
            let df = d as f32;
            let mut segments = vec![GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Linear {
                    weight: w_out.clone(),
                    bias: None,
                    out_features: o,
                    in_features: h,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Activation {
                    lower_slope: vec![0.25 + 0.11 * df; h],
                    upper_slope: vec![0.60 + 0.09 * df; h],
                    lower_intercept: vec![0.03 * df; h],
                    upper_intercept: vec![0.08 + 0.04 * df; h],
                    num_neurons: h,
                },
            ])];
            if with_residual {
                segments.push(GpuResnetSegment::Residual(vec![GpuCrownLayer::Linear {
                    weight: w_f.clone(),
                    bias: None,
                    out_features: h,
                    in_features: h,
                    cert_err: Default::default(),
                }]));
            }
            Dom {
                segments,
                in_lo: (0..h).map(|k| -0.8 - 0.13 * df - 0.03 * k as f32).collect(),
                in_hi: (0..h).map(|k| 0.8 + 0.13 * df + 0.03 * k as f32).collect(),
                beta: vec![vec![0.04 * df; h]],
            }
        };
        // Contiguous runs: domains 0,1 share skeleton A; domains 2,3 share B.
        let doms: Vec<Dom> = vec![
            build(0, true),
            build(1, true),
            build(2, false),
            build(3, false),
        ];
        let empty: Vec<Vec<f32>> = Vec::new();
        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dd| GpuResnetBatchedDomainRef {
                segments: &dd.segments,
                input_lower: &dd.in_lo,
                input_upper: &dd.in_hi,
                beta_signed: &dd.beta,
                frontier_abs: &empty,
                node_abs: &empty,
            })
            .collect();
        // Fixture self-check: the wave really is heterogeneous (differing segment
        // COUNT is what `resnet_skeleton_matches` refuses on), or the test would
        // silently prove nothing about sub-grouping.
        assert_eq!(doms[0].segments.len(), doms[1].segments.len());
        assert_eq!(doms[2].segments.len(), doms[3].segments.len());
        assert_ne!(
            doms[0].segments.len(),
            doms[2].segments.len(),
            "fixture bug: skeletons A and B must differ for this to exercise HOLE 7"
        );

        // (1) GATE OFF ⇒ historical refusal, and NO publication.
        {
            let _off = ScopedEnvVar::unset("NY_BAB_RESNET_WIDE_SUBGROUP");
            let before = super::super::crown_backward::wide_resnet_batched_taken_count();
            assert!(
                device
                    .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
                    .is_err(),
                "with the sub-group gate OFF a heterogeneous batch must still abort \
                 to the caller's serial path (byte-identical scored routing)"
            );
            assert_eq!(
                super::super::crown_backward::wide_resnet_batched_taken_count(),
                before,
                "the dark gate must not publish anything"
            );
        }

        // (2) GATE ON ⇒ the wave PUBLISHES, one wide pass per homogeneous run.
        let _sub_on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE_SUBGROUP", "1");
        let taken_before = super::super::crown_backward::wide_resnet_batched_taken_count();
        let batched = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
            .expect("sub-grouped heterogeneous batch must publish");
        assert_eq!(batched.len(), doms.len(), "one result per domain, in order");
        assert!(
            super::super::crown_backward::wide_resnet_batched_taken_count() >= taken_before + 2,
            "each homogeneous run is one wide publication (>= 2 for A,A|B,B); the \
             counter standing still means the whole wave silently fell back"
        );

        // (3) ENCLOSURE + two-sided parity against the SERIAL per-domain reference.
        for (d, dd) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_beta(
                    &dd.segments,
                    &seed,
                    &dd.in_lo,
                    &dd.in_hi,
                    &dd.beta,
                    &empty,
                    &empty,
                )
                .expect("serial per-domain sound reference");
            for r in 0..num_specs {
                let (wl, wu) = (batched[d].lower_bounds[r], batched[d].upper_bounds[r]);
                let (sl, su) = (serial.lower_bounds[r], serial.upper_bounds[r]);
                let tol = 1e-3 * (1.0 + sl.abs().max(su.abs()));
                assert!(
                    wl <= sl + tol,
                    "domain {d} row {r}: sub-grouped lower {wl} TIGHTER than serial {sl} (+{tol})"
                );
                assert!(
                    wu >= su - tol,
                    "domain {d} row {r}: sub-grouped upper {wu} TIGHTER than serial {su} (-{tol})"
                );
                assert!(
                    (wl - sl).abs() <= tol && (wu - su).abs() <= tol,
                    "domain {d} row {r}: sub-grouped [{wl},{wu}] vs serial [{sl},{su}] > tol {tol}"
                );
            }
        }
    }

    /// #batched-bab HOLE-7 SUB-GROUPING fail-closed pin: arming the sub-group lane
    /// must NOT weaken HOLE 8. A heterogeneous wave in which one run carries a
    /// dual-alpha ReLU (backward shader not domain-block-indexed) must be refused
    /// WHOLESALE — never partially folded, never mixed wide-and-serial — so the
    /// caller's proven per-domain path runs exactly as it does today.
    #[test]
    fn crown_batched_wide_subgroup_still_declines_hole8_runs() {
        // Review defect 1 (mirror): the unit pin in crown_backward/tests.rs
        // writes the same NY_BAB_RESNET_WIDE_SUBGROUP under lock_env() only.
        // Take BOTH guards here too, in the same order, so neither suite can
        // observe the other's value mid-assertion.
        let _env_guard = ny_test_utils::env::lock_env();
        let _g = gpu_test_serial_guard();
        let _wide_on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE", "1");
        let _sub_on = ScopedEnvVar::set("NY_BAB_RESNET_WIDE_SUBGROUP", "1");
        let device = require_device();

        let (o, h) = (3usize, 4usize);
        let num_specs = 1usize;
        let w_out: Arc<[f32]> = (0..o * h)
            .map(|n| 0.02 * n as f32)
            .collect::<Vec<_>>()
            .into();
        let seed_a: Vec<f32> = (0..num_specs * o).map(|n| 0.1 * (n as f32 + 1.0)).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: o,
        };
        let plain = |slope: f32| -> Vec<GpuResnetSegment> {
            vec![GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Linear {
                    weight: w_out.clone(),
                    bias: None,
                    out_features: o,
                    in_features: h,
                    cert_err: Default::default(),
                },
                GpuCrownLayer::Activation {
                    lower_slope: vec![slope; h],
                    upper_slope: vec![slope + 0.3; h],
                    lower_intercept: vec![0.0; h],
                    upper_intercept: vec![0.05; h],
                    num_neurons: h,
                },
            ])]
        };
        // Run 2 is HOLE-8: a dual-alpha ReLU the wide fold cannot domain-block.
        let hole8 = vec![GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Linear {
                weight: w_out.clone(),
                bias: None,
                out_features: o,
                in_features: h,
                cert_err: Default::default(),
            },
            GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope: vec![0.5; h],
                cross_slope: vec![0.6; h],
                upper_neg_slope: vec![0.5; h],
                cross_intercept: vec![0.1; h],
                num_neurons: h,
            },
        ])];
        let (s0, s1, s2) = (plain(0.25), plain(0.35), hole8);
        let (lo, hi) = (vec![-1.0f32; h], vec![1.0f32; h]);
        let beta: Vec<Vec<f32>> = vec![vec![0.0; h]];
        let empty: Vec<Vec<f32>> = Vec::new();
        // Inlined (was a closure): a closure cannot express "the returned ref
        // borrows from the ARGUMENT, not from my frame", so the three domain
        // refs are built directly.
        let dom = |segments: &'static Vec<GpuResnetSegment>| segments;
        let _ = dom;
        let refs = vec![
            GpuResnetBatchedDomainRef {
                segments: &s0,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &beta,
                frontier_abs: &empty,
                node_abs: &empty,
            },
            GpuResnetBatchedDomainRef {
                segments: &s1,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &beta,
                frontier_abs: &empty,
                node_abs: &empty,
            },
            GpuResnetBatchedDomainRef {
                segments: &s2,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &beta,
                frontier_abs: &empty,
                node_abs: &empty,
            },
        ];
        let before = super::super::crown_backward::wide_resnet_batched_taken_count();
        assert!(
            device
                .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
                .is_err(),
            "a wave containing a HOLE-8 run must be refused wholesale even with \
             sub-grouping armed"
        );
        assert_eq!(
            super::super::crown_backward::wide_resnet_batched_taken_count(),
            before,
            "fail-closed means NO run is published when any run is unfoldable"
        );
    }

    /// #metaroom-chain-wide differential oracle: a PURE-CHAIN CONV batch — the exact
    /// segment shape the chain-permitting extractor emits for metaroom's 6cnn conv
    /// chains (`segments = [Chain(conv, act, conv, act, conv)]`, ONE per-segment
    /// frontier_abs entry, per-ReLU node_abs, β on both ReLUs) — must match the serial
    /// per-domain `crown_backward_gpu_resnet_sound_beta` within an f32 GEMM-reorder tol
    /// for every DELIBERATELY DISTINCT domain (slopes/box/β/abs tables all differ per
    /// domain, so any cross-domain mis-index deviates >> tol). CONTAM leg stays
    /// BIT-EXACT (wide-vs-wide): mutating only domain 1's relaxation leaves domains 0
    /// and 2 byte-unchanged. This is the soundness gate for routing pure conv-chain
    /// BaB re-bounds down the wide batched lane (NY_BAB_CHAIN_WIDE).
    #[test]
    fn crown_batched_chain_only_conv_matches_serial_per_domain() {
        let _g = gpu_test_serial_guard();
        // This is verdict-facing: AUTO must carry Conv words and run the full
        // two-sided/contamination oracle below.
        let _taint_words_auto = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        // D = c*hw*hw shared dim; convs are same-padding (k=3,pad=1 → out=hw).
        let (c, hw, k) = (2usize, 3usize, 3usize);
        let d = c * hw * hw; // 18
        let num_specs = 2usize;
        let mut state: u64 = 0xC0DE_C4A1_0FF5;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // SHARED weights (same Arc across every domain → the homogeneity gate holds).
        let conv_w_out: Arc<[f32]> = (0..c * c * k * k)
            .map(|_| rng() * 0.3)
            .collect::<Vec<_>>()
            .into();
        let conv_w_mid: Arc<[f32]> = (0..c * c * k * k)
            .map(|_| rng() * 0.25)
            .collect::<Vec<_>>()
            .into();
        let conv_w_in: Arc<[f32]> = (0..c * c * k * k)
            .map(|_| rng() * 0.2)
            .collect::<Vec<_>>()
            .into();
        let seed_a: Vec<f32> = (0..num_specs * d).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: d,
        };

        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
        }
        let conv = |w: &Arc<[f32]>| GpuCrownLayer::Conv2d {
            weight_col: w.clone(),
            bias_expanded: None,
            out_channels: c,
            in_channels: c,
            kernel_h: k,
            kernel_w: k,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            out_h: hw,
            out_w: hw,
            in_h: hw,
            in_w: hw,
            cert_err: Default::default(),
        };
        let build = |dd: usize| -> Dom {
            let df = dd as f32;
            let act = |o: f32| GpuCrownLayer::Activation {
                lower_slope: vec![0.28 + 0.14 * df + o; d],
                upper_slope: vec![0.60 + 0.12 * df + o; d],
                lower_intercept: vec![0.02 * df + 0.5 * o; d],
                upper_intercept: vec![0.09 + 0.03 * df + o; d],
                num_neurons: d,
            };
            Dom {
                // Backward order (output→input): ONE pure-chain segment, conv-only —
                // the metaroom 6cnn shape (no residual anywhere).
                segments: vec![GpuResnetSegment::Chain(vec![
                    conv(&conv_w_out),
                    act(0.0),
                    conv(&conv_w_mid),
                    act(0.04),
                    conv(&conv_w_in),
                ])],
                in_lo: (0..d).map(|j| -1.0 - 0.2 * df - 0.03 * j as f32).collect(),
                in_hi: (0..d).map(|j| 1.0 + 0.2 * df + 0.03 * j as f32).collect(),
                // 2 ReLUs in fold order; distinct per domain.
                beta: vec![vec![0.05 * df; d], vec![0.03 * df; d]],
                // frontier_abs: ONE entry (one segment; the network-input frontier).
                fa: vec![(0..d).map(|j| 1.0 + 0.2 * df + 0.01 * j as f32).collect()],
                // node_abs: one per ReLU in fold order, distinct per domain.
                na: vec![
                    (0..d).map(|_| 1.1 + 0.25 * df).collect(),
                    (0..d).map(|_| 0.9 + 0.18 * df).collect(),
                ],
            }
        };
        fn make_refs(doms: &[Dom]) -> Vec<GpuResnetBatchedDomainRef<'_>> {
            doms.iter()
                .map(|dd| GpuResnetBatchedDomainRef {
                    segments: &dd.segments,
                    input_lower: &dd.in_lo,
                    input_upper: &dd.in_hi,
                    beta_signed: &dd.beta,
                    frontier_abs: &dd.fa,
                    node_abs: &dd.na,
                })
                .collect()
        }

        let doms: Vec<Dom> = (0..3).map(build).collect();
        let refs = make_refs(&doms);

        // CHECK A (two-sided tol): the wide pass fires (n_domains>1) on the pure-Chain
        // batch; each domain block matches its serial per-domain bound.
        let batched = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
            .expect("worded pure-Conv chain batch");
        assert_eq!(batched.len(), doms.len());
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()));
        for (dd, dom) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_beta(
                    &dom.segments,
                    &seed,
                    &dom.in_lo,
                    &dom.in_hi,
                    &dom.beta,
                    &dom.fa,
                    &dom.na,
                )
                .expect("serial per-domain chain-only");
            for s in 0..num_specs {
                assert!(
                    close(batched[dd].lower_bounds[s], serial.lower_bounds[s]),
                    "domain {dd} spec {s} LOWER: batched={} serial={} (dom mis-index?)",
                    batched[dd].lower_bounds[s],
                    serial.lower_bounds[s]
                );
                assert!(
                    close(batched[dd].upper_bounds[s], serial.upper_bounds[s]),
                    "domain {dd} spec {s} UPPER: batched={} serial={}",
                    batched[dd].upper_bounds[s],
                    serial.upper_bounds[s]
                );
            }
        }

        // CONTAM (bit-exact wide-vs-wide): mutating ONLY domain 1's relaxation +
        // node_abs leaves domains 0 and 2 byte-unchanged.
        let mut doms2: Vec<Dom> = (0..3).map(build).collect();
        if let GpuResnetSegment::Chain(ls) = &mut doms2[1].segments[0] {
            if let GpuCrownLayer::Activation { lower_slope, .. } = &mut ls[1] {
                for s in lower_slope.iter_mut() {
                    *s += 0.35;
                }
            }
        }
        for v in doms2[1].na[0].iter_mut() {
            *v += 0.5;
        }
        let refs2 = make_refs(&doms2);
        let batched2 = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs2, &seed)
            .expect("chain-only batched after domain-1 mutation");
        assert_eq!(
            batched2[0].lower_bounds, batched[0].lower_bounds,
            "domain 0 contaminated by domain 1's mutation"
        );
        assert_eq!(
            batched2[0].upper_bounds, batched[0].upper_bounds,
            "domain 0 (upper) contaminated by domain 1's mutation"
        );
        assert_eq!(
            batched2[2].lower_bounds, batched[2].lower_bounds,
            "domain 2 contaminated by domain 1's mutation"
        );
        assert_eq!(
            batched2[2].upper_bounds, batched[2].upper_bounds,
            "domain 2 (upper) contaminated by domain 1's mutation"
        );
        assert_ne!(
            batched2[1].lower_bounds, batched[1].lower_bounds,
            "domain 1 mutation had no effect (fixture bug — path not exercised)"
        );
    }

    /// #batched-bab increment 3 — the WIDE-PASS two-sided differential oracle over a
    /// MULTI-SEGMENT topology (a Conv2d Chain + an identity Residual, TWO Activations)
    /// with DISTINCT per-domain `frontier_abs`/`node_abs` so the error-concretization
    /// folds FIRE (HOLE 4) and the conv/residual error-composition path is exercised at
    /// width N — the coverage the single-Chain oracle above cannot reach (per the
    /// design's adversarial review). The wide single-pass bound for EACH domain block
    /// must match that domain's serial per-domain bound within an f32-reorder tol; a
    /// dom-mis-index (folding one domain's rows against another's slopes/box/abs-max)
    /// deviates >> tol because every per-domain input is deliberately distinct.
    /// CONTAM stays BIT-EXACT (wide-vs-wide): mutating one domain must not perturb any
    /// other domain's block.
    #[test]
    fn crown_batched_wide_multi_segment_matches_serial_per_domain() {
        let _g = gpu_test_serial_guard();
        // This is a verdict-facing entry: pin today's typed refusal, while
        // retaining the differential oracle for the day Conv words are admitted.
        let _taint_words_auto = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        // D = c*hw*hw shared block dim; conv is same-padding (k=3,pad=1 → out=hw).
        let (c, hw, k) = (2usize, 3usize, 3usize);
        let d = c * hw * hw; // 18
        let num_specs = 2usize;
        let mut state: u64 = 0x1357_2468_ABCD;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        // SHARED weights (same Arc across every domain → Arc::ptr_eq holds).
        let conv_w: Arc<[f32]> = (0..c * c * k * k)
            .map(|_| rng() * 0.3)
            .collect::<Vec<_>>()
            .into();
        let lin_w: Arc<[f32]> = (0..d * d).map(|_| rng() * 0.2).collect::<Vec<_>>().into();
        let seed_a: Vec<f32> = (0..num_specs * d).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: d,
        };

        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
        }
        let build = |dd: usize, conv_w: &Arc<[f32]>, lin_w: &Arc<[f32]>| -> Dom {
            let df = dd as f32;
            let conv = GpuCrownLayer::Conv2d {
                weight_col: conv_w.clone(),
                bias_expanded: None,
                out_channels: c,
                in_channels: c,
                kernel_h: k,
                kernel_w: k,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                out_h: hw,
                out_w: hw,
                in_h: hw,
                in_w: hw,
                cert_err: Default::default(),
            };
            let act = || GpuCrownLayer::Activation {
                lower_slope: vec![0.30 + 0.13 * df; d],
                upper_slope: vec![0.62 + 0.11 * df; d],
                lower_intercept: vec![0.02 * df; d],
                upper_intercept: vec![0.10 + 0.03 * df; d],
                num_neurons: d,
            };
            let lin = GpuCrownLayer::Linear {
                weight: lin_w.clone(),
                bias: None,
                out_features: d,
                in_features: d,
                cert_err: Default::default(),
            };
            Dom {
                // Backward order (output→input): Conv chain, then identity residual.
                segments: vec![
                    GpuResnetSegment::Chain(vec![conv, act()]),
                    GpuResnetSegment::Residual(vec![lin, act()]),
                ],
                in_lo: (0..d).map(|j| -1.0 - 0.2 * df - 0.03 * j as f32).collect(),
                in_hi: (0..d).map(|j| 1.0 + 0.2 * df + 0.03 * j as f32).collect(),
                // 2 ReLUs in fold order; distinct per domain.
                beta: vec![vec![0.05 * df; d], vec![0.03 * df; d]],
                // frontier_abs: one per SEGMENT (length d), distinct per domain.
                fa: vec![
                    (0..d).map(|j| 1.0 + 0.2 * df + 0.01 * j as f32).collect(),
                    (0..d).map(|j| 0.8 + 0.15 * df + 0.01 * j as f32).collect(),
                ],
                // node_abs: one per ReLU in fold order (length d), distinct per domain.
                na: vec![
                    (0..d).map(|_| 1.1 + 0.25 * df).collect(),
                    (0..d).map(|_| 0.9 + 0.18 * df).collect(),
                ],
            }
        };
        fn make_refs(doms: &[Dom]) -> Vec<GpuResnetBatchedDomainRef<'_>> {
            doms.iter()
                .map(|dd| GpuResnetBatchedDomainRef {
                    segments: &dd.segments,
                    input_lower: &dd.in_lo,
                    input_upper: &dd.in_hi,
                    beta_signed: &dd.beta,
                    frontier_abs: &dd.fa,
                    node_abs: &dd.na,
                })
                .collect()
        }

        let doms: Vec<Dom> = (0..3).map(|dd| build(dd, &conv_w, &lin_w)).collect();
        let refs = make_refs(&doms);

        // CHECK A (two-sided tol): the wide pass FIRES (n_domains>1), and each domain
        // block matches its serial per-domain bound within an f32 GEMM-reorder tol.
        let batched = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
            .expect("worded Conv residual batch");
        assert_eq!(batched.len(), doms.len());
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()));
        for (dd, dom) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_beta(
                    &dom.segments,
                    &seed,
                    &dom.in_lo,
                    &dom.in_hi,
                    &dom.beta,
                    &dom.fa,
                    &dom.na,
                )
                .expect("serial per-domain multi-segment");
            for s in 0..num_specs {
                assert!(
                    close(batched[dd].lower_bounds[s], serial.lower_bounds[s]),
                    "domain {dd} spec {s} LOWER: wide={} serial={} (fab dom-mis-index?)",
                    batched[dd].lower_bounds[s],
                    serial.lower_bounds[s]
                );
                assert!(
                    close(batched[dd].upper_bounds[s], serial.upper_bounds[s]),
                    "domain {dd} spec {s} UPPER: wide={} serial={}",
                    batched[dd].upper_bounds[s],
                    serial.upper_bounds[s]
                );
            }
        }

        // CONTAM (bit-exact wide-vs-wide): mutating ONLY domain 1's node_abs must leave
        // domains 0 and 2's blocks byte-unchanged (no cross-domain fab-table leak).
        let mut doms2: Vec<Dom> = (0..3).map(|dd| build(dd, &conv_w, &lin_w)).collect();
        for v in doms2[1].na[0].iter_mut() {
            *v += 0.5;
        }
        let refs2 = make_refs(&doms2);
        let batched2 = device
            .crown_backward_gpu_resnet_sound_beta_batched(&refs2, &seed)
            .expect("wide after domain-1 na mutation");
        assert_eq!(
            batched2[0].lower_bounds, batched[0].lower_bounds,
            "domain 0 block contaminated by domain 1's node_abs mutation"
        );
        assert_eq!(
            batched2[2].upper_bounds, batched[2].upper_bounds,
            "domain 2 block contaminated by domain 1's node_abs mutation"
        );
        assert_ne!(
            batched2[1].lower_bounds, batched[1].lower_bounds,
            "domain 1 node_abs mutation had no effect (HOLE 4 fold not exercised — fixture bug)"
        );
    }

    /// #batched-bab HOLE 8: a batch whose skeleton contains a dual-alpha ReLU (or a
    /// MaxPool2d) must be DECLINED by the batched entry (→ `Err` → serial fallback),
    /// because those backward shaders are not domain-block-indexed and a wide pass would
    /// broadcast domain 0's relaxation/routing (a false VERIFIED).
    #[test]
    fn crown_batched_wide_declines_dual_alpha_and_maxpool() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let (o, h, i) = (3usize, 4usize, 4usize);
        let num_specs = 1usize;
        let w1: Arc<[f32]> = (0..h * i)
            .map(|n| 0.01 * n as f32)
            .collect::<Vec<_>>()
            .into();
        let seed_a: Vec<f32> = (0..num_specs * o).map(|n| 0.1 * (n as f32 + 1.0)).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: o,
        };
        // Two domains sharing a skeleton that contains a dual-alpha ReLU.
        let dual = || GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: vec![0.5; h],
            cross_slope: vec![0.6; h],
            upper_neg_slope: vec![0.5; h],
            cross_intercept: vec![0.1; h],
            num_neurons: h,
        };
        let mk = |_d: usize| -> Vec<GpuResnetSegment> {
            vec![GpuResnetSegment::Chain(vec![
                GpuCrownLayer::Linear {
                    weight: (0..o * h)
                        .map(|n| 0.02 * n as f32)
                        .collect::<Vec<_>>()
                        .into(),
                    bias: None,
                    out_features: o,
                    in_features: h,
                    cert_err: Default::default(),
                },
                dual(),
                GpuCrownLayer::Linear {
                    weight: w1.clone(),
                    bias: None,
                    out_features: h,
                    in_features: i,
                    cert_err: Default::default(),
                },
            ])]
        };
        let s0 = mk(0);
        let s1 = mk(1);
        let (lo, hi) = (vec![-1.0f32; i], vec![1.0f32; i]);
        let (beta, fa, na): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) =
            (vec![vec![0.0; h]], vec![], vec![]);
        let refs = vec![
            GpuResnetBatchedDomainRef {
                segments: &s0,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &beta,
                frontier_abs: &fa,
                node_abs: &na,
            },
            GpuResnetBatchedDomainRef {
                segments: &s1,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &beta,
                frontier_abs: &fa,
                node_abs: &na,
            },
        ];
        assert!(
            device
                .crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
                .is_err(),
            "a dual-alpha batch must be declined (HOLE 8) so the caller uses the serial path"
        );
    }

    /// Dense Complete Clip gather regression: force the compute path just above
    /// [`LEGACY_BETA_GATHER_MAX_COPIES`] and prove it is a bit-exact strided copy
    /// of the pre-activation lower-A matrix. The final requested column is
    /// deliberately out of range and must retain the legacy `+0.0` behavior.
    #[test]
    fn crown_dense_strided_gather_is_bit_exact() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let num_specs = 65usize;
        let num_neurons = 64usize;
        let gather_cols: Vec<u32> = (0..=num_neurons as u32).collect();
        assert!(
            num_specs * gather_cols.len() > LEGACY_BETA_GATHER_MAX_COPIES,
            "fixture must exercise the compute gather"
        );

        let lower_a: Vec<f32> = (0..num_specs * num_neurons)
            .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 257.0)
            .collect();
        let seed = GpuCrownSeed {
            lower_a: lower_a.clone().into(),
            upper_a: lower_a.clone().into(),
            lower_b: vec![0.0; num_specs].into(),
            upper_b: vec![0.0; num_specs].into(),
            num_specs,
            current_dim: num_neurons,
        };
        let segments = vec![GpuResnetSegment::Chain(vec![GpuCrownLayer::Activation {
            lower_slope: vec![0.4; num_neurons],
            upper_slope: vec![0.7; num_neurons],
            lower_intercept: vec![0.0; num_neurons],
            upper_intercept: vec![0.0; num_neurons],
            num_neurons,
        }])];
        let input_lower = vec![-1.0; num_neurons];
        let input_upper = vec![1.0; num_neurons];
        let beta_signed = vec![vec![0.0; num_neurons]];
        let domain = GpuResnetBatchedDomainRef {
            segments: &segments,
            input_lower: &input_lower,
            input_upper: &input_upper,
            beta_signed: &beta_signed,
            frontier_abs: &[],
            node_abs: &[],
        };
        let (_bounds, _alpha_grads, gathered) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(
                &[domain],
                &seed,
                &[gather_cols.as_slice()],
                &[],
            )
            .expect("dense strided gather");
        assert_eq!(gathered.len(), 1);
        assert_eq!(gathered[0].len(), num_specs * gather_cols.len());
        for row in 0..num_specs {
            for slot in 0..gather_cols.len() {
                let actual = gathered[0][row * gather_cols.len() + slot];
                let expected = if slot < num_neurons {
                    lower_a[row * num_neurons + slot]
                } else {
                    0.0
                };
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "row {row} gather slot {slot}: actual={actual} expected={expected}"
                );
            }
        }
    }

    /// #batched-bab part A — the wide-GATHER differential oracle (step 3 of the wide
    /// β-opt plan). The wide-grad batched backward gathers A_lower at the per-ReLU UNION
    /// of all domains' split columns; each domain's OWN columns' values at its OWN rows
    /// (block d = rows [d*nsp,(d+1)*nsp)) must match the serial per-domain grad backward
    /// with THAT domain's columns, within f32 GEMM-reorder tol. Plus SUPERSET (every
    /// per-domain col ∈ the union) + CONTAM (mutating one domain's slopes leaves other
    /// domains' gather blocks byte-unchanged — no cross-domain gather leak).
    #[test]
    fn crown_batched_wide_grad_gather_matches_serial_per_domain() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // This oracle targets domain-block gather indexing, not affine transport.
        // Use an Activation-only topology so this oracle isolates domain-block
        // gather indexing from affine transport.
        let d = 18usize;
        let nsp = 2usize;
        let n_domains = 3usize;
        let n_relu = 2usize;
        let mut state: u64 = 0x9A5C_11EE_2024;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let seed_a: Vec<f32> = (0..nsp * d).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; nsp].into(),
            upper_b: vec![0.0f32; nsp].into(),
            num_specs: nsp,
            current_dim: d,
        };

        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
            gidx: Vec<Vec<u32>>, // per-ReLU split columns (fold order), DISTINCT per domain
        }
        let build = |dd: usize| -> Dom {
            let df = dd as f32;
            let act = |slot: f32| GpuCrownLayer::Activation {
                lower_slope: vec![0.30 + 0.13 * df + 0.04 * slot; d],
                upper_slope: vec![0.62 + 0.11 * df + 0.03 * slot; d],
                lower_intercept: vec![0.02 * df + 0.01 * slot; d],
                upper_intercept: vec![0.10 + 0.03 * df + 0.02 * slot; d],
                num_neurons: d,
            };
            let dd = dd as u32;
            let dm = d as u32;
            Dom {
                segments: vec![
                    GpuResnetSegment::Chain(vec![act(0.0)]),
                    GpuResnetSegment::Residual(vec![act(1.0)]),
                ],
                in_lo: (0..d).map(|j| -1.0 - 0.2 * df - 0.03 * j as f32).collect(),
                in_hi: (0..d).map(|j| 1.0 + 0.2 * df + 0.03 * j as f32).collect(),
                beta: vec![vec![0.05 * df; d], vec![0.03 * df; d]],
                fa: vec![
                    (0..d).map(|j| 1.0 + 0.2 * df + 0.01 * j as f32).collect(),
                    (0..d).map(|j| 0.8 + 0.15 * df + 0.01 * j as f32).collect(),
                ],
                na: vec![
                    (0..d).map(|_| 1.1 + 0.25 * df).collect(),
                    (0..d).map(|_| 0.9 + 0.18 * df).collect(),
                ],
                // Distinct, OVERLAPPING per-domain split columns → a non-trivial union.
                gidx: vec![
                    vec![dd % dm, (dd + 3) % dm, (dd + 6) % dm],
                    vec![(dd + 1) % dm, (dd + 4) % dm],
                ],
            }
        };

        let doms: Vec<Dom> = (0..n_domains).map(build).collect();
        let union_cols: Vec<Vec<u32>> = (0..n_relu)
            .map(|r| {
                let mut u: Vec<u32> = doms
                    .iter()
                    .flat_map(|dm| dm.gidx[r].iter().copied())
                    .collect();
                u.sort_unstable();
                u.dedup();
                u
            })
            .collect();
        let union_refs: Vec<&[u32]> = union_cols.iter().map(|v| v.as_slice()).collect();
        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dm| GpuResnetBatchedDomainRef {
                segments: &dm.segments,
                input_lower: &dm.in_lo,
                input_upper: &dm.in_hi,
                beta_signed: &dm.beta,
                frontier_abs: &dm.fa,
                node_abs: &dm.na,
            })
            .collect();

        let (wide_bounds, _alpha_grads, wide_gathers) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs, &seed, &union_refs, &[])
            .expect("wide grad batched");
        assert_eq!(wide_bounds.len(), n_domains);
        assert_eq!(wide_gathers.len(), n_relu);

        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()));
        for (dd, dm) in doms.iter().enumerate() {
            for r in 0..n_relu {
                for &col in &dm.gidx[r] {
                    assert!(
                        union_cols[r].contains(&col),
                        "dom {dd} relu {r} col {col} not in union"
                    );
                }
            }
            let serial = device
                .crown_backward_gpu_resnet_sound_beta_grad(
                    &dm.segments,
                    &seed,
                    &dm.in_lo,
                    &dm.in_hi,
                    &dm.beta,
                    &dm.gidx,
                    &dm.fa,
                    &dm.na,
                )
                .expect("serial grad per-domain");
            for s in 0..nsp {
                assert!(
                    close(wide_bounds[dd].lower_bounds[s], serial.lower_bounds[s])
                        && close(wide_bounds[dd].upper_bounds[s], serial.upper_bounds[s]),
                    "dom {dd} spec {s} BOUND parity: wide=[{},{}] serial=[{},{}]",
                    wide_bounds[dd].lower_bounds[s],
                    wide_bounds[dd].upper_bounds[s],
                    serial.lower_bounds[s],
                    serial.upper_bounds[s]
                );
            }
            // VALUE parity: wide_gathers[r][(dd*nsp+t)*U_r + upos] == serial[r][t*|gd|+p].
            for r in 0..n_relu {
                let ur = union_cols[r].len();
                let gd = &dm.gidx[r];
                assert_eq!(
                    serial.beta_gather[r].len(),
                    nsp * gd.len(),
                    "serial gather shape r{r}"
                );
                assert_eq!(
                    wide_gathers[r].len(),
                    n_domains * nsp * ur,
                    "wide gather shape r{r}"
                );
                for (p, &col) in gd.iter().enumerate() {
                    let upos = union_cols[r].iter().position(|&x| x == col).unwrap();
                    for t in 0..nsp {
                        let wv = wide_gathers[r][(dd * nsp + t) * ur + upos];
                        let sv = serial.beta_gather[r][t * gd.len() + p];
                        assert!(
                            close(wv, sv),
                            "dom {dd} relu {r} col {col} row {t} GATHER parity: wide={wv} serial={sv} (union/pos mis-map?)"
                        );
                    }
                }
            }
        }

        // CONTAM: mutate ONLY domain 1's slopes → domains 0 and 2's gather blocks byte-exact.
        let mut doms2: Vec<Dom> = (0..n_domains).map(build).collect();
        if let GpuResnetSegment::Chain(ls) = &mut doms2[1].segments[0] {
            if let GpuCrownLayer::Activation { lower_slope, .. } = &mut ls[0] {
                for s in lower_slope.iter_mut() {
                    *s += 0.3;
                }
            }
        }
        let refs2: Vec<GpuResnetBatchedDomainRef> = doms2
            .iter()
            .map(|dm| GpuResnetBatchedDomainRef {
                segments: &dm.segments,
                input_lower: &dm.in_lo,
                input_upper: &dm.in_hi,
                beta_signed: &dm.beta,
                frontier_abs: &dm.fa,
                node_abs: &dm.na,
            })
            .collect();
        let (_b2, _ag2, wg2) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs2, &seed, &union_refs, &[])
            .expect("wide grad after dom-1 mutation");
        for r in 0..n_relu {
            let ur = union_cols[r].len();
            for &dd in &[0usize, 2usize] {
                for i in (dd * nsp * ur)..((dd + 1) * nsp * ur) {
                    assert_eq!(
                        wg2[r][i], wide_gathers[r][i],
                        "dom {dd} relu {r} gather idx {i} contaminated by dom-1 slope mutation"
                    );
                }
            }
        }
    }

    /// #w4 wide α+β ascent oracle: the wide batched pass's per-domain ALPHA gradients
    /// must match the serial single-domain grad kernel (`crown_backward_gpu_resnet_
    /// sound_grad`) domain by domain, and a dom-1 mutation must not contaminate dom
    /// 0/2's gradients. β is ZERO here so the wide and serial (no-β) coefficient
    /// streams are identical — the parity leg then isolates exactly the domain-block
    /// indexing + per-domain row reduction this channel adds. Distinct per-domain
    /// slopes/bounds/pre_lower make cross-domain blending (the risk mode: grads
    /// batch-averaged across domains) fail loudly.
    #[test]
    fn crown_batched_wide_alpha_grads_match_serial_per_domain() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // Alpha-gradient parity needs two relaxation sites and distinct domain
        // blocks, not affine kernels. Activation-only Chain + Residual keeps the
        // fixture inside the admitted worded route.
        let d = 18usize;
        let nsp = 2usize;
        let n_domains = 3usize;
        let n_relu = 2usize;
        let mut state: u64 = 0xA1FA_57EE_2026;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let seed_a: Vec<f32> = (0..nsp * d).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; nsp].into(),
            upper_b: vec![0.0f32; nsp].into(),
            num_specs: nsp,
            current_dim: d,
        };
        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
            pl: Vec<Vec<f32>>, // per-ReLU pre-activation lower (stable masked 0), DISTINCT per domain
        }
        let build = |dd: usize| -> Dom {
            let df = dd as f32;
            let act = |slot: f32| GpuCrownLayer::Activation {
                lower_slope: vec![0.30 + 0.13 * df + 0.04 * slot; d],
                upper_slope: vec![0.62 + 0.11 * df + 0.03 * slot; d],
                lower_intercept: vec![0.02 * df + 0.01 * slot; d],
                upper_intercept: vec![0.10 + 0.03 * df + 0.02 * slot; d],
                num_neurons: d,
            };
            Dom {
                segments: vec![
                    GpuResnetSegment::Chain(vec![act(0.0)]),
                    GpuResnetSegment::Residual(vec![act(1.0)]),
                ],
                in_lo: (0..d).map(|j| -1.0 - 0.2 * df - 0.03 * j as f32).collect(),
                in_hi: (0..d).map(|j| 1.0 + 0.2 * df + 0.03 * j as f32).collect(),
                // β ZERO: the serial reference (`sound_grad`) folds no β, so parity
                // requires identical coefficient streams.
                beta: vec![vec![0.0; d], vec![0.0; d]],
                fa: vec![
                    (0..d).map(|j| 1.0 + 0.2 * df + 0.01 * j as f32).collect(),
                    (0..d).map(|j| 0.8 + 0.15 * df + 0.01 * j as f32).collect(),
                ],
                na: vec![
                    (0..d).map(|_| 1.1 + 0.25 * df).collect(),
                    (0..d).map(|_| 0.9 + 0.18 * df).collect(),
                ],
                // Mixed negative pre-lowers (unstable) with a few masked-stable zeros.
                pl: (0..n_relu)
                    .map(|r| {
                        (0..d)
                            .map(|j| {
                                if j % 5 == 4 {
                                    0.0 // stable-masked
                                } else {
                                    -(0.4 + 0.1 * df + 0.02 * (r + 1) as f32 + 0.01 * j as f32)
                                }
                            })
                            .collect()
                    })
                    .collect(),
            }
        };
        let doms: Vec<Dom> = (0..n_domains).map(build).collect();
        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dm| GpuResnetBatchedDomainRef {
                segments: &dm.segments,
                input_lower: &dm.in_lo,
                input_upper: &dm.in_hi,
                beta_signed: &dm.beta,
                frontier_abs: &dm.fa,
                node_abs: &dm.na,
            })
            .collect();
        let pl_refs: Vec<&[Vec<f32>]> = doms.iter().map(|dm| dm.pl.as_slice()).collect();

        let (wide_bounds, alpha_grads, _gathers) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs, &seed, &[], &pl_refs)
            .expect("wide alpha-grad batched");
        assert_eq!(wide_bounds.len(), n_domains);
        assert_eq!(
            alpha_grads.len(),
            n_relu,
            "one grad vec per ReLU (fold order)"
        );
        for r in 0..n_relu {
            assert_eq!(
                alpha_grads[r].len(),
                n_domains * d,
                "relu {r} grads domain-stacked"
            );
        }

        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()));
        // PARITY: each domain's wide grad block == the serial single-domain kernel.
        for (dd, dm) in doms.iter().enumerate() {
            let serial = device
                .crown_backward_gpu_resnet_sound_grad(
                    &dm.segments,
                    &seed,
                    &dm.in_lo,
                    &dm.in_hi,
                    &dm.pl,
                    &dm.fa,
                    &dm.na,
                )
                .expect("serial per-domain grad");
            assert_eq!(serial.relu_grads.len(), n_relu);
            // Bounds parity too (β=0 ⇒ the streams are identical up to merge policy).
            for s in 0..nsp {
                assert!(
                    close(serial.lower_bounds[s], wide_bounds[dd].lower_bounds[s]),
                    "dom {dd} lo[{s}]: serial {} vs wide {}",
                    serial.lower_bounds[s],
                    wide_bounds[dd].lower_bounds[s]
                );
            }
            for r in 0..n_relu {
                for i in 0..d {
                    let w = alpha_grads[r][dd * d + i];
                    let sg = serial.relu_grads[r][i];
                    assert!(
                        close(sg, w),
                        "dom {dd} relu {r} neuron {i}: serial grad {sg} vs wide {w}"
                    );
                }
            }
        }

        // CONTAMINATION: mutate dom 1's slopes AND pre_lower; doms 0/2 byte-identical.
        let mut doms2: Vec<Dom> = (0..n_domains).map(build).collect();
        for seg in doms2[1].segments.iter_mut() {
            let layers = match seg {
                GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => l,
                GpuResnetSegment::ResidualProj(f, _) => f,
            };
            for l in layers.iter_mut() {
                if let GpuCrownLayer::Activation { lower_slope, .. } = l {
                    for v in lower_slope.iter_mut() {
                        *v = (*v + 0.31).min(1.0);
                    }
                }
            }
        }
        for r in 0..n_relu {
            for v in doms2[1].pl[r].iter_mut() {
                *v *= 3.0;
            }
        }
        let refs2: Vec<GpuResnetBatchedDomainRef> = doms2
            .iter()
            .map(|dm| GpuResnetBatchedDomainRef {
                segments: &dm.segments,
                input_lower: &dm.in_lo,
                input_upper: &dm.in_hi,
                beta_signed: &dm.beta,
                frontier_abs: &dm.fa,
                node_abs: &dm.na,
            })
            .collect();
        let pl_refs2: Vec<&[Vec<f32>]> = doms2.iter().map(|dm| dm.pl.as_slice()).collect();
        let (_b2, ag2, _g2) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs2, &seed, &[], &pl_refs2)
            .expect("wide alpha-grad after dom-1 mutation");
        for r in 0..n_relu {
            for &dd in &[0usize, 2usize] {
                for i in 0..d {
                    assert_eq!(
                        ag2[r][dd * d + i],
                        alpha_grads[r][dd * d + i],
                        "dom {dd} relu {r} neuron {i} alpha grad contaminated by dom-1 mutation"
                    );
                }
            }
        }
    }

    /// #batched-vjp INC4 oracle: the batched exact point-VJP
    /// (`crown_point_vjp_batched`, one wide GPU pass over K restart domains)
    /// must match the SEQUENTIAL exact gradient
    /// (`GraphNetwork::attack_point_gradient`) per restart on a small conv
    /// chain net, ~1e-3 relative. PLUS a CONTAMINATION leg: mutating restart
    /// 1's masks must leave restarts 0/2's gradients byte-identical (each wide
    /// row folds against ITS OWN domain block — any cross-domain bleed fails
    /// loudly) while changing restart 1's.
    #[test]
    fn crown_point_vjp_batched_matches_sequential_exact_gradient() {
        use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
        use ny_propagate::{
            layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer},
            point_vjp_forward_masks, GraphNode, Layer,
        };
        use ny_tensor::BoundedTensor;

        let _g = gpu_test_serial_guard();
        let device = require_device();

        // input [1,4,4] → Conv2d(1→2, 3x3, pad 1) → ReLU → Flatten →
        // Linear(32→3) → ReLU → Linear(3→2). Two ReLU mask slots.
        let mut state: u64 = 0x0B47_C4ED_5EED;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut graph = ny_propagate::GraphNetwork::new();
        let kernel = Array4::from_shape_fn((2, 1, 3, 3), |_| rng()).into_dyn();
        let conv = Conv2dLayer::new(
            kernel,
            Some(Array1::from_vec(vec![0.05, -0.03])),
            (1, 1),
            (1, 1),
        )
        .expect("conv layer");
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["conv1".into()],
        ));
        graph.add_node(GraphNode::new(
            "flat",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["relu1".into()],
        ));
        let w1 = Array2::from_shape_fn((3, 32), |_| rng() * 0.5);
        graph.add_node(GraphNode::new(
            "lin1",
            Layer::Linear(
                LinearLayer::new(w1, Some(Array1::from_vec(vec![0.1, -0.2, 0.05]))).expect("lin1"),
            ),
            vec!["flat".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["lin1".into()],
        ));
        let w2 = Array2::from_shape_fn((2, 3), |_| rng());
        graph.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(w2, None).expect("lin2")),
            vec!["relu2".into()],
        ));
        graph.set_output("lin2");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0_f32),
        )
        .expect("input box");

        let plan = graph.build_point_vjp_batch_plan(&input).expect("wide plan");
        assert_eq!(plan.input_dim, 16);
        assert_eq!(plan.output_dim, 2);
        assert_eq!(plan.mask_positions.len(), 2, "two ReLU mask slots");

        // K=3 restart points; per-restart DIFFERING spec rows.
        let k_restarts = 3usize;
        let points: Vec<Vec<f32>> = (0..k_restarts)
            .map(|_| (0..plan.input_dim).map(|_| rng()).collect())
            .collect();
        let (masks, _outputs) =
            point_vjp_forward_masks(&plan, &points).expect("batched mask forward");
        let spec_rows_per: Vec<Vec<f32>> = vec![vec![1.0, -1.0], vec![-0.5, 2.0], vec![0.7, 0.3]];
        let spec_rows: Vec<f32> = spec_rows_per.iter().flatten().copied().collect();

        let grads = device
            .crown_point_vjp_batched(
                &plan.layers_backward,
                &plan.mask_positions,
                &masks,
                &spec_rows,
                plan.output_dim,
                plan.input_dim,
            )
            .expect("batched point VJP");
        assert_eq!(grads.len(), k_restarts);

        // Sequential oracle per restart: the exact point-Jacobian VJP.
        for kk in 0..k_restarts {
            let x = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), points[kk].clone()).expect("x");
            let row = Array2::from_shape_vec((1, 2), spec_rows_per[kk].clone()).expect("spec row");
            let reference = graph
                .attack_point_gradient(&x, &row, None, None)
                .expect("sequential gradient")
                .expect("in-fragment gradient");
            let reference: Vec<f32> = reference.iter().copied().collect();
            assert_eq!(grads[kk].len(), reference.len());
            for (i, (&b, &r)) in grads[kk].iter().zip(reference.iter()).enumerate() {
                let tol = 1e-3 * (1.0 + r.abs());
                assert!(
                    (b - r).abs() <= tol,
                    "restart {kk} grad[{i}]: batched={b} sequential={r}"
                );
            }
        }

        // CONTAMINATION leg: flip EVERY mask bit of restart 1's first ReLU slot.
        // Restarts 0/2 must stay byte-identical; restart 1 must change.
        let mut masks_mut = masks;
        for m in masks_mut[1][0].iter_mut() {
            *m = 1.0 - *m;
        }
        let grads_mut = device
            .crown_point_vjp_batched(
                &plan.layers_backward,
                &plan.mask_positions,
                &masks_mut,
                &spec_rows,
                plan.output_dim,
                plan.input_dim,
            )
            .expect("batched point VJP (mutated dom 1)");
        for &kk in &[0usize, 2usize] {
            for (i, (&a, &b)) in grads[kk].iter().zip(grads_mut[kk].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "restart {kk} grad[{i}] contaminated by restart 1's mask mutation"
                );
            }
        }
        assert!(
            grads[1]
                .iter()
                .zip(grads_mut[1].iter())
                .any(|(&a, &b)| a.to_bits() != b.to_bits()),
            "restart 1's gradient must respond to its own mask mutation"
        );
    }

    /// #batched-vjp-resnet oracle: the RESNET batched exact point-VJP
    /// (`crown_point_vjp_batched_resnet`, one wide GPU pass over K restart
    /// domains of a chain+Residual segment template) must match the SEQUENTIAL
    /// exact gradient (`GraphNetwork::attack_point_gradient`, which walks the
    /// residual DAG with the certified fan-in-sum accumulator) per restart on a
    /// small conv resnet, ~1e-3 relative. PLUS the same CONTAMINATION leg as
    /// the chain test (per-domain mask isolation across the residual fold).
    #[test]
    fn crown_point_vjp_batched_resnet_matches_sequential_exact_gradient() {
        use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
        use ny_propagate::{
            layers::{AddLayer, Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer},
            point_vjp_resnet_forward_masks, GraphNode, Layer,
        };
        use ny_tensor::BoundedTensor;

        let _g = gpu_test_serial_guard();
        // Keep AUTO armed; the Conv-capable wide helper must execute, and any
        // regression is a hard test failure rather than a vacuous fallback.
        let _taint_words_auto = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();

        // input [1,4,4] → conv1(1→2) → relu1 → [F: conv2(2→2) → relu2] →
        // add(relu2, relu1) → flatten → lin1(32→3) → relu3 → lin2(3→2).
        // One identity residual, three ReLU mask slots.
        let mut state: u64 = 0x0B47_0DA6_5EED;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut graph = ny_propagate::GraphNetwork::new();
        let k1 = Array4::from_shape_fn((2, 1, 3, 3), |_| rng() * 0.4).into_dyn();
        let conv1 = Conv2dLayer::new(
            k1,
            Some(Array1::from_vec(vec![0.05, -0.03])),
            (1, 1),
            (1, 1),
        )
        .expect("conv1");
        graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["conv1".into()],
        ));
        let k2 = Array4::from_shape_fn((2, 2, 3, 3), |_| rng() * 0.4).into_dyn();
        let conv2 = Conv2dLayer::new(
            k2,
            Some(Array1::from_vec(vec![-0.02, 0.04])),
            (1, 1),
            (1, 1),
        )
        .expect("conv2");
        graph.add_node(GraphNode::new(
            "conv2",
            Layer::Conv2d(conv2),
            vec!["relu1".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["conv2".into()],
        ));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["relu2".into(), "relu1".into()],
        ));
        graph.add_node(GraphNode::new(
            "flat",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["add".into()],
        ));
        let w1 = Array2::from_shape_fn((3, 32), |_| rng() * 0.5);
        graph.add_node(GraphNode::new(
            "lin1",
            Layer::Linear(
                LinearLayer::new(w1, Some(Array1::from_vec(vec![0.1, -0.2, 0.05]))).expect("lin1"),
            ),
            vec!["flat".into()],
        ));
        graph.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["lin1".into()],
        ));
        let w2 = Array2::from_shape_fn((2, 3), |_| rng());
        graph.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(LinearLayer::new(w2, None).expect("lin2")),
            vec!["relu3".into()],
        ));
        graph.set_output("lin2");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0_f32),
        )
        .expect("input box");

        let plan = graph
            .build_point_vjp_resnet_plan(&input)
            .expect("resnet wide plan");
        assert_eq!(plan.input_dim, 16);
        assert_eq!(plan.output_dim, 2);
        assert_eq!(plan.mask_flat_positions.len(), 3, "three ReLU mask slots");
        assert_eq!(plan.segments_backward.len(), 3, "Chain + Residual + Chain");

        // K=3 restart points; per-restart DIFFERING spec rows.
        let k_restarts = 3usize;
        let points: Vec<Vec<f32>> = (0..k_restarts)
            .map(|_| (0..plan.input_dim).map(|_| rng()).collect())
            .collect();
        let (masks, _outputs) =
            point_vjp_resnet_forward_masks(&plan, &points).expect("batched mask forward");
        let spec_rows_per: Vec<Vec<f32>> = vec![vec![1.0, -1.0], vec![-0.5, 2.0], vec![0.7, 0.3]];
        let spec_rows: Vec<f32> = spec_rows_per.iter().flatten().copied().collect();

        let grads = device
            .crown_point_vjp_batched_resnet(
                &plan.segments_backward,
                &plan.mask_flat_positions,
                &masks,
                &spec_rows,
                plan.output_dim,
                plan.input_dim,
            )
            .expect("batched Conv residual point VJP");
        assert_eq!(grads.len(), k_restarts);

        // Sequential oracle per restart: the exact point-Jacobian VJP through
        // the residual DAG (fan-in summation via the certified accumulator).
        for kk in 0..k_restarts {
            let x = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), points[kk].clone()).expect("x");
            let row = Array2::from_shape_vec((1, 2), spec_rows_per[kk].clone()).expect("spec row");
            let reference = graph
                .attack_point_gradient(&x, &row, None, None)
                .expect("sequential gradient")
                .expect("in-fragment gradient");
            let reference: Vec<f32> = reference.iter().copied().collect();
            assert_eq!(grads[kk].len(), reference.len());
            for (i, (&b, &r)) in grads[kk].iter().zip(reference.iter()).enumerate() {
                let tol = 1e-3 * (1.0 + r.abs());
                assert!(
                    (b - r).abs() <= tol,
                    "restart {kk} grad[{i}]: batched={b} sequential={r}"
                );
            }
        }

        // CONTAMINATION leg: flip EVERY mask bit of restart 1's residual-branch
        // ReLU slot (slot 1 = relu2, inside the Residual F branch). Restarts
        // 0/2 must stay byte-identical; restart 1 must change.
        let mut masks_mut = masks;
        for m in masks_mut[1][1].iter_mut() {
            *m = 1.0 - *m;
        }
        let grads_mut = device
            .crown_point_vjp_batched_resnet(
                &plan.segments_backward,
                &plan.mask_flat_positions,
                &masks_mut,
                &spec_rows,
                plan.output_dim,
                plan.input_dim,
            )
            .expect("batched resnet point VJP (mutated dom 1)");
        for &kk in &[0usize, 2usize] {
            for (i, (&a, &b)) in grads[kk].iter().zip(grads_mut[kk].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "restart {kk} grad[{i}] contaminated by restart 1's mask mutation"
                );
            }
        }
        assert!(
            grads[1]
                .iter()
                .zip(grads_mut[1].iter())
                .any(|(&a, &b)| a.to_bits() != b.to_bits()),
            "restart 1's gradient must respond to its own mask mutation"
        );
    }

    /// #wg-limit-subchunk VALUE-IDENTITY: the device-limit-safe domain sub-chunking in
    /// `try_wide_resnet_batched_grad` must produce BIT-IDENTICAL per-domain bounds,
    /// alpha-gradients, AND β-gathers to the single wide pass over the same domains.
    /// Proven by forcing the sub-chunk path (`NY_WIDE_MAX_STACKED_ROWS` capped small) and
    /// comparing to the un-capped single pass. This is the moat proof that a large
    /// `NY_MO_GPU_CHUNK` (honored by LOOPING) can never change a bound — the −0.976 vs
    /// −1.31 hole cannot recur, because the sub-chunk result equals the sound single-pass
    /// result exactly.
    #[test]
    fn crown_batched_wide_subchunk_is_bit_identical_to_single_pass() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // Clear any inherited cap so the "single pass" baseline truly runs unchunked.
        let _cap_clear = ScopedEnvVar::unset("NY_WIDE_MAX_STACKED_ROWS");

        // Subchunk stitching is independent of affine transport. Two Activation
        // sites preserve every bounds/gradient/gather channel while keeping the
        // fixture compact.
        let d = 18usize;
        let nsp = 2usize;
        let n_domains = 7usize; // odd, > cap, so groups are ragged (2,2,2,1)
        let n_relu = 2usize;
        let mut state: u64 = 0x5EED_2026_C0DE;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let seed_a: Vec<f32> = (0..nsp * d).map(|_| rng()).collect();
        let seed = GpuCrownSeed {
            lower_a: seed_a.clone().into(),
            upper_a: seed_a.into(),
            lower_b: vec![0.0f32; nsp].into(),
            upper_b: vec![0.0f32; nsp].into(),
            num_specs: nsp,
            current_dim: d,
        };
        struct Dom {
            segments: Vec<GpuResnetSegment>,
            in_lo: Vec<f32>,
            in_hi: Vec<f32>,
            beta: Vec<Vec<f32>>,
            fa: Vec<Vec<f32>>,
            na: Vec<Vec<f32>>,
            pl: Vec<Vec<f32>>,
        }
        // Distinct per-domain relaxation/box so a mis-stitched sub-chunk would diverge.
        let build = |dd: usize| -> Dom {
            let df = dd as f32;
            let act = |slot: f32| GpuCrownLayer::Activation {
                lower_slope: vec![0.28 + 0.06 * df + 0.03 * slot; d],
                upper_slope: vec![0.61 + 0.04 * df + 0.02 * slot; d],
                lower_intercept: vec![0.015 * df + 0.01 * slot; d],
                upper_intercept: vec![0.08 + 0.02 * df + 0.015 * slot; d],
                num_neurons: d,
            };
            Dom {
                segments: vec![
                    GpuResnetSegment::Chain(vec![act(0.0)]),
                    GpuResnetSegment::Residual(vec![act(1.0)]),
                ],
                in_lo: (0..d).map(|j| -1.0 - 0.2 * df - 0.03 * j as f32).collect(),
                in_hi: (0..d).map(|j| 1.0 + 0.2 * df + 0.03 * j as f32).collect(),
                // Non-zero signed β so the fold exercises the per-domain β dual too.
                beta: (0..n_relu)
                    .map(|r| {
                        (0..d)
                            .map(|j| 0.01 * (r + 1) as f32 * (j % 3) as f32)
                            .collect()
                    })
                    .collect(),
                fa: vec![
                    (0..d).map(|j| 1.0 + 0.2 * df + 0.01 * j as f32).collect(),
                    (0..d).map(|j| 0.8 + 0.15 * df + 0.01 * j as f32).collect(),
                ],
                na: vec![
                    (0..d).map(|_| 1.1 + 0.25 * df).collect(),
                    (0..d).map(|_| 0.9 + 0.18 * df).collect(),
                ],
                pl: (0..n_relu)
                    .map(|r| {
                        (0..d)
                            .map(|j| {
                                if j % 5 == 4 {
                                    0.0
                                } else {
                                    -(0.4 + 0.1 * df + 0.02 * (r + 1) as f32 + 0.01 * j as f32)
                                }
                            })
                            .collect()
                    })
                    .collect(),
            }
        };
        let doms: Vec<Dom> = (0..n_domains).map(build).collect();
        // Per-ReLU union gather columns (exercise the gather-concat path too).
        let union_cols: Vec<Vec<u32>> = vec![vec![0u32, 3, 7], vec![1u32, 4]];
        let ug: Vec<&[u32]> = union_cols.iter().map(|v| v.as_slice()).collect();

        let refs: Vec<GpuResnetBatchedDomainRef> = doms
            .iter()
            .map(|dm| GpuResnetBatchedDomainRef {
                segments: &dm.segments,
                input_lower: &dm.in_lo,
                input_upper: &dm.in_hi,
                beta_signed: &dm.beta,
                frontier_abs: &dm.fa,
                node_abs: &dm.na,
            })
            .collect();
        let pl_refs: Vec<&[Vec<f32>]> = doms.iter().map(|dm| dm.pl.as_slice()).collect();

        // (1) SINGLE wide pass (cap cleared ⇒ whole batch in one pass).
        let (bounds_single, grads_single, gathers_single) = device
            .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs, &seed, &ug, &pl_refs)
            .expect("single wide grad pass");
        assert_eq!(bounds_single.len(), n_domains);

        // (2) FORCE the sub-chunk path: cap stacked rows at 2*nsp ⇒ safe_domains=2 ⇒
        // groups (2,2,2,1). Same inputs, so per-domain outputs MUST be bit-identical.
        let (bounds_chunk, grads_chunk, gathers_chunk) = {
            let _cap = ScopedEnvVar::set("NY_WIDE_MAX_STACKED_ROWS", &(2 * nsp).to_string());
            device
                .crown_backward_gpu_resnet_sound_beta_batched_grad(&refs, &seed, &ug, &pl_refs)
                .expect("sub-chunked wide grad pass")
        };

        assert_eq!(bounds_chunk.len(), n_domains, "one result per domain");
        for dd in 0..n_domains {
            for s in 0..nsp {
                assert_eq!(
                    bounds_single[dd].lower_bounds[s].to_bits(),
                    bounds_chunk[dd].lower_bounds[s].to_bits(),
                    "dom {dd} lower[{s}]: single {} vs sub-chunk {} (BIT-IDENTITY required)",
                    bounds_single[dd].lower_bounds[s],
                    bounds_chunk[dd].lower_bounds[s]
                );
                assert_eq!(
                    bounds_single[dd].upper_bounds[s].to_bits(),
                    bounds_chunk[dd].upper_bounds[s].to_bits(),
                    "dom {dd} upper[{s}]: single vs sub-chunk"
                );
            }
        }
        // Advisory α-gradient channel: only the SHAPE is asserted. The gradient capture
        // is a reduction whose value depends on the stacked-batch width (a PRE-EXISTING
        // property of the wide fold, independent of this sub-chunking change — the α
        // ascent already re-derives a valid α every iteration, so a batch-width-dependent
        // gradient never affects soundness; the verdict-deciding BOUNDS above are
        // bit-identical). Asserting grad VALUE identity here would encode a property the
        // fold never had.
        assert_eq!(grads_single.len(), grads_chunk.len(), "same relu count");
        for r in 0..grads_single.len() {
            assert_eq!(
                grads_single[r].len(),
                grads_chunk[r].len(),
                "relu {r} grad block length (domain-stacked)"
            );
        }
        // Gathers are pure COPIES of the (bit-identical) coefficient stream — no
        // reduction — so they must be BIT-IDENTICAL across the grouping.
        assert_eq!(
            gathers_single.len(),
            gathers_chunk.len(),
            "same gather relu count"
        );
        for r in 0..gathers_single.len() {
            assert_eq!(
                gathers_single[r].len(),
                gathers_chunk[r].len(),
                "relu {r} gather length (N×U_r row-major)"
            );
            for (i, (a, b)) in gathers_single[r]
                .iter()
                .zip(gathers_chunk[r].iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "relu {r} gather[{i}]: single {a} vs sub-chunk {b}"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod u1_composed_sequence_integrity {
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};
    use wgpu::util::DeviceExt;

    /// CPU twin of `GEMM_F32_EFT_TWIN_SHADER`, replicating its op sequence
    /// EXACTLY — including the 16-wide k tiling and the zero-padded OOB taps,
    /// because the residual channel measures the sequence it executes.
    fn cpu_twin(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        const TILE: usize = 16;
        const F32_MIN_NORMAL: f32 = 1.1754944e-38;
        const FLOOR: f32 = 3.9443045e-31; // 2^-101
        let mut v = vec![0.0f32; m * n];
        let mut r = vec![0.0f32; m * n];
        let num_tiles = k.div_ceil(TILE);
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                let mut rsum = 0.0f32;
                for t in 0..num_tiles {
                    for kk in 0..TILE {
                        let idx = t * TILE + kk;
                        // Zero-padded exactly as the shader's tile loads do.
                        let av = if idx < k { a[row * k + idx] } else { 0.0 };
                        let wv = if idx < k { w[idx * n + col] } else { 0.0 };
                        let prod = av * wv;
                        let ep = av.mul_add(wv, -prod);
                        let mut eterm = ep.abs();
                        // Match the shader's operand-based guard: a nonzero exact
                        // product may round all the way to zero before this test.
                        if av != 0.0 && wv != 0.0 && prod.abs() < FLOOR {
                            eterm = F32_MIN_NORMAL;
                        }
                        let s = acc + prod;
                        let bb = (-1.0f32).mul_add(acc, s);
                        let sb = (-1.0f32).mul_add(bb, s);
                        let da = (-1.0f32).mul_add(sb, acc);
                        let db = (-1.0f32).mul_add(bb, prod);
                        let es = da + db;
                        rsum = rsum + eterm + es.abs();
                        acc = s;
                    }
                }
                v[row * n + col] = acc;
                r[row * n + col] = rsum;
            }
        }
        (v, r)
    }

    #[test]
    fn cpu_twin_charges_a_nonzero_product_that_rounds_to_zero() {
        let a = [f32::from_bits(1)]; // Smallest positive subnormal.
        let w = [0.5f32];
        assert_ne!(a[0], 0.0);
        assert_ne!(w[0], 0.0);
        assert_eq!(a[0] * w[0], 0.0, "the exact product rounds to zero");

        let (value, residual) = cpu_twin(&a, &w, 1, 1, 1);
        assert_eq!(value[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(
            residual[0].to_bits(),
            f32::MIN_POSITIVE.to_bits(),
            "the operand guard must charge the normal underflow floor"
        );
    }

    /// #s1 U1 — composed-sequence integrity in the PRODUCTION kernel.
    ///
    /// `METAL_EFT_VIABLE_2026-08-04.md` §5 lists this as the big undischarged
    /// obligation: every EFT probe is a `workgroup_size(1)` straight-line shader,
    /// while the shipping twin is a 16x16 tiled GEMM with `var<workgroup>` tiles
    /// and barriers. A passing probe does not prove the production kernel
    /// compiled with the same op sequence.
    ///
    /// Its stated settling test: per-element bit-compare of the twin's `(V, R)`
    /// against a CPU twin executing the identical sequence, at CROWN-shaped
    /// `(m, k, n)`. This is that test.
    ///
    /// BIT-compare, not approximate: an over-estimating `R` would also "enclose",
    /// so only bit-equality discriminates a correctly-compiled sequence from a
    /// reassociated one that happens to be conservative.
    #[test]
    fn eft_twin_matches_cpu_sequence_bitwise_at_crown_shapes() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        // CROWN-shaped, and deliberately including k not a multiple of TILE so
        // the zero-padded tail is exercised.
        for (m, k, n) in [(16usize, 16usize, 16usize), (32, 40, 24), (17, 65, 33)] {
            let mut st: u32 = 0x2F6E_5B11;
            let mut rng = || {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((st >> 8) as f32 / 8_388_608.0) - 1.0
            };
            let a: Vec<f32> = (0..m * k).map(|_| rng() * 0.7).collect();
            let w: Vec<f32> = (0..k * n).map(|_| rng() * 0.7).collect();
            let (cv, cr) = cpu_twin(&a, &w, m, k, n);

            let params = [m as u32, k as u32, n as u32, 0u32];
            let pbuf = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("u1_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let mk = |data: &[f32], rw: bool| {
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("u1_buf"),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | if rw {
                                wgpu::BufferUsages::COPY_SRC
                            } else {
                                wgpu::BufferUsages::empty()
                            },
                    })
            };
            let abuf = mk(&a, false);
            let wbuf = mk(&w, false);
            let vbuf = mk(&vec![0.0f32; m * n], true);
            let rbuf = mk(&vec![0.0f32; m * n], true);

            let pipe = device.create_simple_pipeline(
                crate::wgpu_device::shaders::GEMM_F32_EFT_TWIN_SHADER,
                "u1_eft_twin",
                &[false, false, true, true],
            );
            let mut enc = device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("u1") });
            device.pass_simple_2d(
                &mut enc,
                &pipe,
                &pbuf,
                &[&abuf, &wbuf, &vbuf, &rbuf],
                n.div_ceil(16) as u32,
                m.div_ceil(16) as u32,
            );
            let bytes = (m * n * 4) as u64;
            let stage = |lbl: &str| {
                device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(lbl),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let vstage = stage("u1_v_stage");
            let rstage = stage("u1_r_stage");
            enc.copy_buffer_to_buffer(&vbuf, 0, &vstage, 0, bytes);
            enc.copy_buffer_to_buffer(&rbuf, 0, &rstage, 0, bytes);
            device.queue.submit(std::iter::once(enc.finish()));
            let gv_bits = crate::WgpuDevice::read_u32_buffer(&device.device, &vstage, m * n)
                .expect("read twin V");
            let gr_bits = crate::WgpuDevice::read_u32_buffer(&device.device, &rstage, m * n)
                .expect("read twin R");

            let mut vbad = 0usize;
            let mut rbad = 0usize;
            for i in 0..m * n {
                if gv_bits[i] != cv[i].to_bits() {
                    vbad += 1;
                }
                if gr_bits[i] != cr[i].to_bits() {
                    rbad += 1;
                }
            }
            // DIRECTION is what decides whether an R mismatch matters. R bounds
            // |exact - V|, so a GPU R that is LARGER than the reference is merely
            // loose; one that is SMALLER is an under-charge, and under-charging is
            // the false-proof direction.
            let mut r_under = 0usize;
            let mut worst_under_rel = 0.0f32;
            let mut worst_over_rel = 0.0f32;
            for i in 0..m * n {
                let g = f32::from_bits(gr_bits[i]);
                let c = cr[i];
                if g < c {
                    r_under += 1;
                    worst_under_rel = worst_under_rel.max((c - g) / c.abs().max(f32::MIN_POSITIVE));
                } else if g > c {
                    worst_over_rel = worst_over_rel.max((g - c) / c.abs().max(f32::MIN_POSITIVE));
                }
            }
            eprintln!(
                "#u1 shape=({m},{k},{n}) V_mismatch={vbad}/{tot} R_mismatch={rbad}/{tot} \
                 R_UNDER={r_under} worst_under_rel={worst_under_rel:.3e} \
                 worst_over_rel={worst_over_rel:.3e}",
                tot = m * n
            );
            assert_eq!(
                vbad, 0,
                "({m},{k},{n}): twin V is not bit-identical to the CPU sequence — the \
                 tiled kernel compiled to a DIFFERENT op sequence than the probe measured"
            );
            // R's own f32 accumulation is a plain add chain, NOT fma-barriered, so
            // the compiler may reassociate it — which is what the mismatches above
            // are. `eft_r_slack_f32` recovers that with `1/(1 - gamma_{2k+2})`,
            // the Higham factor for a 2k+2-term non-negative f32 reduction.
            //
            // Compare against THAT, not against the function's `(1+u)^6` factor:
            // the `(1+u)^6` covers the MIN-COMBINE's own six f32 ops (the
            // |V-value| subtract/abs, the R+d add, the *r_slack multiply, the
            // prop*slack product, the cross add, the +flush) and has nothing to do
            // with the residual reduction. Checking the wrong term would pass for
            // the wrong reason.
            //
            // Computed in f64: in f32, `1.0 + 2^-24` rounds to exactly 1.0 (half an
            // ULP at 1.0), so the same expression there evaluates to 0.
            const U: f64 = 5.960_464_477_539_063e-8; // 2^-24, f32 unit roundoff
            let terms = (2 * k + 2) as f64;
            let gamma = terms * U / (1.0 - terms * U);
            let slack = (1.0 / (1.0 - gamma) - 1.0) as f32;
            assert!(
                worst_under_rel < slack,
                "({m},{k},{n}): R under-charges by {worst_under_rel:.3e} relative, EXCEEDING \
                 the 1/(1-gamma_{{2k+2}}) = {slack:.3e} recovery that eft_r_slack_f32 \
                 applies — the residual channel would publish a radius that does not enclose"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod u5_activation_lipschitz {
    //! #u5 — the LIPSCHITZ PROPAGATION SWAP (sound_authority.rs obligation U5).
    //!
    //! `NY_EFT_ERR=1` is not only an error-MEASUREMENT flag: in the activation
    //! kernels it also swaps the propagated coefficient-error factor from the
    //! conservative `|ls|+|us|` to the Lipschitz transport factor of the
    //! piecewise-linear activation map `v ↦ v·sel(v) ∓ β` — `|sel|` when the
    //! coefficient's sign is certain (`|a| > err_in`), `max(|ls|,|us|)`
    //! otherwise. That is an A-PRIORI claim, not an EFT measurement
    //! (`docs/METAL_EFT_VIABLE_2026-08-04.md` U5), and until this module it had
    //! NO dedicated adversarial oracle.
    //!
    //! # The soundness claim under test
    //!
    //! For every realization `a' ∈ [a − err_in, a + err_in]` of the incoming
    //! coefficient, the published error must cover the exact deviation of the
    //! activation map at `a'` from the shipped f32 coefficient:
    //!
    //! ```text
    //!   |g(a') − coeff| ≤ err_out,   g(t) = t·sel(t) ∓ β   (exact reals)
    //! ```
    //!
    //! `g` is continuous piecewise-linear with the only kink at `t = 0`
    //! (`g(0) = ∓β` from both sides), so on each linear piece `|g − coeff|` is
    //! convex and its supremum over the interval is attained at one of the
    //! piece endpoints — i.e. at `a − err_in`, `a + err_in`, or the kink `0`
    //! when the interval straddles it. The f64 oracle evaluates exactly those
    //! candidates; f32 products/f32-pair sums are exact in f64, so the only
    //! oracle noise is one f64 rounding per endpoint (≤ 2⁻⁵³ relative,
    //! absorbed by a 1e-9 allowance far below the kernel's own ×1.000001
    //! SLACK, so it cannot mask a real violation).
    //!
    //! Params are driven directly (`eft_mode` at the uniform level, the
    //! `u1_composed_sequence_integrity` / `tests/u1_tree_settling.rs`
    //! precedent): U5 is a claim about the KERNEL's error algebra, and this
    //! way BOTH modes are exercised on every adapter regardless of the
    //! authority-gate state. Production `additive` (`rung3_flush_safe_additive`)
    //! is used so the fma-subnormal-flush floor the shipped walk charges is
    //! part of what is validated (GB10: fma flushes subnormal RESULTS even
    //! under DenormPreserve — the floor is the cover for the measured
    //! `e_prod`/`e_sub` lanes at 2⁻¹²⁶-edge operands).
    // The certified-coefficient fixtures appended to this module (cert_err_*,
    // resnet_coeff_fixture and its projection twin) build real layer/segment
    // plans, which this module's original two imports did not cover.
    use crate::wgpu_device::test_support::{
        gpu_test_serial_guard, require_device, require_verdict_device,
    };
    use crate::WgpuDevice;
    use ny_core::dd::next_up_f64;
    use ny_core::f32_to_f64_exact;
    use ny_core::GpuCrownBackward;
    use ny_core::{GpuCrownLayer, GpuCrownSeed, GpuResnetSegment};
    use ny_test_utils::env::ScopedEnvVar;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    /// NaN payload pre-written into the read_write outputs: a silently no-op'd
    /// dispatch reads back as a mismatch, never as agreement (u1 discipline).
    const UNWRITTEN_SENTINEL: u32 = 0x7FC0_1234;

    /// Deterministic xorshift64* (idiom copied from the settling probes).
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in [0, 1).
        fn frac(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
        }
        fn sign(&mut self) -> f32 {
            if self.next_u64() & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        }
        /// `± 2^e · (1 + frac)` with `e` uniform in `[lo, hi]` (powi handles
        /// the subnormal-edge exponents exactly).
        fn banded(&mut self, lo: i32, hi: i32) -> f32 {
            let span = (hi - lo + 1) as u64;
            let e = lo + (self.next_u64() % span) as i32;
            self.sign() * 2.0f32.powi(e) * (1.0 + self.frac())
        }
    }

    /// Exact-f64 supremum of `|g(a') − coeff|` over `a' ∈ [a−e_in, a+e_in]`,
    /// `g(t) = t·sel(t) + sb2` with the shader's own branch convention
    /// (`t >= 0` selects the same slope the WGSL `select(..., a >= 0.0)` does)
    /// and `sb2 = +β` (upper) / `−β` (lower). Candidates: the two interval
    /// ends plus the kink at 0 when interior (see the module doc for why that
    /// set is exhaustive).
    #[allow(clippy::too_many_arguments)]
    fn worst_realization_sup(
        a: f32,
        e_in: f32,
        ls: f32,
        us: f32,
        bv: f32,
        is_upper: bool,
        coeff: f32,
    ) -> f64 {
        let (s_nonneg, s_neg) = if is_upper { (us, ls) } else { (ls, us) };
        let sb2 = if is_upper {
            f64::from(bv)
        } else {
            -f64::from(bv)
        };
        let c = f64::from(coeff);
        // Cancellation-safe endpoint deviations (review defect 1): the naive
        // `g(a ± e) − c` rounds the endpoint on the grid of |a| (error up to
        // 2^-53·|a| — relative to |a|, NOT to the deviation) whenever the
        // a/e exponent gap exceeds 29, overshooting the sup by up to ~3% in
        // the small-e/large-a regime and false-failing the eft arm (its only
        // margin is ×1.000001). Instead: `(a·s − c) + σ·e·s + sb2` — the
        // f32×f32 products are exact 48-bit f64 values, leaving ≤3 roundings
        // AT THE DEVIATION SCALE, which 1e-9 genuinely covers. The rounded
        // endpoint is still used for the SIGN branch only (f64 rounding never
        // crosses zero).
        let dev = |sigma: f64| -> f64 {
            let t = f64::from(a) + sigma * f64::from(e_in);
            let s = if t >= 0.0 {
                f64::from(s_nonneg)
            } else {
                f64::from(s_neg)
            };
            // Order matters: sb2 and c are the LARGE near-equal terms
            // (c ≈ fl(a·s + sb2)) — cancel them FIRST (Sterbenz-exact when
            // they cancel at all), then the exact small products join at the
            // deviation scale. `(a·s − c) + ...` would round the tiny a into
            // the grid of |c| and re-introduce exactly the defect-1 bug.
            (((sb2 - c) + f64::from(a) * s) + sigma * (f64::from(e_in) * s)).abs()
        };
        let lo = f64::from(a) - f64::from(e_in);
        let hi = f64::from(a) + f64::from(e_in);
        let mut sup = dev(-1.0).max(dev(1.0));
        if lo < 0.0 && hi > 0.0 {
            // g(0) = sb2 exactly.
            sup = sup.max((sb2 - c).abs());
        }
        sup
    }

    /// Dispatch the PRODUCTION `CROWN_ACTIVATION_RESIDENT_SHADER` (same
    /// `create_simple_pipeline` construction and rw-flag array as
    /// `resident_backward_pipelines().act`) and return `(a_out, err_out)` as
    /// raw bits.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_act(
        device: &WgpuDevice,
        params: super::ActParams,
        a: &[f32],
        err: &[f32],
        ls: &[f32],
        us: &[f32],
        beta: &[f32],
    ) -> (Vec<u32>, Vec<u32>) {
        let total = a.len();
        let pbuf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("u5_act_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mk_ro = |data: &[f32]| {
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("u5_act_ro"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let mk_rw = || {
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("u5_act_rw"),
                    contents: bytemuck::cast_slice(&vec![UNWRITTEN_SENTINEL; total]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        let (ab, eb, lb, ub, bb) = (mk_ro(a), mk_ro(err), mk_ro(ls), mk_ro(us), mk_ro(beta));
        let (aout, eout) = (mk_rw(), mk_rw());
        let pipe = device.create_simple_pipeline(
            crate::wgpu_device::shaders::CROWN_ACTIVATION_RESIDENT_SHADER,
            "u5_act_resident",
            &[false, false, false, false, true, true, false],
        );
        let mut enc = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("u5") });
        device.pass_simple(
            &mut enc,
            &pipe,
            &pbuf,
            &[&ab, &eb, &lb, &ub, &aout, &eout, &bb],
            (total as u32).div_ceil(256),
        );
        let bytes = (total * 4) as u64;
        let stage = |lbl: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(lbl),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (astage, estage) = (stage("u5_a_stage"), stage("u5_e_stage"));
        enc.copy_buffer_to_buffer(&aout, 0, &astage, 0, bytes);
        enc.copy_buffer_to_buffer(&eout, 0, &estage, 0, bytes);
        device.queue.submit(std::iter::once(enc.finish()));
        let a_bits =
            WgpuDevice::read_u32_buffer(&device.device, &astage, total).expect("u5: read a_out");
        let e_bits =
            WgpuDevice::read_u32_buffer(&device.device, &estage, total).expect("u5: read err_out");
        (a_bits, e_bits)
    }

    /// U5 device oracle for the COEFFICIENT activation kernel: adversarial
    /// bands (mixed signs, 2^-30..2^8 magnitudes plus 2^-126 subnormal edges,
    /// slopes in [0,1] incl. exact 0/1, β ≠ 0 arm, exact zeros, sign-certain
    /// AND sign-uncertain err bands), BOTH eft_mode settings, BOTH sides.
    /// Asserts per element: published `err_out` ≥ the exact worst-realization
    /// deviation; the VALUE lane is bit-identical across modes (the swap must
    /// touch only the error channel). Prints the eft/legacy tightness ratio
    /// (eft ≤ legacy on the stable-neuron majority is EXPECTED, printed, not
    /// asserted).
    #[test]
    fn act_eft_err_encloses_worst_realization() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        let num_specs = 4usize;
        let nn = 2048usize;
        let total = num_specs * nn;
        let mut rng = Rng(0x0055_EFAB_2026_0810);

        // Per-neuron slopes in [0,1] incl. EXACT 0 and 1, and β (≠0 on a third).
        let mut ls = vec![0.0f32; nn];
        let mut us = vec![0.0f32; nn];
        let mut beta = vec![0.0f32; nn];
        for i in 0..nn {
            let (l, u) = match i % 5 {
                0 => (0.0, 1.0),
                1 => (1.0, 1.0),
                2 => (0.0, 0.0),
                3 => (rng.frac(), 1.0),
                _ => (rng.frac(), rng.frac()),
            };
            ls[i] = l;
            us[i] = u;
            if i % 3 == 0 {
                beta[i] = rng.banded(-20, 3);
            }
        }
        // Per-element coefficients and incoming errors.
        let mut a = vec![0.0f32; total];
        let mut e_in = vec![0.0f32; total];
        for idx in 0..total {
            a[idx] = if rng.frac() < 0.01 {
                0.0 // ~1% exact zeros
            } else if idx % 97 == 0 {
                rng.banded(-129, -122) // subnormal / 2^-126 edge band
            } else {
                rng.banded(-30, 8)
            };
            e_in[idx] = match idx % 4 {
                0 => 0.0,
                1 => a[idx].abs() * 2.0f32.powi(-20) * (1.0 + rng.frac()), // sign certain
                2 => a[idx].abs() * (1.0 + rng.frac()) + 1e-30,            // sign UNCERTAIN
                _ => rng.banded(-40, 2).abs(),
            };
        }
        // Non-vacuity: every adversarial class must actually be present.
        assert!(a.contains(&0.0), "no exact-zero coefficients");
        assert!(
            a.iter()
                .any(|&v| v != 0.0 && v.abs() < f32::MIN_POSITIVE * 128.0),
            "no 2^-126-edge coefficients"
        );
        assert!(
            (0..total).any(|i| e_in[i] > a[i].abs()),
            "no sign-uncertain elements"
        );
        assert!((0..nn).any(|i| beta[i] != 0.0), "beta arm never exercised");

        let additive = crate::wgpu_device::sound_consts::rung3_flush_safe_additive(1)
            .expect("single-term U5 fixture has a representable rung-3 point count");
        let mk_params = |is_upper: u32, eft_mode: u32| super::ActParams {
            num_specs: num_specs as u32,
            num_neurons: nn as u32,
            is_upper,
            additive,
            num_specs_per_dom: num_specs as u32, // single domain
            eft_mode,
            _p: [0; 2],
        };

        for is_upper in [0u32, 1u32] {
            let (a_leg, e_leg) =
                dispatch_act(&device, mk_params(is_upper, 0), &a, &e_in, &ls, &us, &beta);
            let (a_eft, e_eft) =
                dispatch_act(&device, mk_params(is_upper, 1), &a, &e_in, &ls, &us, &beta);
            assert!(
                !a_leg
                    .iter()
                    .chain(&e_leg)
                    .chain(&a_eft)
                    .chain(&e_eft)
                    .any(|&b| b == UNWRITTEN_SENTINEL),
                "is_upper={is_upper}: an output element was never written (no-op dispatch)"
            );
            // The VALUE lane must be bit-identical across modes: the swap is an
            // error-channel-only change by construction.
            assert_eq!(
                a_leg, a_eft,
                "is_upper={is_upper}: eft_mode changed the VALUE lane of the \
                 activation kernel — the Lipschitz swap must only touch err_out"
            );

            let mut ratio_sum = 0.0f64;
            let mut ratio_n = 0usize;
            let mut eft_tighter = 0usize;
            let mut eft_looser = 0usize;
            let mut stable_ratio_sum = 0.0f64;
            let mut stable_n = 0usize;
            for idx in 0..total {
                let i = idx % nn;
                let coeff = f32::from_bits(a_leg[idx]);
                let sup = worst_realization_sup(
                    a[idx],
                    e_in[idx],
                    ls[i],
                    us[i],
                    beta[i],
                    is_upper == 1,
                    coeff,
                );
                for (mode, bits) in [("legacy", &e_leg), ("eft", &e_eft)] {
                    let e_out = f32::from_bits(bits[idx]);
                    assert!(
                        e_out.is_finite() && e_out >= 0.0,
                        "is_upper={is_upper} {mode} idx={idx}: err_out={e_out} not a \
                         finite nonnegative bound"
                    );
                    // 1e-9 relative allowance = the oracle's OWN f64 endpoint
                    // rounding only (see module doc); three orders below the
                    // kernel's ×1.000001 SLACK, so a real violation cannot hide.
                    assert!(
                        f64::from(e_out) >= sup * (1.0 - 1e-9),
                        "is_upper={is_upper} {mode} idx={idx}: err_out={e_out:e} DOES NOT \
                         ENCLOSE the worst realization {sup:e} \
                         (a={}, e_in={}, ls={}, us={}, beta={}, coeff={coeff})",
                        a[idx],
                        e_in[idx],
                        ls[i],
                        us[i],
                        beta[i],
                    );
                }
                let (l, f) = (
                    f64::from(f32::from_bits(e_leg[idx])),
                    f64::from(f32::from_bits(e_eft[idx])),
                );
                if f < l {
                    eft_tighter += 1;
                } else if f > l {
                    eft_looser += 1;
                }
                if f > 0.0 {
                    ratio_sum += l / f;
                    ratio_n += 1;
                    if a[idx].abs() > e_in[idx] {
                        stable_ratio_sum += l / f;
                        stable_n += 1;
                    }
                }
            }
            println!(
                "[u5-act] is_upper={is_upper} elements={total} eft_tighter={eft_tighter} \
                 eft_looser={eft_looser} mean legacy/eft ratio={:.4} \
                 (sign-certain subset: {:.4} over {stable_n})",
                ratio_sum / ratio_n.max(1) as f64,
                stable_ratio_sum / stable_n.max(1) as f64,
            );
        }
    }

    /// Neumaier-compensated f64 accumulation (u1_tree idiom): keeps the
    /// row-sum oracle's own noise at the 2^-104 class so a flat 1e-12 relative
    /// allowance suffices.
    #[inline]
    fn neumaier_add(sum: &mut f64, comp: &mut f64, term: f64) {
        let t = *sum + term;
        *comp += if sum.abs() >= term.abs() {
            (*sum - t) + term
        } else {
            (term - t) + *sum
        };
        *sum = t;
    }

    /// U5 device oracle for the INTERCEPT-BIAS kernel — the other half of the
    /// swap (`CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER`: Lipschitz factor of
    /// `v ↦ v·int(v)`, `max(|li|,|ui|)` / `|sel_int|`, replacing `|li|+|ui|`).
    /// Per spec row the published increment must satisfy, for EVERY joint
    /// realization `a'_j ∈ [a_j − e_j, a_j + e_j]`:
    /// `|Σ_j g_j(a'_j) − bias_out| ≤ bias_err_out`. The per-element intervals
    /// are independent, so the exact sup is
    /// `max(Σ_j max g_j − B, B − Σ_j min g_j)` with per-element min/max over
    /// the same 3-candidate set as the coefficient oracle. `nn = 1` isolates
    /// the per-element Lipschitz claim from the (U1-settled) tree; larger `nn`
    /// validates its composition with the reduction charges.
    #[test]
    fn act_intercept_bias_eft_err_encloses_worst_realization() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        for &nn in &[1usize, 7, 300] {
            let num_specs = 8usize;
            let total = num_specs * nn;
            let mut rng = Rng(0x0055_1B1A_5000_0000 ^ nn as u64);
            let mut li = vec![0.0f32; nn];
            let mut ui = vec![0.0f32; nn];
            for i in 0..nn {
                // Mixed-sign intercepts incl. exact zeros (the lower intercept
                // of a ReLU relaxation is 0 in production; keep that shape).
                li[i] = if i % 4 == 0 { 0.0 } else { rng.banded(-10, 4) };
                ui[i] = rng.banded(-10, 4).abs();
            }
            let mut a = vec![0.0f32; total];
            let mut e_in = vec![0.0f32; total];
            for idx in 0..total {
                a[idx] = if rng.frac() < 0.02 {
                    0.0
                } else {
                    rng.banded(-30, 6)
                };
                e_in[idx] = match idx % 3 {
                    0 => 0.0,
                    1 => a[idx].abs() * 2.0f32.powi(-18) * (1.0 + rng.frac()),
                    _ => a[idx].abs() * (1.0 + rng.frac()) + 1e-32, // uncertain
                };
            }

            let gamma = crate::wgpu_device::sound_consts::gamma_k_f32(nn).expect("gamma");
            let slack = crate::wgpu_device::sound_consts::combine_slack_f32(nn).expect("slack");
            let eft_slack =
                crate::wgpu_device::sound_consts::eft_r_slack_f32(nn).expect("eft slack");
            let additive = crate::wgpu_device::sound_consts::rung3_flush_safe_additive(
                u32::try_from(nn).unwrap(),
            )
            .expect("U5 activation-bias fixture has a representable rung-3 point count");

            for is_upper in [0u32, 1u32] {
                let mut per_mode: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
                for eft_mode in [0u32, 1u32] {
                    let params = super::ActBiasParams {
                        num_specs: num_specs as u32,
                        num_neurons: nn as u32,
                        is_upper,
                        // Production wiring: γ carries r_slack in EFT mode.
                        gamma_k: if eft_mode == 1 { eft_slack } else { gamma },
                        additive,
                        slack,
                        num_specs_per_dom: num_specs as u32,
                        eft_mode,
                    };
                    let pbuf =
                        device
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("u5_actbias_params"),
                                contents: bytemuck::bytes_of(&params),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });
                    let mk_ro = |data: &[f32]| {
                        device
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("u5_actbias_ro"),
                                contents: bytemuck::cast_slice(data),
                                usage: wgpu::BufferUsages::STORAGE,
                            })
                    };
                    // Read-modify-write outputs: zero preloads (the kernel `+=`s).
                    let mk_rw = || {
                        device
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("u5_actbias_rw"),
                                contents: bytemuck::cast_slice(&vec![0.0f32; num_specs]),
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            })
                    };
                    let (ab, eb, lb, ub) = (mk_ro(&a), mk_ro(&e_in), mk_ro(&li), mk_ro(&ui));
                    let (bout, berr) = (mk_rw(), mk_rw());
                    let pipe = device.create_simple_pipeline(
                        crate::wgpu_device::shaders::CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER,
                        "u5_act_intercept_bias",
                        &[false, false, false, false, true, true],
                    );
                    let mut enc =
                        device
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("u5_bias"),
                            });
                    // One workgroup per spec row — the production dispatch shape.
                    device.pass_simple(
                        &mut enc,
                        &pipe,
                        &pbuf,
                        &[&ab, &eb, &lb, &ub, &bout, &berr],
                        num_specs as u32,
                    );
                    let bytes = (num_specs * 4) as u64;
                    let stage = |lbl: &str| {
                        device.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(lbl),
                            size: bytes,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        })
                    };
                    let (bstage, estage) = (stage("u5_b_stage"), stage("u5_be_stage"));
                    enc.copy_buffer_to_buffer(&bout, 0, &bstage, 0, bytes);
                    enc.copy_buffer_to_buffer(&berr, 0, &estage, 0, bytes);
                    device.queue.submit(std::iter::once(enc.finish()));
                    let b_bits = WgpuDevice::read_u32_buffer(&device.device, &bstage, num_specs)
                        .expect("u5: read bias_out");
                    let e_bits = WgpuDevice::read_u32_buffer(&device.device, &estage, num_specs)
                        .expect("u5: read bias_err_out");
                    per_mode.push((
                        b_bits.iter().map(|&b| f32::from_bits(b)).collect(),
                        e_bits.iter().map(|&b| f32::from_bits(b)).collect(),
                    ));
                }

                for (mode_i, mode) in ["legacy", "eft"].iter().enumerate() {
                    let (bias_out, bias_err) = &per_mode[mode_i];
                    for s in 0..num_specs {
                        let (mut smax, mut cmax) = (0.0f64, 0.0f64);
                        let (mut smin, mut cmin) = (0.0f64, 0.0f64);
                        for j in 0..nn {
                            let idx = s * nn + j;
                            let av = a[idx];
                            let (s_nonneg, s_neg) = if is_upper == 1 {
                                (ui[j], li[j])
                            } else {
                                (li[j], ui[j])
                            };
                            let g = |t: f64| -> f64 {
                                let sl = if t >= 0.0 {
                                    f64::from(s_nonneg)
                                } else {
                                    f64::from(s_neg)
                                };
                                t * sl
                            };
                            let lo = f64::from(av) - f64::from(e_in[idx]);
                            let hi = f64::from(av) + f64::from(e_in[idx]);
                            let mut gmin = g(lo).min(g(hi));
                            let mut gmax = g(lo).max(g(hi));
                            if lo < 0.0 && hi > 0.0 {
                                gmin = gmin.min(0.0);
                                gmax = gmax.max(0.0);
                            }
                            neumaier_add(&mut smax, &mut cmax, gmax);
                            neumaier_add(&mut smin, &mut cmin, gmin);
                        }
                        let b = f64::from(bias_out[s]);
                        let sup = ((smax + cmax) - b).max(b - (smin + cmin)).max(0.0);
                        let e_out = f64::from(bias_err[s]);
                        assert!(
                            e_out.is_finite() && e_out >= 0.0,
                            "nn={nn} is_upper={is_upper} {mode} row {s}: bias_err_out={e_out}"
                        );
                        assert!(
                            e_out >= sup * (1.0 - 1e-12),
                            "nn={nn} is_upper={is_upper} {mode} row {s}: \
                             bias_err_out={e_out:e} DOES NOT ENCLOSE the worst joint \
                             realization {sup:e} (bias_out={b:e})"
                        );
                    }
                }
                let tighter = (0..num_specs)
                    .filter(|&s| per_mode[1].1[s] < per_mode[0].1[s])
                    .count();
                println!(
                    "[u5-actbias] nn={nn} is_upper={is_upper} rows={num_specs} \
                     eft_tighter={tighter} legacy_err[0]={:e} eft_err[0]={:e}",
                    per_mode[0].1[0], per_mode[1].1[0],
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // #cert-err / #cert-coeffs — the device-level soundness oracles
    // -----------------------------------------------------------------

    /// A single-Linear fixture whose published bound is exactly the range of
    /// `y = W x + b` over the input box, so an EXACT f64 oracle exists.
    struct CertErrFixture {
        weight: Vec<f32>,
        bias: Vec<f32>,
        out_features: usize,
        in_features: usize,
        input_lower: Vec<f32>,
        input_upper: Vec<f32>,
        spec: Vec<f32>,
    }

    fn cert_err_fixture() -> CertErrFixture {
        let (out_features, in_features) = (3usize, 4usize);
        // Mixed signs and magnitudes: sign-agnostic charge coverage.
        let weight = vec![
            0.75, -0.40, 0.25, 0.90, //
            -1.20, 0.60, -0.35, 0.15, //
            0.05, 0.80, 1.10, -0.70,
        ];
        let bias = vec![0.20f32, -0.35, 0.05];
        let input_lower = vec![-1.0f32, -0.5, 0.0, -0.25];
        let input_upper = vec![1.0f32, 0.5, 0.75, 0.25];
        let mut spec = vec![0.0f32; out_features * out_features];
        for r in 0..out_features {
            spec[r * out_features + r] = 1.0;
        }
        CertErrFixture {
            weight,
            bias,
            out_features,
            in_features,
            input_lower,
            input_upper,
            spec,
        }
    }

    fn cert_err_layers(
        fx: &CertErrFixture,
        cert_err: ny_core::CertifiedWeightError,
    ) -> Vec<GpuCrownLayer> {
        vec![GpuCrownLayer::Linear {
            weight: Arc::from(fx.weight.clone().into_boxed_slice()),
            bias: Some(Arc::from(fx.bias.clone().into_boxed_slice())),
            out_features: fx.out_features,
            in_features: fx.in_features,
            cert_err,
        }]
    }

    /// The EXACT extreme of `y_i = Σ_j w*_ij x_j + b*_i` over BOTH the input box
    /// and the declared weight/bias band, computed in f64 by enumerating the four
    /// corners of each `(w*, x)` product interval. This is the truth the
    /// published bound must enclose — not an approximation of it.
    fn cert_err_true_extremes(
        fx: &CertErrFixture,
        w_rel: f64,
        bias_abs: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut lo = Vec::with_capacity(fx.out_features);
        let mut hi = Vec::with_capacity(fx.out_features);
        for i in 0..fx.out_features {
            let mut row_lo = f64::from(fx.bias[i]) - bias_abs;
            let mut row_hi = f64::from(fx.bias[i]) + bias_abs;
            for j in 0..fx.in_features {
                let w = f64::from(fx.weight[i * fx.in_features + j]);
                let (wl, wh) = (w - w_rel * w.abs(), w + w_rel * w.abs());
                let (xl, xu) = (f64::from(fx.input_lower[j]), f64::from(fx.input_upper[j]));
                let corners = [wl * xl, wl * xu, wh * xl, wh * xu];
                row_lo += corners.iter().copied().fold(f64::INFINITY, f64::min);
                row_hi += corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }
            lo.push(row_lo);
            hi.push(row_hi);
        }
        (lo, hi)
    }

    /// THE SOUNDNESS ORACLE for the certified weight-error charge.
    ///
    /// Declaring `weight_rel_err`/`bias_abs_err` asserts that the SUPPLIED
    /// weights are only an approximation of an exact real fold. This test takes
    /// that assertion literally: it computes, in exact f64, the true output range
    /// over every weight inside the declared band (plus a set of concrete
    /// samples), and asserts the published GPU bound encloses all of it. It also
    /// asserts the charge is (a) strictly widening and (b) NECESSARY — the
    /// uncharged bound demonstrably fails to enclose the same oracle, so the test
    /// cannot pass vacuously.
    #[test]
    fn cert_err_widens_and_encloses_the_perturbed_weight_oracle() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let fx = cert_err_fixture();

        let (w_rel, bias_abs) = (1e-3f32, 1e-3f32);
        let exact_layers = cert_err_layers(&fx, ny_core::CertifiedWeightError::default());
        let charged_layers = cert_err_layers(
            &fx,
            ny_core::CertifiedWeightError {
                weight_rel_err: w_rel,
                bias_abs_err: bias_abs,
            },
        );

        let (exact_lo, exact_hi) = device
            .crown_backward_sound_resident(
                &exact_layers,
                &fx.spec,
                fx.out_features,
                fx.out_features,
                &fx.input_lower,
                &fx.input_upper,
            )
            .expect("exact-weight resident walk");
        let (charged_lo, charged_hi) = device
            .crown_backward_sound_resident(
                &charged_layers,
                &fx.spec,
                fx.out_features,
                fx.out_features,
                &fx.input_lower,
                &fx.input_upper,
            )
            .expect("charged resident walk");

        let (true_lo, true_hi) = cert_err_true_extremes(&fx, f64::from(w_rel), f64::from(bias_abs));

        for i in 0..fx.out_features {
            // (1) The charge is strictly OUTWARD on both sides.
            assert!(
                charged_lo[i] < exact_lo[i] && charged_hi[i] > exact_hi[i],
                "row {i}: charged bound [{}, {}] must be strictly wider than the \
                 exact-weight bound [{}, {}]",
                charged_lo[i],
                charged_hi[i],
                exact_lo[i],
                exact_hi[i]
            );
            // (2) The charged bound ENCLOSES the true range over the whole band.
            assert!(
                f64::from(charged_lo[i]) <= true_lo[i],
                "row {i}: charged lower {} does NOT enclose the band minimum {} \
                 — a bound below the truth is a false proof",
                charged_lo[i],
                true_lo[i]
            );
            assert!(
                f64::from(charged_hi[i]) >= true_hi[i],
                "row {i}: charged upper {} does NOT enclose the band maximum {}",
                charged_hi[i],
                true_hi[i]
            );
        }

        // (3) TEETH: without the charge the very same oracle is violated, so the
        // enclosure above is not an artifact of slack the walk already had.
        let uncharged_fails = (0..fx.out_features)
            .any(|i| f64::from(exact_lo[i]) > true_lo[i] || f64::from(exact_hi[i]) < true_hi[i]);
        assert!(
            uncharged_fails,
            "the uncharged bound already enclosed the perturbed-weight oracle — \
             this fixture cannot detect a missing charge; widen w_rel"
        );

        // (4) Concrete samples inside the band, evaluated exactly.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next_unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
        };
        for sample in 0..32 {
            let perturbed: Vec<f64> = fx
                .weight
                .iter()
                .map(|&w| {
                    let w = f64::from(w);
                    w + next_unit() * f64::from(w_rel) * w.abs()
                })
                .collect();
            let perturbed_bias: Vec<f64> = fx
                .bias
                .iter()
                .map(|&b| f64::from(b) + next_unit() * f64::from(bias_abs))
                .collect();
            for i in 0..fx.out_features {
                let mut lo = perturbed_bias[i];
                let mut hi = perturbed_bias[i];
                for j in 0..fx.in_features {
                    let w = perturbed[i * fx.in_features + j];
                    let (xl, xu) = (f64::from(fx.input_lower[j]), f64::from(fx.input_upper[j]));
                    lo += (w * xl).min(w * xu);
                    hi += (w * xl).max(w * xu);
                }
                assert!(
                    f64::from(charged_lo[i]) <= lo && f64::from(charged_hi[i]) >= hi,
                    "sample {sample} row {i}: published [{}, {}] does not enclose \
                     the sampled true range [{lo}, {hi}]",
                    charged_lo[i],
                    charged_hi[i]
                );
            }
        }
    }

    /// Host-side OUTWARD concretization of a published `CertifiedCoeffs`
    /// frontier over one input box. Deliberately independent of the device
    /// concretize so the self-consistency check below is a real cross-check.
    fn concretize_certified_coeffs(
        c: &ny_core::CertifiedCoeffs,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let next_down = |x: f64| -next_up_f64(-x);
        let mut lower = Vec::with_capacity(c.num_specs);
        let mut upper = Vec::with_capacity(c.num_specs);
        for s in 0..c.num_specs {
            let mut lo =
                next_down(f32_to_f64_exact(c.lower_b[s]) - f32_to_f64_exact(c.lower_b_err[s]));
            let mut hi =
                next_up_f64(f32_to_f64_exact(c.upper_b[s]) + f32_to_f64_exact(c.upper_b_err[s]));
            for j in 0..c.dim {
                let idx = s * c.dim + j;
                let (xl, xu) = (
                    f32_to_f64_exact(input_lower[j]),
                    f32_to_f64_exact(input_upper[j]),
                );
                let (la, le) = (
                    f32_to_f64_exact(c.lower_a[idx]),
                    f32_to_f64_exact(c.lower_a_err[idx]),
                );
                let (ua, ue) = (
                    f32_to_f64_exact(c.upper_a[idx]),
                    f32_to_f64_exact(c.upper_a_err[idx]),
                );
                let lprod = [
                    (la - le) * xl,
                    (la - le) * xu,
                    (la + le) * xl,
                    (la + le) * xu,
                ]
                .into_iter()
                .fold(f64::INFINITY, f64::min);
                let uprod = [
                    (ua - ue) * xl,
                    (ua - ue) * xu,
                    (ua + ue) * xl,
                    (ua + ue) * xu,
                ]
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
                lo = next_down(lo + lprod);
                hi = next_up_f64(hi + uprod);
            }
            lower.push(lo);
            upper.push(hi);
        }
        (lower, upper)
    }

    /// The coefficient egress must be the SAME walk as the bounds entry, merely
    /// stopped one step earlier: concretizing what it publishes has to reproduce
    /// the bounds entry's answer. It must also actually publish COEFFICIENTS —
    /// `dim` is the input width, not a concretized scalar per row.
    #[test]
    fn certified_coeffs_entry_concretizes_to_the_bounds_entry() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let fx = cert_err_fixture();
        let layers = cert_err_layers(&fx, ny_core::CertifiedWeightError::default());
        let seed = GpuCrownSeed {
            lower_a: Arc::from(fx.spec.clone().into_boxed_slice()),
            upper_a: Arc::from(fx.spec.clone().into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32; fx.out_features].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32; fx.out_features].into_boxed_slice()),
            num_specs: fx.out_features,
            current_dim: fx.out_features,
        };

        let coeffs = device
            .crown_backward_gpu_seeded_sound_coeffs(
                &layers,
                &seed,
                &fx.input_lower,
                &fx.input_upper,
            )
            .expect("coefficient egress must not error on a qualified device")
            .expect("a qualified device must publish the frontier");

        assert_eq!(coeffs.num_specs, fx.out_features);
        assert_eq!(
            coeffs.dim, fx.in_features,
            "the entry must publish INPUT-dim coefficients, not concretized rows"
        );
        for field in [
            &coeffs.lower_a,
            &coeffs.upper_a,
            &coeffs.lower_a_err,
            &coeffs.upper_a_err,
        ] {
            assert_eq!(field.len(), coeffs.num_specs * coeffs.dim);
        }
        for field in [
            &coeffs.lower_b,
            &coeffs.upper_b,
            &coeffs.lower_b_err,
            &coeffs.upper_b_err,
        ] {
            assert_eq!(field.len(), coeffs.num_specs);
        }
        assert!(
            coeffs
                .lower_a_err
                .iter()
                .chain(coeffs.upper_a_err.iter())
                .chain(coeffs.lower_b_err.iter())
                .chain(coeffs.upper_b_err.iter())
                .all(|v| v.is_finite() && *v >= 0.0),
            "published radii must be finite and non-negative"
        );

        let bounds = device
            .crown_backward_gpu_seeded_sound(&layers, &seed, &fx.input_lower, &fx.input_upper)
            .expect("bounds entry");
        let (host_lo, host_hi) =
            concretize_certified_coeffs(&coeffs, &fx.input_lower, &fx.input_upper);

        for s in 0..coeffs.num_specs {
            let (gl, gu) = (
                f64::from(bounds.lower_bounds[s]),
                f64::from(bounds.upper_bounds[s]),
            );
            let tol = 1e-4 * (1.0 + gl.abs().max(gu.abs()));
            assert!(
                (host_lo[s] - gl).abs() <= tol && (host_hi[s] - gu).abs() <= tol,
                "row {s}: concretizing the published frontier gives [{}, {}] but \
                 the bounds entry says [{gl}, {gu}] — the two entries disagree",
                host_lo[s],
                host_hi[s]
            );
            assert!(host_lo[s] <= host_hi[s], "row {s}: inverted frontier");
        }
    }

    /// Publishing coefficients is MORE authority than publishing the bound
    /// derived from them, so the egress must move in lockstep with the sound
    /// CROWN authority gate: no authority, no frontier.
    #[test]
    fn certified_coeffs_entry_publishes_only_with_authority() {
        let _guard = gpu_test_serial_guard();
        let device = require_device();
        let fx = cert_err_fixture();
        let layers = cert_err_layers(&fx, ny_core::CertifiedWeightError::default());
        let seed = GpuCrownSeed {
            lower_a: Arc::from(fx.spec.clone().into_boxed_slice()),
            upper_a: Arc::from(fx.spec.clone().into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32; fx.out_features].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32; fx.out_features].into_boxed_slice()),
            num_specs: fx.out_features,
            current_dim: fx.out_features,
        };
        let published = device
            .crown_backward_gpu_seeded_sound_coeffs(
                &layers,
                &seed,
                &fx.input_lower,
                &fx.input_upper,
            )
            .expect("the egress declines by returning Ok(None), never by erroring");
        assert_eq!(
            published.is_some(),
            GpuCrownBackward::provides_sound_gpu_crown(&*device),
            "the coefficient egress and the sound-CROWN authority gate must \
             move together"
        );
    }

    // -----------------------------------------------------------------------
    // #cert-coeffs-resnet: the SEGMENT coefficient egress
    // -----------------------------------------------------------------------

    /// A three-segment residual fixture with CHANGING widths, so a frontier that
    /// stopped at any single segment is detectable by its `dim` alone:
    /// `3 -(Chain)-> 6 -(Residual, identity skip)-> 6 -(Chain)-> 4`.
    #[allow(clippy::type_complexity)]
    fn resnet_coeff_fixture() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Vec<f32>, Vec<f32>) {
        let (num_specs, seed_dim, mid, in_dim) = (3usize, 3usize, 6usize, 4usize);
        let mk = |n: usize, f: fn(usize) -> f32| -> Arc<[f32]> {
            (0..n).map(f).collect::<Vec<f32>>().into()
        };
        // Backward-order layers. `Linear { out_features, in_features }` maps an
        // (specs x out_features) frontier to (specs x in_features).
        let head = GpuCrownLayer::Linear {
            weight: mk(seed_dim * mid, |i| {
                0.4 - 0.13 * ((i % 7) as f32) + 0.05 * ((i % 3) as f32)
            }),
            bias: Some(mk(seed_dim, |i| 0.1 - 0.07 * (i as f32))),
            out_features: seed_dim,
            in_features: mid,
            cert_err: ny_core::CertifiedWeightError::default(),
        };
        let act = |scale: f32| GpuCrownLayer::Activation {
            lower_slope: (0..mid).map(|j| 0.3 + 0.05 * (j as f32)).collect(),
            upper_slope: (0..mid).map(|j| 0.6 + 0.04 * (j as f32)).collect(),
            lower_intercept: vec![0.0; mid],
            upper_intercept: (0..mid)
                .map(|j| scale * (0.1 + 0.02 * (j as f32)))
                .collect(),
            num_neurons: mid,
        };
        // The residual branch must map the block dim back to itself (6 -> 6).
        let branch = GpuCrownLayer::Linear {
            weight: mk(mid * mid, |i| 0.25 - 0.09 * ((i % 5) as f32)),
            bias: Some(mk(mid, |i| 0.05 * (i as f32) - 0.1)),
            out_features: mid,
            in_features: mid,
            cert_err: ny_core::CertifiedWeightError::default(),
        };
        let tail = GpuCrownLayer::Linear {
            weight: mk(mid * in_dim, |i| 0.5 - 0.11 * ((i % 4) as f32)),
            bias: Some(mk(mid, |i| 0.02 * (i as f32))),
            out_features: mid,
            in_features: in_dim,
            cert_err: ny_core::CertifiedWeightError::default(),
        };
        let segments = vec![
            GpuResnetSegment::Chain(vec![head, act(1.0)]),
            GpuResnetSegment::Residual(vec![branch, act(0.7)]),
            GpuResnetSegment::Chain(vec![tail]),
        ];
        let mut spec = vec![0.0f32; num_specs * seed_dim];
        for r in 0..num_specs {
            spec[r * seed_dim + r] = 1.0;
        }
        let seed = GpuCrownSeed {
            lower_a: Arc::from(spec.clone().into_boxed_slice()),
            upper_a: Arc::from(spec.into_boxed_slice()),
            lower_b: Arc::from(vec![0.0f32; num_specs].into_boxed_slice()),
            upper_b: Arc::from(vec![0.0f32; num_specs].into_boxed_slice()),
            num_specs,
            current_dim: seed_dim,
        };
        let input_lower: Vec<f32> = (0..in_dim).map(|j| -0.5 - 0.1 * (j as f32)).collect();
        let input_upper: Vec<f32> = (0..in_dim).map(|j| 0.4 + 0.1 * (j as f32)).collect();
        (segments, seed, input_lower, input_upper)
    }

    /// #margin-row-gpu-batch: the BATCHED coefficient egress must publish, for
    /// every slot, exactly what the SINGLE-DOMAIN egress publishes for that
    /// slot's own domain.
    ///
    /// The two domains here differ ONLY in their `Activation` relaxation (the
    /// only thing that varies per domain in the margin-row lane) and are
    /// deliberately DISTINGUISHABLE, so a wide fold that mixed the blocks — or
    /// a split that mis-cut the domain-major rows — moves at least one slot off
    /// its reference. That is the failure this test exists for; the shapes
    /// alone would agree either way.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn resnet_batched_certified_coeffs_match_the_single_domain_egress() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (segs_a, seed, lo, hi) = resnet_coeff_fixture();
        // Domain B: same skeleton, same weights (value-equal `Arc`s, which the
        // homogeneity gate accepts), DIFFERENT relaxation.
        let segs_b: Vec<GpuResnetSegment> = segs_a
            .iter()
            .map(|seg| {
                let bump = |layers: &Vec<GpuCrownLayer>| -> Vec<GpuCrownLayer> {
                    layers
                        .iter()
                        .map(|l| match l {
                            GpuCrownLayer::Activation {
                                lower_slope,
                                upper_slope,
                                lower_intercept,
                                upper_intercept,
                                num_neurons,
                            } => GpuCrownLayer::Activation {
                                // A genuinely different, still valid, relaxation.
                                lower_slope: lower_slope.iter().map(|v| v * 0.5).collect(),
                                upper_slope: upper_slope.iter().map(|v| v * 1.3).collect(),
                                lower_intercept: lower_intercept.clone(),
                                upper_intercept: upper_intercept.iter().map(|v| v + 0.25).collect(),
                                num_neurons: *num_neurons,
                            },
                            other => other.clone(),
                        })
                        .collect()
                };
                match seg {
                    GpuResnetSegment::Chain(l) => GpuResnetSegment::Chain(bump(l)),
                    GpuResnetSegment::Residual(l) => GpuResnetSegment::Residual(bump(l)),
                    GpuResnetSegment::ResidualProj(f, p) => {
                        GpuResnetSegment::ResidualProj(bump(f), bump(p))
                    }
                }
            })
            .collect();

        let single = |segs: &[GpuResnetSegment]| {
            device
                .crown_backward_gpu_resnet_sound_coeffs(segs, &seed, &lo, &hi, &[], &[])
                .expect("the single-domain egress must run")
                .expect("the single-domain egress must publish on an armed adapter")
        };
        let want_a = single(&segs_a);
        let want_b = single(&segs_b);
        // The premise: without this, a slot error would be invisible.
        assert!(
            want_a
                .lower_b
                .iter()
                .zip(&want_b.lower_b)
                .any(|(x, y)| (x - y).abs() > 1e-4),
            "the two domains must be distinguishable for this pin to mean anything"
        );

        let domains = vec![
            ny_core::GpuResnetBatchedDomainRef {
                segments: &segs_a,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &[],
                frontier_abs: &[],
                node_abs: &[],
            },
            ny_core::GpuResnetBatchedDomainRef {
                segments: &segs_b,
                input_lower: &lo,
                input_upper: &hi,
                beta_signed: &[],
                frontier_abs: &[],
                node_abs: &[],
            },
        ];
        let got = device
            .crown_backward_gpu_resnet_sound_batched_coeffs(&domains, &seed)
            .expect("the batched egress must run")
            .expect("the batched egress must publish on an armed adapter");
        assert_eq!(got.len(), 2, "one frontier per domain, in domain order");
        // Wide vs serial differ only by f32 GEMM accumulation order; both are
        // independently certified enclosures, so this is a match with a
        // documented tolerance, never a bit-equality.
        let tol = |x: f32, y: f32| {
            (f64::from(x) - f64::from(y)).abs()
                <= 1e-6 + 1e-3 * f64::from(x).abs().max(f64::from(y).abs())
        };
        for (slot, want) in [(0usize, &want_a), (1usize, &want_b)] {
            let cc = &got[slot];
            assert_eq!(cc.num_specs, seed.num_specs, "slot {slot}: per-domain rows");
            assert_eq!(cc.dim, want.dim, "slot {slot}: frontier width");
            for (i, (g, w)) in cc.lower_a.iter().zip(&want.lower_a).enumerate() {
                assert!(tol(*g, *w), "slot {slot} lower_a[{i}]: {g} vs {w}");
            }
            for (i, (g, w)) in cc.lower_b.iter().zip(&want.lower_b).enumerate() {
                assert!(tol(*g, *w), "slot {slot} lower_b[{i}]: {g} vs {w}");
            }
            for (i, (g, w)) in cc.upper_b.iter().zip(&want.upper_b).enumerate() {
                assert!(tol(*g, *w), "slot {slot} upper_b[{i}]: {g} vs {w}");
            }
        }
    }

    /// Review defect D4: every existing fixture emits only `Residual`, so
    /// `merge_streams`, the zero-bias P seeding, the `cf.dim != cp.dim` guard
    /// and the F-before-P `node_abs` split were unreached by any test — and
    /// that is exactly the branch where a fold-order drift is possible. This
    /// fixture emits a genuine `ResidualProj` (BOTH branches non-empty).
    #[cfg(feature = "gpu-tests")]
    fn resnet_proj_coeff_fixture() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Vec<f32>, Vec<f32>) {
        let (segments, seed, lo, hi) = resnet_coeff_fixture();
        let mid = 6usize;
        let mk = |n: usize, f: fn(usize) -> f32| -> Arc<[f32]> {
            (0..n).map(f).collect::<Vec<f32>>().into()
        };
        // P branch: a second mid->mid map, deliberately DIFFERENT from F so a
        // swapped or shared fold would change the composed frontier.
        let proj = GpuCrownLayer::Linear {
            weight: mk(mid * mid, |i| {
                0.17 - 0.06 * ((i % 4) as f32) + 0.01 * ((i % 3) as f32)
            }),
            bias: Some(mk(mid, |i| 0.03 - 0.02 * (i as f32))),
            out_features: mid,
            in_features: mid,
            cert_err: ny_core::CertifiedWeightError::default(),
        };
        let proj_act = GpuCrownLayer::Activation {
            lower_slope: (0..mid).map(|j| 0.2 + 0.06 * (j as f32)).collect(),
            upper_slope: (0..mid).map(|j| 0.55 + 0.03 * (j as f32)).collect(),
            lower_intercept: vec![0.0; mid],
            upper_intercept: (0..mid).map(|j| 0.07 + 0.015 * (j as f32)).collect(),
            num_neurons: mid,
        };
        let segments = segments
            .into_iter()
            .map(|seg| match seg {
                GpuResnetSegment::Residual(f) => {
                    GpuResnetSegment::ResidualProj(f, vec![proj.clone(), proj_act.clone()])
                }
                other => other,
            })
            .collect();
        (segments, seed, lo, hi)
    }

    /// Review defect D4: the `ResidualProj` twin of the self-consistency pin.
    /// The projection branch is where `merge_streams` and the F-before-P
    /// `node_abs` split live, so the egress must agree with the bounds entry
    /// THERE too, not just on the identity-skip shape.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn resnet_proj_certified_coeffs_concretize_to_the_resnet_bounds_entry() {
        let _g = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (segments, seed, lo, hi) = resnet_proj_coeff_fixture();
        let coeffs = device
            .crown_backward_gpu_resnet_sound_coeffs(&segments, &seed, &lo, &hi, &[], &[])
            .expect("projection egress must run")
            .expect("projection egress must publish on an armed adapter");
        // The composed frontier must reach the INPUT width, not a branch width.
        assert_eq!(
            coeffs.dim,
            lo.len(),
            "composed frontier must reach the input"
        );
        assert_eq!(coeffs.num_specs, seed.num_specs);
        for (label, v) in [
            ("lower_a_err", &coeffs.lower_a_err),
            ("upper_a_err", &coeffs.upper_a_err),
        ] {
            assert!(
                v.iter().all(|x| x.is_finite() && *x >= 0.0),
                "{label}: radii must be finite and non-negative"
            );
        }
        let bounds = device
            .crown_backward_gpu_resnet_sound(&segments, &seed, &lo, &hi, &[], &[])
            .expect("projection bounds entry must run");
        // Independent host concretization of the published coefficients must
        // ENCLOSE the bounds entry's own published bounds (identical single
        // pass here, so this is the self-consistency oracle for the PROJ path).
        for r in 0..coeffs.num_specs {
            let (mut lo_acc, mut hi_acc) = (
                f64::from(coeffs.lower_b[r]) - f64::from(coeffs.lower_b_err[r]),
                f64::from(coeffs.upper_b[r]) + f64::from(coeffs.upper_b_err[r]),
            );
            for j in 0..coeffs.dim {
                let k = r * coeffs.dim + j;
                let (al, au) = (
                    f64::from(coeffs.lower_a[k]) - f64::from(coeffs.lower_a_err[k]),
                    f64::from(coeffs.upper_a[k]) + f64::from(coeffs.upper_a_err[k]),
                );
                let (xl, xu) = (f64::from(lo[j]), f64::from(hi[j]));
                lo_acc += (al * xl)
                    .min(al * xu)
                    .min((f64::from(coeffs.lower_a[k]) + f64::from(coeffs.lower_a_err[k])) * xl)
                    .min((f64::from(coeffs.lower_a[k]) + f64::from(coeffs.lower_a_err[k])) * xu);
                hi_acc += (au * xl)
                    .max(au * xu)
                    .max((f64::from(coeffs.upper_a[k]) - f64::from(coeffs.upper_a_err[k])) * xl)
                    .max((f64::from(coeffs.upper_a[k]) - f64::from(coeffs.upper_a_err[k])) * xu);
            }
            let tol = 1e-4 * (1.0 + lo_acc.abs().max(hi_acc.abs()));
            assert!(
                lo_acc <= f64::from(bounds.lower_bounds[r]) + tol,
                "spec {r}: coeff concretization {lo_acc} must not exceed the \
                 bounds entry's lower {}",
                bounds.lower_bounds[r]
            );
            assert!(
                hi_acc >= f64::from(bounds.upper_bounds[r]) - tol,
                "spec {r}: coeff concretization {hi_acc} must not undercut the \
                 bounds entry's upper {}",
                bounds.upper_bounds[r]
            );
        }
    }

    /// The coefficient egress MUST ignore every domain-specific magnitude even
    /// when both bounds-only concretization gates are armed. Folding a radius
    /// against either table and zeroing that coefficient error is sound only for
    /// the supplied domain; publishing it as box-independent
    /// [`ny_core::CertifiedCoeffs`] would violate the trait contract. Different
    /// valid and deliberately understated tables must therefore publish the
    /// exact same nonzero coefficient radii.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn resnet_certified_coeffs_ignore_domain_abs_frontiers() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (segments, seed, in_lo, in_hi) = resnet_coeff_fixture();
        // Two Activations in the fixture (trunk act, branch act), width 6 each,
        // in backward fold order.
        let outward: Vec<Vec<f32>> = vec![vec![4.0f32; 6], vec![4.0f32; 6]];
        let understated: Vec<Vec<f32>> = vec![vec![0.0f32; 6], vec![0.0f32; 6]];
        let _coarse = ScopedEnvVar::set("NY_RESNET_ERR_CONCRETIZE", "1");
        let _fine = ScopedEnvVar::set("NY_RESNET_ERR_CONCRETIZE_FINE", "1");

        let wide = device
            .crown_backward_gpu_resnet_sound_coeffs(
                &segments, &seed, &in_lo, &in_hi, &outward, &outward,
            )
            .expect("outward tables must not error")
            .expect("outward tables must publish");
        let tight = device
            .crown_backward_gpu_resnet_sound_coeffs(
                &segments,
                &seed,
                &in_lo,
                &in_hi,
                &understated,
                &understated,
            )
            .expect("understated tables must not error")
            .expect("understated tables must publish");

        for (label, lhs, rhs) in [
            ("lower_a", &wide.lower_a, &tight.lower_a),
            ("upper_a", &wide.upper_a, &tight.upper_a),
            ("lower_a_err", &wide.lower_a_err, &tight.lower_a_err),
            ("upper_a_err", &wide.upper_a_err, &tight.upper_a_err),
            ("lower_b", &wide.lower_b, &tight.lower_b),
            ("upper_b", &wide.upper_b, &tight.upper_b),
            ("lower_b_err", &wide.lower_b_err, &tight.lower_b_err),
            ("upper_b_err", &wide.upper_b_err, &tight.upper_b_err),
        ] {
            assert_eq!(lhs, rhs, "{label} changed with ignored domain tables");
        }
        assert_eq!((wide.num_specs, wide.dim), (tight.num_specs, tight.dim));
        assert!(
            wide.lower_a_err.iter().any(|radius| *radius > 0.0)
                || wide.upper_a_err.iter().any(|radius| *radius > 0.0),
            "fixture must carry a real coefficient radius or equality is vacuous"
        );
    }

    #[test]
    fn resnet_certified_coeffs_concretize_to_the_resnet_bounds_entry() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let (segments, seed, in_lo, in_hi) = resnet_coeff_fixture();

        let coeffs = device
            .crown_backward_gpu_resnet_sound_coeffs(&segments, &seed, &in_lo, &in_hi, &[], &[])
            .expect("the segment egress must not error on a qualified device")
            .expect("a qualified device must publish the composed frontier");

        assert_eq!(coeffs.num_specs, seed.num_specs);
        assert_eq!(
            coeffs.dim,
            in_lo.len(),
            "the segment egress must publish the COMPOSED (input-width) frontier, \
             not a single segment's intermediate one"
        );
        for field in [
            &coeffs.lower_a,
            &coeffs.upper_a,
            &coeffs.lower_a_err,
            &coeffs.upper_a_err,
        ] {
            assert_eq!(field.len(), coeffs.num_specs * coeffs.dim);
        }
        assert!(
            coeffs
                .lower_a_err
                .iter()
                .chain(coeffs.upper_a_err.iter())
                .chain(coeffs.lower_b_err.iter())
                .chain(coeffs.upper_b_err.iter())
                .all(|v| v.is_finite() && *v >= 0.0),
            "published radii must be finite and non-negative"
        );

        let bounds = device
            .crown_backward_gpu_resnet_sound(&segments, &seed, &in_lo, &in_hi, &[], &[])
            .expect("resnet bounds entry");
        let (host_lo, host_hi) = concretize_certified_coeffs(&coeffs, &in_lo, &in_hi);
        for s in 0..coeffs.num_specs {
            let (gl, gu) = (
                f64::from(bounds.lower_bounds[s]),
                f64::from(bounds.upper_bounds[s]),
            );
            let tol = 1e-4 * (1.0 + gl.abs().max(gu.abs()));
            assert!(
                (host_lo[s] - gl).abs() <= tol && (host_hi[s] - gu).abs() <= tol,
                "row {s}: concretizing the published segment frontier gives \
                 [{}, {}] but the resnet bounds entry says [{gl}, {gu}] — the two \
                 entries disagree",
                host_lo[s],
                host_hi[s]
            );
            assert!(host_lo[s] <= host_hi[s], "row {s}: inverted frontier");
        }
    }

    /// Publishing coefficients is MORE authority than publishing the bound
    /// derived from them, so the SEGMENT egress must move in lockstep with the
    /// sound CROWN authority gate too: no authority, no frontier.
    #[test]
    fn resnet_certified_coeffs_publish_only_with_authority() {
        let _guard = gpu_test_serial_guard();
        let device = require_device();
        let (segments, seed, in_lo, in_hi) = resnet_coeff_fixture();
        let published = device
            .crown_backward_gpu_resnet_sound_coeffs(&segments, &seed, &in_lo, &in_hi, &[], &[])
            .expect("the egress declines by returning Ok(None), never by erroring");
        assert_eq!(
            published.is_some(),
            GpuCrownBackward::provides_sound_gpu_crown(&*device),
            "the segment coefficient egress and the sound-CROWN authority gate \
             must move together"
        );
    }

    /// Modes whose kernels cannot carry the charge must REFUSE a nonzero
    /// declaration rather than silently drop it.
    #[test]
    fn cert_err_refuses_the_modes_that_cannot_charge_it() {
        let _guard = gpu_test_serial_guard();
        let device = require_verdict_device();
        let fx = cert_err_fixture();
        let charged = cert_err_layers(
            &fx,
            ny_core::CertifiedWeightError {
                weight_rel_err: 1e-4,
                bias_abs_err: 0.0,
            },
        );
        let run = |device: &WgpuDevice| {
            device.crown_backward_sound_resident(
                &charged,
                &fx.spec,
                fx.out_features,
                fx.out_features,
                &fx.input_lower,
                &fx.input_upper,
            )
        };
        // Baseline: the charge is accepted in the default mode.
        run(&device).expect("the default per-entry mode charges cert_err");

        {
            let _eft = ScopedEnvVar::set("NY_EFT_ERR", "1");
            // The EFT min-combine has no weight-error term, so min(higham, eft)
            // could erase the charge. Either the gate is inactive on this adapter
            // (walk succeeds, charge intact) or the walk refuses — never a
            // silent tightening.
            if let Ok((lo, hi)) = run(&device) {
                let (exact_lo, exact_hi) = device
                    .crown_backward_sound_resident(
                        &cert_err_layers(&fx, ny_core::CertifiedWeightError::default()),
                        &fx.spec,
                        fx.out_features,
                        fx.out_features,
                        &fx.input_lower,
                        &fx.input_upper,
                    )
                    .expect("exact-weight walk");
                for i in 0..fx.out_features {
                    assert!(
                        lo[i] <= exact_lo[i] && hi[i] >= exact_hi[i],
                        "row {i}: NY_EFT_ERR=1 admitted a charged walk whose bound \
                         is TIGHTER than the uncharged one"
                    );
                }
            }
        }
        {
            let _rowmax = ScopedEnvVar::set("NY_CONV_ERR_ROWMAX", "1");
            assert!(
                run(&device).is_err(),
                "the legacy row-max conv error mode must refuse a nonzero cert_err"
            );
        }
        {
            let _coalesce = ScopedEnvVar::set("NY_FOLD_COALESCE", "1");
            let bias_charged = cert_err_layers(
                &fx,
                ny_core::CertifiedWeightError {
                    weight_rel_err: 0.0,
                    bias_abs_err: 1e-4,
                },
            );
            assert!(
                device
                    .crown_backward_sound_resident(
                        &bias_charged,
                        &fx.spec,
                        fx.out_features,
                        fx.out_features,
                        &fx.input_lower,
                        &fx.input_upper,
                    )
                    .is_err(),
                "the bias charge must refuse the coalesced fold"
            );
        }
    }
}
