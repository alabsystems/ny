// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Carrier-driven resident Cut-CROWN SHADOW for the WGPU backend
//! (observation-only; the CUDA resident cut kernel's charged-Metal twin).
//!
//! This module makes `WgpuDevice` an honest implementor of
//! `GpuCrownBackward::provides_resident_cut_shadow`: the observation-only
//! `crown_backward_gpu_resnet_sound_beta_cut_shadow` trait entry runs the SAME
//! optimized resident beta fold twice — once untouched (the baseline, the only
//! consumable bound) and once with one validated
//! [`ny_core::ResidentLowerCutCarrier`] threaded through the walk's activation
//! cursor — and can attach the binding row's `(baseline, shadow)` pair as
//! telemetry. The shadow NEVER has verdict authority: every disposition returns
//! the exact baseline, upper-lane bit-identity is required before an
//! observation may attach, and any refusal is a telemetry miss, not an error.
//!
//! # Kernel design — what is CHARGED vs REFUSED
//!
//! The cut application itself is ONE device pass, [`CUT_SHADOW_APPLY_SHADER`]:
//! for each compact element `i` (a `(row, target-neuron)` coefficient site or a
//! `(row)` lower-bias site) with stored channel `(value, source)` from the
//! carrier:
//!
//! ```text
//! sum  = center[i] + value                       // A1  f32 RN add (may flush)
//! gap  = |sum| · U                               // A2  bound on A1's rounding
//! e    = (err[i] + source) + gap                 // A3/A4 nonneg error chain
//! err' = round_up_pos(round_up_pos(e·slack) + flush_additive)   // A5/A6
//! ```
//!
//! Enclosure claim (the Cert-2 §S6/S8/S9 obligation): for every real `q` with
//! `|q − (center + value)| ≤ err + source` (the carrier channel's own contract
//! composed with the incoming lane's), `q ∈ [sum − err', sum + err']`.
//! Sufficient: `err' ≥ (err + source) + |sum − (center + value)|`.
//!
//! Charged sites (all of them — the exact-rational oracle
//! [`oracle_tests`] proves the composed bound under BOTH the IEEE model and the
//! pure operand-DAZ + result-FTZ model of `flush_charge_oracle::Hw`):
//! * A1's round-to-nearest residual: `≤ U·|sum|`, charged by A2.
//! * A2/A3/A4/A5's own RN under-rounding: multiplicative, `≤ (1+U)⁴`, charged
//!   by `slack = CUT_APPLY_SLACK` (pinned `≥ (1+2⁻²⁴)⁴` by the oracle).
//! * EVERY flush loss is a magnitude-independent absolute loss `< 2⁻¹²⁶`
//!   (operand-DAZ of `center`/`err`/`value`/`source`, result-FTZ of
//!   `sum`/`gap`/`e`/`e·slack`; ≤ 10 sites), charged by
//!   `flush_additive = CUT_APPLY_FLUSH_ADDITIVE = 32·2⁻¹²⁶` (≥3× margin,
//!   oracle-pinned). Unlike the AW-error combines there is NO channel here
//!   whose flush loss scales with a runtime radius — each element performs one
//!   add of one call-constant channel — so nothing needs a subnormal-input
//!   refusal of its own and the additive floor is a complete cover.
//!
//! Refused (typed, never silently degraded):
//! * every refusal `charged_walk_guard` already imposes on the surrounding walk
//!   (subnormal weights/bias/slopes/intercepts, `cert_err` layers, non-admitted
//!   layer kinds, the EFT channel) — the shadow runs the SAME walk and inherits
//!   them verbatim, as it inherits the charged concretize consult (subnormal
//!   input-box endpoints refused);
//! * a target activation that the walk does not encounter exactly once, a
//!   branch/width mismatch, or a multi-domain (wide) fold — the shadow refuses
//!   rather than degrade to the untouched branch, because a silently unapplied
//!   cut would mislabel `shadow == baseline` as a measured Δ=0;
//! * non-finite kernel output (checked host-side after readback).
//!
//! # Carrier-validation split (host vs GPU)
//!
//! The GPU cannot run bit-exact replay checks atomically, so validation is
//! split exactly as follows; the kernel itself validates NOTHING and only ever
//! executes on data every host check has passed:
//! * **ny-propagate authority context (unchanged, backend-independent):** call
//!   seal pointer identity, snapshot generations, input-box bit-equality,
//!   fresh-octahedron bit-exact replay, target-ReLU resolution
//!   (`certified_cut_authority.rs`). These complete BEFORE the backend is
//!   invoked and the carrier reaches this module.
//! * **host, this backend, pre-dispatch:** `Disabled` off-parity (the baseline
//!   runs before any carrier field is read), carrier presence, binding-row
//!   bounds, nonzero multiplier, `carrier.deadline() == deadline`, live
//!   deadline, `validate_for_call` against the exact resident activation-width
//!   table, channel finiteness, target-column bounds, single-domain shape, and
//!   the walk's "target encountered exactly once" accounting.
//! * **host, post-readback:** finiteness of the mutated lanes, shadow-result
//!   shape, upper-lane bit-identity vs the baseline, and the observation's
//!   bit-binding to the exact baseline row (`ResidentCutShadowOutcome`).
//!
//! # Deadline
//!
//! The trait entry's explicit deadline is armed as a
//! `CallLocalCrownDeadlineScope` around BOTH folds, so the resident walk's
//! per-layer `crown_backward_deadline_expired()` polls bound the whole
//! observation; the branch split additionally polls before each apply phase.

use std::cell::RefCell;
use std::time::Instant;

use ny_core::{
    GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment, NyError,
    ResidentCutShadowObservation, ResidentCutShadowOutcome, ResidentCutShadowPolicy,
    ResidentLowerCutCarrier, Result,
};

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;

/// Outward multiplicative slack on the kernel's error chain: covers the ≤4 RN
/// under-roundings of A2..A5 (`≥ (1+2⁻²⁴)⁴ ≈ 1.00000024`; oracle-pinned).
/// Same value as the audited resident activation SLACK.
pub(in crate::wgpu_device) const CUT_APPLY_SLACK: f32 = 1.000_001;

/// Additive flush floor: `32·2⁻¹²⁶ = 2⁻¹²¹`. The kernel has ≤10 flush sites,
/// each losing `< 2⁻¹²⁶` absolute (see module docs); ≥3× margin, oracle-pinned.
pub(in crate::wgpu_device) const CUT_APPLY_FLUSH_ADDITIVE: f32 = f32::MIN_POSITIVE * 32.0;

/// 32-byte std140-clean uniform. Layout MUST match WGSL `struct Params`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CutApplyParams {
    n: u32,
    slack: f32,
    flush_additive: f32,
    _pad: u32,
}

/// The resident cut-apply pass. One thread per compact element; channel layout
/// is `[value₀, source₀, value₁, source₁, …]`. The oracle test
/// `model_tracks_the_shipped_cut_shader_text` transcribes this text line by
/// line — edit both together.
const CUT_SHADOW_APPLY_SHADER: &str = r#"
struct Params { n: u32, slack: f32, flush_additive: f32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ch: array<f32>;
@group(0) @binding(2) var<storage, read_write> centers: array<f32>;
@group(0) @binding(3) var<storage, read_write> errs: array<f32>;
const U: f32 = 0.00000005960464477539063; // 2^-24
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let value = ch[2u * i];
    let source = ch[2u * i + 1u];
    let base = centers[i];
    let sum = base + value;
    centers[i] = sum;
    let gap = abs(sum) * U;
    let e = (errs[i] + source) + gap;
    errs[i] = round_up_pos(round_up_pos(e * p.slack) + p.flush_additive);
}
"#;

/// One validated `(value, source_abs_error)` channel copied out of a carrier.
#[derive(Clone, Copy, Debug)]
pub(in crate::wgpu_device) struct CutApplyChannel {
    pub(in crate::wgpu_device) value: f32,
    pub(in crate::wgpu_device) source: f32,
}

/// One seed row's complete lower-only contribution.
#[derive(Clone, Debug)]
pub(in crate::wgpu_device) struct CutApplyRow {
    pre: [CutApplyChannel; 2],
    post: [CutApplyChannel; 2],
    bias: CutApplyChannel,
}

/// Which coefficient channel family an apply pass consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::wgpu_device) enum CutChannelKind {
    /// ReLU-OUTPUT frontier channels, applied BEFORE the target relaxation.
    Post,
    /// ReLU-INPUT frontier channels, applied AFTER the target relaxation.
    Pre,
}

/// Call-local arithmetic snapshot of one validated carrier.
///
/// Module-private, non-serializable, and held only inside a
/// [`CutShadowHookScope`] for the duration of one synchronous shadow fold. The
/// authority-bearing `ResidentLowerCutCarrier` stays non-`Clone`; this snapshot
/// carries no multipliers, no deadline, and no identity — it is pure transport
/// for the audited apply kernel and cannot cross back into ny-propagate.
#[derive(Clone, Debug)]
pub(in crate::wgpu_device) struct CutApplySnapshot {
    target_activation: usize,
    target_width: usize,
    ordered_neurons: [usize; 2],
    rows: Vec<CutApplyRow>,
}

impl CutApplySnapshot {
    /// Copy one already-`validate_for_call`-checked carrier into an arithmetic
    /// snapshot, re-validating finiteness and column bounds defensively.
    pub(in crate::wgpu_device) fn from_carrier(carrier: &ResidentLowerCutCarrier) -> Result<Self> {
        let [first, second] = carrier.ordered_neurons();
        let width = carrier.target_width();
        if first == second || first >= width || second >= width {
            return Err(NyError::InvalidSpec(
                "wgpu resident cut shadow: ordered pair is invalid for the target width".into(),
            ));
        }
        let channel = |c: ny_core::ResidentLowerCutChannel| -> Result<CutApplyChannel> {
            if !c.value().is_finite()
                || !c.source_abs_error().is_finite()
                || c.source_abs_error() < 0.0
            {
                return Err(NyError::NumericalInstability(
                    "wgpu resident cut shadow: carrier channel is not finite".into(),
                ));
            }
            Ok(CutApplyChannel {
                value: c.value(),
                source: c.source_abs_error(),
            })
        };
        let rows = carrier
            .rows()
            .iter()
            .map(|row| {
                Ok(CutApplyRow {
                    pre: [channel(row.pre()[0])?, channel(row.pre()[1])?],
                    post: [channel(row.post()[0])?, channel(row.post()[1])?],
                    bias: channel(row.bias())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            target_activation: carrier.target_activation(),
            target_width: width,
            ordered_neurons: [first, second],
            rows,
        })
    }

    pub(in crate::wgpu_device) const fn target_activation(&self) -> usize {
        self.target_activation
    }

    pub(in crate::wgpu_device) const fn target_width(&self) -> usize {
        self.target_width
    }

    pub(in crate::wgpu_device) const fn ordered_neurons(&self) -> [usize; 2] {
        self.ordered_neurons
    }

    pub(in crate::wgpu_device) fn rows(&self) -> &[CutApplyRow] {
        &self.rows
    }
}

struct CutShadowHookState {
    snapshot: CutApplySnapshot,
    applied_walks: u32,
}

thread_local! {
    /// THREAD-LOCAL by design (the `RESIDENT_IO` precedent): the shadow driver
    /// arms this strictly around ONE synchronous resident fold on the calling
    /// thread; concurrent BaB workers each see their own (unarmed) slot, so an
    /// armed snapshot can never leak into an unrelated walk.
    static CUT_SHADOW_HOOK: RefCell<Option<CutShadowHookState>> =
        const { RefCell::new(None) };
}

/// RAII scope arming the thread-local cut-shadow hook. Drop ALWAYS clears the
/// slot, so an erroring walk cannot leave a stale snapshot behind.
#[must_use = "the cut-shadow hook is armed only while this scope is alive"]
pub(in crate::wgpu_device) struct CutShadowHookScope {
    _private: (),
}

impl CutShadowHookScope {
    fn arm(snapshot: CutApplySnapshot) -> Result<Self> {
        CUT_SHADOW_HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(NyError::InternalError(
                    "wgpu resident cut shadow: hook already armed on this thread".into(),
                ));
            }
            *slot = Some(CutShadowHookState {
                snapshot,
                applied_walks: 0,
            });
            Ok(Self { _private: () })
        })
    }

    fn applied_walks(&self) -> u32 {
        CUT_SHADOW_HOOK.with(|slot| {
            slot.borrow()
                .as_ref()
                .map_or(0, |state| state.applied_walks)
        })
    }
}

impl Drop for CutShadowHookScope {
    fn drop(&mut self) {
        CUT_SHADOW_HOOK.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Fold-order target-activation index of the armed hook, if any. `None` on
/// every production walk (the hook arms only inside the shadow driver).
pub(in crate::wgpu_device) fn armed_cut_shadow_target() -> Option<usize> {
    CUT_SHADOW_HOOK.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|state| state.snapshot.target_activation())
    })
}

/// Clone the armed snapshot for one branch application. Rows are ≤64 tiny
/// structs; the clone keeps the `RefCell` borrow out of the branch walk.
pub(in crate::wgpu_device) fn armed_cut_shadow_snapshot() -> Option<CutApplySnapshot> {
    CUT_SHADOW_HOOK.with(|slot| slot.borrow().as_ref().map(|state| state.snapshot.clone()))
}

/// Record one complete (post + pre + bias) application by the current walk.
pub(in crate::wgpu_device) fn note_cut_shadow_walk_applied() {
    CUT_SHADOW_HOOK.with(|slot| {
        if let Some(state) = slot.borrow_mut().as_mut() {
            state.applied_walks = state.applied_walks.saturating_add(1);
        }
    });
}

impl WgpuDevice {
    /// Run the audited cut-apply kernel on one compact element set:
    /// `centers[i] += channels[i].value` with the full error charge landing in
    /// `errs[i]`. Pure arithmetic — every validation is the caller's (host).
    /// Non-finite readback is a typed refusal.
    fn resident_cut_apply_compact(
        &self,
        centers: &[f32],
        errs: &[f32],
        channels: &[CutApplyChannel],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = channels.len();
        if n == 0 || centers.len() != n || errs.len() != n {
            return Err(NyError::InvalidSpec(
                "wgpu resident cut shadow: compact apply shape mismatch".into(),
            ));
        }
        if centers.iter().chain(errs).any(|v| !v.is_finite()) || errs.iter().any(|v| *v < 0.0) {
            return Err(NyError::NumericalInstability(
                "wgpu resident cut shadow: compact apply input is not a finite enclosure".into(),
            ));
        }
        let n_u32 = super::gpu_checked_u32(n, "cut apply elements")?;
        self.run_gpu_checked("resident_cut_apply", || {
            let params = CutApplyParams {
                n: n_u32,
                slack: CUT_APPLY_SLACK,
                flush_additive: CUT_APPLY_FLUSH_ADDITIVE,
                _pad: 0,
            };
            let params_buf = create_buffer(
                &self.device,
                "cut_apply_params",
                size_of::<CutApplyParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            self.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            let ch: Vec<f32> = channels.iter().flat_map(|c| [c.value, c.source]).collect();
            let ch_buf = create_buffer(
                &self.device,
                "cut_apply_channels",
                (ch.len() * size_of::<f32>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.queue
                .write_buffer(&ch_buf, 0, bytemuck::cast_slice(&ch));
            let lane_bytes = (n * size_of::<f32>()) as u64;
            let centers_buf = create_buffer(
                &self.device,
                "cut_apply_centers",
                lane_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.queue
                .write_buffer(&centers_buf, 0, bytemuck::cast_slice(centers));
            let errs_buf = create_buffer(
                &self.device,
                "cut_apply_errs",
                lane_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.queue
                .write_buffer(&errs_buf, 0, bytemuck::cast_slice(errs));

            let pipe = self.create_simple_pipeline(
                CUT_SHADOW_APPLY_SHADER,
                "resident_cut_apply",
                &[false, true, true],
            );
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resident_cut_apply"),
                });
            self.pass_simple(
                &mut encoder,
                &pipe,
                &params_buf,
                &[&ch_buf, &centers_buf, &errs_buf],
                n_u32.div_ceil(64),
            );
            let centers_staging = create_buffer(
                &self.device,
                "cut_apply_centers_staging",
                lane_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let errs_staging = create_buffer(
                &self.device,
                "cut_apply_errs_staging",
                lane_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(&centers_buf, 0, &centers_staging, 0, lane_bytes);
            encoder.copy_buffer_to_buffer(&errs_buf, 0, &errs_staging, 0, lane_bytes);
            self.queue.submit(std::iter::once(encoder.finish()));

            let centers_out = WgpuDevice::read_buffer(&self.device, &centers_staging, n)?;
            let errs_out = WgpuDevice::read_buffer(&self.device, &errs_staging, n)?;
            if centers_out.iter().chain(&errs_out).any(|v| !v.is_finite())
                || errs_out.iter().any(|v| *v < 0.0)
            {
                return Err(NyError::NumericalInstability(
                    "wgpu resident cut shadow: cut-apply kernel produced a non-finite or \
                     negative-radius lane — refusing (fail-closed)"
                        .into(),
                ));
            }
            Ok((centers_out, errs_out))
        })
    }

    /// Apply one coefficient channel family (post or pre) of the snapshot to
    /// the LOWER coefficient lanes at the two target columns, through the
    /// audited device kernel. Host does all shape/bounds validation and the
    /// gather/scatter (pure copies, no arithmetic).
    pub(in crate::wgpu_device) fn resident_cut_apply_lower_pair_columns(
        &self,
        lower_a: &mut [f32],
        lower_err: &mut [f32],
        width: usize,
        num_specs: usize,
        snapshot: &CutApplySnapshot,
        kind: CutChannelKind,
    ) -> Result<()> {
        let expected = num_specs.checked_mul(width).ok_or_else(|| {
            NyError::InvalidSpec("wgpu resident cut shadow: coefficient shape overflow".into())
        })?;
        if lower_a.len() != expected
            || lower_err.len() != expected
            || snapshot.rows().len() != num_specs
            || snapshot.target_width() != width
        {
            return Err(NyError::InvalidSpec(
                "wgpu resident cut shadow: target coefficient shape mismatch".into(),
            ));
        }
        let pair = snapshot.ordered_neurons();
        if pair[0] >= width || pair[1] >= width {
            return Err(NyError::InvalidSpec(
                "wgpu resident cut shadow: target neuron is outside resident width".into(),
            ));
        }
        let mut indices = Vec::with_capacity(num_specs * 2);
        let mut centers = Vec::with_capacity(num_specs * 2);
        let mut errs = Vec::with_capacity(num_specs * 2);
        let mut channels = Vec::with_capacity(num_specs * 2);
        for (row_index, row) in snapshot.rows().iter().enumerate() {
            let family = match kind {
                CutChannelKind::Post => &row.post,
                CutChannelKind::Pre => &row.pre,
            };
            for pair_position in 0..2 {
                let index = row_index * width + pair[pair_position];
                indices.push(index);
                centers.push(lower_a[index]);
                errs.push(lower_err[index]);
                channels.push(family[pair_position]);
            }
        }
        let (centers_out, errs_out) =
            self.resident_cut_apply_compact(&centers, &errs, &channels)?;
        for (position, &index) in indices.iter().enumerate() {
            lower_a[index] = centers_out[position];
            lower_err[index] = errs_out[position];
        }
        Ok(())
    }

    /// Apply the snapshot's per-row lower-bias channel through the same kernel.
    pub(in crate::wgpu_device) fn resident_cut_apply_lower_bias(
        &self,
        lower_b: &mut [f32],
        lower_b_err: &mut [f32],
        snapshot: &CutApplySnapshot,
    ) -> Result<()> {
        let num_specs = snapshot.rows().len();
        if lower_b.len() != num_specs || lower_b_err.len() != num_specs {
            return Err(NyError::InvalidSpec(
                "wgpu resident cut shadow: bias row shape mismatch".into(),
            ));
        }
        let channels: Vec<CutApplyChannel> = snapshot.rows().iter().map(|row| row.bias).collect();
        let (b_out, b_err_out) =
            self.resident_cut_apply_compact(lower_b, lower_b_err, &channels)?;
        lower_b.copy_from_slice(&b_out);
        lower_b_err.copy_from_slice(&b_err_out);
        Ok(())
    }

    /// One-time pinned selfcheck of the cut-apply kernel (the qualification
    /// probe behind `provides_resident_cut_shadow`). All-normal operands, so
    /// the pure-flush and IEEE models agree: centers must be BIT-EXACT against
    /// the host transcription and every error lane must both bit-match the
    /// transcription and contain the exact demand. Any GPU error, mismatch, or
    /// containment breach refuses the capability (fail-closed); the sound
    /// baseline path is untouched either way.
    pub(crate) fn verify_resident_cut_apply(&self) -> bool {
        *self
            .resident_cut_selfcheck
            .get_or_init(|| match self.run_resident_cut_selfcheck() {
                Ok(passed) => {
                    if !passed {
                        tracing::warn!(
                            target: "ny_gpu::wgpu",
                            adapter = %self.adapter_info.name,
                            "resident cut-apply selfcheck FAILED: refusing the \
                             resident Cut-CROWN shadow capability (fail-closed)"
                        );
                    }
                    passed
                }
                Err(e) => {
                    tracing::warn!(
                        target: "ny_gpu::wgpu",
                        adapter = %self.adapter_info.name,
                        error = %e,
                        "resident cut-apply selfcheck could not run: refusing the \
                         resident Cut-CROWN shadow capability (fail-closed)"
                    );
                    false
                }
            })
    }

    fn run_resident_cut_selfcheck(&self) -> Result<bool> {
        let (centers, errs, channels) = cut_selfcheck_fixture();
        let (centers_out, errs_out) =
            self.resident_cut_apply_compact(&centers, &errs, &channels)?;
        for i in 0..channels.len() {
            let (expect_center, expect_err) = model_cut_apply_ieee(
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
                CUT_APPLY_SLACK,
                CUT_APPLY_FLUSH_ADDITIVE,
            );
            if centers_out[i].to_bits() != expect_center.to_bits()
                || errs_out[i].to_bits() != expect_err.to_bits()
            {
                return Ok(false);
            }
            // Exact containment demand in f64 (each f32 is exact in f64; the
            // f64 sum of two f32 is exact).
            let exact = f64::from(centers[i]) + f64::from(channels[i].value);
            let demand = f64::from(errs[i])
                + f64::from(channels[i].source)
                + (f64::from(centers_out[i]) - exact).abs();
            if f64::from(errs_out[i]) < demand {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether this device may honestly claim the resident Cut-CROWN shadow:
    /// verdict authority (fully-qualified OR charged) AND a passing cut-apply
    /// selfcheck. Consulted by `provides_resident_cut_shadow` and re-checked by
    /// the driver (fail-closed); prewarmed by both verdict constructors so the
    /// probe result is qualification-time evidence.
    pub(crate) fn resident_cut_shadow_capability(&self) -> bool {
        (self.sound_gpu_authority_cached() || self.charged_flush_authority_cached().is_some())
            && self.verify_resident_cut_apply()
    }
}

/// Pinned selfcheck fixture: exponent-spread NORMAL values only (flush can
/// never fire, so the IEEE transcription is the exact device expectation on
/// both conformant and pure-flush adapters).
fn cut_selfcheck_fixture() -> (Vec<f32>, Vec<f32>, Vec<CutApplyChannel>) {
    let centers = vec![1.0_f32, -0.75, 3.5e4, -2.0e-5, 0.125, -1.0];
    let errs = vec![0.0_f32, 1.0e-7, 0.5, 3.0e-6, 0.0, 2.5e-3];
    let channels = vec![
        CutApplyChannel {
            value: 0.25,
            source: 0.0,
        },
        CutApplyChannel {
            value: 0.5,
            source: 1.0e-6,
        },
        CutApplyChannel {
            value: -1.25e3,
            source: 2.0e-4,
        },
        CutApplyChannel {
            value: 7.5e-6,
            source: 0.0,
        },
        CutApplyChannel {
            value: -0.125,
            source: 4.0e-8,
        },
        CutApplyChannel {
            value: 1.0,
            source: 0.0,
        },
    ];
    (centers, errs, channels)
}

/// Host round_up_pos twin (bit-identical to the WGSL helper).
fn host_round_up_pos(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if (bits & 0x8000_0000) != 0 || magnitude == 0 {
        return 0.0;
    }
    if magnitude < 0x0080_0000 {
        return f32::MIN_POSITIVE;
    }
    f32::from_bits(bits + 1)
}

/// IEEE transcription of one kernel element (no flush). Kept in lockstep with
/// [`CUT_SHADOW_APPLY_SHADER`] by `model_tracks_the_shipped_cut_shader_text`.
fn model_cut_apply_ieee(
    base: f32,
    err_in: f32,
    value: f32,
    source: f32,
    slack: f32,
    flush_additive: f32,
) -> (f32, f32) {
    const U: f32 = 5.960_464_5e-8; // 2^-24
    let sum = base + value;
    let gap = sum.abs() * U;
    let e = (err_in + source) + gap;
    let err_out = host_round_up_pos(host_round_up_pos(e * slack) + flush_additive);
    (sum, err_out)
}

/// The `GpuCrownBackward::crown_backward_gpu_resnet_sound_beta_cut_shadow`
/// body for `WgpuDevice` — the exact CUDA trait-override shape, on the wgpu
/// resident walk.
#[allow(clippy::too_many_arguments)]
pub(in crate::wgpu_device) fn run_resident_cut_shadow(
    device: &WgpuDevice,
    policy: ResidentCutShadowPolicy,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    beta_signed: &[Vec<f32>],
    frontier_abs: &[Vec<f32>],
    node_abs: &[Vec<f32>],
    carrier: Option<&ResidentLowerCutCarrier>,
    binding_row: usize,
    deadline: Instant,
) -> Result<ResidentCutShadowOutcome> {
    // Load-bearing off parity: exactly the historical call, before reading the
    // carrier, binding row, explicit shadow deadline, or the cut capability.
    if policy == ResidentCutShadowPolicy::Disabled {
        let baseline = device.crown_backward_gpu_resnet_sound_beta(
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
        )?;
        return Ok(ResidentCutShadowOutcome::disabled(baseline));
    }

    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "wgpu resident cut shadow: deadline expired before the baseline replay".into(),
        ));
    }
    // Deadline-bounded baseline replay: a late or failed replay returns Err and
    // the caller keeps its historical result unchanged (the CUDA contract).
    let baseline = {
        let _deadline = super::super::CallLocalCrownDeadlineScope::arm(deadline);
        device.crown_backward_gpu_resnet_sound_beta(
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
        )?
    };
    let Some(carrier) = carrier else {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    };
    if binding_row >= seed.num_specs
        || binding_row >= baseline.lower_bounds.len()
        || baseline.lower_bounds.len() != seed.num_specs
        || baseline.upper_bounds.len() != seed.num_specs
        || !carrier.has_nonzero_multiplier()
        || carrier.deadline() != deadline
        || Instant::now() >= deadline
    {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    }
    if !device.resident_cut_shadow_capability() {
        return Ok(ResidentCutShadowOutcome::backend_unavailable(baseline));
    }
    let Ok(activation_widths) = resident_activation_widths(segments) else {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    };
    let Some(&target_width) = activation_widths.get(carrier.target_activation()) else {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    };
    if carrier
        .validate_for_call(
            activation_widths.len(),
            target_width,
            seed.num_specs,
            deadline,
        )
        .is_err()
    {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    }
    let Ok(snapshot) = CutApplySnapshot::from_carrier(carrier) else {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    };

    let shadow = {
        let _deadline = super::super::CallLocalCrownDeadlineScope::arm(deadline);
        let hook = match CutShadowHookScope::arm(snapshot) {
            Ok(hook) => hook,
            Err(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
        };
        let beta_refs: Vec<&[f32]> = beta_signed.iter().map(|v| v.as_slice()).collect();
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let result = device.crown_backward_gpu_resnet_sound_beta_inner(
            segments,
            seed,
            input_lower,
            input_upper,
            &beta_refs,
            &fa_refs,
            &na_refs,
        );
        let applied_walks = hook.applied_walks();
        drop(hook);
        match result {
            // Every completed fold that consumed the hook applied the complete
            // post/pre/bias mutation exactly once (the compose loop errors
            // otherwise); a fold that somehow never consumed it must not be
            // observed as a Δ=0 cut.
            Ok((lower_bounds, upper_bounds)) if applied_walks >= 1 => GpuCrownResult {
                lower_bounds,
                upper_bounds,
            },
            Ok(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
            Err(NyError::UnsupportedOp(_)) | Err(NyError::UnsupportedConfiguration(_)) => {
                return Ok(ResidentCutShadowOutcome::backend_unavailable(baseline));
            }
            Err(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
        }
    };
    if Instant::now() >= deadline
        || shadow.lower_bounds.len() != seed.num_specs
        || shadow.upper_bounds.len() != seed.num_specs
        || shadow
            .lower_bounds
            .iter()
            .chain(&shadow.upper_bounds)
            .any(|value| !value.is_finite())
        // A lower-only carrier has no authority to perturb the upper channel.
        // Require exact replay identity before observing it.
        || shadow
            .upper_bounds
            .iter()
            .zip(&baseline.upper_bounds)
            .any(|(shadow, baseline)| shadow.to_bits() != baseline.to_bits())
    {
        return Ok(ResidentCutShadowOutcome::rejected(baseline));
    }

    let observation = match ResidentCutShadowObservation::try_new(
        binding_row,
        baseline.lower_bounds[binding_row],
        shadow.lower_bounds[binding_row],
    ) {
        Ok(observation) => observation,
        Err(_) => return Ok(ResidentCutShadowOutcome::rejected(baseline)),
    };
    match ResidentCutShadowOutcome::try_observed(baseline.clone(), observation) {
        Ok(outcome) => Ok(outcome),
        Err(_) => Ok(ResidentCutShadowOutcome::rejected(baseline)),
    }
}

/// Activation widths in exact resident fold order (F before P), mirroring the
/// CUDA `resident_activation_widths` and the ny-propagate authority twin.
fn resident_activation_widths(segments: &[GpuResnetSegment]) -> Result<Vec<usize>> {
    let mut widths = Vec::new();
    let mut visit = |layers: &[GpuCrownLayer]| -> Result<()> {
        for layer in layers {
            match layer {
                GpuCrownLayer::Activation { num_neurons, .. }
                | GpuCrownLayer::ActivationReluDualAlpha { num_neurons, .. } => {
                    if *num_neurons == 0 {
                        return Err(NyError::InvalidSpec(
                            "wgpu resident cut shadow: zero-width activation".into(),
                        ));
                    }
                    widths.push(*num_neurons);
                }
                _ => {}
            }
        }
        Ok(())
    };
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(layers) | GpuResnetSegment::Residual(layers) => {
                visit(layers)?;
            }
            GpuResnetSegment::ResidualProj(function, projection) => {
                visit(function)?;
                visit(projection)?;
            }
        }
    }
    if widths.is_empty() {
        return Err(NyError::InvalidSpec(
            "wgpu resident cut shadow: decomposition has no activation".into(),
        ));
    }
    Ok(widths)
}

// ---------------------------------------------------------------------------
// Exact-rational flush-channel oracle (#flush-charge style; CPU-only).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod oracle_tests {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Signed;

    use super::*;

    const F32_MIN_NORMAL: f32 = 1.1754944e-38; // 2^-126
    const U: f32 = 5.960_464_5e-8; // 2^-24

    fn is_subnormal(x: f32) -> bool {
        x != 0.0 && x.is_finite() && x.abs() < F32_MIN_NORMAL
    }

    /// Core-op flush policy (the kernel uses no FMA). Mirrors
    /// `flush_charge_oracle::Hw` restricted to core add/mul.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct Hw {
        core_ftz: bool,
        core_daz: bool,
    }

    const IEEE: Hw = Hw {
        core_ftz: false,
        core_daz: false,
    };
    const PURE_FLUSH: Hw = Hw {
        core_ftz: true,
        core_daz: true,
    };

    impl Hw {
        fn o(self, x: f32) -> f32 {
            if self.core_daz && is_subnormal(x) {
                if x.is_sign_negative() {
                    -0.0
                } else {
                    0.0
                }
            } else {
                x
            }
        }
        fn r(self, x: f32) -> f32 {
            if self.core_ftz && is_subnormal(x) {
                if x.is_sign_negative() {
                    -0.0
                } else {
                    0.0
                }
            } else {
                x
            }
        }
        fn add(self, a: f32, b: f32) -> f32 {
            self.r(self.o(a) + self.o(b))
        }
        fn mul(self, a: f32, b: f32) -> f32 {
            self.r(self.o(a) * self.o(b))
        }
    }

    /// Line-by-line transcription of [`CUT_SHADOW_APPLY_SHADER`]'s element
    /// body under one hardware model. `round_up_pos` is integer/bit-only in
    /// the shader, so it is flush-immune and shared verbatim.
    fn model_cut_apply(
        hw: Hw,
        base: f32,
        err_in: f32,
        value: f32,
        source: f32,
        slack: f32,
        flush_additive: f32,
    ) -> (f32, f32) {
        let sum = hw.add(base, value);
        let gap = hw.mul(sum.abs(), U);
        let e = hw.add(hw.add(err_in, source), gap);
        let scaled = host_round_up_pos(hw.mul(e, slack));
        let err_out = host_round_up_pos(hw.add(scaled, flush_additive));
        (sum, err_out)
    }

    fn rat(x: f32) -> BigRational {
        BigRational::from_float(x).expect("finite f32 is an exact dyadic")
    }

    /// The exact enclosure demand: `err_out ≥ (err_in + source) + |sum − (base
    /// + value)|`, all rational.
    fn assert_encloses(hw: Hw, base: f32, err_in: f32, value: f32, source: f32) {
        let (sum, err_out) = model_cut_apply(
            hw,
            base,
            err_in,
            value,
            source,
            CUT_APPLY_SLACK,
            CUT_APPLY_FLUSH_ADDITIVE,
        );
        if !sum.is_finite() || !err_out.is_finite() {
            // The host driver refuses non-finite readback; nothing shipped.
            return;
        }
        let exact = rat(base) + rat(value);
        let demand = rat(err_in) + rat(source) + (rat(sum) - exact).abs();
        assert!(
            rat(err_out) >= demand,
            "cut-apply enclosure breach under {hw:?}: base={base:e} err_in={err_in:e} \
             value={value:e} source={source:e} sum={sum:e} err_out={err_out:e}"
        );
    }

    /// Adversarial operand table: subnormals in every slot, cancellation,
    /// exponent spread, exact zeros, boundary normals.
    fn operand_table() -> Vec<f32> {
        vec![
            0.0,
            f32::from_bits(1), // minsub
            -f32::from_bits(1),
            f32::from_bits(0x007f_ffff), // maxsub
            -f32::from_bits(0x007f_ffff),
            F32_MIN_NORMAL,
            -F32_MIN_NORMAL,
            2.0_f32.powi(-100),
            1.0e-8,
            0.999_999_94,
            1.0,
            -1.0,
            1.000_000_1,
            1024.0,
            -1024.0,
            1.0e10,
            -1.0e10,
            3.0e38,
        ]
    }

    fn nonneg_table() -> Vec<f32> {
        operand_table().into_iter().filter(|v| *v >= 0.0).collect()
    }

    #[test]
    fn model_tracks_the_shipped_cut_shader_text() {
        for line in [
            "let sum = base + value;",
            "let gap = abs(sum) * U;",
            "let e = (errs[i] + source) + gap;",
            "errs[i] = round_up_pos(round_up_pos(e * p.slack) + p.flush_additive);",
            "const U: f32 = 0.00000005960464477539063; // 2^-24",
            "if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }",
        ] {
            assert!(
                CUT_SHADOW_APPLY_SHADER.contains(line),
                "the oracle transcription no longer matches the shipped shader: \
                 missing line {line:?}"
            );
        }
    }

    #[test]
    fn slack_and_additive_constants_are_pinned() {
        // slack ≥ (1 + 2^-24)^4 exactly.
        let one_plus_u = BigRational::from_integer(BigInt::from(1))
            + BigRational::new(BigInt::from(1), BigInt::from(1_u64 << 24));
        let needed = one_plus_u.clone() * one_plus_u.clone() * one_plus_u.clone() * one_plus_u;
        assert!(
            rat(CUT_APPLY_SLACK) >= needed,
            "CUT_APPLY_SLACK no longer covers four RN under-roundings"
        );
        // additive = 32·2^-126 exactly, ≥ 2x margin over the 10-site flush demand.
        assert_eq!(
            CUT_APPLY_FLUSH_ADDITIVE.to_bits(),
            (f32::MIN_POSITIVE * 32.0).to_bits()
        );
        let site_demand = rat(F32_MIN_NORMAL) * BigRational::from_integer(BigInt::from(10));
        assert!(
            rat(CUT_APPLY_FLUSH_ADDITIVE)
                >= site_demand * BigRational::from_integer(BigInt::from(2)),
            "CUT_APPLY_FLUSH_ADDITIVE lost its 2x margin over the flush-site demand"
        );
    }

    #[test]
    fn exhaustive_grid_encloses_under_both_hardware_models() {
        let values = operand_table();
        let radii = nonneg_table();
        for &base in &values {
            for &value in &values {
                for &err_in in &radii {
                    for &source in &radii {
                        assert_encloses(IEEE, base, err_in, value, source);
                        assert_encloses(PURE_FLUSH, base, err_in, value, source);
                    }
                }
            }
        }
    }

    #[test]
    fn randomised_flush_hunt_holds_the_enclosure() {
        let values = operand_table();
        let radii = nonneg_table();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..65_536 {
            let base = values[next() as usize % values.len()];
            let value = values[next() as usize % values.len()];
            let err_in = radii[next() as usize % radii.len()];
            let source = radii[next() as usize % radii.len()];
            // Cancellation companion: value = -base exercises exact/near-exact
            // cancellation with subnormal residual mass.
            assert_encloses(PURE_FLUSH, base, err_in, value, source);
            assert_encloses(PURE_FLUSH, base, err_in, -base, source);
            assert_encloses(IEEE, base, err_in, value, source);
        }
    }

    /// The shipped-fails half: WITHOUT the additive floor the flush model
    /// breaches the enclosure (the additive is load-bearing, not decoration).
    #[test]
    fn additive_floor_is_load_bearing_under_flush() {
        // base subnormal, value zero: DAZ zeroes the operand, the exact sum is
        // `base`, the shipped center is 0, and with err_in = source = 0 the
        // whole slack chain reports 0 error.
        let base = f32::from_bits(0x007f_ffff); // maxsub
        let (sum, err_no_additive) =
            model_cut_apply(PURE_FLUSH, base, 0.0, 0.0, 0.0, CUT_APPLY_SLACK, 0.0);
        let exact = rat(base);
        let demand = (rat(sum) - exact).abs();
        assert!(
            rat(err_no_additive) < demand,
            "expected the additive-free model to under-report the DAZ loss"
        );
        // The shipped kernel (additive armed) encloses the same case.
        assert_encloses(PURE_FLUSH, base, 0.0, 0.0, 0.0);
    }

    /// The factor-encloses half: the slack factor alone covers the pure RN
    /// chain (no flush), pinned on a case whose error terms are all normal.
    #[test]
    fn slack_factor_covers_the_rn_chain_without_flush() {
        assert_encloses(IEEE, 1.0, 1.0e-7, 1.000_000_1, 1.0e-7);
        assert_encloses(IEEE, 3.5e4, 0.5, -1.25e3, 2.0e-4);
        // The selfcheck fixture is itself a subset of the oracle domain.
        let (centers, errs, channels) = cut_selfcheck_fixture();
        for i in 0..channels.len() {
            assert_encloses(
                IEEE,
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
            );
            let (model_center, model_err) = model_cut_apply(
                IEEE,
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
                CUT_APPLY_SLACK,
                CUT_APPLY_FLUSH_ADDITIVE,
            );
            let (host_center, host_err) = model_cut_apply_ieee(
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
                CUT_APPLY_SLACK,
                CUT_APPLY_FLUSH_ADDITIVE,
            );
            assert_eq!(model_center.to_bits(), host_center.to_bits());
            assert_eq!(model_err.to_bits(), host_err.to_bits());
        }
    }
}

// ---------------------------------------------------------------------------
// Device tests (real adapter; gpu-tests feature).
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    use super::super::super::test_support::gpu_test_serial_guard;
    use super::*;
    use ny_core::{GpuCrownLayer, GpuCrownSeed, GpuResnetSegment, ResidentCutShadowDisposition};

    static CUT_CAPABLE_DEVICE: OnceLock<std::result::Result<Arc<WgpuDevice>, String>> =
        OnceLock::new();

    /// A device with verdict authority of EITHER kind (fully-qualified or
    /// production charged). Any refusal other than the recognized explicit
    /// `NY_GPU_DENORM_PRESERVE=1` pin is a hard failure, mirroring the
    /// flush-charge acceptance harness.
    fn cut_capable_device() -> Option<Arc<WgpuDevice>> {
        let outcome = CUT_CAPABLE_DEVICE.get_or_init(|| {
            match WgpuDevice::new_for_verdict(
                super::super::sound_authority::WgpuVerdictRequest::new(),
            ) {
                Ok(device) => Ok(Arc::new(device)),
                Err(qualified_err) => WgpuDevice::new_for_verdict_flush_charged(
                    super::super::sound_authority::WgpuChargedVerdictRequest::new(),
                )
                .map(Arc::new)
                .map_err(|charged_err| {
                    format!("uncharged: {qualified_err}; charged: {charged_err}")
                }),
            }
        });
        match outcome {
            Ok(device) => Some(Arc::clone(device)),
            Err(error) => {
                assert!(
                    error.contains("NY_GPU_DENORM_PRESERVE"),
                    "no verdict-capable device armed for a reason other than the \
                     recognized explicit NY_GPU_DENORM_PRESERVE=1 pin: {error}"
                );
                println!(
                    "[cut-shadow gpu tests] PRECONDITION NOT MET: {error} — unset \
                     NY_GPU_DENORM_PRESERVE and re-run"
                );
                None
            }
        }
    }

    #[test]
    fn cut_apply_kernel_matches_host_transcription_and_contains() {
        let _guard = gpu_test_serial_guard();
        let Some(device) = cut_capable_device() else {
            return;
        };
        // Normal-only fixture: bit-exact against the IEEE transcription.
        let (centers, errs, channels) = cut_selfcheck_fixture();
        let (centers_out, errs_out) = device
            .resident_cut_apply_compact(&centers, &errs, &channels)
            .expect("pinned cut-apply dispatch");
        for i in 0..channels.len() {
            let (expect_center, expect_err) = model_cut_apply_ieee(
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
                CUT_APPLY_SLACK,
                CUT_APPLY_FLUSH_ADDITIVE,
            );
            assert_eq!(
                centers_out[i].to_bits(),
                expect_center.to_bits(),
                "center lane {i} diverged from the transcription"
            );
            assert_eq!(
                errs_out[i].to_bits(),
                expect_err.to_bits(),
                "error lane {i} diverged from the transcription"
            );
        }

        // Randomized + adversarial containment at zero tolerance (f64-exact
        // demand; f32 inputs are exact in f64 and the f64 sum of two f32 is
        // exact). Subnormal channel operands are included: on either hardware
        // class the shipped error must contain the exact demand.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let table = [
            0.0_f32,
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            1.0e-8,
            0.999_999_94,
            1.0,
            1024.0,
            1.0e10,
        ];
        let mut centers = Vec::new();
        let mut errs = Vec::new();
        let mut channels = Vec::new();
        for _ in 0..512 {
            let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
            let vsign = if next() & 2 == 0 { 1.0 } else { -1.0 };
            centers.push(sign * table[next() as usize % table.len()]);
            errs.push(table[next() as usize % table.len()].abs());
            channels.push(CutApplyChannel {
                value: vsign * table[next() as usize % table.len()],
                source: table[next() as usize % table.len()].abs(),
            });
        }
        let (centers_out, errs_out) = device
            .resident_cut_apply_compact(&centers, &errs, &channels)
            .expect("randomized cut-apply dispatch");
        for i in 0..channels.len() {
            let exact = f64::from(centers[i]) + f64::from(channels[i].value);
            let demand = f64::from(errs[i])
                + f64::from(channels[i].source)
                + (f64::from(centers_out[i]) - exact).abs();
            assert!(
                f64::from(errs_out[i]) >= demand,
                "containment breach at element {i}: base={} err={} value={} source={} \
                 center_out={} err_out={} demand={demand:e}",
                centers[i],
                errs[i],
                channels[i].value,
                channels[i].source,
                centers_out[i],
                errs_out[i],
            );
        }
    }

    #[test]
    fn selfcheck_pin_and_capability_wiring() {
        let _guard = gpu_test_serial_guard();
        let Some(device) = cut_capable_device() else {
            return;
        };
        assert!(
            device.verify_resident_cut_apply(),
            "the pinned cut-apply selfcheck must pass on a verdict-capable adapter"
        );
        assert!(device.resident_cut_shadow_capability());
        assert!(device.provides_resident_cut_shadow());

        // An ordinary (never-qualified) device must refuse the capability even
        // though its kernel arithmetic is identical.
        let ordinary = super::super::super::test_support::require_device();
        assert!(!ordinary.resident_cut_shadow_capability());
        assert!(!ordinary.provides_resident_cut_shadow());
    }

    /// The exact diamond fixture of the CUDA resident cut test, on the wgpu
    /// walk: relu(z0)+relu(z1) with the certified coupling cut tightens the
    /// binding row by ~1.0 while the baseline stays the only consumable bound.
    #[test]
    fn resident_cut_shadow_end_to_end_diamond() {
        let _guard = gpu_test_serial_guard();
        let Some(device) = cut_capable_device() else {
            return;
        };
        const ROWS: usize = 4;
        const WIDTH: usize = 8;
        let mut weight = vec![0.0_f32; WIDTH * WIDTH];
        for i in 0..WIDTH {
            weight[i * WIDTH + i] = 1.0;
        }
        // z0 = x0 + x1, z1 = x0 - x1: the exact diamond of the host oracle.
        weight[1] = 1.0;
        weight[WIDTH] = 1.0;
        weight[WIDTH + 1] = -1.0;
        let segments = vec![GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Activation {
                lower_slope: vec![0.0; WIDTH],
                upper_slope: vec![0.5; WIDTH],
                lower_intercept: vec![0.0; WIDTH],
                upper_intercept: vec![1.0; WIDTH],
                num_neurons: WIDTH,
            },
            GpuCrownLayer::Linear {
                weight: Arc::from(weight),
                bias: None,
                out_features: WIDTH,
                in_features: WIDTH,
                cert_err: Default::default(),
            },
        ])];
        let mut seed_rows = vec![0.0_f32; ROWS * WIDTH];
        for row in 0..ROWS {
            seed_rows[row * WIDTH] = -1.0;
            seed_rows[row * WIDTH + 1] = -1.0;
        }
        let seed = GpuCrownSeed {
            lower_a: Arc::from(seed_rows.clone()),
            upper_a: Arc::from(seed_rows),
            lower_b: Arc::from(vec![0.0_f32; ROWS]),
            upper_b: Arc::from(vec![0.0_f32; ROWS]),
            num_specs: ROWS,
            current_dim: WIDTH,
        };
        let input_lower = vec![-1.0_f32; WIDTH];
        let input_upper = vec![1.0_f32; WIDTH];
        let beta_signed = vec![vec![0.0_f32; WIDTH]];

        let baseline = device
            .crown_backward_gpu_resnet_sound_beta(
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
            )
            .expect("ordinary wgpu beta baseline");

        let expired_deadline = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one second of monotonic history");
        let disabled = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Disabled,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                None,
                usize::MAX,
                expired_deadline,
            )
            .expect("disabled wgpu cut route");
        assert_eq!(
            disabled.disposition(),
            ResidentCutShadowDisposition::Disabled
        );
        for (got, expected) in disabled
            .baseline()
            .lower_bounds
            .iter()
            .chain(&disabled.baseline().upper_bounds)
            .zip(baseline.lower_bounds.iter().chain(&baseline.upper_bounds))
        {
            assert_eq!(got.to_bits(), expected.to_bits());
        }

        let channel = |value| {
            ny_core::ResidentLowerCutChannel::try_new(value, 0.0).expect("exact test channel")
        };
        let carrier_at = |deadline| {
            ResidentLowerCutCarrier::try_new(
                0,
                WIDTH,
                [0, 1],
                (0..ROWS)
                    .map(|_| {
                        ny_core::ResidentLowerCutRow::try_new(
                            vec![1.0],
                            [channel(-0.5), channel(-0.5)],
                            [channel(1.0), channel(1.0)],
                            channel(-1.0),
                        )
                        .expect("complete diamond cut row")
                    })
                    .collect(),
                deadline,
            )
            .expect("complete diamond cut carrier")
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        let carrier = carrier_at(deadline);
        let observed = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&carrier),
                0,
                deadline,
            )
            .expect("complete resident cut observation");
        assert_eq!(
            observed.disposition(),
            ResidentCutShadowDisposition::Observed
        );
        let observation = observed.observation().expect("complete telemetry");
        assert!(
            observation.delta() > 0.5,
            "diamond cut must materially tighten the observed lower row: {observation:?}"
        );
        for (got, expected) in observed
            .baseline()
            .lower_bounds
            .iter()
            .chain(&observed.baseline().upper_bounds)
            .zip(baseline.lower_bounds.iter().chain(&baseline.upper_bounds))
        {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "the consumable baseline must stay bit-identical to the plain call"
            );
        }

        // Zero-multiplier carriers, wrong-deadline carriers, and out-of-range
        // binding rows are telemetry misses that retain the exact baseline.
        let zero_carrier = ResidentLowerCutCarrier::try_new(
            0,
            WIDTH,
            [0, 1],
            (0..ROWS)
                .map(|_| {
                    ny_core::ResidentLowerCutRow::try_new(
                        vec![0.0],
                        [channel(0.0), channel(0.0)],
                        [channel(0.0), channel(0.0)],
                        channel(0.0),
                    )
                    .expect("zero cut row")
                })
                .collect(),
            deadline,
        )
        .expect("zero cut carrier");
        let rejected = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&zero_carrier),
                0,
                deadline,
            )
            .expect("zero-multiplier shadow retains baseline");
        assert_eq!(
            rejected.disposition(),
            ResidentCutShadowDisposition::Rejected
        );
        assert!(rejected.observation().is_none());

        let other_deadline = deadline + Duration::from_secs(5);
        let mismatched = carrier_at(other_deadline);
        let rejected = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&mismatched),
                0,
                deadline,
            )
            .expect("deadline-mismatched shadow retains baseline");
        assert_eq!(
            rejected.disposition(),
            ResidentCutShadowDisposition::Rejected
        );

        let rejected = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&carrier_at(deadline)),
                ROWS,
                deadline,
            )
            .expect("out-of-range binding row retains baseline");
        assert_eq!(
            rejected.disposition(),
            ResidentCutShadowDisposition::Rejected
        );

        // An expired explicit deadline refuses before any replay or mutation.
        let expired_carrier = carrier_at(expired_deadline);
        let err = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&expired_carrier),
                0,
                expired_deadline,
            )
            .expect_err("late shadow must return before any replay");
        assert!(err.is_deadline_exceeded());

        // A target activation outside the fold is a telemetry miss.
        let out_of_fold = ResidentLowerCutCarrier::try_new(
            7,
            WIDTH,
            [0, 1],
            (0..ROWS)
                .map(|_| {
                    ny_core::ResidentLowerCutRow::try_new(
                        vec![1.0],
                        [channel(-0.5), channel(-0.5)],
                        [channel(1.0), channel(1.0)],
                        channel(-1.0),
                    )
                    .expect("out-of-fold cut row")
                })
                .collect(),
            deadline,
        )
        .expect("out-of-fold carrier");
        let rejected = device
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
                Some(&out_of_fold),
                0,
                deadline,
            )
            .expect("out-of-fold target retains baseline");
        assert_eq!(
            rejected.disposition(),
            ResidentCutShadowDisposition::Rejected
        );

        // The hook slot is clear after every disposition: a fresh plain beta
        // call must reproduce the baseline bit-for-bit.
        let replay = device
            .crown_backward_gpu_resnet_sound_beta(
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &beta_signed,
                &[],
                &[],
            )
            .expect("post-shadow plain beta replay");
        for (got, expected) in replay
            .lower_bounds
            .iter()
            .chain(&replay.upper_bounds)
            .zip(baseline.lower_bounds.iter().chain(&baseline.upper_bounds))
        {
            assert_eq!(got.to_bits(), expected.to_bits());
        }
    }
}
