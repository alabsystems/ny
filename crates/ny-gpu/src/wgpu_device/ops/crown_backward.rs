// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward pass orchestration (#3397).
//!
//! Keeps A-matrices on GPU across the entire backward loop, dispatching
//! activation backward, bias accumulation, GEMM, and concretization shaders.
//! Only the final concretized bounds are read back to the host.
//!
//! **Batched command encoding (#3397):** All dispatch passes are encoded into
//! a single `CommandEncoder` and submitted with one `queue.submit` call. Static
//! params/weights/biases now live in a cached device staging buffer keyed by
//! layer topology + static data; each invocation only refreshes activation
//! relaxations before the encoder copies staged data into the working buffers.
//!
//! Conv2d support (#3397): Conv2d layers are handled via reshape + GEMM + col2im
//! gather, all on GPU. No host roundtrip for Conv2d layers.
//!
//! Reference: designs/2026-03-06-gpu-crown-backward.md
//! Reference: alpha-beta-CROWN `auto_LiRPA/backward_bound.py`

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ny_core::wide_lane_telemetry::{
    note_wide_lane_attempt, note_wide_lane_decline, note_wide_lane_decline_none, WideLaneDecline,
};
use ny_core::{
    GpuCrownBackward, GpuCrownGradResult, GpuCrownLayer, GpuCrownResult, GpuCrownSeed,
    GpuCrownTrajectoryResult, GpuIntermediateSweepRequest, GpuIntermediateSweepResourcePolicy,
    GpuIntermediateSweepResult, GpuResidentCoeffBatched, GpuResnetBatchedDomainRef,
    GpuResnetSegment, NyError, Result,
};

use super::super::WgpuDevice;
use super::crown_backward_encode::count_encoded_compute_passes;
use super::crown_backward_types::{layer_input_dim, layer_output_dim};
use super::crown_host_profile::CrownHostTimingProfile;
use super::crown_memory_estimate::{
    estimate_crown_backward_memory, gpu_memory_budget_bytes, max_specs_per_budget,
};
use super::crown_timestamps::{CrownGpuTimingProfile, CrownTimestampProfiler};
use super::gemm::{GEMM_SMALL_K_THRESHOLD, GEMM_TILE_DIM, SMALL_K_ROWS_PER_THREAD};
use super::sanitize_readback;

mod batching;

/// wgpu default `max_storage_buffer_binding_size` (128 MB). This is more
/// restrictive than `max_buffer_size` (256 MB) because each bind group entry
/// references a buffer range that must fit within the binding limit.
const WGPU_MAX_BINDING_BYTES: usize = 134_217_728;
const WGPU_MAX_DISPATCH_WORKGROUPS: usize = 65_535;
const WGPU_1D_DISPATCH_THREADS: usize = WGPU_MAX_DISPATCH_WORKGROUPS * 256;

const SWEEP_MIB: usize = 1024 * 1024;
const SWEEP_MIN_BINDING_BYTES: usize = SWEEP_MIB;
const SWEEP_MIN_ROWS_PER_TARGET: usize = 8;

/// Build a conservative automatic scheduling profile from live WGPU facts.
///
/// WGPU does not expose physical VRAM. Device type therefore supplies only a
/// conservative class ceiling, while the granted single-buffer/binding limits
/// choose the useful starting row tier. The exact sweep planner remains the
/// final capacity authority and may cleanly decline any whole request before
/// allocation or dispatch.
fn automatic_intermediate_sweep_resource_policy(
    device_type: wgpu::DeviceType,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    backend_budget_bytes: usize,
) -> Option<GpuIntermediateSweepResourcePolicy> {
    if device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    let max_buffer_size = usize::try_from(max_buffer_size).ok()?;
    let binding_bytes = max_buffer_size.min(usize::try_from(max_storage_buffer_binding_size).ok()?);
    if binding_bytes < SWEEP_MIN_BINDING_BYTES {
        return None;
    }

    let (class_bytes, class_rows) = match device_type {
        wgpu::DeviceType::DiscreteGpu => (8 * 1024 * SWEEP_MIB, 32),
        wgpu::DeviceType::IntegratedGpu => (2 * 1024 * SWEEP_MIB, 16),
        wgpu::DeviceType::VirtualGpu => (1024 * SWEEP_MIB, 8),
        wgpu::DeviceType::Other => (512 * SWEEP_MIB, 8),
        wgpu::DeviceType::Cpu => unreachable!("CPU adapters returned above"),
    };
    // #comprehensive-rows-probe (measurement-only): the class table is a proxy
    // for "how much memory does this device class usually have", and on a unified
    // -memory part it is badly wrong in one direction — an `IntegratedGpu` here
    // gets 2 GiB / 16 rows on a board with 121 GiB shared. The comprehensive root
    // sweep is memory-bound in ROWS (1.4 GiB peak at 144 rows), and 16 rows/target
    // is 0.26% coverage of the eligible neurons, which is why the sweep completes
    // atomically and still moves the root census by nothing.
    //
    // Raising this is NOT obviously safe on this box: the 121 GiB is shared with
    // the host, and an earlier over-allocation caused a global OOM. So it is
    // deliberately an explicit opt-in for measurement, never a default, and the
    // backend preflight still computes exact simultaneous liveness and refuses
    // anything it cannot honour. Its purpose is to establish the memory-vs-rows
    // SCALING LAW, which decides between one wide sweep and row-chunked
    // accumulation. Absent/malformed values leave the shipped class policy.
    let (class_bytes, class_rows) = match (
        ny_levers::read(&ny_levers::decls::comprehensive_rows::SWEEP_CLASS_MIB)
            .value
            .as_u64()
            .and_then(|mib| usize::try_from(mib).ok())
            .filter(|mib| *mib > 0),
        ny_levers::read(&ny_levers::decls::comprehensive_rows::SWEEP_CLASS_ROWS)
            .value
            .as_u64()
            .and_then(|rows| usize::try_from(rows).ok())
            .filter(|rows| *rows > 0),
    ) {
        (None, None) => (class_bytes, class_rows),
        (mib, rows) => (
            mib.map_or(class_bytes, |mib| mib.saturating_mul(SWEEP_MIB)),
            rows.unwrap_or(class_rows),
        ),
    };
    // A sweep carrier is a bounded family of independently limit-checked
    // buffers, not one giant binding. This multiplier merely prevents a tiny
    // granted binding from inheriting a much larger class-wide ceiling; exact
    // simultaneous liveness is still computed by the backend preflight.
    let binding_scaled_ceiling = binding_bytes.saturating_mul(16);
    let policy = GpuIntermediateSweepResourcePolicy {
        max_device_bytes: class_bytes
            .min(backend_budget_bytes)
            .min(binding_scaled_ceiling),
        preferred_rows_per_target: class_rows,
        minimum_rows_per_target: SWEEP_MIN_ROWS_PER_TARGET,
    };
    policy.is_valid().then_some(policy)
}

/// Honest bounded-rows capacity for the deadline-bounded ResNet sound entries
/// (`deadline_bounded_resnet_sound_max_rows`) = the FULL audited K=8 contract
/// cap ([`ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS`]).
///
/// HONESTY ARGUMENT (why 8 rows, not fewer):
/// - Row capacity: the resident sound fold is ROW-BATCHED — the batched-BaB
///   wide lane already drives the SAME kernels + certified-error folds with
///   `n_domains × num_specs` STACKED rows (admitted up to 512 rows by
///   ny-propagate's `try_gpu_beta_batched_resnet_opt`, with
///   `sound_spec_row_chunk` splitting anything past a device binding/dispatch
///   limit). Every row is an independent certified enclosure, so 8 rows sit far
///   inside limits this backend already validates in production.
/// - Deadline: `honors_crown_backward_deadline()` is `true` — the resident fold
///   polls `crown_backward_deadline_expired()` between layers
///   (crown_backward_sound_resident.rs) and between spec-row chunks
///   (crown_backward/batching.rs). The deadline-bounded entries below arm a
///   CALL-LOCAL scope feeding that same poll, pre-check before dispatch, and
///   refuse to publish a late result — the trait's documented contract.
/// - Soundness: the entries delegate to `crown_backward_gpu_resnet_sound_inner`,
///   i.e. the certified-error resident path (`γ_k·S` combine + outward-rounded
///   host folds) — NEVER the fast round-to-nearest tier.
pub(crate) const WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS: usize =
    ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;

/// #batched-bab wide-lane taken counter — PRODUCTION-ARMED (not test-only):
/// incremented exactly once per batch whenever the ONE-pass wide resident fold
/// produced the published result (bound-only wrapper or the β-opt grad lane;
/// single-group or sub-chunked). Device tests assert lane-taken against the
/// REAL routing instead of trusting probe stderr; monotonic, never reset. The
/// independently gated margin-row coefficient batch uses its own profile
/// counters and is deliberately excluded from this candidate denominator.
static WIDE_RESNET_BATCHED_TAKEN: AtomicU64 = AtomicU64::new(0);

/// Monotonic count of wide one-pass batched ResNet dispatches that produced the
/// published result (see [`WIDE_RESNET_BATCHED_TAKEN`]).
/// Public for the CLI's dark `[wide-lane]` readout (`NY_BETA_GPU_PROBE=1`):
/// a lane that silently never fires must be distinguishable from one doing
/// the work. Observability only — reading it changes nothing.
pub fn wide_resnet_batched_taken_count() -> u64 {
    WIDE_RESNET_BATCHED_TAKEN.load(Ordering::Relaxed)
}

/// #batched-bab HOLE 7 homogeneity gate: two domains may batch together only if
/// their segment lists share the SAME network skeleton — identical variant
/// sequence, per-layer variant + dims, and the SAME shared-weight `Arc`s
/// (`Arc::ptr_eq`) — differing ONLY in per-domain relaxation VALUES (Activation
/// slopes/intercepts, MaxPool routing/ibp). Any structural difference or a
/// distinct weight `Arc` returns false → the batched call aborts to the serial
/// fallback (never packs ragged rows). Conservative: an unrecognized layer-pair
/// returns false (batch declined, serial still sound).
fn resnet_skeleton_matches(a: &[GpuResnetSegment], b: &[GpuResnetSegment]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(sa, sb)| resnet_segment_matches(sa, sb))
}

fn resnet_segment_matches(a: &GpuResnetSegment, b: &GpuResnetSegment) -> bool {
    use GpuResnetSegment::{Chain, Residual, ResidualProj};
    match (a, b) {
        (Chain(la), Chain(lb)) | (Residual(la), Residual(lb)) => layers_skeleton_match(la, lb),
        (ResidualProj(fa, pa), ResidualProj(fb, pb)) => {
            layers_skeleton_match(fa, fb) && layers_skeleton_match(pa, pb)
        }
        _ => false,
    }
}

fn layers_skeleton_match(a: &[GpuCrownLayer], b: &[GpuCrownLayer]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(la, lb)| layer_skeleton_matches(la, lb))
}

/// #batched-bab: two shared-weight Arcs match if they are the SAME allocation
/// (`ptr_eq`, the fast path) OR by-value equal. The by-value fallback is load-bearing
/// for the BaB per-domain path: the resnet extraction
/// (`network/core/sequential/crown/gpu_extraction.rs`) mints a FRESH `Arc<[f32]>` per
/// domain via `kernel_slice.to_vec().into()`, so `ptr_eq` fails across domains even
/// though every domain is the SAME network (BaB splits the input/ReLU state, never the
/// weights). Without the fallback the homogeneity gate rejects EVERY real batch → the
/// batched/wide path is dark. SOUND: the wide fold uses domain 0's weights for all rows,
/// which is correct precisely when the weights are equal (what this verifies). Cost:
/// an O(len) compare only when `ptr_eq` misses; negligible vs the batched GPU pass it
/// enables. (A future extraction that Arc-shares weights restores the free `ptr_eq`.)
fn arc_slice_eq(a: &std::sync::Arc<[f32]>, b: &std::sync::Arc<[f32]>) -> bool {
    std::sync::Arc::ptr_eq(a, b) || a == b
}

fn arc_opt_slice_eq(a: &Option<std::sync::Arc<[f32]>>, b: &Option<std::sync::Arc<[f32]>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => arc_slice_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn layer_skeleton_matches(a: &GpuCrownLayer, b: &GpuCrownLayer) -> bool {
    use GpuCrownLayer::{Activation, ActivationReluDualAlpha, Conv2d, Linear, MaxPool2d};
    match (a, b) {
        (
            Linear {
                weight: wa,
                bias: ba,
                out_features: oa,
                in_features: ia,
                cert_err: cea,
            },
            Linear {
                weight: wb,
                bias: bb,
                out_features: ob,
                in_features: ib,
                cert_err: ceb,
            },
            // #cert-err is part of the SKELETON, not a per-domain value: two layers
            // that differ only in their declared BN-fold error charge different
            // certified radii, so reusing one's plan for the other would publish a
            // bound built from the wrong charge.
        ) => oa == ob && ia == ib && cea == ceb && arc_slice_eq(wa, wb) && arc_opt_slice_eq(ba, bb),
        // Relaxation slopes/intercepts are per-domain VALUES — only the shape matters.
        (
            Activation {
                num_neurons: na, ..
            },
            Activation {
                num_neurons: nb, ..
            },
        ) => na == nb,
        (
            ActivationReluDualAlpha {
                num_neurons: na, ..
            },
            ActivationReluDualAlpha {
                num_neurons: nb, ..
            },
        ) => na == nb,
        (
            Conv2d {
                weight_col: wa,
                bias_expanded: ba,
                out_channels: oca,
                in_channels: ica,
                kernel_h: kha,
                kernel_w: kwa,
                stride_h: sha,
                stride_w: swa,
                pad_h: pha,
                pad_w: pwa,
                out_h: oha,
                out_w: owa,
                in_h: iha,
                in_w: iwa,
                cert_err: cea,
            },
            Conv2d {
                weight_col: wb,
                bias_expanded: bb,
                out_channels: ocb,
                in_channels: icb,
                kernel_h: khb,
                kernel_w: kwb,
                stride_h: shb,
                stride_w: swb,
                pad_h: phb,
                pad_w: pwb,
                out_h: ohb,
                out_w: owb,
                in_h: ihb,
                in_w: iwb,
                cert_err: ceb,
            },
            // #cert-err is part of the SKELETON (see the Linear arm above).
        ) => {
            cea == ceb
                && oca == ocb
                && ica == icb
                && kha == khb
                && kwa == kwb
                && sha == shb
                && swa == swb
                && pha == phb
                && pwa == pwb
                && oha == ohb
                && owa == owb
                && iha == ihb
                && iwa == iwb
                && arc_slice_eq(wa, wb)
                && arc_opt_slice_eq(ba, bb)
        }
        // MaxPool routing/ibp are per-domain (depend on this domain's bounds);
        // only the pool geometry (dims) is shared.
        (
            MaxPool2d {
                input_dim: ia,
                output_dim: oa,
                ..
            },
            MaxPool2d {
                input_dim: ib,
                output_dim: ob,
                ..
            },
        ) => ia == ib && oa == ob,
        _ => false,
    }
}

/// #batched-bab HOLE 8: the wide resident fold handles ONLY Linear / Activation /
/// Conv2d. `ActivationReluDualAlpha` (packed 4×slopes) and `MaxPool2d` (per-domain
/// routing + ibp bounds) have backward shaders that are NOT domain-block-indexed, so
/// a wide pass over them would broadcast domain 0's relaxation/routing → a false
/// VERIFIED. Decline any batch containing them (→ serial per-domain fallback, which
/// dispatches each domain's own shader soundly). The single-domain resnet fold also
/// hard-errors on these variants, so this is a fail-closed early decline, not the only
/// backstop.
fn segments_contain_unbatchable(segments: &[GpuResnetSegment]) -> bool {
    let has = |ls: &[GpuCrownLayer]| {
        ls.iter().any(|l| {
            matches!(
                l,
                GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. }
            )
        })
    };
    segments.iter().any(|s| match s {
        GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => has(l),
        GpuResnetSegment::ResidualProj(f, p) => has(f) || has(p),
    })
}

/// #wg-limit-subchunk (SOUNDNESS): the widest 1-D compute dispatch the resident wide
/// fold issues for a domain batch is `ceil(N * W / 256)` workgroups — `N` = total
/// stacked spec rows (`n_domains * num_specs_per_dom`), `W` = the widest per-domain
/// layer coefficient width — plus the sound concretize's `N`-workgroup dispatch. `W`
/// is the max over Linear in/out features, Activation neurons, and each Conv2d's
/// im2col width (`out_channels*out_h*out_w`) / col2im width (`in_channels*in_h*in_w`).
/// This returns that per-domain `W` so the caller can bound `N` against
/// `max_compute_workgroups_per_dimension` and sub-chunk the batch to fit. The conv
/// GEMM is 2-D-dispatched (`select_gemm_dispatch`, which caps its own M via the small-K
/// shader), so it stays within the per-dimension cap for any `N` in the sound-BaB range
/// and is deliberately not counted here.
fn resnet_segments_max_1d_dispatch_dim(segments: &[GpuResnetSegment]) -> usize {
    fn visit(ls: &[GpuCrownLayer], m: &mut usize) {
        for l in ls {
            match l {
                GpuCrownLayer::Linear {
                    out_features,
                    in_features,
                    ..
                } => {
                    *m = (*m).max(*out_features).max(*in_features);
                }
                GpuCrownLayer::Activation { num_neurons, .. } => {
                    *m = (*m).max(*num_neurons);
                }
                GpuCrownLayer::Conv2d {
                    out_channels,
                    in_channels,
                    out_h,
                    out_w,
                    in_h,
                    in_w,
                    ..
                } => {
                    let reshape = out_channels.saturating_mul(*out_h).saturating_mul(*out_w);
                    let col2im = in_channels.saturating_mul(*in_h).saturating_mul(*in_w);
                    *m = (*m).max(reshape).max(col2im);
                }
                _ => {}
            }
        }
    }
    let mut m = 1usize;
    for s in segments {
        match s {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => visit(l, &mut m),
            GpuResnetSegment::ResidualProj(f, p) => {
                visit(f, &mut m);
                visit(p, &mut m);
            }
        }
    }
    m
}

/// #wg-limit-subchunk (SOUNDNESS): max stacked spec rows `N` the resident wide fold may
/// dispatch WITHOUT exceeding the wgpu `max_compute_workgroups_per_dimension` limit,
/// given the widest per-domain 1-D dispatch width `W` (see
/// [`resnet_segments_max_1d_dispatch_dim`]). Bounds BOTH the elementwise passes
/// (`ceil(N*W/256) <= max_wg`) and the sound concretize (`N <= max_wg`). The device
/// keeps this cap at the wgpu default (65535) even under `NY_GPU_BIG_BINDINGS` (which
/// only raises the *binding-size* limit), so overrunning it is UB on some drivers
/// (silent over-tight — UNSOUND — bound, or a crash). `NY_WIDE_MAX_STACKED_ROWS=<n>`
/// may only LOWER the cap (diagnostics / forcing the sub-chunk path in tests); it never
/// raises it.
fn wide_max_safe_stacked_rows(max_wg: usize, width: usize) -> usize {
    let width = width.max(1);
    let elem_bound = max_wg.saturating_mul(256) / width;
    // Zero is meaningful: even one stacked row would exceed the elementwise
    // dispatch limit.  The caller must decline the wide path in that case.
    let mut n = elem_bound.min(max_wg);
    if let Some(cap) = std::env::var("NY_WIDE_MAX_STACKED_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| c >= 1)
    {
        n = n.min(cap);
    }
    n
}

/// #batched-bab HOLE-7 SUB-GROUPING gate — DARK, default OFF.
///
/// ON (`NY_BAB_RESNET_WIDE_SUBGROUP=1`) lets a HETEROGENEOUS batch be split into
/// maximal homogeneous runs and folded wide run-by-run instead of abandoning the
/// whole wave to the serial path (see
/// [`WgpuDevice::try_wide_resnet_batched_subgrouped`]). OFF ⇒ byte-identical
/// routing to today: a heterogeneous batch returns the historical `Err` and the
/// caller runs its proven per-domain loop.
///
/// It is deliberately not `env_gate_default_on`-style: this lane changes WHICH
/// kernel produces a verdict-bearing bound, so arming it is a measured-session
/// decision, not a default.
fn wide_subgroup_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::wide_lane::BAB_RESNET_WIDE_SUBGROUP)
        .value
        .as_bool()
}

/// Legacy-armed global selector for every wide ResNet CROWN dispatch.
///
/// Exact `0` disables the lane; absence and every other spelling retain the
/// shipped ON behavior. The declaration preserves that compatibility contract
/// while keeping its unqualified-default debt explicit in the central registry.
fn wide_resnet_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::wide_lane::BAB_RESNET_WIDE)
        .value
        .as_bool()
}

/// Number of whole domains that fit in a device-safe wide group.  A domain is
/// indivisible because all of its specification rows share one relaxation
/// block, so an oversized single domain must fall back instead of being issued
/// as an invalid dispatch.
fn wide_safe_domain_count(max_wg: usize, width: usize, specs_per_domain: usize) -> Option<usize> {
    if specs_per_domain == 0 {
        return None;
    }
    let safe_rows = wide_max_safe_stacked_rows(max_wg, width);
    (safe_rows >= specs_per_domain).then_some(safe_rows / specs_per_domain)
}

/// Select the per-domain auxiliary rows corresponding to one domain chunk.
/// Empty means that the optional channel is disabled; any other outer shape
/// must cover the complete domain batch before it may be sliced.
fn wide_domain_table_chunk<T>(
    table: &[T],
    total_domains: usize,
    start: usize,
    end: usize,
) -> Option<&[T]> {
    if table.is_empty() {
        Some(table)
    } else if table.len() == total_domains {
        table.get(start..end)
    } else {
        None
    }
}

/// #batched-bab: stack `n_domains` domains' per-layer state into ONE wide layer vec.
/// `Activation` slopes/intercepts are concatenated CONTIGUOUSLY (domain `d`'s block at
/// `d*num_neurons`), so `CROWN_ACTIVATION_RESIDENT_SHADER` reads domain `dom`'s slope
/// at `dom*num_neurons + i`. Shared `Linear`/`Conv2d` layers (weights `Arc::ptr_eq`
/// across domains under the homogeneity gate) are cloned from domain 0. Returns `None`
/// on any structural mismatch (caller falls back to the serial path — always sound).
fn stack_wide_layers(per_domain: &[&[GpuCrownLayer]]) -> Option<Vec<GpuCrownLayer>> {
    let template = per_domain.first()?;
    let mut out = Vec::with_capacity(template.len());
    for (li, l0) in template.iter().enumerate() {
        match l0 {
            GpuCrownLayer::Activation { num_neurons, .. } => {
                let nn = *num_neurons;
                let (mut ls, mut us, mut lint, mut uint) = (
                    Vec::with_capacity(per_domain.len() * nn),
                    Vec::with_capacity(per_domain.len() * nn),
                    Vec::with_capacity(per_domain.len() * nn),
                    Vec::with_capacity(per_domain.len() * nn),
                );
                for dom in per_domain {
                    match dom.get(li)? {
                        GpuCrownLayer::Activation {
                            lower_slope,
                            upper_slope,
                            lower_intercept,
                            upper_intercept,
                            num_neurons: nnd,
                        } => {
                            if *nnd != nn
                                || lower_slope.len() != nn
                                || upper_slope.len() != nn
                                || lower_intercept.len() != nn
                                || upper_intercept.len() != nn
                            {
                                return None;
                            }
                            ls.extend_from_slice(lower_slope);
                            us.extend_from_slice(upper_slope);
                            lint.extend_from_slice(lower_intercept);
                            uint.extend_from_slice(upper_intercept);
                        }
                        _ => return None,
                    }
                }
                out.push(GpuCrownLayer::Activation {
                    lower_slope: ls,
                    upper_slope: us,
                    lower_intercept: lint,
                    upper_intercept: uint,
                    num_neurons: nn,
                });
            }
            // Shared network layers — weights are Arc::ptr_eq across the batch.
            GpuCrownLayer::Linear { .. } | GpuCrownLayer::Conv2d { .. } => out.push(l0.clone()),
            // Unbatchable variants (declined earlier); bail defensively.
            _ => return None,
        }
    }
    Some(out)
}

/// #batched-bab: build the shared wide skeleton (stacked per-domain Activation slopes)
/// from all domains' segment lists. Requires the homogeneity gate to have passed
/// (identical variant sequence). Returns `None` on any mismatch → serial fallback.
fn stack_wide_segments(domains: &[GpuResnetBatchedDomainRef<'_>]) -> Option<Vec<GpuResnetSegment>> {
    let template = domains.first()?.segments;
    let mut out = Vec::with_capacity(template.len());
    for (si, seg0) in template.iter().enumerate() {
        // Collect each domain's branch layer-slices for this segment position.
        let branch = |pick: fn(&GpuResnetSegment) -> Option<&[GpuCrownLayer]>| -> Option<Vec<&[GpuCrownLayer]>> {
            domains
                .iter()
                .map(|d| pick(d.segments.get(si)?))
                .collect::<Option<Vec<_>>>()
        };
        let seg = match seg0 {
            GpuResnetSegment::Chain(_) => {
                let per = branch(|s| match s {
                    GpuResnetSegment::Chain(l) => Some(l.as_slice()),
                    _ => None,
                })?;
                GpuResnetSegment::Chain(stack_wide_layers(&per)?)
            }
            GpuResnetSegment::Residual(_) => {
                let per = branch(|s| match s {
                    GpuResnetSegment::Residual(l) => Some(l.as_slice()),
                    _ => None,
                })?;
                GpuResnetSegment::Residual(stack_wide_layers(&per)?)
            }
            GpuResnetSegment::ResidualProj(_, _) => {
                let per_f = branch(|s| match s {
                    GpuResnetSegment::ResidualProj(f, _) => Some(f.as_slice()),
                    _ => None,
                })?;
                let per_p = branch(|s| match s {
                    GpuResnetSegment::ResidualProj(_, p) => Some(p.as_slice()),
                    _ => None,
                })?;
                GpuResnetSegment::ResidualProj(
                    stack_wide_layers(&per_f)?,
                    stack_wide_layers(&per_p)?,
                )
            }
        };
        out.push(seg);
    }
    Some(out)
}

/// #batched-bab: stack a per-domain `&[Vec<f32>]` table (per-ReLU β or per-node/segment
/// abs-max, aligned in fold order across domains) into `n_domains`-block wide entries —
/// entry `k` = domain 0's block ++ domain 1's block ++ … VALIDATES that every domain has
/// the same entry count AND the same per-entry block length (they must, under the
/// homogeneity gate); returns `None` on any mismatch so the caller falls back to the
/// serial path rather than fold one domain's rows against a mis-sized block (which the
/// fold's `fab.len() == d*n_domains` guard would silently skip, dropping the error term
/// ⇒ a tighter, unsound bound). Empty input (all domains empty) → `Some(empty)`.
fn stack_wide_table(per_domain: &[&[Vec<f32>]]) -> Option<Vec<Vec<f32>>> {
    let first = per_domain.first()?;
    let n_entries = first.len();
    if per_domain.iter().any(|d| d.len() != n_entries) {
        return None;
    }
    let mut out = Vec::with_capacity(n_entries);
    for k in 0..n_entries {
        let block_len = per_domain[0][k].len();
        let mut v = Vec::with_capacity(per_domain.len() * block_len);
        for d in per_domain {
            if d[k].len() != block_len {
                return None;
            }
            v.extend_from_slice(&d[k]);
        }
        out.push(v);
    }
    Some(out)
}

/// Compute the max specs that fit in a single GPU dispatch without exceeding
/// the wgpu max storage buffer binding size.  Returns `num_specs` when no
/// batching is needed.
fn max_specs_per_batch(
    layers: &[GpuCrownLayer],
    num_specs: usize,
    first_dim: usize,
    max_binding_bytes: usize,
) -> usize {
    // BufferPool applies a 1.2× growth factor, so effective max is smaller.
    //
    // #hard-caps: `max_binding_bytes` is the LIVE device limit, not the
    // hard-coded 128 MiB this used to read. `WgpuDevice::new` requests the
    // adapter's real limits, and on an Apple M4 Pro that is 4095 MiB — so this
    // was batching against a 32x under-estimate and issuing far more dispatches
    // than the device required. Its sibling at `resident_spec_cap` already read
    // the live limit; this one did not.
    let effective_max = (max_binding_bytes as f64 / 1.2) as usize;
    let max_elems_for_f32 = effective_max / size_of::<f32>();

    // Per-spec element counts for the largest buffers:
    let max_dim = layers
        .iter()
        .filter_map(|l| layer_input_dim(l).ok())
        .chain(std::iter::once(first_dim))
        .max()
        .unwrap_or(first_dim);

    // A-matrix: num_specs × max_dim
    let a_per_spec = max_dim;

    // Conv GEMM output: worst case per spec
    let conv_gemm_per_spec = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Conv2d {
                in_channels,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
                ..
            } => Some(out_h * out_w * in_channels * kernel_h * kernel_w),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    // Conv reshaped: worst case per spec
    let conv_reshaped_per_spec = layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Conv2d {
                out_channels,
                out_h,
                out_w,
                ..
            } => Some(out_h * out_w * out_channels),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let worst_per_spec = a_per_spec
        .max(conv_gemm_per_spec)
        .max(conv_reshaped_per_spec)
        .max(1);

    let max_batch = max_elems_for_f32 / worst_per_spec;
    max_batch.clamp(1, num_specs)
}

fn max_specs_for_1d_workgroups(per_spec_threads: usize, num_specs: usize) -> usize {
    if per_spec_threads == 0 {
        return num_specs;
    }
    if per_spec_threads > WGPU_1D_DISPATCH_THREADS {
        return 0;
    }
    (WGPU_1D_DISPATCH_THREADS / per_spec_threads).clamp(1, num_specs)
}

fn max_specs_for_gemm_dispatch(
    k: usize,
    n: usize,
    per_spec_rows: usize,
    num_specs: usize,
) -> usize {
    if per_spec_rows == 0 {
        return num_specs;
    }
    if n.div_ceil(GEMM_TILE_DIM) > WGPU_MAX_DISPATCH_WORKGROUPS {
        return 0;
    }

    let max_rows = if k <= GEMM_SMALL_K_THRESHOLD as usize {
        WGPU_MAX_DISPATCH_WORKGROUPS * GEMM_TILE_DIM * SMALL_K_ROWS_PER_THREAD
    } else {
        WGPU_MAX_DISPATCH_WORKGROUPS * GEMM_TILE_DIM
    };
    (max_rows / per_spec_rows).clamp(1, num_specs)
}

fn max_specs_per_dispatch(layers: &[GpuCrownLayer], num_specs: usize) -> usize {
    if num_specs == 0 {
        return 0;
    }

    let mut max_batch = num_specs.min(WGPU_MAX_DISPATCH_WORKGROUPS);
    for layer in layers {
        let layer_limit = match layer {
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                max_batch
            }
            GpuCrownLayer::MaxPool2d { .. } => max_batch,
            GpuCrownLayer::Linear {
                out_features,
                in_features,
                ..
            } => max_specs_for_gemm_dispatch(*out_features, *in_features, 1, num_specs),
            GpuCrownLayer::Conv2d {
                out_channels,
                in_channels,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
                in_h,
                in_w,
                ..
            } => {
                let spatial = out_h * out_w;
                let kernel_cols = in_channels * kernel_h * kernel_w;
                let flat_input_dim = in_channels * in_h * in_w;
                let reshape_limit = max_specs_for_1d_workgroups(spatial * out_channels, num_specs);
                let gemm_limit =
                    max_specs_for_gemm_dispatch(*out_channels, kernel_cols, spatial, num_specs);
                let col2im_limit = max_specs_for_1d_workgroups(flat_input_dim, num_specs);
                reshape_limit.min(gemm_limit).min(col2im_limit)
            }
        };
        if layer_limit == 0 {
            return 0;
        }
        max_batch = max_batch.min(layer_limit);
    }
    max_batch
}

fn record_host_phase<T>(
    profile: &mut Option<CrownHostTimingProfile>,
    label: &'static str,
    f: impl FnOnce() -> T,
) -> T {
    if let Some(profile) = profile.as_mut() {
        let start = Instant::now();
        let result = f();
        profile.record(label, start.elapsed().as_secs_f64());
        result
    } else {
        f()
    }
}

impl GpuCrownBackward for WgpuDevice {
    fn clear_crown_working_set(&self) -> Result<()> {
        // Fully qualify the inherent method: the trait hook is the public
        // type-erased seam used by long-lived attack engines, while the
        // inherent implementation owns the actual cache/pool teardown.
        WgpuDevice::clear_crown_working_set(self)
    }

    fn provides_sound_intermediate_sweep(&self) -> bool {
        self.provides_intermediate_sweep()
    }

    fn intermediate_sweep_resource_policy(&self) -> Option<GpuIntermediateSweepResourcePolicy> {
        if !self.provides_intermediate_sweep() {
            return None;
        }
        let limits = self.device.limits();
        automatic_intermediate_sweep_resource_policy(
            self.adapter_info.device_type,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
            gpu_memory_budget_bytes(),
        )
    }

    fn crown_backward_gpu_sound_intermediate_sweep(
        &self,
        request: &GpuIntermediateSweepRequest<'_>,
    ) -> Result<Option<GpuIntermediateSweepResult>> {
        // Validation deliberately precedes even the default-dark capability
        // decline: malformed/late typed requests never disappear as a benign
        // backend miss.
        request.validate()?;
        self.run_intermediate_sweep(request)
    }

    fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
        // A separate, default-closed source/selfcheck qualification is bound
        // to this exact device's stable registration epoch. The predicate also
        // re-reads ordinary WGPU verdict/loading-path authority on every call.
        self.bab_bound_authority_cached()
    }

    fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn ny_core::GpuBabBoundNumericalTcb> {
        // This accessor and the boolean gate share one predicate. Ordinary,
        // charged-flush, authority-lost, forced-fail, and unfinished-kernel
        // devices all remain `None`.
        self.bab_bound_numerical_tcb_cached()
    }

    fn crown_backward_gpu(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu: empty layer list".into(),
            ));
        }
        let first_dim = layer_output_dim(&layers[0])?;
        if spec.len() != num_specs * first_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, first_dim],
                vec![spec.len()],
            ));
        }
        let seed = GpuCrownSeed {
            lower_a: spec.to_vec().into(),
            upper_a: spec.to_vec().into(),
            lower_b: vec![0.0; num_specs].into(),
            upper_b: vec![0.0; num_specs].into(),
            num_specs,
            current_dim: first_dim,
        };
        self.crown_backward_gpu_seeded(layers, &seed, input_lower, input_upper)
    }

    fn crown_backward_gpu_seeded(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        // Wrap the entire seeded backward pass in a wgpu error scope so any
        // validation/internal/OOM error returns Err — the sequential CROWN
        // caller (network/.../crown.rs) then falls back to the sound CPU
        // backward — instead of aborting via wgpu's panicking uncaptured
        // handler (#live bug).
        self.run_gpu_checked("crown_backward_gpu_seeded", || {
            self.crown_backward_gpu_seeded_inner(layers, seed, input_lower, input_upper)
        })
    }

    fn crown_backward_gpu_sound(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_sound: empty layer list".into(),
            ));
        }
        let output_dim = layer_output_dim(&layers[0])?;

        // FIT-PRESERVING attempt: run the single (unchunked) wide dispatch first.
        // Every node whose A-buffer + dispatch already fit this adapter's limits
        // succeeds HERE and is byte-identical to main (no chunk overhead) — even with
        // the gate on. Only a node that OVERFLOWS a wgpu device limit returns `Err`.
        match self.crown_backward_sound_resident(
            layers,
            spec,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
        ) {
            Ok((lower_bounds, upper_bounds)) => Ok(GpuCrownResult {
                lower_bounds,
                upper_bounds,
            }),
            Err(e) => {
                // #wg-limit-subchunk (spec-row chunking, DEFAULT-ON since 2026-07-24;
                // kill switch `NY_GPU_BATCHED_COLLECT=0`): the sound-resident backward
                // allocates an A-coefficient buffer of `num_specs × max_layer_width`
                // f32 and dispatches `ceil(num_specs × max_layer_width / 256)`
                // workgroups in X. On a wide node these exceed wgpu's per-binding
                // `max_storage_buffer_binding_size` and/or
                // `max_compute_workgroups_per_dimension` — the latter is 65535 on BOTH
                // Metal AND the GB10 Vulkan stack (MEASURED; the old
                // Metal-only-hits-this belief is what made this a latent trap) — so the
                // whole node errors above and, with the kill switch set, falls back to
                // the CPU sound path. By default we instead split the SPEC ROWS (an EXACT
                // batch dimension: CROWN backward has no cross-row reduction — each row
                // is an independent linear functional of the output, and the ReLU
                // relaxation slopes are per-neuron and shared across rows) so each
                // sub-dispatch fits the device, then concatenate the per-row bounds.
                // This is byte-identical to the single wide dispatch — no soundness
                // change, only "fits the device". Any chunk `Err`/NaN still surfaces to
                // the caller's CPU sound fallback (the 0-wrong moat holds).
                match self.sound_spec_row_chunk(layers, num_specs) {
                    Some(chunk) if chunk < num_specs => self.crown_backward_sound_chunked(
                        layers,
                        spec,
                        num_specs,
                        output_dim,
                        chunk,
                        input_lower,
                        input_upper,
                    ),
                    // Kill switch set, or this adapter's limits already admit the full
                    // row batch (so the Err was NOT a dispatch/binding overflow and
                    // chunking cannot help) → propagate the original error so the
                    // caller takes the proven CPU sound path. Fail-closed either way.
                    _ => Err(e),
                }
            }
        }
    }

    /// Raw `WgpuDevice` CROWN authority passed the U1/U3/U4/U5/U6 ledger and B0
    /// source review. The resident word route is AUTO/default-on when its twins
    /// are available, ResNet segments compose row words, and the armed C1
    /// consult fails closed. Unsupported configurations typed-refuse. See the
    /// live ledger in `ops/sound_authority.rs`.
    ///
    /// Delegates to the immutable per-device verdict report
    /// (`ops/sound_authority.rs`). The seam returns true only on a device built
    /// through the typed explicit constructor after a passing five-rung ladder;
    /// ordinary devices, failed/uninitialized probes, and unsupported operations
    /// refuse. The public `ComputeDevice` wrapper exposes this CROWN seam only
    /// through its matching proof constructor.
    ///
    /// This MUST stay in lockstep with `impl GemmEngine for WgpuDevice`'s
    /// `as_gpu_crown_backward` — `test_support::sound_gpu_crown_quarantined`
    /// asserts the two agree, and `sound_gpu_gate` only ever reaches this
    /// predicate through that accessor.
    ///
    /// #flush-charge: charged-flush authority ALSO opens this seam — that is
    /// the entire point of the charged mode: the same walk runs with the
    /// oracle-derived widenings/refusals armed by `charged_walk_guard` and the
    /// charge sites. Unreachable until the reviewed charged source gate opens
    /// (`PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED`, compile-time
    /// `false`); the provenance string distinguishes the two modes on every
    /// ledger row.
    fn provides_sound_gpu_crown(&self) -> bool {
        self.sound_gpu_authority_cached() || self.charged_flush_authority_cached().is_some()
    }

    fn crown_backward_gpu_seeded_sound(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_seeded_sound: empty layer list".into(),
            ));
        }
        let (lower_bounds, upper_bounds) = self.crown_backward_sound_resident_seeded(
            layers,
            &seed.lower_a,
            &seed.upper_a,
            &seed.lower_b,
            &seed.upper_b,
            seed.num_specs,
            seed.current_dim,
            input_lower,
            input_upper,
        )?;
        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    /// COEFFICIENT egress for the seeded sound resident walk (#cert-coeffs).
    ///
    /// Same walk, same layers, same seed as
    /// [`Self::crown_backward_gpu_seeded_sound`] — but the certified frontier is
    /// published instead of concretized, because the margin-row lane's hot step
    /// consumes coefficients and cannot use a concretized bound at all.
    ///
    /// Gated on exactly the same authority as the bounds entry: without
    /// `provides_sound_gpu_crown()` this DECLINES (`Ok(None)`) rather than
    /// handing back a frontier a verdict might trust. `input_lower`/
    /// `input_upper` are accepted for signature parity with the bounds entry and
    /// are deliberately unused — the whole point is that the caller chooses the
    /// concretization.
    fn crown_backward_gpu_seeded_sound_coeffs(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<Option<ny_core::CertifiedCoeffs>> {
        if !self.provides_sound_gpu_crown() {
            return Ok(None);
        }
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_seeded_sound_coeffs: empty layer list".into(),
            ));
        }
        self.crown_backward_sound_resident_certified_coeffs(layers, seed)
            .map(Some)
    }

    fn crown_backward_gpu_resnet_sound(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound: empty segment list".into(),
            ));
        }
        // NOTE: do NOT wrap in `run_gpu_checked` here. The inner per-segment
        // `coeff_seeded_err` and the final `concretize_resident_coeff` each take the
        // non-reentrant `gpu_serialize` lock via their own `run_gpu_checked`; an outer
        // wrapper would re-enter that lock and DEADLOCK on the first segment. This
        // mirrors `crown_backward_gpu_seeded_sound`, which also calls the resident
        // path directly. Each inner GPU op already returns `Err` on
        // validation/internal/OOM faults, which propagates to the suffix caller's
        // CPU fallback (the 0-wrong moat holds).
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        // Per-ReLU pre-activation abs-max bounds (fold order). Threaded so the AUTO-
        // FALLBACK can prefer the finer per-ReLU concretization on an error-explosion;
        // empty ⇒ the fallback degrades to the per-segment frontier_abs path (verdict
        // default for non-exploding nets unchanged — the fine path only runs when the
        // cheap un-concretized bound already failed).
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let (lower_bounds, upper_bounds) = self.crown_backward_gpu_resnet_sound_inner(
            segments,
            seed,
            input_lower,
            input_upper,
            &fa_refs,
            &na_refs,
        )?;
        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    /// COEFFICIENT egress for the RESNET sound resident backward
    /// (#cert-coeffs-resnet).
    ///
    /// Same segments and seed as [`Self::crown_backward_gpu_resnet_sound`] — but
    /// the COMPOSED certified frontier (across every segment) is published
    /// instead of concretized, because the margin-row lane's hot step consumes
    /// coefficients and every cifar100/tinyimagenet net it must accelerate is a
    /// resnet. The box and abs-frontier arguments are deliberately ignored:
    /// using them to move coefficient error into bias would produce a
    /// domain-bound functional enclosure, not a coefficient-wise
    /// [`ny_core::CertifiedCoeffs`] enclosure.
    ///
    /// Gated on exactly the same authority as the bounds entry: without
    /// `provides_sound_gpu_crown()` this DECLINES (`Ok(None)`) rather than
    /// handing back a frontier a verdict might trust. `input_lower`/
    /// `input_upper` are accepted for signature parity and deliberately unused.
    ///
    /// NOTE (same reason as the bounds entry): do NOT wrap in `run_gpu_checked`
    /// — the inner per-segment folds take the non-reentrant `gpu_serialize`
    /// lock themselves and an outer wrapper would deadlock on segment one.
    fn crown_backward_gpu_resnet_sound_coeffs(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        _input_lower: &[f32],
        _input_upper: &[f32],
        _frontier_abs: &[Vec<f32>],
        _node_abs: &[Vec<f32>],
    ) -> Result<Option<ny_core::CertifiedCoeffs>> {
        if !self.provides_sound_gpu_crown() {
            return Ok(None);
        }
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_coeffs: empty segment list".into(),
            ));
        }
        self.crown_backward_gpu_resnet_sound_certified_coeffs(segments, seed)
            .map(Some)
    }

    /// BATCHED COEFFICIENT EGRESS (#margin-row-gpu-batch): ONE wide resident
    /// pass over `N = n_domains * seed.num_specs` stacked rows, publishing one
    /// COMPOSED certified frontier PER DOMAIN.
    ///
    /// This is the entry the margin-row twin-wall lane needs to stop processing
    /// BaB domains one at a time. It reuses, unchanged, the machinery the wide
    /// bounds lane already proved out:
    ///
    /// * the HOLE-7 homogeneity gate ([`resnet_skeleton_matches`]) — identical
    ///   variant sequence, dims, `CertifiedWeightError` charges and shared
    ///   weights, differing ONLY in per-domain relaxation VALUES;
    /// * the HOLE-8 unbatchable-layer gate ([`segments_contain_unbatchable`]);
    /// * [`stack_wide_segments`] to build the ONE shared skeleton with each
    ///   domain's Activation block at `d*num_neurons`;
    /// * the device dispatch-limit check ([`wide_safe_domain_count`]) — the
    ///   `max_compute_workgroups_per_dimension` overrun is a latent
    ///   false-VERIFY hole, so an over-limit batch returns the typed,
    ///   pre-dispatch `GpuBatchCapacityExceeded` signal. The margin-row caller
    ///   alone may use that signal to narrow its chunk width.
    ///
    /// Everything after that is the SAME un-concretized composition and the
    /// SAME fail-closed firewall the single-domain egress runs
    /// (`resnet_certified_coeffs_unconcretized`), followed by the pinned
    /// domain-major split (`split_batched_certified_coeffs`). Domain boxes and
    /// abs frontiers remain unused, as required by the coefficient contract.
    ///
    /// `Ok(None)` = unsupported/declined (no authority, gate or stacking
    /// refusal). `GpuBatchCapacityExceeded` is the only retryable `Err`, and is
    /// guaranteed pre-dispatch. Every other `Err` is a terminal failure of an
    /// accepted request. All outcomes leave the exact CPU fallback available.
    ///
    /// NOTE (same reason as the other resnet methods): do NOT wrap in
    /// `run_gpu_checked` — the inner folds take the non-reentrant
    /// `gpu_serialize` lock themselves.
    fn crown_backward_gpu_resnet_sound_batched_coeffs(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Result<Option<Vec<ny_core::CertifiedCoeffs>>> {
        if !self.provides_sound_gpu_crown() {
            return Ok(None);
        }
        let n_domains = domains.len();
        let nsp = seed.num_specs;
        if n_domains == 0 || nsp == 0 {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_batched_coeffs: empty batch".into(),
            ));
        }
        if domains[0].segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_batched_coeffs: empty segment list".into(),
            ));
        }
        // HOLE 7: one shared skeleton, per-domain relaxation VALUES only.
        for d in &domains[1..] {
            if !resnet_skeleton_matches(domains[0].segments, d.segments) {
                return Ok(None);
            }
        }
        // HOLE 8: the backward shaders for these kinds are not domain-block
        // indexed, so a wide fold would read another domain's state.
        if segments_contain_unbatchable(domains[0].segments) {
            return Ok(None);
        }
        // #wg-limit-subchunk: an overrun of `max_compute_workgroups_per_dimension`
        // is a latent FALSE-VERIFY hole (some drivers silently return an
        // over-tight bound instead of erroring). Unlike the bounds lane this
        // entry does not sub-chunk — a coefficient frontier must stay ONE
        // domain-major object — so an over-limit batch declines and the caller
        // narrows its own batch.
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1) as usize;
        let width = resnet_segments_max_1d_dispatch_dim(domains[0].segments);
        let Some(safe_domains) = wide_safe_domain_count(max_wg, width, nsp) else {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: n_domains,
                capacity: 0,
                unit: "domains",
                site: "coefficient dispatch-limit preflight",
            });
        };
        if n_domains > safe_domains {
            return Err(NyError::GpuBatchCapacityExceeded {
                requested: n_domains,
                capacity: safe_domains,
                unit: "domains",
                site: "coefficient dispatch-limit preflight",
            });
        }
        // ONE shared skeleton carrying every domain's Activation block.
        let Some(wide_segments) = stack_wide_segments(domains) else {
            return Ok(None);
        };
        // Tile the SHARED spec seed n_domains times → N rows, so each domain
        // block starts from the identical seed exactly as N serial calls do.
        let od = seed.current_dim;
        let Some(n) = n_domains.checked_mul(nsp) else {
            return Ok(None);
        };
        if seed.lower_a.len() != nsp * od
            || seed.upper_a.len() != nsp * od
            || seed.lower_b.len() != nsp
            || seed.upper_b.len() != nsp
        {
            return Ok(None);
        }
        let mut wl_a = Vec::with_capacity(n * od);
        let mut wu_a = Vec::with_capacity(n * od);
        let mut wl_b = Vec::with_capacity(n);
        let mut wu_b = Vec::with_capacity(n);
        for _ in 0..n_domains {
            wl_a.extend_from_slice(&seed.lower_a);
            wu_a.extend_from_slice(&seed.upper_a);
            wl_b.extend_from_slice(&seed.lower_b);
            wu_b.extend_from_slice(&seed.upper_b);
        }
        let wide_seed = GpuCrownSeed {
            lower_a: wl_a.into(),
            upper_a: wu_a.into(),
            lower_b: wl_b.into(),
            upper_b: wu_b.into(),
            num_specs: n,
            current_dim: od,
        };
        let out = self.crown_backward_gpu_resnet_sound_batched_certified_coeffs(
            &wide_segments,
            &wide_seed,
            nsp,
            n_domains,
        )?;
        // The caller indexes results BY DOMAIN; a count drift must refuse
        // rather than associate a frontier with another domain's relaxation.
        if out.len() != n_domains {
            return Err(NyError::InvalidSpec(format!(
                "crown_backward_gpu_resnet_sound_batched_coeffs: produced {} frontiers for {} \
                 domains",
                out.len(),
                n_domains
            )));
        }
        Ok(Some(out))
    }

    /// The deadline-bounded ResNet sound entries below are REAL on this backend
    /// (#batched-bab arming, 2026-08-11). Historically WgpuDevice inherited the
    /// ny-core defaults — `provides… = false`, `max_rows = 0`
    /// (ny-core/src/gemm.rs:2052/:2087) — so EVERY bounded-rows admission seam
    /// (`RootJointDeadlineGpu::from_engine`, resnet_decompose's
    /// `DeadlineBoundedRows` dispatch, the active-set/bounded-shared K≤8
    /// selectors) refused the prewarmed sound WGPU backend even with the
    /// authority ladder green. See `WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS`
    /// for the honest-capacity argument.
    fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
        true
    }

    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
    }

    fn crown_backward_gpu_resnet_sound_single_row_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        if seed.num_specs != 1 {
            return Err(NyError::InvalidSpec(format!(
                "crown_backward_gpu_resnet_sound_single_row_with_deadline: exactly one \
                 spec row required, got {}",
                seed.num_specs
            )));
        }
        self.resnet_sound_rows_with_call_local_deadline(
            segments,
            seed,
            input_lower,
            input_upper,
            frontier_abs,
            node_abs,
            deadline,
        )
    }

    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        if seed.num_specs == 1 {
            // Contract (ny-core): K=1 delegates to the single-row entry and
            // inherits its validation/result contract exactly.
            return self.crown_backward_gpu_resnet_sound_single_row_with_deadline(
                segments,
                seed,
                input_lower,
                input_upper,
                frontier_abs,
                node_abs,
                deadline,
            );
        }
        if !(2..=WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&seed.num_specs) {
            return Err(NyError::InvalidSpec(format!(
                "crown_backward_gpu_resnet_sound_bounded_rows_with_deadline: row count {} \
                 outside the advertised 2..={} capacity",
                seed.num_specs, WGPU_DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
            )));
        }
        self.resnet_sound_rows_with_call_local_deadline(
            segments,
            seed,
            input_lower,
            input_upper,
            frontier_abs,
            node_abs,
            deadline,
        )
    }

    fn crown_backward_gpu_resnet_sound_grad(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        relu_pre_lower: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownGradResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_grad: empty segment list".into(),
            ));
        }
        // Same no-`run_gpu_checked` rationale as `crown_backward_gpu_resnet_sound`:
        // the inner per-segment ops take the non-reentrant gpu_serialize lock. The
        // captured gradients are non-soundness-critical (they only steer alpha), so
        // a fault still falls back to the CPU gradient path with the moat intact.
        let refs: Vec<&[f32]> = relu_pre_lower.iter().map(|v| v.as_slice()).collect();
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let (lower_bounds, upper_bounds, relu_grads) = self
            .crown_backward_gpu_resnet_sound_grad_inner(
                segments,
                seed,
                input_lower,
                input_upper,
                &refs,
                &fa_refs,
                &na_refs,
            )?;
        Ok(GpuCrownGradResult {
            lower_bounds,
            upper_bounds,
            relu_grads,
        })
    }

    fn crown_backward_gpu_resnet_sound_beta(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta: empty segment list".into(),
            ));
        }
        // Same no-`run_gpu_checked` rationale as the other resnet methods (inner ops take
        // the non-reentrant lock). The bound is sound for any β≥0; any fault falls back to
        // the CPU beta-CROWN per-domain path with the 0-wrong moat intact.
        let refs: Vec<&[f32]> = beta_signed.iter().map(|v| v.as_slice()).collect();
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let (lower_bounds, upper_bounds) = self.crown_backward_gpu_resnet_sound_beta_inner(
            segments,
            seed,
            input_lower,
            input_upper,
            &refs,
            &fa_refs,
            &na_refs,
        )?;
        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    /// Observation-only, deadline-bounded Cut-CROWN fold on the actual wgpu
    /// resident walk (the CUDA override's charged-Metal twin;
    /// `ops/cut_shadow_resident.rs`). The ordinary beta result remains the
    /// sole consumable baseline; a complete cut fold can only attach
    /// telemetry.
    #[allow(clippy::too_many_arguments)]
    fn crown_backward_gpu_resnet_sound_beta_cut_shadow(
        &self,
        policy: ny_core::ResidentCutShadowPolicy,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        carrier: Option<&ny_core::ResidentLowerCutCarrier>,
        binding_row: usize,
        deadline: Instant,
    ) -> Result<ny_core::ResidentCutShadowOutcome> {
        super::cut_shadow_resident::run_resident_cut_shadow(
            self,
            policy,
            segments,
            seed,
            input_lower,
            input_upper,
            beta_signed,
            frontier_abs,
            node_abs,
            carrier,
            binding_row,
            deadline,
        )
    }

    /// The resident Cut-CROWN shadow capability is claimed ONLY when this
    /// device holds verdict authority (fully-qualified OR charged-flush) AND
    /// the audited cut-apply kernel's pinned selfcheck passed at qualification
    /// (`ops/cut_shadow_resident.rs`). This does not grant verdict authority
    /// — the shadow is observation-only by type.
    fn provides_resident_cut_shadow(&self) -> bool {
        self.resident_cut_shadow_capability()
    }

    /// #batched-bab INCREMENT 1 — REFERENCE STACKER (byte-identical to N serial
    /// [`Self::crown_backward_gpu_resnet_sound_beta`] calls). Runs the homogeneity
    /// gate, then computes each domain-block's bounds by dispatching the EXISTING
    /// sound per-domain kernel on THAT block's own operands. No shared row buffer
    /// exists, so cross-domain contamination is not representable — the result is
    /// bound-for-bound identical to the serial path (the differential oracle's
    /// tol=0 anchor). A later increment replaces ONLY this loop with a single wide
    /// GPU pass, behind the SAME signature + SAME oracle. Any heterogeneous batch
    /// aborts to `Err(UnsupportedOp)` → the caller's serial fallback (0-wrong moat).
    fn crown_backward_gpu_resnet_sound_beta_batched(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Result<Vec<GpuCrownResult>> {
        // #wide-decline-tally: one relaxed increment marking that a candidate
        // batch reached a batched GPU trait entry. `attempts - published` is the
        // coverage gap the reasons below explain. Observability only.
        note_wide_lane_attempt();
        if domains.is_empty() {
            note_wide_lane_decline(WideLaneDecline::GpuEmptyBatch);
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_batched: empty batch".into(),
            ));
        }
        // HOMOGENEITY GATE (#batched-bab HOLE 7): all domains must share the SAME
        // network skeleton (variant sequence, per-layer dims, Arc::ptr_eq of the
        // shared weights) — differing ONLY in per-domain relaxation VALUES. The
        // reference stacker is sound even for a heterogeneous batch (it dispatches
        // each domain's own segments), but landing the gate here gives increment
        // 2's wide kernel — which ASSUMES one shared skeleton — its soundness
        // precondition. A mismatch aborts the whole batch to the serial fallback;
        // it never packs ragged rows.
        let wprobe = std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1");
        let homogeneous = domains[1..]
            .iter()
            .all(|d| resnet_skeleton_matches(domains[0].segments, d.segments));
        if !homogeneous {
            // #batched-bab HOLE-7 SUB-GROUPING (DARK, `NY_BAB_RESNET_WIDE_SUBGROUP=1`):
            // rather than abandoning the WHOLE wave because two domains disagree,
            // split it into maximal contiguous homogeneous runs and run one wide
            // pass per run. Each run satisfies the wide fold's own precondition
            // (one shared skeleton) exactly as a homogeneous batch does, and runs
            // are visited in domain order so the concatenation preserves the
            // caller's per-domain result layout. Any run this backend cannot fold
            // soundly (HOLE-8 layer kinds, or a wide-assembly refusal) declines the
            // WHOLE batch back to the historical `Err` — never a partial answer,
            // never a serial/wide mixture. DEFAULT OFF: until the device-level
            // enclosure oracle is measured on a real heterogeneous wave, the gate
            // keeps scored routing byte-identical.
            if let Some(res) = self.try_wide_resnet_batched_subgrouped(domains, seed) {
                // Review defect 2: this is the SUCCESS path — count it as a
                // published sub-group run, never as a decline (a decline entry
                // here files a covered batch under `declines:` and drives the
                // documented `attempts - published` gap negative).
                ny_core::wide_lane_telemetry::note_wide_lane_subgrouped_run();
                if wprobe {
                    eprintln!(
                        "[wide] GATE homogeneity mismatch SUBGROUPED (n_domains={})",
                        domains.len()
                    );
                }
                return Ok(res);
            }
            note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
            if wprobe {
                eprintln!(
                    "[wide] GATE homogeneity mismatch (n_domains={})",
                    domains.len()
                );
            }
            return Err(NyError::UnsupportedOp(
                "crown_backward_gpu_resnet_sound_beta_batched: heterogeneous resnet skeleton \
                 across domains — falling back to the per-domain path"
                    .into(),
            ));
        }
        // #batched-bab HOLE 8: decline dual-alpha / maxpool (backward shaders not
        // domain-block-indexed) → serial per-domain fallback (fail-closed).
        if segments_contain_unbatchable(domains[0].segments) {
            if wprobe {
                let kinds: Vec<&str> = domains[0]
                    .segments
                    .iter()
                    .flat_map(|s| match s {
                        GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => vec![l],
                        GpuResnetSegment::ResidualProj(f, p) => vec![f, p],
                    })
                    .flatten()
                    .map(|l| match l {
                        GpuCrownLayer::Linear { .. } => "Lin",
                        GpuCrownLayer::Activation { .. } => "Act",
                        GpuCrownLayer::ActivationReluDualAlpha { .. } => "DualAlpha",
                        GpuCrownLayer::Conv2d { .. } => "Conv",
                        GpuCrownLayer::MaxPool2d { .. } => "MaxPool",
                    })
                    .collect();
                eprintln!("[wide] GATE hole8 unbatchable; layer kinds: {kinds:?}");
            }
            note_wide_lane_decline(WideLaneDecline::GpuUnbatchableLayer);
            return Err(NyError::UnsupportedOp(
                "crown_backward_gpu_resnet_sound_beta_batched: batch contains \
                 ActivationReluDualAlpha/MaxPool2d — not wide-batchable, using the serial path"
                    .into(),
            ));
        }
        // #batched-bab increment 3: the WIDE single-pass path — run ALL domains in ONE
        // resident backward over N = n_domains*num_specs stacked rows. Sound-checked by
        // the two-sided differential oracle (wide bound matches serial per-domain within
        // f32-reorder tol). Any build/shape failure falls THROUGH to the byte-identical
        // reference stacker below (0-wrong moat). Opt out with NY_BAB_RESNET_WIDE=0 (A/B).
        let wide_disabled = !wide_resnet_enabled();
        if domains.len() > 1 && !wide_disabled {
            if let Some(res) = self.try_wide_resnet_batched(domains, seed) {
                return Ok(res);
            }
            // The wide assembly itself declined; `try_wide_resnet_batched_grad`
            // already recorded WHICH predicate refused, so do not double-count
            // here — fall through to the byte-identical reference stacker.
        } else if wide_disabled {
            note_wide_lane_decline(WideLaneDecline::GpuWideEnvDisabled);
        } else {
            // A one-domain batch has nothing to stack: the per-domain kernel below
            // IS the wide pass for it. Counted so the denominator stays honest.
            note_wide_lane_decline(WideLaneDecline::GpuSingleDomainBatch);
        }
        let mut out = Vec::with_capacity(domains.len());
        for d in domains {
            if d.segments.is_empty() {
                return Err(NyError::InvalidSpec(
                    "crown_backward_gpu_resnet_sound_beta_batched: empty segment list".into(),
                ));
            }
            let bs: Vec<&[f32]> = d.beta_signed.iter().map(|v| v.as_slice()).collect();
            let fa: Vec<&[f32]> = d.frontier_abs.iter().map(|v| v.as_slice()).collect();
            let na: Vec<&[f32]> = d.node_abs.iter().map(|v| v.as_slice()).collect();
            let (lower_bounds, upper_bounds) = self.crown_backward_gpu_resnet_sound_beta_inner(
                d.segments,
                seed,
                d.input_lower,
                d.input_upper,
                &bs,
                &fa,
                &na,
            )?;
            out.push(GpuCrownResult {
                lower_bounds,
                upper_bounds,
            });
        }
        Ok(out)
    }

    /// #batched-bab part A: the GRADIENT-capturing wide batched backward. Same
    /// homogeneity + HOLE-8 gates as the bound path, then ONE wide resident pass over
    /// N = n_domains*num_specs rows that ALSO gathers the per-ReLU union columns'
    /// A_lower for the analytic β gradient. NO reference-stacker fallback here (the
    /// CALLER's serial per-domain β ascent is the fallback): any gate/shape/GPU failure
    /// returns `Err` → the caller drops to the serial path (0-wrong moat preserved,
    /// since β-opt only steers β and the bound comes from the sound wide fold).
    fn crown_backward_gpu_resnet_sound_beta_batched_grad(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        // #wide-decline-tally: see the bound entry. The β-opt lane calls this once
        // per ascent ITERATION, so attempts here are iteration-granular — the same
        // granularity as the published counter it is compared against.
        note_wide_lane_attempt();
        if domains.is_empty() {
            note_wide_lane_decline(WideLaneDecline::GpuEmptyBatch);
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_batched_grad: empty batch".into(),
            ));
        }
        for d in &domains[1..] {
            if !resnet_skeleton_matches(domains[0].segments, d.segments) {
                note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
                return Err(NyError::UnsupportedOp(
                    "batched-grad: heterogeneous resnet skeleton — using the serial β ascent"
                        .into(),
                ));
            }
        }
        if segments_contain_unbatchable(domains[0].segments) {
            note_wide_lane_decline(WideLaneDecline::GpuUnbatchableLayer);
            return Err(NyError::UnsupportedOp(
                "batched-grad: ActivationReluDualAlpha/MaxPool2d — using the serial β ascent"
                    .into(),
            ));
        }
        self.try_wide_resnet_batched_grad(
            domains,
            seed,
            union_gather_idx,
            relu_pre_lower,
            None,
            false,
        )
        .ok_or_else(|| {
            NyError::UnsupportedOp(
                "batched-grad: wide assembly/shape failure — using the serial β ascent".into(),
            )
        })
    }

    /// Capture bounds, both dual-gradient channels, and the affine input
    /// frontier from the first resident pass. Unlike the clip-specific coeff
    /// entry, trajectory capture does not force a second fine-concretization
    /// backward merely to recover coefficients; their still-live certified
    /// error is exported and FacetBank discharges it over the input box.
    fn crown_backward_gpu_resnet_sound_beta_batched_trajectory(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
    ) -> Result<GpuCrownTrajectoryResult> {
        // #wide-decline-tally: see the bound entry.
        note_wide_lane_attempt();
        if domains.is_empty() {
            note_wide_lane_decline(WideLaneDecline::GpuEmptyBatch);
            return Err(NyError::InvalidSpec(
                "batched-trajectory: empty batch".into(),
            ));
        }
        for d in &domains[1..] {
            if !resnet_skeleton_matches(domains[0].segments, d.segments) {
                note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
                return Err(NyError::UnsupportedOp(
                    "batched-trajectory: heterogeneous resnet skeleton".into(),
                ));
            }
        }
        if segments_contain_unbatchable(domains[0].segments) {
            note_wide_lane_decline(WideLaneDecline::GpuUnbatchableLayer);
            return Err(NyError::UnsupportedOp(
                "batched-trajectory: ActivationReluDualAlpha/MaxPool2d is not wide-batchable"
                    .into(),
            ));
        }
        let mut coeff = GpuResidentCoeffBatched {
            lower_a: Vec::new(),
            upper_a: Vec::new(),
            lower_err: Vec::new(),
            upper_err: Vec::new(),
            lower_b: Vec::new(),
            upper_b: Vec::new(),
            lower_b_err: Vec::new(),
            upper_b_err: Vec::new(),
            dim: 0,
            num_specs: 0,
            num_specs_per_dom: 0,
        };
        let (bounds, alpha_grads, beta_gather) = self
            .try_wide_resnet_batched_grad(
                domains,
                seed,
                union_gather_idx,
                relu_pre_lower,
                Some(&mut coeff),
                false,
            )
            .ok_or_else(|| {
                NyError::UnsupportedOp(
                    "batched-trajectory: wide assembly/shape/coeff failure".into(),
                )
            })?;
        let expected_specs = domains
            .len()
            .checked_mul(seed.num_specs)
            .ok_or_else(|| NyError::InvalidSpec("batched-trajectory: row overflow".into()))?;
        let expected_coeffs = expected_specs
            .checked_mul(coeff.dim)
            .ok_or_else(|| NyError::InvalidSpec("batched-trajectory: coeff overflow".into()))?;
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        let valid_error = |v: &[f32]| v.iter().all(|x| x.is_finite() && *x >= 0.0);
        if coeff.dim == 0
            || coeff.num_specs != expected_specs
            || coeff.num_specs_per_dom != seed.num_specs
            || coeff.lower_a.len() != expected_coeffs
            || coeff.upper_a.len() != expected_coeffs
            || coeff.lower_err.len() != expected_coeffs
            || coeff.upper_err.len() != expected_coeffs
            || coeff.lower_b.len() != expected_specs
            || coeff.upper_b.len() != expected_specs
            || coeff.lower_b_err.len() != expected_specs
            || coeff.upper_b_err.len() != expected_specs
            || !finite(&coeff.lower_a)
            || !finite(&coeff.upper_a)
            || !finite(&coeff.lower_b)
            || !finite(&coeff.upper_b)
            || !valid_error(&coeff.lower_err)
            || !valid_error(&coeff.upper_err)
            || !valid_error(&coeff.lower_b_err)
            || !valid_error(&coeff.upper_b_err)
        {
            return Err(NyError::InvalidSpec(
                "batched-trajectory: malformed captured coefficient frontier".into(),
            ));
        }
        Ok(GpuCrownTrajectoryResult {
            bounds,
            alpha_grads,
            beta_gather,
            coeff,
        })
    }

    /// #clip-interm-resnet-batched: run the wide batched sound resnet backward AND
    /// return the downloaded input-relative coefficient frontier for the batched clip.
    /// Same homogeneity + unbatchable gates as
    /// [`Self::crown_backward_gpu_resnet_sound_beta_batched`]; a gate/shape/GPU failure
    /// returns `Err` → the caller keeps the frozen intermediates (sound, no clip).
    fn crown_backward_gpu_resnet_sound_beta_batched_coeff(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Result<(Vec<GpuCrownResult>, GpuResidentCoeffBatched)> {
        // #wide-decline-tally: see the bound entry.
        note_wide_lane_attempt();
        if domains.is_empty() {
            note_wide_lane_decline(WideLaneDecline::GpuEmptyBatch);
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_batched_coeff: empty batch".into(),
            ));
        }
        for d in &domains[1..] {
            if !resnet_skeleton_matches(domains[0].segments, d.segments) {
                note_wide_lane_decline(WideLaneDecline::GpuHomogeneityMismatch);
                return Err(NyError::UnsupportedOp(
                    "batched-coeff: heterogeneous resnet skeleton — no batched clip".into(),
                ));
            }
        }
        if segments_contain_unbatchable(domains[0].segments) {
            note_wide_lane_decline(WideLaneDecline::GpuUnbatchableLayer);
            return Err(NyError::UnsupportedOp(
                "batched-coeff: ActivationReluDualAlpha/MaxPool2d — no batched clip".into(),
            ));
        }
        self.try_wide_resnet_batched_coeff(domains, seed)
            .ok_or_else(|| {
                NyError::UnsupportedOp(
                    "batched-coeff: wide assembly/shape/coeff failure — no batched clip".into(),
                )
            })
    }

    fn crown_joint_alpha_gradient_resident(
        &self,
        segments: &[GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        // Delegates to the resident forward+adjoint (crown_backward_sound_resident.rs).
        // Non-soundness-critical: the returned gradient only steers α∈[0,1]; the
        // verdict bound is always the sound fold, so any fault falls back to the CPU
        // oracle in the caller (still the correct gradient).
        self.crown_joint_alpha_gradient_resident(
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
        )
    }

    fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
        true
    }

    fn crown_joint_alpha_gradient_resident_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed_lower_a: &[f32],
        num_specs: usize,
        output_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
        deadline: Instant,
    ) -> Result<Vec<Vec<f32>>> {
        self.crown_joint_alpha_gradient_resident_with_deadline(
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            input_lower,
            input_upper,
            deadline,
        )
    }

    fn crown_backward_gpu_resnet_sound_beta_grad(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        beta_signed: &[Vec<f32>],
        beta_gather_idx: &[Vec<u32>],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
    ) -> Result<ny_core::GpuCrownBetaGradResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_resnet_sound_beta_grad: empty segment list".into(),
            ));
        }
        // Same no-`run_gpu_checked` rationale as the other resnet methods (inner ops take
        // the non-reentrant lock). The bound is sound for any β≥0; the gathered A-values
        // only steer β. Any fault falls back to the single-shot beta path (0-wrong moat).
        let refs: Vec<&[f32]> = beta_signed.iter().map(|v| v.as_slice()).collect();
        let gi_refs: Vec<&[u32]> = beta_gather_idx.iter().map(|v| v.as_slice()).collect();
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let (lower_bounds, upper_bounds, beta_gather) = self
            .crown_backward_gpu_resnet_sound_beta_grad_inner(
                segments,
                seed,
                input_lower,
                input_upper,
                &refs,
                &gi_refs,
                &fa_refs,
                &na_refs,
            )?;
        Ok(ny_core::GpuCrownBetaGradResult {
            lower_bounds,
            upper_bounds,
            beta_gather,
        })
    }

    /// #batched-vjp: exact point-VJP for K restarts in ONE wide resident pass.
    /// Encodes each restart as a wide-lane "domain": one shared-weight Chain
    /// skeleton whose `Activation` mask slots are stacked per-restart
    /// (`lower_slope == upper_slope == mask_k`, zero intercepts) and whose
    /// static `Activation` entries (constant arithmetic) are tiled; the seed is
    /// the K per-restart spec rows (num_specs_per_dom = 1, so wide row k IS
    /// restart k). The folded input-level LOWER coefficient rows returned by
    /// the lean wide inner are the exact per-restart gradients. ATTACK-ONLY
    /// (never verdict-feeding); any shape/assembly/GPU failure returns `Err`
    /// and the caller falls back to the sequential exact gradient.
    fn crown_point_vjp_batched(
        &self,
        layers_backward: &[GpuCrownLayer],
        mask_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let k = masks.len();
        if layers_backward.is_empty() || k == 0 || input_dim == 0 || output_dim == 0 {
            return Err(NyError::InvalidSpec(
                "crown_point_vjp_batched: empty layers/masks or zero dims".into(),
            ));
        }
        if spec_rows.len() != k * output_dim {
            return Err(NyError::shape_mismatch(
                vec![k, output_dim],
                vec![spec_rows.len()],
            ));
        }
        if layer_output_dim(&layers_backward[0])? != output_dim {
            return Err(NyError::shape_mismatch(
                vec![output_dim],
                vec![layer_output_dim(&layers_backward[0])?],
            ));
        }
        // #vjp-resident (#sb-rebank lever 3): the device-resident wide-template
        // cache folds the same lower-coefficient chain with one submit and
        // per-step uploads of only the K mask slabs + spec rows — bit-identical
        // gradients (oracle-tested in ops/point_vjp_resident.rs). Any error
        // falls through to the audited un-cached stacking path below.
        // `NY_VJP_RESIDENT=0` disables.
        if super::point_vjp_resident::resident_vjp_enabled() {
            match self.crown_point_vjp_batched_resident(
                layers_backward,
                mask_positions,
                masks,
                spec_rows,
                output_dim,
                input_dim,
            ) {
                Ok(grads) => return Ok(grads),
                Err(err) => {
                    if std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1") {
                        eprintln!("[vjp-resident] falling back to un-cached wide fold: {err}");
                    }
                }
            }
        }
        let wide_layers = stack_point_vjp_wide_layers(layers_backward, mask_positions, masks)?;
        let wide_segments = vec![GpuResnetSegment::Chain(wide_layers)];
        // Seed: row k = restart k's cotangent row, symmetric (lower == upper),
        // zero bias. num_specs_per_dom = 1 ⇒ wide row k folds against ITS OWN
        // domain block (mask_k) in the domain-block-indexed activation shader.
        let seed = GpuCrownSeed {
            lower_a: spec_rows.to_vec().into(),
            upper_a: spec_rows.to_vec().into(),
            lower_b: vec![0.0; k].into(),
            upper_b: vec![0.0; k].into(),
            num_specs: k,
            current_dim: output_dim,
        };
        // Dummy per-domain input box (K blocks of input_dim): the concretized
        // bounds are discarded — only the pre-concretize coefficient is read.
        let dummy_box = vec![0.0f32; k * input_dim];
        let coeff = self.crown_backward_gpu_point_vjp_wide_inner(
            &wide_segments,
            &seed,
            1,
            &dummy_box,
            &dummy_box,
        )?;
        if coeff.len() != k * input_dim {
            return Err(NyError::shape_mismatch(
                vec![k, input_dim],
                vec![coeff.len()],
            ));
        }
        Ok((0..k)
            .map(|d| coeff[d * input_dim..(d + 1) * input_dim].to_vec())
            .collect())
    }

    /// #batched-vjp-resnet: exact point-VJP for K restarts over a RESNET-DAG
    /// segment template in ONE wide resident pass. Same wide-lane encoding as
    /// [`Self::crown_point_vjp_batched`] (per-restart mask slopes stacked
    /// domain-blocked, static affines tiled, shared weights cloned), but the
    /// template is a backward-order [`GpuResnetSegment`] list so residual
    /// blocks fold with the exact reverse-mode fan-in ADD
    /// (`A_in = backward_F(A) + A` / `backward_F(A) + backward_P(A)`).
    /// ATTACK-ONLY; any failure returns `Err` (caller falls back to the
    /// sequential exact gradient).
    fn crown_point_vjp_batched_resnet(
        &self,
        segments_backward: &[GpuResnetSegment],
        mask_flat_positions: &[usize],
        masks: &[Vec<Vec<f32>>],
        spec_rows: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let k = masks.len();
        if segments_backward.is_empty() || k == 0 || input_dim == 0 || output_dim == 0 {
            return Err(NyError::InvalidSpec(
                "crown_point_vjp_batched_resnet: empty segments/masks or zero dims".into(),
            ));
        }
        if spec_rows.len() != k * output_dim {
            return Err(NyError::shape_mismatch(
                vec![k, output_dim],
                vec![spec_rows.len()],
            ));
        }
        for mk in masks {
            if mk.len() != mask_flat_positions.len() {
                return Err(NyError::shape_mismatch(
                    vec![mask_flat_positions.len()],
                    vec![mk.len()],
                ));
            }
        }
        // The seed's current_dim must match the output-side segment's first layer.
        let first_branch = match &segments_backward[0] {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => l,
            GpuResnetSegment::ResidualProj(f, _) => f,
        };
        let first_layer = first_branch.first().ok_or_else(|| {
            NyError::InvalidSpec("crown_point_vjp_batched_resnet: empty first branch".into())
        })?;
        if layer_output_dim(first_layer)? != output_dim {
            return Err(NyError::shape_mismatch(
                vec![output_dim],
                vec![layer_output_dim(first_layer)?],
            ));
        }
        // Stack the wide template: flat traversal = for each segment in order,
        // Chain/Residual branch layers in stored order; ResidualProj F then P.
        let mut flat = 0usize;
        let mut wide_segments = Vec::with_capacity(segments_backward.len());
        for seg in segments_backward {
            wide_segments.push(match seg {
                GpuResnetSegment::Chain(l) => GpuResnetSegment::Chain(stack_point_vjp_wide_branch(
                    l,
                    &mut flat,
                    mask_flat_positions,
                    masks,
                )?),
                GpuResnetSegment::Residual(l) => GpuResnetSegment::Residual(
                    stack_point_vjp_wide_branch(l, &mut flat, mask_flat_positions, masks)?,
                ),
                GpuResnetSegment::ResidualProj(f, p) => {
                    let wf = stack_point_vjp_wide_branch(f, &mut flat, mask_flat_positions, masks)?;
                    let wp = stack_point_vjp_wide_branch(p, &mut flat, mask_flat_positions, masks)?;
                    GpuResnetSegment::ResidualProj(wf, wp)
                }
            });
        }
        // Every mask slot must have been consumed by the traversal.
        if let Some(&bad) = mask_flat_positions.iter().find(|&&p| p >= flat) {
            return Err(NyError::InvalidSpec(format!(
                "crown_point_vjp_batched_resnet: mask position {bad} beyond flat layer count {flat}"
            )));
        }
        let seed = GpuCrownSeed {
            lower_a: spec_rows.to_vec().into(),
            upper_a: spec_rows.to_vec().into(),
            lower_b: vec![0.0; k].into(),
            upper_b: vec![0.0; k].into(),
            num_specs: k,
            current_dim: output_dim,
        };
        // Dummy per-domain input box: concretized bounds are discarded — only the
        // pre-concretize folded lower coefficient is read.
        let dummy_box = vec![0.0f32; k * input_dim];
        let coeff = self.crown_backward_gpu_point_vjp_wide_inner(
            &wide_segments,
            &seed,
            1,
            &dummy_box,
            &dummy_box,
        )?;
        if coeff.len() != k * input_dim {
            return Err(NyError::shape_mismatch(
                vec![k, input_dim],
                vec![coeff.len()],
            ));
        }
        Ok((0..k)
            .map(|d| coeff[d * input_dim..(d + 1) * input_dim].to_vec())
            .collect())
    }

    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        self.store_crown_backward_deadline(deadline);
    }

    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }
}

/// #batched-vjp: stack the shared backward template into ONE wide layer vec for
/// `n_restarts` domains. `Activation` MASK slots (listed in `mask_positions`, the
/// per-restart ReLU masks) get `lower_slope == upper_slope == masks[k][r]` with
/// zero intercepts, concatenated CONTIGUOUSLY per domain (`d*num_neurons + i`, the
/// layout `CROWN_ACTIVATION_RESIDENT_SHADER` domain-block-indexes). Static
/// `Activation` entries (constant arithmetic, identical across restarts) are TILED
/// `n_restarts` times. Shared `Linear`/`Conv2d` layers are cloned (Arc weights).
/// Any unsupported variant (`ActivationReluDualAlpha`/`MaxPool2d` — backward
/// shaders not domain-block-indexed) or shape mismatch returns `Err` (fail-closed
/// → the caller's sequential fallback).
pub(in crate::wgpu_device) fn stack_point_vjp_wide_layers(
    layers_backward: &[GpuCrownLayer],
    mask_positions: &[usize],
    masks: &[Vec<Vec<f32>>],
) -> Result<Vec<GpuCrownLayer>> {
    for &p in mask_positions {
        if p >= layers_backward.len()
            || !matches!(layers_backward[p], GpuCrownLayer::Activation { .. })
        {
            return Err(NyError::InvalidSpec(format!(
                "crown_point_vjp_batched: mask position {p} is not an Activation layer"
            )));
        }
    }
    for mk in masks {
        if mk.len() != mask_positions.len() {
            return Err(NyError::shape_mismatch(
                vec![mask_positions.len()],
                vec![mk.len()],
            ));
        }
    }
    // The chain template IS one flat branch: flat index == layer index.
    let mut flat = 0usize;
    stack_point_vjp_wide_branch(layers_backward, &mut flat, mask_positions, masks)
}

/// #batched-vjp-resnet: stack ONE branch's layers into their wide (K-domain) form,
/// advancing `flat` (the running index in the template's flattened layer traversal)
/// by one per layer. A layer whose flat index appears in `mask_flat_positions` is a
/// per-restart ReLU MASK slot (`masks[k][r]`, r = its position in that list) baked
/// as domain-block-stacked `lower_slope == upper_slope == mask_k`, zero intercepts;
/// static `Activation` entries are tiled; shared `Linear`/`Conv2d` are cloned (Arc
/// weights). Shared by the chain and resnet-segment wide entries.
fn stack_point_vjp_wide_branch(
    branch: &[GpuCrownLayer],
    flat: &mut usize,
    mask_flat_positions: &[usize],
    masks: &[Vec<Vec<f32>>],
) -> Result<Vec<GpuCrownLayer>> {
    let k = masks.len();
    let slot_of = |idx: usize| mask_flat_positions.iter().position(|&p| p == idx);
    let mut out = Vec::with_capacity(branch.len());
    for layer in branch {
        let idx = *flat;
        *flat += 1;
        match layer {
            GpuCrownLayer::Linear { .. } | GpuCrownLayer::Conv2d { .. } => {
                if slot_of(idx).is_some() {
                    return Err(NyError::InvalidSpec(format!(
                        "crown_point_vjp: mask position {idx} is not an Activation layer"
                    )));
                }
                out.push(layer.clone());
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
            } => {
                let nn = *num_neurons;
                if let Some(r) = slot_of(idx) {
                    // Per-restart ReLU mask slot: domain-block stacked masks.
                    let mut slopes = Vec::with_capacity(k * nn);
                    for mk in masks {
                        let m = &mk[r];
                        if m.len() != nn {
                            return Err(NyError::shape_mismatch(vec![nn], vec![m.len()]));
                        }
                        slopes.extend_from_slice(m);
                    }
                    out.push(GpuCrownLayer::Activation {
                        lower_slope: slopes.clone(),
                        upper_slope: slopes,
                        lower_intercept: vec![0.0; k * nn],
                        upper_intercept: vec![0.0; k * nn],
                        num_neurons: nn,
                    });
                } else {
                    // Static affine Activation (constant arithmetic): tiled.
                    if lower_slope.len() != nn
                        || upper_slope.len() != nn
                        || lower_intercept.len() != nn
                        || upper_intercept.len() != nn
                    {
                        return Err(NyError::shape_mismatch(vec![nn], vec![lower_slope.len()]));
                    }
                    let tile = |v: &[f32]| {
                        let mut w = Vec::with_capacity(k * nn);
                        for _ in 0..k {
                            w.extend_from_slice(v);
                        }
                        w
                    };
                    out.push(GpuCrownLayer::Activation {
                        lower_slope: tile(lower_slope),
                        upper_slope: tile(upper_slope),
                        lower_intercept: tile(lower_intercept),
                        upper_intercept: tile(upper_intercept),
                        num_neurons: nn,
                    });
                }
            }
            // Backward shaders for these are NOT domain-block-indexed (HOLE 8).
            GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => {
                return Err(NyError::UnsupportedOp(
                    "crown_point_vjp: DualAlpha/MaxPool2d not wide-batchable".into(),
                ));
            }
        }
    }
    Ok(out)
}

impl WgpuDevice {
    /// Shared body of the two deadline-bounded ResNet sound trait entries.
    ///
    /// Contract discharge (ny-core `GpuCrownBackward` docs):
    /// - PRE-CHECK: refuses before any dispatch once `deadline` has passed.
    /// - MALFORMED INPUT: rejects non-finite seed coefficients/biases and a
    ///   non-finite or inverted input box before publication.
    /// - MID-FLIGHT: arms the thread-local [`CallLocalCrownDeadlineScope`], which
    ///   `crown_backward_deadline_expired()` folds into the SAME between-layer /
    ///   between-chunk polls the resident sound fold already runs — without
    ///   touching the lease-owned backend deadline slot.
    /// - LATE PUBLICATION: re-checks after the fold and returns
    ///   `DeadlineExceeded` instead of publishing a late result.
    /// - RESULT SHAPE: exactly one finite, ordered interval per row, else `Err`.
    /// - SOUNDNESS: delegates to `crown_backward_gpu_resnet_sound_inner` — the
    ///   certified-error RESIDENT path (γ_k·S combine, outward-rounded host
    ///   folds), never the fast round-to-nearest tier.
    #[allow(clippy::too_many_arguments)]
    fn resnet_sound_rows_with_call_local_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        if segments.is_empty() {
            return Err(NyError::InvalidSpec(
                "deadline-bounded resnet sound backward: empty segment list".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "deadline-bounded resnet sound backward: deadline passed before dispatch".into(),
            ));
        }
        if seed
            .lower_a
            .iter()
            .chain(seed.upper_a.iter())
            .chain(seed.lower_b.iter())
            .chain(seed.upper_b.iter())
            .any(|value| !value.is_finite())
        {
            return Err(NyError::InvalidSpec(
                "deadline-bounded resnet sound backward: non-finite seed".into(),
            ));
        }
        if input_lower.is_empty()
            || input_lower.len() != input_upper.len()
            || input_lower
                .iter()
                .zip(input_upper)
                .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err(NyError::InvalidSpec(
                "deadline-bounded resnet sound backward: malformed input box".into(),
            ));
        }
        let _deadline_scope = crate::wgpu_device::CallLocalCrownDeadlineScope::arm(deadline);
        let fa_refs: Vec<&[f32]> = frontier_abs.iter().map(|v| v.as_slice()).collect();
        let na_refs: Vec<&[f32]> = node_abs.iter().map(|v| v.as_slice()).collect();
        let (lower_bounds, upper_bounds) = self.crown_backward_gpu_resnet_sound_inner(
            segments,
            seed,
            input_lower,
            input_upper,
            &fa_refs,
            &na_refs,
        )?;
        if Instant::now() >= deadline {
            // Never publish a late result: the caller's schedule assumed this
            // budget and a sound refusal is always available (CPU fallback).
            return Err(NyError::DeadlineExceeded(
                "deadline-bounded resnet sound backward: refusing to publish a late result".into(),
            ));
        }
        let rows = seed.num_specs;
        let publishable = lower_bounds.len() == rows
            && upper_bounds.len() == rows
            && lower_bounds
                .iter()
                .zip(&upper_bounds)
                .all(|(&lo, &hi)| lo.is_finite() && hi.is_finite() && lo <= hi);
        if !publishable {
            return Err(NyError::InternalError(
                "deadline-bounded resnet sound backward: result is not one finite ordered \
                 interval per row"
                    .into(),
            ));
        }
        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    /// #wg-limit-subchunk (DEFAULT-ON, auto-detected): the largest spec-row chunk for
    /// the sound-resident backward that keeps every per-dispatch buffer binding under
    /// this adapter's `max_storage_buffer_binding_size` AND every workgroup dimension
    /// under its `max_compute_workgroups_per_dimension`. Returns `Some(chunk)` with
    /// `1 ≤ chunk ≤ num_specs`; the caller only *uses* it when `chunk < num_specs`
    /// (i.e. when this adapter's own reported limits say the single wide dispatch does
    /// not fit), so on any adapter/shape that fits, this is a no-op.
    ///
    /// AUTO-DETECT, NOT A PLATFORM ASSUMPTION. Both caps are read from
    /// `self.device.limits()` — the exact values wgpu validates the dispatch against.
    /// `docs/METAL_CUDA_SPECIALIZATION_DECISION.md` used to claim only Metal's tighter
    /// caps trigger this and that "CUDA/Vulkan report larger limits"; MEASURED FALSE on
    /// the GB10 Vulkan stack, which reports `max_compute_workgroups_per_dimension =
    /// 65535` — the wgpu default — exactly like Metal. Hard-coding a platform belief is
    /// what made this a latent trap; asking the device is the fix.
    ///
    /// KILL SWITCH: `NY_GPU_BATCHED_COLLECT=0` returns `None` (never chunk), restoring
    /// the pre-2026-07-25 behavior in which an over-limit node hard-fails
    /// (`crown_backward_sound_resident: 1-D dispatch … exceeds
    /// max_compute_workgroups_per_dimension …`) and the caller falls back to the CPU
    /// sound path. Kept only as an escape hatch for bisecting a suspected chunking bug.
    /// Measured cost of setting it, 16-row A/B on relusplitter/cifar_biasfield at official
    /// budgets (`scratchpad/wg_cap_default_flip_2026-07-25/`): 1–9 CPU fallbacks instead
    /// of 0, CROWN-IBP collection 14.6–97.1 s instead of 13.0–20.9 s, root objective
    /// −2.6e7…−1.2e8 instead of −1.6e5…−6.8e5, and 3 of 8 unsat rows never reaching the
    /// BaB root report at all. Verdicts are identical either way (0 flips, delta +0), so
    /// this is a robustness/bound-quality knob, not a scorecard one.
    ///
    /// Sizing model: the peak A-coefficient buffer is `num_specs × W` f32 and the
    /// resident elementwise passes dispatch `ceil(num_specs × W / 256)` workgroups in X,
    /// where `W = max` layer in/out width in this sub-network. Both are linear in the
    /// row count, so a row-chunk of
    ///   `min( 0.9·max_binding_bytes / (4·W),  0.9·(max_wg·256) / W )`
    /// respects both limits with a 10% margin.
    ///
    /// Chunking is EXACT (not merely sound): spec rows carry no cross-row reduction, so
    /// each chunk's per-row bounds equal the single-dispatch bounds — verified
    /// bit-for-bit by `spec_row_chunk_is_exact_vs_unchunked` (Linear/ReLU chain) and
    /// `spec_row_chunk_is_exact_vs_unchunked_conv_chain` (Conv2d chain, prime row count
    /// so every chunk size exercises a short tail).
    pub(crate) fn sound_spec_row_chunk(
        &self,
        layers: &[GpuCrownLayer],
        num_specs: usize,
    ) -> Option<usize> {
        // Kill switch: `NY_GPU_BATCHED_COLLECT=0` ⇒ never chunk (pre-fix behavior:
        // over-limit node → Err → caller's CPU sound fallback). Any other value
        // (including unset, the default) ⇒ size the chunk from the device's limits.
        if std::env::var("NY_GPU_BATCHED_COLLECT").ok().as_deref() == Some("0") {
            return None;
        }
        if num_specs == 0 {
            return None;
        }
        // BLAST-RADIUS FENCE (matters now that this is the default): the caller only
        // reaches us AFTER the unchunked attempt returned SOME `Err`, and not every
        // `Err` is a device-limit overflow. `crown_backward_sound_resident` also
        // rejects any layer that is not Linear/Activation/Conv2d ("R4"), which is what
        // a MaxPool2d / dual-alpha graph hits. Chunking cannot fix an R4 rejection —
        // every chunk would re-reject — so decline here and let the original `Err` go
        // straight to the CPU sound path instead of spinning `num_specs/chunk` futile
        // sub-calls. Mirrors R4 exactly, so it can never suppress a legitimate chunk.
        if !layers.iter().all(|l| {
            matches!(
                l,
                GpuCrownLayer::Linear { .. }
                    | GpuCrownLayer::Activation { .. }
                    | GpuCrownLayer::Conv2d { .. }
            )
        }) {
            return None;
        }
        let max_dim = layers
            .iter()
            .filter_map(|l| layer_input_dim(l).ok())
            .chain(layers.iter().filter_map(|l| layer_output_dim(l).ok()))
            .max()
            .unwrap_or(0)
            .max(1);
        let limits = self.device.limits();
        // Use the adapter's OWN reported limits (authoritative — this is exactly what
        // wgpu validates the dispatch against), falling back to the known wgpu
        // defaults only if a field reports 0. Backend-agnostic and adapts
        // automatically. (The note here used to say "Metal reports 128 MiB" —
        // that is wgpu's DEFAULT, not Metal's capability. With
        // `NY_GPU_BIG_BINDINGS` requesting adapter limits, an Apple M4 Pro
        // reports 4095 MiB; measured 2026-08-06.)
        let max_bind = match limits.max_storage_buffer_binding_size as usize {
            0 => WGPU_MAX_BINDING_BYTES,
            r => r,
        };
        let max_wg = match limits.max_compute_workgroups_per_dimension as usize {
            0 => WGPU_MAX_DISPATCH_WORKGROUPS,
            r => r,
        };
        // Single A-buffer binding: rows × max_dim × 4 bytes ≤ max_storage_buffer_binding_size.
        let cap_buf = (((max_bind / 4) * 9 / 10) / max_dim).max(1);
        // 1-D elementwise dispatch bound (the true LIMIT-1 constraint). The resident
        // GEMM itself is 2-D (`select_gemm_dispatch`: wg_x=ceil(N/16), wg_y=ceil(M/16)),
        // so it never overflows the per-dimension cap here. What DOES dispatch 1-D over
        // `rows × width` elements are the resident elementwise passes (`pass_simple`):
        // ABS_COPY / AW_ERROR_COMBINE / ACTIVATION_RESIDENT / *_BIAS, every one of which
        // is `@workgroup_size(256)` with ONE element per thread — so each dispatches
        // `ceil(rows × width / 256)` workgroups in X (`width ≤ max_dim`). The prior
        // `/64` sizing conflated this with a 64-wide GEMM group and under-sized the
        // chunk ~4× (11 chunks for the 6272 node instead of 3), re-paying the fixed
        // per-dispatch overhead 4× over. The correct bound is
        //   ceil(rows × max_dim / 256) ≤ max_compute_workgroups_per_dimension.
        // Still fail-closed: if any pass had a smaller group the over-large dispatch
        // would `Err` and surface to the CPU sound fallback (never an unsound bound).
        let cap_disp = (((max_wg * 256) * 9 / 10) / max_dim).max(1);
        let chunk = cap_buf.min(cap_disp).min(num_specs).max(1);
        Some(chunk)
    }

    /// #wg-limit-subchunk: run the sound-resident backward in `chunk`-row batches and
    /// concatenate the per-row bounds. Each chunk carries the SAME layers / input box;
    /// only the spec-row slice differs. EXACT (each row's bound equals its
    /// single-dispatch value — no cross-row reduction). Any chunk `Err` propagates
    /// (→ caller's CPU sound fallback).
    ///
    /// TAIL: the loop is a half-open walk `[start, min(start+chunk, num_specs))`, so a
    /// `num_specs` that is NOT a multiple of `chunk` yields a final short chunk of
    /// `num_specs % chunk` rows — never a truncated, over-read, or duplicated row. Each
    /// chunk asserts it returned exactly `rows` lower AND `rows` upper bounds before
    /// appending, so a kernel that silently returned the wrong row count `Err`s here
    /// instead of shifting every later row's bound. The concatenation preserves spec
    /// order (`lower_bounds[s]` is spec row `s`), which the callers index positionally.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crown_backward_sound_chunked(
        &self,
        layers: &[GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        output_dim: usize,
        chunk: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        let chunk = chunk.max(1);
        // PANIC GUARD (required by the default flip): the row slice below indexes
        // `spec[start*output_dim .. end*output_dim]`, so an under-sized `spec` would
        // PANIC instead of returning `Err`. Reachable because the caller only lands
        // here after the unchunked attempt already returned `Err` — and one of the
        // errors it can return is exactly a `spec`-length `shape_mismatch` (the
        // resident fold checks `lower_a.len() != num_specs*output_dim` BEFORE the
        // dispatch-cap guard). While the chunking was opt-in this path was unreachable
        // in production; as the default it must fail closed, not abort the verifier.
        if output_dim == 0 || spec.len() < num_specs * output_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, output_dim],
                vec![spec.len()],
            ));
        }
        let probe = std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1");
        if probe {
            eprintln!(
                "[chunk] num_specs={num_specs} output_dim={output_dim} chunk={chunk} n_chunks={}",
                num_specs.div_ceil(chunk)
            );
        }
        let mut lower_bounds = Vec::with_capacity(num_specs);
        let mut upper_bounds = Vec::with_capacity(num_specs);
        let mut start = 0usize;
        while start < num_specs {
            let end = (start + chunk).min(num_specs);
            let rows = end - start;
            let spec_chunk = &spec[start * output_dim..end * output_dim];
            let t0 = Instant::now();
            let (lo, hi) = self.crown_backward_sound_resident(
                layers,
                spec_chunk,
                rows,
                output_dim,
                input_lower,
                input_upper,
            )?;
            if probe {
                eprintln!(
                    "[chunk]   rows={rows} took {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            if lo.len() != rows || hi.len() != rows {
                return Err(NyError::InvalidSpec(format!(
                    "crown_backward_sound_chunked: chunk returned {} lo / {} hi bounds, expected {}",
                    lo.len(),
                    hi.len(),
                    rows
                )));
            }
            lower_bounds.extend_from_slice(&lo);
            upper_bounds.extend_from_slice(&hi);
            start = end;
        }
        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }

    /// #batched-bab increment 3: assemble the wide inputs (stacked skeleton, tiled seed,
    /// per-domain-block box / β / abs-max tables) and run ONE wide resident β-CROWN pass
    /// over `N = n_domains * num_specs` rows, then unpack the N rows back into one
    /// [`GpuCrownResult`] per domain (block `d` = rows `[d*nsp, (d+1)*nsp)`). Returns
    /// `None` on any assembly/shape failure OR a GPU error so the caller uses the
    /// byte-identical serial reference stacker (the 0-wrong moat). SOUND: the wide pass
    /// dom-block-indexes every per-domain input, so each block's bound equals its serial
    /// per-domain bound (up to f32 GEMM-reorder); the two-sided differential oracle
    /// verifies this. Assumes the homogeneity + HOLE-8 gates already passed.
    ///
    /// #batched-bab part A: `union_gather_idx` (per-ReLU union split columns, fold order;
    /// `&[]` = bound-only) is passed to the wide inner; the returned `gathers[r]` =
    /// N×|union_gather_idx[r]| row-major (A_lower at the union columns for every wide row).
    /// The bound-only entry [`try_wide_resnet_batched`] is a thin wrapper passing `&[]`.
    /// #wg-limit-subchunk (SOUNDNESS + throughput): device-limit-safe wrapper around
    /// [`Self::try_wide_resnet_batched_grad_group`]. The single wide pass dispatches
    /// `ceil(N*W/256)` workgroups (`N = n_domains * nsp`), and the sound concretize
    /// dispatches `N` — both capped by `max_compute_workgroups_per_dimension` (kept at
    /// the wgpu default 65535; `NY_GPU_BIG_BINDINGS` only raises binding SIZE, not this).
    /// Overrunning it is a latent false-VERIFY hole (some drivers silently return an
    /// over-tight bound instead of erroring) AND a crash risk. So when the batch would
    /// overflow, we sub-chunk the DOMAINS into device-safe groups and run one wide pass
    /// per group — each group folds against its OWN per-domain state with no cross-domain
    /// reduction, so per-domain bounds/grads/gathers are BIT-IDENTICAL to the single-pass
    /// result. This honors an arbitrarily large `NY_MO_GPU_CHUNK` by LOOPING, never by
    /// overrunning the limit (and never by falling back to the slow per-child path).
    fn try_wide_resnet_batched_grad(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        relu_pre_lower: &[&[Vec<f32>]],
        coeff_full_out: Option<&mut GpuResidentCoeffBatched>,
        force_fine_coeff: bool,
    ) -> Option<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n_domains = domains.len();
        let nsp = seed.num_specs; // per-domain spec-row count
        if n_domains == 0 || nsp == 0 {
            return note_wide_lane_decline_none(WideLaneDecline::GpuEmptyBatch);
        }
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1) as usize;
        let width = resnet_segments_max_1d_dispatch_dim(domains[0].segments);
        let Some(safe_domains) = wide_safe_domain_count(max_wg, width, nsp) else {
            return note_wide_lane_decline_none(WideLaneDecline::GpuDispatchLimitTooWide);
        };
        if !relu_pre_lower.is_empty() && relu_pre_lower.len() != n_domains {
            return note_wide_lane_decline_none(WideLaneDecline::GpuPreLowerShapeMismatch);
        }

        // Common case: the whole batch fits in one wide pass (byte-identical to the
        // pre-fix path). Coefficient captures below decline when the whole batch does
        // not fit because concatenating `GpuResidentCoeffBatched` across sub-chunks is
        // not yet implemented in this backend.
        if n_domains <= safe_domains {
            let out = self.try_wide_resnet_batched_grad_group(
                domains,
                seed,
                union_gather_idx,
                relu_pre_lower,
                coeff_full_out,
                force_fine_coeff,
            );
            if out.is_some() {
                WIDE_RESNET_BATCHED_TAKEN.fetch_add(1, Ordering::Relaxed);
            }
            return out;
        }
        // A coefficient frontier must remain one domain-major object.  Until
        // this backend has a checked concatenator, decline an over-limit capture
        // instead of issuing an invalid oversized dispatch.
        if coeff_full_out.is_some() {
            return note_wide_lane_decline_none(WideLaneDecline::GpuCoeffOverDispatchLimit);
        }

        if std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1") {
            eprintln!(
                "[wide] SUBCHUNK n_domains={n_domains} nsp={nsp} width={width} max_wg={max_wg} \
                 safe_domains={safe_domains} n_groups={}",
                n_domains.div_ceil(safe_domains)
            );
        }
        let mut out_bounds: Vec<GpuCrownResult> = Vec::with_capacity(n_domains);
        let mut out_alpha: Vec<Vec<f32>> = Vec::new();
        let mut out_gathers: Vec<Vec<f32>> = Vec::new();
        let mut first = true;
        for (group_index, group) in domains.chunks(safe_domains).enumerate() {
            let start = group_index * safe_domains;
            let end = start + group.len();
            let Some(group_pre_lower) =
                wide_domain_table_chunk(relu_pre_lower, n_domains, start, end)
            else {
                return note_wide_lane_decline_none(WideLaneDecline::GpuPreLowerShapeMismatch);
            };
            let Some((b, ag, gv)) = self.try_wide_resnet_batched_grad_group(
                group,
                seed,
                union_gather_idx,
                group_pre_lower,
                None,
                false,
            ) else {
                // The group recorded its own precise reason; this marks that a
                // partially-completed sub-chunked batch was abandoned wholesale.
                return note_wide_lane_decline_none(WideLaneDecline::GpuSubchunkGroupFailed);
            };
            out_bounds.extend(b);
            if first {
                out_alpha = ag;
                out_gathers = gv;
                first = false;
            } else {
                // Per-ReLU domain-stacked concat: groups are visited in domain order,
                // so appending each group's block preserves the global domain layout
                // (`alpha_grads[r][dom*nn + i]`, `gathers[r]` row-major N×U_r).
                if ag.len() != out_alpha.len() || gv.len() != out_gathers.len() {
                    return note_wide_lane_decline_none(WideLaneDecline::GpuSubchunkGroupFailed);
                }
                for (dst, src) in out_alpha.iter_mut().zip(ag) {
                    dst.extend(src);
                }
                for (dst, src) in out_gathers.iter_mut().zip(gv) {
                    dst.extend(src);
                }
            }
        }
        WIDE_RESNET_BATCHED_TAKEN.fetch_add(1, Ordering::Relaxed);
        Some((out_bounds, out_alpha, out_gathers))
    }

    /// One wide resident β-CROWN pass over ALL given domains stacked into `N =
    /// n_domains * nsp` rows. Assumes the batch already fits the device dispatch limit
    /// (the [`Self::try_wide_resnet_batched_grad`] wrapper guarantees this by
    /// sub-chunking). See that wrapper for the soundness rationale.
    fn try_wide_resnet_batched_grad_group(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
        union_gather_idx: &[&[u32]],
        // #w4 wide α+β ascent: per-domain per-ReLU (fold order) pre-activation lower
        // bounds with stable neurons masked to 0 — the alpha-gradient request. Empty
        // ⇒ no capture (returned alpha grads empty), bounds byte-for-byte unchanged.
        relu_pre_lower: &[&[Vec<f32>]],
        // #clip-interm-resnet-batched: when Some, receives the batched coeff frontier
        // (all rows N×dim). `force_fine_coeff` distinguishes the clip's force-fine
        // carrier from Hydra's first-pass trajectory carrier. None ⇒ no capture.
        coeff_full_out: Option<&mut GpuResidentCoeffBatched>,
        force_fine_coeff: bool,
    ) -> Option<(Vec<GpuCrownResult>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n_domains = domains.len();
        let nsp = seed.num_specs; // per-domain spec-row count
        if nsp == 0 {
            return note_wide_lane_decline_none(WideLaneDecline::GpuSeedShapeRefused);
        }
        let Some(n) = n_domains.checked_mul(nsp) else {
            return note_wide_lane_decline_none(WideLaneDecline::GpuSeedShapeRefused);
        };
        let od = seed.current_dim;
        let probe = std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1");
        // Wide skeleton: shared weights + per-domain-STACKED Activation slopes.
        let Some(wide_segments) = stack_wide_segments(domains) else {
            if probe {
                eprintln!("[wide] BAIL stack_wide_segments None (n_domains={n_domains})");
            }
            return note_wide_lane_decline_none(WideLaneDecline::GpuSegmentStackRefused);
        };
        // Tile the SHARED spec seed n_domains times → N rows (each domain block starts
        // from the identical seed, exactly as the serial per-domain calls do).
        if seed.lower_a.len() != nsp * od
            || seed.upper_a.len() != nsp * od
            || seed.lower_b.len() != nsp
            || seed.upper_b.len() != nsp
        {
            if probe {
                eprintln!(
                    "[wide] BAIL seed shape: lower_a={} nsp*od={} lower_b={} nsp={nsp}",
                    seed.lower_a.len(),
                    nsp * od,
                    seed.lower_b.len()
                );
            }
            return note_wide_lane_decline_none(WideLaneDecline::GpuSeedShapeRefused);
        }
        let mut wl_a = Vec::with_capacity(n * od);
        let mut wu_a = Vec::with_capacity(n * od);
        let mut wl_b = Vec::with_capacity(n);
        let mut wu_b = Vec::with_capacity(n);
        for _ in 0..n_domains {
            wl_a.extend_from_slice(&seed.lower_a);
            wu_a.extend_from_slice(&seed.upper_a);
            wl_b.extend_from_slice(&seed.lower_b);
            wu_b.extend_from_slice(&seed.upper_b);
        }
        let wide_seed = GpuCrownSeed {
            lower_a: wl_a.into(),
            upper_a: wu_a.into(),
            lower_b: wl_b.into(),
            upper_b: wu_b.into(),
            num_specs: n,
            current_dim: od,
        };
        // Wide input box: n_domains contiguous blocks of input_dim (HOLE 3).
        let input_dim = domains[0].input_lower.len();
        let mut w_lo = Vec::with_capacity(n_domains * input_dim);
        let mut w_hi = Vec::with_capacity(n_domains * input_dim);
        for d in domains {
            if d.input_lower.len() != input_dim || d.input_upper.len() != input_dim {
                if probe {
                    eprintln!(
                        "[wide] BAIL input box: dom lo={} hi={} input_dim={input_dim}",
                        d.input_lower.len(),
                        d.input_upper.len()
                    );
                }
                return note_wide_lane_decline_none(WideLaneDecline::GpuInputBoxMismatch);
            }
            w_lo.extend_from_slice(d.input_lower);
            w_hi.extend_from_slice(d.input_upper);
        }
        // Wide β (per ReLU) / frontier (per segment) / node (per ReLU) abs-max tables,
        // each stacked into n_domains blocks (HOLES 1/4). Validated equal-length.
        let beta_tbls: Vec<&[Vec<f32>]> = domains.iter().map(|d| d.beta_signed).collect();
        let fa_tbls: Vec<&[Vec<f32>]> = domains.iter().map(|d| d.frontier_abs).collect();
        let na_tbls: Vec<&[Vec<f32>]> = domains.iter().map(|d| d.node_abs).collect();
        let bail_tbl = |which: &str| {
            if probe {
                let shapes = |t: &[&[Vec<f32>]]| {
                    t.iter()
                        .map(|d| d.iter().map(|v| v.len()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "[wide] BAIL stack_wide_table({which}): beta={:?} fa={:?} na={:?}",
                    shapes(&beta_tbls),
                    shapes(&fa_tbls),
                    shapes(&na_tbls)
                );
            }
        };
        let Some(wide_beta) = stack_wide_table(&beta_tbls) else {
            bail_tbl("beta");
            return note_wide_lane_decline_none(WideLaneDecline::GpuBetaTableStackRefused);
        };
        let Some(wide_fa) = stack_wide_table(&fa_tbls) else {
            bail_tbl("fa");
            return note_wide_lane_decline_none(WideLaneDecline::GpuFrontierTableStackRefused);
        };
        let Some(wide_na) = stack_wide_table(&na_tbls) else {
            bail_tbl("na");
            return note_wide_lane_decline_none(WideLaneDecline::GpuNodeTableStackRefused);
        };
        let bs: Vec<&[f32]> = wide_beta.iter().map(|v| v.as_slice()).collect();
        let fa: Vec<&[f32]> = wide_fa.iter().map(|v| v.as_slice()).collect();
        let na: Vec<&[f32]> = wide_na.iter().map(|v| v.as_slice()).collect();
        // #w4 wide α+β ascent: stack the per-domain per-ReLU pre-lower tables into
        // n_domains blocks (same layout contract as beta/node tables). Empty ⇒ no
        // alpha-gradient capture.
        let wide_pl = if relu_pre_lower.is_empty() {
            Vec::new()
        } else {
            let Some(w) = stack_wide_table(relu_pre_lower) else {
                bail_tbl("pre_lower");
                return note_wide_lane_decline_none(WideLaneDecline::GpuPreLowerTableStackRefused);
            };
            w
        };
        let pl: Vec<&[f32]> = wide_pl.iter().map(|v| v.as_slice()).collect();
        // `union_gather_idx` = &[] for the bound-only wrapper (no-op copy) or the per-ReLU
        // union columns for the wide β-opt grad path.
        let wide_res = self.crown_backward_gpu_resnet_sound_beta_wide_inner(
            &wide_segments,
            &wide_seed,
            nsp,
            &w_lo,
            &w_hi,
            &bs,
            &fa,
            &na,
            union_gather_idx,
            &pl,
            coeff_full_out,
            force_fine_coeff,
        );
        let (lo, hi, alpha_grads, gathers) = match wide_res {
            Ok(v) => v,
            Err(e) => {
                if std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "[wide] ERR n_domains={n_domains} nsp={nsp} N={n} od={od} segs={} beta={} fa={} na={} in_dim={input_dim}: {e}",
                        wide_segments.len(),
                        wide_beta.len(),
                        wide_fa.len(),
                        wide_na.len()
                    );
                }
                // Structurally classify the refusal so the tally distinguishes a
                // budget expiry (schedule) from a host memory cap (sizing) from a
                // genuine device/shape fault — three different fixes.
                return note_wide_lane_decline_none(if e.is_deadline_exceeded() {
                    WideLaneDecline::GpuWideInnerDeadline
                } else if e.is_cpu_memory_exceeded() {
                    WideLaneDecline::GpuWideInnerMemoryCap
                } else {
                    WideLaneDecline::GpuWideInnerError
                });
            }
        };
        if lo.len() != n || hi.len() != n {
            return note_wide_lane_decline_none(WideLaneDecline::GpuWideOutputLenMismatch);
        }
        if std::env::var("NY_WIDE_PROBE").ok().as_deref() == Some("1") {
            eprintln!(
                "[wide] FIRED n_domains={n_domains} nsp={nsp} N={n} segs={} beta={} fa={} na={}",
                wide_segments.len(),
                wide_beta.len(),
                wide_fa.len(),
                wide_na.len()
            );
        }
        // Unpack: block d = rows [d*nsp, (d+1)*nsp).
        let mut out = Vec::with_capacity(n_domains);
        for d in 0..n_domains {
            let s = d * nsp;
            out.push(GpuCrownResult {
                lower_bounds: lo[s..s + nsp].to_vec(),
                upper_bounds: hi[s..s + nsp].to_vec(),
            });
        }
        Some((out, alpha_grads, gathers))
    }

    /// Bound-only wide batched pass (thin wrapper): the wide β-opt grad path with
    /// EMPTY gather + alpha channels ⇒ byte-identical bounds, no capture copies. See
    /// [`Self::try_wide_resnet_batched_grad`].
    fn try_wide_resnet_batched(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Option<Vec<GpuCrownResult>> {
        self.try_wide_resnet_batched_grad(domains, seed, &[], &[], None, false)
            .map(|(bounds, _alpha_grads, _gathers)| bounds)
    }

    /// #clip-interm-resnet-batched: coeff-capturing wide batched pass. Runs the SAME
    /// single wide resident backward as [`Self::try_wide_resnet_batched`] over the whole
    /// domain frontier and ALSO downloads the input-relative coefficient frontier
    /// ([`GpuResidentCoeffBatched`], `N × dim` rows, `num_specs_per_dom = nsp`), captured
    /// from the force-fine pass. `None` on any wide-assembly/shape/GPU failure ⇒ the
    /// caller drops the clip for this batch (sound: keeps frozen intermediates).
    fn try_wide_resnet_batched_coeff(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Option<(Vec<GpuCrownResult>, GpuResidentCoeffBatched)> {
        let mut coeff = GpuResidentCoeffBatched {
            lower_a: Vec::new(),
            upper_a: Vec::new(),
            lower_err: Vec::new(),
            upper_err: Vec::new(),
            lower_b: Vec::new(),
            upper_b: Vec::new(),
            lower_b_err: Vec::new(),
            upper_b_err: Vec::new(),
            dim: 0,
            num_specs: 0,
            num_specs_per_dom: 0,
        };
        let (bounds, _alpha_grads, _gathers) =
            self.try_wide_resnet_batched_grad(domains, seed, &[], &[], Some(&mut coeff), true)?;
        // Guard: a coeff request that produced an empty frontier is unusable.
        if coeff.dim == 0 || coeff.num_specs == 0 || coeff.num_specs_per_dom == 0 {
            return note_wide_lane_decline_none(WideLaneDecline::GpuCoeffFrontierEmpty);
        }
        Some((bounds, coeff))
    }

    /// #batched-bab HOLE-7 SUB-GROUPING (DARK, `NY_BAB_RESNET_WIDE_SUBGROUP=1`) —
    /// the coverage increment for heterogeneous waves.
    ///
    /// Today a single odd domain sends the WHOLE batch to the serial path. This
    /// splits `domains` into MAXIMAL CONTIGUOUS runs of skeleton-equal domains and
    /// runs one wide pass per run. Soundness rests on exactly the argument the
    /// homogeneous lane already relies on:
    /// - every run satisfies the wide fold's precondition (one shared skeleton,
    ///   per-domain relaxation VALUES only), so each run's per-domain block is
    ///   folded against its OWN box / β / abs-max tables — a run is not "part of"
    ///   a bigger reduction, and no state crosses runs;
    /// - runs are visited in domain order and their results appended in order, so
    ///   output slot `i` is domain `i`'s own bound (the caller's contract);
    /// - a run whose head carries HOLE-8 layer kinds, or whose wide assembly
    ///   refuses for any reason, declines the ENTIRE batch (`None`) so the caller
    ///   falls back exactly as today. There is never a partial result and never a
    ///   mixture of a wide run with a serial one. (HOLE 8 is decided for EVERY run
    ///   before the first dispatch, so an unfoldable wave costs zero GPU work; a
    ///   late assembly refusal in phase 2 can still discard already-computed runs —
    ///   wasted work, never an unsound or partial answer.)
    ///
    /// A run of length 1 is admissible: the wide fold with `n_domains == 1`
    /// degenerates to the same single-block computation the per-domain sound inner
    /// performs, so it neither tightens nor loosens.
    ///
    /// DEFAULT OFF. Enabling it changes which kernel produces a verdict-bearing
    /// bound on heterogeneous waves, so it stays dark until the device-level
    /// enclosure oracle (`wide_subgrouped_encloses_serial_reference`) has been run
    /// on real heterogeneous waves on the target adapter.
    fn try_wide_resnet_batched_subgrouped(
        &self,
        domains: &[GpuResnetBatchedDomainRef<'_>],
        seed: &GpuCrownSeed,
    ) -> Option<Vec<GpuCrownResult>> {
        if !wide_subgroup_enabled() {
            return None;
        }
        // The A/B kill-switch for the wide kernel outranks the sub-grouping lane:
        // `NY_BAB_RESNET_WIDE=0` must mean "no wide dispatches at all".
        if !wide_resnet_enabled() {
            return None;
        }
        if domains.len() < 2 {
            return None;
        }
        // PHASE 1 — partition and validate WITHOUT dispatching. Deciding HOLE 8 for
        // every run up front is what makes the refusal genuinely fail-closed: a
        // wave with one unfoldable run must cost zero wide dispatches, not publish
        // the earlier runs and then abandon them.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < domains.len() {
            let head = &domains[start];
            if segments_contain_unbatchable(head.segments) {
                return None;
            }
            let mut end = start + 1;
            while end < domains.len()
                && resnet_skeleton_matches(head.segments, domains[end].segments)
            {
                end += 1;
            }
            runs.push((start, end));
            start = end;
        }
        // A wave that turned out to be one homogeneous run does not belong here:
        // the caller's ordinary wide path already handles it (and calling it again
        // would double-count a publication).
        if runs.len() < 2 {
            return None;
        }
        // PHASE 2 — one wide pass per run, appended in domain order.
        let mut out: Vec<GpuCrownResult> = Vec::with_capacity(domains.len());
        for (run_start, run_end) in runs {
            out.extend(self.try_wide_resnet_batched(&domains[run_start..run_end], seed)?);
        }
        // Defensive: the caller indexes results by domain, so a count drift must
        // decline rather than silently re-associate bounds with other domains.
        (out.len() == domains.len()).then_some(out)
    }

    fn crown_backward_gpu_seeded_inner(
        &self,
        layers: &[GpuCrownLayer],
        seed: &GpuCrownSeed,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        if layers.is_empty() {
            return Err(NyError::InvalidSpec(
                "crown_backward_gpu_seeded: empty layer list".into(),
            ));
        }
        let num_specs = seed.num_specs;
        let first_dim = layer_output_dim(&layers[0])?;
        if seed.current_dim != first_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, first_dim],
                vec![num_specs, seed.current_dim],
            ));
        }
        if seed.lower_a.len() != num_specs * first_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, first_dim],
                vec![seed.lower_a.len()],
            ));
        }
        if seed.upper_a.len() != num_specs * first_dim {
            return Err(NyError::shape_mismatch(
                vec![num_specs, first_dim],
                vec![seed.upper_a.len()],
            ));
        }
        if seed.lower_b.len() != num_specs {
            return Err(NyError::shape_mismatch(
                vec![num_specs],
                vec![seed.lower_b.len()],
            ));
        }
        if seed.upper_b.len() != num_specs {
            return Err(NyError::shape_mismatch(
                vec![num_specs],
                vec![seed.upper_b.len()],
            ));
        }
        if input_lower.len() != input_upper.len() {
            return Err(NyError::shape_mismatch(
                vec![input_lower.len()],
                vec![input_upper.len()],
            ));
        }
        let profiling_enabled = self.crown_timestamp_profiling_enabled()?;
        let host_profiling_enabled = self.crown_host_timing_profiling_enabled()?;
        self.store_crown_host_timing_profile(None)?;
        self.store_crown_timestamp_profile(None)?;

        // Memory budget gate (#3515): intersect the existing binding-size batch
        // cap with the total-working-set budget so oversized workloads are
        // reduced to safe spec batches before any GPU allocations occur.
        let estimate = estimate_crown_backward_memory(layers, num_specs);
        let budget = gpu_memory_budget_bytes();
        let binding_batch =
            max_specs_per_batch(layers, num_specs, first_dim, self.max_binding_bytes_live());
        let budget_batch = max_specs_per_budget(layers, num_specs, budget);
        let dispatch_batch = max_specs_per_dispatch(layers, num_specs);

        if budget_batch == 0 {
            let minimum_required = estimate_crown_backward_memory(layers, 1);
            tracing::warn!(
                required_mb = estimate.total_bytes / (1024 * 1024),
                minimum_batch_mb = minimum_required.total_bytes / (1024 * 1024),
                budget_mb = budget / (1024 * 1024),
                a_matrix_mb = estimate.a_matrix_bytes / (1024 * 1024),
                conv_mb = estimate.conv_bytes / (1024 * 1024),
                misc_mb = estimate.misc_bytes / (1024 * 1024),
                "GPU CROWN backward exceeds memory budget — falling back to CPU (#3515)",
            );
            return Err(NyError::GpuMemoryExceeded {
                required_bytes: minimum_required.total_bytes,
                budget_bytes: budget,
            });
        }
        if dispatch_batch == 0 {
            return Err(NyError::UnsupportedConfiguration(
                "GPU CROWN backward dispatch limit exceeded even for one spec batch".into(),
            ));
        }

        let batch_size = binding_batch.min(budget_batch).min(dispatch_batch);
        if batch_size < num_specs {
            tracing::info!(
                requested_specs = num_specs,
                binding_batch,
                budget_batch,
                dispatch_batch,
                chosen_batch = batch_size,
                budget_mb = budget / (1024 * 1024),
                estimated_full_mb = estimate.total_bytes / (1024 * 1024),
                "GPU CROWN backward reduced to spec batches before allocation (#3515, #3599, #3813)",
            );
            let (result, profile, host_profile) = self.crown_backward_gpu_batched_seeded(
                layers,
                seed,
                first_dim,
                batch_size,
                input_lower,
                input_upper,
                profiling_enabled,
                host_profiling_enabled,
            )?;
            self.store_crown_host_timing_profile(host_profile)?;
            self.store_crown_timestamp_profile(profile)?;
            return Ok(result);
        }

        let mut host_profile = host_profiling_enabled.then(CrownHostTimingProfile::default);
        let plan = record_host_phase(&mut host_profile, "plan_prepare", || {
            self.get_or_prepare_crown_plan(layers, num_specs, first_dim)
        })?;
        let mut profiler = if profiling_enabled {
            CrownTimestampProfiler::new(self, count_encoded_compute_passes(&plan.steps))?
        } else {
            None
        };
        let w = &plan.working;

        // Upload initial data to plan's dedicated working buffers (#3397 Step 4).
        // No pool lock needed — buffers are owned by the plan and stable.
        record_host_phase(&mut host_profile, "write_initial_state", || {
            self.queue
                .write_buffer(&w.a_lower_0, 0, bytemuck::cast_slice(seed.lower_a.as_ref()));
            self.queue
                .write_buffer(&w.a_upper_0, 0, bytemuck::cast_slice(seed.upper_a.as_ref()));
            self.queue.write_buffer(
                &w.bias_lower,
                0,
                bytemuck::cast_slice(seed.lower_b.as_ref()),
            );
            self.queue.write_buffer(
                &w.bias_upper,
                0,
                bytemuck::cast_slice(seed.upper_b.as_ref()),
            );
            self.queue
                .write_buffer(&w.inp_lower_buf, 0, bytemuck::cast_slice(input_lower));
            self.queue
                .write_buffer(&w.inp_upper_buf, 0, bytemuck::cast_slice(input_upper));
        });

        // Refresh dynamic activation slopes/intercepts in the staging buffer.
        record_host_phase(&mut host_profile, "refresh_dynamics", || {
            self.refresh_crown_plan_dynamic_layers(&plan, layers)
        })?;

        // Encode all dispatch steps using pre-built bind groups (#3397 Step 4).
        let mut encoder = record_host_phase(&mut host_profile, "create_encoder", || {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("crown_backward_cached"),
                })
        });

        record_host_phase(&mut host_profile, "encode_steps", || {
            self.encode_crown_steps_cached(&mut encoder, &plan, profiler.as_mut())
        })?;

        // Readback copies in the same encoder
        let out_bytes = (num_specs * size_of::<f32>()) as u64;
        record_host_phase(&mut host_profile, "encode_readback", || {
            encoder.copy_buffer_to_buffer(&w.out_lower, 0, &w.readback_lower, 0, out_bytes);
            encoder.copy_buffer_to_buffer(&w.out_upper, 0, &w.readback_upper, 0, out_bytes);
            if let Some(profiler) = profiler.as_ref() {
                profiler.encode_resolve(&mut encoder);
            }
        });

        // Single submit for entire backward pass + readback
        record_host_phase(&mut host_profile, "queue_submit", || {
            self.queue.submit(std::iter::once(encoder.finish()))
        });

        // Concurrent readback: map both buffers with a single device.poll (#3397).
        let readback_start = Instant::now();
        let (mut lower_bounds, mut upper_bounds, readback_timing) =
            Self::read_two_buffers_profiled(
                &self.device,
                &w.readback_lower,
                &w.readback_upper,
                num_specs,
                num_specs,
            )?;
        if let Some(profile) = host_profile.as_mut() {
            profile.record(
                "readback_map_requests",
                readback_timing.map_requests_seconds,
            );
            profile.record("readback_poll_wait", readback_timing.poll_wait_seconds);
            profile.record("readback_copy", readback_timing.copy_seconds);
            let measured_readback = readback_timing.map_requests_seconds
                + readback_timing.poll_wait_seconds
                + readback_timing.copy_seconds;
            let residual = (readback_start.elapsed().as_secs_f64() - measured_readback).max(0.0);
            if residual > 0.0 {
                profile.record("readback_runtime_overhead", residual);
            }
        }
        record_host_phase(&mut host_profile, "sanitize_readback", || {
            sanitize_readback(&mut lower_bounds, &mut upper_bounds)
        });
        let profile = record_host_phase(&mut host_profile, "timestamp_profile_readback", || {
            profiler.map(|profiler| profiler.finish(self)).transpose()
        })?;
        self.store_crown_host_timing_profile(host_profile)?;
        self.store_crown_timestamp_profile(profile)?;

        Ok(GpuCrownResult {
            lower_bounds,
            upper_bounds,
        })
    }
}

#[cfg(test)]
mod tests;
