// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-linear intermediate bounds for graph/DAG networks.
//! Collect per-node `LinearBounds` relative to the original input, then concretize them
//! back to `BoundedTensor` node bounds. The first packet stays intentionally narrow:
//! support the nn4sys-style DAG operator surface and fail closed instead of degrading to IBP.

pub(crate) mod alpha_opt;
mod binary;
mod concat;
mod image;

use std::borrow::Cow;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use ndarray::{Array1, Array2, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer};

use super::{
    ForwardLinearCacheEntry, ForwardLinearCacheFingerprint, GraphNetwork, MarginOptMemoFingerprint,
    NETWORK_INPUT,
};

#[cfg(test)]
thread_local! {
    static FORWARD_LINEAR_COLLECTION_REQUESTS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only observation scope for fixed-slope cached collector requests.
/// Counts requests, including cache hits, so regressions can detect a redundant
/// same-episode retry that would become an expensive build after any refusal.
#[cfg(test)]
pub(crate) struct ForwardLinearCollectionRequestCounter {
    previous: Option<usize>,
}

#[cfg(test)]
impl ForwardLinearCollectionRequestCounter {
    pub(crate) fn start() -> Self {
        let previous = FORWARD_LINEAR_COLLECTION_REQUESTS.with(|slot| slot.replace(Some(0)));
        Self { previous }
    }

    pub(crate) fn requests(&self) -> usize {
        FORWARD_LINEAR_COLLECTION_REQUESTS.with(|slot| {
            slot.get()
                .expect("forward-linear request counter scope must still be active")
        })
    }
}

#[cfg(test)]
impl Drop for ForwardLinearCollectionRequestCounter {
    fn drop(&mut self) {
        FORWARD_LINEAR_COLLECTION_REQUESTS.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
fn record_forward_linear_collection_request() {
    FORWARD_LINEAR_COLLECTION_REQUESTS.with(|slot| {
        if let Some(requests) = slot.get() {
            slot.set(Some(requests.saturating_add(1)));
        }
    });
}

/// Minimum remaining wall time required to start a cold image forward-linear
/// reference build.
///
/// A CIFAR-sized cold build contains f64 GEMMs that cannot be interrupted once
/// submitted.  The full pass is measured at roughly 22--25 seconds, so starting
/// it inside a 10-second verifier slice can hold a scoped cache warmer until the
/// competition watchdog fires.  Cached hits remain admissible at any deadline;
/// this floor only refuses optional cold work, whose callers already fail closed
/// to IBP/CROWN.  Thirty seconds retains five seconds of safety margin over the
/// slow end of the measured pass.
const FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM: std::time::Duration =
    std::time::Duration::from_secs(30);

fn forward_linear_cold_build_admitted_at(deadline: Option<Instant>, now: Instant) -> bool {
    deadline
        .is_none_or(|d| d.saturating_duration_since(now) >= FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM)
}

/// Last-resort forward-linear throughput constant, in MACs per second, used
/// only when the startup micro-calibration probe cannot run or fails
/// (#forward-linear-cost-gate).
///
/// Measured ONCE on `CIFAR100_resnet_medium`: the cold build is 559.4 G f64
/// MACs (19 convs + 2 Gemms, each costing `input_numel · out_h·out_w ·
/// in_c·kh·kw · out_c`, twice for the center and radius passes) and completed
/// in ~102 s on the original calibration host, i.e. ~5.5 GMAC/s. That number
/// proved ~4-5x stale for the current host class (the same build measures
/// ~24 s, `docs/FL_FIRST_MEASUREMENT_2026-08-02.md`), which is exactly why the
/// probe below replaced it as the primary source. Deliberately conservative:
/// a stale-slow fallback under-admits (status quo), never over-admits.
const FORWARD_LINEAR_F64_MACS_PER_SEC_FALLBACK: u128 = 5_500_000_000;

/// Admission safety margin: admit a cold build only when the remaining budget
/// covers `predicted x 5/4`. Over-admission is the expensive failure (the
/// floor-smoke shape: the pass runs to the deadline, returns nothing, and BaB
/// starves), so a marginal admit must still leave BaB a floor.
const FORWARD_LINEAR_ADMISSION_MARGIN_NUM: u128 = 5;
const FORWARD_LINEAR_ADMISSION_MARGIN_DEN: u128 = 4;

/// Derate applied to the probe's measured rate before it is used for
/// admission: covers the ~7% non-GEMM glue of a full pass plus the probe's
/// hot-cache optimism (a small resident GEMM vs 256 MB im2col chunking).
const FORWARD_LINEAR_CALIBRATION_DERATE: f64 = 0.8;

/// Wall-clock ceiling for one calibration rep. The probe runs through the
/// deadline dispatch chain (the branch every scored run executes), so this
/// also guarantees a stuck probe costs bounded time before falling back.
const FORWARD_LINEAR_CALIBRATION_REP_DEADLINE: std::time::Duration =
    std::time::Duration::from_secs(5);

/// Whether forward-linear admission may use the published instance deadline.
///
/// Exact `NY_FL_INSTANCE_BUDGET="1"`, read once, default OFF. With the gate
/// closed, admission retains the historical phase-local deadline.
fn forward_linear_instance_budget_enabled() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("NY_FL_INSTANCE_BUDGET").is_ok_and(|v| v == "1"))
}

/// Whether the sound f32 value-GEMM seam is armed for forward-linear builds.
///
/// Mirrors `image::forward_linear_f32_gemm_enabled` (private to the image
/// module, which is code-frozen); the flag contract is identical: opt-in via
/// the existing `NY_FORWARD_LINEAR_F32=1`, uncached environment read. The seam
/// changes which GEMM implementation the deadline dispatch chain reaches, so
/// both the calibrated rate and the cache fingerprint are keyed on it.
fn forward_linear_f32_seam_armed() -> bool {
    matches!(
        std::env::var("NY_FORWARD_LINEAR_F32").ok().as_deref(),
        Some("1")
    )
}

/// One process-wide calibration of the forward-linear build throughput
/// (#forward-linear-cost-gate). `macs_per_sec` is already derated and ready
/// for admission arithmetic.
#[derive(Debug, Clone, Copy)]
struct ForwardLinearRateCalibration {
    macs_per_sec: u128,
    /// "env" (manual `NY_FORWARD_LINEAR_MACS_PER_SEC` override), "probe"
    /// (measured by [`calibrate_forward_linear_rate`]), or "fallback" (the
    /// shipped constant, used when the probe could not run).
    source: &'static str,
    seam_f32: bool,
    /// Probe work and best rep duration; zero for "env"/"fallback".
    probe_macs: u64,
    probe_secs: f64,
}

/// Effective forward-linear build throughput for admission prediction.
///
/// Order of authority:
/// 1. `NY_FORWARD_LINEAR_MACS_PER_SEC` — the existing manual escape hatch,
///    absolute precedence, no probe runs.
/// 2. A per-process measured rate from [`calibrate_forward_linear_rate`],
///    cached in a `OnceLock` PER SEAM MODE (the f32 seam reaches a different
///    GEMM implementation under deadline, so the two rates differ ~2x on CPU
///    hosts and more where an accelerator serves only one of them).
/// 3. [`FORWARD_LINEAR_F64_MACS_PER_SEC_FALLBACK`] when the probe fails.
fn forward_linear_rate_calibration(
    engine: Option<&dyn GemmEngine>,
) -> ForwardLinearRateCalibration {
    let seam_f32 = forward_linear_f32_seam_armed();
    if let Some(rate) = std::env::var("NY_FORWARD_LINEAR_MACS_PER_SEC")
        .ok()
        .and_then(|v| v.parse::<u128>().ok())
        .filter(|&v| v > 0)
    {
        return ForwardLinearRateCalibration {
            macs_per_sec: rate,
            source: "env",
            seam_f32,
            probe_macs: 0,
            probe_secs: 0.0,
        };
    }
    static RATE_SEAM_OFF: std::sync::OnceLock<ForwardLinearRateCalibration> =
        std::sync::OnceLock::new();
    static RATE_SEAM_ON: std::sync::OnceLock<ForwardLinearRateCalibration> =
        std::sync::OnceLock::new();
    let slot = if seam_f32 {
        &RATE_SEAM_ON
    } else {
        &RATE_SEAM_OFF
    };
    *slot.get_or_init(|| calibrate_forward_linear_rate(engine, seam_f32))
}

/// Census-representative single-conv probe workload: contraction 1152
/// (= 128 in-channels x 3x3, the k>=1152 layers are ~98% of
/// `CIFAR100_resnet_medium`'s 559 GMAC), 16 out-channels over an 8x8 grid
/// (pad 1, stride 1) = 4.84 G value-GEMM MACs per pass — ~880 ms at the
/// fallback 5.5 GMAC/s, ~210 ms at the ~23 GMAC/s this host sustains.
///
/// Sized so fixed costs (rayon spin-up, im2col setup, graph plumbing) are
/// amortized: the original 1.21 G / ~80 ms probe measured 11-12 GMAC/s on a
/// host whose real 559 GMAC build sustains 23.3 (MEASURED, ~24 s under
/// load-21) — a 2x underestimate that kept rule 6 (#fl-phase-budget) below
/// its 17.48 GMAC/s widening threshold on hardware that qualifies.
fn forward_linear_calibration_fixture(seam_f32: bool) -> Option<(GraphNetwork, BoundedTensor)> {
    const IN_C: usize = 128;
    const HW: usize = 8;
    // Seam-on probes use the GPU-regime shape: OUT_C 64 puts each value GEMM
    // at ~2.1 GMAC — inside the measured 8.8-15.5x wgpu win band (m7,
    // 1bb88165) that dominates the real build, instead of the ~0.5 GMAC
    // break-even shape where a GPU-assisted chain measures CPU-flat and
    // under-admits FL on hosts that qualify. Seam-off keeps the cheap CPU
    // fixture (probe cost matters on CPU-only hosts).
    let out_c = if seam_f32 { 64 } else { 16 };
    let kernel_len = out_c * IN_C * 3 * 3;
    let kernel = ndarray::ArrayD::from_shape_vec(
        IxDyn(&[out_c, IN_C, 3, 3]),
        (0..kernel_len)
            .map(|i| ((i % 7) as f32 - 3.0) / 64.0)
            .collect(),
    )
    .ok()?;
    let conv = crate::layers::Conv2dLayer::with_input_shape(
        kernel,
        Some(Array1::zeros(out_c)),
        (1, 1),
        (1, 1),
        HW,
        HW,
    )
    .ok()?;
    let mut graph = GraphNetwork::new();
    graph.add_node(super::GraphNode::from_input(
        "fl_calibration_conv",
        Layer::Conv2d(conv),
    ));
    graph.set_output("fl_calibration_conv");
    let lower = ndarray::ArrayD::from_elem(IxDyn(&[IN_C, HW, HW]), -0.01f32);
    let upper = ndarray::ArrayD::from_elem(IxDyn(&[IN_C, HW, HW]), 0.01f32);
    let input = BoundedTensor::new(lower, upper).ok()?;
    Some((graph, input))
}
/// #fl-calib-diag: force every calibration rep and report each, so the
/// cold-vs-warm ratio can be measured instead of assumed. Diagnostics only --
/// it changes how many reps run, never a bound.
fn calibration_diag_enabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("NY_FL_CALIB_DIAG").is_ok_and(|v| v == "1"))
}

/// Startup micro-calibration: time the calibration fixture through
/// [`collect_forward_linear_state_dag`] — the EXACT production build path, so
/// the measured rate captures the full deadline dispatch precedence
/// (process-global sound-f64 engine > per-call engine > f32-tiled-if-armed >
/// f64-tiled CPU) plus im2col/scatter/composition glue, in the same MAC units
/// [`GraphNetwork::forward_linear_cold_build_macs`] predicts with. The seam
/// gate inside the pass reads `NY_FORWARD_LINEAR_F32` itself; `seam_f32` here
/// records which mode was measured (the caller keys the cache on it).
///
/// Up to three reps: rep 0 is a discarded warm-up (thread pools, allocator,
/// caches) whenever a later rep completes; best-of the measured reps. Reps
/// after the first are skipped when rep 0 already cost >1 s (slow host —
/// warm-up discard is then waived so the probe still reports a measurement).
/// Each rep is bounded by a 5 s deadline. Any error falls back to the
/// conservative shipped constant.
fn calibrate_forward_linear_rate(
    engine: Option<&dyn GemmEngine>,
    seam_f32: bool,
) -> ForwardLinearRateCalibration {
    let fallback = ForwardLinearRateCalibration {
        macs_per_sec: FORWARD_LINEAR_F64_MACS_PER_SEC_FALLBACK,
        source: "fallback",
        seam_f32,
        probe_macs: 0,
        probe_secs: 0.0,
    };
    let Some((graph, input)) = forward_linear_calibration_fixture(seam_f32) else {
        return fallback;
    };
    let Some(macs) = graph.forward_linear_cold_build_macs(input.len()) else {
        return fallback;
    };
    let mut best: Option<f64> = None;
    let mut warmup_elapsed: Option<f64> = None;
    for rep in 0..3 {
        let deadline = Instant::now() + FORWARD_LINEAR_CALIBRATION_REP_DEADLINE;
        let started = Instant::now();
        if collect_forward_linear_state_dag(
            &graph,
            &input,
            engine,
            Some(deadline),
            None,
            false,
            false,
        )
        .is_err()
        {
            return fallback;
        }
        let elapsed = started.elapsed().as_secs_f64();
        if calibration_diag_enabled() {
            // #fl-calib-diag: report every rep so the cold/warm ratio on THIS
            // host is a measurement rather than an assumption. The slow-host
            // waiver below means rep 0 is normally the only sample taken on any
            // CPU host, and rep 0 is by definition the cold one.
            eprintln!("[fl-calib] rep={rep} elapsed={elapsed:.4}s macs={macs}");
        }
        if rep == 0 {
            // Warm-up rep: discarded unless it turns out to be the only one.
            warmup_elapsed = Some(elapsed);
            if elapsed > 1.0 && !calibration_diag_enabled() {
                break;
            }
            continue;
        }
        best = Some(best.map_or(elapsed, |b: f64| b.min(elapsed)));
    }
    // Slow-host waiver: if no measured rep ran, fall back to the warm-up time.
    let best = best.or(warmup_elapsed);
    let Some(best) = best.filter(|b| b.is_finite() && *b > 0.0) else {
        return fallback;
    };
    let rate = (macs as f64 / best) * FORWARD_LINEAR_CALIBRATION_DERATE;
    if !rate.is_finite() || rate < 1.0 {
        return fallback;
    }
    let calibration = ForwardLinearRateCalibration {
        macs_per_sec: rate as u128,
        source: "probe",
        seam_f32,
        probe_macs: u64::try_from(macs).unwrap_or(u64::MAX),
        probe_secs: best,
    };
    info!(
        macs_per_sec = calibration.macs_per_sec as u64,
        probe_macs = calibration.probe_macs,
        probe_secs = calibration.probe_secs,
        seam_f32,
        "forward-linear affordability rate calibrated (#forward-linear-cost-gate)"
    );
    calibration
}

/// Measured forward-linear build throughput, exposed for pre-run phase
/// budgeting (#fl-phase-budget, I10).
///
/// The plan resolver needs the SAME rate the admission gate will later use so
/// its window arithmetic and the gate's admission arithmetic cannot disagree.
/// This returns the per-process calibration (env override > measured probe >
/// shipped fallback constant) — the probe runs at most once per process per
/// seam mode and is cached in the same `OnceLock` the gate reads, so calling
/// this at plan-resolution time PRE-PAYS the probe the gate would have paid
/// (~0.1s quiet, bounded by the 5s rep deadline).
#[derive(Debug, Clone, Copy)]
pub struct ForwardLinearRateObservation {
    /// Derated MACs/second, ready for admission arithmetic.
    pub macs_per_sec: u64,
    /// "env" | "probe" | "fallback" — see `forward_linear_rate_calibration`.
    pub source: &'static str,
    /// Which seam mode was measured (`NY_FORWARD_LINEAR_F32`).
    pub seam_f32: bool,
    /// Best probe rep duration; 0.0 unless `source == "probe"`.
    pub probe_secs: f64,
}

/// Run (or reuse) the per-process forward-linear rate calibration and return
/// the observation (#fl-phase-budget, I10). Uses the default engine
/// resolution path (`None`), i.e. the same deadline dispatch chain a scored
/// cold build executes.
pub fn forward_linear_measured_rate() -> ForwardLinearRateObservation {
    let calibration = forward_linear_rate_calibration(None);
    ForwardLinearRateObservation {
        macs_per_sec: u64::try_from(calibration.macs_per_sec).unwrap_or(u64::MAX),
        source: calibration.source,
        seam_f32: calibration.seam_f32,
        probe_secs: calibration.probe_secs,
    }
}

/// Flight-recorder snapshot of the most recent forward-linear cold-build
/// admission decision (#forward-linear-cost-gate, I7): the calibrated rate
/// with its provenance, the seam mode, and predicted-vs-remaining seconds.
/// Written at BOTH outcomes of the affordability gate; `None` until a
/// deadline-carrying cold build has been considered in this process.
#[derive(Debug, Clone)]
pub struct ForwardLinearAdmissionRecord {
    pub admitted: bool,
    pub seam_f32: bool,
    pub macs_per_sec: u64,
    /// "env" | "probe" | "fallback" — see [`forward_linear_rate_calibration`].
    pub rate_source: &'static str,
    pub predicted_secs: u64,
    pub remaining_secs: u64,
    /// Estimated MACs of the build that was being admitted.
    pub build_macs: u64,
    /// Probe workload and best rep duration (zero unless `rate_source` is
    /// "probe").
    pub probe_macs: u64,
    pub probe_secs: f64,
}

static LAST_FORWARD_LINEAR_ADMISSION: std::sync::Mutex<Option<ForwardLinearAdmissionRecord>> =
    std::sync::Mutex::new(None);

fn record_forward_linear_admission(record: ForwardLinearAdmissionRecord) {
    if let Ok(mut guard) = LAST_FORWARD_LINEAR_ADMISSION.lock() {
        *guard = Some(record);
    }
}

/// The most recent admission decision, for the CLI flight recorder.
pub fn forward_linear_admission_record() -> Option<ForwardLinearAdmissionRecord> {
    LAST_FORWARD_LINEAR_ADMISSION
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

impl GraphNetwork {
    /// Collect forward-linear intermediate bounds for supported DAG operators.
    ///
    /// Unlike `collect_node_bounds_with_engine`, this preserves affine
    /// correlations with the original input box instead of repeatedly
    /// concretizing to IBP at each node.
    pub fn collect_forward_linear_bounds_dag_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            None,
            None,
            Self::forward_linear_conv_transpose_reference_enabled(),
            false,
        )?;
        Ok(node_bounds)
    }

    /// Collect forward-linear intermediate bounds for supported DAG operators,
    /// aborting when the deadline is exceeded.
    pub fn collect_forward_linear_bounds_dag_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            deadline,
            None,
            Self::forward_linear_conv_transpose_reference_enabled(),
            false,
        )?;
        Ok(node_bounds)
    }

    /// Alpha-fed variant of
    /// [`Self::collect_forward_linear_bounds_dag_with_engine`]
    /// (#w4-root-alpha): image-mode ReLU nodes present in `relu_alphas` use
    /// the given per-neuron LOWER slopes (clamped to [0, 1], sound intercept
    /// 0 on crossing neurons — see `image::compose_relu_diag_forward`);
    /// absent nodes keep the adaptive rule. Uncached.
    ///
    /// Retained as an alpha-aware state-returning reference/audit seam. There
    /// is no production caller: the current #envelope-grad gate uses the
    /// point-forward surrogate in `backward/gradients.rs`.
    ///
    /// Per-node LINEAR functions over the input box under the given per-neuron
    /// lower slopes — the same pass as
    /// [`Self::collect_forward_linear_bounds_dag_with_alphas`], returning the
    /// affine state instead of its concretization.
    ///
    /// A future exact relaxed-linear implementation would need the FUNCTIONS,
    /// not the bounds: evaluating a node's lower linear function at the
    /// concretization argmin `x*` is the relaxed-linear forward value `ĥ(x*)`.
    /// Concretizing at a degenerate box `[x*, x*]` does NOT compute this — every
    /// neuron's own pre-activation is a point there, so the pass classifies it
    /// stable and applies the EXACT ReLU, silently ignoring the supplied slopes.
    #[allow(dead_code)] // Explicit state-returning seam retained for envelope-gradient audits.
    pub(crate) fn collect_forward_linear_state_dag_with_alphas(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, LinearBounds>> {
        let (_, linear) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            None,
            Some(relu_alphas),
            Self::forward_linear_conv_transpose_reference_enabled(),
            false,
        )?;
        Ok(linear)
    }

    /// NOTE: no production caller since 2026-08-12 — `envelope_binding_points`
    /// was its only one, and it now uses the much cheaper
    /// `collect_node_activations_pointwise` surrogate. The point forward matches
    /// the relaxed factor at the first ReLU but is heuristic deeper; this was a
    /// cost/approximation tradeoff, not a semantics-preserving reduction.
    /// Retained because the forward-linear image tests exercise it as the
    /// reference composition.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn collect_forward_linear_bounds_dag_with_alphas(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            None,
            Some(relu_alphas),
            Self::forward_linear_conv_transpose_reference_enabled(),
            false,
        )?;
        Ok(node_bounds)
    }

    /// Test-only direct entry for the dark ConvTranspose image surface.  This
    /// bypasses the production enable flag without mutating process-global env
    /// state (cargo tests run concurrently).
    #[cfg(test)]
    pub(crate) fn collect_forward_linear_bounds_dag_with_conv_transpose_for_test(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) =
            collect_forward_linear_state_dag(self, input, engine, None, None, true, false)?;
        Ok(node_bounds)
    }

    /// Test-only compatibility entry. Forces ConvTranspose composition OFF
    /// regardless of the process environment so tests can prove legacy routing.
    #[cfg(test)]
    pub(crate) fn collect_forward_linear_bounds_dag_without_conv_transpose_for_test(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let (node_bounds, _) =
            collect_forward_linear_state_dag(self, input, engine, None, None, false, false)?;
        Ok(node_bounds)
    }

    /// Batteries-included gate for the conv-DAG forward-linear reference-bounds
    /// source (#vnncomp-image-forward-linear): ON by default, opt out with
    /// `NY_NO_FORWARD_LINEAR_REF=1` (disable-flag principle). Shared by the
    /// alpha reference collection, the spec-propagation setup, and the CLI
    /// attack-phase cache warmer (#w5-bab-throughput) so all consult ONE policy.
    pub fn forward_linear_reference_enabled() -> bool {
        !matches!(
            std::env::var("NY_NO_FORWARD_LINEAR_REF").ok().as_deref(),
            Some("1")
        )
    }

    /// Gate for ConvTranspose2d/BatchNorm image forward-linear references:
    /// ON by default, opt out with `NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF=1`
    /// (disable-flag principle, as for `forward_linear_reference_enabled`).
    ///
    /// This was a dark enable-gate while the composition still routed through
    /// an f32 interval step, whose subnormal-flush fail-open capped the gain at
    /// "~40x without making the root decisive". That step is gone — the
    /// composition is now certified dense f64 center-radius — and the measured
    /// effect is much larger than the old comment recorded:
    ///
    /// - `cGAN_imgSz32_nCh_1 prop_1` root: `[-1048.89, 2204.65]` -> `[-2.033398,
    ///   0.752635]` (~1000x), with per-node widths within 1.002x of DeepPoly.
    /// - Enclosure audit over 3277 rows: 0 violations, against a mutation
    ///   control that does trip the same check.
    /// - Byte-inert on all 28 non-cGAN 2025 benchmarks (none carries a
    ///   ConvTranspose2d on a reference path). cgan2026 deliberately shares
    ///   this route because its seven model variants are byte-identical to the
    ///   2025 cGAN models; no broader cross-category inertness is claimed.
    ///
    /// Tightening a reference bound is sound-by-construction here: the value is
    /// an enclosure either way, and a narrower enclosure can only remove
    /// spurious relaxations. The unsealed legacy cgan_2023 projection currently
    /// records 12 official SAT rows, 7 timeouts, and 2 unknowns; the separate
    /// `test_nano` UNSAT is a synthetic regression moat. Those records guide
    /// regression testing but are not an audited verdict-parity claim.
    pub fn forward_linear_conv_transpose_reference_enabled() -> bool {
        !matches!(
            std::env::var("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF")
                .ok()
                .as_deref(),
            Some("1")
        )
    }

    /// Whether intermediate-bound collection should attempt the cached
    /// forward-linear reference source.
    ///
    /// Preserve the historical route for every non-sequential convolution
    /// graph, including Conv1d surfaces that may fail closed inside the
    /// collector. Additionally admit the sequential cGAN image surface when
    /// it contains both ConvTranspose2d and Conv2d. The latter remains behind
    /// the ConvTranspose-specific kill switch as well as the shared reference
    /// kill switch.
    pub(crate) fn should_collect_forward_linear_intermediate_reference(&self) -> bool {
        if !Self::forward_linear_reference_enabled() {
            return false;
        }

        let Ok(order) = self.exec_order() else {
            return false;
        };
        let sequential = self.is_sequential_graph(order);
        (!sequential && self.has_conv_layers())
            || (sequential
                && Self::forward_linear_conv_transpose_reference_enabled()
                && self.has_conv2d_layers()
                && self.has_conv_transpose2d_layers())
    }

    /// Whether an image-only forward-linear consumer should attempt the
    /// reference source.
    ///
    /// Root C-margin composition and alpha's Conv2d-DAG reference path must
    /// exclude the legacy Conv1d-only route admitted by
    /// [`Self::should_collect_forward_linear_intermediate_reference`]. Keeping
    /// that final image constraint here makes their kill-switch and sequential
    /// ConvTranspose routing identical to Step-1 intermediate collection.
    pub(crate) fn should_collect_forward_linear_image_reference(&self) -> bool {
        self.should_collect_forward_linear_intermediate_reference() && self.has_conv2d_layers()
    }

    /// Whether a cold forward-linear reference build has enough wall-clock
    /// headroom to start.  This is public so the CLI can avoid spawning a
    /// scoped optional warmer that the cache implementation would immediately
    /// refuse.  A warm cache is checked before this admission gate.
    pub fn forward_linear_cold_build_admitted(deadline: Option<Instant>) -> bool {
        forward_linear_cold_build_admitted_at(deadline, Instant::now())
    }

    /// Predicted f64 multiply-accumulate count of a cold forward-linear image
    /// pass over this graph (#forward-linear-cost-gate).
    ///
    /// The pass composes every node's affine map against the NETWORK INPUT, so
    /// each weighted node contributes `input_numel · (output spatial) ·
    /// (contraction) · (out channels)` MACs, and runs twice — once for the
    /// center and once for the radius (`conv_apply_rows_f64` is called for
    /// each). Non-weighted nodes (ReLU, Add, Flatten, …) are elementwise and
    /// negligible against the GEMMs.
    ///
    /// `None` when the graph has no shape information to estimate from, which
    /// keeps the caller on the fixed floor.
    pub(crate) fn forward_linear_cold_build_macs(&self, input_numel: usize) -> Option<u128> {
        let input_numel = u128::try_from(input_numel).ok()?;
        if input_numel == 0 {
            return None;
        }
        let mut total: u128 = 0;
        let mut saw_weighted = false;
        for name in self.exec_order().ok()? {
            let Some(node) = self.nodes.get(name) else {
                continue;
            };
            let per_row = match &node.layer {
                Layer::Conv2d(conv) => {
                    let (in_h, in_w) = conv.input_shape?;
                    let (kh, kw) = conv.kernel_size();
                    let (sh, sw) = conv.stride;
                    let (ph, pw) = conv.padding;
                    // Same out-dim formula the conv geometry resolver uses.
                    let out_h = (in_h + 2 * ph).checked_sub(kh)? / sh.max(1) + 1;
                    let out_w = (in_w + 2 * pw).checked_sub(kw)? / sw.max(1) + 1;
                    let contraction = conv.in_channels().checked_mul(kh)?.checked_mul(kw)?;
                    (out_h as u128)
                        .checked_mul(out_w as u128)?
                        .checked_mul(contraction as u128)?
                        .checked_mul(conv.out_channels() as u128)?
                }
                Layer::Linear(linear) => {
                    (linear.out_features() as u128).checked_mul(linear.in_features() as u128)?
                }
                _ => continue,
            };
            saw_weighted = true;
            // x2: the center and radius passes each run the full GEMM.
            total = total.checked_add(per_row.checked_mul(input_numel)?.checked_mul(2)?)?;
        }
        saw_weighted.then_some(total)
    }

    /// Whether a cold forward-linear build can be EXPECTED TO FINISH in the
    /// remaining budget (#forward-linear-cost-gate).
    ///
    /// The fixed [`FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM`] floor is blind to
    /// graph size: its comment calibrates it against a pass "measured at roughly
    /// 22--25 seconds", but `CIFAR100_resnet_medium` needs ~102 s. The
    /// consequence was that at the scored 100 s budget the pass STARTED, ran for
    /// 45 s — 45% of the whole budget — hit its deadline mid-GEMM, and returned
    /// nothing, falling back to plain IBP. That time is pure loss: the caller
    /// ends up exactly where it would have been had the pass never run, minus
    /// the budget.
    ///
    /// So require headroom for the ESTIMATED cost as well as the floor. This is
    /// verdict-neutral by construction — it only ever refuses a build that the
    /// budget could not have completed, and every caller of the cold build
    /// already fails closed to IBP/CROWN.
    ///
    /// Pure decision core: `remaining >= predicted x 5/4` at the given rate.
    /// The production gate in [`Self::collect_forward_linear_state_cached`]
    /// resolves the rate through [`forward_linear_rate_calibration`] (env
    /// override > per-process probe > shipped fallback constant); tests
    /// inject explicit rates here so admission logic is checked independently
    /// of what this host happens to measure.
    pub(crate) fn forward_linear_cold_build_affordable_with_rate(
        macs: Option<u128>,
        deadline: Option<Instant>,
        now: Instant,
        macs_per_sec: u128,
    ) -> bool {
        let Some(deadline) = deadline else {
            return true;
        };
        let Some(macs) = macs else {
            return true;
        };
        let predicted_secs = macs / macs_per_sec.max(1);
        let Some(padded_secs) = predicted_secs.checked_mul(FORWARD_LINEAR_ADMISSION_MARGIN_NUM)
        else {
            return false;
        };
        let padded_secs = padded_secs / FORWARD_LINEAR_ADMISSION_MARGIN_DEN;
        let padded = std::time::Duration::from_secs(u64::try_from(padded_secs).unwrap_or(u64::MAX));
        deadline.saturating_duration_since(now) >= padded
    }

    /// Cached variant of
    /// [`Self::collect_forward_linear_bounds_dag_with_engine_and_deadline`]:
    /// single-entry cache keyed by a bit-exact hash of the input bounds (see
    /// [`super::ForwardLinearMapCache`]). The root input recurs across the PGD
    /// spec-CROWN prechecks, the alpha reference collection, and the
    /// spec-propagation setup — each paid the full O(L) certified pass (~22s
    /// on cifar100 release) before this cache.
    ///
    /// Errors (deadline/unsupported/mem-cap) are NOT cached: a later call with
    /// more budget may succeed.
    pub fn collect_forward_linear_bounds_dag_cached(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::sync::Arc<HashMap<String, BoundedTensor>>> {
        #[cfg(test)]
        record_forward_linear_collection_request();
        Ok(self
            .collect_forward_linear_state_cached_impl(input, engine, deadline, false)?
            .0)
    }

    /// Typed cGAN reference-map request. The extra Tanh arm exists only for
    /// this explicit root transaction; ordinary callers retain their previous
    /// fail-closed surface. Its cache key is disjoint from the ordinary map so
    /// a typed entry can never silently expand a later ordinary request.
    pub(crate) fn collect_forward_linear_bounds_dag_cached_for_typed_cgan(
        &self,
        input: &BoundedTensor,
        config: &crate::bounds::AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<std::sync::Arc<HashMap<String, BoundedTensor>>> {
        let exec_order = self.exec_order()?;
        if !self.cgan_complete_crown_ibp_root_eligible(config, exec_order)
            && !self.cgan_sparse_target_complete_root_eligible(config, exec_order)
        {
            return Err(NyError::UnsupportedConfiguration(
                "typed cGAN Tanh forward-linear request requires the exact root policy and \
                 sequential ConvTranspose2d+Conv2d structure"
                    .to_string(),
            ));
        }
        #[cfg(test)]
        record_forward_linear_collection_request();
        Ok(self
            .collect_forward_linear_state_cached_impl(input, engine, deadline, true)?
            .0)
    }

    /// Shared cached forward-linear state: the concretized per-node bounds map
    /// plus the OUTPUT node's certified `LinearBounds` (#w4-root-margin) when
    /// retained by the pass. One O(L) certified computation per root input.
    #[allow(clippy::type_complexity)]
    pub(crate) fn collect_forward_linear_state_cached(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        self.collect_forward_linear_state_cached_impl(input, engine, deadline, false)
    }

    #[allow(clippy::type_complexity)]
    fn collect_forward_linear_state_cached_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        allow_typed_tanh: bool,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let allow_conv_transpose = Self::forward_linear_conv_transpose_reference_enabled();
        let typed_tanh_active = allow_typed_tanh
            && self
                .nodes
                .values()
                .any(|node| matches!(node.layer, Layer::Tanh(_)));
        let fingerprint = forward_linear_cache_fingerprint_with_typed_tanh(
            input,
            None,
            allow_conv_transpose,
            forward_linear_f32_seam_armed(),
            typed_tanh_active,
        );

        if let Ok(guard) = self.cached_forward_linear_map.fixed.read() {
            if let Some(entry) = guard.as_ref() {
                if entry.fingerprint.exact_match(&fingerprint) {
                    return Ok((std::sync::Arc::clone(&entry.map), entry.output_lb.clone()));
                }
            }
        }

        // #instance-budget + I1: the `deadline` in scope here is whatever
        // phase-local deadline the caller happened to have -- on cifar100 that
        // is the 40 s root-alpha cap, and the gate measurably saw 37-38 s while
        // 186-226 s was actually live. When the instance has published its
        // authoritative deadline, admit against a FRACTION of the REAL
        // remaining budget (invariant I1) instead.
        //
        // Fail-closed by construction: the effective deadline is the LATER of
        // the two only when the instance budget says so, and if nothing is
        // published this is exactly today's behaviour. A gate must never become
        // more permissive merely because information is missing.
        let deadline = match (deadline, ny_core::instance_budget::deadline()) {
            (Some(phase), Some(instance)) if instance > phase => {
                if forward_linear_instance_budget_enabled() {
                    Some(instance)
                } else {
                    Some(phase)
                }
            }
            (d, _) => d,
        };
        if !Self::forward_linear_cold_build_admitted(deadline) {
            return Err(NyError::DeadlineExceeded(format!(
                "forward-linear cold build requires at least {}s headroom",
                FORWARD_LINEAR_COLD_BUILD_MIN_HEADROOM.as_secs()
            )));
        }
        // Affordability gate (#forward-linear-cost-gate). The rate probe only
        // runs here — after the 30 s floor passed AND a finite deadline plus a
        // MAC estimate exist — so deadline-free (offline/test) callers never
        // pay it, and a scored run pays it once per process inside guaranteed
        // headroom.
        if let (Some(deadline_at), Some(macs)) =
            (deadline, self.forward_linear_cold_build_macs(input.len()))
        {
            let calibration = forward_linear_rate_calibration(engine);
            let now = Instant::now();
            let remaining = deadline_at.saturating_duration_since(now);
            let predicted_secs = macs / calibration.macs_per_sec.max(1);
            let admitted = Self::forward_linear_cold_build_affordable_with_rate(
                Some(macs),
                Some(deadline_at),
                now,
                calibration.macs_per_sec,
            );
            record_forward_linear_admission(ForwardLinearAdmissionRecord {
                admitted,
                seam_f32: calibration.seam_f32,
                macs_per_sec: u64::try_from(calibration.macs_per_sec).unwrap_or(u64::MAX),
                rate_source: calibration.source,
                predicted_secs: u64::try_from(predicted_secs).unwrap_or(u64::MAX),
                remaining_secs: remaining.as_secs(),
                build_macs: u64::try_from(macs).unwrap_or(u64::MAX),
                probe_macs: calibration.probe_macs,
                probe_secs: calibration.probe_secs,
            });
            if !admitted {
                return Err(NyError::DeadlineExceeded(format!(
                    "forward-linear cold build needs ~{predicted_secs}s (x{}/{} admission \
                     margin) for {} Gf64-MAC over this graph at {} MAC/s (seam={}, rate \
                     source={}), more than the remaining {}s budget; skipping so the budget \
                     goes to CROWN/BaB instead of being spent on a pass that could not finish",
                    FORWARD_LINEAR_ADMISSION_MARGIN_NUM,
                    FORWARD_LINEAR_ADMISSION_MARGIN_DEN,
                    macs / 1_000_000_000,
                    calibration.macs_per_sec,
                    if calibration.seam_f32 { "f32" } else { "f64" },
                    calibration.source,
                    remaining.as_secs(),
                )));
            }
            info!(
                predicted_secs = u64::try_from(predicted_secs).unwrap_or(u64::MAX),
                remaining_secs = remaining.as_secs(),
                macs_per_sec = u64::try_from(calibration.macs_per_sec).unwrap_or(u64::MAX),
                rate_source = calibration.source,
                seam_f32 = calibration.seam_f32,
                "forward-linear cold build admitted (#forward-linear-cost-gate)"
            );
        }

        let build_start = Instant::now();
        let (map, output_lb) = self.collect_forward_linear_state_fresh(
            input,
            engine,
            deadline,
            None,
            allow_conv_transpose,
            typed_tanh_active,
        )?;
        let build_cost = build_start.elapsed();
        check_forward_linear_deadline(deadline, "fixed-cache publication")?;
        if let Ok(mut guard) = self.cached_forward_linear_map.fixed.write() {
            check_forward_linear_deadline(deadline, "fixed-cache publication")?;
            *guard = Some(ForwardLinearCacheEntry {
                fingerprint,
                map: std::sync::Arc::clone(&map),
                output_lb: output_lb.clone(),
                build_cost,
            });
        }
        Ok((map, output_lb))
    }

    /// Alpha-fed variant of [`Self::collect_forward_linear_state_cached`]
    /// (#w4-root-alpha): the image-mode diagonal ReLU compositions use the
    /// given per-neuron lower slopes (the warmup's optimized alphas). Cached
    /// in a SEPARATE single-entry slot whose key includes a bit-exact
    /// fingerprint of the alpha map, so the fixed-slope entry is never
    /// clobbered and a stale alpha map can never be served.
    #[allow(clippy::type_complexity)]
    #[allow(dead_code)] // Alpha-state cache face retained for unwired root-alpha experiments.
    pub(crate) fn collect_forward_linear_state_cached_with_alphas(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let allow_conv_transpose = Self::forward_linear_conv_transpose_reference_enabled();
        self.collect_forward_linear_state_cached_with_alphas_and_policy(
            input,
            relu_alphas,
            engine,
            deadline,
            allow_conv_transpose,
        )
    }

    #[allow(clippy::type_complexity)]
    fn collect_forward_linear_state_cached_with_alphas_and_policy(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        allow_conv_transpose: bool,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let fingerprint = forward_linear_cache_fingerprint(
            input,
            Some(relu_alphas),
            allow_conv_transpose,
            forward_linear_f32_seam_armed(),
        );

        if let Ok(guard) = self.cached_forward_linear_map.alpha.read() {
            if let Some(entry) = guard.as_ref() {
                if entry.fingerprint.exact_match(&fingerprint) {
                    return Ok((std::sync::Arc::clone(&entry.map), entry.output_lb.clone()));
                }
            }
        }

        check_forward_linear_deadline(deadline, "alpha rebuild admission")?;
        let build_start = Instant::now();
        let (map, output_lb) = self.collect_forward_linear_state_fresh(
            input,
            engine,
            deadline,
            Some(relu_alphas),
            allow_conv_transpose,
            false,
        )?;
        let build_cost = build_start.elapsed();
        check_forward_linear_deadline(deadline, "alpha-cache publication")?;
        if let Ok(mut guard) = self.cached_forward_linear_map.alpha.write() {
            check_forward_linear_deadline(deadline, "alpha-cache publication")?;
            *guard = Some(ForwardLinearCacheEntry {
                fingerprint,
                map: std::sync::Arc::clone(&map),
                output_lb: output_lb.clone(),
                build_cost,
            });
        }
        Ok((map, output_lb))
    }

    /// Run one full forward-linear pass and split off the OUTPUT node's
    /// retained `LinearBounds` (the margin composition seed).
    #[allow(clippy::type_complexity)]
    fn collect_forward_linear_state_fresh(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
        allow_conv_transpose: bool,
        allow_typed_tanh: bool,
    ) -> Result<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let (node_bounds, mut linear_map) = collect_forward_linear_state_dag(
            self,
            input,
            engine,
            deadline,
            relu_alphas,
            allow_conv_transpose,
            allow_typed_tanh,
        )?;
        // The output node's affine map w.r.t. the original input — the margin
        // composition seed. Retained by the pass (the output has no consumer,
        // so image-mode liveness never frees it).
        let output_name = if self.output_node.is_empty() {
            self.topological_sort()?.last().cloned().unwrap_or_default()
        } else {
            self.output_node.clone()
        };
        let output_lb = linear_map.remove(&output_name).map(std::sync::Arc::new);
        Ok((std::sync::Arc::new(node_bounds), output_lb))
    }

    /// Certified spec-margin bounds from the forward-linear output map
    /// (#w4-root-margin): compose the spec matrix `C` (an exact affine map, no
    /// bias) with the output node's certified forward-linear `LinearBounds`
    /// using the SAME certified dense-affine composition the pass uses for
    /// Gemm layers (f64 GEMM + outward coefficient-cast gap + γ·S discharge),
    /// then sound-concretize on the input box.
    ///
    /// This keeps the CROSS-OUTPUT correlation that the per-logit projection
    /// destroys: a margin row `e_i − e_j` composes to `w_i − w_j` coefficient
    /// CANCELLATION before concretization, instead of `lower_i − upper_j`
    /// interval subtraction after. Measured on cifar100 prop_idx_7641 this is
    /// the difference between obj[0] = −23.85 (projection) and a decidable
    /// root bound.
    ///
    /// Errors mirror the forward-linear reference collection: refusal classes
    /// (unsupported op / deadline / memory cap) surface as their `NyError`s so
    /// the caller can fail closed to the CPU spec loop.
    pub(crate) fn forward_linear_spec_margin_bounds(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (_, output_lb) = self.collect_forward_linear_state_cached(input, engine, deadline)?;
        compose_spec_margin(input, spec_matrix, output_lb.as_deref(), engine, deadline)
    }

    /// Alpha-fed variant of [`Self::forward_linear_spec_margin_bounds`]
    /// (#w4-root-alpha): the forward-linear map is rebuilt with the given
    /// per-neuron lower ReLU slopes (sound for any α ∈ [0, 1] —
    /// see `image::compose_relu_diag_forward`), then composed with `C`
    /// through the same certified dense-affine composition. The result is a
    /// sound enclosure of the same spec values as the fixed-slope route, so
    /// callers may intersect the two element-wise. Production traffic goes
    /// through [`Self::forward_linear_alpha_optimized_spec_margin_bounds`];
    /// this direct variant remains for the soundness test suite.
    #[cfg(test)]
    pub(crate) fn forward_linear_spec_margin_bounds_with_alphas(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (_, output_lb) = self.collect_forward_linear_state_cached_with_alphas(
            input,
            relu_alphas,
            engine,
            deadline,
        )?;
        compose_spec_margin(input, spec_matrix, output_lb.as_deref(), engine, deadline)
    }

    /// Measured wall cost of the cached fixed-slope forward-linear pass for
    /// THIS input (`None` when the cache is cold). The alpha-fed rebuild
    /// costs the same O(L) pass, so this is the budget quantum the root
    /// warmup cap and the optimizer's self-budgeting both consult
    /// (#w4-root-alpha-opt).
    pub(crate) fn forward_linear_fixed_pass_cost(
        &self,
        input: &BoundedTensor,
    ) -> Option<std::time::Duration> {
        self.forward_linear_fixed_state_if_cached(input)
            .map(|(.., cost)| cost)
    }

    /// Cached fixed-slope forward-linear state for THIS input — `None` when
    /// the cache is cold (#w4-root-alpha-opt: the optimizer must never pay
    /// the fresh O(L) pass itself; it only runs where the fixed pass already
    /// did, i.e. on the root input).
    #[allow(clippy::type_complexity)]
    fn forward_linear_fixed_state_if_cached(
        &self,
        input: &BoundedTensor,
    ) -> Option<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
        std::time::Duration,
    )> {
        self.forward_linear_fixed_state_if_cached_with_policy(
            input,
            Self::forward_linear_conv_transpose_reference_enabled(),
        )
    }

    #[allow(clippy::type_complexity)]
    fn forward_linear_fixed_state_if_cached_with_policy(
        &self,
        input: &BoundedTensor,
        allow_conv_transpose: bool,
    ) -> Option<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
        std::time::Duration,
    )> {
        let fingerprint = forward_linear_cache_fingerprint(
            input,
            None,
            allow_conv_transpose,
            forward_linear_f32_seam_armed(),
        );
        let guard = self.cached_forward_linear_map.fixed.read().ok()?;
        guard
            .as_ref()
            .filter(|entry| entry.fingerprint.exact_match(&fingerprint))
            .map(|entry| {
                (
                    std::sync::Arc::clone(&entry.map),
                    entry.output_lb.clone(),
                    entry.build_cost,
                )
            })
    }

    /// Cached alpha-fed state for this exact `(input, alpha map, operator
    /// policy)` request. A hash match without canonical equality is a miss.
    #[allow(clippy::type_complexity)]
    fn forward_linear_alpha_state_if_cached_with_policy(
        &self,
        input: &BoundedTensor,
        relu_alphas: &std::collections::BTreeMap<String, Array1<f32>>,
        allow_conv_transpose: bool,
    ) -> Option<(
        std::sync::Arc<HashMap<String, BoundedTensor>>,
        Option<std::sync::Arc<LinearBounds>>,
    )> {
        let fingerprint = forward_linear_cache_fingerprint(
            input,
            Some(relu_alphas),
            allow_conv_transpose,
            forward_linear_f32_seam_armed(),
        );
        let guard = self.cached_forward_linear_map.alpha.read().ok()?;
        guard
            .as_ref()
            .filter(|entry| entry.fingerprint.exact_match(&fingerprint))
            .map(|entry| (std::sync::Arc::clone(&entry.map), entry.output_lb.clone()))
    }

    /// Forward-map ALPHA OPTIMIZER + certified rebuild (#w4-root-alpha-opt):
    /// optimize per-neuron lower ReLU slopes against the C-margin objective of
    /// the unverified spec rows (see [`alpha_opt`] module docs), then rebuild
    /// the forward-linear map ONCE with the optimized alphas through the
    /// certified machinery and compose the margin. Returns `Ok(None)` when the
    /// fixed cache is cold for this input, the headroom cannot fit the
    /// rebuild, or the optimizer finds no predicted improvement (in which case
    /// the ~`fixed_cost` rebuild is skipped entirely and the budget returns to
    /// BaB).
    ///
    /// Soundness: the returned bounds come from the same certified alpha-fed
    /// pass as any other alpha map (sound for any α ∈ [0, 1]); the optimizer
    /// itself never touches the verdict path.
    pub(crate) fn forward_linear_alpha_optimized_spec_margin_bounds(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        current_lower: Option<&BoundedTensor>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Option<(BoundedTensor, alpha_opt::AlphaOptStats)>> {
        use std::time::Duration;

        // Typed, graph-local and default OFF. Keep this check at the authority
        // boundary as defense in depth even though production spec propagation
        // also avoids calling the optimizer while the lane is dark.
        if !self.forward_linear_spec_alpha_enabled() {
            return Ok(None);
        }

        // Root-class requests only: the 1-row spec calls on the root input are
        // the PGD margin PRECHECKS (many per instance, each with a distinct C
        // row, so the memo below cannot amortize them) — running sweeps plus a
        // ~20s certified rebuild per precheck would eat the attack phase. The
        // multi-row root C-matrix is the single call this lever exists for.
        if spec_matrix.nrows() < 2 {
            return Ok(None);
        }

        let allow_conv_transpose = Self::forward_linear_conv_transpose_reference_enabled();
        let Some((map, output_lb, fixed_cost)) =
            self.forward_linear_fixed_state_if_cached_with_policy(input, allow_conv_transpose)
        else {
            tracing::debug!(
                "forward-linear alpha-opt: fixed cache cold for this input, skipping (#w4-root-alpha-opt)"
            );
            return Ok(None);
        };
        let Some(output_lb) = output_lb else {
            return Ok(None);
        };

        const OPT_FLOOR: Duration = Duration::from_millis(1500);
        const OPT_CAP: Duration = Duration::from_secs(12);
        let rebuild_reserve = fixed_cost.mul_f64(1.15);
        let rebuild_fits =
            |extra: Duration| alpha_rebuild_fits(deadline, Instant::now(), rebuild_reserve, extra);

        // Remove the single incumbent entry before constructing a replacement.
        // Its canonical allocation remains live locally so exact hits preserve
        // their memo value, but its retained bytes reduce the new fingerprint's
        // admission ceiling. Thus two request identities can never coexist
        // above the optimizer's shared 256 MiB envelope. Any refusal restores
        // the incumbent if no concurrent caller has filled the slot.
        let cached_memo = self
            .cached_forward_linear_map
            .alpha_opt
            .write()
            .ok()
            .and_then(|mut guard| guard.take());
        let incumbent_fingerprint_bytes = cached_memo
            .as_ref()
            .map_or(0, |(fingerprint, _)| fingerprint.retained_bytes());
        let fingerprint_budget =
            alpha_opt::MAX_SURROGATE_BYTES.saturating_sub(incumbent_fingerprint_bytes);
        let publish_if_empty = |entry| {
            if let Ok(mut guard) = self.cached_forward_linear_map.alpha_opt.write() {
                if guard.is_none() {
                    *guard = Some(entry);
                }
            }
        };

        // Memo: one optimizer run per exact request. The hash accelerates the
        // comparison, but canonical bytes authorize the hit.
        let memo_fingerprint = match margin_opt_memo_fingerprint(
            input,
            spec_matrix,
            current_lower,
            allow_conv_transpose,
            self.forward_linear_spec_alpha_enabled(),
            deadline,
            fingerprint_budget,
        ) {
            Ok(Some(fingerprint)) => fingerprint,
            Ok(None) | Err(NyError::DeadlineExceeded(_)) => {
                if let Some(entry) = cached_memo {
                    publish_if_empty(entry);
                }
                tracing::info!(
                    incumbent_fingerprint_bytes,
                    "forward-linear alpha-opt: memo fingerprint resource/deadline refusal"
                );
                return Ok(None);
            }
            Err(error) => {
                if let Some(entry) = cached_memo {
                    publish_if_empty(entry);
                }
                return Err(error);
            }
        };
        let retained_request_bytes = memo_fingerprint.retained_bytes();
        if let Some((cached_fingerprint, memo)) = cached_memo {
            if cached_fingerprint.exact_match(&memo_fingerprint) {
                // Drop the duplicate old canonical allocation before any
                // certified rebuild, then republish the equivalent value under
                // the newly constructed identity. Every early return below
                // therefore preserves the exact-hit memo.
                drop(cached_fingerprint);
                return match memo {
                    None => {
                        publish_if_empty((memo_fingerprint, None));
                        Ok(None)
                    }
                    Some((alphas, stats)) => {
                        publish_if_empty((
                            memo_fingerprint,
                            Some((std::sync::Arc::clone(&alphas), stats)),
                        ));
                        // The alpha slot is normally warm. If another alpha
                        // request displaced it, reserve the measured rebuild
                        // cost BEFORE doing any cold work; a later call with
                        // more budget may retry the same memo.
                        let alpha_lb = if let Some((_, alpha_lb)) = self
                            .forward_linear_alpha_state_if_cached_with_policy(
                                input,
                                &alphas,
                                allow_conv_transpose,
                            ) {
                            alpha_lb
                        } else {
                            if !rebuild_fits(Duration::ZERO) {
                                tracing::info!(
                                    rebuild_reserve_ms = rebuild_reserve.as_millis() as u64,
                                    "forward-linear alpha-opt memo hit: cold rebuild deferred \
                                     for insufficient headroom"
                                );
                                return Ok(None);
                            }
                            self.collect_forward_linear_state_cached_with_alphas_and_policy(
                                input,
                                &alphas,
                                engine,
                                deadline,
                                allow_conv_transpose,
                            )?
                            .1
                        };
                        let bounds = compose_spec_margin(
                            input,
                            spec_matrix,
                            alpha_lb.as_deref(),
                            engine,
                            deadline,
                        )?;
                        Ok(Some((bounds, stats)))
                    }
                };
            }
            // A miss deliberately drops the displaced entry here before the
            // new fingerprint becomes part of the optimizer resource plan.
        }

        // Self-budgeting (#w4-root-alpha): the certified rebuild costs the
        // same O(L) pass as the fixed map (measured via the cache entry).
        // Reserve it with margin; give the optimizer a bounded slice of the
        // rest. When even the rebuild does not fit, skip everything (the
        // fixed-slope candidates stand).
        let now = Instant::now();
        let opt_budget = match deadline.map(|d| d.saturating_duration_since(now)) {
            Some(remaining) => {
                if remaining < rebuild_reserve + OPT_FLOOR {
                    tracing::info!(
                        headroom_ms = remaining.as_millis() as u64,
                        rebuild_reserve_ms = rebuild_reserve.as_millis() as u64,
                        "forward-linear alpha-opt: skipping (insufficient headroom, #w4-root-alpha-opt)"
                    );
                    return Ok(None);
                }
                // Cannot underflow: the guard above ensures
                // `remaining >= rebuild_reserve + OPT_FLOOR`.
                remaining
                    .saturating_sub(rebuild_reserve)
                    .min(remaining.mul_f32(0.35))
                    .min(OPT_CAP)
                    .max(OPT_FLOOR)
            }
            None => OPT_CAP,
        };
        let opt_deadline = Some(now + opt_budget);

        match alpha_opt::optimize_margin_alphas(
            self,
            input,
            spec_matrix,
            current_lower,
            &map,
            &output_lb,
            engine,
            opt_deadline,
            retained_request_bytes,
        )? {
            None => {
                // Only budget-independent declines may become authoritative.
                // The optimizer propagates observed deadline errors, and this
                // final clock guard also protects future refusal paths from
                // accidentally making a timeout permanent.
                if alpha_decline_is_memoizable(opt_deadline, Instant::now()) {
                    publish_if_empty((memo_fingerprint, None));
                }
                Ok(None)
            }
            Some((alphas, stats)) => {
                let alphas = std::sync::Arc::new(alphas);
                let (_, alpha_lb) = self
                    .collect_forward_linear_state_cached_with_alphas_and_policy(
                        input,
                        &alphas,
                        engine,
                        deadline,
                        allow_conv_transpose,
                    )?;
                let bounds =
                    compose_spec_margin(input, spec_matrix, alpha_lb.as_deref(), engine, deadline)?;
                publish_if_empty((memo_fingerprint, Some((alphas, stats))));
                Ok(Some((bounds, stats)))
            }
        }
    }
}

/// Compose the spec matrix `C` with the OUTPUT node's certified
/// forward-linear map and sound-concretize on the input box (shared by the
/// fixed-slope and alpha-fed margin routes).
fn compose_spec_margin(
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    output_lb: Option<&LinearBounds>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear spec margin: deadline exceeded before composition".to_string(),
        ));
    }
    let Some(output_lb) = output_lb else {
        return Err(NyError::UnsupportedConfiguration(
            "forward-linear spec margin: output linear map not retained".to_string(),
        ));
    };
    if spec_matrix.ncols() != output_lb.num_outputs() {
        return Err(NyError::shape_mismatch(
            vec![output_lb.num_outputs()],
            vec![spec_matrix.ncols()],
        ));
    }
    // Worst-case input magnitude per coordinate (the certified-error
    // discharge weights), exactly as the forward pass computes them.
    let input_flat = input.flatten();
    let input_mag: Vec<f64> = input_flat
        .lower()
        .iter()
        .zip(input_flat.upper().iter())
        .map(|(&l, &u)| f64::from(l).abs().max(f64::from(u).abs()))
        .collect();
    let composed = image::compose_dense_affine_forward(
        "spec-margin",
        spec_matrix,
        None,
        output_lb,
        &input_mag,
        engine,
        deadline,
        None,
    )?;
    let bounds = composed
        .concretize_checked(input)?
        .reshape(&[spec_matrix.nrows()])?;
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear spec margin: deadline exceeded before return".to_string(),
        ));
    }
    Ok(bounds)
}

fn check_forward_linear_deadline(deadline: Option<Instant>, context: &str) -> Result<()> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        Err(NyError::DeadlineExceeded(format!(
            "forward-linear: deadline exceeded during {context}"
        )))
    } else {
        Ok(())
    }
}

/// A `None` returned at/after the optimizer's private deadline may be a
/// timeout-derived refusal and therefore must not suppress a later,
/// larger-budget retry.
#[inline]
fn alpha_decline_is_memoizable(opt_deadline: Option<Instant>, now: Instant) -> bool {
    opt_deadline.is_none_or(|value| now < value)
}

#[inline]
fn alpha_rebuild_fits(
    deadline: Option<Instant>,
    now: Instant,
    rebuild_reserve: std::time::Duration,
    extra: std::time::Duration,
) -> bool {
    deadline.is_none_or(|value| {
        now < value && value.saturating_duration_since(now) >= rebuild_reserve + extra
    })
}

#[inline]
fn append_len(canonical: &mut Vec<u8>, len: usize) {
    canonical.extend_from_slice(&(len as u64).to_le_bytes());
}

#[inline]
fn append_f32_bits<'a>(canonical: &mut Vec<u8>, values: impl Iterator<Item = &'a f32>) {
    for value in values {
        canonical.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn fingerprint_from_canonical(canonical: Vec<u8>) -> ForwardLinearCacheFingerprint {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(&canonical);
    ForwardLinearCacheFingerprint {
        hash: hasher.finish(),
        canonical: canonical.into(),
    }
}

/// Collision-proof cache identity for a forward-linear map request.
///
/// The exact canonical encoding contains:
///
/// - input rank/dimensions;
/// - lower and upper endpoint lengths and every f32 payload bit;
/// - the ConvTranspose operator-surface policy;
/// - the f32 value-GEMM seam state (v2: a seam-on-built map must never serve
///   a seam-off request in-process — the numeric values differ);
/// - the typed-Tanh operator-surface policy (v3);
/// - alpha-map presence and sorted (`BTreeMap`) length-delimited node names,
///   vector lengths, and every alpha payload bit.
///
/// `hash` accelerates the common miss, but every cache hit also compares the
/// canonical bytes. This prevents both deterministic same-numel shape aliases
/// and adversarial/accidental u64 collisions from becoming proof authority.
fn forward_linear_cache_fingerprint(
    input: &BoundedTensor,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
    allow_conv_transpose: bool,
    f32_seam: bool,
) -> ForwardLinearCacheFingerprint {
    forward_linear_cache_fingerprint_with_typed_tanh(
        input,
        relu_alphas,
        allow_conv_transpose,
        f32_seam,
        false,
    )
}

fn forward_linear_cache_fingerprint_with_typed_tanh(
    input: &BoundedTensor,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
    allow_conv_transpose: bool,
    f32_seam: bool,
    allow_typed_tanh: bool,
) -> ForwardLinearCacheFingerprint {
    let alpha_capacity = relu_alphas
        .map(|alphas| {
            alphas.iter().fold(0usize, |total, (name, alpha)| {
                total
                    .saturating_add(16)
                    .saturating_add(name.len())
                    .saturating_add(alpha.len().saturating_mul(4))
            })
        })
        .unwrap_or(0);
    let mut canonical = Vec::with_capacity(
        48usize
            .saturating_add(input.len().saturating_mul(8))
            .saturating_add(alpha_capacity),
    );
    // v3: typed-Tanh eligibility joined the exact operator policy.
    canonical.extend_from_slice(b"NYFLMAP\x03");
    append_len(&mut canonical, input.shape().len());
    for &dim in input.shape() {
        append_len(&mut canonical, dim);
    }
    append_len(&mut canonical, input.lower().len());
    append_f32_bits(&mut canonical, input.lower().iter());
    append_len(&mut canonical, input.upper().len());
    append_f32_bits(&mut canonical, input.upper().iter());
    canonical.push(u8::from(allow_conv_transpose));
    canonical.push(u8::from(f32_seam));
    canonical.push(u8::from(allow_typed_tanh));
    match relu_alphas {
        Some(alphas) => {
            canonical.push(1);
            append_len(&mut canonical, alphas.len());
            // BTreeMap iteration is lexicographically sorted. Explicit lengths
            // make (`a`,`bc`) unambiguous from (`ab`,`c`).
            for (name, alpha) in alphas {
                append_len(&mut canonical, name.len());
                canonical.extend_from_slice(name.as_bytes());
                append_len(&mut canonical, alpha.len());
                append_f32_bits(&mut canonical, alpha.iter());
            }
        }
        None => canonical.push(0),
    }
    fingerprint_from_canonical(canonical)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MarginOptFingerprintLayout {
    canonical_bytes: usize,
    scan_work: u64,
}

fn checked_add_bytes(total: &mut usize, bytes: usize) -> Option<()> {
    *total = total.checked_add(bytes)?;
    Some(())
}

fn checked_add_f32_payload(total: &mut usize, len: usize) -> Option<()> {
    checked_add_bytes(total, len.checked_mul(size_of::<f32>())?)
}

/// Exact byte/work plan for the optimizer memo identity. This runs solely on
/// scalar lengths: no request payload is scanned and no heap memory is
/// allocated until the complete plan has fit the shared surrogate envelope.
fn margin_opt_fingerprint_layout(
    input_rank: usize,
    input_lower_len: usize,
    input_upper_len: usize,
    spec_rows: usize,
    spec_cols: usize,
    spec_len: usize,
    incumbent: Option<(usize, usize, usize)>,
    max_bytes: usize,
) -> Option<MarginOptFingerprintLayout> {
    if spec_rows.checked_mul(spec_cols)? != spec_len {
        return None;
    }

    // Embedded NYFLMAP v1 canonical input identity.
    let mut input_bytes = 8usize; // magic/version
    checked_add_bytes(&mut input_bytes, 8)?; // rank
    checked_add_bytes(&mut input_bytes, input_rank.checked_mul(8)?)?;
    checked_add_bytes(&mut input_bytes, 8)?; // lower length
    checked_add_f32_payload(&mut input_bytes, input_lower_len)?;
    checked_add_bytes(&mut input_bytes, 8)?; // upper length
    checked_add_f32_payload(&mut input_bytes, input_upper_len)?;
    checked_add_bytes(&mut input_bytes, 2)?; // operator policy + absent alpha map

    let mut bytes = 8usize; // NYFLOPT v1 magic/version
    checked_add_bytes(&mut bytes, 8)?; // embedded-input length
    checked_add_bytes(&mut bytes, input_bytes)?;
    checked_add_bytes(&mut bytes, 1)?; // typed candidate policy
    checked_add_bytes(&mut bytes, 16)?; // spec rows + columns
    checked_add_f32_payload(&mut bytes, spec_len)?;
    match incumbent {
        Some((rank, lower_len, upper_len)) => {
            checked_add_bytes(&mut bytes, 1 + 8)?; // present + rank
            checked_add_bytes(&mut bytes, rank.checked_mul(8)?)?;
            checked_add_bytes(&mut bytes, 8)?; // lower length
            checked_add_f32_payload(&mut bytes, lower_len)?;
            checked_add_bytes(&mut bytes, 8)?; // upper length
            checked_add_f32_payload(&mut bytes, upper_len)?;
        }
        None => checked_add_bytes(&mut bytes, 1)?,
    }
    if bytes > max_bytes {
        return None;
    }

    // Every canonical byte is written once and hashed once. Keep this work
    // under the optimizer's existing pass ceiling as well as its byte ceiling.
    let scan_work = u64::try_from(bytes).ok()?.checked_mul(2)?;
    if scan_work > alpha_opt::MAX_SURROGATE_PASS_MACS {
        return None;
    }
    Some(MarginOptFingerprintLayout {
        canonical_bytes: bytes,
        scan_work,
    })
}

/// Capacity-checked writer. Once `try_reserve_exact` succeeds, every append is
/// proved to fit the accepted byte plan before touching the Vec, so no append
/// can silently grow the allocation beyond the surrogate envelope.
struct MarginOptCanonicalWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl MarginOptCanonicalWriter {
    fn try_new(limit: usize, max_capacity: usize) -> Option<Self> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(limit).ok()?;
        if bytes.capacity() > max_capacity {
            return None;
        }
        Some(Self { bytes, limit })
    }

    fn extend(&mut self, value: &[u8]) -> Option<()> {
        if value.len() > self.limit.checked_sub(self.bytes.len())? {
            return None;
        }
        self.bytes.extend_from_slice(value);
        Some(())
    }

    fn push(&mut self, value: u8) -> Option<()> {
        if self.bytes.len() == self.limit {
            return None;
        }
        self.bytes.push(value);
        Some(())
    }

    fn len(&mut self, value: usize) -> Option<()> {
        self.extend(&(value as u64).to_le_bytes())
    }

    fn finish(self) -> Option<Vec<u8>> {
        (self.bytes.len() == self.limit).then_some(self.bytes)
    }
}

fn append_margin_opt_dims(
    writer: &mut MarginOptCanonicalWriter,
    dims: &[usize],
    checkpoint: &mut impl FnMut(&str) -> Result<()>,
) -> Result<Option<()>> {
    if writer.len(dims.len()).is_none() {
        return Ok(None);
    }
    for (index, &dim) in dims.iter().enumerate() {
        if index.is_multiple_of(alpha_opt::DEADLINE_POLL_WORK as usize) {
            checkpoint("optimizer memo shape fingerprint")?;
        }
        if writer.len(dim).is_none() {
            return Ok(None);
        }
    }
    Ok(Some(()))
}

fn append_margin_opt_f32_bits<'a>(
    writer: &mut MarginOptCanonicalWriter,
    values: impl Iterator<Item = &'a f32>,
    checkpoint: &mut impl FnMut(&str) -> Result<()>,
) -> Result<Option<()>> {
    for (index, value) in values.enumerate() {
        if index.is_multiple_of(alpha_opt::DEADLINE_POLL_WORK as usize) {
            checkpoint("optimizer memo payload fingerprint")?;
        }
        if writer.extend(&value.to_bits().to_le_bytes()).is_none() {
            return Ok(None);
        }
    }
    Ok(Some(()))
}

/// Fallible core used by production and deterministic deadline/cap tests.
fn margin_opt_memo_fingerprint_with(
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    current_lower: Option<&BoundedTensor>,
    allow_conv_transpose: bool,
    forward_linear_spec_alpha: bool,
    max_bytes: usize,
    mut checkpoint: impl FnMut(&str) -> Result<()>,
) -> Result<Option<MarginOptMemoFingerprint>> {
    let layout = match margin_opt_fingerprint_layout(
        input.shape().len(),
        input.lower().len(),
        input.upper().len(),
        spec_matrix.nrows(),
        spec_matrix.ncols(),
        spec_matrix.len(),
        current_lower.map(|bounds| {
            (
                bounds.shape().len(),
                bounds.lower().len(),
                bounds.upper().len(),
            )
        }),
        max_bytes,
    ) {
        Some(layout) => layout,
        None => return Ok(None),
    };
    checkpoint("optimizer memo allocation")?;
    let Some(mut writer) = MarginOptCanonicalWriter::try_new(layout.canonical_bytes, max_bytes)
    else {
        return Ok(None);
    };
    checkpoint("optimizer memo allocation")?;

    if writer.extend(b"NYFLOPT\x01").is_none() || writer.len(0).is_none() {
        return Ok(None);
    }
    let embedded_len_offset = 8;
    let embedded_start = writer.bytes.len();
    if writer.extend(b"NYFLMAP\x01").is_none()
        || append_margin_opt_dims(&mut writer, input.shape(), &mut checkpoint)?.is_none()
        || writer.len(input.lower().len()).is_none()
        || append_margin_opt_f32_bits(&mut writer, input.lower().iter(), &mut checkpoint)?.is_none()
        || writer.len(input.upper().len()).is_none()
        || append_margin_opt_f32_bits(&mut writer, input.upper().iter(), &mut checkpoint)?.is_none()
        || writer.push(u8::from(allow_conv_transpose)).is_none()
        || writer.push(0).is_none()
    {
        return Ok(None);
    }
    let embedded_len = writer
        .bytes
        .len()
        .checked_sub(embedded_start)
        .ok_or_else(|| {
            NyError::InternalError("optimizer memo embedded length underflow".to_string())
        })?;
    writer.bytes[embedded_len_offset..embedded_len_offset + 8]
        .copy_from_slice(&(embedded_len as u64).to_le_bytes());

    if writer.push(u8::from(forward_linear_spec_alpha)).is_none()
        || writer.len(spec_matrix.nrows()).is_none()
        || writer.len(spec_matrix.ncols()).is_none()
        || append_margin_opt_f32_bits(&mut writer, spec_matrix.iter(), &mut checkpoint)?.is_none()
    {
        return Ok(None);
    }
    match current_lower {
        // Guard form, matching the `None` arm below: the appends run left to
        // right and the first refusal declines the memo (the partially filled
        // writer is dropped with it).
        Some(bounds)
            if writer.push(1).is_none()
                || append_margin_opt_dims(&mut writer, bounds.shape(), &mut checkpoint)?
                    .is_none()
                || writer.len(bounds.lower().len()).is_none()
                || append_margin_opt_f32_bits(
                    &mut writer,
                    bounds.lower().iter(),
                    &mut checkpoint,
                )?
                .is_none()
                || writer.len(bounds.upper().len()).is_none()
                || append_margin_opt_f32_bits(
                    &mut writer,
                    bounds.upper().iter(),
                    &mut checkpoint,
                )?
                .is_none() =>
        {
            return Ok(None);
        }
        Some(_) => {}
        None if writer.push(0).is_none() => return Ok(None),
        None => {}
    }
    checkpoint("optimizer memo canonical completion")?;
    let Some(canonical) = writer.finish() else {
        return Ok(None);
    };

    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for chunk in canonical.chunks(alpha_opt::DEADLINE_POLL_WORK as usize) {
        checkpoint("optimizer memo canonical hash")?;
        hasher.write(chunk);
    }
    checkpoint("optimizer memo canonical hash")?;
    debug_assert_eq!(
        layout.scan_work,
        u64::try_from(canonical.len()).unwrap_or(u64::MAX) * 2
    );
    Ok(Some(MarginOptMemoFingerprint {
        hash: hasher.finish(),
        canonical,
    }))
}

/// Exact optimizer-memo request identity. Any byte/work overflow, cap refusal,
/// failed reservation, or observed deadline declines this optional heuristic
/// before optimizer work and leaves the memo cold for a later retry.
fn margin_opt_memo_fingerprint(
    input: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    current_lower: Option<&BoundedTensor>,
    allow_conv_transpose: bool,
    forward_linear_spec_alpha: bool,
    deadline: Option<Instant>,
    max_bytes: usize,
) -> Result<Option<MarginOptMemoFingerprint>> {
    margin_opt_memo_fingerprint_with(
        input,
        spec_matrix,
        current_lower,
        allow_conv_transpose,
        forward_linear_spec_alpha,
        max_bytes,
        |context| check_forward_linear_deadline(deadline, context),
    )
}

/// Log-only per-node width trace for the forward-linear pass (`NY_FL_TRACE=1`,
/// default OFF). Emits no verdict-affecting state; used to localize which op in
/// an image DAG loses tightness relative to plain IBP.
fn forward_linear_width_trace_enabled() -> bool {
    matches!(std::env::var("NY_FL_TRACE").ok().as_deref(), Some("1"))
}

fn collect_forward_linear_state_dag(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
    allow_conv_transpose: bool,
    allow_typed_tanh: bool,
) -> Result<(
    HashMap<String, BoundedTensor>,
    HashMap<String, LinearBounds>,
)> {
    let exec_order = graph.topological_sort()?;

    // Image mode (#vnncomp-image-forward-linear): conv DAGs route through the
    // certified compositions in `image.rs` (Conv2d / ConvTranspose2d /
    // BatchNorm / diagonal ReLU and typed-cGAN Tanh / Add / Linear / shape
    // pass-through). The
    // generic identity-trick path below is
    // O(N²) memory per activation node — infeasible at image scale — and
    // never supported Conv2d, so conv graphs previously always failed closed.
    // Graphs WITHOUT a 2-D convolution keep the legacy path byte-identical.
    let has_conv2d = exec_order.iter().any(|name| {
        graph
            .nodes
            .get(name)
            .is_some_and(|node| matches!(node.layer, Layer::Conv2d(_)))
    });
    let has_conv_transpose = exec_order.iter().any(|name| {
        graph
            .nodes
            .get(name)
            .is_some_and(|node| matches!(node.layer, Layer::ConvTranspose2d(_)))
    });
    let image_mode = has_conv2d || (allow_conv_transpose && has_conv_transpose);
    if image_mode {
        // Fail closed BEFORE any expensive work if the graph leaves the
        // certified image op surface: the caller falls back to plain IBP.
        for name in &exec_order {
            if let Some(node) = graph.nodes.get(name) {
                if !image_mode_supported(&node.layer, allow_conv_transpose, allow_typed_tanh) {
                    return Err(unsupported_forward_linear_node(
                        name,
                        &node.layer,
                        "operator is outside the certified image forward-linear surface",
                    ));
                }
            }
        }
    }

    // #w4-root-alpha-opt profile: which parts of this pass are alpha-
    // independent (cacheable across alpha-fed rebuilds) vs alpha-dependent.
    // The IBP prepass never depends on alpha; every coefficient composition
    // downstream of the first crossing ReLU does (the ReLU diagonal feeds the
    // conv im2col+GEMM inputs), so it must be re-done per alpha map.
    let pass_start = Instant::now();
    let ibp_node_bounds =
        graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)?;
    let ibp_elapsed = pass_start.elapsed();
    let mut profile = image_mode.then(ForwardLinearProfile::default);
    let input_dim = input.len();

    if image_mode {
        // Dense-coefficient memory guard: each node carries two f32 matrices
        // of `node_numel × input_dim`. Refuse (fail closed to IBP) when the
        // largest exceeds ~128M entries (512 MB per matrix) — cifar100-scale
        // (16384×3072 ≈ 50M) passes; tinyimagenet-scale (12288-dim inputs)
        // stays on its existing IBP gate until column-block streaming lands.
        const MAX_COEFF_ENTRIES: usize = 1 << 27;
        let max_numel = ibp_node_bounds.values().map(|b| b.len()).max().unwrap_or(0);
        if max_numel.saturating_mul(input_dim) > MAX_COEFF_ENTRIES {
            return Err(NyError::UnsupportedConfiguration(format!(
                "forward-linear image bounds: coefficient state {max_numel}x{input_dim} exceeds \
                 the dense memory cap ({MAX_COEFF_ENTRIES} entries)"
            )));
        }
    }

    // max(|x_l|, |x_u|) per input coordinate: the worst-case input magnitude
    // used to discharge certified coefficient errors into the bias (image mode).
    let input_flat = input.flatten();
    let input_mag: Vec<f64> = input_flat
        .lower()
        .iter()
        .zip(input_flat.upper().iter())
        .map(|(&l, &u)| (l as f64).abs().max((u as f64).abs()))
        .collect();

    // Liveness: last consumer index per node, so image mode can drop dense
    // coefficient matrices as soon as no downstream node needs them.
    let mut last_use: HashMap<&str, usize> = HashMap::new();
    for (t, name) in exec_order.iter().enumerate() {
        if let Some(node) = graph.nodes.get(name) {
            for input_name in &node.inputs {
                last_use.insert(input_name.as_str(), t);
            }
        }
    }

    let mut node_bounds = HashMap::with_capacity(exec_order.len());
    let mut linear_bounds = HashMap::with_capacity(exec_order.len());

    for (exec_idx, node_name) in exec_order.iter().enumerate() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "Graph forward-linear: deadline exceeded before node '{node_name}'"
            )));
        }
        let node = graph.nodes.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: unknown node '{node_name}' in execution order"
            ))
        })?;
        let output_shape = ibp_node_bounds
            .get(node_name)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "forward-linear bounds: missing IBP output shape for node '{node_name}'"
                ))
            })?
            .shape()
            .to_vec();
        let output_dim = ibp_node_bounds[node_name].len();

        let compose_started = Instant::now();
        let node_linear = if image_mode {
            compose_image_node(
                node_name,
                &node.layer,
                &node.inputs,
                output_dim,
                &linear_bounds,
                &node_bounds,
                &ibp_node_bounds,
                input,
                input_dim,
                &input_mag,
                engine,
                deadline,
                relu_alphas,
            )?
        } else {
            match &node.layer {
                Layer::Concat(layer) => concat::compose_concat_forward(
                    node_name,
                    layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                )?,
                // Match auto_LiRPA's forward-mode BoundMul middle relaxation for
                // graph warmup so forward+crown uses the same fixed bilinear
                // interpolation on non-optimized MulBinary nodes.
                // Source: auto_LiRPA/operators/bivariate.py::MulHelper.get_forward_relaxation.
                Layer::MulBinary(_) => binary::compose_binary_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                    |identity, input_a_bounds, input_b_bounds| {
                        crate::layers::MulBinaryLayer.propagate_linear_binary(
                            identity,
                            input_a_bounds,
                            input_b_bounds,
                            crate::MulBinaryRelaxationMode::Middle,
                        )
                    },
                )?,
                Layer::Sub(layer) => binary::compose_binary_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                    |identity, _, _| layer.propagate_linear_binary(identity),
                )?,
                Layer::Div(_) => binary::compose_div_forward(
                    node_name,
                    &node.layer,
                    &node.inputs,
                    output_dim,
                    &linear_bounds,
                    &ibp_node_bounds,
                    input,
                    input_dim,
                )?,
                _ => {
                    if node.inputs.len() != 1 {
                        return Err(unsupported_forward_linear_node(
                            node_name,
                            &node.layer,
                            "only unary nodes and Concat are supported in this packet",
                        ));
                    }

                    let pred_name = node.inputs.first().ok_or_else(|| {
                        NyError::InternalError(
                            "validated unary node must have exactly one input".into(),
                        )
                    })?;
                    let upstream = resolve_upstream_linear_bounds(
                        pred_name,
                        None,
                        &linear_bounds,
                        input_dim,
                        node_name,
                    )?;
                    let layer_name = layer_debug_name(&node.layer);
                    let pre_activation = resolve_pre_activation_bounds(
                        pred_name,
                        &ibp_node_bounds,
                        input,
                        node_name,
                        &layer_name,
                    )?;
                    let local = local_forward_relaxation(
                        node_name,
                        &node.layer,
                        output_dim,
                        Some(pre_activation),
                    )?;
                    compose_forward_relaxation(&local, &upstream)?
                }
            }
        };

        if let Some(profile) = profile.as_mut() {
            profile.record(&node.layer, compose_started.elapsed());
        }

        let concretize_started = Instant::now();
        let concretized = concretize_to_node_shape(&node_linear, input, &output_shape, node_name)?;
        // Intersect element-wise with IBP: forward-linear preserves correlations but may be
        // looser per element after nonlinear relaxations (e.g., ReLU triangle).
        let ibp_bounds = &ibp_node_bounds[node_name];
        // #fl-trace (NY_FL_TRACE=1, log-only): per-node forward-vs-IBP width so a
        // looseness regression can be localized to the exact op that loses it.
        let trace_widths = forward_linear_width_trace_enabled();
        let forward_max_width = trace_widths.then(|| concretized.max_width());
        let tightened = if concretized.shape() == ibp_bounds.shape() {
            tighten_with_ibp(&concretized, ibp_bounds)
        } else {
            concretized
        };
        if let Some(forward_max_width) = forward_max_width {
            info!(
                node = %node_name,
                op = %layer_debug_name(&node.layer),
                numel = tightened.len(),
                forward_max_width,
                ibp_max_width = ibp_bounds.max_width(),
                kept_max_width = tightened.max_width(),
                "#fl-trace forward-linear per-node width"
            );
        }
        if let Some(profile) = profile.as_mut() {
            profile.concretize += concretize_started.elapsed();
        }
        node_bounds.insert(node_name.clone(), tightened);
        linear_bounds.insert(node_name.clone(), node_linear);

        // Image mode: free dense coefficient matrices whose consumers have all
        // executed (the returned linear map is unused by both public wrappers;
        // node_bounds keeps the concretized per-node boxes).
        if image_mode {
            for input_name in &node.inputs {
                if last_use.get(input_name.as_str()) == Some(&exec_idx) {
                    linear_bounds.remove(input_name);
                }
            }
        }
    }

    if let Some(profile) = profile {
        info!(
            total_ms = pass_start.elapsed().as_millis() as u64,
            ibp_prepass_ms = ibp_elapsed.as_millis() as u64,
            conv_ms = profile.conv.as_millis() as u64,
            relu_ms = profile.relu.as_millis() as u64,
            add_ms = profile.add.as_millis() as u64,
            linear_ms = profile.linear.as_millis() as u64,
            shape_ms = profile.shape.as_millis() as u64,
            concretize_ms = profile.concretize.as_millis() as u64,
            alpha_fed = relu_alphas.is_some(),
            "forward-linear image pass profile (#w4-root-alpha-opt): only the IBP prepass is alpha-independent"
        );
    }

    check_forward_linear_deadline(deadline, "completed-map return")?;
    Ok((node_bounds, linear_bounds))
}

/// Per-op-class wall-time accumulator for the image pass (#w4-root-alpha-opt
/// profile): answers which fraction of an alpha-fed rebuild re-does
/// alpha-independent work.
#[derive(Default)]
struct ForwardLinearProfile {
    conv: std::time::Duration,
    relu: std::time::Duration,
    add: std::time::Duration,
    linear: std::time::Duration,
    shape: std::time::Duration,
    concretize: std::time::Duration,
}

impl ForwardLinearProfile {
    fn record(&mut self, layer: &Layer, elapsed: std::time::Duration) {
        match layer {
            Layer::Conv2d(_) | Layer::ConvTranspose2d(_) => self.conv += elapsed,
            Layer::ReLU(_) => self.relu += elapsed,
            Layer::Add(_) => self.add += elapsed,
            Layer::Linear(_) => self.linear += elapsed,
            _ => self.shape += elapsed,
        }
    }
}

/// Certified image op surface (#vnncomp-image-forward-linear): the conv-DAG
/// allowlist. Anything else fails closed (caller falls back to plain IBP).
fn image_mode_supported(layer: &Layer, allow_conv_transpose: bool, allow_typed_tanh: bool) -> bool {
    matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::ReLU(_)
            | Layer::Add(_)
            | Layer::Linear(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
    ) || (allow_conv_transpose && matches!(layer, Layer::ConvTranspose2d(_) | Layer::BatchNorm(_)))
        || (allow_typed_tanh && matches!(layer, Layer::Tanh(_)))
}

/// Resolve an upstream forward-linear map without cloning stored matrices
/// (image-scale coefficient state is 100s of MB per node).
fn resolve_upstream_linear_ref<'a>(
    input_name: &str,
    forward_bounds: &'a HashMap<String, LinearBounds>,
    input_dim: usize,
    node_name: &str,
) -> Result<Cow<'a, LinearBounds>> {
    if input_name == NETWORK_INPUT {
        return Ok(Cow::Owned(LinearBounds::identity(input_dim)));
    }
    forward_bounds
        .get(input_name)
        .map(Cow::Borrowed)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: node '{node_name}' references unknown upstream input '{input_name}'"
            ))
        })
}

/// Resolve the tightened running bounds (forward∩IBP) for a predecessor —
/// the pre-activation source for image-mode relaxations. Falls back to the
/// IBP prepass map only when the running map has no entry (never happens in
/// topological order, kept as a sound fallback).
fn resolve_running_bounds<'a>(
    pred_name: &str,
    running_bounds: &'a HashMap<String, BoundedTensor>,
    ibp_node_bounds: &'a HashMap<String, BoundedTensor>,
    input: &'a BoundedTensor,
    node_name: &str,
) -> Result<&'a BoundedTensor> {
    if pred_name == NETWORK_INPUT {
        return Ok(input);
    }
    running_bounds
        .get(pred_name)
        .or_else(|| ibp_node_bounds.get(pred_name))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: node '{node_name}' is missing predecessor bounds for '{pred_name}'"
            ))
        })
}

/// Route one node through the certified image compositions (#vnncomp-image-
/// forward-linear). Every concretization downstream goes through
/// `concretize_sound`; every rounding inside these compositions is certified
/// and discharged outward (see `image.rs` module docs).
#[allow(clippy::too_many_arguments)]
fn compose_image_node(
    node_name: &str,
    layer: &Layer,
    inputs: &[String],
    output_dim: usize,
    linear_bounds: &HashMap<String, LinearBounds>,
    node_bounds: &HashMap<String, BoundedTensor>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    input_dim: usize,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    relu_alphas: Option<&std::collections::BTreeMap<String, Array1<f32>>>,
) -> Result<LinearBounds> {
    let single_input = |layer: &Layer| -> Result<&str> {
        if inputs.len() == 1 {
            Ok(inputs[0].as_str())
        } else {
            Err(unsupported_forward_linear_node(
                node_name,
                layer,
                "expected exactly one input",
            ))
        }
    };

    match layer {
        Layer::Conv2d(conv) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pred_shape =
                resolve_input_shape(pred, None, None, ibp_node_bounds, input, node_name)?;
            image::compose_conv2d_forward(
                node_name,
                conv,
                &upstream,
                &pred_shape,
                output_dim,
                input_mag,
                engine,
                deadline,
                None,
            )
        }
        Layer::ConvTranspose2d(conv) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pred_shape =
                resolve_input_shape(pred, None, None, ibp_node_bounds, input, node_name)?;
            image::compose_conv_transpose2d_forward(
                node_name,
                conv,
                &upstream,
                &pred_shape,
                output_dim,
                input_mag,
                engine,
                deadline,
            )
        }
        Layer::BatchNorm(batch_norm) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pre_activation =
                resolve_running_bounds(pred, node_bounds, ibp_node_bounds, input, node_name)?;
            image::compose_batch_norm_forward(
                node_name,
                batch_norm,
                &upstream,
                pre_activation,
                output_dim,
                input_mag,
            )
        }
        Layer::ReLU(_) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            // Pre-activation from the RUNNING tightened map (forward∩IBP), not
            // the exploding raw-IBP prepass — this is what keeps relaxation
            // slopes sane on deep conv stacks (design step 1d).
            let pre_activation =
                resolve_running_bounds(pred, node_bounds, ibp_node_bounds, input, node_name)?;
            // #w4-root-alpha: optimized per-neuron lower slopes when supplied.
            // Length mismatches fail OPEN to the adaptive rule (sound — the
            // adaptive relaxation is always valid); contiguity is guaranteed
            // for freshly-built Array1 but checked defensively.
            let alpha_lower = relu_alphas
                .and_then(|m| m.get(node_name))
                .filter(|a| a.len() == output_dim)
                .and_then(|a| a.as_slice());
            image::compose_relu_diag_forward(
                node_name,
                &upstream,
                pre_activation,
                input_mag,
                alpha_lower,
            )
        }
        Layer::Tanh(_) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            let pre_activation =
                resolve_running_bounds(pred, node_bounds, ibp_node_bounds, input, node_name)?;
            image::compose_tanh_diag_forward(node_name, &upstream, pre_activation, input_mag)
        }
        Layer::Add(_) => {
            if inputs.len() != 2 {
                return Err(unsupported_forward_linear_node(
                    node_name,
                    layer,
                    "binary Add must have exactly 2 inputs",
                ));
            }
            let a = resolve_upstream_linear_ref(&inputs[0], linear_bounds, input_dim, node_name)?;
            let b = resolve_upstream_linear_ref(&inputs[1], linear_bounds, input_dim, node_name)?;
            image::compose_add_forward(node_name, &a, &b, input_mag)
        }
        Layer::Linear(linear) => {
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            image::compose_dense_affine_forward(
                node_name,
                linear.weight(),
                linear.bias(),
                &upstream,
                input_mag,
                engine,
                deadline,
                None,
            )
        }
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
            // Pure C-order shape ops: the flattened coefficient layout is
            // unchanged, so the linear map passes through exactly.
            let pred = single_input(layer)?;
            let upstream = resolve_upstream_linear_ref(pred, linear_bounds, input_dim, node_name)?;
            if upstream.num_outputs() != output_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![output_dim],
                    got: vec![upstream.num_outputs()],
                });
            }
            Ok(upstream.into_owned())
        }
        _ => Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "operator is outside the certified image forward-linear surface",
        )),
    }
}

fn local_forward_relaxation(
    node_name: &str,
    layer: &Layer,
    output_dim: usize,
    pre_activation: Option<&BoundedTensor>,
) -> Result<LinearBounds> {
    let identity = LinearBounds::identity(output_dim);
    let pre_activation = pre_activation.ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear bounds: node '{node_name}' ({}) is missing pre-activation bounds",
            layer_debug_name(layer),
        ))
    })?;

    let result = match layer {
        Layer::Linear(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Conv1d(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::AddConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::MulConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::DivConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::SubConstant(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Reshape(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Flatten(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Transpose(layer) => {
            let mut layer = layer.clone();
            layer.set_input_shape(pre_activation.shape().to_vec());
            layer
                .propagate_linear(&identity)
                .map(|bounds| bounds.into_owned())
        }
        Layer::Squeeze(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Unsqueeze(layer) => layer
            .propagate_linear(&identity)
            .map(|bounds| bounds.into_owned()),
        Layer::Slice(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::Gather(layer) => {
            let mut layer = layer.clone();
            layer.set_input_shape(pre_activation.shape().to_vec());
            layer
                .propagate_linear(&identity)
                .map(|bounds| bounds.into_owned())
        }
        Layer::ReLU(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::Sigmoid(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::PowConstant(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        Layer::ReduceSum(layer) => layer.propagate_linear_with_bounds(&identity, pre_activation),
        _ => Err(unsupported_forward_linear_node(
            node_name,
            layer,
            "operator is outside the forward-linear packet surface",
        )),
    };

    result.map_err(|error| wrap_forward_linear_error(node_name, layer, error))
}

fn compose_forward_relaxation(
    local: &LinearBounds,
    upstream: &LinearBounds,
) -> Result<LinearBounds> {
    if local.num_inputs() != upstream.num_outputs() {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![local.num_inputs()],
        });
    }

    let local_lower_pos = local.lower_a().mapv(|value| value.max(0.0));
    let local_lower_neg = local.lower_a().mapv(|value| value.min(0.0));
    let local_upper_pos = local.upper_a().mapv(|value| value.max(0.0));
    let local_upper_neg = local.upper_a().mapv(|value| value.min(0.0));

    let lower_a = local_lower_pos.dot(upstream.lower_a()) + local_lower_neg.dot(upstream.upper_a());
    let upper_a = local_upper_pos.dot(upstream.upper_a()) + local_upper_neg.dot(upstream.lower_a());

    let lower_b = local_lower_pos.dot(upstream.lower_b())
        + local_lower_neg.dot(upstream.upper_b())
        + local.lower_b();
    let upper_b = local_upper_pos.dot(upstream.upper_b())
        + local_upper_neg.dot(upstream.lower_b())
        + local.upper_b();

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

fn sum_linear_bounds(parts: &[LinearBounds]) -> Result<LinearBounds> {
    let first = parts.first().ok_or_else(|| {
        NyError::InvalidSpec("forward-linear bounds: empty linear-bounds sum".to_string())
    })?;
    let num_outputs = first.num_outputs();
    let num_inputs = first.num_inputs();

    let mut lower_a = Array2::zeros((num_outputs, num_inputs));
    let mut lower_b = Array1::zeros(num_outputs);
    let mut upper_a = Array2::zeros((num_outputs, num_inputs));
    let mut upper_b = Array1::zeros(num_outputs);

    for part in parts {
        if part.num_outputs() != num_outputs || part.num_inputs() != num_inputs {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, num_inputs],
                got: vec![part.num_outputs(), part.num_inputs()],
            });
        }
        lower_a += part.lower_a();
        lower_b += part.lower_b();
        upper_a += part.upper_a();
        upper_b += part.upper_b();
    }

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

fn resolve_upstream_linear_bounds(
    input_name: &str,
    constant_input: Option<&BoundedTensor>,
    forward_bounds: &HashMap<String, LinearBounds>,
    input_dim: usize,
    node_name: &str,
) -> Result<LinearBounds> {
    if input_name == NETWORK_INPUT {
        return Ok(LinearBounds::identity(input_dim));
    }
    if let Some(bounds) = forward_bounds.get(input_name) {
        return Ok(bounds.clone());
    }
    if let Some(constant_input) = constant_input {
        return constant_linear_bounds(constant_input, input_dim);
    }

    Err(NyError::InvalidSpec(format!(
        "forward-linear bounds: node '{node_name}' references unknown upstream input '{input_name}'"
    )))
}

fn resolve_input_shape(
    input_name: &str,
    constant_input: Option<&BoundedTensor>,
    stored_shape: Option<&[usize]>,
    ibp_node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    node_name: &str,
) -> Result<Vec<usize>> {
    if input_name == NETWORK_INPUT {
        return Ok(input.shape().to_vec());
    }
    if let Some(bounds) = ibp_node_bounds.get(input_name) {
        return Ok(bounds.shape().to_vec());
    }
    if let Some(constant_input) = constant_input {
        return Ok(constant_input.shape().to_vec());
    }
    if let Some(stored_shape) = stored_shape {
        return Ok(stored_shape.to_vec());
    }

    Err(NyError::InvalidSpec(format!(
        "forward-linear bounds: node '{node_name}' is missing shape metadata for input '{input_name}'"
    )))
}

fn resolve_pre_activation_bounds<'a>(
    pred_name: &str,
    ibp_node_bounds: &'a HashMap<String, BoundedTensor>,
    input: &'a BoundedTensor,
    node_name: &str,
    layer_name: &str,
) -> Result<&'a BoundedTensor> {
    if pred_name == NETWORK_INPUT {
        return Ok(input);
    }
    ibp_node_bounds.get(pred_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear bounds: node '{node_name}' ({layer_name}) is missing predecessor bounds for '{pred_name}'"
        ))
    })
}

fn constant_linear_bounds(bounds: &BoundedTensor, input_dim: usize) -> Result<LinearBounds> {
    let flat = bounds.flatten();
    LinearBounds::new_or_conservative(
        Array2::zeros((flat.len(), input_dim)),
        Array1::from_iter(flat.lower().iter().copied()),
        Array2::zeros((flat.len(), input_dim)),
        Array1::from_iter(flat.upper().iter().copied()),
    )
}

fn concretize_to_node_shape(
    bounds: &LinearBounds,
    input: &BoundedTensor,
    output_shape: &[usize],
    node_name: &str,
) -> Result<BoundedTensor> {
    // SOUNDNESS (#concretize-soundness-hardening): use the directed-rounding
    // `concretize_sound` (lower rounds toward -∞, upper toward +∞) rather than the
    // plain round-to-nearest `concretize`. These concretized forward-linear node
    // bounds are *intermediate* bounds: they are intersected with IBP via
    // `tighten_with_ibp` (max(lower)/min(upper)) and used to constrain downstream
    // relaxations (e.g. pre-activation bounds feeding ReLU/activation planes), which
    // in turn feed the certified verdict. A round-to-nearest f64→f32 cast can land up
    // to 0.5 ULP *inside* the true range, producing an optimistically narrow
    // intermediate bound; `tighten_with_ibp`'s comment ("both sets are sound")
    // depends on this being a sound over-approximation. `concretize_sound` guarantees
    // it. The forward-linear path is not the hot per-domain tightening loop, so the
    // 1-ULP directed cast has no measurable cost here.
    let flat = bounds.concretize_sound(input);
    let lower = flat
        .lower()
        .clone()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|error| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: reshape lower failed for node '{node_name}': {error}"
            ))
        })?;
    let upper = flat
        .upper()
        .clone()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|error| {
            NyError::InvalidSpec(format!(
                "forward-linear bounds: reshape upper failed for node '{node_name}': {error}"
            ))
        })?;

    if lower.iter().all(|value| value.is_finite()) && upper.iter().all(|value| value.is_finite()) {
        BoundedTensor::new(lower, upper)
    } else {
        BoundedTensor::new_allow_infinite(lower, upper)
    }
}

/// Tighten forward-linear bounds by intersecting element-wise with IBP bounds.
/// Both sets are sound, so `max(lower)` / `min(upper)` per element is also sound.
fn tighten_with_ibp(forward: &BoundedTensor, ibp: &BoundedTensor) -> BoundedTensor {
    let mut lower = forward.lower().clone();
    let mut upper = forward.upper().clone();
    for (fl, il) in lower.iter_mut().zip(ibp.lower().iter()) {
        *fl = fl.max(*il);
    }
    for (fu, iu) in upper.iter_mut().zip(ibp.upper().iter()) {
        *fu = fu.min(*iu);
    }
    // Clamp: if intersection is empty on any element, use IBP (always sound).
    for ((l, u), (il, iu)) in lower
        .iter_mut()
        .zip(upper.iter_mut())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
    {
        if *l > *u {
            *l = *il;
            *u = *iu;
        }
    }
    if lower.iter().all(|v| v.is_finite()) && upper.iter().all(|v| v.is_finite()) {
        BoundedTensor::new(lower, upper).unwrap_or_else(|_| ibp.clone())
    } else {
        BoundedTensor::new_allow_infinite(lower, upper).unwrap_or_else(|_| ibp.clone())
    }
}

fn wrap_forward_linear_error(node_name: &str, layer: &Layer, error: NyError) -> NyError {
    match error {
        NyError::UnsupportedOp(_) | NyError::UnsupportedConfiguration(_) => {
            unsupported_forward_linear_node(node_name, layer, &error.to_string())
        }
        other => other,
    }
}

fn unsupported_forward_linear_node(node_name: &str, layer: &Layer, reason: &str) -> NyError {
    NyError::UnsupportedConfiguration(format!(
        "forward-linear bounds do not support node '{node_name}' ({}){separator}{reason}",
        layer_debug_name(layer),
        separator = if reason.is_empty() { "" } else { ": " },
    ))
}

fn layer_debug_name(layer: &Layer) -> String {
    let debug = format!("{layer:?}");
    debug.split('(').next().unwrap_or("Unknown").to_string()
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_image;
