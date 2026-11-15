// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown`: a certified patches-mode CROWN root pass whose coefficient
//! products run on the GPU (typed key `bab.gpu_patches_crown`, DARK).
//!
//! # What it is for
//!
//! `vggnet16_2022` spec15/16/17 perturb EVERY one of the 150528 input pixels
//! (widths 2e-7, 2e-6, 2e-5). The field proves them with plain CROWN
//! (CROWN intermediates for every ReLU layer, patches-mode coefficients) on a
//! GPU in ~1100 s of 1200 (abc 1107/1057/1109 s; its worst margin +5.33); ny's
//! CPU patches CROWN cannot finish, IBP intermediates are vacuous (~1e13
//! gain), and the double-double zonotope is capped at `k <= 128` perturbed
//! inputs (`scratchpad/vbs/diag6/cgan_vgg.md` §6, Vibecheck idea 9). Each row
//! is 5.56 normalized points.
//!
//! # The pass
//!
//! Over an admitted sequential chain (`plan`: Conv2d / Linear / ReLU /
//! non-overlapping MaxPool / reshapes — VGG16's whole op surface), forward in
//! value order:
//!
//! 1. every affine value gets the certified one-step interval SCREEN from its
//!    input's box (`boxes::screen_affine`);
//! 2. if it feeds a ReLU, its neurons that the screen leaves crossing (all of
//!    them with `screen: false`) get two CROWN rows each (`+e_j`, `-e_j`),
//!    walked backward to the input (or `depth` affine ops) in chunks, and the
//!    results are INTERSECTED into the screen box — shrink-only; a crossed
//!    result refuses the whole pass (it would mean an unsound enclosure);
//! 3. ReLU / reshape boxes are views, max-pool boxes are exact;
//! 4. finally every spec row is walked from the output to the input at full
//!    depth, always with an f64 signed product.
//!
//! The verdict (CLI side) is `f64_to_f32_down(lower_i) > threshold_i` for every
//! objective, exactly the `#dd-zonotope` root rule; anything else falls
//! through to the standard pipeline with nothing changed.
//!
//! # Soundness
//!
//! Every published number is a certified bound on the STORED operands:
//!
//! * the backward keeps `c . v_t >= lambda . v + beta - kappa` with exact
//!   algebra (`backward` module docs); ReLU/max-pool relaxations are valid for
//!   the stored boxes with directed rounding of their slopes/intercepts;
//! * each coefficient product carries a certified per-entry error
//!   `err >= |mu - W^T lambda|` — the a-posteriori `gamma_k * |A||B|` of the
//!   GEMM that produced it (f64 or f32 cuBLAS on the GPU, f64 on the CPU),
//!   outward-rounded to binary64 (`product` module docs) — and that error is
//!   CHARGED AGAINST THE BOX OF THE VALUE IT MULTIPLIES, never transported;
//! * the final `beta` / concretization sums are charged Higham `gamma_n`
//!   bounds and the result is rounded toward `-inf`.
//!
//! The pass requires a qualified binary64 environment (round-to-nearest,
//! gradual underflow) on the caller and every Rayon worker, like
//! `#dd-zonotope`. Intersection with the screen box can only shrink a valid
//! enclosure, so the pass can turn `unknown` into `unsat` and never the
//! reverse.
//!
//! # GPU and memory admission
//!
//! Products run on the process-global sound f64 engine (cuBLAS on a CUDA
//! build; admitted per product, see `product`), tiled over OUTPUT columns so
//! each engine call's operands + result stay under `gpu_tile_mb` (the
//! device-memory cap). Rows are chunked so a chunk's host estimate — the
//! largest `lambda + mu + err` window along its walk times the rows, plus
//! three tiles (host gather + device copies: on this unified-memory host a
//! device allocation IS host memory) — fits `min(max_mb, headroom_frac *
//! live process headroom)` (`crown_memory::process_memory_headroom_bytes`,
//! which includes the `MemAvailable` governor that competition runs rely on).
//! A single row that cannot fit refuses the pass. Any GPU refusal runs the
//! deterministic CPU product for that chunk — bit-identical to the dark CPU
//! route — and is counted.
//!
//! # Observable (fixed schema, every armed run)
//!
//! `[gpu-patches-crown] armed=1 layers= gpu_passes= cpu_fallback= secs=
//! peak_mb= ...` (see [`GpuPatchesCrownOutcome::observable_line`]), plus one
//! `[gpu-patches-crown] layer ...` line per tightened layer. An armed run
//! with `gpu_passes=0` on the global route never engaged the GPU: VOID, not a
//! negative. The dark route prints nothing and runs nothing.

mod backward;
mod boxes;
mod plan;
mod product;
#[cfg(test)]
mod tests;

use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use ny_core::{f32_to_f64_exact, f64_to_f32_down, require_f64_interval_proof_environment};
use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};

use crate::GraphNetwork;

use backward::{row_bytes, run_rows, seed_neurons, seed_objectives};
use boxes::{box_is_valid, pool_box, screen_affine, Boxes, ValueBox};
use plan::{input_shape, ChainPlan, OpKind};
use product::{Ctx, Refusal, Route, Stats, Stop};

const MIB: f64 = 1_048_576.0;

/// Precision of the SIGNED coefficient GEMM on the GPU route (the `|A||B|`
/// certificate GEMM is always f32 over upward-rounded operands; the CPU route
/// and the final spec rows are always f64).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuPatchesPrecision {
    /// cuBLAS `Dgemm`: certified charge `~gamma64(k)`, negligible.
    #[default]
    F64,
    /// cuBLAS `Sgemm`: `~30x` the GB10 f64 rate, certified charge
    /// `~gamma32(k+1)` (about `k * 6e-8` of `|A||B|` per layer).
    F32,
}

impl GpuPatchesPrecision {
    /// Stop-line tag.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
        }
    }
}

/// The typed `bab.gpu_patches_crown` arming. Presence arms it; every field
/// has a default; unknown sub-keys are refused.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuPatchesCrownConfig {
    /// Rows (two per tightened neuron) per chunk, before memory admission.
    #[serde(default = "default_chunk_rows")]
    pub chunk_rows: usize,
    /// Host-memory cap on one chunk (MiB).
    #[serde(default = "default_max_mb")]
    pub max_mb: usize,
    /// Share of the LIVE process headroom a chunk may claim, in `(0, 1]`.
    #[serde(default = "default_headroom_frac")]
    pub headroom_frac: f64,
    /// Device-memory cap on one engine call's operands + result (MiB).
    #[serde(default = "default_gpu_tile_mb")]
    pub gpu_tile_mb: usize,
    /// Signed-GEMM precision on the GPU route.
    #[serde(default)]
    pub precision: GpuPatchesPrecision,
    /// Affine ops an intermediate row walks before concretizing against the
    /// box it reached; `0` = all the way to the input (full CROWN).
    #[serde(default)]
    pub depth: usize,
    /// Tighten only the neurons the one-step screen leaves crossing.
    #[serde(default = "default_screen")]
    pub screen: bool,
    /// Share of the remaining BaB window the pass may spend, in `(0, 0.95]`.
    #[serde(default = "default_budget_frac")]
    pub budget_frac: f64,
    /// Absolute cap on the pass (seconds).
    #[serde(default = "default_max_secs")]
    pub max_secs: f64,
    /// Admission: run only when at least this many input coordinates are
    /// perturbed (counted on the exact-decimal box when one is registered),
    /// so the sparse-input rows keep their existing lanes.
    #[serde(default = "default_min_perturbed")]
    pub min_perturbed: usize,
}

fn default_chunk_rows() -> usize {
    256
}

fn default_max_mb() -> usize {
    16 * 1024
}

fn default_headroom_frac() -> f64 {
    0.5
}

fn default_gpu_tile_mb() -> usize {
    256
}

fn default_screen() -> bool {
    true
}

fn default_budget_frac() -> f64 {
    0.9
}

fn default_max_secs() -> f64 {
    1100.0
}

fn default_min_perturbed() -> usize {
    1024
}

impl Default for GpuPatchesCrownConfig {
    fn default() -> Self {
        Self {
            chunk_rows: default_chunk_rows(),
            max_mb: default_max_mb(),
            headroom_frac: default_headroom_frac(),
            gpu_tile_mb: default_gpu_tile_mb(),
            precision: GpuPatchesPrecision::default(),
            depth: 0,
            screen: default_screen(),
            budget_frac: default_budget_frac(),
            max_secs: default_max_secs(),
            min_perturbed: default_min_perturbed(),
        }
    }
}

impl GpuPatchesCrownConfig {
    /// Largest admissible `chunk_rows`.
    pub const MAX_CHUNK_ROWS: usize = 1 << 20;
    /// Largest admissible `max_mb` (1 TiB).
    pub const MAX_MB: usize = 1 << 20;
    /// Largest admissible `gpu_tile_mb` (64 GiB).
    pub const MAX_TILE_MB: usize = 1 << 16;
    /// Largest admissible `depth`.
    pub const MAX_DEPTH: usize = 64;
    /// Largest admissible `budget_frac`: the standard pipeline keeps a share.
    pub const MAX_BUDGET_FRAC: f64 = 0.95;
    /// Largest admissible `max_secs`.
    pub const MAX_SECS: f64 = 86_400.0;

    /// Refuse an out-of-range arming instead of clamping it. Every message
    /// starts with `gpu_patches_crown.` so the preset loader can prefix
    /// `bab.`.
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_rows == 0 || self.chunk_rows > Self::MAX_CHUNK_ROWS {
            return Err(format!(
                "gpu_patches_crown.chunk_rows must be in 1..={}, got {}",
                Self::MAX_CHUNK_ROWS,
                self.chunk_rows
            ));
        }
        if self.max_mb == 0 || self.max_mb > Self::MAX_MB {
            return Err(format!(
                "gpu_patches_crown.max_mb must be in 1..={}, got {}",
                Self::MAX_MB,
                self.max_mb
            ));
        }
        if !(self.headroom_frac.is_finite()
            && self.headroom_frac > 0.0
            && self.headroom_frac <= 1.0)
        {
            return Err(format!(
                "gpu_patches_crown.headroom_frac must be finite in (0, 1], got {}",
                self.headroom_frac
            ));
        }
        if self.gpu_tile_mb == 0 || self.gpu_tile_mb > Self::MAX_TILE_MB {
            return Err(format!(
                "gpu_patches_crown.gpu_tile_mb must be in 1..={}, got {}",
                Self::MAX_TILE_MB,
                self.gpu_tile_mb
            ));
        }
        if self.depth > Self::MAX_DEPTH {
            return Err(format!(
                "gpu_patches_crown.depth must be in 0..={} (0 = full), got {}",
                Self::MAX_DEPTH,
                self.depth
            ));
        }
        if !(self.budget_frac.is_finite()
            && self.budget_frac > 0.0
            && self.budget_frac <= Self::MAX_BUDGET_FRAC)
        {
            return Err(format!(
                "gpu_patches_crown.budget_frac must be finite in (0, {}], got {}",
                Self::MAX_BUDGET_FRAC,
                self.budget_frac
            ));
        }
        if !(self.max_secs.is_finite() && self.max_secs > 0.0 && self.max_secs <= Self::MAX_SECS) {
            return Err(format!(
                "gpu_patches_crown.max_secs must be finite in (0, {}], got {}",
                Self::MAX_SECS,
                self.max_secs
            ));
        }
        Ok(())
    }
}

/// Result and telemetry of one armed pass.
#[derive(Debug, Clone)]
pub struct GpuPatchesCrownOutcome {
    /// Certified lower bound of each objective row (`-inf` when refused).
    pub lower: Vec<f64>,
    /// Why the pass refused (fail-closed), if it did.
    pub refused: Option<String>,
    /// Layers whose boxes were CROWN-tightened.
    pub layers: usize,
    /// Coefficient products served whole by the GPU.
    pub gpu_passes: u64,
    /// Engine GEMM calls.
    pub gpu_calls: u64,
    /// Wall seconds inside GPU-served products.
    pub gpu_secs: f64,
    /// Products the CPU computed because the route is CPU-only.
    pub cpu_products: u64,
    /// Products the CPU computed after a GPU refusal (bit-identical fallback).
    pub cpu_fallback: u64,
    /// Per-reason fallback counts.
    pub fallback: Vec<(&'static str, u64)>,
    /// Pass wall seconds.
    pub secs: f64,
    /// Peak admitted chunk estimate (MiB).
    pub peak_mb: f64,
    /// Materialised box storage (MiB).
    pub boxes_mb: f64,
    /// CROWN rows walked for intermediate bounds.
    pub rows: u64,
    /// Crossing neurons under the screen (summed over tightened layers).
    pub unstable_before: u64,
    /// Crossing neurons after the CROWN intersection.
    pub unstable_after: u64,
    /// `cpu` | `global` | `engine`.
    pub route: &'static str,
    /// Engine provenance (`none` when no engine was admitted).
    pub backend: &'static str,
    /// Configured signed-GEMM precision.
    pub precision: GpuPatchesPrecision,
    /// Configured truncation depth.
    pub depth: usize,
    /// Configured screen policy.
    pub screen: bool,
}

impl GpuPatchesCrownOutcome {
    /// Every objective certified above its threshold (the root verdict rule).
    #[must_use]
    pub fn certifies(&self, thresholds: &[f32]) -> bool {
        self.refused.is_none()
            && !self.lower.is_empty()
            && self.lower.len() == thresholds.len()
            && self
                .lower
                .iter()
                .zip(thresholds)
                .all(|(&lo, &t)| lo.is_finite() && f64_to_f32_down(lo) > t)
    }

    /// A refusal decided BEFORE the pass could run (spec shape, config,
    /// objectives): `-inf` lowers, nothing engaged, and the SAME fixed-schema
    /// stop line as a run, so an armed row is never silent.
    #[must_use]
    pub fn refused_before_run(objectives: usize, why: &str, cfg: &GpuPatchesCrownConfig) -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY; objectives],
            refused: Some(why.to_string()),
            layers: 0,
            gpu_passes: 0,
            gpu_calls: 0,
            gpu_secs: 0.0,
            cpu_products: 0,
            cpu_fallback: 0,
            fallback: Refusal::ALL.iter().map(|r| (r.tag(), 0)).collect(),
            secs: 0.0,
            peak_mb: 0.0,
            boxes_mb: 0.0,
            rows: 0,
            unstable_before: 0,
            unstable_after: 0,
            route: Route::Global.tag(),
            backend: "none",
            precision: cfg.precision,
            depth: cfg.depth,
            screen: cfg.screen,
        }
    }

    /// The fixed-schema stop line.
    #[must_use]
    pub fn observable_line(&self, thresholds: &[f32]) -> String {
        let mut fallback = String::new();
        for (tag, n) in &self.fallback {
            fallback.push_str(&format!(" fb_{tag}={n}"));
        }
        let verdict = if self.refused.is_some() {
            "refused"
        } else if self.certifies(thresholds) {
            "verified"
        } else {
            "open"
        };
        let lower: Vec<String> = self.lower.iter().map(|v| format!("{v:.6e}")).collect();
        let threshold: Vec<String> = thresholds.iter().map(|t| format!("{t}")).collect();
        let refused = self
            .refused
            .as_deref()
            .map_or_else(|| "none".to_string(), |why| why.replace(' ', "_"));
        format!(
            "[gpu-patches-crown] armed=1 layers={} gpu_passes={} cpu_fallback={} secs={:.3} \
             peak_mb={:.1} route={} backend={} precision={} depth={} screen={} rows={} \
             unstable_before={} unstable_after={} gpu_calls={} gpu_secs={:.3} cpu_products={} \
             boxes_mb={:.1}{fallback} refused={refused} verdict={verdict} lower={} threshold={}",
            self.layers,
            self.gpu_passes,
            self.cpu_fallback,
            self.secs,
            self.peak_mb,
            self.route,
            self.backend,
            self.precision.tag(),
            self.depth,
            u8::from(self.screen),
            self.rows,
            self.unstable_before,
            self.unstable_after,
            self.gpu_calls,
            self.gpu_secs,
            self.cpu_products,
            self.boxes_mb,
            lower.join(","),
            threshold.join(","),
        )
    }
}

/// Run the armed pass on the process-global GPU route.
///
/// Never returns a degraded bound: every refusal publishes `-inf` lowers and
/// a `refused` reason.
#[must_use]
pub fn run_root(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    cfg: &GpuPatchesCrownConfig,
    deadline: Instant,
) -> GpuPatchesCrownOutcome {
    run_root_with_route(graph, input, objectives, cfg, deadline, &Route::Global)
}

/// Per-pass structural tallies (the product counters live in `Stats`).
#[derive(Default)]
pub(crate) struct Tally {
    pub(crate) layers: usize,
    pub(crate) rows: u64,
    pub(crate) unstable_before: u64,
    pub(crate) unstable_after: u64,
    pub(crate) peak_bytes: usize,
    pub(crate) boxes_bytes: usize,
    /// Engine provenance, set only once the cheap admission checks have
    /// passed (a refused run never materialises the engine).
    pub(crate) backend: Option<&'static str>,
}

/// What a successful pass produced (the boxes are kept for the tests).
pub(crate) struct Inner {
    pub(crate) lower: Vec<f64>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) boxes: Boxes,
}

pub(crate) fn run_root_with_route(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    cfg: &GpuPatchesCrownConfig,
    deadline: Instant,
    route: &Route,
) -> GpuPatchesCrownOutcome {
    let t0 = Instant::now();
    let stats = Stats::default();
    let mut tally = Tally::default();
    let (lower, refused) = match run_inner(
        graph, input, objectives, cfg, deadline, route, &stats, &mut tally,
    ) {
        Ok(inner) => (inner.lower, None),
        Err(why) => (vec![f64::NEG_INFINITY; objectives.len()], Some(why)),
    };
    GpuPatchesCrownOutcome {
        lower,
        refused,
        layers: tally.layers,
        gpu_passes: stats.gpu_passes.load(Relaxed),
        gpu_calls: stats.gpu_calls.load(Relaxed),
        gpu_secs: stats.gpu_ns.load(Relaxed) as f64 / 1e9,
        cpu_products: stats.cpu_products.load(Relaxed),
        cpu_fallback: stats.cpu_fallback.load(Relaxed),
        fallback: Refusal::ALL
            .iter()
            .map(|r| (r.tag(), stats.refused[r.index()].load(Relaxed)))
            .collect(),
        secs: t0.elapsed().as_secs_f64(),
        peak_mb: tally.peak_bytes as f64 / MIB,
        boxes_mb: tally.boxes_bytes as f64 / MIB,
        rows: tally.rows,
        unstable_before: tally.unstable_before,
        unstable_after: tally.unstable_after,
        route: route.tag(),
        backend: tally.backend.unwrap_or("none"),
        precision: cfg.precision,
        depth: cfg.depth,
        screen: cfg.screen,
    }
}

/// Materialise (bounded wait) the global engine so per-product admission can
/// use the non-blocking accessor; report its provenance.
fn admit_backend(route: &Route, deadline: Instant) -> &'static str {
    match route {
        Route::Cpu => "none",
        Route::Engine(engine) => engine.backend_provenance(),
        Route::Global => {
            let by = deadline.min(Instant::now() + Duration::from_secs(3));
            match crate::sound_f64_gemm::with_engine_deadline(by, |e| e.backend_provenance()) {
                Ok(Some(name)) => name,
                _ => "none",
            }
        }
    }
}

/// The caller and every Rayon worker run binary64 round-to-nearest with
/// gradual underflow (the `#dd-zonotope` qualification).
fn f64_environment_qualified_for_rayon() -> bool {
    require_f64_interval_proof_environment().is_ok()
        && rayon::broadcast(|_| require_f64_interval_proof_environment())
            .into_iter()
            .all(|result| result.is_ok())
}

/// The input box (the exact-decimal box when one is registered for THIS
/// instance — tighter than the outward f32 box it is contained in) and the
/// number of perturbed coordinates.
fn input_box(input: &BoundedTensor) -> Option<(Vec<f64>, Vec<f64>, usize)> {
    if let Some(exact) = crate::dd_zonotope::certified_box::lookup(input) {
        if exact.lower.len() == input.len()
            && exact.half_width.len() == input.len()
            && box_is_valid(&exact.lower, &exact.upper)
        {
            let perturbed = exact.half_width.iter().filter(|&&h| h > 0.0).count();
            return Some((exact.lower, exact.upper, perturbed));
        }
    }
    let lo: Vec<f64> = input.lower().iter().map(|&v| f32_to_f64_exact(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f32_to_f64_exact(v)).collect();
    if !box_is_valid(&lo, &hi) {
        return None;
    }
    let perturbed = lo.iter().zip(&hi).filter(|&(l, h)| h > l).count();
    Some((lo, hi, perturbed))
}

fn stop_reason(stop: Stop) -> String {
    match stop {
        Stop::Expired => "deadline".to_string(),
        Stop::Refused(why) => why.to_string(),
    }
}

/// Rows per chunk under the memory admission, or the refusal.
pub(crate) fn chunk_rows_for_budget(
    budget: usize,
    fixed: usize,
    per_row: usize,
    chunk_rows: usize,
    want: usize,
) -> usize {
    (budget.saturating_sub(fixed) / per_row.max(1))
        .min(chunk_rows)
        .min(want)
}

fn admit_rows(
    cfg: &GpuPatchesCrownConfig,
    route: &Route,
    per_row: usize,
    want: usize,
    tally: &mut Tally,
) -> Result<usize, String> {
    let cap = cfg.max_mb.saturating_mul(1 << 20);
    let budget = match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => cap.min((headroom as f64 * cfg.headroom_frac) as usize),
        None => cap,
    };
    let fixed = if route.is_gpu() {
        cfg.gpu_tile_mb.saturating_mul(1 << 20).saturating_mul(3)
    } else {
        0
    };
    let rows = chunk_rows_for_budget(budget, fixed, per_row, cfg.chunk_rows, want);
    if rows == 0 {
        return Err(format!(
            "memory: one row needs {per_row} B plus {fixed} B of tiles, budget {budget} B"
        ));
    }
    tally.peak_bytes = tally
        .peak_bytes
        .max(fixed.saturating_add(rows.saturating_mul(per_row)));
    Ok(rows)
}

/// CROWN-tighten the crossing neurons of affine value `vt` (shrink-only).
#[allow(clippy::too_many_arguments)]
fn tighten(
    plan: &ChainPlan<'_>,
    boxes: &Boxes,
    ctx: &Ctx<'_>,
    cfg: &GpuPatchesCrownConfig,
    vt: usize,
    lo: &mut [f64],
    hi: &mut [f64],
    tally: &mut Tally,
) -> Result<(), String> {
    let t0 = Instant::now();
    let op = &plan.ops[vt - 1];
    if !op.is_affine() {
        return Err("tighten on a non-affine value".into());
    }
    let n = lo.len();
    let crossing = |lo: &[f64], hi: &[f64]| {
        (0..lo.len())
            .filter(|&j| lo[j] < 0.0 && hi[j] > 0.0)
            .count()
    };
    let unstable_before = crossing(&*lo, &*hi);
    let neurons: Vec<usize> = (0..n)
        .filter(|&j| !cfg.screen || (lo[j] < 0.0 && hi[j] > 0.0))
        .collect();
    let targets: Vec<(usize, bool)> = neurons
        .iter()
        .flat_map(|&j| [(j, false), (j, true)])
        .collect();
    let (qh, qw) = match &op.kind {
        OpKind::Conv(spec) => (spec.kh.min(op.input.h), spec.kw.min(op.input.w)),
        _ => (op.input.h, op.input.w),
    };
    let per_row = row_bytes(plan, vt - 1, qh, qw, cfg.depth, 1);
    let mut idx = 0usize;
    let mut chunks = 0usize;
    while idx < targets.len() {
        if Instant::now() >= ctx.deadline {
            return Err("deadline".into());
        }
        let rows = admit_rows(cfg, ctx.route, per_row, targets.len() - idx, tally)?;
        let chunk = &targets[idx..idx + rows];
        let seeded = seed_neurons(plan, vt, chunk).map_err(stop_reason)?;
        let bounds = run_rows(plan, boxes, ctx, seeded, cfg.depth, 1).map_err(stop_reason)?;
        for (&(j, negative), &b) in chunk.iter().zip(&bounds) {
            if negative {
                hi[j] = hi[j].min(-b);
            } else {
                lo[j] = lo[j].max(b);
            }
        }
        idx += rows;
        chunks += 1;
    }
    if !box_is_valid(&*lo, &*hi) {
        return Err(format!(
            "crossed or non-finite enclosure at value {vt} ('{}'): refusing",
            op.name
        ));
    }
    let unstable_after = crossing(&*lo, &*hi);
    tally.layers += 1;
    tally.rows += targets.len() as u64;
    tally.unstable_before += unstable_before as u64;
    tally.unstable_after += unstable_after as u64;
    eprintln!(
        "[gpu-patches-crown] layer node={} value={vt} neurons={n} screen_unstable={unstable_before} \
         rows={} unstable_after={unstable_after} row_kb={} chunks={chunks} secs={:.3}",
        op.name,
        targets.len(),
        per_row / 1024,
        t0.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// The pass proper. `Err` is a fail-closed refusal reason.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_inner(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    cfg: &GpuPatchesCrownConfig,
    deadline: Instant,
    route: &Route,
    stats: &Stats,
    tally: &mut Tally,
) -> Result<Inner, String> {
    cfg.validate().map_err(|why| format!("config: {why}"))?;
    if !f64_environment_qualified_for_rayon() {
        return Err("binary64 environment is not round-to-nearest with gradual underflow".into());
    }
    let (lo0, hi0, perturbed) = input_box(input).ok_or("input box")?;
    if perturbed < cfg.min_perturbed {
        return Err(format!(
            "admission: {perturbed} perturbed inputs < min_perturbed {}",
            cfg.min_perturbed
        ));
    }
    let shape = input_shape(graph, input).ok_or("input shape")?;
    if shape.numel() != lo0.len() {
        return Err("input shape arity".into());
    }
    let plan = ChainPlan::build(graph, shape).map_err(|why| format!("plan: {why}"))?;
    let last = plan.output_value();
    let out_len = plan.value_shape(last).numel();
    if objectives.is_empty() || objectives.iter().any(|o| o.len() != out_len) {
        return Err("objective arity".into());
    }
    // Materialise the engine only now: every cheap refusal above (config,
    // environment, admission, plan, arity) must not pay the bounded engine
    // wait, nor risk marking the process-global initialisation abandoned.
    tally.backend = Some(admit_backend(route, deadline));
    let ctx = Ctx {
        route,
        stats,
        precision: cfg.precision,
        tile_bytes: cfg.gpu_tile_mb.saturating_mul(1 << 20),
        deadline,
    };

    let mut boxes = Boxes::default();
    boxes.push(ValueBox::Stored { lo: lo0, hi: hi0 });
    for (i, op) in plan.ops.iter().enumerate() {
        if Instant::now() >= deadline {
            return Err("deadline".into());
        }
        let value_box = match &op.kind {
            OpKind::Relu => ValueBox::ReluOf(i),
            OpKind::Reshape => ValueBox::Same(i),
            OpKind::MaxPool(spec) => {
                let (lo, hi) = pool_box(spec, op.input, op.output, boxes.view(i));
                if !box_is_valid(&lo, &hi) {
                    return Err("max-pool box".into());
                }
                ValueBox::Stored { lo, hi }
            }
            OpKind::Conv(_) | OpKind::Linear(_) => {
                let (mut lo, mut hi) = screen_affine(op, boxes.view(i), deadline)
                    .ok_or("screen (deadline or non-finite)")?;
                let v = i + 1;
                if v != last && plan.feeds_relu(v) {
                    tighten(&plan, &boxes, &ctx, cfg, v, &mut lo, &mut hi, tally)?;
                }
                ValueBox::Stored { lo, hi }
            }
        };
        boxes.push(value_box);
    }
    if boxes.len() != last + 1 {
        return Err("box bookkeeping".into());
    }
    tally.boxes_bytes = boxes.stored_bytes();

    // The spec rows: full depth, always an f64 signed product.
    let obj_ctx = Ctx {
        precision: GpuPatchesPrecision::F64,
        ..ctx
    };
    let shape_out = plan.value_shape(last);
    let per_row = row_bytes(&plan, last, shape_out.h, shape_out.w, 0, 0);
    let mut lower = Vec::with_capacity(objectives.len());
    let mut idx = 0usize;
    while idx < objectives.len() {
        let rows = admit_rows(cfg, route, per_row, objectives.len() - idx, tally)?;
        let seeded = seed_objectives(&plan, &objectives[idx..idx + rows]).map_err(stop_reason)?;
        lower.extend(run_rows(&plan, &boxes, &obj_ctx, seeded, 0, 0).map_err(stop_reason)?);
        idx += rows;
    }
    Ok(Inner { lower, boxes })
}
