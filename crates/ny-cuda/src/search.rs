// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `CudaSearchBackend` — ENGINE_PROGRAM M4's DEVICE level search for the
//! frozen-dual tier (`margin_row.frozen_dual_bab_search: cuda | auto`) and
//! M6's DEVICE wave search for the margin-row tree (`margin_row.tree_mode:
//! wave_cuda`), UNTRUSTED like every `ny_core::search::SearchBackend`.
//!
//! The wave search (`upload_wave` / `backward_batch` / `branch_scores`) is
//! documented in `search/wave_plan.rs` (lowering, ABI, lease) and
//! `search/wave_run.rs` (launch sequence): ny-bbe's CPU f32 wave search moved
//! onto `kernels/wave_search.cu` BIT FOR BIT, its image loaded into this
//! backend's own context on the first wave upload and known-answer tested
//! there (a failure refuses the wave search only, latched), its buffers
//! admitted against the lease before the device is touched, every launch
//! checked against the buffers actually allocated. The sections below apply to
//! both searches unless they name one.
//!
//! # What it is
//!
//! The CPU f32 level search of ny-bbe (`fd.rs`), moved onto the device BIT
//! FOR BIT: `kernels/fd_search.cu` performs the same IEEE binary32 operations
//! on the same operands in the same order (explicit `__fmul_rn` /
//! `__fadd_rn`, no contraction, no flush-to-zero; the column sums in the
//! shared `FD_SUM_TILE` tiles; the line search's `ln` / `exp` the same
//! platform-independent polynomials), so for every node it proposes exactly
//! the values, multipliers and children's warm start the CPU backend
//! proposes. The CPU backend is the REFERENCE: ny-cli's `cuda_fd_*` tests
//! compare the two bit for bit (NaN = NaN; the sign of a zero is not an
//! output), and a lane that runs `cuda` searches the same tree as one that
//! runs `cpu_f32`.
//!
//! # Trust
//!
//! THE DEVICE NEVER CERTIFIES. Every output is a [`Untrusted`] hint the tier
//! re-scores in CPU f64 with its own certified evaluator (weak duality makes
//! every non-negative point valid, so a wrong proposal can only fail to
//! certify), behind the lane's per-class known-answer probe (G6) and the M2
//! `UnsatToken` / cover gate — none of which this module touches or can name.
//!
//! # Packaging, context, failure (ENGINE_PROGRAM §1.9, JS F2)
//!
//! * The kernels are a committed fatbin (SASS sm_86 + sm_121, PTX compute_80)
//!   built offline by `scripts/build_search_kernels.sh`, embedded as bytes and
//!   loaded with cudarc's `nvrtc`-feature `Ptx::from_binary` — no `build.rs`,
//!   no runtime compilation (libnvrtc is never loaded), no cuBLAS. A manifest
//!   ratchet ties the image to its source.
//! * The backend runs in a CUDA context of its OWN (`cuCtxCreate`, not the
//!   primary context the verdict-feeding `CudaEngine` retains), and every
//!   method restores the calling thread's current context on return.
//! * Construction touches nothing. The first [`SearchBackend::upload_fd`]
//!   opens the device (driver library, context, compute capability >= 8.0,
//!   module) and runs a KNOWN-ANSWER SELF-TEST (the kernels' `exp` / `ln`
//!   against their host copies, and a dyadic level whose warm values have one
//!   exact answer); the outcome is LATCHED. Any failure — no driver, no
//!   device, a failed self-test, a later device fault or panic — is a
//!   [`Refusal`], never a panic and never a verdict: `cuda` then runs the
//!   class on its f64 search and `auto` falls through to `cpu_f32`. Only a
//!   DEVICE fault (a driver error, a failed device check, a panic in the device
//!   code) latches the backend off for the process, and it refuses as
//!   [`Refusal::Fault`]; an out-of-memory is a [`Refusal::Memory`] only; a
//!   HOST-side refusal (a launch bounds check, a size or index conversion, a
//!   buffer the charged layout does not carry, a poisoned lock) refuses that
//!   call only (`DevKind::Host`), because the level and the wave search share
//!   this backend and neither may switch the device off for the other.
//! * Memory (§1.8): an upload holds exactly the buffers of
//!   `plan::upload_buffers` + `plan::slab_buffers` (the planner and the
//!   allocator read the same layout), admitted against the lease BEFORE the
//!   device is touched and against `min(lease, free - 1.5 GiB, 0.85 total)`
//!   (a fresh `cuMemGetInfo` per upload) after; every launch is checked
//!   against the buffers actually allocated (`plan::launch_needs`) before it
//!   is issued; a level is searched in slabs of that capacity (about 50 ms of
//!   work each, the deadline polled between them), results scattered back in
//!   node order. MEASURED on the GB10 (review probe, 2026-09-29): a device
//!   allocation there is NOT in the process's `VmRSS` (1 GiB `alloc_zeros`:
//!   RSS +15 MB, free -1063 MB), and this backend's own context costs about
//!   73 MB of RSS and 226 MB of free memory that no lease charges.
//! * Determinism (R6): fixed-order sequential float sums, exact min / max,
//!   integer scans, no atomics — outputs depend on neither the grid, the slab
//!   size nor the chunk cut (`cuda_fd_outputs_do_not_depend_on_the_slab_or_the_chunk_cut`).
//! * The WAVE search: the first [`SearchBackend::upload_wave`] opens the
//!   device as above if nothing has (so the level search's self-test runs
//!   too), then loads the wave image into the same own context and runs the
//!   WAVE self-test (`wave_run::self_test`: an exact dyadic wave checked bit
//!   for bit against a host oracle, in one slab and in two-domain slabs, plus
//!   a short well-formed ascent). A wave-image or wave-self-test failure is
//!   latched on the wave side only ([`CudaSearchBackend::wave_unavailable`]);
//!   a DEVICE fault during a wave launch latches the whole backend (the
//!   context may be poisoned) and refuses as [`Refusal::Fault`]; a host-side
//!   refusal of a wave call refuses that call only. `margin_row.tree_mode:
//!   wave_cuda` runs this backend behind `ny_core::search::FallbackBackend`,
//!   which falls through to the CPU f32 wave search on a refused upload and
//!   switches a tree to it on a `Fault` (never on another refusal). Memory:
//!   the upload holds exactly
//!   `wave_plan::upload_buffers` + `wave_plan::slab_buffers`, admitted against
//!   the lease before the device is touched and against the live device cap
//!   after; one iteration of one slab is sized to about
//!   `wave_plan::PASS_WORK_BUDGET` multiply-adds and the deadline is polled
//!   between iterations. Determinism: every float reduction is a fixed-order
//!   sequential sum on one thread (`cuda_wave_outputs_do_not_depend_on_the_slab_or_the_batch_cut`).
//!
//! # Not built here (and why)
//!
//! Device-side compaction of the kept nodes' CHILDREN (so a level never
//! round-trips the host) would move tree construction off the lane, which the
//! soundness argument keeps on the lane ("the backend never sees or changes
//! the tree"), and needs a trait method M4 does not have; `fd_compact`
//! compacts the far-first WORK LISTS inside a level instead. Measurement (S1
//! nodes/s on GB10 and A10G, the M4 kill line CUDA >= 2x CPU f32 end to end)
//! and `compute-sanitizer memcheck` on the x86 A10G box are the integrator's.
//!
//! The wave kernels are one launch per op per pass (no fused multi-op kernel,
//! no shared-memory tiling of the conv gathers, a per-domain chain for every
//! bias sum): the simplest shape that is bit for bit the CPU reference. Their
//! throughput (the M6 target of >= 200 certified closes/s on cifar-medium, and
//! the device >= 2x the CPU f32 wave kill line) is UNMEASURED; `compute-sanitizer
//! memcheck` has not run on them.

mod device;
mod kernels;
mod plan;
#[cfg(test)]
mod tests;
mod wave_device;
mod wave_plan;
mod wave_run;
#[cfg(test)]
mod wave_tests;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ny_core::search::{
    BackendCaps, BranchOut, DeviceLease, FdHandle, FdSearchProblem, LevelChunk, LevelOut,
    LevelStats, Refusal, SearchBackend, Untrusted, WaveBatch, WaveHandle, WaveOut, WaveProblem,
    WaveSearchCfg, WaveStats, FD_WARM,
};

use device::{CurrentGuard, DevError, DevKind, Device, Upload};
use wave_device::{WaveKernels, WaveUpload};
use wave_plan::WaveProg;
use wave_run::RunErr;

/// The prefix of every refusal of the device wave search that is the
/// device's (unavailable, failed self-test, fault), for telemetry
/// (`[bbe] phase=wave ... device_refused=`).
pub const CUDA_WAVE_UNAVAILABLE: &str = "cuda wave search unavailable: ";

/// The backend's telemetry name (`search=cuda`).
pub const CUDA_SEARCH_BACKEND: &str = "cuda";

/// The device level search (see the module docs).
pub struct CudaSearchBackend {
    ordinal: usize,
    /// The opened, self-tested device, or why it is unavailable (latched).
    device: OnceLock<Result<Arc<Device>, String>>,
    /// A fault after a successful open (latched): every later call refuses.
    dead: OnceLock<String>,
    /// The wave kernels, loaded into the opened device and self-tested on
    /// the first wave upload, or why the wave search refuses (latched; the
    /// level search is unaffected).
    wave: OnceLock<Result<Arc<WaveKernels>, String>>,
    /// Test hook: nodes per slab (launch-configuration independence).
    #[cfg(test)]
    slab_override: Option<usize>,
    /// Test hook: the answer of the libcuda presence probe (fault injection
    /// of a host without a driver, on any host).
    #[cfg(test)]
    driver_override: Option<bool>,
}

impl std::fmt::Debug for CudaSearchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSearchBackend")
            .field("ordinal", &self.ordinal)
            .field("device", &self.device.get())
            .field("dead", &self.dead.get())
            .field("wave", &self.wave.get().map(|w| w.as_ref().err()))
            .finish_non_exhaustive()
    }
}

impl Default for CudaSearchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaSearchBackend {
    /// A backend on device 0. Touches no driver: the device is opened lazily,
    /// on the first upload.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ordinal: 0,
            device: OnceLock::new(),
            dead: OnceLock::new(),
            wave: OnceLock::new(),
            #[cfg(test)]
            slab_override: None,
            #[cfg(test)]
            driver_override: None,
        }
    }

    /// Whether libcuda can be loaded at all. cudarc PANICS (it does not err)
    /// on the first driver entry point it cannot resolve — `cuCtxGetCurrent`
    /// included — so no driver call may run before this holds.
    fn driver_present(&self) -> bool {
        #[cfg(test)]
        if let Some(present) = self.driver_override {
            return present;
        }
        crate::CudaGemmEngine::runtime_driver_library_present()
    }

    /// Whether the device has been opened (successfully or not).
    #[must_use]
    pub fn probed(&self) -> bool {
        self.device.get().is_some()
    }

    /// Why the backend refuses every request, if it does (the latched
    /// open or self-test failure, else the latched fault).
    #[must_use]
    pub fn unavailable(&self) -> Option<String> {
        match self.device.get() {
            Some(Err(why)) => Some(why.clone()),
            _ => self.dead.get().cloned(),
        }
    }

    /// The opened device (opening and self-testing it on first use).
    fn device(&self) -> Result<Arc<Device>, String> {
        let ordinal = self.ordinal;
        self.device
            .get_or_init(|| {
                let _restore = CurrentGuard::save();
                match catch_unwind(move || open_checked(ordinal)) {
                    Ok(Ok(dev)) => Ok(dev),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(_) => Err("opening the device panicked".to_string()),
                }
            })
            .clone()
    }

    fn latch(&self, why: &str) {
        let _ = self.dead.set(why.to_string());
    }

    /// Why the WAVE search refuses every request, if it does: the backend's
    /// own unavailability, or the wave kernels' latched load / self-test
    /// failure.
    #[must_use]
    pub fn wave_unavailable(&self) -> Option<String> {
        self.unavailable()
            .or_else(|| self.wave.get().and_then(|w| w.as_ref().err().cloned()))
    }

    /// The wave kernels on the opened `dev` (loading and self-testing them on
    /// first use; the outcome is latched).
    fn wave_kernels(&self, dev: &Arc<Device>) -> Result<Arc<WaveKernels>, String> {
        self.wave
            .get_or_init(|| {
                let _restore = CurrentGuard::save();
                match catch_unwind(AssertUnwindSafe(|| wave_device::open_wave(dev))) {
                    Ok(Ok(kern)) => Ok(kern),
                    Ok(Err(why)) => Err(format!("wave kernels: {why}")),
                    Err(_) => Err("wave kernels: loading or self-testing panicked".to_string()),
                }
            })
            .clone()
    }

    /// Refuse when the wave search cannot serve: a LATCHED FAULT (this
    /// backend's context may be poisoned; every later call refuses) is a
    /// [`Refusal::Fault`], so a composite may switch the tree away; a device
    /// that never opened, or wave kernels that never loaded, is
    /// [`Refusal::Invalid`] (only an upload can meet it).
    fn check_wave_alive(&self) -> Result<(), Refusal> {
        if let Some(why) = self.dead.get() {
            return Err(Refusal::Fault(format!("{CUDA_WAVE_UNAVAILABLE}{why}")));
        }
        match self.wave_unavailable() {
            Some(why) => Err(Refusal::Invalid(format!("{CUDA_WAVE_UNAVAILABLE}{why}"))),
            None => Ok(()),
        }
    }

    /// Latch `why` and refuse with it as a device fault (a driver error, or a
    /// panic in the device code: the context may be poisoned).
    fn fault(&self, search: &str, why: &str) -> Refusal {
        self.latch(why);
        Refusal::Fault(format!("cuda {search} search: {why}"))
    }

    /// A wave run's failure as a refusal: the deadline is a deadline; a
    /// host-side refusal (the bounds check refusing a slab before any launch,
    /// a size mismatch, a poisoned lock) refuses THAT CALL without latching —
    /// the context was never touched, and this backend also serves the level
    /// search; only a DEVICE fault (a driver error) latches the backend off,
    /// because the context may be poisoned, and refuses as
    /// [`Refusal::Fault`].
    fn wave_refusal(&self, e: RunErr) -> Refusal {
        match e {
            RunErr::Deadline => Refusal::Deadline,
            RunErr::Invalid(why) => Refusal::Invalid(format!("cuda wave search: {why}")),
            RunErr::Fault(why) => self.fault("wave", &format!("device fault: {why}")),
        }
    }

    /// A device error of `search` (`fd` / `wave`) as a refusal: an
    /// out-of-memory is a lease refusal; a HOST-side refusal refuses that call
    /// only; only a driver error (or a failed device check) latches the
    /// backend off, as a [`Refusal::Fault`]. The level and the wave search
    /// share this backend, so a host-side refusal of either must never switch
    /// the device off for both.
    fn refusal(&self, search: &str, e: &DevError, need: u64, cap: u64) -> Refusal {
        match e.kind {
            DevKind::Oom => Refusal::Memory { need, cap },
            DevKind::Host => Refusal::Invalid(format!("cuda {search} search: {}", e.what)),
            DevKind::Driver => self.fault(search, &format!("device fault: {}", e.what)),
        }
    }

    /// Refuse when the level search cannot serve: a latched fault is a
    /// [`Refusal::Fault`]; a device that never opened is [`Refusal::Invalid`].
    fn check_alive(&self) -> Result<(), Refusal> {
        if let Some(why) = self.dead.get() {
            return Err(Refusal::Fault(format!("cuda fd search unavailable: {why}")));
        }
        match self.unavailable() {
            Some(why) => Err(Refusal::Invalid(format!(
                "cuda fd search unavailable: {why}"
            ))),
            None => Ok(()),
        }
    }

    /// Nodes per launch for an upload of `n` generators at `depth`.
    fn slab_nodes(&self, n: usize, depth: usize, admitted: &plan::Plan) -> usize {
        #[cfg(test)]
        if let Some(k) = self.slab_override {
            let room = admitted
                .slab_nodes
                .min(admitted.slab_nd / depth.max(1))
                .max(1);
            return k.clamp(1, room);
        }
        plan::slab_nodes(n, depth, admitted)
    }
}

/// Open device `ordinal` and run the known-answer self-test on it.
fn open_checked(ordinal: usize) -> Result<Arc<Device>, DevError> {
    let dev = Arc::new(Device::open(ordinal)?);
    self_test(&dev)?;
    Ok(dev)
}

/// The backend's lane-independent known-answer self-test (G6): the kernels'
/// `exp` / `ln` bit for bit against their host copies, and a two-CTA,
/// two-tile dyadic level whose warm values and sums have ONE exact answer
/// (every partial sum is representable), plus well-formed proposals.
fn self_test(dev: &Arc<Device>) -> Result<(), DevError> {
    let xs = plan::math_kat_inputs();
    let (exp, ln) = dev.math_kat(&xs)?;
    for (k, &x) in xs.iter().enumerate() {
        let (want_e, want_l) = (plan::host_fast_exp(x), plan::host_fast_ln(x));
        if !plan::same_bits(exp[k], want_e) || !plan::same_bits(ln[k], want_l) {
            return Err(DevError::fault(format!(
                "self-test: exp/ln({x:e}) = ({:e}, {:e}) on the device vs ({want_e:e}, {want_l:e})",
                exp[k], ln[k]
            )));
        }
    }
    let (n, depth, count) = (300usize, 5usize, 11usize);
    let p = plan::kat_problem(n, depth, 0x5E1F);
    let (sides, l0, l1) = plan::kat_level(count, depth, 0x5E1F);
    let admitted = plan::Plan::exact(n, depth, 16, 16 * depth);
    let up = Upload::new(Arc::clone(dev), &p, admitted)?;
    let chunk = LevelChunk {
        depth,
        count,
        sides: &sides,
        l0: &l0,
        l1: &l1,
        deadline: None,
    };
    let mut out = LevelOut::default();
    out.prepare(count, depth);
    up.run_slab(&chunk, 0, count, &mut out.writer())?;
    drop(up);
    let view = out.hint().into_ingest();
    for k in 0..count {
        let r = k * depth..(k + 1) * depth;
        let (v, s) = plan::exact_warm(&p, &sides[r.clone()], &l0[r.clone()], &l1[r]);
        let (gv, gs) = (view.value(k, FD_WARM), view.scale(k, FD_WARM));
        if gv.to_bits() != v.to_bits() || gs.to_bits() != s.to_bits() {
            return Err(DevError::fault(format!(
                "self-test: node {k} warm (value, S) = ({gv:e}, {gs:e}) on the device vs the \
                 exact ({v:e}, {s:e})"
            )));
        }
        for (a, b) in [view.clamped(k), view.far(k), view.next(k)] {
            if !a.iter().chain(b).all(|m| m.is_finite() && *m >= 0.0) {
                return Err(DevError::fault(format!(
                    "self-test: node {k} proposed a non-finite or negative multiplier"
                )));
            }
        }
    }
    Ok(())
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

impl SearchBackend for CudaSearchBackend {
    fn name(&self) -> &'static str {
        CUDA_SEARCH_BACKEND
    }

    fn caps(&self) -> BackendCaps {
        match self.device.get() {
            Some(Ok(dev)) => BackendCaps::Cuda {
                sm: dev.sm,
                unified: dev.unified,
                mem_bytes: dev.cap,
            },
            // Not probed, or unavailable.
            _ => BackendCaps::Cuda {
                sm: 0,
                unified: false,
                mem_bytes: 0,
            },
        }
    }

    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        problem.validate().map_err(Refusal::Invalid)?;
        let (n, depth) = (problem.n, problem.depth);
        if n > plan::MAX_DIM
            || depth > plan::MAX_DIM
            || depth.saturating_mul(n) > isize::MAX as usize / 4
        {
            return Err(Refusal::Invalid(format!(
                "cuda fd search: {n} generators x {depth} rows exceed the kernels' index range"
            )));
        }
        // Admission against the lease BEFORE the device is touched.
        plan::plan(n, depth, lease.bytes)?;
        self.check_alive()?;
        // No libcuda: LATCH the absence and refuse, before any cudarc driver
        // symbol (even the context guard's `cuCtxGetCurrent`) is resolved —
        // cudarc panics on a missing library, and a refusal must not panic.
        if self.device.get().is_none() && !self.driver_present() {
            let why = "libcuda is not loadable on this host";
            let _ = self.device.set(Err(why.to_string()));
            return Err(Refusal::Invalid(format!(
                "cuda fd search unavailable: {why}"
            )));
        }
        let _restore = CurrentGuard::save();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let dev = self
                .device()
                .map_err(|why| Refusal::Invalid(format!("cuda fd search unavailable: {why}")))?;
            let live = dev.live_cap().map_err(|e| self.refusal("fd", &e, 0, 0))?;
            let cap = lease.bytes.min(live);
            let admitted = plan::plan(n, depth, cap)?;
            Upload::new(dev, problem, admitted)
                .map_err(|e| self.refusal("fd", &e, admitted.total(), cap))
        }));
        match result {
            Ok(Ok(upload)) => Ok(FdHandle::new(CUDA_SEARCH_BACKEND, upload)),
            Ok(Err(refusal)) => Err(refusal),
            Err(_) => Err(self.fault("fd", "the device upload panicked")),
        }
    }

    fn fd_level(
        &self,
        handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<Untrusted<LevelStats>, Refusal> {
        let t0 = Instant::now();
        let up = handle
            .payload::<Upload>()
            .ok_or_else(|| Refusal::Invalid(format!("foreign fd handle ({})", handle.backend())))?;
        chunk.validate().map_err(Refusal::Invalid)?;
        if chunk.depth > up.depth {
            return Err(Refusal::Invalid(format!(
                "fd level chunk: depth {} beyond the uploaded {} rows",
                chunk.depth, up.depth
            )));
        }
        if past(chunk.deadline) {
            return Err(Refusal::Deadline);
        }
        self.check_alive()?;
        let (count, depth) = (chunk.count, chunk.depth);
        out.prepare(count, depth);
        if count == 0 {
            return Ok(Untrusted::new(LevelStats::default()));
        }
        let per = self.slab_nodes(up.n, depth, &up.plan);
        let _restore = CurrentGuard::save();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut w = out.writer();
            let mut stats = LevelStats {
                nodes: count as u64,
                threads: 1,
                ..LevelStats::default()
            };
            let mut lo = 0usize;
            while lo < count {
                if past(chunk.deadline) {
                    return Err(Refusal::Deadline);
                }
                let hi = lo.saturating_add(per).min(count);
                let c = up
                    .run_slab(chunk, lo, hi, &mut w)
                    .map_err(|e| self.refusal("fd", &e, up.plan.total(), up.plan.total()))?;
                stats.evals += c.evals;
                stats.refused += c.refused;
                lo = hi;
            }
            stats.secs = t0.elapsed().as_secs_f64();
            Ok(stats)
        }));
        match result {
            Ok(r) => r.map(Untrusted::new),
            Err(_) => Err(self.fault("fd", "the device level search panicked")),
        }
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        // Lowering and admission against the lease BEFORE the device is
        // touched (the CPU search's refusal order: no ReLU, then validation).
        let prog = WaveProg::lower(problem)?;
        wave_plan::plan(&prog, lease.bytes)?;
        self.check_wave_alive()?;
        if self.device.get().is_none() && !self.driver_present() {
            let why = "libcuda is not loadable on this host";
            let _ = self.device.set(Err(why.to_string()));
            return Err(Refusal::Invalid(format!("{CUDA_WAVE_UNAVAILABLE}{why}")));
        }
        let _restore = CurrentGuard::save();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let dev = self
                .device()
                .map_err(|why| Refusal::Invalid(format!("{CUDA_WAVE_UNAVAILABLE}{why}")))?;
            let kern = self
                .wave_kernels(&dev)
                .map_err(|why| Refusal::Invalid(format!("{CUDA_WAVE_UNAVAILABLE}{why}")))?;
            let live = dev.live_cap().map_err(|e| self.refusal("wave", &e, 0, 0))?;
            let cap = lease.bytes.min(live);
            let admitted = wave_plan::plan(&prog, cap)?;
            let stride = prog.stride;
            WaveUpload::new(dev, kern, prog, admitted)
                .map(|up| WaveHandle::new(CUDA_SEARCH_BACKEND, stride, up))
                .map_err(|e| self.refusal("wave", &e, admitted.total(), cap))
        }));
        match result {
            Ok(r) => r,
            Err(_) => Err(self.fault("wave", "the device upload panicked")),
        }
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        let t0 = Instant::now();
        let up = wave_payload(handle, batch)?;
        self.check_wave_alive()?;
        let stride = up.prog.stride;
        out.prepare(batch.count, stride, batch.splits.len());
        if batch.count == 0 {
            return Ok(Untrusted::new(WaveStats::default()));
        }
        let _restore = CurrentGuard::save();
        let result = catch_unwind(AssertUnwindSafe(|| {
            up.search(batch, cfg, &mut out.writer())
        }));
        match result {
            Ok(Ok(counts)) => Ok(Untrusted::new(WaveStats {
                domains: batch.count as u64,
                passes: counts.passes,
                early_stopped: counts.early,
                secs: t0.elapsed().as_secs_f64(),
                threads: 1,
            })),
            Ok(Err(e)) => Err(self.wave_refusal(e)),
            Err(_) => Err(self.fault("wave", "the device wave search panicked")),
        }
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        let up = wave_payload(handle, batch)?;
        if k == 0 || k > wave_plan::WAVE_MAX_K {
            return Err(Refusal::Invalid(format!(
                "branch_scores: k = {k} outside 1..={}",
                wave_plan::WAVE_MAX_K
            )));
        }
        self.check_wave_alive()?;
        out.prepare(batch.count, k);
        if batch.count == 0 {
            return Ok(Untrusted::new(()));
        }
        let _restore = CurrentGuard::save();
        let result = catch_unwind(AssertUnwindSafe(|| up.branch(batch, k, &mut out.writer())));
        match result {
            Ok(Ok(_)) => Ok(Untrusted::new(())),
            Ok(Err(e)) => Err(self.wave_refusal(e)),
            Err(_) => Err(self.fault("wave", "the device branch scoring panicked")),
        }
    }
}

/// The CPU search's `upload_of`: this backend's upload in `handle`, the batch
/// validated against it, and the deadline not yet passed.
fn wave_payload<'h>(
    handle: &'h WaveHandle,
    batch: &WaveBatch<'_>,
) -> Result<&'h WaveUpload, Refusal> {
    let up = handle
        .payload::<WaveUpload>()
        .ok_or_else(|| Refusal::Invalid(format!("foreign wave handle ({})", handle.backend())))?;
    batch
        .validate(up.prog.stride, up.prog.classes.len())
        .map_err(Refusal::Invalid)?;
    if batch.deadline.is_some_and(|d| Instant::now() > d) {
        return Err(Refusal::Deadline);
    }
    Ok(up)
}
