// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The CPU f32 search backend (ENGINE_PROGRAM M4/M5a/V2/M6/M8 `CpuF32Backend`).

use std::time::Instant;

use ny_core::search::{
    BackendCaps, BranchOut, DeviceLease, DualSearchOutcome, DualSearchRequest, FdHandle,
    FdSearchProblem, FlBatch, FlHandle, FlOut, FlProblem, FlStats, GraphBabOutcome,
    GraphBabRequest, GraphRootOutcome, GraphRootRequest, GraphSearchNet, LevelChunk, LevelOut,
    LevelStats, Refusal, RootAlphaHint, RootSearchOutcome, RootSearchRequest, RootSearchStats,
    SearchBackend, SearchNet, SearchPrecision, Untrusted, WaveBatch, WaveHandle, WaveOut,
    WaveProblem, WaveSearchCfg, WaveStats,
};

use crate::fd::{self, FdProgram, FdReal, Upload};
use crate::fl;
use crate::graph_bab;
use crate::graph_root;
use crate::joint_root::{self, Config, LANES};
use crate::program::Program;
use crate::real::Real;
use crate::wave;

/// UNTRUSTED f32 root search on host threads, SIMD across [`LANES`] rows.
/// Dual breakpoint refinement searches in f64 and returns only hints.
///
/// Stateless: every call lowers the net, runs, and drops its buffers. `threads
/// = 0` uses the host's available parallelism (a request's own nonzero
/// `threads` wins).
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuF32Backend {
    threads: usize,
}

impl CpuF32Backend {
    /// A backend on the host's available parallelism.
    #[must_use]
    pub fn new() -> Self {
        Self { threads: 0 }
    }

    /// A backend pinned to `threads` workers (0 = available parallelism).
    #[must_use]
    pub fn with_threads(threads: usize) -> Self {
        Self { threads }
    }
}

impl SearchBackend for CpuF32Backend {
    fn name(&self) -> &'static str {
        "cpu_f32"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: self.threads,
            lanes: LANES,
            precision: SearchPrecision::F32,
        }
    }

    fn root_joint_alpha(
        &self,
        net: &SearchNet,
        request: &RootSearchRequest,
    ) -> Result<RootSearchOutcome, Refusal> {
        run_root::<f32>(net, request, self.threads, false)
    }

    fn refine_dual(&self, request: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
        crate::dual_breakpoint::search(request)
    }

    fn graph_root_alpha(
        &self,
        net: &GraphSearchNet,
        request: &GraphRootRequest,
    ) -> Result<GraphRootOutcome, Refusal> {
        graph_root::run_graph_root::<f32>(
            net,
            request,
            resolve_threads(request.threads, self.threads),
        )
    }

    fn graph_bab_batch(
        &self,
        net: &GraphSearchNet,
        request: &GraphBabRequest,
    ) -> Result<GraphBabOutcome, Refusal> {
        graph_bab::run_graph_bab::<f32>(
            net,
            request,
            resolve_threads(request.threads, self.threads),
        )
    }

    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        upload_fd::<f32>(self.name(), problem, lease, self.threads)
    }

    fn fd_level(
        &self,
        handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<Untrusted<LevelStats>, Refusal> {
        fd_level::<f32>(handle, chunk, out)
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        wave::upload::<f32>(
            self.name(),
            problem,
            lease,
            resolve_threads(0, self.threads),
        )
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        wave::backward_batch::<f32>(handle, batch, cfg, out).map(Untrusted::new)
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        wave::branch_scores::<f32>(handle, batch, k, out).map(Untrusted::new)
    }

    fn upload_forward_linear(
        &self,
        problem: &FlProblem,
        lease: &DeviceLease,
    ) -> Result<FlHandle, Refusal> {
        fl::upload::<f32>(
            self.name(),
            problem,
            lease,
            resolve_threads(0, self.threads),
        )
    }

    fn forward_linear_batch(
        &self,
        handle: &FlHandle,
        batch: &FlBatch<'_>,
        out: &mut FlOut,
    ) -> Result<Untrusted<FlStats>, Refusal> {
        fl::forward_linear_batch::<f32>(handle, batch, out, 0).map(Untrusted::new)
    }
}

/// M4: lower the frozen-dual search copy into `R` for `threads` workers,
/// refusing when the copy plus every worker's working set exceeds the lease.
pub(crate) fn upload_fd<R: FdReal>(
    backend: &'static str,
    problem: &FdSearchProblem,
    lease: &DeviceLease,
    backend_threads: usize,
) -> Result<FdHandle, Refusal> {
    let prog = FdProgram::<R>::lower(problem)?;
    let threads = resolve_threads(0, backend_threads);
    let need = prog
        .bytes()
        .saturating_add((threads as u64).saturating_mul(fd::work_bytes::<R>(prog.n, prog.depth)));
    if need > lease.bytes {
        return Err(Refusal::Memory {
            need,
            cap: lease.bytes,
        });
    }
    Ok(FdHandle::new(backend, Upload { prog, threads }))
}

/// M4: one level chunk on the upload in `handle` (a handle of another
/// backend or scalar is refused).
pub(crate) fn fd_level<R: FdReal>(
    handle: &FdHandle,
    chunk: &LevelChunk<'_>,
    out: &mut LevelOut,
) -> Result<Untrusted<LevelStats>, Refusal> {
    let up = handle
        .payload::<Upload<R>>()
        .ok_or_else(|| Refusal::Invalid(format!("foreign fd handle ({})", handle.backend())))?;
    fd::fd_level(&up.prog, chunk, out, up.threads).map(Untrusted::new)
}

/// Resolve the worker count: the request's, else the backend's, else the host.
pub(crate) fn resolve_threads(requested: usize, backend: usize) -> usize {
    if requested > 0 {
        requested
    } else if backend > 0 {
        backend
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    }
}

/// Lower, search in `R`, and wrap the result as untrusted outputs.
pub(crate) fn run_root<R: Real>(
    net: &SearchNet,
    request: &RootSearchRequest,
    backend_threads: usize,
    force_shared: bool,
) -> Result<RootSearchOutcome, Refusal> {
    let t0 = Instant::now();
    let program = Program::<R>::lower(net).map_err(Refusal::Invalid)?;
    let config = Config {
        iters: request.iters,
        lr: if request.lr.is_finite() && request.lr > 0.0 {
            request.lr
        } else {
            0.25
        },
        deadline: request.deadline,
        threads: resolve_threads(request.threads, backend_threads),
        mem_cap: request.mem_cap_bytes,
        force_shared,
        chunk_blocks: None,
    };
    let result = joint_root::search(&program, request, &config)?;
    Ok(RootSearchOutcome {
        alpha: Untrusted::new(RootAlphaHint {
            trunk: result.hint_trunk,
            head: result.hint_head,
        }),
        stats: RootSearchStats {
            iters: result.iters,
            passes: result.passes,
            search_secs: t0.elapsed().as_secs_f64(),
            crown_lb: Untrusted::new(result.crown_lb),
            root_lb: Untrusted::new(result.root_lb),
            specs: result.specs,
            pruned: result.pruned,
            rows: result.rows,
            share_rows: result.share_rows,
            threads: result.threads,
            tightened: result.tightened,
            stabilized: result.stabilized,
            stop: result.stop,
        },
    })
}
