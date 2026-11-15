// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device side of the level search: the backend's OWN CUDA context
//! (`cuCtxCreate`, never the primary context `CudaEngine` retains), the
//! kernel module from the embedded fatbin, the uploaded problem, the reusable
//! slab buffers and the launch sequence. Every driver error is returned as a
//! `String`; the caller turns it into a `Refusal` and latches the backend.
//! Nothing here is trusted: every output is a hint the lane re-scores in f64.

use std::sync::{Arc, Mutex};

use cudarc::driver::result::DriverError;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use ny_core::search::{FdSearchProblem, LevelChunk, LevelOutWriter, FD_POINTS};

use super::kernels::{FD_SEARCH_FATBIN, KERNELS};
use super::plan::{
    self, constants, rel_threshold, slab_buffers, upload_buffers, Plan, EVALS_REFUSED,
    FD_COMPACT_THREADS, FD_FINISH_THREADS, FD_NB, FD_THREADS,
};

/// What kind of failure a [`DevError`] is, which decides whether it latches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DevKind {
    /// An allocation the device could not satisfy: a lease refusal, not a
    /// fault (the backend stays usable).
    Oom,
    /// A HOST-side refusal that never touched the device (a launch bounds
    /// check, a size or index conversion, a buffer the charged layout does not
    /// carry, a poisoned lock, a released upload): that call refuses, the
    /// backend stays usable.
    Host,
    /// A driver error or a failed device self-test: the context may be
    /// poisoned, so it latches the backend off for the process.
    Driver,
}

/// A device-side failure, classified ([`DevKind`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DevError {
    pub(crate) kind: DevKind,
    pub(crate) what: String,
}

impl DevError {
    /// A driver-level failure (latches).
    pub(crate) fn fault(what: impl Into<String>) -> Self {
        Self {
            kind: DevKind::Driver,
            what: what.into(),
        }
    }

    /// A host-side refusal (never latches).
    pub(crate) fn host(what: impl Into<String>) -> Self {
        Self {
            kind: DevKind::Host,
            what: what.into(),
        }
    }
}

impl std::fmt::Display for DevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.what)
    }
}

/// A driver call's error, classified.
pub(super) fn drv(what: &str) -> impl Fn(DriverError) -> DevError + '_ {
    move |e| DevError {
        kind: if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            DevKind::Oom
        } else {
            DevKind::Driver
        },
        what: format!("{what}: {e:?}"),
    }
}

pub(super) type DevResult<T> = Result<T, DevError>;

/// Saves the calling thread's current CUDA context and restores it on drop,
/// so the backend never leaves ITS context current on a lane thread (the
/// verdict-feeding `CudaEngine` binds its own before every call, but the
/// search must not depend on that).
///
/// When the current context could not be read — cudarc PANICS, rather than
/// erring, when libcuda cannot be loaded, even for `cuCtxGetCurrent` — the
/// guard restores nothing (no driver call on drop either).
pub(crate) struct CurrentGuard {
    /// The context was read (else nothing is restored).
    read: bool,
    /// The context current when the guard was made (null = none).
    ctx: cudarc::driver::sys::CUcontext,
}

impl CurrentGuard {
    pub(crate) fn save() -> Self {
        match std::panic::catch_unwind(|| cudarc::driver::result::ctx::get_current().ok().flatten())
        {
            Ok(ctx) => Self {
                read: true,
                ctx: ctx.unwrap_or(std::ptr::null_mut()),
            },
            Err(_) => Self {
                read: false,
                ctx: std::ptr::null_mut(),
            },
        }
    }
}

impl Drop for CurrentGuard {
    fn drop(&mut self) {
        if !self.read {
            return;
        }
        // SAFETY: `ctx` was current on this thread when the guard was made (or
        // is null, which unbinds); nothing the backend does destroys it.
        let _ = unsafe { cudarc::driver::result::ctx::set_current(self.ctx) };
    }
}

/// The opened device: an own context, its stream and the loaded kernels.
pub(crate) struct Device {
    pub(super) ctx: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    warm: CudaFunction,
    compact: CudaFunction,
    points: CudaFunction,
    finish: CudaFunction,
    math: CudaFunction,
    /// `major * 10 + minor`.
    pub(crate) sm: u32,
    /// The device shares the host's memory.
    pub(crate) unified: bool,
    /// Bytes the backend may lease on this device ([`plan::device_cap`]).
    pub(crate) cap: u64,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("sm", &self.sm)
            .field("unified", &self.unified)
            .field("cap", &self.cap)
            .finish_non_exhaustive()
    }
}

/// Lowest compute capability the embedded PTX (`compute_80`) serves.
const MIN_SM: u32 = 80;

impl Device {
    /// Open device `ordinal`: driver library, own context, capability, the
    /// fatbin module and its kernels. Fail-soft: every failure is `Err`.
    pub(crate) fn open(ordinal: usize) -> DevResult<Self> {
        if !crate::CudaGemmEngine::runtime_driver_library_present() {
            return Err(DevError::fault("libcuda is not loadable on this host"));
        }
        // Pin cudarc's retained driver handle; a loader panic becomes an error.
        std::panic::catch_unwind(|| {
            // SAFETY: initialises cudarc's process-global driver handle.
            let _ = unsafe { cudarc::driver::sys::culib() };
        })
        .map_err(|_| DevError::fault("libcuda failed to load"))?;
        let ctx = CudaContext::new_non_primary(ordinal, 0).map_err(drv("own CUDA context"))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(drv("compute capability"))?;
        let sm = u32::try_from(major * 10 + minor).unwrap_or(0);
        if sm < MIN_SM {
            return Err(DevError::fault(format!(
                "compute capability {major}.{minor} is below the kernels' {}.{}",
                MIN_SM / 10,
                MIN_SM % 10
            )));
        }
        // One stream, so no cross-stream event tracking is needed.
        // SAFETY: every slice of this context is used on its default stream only.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let image = FD_SEARCH_FATBIN.to_vec();
        if image.as_ptr().align_offset(8) != 0 {
            return Err(DevError::fault("the kernel image is not 8-byte aligned"));
        }
        let module = ctx
            .load_module(Ptx::from_binary(image))
            .map_err(drv("load the search kernels"))?;
        let load = |name: &str| module.load_function(name).map_err(drv(name));
        let [warm, compact, points, finish, math] = KERNELS;
        let (warm, compact, points, finish, math) = (
            load(warm)?,
            load(compact)?,
            load(points)?,
            load(finish)?,
            load(math)?,
        );
        let (free, total) = ctx.mem_get_info().map_err(drv("memory info"))?;
        let attr = |a: CUdevice_attribute| ctx.attribute(a).map(|v| v != 0).unwrap_or(false);
        let unified = attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            || attr(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
            );
        Ok(Self {
            ctx,
            stream,
            _module: module,
            warm,
            compact,
            points,
            finish,
            math,
            sm,
            unified,
            cap: plan::device_cap(free as u64, total as u64),
        })
    }

    /// The lease cap NOW (`min(free - reserve, 0.85 total)` from a fresh
    /// `cuMemGetInfo`): free device memory moves between classes — other
    /// contexts of this process (the verdict-feeding engine's among them) and
    /// other processes allocate — so an upload is admitted against the live
    /// figure, not the one read when the device was opened.
    pub(crate) fn live_cap(&self) -> DevResult<u64> {
        self.bind()?;
        let (free, total) = self.ctx.mem_get_info().map_err(drv("memory info"))?;
        Ok(plan::device_cap(free as u64, total as u64))
    }

    pub(super) fn bind(&self) -> DevResult<()> {
        self.ctx
            .bind_to_thread()
            .map_err(drv("bind the search context"))
    }

    fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> DevResult<CudaSlice<T>> {
        self.stream
            .alloc_zeros::<T>(len.max(1))
            .map_err(drv("device allocation"))
    }

    fn copy<T: DeviceRepr>(&self, host: &[T]) -> DevResult<CudaSlice<T>> {
        let mut dst = self.zeros_raw::<T>(host.len())?;
        if !host.is_empty() {
            self.stream
                .memcpy_htod(host, &mut dst)
                .map_err(drv("host to device copy"))?;
        }
        Ok(dst)
    }

    fn zeros_raw<T: DeviceRepr>(&self, len: usize) -> DevResult<CudaSlice<T>> {
        // SAFETY: every element is written (copy) before any kernel reads it.
        unsafe { self.stream.alloc::<T>(len.max(1)) }.map_err(drv("device allocation"))
    }

    /// The math known-answer probe: `(exp, ln)` of every input.
    pub(crate) fn math_kat(&self, xs: &[f32]) -> DevResult<(Vec<f32>, Vec<f32>)> {
        self.bind()?;
        let count = xs.len();
        let x = self.copy(xs)?;
        let mut out = self.zeros::<f32>(2 * count)?;
        let count_i = i32::try_from(count).map_err(|_| DevError::fault("math probe too large"))?;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(128).max(1) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.math);
        b.arg(&x).arg(&count_i).arg(&mut out);
        // SAFETY: the kernel reads `count` floats of `x` and writes `2 count`
        // of `out`, both allocated at those sizes.
        unsafe { b.launch(cfg) }.map_err(drv("launch fd_math_kat"))?;
        let host = self
            .stream
            .clone_dtoh(&out)
            .map_err(drv("device to host copy"))?;
        self.stream.synchronize().map_err(drv("synchronize"))?;
        Ok((host[..count].to_vec(), host[count..2 * count].to_vec()))
    }
}

/// The uploaded problem.
pub(crate) struct DevProblem {
    pub(crate) n: usize,
    pub(crate) depth: usize,
    c0: f32,
    abs_d: f32,
    a: CudaSlice<f32>,
    own: CudaSlice<i32>,
    dv: CudaSlice<f32>,
    pairs: CudaSlice<f32>,
    row_abs: CudaSlice<f32>,
    k: CudaSlice<f32>,
}

/// The reusable per-call buffers of one upload.
pub(crate) struct Slab {
    /// Nodes it holds.
    nodes: usize,
    /// Node-rows it holds.
    nd: usize,
    sides: CudaSlice<u8>,
    l0: CudaSlice<f32>,
    l1: CudaSlice<f32>,
    vec: CudaSlice<f32>,
    lm: CudaSlice<f32>,
    val: CudaSlice<f32>,
    abs: CudaSlice<f32>,
    flags: CudaSlice<u32>,
    slope: CudaSlice<f32>,
    eta: CudaSlice<f32>,
    evals: CudaSlice<u32>,
    items: CudaSlice<u32>,
    item_count: CudaSlice<u32>,
    out_clamped: CudaSlice<f32>,
    out_far: CudaSlice<f32>,
    out_next: CudaSlice<f32>,
}

impl Slab {
    /// Elements ACTUALLY allocated for buffer `name` (0 for a name this slab
    /// does not hold, so a check against it fails closed).
    fn len_of(&self, name: &str) -> usize {
        match name {
            "sides" => self.sides.len(),
            "l0" => self.l0.len(),
            "l1" => self.l1.len(),
            "vec" => self.vec.len(),
            "lm" => self.lm.len(),
            "val" => self.val.len(),
            "abs" => self.abs.len(),
            "flags" => self.flags.len(),
            "slope" => self.slope.len(),
            "eta" => self.eta.len(),
            "evals" => self.evals.len(),
            "items" => self.items.len(),
            "item_count" => self.item_count.len(),
            "out_clamped" => self.out_clamped.len(),
            "out_far" => self.out_far.len(),
            "out_next" => self.out_next.len(),
            _ => 0,
        }
    }
}

/// One slab's untrusted counters.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SlabCounts {
    pub(crate) evals: u64,
    pub(crate) refused: u64,
}

impl Device {
    /// Upload `p` (validated by the caller) and allocate the slab `admitted`
    /// sized. The buffers are exactly [`upload_buffers`] and
    /// [`slab_buffers`], so the lease the planner charged is what the device
    /// holds.
    pub(crate) fn upload(
        &self,
        p: &FdSearchProblem,
        admitted: &Plan,
    ) -> DevResult<(DevProblem, Slab)> {
        self.bind()?;
        let (n, depth) = (p.n, p.depth);
        let mut own = vec![-1i32; n];
        for (i, &e) in p.enc.iter().enumerate() {
            own[e as usize] = i32::try_from(i).map_err(|_| DevError::host("depth beyond i32"))?;
        }
        let mut pairs = Vec::with_capacity(depth * 8);
        for i in 0..depth {
            pairs.extend_from_slice(&[
                p.rho[i][0],
                p.rho[i][1],
                p.c0s[i][0],
                p.c0s[i][1],
                p.b0[i][0],
                p.b0[i][1],
                p.b1[i][0],
                p.b1[i][1],
            ]);
        }
        let layout = upload_buffers(n, depth);
        debug_assert_eq!(
            layout.map(|(_, _, len)| len),
            [
                p.a.len(),
                own.len(),
                p.d.len(),
                pairs.len(),
                p.row_abs.len(),
                plan::FD_NK
            ]
        );
        let problem = DevProblem {
            n,
            depth,
            c0: p.c0,
            abs_d: p.abs_d,
            a: self.copy(&p.a)?,
            own: self.copy(&own)?,
            dv: self.copy(&p.d)?,
            pairs: self.copy(&pairs)?,
            row_abs: self.copy(&p.row_abs)?,
            k: self.copy(&constants())?,
        };
        let layout = slab_buffers(admitted.slab_nodes, admitted.slab_nd);
        // Fail CLOSED: a name the charged layout does not carry is a fault,
        // never a 1-element buffer (the kernels would index past it — and on a
        // unified-memory device an out-of-bounds device write can reach host
        // pages, the certified state's included).
        let len = |name: &str| -> DevResult<usize> {
            layout
                .iter()
                .find(|(b, _, _)| *b == name)
                .map(|&(_, _, l)| l)
                .ok_or_else(|| {
                    DevError::host(format!("slab buffer {name} is not in the charged layout"))
                })
        };
        let slab = Slab {
            nodes: admitted.slab_nodes,
            nd: admitted.slab_nd,
            sides: self.zeros::<u8>(len("sides")?)?,
            l0: self.zeros::<f32>(len("l0")?)?,
            l1: self.zeros::<f32>(len("l1")?)?,
            vec: self.zeros::<f32>(len("vec")?)?,
            lm: self.zeros::<f32>(len("lm")?)?,
            val: self.zeros::<f32>(len("val")?)?,
            abs: self.zeros::<f32>(len("abs")?)?,
            flags: self.zeros::<u32>(len("flags")?)?,
            slope: self.zeros::<f32>(len("slope")?)?,
            eta: self.zeros::<f32>(len("eta")?)?,
            evals: self.zeros::<u32>(len("evals")?)?,
            items: self.zeros::<u32>(len("items")?)?,
            item_count: self.zeros::<u32>(len("item_count")?)?,
            out_clamped: self.zeros::<f32>(len("out_clamped")?)?,
            out_far: self.zeros::<f32>(len("out_far")?)?,
            out_next: self.zeros::<f32>(len("out_next")?)?,
        };
        self.stream.synchronize().map_err(drv("synchronize"))?;
        Ok((problem, slab))
    }

    /// Search nodes `lo..hi` of `chunk` (at most the slab's nodes and
    /// node-rows) and write their outputs into `out` at the same node indices.
    /// The kernels lay this launch's per-node vectors out with `count` (the
    /// launch's own node count) as the stride, so `count * d <= nd` bounds
    /// every index.
    pub(crate) fn run_slab(
        &self,
        p: &DevProblem,
        slab: &mut Slab,
        chunk: &LevelChunk<'_>,
        lo: usize,
        hi: usize,
        out: &mut LevelOutWriter<'_>,
    ) -> DevResult<SlabCounts> {
        let d = chunk.depth;
        let count = hi - lo;
        if count == 0 {
            return Ok(SlabCounts::default());
        }
        if count > slab.nodes || count.saturating_mul(d) > slab.nd || d > p.depth || d == 0 {
            return Err(DevError::host(format!(
                "a launch of {count} nodes at depth {d} exceeds the slab ({} nodes, {} \
                 node-rows) or the upload ({} rows)",
                slab.nodes, slab.nd, p.depth
            )));
        }
        // Every buffer the kernels index, against what was ACTUALLY allocated
        // (not the plan's arithmetic): the launch is refused before any
        // device write could land past an allocation.
        for (name, need) in plan::launch_needs(count, d) {
            let have = slab.len_of(name);
            if have < need {
                return Err(DevError::host(format!(
                    "a launch of {count} nodes at depth {d} needs {need} elements of slab buffer \
                     {name}, which holds {have}"
                )));
            }
        }
        self.bind()?;
        let r = lo * d..hi * d;
        self.stream
            .memcpy_htod(
                &chunk.sides[r.clone()],
                &mut slab.sides.slice_mut(0..count * d),
            )
            .map_err(drv("upload sides"))?;
        self.stream
            .memcpy_htod(&chunk.l0[r.clone()], &mut slab.l0.slice_mut(0..count * d))
            .map_err(drv("upload l0"))?;
        self.stream
            .memcpy_htod(&chunk.l1[r], &mut slab.l1.slice_mut(0..count * d))
            .map_err(drv("upload l1"))?;
        let as_i32 = |v: usize, what: &str| {
            i32::try_from(v).map_err(|_| DevError::host(format!("{what} beyond i32")))
        };
        let (n_i, d_i, count_i) = (
            as_i32(p.n, "n")?,
            as_i32(d, "depth")?,
            as_i32(count, "count")?,
        );
        // The vectors' layout stride: this launch's node count.
        let cap_i = count_i;
        let rel = rel_threshold(d);
        let blocks = |nodes: usize| (nodes.div_ceil(FD_NB).max(1)) as u32;
        let level = |grid: u32| LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (FD_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        // 1. warm + line search + far point.
        {
            let mut b = self.stream.launch_builder(&self.warm);
            b.arg(&p.a)
                .arg(&p.own)
                .arg(&p.dv)
                .arg(&p.pairs)
                .arg(&p.row_abs)
                .arg(&n_i)
                .arg(&p.c0)
                .arg(&p.abs_d)
                .arg(&d_i)
                .arg(&count_i)
                .arg(&cap_i)
                .arg(&slab.sides)
                .arg(&slab.l0)
                .arg(&slab.l1)
                .arg(&p.k)
                .arg(&mut slab.vec)
                .arg(&mut slab.val)
                .arg(&mut slab.abs)
                .arg(&mut slab.flags)
                .arg(&mut slab.slope)
                .arg(&mut slab.eta);
            // SAFETY: argument order and types are `fd_level_warm`'s; the
            // kernel indexes nodes `< count <= slab.nodes` and per-node vectors
            // `< FD_NVEC * count * d <= FD_NVEC * slab.nd` (checked above).
            unsafe { b.launch(level(blocks(count))) }.map_err(drv("launch fd_level_warm"))?;
        }
        // 2. far points, 3. clamped / floor points (far-first).
        for phase in [0i32, 1] {
            {
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (FD_COMPACT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(&self.compact);
                b.arg(&phase)
                    .arg(&count_i)
                    .arg(&rel)
                    .arg(&mut slab.flags)
                    .arg(&mut slab.val)
                    .arg(&mut slab.abs)
                    .arg(&mut slab.items)
                    .arg(&mut slab.item_count);
                // SAFETY: `fd_compact`'s signature; it writes at most
                // `2 count <= 2 slab.nodes` item ids.
                unsafe { b.launch(cfg) }.map_err(drv("launch fd_compact"))?;
            }
            let max_items = if phase == 0 { count } else { 2 * count };
            let mut b = self.stream.launch_builder(&self.points);
            b.arg(&p.a)
                .arg(&p.own)
                .arg(&p.dv)
                .arg(&p.pairs)
                .arg(&p.row_abs)
                .arg(&n_i)
                .arg(&p.c0)
                .arg(&p.abs_d)
                .arg(&d_i)
                .arg(&cap_i)
                .arg(&slab.sides)
                .arg(&mut slab.vec)
                .arg(&mut slab.lm)
                .arg(&mut slab.val)
                .arg(&mut slab.abs)
                .arg(&slab.items)
                .arg(&slab.item_count);
            // SAFETY: `fd_level_points`' signature; CTAs past the device-side
            // item count exit, and item rows `< 2 count * d <= 2 slab.nd`.
            unsafe { b.launch(level(blocks(max_items))) }.map_err(drv("launch fd_level_points"))?;
        }
        // 4. the children's warm start and the stored proposals.
        {
            let cfg = LaunchConfig {
                grid_dim: (
                    count.div_ceil(FD_FINISH_THREADS as usize).max(1) as u32,
                    1,
                    1,
                ),
                block_dim: (FD_FINISH_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.finish);
            b.arg(&d_i)
                .arg(&count_i)
                .arg(&cap_i)
                .arg(&p.k)
                .arg(&slab.vec)
                .arg(&slab.val)
                .arg(&slab.flags)
                .arg(&slab.slope)
                .arg(&slab.eta)
                .arg(&mut slab.out_clamped)
                .arg(&mut slab.out_far)
                .arg(&mut slab.out_next)
                .arg(&mut slab.evals);
            // SAFETY: `fd_level_finish`'s signature; it writes `2 count d <=
            // 2 slab.nd` multipliers per output.
            unsafe { b.launch(cfg) }.map_err(drv("launch fd_level_finish"))?;
        }
        // 5. download into the caller's view, in node order.
        let (vp, mp) = (lo * FD_POINTS..hi * FD_POINTS, lo * 2 * d..hi * 2 * d);
        let points = count * FD_POINTS;
        let mults = count * 2 * d;
        self.stream
            .memcpy_dtoh(&slab.val.slice(0..points), &mut out.value[vp.clone()])
            .map_err(drv("download values"))?;
        self.stream
            .memcpy_dtoh(&slab.abs.slice(0..points), &mut out.scale[vp])
            .map_err(drv("download sums"))?;
        self.stream
            .memcpy_dtoh(
                &slab.out_clamped.slice(0..mults),
                &mut out.clamped[mp.clone()],
            )
            .map_err(drv("download clamped"))?;
        self.stream
            .memcpy_dtoh(&slab.out_far.slice(0..mults), &mut out.far[mp.clone()])
            .map_err(drv("download far"))?;
        self.stream
            .memcpy_dtoh(&slab.out_next.slice(0..mults), &mut out.next[mp])
            .map_err(drv("download next"))?;
        let mut evals = vec![0u32; count];
        self.stream
            .memcpy_dtoh(&slab.evals.slice(0..count), &mut evals[..])
            .map_err(drv("download counters"))?;
        self.stream.synchronize().map_err(drv("synchronize"))?;
        let mut counts = SlabCounts::default();
        for e in evals {
            if e == EVALS_REFUSED {
                counts.refused += 1;
            } else {
                counts.evals += u64::from(e);
            }
        }
        Ok(counts)
    }
}

/// One class's upload: the `FdHandle` payload. It owns the problem's and the
/// slab's device buffers for the class's lifetime and frees them in THIS
/// device's context whichever thread drops it (cudarc's slice drop frees on
/// the current context's stream without binding, so the drop binds first).
pub(crate) struct Upload {
    pub(crate) dev: Arc<Device>,
    /// Generators.
    pub(crate) n: usize,
    /// Uploaded split rows (a chunk may be at most this deep).
    pub(crate) depth: usize,
    /// The admission the lease was charged for.
    pub(crate) plan: Plan,
    bufs: Mutex<Option<(DevProblem, Slab)>>,
}

impl Upload {
    /// Upload `p` to `dev` under the admitted `plan`.
    pub(crate) fn new(dev: Arc<Device>, p: &FdSearchProblem, plan: Plan) -> DevResult<Self> {
        let bufs = dev.upload(p, &plan)?;
        Ok(Self {
            n: p.n,
            depth: p.depth,
            plan,
            bufs: Mutex::new(Some(bufs)),
            dev,
        })
    }

    /// Search nodes `lo..hi` of `chunk` (see [`Device::run_slab`]).
    pub(crate) fn run_slab(
        &self,
        chunk: &LevelChunk<'_>,
        lo: usize,
        hi: usize,
        out: &mut LevelOutWriter<'_>,
    ) -> DevResult<SlabCounts> {
        let mut guard = self
            .bufs
            .lock()
            .map_err(|_| DevError::host("the upload's buffers are poisoned by an earlier panic"))?;
        let (problem, slab) = guard
            .as_mut()
            .ok_or_else(|| DevError::host("the upload was released"))?;
        self.dev.run_slab(problem, slab, chunk, lo, hi, out)
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        let _restore = CurrentGuard::save();
        let bufs = match self.bufs.get_mut() {
            Ok(b) => b.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if self.dev.bind().is_ok() {
            drop(bufs);
            // The frees are stream-ordered: release the lease before return.
            let _ = self.dev.stream.synchronize();
        } else {
            // A context that cannot be bound cannot free: leak rather than
            // free on whichever context is current (the driver reclaims the
            // memory with the context at exit).
            std::mem::forget(bufs);
        }
    }
}
