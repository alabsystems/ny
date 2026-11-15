// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device side of the WAVE search (#wave-beta-bab, ENGINE_PROGRAM M6): the
//! wave kernels' module, loaded into the backend's OWN context (the one the
//! level search opened; never the primary context the verdict-feeding
//! `CudaEngine` retains) on the first wave upload and self-tested there; the
//! uploaded problem and its slab buffers, allocated from the one layout the
//! planner charged; and the cudarc executor of `wave_run`'s launch sequence.
//! Every driver error comes back as a `String` / [`DevError`]; the backend
//! turns it into a `Refusal` and latches. Nothing here is trusted: every output
//! is a hint the lane certifies in CPU f64.

use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use ny_core::search::{BranchOutWriter, WaveBatch, WaveOutWriter, WaveSearchCfg};

use super::device::{drv, CurrentGuard, DevError, DevResult, Device};
use super::kernels::{WAVE_KERNELS, WAVE_SEARCH_FATBIN};
use super::wave_plan::{
    literals_for, plan, slab_buffers, slab_bytes, upload_buffers, AdamConsts, Elem, WaveArgs,
    WaveBuf, WavePlan, WaveProg, WAVE_ARGS_POINTERS,
};
use super::wave_run::{self, Counts, ExecError, KArg, Kernel, Launch, RunErr, SlabHost, WaveExec};

// SAFETY: plain `#[repr(C)]` structs of integers and floats with no padding
// the kernels could misread: exactly the kernels' by-value parameters
// (`static_assert(sizeof ...)` in the kernel source against the const asserts
// in wave_plan.rs; the field lists are pinned by
// `wave_kernel_abi_matches_the_rust_side`).
unsafe impl DeviceRepr for WaveArgs {}
unsafe impl DeviceRepr for AdamConsts {}

/// The loaded wave module and its entry points.
pub(crate) struct WaveKernels {
    _module: Arc<CudaModule>,
    /// In [`WAVE_KERNELS`] order.
    funcs: Vec<CudaFunction>,
}

impl std::fmt::Debug for WaveKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveKernels").finish_non_exhaustive()
    }
}

/// `kernel`'s index in [`WAVE_KERNELS`] (pinned by
/// `wave_kernel_names_are_the_image_entry_points`).
#[must_use]
pub(crate) const fn kernel_index(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Init => 0,
        Kernel::Backward => 1,
        Kernel::Concretize => 2,
        Kernel::Forward => 3,
        Kernel::Adam => 4,
        Kernel::Babsr => 5,
    }
}

impl WaveKernels {
    /// Load the committed wave image into `dev`'s own context.
    pub(crate) fn load(dev: &Device) -> DevResult<Self> {
        dev.bind()?;
        let image = WAVE_SEARCH_FATBIN.to_vec();
        if image.as_ptr().align_offset(8) != 0 {
            return Err(DevError::fault(
                "the wave kernel image is not 8-byte aligned",
            ));
        }
        let module = dev
            .ctx
            .load_module(Ptx::from_binary(image))
            .map_err(drv("load the wave kernels"))?;
        let mut funcs = Vec::with_capacity(WAVE_KERNELS.len());
        for name in WAVE_KERNELS {
            funcs.push(module.load_function(name).map_err(drv(name))?);
        }
        Ok(Self {
            _module: module,
            funcs,
        })
    }

    fn func(&self, kernel: Kernel) -> &CudaFunction {
        &self.funcs[kernel_index(kernel)]
    }
}

/// One device buffer, whatever its element type.
enum DevBuf {
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    I32(CudaSlice<i32>),
    U32(CudaSlice<u32>),
}

impl DevBuf {
    fn len(&self) -> usize {
        match self {
            Self::F32(b) => b.len(),
            Self::F64(b) => b.len(),
            Self::I32(b) => b.len(),
            Self::U32(b) => b.len(),
        }
    }

    /// The device address (event tracking is disabled in the backend's
    /// context — one stream — so the guard records nothing; the address lives
    /// as long as the buffer).
    fn addr(&self, stream: &cudarc::driver::CudaStream) -> u64 {
        fn raw<T>(b: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> u64 {
            let (ptr, _sync) = b.device_ptr(stream);
            ptr
        }
        match self {
            Self::F32(b) => raw(b, stream),
            Self::F64(b) => raw(b, stream),
            Self::I32(b) => raw(b, stream),
            Self::U32(b) => raw(b, stream),
        }
    }
}

/// Every device buffer of one upload (the problem's tables and one slab),
/// indexed by [`WaveBuf`] — the one buffer table of `wave_plan.rs` names,
/// types and sizes them; nothing here lists them again.
struct WaveBufs {
    bufs: Vec<DevBuf>,
}

/// A typed view of buffer `$b` (a HOST refusal if the table gives it another
/// element type — the context is never touched).
macro_rules! typed {
    ($self:expr, $b:expr, $variant:ident, $how:ident) => {
        match $self.bufs.$how($b as usize) {
            Some(DevBuf::$variant(x)) => Ok(x),
            _ => Err(ExecError::Host(format!(
                "buffer {} is not a {} buffer of this upload",
                $b.name(),
                stringify!($variant)
            ))),
        }
    };
}

impl WaveBufs {
    /// Elements allocated for `name` (0 for a name this upload does not
    /// hold, so a check against it fails closed).
    fn len_of(&self, name: &str) -> usize {
        WaveBuf::from_name(name)
            .and_then(|b| self.bufs.get(b as usize))
            .map_or(0, DevBuf::len)
    }

    /// Device address of `buf` (0 for a buffer this upload does not hold; a
    /// launch refuses a zero pointer).
    fn ptr(&self, buf: WaveBuf, stream: &cudarc::driver::CudaStream) -> u64 {
        self.bufs.get(buf as usize).map_or(0, |b| b.addr(stream))
    }

    fn f32_ref(&self, b: WaveBuf) -> Result<&CudaSlice<f32>, ExecError> {
        typed!(self, b, F32, get)
    }

    fn u32_ref(&self, b: WaveBuf) -> Result<&CudaSlice<u32>, ExecError> {
        typed!(self, b, U32, get)
    }

    fn f32_mut(&mut self, b: WaveBuf) -> Result<&mut CudaSlice<f32>, ExecError> {
        typed!(self, b, F32, get_mut)
    }

    fn u32_mut(&mut self, b: WaveBuf) -> Result<&mut CudaSlice<u32>, ExecError> {
        typed!(self, b, U32, get_mut)
    }

    /// Allocate exactly [`upload_buffers`] and [`slab_buffers`] at `admitted`
    /// (in [`WaveBuf::ALL`] order) and copy the problem's tables. Fail
    /// CLOSED: a table whose length the charged layout does not match is a
    /// HOST refusal — never a buffer a kernel would index past.
    fn alloc(dev: &Device, prog: &WaveProg, admitted: &WavePlan) -> DevResult<Self> {
        dev.bind()?;
        let layout: Vec<(&'static str, Elem, usize)> = upload_buffers(prog)
            .into_iter()
            .chain(slab_buffers(prog.t, prog.stride, admitted.nd, admitted.nl))
            .collect();
        if layout.len() != WAVE_ARGS_POINTERS {
            return Err(DevError::host(format!(
                "the charged layout holds {} buffers, the ABI {WAVE_ARGS_POINTERS}",
                layout.len()
            )));
        }
        let mut bufs = Vec::with_capacity(WAVE_ARGS_POINTERS);
        for (buf, (name, elem, len)) in WaveBuf::ALL.into_iter().zip(layout) {
            if name != buf.name() || elem != buf.elem() {
                return Err(DevError::host(format!(
                    "buffer {} ({:?}) is not the charged layout's {name} ({elem:?})",
                    buf.name(),
                    buf.elem()
                )));
            }
            // The upload's tables carry data; the slab's start zeroed.
            let table = |have: usize| -> DevResult<()> {
                if have == len {
                    Ok(())
                } else {
                    Err(DevError::host(format!(
                        "{name}: the layout charges {len} elements, the program has {have}"
                    )))
                }
            };
            bufs.push(match buf {
                WaveBuf::Ops => {
                    table(prog.ops.len())?;
                    DevBuf::I32(copy(dev, &prog.ops)?)
                }
                WaveBuf::Fp => {
                    table(prog.fp.len())?;
                    DevBuf::F32(copy(dev, &prog.fp)?)
                }
                WaveBuf::Ip => {
                    table(prog.ip.len())?;
                    DevBuf::U32(copy(dev, &prog.ip)?)
                }
                WaveBuf::Classes => {
                    table(prog.classes.len())?;
                    DevBuf::U32(copy(dev, &prog.classes)?)
                }
                _ => match elem {
                    Elem::F32 => DevBuf::F32(zeros(dev, len)?),
                    Elem::F64 => DevBuf::F64(zeros(dev, len)?),
                    Elem::I32 => DevBuf::I32(zeros(dev, len)?),
                    Elem::U32 => DevBuf::U32(zeros(dev, len)?),
                },
            });
        }
        dev.stream.synchronize().map_err(drv("synchronize"))?;
        Ok(Self { bufs })
    }
}

fn zeros<T: DeviceRepr + ValidAsZeroBits>(dev: &Device, len: usize) -> DevResult<CudaSlice<T>> {
    dev.stream
        .alloc_zeros::<T>(len.max(1))
        .map_err(drv("device allocation"))
}

fn copy<T: DeviceRepr + ValidAsZeroBits>(dev: &Device, host: &[T]) -> DevResult<CudaSlice<T>> {
    let mut dst = zeros::<T>(dev, host.len())?;
    if !host.is_empty() {
        dev.stream
            .memcpy_htod(host, &mut dst.slice_mut(0..host.len()))
            .map_err(drv("host to device copy"))?;
    }
    Ok(dst)
}

/// Copy `host` into the first `host.len()` elements of `dst`.
fn put_into<T: DeviceRepr>(
    dev: &Device,
    dst: &mut CudaSlice<T>,
    host: &[T],
    name: &str,
) -> Result<(), ExecError> {
    if host.is_empty() {
        return Ok(());
    }
    if host.len() > dst.len() {
        return Err(ExecError::Host(format!(
            "slab input {name}: {} elements into a buffer of {}",
            host.len(),
            dst.len()
        )));
    }
    dev.stream
        .memcpy_htod(host, &mut dst.slice_mut(0..host.len()))
        .map_err(|e| ExecError::Device(format!("upload {name}: {e:?}")))
}

/// Copy the first `host.len()` elements of `src` back.
fn get_from<T: DeviceRepr>(
    dev: &Device,
    src: &CudaSlice<T>,
    host: &mut [T],
    name: &str,
) -> Result<(), ExecError> {
    if host.is_empty() {
        return Ok(());
    }
    if host.len() > src.len() {
        return Err(ExecError::Host(format!(
            "download {name}: {} elements from a buffer of {}",
            host.len(),
            src.len()
        )));
    }
    dev.stream
        .memcpy_dtoh(&src.slice(0..host.len()), host)
        .map_err(|e| ExecError::Device(format!("download {name}: {e:?}")))
}

/// Wait for the stream (a driver error is a device fault).
fn sync(dev: &Device) -> Result<(), ExecError> {
    dev.stream
        .synchronize()
        .map_err(|e| ExecError::Device(format!("synchronize: {e:?}")))
}

/// The cudarc executor of one upload's buffers.
struct Exec<'a> {
    dev: &'a Device,
    kern: &'a WaveKernels,
    b: &'a mut WaveBufs,
}

impl Exec<'_> {
    /// Copy the first `host.len()` elements of the f32 buffer `buf` back.
    fn get_f32(&self, buf: WaveBuf, host: &mut [f32]) -> Result<(), ExecError> {
        get_from(self.dev, self.b.f32_ref(buf)?, host, buf.name())
    }

    /// Copy the first `host.len()` elements of the u32 buffer `buf` back.
    fn get_u32(&self, buf: WaveBuf, host: &mut [u32]) -> Result<(), ExecError> {
        get_from(self.dev, self.b.u32_ref(buf)?, host, buf.name())
    }
}

impl WaveExec for Exec<'_> {
    fn len_of(&self, name: &str) -> usize {
        self.b.len_of(name)
    }

    fn put(&mut self, h: &SlabHost) -> Result<(), ExecError> {
        let (dev, b) = (self.dev, &mut *self.b);
        for (buf, host) in [
            (WaveBuf::Rows, &h.rows),
            (WaveBuf::LitPos, &h.lit_pos),
            (WaveBuf::LitAct, &h.lit_act),
            (WaveBuf::LitDom, &h.lit_dom),
        ] {
            put_into(dev, b.u32_mut(buf)?, host, buf.name())?;
        }
        for (buf, host) in [(WaveBuf::Beta, &h.beta), (WaveBuf::Alpha, &h.alpha)] {
            put_into(dev, b.f32_mut(buf)?, host, buf.name())?;
        }
        Ok(())
    }

    fn zero_coef(&mut self, n: usize) -> Result<(), ExecError> {
        if n == 0 {
            return Ok(());
        }
        let dev = self.dev;
        let coef = self.b.f32_mut(WaveBuf::Coef)?;
        if n > coef.len() {
            return Err(ExecError::Host(format!(
                "zeroing {n} coefficients of a buffer of {}",
                coef.len()
            )));
        }
        dev.stream
            .memset_zeros(&mut coef.slice_mut(0..n))
            .map_err(|e| ExecError::Device(format!("zero the coefficients: {e:?}")))
    }

    fn launch(&mut self, l: &Launch) -> Result<(), ExecError> {
        let stream = &self.dev.stream;
        let args = l.args.with_ptrs(|buf| self.b.ptr(buf, stream));
        if args.pointers().contains(&0) {
            return Err(ExecError::Host(format!(
                "{}: a kernel argument has no buffer",
                l.kernel.name()
            )));
        }
        let (int, last, stop, adam) = match l.arg {
            KArg::Int(v) => (v, 0i32, 0f64, AdamConsts::default()),
            KArg::Concretize { last, stop_above } => (0, last, stop_above, AdamConsts::default()),
            KArg::Adam(c) => (0, 0, 0.0, c),
            KArg::None => (0, 0, 0.0, AdamConsts::default()),
        };
        let cfg = LaunchConfig {
            grid_dim: (l.grid.0, l.grid.1, 1),
            block_dim: (l.block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(self.kern.func(l.kernel));
        b.arg(&args);
        match l.arg {
            KArg::Int(_) => {
                b.arg(&int);
            }
            KArg::Concretize { .. } => {
                b.arg(&last).arg(&stop);
            }
            KArg::Adam(_) => {
                b.arg(&adam);
            }
            KArg::None => {}
        }
        // SAFETY: the arguments are the kernel's signature (`const WaveArgs`,
        // then `int` | `int, double` | `const AdamConsts` | nothing); every
        // pointer in `args` is a live buffer of this upload, and `wave_run`
        // checked, before the slab's first launch, that every buffer holds at
        // least what a launch of this slab's shape indexes.
        unsafe { b.launch(cfg) }
            .map(|_| ())
            .map_err(|e| ExecError::Device(format!("launch {}: {e:?}", l.kernel.name())))
    }

    fn get_dom(&mut self, dom_u: &mut [u32], dom_f: &mut [f32]) -> Result<(), ExecError> {
        self.get_u32(WaveBuf::DomU, dom_u)?;
        self.get_f32(WaveBuf::DomF, dom_f)?;
        sync(self.dev)
    }

    fn get_best(&mut self, alpha: &mut [f32], beta: &mut [f32]) -> Result<(), ExecError> {
        self.get_f32(WaveBuf::BestAlpha, alpha)?;
        self.get_f32(WaveBuf::BestBeta, beta)?;
        sync(self.dev)
    }

    fn get_branch(&mut self, pos: &mut [u32], score: &mut [f32]) -> Result<(), ExecError> {
        self.get_u32(WaveBuf::Bpos, pos)?;
        self.get_f32(WaveBuf::Bscore, score)?;
        sync(self.dev)
    }
}

/// One wave tree's upload: the `WaveHandle` payload. It owns the problem's and
/// the slab's device buffers for the tree's lifetime and frees them in THIS
/// device's context whichever thread drops it.
pub(crate) struct WaveUpload {
    pub(crate) dev: Arc<Device>,
    kern: Arc<WaveKernels>,
    /// The lowered problem (host tables dropped once uploaded).
    pub(crate) prog: WaveProg,
    /// The admission the lease was charged for.
    pub(crate) plan: WavePlan,
    bufs: Mutex<Option<WaveBufs>>,
}

impl WaveUpload {
    /// Upload `prog` to `dev` under the admitted `plan`.
    pub(crate) fn new(
        dev: Arc<Device>,
        kern: Arc<WaveKernels>,
        mut prog: WaveProg,
        plan: WavePlan,
    ) -> DevResult<Self> {
        let bufs = WaveBufs::alloc(&dev, &prog, &plan)?;
        prog.strip();
        Ok(Self {
            dev,
            kern,
            prog,
            plan,
            bufs: Mutex::new(Some(bufs)),
        })
    }

    /// Run `f` on this upload's executor (the buffers locked, the context
    /// bound).
    fn with_exec<T>(
        &self,
        f: impl FnOnce(&mut Exec<'_>, &WaveProg, &WavePlan) -> T,
    ) -> Result<T, RunErr> {
        let mut guard = self.bufs.lock().map_err(|_| {
            RunErr::Invalid("the wave upload's buffers are poisoned by an earlier panic".into())
        })?;
        let b = guard
            .as_mut()
            .ok_or_else(|| RunErr::Invalid("the wave upload was released".into()))?;
        self.dev.bind().map_err(|e| RunErr::Fault(e.what))?;
        let mut x = Exec {
            dev: &self.dev,
            kern: &self.kern,
            b,
        };
        Ok(f(&mut x, &self.prog, &self.plan))
    }

    /// `backward_batch` on this upload (see `wave_run::run_search`).
    pub(crate) fn search(
        &self,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        w: &mut WaveOutWriter<'_>,
    ) -> Result<Counts, RunErr> {
        self.with_exec(|x, prog, plan| wave_run::run_search(x, prog, plan, batch, cfg, w))?
    }

    /// `branch_scores` on this upload (see `wave_run::run_branch`).
    pub(crate) fn branch(
        &self,
        batch: &WaveBatch<'_>,
        k: usize,
        w: &mut BranchOutWriter<'_>,
    ) -> Result<Counts, RunErr> {
        self.with_exec(|x, prog, plan| wave_run::run_branch(x, prog, plan, batch, k, w))?
    }
}

impl Drop for WaveUpload {
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

/// Device bytes the self-test's one-slab upload may hold (hundreds of
/// fixture domains; the self-test runs nine). Not charged to any lease: it is
/// released before the first real upload.
const SELF_TEST_BYTES: u64 = 4 << 20;

/// Load the wave kernels into `dev` and run their known-answer self-test
/// (G6): the exact dyadic fixture in one slab and in two-domain slabs (so the
/// test crosses slab boundaries). `Err` is why the wave search must refuse.
pub(crate) fn open_wave(dev: &Arc<Device>) -> Result<Arc<WaveKernels>, String> {
    let kern = Arc::new(WaveKernels::load(dev).map_err(|e| e.what)?);
    let prog = wave_run::kat_prog()?;
    let ample = plan(&prog, SELF_TEST_BYTES).map_err(|e| format!("self-test admission: {e}"))?;
    let stride = prog.stride;
    let nl = literals_for(stride, 2);
    let two = WavePlan {
        nd: 2,
        nl,
        slab: slab_bytes(prog.t, stride, 2, nl),
        ..ample
    };
    for admitted in [ample, two] {
        let up = WaveUpload::new(Arc::clone(dev), Arc::clone(&kern), prog.clone(), admitted)
            .map_err(|e| format!("self-test upload: {}", e.what))?;
        up.with_exec(|x, _, plan| {
            wave_run::self_test(x, &prog, plan, wave_run::KAT_DOMAINS, wave_run::KAT_SEED)
        })
        .map_err(|e| format!("self-test executor: {e:?}"))??;
    }
    Ok(kern)
}
