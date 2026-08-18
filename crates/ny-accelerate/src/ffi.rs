// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE SINGLE AUDITED `unsafe` MODULE OF THE ACCELERATE SEAM.
//!
//! Nothing outside this file uses `unsafe`. Every function here re-validates,
//! locally and unconditionally, every precondition the C ABI needs — it does
//! **not** trust a caller-side check, so the audit surface is exactly this file.
//!
//! # What is bound (G3, symbol identity)
//!
//! The **legacy** LP64 CBLAS entry points `cblas_dgemm` / `cblas_sgemm`. Apple's
//! headers rebind these to `cblas_dgemm$NEWLAPACK` (a *different* address) when
//! the C preprocessor macro `ACCELERATE_NEW_LAPACK` is defined; Rust has no
//! preprocessor, so this `extern` block always resolves the legacy symbol.
//! [`dgemm_provenance`] reports `dladdr`'s view of whatever actually got bound,
//! so the install log records the real symbol/dylib rather than an assumption.
//!
//! # ABI
//!
//! The legacy CBLAS ABI is **LP64**: every dimension and leading dimension is a
//! 32-bit `int`. `c_int::try_from` is therefore load-bearing, not decorative —
//! a `usize` dimension above `i32::MAX` must DECLINE, never truncate.

// `unsafe_code` is lifted for THIS module only, at the single `#[allow]` on
// `mod ffi;` in lib.rs. The crate root stays `#![deny(unsafe_code)]`.

use std::ffi::CStr;
use std::os::raw::{c_int, c_void};

/// CBLAS `CBLAS_ORDER::CblasRowMajor`.
const CBLAS_ROW_MAJOR: c_int = 101;
/// CBLAS `CBLAS_TRANSPOSE::CblasNoTrans`.
const CBLAS_NO_TRANS: c_int = 111;

/// `BLAS_THREADING_SINGLE_THREADED` (`vecLib/thread_api.h`, macOS 15+).
const BLAS_THREADING_SINGLE_THREADED: u32 = 1;

type DgemmFn = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f64,
    *const f64,
    c_int,
    *const f64,
    c_int,
    f64,
    *mut f64,
    c_int,
);

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );

    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// Dimensions validated against the LP64 ABI and the caller's slice lengths.
#[derive(Clone, Copy)]
struct Lp64Dims {
    m: c_int,
    k: c_int,
    n: c_int,
}

/// G1 (SHAPE/STRIDE), enforced *inside* the unsafe module.
///
/// Returns `None` — meaning DECLINE, never truncate — unless
/// `1 <= m,k,n <= i32::MAX`, the three products `m*k`, `k*n`, `m*n` are
/// representable in `usize`, and the three contiguous row-major slices have
/// EXACTLY those lengths. `lda=k`, `ldb=n`, `ldc=n` then describe the buffers
/// exactly, so the C side reads `a[0..m*k]`, `b[0..k*n]` and writes
/// `c[0..m*n]` and nothing else.
fn lp64_dims(m: usize, k: usize, n: usize, a: usize, b: usize, c: usize) -> Option<Lp64Dims> {
    if m == 0 || k == 0 || n == 0 {
        return None;
    }
    let mk = m.checked_mul(k)?;
    let kn = k.checked_mul(n)?;
    let mn = m.checked_mul(n)?;
    if a != mk || b != kn || c != mn {
        return None;
    }
    Some(Lp64Dims {
        m: c_int::try_from(m).ok()?,
        k: c_int::try_from(k).ok()?,
        n: c_int::try_from(n).ok()?,
    })
}

/// Row-major `C = A·B` (alpha=1, beta=0) via the legacy Accelerate `cblas_dgemm`.
///
/// Returns `false` without calling into C when any ABI precondition fails.
/// `beta = 0.0`: the probe's KA-1/KA-7 check proves Accelerate does not READ
/// `c`, so a NaN-poisoned destination is legal and callers may pass an
/// arbitrary (initialized) buffer.
#[must_use]
pub(crate) fn dgemm_row_major(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) -> bool {
    let Some(d) = lp64_dims(m, k, n, a.len(), b.len(), c.len()) else {
        return false;
    };
    // SAFETY: `d` proves 1 <= m,k,n <= i32::MAX and that `a`, `b`, `c` are
    // contiguous row-major buffers of exactly `m*k`, `k*n`, `m*n` f64s. With
    // RowMajor/NoTrans/NoTrans and lda=k, ldb=n, ldc=n, `cblas_dgemm` reads
    // strictly within `a`/`b` and writes strictly within `c`. `&mut [f64]`
    // guarantees `c` is uniquely borrowed and non-overlapping with `a`/`b`.
    // `alpha=1, beta=0` means no read of `c`. The call is thread-safe and
    // re-entrant (measured: 18 concurrent pthreads x 40 reps, 0 mismatches).
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            d.m,
            d.n,
            d.k,
            1.0,
            a.as_ptr(),
            d.k,
            b.as_ptr(),
            d.n,
            0.0,
            c.as_mut_ptr(),
            d.n,
        );
    }
    true
}

/// f32 twin of [`dgemm_row_major`]. Same ABI guard, same call form.
#[must_use]
pub(crate) fn sgemm_row_major(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> bool {
    let Some(d) = lp64_dims(m, k, n, a.len(), b.len(), c.len()) else {
        return false;
    };
    // SAFETY: identical argument to `dgemm_row_major`, with f32 elements.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            d.m,
            d.n,
            d.k,
            1.0,
            a.as_ptr(),
            d.k,
            b.as_ptr(),
            d.n,
            0.0,
            c.as_mut_ptr(),
            d.n,
        );
    }
    true
}

/// Read AArch64 `FPCR` on the calling thread.
///
/// KA-9 uses this to prove Accelerate leaves the floating-point control
/// register alone. A `dgemm` that set `FPCR.FZ` and forgot to restore it would
/// silently break every other f64 operation on the thread — including NY's
/// error-free-transformation primitives, which explicitly require no-FTZ.
#[must_use]
pub(crate) fn read_fpcr() -> u64 {
    let value: u64;
    // SAFETY: `mrs Xt, FPCR` is an unprivileged register read with no memory
    // operands and no side effects; it is available at EL0 on every AArch64
    // core. `nomem`/`nostack`/`preserves_flags` are all satisfied.
    unsafe {
        core::arch::asm!(
            "mrs {v}, fpcr",
            v = out(reg) value,
            options(nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Ask vecLib to keep BLAS on the calling thread (`BLASSetThreading`, macOS 15+).
///
/// This is the *correct* mechanism for the oversubscription hygiene the plan
/// asked `VECLIB_MAXIMUM_THREADS=1` to provide: it is a documented per-thread
/// thread-local, so it cannot race the way `setenv` can, and unlike the env var
/// it still takes effect after the framework has already been loaded and used
/// (which it has been on macOS — `blas-src` links Accelerate into every ny
/// binary today). Resolved with `dlsym` rather than a link-time `extern` so a
/// pre-macOS-15 host degrades to "multi-threaded vecLib" instead of failing to
/// launch. Returns `true` only when the call reported success.
#[must_use]
pub(crate) fn request_single_threaded_blas() -> bool {
    // SAFETY: `dlsym(RTLD_DEFAULT, ...)` with a NUL-terminated name is always
    // sound; it returns null when the symbol is absent, which is checked.
    let sym = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"BLASSetThreading".as_ptr()) };
    if sym.is_null() {
        return false;
    }
    // SAFETY: `vecLib/thread_api.h` declares
    // `int BLASSetThreading(const enum BLAS_THREADING)`; the enum has an
    // explicit `unsigned int` base type, so this is the exact C signature.
    let f: unsafe extern "C" fn(u32) -> c_int = unsafe { std::mem::transmute(sym) };
    // SAFETY: the argument is a valid enumerator
    // (`BLAS_THREADING_SINGLE_THREADED`); the function only writes a
    // thread-local and returns a status code.
    unsafe { f(BLAS_THREADING_SINGLE_THREADED) == 0 }
}

/// `dladdr` provenance for the `cblas_dgemm` symbol this build actually bound.
///
/// Returns `(dli_fname, dli_sname)`. Logged at install so the record shows the
/// real dylib and symbol (legacy vs `$NEWLAPACK`), mirroring
/// `GemmEngine::backend_provenance`'s "truthful identity" rule.
#[must_use]
pub(crate) fn dgemm_provenance() -> Option<(String, String)> {
    let f: DgemmFn = cblas_dgemm;
    let addr = f as *const c_void;
    // SAFETY: `Dl_info` is a plain-old-data struct of pointers/ints, so an
    // all-zero value is a valid initial state; `dladdr` fully initializes it on
    // success (nonzero return) and we read it only in that case.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: `addr` is the address of a symbol in a loaded image, and `info`
    // is a valid, exclusively-borrowed `Dl_info`.
    if unsafe { libc::dladdr(addr, &raw mut info) } == 0 {
        return None;
    }
    let read = |p: *const std::os::raw::c_char| -> String {
        if p.is_null() {
            return "?".to_string();
        }
        // SAFETY: on a successful `dladdr`, `dli_fname`/`dli_sname` are either
        // null (checked above) or NUL-terminated C strings owned by dyld and
        // valid for the lifetime of the loaded image.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    };
    Some((read(info.dli_fname), read(info.dli_sname)))
}
