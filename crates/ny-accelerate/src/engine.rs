// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The Accelerate-backed [`GemmEngine`] and its gates.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use ny_core::{GemmEngine, NyError, Result};

use crate::ffi;
use crate::probe::{self, ProbeReport};

/// Kill switch. Presence (any value) disables the seam entirely, exactly like
/// `NY_NO_CUDA`.
pub const KILL_SWITCH_ENV: &str = "NY_NO_ACCELERATE";
/// Arms the SOUND f64 seam (verdict path). Default OFF: without `=1` the engine
/// is never installed and every published bound is byte-identical to today.
pub const F64_GATE_ENV: &str = "NY_ACCELERATE_F64";
/// Arms the f32 free-rider (IBP / PGD / BaB — non-verdict). Default OFF.
pub const F32_GATE_ENV: &str = "NY_ACCELERATE_F32";
/// Minimum `m*k*n` for an f64 offload. Below it `gemm_f64` DECLINES so the
/// caller keeps faer: measured, Accelerate runs 8x64x8 at 33-39 GFLOP/s versus
/// 353-454 GFLOP/s at blocked shapes, so tiny products are not worth the call.
pub const MIN_MACS_ENV: &str = "NY_ACCELERATE_MIN_MACS";
/// Set to `multi` to leave vecLib's own threading alone. Default: the engine
/// asks for `BLAS_THREADING_SINGLE_THREADED` on each thread that uses it, so
/// libdispatch cannot oversubscribe against rayon's domain workers.
pub const BLAS_THREADS_ENV: &str = "NY_ACCELERATE_BLAS_THREADS";

/// Default f64 offload floor: 2^15 MACs (e.g. 32x32x32).
pub const DEFAULT_MIN_MACS: usize = 1 << 15;

/// G2 floor. `min nonzero |a| * min nonzero |b| >= 2^-969` guarantees no
/// product and no partial sum is subnormal, closing the one regime the
/// multiplicative `γ_n·S` envelope cannot cover (a subnormal product has
/// `fl(a·b) = ab(1+δ) + η`, and the ADDITIVE `η` has no room in `γ_n·S`).
const UNDERFLOW_DOMAIN_MIN_EXP2: i32 = -969;

static CALLS_F64: AtomicU64 = AtomicU64::new(0);
static CALLS_F32_ACCELERATE: AtomicU64 = AtomicU64::new(0);
static CALLS_F32_FAER: AtomicU64 = AtomicU64::new(0);
static DECLINED_SHAPE: AtomicU64 = AtomicU64::new(0);
static DECLINED_SMALL: AtomicU64 = AtomicU64::new(0);
static DECLINED_UNDERFLOW_DOMAIN: AtomicU64 = AtomicU64::new(0);
static DECLINED_NON_FINITE: AtomicU64 = AtomicU64::new(0);

/// Non-forcing view of what the seam actually did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AccelerateTelemetry {
    /// `gemm_f64` calls that reached `cblas_dgemm`.
    pub f64_calls: u64,
    /// `gemm_f32` calls that reached `cblas_sgemm`.
    pub f32_accelerate_calls: u64,
    /// `gemm_f32` calls that took the incumbent faer kernel.
    pub f32_faer_calls: u64,
    /// G1 refusals (dimension / stride / length).
    pub declined_shape: u64,
    /// Below the MAC floor.
    pub declined_small: u64,
    /// G2 refusals (operand magnitudes could produce a subnormal product).
    pub declined_underflow_domain: u64,
    /// Refusals because an operand was NaN or infinite.
    pub declined_non_finite: u64,
}

/// Read the seam counters without touching the engine.
#[must_use]
pub fn telemetry() -> AccelerateTelemetry {
    AccelerateTelemetry {
        f64_calls: CALLS_F64.load(Ordering::Relaxed),
        f32_accelerate_calls: CALLS_F32_ACCELERATE.load(Ordering::Relaxed),
        f32_faer_calls: CALLS_F32_FAER.load(Ordering::Relaxed),
        declined_shape: DECLINED_SHAPE.load(Ordering::Relaxed),
        declined_small: DECLINED_SMALL.load(Ordering::Relaxed),
        declined_underflow_domain: DECLINED_UNDERFLOW_DOMAIN.load(Ordering::Relaxed),
        declined_non_finite: DECLINED_NON_FINITE.load(Ordering::Relaxed),
    }
}

fn env_present(key: &str) -> bool {
    std::env::var_os(key).is_some()
}

fn env_is_one(key: &str) -> bool {
    matches!(std::env::var(key).ok().as_deref(), Some("1"))
}

/// `NY_NO_ACCELERATE` is set: nothing is installed, on any gate.
#[must_use]
pub fn kill_switch_engaged() -> bool {
    env_present(KILL_SWITCH_ENV)
}

/// Whether the SOUND f64 seam is armed (`NY_ACCELERATE_F64=1`, no kill switch).
#[must_use]
pub fn f64_seam_armed() -> bool {
    !kill_switch_engaged() && env_is_one(F64_GATE_ENV)
}

/// Whether the f32 free-rider is armed (`NY_ACCELERATE_F32=1`, no kill switch).
#[must_use]
pub fn f32_seam_armed() -> bool {
    !kill_switch_engaged() && env_is_one(F32_GATE_ENV)
}

fn min_macs_from_env() -> usize {
    std::env::var(MIN_MACS_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIN_MACS)
}

fn single_threaded_blas_requested() -> bool {
    !matches!(
        std::env::var(BLAS_THREADS_ENV).ok().as_deref(),
        Some("multi")
    )
}

thread_local! {
    static BLAS_THREADING_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Apply vecLib's per-thread single-threading hint once per thread.
fn apply_thread_blas_policy(enabled: bool) {
    if !enabled {
        return;
    }
    BLAS_THREADING_SET.with(|done| {
        if !done.get() {
            let _ = ffi::request_single_threaded_blas();
            done.set(true);
        }
    });
}

/// Whether vecLib accepted the per-thread single-threading request on THIS
/// thread (`BLASSetThreading`, macOS 15+).
///
/// DEVIATION FROM THE PLAN, DELIBERATE. The plan asked for
/// `VECLIB_MAXIMUM_THREADS=1`. This engine does not call `std::env::set_var`:
/// `setenv(3)` racing a concurrent `getenv(3)` is undefined behaviour in libc,
/// and NY's install point already coexists with rayon/ORT/wgpu threads, so a
/// process-wide env mutation is a real hazard for a zero-incorrect verifier.
/// `BLASSetThreading` is the documented replacement — a per-thread thread-local
/// that also works AFTER the framework has been loaded and used (which it has:
/// `blas-src` links Accelerate into every macOS ny binary today, so the env var
/// may already be too late to read). Qualification also failed to reproduce the
/// prior doc's "MEASURED FREE" 243.7-vs-218.8 GFLOP/s claim for the env var
/// (353-385 with vs 358-381 without, bit-identical either way), so nothing of
/// value is given up. Operators who still want the env var can export it in the
/// launcher, where it is set before any thread exists.
#[must_use]
pub fn single_threaded_blas_available() -> bool {
    ffi::request_single_threaded_blas()
}

/// `floor(log2 |x|)`, or `None` for zero / non-finite.
fn floor_log2_abs(x: f64) -> Option<i32> {
    if !x.is_finite() || x == 0.0 {
        return None;
    }
    let bits = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let biased = (bits >> 52) as i32;
    if biased > 0 {
        Some(biased - 1023)
    } else {
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        let highest = 63 - mantissa.leading_zeros() as i32;
        Some(highest - 1074)
    }
}

enum DomainScan {
    /// An operand was NaN or infinite.
    NonFinite,
    /// Every operand is exactly zero — the product is exactly zero.
    AllZero,
    /// `floor(log2)` of the smallest nonzero magnitude.
    MinExp2(i32),
}

fn scan_domain(v: &[f64]) -> DomainScan {
    let mut min: Option<i32> = None;
    for &x in v {
        if !x.is_finite() {
            return DomainScan::NonFinite;
        }
        if let Some(e) = floor_log2_abs(x) {
            min = Some(min.map_or(e, |m: i32| m.min(e)));
        }
    }
    min.map_or(DomainScan::AllZero, DomainScan::MinExp2)
}

/// G2 (UNDERFLOW DOMAIN) + a non-finite refusal.
///
/// Exact integer test on exponents, never a float multiply that could itself
/// round across the boundary: with `|a| >= 2^ea` and `|b| >= 2^eb` for every
/// nonzero entry, every nonzero product is `>= 2^(ea+eb)`. Requiring
/// `ea + eb >= -969` therefore makes every product normal, hence every product
/// a multiple of `2^-1022`, hence every exact partial sum a multiple of
/// `2^-1022` — so no partial sum is subnormal either, and the multiplicative
/// `γ_n·S` envelope is complete.
///
/// ON THE CROWN SEAM THIS IS FREE INSURANCE, NOT A CONSTRAINT. `aw_via_engine`
/// feeds `f32_to_f64_exact` widenings, so every nonzero operand has
/// `|x| >= 2^-149` and `ea + eb >= -298`: the guard can never fire there. It
/// exists for the GENUINE f64 surfaces (`forward_linear/image.rs`, conv
/// `ops_gemm` / `ops_transpose_gemm`), where operands are real f64 and the
/// regime is reachable. The `O(mk + kn)` scan is ~0.4% of the `O(mkn)` GEMM.
fn underflow_domain_admissible(a: &[f64], b: &[f64]) -> std::result::Result<(), Refusal> {
    let ea = match scan_domain(a) {
        DomainScan::NonFinite => return Err(Refusal::NonFinite),
        DomainScan::AllZero => return Ok(()),
        DomainScan::MinExp2(e) => e,
    };
    let eb = match scan_domain(b) {
        DomainScan::NonFinite => return Err(Refusal::NonFinite),
        DomainScan::AllZero => return Ok(()),
        DomainScan::MinExp2(e) => e,
    };
    if ea.saturating_add(eb) >= UNDERFLOW_DOMAIN_MIN_EXP2 {
        Ok(())
    } else {
        Err(Refusal::UnderflowDomain)
    }
}

enum Refusal {
    NonFinite,
    UnderflowDomain,
}

/// Apple Accelerate (vecLib BLAS) GEMM engine.
///
/// # Soundness
///
/// `cblas_dgemm` is an IEEE-f64 GEMM. NY's CROWN certificate charges
/// `γ_n·S` with `S = Σ|a||b|`, a bound that is valid for ANY summation order
/// (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 3.1) — which
/// is exactly the property an opaque vendor BLAS needs. That is the identical
/// argument `ny-cuda` already banks for `cublasDgemm` and `faer_parallelism`
/// banks for faer's blocked/threaded reduction. This engine adds NO new
/// soundness theory; it adds a new *implementation* under the existing theory,
/// plus three guards (G1/G2/G3) and a one-shot conformance probe that check the
/// theory's preconditions actually hold on this host.
///
/// The engine is default-OFF. Until `NY_ACCELERATE_F64=1`, `new()` is never
/// called from a factory and every bound is byte-identical to today.
pub struct AccelerateGemmEngine {
    f64_enabled: bool,
    f32_via_accelerate: bool,
    min_macs: usize,
    single_threaded_blas: bool,
    dgemm_report: ProbeReport,
    sgemm_report: Option<ProbeReport>,
    provenance: Option<(String, String)>,
}

impl std::fmt::Debug for AccelerateGemmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelerateGemmEngine")
            .field("f64_enabled", &self.f64_enabled)
            .field("f32_via_accelerate", &self.f32_via_accelerate)
            .field("min_macs", &self.min_macs)
            .field("single_threaded_blas", &self.single_threaded_blas)
            .field("provenance", &self.provenance)
            .field("dgemm_checks", &self.dgemm_report.checks.len())
            .field(
                "sgemm_checks",
                &self.sgemm_report.as_ref().map(|r| r.checks.len()),
            )
            .finish()
    }
}

impl AccelerateGemmEngine {
    /// Construct the engine, running the one-shot conformance probe.
    ///
    /// FAIL-CLOSED. Returns `None` when the kill switch is set, when neither
    /// gate is armed, or when ANY conformance check fails — the caller then
    /// installs nothing and the incumbent faer engine keeps the seam. The f32
    /// free-rider is additionally armed only if its own probe passes; if it
    /// fails, `gemm_f32` silently keeps faer while the f64 seam continues.
    #[must_use]
    pub fn new() -> Option<Self> {
        Self::new_with_gates(f64_seam_armed(), f32_seam_armed())
    }

    /// Gate-explicit constructor (tests; the CLI uses [`Self::new`]).
    ///
    /// Still fail-closed on the probe and still honours `NY_NO_ACCELERATE`.
    #[must_use]
    pub fn new_with_gates(f64_armed: bool, f32_armed: bool) -> Option<Self> {
        if kill_switch_engaged() || (!f64_armed && !f32_armed) {
            return None;
        }
        let single_threaded_blas = single_threaded_blas_requested();
        apply_thread_blas_policy(single_threaded_blas);

        let dgemm_report = probe::dgemm_conformance_probe();
        if !dgemm_report.accepted() {
            tracing::warn!(
                failures = ?dgemm_report.failures(),
                "Accelerate conformance probe REFUSED cblas_dgemm; keeping the incumbent \
                 f64 GEMM engine"
            );
            return None;
        }
        let sgemm_report = if f32_armed {
            let report = probe::sgemm_conformance_probe();
            if !report.accepted() {
                tracing::warn!(
                    failures = ?report.failures(),
                    "Accelerate conformance probe REFUSED cblas_sgemm; the f32 free-rider \
                     stays on faer"
                );
            }
            Some(report)
        } else {
            None
        };
        let f32_via_accelerate = sgemm_report.as_ref().is_some_and(ProbeReport::accepted);

        Some(Self {
            f64_enabled: f64_armed,
            f32_via_accelerate,
            min_macs: min_macs_from_env(),
            single_threaded_blas,
            dgemm_report,
            sgemm_report,
            provenance: ffi::dgemm_provenance(),
        })
    }

    /// The dgemm conformance report this engine was admitted on.
    #[must_use]
    pub fn dgemm_report(&self) -> &ProbeReport {
        &self.dgemm_report
    }

    /// The sgemm conformance report, when the f32 gate was armed.
    #[must_use]
    pub fn sgemm_report(&self) -> Option<&ProbeReport> {
        self.sgemm_report.as_ref()
    }

    /// Whether `gemm_f32` routes to `cblas_sgemm` (vs the incumbent faer path).
    #[must_use]
    pub fn f32_via_accelerate(&self) -> bool {
        self.f32_via_accelerate
    }

    /// `dladdr` view of the bound `cblas_dgemm` (G3 provenance): `(dylib, symbol)`.
    #[must_use]
    pub fn symbol_provenance(&self) -> Option<(&str, &str)> {
        self.provenance
            .as_ref()
            .map(|(file, name)| (file.as_str(), name.as_str()))
    }

    /// One-line install record, mirroring `backend_provenance()`'s truthfulness rule.
    #[must_use]
    pub fn install_summary(&self) -> String {
        let (file, sym) = self
            .symbol_provenance()
            .unwrap_or(("<dladdr unavailable>", "?"));
        format!(
            "accelerate seam: f64={} f32={} min_macs={} single_threaded_blas={} checks={} \
             symbol={sym} image={file}",
            self.f64_enabled,
            if self.f32_via_accelerate {
                "cblas_sgemm"
            } else {
                "faer"
            },
            self.min_macs,
            self.single_threaded_blas,
            self.dgemm_report.checks.len(),
        )
    }

    fn faer_gemm_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        // Mirrors `ny_propagate::faer_parallelism::{FaerCpuGemmEngine::gemm_f32,
        // current_par}`: inside a rayon worker force `Par::Seq` (the #4392
        // nested-Rayon rule), otherwise use the configured global parallelism.
        let par = if rayon::current_thread_index().is_some() {
            Par::Seq
        } else {
            faer::get_global_parallelism()
        };
        let a_mat = MatRef::from_row_major_slice(a, m, k);
        let b_mat = MatRef::from_row_major_slice(b, k, n);
        let mut c = Mat::<f32>::zeros(m, n);
        matmul(&mut c, Accum::Replace, a_mat, b_mat, 1.0, par);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        out
    }
}

/// Checked `m*k`, `k*n`, `m*n` with the caller's slice lengths (G1, caller side).
struct Shape {
    lhs: usize,
    rhs: usize,
    out: usize,
    macs: usize,
}

fn shape(m: usize, k: usize, n: usize, site: &'static str) -> Result<Shape> {
    let bad = |what: &str| NyError::InvalidSpec(format!("{site}: {what}"));
    let lhs = m.checked_mul(k).ok_or_else(|| bad("m*k overflows usize"))?;
    let rhs = k.checked_mul(n).ok_or_else(|| bad("k*n overflows usize"))?;
    let out = m.checked_mul(n).ok_or_else(|| bad("m*n overflows usize"))?;
    let macs = lhs.saturating_mul(n);
    Ok(Shape {
        lhs,
        rhs,
        out,
        macs,
    })
}

fn try_zeroed<T: Copy>(len: usize, zero: T, site: &'static str) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: len.saturating_mul(size_of::<T>()),
            budget_bytes: usize::MAX,
            site,
        })?;
    out.resize(len, zero);
    Ok(out)
}

impl GemmEngine for AccelerateGemmEngine {
    fn backend_provenance(&self) -> &'static str {
        "accelerate-veclib-cpu"
    }

    /// IEEE round-to-nearest f32 GEMM.
    ///
    /// Routes to `cblas_sgemm` ONLY when the f32 gate is armed AND the sgemm
    /// conformance probe accepted; otherwise it is the incumbent faer blocked
    /// kernel, so installing this engine for its f64 seam alone leaves f32
    /// arithmetic exactly as it is today.
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        const SITE: &str = "AccelerateGemmEngine::gemm_f32";
        let s = shape(m, k, n, SITE)?;
        if a.len() != s.lhs || b.len() != s.rhs {
            return Err(NyError::InvalidSpec(format!(
                "{SITE}: a.len()={} (want {}) b.len()={} (want {})",
                a.len(),
                s.lhs,
                b.len(),
                s.rhs
            )));
        }
        if m == 0 || k == 0 || n == 0 {
            return try_zeroed(s.out, 0.0f32, SITE);
        }
        if self.f32_via_accelerate && s.macs >= self.min_macs {
            apply_thread_blas_policy(self.single_threaded_blas);
            let mut out = try_zeroed(s.out, 0.0f32, SITE)?;
            if ffi::sgemm_row_major(m, k, n, a, b, &mut out) {
                CALLS_F32_ACCELERATE.fetch_add(1, Ordering::Relaxed);
                return Ok(out);
            }
            DECLINED_SHAPE.fetch_add(1, Ordering::Relaxed);
        }
        CALLS_F32_FAER.fetch_add(1, Ordering::Relaxed);
        Ok(Self::faer_gemm_f32(m, k, n, a, b))
    }

    /// SOUND IEEE-f64 GEMM through `cblas_dgemm`.
    ///
    /// Every refusal path returns a TYPED error and never a value, so every
    /// caller falls back to its existing faer/CPU route unchanged.
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        const SITE: &str = "AccelerateGemmEngine::gemm_f64";
        if !self.f64_enabled {
            return Err(NyError::UnsupportedOp(
                "accelerate f64 seam is not armed (NY_ACCELERATE_F64)".into(),
            ));
        }
        let s = shape(m, k, n, SITE)?;
        if a.len() != s.lhs || b.len() != s.rhs {
            return Err(NyError::InvalidSpec(format!(
                "{SITE}: a.len()={} (want {}) b.len()={} (want {})",
                a.len(),
                s.lhs,
                b.len(),
                s.rhs
            )));
        }
        if m == 0 || k == 0 || n == 0 {
            return try_zeroed(s.out, 0.0f64, SITE);
        }
        if s.macs < self.min_macs {
            DECLINED_SMALL.fetch_add(1, Ordering::Relaxed);
            return Err(NyError::UnsupportedOp(format!(
                "{SITE}: {}x{}x{} below the offload floor ({} MACs)",
                m, k, n, self.min_macs
            )));
        }
        match underflow_domain_admissible(a, b) {
            Ok(()) => {}
            Err(Refusal::NonFinite) => {
                DECLINED_NON_FINITE.fetch_add(1, Ordering::Relaxed);
                return Err(NyError::UnsupportedOp(format!(
                    "{SITE}: non-finite operand; declining to the incumbent engine"
                )));
            }
            Err(Refusal::UnderflowDomain) => {
                DECLINED_UNDERFLOW_DOMAIN.fetch_add(1, Ordering::Relaxed);
                return Err(NyError::UnsupportedOp(format!(
                    "{SITE}: operand magnitudes admit a subnormal product (G2); declining"
                )));
            }
        }
        apply_thread_blas_policy(self.single_threaded_blas);
        let mut out = try_zeroed(s.out, 0.0f64, SITE)?;
        if !ffi::dgemm_row_major(m, k, n, a, b, &mut out) {
            DECLINED_SHAPE.fetch_add(1, Ordering::Relaxed);
            return Err(NyError::UnsupportedOp(format!(
                "{SITE}: {m}x{k}x{n} outside the LP64 CBLAS ABI (G1); declining"
            )));
        }
        if CALLS_F64.fetch_add(1, Ordering::Relaxed) == 0 {
            // Provenance, once: "installed" and "actually used" are different
            // facts, and a run log that cannot distinguish them is a run log
            // that cannot be banked. The caller-side dispatch floors
            // (`SOUND_F64_GEMM_MIN_MACS = 1 << 24`) can keep an installed engine
            // completely dark on small models.
            tracing::info!(m, k, n, "Accelerate sound f64 seam took its FIRST dispatch");
        }
        Ok(out)
    }
}

/// Process-global, lazily probed Accelerate engine.
///
/// The probe (~0.4 ms) runs at most once per process, on the first consult.
/// `None` is cached, so a refused host pays nothing after the first attempt.
#[must_use]
pub fn shared_engine() -> Option<Arc<AccelerateGemmEngine>> {
    static ENGINE: OnceLock<Option<Arc<AccelerateGemmEngine>>> = OnceLock::new();
    ENGINE
        .get_or_init(|| AccelerateGemmEngine::new().map(Arc::new))
        .clone()
}

/// What an install attempt did — for the CLI's log line and for tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InstallOutcome {
    /// Kill switch set.
    Disabled,
    /// Neither gate armed (the default).
    NotArmed,
    /// Probe refused the host's BLAS.
    ProbeRefused(Vec<&'static str>),
    /// Installed; the string is [`AccelerateGemmEngine::install_summary`].
    Installed {
        /// Sound f64 seam armed.
        f64_seam: bool,
        /// f32 free-rider armed AND probe-accepted.
        f32_seam: bool,
        /// Human-readable provenance record.
        summary: String,
    },
}

/// Resolve the engine for installation, reporting exactly what happened.
///
/// This does NOT touch `ny-propagate` (this crate must stay a leaf, like
/// `ny-cuda`); the CLI takes the returned `Arc` and calls
/// `sound_f64_gemm::set_sound_f64_gemm_factory` /
/// `fast_f32_gemm::set_fast_f32_gemm_factory` with it.
#[must_use]
pub fn resolve_for_install() -> (Option<Arc<AccelerateGemmEngine>>, InstallOutcome) {
    if kill_switch_engaged() {
        return (None, InstallOutcome::Disabled);
    }
    if !f64_seam_armed() && !f32_seam_armed() {
        return (None, InstallOutcome::NotArmed);
    }
    match shared_engine() {
        Some(engine) => {
            let outcome = InstallOutcome::Installed {
                f64_seam: engine.f64_enabled,
                f32_seam: engine.f32_via_accelerate,
                summary: engine.install_summary(),
            };
            (Some(engine), outcome)
        }
        None => (
            None,
            InstallOutcome::ProbeRefused(probe::dgemm_conformance_probe().failures()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_log2_abs_matches_definition() {
        assert_eq!(floor_log2_abs(1.0), Some(0));
        assert_eq!(floor_log2_abs(-3.0), Some(1));
        assert_eq!(floor_log2_abs(f64::MIN_POSITIVE), Some(-1022));
        // smallest subnormal
        assert_eq!(floor_log2_abs(f64::from_bits(1)), Some(-1074));
        assert_eq!(floor_log2_abs(f64::from_bits(3)), Some(-1073));
        assert_eq!(floor_log2_abs(0.0), None);
        assert_eq!(floor_log2_abs(f64::NAN), None);
        assert_eq!(floor_log2_abs(f64::INFINITY), None);
    }

    #[test]
    fn g2_admits_f32_widened_operands_and_refuses_tiny_ones() {
        // The CROWN seam's domain: f32 widenings, |x| >= 2^-149.
        let a = [f64::from(f32::from_bits(1)), 0.0, 1.0];
        let b = [f64::from(f32::from_bits(1)), 3.5];
        assert!(underflow_domain_admissible(&a, &b).is_ok());

        // Genuine f64 operands that would produce a subnormal product.
        let tiny = [1e-300f64, 2.0];
        assert!(matches!(
            underflow_domain_admissible(&tiny, &tiny),
            Err(Refusal::UnderflowDomain)
        ));

        // Non-finite refuses.
        assert!(matches!(
            underflow_domain_admissible(&[f64::NAN], &[1.0]),
            Err(Refusal::NonFinite)
        ));
        assert!(matches!(
            underflow_domain_admissible(&[1.0], &[f64::INFINITY]),
            Err(Refusal::NonFinite)
        ));

        // All-zero is admissible: the product is exactly zero.
        assert!(underflow_domain_admissible(&[0.0, 0.0], &[1e-300]).is_ok());
    }

    #[test]
    fn g2_boundary_is_the_exact_exponent_sum() {
        // 2^-484 * 2^-485 = 2^-969  -> admissible (exactly at the floor).
        let a = [f64::from_bits(((-484i32 + 1023) as u64) << 52)];
        let b = [f64::from_bits(((-485i32 + 1023) as u64) << 52)];
        assert!(underflow_domain_admissible(&a, &b).is_ok());
        // one binade lower -> refused
        let b2 = [f64::from_bits(((-486i32 + 1023) as u64) << 52)];
        assert!(matches!(
            underflow_domain_admissible(&a, &b2),
            Err(Refusal::UnderflowDomain)
        ));
    }
}
