// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-collector-throughput / #vit-batched-bilinear: process-wide arming and
//! engagement census for the CROWN-collector throughput kernels.
//!
//! # What is armed here
//!
//! | typed key (`bab.*`)            | `BetaCrownConfig` field          | kernel |
//! |--------------------------------|----------------------------------|--------|
//! | `bilinear_rowpar_kernel`       | `bilinear_rowpar_kernel`         | row-parallel, plane-hoisted McCormick backward for var×var `MatMul` / `BilinearCrown` (both the α-free and the α arm) |
//! | `linear_position_stack`        | `linear_position_stack`          | one certified f64 `A·W` over ALL token positions of a sequence Linear instead of one per position |
//! | `reciprocal_unbounded_upper`   | `reciprocal_unbounded_upper`     | a sound `1/x` relaxation on `[l, +inf)` and the non-aborting `UnsupportedOp` refusal class |
//! | `bilinear_gemm_kernel`         | `bilinear_gemm_kernel`           | #vit-batched-bilinear (ENGINE_PROGRAM V1): the batched, GEMM-shaped McCormick backward for var×var `MatMul` / `BilinearCrown` (both arms); tried before `bilinear_rowpar_kernel` |
//!
//! # Why a process-wide latch rather than a threaded parameter
//!
//! The three kernels sit at the bottom of every backward walk (collector,
//! alpha-post pass, margin ascent, BaB child walks), reached through the shared
//! backward dispatcher, which carries no configuration. The latch is written
//! from the typed `BetaCrownConfig` fields when a `BetaCrownVerifier` is
//! constructed (`beta_crown/engine/core.rs`), before that verifier runs
//! anything. It is STICKY-ON: a config with a key unset never disarms a key a
//! previous config armed, so a sub-verifier built from a partial config cannot
//! silently turn a lane dark mid-run. One `ny vnncomp v1` process verifies one
//! instance, so there is nothing to leak across runs.
//!
//! # Dark by default
//!
//! Every flag starts `false` and is only ever raised by an explicit `true` in a
//! typed preset key. No shipped yaml sets any of the four keys (pinned by
//! `no_shipped_preset_arms_vit_collector_kernel_keys`), so on every shipped
//! configuration every kernel runs its historical code path byte-for-byte.
//!
//! # Soundness of each kernel (details at each kernel)
//!
//! * `bilinear_rowpar_kernel`: BIT-IDENTICAL to the serial nest. Rows are
//!   independent and every row performs the SAME floating-point operation
//!   sequence in the SAME order; only the interleaving across rows and the
//!   number of times a (pure) McCormick plane is recomputed change.
//! * `bilinear_gemm_kernel`: BIT-IDENTICAL to the serial nest by the same
//!   argument, per ACCUMULATOR rather than per row: every coefficient cell,
//!   certified-`S` cell and constant fold receives the nest's updates in the
//!   nest's order; the planes are computed once per call
//!   (`binary_ops/matmul/crown_gemm.rs`). When both bilinear keys are armed
//!   the GEMM-shaped kernel is tried first, then the row-parallel one, then
//!   the nest; all three publish the same bits.
//! * `linear_position_stack`: BIT-IDENTICAL on the CPU reducer (every output
//!   row keeps its full `k` contraction inside one GEMM call and faer's
//!   per-element reduction does not depend on the row count, the property
//!   already pinned for the chunked and domain-batched twins). Once the
//!   stacked product crosses `SOUND_F64_GEMM_MIN_MACS` it may run on the
//!   process-global sound f64 engine, which is sound through the same
//!   summation-order-independent `γ_n·S` certificate but NOT bit-identical.
//! * `reciprocal_unbounded_upper`: a strictly sound relaxation that only
//!   replaces a refusal, and a refusal class every collector already treats as
//!   "fall back to the certified IBP box for this target".
//!
//! # Observables
//!
//! * `[collector-kernels] armed bilinear_rowpar=<0|1> linear_pos_stack=<0|1>
//!   reciprocal_unbounded=<0|1> bilinear_gemm=<0|1> source=config` — printed
//!   when the latch changes (at most four times per process).
//! * `[bilinear-kernel] … form=rowpar …`, `[bilinear-kernel] … form=gemm …`,
//!   `[linear-pos-stack] …`, `[reciprocal-unbounded] …`
//!   — cumulative engagement census, printed at every retained CROWN-IBP
//!   collection summary and alpha-post pass while the matching key is armed
//!   (`emit_census`). A key that is armed and shows `calls=0` never engaged:
//!   a null read from such a run is VOID, not negative.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

static BILINEAR_ROWPAR: AtomicBool = AtomicBool::new(false);
static LINEAR_POSITION_STACK: AtomicBool = AtomicBool::new(false);
static RECIPROCAL_UNBOUNDED: AtomicBool = AtomicBool::new(false);
static BILINEAR_GEMM: AtomicBool = AtomicBool::new(false);

// Engagement counters (cumulative, process-wide). Relaxed ordering: these are
// telemetry only and never decide anything.
static BILINEAR_CALLS: AtomicU64 = AtomicU64::new(0);
static BILINEAR_ALPHA_CALLS: AtomicU64 = AtomicU64::new(0);
static BILINEAR_ROWS: AtomicU64 = AtomicU64::new(0);
static BILINEAR_NANOS: AtomicU64 = AtomicU64::new(0);
static BILINEAR_DECLINED: AtomicU64 = AtomicU64::new(0);
static BILINEAR_PARALLEL_CALLS: AtomicU64 = AtomicU64::new(0);

static GEMM_CALLS: AtomicU64 = AtomicU64::new(0);
static GEMM_ALPHA_CALLS: AtomicU64 = AtomicU64::new(0);
static GEMM_ROWS: AtomicU64 = AtomicU64::new(0);
static GEMM_NANOS: AtomicU64 = AtomicU64::new(0);
static GEMM_TABLE_NANOS: AtomicU64 = AtomicU64::new(0);
static GEMM_DECLINED: AtomicU64 = AtomicU64::new(0);
static GEMM_PARALLEL_CALLS: AtomicU64 = AtomicU64::new(0);

// Test-only engagement probe for the GEMM-shaped kernel, per CALLING thread.
// The kernel records on the thread that called it, after its parallel section
// has joined, so a test can prove that a ROUTED entry actually took the
// GEMM-shaped kernel — every fallback is bit-identical, so bit-identity alone
// cannot tell "engaged" from "fell through" — without racing the process-wide
// counters that concurrently running tests also bump.
#[cfg(test)]
thread_local! {
    static GEMM_CALLS_THIS_THREAD: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only: GEMM-shaped completions recorded on the calling thread.
#[cfg(test)]
pub(crate) fn gemm_calls_this_thread() -> u64 {
    GEMM_CALLS_THIS_THREAD.with(std::cell::Cell::get)
}

static LINEAR_STACK_CALLS: AtomicU64 = AtomicU64::new(0);
static LINEAR_STACK_ROWS: AtomicU64 = AtomicU64::new(0);
static LINEAR_STACK_MAX_POSITIONS: AtomicU64 = AtomicU64::new(0);
static LINEAR_STACK_ROUTE_CPU: AtomicU64 = AtomicU64::new(0);
static LINEAR_STACK_ROUTE_ENGINE: AtomicU64 = AtomicU64::new(0);
static LINEAR_STACK_DECLINED: AtomicU64 = AtomicU64::new(0);

static RECIPROCAL_CALLS: AtomicU64 = AtomicU64::new(0);
static RECIPROCAL_ELEMS: AtomicU64 = AtomicU64::new(0);
static RECIPROCAL_REFUSALS: AtomicU64 = AtomicU64::new(0);

/// The typed arming of the kernels, as read from one `BetaCrownConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectorKernelArming {
    /// `bab.bilinear_rowpar_kernel`.
    pub bilinear_rowpar: bool,
    /// `bab.linear_position_stack`.
    pub linear_pos_stack: bool,
    /// `bab.reciprocal_unbounded_upper`.
    pub reciprocal_unbounded: bool,
    /// `bab.bilinear_gemm_kernel` (#vit-batched-bilinear).
    pub bilinear_gemm: bool,
}

impl CollectorKernelArming {
    /// Every kernel on its historical path. What `Default` returns, and what
    /// `BetaCrownConfig::default()` maps to.
    pub const DARK: Self = Self {
        bilinear_rowpar: false,
        linear_pos_stack: false,
        reciprocal_unbounded: false,
        bilinear_gemm: false,
    };

    /// Whether any kernel is requested.
    #[must_use]
    pub fn any(self) -> bool {
        self.bilinear_rowpar
            || self.linear_pos_stack
            || self.reciprocal_unbounded
            || self.bilinear_gemm
    }
}

impl Default for CollectorKernelArming {
    fn default() -> Self {
        Self::DARK
    }
}

/// Raise (never lower) the process-wide latch for every kernel `arming` asks
/// for. A DARK arming is a no-op that touches nothing and prints nothing.
///
/// Prints `[collector-kernels] armed …` whenever this call newly armed at
/// least one kernel, so the log records WHEN arming happened relative to the
/// first collection.
pub fn arm(arming: CollectorKernelArming) {
    if !arming.any() {
        return;
    }
    let mut changed = false;
    if arming.bilinear_rowpar {
        changed |= !BILINEAR_ROWPAR.swap(true, Ordering::Relaxed);
    }
    if arming.linear_pos_stack {
        changed |= !LINEAR_POSITION_STACK.swap(true, Ordering::Relaxed);
    }
    if arming.reciprocal_unbounded {
        changed |= !RECIPROCAL_UNBOUNDED.swap(true, Ordering::Relaxed);
    }
    if arming.bilinear_gemm {
        changed |= !BILINEAR_GEMM.swap(true, Ordering::Relaxed);
    }
    if changed {
        eprintln!(
            "[collector-kernels] armed bilinear_rowpar={} linear_pos_stack={} \
             reciprocal_unbounded={} bilinear_gemm={} source=config",
            u8::from(bilinear_rowpar_armed()),
            u8::from(linear_position_stack_armed()),
            u8::from(reciprocal_unbounded_armed()),
            u8::from(bilinear_gemm_armed()),
        );
    }
}

/// `bab.bilinear_rowpar_kernel` is armed in this process.
#[inline]
#[must_use]
pub(crate) fn bilinear_rowpar_armed() -> bool {
    BILINEAR_ROWPAR.load(Ordering::Relaxed)
}

/// `bab.bilinear_gemm_kernel` is armed in this process.
#[inline]
#[must_use]
pub(crate) fn bilinear_gemm_armed() -> bool {
    BILINEAR_GEMM.load(Ordering::Relaxed)
}

/// Which McCormick backward kernels one var×var `MatMul` / `BilinearCrown`
/// call may try, in order: the GEMM-shaped kernel (`gemm`), the row-parallel
/// kernel (`rowpar`), then always the serial nest. All three are
/// BIT-IDENTICAL, so the choice changes wall time only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct BilinearKernelChoice {
    /// Try the #vit-batched-bilinear GEMM-shaped kernel first.
    pub(crate) gemm: bool,
    /// Then try the #vit-collector-throughput row-parallel kernel.
    pub(crate) rowpar: bool,
}

impl BilinearKernelChoice {
    /// The historical serial nest only (the tests' reference).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const SERIAL: Self = Self {
        gemm: false,
        rowpar: false,
    };

    /// The process-wide typed arming (`bab.bilinear_gemm_kernel`,
    /// `bab.bilinear_rowpar_kernel`): what every production entry passes.
    #[inline]
    #[must_use]
    pub(crate) fn armed() -> Self {
        Self {
            gemm: bilinear_gemm_armed(),
            rowpar: bilinear_rowpar_armed(),
        }
    }
}

/// `bab.linear_position_stack` is armed in this process.
#[inline]
#[must_use]
pub(crate) fn linear_position_stack_armed() -> bool {
    LINEAR_POSITION_STACK.load(Ordering::Relaxed)
}

/// `bab.reciprocal_unbounded_upper` is armed in this process.
#[inline]
#[must_use]
pub(crate) fn reciprocal_unbounded_armed() -> bool {
    RECIPROCAL_UNBOUNDED.load(Ordering::Relaxed)
}

fn saturating_nanos(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// One row-parallel McCormick backward completed (`alpha` = the interpolated
/// `BilinearCrown` arm, else the midpoint `MatMul` arm).
pub(crate) fn record_bilinear_rowpar(alpha: bool, rows: usize, parallel: bool, elapsed: Duration) {
    BILINEAR_CALLS.fetch_add(1, Ordering::Relaxed);
    if alpha {
        BILINEAR_ALPHA_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if parallel {
        BILINEAR_PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    BILINEAR_ROWS.fetch_add(to_u64(rows), Ordering::Relaxed);
    BILINEAR_NANOS.fetch_add(saturating_nanos(elapsed), Ordering::Relaxed);
}

/// The armed row-parallel kernel declined a shape and ran the serial nest.
pub(crate) fn record_bilinear_declined() {
    BILINEAR_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// One GEMM-shaped McCormick backward completed (`alpha` = the interpolated
/// `BilinearCrown` arm, else the midpoint `MatMul` arm); `table` is the part of
/// `elapsed` spent building the plane table.
pub(crate) fn record_bilinear_gemm(
    alpha: bool,
    rows: usize,
    parallel: bool,
    elapsed: Duration,
    table: Duration,
) {
    #[cfg(test)]
    GEMM_CALLS_THIS_THREAD.with(|calls| calls.set(calls.get() + 1));
    GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
    if alpha {
        GEMM_ALPHA_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if parallel {
        GEMM_PARALLEL_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    GEMM_ROWS.fetch_add(to_u64(rows), Ordering::Relaxed);
    GEMM_NANOS.fetch_add(saturating_nanos(elapsed), Ordering::Relaxed);
    GEMM_TABLE_NANOS.fetch_add(saturating_nanos(table), Ordering::Relaxed);
}

/// The armed GEMM-shaped kernel declined a shape; the next choice (row-parallel
/// kernel if armed, else the serial nest) ran.
pub(crate) fn record_bilinear_gemm_declined() {
    GEMM_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// One position-stacked Linear backward completed.
pub(crate) fn record_linear_stack(positions: usize, stacked_rows: usize, engine_route: bool) {
    LINEAR_STACK_CALLS.fetch_add(1, Ordering::Relaxed);
    LINEAR_STACK_ROWS.fetch_add(to_u64(stacked_rows), Ordering::Relaxed);
    LINEAR_STACK_MAX_POSITIONS.fetch_max(to_u64(positions), Ordering::Relaxed);
    if engine_route {
        LINEAR_STACK_ROUTE_ENGINE.fetch_add(1, Ordering::Relaxed);
    } else {
        LINEAR_STACK_ROUTE_CPU.fetch_add(1, Ordering::Relaxed);
    }
}

/// The armed position stack declined (size cap) and ran the per-position loop.
pub(crate) fn record_linear_stack_declined() {
    LINEAR_STACK_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// One armed Reciprocal backward relaxed `elems` elements on `[l, +inf)`.
pub(crate) fn record_reciprocal_unbounded(elems: usize) {
    RECIPROCAL_CALLS.fetch_add(1, Ordering::Relaxed);
    RECIPROCAL_ELEMS.fetch_add(to_u64(elems), Ordering::Relaxed);
}

/// One armed Reciprocal backward refused with the non-aborting class.
pub(crate) fn record_reciprocal_refusal() {
    RECIPROCAL_REFUSALS.fetch_add(1, Ordering::Relaxed);
}

/// Print the cumulative engagement census for every ARMED kernel, tagged with
/// the caller's `stage`. Prints nothing when every kernel is dark, so the
/// shipped configurations' stderr is unchanged.
///
/// Callers rate-limit (they print only at retained occurrences).
pub(crate) fn emit_census(stage: &str) {
    if bilinear_rowpar_armed() {
        eprintln!(
            "[bilinear-kernel] stage={stage} armed=1 form=rowpar calls={} alpha_calls={} \
             parallel_calls={} rows={} secs={:.3} declined={}",
            BILINEAR_CALLS.load(Ordering::Relaxed),
            BILINEAR_ALPHA_CALLS.load(Ordering::Relaxed),
            BILINEAR_PARALLEL_CALLS.load(Ordering::Relaxed),
            BILINEAR_ROWS.load(Ordering::Relaxed),
            Duration::from_nanos(BILINEAR_NANOS.load(Ordering::Relaxed)).as_secs_f64(),
            BILINEAR_DECLINED.load(Ordering::Relaxed),
        );
    }
    if bilinear_gemm_armed() {
        eprintln!(
            "[bilinear-kernel] stage={stage} armed=1 form=gemm calls={} alpha_calls={} \
             parallel_calls={} rows={} secs={:.3} table_secs={:.3} declined={}",
            GEMM_CALLS.load(Ordering::Relaxed),
            GEMM_ALPHA_CALLS.load(Ordering::Relaxed),
            GEMM_PARALLEL_CALLS.load(Ordering::Relaxed),
            GEMM_ROWS.load(Ordering::Relaxed),
            Duration::from_nanos(GEMM_NANOS.load(Ordering::Relaxed)).as_secs_f64(),
            Duration::from_nanos(GEMM_TABLE_NANOS.load(Ordering::Relaxed)).as_secs_f64(),
            GEMM_DECLINED.load(Ordering::Relaxed),
        );
    }
    if linear_position_stack_armed() {
        let engine = LINEAR_STACK_ROUTE_ENGINE.load(Ordering::Relaxed);
        let cpu = LINEAR_STACK_ROUTE_CPU.load(Ordering::Relaxed);
        let route = if engine > 0 {
            "engine"
        } else if cpu > 0 {
            "cpu"
        } else {
            "none"
        };
        eprintln!(
            "[linear-pos-stack] stage={stage} armed=1 calls={} positions={} rows={} \
             route={route} route_cpu={cpu} route_engine={engine} declined={}",
            LINEAR_STACK_CALLS.load(Ordering::Relaxed),
            LINEAR_STACK_MAX_POSITIONS.load(Ordering::Relaxed),
            LINEAR_STACK_ROWS.load(Ordering::Relaxed),
            LINEAR_STACK_DECLINED.load(Ordering::Relaxed),
        );
    }
    if reciprocal_unbounded_armed() {
        eprintln!(
            "[reciprocal-unbounded] stage={stage} armed=1 calls={} elems={} \
             route=const-upper(lower=zero) refusals_unsupported_op={}",
            RECIPROCAL_CALLS.load(Ordering::Relaxed),
            RECIPROCAL_ELEMS.load(Ordering::Relaxed),
            RECIPROCAL_REFUSALS.load(Ordering::Relaxed),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{BilinearKernelChoice, CollectorKernelArming};

    /// DARK BY DEFAULT: the default arming requests nothing, and `DARK` is
    /// exactly the default. (The latch itself is process-global, so this test
    /// deliberately inspects the pure value rather than mutating the latch.)
    #[test]
    fn collector_kernel_arming_default_is_dark() {
        assert_eq!(
            CollectorKernelArming::default(),
            CollectorKernelArming::DARK
        );
        assert!(!CollectorKernelArming::DARK.any());
        assert!(CollectorKernelArming {
            bilinear_rowpar: true,
            ..CollectorKernelArming::DARK
        }
        .any());
        assert!(CollectorKernelArming {
            linear_pos_stack: true,
            ..CollectorKernelArming::DARK
        }
        .any());
        assert!(CollectorKernelArming {
            reciprocal_unbounded: true,
            ..CollectorKernelArming::DARK
        }
        .any());
        const { assert!(!CollectorKernelArming::DARK.bilinear_gemm) };
        assert!(CollectorKernelArming {
            bilinear_gemm: true,
            ..CollectorKernelArming::DARK
        }
        .any());
        // The kernel choice's default is the serial nest, i.e. what an
        // unarmed process passes.
        assert_eq!(
            BilinearKernelChoice::default(),
            BilinearKernelChoice::SERIAL
        );
        const { assert!(!BilinearKernelChoice::SERIAL.gemm && !BilinearKernelChoice::SERIAL.rowpar) };
    }
}
