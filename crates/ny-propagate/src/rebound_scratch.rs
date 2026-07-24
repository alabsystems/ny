// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Thread-local scratch-buffer pool for the CROWN backward rebound
//! (#rebound-scratch).
//!
//! The per-domain input-split rebound (the ~90% of the relational / iso
//! difference-net BaB compute) re-allocates the same shape of temporary
//! matrices on EVERY layer of EVERY domain. This module recycles the four faer
//! f64 operands and two products inside
//! [`crate::layers::linear::crown_single::aw_f64_with_abssum`]. The two output
//! `Array2`s are still allocated normally. On the iso ACAS diff-net regime
//! (~600 neurons, 5-D input split, thousands of sub-boxes), profiling attributes
//! a material part of rebound time to this temporary-buffer churn.
//!
//! # Soundness — BIT IDENTICAL
//!
//! A recycled buffer is handed back to the caller in one of two states, each
//! byte-for-byte identical to a fresh allocation:
//!  * [`take_f64`] resizes the buffer with zeroes, so the caller sees exactly
//!    the `vec![0.0; len]` bytes it asked for; or
//!  * the faer-view helpers overwrite EVERY element before the buffer is read
//!    (the `from_fn`-style column-major fill, then `matmul` with
//!    `Accum::Replace`, which writes the entire product regardless of the
//!    destination's prior contents).
//!
//! The arithmetic is unchanged: the same operands feed the same faer `matmul`
//! under the same [`crate::faer_parallelism::current_par`] policy (identical
//! reduction order → identical f64 rounding), and the ndarray temporaries are
//! filled by the same expressions. So the stored coefficient matrices, the
//! certified-error matrices, the biases, and every verdict-feeding byte are
//! IDENTICAL whether the pool is on or off. That equality is the moat: the
//! parity tests assert raw-f32-bit equality of every output, gate ON vs OFF.
//!
//! # Gate
//!
//! [`enabled`] reads `NY_REBOUND_SCRATCH` once (cached in an atomic). Once the
//! parity moat is green the pool is DEFAULT-ON; opt OUT with
//! `NY_REBOUND_SCRATCH=0` (`false`) for the A/B reference. When OFF, every
//! `take_*` allocates fresh and every `recycle_*` drops — i.e. the historical
//! behaviour, byte-for-byte. Tests use a thread-local override that never
//! mutates the process-global production gate.

use std::cell::RefCell;
use std::mem::size_of;
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(test)]
use std::cell::Cell;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

/// Tri-state gate cache: `0` = off, `1` = on, `2` = uninitialised.
static GATE: AtomicU8 = AtomicU8::new(2);

#[cfg(test)]
thread_local! {
    /// Per-test-thread gate override; `None` preserves the production cache.
    static TEST_GATE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Proof that an end-to-end parity case actually reached this path.
    static POOLED_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Whether the rebound scratch pool is active (cached read of
/// `NY_REBOUND_SCRATCH`). DEFAULT-ON once the parity moat proved raw-f32-bit
/// equality gate ON vs OFF on both the iso-shaped and lsnc-shaped batches; opt
/// OUT with `NY_REBOUND_SCRATCH=0`.
#[inline]
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_GATE_OVERRIDE.with(Cell::get) {
        return on;
    }

    match GATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_REBOUND_SCRATCH").ok().as_deref(),
                Some("0") | Some("false")
            );
            GATE.store(u8::from(on), Ordering::Relaxed);
            on
        }
    }
}

#[cfg(test)]
#[must_use]
pub(crate) struct TestGateOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl TestGateOverride {
    /// Override the gate only on the current test thread.
    pub(crate) fn new(on: bool) -> Self {
        let previous = TEST_GATE_OVERRIDE.with(|override_cell| override_cell.replace(Some(on)));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for TestGateOverride {
    fn drop(&mut self) {
        TEST_GATE_OVERRIDE.with(|override_cell| override_cell.set(self.previous));
    }
}

#[cfg(test)]
pub(crate) fn pooled_call_count_for_test() -> usize {
    POOLED_CALLS.with(Cell::get)
}

/// Bound both the number and retained capacity of each thread-local pool.
/// Six buffers are live in the current operation; two spare slots avoid churn
/// across changing shapes. Every participating worker retains at most 16 MiB
/// (so 20 workers have a 320 MiB aggregate ceiling), and a pathological wider
/// buffer is dropped rather than retained.
const MAX_POOLED: usize = 8;
const MAX_POOLED_BYTES_PER_THREAD: usize = 16 * 1024 * 1024;

thread_local! {
    static F64_POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a zero-initialised `Vec<f64>` of exactly `len` elements. Reuses a
/// recycled buffer's capacity when available; otherwise allocates. The bytes
/// are IDENTICAL to `vec![0.0; len]` (`clear` + `resize(len, 0.0)` writes `0.0`
/// to every live slot).
#[inline]
pub(crate) fn take_f64(len: usize) -> Vec<f64> {
    if enabled() {
        if let Some(mut v) = F64_POOL.with(|p| p.borrow_mut().pop()) {
            v.clear();
            v.resize(len, 0.0);
            return v;
        }
    }
    vec![0.0f64; len]
}

/// Return an f64 buffer to the per-thread free list (dropped when the gate is
/// off or the list is full). The buffer keeps its capacity for reuse.
#[inline]
pub(crate) fn recycle_f64(v: Vec<f64>) {
    if enabled() {
        F64_POOL.with(|p| {
            let mut p = p.borrow_mut();
            let pooled_bytes = p.iter().fold(0usize, |total, buffer| {
                total.saturating_add(buffer.capacity().saturating_mul(size_of::<f64>()))
            });
            if pool_has_capacity_for(p.len(), pooled_bytes, v.capacity()) {
                p.push(v);
            }
        });
    }
}

#[inline]
fn pool_has_capacity_for(count: usize, pooled_bytes: usize, candidate_capacity: usize) -> bool {
    let candidate_bytes = candidate_capacity.saturating_mul(size_of::<f64>());
    count < MAX_POOLED
        && candidate_bytes <= MAX_POOLED_BYTES_PER_THREAD.saturating_sub(pooled_bytes)
}

/// Sound, pooled f64 twin of [`crate::faer_parallelism::mat_mul_f64`] plus its
/// abs-sum companion, computing BOTH `A·W` and `|A|·|W|` from pooled operands.
///
/// Given `a(i,l)` and `w(l,j)` readers (both indexing the logical `(row, col)`
/// element), returns `(aw, s)` as column-major `Vec<f64>` of length `m·p`
/// where `aw[j*m+i] = Σ_l a(i,l)·w(l,j)` and `s[j*m+i] = Σ_l |a(i,l)|·|w(l,j)|`.
/// The four operand buffers come from / return to the thread-local pool; the
/// two product buffers are taken from the pool and returned to the CALLER (who
/// reads then recycles them). The products are written by faer `matmul`
/// (`Accum::Replace`) under `par`, so they are bit-identical to the owned-`Mat`
/// path (same operands, same reduction order).
#[allow(clippy::type_complexity)]
pub(crate) fn pooled_aw_and_abssum<FA, FW>(
    m: usize,
    k: usize,
    p: usize,
    par: Par,
    a: FA,
    w: FW,
) -> (Vec<f64>, Vec<f64>)
where
    FA: Fn(usize, usize) -> f64,
    FW: Fn(usize, usize) -> f64,
{
    #[cfg(test)]
    POOLED_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));

    // Column-major operand fills (element (i,j) at index j*rows + i), matching
    // `Mat::<f64>::from_fn(rows, cols, |i, j| ..)`'s storage exactly.
    let mut af = take_f64(m * k);
    let mut aabs = take_f64(m * k);
    for j in 0..k {
        let base = j * m;
        for i in 0..m {
            let v = a(i, j);
            af[base + i] = v;
            aabs[base + i] = v.abs();
        }
    }
    let mut wf = take_f64(k * p);
    let mut wabs = take_f64(k * p);
    for j in 0..p {
        let base = j * k;
        for i in 0..k {
            let v = w(i, j);
            wf[base + i] = v;
            wabs[base + i] = v.abs();
        }
    }

    let mut aw = take_f64(m * p);
    let mut s = take_f64(m * p);
    {
        let a_ref = MatRef::from_column_major_slice(&af[..m * k], m, k);
        let w_ref = MatRef::from_column_major_slice(&wf[..k * p], k, p);
        let aw_dst = MatMut::from_column_major_slice_mut(&mut aw[..m * p], m, p);
        matmul(aw_dst, Accum::Replace, a_ref, w_ref, 1.0, par);
    }
    {
        let aabs_ref = MatRef::from_column_major_slice(&aabs[..m * k], m, k);
        let wabs_ref = MatRef::from_column_major_slice(&wabs[..k * p], k, p);
        let s_dst = MatMut::from_column_major_slice_mut(&mut s[..m * p], m, p);
        matmul(s_dst, Accum::Replace, aabs_ref, wabs_ref, 1.0, par);
    }

    recycle_f64(af);
    recycle_f64(aabs);
    recycle_f64(wf);
    recycle_f64(wabs);
    (aw, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn reset() {
        F64_POOL.with(|p| p.borrow_mut().clear());
    }

    #[test]
    fn take_returns_zeroed_len() {
        let _gate = TestGateOverride::new(true);
        reset();
        let a = take_f64(7);
        assert_eq!(a.len(), 7);
        assert!(a.iter().all(|&x| x == 0.0));
        recycle_f64(a);
        // Reused buffer is still exactly len and zeroed.
        let b = take_f64(3);
        assert_eq!(b.len(), 3);
        assert!(b.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pool_is_bounded_by_count_and_bytes() {
        assert!(pool_has_capacity_for(0, 0, 1));
        assert!(!pool_has_capacity_for(MAX_POOLED, 0, 1));
        assert!(!pool_has_capacity_for(
            0,
            0,
            MAX_POOLED_BYTES_PER_THREAD / size_of::<f64>() + 1
        ));
        assert!(!pool_has_capacity_for(1, MAX_POOLED_BYTES_PER_THREAD, 1));
    }

    #[test]
    fn pooled_aw_matches_owned_faer_bitwise() {
        // Small block; compare pooled column-major products against the
        // owned-`Mat` `matmul` the production path used before.
        let m = 5;
        let k = 6;
        let p = 4;
        let a_val = |i: usize, j: usize| ((i * 3 + j) as f64 * 0.5 - 4.0) / 7.0;
        let w_val = |i: usize, j: usize| ((i + j * 2) as f64 - 3.0) * 0.25;

        let a_mat = Mat::<f64>::from_fn(m, k, &a_val);
        let w_mat = Mat::<f64>::from_fn(k, p, &w_val);
        let a_abs = Mat::<f64>::from_fn(m, k, |i, j| a_val(i, j).abs());
        let w_abs = Mat::<f64>::from_fn(k, p, |i, j| w_val(i, j).abs());
        for par in [Par::Seq, crate::faer_parallelism::current_par()] {
            let mut aw_ref = Mat::<f64>::zeros(m, p);
            matmul(&mut aw_ref, Accum::Replace, &a_mat, &w_mat, 1.0, par);
            let mut s_ref = Mat::<f64>::zeros(m, p);
            matmul(&mut s_ref, Accum::Replace, &a_abs, &w_abs, 1.0, par);

            for &on in &[false, true] {
                let _gate = TestGateOverride::new(on);
                reset();
                let (aw, s) = pooled_aw_and_abssum(m, k, p, par, a_val, w_val);
                for j in 0..p {
                    for i in 0..m {
                        assert_eq!(aw[j * m + i].to_bits(), aw_ref[(i, j)].to_bits());
                        assert_eq!(s[j * m + i].to_bits(), s_ref[(i, j)].to_bits());
                    }
                }
            }
        }
    }
}
