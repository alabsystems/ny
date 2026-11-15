// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::incoming_error_product_with_poll_quantum;
use super::*;
use crate::amortise::{NoPoll, WeightedPoll, POLL_CHUNK_UNITS};
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLease, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::layers::linear::bias::f32_to_f64_exact;
use faer::Mat;
use ndarray::{s, Array2};
use num_rational::BigRational;
use std::time::Duration;

fn matrix_bits(a: &Array2<f32>) -> Vec<u32> {
    a.iter().map(|v| v.to_bits()).collect()
}
fn weight_bits(w: &Mat<f32>) -> Vec<u32> {
    (0..w.ncols())
        .flat_map(|j| (0..w.nrows()).map(move |i| w[(i, j)].to_bits()))
        .collect()
}
fn fixture(rows: usize, k: usize, cols: usize) -> (Array2<f32>, Mat<f32>) {
    let bits = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800001, 0xbf800000, 0x4f800001, 0x7f800000,
        0xff800000, 0x7fc01234, 0xffc05678,
    ];
    (
        Array2::from_shape_fn((rows, k + 4), |(i, j)| {
            f32::from_bits(bits[(i * 3 + j * 5) % bits.len()])
        }),
        Mat::from_fn(k, cols, |i, j| {
            f32::from_bits(bits[(i * 7 + j * 3 + 1) % bits.len()])
        }),
    )
}

#[test]
fn cooperative_incoming_error_original_signed_weight_matches_legacy_scalar_bits() {
    for (rows, k, cols) in [
        (0, 0, 0),
        (0, 3, 5),
        (3, 0, 5),
        (3, 5, 0),
        (1, 1, 1),
        (3, 50, 50),
        (5, 129, 7),
    ] {
        let (error, weights) = fixture(rows, k, cols);
        let original = matrix_bits(&error);
        let wb = weight_bits(&weights);
        let abs = Mat::from_fn(k, cols, |i, j| weights[(i, j)].abs());
        for quantum in [1, 4096] {
            let expected =
                incoming_error_product_with_poll_quantum(&error, 2, k, &abs, None, quantum)
                    .unwrap();
            let mut actual = Array2::from_elem((rows, cols), -12345.0);
            into_with_work(
                error.view(),
                2,
                k,
                weights.as_ref(),
                actual.view_mut(),
                &mut NoPoll,
            )
            .unwrap();
            assert_eq!(matrix_bits(&actual), matrix_bits(&expected));
        }
        assert_eq!(matrix_bits(&error), original);
        assert_eq!(weight_bits(&weights), wb);
    }
}

#[test]
fn cooperative_incoming_error_encloses_exact_rational_nonnegative_product() {
    let ebits = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800001, 0x47800001, 0x30801234,
    ];
    let wbits = [
        0, 0x80000000, 1, 0x807fffff, 0x00800000, 0xbf800001, 0x47800001, 0xb0801234,
    ];
    for (rows, k, cols) in [(3, 50, 5), (2, 5, 50), (3, 17, 7)] {
        let error = Array2::from_shape_fn((rows, k + 3), |(i, j)| {
            f32::from_bits(ebits[(i * 7 + j * 3) % ebits.len()])
        });
        let weights = Mat::from_fn(k, cols, |i, j| {
            f32::from_bits(wbits[(i * 3 + j * 5 + 1) % wbits.len()])
        });
        let mut actual = Array2::zeros((rows, cols));
        into_with_work(
            error.view(),
            1,
            k,
            weights.as_ref(),
            actual.view_mut(),
            &mut NoPoll,
        )
        .unwrap();
        for i in 0..rows {
            for j in 0..cols {
                let mut exact = BigRational::from_integer(0.into());
                for kk in 0..k {
                    let e = BigRational::from_float(f32_to_f64_exact(error[[i, 1 + kk]])).unwrap();
                    let w = f32::from_bits(weights[(kk, j)].to_bits() & 0x7fffffff);
                    exact += e * BigRational::from_float(f32_to_f64_exact(w)).unwrap();
                }
                assert!(actual[[i, j]].is_finite());
                assert!(
                    BigRational::from_float(f32_to_f64_exact(actual[[i, j]])).unwrap() >= exact
                );
            }
        }
    }
}

#[test]
fn cooperative_incoming_error_zero_dominates_poison_without_poisoning_other_columns() {
    let error =
        Array2::from_shape_vec((1, 5), vec![0.0, f32::NAN, 1.0, -1.0, f32::from_bits(1)]).unwrap();
    let mut weights = Mat::zeros(5, 4);
    weights[(0, 1)] = f32::NEG_INFINITY; // exact zero error dominates poison
    weights[(1, 2)] = -1.0; // NaN error becomes +Inf, nonzero weight propagates it
    weights[(2, 3)] = -2.0;
    let mut output = Array2::from_elem((1, 4), -999.0);
    into_with_work(
        error.view(),
        0,
        5,
        weights.as_ref(),
        output.view_mut(),
        &mut NoPoll,
    )
    .unwrap();
    assert_eq!(output[[0, 0]].to_bits(), 0);
    assert_eq!(output[[0, 1]].to_bits(), 0);
    assert_eq!(output[[0, 2]], f32::INFINITY);
    assert!(output[[0, 3]] >= 2.0 && output[[0, 3]].is_finite());
}

#[test]
fn cooperative_incoming_error_strided_inputs_output_and_offset_preserve_all_gaps() {
    let (rows, k, cols) = (3, 50, 7);
    let error = Array2::from_shape_fn((rows * 2 + 2, (k + 4) * 2 + 2), |(i, j)| {
        (i + j) as f32 * 0.125
    });
    let view = error.slice(s![1..1+rows*2;2,1..1+(k+4)*2;2]);
    let stride = cols + 5;
    let raw: Vec<f32> = (0..k * stride).map(|i| i as f32 * 0.03125 - 10.0).collect();
    let weights = MatRef::from_row_major_slice_with_stride(&raw, k, cols, stride);
    let abs = Mat::from_fn(k, cols, |i, j| weights[(i, j)].abs());
    let expected =
        incoming_error_product_with_poll_quantum(&view.to_owned(), 2, k, &abs, None, 1).unwrap();
    let mut output = Array2::from_elem((rows * 2 + 2, cols * 3 + 3), -12345.0);
    into_with_work(
        view,
        2,
        k,
        weights,
        output.slice_mut(s![1..1+rows*2;2,2..2+cols*3;3]),
        &mut NoPoll,
    )
    .unwrap();
    for i in 0..output.nrows() {
        for j in 0..output.ncols() {
            if i >= 1
                && i < 1 + rows * 2
                && (i - 1) % 2 == 0
                && j >= 2
                && j < 2 + cols * 3
                && (j - 2) % 3 == 0
            {
                assert_eq!(
                    output[[i, j]].to_bits(),
                    expected[[(i - 1) / 2, (j - 2) / 3]].to_bits()
                );
            } else {
                assert_eq!(output[[i, j]], -12345.0);
            }
        }
    }
}

#[test]
fn cooperative_incoming_error_shape_failures_precede_every_output_write() {
    let (error, weights) = fixture(3, 5, 7);
    let original = matrix_bits(&error);
    let wb = weight_bits(&weights);
    for (offset, k, rows, cols) in [
        (8, 5, 3, 7),
        (usize::MAX, 5, 3, 7),
        (0, 4, 3, 7),
        (0, 5, 2, 7),
        (0, 5, 3, 6),
    ] {
        let mut output = Array2::from_elem((rows, cols), -12345.0);
        assert!(matches!(
            into_with_work(
                error.view(),
                offset,
                k,
                weights.as_ref(),
                output.view_mut(),
                &mut NoPoll
            ),
            Err(NyError::ShapeMismatch { .. })
        ));
        assert!(output.iter().all(|v| *v == -12345.0));
    }
    assert_eq!(matrix_bits(&error), original);
    assert_eq!(weight_bits(&weights), wb);
}

#[test]
fn cooperative_incoming_error_prices_zero_contractions_and_every_entry_mac_and_publication() {
    assert_eq!(POLL_CHUNK_UNITS, 4096);
    for (rows, k, cols) in [(0, 50, 7), (3, 0, 7), (3, 50, 0), (3, 50, 7), (1, 1, 1)] {
        let (error, weights) = fixture(rows, k, cols);
        let mut output = Array2::zeros((rows, cols));
        let mut total = 0usize;
        let mut calls = 0;
        let mut work = WeightedPoll::new(|units| {
            assert!(units <= 4096);
            total += units;
            calls += 1;
            Ok(())
        });
        into_with_work(
            error.view(),
            2,
            k,
            weights.as_ref(),
            output.view_mut(),
            &mut work,
        )
        .unwrap();
        assert_eq!(total, 64 + 64 * rows * cols * (k + 2));
        assert!(calls >= 2);
    }
}

struct Boundary {
    calls: usize,
    stop: usize,
}
impl CooperativeWork for Boundary {
    fn checkpoint(&mut self) -> Result<()> {
        self.calls += 1;
        if self.calls == self.stop {
            Err(NyError::DeadlineExceeded("forced incoming boundary".into()))
        } else {
            Ok(())
        }
    }
    fn before(&mut self, units: usize) -> Result<()> {
        assert_eq!(units, 64);
        self.checkpoint()
    }
}

#[test]
fn cooperative_incoming_error_every_refusal_boundary_keeps_sources_and_returns_no_owned_output() {
    for (rows, k, cols) in [(2, 3, 2), (2, 0, 2), (0, 3, 2)] {
        let (error, weights) = fixture(rows, k, cols);
        let original = matrix_bits(&error);
        let wb = weight_bits(&weights);
        let terminal = 3 + rows * cols * (k + 2);
        for stop in 1..=terminal {
            let mut output = Array2::from_elem((rows, cols), -12345.0);
            let mut work = Boundary { calls: 0, stop };
            let result = into_with_work(
                error.view(),
                2,
                k,
                weights.as_ref(),
                output.view_mut(),
                &mut work,
            );
            assert!(matches!(result,Err(e) if e.is_deadline_exceeded()));
            assert_eq!(work.calls, stop);
            if stop <= 2 {
                assert!(output.iter().all(|v| *v == -12345.0));
            }
            assert_eq!(matrix_bits(&error), original);
            assert_eq!(weight_bits(&weights), wb);
        }
    }
}

fn refusal(error: WorkRefusal) -> NyError {
    match error {
        WorkRefusal::Cancelled | WorkRefusal::Deadline => {
            NyError::DeadlineExceeded(format!("original query {error}"))
        }
        _ => NyError::UnsupportedOp(format!("original query {error}")),
    }
}
struct QueryWork<'a> {
    query: &'a WorkBudget,
    phase: WorkLease,
    cancel: &'a CancellationToken,
    calls: usize,
    stop: usize,
}
impl CooperativeWork for QueryWork<'_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.calls += 1;
        if self.calls == self.stop {
            self.cancel.cancel();
        }
        self.query
            .checkpoint_for(LeaseClass::Fallback)
            .map_err(refusal)
    }
    fn before(&mut self, units: usize) -> Result<()> {
        self.checkpoint()?;
        self.phase.reserve_more_work(units as u64).map_err(refusal)
    }
}

#[test]
fn cooperative_incoming_error_original_query_work_limit_and_cancel_cannot_renew_at_seams() {
    let (error, weights) = fixture(1, 3, 2);
    let total = 704;
    for (limit, stop) in (0..=total)
        .step_by(64)
        .map(|n| (n, 0))
        .chain([(total, 1), (total, 13)])
    {
        let cancel = CancellationToken::default();
        let memory = MemoryAuthority::new(4096);
        let query = WorkBudget::new_in(
            &memory,
            &cancel,
            WorkLimits {
                deadline: None,
                scratch_bytes: 4096,
                persistent_bytes: 0,
                max_work: limit as u64,
                max_proposals: 0,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap();
        let phase = query
            .lease_for(LeaseClass::Fallback, WorkRequest::default())
            .unwrap();
        let mut work = QueryWork {
            query: &query,
            phase,
            cancel: &cancel,
            calls: 0,
            stop,
        };
        let mut output = Array2::from_elem((1, 2), -12345.0);
        let result = into_with_work(
            error.view(),
            2,
            3,
            weights.as_ref(),
            output.view_mut(),
            &mut work,
        );
        assert_eq!(result.is_ok(), limit == total && stop == 0);
        assert!(query.receipt().unwrap().work_reserved <= limit as u64);
        assert_eq!(query.receipt().unwrap().admitted, 1);
        if limit == total && stop == 0 {
            assert_eq!(query.receipt().unwrap().work_reserved, 704);
        }
        if stop != 0 {
            assert!(matches!(result,Err(e) if e.is_deadline_exceeded()));
            assert_eq!(work.calls, stop);
        }
    }
}
