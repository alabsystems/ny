// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::{aw_f64_with_abssum_cpu_pollable_scalar, gamma_n_f64, publish_error_up_normal};
use super::*;
use crate::amortise::NoPoll;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLease, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use faer::Mat;
use ndarray::{s, Array2};
use num_rational::BigRational;
use std::time::Duration;

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}
fn source_bits(a: &Mat<f32>) -> Vec<u32> {
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| a[(i, j)].to_bits()))
        .collect()
}
fn fixture(m: usize, k: usize, p: usize) -> (Mat<f32>, Mat<f32>) {
    let v = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800001, 0xbf800000, 0x4f800001, 0x7f800000,
        0xff800000, 0x7fc01234,
    ];
    (
        Mat::from_fn(m, k, |i, j| f32::from_bits(v[(i * 5 + j * 7) % v.len()])),
        Mat::from_fn(k, p, |i, j| {
            f32::from_bits(v[(i * 3 + j * 5 + 1) % v.len()])
        }),
    )
}

#[test]
fn cooperative_scalar_aw_all_shape_classes_keep_original_legacy_bits() {
    for (m, k, p) in [
        (0, 0, 0),
        (0, 3, 5),
        (3, 0, 5),
        (3, 5, 0),
        (1, 1, 1),
        (2, 50, 50),
        (5, 50, 5),
        (5, 5, 50),
        (2, 257, 2),
        (2, 1025, 3),
        (1, 3, 4097),
    ] {
        let (a, w) = fixture(m, k, p);
        let ab = source_bits(&a);
        let wb = source_bits(&w);
        let (expected, expected_s) = original_scalar_aw(&a, &w, None).unwrap();
        let (legacy, legacy_s) = aw_f64_with_abssum_cpu_pollable_scalar(&a, &w, None).unwrap();
        assert_eq!(bits(&legacy), bits(&expected));
        assert_eq!(bits(&legacy_s), bits(&expected_s));
        let mut out = Array2::from_elem((m, p), -12345.0);
        let mut sum = out.clone();
        into_with_work(
            a.as_ref(),
            w.as_ref(),
            out.view_mut(),
            sum.view_mut(),
            &mut NoPoll,
        )
        .unwrap();
        assert_eq!(bits(&out), bits(&expected), "shape{m}x{k}x{p}");
        assert_eq!(bits(&sum), bits(&expected_s));
        assert_eq!(source_bits(&a), ab);
        assert_eq!(source_bits(&w), wb);
    }
}

#[test]
fn cooperative_scalar_aw_existing_certificate_encloses_exact_rational_dots() {
    let av = [
        1.0,
        -1.0,
        0.375,
        2f32.powi(-32),
        2f32.powi(32),
        f32::from_bits(1),
    ];
    let r = |v: f32| BigRational::from_float(f32_to_f64_exact(v)).unwrap();
    for (m, k, p) in [(2, 50, 50), (5, 50, 5), (3, 5, 50), (2, 257, 3)] {
        let a = Mat::from_fn(m, k, |i, j| av[(i * 3 + j * 5) % av.len()]);
        let w = Mat::from_fn(k, p, |i, j| av[(i * 5 + j * 3 + 1) % av.len()]);
        let mut out = Array2::zeros((m, p));
        let mut sum = out.clone();
        into_with_work(
            a.as_ref(),
            w.as_ref(),
            out.view_mut(),
            sum.view_mut(),
            &mut NoPoll,
        )
        .unwrap();
        for i in 0..m {
            for j in 0..p {
                let mut exact = BigRational::from_integer(0.into());
                for kk in 0..k {
                    exact += r(a[(i, kk)]) * r(w[(kk, j)]);
                }
                let stored = out[[i, j]] as f32;
                let error = publish_error_up_normal(
                    (out[[i, j]] - f32_to_f64_exact(stored)).abs() + gamma_n_f64(k) * sum[[i, j]],
                );
                assert!(error.is_finite());
                assert!(r(stored) - r(error) <= exact && exact <= r(stored) + r(error));
            }
        }
    }
}

#[test]
fn cooperative_scalar_aw_strided_sources_and_private_outputs_preserve_all_gaps() {
    let (m, k, p) = (3, 17, 5);
    let (base_a, base_w) = fixture(m + 2, k + 2, p + 2);
    let a = base_a.as_ref().submatrix(1, 1, m, k);
    let w = base_w.as_ref().submatrix(1, 1, k, p);
    let copied_a = Mat::from_fn(m, k, |i, j| a[(i, j)]);
    let copied_w = Mat::from_fn(k, p, |i, j| w[(i, j)]);
    let (expected, expected_s) = original_scalar_aw(&copied_a, &copied_w, None).unwrap();
    let mut data = Array2::from_elem((2 * m + 2, 2 * p + 2), -12345.0);
    let mut sum = Array2::from_elem((3 * m + 2, 3 * p + 2), -23456.0);
    into_with_work(
        a,
        w,
        data.slice_mut(s![1..1+2*m;2,1..1+2*p;2]),
        sum.slice_mut(s![1..1+3*m;3,1..1+3*p;3]),
        &mut NoPoll,
    )
    .unwrap();
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            if i >= 1
                && i < 1 + 2 * m
                && (i - 1) % 2 == 0
                && j >= 1
                && j < 1 + 2 * p
                && (j - 1) % 2 == 0
            {
                assert_eq!(
                    data[[i, j]].to_bits(),
                    expected[[(i - 1) / 2, (j - 1) / 2]].to_bits()
                );
            } else {
                assert_eq!(data[[i, j]], -12345.0);
            }
        }
    }
    for i in 0..sum.nrows() {
        for j in 0..sum.ncols() {
            if i >= 1
                && i < 1 + 3 * m
                && (i - 1) % 3 == 0
                && j >= 1
                && j < 1 + 3 * p
                && (j - 1) % 3 == 0
            {
                assert_eq!(
                    sum[[i, j]].to_bits(),
                    expected_s[[(i - 1) / 3, (j - 1) / 3]].to_bits()
                );
            } else {
                assert_eq!(sum[[i, j]], -23456.0);
            }
        }
    }
}

struct Boundary {
    calls: usize,
    stop: usize,
    units: usize,
    max: usize,
}
impl CooperativeWork for Boundary {
    fn checkpoint(&mut self) -> Result<()> {
        self.calls += 1;
        if self.calls == self.stop {
            Err(NyError::DeadlineExceeded(
                "original scalar AW observer".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn before(&mut self, units: usize) -> Result<()> {
        assert!(units <= 4096);
        self.checkpoint()?;
        self.units += units;
        self.max = self.max.max(units);
        Ok(())
    }
}

#[test]
fn cooperative_scalar_aw_exact_prework_units_and_every_refusal_boundary() {
    let a = Mat::from_fn(2, 3, |i, j| 1.0 + (i + j) as f32 * 0.125);
    let w = Mat::from_fn(3, 65, |i, j| (i + j) as f32 * 0.0625);
    let ab = source_bits(&a);
    let wb = source_bits(&w);
    let mut out = Array2::zeros((2, 65));
    let mut sum = out.clone();
    let mut baseline = Boundary {
        calls: 0,
        stop: 0,
        units: 0,
        max: 0,
    };
    into_with_work(
        a.as_ref(),
        w.as_ref(),
        out.view_mut(),
        sum.view_mut(),
        &mut baseline,
    )
    .unwrap();
    //64metadata +130pairs*64initialization +6A-decodes*64 +390MACs*64.
    assert_eq!(baseline.units, 33728);
    assert_eq!(baseline.max, 4096);
    assert!(baseline.calls > 130);
    for stop in 1..=baseline.calls {
        let mut data = Array2::from_elem((2, 65), -12345.0);
        let mut abs = data.clone();
        let mut work = Boundary {
            calls: 0,
            stop,
            units: 0,
            max: 0,
        };
        let result = into_with_work(
            a.as_ref(),
            w.as_ref(),
            data.view_mut(),
            abs.view_mut(),
            &mut work,
        );
        assert!(
            matches!(result,Err(NyError::DeadlineExceeded(ref e))if e=="original scalar AW observer")
        );
        assert_eq!(work.calls, stop);
        assert_eq!(source_bits(&a), ab);
        assert_eq!(source_bits(&w), wb);
        if stop <= 2 {
            assert!(data.iter().chain(abs.iter()).all(|&v| v == -12345.0));
        }
    }
}

#[test]
fn cooperative_scalar_aw_shape_and_offset_errors_precede_all_destination_writes() {
    let a = Mat::zeros(2, 3);
    let w = Mat::zeros(3, 5);
    let wrong_w = Mat::zeros(2, 5);
    for wrong in [0, 1, 2] {
        let shape = if wrong == 1 { (2, 4) } else { (2, 5) };
        let mut data = Array2::from_elem(shape, -12345.0);
        let mut sum = Array2::from_elem((2, 5), -23456.0);
        let mut work = Boundary {
            calls: 0,
            stop: 0,
            units: 0,
            max: 0,
        };
        let result = into_with_work(
            a.as_ref(),
            if wrong == 0 {
                wrong_w.as_ref()
            } else {
                w.as_ref()
            },
            data.view_mut(),
            if wrong == 2 {
                sum.slice_mut(s![.., 0..4])
            } else {
                sum.view_mut()
            },
            &mut work,
        );
        assert!(result.is_err());
        assert_eq!(work.units, 64);
        assert!(data.iter().all(|&v| v == -12345.0));
        assert!(sum.iter().all(|&v| v == -23456.0));
    }
    let lower = Array2::zeros((2, 3));
    let upper = lower.clone();
    let mut data = Array2::from_elem((4, 5), -12345.0);
    let mut sum = data.clone();
    for offset in [2, usize::MAX] {
        assert!(stacked_into_with_work(
            lower.view(),
            upper.view(),
            offset,
            3,
            w.as_ref(),
            data.view_mut(),
            sum.view_mut(),
            &mut NoPoll
        )
        .is_err());
        assert!(data.iter().chain(sum.iter()).all(|&v| v == -12345.0));
    }
    let bad_upper = Array2::zeros((1, 3));
    assert!(stacked_into_with_work(
        lower.view(),
        bad_upper.view(),
        0,
        3,
        w.as_ref(),
        data.view_mut(),
        sum.view_mut(),
        &mut NoPoll
    )
    .is_err());
}

#[test]
fn cooperative_scalar_aw_zero_skips_poison_and_empty_shapes_keep_final_authority() {
    let a = Mat::zeros(2, 257);
    let w = Mat::from_fn(257, 3, |i, j| {
        if (i + j) % 2 == 0 {
            f32::NAN
        } else {
            f32::INFINITY
        }
    });
    let mut data = Array2::from_elem((2, 3), -12345.0);
    let mut sum = data.clone();
    let mut work = Boundary {
        calls: 0,
        stop: 0,
        units: 0,
        max: 0,
    };
    into_with_work(
        a.as_ref(),
        w.as_ref(),
        data.view_mut(),
        sum.view_mut(),
        &mut work,
    )
    .unwrap();
    assert!(data.iter().chain(sum.iter()).all(|&v| v.to_bits() == 0));
    assert_eq!(work.units, 64 + 6 * 64 + 2 * 257 * 64);
    let huge = 1usize << 30;
    let empty_a = MatRef::from_column_major_slice(&[], huge, 0);
    let empty_w = Mat::<f32>::zeros(0, 0);
    let mut data = Array2::zeros((huge, 0));
    let mut sum = data.clone();
    let mut work = Boundary {
        calls: 0,
        stop: 0,
        units: 0,
        max: 0,
    };
    into_with_work(
        empty_a,
        empty_w.as_ref(),
        data.view_mut(),
        sum.view_mut(),
        &mut work,
    )
    .unwrap();
    assert_eq!(work.units, 64);
    assert_eq!(work.calls, 3);
    let mut work = Boundary {
        calls: 0,
        stop: 3,
        units: 0,
        max: 0,
    };
    assert!(
        matches!(into_with_work(empty_a,empty_w.as_ref(),data.view_mut(),sum.view_mut(),&mut work),Err(e)if e.is_deadline_exceeded())
    );
}

#[test]
fn cooperative_scalar_aw_borrowed_lower_upper_position_blocks_match_owned_stack_bits() {
    let (rows, k, p) = (3, 50, 5);
    let mut lower = Array2::zeros((rows * 2 + 2, 2 * (17 * k + 4) + 2));
    let mut upper = lower.clone();
    for i in 0..lower.nrows() {
        for j in 0..lower.ncols() {
            lower[[i, j]] = (i * 3 + j * 5) as f32 * 0.0625 - 1.0;
            upper[[i, j]] = -lower[[i, j]];
        }
    }
    let lo = lower.slice(s![1..1+2*rows;2,1..1+2*(17*k+4);2]);
    let up = upper.slice(s![1..1+2*rows;2,1..1+2*(17*k+4);2]);
    let offset = 16 * k + 2;
    let w = Mat::from_fn(k, p, |i, j| ((i + j) % 7) as f32 * 0.125 - 0.375);
    let stack = Mat::from_fn(2 * rows, k, |i, j| {
        if i < rows {
            lo[[i, offset + j]]
        } else {
            up[[i - rows, offset + j]]
        }
    });
    let (expected, expected_s) = original_scalar_aw(&stack, &w, None).unwrap();
    let mut data = Array2::from_elem((2 * rows, p), -12345.0);
    let mut sum = data.clone();
    stacked_into_with_work(
        lo,
        up,
        offset,
        k,
        w.as_ref(),
        data.view_mut(),
        sum.view_mut(),
        &mut NoPoll,
    )
    .unwrap();
    assert_eq!(bits(&data), bits(&expected));
    assert_eq!(bits(&sum), bits(&expected_s));
    assert_eq!(lower[[0, 0]], -1.0);
    assert_eq!(upper[[0, 0]], 1.0);
}

fn refused(e: WorkRefusal) -> NyError {
    match e {
        WorkRefusal::Deadline | WorkRefusal::Cancelled => {
            NyError::DeadlineExceeded(format!("original scalar query{e}"))
        }
        _ => NyError::UnsupportedOp(format!("original scalar query{e}")),
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
            .map_err(refused)
    }
    fn before(&mut self, units: usize) -> Result<()> {
        assert!(units <= 4096);
        self.checkpoint()?;
        self.phase.reserve_more_work(units as u64).map_err(refused)
    }
}

#[test]
fn cooperative_scalar_aw_original_query_work_and_cancel_do_not_renew_at_chunk_seams() {
    let a = Mat::from_fn(2, 3, |i, j| 1.0 + (i + j) as f32 * 0.125);
    let w = Mat::from_fn(3, 65, |i, j| (i + j) as f32 * 0.0625);
    let mut data = Array2::zeros((2, 65));
    let mut sum = data.clone();
    let mut baseline = Boundary {
        calls: 0,
        stop: 0,
        units: 0,
        max: 0,
    };
    into_with_work(
        a.as_ref(),
        w.as_ref(),
        data.view_mut(),
        sum.view_mut(),
        &mut baseline,
    )
    .unwrap();
    assert_eq!(baseline.units, 33728);
    for (limit, stop) in [
        (0, 0),
        (63, 0),
        (64, 0),
        (33727, 0),
        (33728, 0),
        (33728, 1),
        (33728, baseline.calls),
    ] {
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
        let mut data = Array2::from_elem((2, 65), -12345.0);
        let mut sum = data.clone();
        let result = into_with_work(
            a.as_ref(),
            w.as_ref(),
            data.view_mut(),
            sum.view_mut(),
            &mut work,
        );
        assert_eq!(result.is_ok(), limit == 33728 && stop == 0);
        assert_eq!(query.receipt().unwrap().admitted, 1);
        assert!(query.receipt().unwrap().work_reserved <= limit as u64);
        if result.is_ok() {
            assert_eq!(query.receipt().unwrap().work_reserved, 33728);
        }
        if stop != 0 {
            assert!(matches!(result,Err(e)if e.is_deadline_exceeded()));
            assert_eq!(work.calls, stop);
        }
    }
}

// Independent cfg(test) snapshot of the historical helper, without edits to
// arithmetic, allocation, shapes, polling or deadline behavior.
fn original_scalar_aw(
    a_block: &Mat<f32>,
    w: &Mat<f32>,
    deadline: Option<Instant>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    const DEADLINE_POLL_MACS: usize = 1 << 12;
    const DEADLINE_POLL_SKIPPED_K: usize = 1 << 12;

    check_optional_aw_deadline(deadline, "during certified f64 A·W")?;
    let m = a_block.nrows();
    let k = a_block.ncols();
    let p = w.ncols();
    if w.nrows() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k, p],
            got: vec![w.nrows(), p],
        });
    }

    let mut a64 = Array2::<f64>::zeros((m, p));
    let mut s = Array2::<f64>::zeros((m, p));
    // Large zero-fills may themselves consume the remaining budget.
    check_optional_aw_deadline(deadline, "during certified f64 A·W")?;

    let a64_buf = a64
        .as_slice_mut()
        .expect("a64 is freshly allocated row-major contiguous");
    let s_buf = s
        .as_slice_mut()
        .expect("s is freshly allocated row-major contiguous");
    let mut skipped_k_since_poll = 0usize;
    for i in 0..m {
        check_optional_aw_deadline(deadline, "during certified f64 A·W")?;
        let a64_row = &mut a64_buf[i * p..i * p + p];
        let s_row = &mut s_buf[i * p..i * p + p];
        for kk in 0..k {
            let av = f32_to_f64_exact(a_block[(i, kk)]);
            if av == 0.0 || p == 0 {
                skipped_k_since_poll += 1;
                if skipped_k_since_poll >= DEADLINE_POLL_SKIPPED_K {
                    check_optional_aw_deadline(deadline, "during certified f64 A·W")?;
                    skipped_k_since_poll = 0;
                }
                continue;
            }

            let av_abs = av.abs();
            let mut j_start = 0usize;
            while j_start < p {
                let j_end = (j_start + DEADLINE_POLL_MACS).min(p);
                for j in j_start..j_end {
                    let wv = f32_to_f64_exact(w[(kk, j)]);
                    a64_row[j] += av * wv;
                    s_row[j] += av_abs * wv.abs();
                }
                check_optional_aw_deadline(deadline, "during certified f64 A·W")?;
                j_start = j_end;
            }
        }
    }
    check_optional_aw_deadline(deadline, "during certified f64 A·W")?;
    Ok((a64, s))
}
