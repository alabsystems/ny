// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{array, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Clone, Copy)]
enum Fault {
    None,
    RefuseSecond,
    ShortSecond,
    NanSecond,
    PanicSecond,
    DeadlineFirst,
    DeadlineSecond,
    ExpireSecond,
}

struct BoundedOnly {
    calls: AtomicUsize,
    expired: AtomicBool,
    fault: Fault,
    reverse: bool,
    contraction: usize,
}

impl BoundedOnly {
    fn new(contraction: usize, fault: Fault) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            expired: AtomicBool::new(false),
            fault,
            reverse: true,
            contraction,
        }
    }
}

impl GemmEngine for BoundedOnly {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("finite-authority route must never enter ordinary f32 GEMM")
    }
    fn gemm_f64(&self, _: usize, _: usize, _: usize, _: &[f64], _: &[f64]) -> Result<Vec<f64>> {
        panic!("finite-authority route must never enter ordinary f64 GEMM")
    }
    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        _: DeadlineInstant,
        cap: usize,
    ) -> Result<Vec<f64>> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        assert_eq!(
            k, self.contraction,
            "every output retains the whole contraction"
        );
        assert_eq!(cap, 1 << 24);
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        if call == 1 && matches!(self.fault, Fault::DeadlineFirst) {
            return Err(NyError::DeadlineExceeded("first dispatch".into()));
        }
        if call == 2 {
            match self.fault {
                Fault::RefuseSecond => return Err(NyError::UnsupportedOp("injected".into())),
                Fault::ShortSecond => return Ok(vec![0.0; m * n - 1]),
                Fault::NanSecond => return Ok(vec![f64::NAN; m * n]),
                Fault::PanicSecond => panic!("injected second bounded call panic"),
                Fault::DeadlineSecond => return Err(NyError::DeadlineExceeded("injected".into())),
                Fault::ExpireSecond => self.expired.store(true, Ordering::Relaxed),
                Fault::None | Fault::DeadlineFirst => {}
            }
        }
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for index in 0..k {
                    let t = if self.reverse { k - 1 - index } else { index };
                    result[i * n + j] += a[i * k + t] * b[t * n + j];
                }
            }
        }
        Ok(result)
    }
}

fn fixture() -> (Array2<f32>, Array2<f32>, ArrayD<f32>, DirectF64PairGeometry) {
    let lower = Array2::from_shape_fn((3, 2 * 4 * 5), |(r, c)| {
        ((r * 13 + c * 7) % 23) as f32 / 8.0 - 1.5
    });
    let upper = lower.mapv(|x| -x + 0.25);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[3, 2, 2, 2]), |i| {
        ((i[0] * 11 + i[1] * 7 + i[2] * 5 + i[3]) % 17) as f32 / 16.0 - 0.5
    });
    let g = DirectF64PairGeometry::checked(&lower, &upper, &kernel, (2, 1), (1, 0), (1, 2), (4, 5))
        .expect("rectangular strided/dilated fixture");
    (lower, upper, kernel, g)
}

fn far() -> DeadlineInstant {
    DeadlineInstant::now() + Duration::from_secs(60)
}

#[test]
fn bounded_convtranspose_pair_default_on_policy_retains_literal_zero_kill_switch() {
    assert!(enabled(None));
    assert!(enabled(Some(OsStr::new("1"))));
    for raw in ["0", "", "true", "01", "1 "] {
        assert!(!enabled(Some(OsStr::new(raw))));
    }
}

// Independent direct convolution: integer numerators in units of 1/128.
// No im2col/scatter helper or floating reduction from the implementation.
fn integer_oracle(a: &Array2<f32>, kernel: &ArrayD<f32>) -> Array2<f64> {
    let mut out = Array2::zeros((3, 3 * 3 * 3));
    for obj in 0..3 {
        for oc in 0..3 {
            for oh in 0..3 {
                for ow in 0..3 {
                    let mut numerator = 0i64;
                    for ic in 0..2 {
                        for kh in 0..2 {
                            for kw in 0..2 {
                                let y = 2 * oh as isize + kh as isize - 1;
                                let x = ow + 2 * kw;
                                if (0..4).contains(&y) && x < 5 {
                                    let av = (a[[obj, ic * 20 + y as usize * 5 + x]] * 8.0) as i64;
                                    let wv = (kernel[[oc, ic, kh, kw]] * 16.0) as i64;
                                    numerator += av * wv;
                                }
                            }
                        }
                    }
                    out[[obj, oc * 9 + oh * 3 + ow]] = numerator as f64 / 128.0;
                }
            }
        }
    }
    out
}

#[test]
fn bounded_convtranspose_pair_matches_exact_integer_convolution_and_keeps_full_k() {
    let (lower, upper, kernel, g) = fixture();
    let engine = BoundedOnly::new(8, Fault::None);
    let result = run(
        &lower,
        &upper,
        &kernel,
        g,
        quote(g).unwrap(),
        &engine,
        far(),
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap();
    assert_eq!(result.0, integer_oracle(&lower, &kernel));
    assert_eq!(result.1, integer_oracle(&upper, &kernel));
    assert_eq!(engine.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn bounded_convtranspose_pair_accepts_strided_source_layout_without_reinterpreting_weights() {
    let (lower, upper, kernel, _) = fixture();
    // Equal-sized kernel spatial axes have different contents. Swapping them
    // changes the convolution and creates a non-contiguous source layout.
    let kernel = kernel.permuted_axes(IxDyn(&[0, 1, 3, 2]));
    assert!(!kernel.is_standard_layout());
    let g = DirectF64PairGeometry::checked(&lower, &upper, &kernel, (2, 1), (1, 0), (1, 2), (4, 5))
        .unwrap();
    let result = run(
        &lower,
        &upper,
        &kernel,
        g,
        quote(g).unwrap(),
        &BoundedOnly::new(8, Fault::None),
        far(),
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap();
    assert_eq!(result.0, integer_oracle(&lower, &kernel));
    assert_eq!(result.1, integer_oracle(&upper, &kernel));
}

#[test]
fn bounded_convtranspose_pair_gamma_enclosure_covers_exact_cancellation() {
    let big = 2.0f32.powi(53);
    let lower = array![[big, 1.0, -big]];
    let upper = array![[big, -big, 1.0]];
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1]), vec![1.0; 3]).unwrap();
    let g = DirectF64PairGeometry::checked(&lower, &upper, &kernel, (1, 1), (0, 0), (1, 1), (1, 1))
        .unwrap();
    let gamma = BigRational::new(BigInt::from(3), (BigInt::from(1) << 53) - 3);
    let s = BigRational::from_integer((BigInt::from(1) << 54) + 1);
    let exact = BigRational::from_integer(BigInt::from(1));
    for reverse in [false, true] {
        let mut engine = BoundedOnly::new(3, Fault::None);
        engine.reverse = reverse;
        let result = run(
            &lower,
            &upper,
            &kernel,
            g,
            quote(g).unwrap(),
            &engine,
            far(),
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
        for x in [result.0[[0, 0]], result.1[[0, 0]]] {
            let actual = BigRational::from_float(x).unwrap();
            let error = if actual >= exact {
                &actual - &exact
            } else {
                &exact - &actual
            };
            assert!(error <= &gamma * &s);
        }
        assert_actual_error_channel(&lower, &result.0, &exact);
        assert_actual_error_channel(&upper, &result.1, &exact);
    }
    // The real caller narrows f64 coefficients to f32. The separate cast term
    // is necessary here: exact 2^24+1 cannot be stored as a binary32 value.
    let lower = array![[2.0f32.powi(24), 1.0, 0.0]];
    let upper = array![[1.0, 2.0f32.powi(24), 0.0]];
    let result = run(
        &lower,
        &upper,
        &kernel,
        g,
        quote(g).unwrap(),
        &BoundedOnly::new(3, Fault::None),
        far(),
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap();
    let exact = BigRational::from_integer((BigInt::from(1) << 24) + 1);
    assert_ne!(
        BigRational::from_float(f64::from(result.0[[0, 0]] as f32)).unwrap(),
        exact
    );
    assert_actual_error_channel(&lower, &result.0, &exact);
    assert_actual_error_channel(&upper, &result.1, &exact);
}

fn assert_actual_error_channel(in_a: &Array2<f32>, coeff: &Array2<f64>, exact: &BigRational) {
    let stored = coeff.mapv(|x| x as f32);
    let errors = crate::layers::convolution::crown_helpers::conv_coeff_err_matrix(
        in_a,
        None,
        &stored,
        Some(coeff),
        3.0,
        3,
        None,
        None,
    );
    assert!(errors[[0, 0]].is_finite());
    let center = BigRational::from_float(f64::from(stored[[0, 0]])).unwrap();
    let radius = BigRational::from_float(f64::from(errors[[0, 0]])).unwrap();
    assert!(&center - &radius <= *exact);
    assert!(&center + &radius >= *exact);
}

#[test]
fn bounded_convtranspose_pair_discards_first_side_when_second_side_refuses_or_is_invalid() {
    let (lower, upper, kernel, g) = fixture();
    for fault in [
        Fault::RefuseSecond,
        Fault::ShortSecond,
        Fault::NanSecond,
        Fault::PanicSecond,
    ] {
        let engine = BoundedOnly::new(8, fault);
        let result = run(
            &lower,
            &upper,
            &kernel,
            g,
            quote(g).unwrap(),
            &engine,
            far(),
            &mut || Ok(()),
        )
        .unwrap();
        assert!(
            result.is_none(),
            "no first-side or partial result may escape"
        );
        assert_eq!(engine.calls.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn bounded_convtranspose_pair_deadline_error_or_post_dispatch_expiry_is_terminal() {
    let (lower, upper, kernel, g) = fixture();
    for fault in [
        Fault::DeadlineFirst,
        Fault::DeadlineSecond,
        Fault::ExpireSecond,
    ] {
        let engine = BoundedOnly::new(8, fault);
        let mut poll = || {
            if engine.expired.load(Ordering::Relaxed) {
                Err(NyError::DeadlineExceeded("post-dispatch clock".into()))
            } else {
                Ok(())
            }
        };
        let error = run(
            &lower,
            &upper,
            &kernel,
            g,
            quote(g).unwrap(),
            &engine,
            far(),
            &mut poll,
        )
        .unwrap_err();
        assert!(error.is_deadline_exceeded());
        let expected_calls = if matches!(fault, Fault::DeadlineFirst) {
            1
        } else {
            2
        };
        assert_eq!(engine.calls.load(Ordering::Relaxed), expected_calls);
    }
}

#[test]
fn bounded_convtranspose_pair_every_poll_failure_including_final_publication_is_terminal() {
    let (lower, upper, kernel, g) = fixture();
    let count = Cell::new(0usize);
    run(
        &lower,
        &upper,
        &kernel,
        g,
        quote(g).unwrap(),
        &BoundedOnly::new(8, Fault::None),
        far(),
        &mut || {
            count.set(count.get() + 1);
            Ok(())
        },
    )
    .unwrap()
    .unwrap();
    assert!(count.get() > 20);
    for stop in 1..=count.get() {
        let calls = Cell::new(0usize);
        let result = run(
            &lower,
            &upper,
            &kernel,
            g,
            quote(g).unwrap(),
            &BoundedOnly::new(8, Fault::None),
            far(),
            &mut || {
                calls.set(calls.get() + 1);
                // One transient failure, not a permanently expired mock.
                if calls.get() == stop {
                    Err(NyError::DeadlineExceeded("injected poll".into()))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.unwrap_err().is_deadline_exceeded(), "poll {stop}");
    }
}

#[test]
fn bounded_convtranspose_pair_nonfinite_or_subnormal_source_declines_before_dispatch() {
    let (lower, upper, kernel, g) = fixture();
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(1),
        -f32::from_bits(1),
    ] {
        let mut bad_kernel = kernel.clone();
        bad_kernel[[0, 0, 0, 0]] = value;
        let engine = BoundedOnly::new(8, Fault::None);
        assert!(run(
            &lower,
            &upper,
            &bad_kernel,
            g,
            quote(g).unwrap(),
            &engine,
            far(),
            &mut || Ok(())
        )
        .unwrap()
        .is_none());
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
        let mut bad_lower = lower.clone();
        bad_lower[[0, 0]] = value;
        let engine = BoundedOnly::new(8, Fault::None);
        assert!(run(
            &bad_lower,
            &upper,
            &kernel,
            g,
            quote(g).unwrap(),
            &engine,
            far(),
            &mut || Ok(())
        )
        .unwrap()
        .is_none());
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn bounded_convtranspose_pair_cpu_only_scope_never_observes_the_global_engine() {
    let _lock = crate::sound_f64_gemm::accessor_counter_test_lock();
    crate::sound_f64_gemm::reset_cpu_only_global_accessor_attempts();
    let input = Array2::zeros((1, 8 * 64 * 64));
    let kernel = ArrayD::zeros(IxDyn(&[64, 8, 3, 3]));
    let g =
        DirectF64PairGeometry::checked(&input, &input, &kernel, (1, 1), (0, 0), (1, 1), (64, 64))
            .unwrap();
    assert!(
        g.spatial_per_obj * g.col_width * g.conv_out_c
            >= super::super::super::ops_transpose_gemm::CONV_SOUND_F64_GEMM_MIN_MACS
    );
    let _cpu = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
    assert!(try_pair(
        &input,
        &input,
        &kernel,
        (1, 1),
        (0, 0),
        (1, 1),
        (64, 64),
        far()
    )
    .unwrap()
    .is_none());
    assert_eq!(
        crate::sound_f64_gemm::cpu_only_global_accessor_attempts(),
        0
    );
}

#[test]
fn bounded_convtranspose_pair_quote_caps_peak_and_contraction_without_allocating() {
    let (_, _, _, g) = fixture();
    let q = quote(g).unwrap();
    let rows = q.block_objs * g.spatial_per_obj;
    assert_eq!(
        q.peak_bytes,
        8 * (2 * g.result_len
            + rows * g.col_width
            + rows * g.conv_out_c
            + g.col_width * g.conv_out_c)
            + 1_024
    );
    assert!(q.peak_bytes <= MAX_OWNED_BYTES);
    let mut overflow = g;
    overflow.result_len = usize::MAX;
    assert!(quote(overflow).is_none());
    let mut excessive = g;
    excessive.result_len = MAX_OWNED_BYTES / 16;
    assert!(quote(excessive).is_none());
    let mut contraction = g;
    contraction.col_width = MAX_DISPATCH_MACS + 1;
    assert!(quote(contraction).is_none());
    let (_, _, kernel, _) = fixture();
    let empty = Array2::zeros((0, 2 * 4 * 5));
    let empty_g =
        DirectF64PairGeometry::checked(&empty, &empty, &kernel, (2, 1), (1, 0), (1, 2), (4, 5))
            .unwrap();
    assert!(
        quote(empty_g).is_none(),
        "empty requests retain the historical route"
    );
}
