// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for GEMM engine trait and NaiveCpuGemmEngine implementation.

use super::*;

#[test]
fn directional_f32_steps_cross_the_opposite_infinity() {
    assert_eq!(next_up_f32(f32::NEG_INFINITY), -f32::MAX);
    assert_eq!(next_down_f32(f32::INFINITY), f32::MAX);
    assert_eq!(next_up_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(next_down_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
}

#[test]
fn gemm_f64_with_deadline_default_never_delegates_to_ordinary_gemm() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct OrdinaryCallCounter {
        ordinary_calls: AtomicUsize,
    }

    impl GemmEngine for OrdinaryCallCounter {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            unreachable!("deadline-f64 default must not call f32 GEMM")
        }

        fn gemm_f64(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0; m * n])
        }
    }

    let engine = OrdinaryCallCounter::default();
    let error = engine
        .gemm_f64_with_deadline(
            1,
            1,
            1,
            &[2.0],
            &[3.0],
            Instant::now() + Duration::from_secs(1),
            16,
        )
        .expect_err("default deadline GEMM must explicitly decline");
    assert!(matches!(error, NyError::UnsupportedOp(_)));
    assert_eq!(engine.ordinary_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn gemm_f64_pair_shared_rhs_default_is_two_ordered_fallback_calls() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingF64 {
        calls: Mutex<Vec<(f64, f64)>>,
    }

    impl GemmEngine for RecordingF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            unreachable!("the f64 pair fallback must not call f32 GEMM")
        }

        fn gemm_f64(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            a: &[f64],
            b: &[f64],
        ) -> Result<Vec<f64>> {
            let markers = (a[0], b[0]);
            self.calls.lock().expect("recording lock").push(markers);
            Ok(vec![markers.0 + markers.1; m * n])
        }
    }

    let engine = RecordingF64::default();
    let lower = [1.0, 2.0];
    let upper = [3.0, 4.0];
    let shared_rhs = [7.0, 8.0];
    let got = engine
        .gemm_f64_pair_shared_rhs(1, 2, 1, [&lower, &upper], &shared_rhs)
        .expect("default shared-RHS pair fallback");

    assert_eq!(got, [vec![8.0], vec![10.0]]);
    assert_eq!(
        *engine.calls.lock().expect("recording lock"),
        vec![(1.0, 7.0), (3.0, 7.0)]
    );
}

#[test]
fn gemm_f64_triplet_default_is_three_ordered_fallback_calls() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingF64 {
        calls: Mutex<Vec<f64>>,
    }

    impl GemmEngine for RecordingF64 {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            unreachable!("the f64 triplet fallback must not call f32 GEMM")
        }

        fn gemm_f64(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            let marker = a[0];
            self.calls.lock().expect("recording lock").push(marker);
            Ok(vec![marker; m * n])
        }
    }

    let engine = RecordingF64::default();
    let a0 = [1.0, 2.0];
    let a1 = [3.0, 4.0];
    let a2 = [5.0, 6.0];
    let b = [7.0, 8.0];
    let got = engine
        .gemm_f64_triplet(1, 2, 1, [&a0, &a1, &a2], [&b, &b, &b])
        .expect("default triplet fallback");

    assert_eq!(got, [vec![1.0], vec![3.0], vec![5.0]]);
    assert_eq!(
        *engine.calls.lock().expect("recording lock"),
        vec![1.0, 3.0, 5.0]
    );
}

#[test]
fn test_gemm_2x2_identity() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![3.0, 4.0, 5.0, 6.0];
    let c = engine.gemm_f32(2, 2, 2, &a, &b).unwrap();
    assert_eq!(c, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_gemm_2x2_known_product() {
    let engine = NaiveCpuGemmEngine;
    // C = [[19,22],[43,50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = engine.gemm_f32(2, 2, 2, &a, &b).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_3x3_known_product() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let c = engine.gemm_f32(3, 3, 3, &a, &b).unwrap();
    assert_eq!(c, b);
}

#[test]
fn test_gemm_non_square() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let c = engine.gemm_f32(2, 3, 2, &a, &b).unwrap();
    assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_gemm_dimension_mismatch_a() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let result = engine.gemm_f32(2, 2, 2, &a, &b);
    assert!(
        result.is_err(),
        "gemm_f32 should reject a.len()={} for 2x2 * 2x2 input",
        a.len()
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("a.len()=3"),
        "dimension mismatch should mention the short lhs buffer: {err}"
    );
}

#[test]
fn test_gemm_dimension_mismatch_b() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0];
    let result = engine.gemm_f32(2, 2, 2, &a, &b);
    assert!(
        result.is_err(),
        "gemm_f32 should reject b.len()={} for 2x2 * 2x2 input",
        b.len()
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("b.len()=2"),
        "dimension mismatch should mention the short rhs buffer: {err}"
    );
}

#[test]
fn test_gemm_empty_m_zero() {
    let engine = NaiveCpuGemmEngine;
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let c = engine.gemm_f32(0, 0, 0, &a, &b).unwrap();
    assert!(
        c.is_empty(),
        "zero-sized GEMM should return an empty output, got {c:?}"
    );
}

#[test]
fn test_gemm_single_element() {
    let engine = NaiveCpuGemmEngine;
    let a = vec![3.0];
    let b = vec![4.0];
    let c = engine.gemm_f32(1, 1, 1, &a, &b).unwrap();
    assert_eq!(c, vec![12.0]);
}

#[test]
fn test_gemm_k_zero() {
    let engine = NaiveCpuGemmEngine;
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let c = engine.gemm_f32(2, 0, 3, &a, &b).unwrap();
    assert_eq!(c, vec![0.0; 6]);
}

#[test]
fn test_gemm_dimension_overflow_returns_error_instead_of_panicking() {
    let engine = NaiveCpuGemmEngine;
    assert!(
        engine.gemm_f32(usize::MAX, 2, 0, &[], &[]).is_err(),
        "f32 GEMM must reject an overflowing lhs shape"
    );
    assert!(
        engine.gemm_f64(usize::MAX, 2, 0, &[], &[]).is_err(),
        "f64 GEMM must reject an overflowing lhs shape"
    );
}

#[test]
fn test_sound_gemm_helpers_preserve_empty_contraction_shape() {
    let engine = NaiveCpuGemmEngine;

    let (lower, upper) = engine
        .gemm_interval_sound(2, 0, 3, &[], &[], &[], &[])
        .expect("empty interval contraction");
    assert_eq!(lower, vec![0.0; 6]);
    assert_eq!(upper, vec![0.0; 6]);

    let (coeff, error) = engine
        .crown_aw_error_step(2, 0, 3, &[], &[], &[])
        .expect("empty CROWN contraction");
    assert_eq!(coeff, vec![0.0; 6]);
    assert_eq!(error, vec![0.0; 6]);
}

#[test]
fn test_sound_gemm_helpers_reject_dimension_overflow() {
    let engine = NaiveCpuGemmEngine;
    assert!(
        engine
            .gemm_interval_sound(usize::MAX, 2, 0, &[], &[], &[], &[])
            .is_err(),
        "sound interval GEMM must reject an overflowing lhs shape"
    );
    assert!(
        engine
            .crown_aw_error_step(usize::MAX, 2, 0, &[], &[], &[])
            .is_err(),
        "CROWN error propagation must reject an overflowing lhs shape"
    );
    assert!(
        crown_activation_error_step(usize::MAX, 2, &[], &[], &[], &[], &[], &[]).is_err(),
        "activation error propagation must reject an overflowing coefficient shape"
    );
}

#[test]
fn test_sound_gemm_helpers_reject_short_backend_results() {
    struct ShortOutputEngine;

    impl GemmEngine for ShortOutputEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Ok(vec![])
        }
    }

    let engine = ShortOutputEngine;
    assert!(
        engine
            .gemm_interval_sound(1, 1, 1, &[0.0], &[0.0], &[0.0], &[0.0])
            .is_err(),
        "interval helper must not index a malformed backend result"
    );
    assert!(
        engine
            .crown_aw_error_step(1, 1, 1, &[0.0], &[0.0], &[0.0])
            .is_err(),
        "CROWN helper must not index a malformed backend result"
    );
}

/// Verify conv_transpose_2d for a 1-spec, 1-channel, 3x3 kernel, stride=1, pad=0 case.
#[test]
fn test_conv_transpose_2d_simple_3x3() {
    let engine = NaiveCpuGemmEngine;
    let params = ConvTranspose2dParams {
        num_specs: 1,
        out_channels: 1,
        in_channels: 1,
        out_h: 2,
        out_w: 2,
        in_h: 4,
        in_w: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    let a_reshaped = vec![1.0, 0.0, 0.0, 0.0];
    let weight_col = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = engine
        .conv_transpose_2d(&a_reshaped, &weight_col, &params)
        .unwrap();
    assert_eq!(result.len(), 16);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0, 0.0,
        4.0, 5.0, 6.0, 0.0,
        7.0, 8.0, 9.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result, expected);
}

/// Verify conv_transpose_2d matches manual GEMM + col2im for a multi-spec case.
#[test]
fn test_conv_transpose_2d_multi_spec() {
    let engine = NaiveCpuGemmEngine;
    let params = ConvTranspose2dParams {
        num_specs: 2,
        out_channels: 2,
        in_channels: 1,
        out_h: 1,
        out_w: 1,
        in_h: 3,
        in_w: 3,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    let a_reshaped = vec![1.0, 0.0, 0.0, 1.0];
    #[rustfmt::skip]
    let weight_col = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let result = engine
        .conv_transpose_2d(&a_reshaped, &weight_col, &params)
        .unwrap();
    assert_eq!(result.len(), 18);
    assert_eq!(
        &result[0..9],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        &result[9..18],
        &[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    );
}

#[test]
fn test_conv_transpose_2d_dimension_mismatch() {
    let engine = NaiveCpuGemmEngine;
    let params = ConvTranspose2dParams {
        num_specs: 1,
        out_channels: 2,
        in_channels: 1,
        out_h: 2,
        out_w: 2,
        in_h: 4,
        in_w: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    let result = engine.conv_transpose_2d(&[1.0; 4], &[1.0; 18], &params);
    assert!(
        result.is_err(),
        "conv_transpose_2d should reject a_reshaped.len()=4 for params {params:?}"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("a_reshaped.len()=4"),
        "dimension mismatch should mention the short activation buffer: {err}"
    );
}

#[test]
fn test_conv_transpose_2d_rejects_dimension_and_coordinate_overflow() {
    let engine = NaiveCpuGemmEngine;
    let dimension_overflow = ConvTranspose2dParams {
        num_specs: usize::MAX,
        out_channels: 2,
        in_channels: 1,
        out_h: 1,
        out_w: 1,
        in_h: 1,
        in_w: 1,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    assert!(engine
        .conv_transpose_2d(&[], &[], &dimension_overflow)
        .is_err());

    let coordinate_overflow = ConvTranspose2dParams {
        num_specs: 0,
        out_channels: 0,
        in_channels: 0,
        out_h: 2,
        out_w: 1,
        in_h: 0,
        in_w: 0,
        kernel_h: 2,
        kernel_w: 1,
        stride_h: usize::MAX,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    assert!(engine
        .conv_transpose_2d(&[], &[], &coordinate_overflow)
        .is_err());
}

#[test]
fn test_conv_transpose_2d_zero_volume_skips_large_scatter_loop() {
    let engine = NaiveCpuGemmEngine;
    let params = ConvTranspose2dParams {
        num_specs: 1,
        out_channels: 0,
        in_channels: 0,
        out_h: usize::MAX,
        out_w: 1,
        in_h: 0,
        in_w: 0,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };

    let result = engine
        .conv_transpose_2d(&[], &[], &params)
        .expect("zero-volume transpose convolution is empty");
    assert!(result.is_empty());
}

#[test]
fn test_is_crown_coeff_safe_normal_values() {
    for value in [0.0, 1.0, -1.0, 1e9, -1e9] {
        assert!(
            is_crown_coeff_safe(value),
            "value {value} should stay within the safe CROWN coefficient envelope"
        );
    }
}

#[test]
fn test_is_crown_coeff_safe_overflow_values() {
    for value in [
        CROWN_COEFF_MAX,
        -CROWN_COEFF_MAX,
        CROWN_COEFF_MAX * 1.001,
        -CROWN_COEFF_MAX * 1.001,
        1e11,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ] {
        assert!(
            !is_crown_coeff_safe(value),
            "value {value:?} should be rejected as an unsafe CROWN coefficient"
        );
    }
}

// ===== gemm_interval_sound: enclosure soundness =====

#[test]
fn sound_dot_gamma_refuses_unrecoverable_contraction_lengths() {
    let last_supported = (1usize << 23) - 1;
    assert!(sound_dot_gamma(last_supported).is_ok());
    assert!(matches!(
        sound_dot_gamma(1usize << 23),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert!(matches!(
        sound_dot_gamma(usize::MAX),
        Err(NyError::UnsupportedConfiguration(_))
    ));
}

/// Exact-ish reference enclosure of `{A@B : A∈[a_lo,a_hi], B∈[b_lo,b_hi]}`.
///
/// For each output (i,j), the global min/max over the box is attained at the
/// per-term corners (each `a_ik·b_kj` is bilinear and the terms are independent
/// across `k`), so summing the per-term min/max gives the *exact* output range.
/// `f32·f32` is exact in `f64`, and the `k`-term `f64` sum error (~k·2⁻⁵³) is
/// dwarfed by `gemm_interval_sound`'s f32 radius (~2⁻²³·magnitude), so a nearest
/// `f64` accumulation is a faithful oracle for the containment assertions below.
fn interval_matmul_oracle(
    m: usize,
    k: usize,
    n: usize,
    a_lo: &[f32],
    a_hi: &[f32],
    b_lo: &[f32],
    b_hi: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![0.0f64; m * n];
    let mut hi = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut slo = 0.0f64;
            let mut shi = 0.0f64;
            for l in 0..k {
                let al = f64::from(a_lo[i * k + l]);
                let ah = f64::from(a_hi[i * k + l]);
                let bl = f64::from(b_lo[l * n + j]);
                let bh = f64::from(b_hi[l * n + j]);
                let p = [al * bl, al * bh, ah * bl, ah * bh];
                let mut tmin = p[0];
                let mut tmax = p[0];
                for &v in &p[1..] {
                    tmin = tmin.min(v);
                    tmax = tmax.max(v);
                }
                slo += tmin;
                shi += tmax;
            }
            lo[i * n + j] = slo;
            hi[i * n + j] = shi;
        }
    }
    (lo, hi)
}

/// Tiny deterministic LCG so the randomized soundness sweep is reproducible
/// without a `proptest`/`rand` dependency in ny-core.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform f32 in [-scale, scale].
    fn signed(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (u * 2.0 - 1.0) * scale
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

#[test]
fn gemm_interval_sound_encloses_naive_cpu_engine() {
    let engine = NaiveCpuGemmEngine;
    let mut rng = Lcg(0x5DEECE66D ^ 0xC0FFEE);
    // Span normal, subnormal-producing, and overflow-producing magnitudes so the
    // property covers the underflow/overflow regimes that broke earlier versions.
    let scales = [
        0.0f32,
        1.0,
        8.0,
        1e3,
        1e-20,
        1e-30,
        2.0f32.powi(-80),
        1e20,
        1e30,
    ];

    for case in 0..4000u32 {
        let m = 1 + rng.below(5);
        let k = 1 + rng.below(12);
        let n = 1 + rng.below(5);
        let scale = scales[rng.below(scales.len())];

        // Random box; sometimes a point interval (zero width) per element, and
        // sometimes a wide one. width is nonnegative so lo<=hi always holds.
        let mut a_lo = vec![0.0f32; m * k];
        let mut a_hi = vec![0.0f32; m * k];
        let mut b_lo = vec![0.0f32; k * n];
        let mut b_hi = vec![0.0f32; k * n];
        let fill = |rng: &mut Lcg, lo: &mut [f32], hi: &mut [f32]| {
            for idx in 0..lo.len() {
                let c = rng.signed(scale);
                // 25% of the time a degenerate point interval.
                let w = if rng.below(4) == 0 {
                    0.0
                } else {
                    rng.signed(scale).abs()
                };
                lo[idx] = c - w;
                hi[idx] = c + w;
            }
        };
        fill(&mut rng, &mut a_lo, &mut a_hi);
        fill(&mut rng, &mut b_lo, &mut b_hi);

        let (c_lo, c_hi) = engine
            .gemm_interval_sound(m, k, n, &a_lo, &a_hi, &b_lo, &b_hi)
            .expect("interval gemm");
        let (o_lo, o_hi) = interval_matmul_oracle(m, k, n, &a_lo, &a_hi, &b_lo, &b_hi);

        for idx in 0..(m * n) {
            // NaN is never an acceptable bound; ±inf is sound for overflow.
            assert!(
                !c_lo[idx].is_nan() && !c_hi[idx].is_nan(),
                "case {case}: NaN bound at {idx}"
            );
            assert!(
                c_lo[idx] <= c_hi[idx],
                "case {case}: lower exceeds upper at {idx}: {} > {}",
                c_lo[idx],
                c_hi[idx]
            );
            // The enclosure must contain the true output range (the moat: a
            // sound bound never excludes a reachable product).
            assert!(
                f64::from(c_lo[idx]) <= o_lo[idx],
                "case {case} idx {idx}: UNSOUND lower {} > true min {} (m={m},k={k},n={n})",
                c_lo[idx],
                o_lo[idx]
            );
            assert!(
                f64::from(c_hi[idx]) >= o_hi[idx],
                "case {case} idx {idx}: UNSOUND upper {} < true max {} (m={m},k={k},n={n})",
                c_hi[idx],
                o_hi[idx]
            );
        }
    }
}

#[test]
fn gemm_interval_sound_point_interval_matches_gemm_within_radius() {
    // For a point interval (lo == hi) the enclosure must bracket the exact
    // real product, and be tight (radius is only the f32 rounding term).
    let engine = NaiveCpuGemmEngine;
    let a = vec![0.5f32, -1.5, 2.0, 0.25, -0.75, 1.0]; // 2x3
    let b = vec![
        1.0f32, -2.0, 0.5, 3.0, // 3x4
        -1.0, 0.0, 2.5, -0.5, //
        0.75, -1.25, 1.5, 2.0,
    ];
    let (m, k, n) = (2, 3, 4);
    let (c_lo, c_hi) = engine
        .gemm_interval_sound(m, k, n, &a, &a, &b, &b)
        .expect("interval gemm");
    let exact = engine
        .gemm_f64(
            m,
            k,
            n,
            &a.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &b.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        )
        .expect("f64 ref");
    for idx in 0..(m * n) {
        assert!(
            f64::from(c_lo[idx]) <= exact[idx] && f64::from(c_hi[idx]) >= exact[idx],
            "idx {idx}: exact {} not in [{}, {}]",
            exact[idx],
            c_lo[idx],
            c_hi[idx]
        );
        // Tightness: width is a few ULPs, not blown up.
        let width = f64::from(c_hi[idx]) - f64::from(c_lo[idx]);
        assert!(
            width <= 1e-3,
            "idx {idx}: point-interval width too loose: {width}"
        );
    }
}

/// Regression: adversarial inputs that the soundness verification found broke an
/// earlier (relative-error-only, no overflow guard) version of the enclosure.
/// Two classes:
///   - subnormal underflow: tiny products flush to 0 in f32, so a purely
///     multiplicative γ_k radius collapsed to 0 and EXCLUDED the true positive
///     product. The additive `c·k·η` term must now cover it.
///   - overflow: products past f32::MAX must yield the trivial sound enclosure
///     [-inf, +inf], never NaN.
#[test]
fn gemm_interval_sound_survives_adversarial_underflow_and_overflow() {
    let engine = NaiveCpuGemmEngine;
    let smallest_sub = f32::from_bits(1); // 2^-149
    let min_normal = f32::MIN_POSITIVE; // 2^-126
    let p = |x: f32| (1usize, 1usize, 1usize, vec![x], vec![x], vec![x], vec![x]);

    // (m, k, n, a_lo, a_hi, b_lo, b_hi)
    let mut cases: Vec<(usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
        // subnormal-product underflow (point intervals: radius is rounding-only)
        p(2.0f32.powi(-80)),            // a=b=2^-80 -> 2^-160 flushes
        p(2.0f32.powi(-100)),           // 2^-200 flushes
        p(smallest_sub),                // (2^-149)^2 flushes
        p(min_normal),                  // (2^-126)^2 = 2^-252 flushes
        p(f32::from_bits(0x0010_0000)), // a subnormal
        // asymmetric tiny mix
        (
            1,
            1,
            1,
            vec![2.0f32.powi(-80)],
            vec![2.0f32.powi(-80)],
            vec![2.0f32.powi(-75)],
            vec![2.0f32.powi(-75)],
        ),
        // width-3-ulp interval just above min-normal
        (
            1,
            1,
            1,
            vec![min_normal],
            vec![f32::from_bits(min_normal.to_bits() + 3)],
            vec![min_normal],
            vec![f32::from_bits(min_normal.to_bits() + 3)],
        ),
        // overflow: f32::MAX squared, and mixed huge+tiny dot
        p(f32::MAX),
        (
            1,
            2,
            1,
            vec![1e20, 1.0],
            vec![1e20, 1.0],
            vec![1e20, 1e-20],
            vec![1e20, 1e-20],
        ),
    ];
    // bulk subnormal-range dot product, k = 100
    {
        let k = 100;
        let lo = vec![min_normal; k];
        let hi = vec![f32::from_bits(min_normal.to_bits() + min_normal.to_bits()); k]; // ~2*min_normal
        cases.push((1, k, 1, lo.clone(), hi.clone(), lo, hi));
    }

    for (ci, (m, k, n, a_lo, a_hi, b_lo, b_hi)) in cases.into_iter().enumerate() {
        let (c_lo, c_hi) = engine
            .gemm_interval_sound(m, k, n, &a_lo, &a_hi, &b_lo, &b_hi)
            .expect("interval gemm");
        let (o_lo, o_hi) = interval_matmul_oracle(m, k, n, &a_lo, &a_hi, &b_lo, &b_hi);
        for idx in 0..(m * n) {
            assert!(
                !c_lo[idx].is_nan() && !c_hi[idx].is_nan(),
                "attack {ci} idx {idx}: NaN bound ({}, {})",
                c_lo[idx],
                c_hi[idx]
            );
            assert!(
                c_lo[idx] <= c_hi[idx],
                "attack {ci} idx {idx}: lower {} > upper {}",
                c_lo[idx],
                c_hi[idx]
            );
            // The enclosure must contain the true output range. (-inf/+inf bounds
            // from the overflow guard trivially satisfy this for finite oracle.)
            assert!(
                f64::from(c_lo[idx]) <= o_lo[idx],
                "attack {ci} idx {idx}: UNSOUND lower {} > true min {:e}",
                c_lo[idx],
                o_lo[idx]
            );
            assert!(
                f64::from(c_hi[idx]) >= o_hi[idx],
                "attack {ci} idx {idx}: UNSOUND upper {} < true max {:e}",
                c_hi[idx],
                o_hi[idx]
            );
        }
    }
}

/// A DAZ backend can erase a subnormal midpoint before multiplying it by a
/// large normal operand. The amplified FLT_MIN floor must still contain the
/// real product; an `8*k*ETA` floor is many orders of magnitude too small.
#[test]
fn gemm_interval_sound_survives_daz_operand_flush() {
    struct MockDazGemmEngine;
    impl GemmEngine for MockDazGemmEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            let z = |x: f32| {
                if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                    0.0
                } else {
                    x
                }
            };
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        acc = z(acc + z(z(a[i * k + l]) * z(b[l * n + j])));
                    }
                    out[i * n + j] = acc;
                }
            }
            Ok(out)
        }
    }

    let cases = [
        (2.0f32.powi(-130), 2.0f32.powi(20)),
        (2.0f32.powi(-130), -(2.0f32.powi(24))),
        (f32::from_bits(1), 2.0f32.powi(120)),
        (2.0f32.powi(-140), 2.0f32.powi(60)),
    ];
    for (a, b) in cases {
        let (lo, hi) = MockDazGemmEngine
            .gemm_interval_sound(1, 1, 1, &[a], &[a], &[b], &[b])
            .expect("DAZ interval GEMM");
        let exact = f64::from(a) * f64::from(b);
        assert!(
            f64::from(lo[0]) <= exact && exact <= f64::from(hi[0]),
            "DAZ operand flush excluded {a:e}*{b:e}={exact:e} from [{:e}, {:e}]",
            lo[0],
            hi[0]
        );
    }
}

#[cfg(test)]
mod moat_soundness_reverify {
    use super::*;
    use crate::gemm::GemmEngine;

    const ETA: f64 = f64::from_bits(0x36A0_0000_0000_0000); // 2^-149 (min f32 subnormal)
    const F32MAX: f32 = f32::MAX;

    // Exact-ish reference: true product range of [a_lo,a_hi]*[b_lo,b_hi] summed over k,
    // computed in f64 (exact for these magnitudes) for a single 1x1 output (k terms).
    fn true_range(a_lo: &[f32], a_hi: &[f32], b_lo: &[f32], b_hi: &[f32]) -> (f64, f64) {
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        for i in 0..a_lo.len() {
            let (al, ah, bl, bh) = (
                a_lo[i] as f64,
                a_hi[i] as f64,
                b_lo[i] as f64,
                b_hi[i] as f64,
            );
            let c = [al * bl, al * bh, ah * bl, ah * bh];
            lo += c.iter().cloned().fold(f64::INFINITY, f64::min);
            hi += c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }
        (lo, hi)
    }

    fn check(
        a_lo: Vec<f32>,
        a_hi: Vec<f32>,
        b_lo: Vec<f32>,
        b_hi: Vec<f32>,
        k: usize,
        label: &str,
    ) {
        let eng = NaiveCpuGemmEngine;
        let (clo, chi) = eng
            .gemm_interval_sound(1, k, 1, &a_lo, &a_hi, &b_lo, &b_hi)
            .unwrap();
        let (tlo, thi) = true_range(&a_lo, &a_hi, &b_lo, &b_hi);
        assert!(!clo[0].is_nan() && !chi[0].is_nan(), "{label}: NaN bound");
        // lower bound must not exceed true lo; upper must not fall below true hi
        if clo[0].is_finite() {
            assert!(
                clo[0] as f64 <= tlo + tlo.abs() * 1e-12 + 1e-300,
                "{label}: clo {} > true_lo {}",
                clo[0],
                tlo
            );
        }
        if chi[0].is_finite() {
            assert!(
                chi[0] as f64 >= thi - thi.abs() * 1e-12 - 1e-300,
                "{label}: chi {} < true_hi {}",
                chi[0],
                thi
            );
        }
    }

    #[test]
    fn subnormal_point_intervals() {
        let e = ETA as f32;
        for k in [1usize, 4, 64, 1024] {
            check(
                vec![e; k],
                vec![e; k],
                vec![e; k],
                vec![e; k],
                k,
                "eta point",
            );
        }
    }

    #[test]
    fn mixed_subnormal_and_near_max() {
        // (d) one operand subnormal, another near f32::MAX in same dot
        let sub = ETA as f32;
        let big = f32::from_bits(0x7E00_0000); // ~2^125
        let k = 4;
        check(
            vec![sub, big, sub, big],
            vec![sub, big, sub, big],
            vec![big, sub, big, sub],
            vec![big, sub, big, sub],
            k,
            "mixed sub/near-max",
        );
    }

    #[test]
    fn overflow_to_inf_is_sound() {
        // product overflows f32 -> guard must give [-inf,+inf]
        let (clo, chi) = NaiveCpuGemmEngine
            .gemm_interval_sound(1, 4, 1, &[1e30; 4], &[1e30; 4], &[1e30; 4], &[1e30; 4])
            .unwrap();
        assert_eq!(chi[0], f32::INFINITY);
        assert_eq!(clo[0], f32::NEG_INFINITY);
        assert!(!clo[0].is_nan() && !chi[0].is_nan());
    }

    #[test]
    fn true_hi_exceeds_f32max() {
        // a=[1,F32MAX], b=2 -> true hi = 2*F32MAX > F32MAX; must give +inf (sound), never NaN
        check(
            vec![1.0],
            vec![F32MAX],
            vec![2.0],
            vec![2.0],
            1,
            "overflow hi",
        );
    }

    #[test]
    fn widely_separated_exponents() {
        let lo = -(2f32.powi(-30));
        let hi = 2f32.powi(100);
        check(vec![lo], vec![hi], vec![1.0], vec![1.0], 1, "wide a, b=1");
        check(
            vec![-(ETA as f32)],
            vec![2f32.powi(127)],
            vec![1.0],
            vec![1.0],
            1,
            "a=[-eta,2^127]",
        );
    }

    #[test]
    fn adjacent_f32_intervals_near_max() {
        let l = f32::from_bits(0x7F7F_FFFE);
        let h = F32MAX;
        check(
            vec![l],
            vec![h],
            vec![1.0],
            vec![1.0],
            1,
            "adjacent near max",
        );
    }

    #[test]
    fn point_f32max_carries_eta_floor() {
        // [F32MAX]*[1]: pi=F32MAX, radius>=ETA pushes pi+rad past F32MAX -> chi=+inf (sound)
        check(
            vec![F32MAX],
            vec![F32MAX],
            vec![1.0],
            vec![1.0],
            1,
            "F32MAX point",
        );
    }
}

// ===== crown_aw_error_step: per-layer coefficient-error propagation =====

#[test]
fn crown_aw_error_step_encloses_true_coefficient_range() {
    let engine = NaiveCpuGemmEngine;
    let mut rng = Lcg(0xA5A5_1234 ^ 0xBEEF);

    for case in 0..2000u32 {
        let m = 1 + rng.below(5);
        let k = 1 + rng.below(14);
        let n = 1 + rng.below(5);
        let scale = [1.0f32, 8.0, 1e-20, 1e3][rng.below(4)];

        let a: Vec<f32> = (0..m * k).map(|_| rng.signed(scale)).collect();
        // Nonnegative incoming error (the exact coeff is in [a - a_err, a + a_err]).
        let a_err: Vec<f32> = (0..m * k).map(|_| rng.signed(scale * 0.1).abs()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| rng.signed(scale)).collect();

        let (a_new, a_err_new) = engine
            .crown_aw_error_step(m, k, n, &a, &a_err, &w)
            .expect("aw error step");
        assert_eq!(a_new.len(), m * n);
        assert_eq!(a_err_new.len(), m * n);

        // Exact f64 corner range of Σ_l a_exact[i,l]·w[l,j] over
        // a_exact[i,l] ∈ [a−a_err, a+a_err]. (f32·f32 is exact in f64.)
        for i in 0..m {
            for j in 0..n {
                let (mut tmin, mut tmax) = (0.0f64, 0.0f64);
                for l in 0..k {
                    let amin = f64::from(a[i * k + l]) - f64::from(a_err[i * k + l]);
                    let amax = f64::from(a[i * k + l]) + f64::from(a_err[i * k + l]);
                    let wv = f64::from(w[l * n + j]);
                    tmin += (amin * wv).min(amax * wv);
                    tmax += (amin * wv).max(amax * wv);
                }
                let idx = i * n + j;
                assert!(a_err_new[idx] >= 0.0 && a_err_new[idx].is_finite());
                let lo = f64::from(a_new[idx]) - f64::from(a_err_new[idx]);
                let hi = f64::from(a_new[idx]) + f64::from(a_err_new[idx]);
                assert!(
                    lo <= tmin,
                    "case {case} ({m}x{k}x{n}) [{i},{j}]: UNSOUND lower {lo} > true min {tmin}"
                );
                assert!(
                    hi >= tmax,
                    "case {case} ({m}x{k}x{n}) [{i},{j}]: UNSOUND upper {hi} < true max {tmax}"
                );
            }
        }
    }
}

/// A point coefficient (`a_err = 0`) still gets a nonzero error covering the
/// GEMM's own f32 rounding, so `a_new ± a_err_new` brackets the exact `a@w`.
#[test]
fn crown_aw_error_step_point_coeff_brackets_exact_product() {
    let engine = NaiveCpuGemmEngine;
    let (m, k, n) = (3, 6, 4);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25)
        .collect();
    let zero_err = vec![0.0f32; m * k];
    let (a_new, a_err_new) = engine
        .crown_aw_error_step(m, k, n, &a, &zero_err, &w)
        .expect("aw");
    let exact = engine
        .gemm_f64(
            m,
            k,
            n,
            &a.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &w.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        )
        .expect("f64");
    for idx in 0..(m * n) {
        let lo = f64::from(a_new[idx]) - f64::from(a_err_new[idx]);
        let hi = f64::from(a_new[idx]) + f64::from(a_err_new[idx]);
        assert!(
            lo <= exact[idx] && hi >= exact[idx],
            "idx {idx}: {exact:?} not bracketed"
        );
    }
}

/// DAZ/FTZ soundness regression (#gpu-metal-daz): a Metal-style engine that flushes
/// subnormal OPERANDS to zero *before* the multiply (denormals-are-zero) loses the
/// entire product `|a|·|w|` — up to `|w|·FLT_MIN`, arbitrarily large with `|w|`. The
/// weight-INDEPENDENT underflow floor cannot cover this; only the weight-AMPLIFIED
/// `flushacc·slack·FLT_MIN` term now added to `crown_aw_error_step` does.
///
/// Pre-fix this test FAILS: the point coefficient flushes to 0 and the certified
/// error was only ~`8k·2^-149`, so the interval `[−2^-146, 2^-146]` excluded the true
/// (normal-range) product. Post-fix the interval stays OUTWARD. `NaiveCpuGemmEngine`
/// uses gradual underflow, so it does not exercise this — hence the DAZ mock.
#[test]
fn crown_aw_error_step_daz_operand_flush_stays_outward() {
    struct MockDazGemmEngine;
    impl GemmEngine for MockDazGemmEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            // subnormals -> 0, both as operands (DAZ) and as products/partials (FTZ).
            let z = |x: f32| {
                if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                    0.0
                } else {
                    x
                }
            };
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        acc = z(acc + z(z(a[i * k + l]) * z(b[l * n + j])));
                    }
                    out[i * n + j] = acc;
                }
            }
            Ok(out)
        }
    }

    // (subnormal operand a, large weight w): the exact product is a NORMAL f32 that
    // DAZ silently drops to 0. Exact in f64 (single k=1 term, f32·f32 ⊂ f64).
    let cases: &[(f32, f32)] = &[
        (2.0f32.powi(-130), 2.0f32.powi(20)),    // -> 2^-110
        (2.0f32.powi(-130), -(2.0f32.powi(24))), // -> -2^-106
        (f32::from_bits(1), 2.0f32.powi(120)),   // smallest subnormal 2^-149 -> 2^-29
        (2.0f32.powi(-140), 2.0f32.powi(60)),    // -> 2^-80
    ];
    for &(av, wv) in cases {
        let (a_new, a_err_new) = MockDazGemmEngine
            .crown_aw_error_step(1, 1, 1, &[av], &[0.0], &[wv])
            .expect("aw daz");
        let exact = f64::from(av) * f64::from(wv); // exact (f32·f32 ⊂ f64)
        let lo = f64::from(a_new[0]) - f64::from(a_err_new[0]);
        let hi = f64::from(a_new[0]) + f64::from(a_err_new[0]);
        assert!(
            a_err_new[0].is_finite() && lo <= exact && hi >= exact,
            "DAZ operand flush UNSOUND: a={av:e} w={wv:e} exact={exact:e} \
             not in [{lo:e}, {hi:e}] (a_new={}, err={})",
            a_new[0],
            a_err_new[0]
        );
    }
}

// ===== crown_activation_error_step: per-neuron relaxation error =====

#[test]
fn crown_activation_error_step_brackets_relaxed_coefficient() {
    let mut rng = Lcg(0x4C71_0A75 ^ 0x1234);
    for case in 0..3000u32 {
        let num_outputs = 1 + rng.below(4);
        let num_neurons = 1 + rng.below(6);
        let n = num_outputs * num_neurons;
        let scale = [1.0f32, 8.0, 1e3][rng.below(3)];

        let lower_a: Vec<f32> = (0..n).map(|_| rng.signed(scale)).collect();
        let upper_a: Vec<f32> = (0..n).map(|_| rng.signed(scale)).collect();
        let lower_a_err: Vec<f32> = (0..n).map(|_| rng.signed(scale * 0.1).abs()).collect();
        let upper_a_err: Vec<f32> = (0..n).map(|_| rng.signed(scale * 0.1).abs()).collect();
        let lower_slope: Vec<f32> = (0..num_neurons).map(|_| rng.signed(2.0)).collect();
        let upper_slope: Vec<f32> = (0..num_neurons).map(|_| rng.signed(2.0)).collect();

        let (nla, nua, nle, nue) = crown_activation_error_step(
            num_outputs,
            num_neurons,
            &lower_a,
            &upper_a,
            &lower_a_err,
            &upper_a_err,
            &lower_slope,
            &upper_slope,
        )
        .expect("activation error step");

        // compose_lower(a) = a·(a>=0?ls:us); compose_upper(a) = a·(a>=0?us:ls).
        // Over a_exact ∈ [a−err, a+err] the relaxed coeff (piecewise-linear, kink
        // at 0) is extremal at the endpoints or at 0; new_a ± new_err must contain
        // all of them.
        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let idx = j * num_neurons + i;
                let ls = f64::from(lower_slope[i]);
                let us = f64::from(upper_slope[i]);
                let cl = |a: f64| a * if a >= 0.0 { ls } else { us };
                let cu = |a: f64| a * if a >= 0.0 { us } else { ls };

                for (a0, err, new_a, new_err, comp) in [
                    (
                        lower_a[idx],
                        lower_a_err[idx],
                        nla[idx],
                        nle[idx],
                        &cl as &dyn Fn(f64) -> f64,
                    ),
                    (
                        upper_a[idx],
                        upper_a_err[idx],
                        nua[idx],
                        nue[idx],
                        &cu as &dyn Fn(f64) -> f64,
                    ),
                ] {
                    let a = f64::from(a0);
                    let e = f64::from(err);
                    let lo = f64::from(new_a) - f64::from(new_err);
                    let hi = f64::from(new_a) + f64::from(new_err);
                    let mut samples = vec![a - e, a + e, a];
                    if a - e <= 0.0 && 0.0 <= a + e {
                        samples.push(0.0);
                    }
                    for s in samples {
                        let v = comp(s);
                        assert!(
                            lo <= v + 1e-9 && v <= hi + 1e-9,
                            "case {case} [{j},{i}]: relaxed coeff {v} (a_exact={s}) not in [{lo}, {hi}]"
                        );
                    }
                }
            }
        }
    }
}

// ── FTZ-safe underflow floor (Metal soundness, #gpu-metal) ──────────────────
// Property (1) FTZ-survival is the Trust `#[ensures]`; property (2) the over-bound
// (>= flush_points·FLT_MIN AND >= the prior 8·ETA subnormal floor) is checked here.

#[test]
fn ftz_floor_is_normal_range_survives_flush_to_zero() {
    // Every result must be a NORMAL f32 (>= smallest normal 2^-126), so Metal's
    // flush-to-zero cannot silently zero it. Covers 1, small, and large counts.
    for &n in &[0u32, 1, 2, 8, 64, 1024, 1 << 20, u32::MAX] {
        let f = ftz_safe_underflow_floor(n);
        assert!(
            f >= f32::MIN_POSITIVE,
            "floor {f} for n={n} is subnormal/zero — would flush on Metal (UNSOUND)"
        );
        assert!(
            f.is_finite() && f > 0.0,
            "floor must be finite positive, got {f}"
        );
    }
}

#[test]
fn ftz_floor_over_bounds_flush_loss_and_prior_eta_floor() {
    // Prior subnormal floor the wgpu resident CROWN used: 8·ETA (ETA = 2^-149).
    const ETA: f64 = 1.401_298_5e-45; // 2^-149, smallest positive f32 subnormal
    for &n in &[1u32, 8, 64, 4096] {
        let f = f64::from(ftz_safe_underflow_floor(n));
        let nn = f64::from(n);
        // (2) over-bounds the max FTZ flush loss (<= n·FLT_MIN).
        assert!(
            f >= nn * f64::from(f32::MIN_POSITIVE),
            "floor must cover n·FLT_MIN flush loss for n={n}"
        );
        // dominates the old 8·n·ETA subnormal floor (so still sound on Vulkan).
        assert!(
            f >= 8.0 * nn * ETA,
            "floor must dominate the prior 8·n·ETA floor for n={n}"
        );
    }
}

/// The DEFAULT declaration every engine inherits must be pointwise identical to
/// the historical single-constant policy `m*k*p >= SOUND_F64_GEMM_DEFAULT_MIN_MACS`.
/// If this drifts, arming the engine-aware floor silently changes cuBLAS.
#[test]
fn constant_floor_admission_is_exactly_the_historical_mac_product_test() {
    let policy = |m: usize, k: usize, p: usize| {
        m.saturating_mul(k).saturating_mul(p) >= SOUND_F64_GEMM_DEFAULT_MIN_MACS
    };
    let shapes: &[(usize, usize, usize)] = &[
        (0, 0, 0),
        (0, 4096, 4096),
        (4096, 0, 4096),
        (1, 1, 1),
        (1, 1, SOUND_F64_GEMM_DEFAULT_MIN_MACS - 1),
        (1, 1, SOUND_F64_GEMM_DEFAULT_MIN_MACS),
        (1, 1, SOUND_F64_GEMM_DEFAULT_MIN_MACS + 1),
        (4, 16, 16),
        (9, 1, 9),
        (64, 512, 512),
        (63, 512, 512),
        (1, 4096, 4096),
        (usize::MAX, usize::MAX, usize::MAX),
        (usize::MAX, 1, 1),
    ];
    for &(m, k, p) in shapes {
        assert_eq!(
            SoundF64GemmAdmission::CONSTANT_FLOOR.admits(m, k, p),
            policy(m, k, p),
            "{m}x{k}x{p}"
        );
        // `sanitized()` must not perturb the default either.
        assert_eq!(
            SoundF64GemmAdmission::CONSTANT_FLOOR
                .sanitized()
                .admits(m, k, p),
            policy(m, k, p),
            "{m}x{k}x{p} (sanitized)"
        );
    }
}

/// An engine that does not override the trait method gets the constant floor.
#[test]
fn undeclared_engine_inherits_the_constant_floor() {
    struct Undeclared;
    impl GemmEngine for Undeclared {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test".into()))
        }
    }
    assert_eq!(
        Undeclared.sound_f64_deadline_admission(),
        SoundF64GemmAdmission::CONSTANT_FLOOR
    );
    assert_eq!(
        NaiveCpuGemmEngine.sound_f64_deadline_admission(),
        SoundF64GemmAdmission::CONSTANT_FLOOR
    );
}

/// A zeroed declaration must not become an "admit everything" declaration.
#[test]
fn sanitized_declaration_refuses_degenerate_operands() {
    let zeroed = SoundF64GemmAdmission {
        min_macs: 0,
        min_rows: 0,
        min_contraction: 0,
        min_columns: 0,
        small_contraction_below: 0,
        small_contraction_max_output: usize::MAX,
    }
    .sanitized();
    assert!(!zeroed.admits(0, 4, 4), "empty A rows");
    assert!(!zeroed.admits(4, 0, 4), "empty contraction");
    assert!(!zeroed.admits(4, 4, 0), "empty output columns");
    assert!(zeroed.admits(1, 1, 1));
}

/// The shape rules are independent: each one alone can decline a product the
/// MAC count alone would admit.
#[test]
fn shape_rules_decline_independently_of_the_mac_count() {
    let declaration = SoundF64GemmAdmission {
        min_macs: 1 << 10,
        min_rows: 4,
        min_contraction: 4,
        min_columns: 1,
        small_contraction_below: 16,
        small_contraction_max_output: 1 << 16,
    };
    // 16.7M MACs — far above min_macs — declined for shape reasons alone.
    assert!(!declaration.admits(4096, 1, 4096), "k==1");
    assert!(!declaration.admits(131072, 2, 64), "k==2");
    assert!(!declaration.admits(1, 4096, 4096), "m==1");
    assert!(!declaration.admits(2048, 4, 2048), "small k, large output");
    // The same small k is fine when the output is small.
    assert!(declaration.admits(256, 4, 256));
    assert!(declaration.admits(64, 4, 64));
    // A large contraction is never subject to the small-k rule.
    assert!(declaration.admits(4, 4194304, 1));
    // Below the declared crossover.
    assert!(!declaration.admits(9, 8, 8), "576 MACs < 1024");
    assert!(declaration.admits(4, 16, 16), "1024 MACs");
}
