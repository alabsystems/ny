// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Charge declared MODEL error before the ordinary f64 weight contraction.
//!
//! For |A*-A|<=E and |W*-W|<=rho|W|, the coefficient deviation before
//! contraction is bounded by E'=(1+rho)E+rho|A|. The existing contraction
//! then encloses A*W* using its unchanged rounding term and E'@|W|.
//! The true-bias uncertainty is separately delta_b*sum(|A|+E), with the
//! ORIGINAL incoming E. It applies even when no stored bias vector exists.
//! Every scalar charge rounds upward; invalid or overflowing arithmetic
//! refuses. This does not authenticate a loader that discarded its error.

use ny_core::{CertifiedWeightError, GemmEngine, GpuCrownLayer, NyError, Result};

fn invalid() -> NyError {
    NyError::SoundnessRefusal("cuda certified weight charge: invalid/overflowed error".into())
}

fn finite_nonnegative(x: f64) -> Result<f64> {
    if x.is_finite() && x >= 0.0 {
        Ok(x)
    } else {
        Err(invalid())
    }
}

// On the existing IEEE binary64 host arithmetic contract, one outward step
// encloses the exact nonnegative operation, including gradual underflow.
// Exact-zero operands are kept exact rather than accumulating artificial debt.
pub(super) fn add_up(a: f64, b: f64) -> Result<f64> {
    finite_nonnegative(a)?;
    finite_nonnegative(b)?;
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    finite_nonnegative(ny_core::dd::next_up_f64(a + b))
}

pub(super) fn mul_up(a: f64, b: f64) -> Result<f64> {
    finite_nonnegative(a)?;
    finite_nonnegative(b)?;
    if a == 0.0 || b == 0.0 {
        return Ok(0.0);
    }
    finite_nonnegative(ny_core::dd::next_up_f64(a * b))
}

// The legacy f32 predicates/comparison conversion can erase a nonzero
// subnormal under DAZ. Classify by representation before ANY FP operation.
// Signed zero is the exact legacy path and retains its arithmetic/poll order.
fn radii(declared: CertifiedWeightError) -> Result<Option<(f64, f64)>> {
    let bits = [
        declared.weight_rel_err.to_bits(),
        declared.bias_abs_err.to_bits(),
    ];
    if bits.iter().all(|&b| b & 0x7fff_ffff == 0) {
        return Ok(None);
    }
    for b in bits {
        let magnitude = b & 0x7fff_ffff;
        if magnitude >= 0x7f80_0000 || (b >> 31 != 0 && magnitude != 0) {
            return Err(invalid());
        }
    }
    ny_core::require_f64_interval_proof_environment()?;
    Ok(Some((
        ny_core::f32_to_f64_exact(declared.weight_rel_err),
        ny_core::f32_to_f64_exact(declared.bias_abs_err),
    )))
}

/// Upward computation of the existing center-rounding allowance, followed by
/// an upward addition to the independently propagated error. The centers and
/// existing gamma/real-factor formula are unchanged.
pub(super) fn rounding_up(gamma: f64, magnitude: f64, factor: f64, additive: f64) -> Result<f64> {
    add_up(mul_up(mul_up(gamma, magnitude)?, factor)?, additive)
}

pub(super) fn validate_layers(layers: &[GpuCrownLayer]) -> Result<()> {
    for layer in layers {
        let error = match layer {
            GpuCrownLayer::Linear { cert_err, .. } | GpuCrownLayer::Conv2d { cert_err, .. } => {
                cert_err
            }
            _ => continue,
        };
        radii(*error)?;
    }
    Ok(())
}

/// Mutates only the uncommitted frontier. No allocation, backend dispatch,
/// or independently publishable result. Caller already folded the stored
/// bias with the original E; all later failures discard this entire frontier.
pub(super) fn charge<E: GemmEngine + ?Sized>(
    eng: &E,
    rows: usize,
    width: usize,
    centers: &[f64],
    errors: &mut [f64],
    bias_errors: &mut [f64],
    declared: CertifiedWeightError,
) -> Result<()> {
    // Bit-only exactness precedes validation, conversion, and environment
    // probing; nonzero radii require the IEEE host contract before mutation.
    let Some((rho, delta)) = radii(declared)? else {
        return Ok(());
    };
    let cells = rows.checked_mul(width).ok_or_else(invalid)?;
    if rows == 0
        || width == 0
        || centers.len() != cells
        || errors.len() != cells
        || bias_errors.len() != rows
    {
        return Err(NyError::InvalidSpec(
            "cuda certified weight charge: shape mismatch".into(),
        ));
    }
    eng.poll_crown_backward_deadline()?;
    for (row, bias_error) in bias_errors.iter_mut().enumerate() {
        finite_nonnegative(*bias_error)?;
        let mut model_bias = 0.0;
        let base = row * width;
        for col in 0..width {
            if col % super::BOUNDED_ROWS_VALIDATION_CHUNK == 0 {
                eng.poll_crown_backward_deadline()?;
            }
            let index = base + col;
            let original_error = finite_nonnegative(errors[index])?;
            let magnitude = finite_nonnegative(centers[index].abs())?;
            let exact_magnitude = add_up(magnitude, original_error)?;
            if delta != 0.0 {
                model_bias = add_up(model_bias, mul_up(delta, exact_magnitude)?)?;
            }
            if rho != 0.0 {
                errors[index] = add_up(original_error, mul_up(rho, exact_magnitude)?)?;
            }
        }
        if delta != 0.0 {
            *bias_error = add_up(*bias_error, model_bias)?;
        }
    }
    eng.poll_crown_backward_deadline()
}

#[cfg(test)]
mod tests {
    use super::super::{backward_layers_f64, concretize_frontier, Frontier};
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Default)]
    struct Cpu {
        polls: AtomicUsize,
        fail_at: AtomicUsize,
        calls: AtomicUsize,
    }
    impl GemmEngine for Cpu {
        fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
            panic!("certified f64 fold must not use f32 GEMM")
        }
        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    for h in 0..k {
                        result[i * n + j] += a[i * k + h] * b[h * n + j];
                    }
                }
            }
            Ok(result)
        }
        fn poll_crown_backward_deadline(&self) -> Result<()> {
            let p = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if p == self.fail_at.load(Ordering::SeqCst) {
                Err(NyError::DeadlineExceeded(
                    "injected model-error charge deadline".into(),
                ))
            } else {
                Ok(())
            }
        }
    }
    fn error(rho: f32, bias: f32) -> CertifiedWeightError {
        CertifiedWeightError {
            weight_rel_err: rho,
            bias_abs_err: bias,
        }
    }
    fn frontier(a: &[f64], e: &[f64]) -> Frontier {
        Frontier {
            lower_a: a.to_vec(),
            upper_a: a.to_vec(),
            lower_err: e.to_vec(),
            upper_err: e.to_vec(),
            lb: vec![0.0],
            ub: vec![0.0],
            lb_err: vec![0.0],
            ub_err: vec![0.0],
            dim: a.len(),
        }
    }
    fn linear(
        w: &[f32],
        bias: Option<&[f32]>,
        out: usize,
        input: usize,
        cert_err: CertifiedWeightError,
    ) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(w),
            bias: bias.map(Arc::from),
            out_features: out,
            in_features: input,
            cert_err,
        }
    }
    fn fold(
        layer: GpuCrownLayer,
        a: &[f64],
        e: &[f64],
        lo: &[f32],
        hi: &[f32],
    ) -> ny_core::GpuCrownResult {
        let f =
            backward_layers_f64(&Cpu::default(), &[layer], frontier(a, e), 1, &mut None).unwrap();
        concretize_frontier(&f, 1, lo, hi)
    }

    #[test]
    fn exact_model_error_is_bit_identical_without_new_polls() {
        let cpu = Cpu::default();
        let mut errors = [-0.0, 0.25, f64::from_bits(1)];
        let mut bias = [-0.0];
        let before = errors.map(f64::to_bits);
        charge(
            &cpu,
            1,
            3,
            &[2.0, -3.0, 0.0],
            &mut errors,
            &mut bias,
            error(-0.0, 0.0),
        )
        .unwrap();
        assert_eq!(errors.map(f64::to_bits), before);
        assert_eq!(bias[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(cpu.polls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn weight_charge_includes_cross_term_and_original_bias_magnitude() {
        let mut errors = [0.25];
        let mut bias = [0.0];
        charge(
            &Cpu::default(),
            1,
            1,
            &[-2.0],
            &mut errors,
            &mut bias,
            error(0.5, 0.125),
        )
        .unwrap();
        // Exact dyadic radii: E+rho*(|A|+E)=11/8, delta*(|A|+E)=9/32.
        assert!(errors[0] >= 11.0 / 8.0 && errors[0] < 11.0 / 8.0 + 1e-14);
        assert!(bias[0] >= 9.0 / 32.0 && bias[0] < 9.0 / 32.0 + 1e-14);
        assert!(errors[0] > 0.25 + 0.5 * 2.0); // omission of rho*E is observable.
    }

    #[test]
    fn linear_charge_encloses_all_exact_dyadic_weight_bias_and_input_corners() {
        let got = fold(
            linear(
                &[1.0, -2.0, -0.5, 0.25],
                Some(&[0.25, -0.5]),
                2,
                2,
                error(0.125, 0.0625),
            ),
            &[0.5, -1.0],
            &[0.125, 0.25],
            &[-1.0, -2.0],
            &[1.0, 2.0],
        );
        // Every operation below is exact binary64: bounded dyadic factors and <=8 additions.
        let weights = [1.0f64, -2.0, -0.5, 0.25];
        let centers = [0.5f64, -1.0];
        let radii = [0.125f64, 0.25];
        let biases = [0.25f64, -0.5];
        for bits in 0usize..1024 {
            let sign = |bit: usize| if bits & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let x = [sign(0), 2.0 * sign(1)];
            let mut truth = 0.0;
            for i in 0..2 {
                let a = centers[i] + sign(2 + i) * radii[i];
                let mut y = biases[i] + sign(8 + i) * 0.0625;
                for j in 0..2 {
                    let p = i * 2 + j;
                    y += (weights[p] + sign(4 + p) * 0.125 * weights[p].abs()) * x[j];
                }
                truth += a * y;
            }
            assert!(
                f64::from(got.lower_bounds[0]) <= truth && truth <= f64::from(got.upper_bounds[0])
            );
        }
    }

    #[test]
    fn absent_stored_bias_still_pays_real_bias_uncertainty() {
        let got = fold(
            linear(&[0.0], None, 1, 1, error(0.0, 0.5)),
            &[-2.0],
            &[0.25],
            &[0.0],
            &[0.0],
        );
        assert!(got.lower_bounds[0] <= -1.125 && got.upper_bounds[0] >= 1.125);
        assert!(got.lower_bounds[0] > -1.126 && got.upper_bounds[0] < 1.126);
    }

    #[test]
    fn conv_overlap_charge_encloses_exact_endpoint_model_with_zero_bias_storage() {
        let layer = GpuCrownLayer::Conv2d {
            weight_col: Arc::from([1.0f32, -0.5]),
            bias_expanded: None,
            out_channels: 1,
            in_channels: 1,
            kernel_h: 1,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            out_h: 1,
            out_w: 2,
            in_h: 1,
            in_w: 3,
            cert_err: error(0.25, 0.125),
        };
        let got = fold(layer, &[1.0, -2.0], &[0.25, 0.125], &[-1.0; 3], &[1.0; 3]);
        // 3 input signs, 2 incoming-coefficient signs, 2 shared kernel signs,
        // and 2 independently uncertain expanded biases. Shared-weight cases
        // are a subset of the certified elementwise contract, fully enumerated.
        for bits in 0usize..512 {
            let sign = |bit: usize| if bits & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let x = [sign(0), sign(1), sign(2)];
            let a = [1.0 + 0.25 * sign(3), -2.0 + 0.125 * sign(4)];
            let w = [1.0 + 0.25 * sign(5), -0.5 + 0.125 * sign(6)];
            let truth = a[0] * (w[0] * x[0] + w[1] * x[1] + 0.125 * sign(7))
                + a[1] * (w[0] * x[1] + w[1] * x[2] + 0.125 * sign(8));
            assert!(
                f64::from(got.lower_bounds[0]) <= truth && truth <= f64::from(got.upper_bounds[0])
            );
        }
    }

    #[test]
    fn malformed_model_errors_refuse_before_any_backend_dispatch() {
        for bad in [
            error(-1.0, 0.0),
            error(f32::NAN, 0.0),
            error(0.0, f32::INFINITY),
            error(0.0, -1.0),
        ] {
            let cpu = Cpu::default();
            assert!(backward_layers_f64(
                &cpu,
                &[linear(&[1.0], None, 1, 1, bad)],
                frontier(&[1.0], &[0.0]),
                1,
                &mut None
            )
            .is_err());
            assert_eq!(cpu.calls.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn overflowing_charge_refuses_without_a_completed_frontier() {
        let mut errors = [f64::MAX / 2.0];
        let mut bias = [0.0];
        assert!(charge(
            &Cpu::default(),
            1,
            1,
            &[f64::MAX / 2.0],
            &mut errors,
            &mut bias,
            error(2.0, 0.0)
        )
        .is_err());
        let mut errors = [0.0];
        let mut bias = [f64::MAX];
        assert!(charge(
            &Cpu::default(),
            1,
            1,
            &[1.0],
            &mut errors,
            &mut bias,
            error(0.0, 1.0)
        )
        .is_err());
    }

    #[test]
    fn tiny_positive_product_is_rounded_up_not_dropped() {
        let tiny = f64::from_bits(1);
        assert!(mul_up(tiny, 0.5).unwrap() >= tiny);
        assert!(add_up(1.0, 2.0f64.powi(-100)).unwrap() > 1.0);
    }

    #[test]
    fn late_charge_cancellation_returns_no_completion() {
        let cells = super::super::BOUNDED_ROWS_VALIDATION_CHUNK + 1;
        let cpu = Cpu::default();
        let mut errors = vec![0.25; cells];
        let mut bias = [0.0];
        charge(
            &cpu,
            1,
            cells,
            &vec![1.0; cells],
            &mut errors,
            &mut bias,
            error(0.125, 0.125),
        )
        .unwrap();
        let failed = Cpu::default();
        failed
            .fail_at
            .store(cpu.polls.load(Ordering::SeqCst), Ordering::SeqCst);
        let mut errors = vec![0.25; cells];
        let mut bias = [0.0];
        assert!(matches!(
            charge(
                &failed,
                1,
                cells,
                &vec![1.0; cells],
                &mut errors,
                &mut bias,
                error(0.125, 0.125)
            ),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(failed.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn subnormal_radius_is_nonzero_and_negative_subnormal_is_invalid() {
        let tiny = f32::from_bits(1);
        let decoded = radii(error(tiny, tiny)).unwrap().unwrap();
        assert_eq!(
            decoded.0.to_bits(),
            ny_core::f32_to_f64_exact(tiny).to_bits()
        );
        assert_eq!(
            decoded.1.to_bits(),
            ny_core::f32_to_f64_exact(tiny).to_bits()
        );
        assert!(decoded.0 > 0.0);
        let mut errors = [0.0];
        let mut bias = [0.0];
        charge(
            &Cpu::default(),
            1,
            1,
            &[1.0],
            &mut errors,
            &mut bias,
            error(tiny, tiny),
        )
        .unwrap();
        assert!(errors[0] >= decoded.0 && bias[0] >= decoded.1);
        for bad in [
            error(f32::from_bits(0x8000_0001), 0.0),
            error(0.0, f32::from_bits(0x8000_0001)),
        ] {
            assert!(radii(bad).is_err());
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn non_ieee_host_refuses_nonzero_model_error_before_mutation_or_dispatch() {
        // MXCSR is thread-local. Always restore it during unwinding as well;
        // the test never changes another test's arithmetic environment.
        std::thread::spawn(|| {
            unsafe fn get() -> u32 {
                let mut value = 0u32;
                unsafe {
                    std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut value,
                    options(nostack, preserves_flags));
                }
                value
            }
            unsafe fn set(value: u32) {
                unsafe {
                    std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &value,
                    options(nostack, preserves_flags));
                }
            }
            struct Restore(u32);
            impl Drop for Restore {
                fn drop(&mut self) {
                    unsafe {
                        set(self.0);
                    }
                }
            }
            let saved = unsafe { get() };
            let _restore = Restore(saved);
            let ieee = saved & !((1 << 15) | (1 << 6) | (3 << 13));
            unsafe {
                set(ieee);
            }
            assert!(ny_core::require_f64_interval_proof_environment().is_ok());
            for mode in [
                1 << 6,
                1 << 15,
                (1 << 6) | (1 << 15),
                1 << 13,
                2 << 13,
                3 << 13,
            ] {
                unsafe {
                    set(ieee | mode);
                }
                for declaration in [
                    error(f32::from_bits(1), 0.0),
                    error(0.0, f32::from_bits(1)),
                    error(0.125, 0.25),
                ] {
                    let cpu = Cpu::default();
                    let mut e = [0.25];
                    let mut b = [-0.0];
                    assert!(charge(&cpu, 1, 1, &[1.0], &mut e, &mut b, declaration).is_err());
                    assert_eq!(e[0].to_bits(), 0.25f64.to_bits());
                    assert_eq!(b[0].to_bits(), (-0.0f64).to_bits());
                    assert_eq!(cpu.polls.load(Ordering::SeqCst), 0);
                    assert!(backward_layers_f64(
                        &cpu,
                        &[linear(&[1.0], Some(&[0.25]), 1, 1, declaration)],
                        frontier(&[1.0], &[0.0]),
                        1,
                        &mut None
                    )
                    .is_err());
                    assert_eq!(cpu.calls.load(Ordering::SeqCst), 0);
                }
                // Compatibility is deliberately narrower: all-signed-zero
                // model error still takes the unchanged legacy exact path.
                let cpu = Cpu::default();
                let mut e = [-0.0];
                let mut b = [-0.0];
                charge(&cpu, 1, 1, &[1.0], &mut e, &mut b, error(-0.0, 0.0)).unwrap();
                assert_eq!(e[0].to_bits(), (-0.0f64).to_bits());
                assert_eq!(b[0].to_bits(), (-0.0f64).to_bits());
                assert_eq!(cpu.polls.load(Ordering::SeqCst), 0);
            }
        })
        .join()
        .expect("isolated FP-state refusal checks");
    }

    #[test]
    fn bias_error_reduction_keeps_small_radii_after_a_unit_radius() {
        let n = 129usize;
        let mut errors = vec![2.0f64.powi(-54); n];
        errors[0] = 1.0;
        let legacy: f64 = errors.iter().copied().fold(0.0, |sum, x| sum + x);
        let exact = 1.0 + 2.0f64.powi(-47); //128*2^-54 is exactly representable.
        assert_eq!(legacy, 1.0);
        assert!(legacy < exact);
        let mut acc = [0.0];
        let mut radius = [0.0];
        super::super::bias_fold_f64(
            1,
            n,
            &vec![0.0; n],
            &errors,
            &vec![1.0; n],
            &mut acc,
            &mut radius,
        )
        .unwrap();
        assert_eq!(acc[0].to_bits(), 0.0f64.to_bits());
        assert!(radius[0] >= exact);
    }

    #[test]
    fn activation_intercept_reduction_preserves_all_independent_radii() {
        let n = 129usize;
        let mut errors = vec![2.0f64.powi(-54); n];
        errors[0] = 1.0;
        let exact = 1.0 + 2.0f64.powi(-47);
        let layer = GpuCrownLayer::Activation {
            num_neurons: n,
            lower_slope: vec![0.0; n],
            upper_slope: vec![0.0; n],
            lower_intercept: vec![0.0; n],
            upper_intercept: vec![1.0; n],
        };
        let got = backward_layers_f64(
            &Cpu::default(),
            &[layer],
            frontier(&vec![0.0; n], &errors),
            1,
            &mut None,
        )
        .unwrap();
        assert_eq!(got.lb[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(got.ub[0].to_bits(), 0.0f64.to_bits());
        assert!(got.lb_err[0] >= exact && got.ub_err[0] >= exact);
    }

    #[test]
    fn activation_slope_error_product_is_outward_even_when_centers_are_zero() {
        // (1+2^-52)*(1+2^-23) has an additional exact2^-75 term.
        // Ordinary f64 multiplication drops that term; the directed result
        // must be strictly above the exactly representable leading sum.
        let e = 1.0 + 2.0f64.powi(-52);
        let leading = 1.0 + 2.0f64.powi(-23) + 2.0f64.powi(-52);
        let (_, _, le, ue) = super::super::activation_step_f64(
            1,
            1,
            &[0.0],
            &[0.0],
            &[e],
            &[e],
            &[1.0],
            &[2.0f32.powi(-23)],
        )
        .unwrap();
        assert!(le[0] > leading && ue[0] > leading);
    }

    #[test]
    fn residual_error_merge_keeps_tiny_radius_when_centers_cancel() {
        let mut center = [1.0];
        let mut error = [1.0];
        super::super::merge_add(&mut center, &[-1.0], &mut error, &[2.0f64.powi(-54)]).unwrap();
        assert_eq!(center[0].to_bits(), 0.0f64.to_bits());
        assert!(error[0] > 1.0); // exact radius1+2^-54 cannot round backto1.
        let mut center = [0.0];
        let mut error = [f64::MAX];
        assert!(super::super::merge_add(&mut center, &[0.0], &mut error, &[f64::MAX]).is_err());
    }

    #[test]
    fn beta_rounding_radius_is_added_outward_to_a_large_existing_radius() {
        let layer = GpuCrownLayer::Activation {
            num_neurons: 1,
            lower_slope: vec![1.0],
            upper_slope: vec![1.0],
            lower_intercept: vec![0.0],
            upper_intercept: vec![0.0],
        };
        let without = backward_layers_f64(
            &Cpu::default(),
            &[layer.clone()],
            frontier(&[0.0], &[1.0]),
            1,
            &mut None,
        )
        .unwrap();
        let beta = 2.0f32.powi(-20);
        let signed = vec![vec![beta]];
        let mut aux = Some(super::super::ActAux::beta(&signed));
        let with = backward_layers_f64(
            &Cpu::default(),
            &[layer],
            frontier(&[0.0], &[1.0]),
            1,
            &mut aux,
        )
        .unwrap();
        assert_eq!(with.lower_a[0], -f64::from(beta));
        assert_eq!(with.upper_a[0], f64::from(beta));
        assert!(with.lower_err[0] > without.lower_err[0]);
        assert!(with.upper_err[0] > without.upper_err[0]);
    }
}
