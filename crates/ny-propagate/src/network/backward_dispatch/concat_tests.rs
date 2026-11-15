// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original-coordinate rational controls for the native scalar Concat arm.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

use super::*;
use crate::layers::{ConcatLayer, Layer};
use crate::network::backward_dispatch::dispatch::{
    dispatch_backward_layer, dispatch_backward_layer_finite_boundary,
};
use crate::MulBinaryRelaxationMode;

fn q(value: f32) -> Q {
    assert!(value.is_finite());
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    let (mantissa, shift) = if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | (1 << 23), exponent as i32 - 150)
    };
    let mut integer = BigInt::from(mantissa);
    if bits >> 31 != 0 {
        integer = -integer;
    }
    if shift >= 0 {
        Q::from_integer(integer << shift as usize)
    } else {
        Q::new(integer, BigInt::from(1) << (-shift) as usize)
    }
}

fn tensor(shape: &[usize], lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new_allow_infinite(
        ArrayD::from_shape_vec(IxDyn(shape), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), hi.to_vec()).unwrap(),
    )
    .unwrap()
}

fn bounds(lower: Array2<f32>, upper: Array2<f32>) -> LinearBounds {
    let rows = lower.nrows();
    LinearBounds::new(lower, Array1::zeros(rows), upper, Array1::zeros(rows)).unwrap()
}

fn run(
    concat: &ConcatLayer,
    input: &BoundedTensor,
    incoming: &LinearBounds,
    names: &[String],
    original_dispatch: bool,
    deadline: Option<Instant>,
) -> Result<BackwardDispatchResult> {
    let layer = Layer::Concat(concat.clone());
    let cached: HashMap<String, BoundedTensor> = HashMap::new();
    let ctx = DispatchContext {
        node_name: "original_concat",
        layer: &layer,
        inputs: names,
        pre_activation: input,
        network_input: input,
        node_bounds: (&cached).into(),
        engine: None,
        deadline,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };
    if original_dispatch {
        dispatch_backward_layer(&ctx, incoming)
    } else {
        dispatch_concat(concat, &ctx, incoming)
    }
}

fn support(bounds: &LinearBounds, row: usize, point: &[f32], lower: bool) -> Q {
    let (a, e, b) = if lower {
        (bounds.lower_a(), bounds.lower_a_err(), bounds.lower_b())
    } else {
        (bounds.upper_a(), bounds.upper_a_err(), bounds.upper_b())
    };
    let mut value = q(b[row]);
    for (column, &x) in point.iter().enumerate() {
        value += q(a[[row, column]]) * q(x);
        let uncertainty = e.map_or_else(Q::zero, |e| q(e[[row, column]])) * q(x).abs();
        if lower {
            value -= uncertainty;
        } else {
            value += uncertainty;
        }
    }
    value
}

/// Independently authored original axis1 order is
/// [c0,x00,x01,c1,x10,x11], not [c0,c1,x00,x01,x10,x11].
fn axis1_control(original_dispatch: bool) {
    let constant = tensor(&[2, 1], &[1.0, -3.0], &[2.0, -1.0]);
    let concat =
        ConcatLayer::with_constants(1, vec![vec![2, 1], vec![2, 2]], vec![Some(constant), None]);
    let input = tensor(&[2, 2], &[-2.0; 4], &[3.0; 4]);
    let mut incoming = bounds(
        array![
            [2.0, -1.0, 3.0, -4.0, 0.5, -2.0],
            [-1.0, 2.0, 0.0, 1.0, -3.0, 4.0]
        ],
        array![
            [-3.0, 2.0, -1.0, 2.0, -4.0, 0.5],
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
        ],
    );
    incoming.lower_b_mut().assign(&array![0.25, -0.5]);
    incoming.upper_b_mut().assign(&array![1.25, 0.75]);
    incoming.set_coeff_err(
        array![
            [0.5, 0.125, 0.25, 0.75, 0.5, 0.25],
            [0.25, 0.5, 0.125, 0.5, 0.25, 0.75]
        ],
        array![
            [0.25, 0.5, 0.125, 0.5, 0.25, 0.75],
            [0.5, 0.125, 0.25, 0.75, 0.5, 0.25]
        ],
    );
    let BackwardDispatchResult::Nary {
        bounds: outputs,
        bias_lower,
        bias_upper,
    } = run(
        &concat,
        &input,
        &incoming,
        &[NETWORK_INPUT.into()],
        original_dispatch,
        None,
    )
    .unwrap()
    else {
        panic!("expected original ONNX ordered Nary result")
    };
    assert!(outputs.onnx_slots()[0].is_none());
    let dynamic = outputs.onnx_slots()[1].as_ref().unwrap();
    for (new, old) in [0usize, 1, 2, 3].into_iter().zip([1usize, 2, 4, 5]) {
        for row in 0..2 {
            assert_eq!(
                dynamic.lower_a()[[row, new]],
                incoming.lower_a()[[row, old]]
            );
            assert_eq!(
                dynamic.upper_a()[[row, new]],
                incoming.upper_a()[[row, old]]
            );
            assert_eq!(
                dynamic.lower_a_err().unwrap()[[row, new]],
                incoming.lower_a_err().unwrap()[[row, old]]
            );
            assert_eq!(
                dynamic.upper_a_err().unwrap()[[row, new]],
                incoming.upper_a_err().unwrap()[[row, old]]
            );
        }
    }
    assert!(dynamic
        .lower_b()
        .iter()
        .chain(dynamic.upper_b())
        .all(|&b| b == 0.0));
    for mask in 0usize..64 {
        let c0 = if mask & 1 == 0 { 1.0 } else { 2.0 };
        let c1 = if mask & 2 == 0 { -3.0 } else { -1.0 };
        let x: [f32; 4] = std::array::from_fn(|i| {
            if mask & (1 << (i + 2)) == 0 {
                -2.0
            } else {
                3.0
            }
        });
        let original = [c0, x[0], x[1], c1, x[2], x[3]];
        for row in 0..2 {
            assert!(
                q(bias_lower[row]) + support(dynamic, row, &x, true)
                    <= support(&incoming, row, &original, true)
            );
            assert!(
                q(bias_upper[row]) + support(dynamic, row, &x, false)
                    >= support(&incoming, row, &original, false)
            );
        }
    }
}

#[test]
fn concat_constant_original_axis1_signed_four_planes_match_independent_rational_corners() {
    axis1_control(false);
}

#[test]
fn concat_constant_normal_scalar_dispatch_is_native_without_enabling_other_concat_arms() {
    // The scalar graph dispatcher alone recognizes the new native carrier.
    // Global Layer::propagates_coeff_err must remain unchanged for older arms.
    assert!(!Layer::Concat(ConcatLayer::new(0)).propagates_coeff_err());
    axis1_control(true);
}

#[test]
fn concat_constant_signed_interval_endpoints_use_each_coefficient_and_side() {
    let constant = tensor(&[2], &[-2.0, 1.0], &[3.0, 4.0]);
    let source = bounds(array![[-2.0, 3.0]], array![[4.0, -5.0]]);
    let mut lo = array![0.25];
    let mut hi = array![-0.5];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    for x in [[-2.0, 1.0], [-2.0, 4.0], [3.0, 1.0], [3.0, 4.0]] {
        assert!(q(lo[0]) <= q(0.25) + support(&source, 0, &x, true));
        assert!(q(hi[0]) >= q(-0.5) + support(&source, 0, &x, false));
    }
}

#[test]
fn concat_constant_cancellation_preserves_actual_positive_and_negative_unit_terms() {
    let large = f32::from_bits((127u32 + 60) << 23);
    for unit in [-1.0, 1.0] {
        let constant = tensor(&[3], &[large, unit, -large], &[large, unit, -large]);
        let source = bounds(Array2::ones((1, 3)), Array2::ones((1, 3)));
        let mut lo = array![0.0];
        let mut hi = array![0.0];
        fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
        assert!(q(lo[0]) <= q(unit) && q(hi[0]) >= q(unit));
        assert!(lo[0] > unit - 0.01 && hi[0] < unit + 0.01);
    }
}

#[test]
fn concat_constant_zero_centers_still_pay_error_magnitude_without_signed_cancellation() {
    let constant = tensor(&[2], &[2.0, -2.0], &[2.0, -2.0]);
    let mut source = bounds(Array2::zeros((1, 2)), Array2::zeros((1, 2)));
    source.set_coeff_err(array![[0.5, 0.5]], array![[0.25, 0.75]]);
    let mut lo = array![0.0];
    let mut hi = array![0.0];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    assert!(q(lo[0]) <= q(-2.0) && q(hi[0]) >= q(2.0));
}

#[test]
fn concat_constant_full_incoming_bias_participates_in_certified_reduction() {
    let large = f32::from_bits((127u32 + 60) << 23);
    let constant = tensor(&[2], &[-large, 1.0], &[-large, 1.0]);
    let source = bounds(Array2::ones((1, 2)), Array2::ones((1, 2)));
    let mut lo = array![large];
    let mut hi = array![large];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    assert!(q(lo[0]) <= q(1.0) && q(hi[0]) >= q(1.0));
    assert!(lo[0] > 0.99 && hi[0] < 1.01);
}

#[test]
fn concat_constant_subnormal_bits_and_signed_zero_are_not_erased() {
    let tiny = f32::from_bits(1);
    let constant = tensor(&[2], &[tiny, -tiny], &[tiny, -tiny]);
    let source = bounds(array![[-tiny, tiny]], array![[tiny, -tiny]]);
    let mut lo = array![-0.0];
    let mut hi = array![0.0];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    let exact = q(tiny) * q(tiny) * Q::from_integer(BigInt::from(2));
    assert!(q(lo[0]) <= -&exact && q(hi[0]) >= exact);
    assert!(lo[0].is_normal() && hi[0].is_normal());
}

#[test]
fn concat_constant_explicit_zero_error_lanes_and_zero_times_infinite_are_exact_zero() {
    let constant = tensor(&[2], &[f32::NEG_INFINITY, -0.0], &[f32::INFINITY, 0.0]);
    let mut source = bounds(array![[0.0, -0.0]], array![[-0.0, 0.0]]);
    source.set_coeff_err(Array2::zeros((1, 2)), Array2::zeros((1, 2)));
    let mut lo = array![-0.0];
    let mut hi = array![0.0];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    assert_eq!(lo[0], 0.0);
    assert_eq!(hi[0], 0.0);
    assert!(lo[0].is_finite() && hi[0].is_finite());
}

#[test]
fn concat_constant_one_sided_errors_and_poisoned_negative_subnormal_refuse_tightness() {
    let constant = tensor(&[1], &[2.0], &[2.0]);
    let mut source = bounds(array![[0.0]], array![[0.0]]);
    source.lower_a_err = Some(array![[0.5]]);
    let mut lo = array![0.0];
    let mut hi = array![0.0];
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    assert!(q(lo[0]) <= q(-1.0));
    assert_eq!(hi[0], 0.0);
    source.lower_a_err.as_mut().unwrap()[[0, 0]] = f32::from_bits(0x8000_0001);
    fold_constant(&source, &constant, &mut lo, &mut hi, None).unwrap();
    assert_eq!(lo[0], f32::NEG_INFINITY);
    assert_eq!(hi[0], 0.0);
}

#[test]
fn concat_constant_expired_original_dispatch_clock_refuses_before_carrier_clone() {
    let constant = tensor(&[1], &[1.0], &[1.0]);
    let concat = ConcatLayer::with_constants(0, vec![vec![1]], vec![Some(constant)]);
    let input = tensor(&[1], &[0.0], &[0.0]);
    let source = bounds(array![[1.0]], array![[1.0]]);
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        run(&concat, &input, &source, &[], false, Some(expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn concat_constant_shape_mismatch_refuses_without_partial_bias_changes() {
    let source = bounds(array![[1.0, 2.0]], array![[1.0, 2.0]]);
    let constant = tensor(&[1], &[1.0], &[1.0]);
    let mut lo = array![0.25];
    let mut hi = array![0.75];
    assert!(fold_constant(&source, &constant, &mut lo, &mut hi, None).is_err());
    assert_eq!(lo, array![0.25]);
    assert_eq!(hi, array![0.75]);
}

#[test]
fn concat_constant_native_scalar_does_not_grant_legacy_finite_boundary_authority() {
    const CHILD: &str = "NY_CONCAT_BOUNDARY_TEST_CHILD";
    let strict_child = std::env::var_os(CHILD).is_some();
    let expiry = crate::network::core::sequential::crown::patches_step::expiry_authority_armed();
    assert_eq!(
        expiry, !strict_child,
        "control requires shipped expiry authority, or its explicit strict child"
    );
    let constant = tensor(&[1], &[2.0], &[2.0]);
    let layer = Layer::Concat(ConcatLayer::with_constants(
        0,
        vec![vec![1]],
        vec![Some(constant)],
    ));
    let input = tensor(&[1], &[0.0], &[0.0]);
    let source = bounds(array![[1.0]], array![[1.0]]);
    let cached: HashMap<String, BoundedTensor> = HashMap::new();
    let ctx = DispatchContext {
        node_name: "original_concat",
        layer: &layer,
        inputs: &[],
        pre_activation: &input,
        network_input: &input,
        node_bounds: (&cached).into(),
        engine: None,
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };
    assert!(!layer.propagates_coeff_err());
    let result = dispatch_backward_layer_finite_boundary(&ctx, &source).unwrap();
    if strict_child {
        assert!(matches!(result, BackwardDispatchResult::Unsupported(_)));
    } else {
        // Shipped expiry authority already admits a still-live legacy walk.
        // The native scalar repair must preserve that policy without changing
        // the shared capability or granting the historical strict boundary.
        let BackwardDispatchResult::Nary {
            bias_lower,
            bias_upper,
            ..
        } = result
        else {
            panic!("shipped live expiry authority lost original Concat arrival");
        };
        assert_eq!(bias_lower.len(), 1);
        assert_eq!(bias_upper.len(), 1);
        // The certified reduction rounds outward even for the exact source
        // constant. Check exact containment and one-ulp tightness instead of
        // requiring an unsound equality to the nominal rounded constant.
        let exact = q(2.0);
        assert!(q(bias_lower[0]) <= exact);
        assert!(q(bias_upper[0]) >= exact);
        assert!(q(bias_lower[0]) >= q(f32::from_bits(2.0_f32.to_bits() - 1)));
        assert!(q(bias_upper[0]) <= q(f32::from_bits(2.0_f32.to_bits() + 1)));
        // The policy is process-latched. A fresh invocation exercises the
        // real strict entry rather than mutating an already initialized latch.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "network::backward_dispatch::concat::tests::concat_constant_native_scalar_does_not_grant_legacy_finite_boundary_authority",
                "--test-threads=1",
            ])
            .env("NY_PATCHES_FINITE_EXPIRY", "0")
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "strict original boundary control failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("1 passed; 0 failed"));
        assert!(stdout.contains(
            "concat_constant_native_scalar_does_not_grant_legacy_finite_boundary_authority ... ok"
        ));
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3u64 << 22)) | (1u64 << 22);
            std::arch::asm!("msr fpcr, {directed}","isb",directed=in(reg)directed,options(nomem,nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}","isb",saved=in(reg)self.saved,options(nomem,nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn concat_constant_unqualified_live_environment_refuses_before_bias_publication() {
    let _env = crate::tests::lock_env_shared();
    let source = bounds(array![[1.0]], array![[1.0]]);
    let constant = tensor(&[1], &[1.0], &[1.0]);
    let mut lo = array![0.25];
    let mut hi = array![0.75];
    {
        let _rounding = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            fold_constant(&source, &constant, &mut lo, &mut hi, None),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(lo, array![0.25]);
    assert_eq!(hi, array![0.75]);
    assert!(ny_core::has_f64_interval_proof_environment());
}
