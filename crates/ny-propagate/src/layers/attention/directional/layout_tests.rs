// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use ndarray::{ArrayD, IxDyn};
use num_rational::BigRational;
use num_traits::Signed;
use std::time::Duration;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 64 << 20, 64 << 20, 1_000_000_000, 64, Duration::ZERO).unwrap()
}

fn original_box(shape: &[usize], seed: usize, radius: f32) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let center: Vec<_> = (0..n)
        .map(|i| {
            let value = ((i * 7 + seed) % 19) as f32 * 0.0625 - 0.5;
            // Original binary32 low bits, not a convenient rounded f64 row.
            if value == 0.0 {
                value
            } else {
                f32::from_bits(value.to_bits() + 1)
            }
        })
        .collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|x| x - radius).collect()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), center.iter().map(|x| x + radius).collect()).unwrap(),
    )
    .unwrap()
}

fn authored_kt(k: &BoundedTensor) -> BoundedTensor {
    let shape = k.shape();
    let rank = shape.len();
    let (sk, dk) = (shape[rank - 2], shape[rank - 1]);
    let swap = |source: &ArrayD<f32>| {
        let old = source.as_slice().unwrap();
        let values: Vec<_> = (0..old.len())
            .map(|i| {
                let slice = i / (sk * dk);
                let channel = (i / sk) % dk;
                let key = i % sk;
                old[slice * sk * dk + key * dk + channel]
            })
            .collect();
        let mut shape = shape.to_vec();
        shape.swap(rank - 2, rank - 1);
        ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap()
    };
    BoundedTensor::new(swap(k.lower()), swap(k.upper())).unwrap()
}

fn objective(rows: usize, columns: usize) -> LinearBounds {
    let a = Array2::from_shape_fn((rows, columns), |(r, c)| {
        let x = ((r * 5 + c * 7) % 17) as f32 * 0.0625 - 0.5;
        if x == 0.0 {
            x
        } else {
            f32::from_bits(x.to_bits() + 1)
        }
    });
    let mut node = LinearBounds::new(
        a.clone(),
        Array1::from_shape_fn(rows, |r| r as f32 * 0.03125 - 0.25),
        -a,
        Array1::from_shape_fn(rows, |r| 0.375 - r as f32 * 0.015625),
    )
    .unwrap();
    node.set_coeff_err(
        Array2::from_shape_fn((rows, columns), |(r, c)| ((r + c) % 3) as f32 * 0.0078125),
        Array2::from_shape_fn((rows, columns), |(r, c)| {
            ((r + 2 * c) % 5) as f32 * 0.00390625
        }),
    );
    node
}

fn bits(values: impl Iterator<Item = f32>) -> Vec<u32> {
    values.map(f32::to_bits).collect()
}

fn assert_array_bits(a: &[f32], b: &[f32]) {
    assert_eq!(bits(a.iter().copied()), bits(b.iter().copied()));
}

fn assert_result_bits(old: &AttnTernaryResult, direct: &AttnTernaryResult, sk: usize, dk: usize) {
    assert_array_bits(old.1.as_slice().unwrap(), direct.1.as_slice().unwrap());
    assert_array_bits(old.2.as_slice().unwrap(), direct.2.as_slice().unwrap());
    for (source, (old, direct)) in old.0.iter().zip(&direct.0).enumerate() {
        let (old, direct) = (old.as_ref().unwrap(), direct.as_ref().unwrap());
        assert_array_bits(
            old.lower_b().as_slice().unwrap(),
            direct.lower_b().as_slice().unwrap(),
        );
        assert_array_bits(
            old.upper_b().as_slice().unwrap(),
            direct.upper_b().as_slice().unwrap(),
        );
        for (a, b) in [
            (old.lower_a(), direct.lower_a()),
            (old.upper_a(), direct.upper_a()),
        ] {
            for row in 0..a.nrows() {
                for canonical in 0..a.ncols() {
                    let destination = if source == 1 {
                        let slice = canonical / (sk * dk);
                        let key = (canonical / dk) % sk;
                        let channel = canonical % dk;
                        slice * sk * dk + channel * sk + key
                    } else {
                        canonical
                    };
                    assert_eq!(
                        a[[row, canonical]].to_bits(),
                        b[[row, destination]].to_bits(),
                        "source={source} row={row} canonical={canonical}"
                    );
                }
            }
        }
        // Current producer discharges all coefficient storage uncertainty into
        // shared bias. The layout adapter must preserve that exact profile.
        assert_eq!(old.lower_a_err(), direct.lower_a_err());
        assert_eq!(old.upper_a_err(), direct.upper_a_err());
    }
}

#[test]
fn authored_kt_direct_layout_bit_exact_asymmetric_signed_rows_and_masks() {
    let q = original_box(&[3, 2, 2], 1, 0.015625);
    let k = original_box(&[3, 3, 2], 4, 0.03125);
    let kt = authored_kt(&k);
    let v = original_box(&[3, 3, 4], 7, 0.03125);
    let node = objective(9, 3 * 2 * 4);
    for (mask, window, scale) in [
        (AttentionMask::Standard, None, 0.25),
        (AttentionMask::Standard, None, -0.5),
        (AttentionMask::Causal, None, 0.25),
        (AttentionMask::Causal, Some(1), -0.5),
    ] {
        let mut layer = SelfAttentionLayer::new(mask, Some(scale));
        layer.window_size = window;
        let old_budget = budget();
        let direct_budget = budget();
        let old = layer
            .propagate_crown_ternary_with_budget_and_deadline(&node, &q, &k, &v, &old_budget, None)
            .unwrap();
        let direct = layer
            .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
                &node,
                &q,
                &kt,
                &v,
                &direct_budget,
                None,
            )
            .unwrap();
        assert_result_bits(&old, &direct, 3, 2);
        assert!(direct.0[1]
            .as_ref()
            .unwrap()
            .lower_a()
            .iter()
            .any(|&x| x != 0.0));
        // Same scalar kernel work/persistent predictions; no budget refills or
        // altered arithmetic are hidden behind a different storage layout.
        assert_eq!(
            old_budget.receipt().unwrap(),
            direct_budget.receipt().unwrap()
        );
    }
}

fn rat(value: f32) -> BigRational {
    BigRational::from_float(ny_core::f32_to_f64_exact(value)).unwrap()
}

#[test]
fn authored_kt_direct_layout_nonzero_key_gradient_original_rational_uniform_control() {
    let q = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.25, -0.5, 0.75, 0.125]).unwrap(),
    )
    .unwrap();
    // All ORIGINAL keys equal, hence scores and probabilities are exactly
    // uniform independently of the checker exp implementation. K gradients
    // remain nonzero because original V differs by key and Q is nonzero.
    let k = BoundedTensor::concrete(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 2]),
            vec![0.375, -0.25, 0.375, -0.25, 0.375, -0.25],
        )
        .unwrap(),
    )
    .unwrap();
    let kt = authored_kt(&k);
    let v = BoundedTensor::concrete(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 2]),
            vec![0.5, -0.25, -0.75, 0.125, 0.25, 0.75],
        )
        .unwrap(),
    )
    .unwrap();
    let node = objective(9, 4);
    let result = SelfAttentionLayer::new(AttentionMask::Standard, Some(0.25))
        .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &node,
            &q,
            &kt,
            &v,
            &budget(),
            None,
        )
        .unwrap();
    assert!(result.0[1]
        .as_ref()
        .unwrap()
        .lower_a()
        .iter()
        .any(|&x| x != 0.0));
    let y: Vec<_> = (0..4)
        .map(|c| {
            (0..3)
                .map(|key| rat(v.lower().as_slice().unwrap()[key * 2 + c % 2]))
                .sum::<BigRational>()
                / BigRational::from_integer(3.into())
        })
        .collect();
    for row in 0..9 {
        for upper in [false, true] {
            let (a, b, error) = if upper {
                (node.upper_a(), node.upper_b(), node.upper_a_err())
            } else {
                (node.lower_a(), node.lower_b(), node.lower_a_err())
            };
            let mut original = rat(b[row]);
            for c in 0..4 {
                original += rat(a[[row, c]]) * &y[c];
                let e = rat(error.unwrap()[[row, c]]) * y[c].abs();
                if upper {
                    original += e;
                } else {
                    original -= e;
                }
            }
            let mut carried = rat(if upper { result.2[row] } else { result.1[row] });
            for (bounds, input) in result.0.iter().zip([&q, &kt, &v]) {
                let bounds = bounds.as_ref().unwrap();
                let a = if upper {
                    bounds.upper_a()
                } else {
                    bounds.lower_a()
                };
                for (c, &x) in input.lower().as_slice().unwrap().iter().enumerate() {
                    carried += rat(a[[row, c]]) * rat(x);
                }
            }
            if upper {
                assert!(carried >= original);
            } else {
                assert!(carried <= original);
            }
        }
    }
}

#[test]
fn authored_kt_direct_layout_refuses_wrong_axes_prefix_nonfinite_and_mask_profile() {
    let layer = SelfAttentionLayer::standard();
    let q = original_box(&[1, 2, 2], 0, 0.015625);
    let k = original_box(&[1, 3, 2], 2, 0.015625);
    let kt = authored_kt(&k);
    let v = original_box(&[1, 3, 4], 4, 0.015625);
    let node = objective(2, 8);
    for wrong in [
        k.clone(),
        original_box(&[2, 2, 3], 5, 0.015625),
        original_box(&[1, 2, 4], 5, 0.015625),
    ] {
        let budget = budget();
        assert!(layer
            .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
                &node, &q, &wrong, &v, &budget, None
            )
            .is_err());
        assert_eq!(budget.receipt().unwrap().work_reserved, 0);
    }
    let zero_q = BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 0, 2]))).unwrap();
    let zero_budget = budget();
    assert!(matches!(
        layer.propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &node,
            &zero_q,
            &kt,
            &v,
            &zero_budget,
            None
        ),
        Err(NyError::UnsupportedOp(_))
    ));
    assert_eq!(zero_budget.receipt().unwrap().work_reserved, 0);
    let nonfinite = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1, 2, 3]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 2, 3]), f32::INFINITY),
    )
    .unwrap();
    assert!(layer
        .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &node,
            &q,
            &nonfinite,
            &v,
            &budget(),
            None
        )
        .is_err());
    let mut l = k.lower().clone();
    let mut u = k.upper().clone();
    l.swap_axes(1, 2);
    u.swap_axes(1, 2);
    let nonstandard = BoundedTensor::new(l, u).unwrap();
    assert_eq!(nonstandard.shape(), kt.shape());
    assert!(!nonstandard.lower().is_standard_layout());
    assert!(layer
        .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &node,
            &q,
            &nonstandard,
            &v,
            &budget(),
            None
        )
        .is_err());
    // The durable windowed-causal sq>sk soundness gate is storage independent.
    let long_q = original_box(&[1, 4, 2], 0, 0.015625);
    let long_node = objective(2, 16);
    let causal = SelfAttentionLayer::causal().with_window_size(1);
    assert!(matches!(
        causal.propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &long_node,
            &long_q,
            &kt,
            &v,
            &budget(),
            None
        ),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(layer
        .propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
            &node,
            &q,
            &kt,
            &v,
            &budget(),
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap()
            )
        )
        .is_err());
}

#[test]
fn authored_kt_direct_layout_mapped_fill_cancellation_polls_logical_cells() {
    let (sk, dk) = (3usize, 2048usize);
    let lower: Vec<_> = (0..sk * dk)
        .map(|i| i as f32 * f32::from_bits(0x3900_0000))
        .collect();
    let upper: Vec<_> = lower.iter().map(|x| x + 0.03125).collect();
    let mut center = vec![f64::NAN; sk * dk];
    let mut radius = vec![f64::NAN; sk * dk];
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &MemoryAuthority::new(1 << 20),
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: 1 << 20,
            persistent_bytes: 1 << 20,
            max_work: 100_000,
            max_proposals: 1,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let before = budget.receipt().unwrap();
    let _lease = budget
        .lease(WorkRequest {
            scratch_bytes: sk * dk * 24,
            work: (sk * dk) as u64 * 8,
            ..WorkRequest::default()
        })
        .unwrap();
    let result = fill_authored_key_centers(
        &lower,
        &upper,
        &mut center,
        &mut radius,
        0,
        0,
        sk,
        dk,
        |logical| {
            if logical >= 1024 {
                cancel.cancel();
            }
            checkpoint(None, Some(&budget))
        },
    );
    assert!(result.is_err());
    for channel in 0..1024 {
        let original = channel * sk;
        let expected =
            lower[original] as f64 + (upper[original] as f64 - lower[original] as f64) * 0.5;
        assert_eq!(center[channel].to_bits(), expected.to_bits());
        assert!(radius[channel].is_finite());
    }
    assert!(center[1024..].iter().all(|x| x.is_nan()));
    assert!(radius[1024..].iter().all(|x| x.is_nan()));
    assert!(budget.receipt().unwrap().work_reserved > before.work_reserved);
}
