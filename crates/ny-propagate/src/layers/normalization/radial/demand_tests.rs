// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::{LayerNormLayer, RmsNormLayer};
use ndarray::ShapeBuilder;
use std::time::{Duration, Instant};

fn zero_bounds(
    groups: usize,
    outputs: usize,
    n: usize,
    fortran_errors: bool,
) -> BatchedLinearBounds {
    let a = ArrayD::from_shape_fn(IxDyn(&[groups, outputs, n]), |i| {
        if (i[0] + i[1] + i[2]) % 2 == 0 {
            0.0
        } else {
            -0.0
        }
    });
    let lower = ArrayD::from_shape_fn(IxDyn(&[groups, outputs]), |i| {
        if i[0] == 0 && i[1] == 0 {
            -0.0
        } else {
            -2.0 + (i[0] + i[1]) as f32 / 8.0
        }
    });
    let upper = ArrayD::from_shape_fn(IxDyn(&[groups, outputs]), |i| {
        if i[0] == 0 && i[1] == 0 {
            -0.0
        } else {
            2.0 + (i[0] + i[1]) as f32 / 8.0
        }
    });
    let mut bounds = BatchedLinearBounds::new(
        a.clone(),
        lower,
        a,
        upper,
        vec![groups, n],
        vec![groups, outputs],
    )
    .unwrap();
    let errors = ArrayD::from_shape_fn(IxDyn(&[groups, outputs, n]).f(), |i| {
        if i[2] % 2 == 0 {
            -0.0
        } else {
            0.0
        }
    });
    let errors = if fortran_errors {
        errors
    } else {
        errors.as_standard_layout().to_owned()
    };
    bounds.lower_a_err = Some(errors.clone());
    bounds.upper_a_err = Some(errors);
    bounds
}
fn boxed(groups: usize, n: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[groups, n]), -1.0),
        ArrayD::from_elem(IxDyn(&[groups, n]), 1.0),
    )
    .unwrap()
}
fn budget(work: u64, scratch: usize) -> WorkBudget {
    WorkBudget::new(None, scratch, 0, work, 1, Duration::ZERO).unwrap()
}
fn assert_bias_passthrough(result: &BatchedLinearBounds, incoming: &BatchedLinearBounds) {
    assert_eq!(result.lower_a.shape(), incoming.lower_a.shape());
    assert!(result
        .lower_a
        .iter()
        .chain(result.upper_a.iter())
        .all(|&v| demand::signed_zero(v)));
    assert_eq!(
        result
            .lower_b
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        incoming
            .lower_b
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        result
            .upper_b
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        incoming
            .upper_b
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn radial_demand_signed_zero_groups_preserve_bias_bits_without_geometry() {
    for fortran in [false, true] {
        let source = zero_bounds(5, 3, 4, fortran);
        let pre = boxed(5, 4);
        let gamma = Array1::from_vec(vec![-2.0, 0.5, 1.25, -0.75]);
        let beta = Array1::from_vec(vec![0.25, -0.5, 1.0, -0.25]);
        for centered in [false, true] {
            geometry::reset_constructions();
            let result =
                baseline::checked_baseline(&source, &gamma, Some(&beta), 3.0, &pre, centered, None)
                    .unwrap()
                    .bounds;
            assert_eq!(geometry::constructions(), 0);
            assert_bias_passthrough(&result, &source);
            geometry::reset_constructions();
            let result =
                candidate_batched(&source, &pre, &gamma, Some(&beta), 3.0, centered, None).unwrap();
            assert_eq!(geometry::constructions(), 0);
            assert_bias_passthrough(&result, &source);
        }
    }
}

#[test]
fn radial_demand_default_layernorm_and_rmsnorm_routes_engage() {
    let source = zero_bounds(32, 4, 8, true);
    let pre = boxed(32, 8);
    let ln =
        LayerNormLayer::new(Array1::from_elem(8, 1.25), Array1::from_elem(8, -0.75), 3.0).unwrap();
    let rms = RmsNormLayer::new(Array1::from_elem(8, -0.5), 3.0).unwrap();
    geometry::reset_constructions();
    let result = ln
        .propagate_linear_batched_with_bounds(&source, &pre)
        .unwrap();
    assert_eq!(
        geometry::constructions(),
        0,
        "both checked baseline and default radial refinement skip absent groups"
    );
    assert_bias_passthrough(&result, &source);
    geometry::reset_constructions();
    let result = rms
        .propagate_linear_batched_with_bounds(&source, &pre)
        .unwrap();
    assert_eq!(geometry::constructions(), 0);
    assert_bias_passthrough(&result, &source);
}

#[test]
fn radial_demand_nonzero_errors_and_minimum_subnormals_construct_geometry() {
    let pre = boxed(3, 4);
    let gamma = Array1::ones(4);
    for selector in 0..4 {
        let mut source = zero_bounds(3, 3, 4, true);
        match selector {
            0 => source.lower_a[[2, 1, 0]] = f32::from_bits(1),
            1 => source.upper_a[[2, 1, 0]] = f32::from_bits(0x8000_0001),
            2 => source.lower_a_err.as_mut().unwrap()[[2, 1, 0]] = f32::from_bits(1),
            _ => source.upper_a_err.as_mut().unwrap()[[2, 1, 0]] = 0.5,
        }
        geometry::reset_constructions();
        baseline::checked_baseline(&source, &gamma, None, 3.0, &pre, false, None).unwrap();
        assert_eq!(geometry::constructions(), 1, "selector={selector}");
        geometry::reset_constructions();
        let result = candidate_batched(&source, &pre, &gamma, None, 3.0, false, None).unwrap();
        assert_eq!(geometry::constructions(), 1, "selector={selector}");
        assert_eq!(
            result.lower_b[[0, 0]].to_bits(),
            source.lower_b[[0, 0]].to_bits()
        );
        if selector == 3 {
            assert!(
                result.upper_b[[2, 1]] > source.upper_b[[2, 1]],
                "error-only upper row cannot become bias-only"
            );
        }
    }
}

#[test]
fn radial_demand_active_rows_remain_bit_identical_when_other_groups_gain_demand() {
    let pre = boxed(4, 4);
    let gamma = Array1::from_vec(vec![1.25, -0.5, 0.75, 2.0]);
    let beta = Array1::from_vec(vec![-0.25, 0.75, 0.0, -0.5]);
    let mut sparse = zero_bounds(4, 3, 4, true);
    for k in 0..4 {
        sparse.lower_a[[3, 2, k]] = [1.0, -0.25, 0.5, -1.5][k];
        sparse.upper_a[[3, 2, k]] = sparse.lower_a[[3, 2, k]];
        sparse.lower_a_err.as_mut().unwrap()[[3, 2, k]] = 0.125;
        sparse.upper_a_err.as_mut().unwrap()[[3, 2, k]] = 0.125;
    }
    let mut dense = sparse.clone();
    for group in 0..4 {
        for output in 0..3 {
            if group != 3 || output != 2 {
                dense.lower_a[[group, output, 0]] = 0.25;
                dense.upper_a[[group, output, 0]] = 0.25;
            }
        }
    }
    for centered in [false, true] {
        for baseline_route in [false, true] {
            geometry::reset_constructions();
            let a = if baseline_route {
                baseline::checked_baseline(&sparse, &gamma, Some(&beta), 3.0, &pre, centered, None)
                    .unwrap()
                    .bounds
            } else {
                candidate_batched(&sparse, &pre, &gamma, Some(&beta), 3.0, centered, None).unwrap()
            };
            assert_eq!(geometry::constructions(), 1);
            geometry::reset_constructions();
            let b = if baseline_route {
                baseline::checked_baseline(&dense, &gamma, Some(&beta), 3.0, &pre, centered, None)
                    .unwrap()
                    .bounds
            } else {
                candidate_batched(&dense, &pre, &gamma, Some(&beta), 3.0, centered, None).unwrap()
            };
            assert_eq!(geometry::constructions(), 4);
            for (left, right) in [(&a.lower_a, &b.lower_a), (&a.upper_a, &b.upper_a)] {
                for k in 0..4 {
                    assert_eq!(left[[3, 2, k]].to_bits(), right[[3, 2, k]].to_bits());
                }
            }
            assert_eq!(a.lower_b[[3, 2]].to_bits(), b.lower_b[[3, 2]].to_bits());
            assert_eq!(a.upper_b[[3, 2]].to_bits(), b.upper_b[[3, 2]].to_bits());
        }
    }
}

#[test]
fn radial_demand_scan_is_charged_before_active_work_refusal_and_dense_memory_stays_priced() {
    let mut sparse = zero_bounds(64, 2, 128, true);
    sparse.lower_a[[63, 1, 0]] = 0.5;
    sparse.upper_a[[63, 1, 0]] = 0.5;
    let pre = boxed(64, 128);
    let gamma = Array1::ones(128);
    let cap = 1_000_000;
    let sparse_budget = budget(cap, 32 << 20);
    geometry::reset_constructions();
    let result = candidate_batched(
        &sparse,
        &pre,
        &gamma,
        None,
        3.0,
        false,
        Some(&sparse_budget),
    )
    .unwrap();
    assert_eq!(geometry::constructions(), 1);
    assert_eq!(
        result.lower_a.len(),
        sparse.lower_a.len(),
        "returned carrier remains dense"
    );
    let receipt = sparse_budget.receipt().unwrap();
    assert_eq!(receipt.admitted, 2);
    assert_eq!(receipt.completed, 2);
    assert_eq!(
        receipt.proposals_reserved, 1,
        "mandatory inspection does not spend a second proposal"
    );
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert!(receipt.work_reserved < cap);
    let mut dense = sparse.clone();
    for group in 0..64 {
        for output in 0..2 {
            dense.lower_a[[group, output, 0]] = 0.5;
            dense.upper_a[[group, output, 0]] = 0.5;
        }
    }
    let dense_budget = budget(cap, 32 << 20);
    geometry::reset_constructions();
    assert!(
        candidate_batched(&dense, &pre, &gamma, None, 3.0, false, Some(&dense_budget)).is_err()
    );
    assert_eq!(geometry::constructions(), 0);
    let refusal = dense_budget.receipt().unwrap();
    assert_eq!(refusal.attempted, 2);
    assert_eq!(
        refusal.completed, 1,
        "paid scan completed even when active construction refused"
    );
    assert_eq!(refusal.refused, 1);
    assert!(refusal.work_reserved > 0);
    assert_eq!(refusal.live_scratch_bytes, 0);
    let zero = zero_bounds(64, 2, 128, true);
    let memory_budget = budget(10_000_000, 512);
    geometry::reset_constructions();
    assert!(
        candidate_batched(&zero, &pre, &gamma, None, 3.0, false, Some(&memory_budget)).is_err(),
        "zero demand does not bypass dense-output admission"
    );
    assert_eq!(geometry::constructions(), 0);
    assert_eq!(memory_budget.receipt().unwrap().completed, 1);
}

#[test]
fn radial_demand_zero_directions_preserve_profile_finite_error_and_deadline_firewalls() {
    let source = zero_bounds(2, 2, 4, true);
    let pre = boxed(2, 4);
    let gamma = Array1::ones(4);
    for value in [-0.25, f32::NAN, f32::INFINITY] {
        let mut poison = source.clone();
        poison.upper_a_err.as_mut().unwrap()[[1, 1, 3]] = value;
        geometry::reset_constructions();
        assert!(baseline::checked_baseline(&poison, &gamma, None, 3.0, &pre, false, None).is_err());
        assert!(candidate_batched(&poison, &pre, &gamma, None, 3.0, false, None).is_err());
        assert_eq!(geometry::constructions(), 0);
    }
    let mut poisoned_gamma = gamma.clone();
    poisoned_gamma[3] = f32::INFINITY;
    assert!(candidate_batched(&source, &pre, &poisoned_gamma, None, 3.0, false, None).is_err());
    assert!(baseline::checked_baseline(&source, &gamma, None, 0.0, &pre, false, None).is_err());
    let unbounded = BoundedTensor::new_repaired(
        ArrayD::from_elem(IxDyn(&[2, 4]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[2, 4]), f32::INFINITY),
        ny_tensor::RepairStrategy::Conservative,
    )
    .unwrap();
    assert!(
        candidate_batched(&source, &unbounded, &gamma, None, 3.0, false, None).is_err(),
        "zero demand cannot legitimize an unsupported input enclosure"
    );
    let mut poison = source.clone();
    poison.lower_b[[1, 1]] = f32::NAN;
    assert!(candidate_batched(&poison, &pre, &gamma, None, 3.0, false, None).is_err());
    let mut conservative = source.clone();
    conservative.lower_b[[1, 1]] = f32::NEG_INFINITY;
    conservative.upper_b[[1, 1]] = f32::INFINITY;
    let baseline = baseline::checked_baseline(&conservative, &gamma, None, 3.0, &pre, false, None)
        .unwrap()
        .bounds;
    assert_bias_passthrough(&baseline, &conservative);
    assert!(
        candidate_batched(&conservative, &pre, &gamma, None, 3.0, false, None).is_err(),
        "radial retains its finite-bias contract"
    );
    // The scalar profile check takes rank-two coefficient carriers. The
    // rank-three group fixture above belongs to the separate batched route.
    let scalar = crate::LinearBounds::new(
        ndarray::Array2::zeros((1, 4)),
        Array1::zeros(1),
        ndarray::Array2::zeros((1, 4)),
        Array1::zeros(1),
    )
    .unwrap();
    let mut ln = LayerNormLayer::new_default(4, 3.0).unwrap();
    ln.forward_mode = true;
    assert!(ln
        .propagate_radial_linear_with_bounds(&scalar, &boxed(1, 4), None)
        .is_err());
    let mut rms = RmsNormLayer::new_default(4, 3.0).unwrap();
    rms.forward_mode = true;
    assert!(rms
        .propagate_radial_linear_with_bounds(&scalar, &boxed(1, 4), None)
        .is_err());
    let expired = WorkBudget::new(
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        1 << 20,
        0,
        10_000_000,
        10,
        Duration::ZERO,
    )
    .unwrap();
    assert!(matches!(
        candidate_batched(&source, &pre, &gamma, None, 3.0, false, Some(&expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn radial_demand_zero_writes_cancel_between_bounded_chunks() {
    let mut row = Array1::from_elem(4 * ZERO_WRITE_CHUNK, 2.0);
    let mut polls = 0;
    let result = write_zero_row_with_poll(row.view_mut(), || {
        polls += 1;
        if polls == 3 {
            Err(NyError::DeadlineExceeded(
                "injected zero-row deadline".into(),
            ))
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert!(row
        .iter()
        .take(2 * ZERO_WRITE_CHUNK)
        .all(|&value| value == 0.0));
    assert!(row
        .iter()
        .skip(2 * ZERO_WRITE_CHUNK)
        .all(|&value| value == 2.0));
    let mut output = vec![-0.0];
    let mut polls = 0;
    let result = append_zero_row_with_poll(&mut output, 4 * ZERO_WRITE_CHUNK, || {
        polls += 1;
        if polls == 3 {
            Err(NyError::DeadlineExceeded(
                "injected appended-row deadline".into(),
            ))
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert_eq!(output.len(), 1 + 2 * ZERO_WRITE_CHUNK);
    assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());
    assert!(output[1..].iter().all(|&value| value == 0.0));
}
