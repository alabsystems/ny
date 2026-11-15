// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::linear::crown_batched_soa::domain_pending_bounds;
use ny_core::NaiveCpuGemmEngine;
use std::time::Duration;

fn compare(a: &LinearBounds, b: &LinearBounds) {
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for (a, b) in [(a.lower_a(), b.lower_a()), (a.upper_a(), b.upper_a())] {
        assert_eq!(a.dim(), b.dim());
        assert_eq!(
            bits(a.iter().copied().collect()),
            bits(b.iter().copied().collect())
        );
    }
    for (a, b) in [(a.lower_b(), b.lower_b()), (a.upper_b(), b.upper_b())] {
        assert_eq!(
            bits(a.iter().copied().collect()),
            bits(b.iter().copied().collect())
        );
    }
    for (a, b) in [
        (a.lower_a_err(), b.lower_a_err()),
        (a.upper_a_err(), b.upper_a_err()),
    ] {
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.dim(), b.dim());
            assert_eq!(
                bits(a.iter().copied().collect()),
                bits(b.iter().copied().collect())
            );
        }
    }
}

#[test]
fn coalesced_domains_match_reference_bits_and_exact_markers() {
    for (cols, out, rows) in [
        (1, 1, 1),
        (5, 7, 3),
        (50, 50, 5),
        (65, 33, 4),
        (120, 120, 7),
    ] {
        for biased in [false, true] {
            for stressed in [false, true] {
                let layer = LinearLayer::new(
                    Array2::from_shape_fn((cols, out), |(i, j)| {
                        ((i * 17 + j * 13 + 1) % 31) as f32 / 17.0 - 0.7
                    }),
                    biased.then(|| Array1::from_shape_fn(cols, |i| i as f32 / 11.0 - 0.3)),
                )
                .unwrap();
                // LinearLayer weights are [output,input]; backward cols=output.
                let domains = 13;
                let cells = domains * rows * cols;
                let lower: Vec<f32> = (0..cells)
                    .map(|i| match (i % 113, stressed) {
                        (0, _) => -0.0,
                        (1, true) => 1.0e30,
                        (2, true) => -1.0e30,
                        (3, true) => f32::NAN,
                        (4, true) => f32::INFINITY,
                        (5, true) => f32::NEG_INFINITY,
                        _ => (i % 37) as f32 / 19.0 - 0.9,
                    })
                    .collect();
                let upper: Vec<f32> = lower.iter().map(|v| -v * 0.7).collect();
                let lower_b: Vec<f32> =
                    (0..domains * rows).map(|i| i as f32 / 13.0 - 0.2).collect();
                let upper_b: Vec<f32> = lower_b.iter().map(|v| -v).collect();
                let active: Vec<bool> = (0..domains).map(|d| d != 12).collect();
                let lp: Vec<bool> = (0..domains).map(|d| (d % 4) & 1 != 0).collect();
                let up: Vec<bool> = (0..domains).map(|d| (d % 4) & 2 != 0).collect();
                let le = vec![0.00003; cells];
                let ue = vec![0.00007; cells];
                let input = SoaLinearBackwardInput {
                    n_domains: domains,
                    rows,
                    cols,
                    active: &active,
                    lower_a: &lower,
                    upper_a: &upper,
                    lower_b: &lower_b,
                    upper_b: &upper_b,
                    lower_a_err: Some(&le),
                    upper_a_err: Some(&ue),
                    lower_err_present: &lp,
                    upper_err_present: &up,
                };
                for deadline in [None, Some(Instant::now() + Duration::from_secs(30))] {
                    let (mut actual, work) =
                        try_coalesced(&layer, &input, &NaiveCpuGemmEngine, deadline, usize::MAX)
                            .unwrap()
                            .unwrap();
                    assert_eq!(work.shared_calls, 4);
                    assert_eq!(work.packed_domains, 12);
                    if !stressed {
                        let bounds: Vec<&LinearBounds> = actual
                            .iter()
                            .filter_map(|slot| slot.as_ref())
                            .map(|result| result.as_ref().unwrap())
                            .collect();
                        assert!(
                            bounds.iter().all(|b| b
                                .lower_b()
                                .iter()
                                .chain(b.upper_b().iter())
                                .all(|v| v.is_finite())),
                            "ordinary finite rows must not pass by conservative fallback"
                        );
                        assert!(
                            bounds.iter().any(|b| b
                                .lower_a()
                                .iter()
                                .chain(b.upper_a().iter())
                                .any(|&v| v != 0.0)),
                            "ordinary finite phase must exercise nonzero coefficient products"
                        );
                    }
                    for d in 0..domains {
                        if !active[d] {
                            assert!(actual[d].is_none());
                            continue;
                        }
                        let pending = domain_pending_bounds(&input, d, rows, cols).unwrap();
                        let _par = NestedFaerParGuard::new();
                        let reference = layer
                            .propagate_linear_pollable_cpu(
                                &pending,
                                Some(&NaiveCpuGemmEngine),
                                deadline,
                            )
                            .unwrap();
                        compare(&actual[d].take().unwrap().unwrap(), &reference);
                    }
                }
            }
        }
    }
}

#[test]
fn coalescing_declines_budget_and_sequence_layout_without_changing_reference() {
    let layer = LinearLayer::new(Array2::eye(2), None).unwrap();
    let a = vec![1.0; 8];
    let b = vec![0.0; 4];
    let active = [true, true];
    let present = [false, false];
    let mut input = SoaLinearBackwardInput {
        n_domains: 2,
        rows: 2,
        cols: 2,
        active: &active,
        lower_a: &a,
        upper_a: &a,
        lower_b: &b,
        upper_b: &b,
        lower_a_err: None,
        upper_a_err: None,
        lower_err_present: &present,
        upper_err_present: &present,
    };
    assert!(try_coalesced(&layer, &input, &NaiveCpuGemmEngine, None, 0)
        .unwrap()
        .is_none());
    input.cols = 4;
    assert!(
        try_coalesced(&layer, &input, &NaiveCpuGemmEngine, None, usize::MAX)
            .unwrap()
            .is_none()
    );
}

#[test]
fn coalesced_original_deadline_and_malformed_planes_refuse_before_kernel() {
    let layer = LinearLayer::new(Array2::eye(2), None).unwrap();
    let a = vec![1.0; 8];
    let b = vec![0.0; 4];
    let active = [true, true];
    let present = [false, false];
    let mut input = SoaLinearBackwardInput {
        n_domains: 2,
        rows: 2,
        cols: 2,
        active: &active,
        lower_a: &a,
        upper_a: &a,
        lower_b: &b,
        upper_b: &b,
        lower_a_err: None,
        upper_a_err: None,
        lower_err_present: &present,
        upper_err_present: &present,
    };
    assert!(matches!(
        try_coalesced(
            &layer,
            &input,
            &NaiveCpuGemmEngine,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            usize::MAX
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    input.lower_a = &a[..7];
    assert!(matches!(
        try_coalesced(&layer, &input, &NaiveCpuGemmEngine, None, usize::MAX),
        Err(NyError::InvalidSpec(_))
    ));
    input.lower_a = &a;
    input.lower_err_present = &[true, false];
    assert!(matches!(
        try_coalesced(&layer, &input, &NaiveCpuGemmEngine, None, usize::MAX),
        Err(NyError::InvalidSpec(_))
    ));
}
