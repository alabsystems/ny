// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use num_rational::BigRational;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ZeroWeightEngine(AtomicUsize);
impl GemmEngine for ZeroWeightEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert!(b.iter().all(|&v| v == 0.0));
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(vec![0.0; m * n])
    }
}

fn fixture(positions: usize, position: usize, mask: usize, bias: f32, error: f32) -> LinearBounds {
    let mut b = LinearBounds::new(
        Array2::ones((1, positions)),
        Array1::from_vec(vec![-(positions as f32) * bias]),
        Array2::ones((1, positions)),
        Array1::from_vec(vec![-(positions as f32) * bias]),
    )
    .unwrap();
    let mut e = Array2::zeros((1, positions));
    e[[0, position]] = error;
    b.lower_a_err = (mask & 1 != 0).then(|| e.clone());
    b.upper_a_err = (mask & 2 != 0).then_some(e);
    b
}

#[test]
fn sequence_bias_error_multi_domain_counterexample_contains_later_carrier_for_each_domain() {
    let layer = LinearLayer::new(
        Array2::zeros((1, 1)),
        Some(Array1::from_vec(vec![1048576.0])),
    )
    .unwrap();
    let bounds: Vec<_> = [3, 1, 2]
        .into_iter()
        .map(|mask| fixture(2, 1, mask, 1048576.0, 2f32.powi(-23)))
        .collect();
    let refs: Vec<_> = bounds.iter().collect();
    let engine = ZeroWeightEngine(AtomicUsize::new(0));
    let out = propagate_linear_batched_with_engine(&layer, &refs, &engine).unwrap();
    assert_eq!(engine.0.load(Ordering::Relaxed), 4);
    assert_eq!(out.len(), 3);
    for (b, mask) in out.iter().zip([3, 1, 2]) {
        if mask & 1 != 0 {
            assert!(
                b.lower_b()[0] <= -0.0625,
                "later multi-domain lower carrier excluded: bias={} exact=-0.0625",
                b.lower_b()[0]
            );
        }
        if mask & 2 != 0 {
            assert!(
                b.upper_b()[0] >= 0.0625,
                "later multi-domain upper carrier excluded: bias={} exact=0.0625",
                b.upper_b()[0]
            );
        }
    }
}

#[test]
fn sequence_bias_error_multi_domain_all_positions_masks_poison_zero_bias_and_legacy_bits() {
    for positions in [1, 2, 3, 17] {
        for position in 0..positions {
            for bias in [0.0, 1048576.0] {
                for error in [2f32.powi(-23), f32::NAN, -1.0, f32::INFINITY] {
                    let layer =
                        LinearLayer::new(Array2::zeros((1, 1)), Some(Array1::from_vec(vec![bias])))
                            .unwrap();
                    let bounds: Vec<_> = [3, 1, 2]
                        .into_iter()
                        .map(|mask| fixture(positions, position, mask, bias, error))
                        .collect();
                    let refs: Vec<_> = bounds.iter().collect();
                    let engine = ZeroWeightEngine(AtomicUsize::new(0));
                    let out = propagate_linear_batched_with_engine(&layer, &refs, &engine).unwrap();
                    assert_eq!(engine.0.load(Ordering::Relaxed), 2 * positions);
                    for (b, mask) in out.iter().zip([3, 1, 2]) {
                        assert!(b
                            .lower_a()
                            .iter()
                            .chain(b.upper_a().iter())
                            .all(|&v| v == 0.0));
                        assert!(b
                            .lower_a_err()
                            .unwrap()
                            .iter()
                            .chain(b.upper_a_err().unwrap().iter())
                            .all(|&v| v == 0.0));
                        if bias == 0.0 {
                            assert_eq!(b.lower_b()[0], 0.0);
                            assert_eq!(b.upper_b()[0], 0.0);
                        } else if !error.is_finite() || error < 0.0 {
                            if mask & 1 != 0 {
                                assert_eq!(b.lower_b()[0], f32::NEG_INFINITY);
                            }
                            if mask & 2 != 0 {
                                assert_eq!(b.upper_b()[0], f32::INFINITY);
                            }
                        } else {
                            if mask & 1 != 0 {
                                assert!(b.lower_b()[0] <= -0.0625);
                            }
                            if mask & 2 != 0 {
                                assert!(b.upper_b()[0] >= 0.0625);
                            }
                            if positions == 1 && mask == 3 {
                                assert_eq!(b.lower_b()[0].to_bits(), 0xbe000001);
                                assert_eq!(b.upper_b()[0].to_bits(), 0x3e000001);
                            }
                        }
                    }
                }
            }
        }
    }
    let layer = LinearLayer::new(Array2::zeros((1, 1)), Some(Array1::zeros(1))).unwrap();
    let engine = ZeroWeightEngine(AtomicUsize::new(0));
    assert!(propagate_linear_batched_with_engine(&layer, &[], &engine)
        .unwrap()
        .is_empty());
    assert_eq!(engine.0.load(Ordering::Relaxed), 0);
}

#[test]
fn sequence_bias_error_multi_domain_exact_rational_shared_bias_matches_corrected_cpu() {
    let positions = 3;
    let rows = 2;
    let k = 3;
    let bias = Array1::from_vec(vec![1.0e20, -1.0e20, 0.375]);
    let layer = LinearLayer::new(Array2::zeros((k, 2)), Some(bias.clone())).unwrap();
    let lower = Array2::from_shape_fn((rows, k * positions), |(i, j)| ((i + j) % 4) as f32 - 1.0);
    let upper = -&lower;
    let error = Array2::from_elem((rows, k * positions), 0.03125);
    let bounds: Vec<_> = [3, 1, 2]
        .into_iter()
        .map(|mask| {
            let mut b = LinearBounds::new(
                lower.clone(),
                Array1::zeros(rows),
                upper.clone(),
                Array1::zeros(rows),
            )
            .unwrap();
            b.lower_a_err = (mask & 1 != 0).then(|| error.clone());
            b.upper_a_err = (mask & 2 != 0).then(|| error.clone());
            b
        })
        .collect();
    let refs: Vec<_> = bounds.iter().collect();
    let engine = ZeroWeightEngine(AtomicUsize::new(0));
    let out = propagate_linear_batched_with_engine(&layer, &refs, &engine).unwrap();
    assert_eq!(engine.0.load(Ordering::Relaxed), 2 * positions);
    let r = |v: f32| BigRational::from_float(f32_to_f64_exact(v)).unwrap();
    for ((b, input), mask) in out.iter().zip(bounds.iter()).zip([3, 1, 2]) {
        let cpu = layer
            .propagate_linear_pollable_cpu(input, None, None)
            .unwrap();
        for i in 0..rows {
            let mut lo = BigRational::from_integer(0.into());
            let mut up = lo.clone();
            let mut penalty = lo.clone();
            for pos in 0..positions {
                for j in 0..k {
                    lo += r(lower[[i, pos * k + j]]) * r(bias[j]);
                    up += r(upper[[i, pos * k + j]]) * r(bias[j]);
                    penalty += r(error[[i, pos * k + j]]) * r(bias[j].abs());
                }
            }
            assert!(
                r(b.lower_b()[i])
                    <= lo
                        - if mask & 1 != 0 {
                            penalty.clone()
                        } else {
                            BigRational::from_integer(0.into())
                        }
            );
            assert!(
                r(b.upper_b()[i])
                    >= up
                        + if mask & 2 != 0 {
                            penalty
                        } else {
                            BigRational::from_integer(0.into())
                        }
            );
            assert_eq!(b.lower_b()[i].to_bits(), cpu.lower_b()[i].to_bits());
            assert_eq!(b.upper_b()[i].to_bits(), cpu.upper_b()[i].to_bits());
        }
    }
}
