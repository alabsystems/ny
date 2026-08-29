// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Containment tests for the interval-constant layer.
//!
//! Every test here asks the SAME question in a different propagation mode: does
//! the bound the layer publishes hold for EVERY constant inside the carried
//! enclosure, and every activation inside the input box? A carrier that is
//! sound at the fold and then dropped by a propagation path is indistinguishable
//! in its output from the unsound point fold, so the fold-side tests in
//! `ny-onnx` are only half the argument — these are the other half.

use super::{IntervalConstKind, IntervalConstLayer};
use crate::layers::common::BoundPropagation;
use crate::layers::linear::LinearLayer;
use crate::{Layer, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Deterministic xorshift in [-1, 1); no dev-dependency, reproducible failures.
struct Rng(u64);

impl Rng {
    fn next_unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        f32::from(((self.0 >> 33) & 0xFFFF) as u16) / 32768.0 - 1.0
    }
}

fn sample_layer() -> (LinearLayer, Array2<f32>) {
    // 3 outputs x 4 inputs, radii deliberately several orders of magnitude
    // apart so a rule that used a single scalar radius would be caught.
    let center = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.5, -1.25, 2.0, 0.125, -0.75, 0.25, -3.5, 1.0, 2.25, 0.0, -0.5, 4.0,
        ],
    )
    .expect("shape");
    let radius = Array2::from_shape_vec(
        (3, 4),
        vec![
            1e-3, 5e-5, 2e-4, 1e-6, 3e-4, 1e-5, 7e-4, 2e-5, 4e-4, 1e-7, 6e-5, 8e-4,
        ],
    )
    .expect("shape");
    (
        LinearLayer::new(center, None).expect("center layer"),
        radius,
    )
}

/// IBP: for EVERY weight in the enclosure and EVERY activation in the box, the
/// concrete output lies inside the published interval.
#[test]
fn interval_linear_ibp_contains_every_weight_and_every_activation() {
    let (center_layer, radius) = sample_layer();
    let center = center_layer.weight().clone();
    let layer = IntervalConstLayer::try_new_linear(center_layer, radius.clone()).expect("layer");

    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, -0.5, 0.25, -2.0]).expect("shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.75, -0.5]).expect("shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input box");
    let out = layer.propagate_ibp(&input).expect("ibp");

    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    for _ in 0..20_000 {
        // A weight anywhere in [center - radius, center + radius].
        let mut weight = center.clone();
        for (w, r) in weight.iter_mut().zip(radius.iter()) {
            *w += rng.next_unit() * *r;
        }
        // An activation anywhere in the box.
        let mut x = Array1::<f32>::zeros(4);
        for j in 0..4 {
            let t = f32::midpoint(rng.next_unit(), 1.0);
            x[j] = lower[j] + t * (upper[j] - lower[j]);
        }
        for i in 0..3 {
            let mut acc = 0.0_f64;
            for j in 0..4 {
                acc += f64::from(weight[[i, j]]) * f64::from(x[j]);
            }
            assert!(
                f64::from(out.lower()[i]) <= acc && acc <= f64::from(out.upper()[i]),
                "IBP interval [{}, {}] excludes a realizable output {acc:e} at row {i}",
                out.lower()[i],
                out.upper()[i]
            );
        }
    }
}

/// CROWN backward: compose an identity objective through the layer, concretize
/// over the input box, and check the same realizable outputs are contained.
///
/// This is the test that would FAIL if the enclosure were dropped: the composed
/// coefficients are the MIDPOINT product `A * Wc`, and only the certified
/// per-coefficient error channel carries the `|A| * R` term that makes the
/// concretized bound valid for the whole enclosure.
#[test]
fn interval_linear_crown_backward_contains_every_weight_and_every_activation() {
    let (center_layer, radius) = sample_layer();
    let center = center_layer.weight().clone();
    let layer = IntervalConstLayer::try_new_linear(center_layer, radius.clone()).expect("layer");

    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, -0.5, 0.25, -2.0]).expect("shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.75, -0.5]).expect("shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input box");

    // Identity objective over the layer's 3 outputs.
    let identity = Array2::<f32>::from_shape_fn((3, 3), |(i, j)| f32::from(u8::from(i == j)));
    let bounds = LinearBounds::new(
        identity.clone(),
        Array1::zeros(3),
        identity,
        Array1::zeros(3),
    )
    .expect("bounds");

    let composed = layer.propagate_linear(&bounds).expect("crown backward");
    assert!(
        composed.has_coeff_err(),
        "the enclosure must land in the certified coefficient-error channel; without it the \
         composed coefficients are the midpoint product and the bound is too tight"
    );
    let concrete = composed.concretize_sound(&input);

    let mut rng = Rng(0x0FED_CBA9_8765_4321);
    for _ in 0..20_000 {
        let mut weight = center.clone();
        for (w, r) in weight.iter_mut().zip(radius.iter()) {
            *w += rng.next_unit() * *r;
        }
        let mut x = Array1::<f32>::zeros(4);
        for j in 0..4 {
            let t = f32::midpoint(rng.next_unit(), 1.0);
            x[j] = lower[j] + t * (upper[j] - lower[j]);
        }
        for i in 0..3 {
            let mut acc = 0.0_f64;
            for j in 0..4 {
                acc += f64::from(weight[[i, j]]) * f64::from(x[j]);
            }
            assert!(
                f64::from(concrete.lower()[i]) <= acc && acc <= f64::from(concrete.upper()[i]),
                "concretized CROWN bound [{}, {}] excludes a realizable output {acc:e} at row {i}",
                concrete.lower()[i],
                concrete.upper()[i]
            );
        }
    }
}

/// The discharge must be STRICTLY WIDER than the midpoint-only bound. A rule
/// that computed the error term but never attached it would pass the
/// containment tests by luck on a small radius; this pins the difference.
#[test]
fn the_enclosure_actually_widens_the_bound_relative_to_the_midpoint() {
    let (center_layer, radius) = sample_layer();
    let midpoint_only = center_layer.clone();
    let layer = IntervalConstLayer::try_new_linear(center_layer, radius).expect("layer");

    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, -0.5, 0.25, -2.0]).expect("shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.75, -0.5]).expect("shape");
    let input = BoundedTensor::new(lower, upper).expect("input box");

    let exact = midpoint_only.propagate_ibp(&input).expect("midpoint ibp");
    let carried = layer.propagate_ibp(&input).expect("interval ibp");
    for i in 0..3 {
        assert!(
            carried.lower()[i] < exact.lower()[i] && carried.upper()[i] > exact.upper()[i],
            "row {i}: the carried IBP interval is not strictly wider than the midpoint's"
        );
    }
}

/// The offset form: `y = x + c` for every `c` in the enclosure, in IBP and in
/// CROWN backward alike.
#[test]
fn interval_offset_contains_every_constant_in_both_modes() {
    let center = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, -2.0, 7.25]).expect("shape");
    let radius = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1e-4, 3e-6, 5e-5]).expect("shape");
    let layer =
        IntervalConstLayer::try_new_offset(center.clone(), radius.clone(), 3).expect("layer");

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 2.5]).expect("shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 4.0, 3.5]).expect("shape");
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input box");
    let ibp = layer.propagate_ibp(&input).expect("ibp");

    let identity = Array2::<f32>::from_shape_fn((3, 3), |(i, j)| f32::from(u8::from(i == j)));
    let bounds = LinearBounds::new(
        identity.clone(),
        Array1::zeros(3),
        identity,
        Array1::zeros(3),
    )
    .expect("bounds");
    let crown = layer
        .propagate_linear(&bounds)
        .expect("crown")
        .concretize_sound(&input);

    let mut rng = Rng(0x2222_3333_4444_5555);
    for _ in 0..20_000 {
        for i in 0..3 {
            let c = f64::from(center[i]) + f64::from(rng.next_unit()) * f64::from(radius[i]);
            let t = f64::from(f32::midpoint(rng.next_unit(), 1.0));
            let x = f64::from(lower[i]) + t * (f64::from(upper[i]) - f64::from(lower[i]));
            let y = x + c;
            assert!(
                f64::from(ibp.lower()[i]) <= y && y <= f64::from(ibp.upper()[i]),
                "IBP offset interval excludes a realizable output at {i}"
            );
            assert!(
                f64::from(crown.lower()[i]) <= y && y <= f64::from(crown.upper()[i]),
                "CROWN offset interval excludes a realizable output at {i}"
            );
        }
    }
}

/// A ZERO radius is refused at construction in both forms: that is the exact
/// case and must be lowered as an ordinary point layer, or the "an exact fold is
/// indistinguishable from today" property is lost.
#[test]
fn a_zero_radius_is_refused_rather_than_carried() {
    let (center_layer, _) = sample_layer();
    assert!(
        IntervalConstLayer::try_new_linear(center_layer, Array2::zeros((3, 4))).is_err(),
        "a zero-radius interval Linear must be refused"
    );
    let center = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("shape");
    assert!(
        IntervalConstLayer::try_new_offset(center, ArrayD::zeros(IxDyn(&[2])), 2).is_err(),
        "a zero-radius interval Offset must be refused"
    );
}

/// A negative or non-finite radius is refused: it is not an enclosure.
#[test]
fn a_malformed_radius_is_refused() {
    let (center_layer, _) = sample_layer();
    let mut bad = Array2::<f32>::from_elem((3, 4), 1e-4);
    bad[[1, 2]] = -1e-4;
    assert!(IntervalConstLayer::try_new_linear(center_layer.clone(), bad).is_err());
    let mut infinite = Array2::<f32>::from_elem((3, 4), 1e-4);
    infinite[[0, 0]] = f32::INFINITY;
    assert!(IntervalConstLayer::try_new_linear(center_layer, infinite).is_err());
}

/// (e) EVERY UNTAUGHT CONSUMER REFUSES. The batched CROWN backward is the one
/// the adversarial pass flagged as most likely to be quietly wrong (its error
/// channel is selected by a `matches!` predicate that does not fail closed), so
/// it is pinned as an explicit refusal rather than left to inspection.
#[test]
fn batched_crown_backward_refuses_an_interval_constant() {
    use crate::BatchedLinearBounds;
    let (center_layer, radius) = sample_layer();
    let layer = Layer::IntervalConst(
        IntervalConstLayer::try_new_linear(center_layer, radius).expect("layer"),
    );
    let bounds = BatchedLinearBounds::new(
        ArrayD::zeros(IxDyn(&[1, 3, 3])),
        ArrayD::zeros(IxDyn(&[1, 3])),
        ArrayD::zeros(IxDyn(&[1, 3, 3])),
        ArrayD::zeros(IxDyn(&[1, 3])),
        vec![1, 3],
        vec![1, 3],
    )
    .expect("batched bounds");
    let result = layer.propagate_crown_backward_batched(&bounds, None, None);
    assert!(
        matches!(result, Err(ny_core::NyError::UnsupportedOp(_))),
        "batched CROWN must REFUSE an interval constant rather than silently \
         propagating the midpoint; got {result:?}"
    );
}

/// The layer reports itself distinctly, so a trace or a diagnostic cannot
/// mistake a carried enclosure for an ordinary Linear.
#[test]
fn the_layer_type_is_distinguishable_from_a_plain_linear() {
    let (center_layer, radius) = sample_layer();
    let layer = Layer::IntervalConst(
        IntervalConstLayer::try_new_linear(center_layer, radius).expect("layer"),
    );
    assert_eq!(layer.layer_type(), "IntervalLinear");
    assert!(matches!(
        layer,
        Layer::IntervalConst(ref ic) if matches!(ic.kind, IntervalConstKind::Linear { .. })
    ));
}

/// An offset whose constant would EXPAND the activation is refused at
/// construction: its CROWN backward keeps `A' = A`, which is only an identity
/// when the output width equals the activation width.
#[test]
fn an_expanding_offset_is_refused() {
    let center = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("shape");
    let radius = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1e-5, 1e-5, 1e-5]).expect("shape");
    assert!(
        IntervalConstLayer::try_new_offset(center, radius, 2).is_err(),
        "a constant of 3 elements over an activation of 2 must be refused"
    );
}
