// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::layer::SoftmaxLayer;
use super::super::utils;
use super::sound::{softmax_objective_envelope_gate, with_softmax_objective_envelope_for_test};
use crate::layers::softmax::bounds::constant_bounds_from_output;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::VerificationSoundnessMode;
use ny_tensor::BoundedTensor;

/// Helper: check that LSE affine bounds contain softmax at all vertex samples.
fn assert_lse_bounds_enclose_vertices(
    layer: &SoftmaxLayer,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
    tol: f32,
) {
    let (lower_a, lower_b, upper_a, upper_b) = layer
        .softmax_lse_affine_bounds(pre_lower, pre_upper)
        .expect("LSE affine bounds should succeed");

    let n = pre_lower.len();
    let mut samples = vec![(pre_lower + pre_upper) / 2.0];
    for mask in 0..(1usize << n) {
        let mut sample = pre_lower.clone();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                sample[i] = pre_upper[i];
            }
        }
        samples.push(sample);
    }

    for sample in &samples {
        let softmax = utils::softmax_1d(sample);
        for i in 0..n {
            let lb = lower_a.row(i).dot(sample) + lower_b[i];
            let ub = upper_a.row(i).dot(sample) + upper_b[i];
            assert!(
                lb <= softmax[i] + tol,
                "lower bound violated at dim {}: {} > softmax {}",
                i,
                lb,
                softmax[i]
            );
            assert!(
                ub + tol >= softmax[i],
                "upper bound violated at dim {}: {} < softmax {}",
                i,
                ub,
                softmax[i]
            );
        }
    }
}

// Audited logit-box endpoints, recorded at the precision they were audited at.
// Truncating a literal moves the fixture's box and silently re-points every
// envelope soundness/dominance assertion below at a different problem.
#[allow(clippy::excessive_precision)]
const OBJECTIVE_ENVELOPE_LOGIT_LOWER: [f32; 5] = [
    -2.320_387,
    -1.335_145_2,
    -1.678_693,
    2.556_615_1,
    0.226_314_49,
];
const OBJECTIVE_ENVELOPE_LOGIT_UPPER: [f32; 5] = [
    -2.171_673_3,
    -1.209_960_5,
    -1.465_777_4,
    2.670_337_2,
    0.479_345_98,
];

// Audited split-direction witnesses, plus deterministic both-LSE and
// both-constant witnesses over the same logit box.
const OBJECTIVE_ENVELOPE_ROWS: [[f32; 5]; 4] = [
    [-2.759_585, -4.884_014, 3.840_507, -4.971, 4.705_489],
    [3.781_244, 4.281_342, -2.547_796, 4.963_99, -4.977_438],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [
        -3.202_033_5,
        -3.738_530_4,
        -0.248_358_38,
        3.513_572_2,
        -4.946_847,
    ],
];

fn objective_envelope_fixture() -> (SoftmaxLayer, BoundedTensor, LinearBounds) {
    let layer = SoftmaxLayer::new(-1);
    let pre = BoundedTensor::new(
        Array1::from_vec(OBJECTIVE_ENVELOPE_LOGIT_LOWER.to_vec()).into_dyn(),
        Array1::from_vec(OBJECTIVE_ENVELOPE_LOGIT_UPPER.to_vec()).into_dyn(),
    )
    .expect("objective-envelope logit box");
    let rows: Vec<f32> = OBJECTIVE_ENVELOPE_ROWS.into_iter().flatten().collect();
    let a = Array2::from_shape_vec((4, 5), rows).expect("four objective rows");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(4), a, Array1::zeros(4))
        .expect("objective-envelope incoming bounds");
    (layer, pre, bounds)
}

fn objective_envelope_candidates(
    layer: &SoftmaxLayer,
    pre: &BoundedTensor,
    bounds: &LinearBounds,
) -> (LinearBounds, LinearBounds, LinearBounds) {
    let pre_lower = pre
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1-D lower");
    let pre_upper = pre
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .expect("1-D upper");
    let lse = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_with_bounds_1d_sound(bounds, &pre_lower, &pre_upper)
    })
    .expect("LSE candidate");
    let softmax_box = SoftmaxLayer::propagate_ibp_with_axis(pre, 0).expect("exact 1-D IBP");
    let constant = constant_bounds_from_output(bounds, &softmax_box).expect("constant candidate");
    let envelope = with_softmax_objective_envelope_for_test(true, || {
        layer.propagate_linear_with_bounds_1d_sound(bounds, &pre_lower, &pre_upper)
    })
    .expect("objective envelope");
    (lse, constant, envelope)
}

fn assert_lower_row_exact(actual: &LinearBounds, expected: &LinearBounds, row: usize) {
    assert_eq!(actual.lower_a.row(row), expected.lower_a.row(row));
    assert_eq!(
        actual.lower_b[row].to_bits(),
        expected.lower_b[row].to_bits()
    );
}

fn assert_upper_row_exact(actual: &LinearBounds, expected: &LinearBounds, row: usize) {
    assert_eq!(actual.upper_a.row(row), expected.upper_a.row(row));
    assert_eq!(
        actual.upper_b[row].to_bits(),
        expected.upper_b[row].to_bits()
    );
}

fn grouped_objective_envelope_fixture(shape: &[usize]) -> (BoundedTensor, LinearBounds) {
    let num_groups = shape.iter().product::<usize>() / 5;
    let mut lower = Vec::with_capacity(num_groups * 5);
    let mut upper = Vec::with_capacity(num_groups * 5);
    for _ in 0..num_groups {
        lower.extend_from_slice(&OBJECTIVE_ENVELOPE_LOGIT_LOWER);
        upper.extend_from_slice(&OBJECTIVE_ENVELOPE_LOGIT_UPPER);
    }
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).expect("grouped lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), upper).expect("grouped upper shape"),
    )
    .expect("grouped logit box");

    let mut rows = Vec::with_capacity(4 * num_groups * 5);
    for objective in OBJECTIVE_ENVELOPE_ROWS {
        for _ in 0..num_groups {
            rows.extend_from_slice(&objective);
        }
    }
    let a = Array2::from_shape_vec((4, num_groups * 5), rows).expect("grouped objectives");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(4), a, Array1::zeros(4))
        .expect("grouped incoming bounds");
    (pre, bounds)
}

fn assert_grouped_envelope_dominates_and_is_sound(
    baseline: &LinearBounds,
    envelope: &LinearBounds,
    pre: &BoundedTensor,
) {
    let baseline_concrete = baseline.concretize_sound(pre);
    let envelope_concrete = envelope.concretize_sound(pre);
    let mut strict = false;
    for row in 0..4 {
        let baseline_lower = baseline_concrete.lower()[[row]];
        let baseline_upper = baseline_concrete.upper()[[row]];
        let envelope_lower = envelope_concrete.lower()[[row]];
        let envelope_upper = envelope_concrete.upper()[[row]];
        assert!(
            envelope_lower + 1e-6 >= baseline_lower,
            "row {row}: envelope lower {envelope_lower} < LSE lower {baseline_lower}"
        );
        assert!(
            envelope_upper <= baseline_upper + 1e-6,
            "row {row}: envelope upper {envelope_upper} > LSE upper {baseline_upper}"
        );
        strict |= envelope_lower > baseline_lower + 1e-5 || envelope_upper + 1e-5 < baseline_upper;
    }
    assert!(
        strict,
        "grouped envelope did not make any strict improvement"
    );

    let lower: Vec<f32> = pre.lower().iter().copied().collect();
    let upper: Vec<f32> = pre.upper().iter().copied().collect();
    let num_groups = lower.len() / 5;
    for mask in 0..(1usize << lower.len()) {
        let sample: Vec<f32> = (0..lower.len())
            .map(|idx| {
                if (mask >> idx) & 1 == 1 {
                    upper[idx]
                } else {
                    lower[idx]
                }
            })
            .collect();
        let mut softmax_groups = Vec::with_capacity(sample.len());
        for group in 0..num_groups {
            let start = group * 5;
            let values = Array1::from_vec(sample[start..start + 5].to_vec());
            softmax_groups.extend(utils::softmax_1d(&values));
        }
        let sample = Array1::from_vec(sample);
        for (row, objective) in OBJECTIVE_ENVELOPE_ROWS.iter().enumerate() {
            #[allow(unknown_lints)] // stock 1.95 clippy (public pin) does not know the lint below
            #[allow(clippy::chunks_exact_to_as_chunks)]
            let true_value: f32 = softmax_groups
                .chunks_exact(5)
                .map(|group| {
                    group
                        .iter()
                        .zip(objective)
                        .map(|(&value, &coefficient)| value * coefficient)
                        .sum::<f32>()
                })
                .sum();
            let lower_bound = envelope.lower_a.row(row).dot(&sample) + envelope.lower_b[row];
            let upper_bound = envelope.upper_a.row(row).dot(&sample) + envelope.upper_b[row];
            assert!(
                lower_bound <= true_value + 1e-3,
                "row {row}, mask {mask}: lower {lower_bound} > true {true_value}"
            );
            assert!(
                upper_bound + 1e-3 >= true_value,
                "row {row}, mask {mask}: upper {upper_bound} < true {true_value}"
            );
        }
    }
}

// =========================================================================
// LSE affine bounds (softmax_lse_affine_bounds)
// =========================================================================

#[test]
fn lse_affine_bounds_enclose_samples_basic() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-1.2, 0.1, -0.4]);
    let pre_upper = Array1::from_vec(vec![1.1, 1.4, 0.9]);
    assert_lse_bounds_enclose_vertices(&layer, &pre_lower, &pre_upper, 1e-4);
}

#[test]
fn lse_affine_bounds_near_uniform() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![0.99, 1.0, 1.01]);
    let pre_upper = Array1::from_vec(vec![1.01, 1.02, 1.03]);
    assert_lse_bounds_enclose_vertices(&layer, &pre_lower, &pre_upper, 1e-4);
}

#[test]
fn lse_affine_bounds_dominant_element() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-5.0, -5.0, 5.0]);
    let pre_upper = Array1::from_vec(vec![0.0, 0.0, 10.0]);
    assert_lse_bounds_enclose_vertices(&layer, &pre_lower, &pre_upper, 1e-3);
}

#[test]
fn lse_affine_bounds_wide_spread() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-10.0, -5.0, 0.0, 5.0]);
    let pre_upper = Array1::from_vec(vec![-5.0, 0.0, 5.0, 10.0]);
    assert_lse_bounds_enclose_vertices(&layer, &pre_lower, &pre_upper, 1e-3);
}

#[test]
fn lse_affine_bounds_2_elements() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-1.0, 0.0]);
    let pre_upper = Array1::from_vec(vec![1.0, 2.0]);
    assert_lse_bounds_enclose_vertices(&layer, &pre_lower, &pre_upper, 1e-4);
}

#[test]
fn lse_affine_bounds_point_interval() {
    let layer = SoftmaxLayer::new(-1);
    let vals = Array1::from_vec(vec![0.5, 1.0, -0.5]);
    let (lower_a, lower_b, upper_a, upper_b) = layer
        .softmax_lse_affine_bounds(&vals, &vals)
        .expect("point interval should succeed");
    let softmax = utils::softmax_1d(&vals);
    for i in 0..3 {
        let lb = lower_a.row(i).dot(&vals) + lower_b[i];
        let ub = upper_a.row(i).dot(&vals) + upper_b[i];
        // Point interval (lower == upper): LSE affine bounds should match softmax to
        // f32 rounding precision. Softmax values are ~0.2-0.5, so ~1e-6 relative.
        assert!(
            (lb - softmax[i]).abs() < 1e-4,
            "point: lower {} != softmax {} (diff={})",
            lb,
            softmax[i],
            (lb - softmax[i]).abs()
        );
        assert!(
            (ub - softmax[i]).abs() < 1e-4,
            "point: upper {} != softmax {} (diff={})",
            ub,
            softmax[i],
            (ub - softmax[i]).abs()
        );
    }
}

// =========================================================================
// propagate_linear_with_bounds — 1D sound
// =========================================================================

#[test]
fn objective_envelope_gate_is_exact_string_and_default_dark() {
    assert!(!softmax_objective_envelope_gate(None));
    assert!(!softmax_objective_envelope_gate(Some("")));
    assert!(!softmax_objective_envelope_gate(Some("0")));
    assert!(!softmax_objective_envelope_gate(Some("true")));
    assert!(!softmax_objective_envelope_gate(Some("01")));
    assert!(softmax_objective_envelope_gate(Some("1")));
}

#[test]
fn objective_envelope_audit_fixture_exercises_all_four_row_choices() {
    let (layer, pre, bounds) = objective_envelope_fixture();
    let (lse, constant, envelope) = objective_envelope_candidates(&layer, &pre, &bounds);

    // Row 0 is the audited constant-lower / affine-upper witness.
    assert_lower_row_exact(&envelope, &constant, 0);
    assert_upper_row_exact(&envelope, &lse, 0);
    // Row 1 is the audited affine-lower / constant-upper witness.
    assert_lower_row_exact(&envelope, &lse, 1);
    assert_upper_row_exact(&envelope, &constant, 1);
    // Equal positive weights exploit sum(softmax)=1 and retain LSE on both
    // sides for this fixture; row 3 is independently constant on both sides.
    assert_lower_row_exact(&envelope, &lse, 2);
    assert_upper_row_exact(&envelope, &lse, 2);
    assert_lower_row_exact(&envelope, &constant, 3);
    assert_upper_row_exact(&envelope, &constant, 3);

    // Non-vacuity: every direction above genuinely distinguishes candidates.
    for row in 0..4 {
        assert_ne!(lse.lower_a.row(row), constant.lower_a.row(row));
        assert_ne!(lse.upper_a.row(row), constant.upper_a.row(row));
    }

    let lse_concrete = lse.concretize_sound(&pre);
    let constant_concrete = constant.concretize_sound(&pre);
    let envelope_concrete = envelope.concretize_sound(&pre);
    for row in 0..4 {
        assert!(
            envelope_concrete.lower()[[row]] >= lse_concrete.lower()[[row]]
                && envelope_concrete.lower()[[row]] >= constant_concrete.lower()[[row]],
            "row {row}: envelope lower did not dominate both candidates"
        );
        assert!(
            envelope_concrete.upper()[[row]] <= lse_concrete.upper()[[row]]
                && envelope_concrete.upper()[[row]] <= constant_concrete.upper()[[row]],
            "row {row}: envelope upper did not dominate both candidates"
        );
    }
}

#[test]
fn objective_envelope_default_off_is_bit_identical_to_lse_composition() {
    let (layer, pre, bounds) = objective_envelope_fixture();
    let pre_lower = pre
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let pre_upper = pre
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let (lower_a, lower_b, upper_a, upper_b) = layer
        .softmax_lse_affine_bounds(&pre_lower, &pre_upper)
        .expect("fixture has a finite LSE relaxation");
    let historical = layer
        .apply_affine_bounds(&bounds, &lower_a, &lower_b, &upper_a, &upper_b)
        .expect("historical LSE composition");
    let gated_off = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_with_bounds_1d_sound(&bounds, &pre_lower, &pre_upper)
    })
    .expect("default-off composition");

    assert_eq!(gated_off.lower_a, historical.lower_a);
    assert_eq!(gated_off.upper_a, historical.upper_a);
    assert_eq!(
        gated_off.lower_b.mapv(f32::to_bits),
        historical.lower_b.mapv(f32::to_bits)
    );
    assert_eq!(
        gated_off.upper_b.mapv(f32::to_bits),
        historical.upper_b.mapv(f32::to_bits)
    );
}

#[test]
fn objective_envelope_randomized_soundness_and_local_dominance() {
    fn next_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32
    }

    let layer = SoftmaxLayer::new(-1);
    let mut state = 0x5eed_5eed_cafe_babe_u64;
    for case in 0..64 {
        let mut lower = Array1::<f32>::zeros(3);
        let mut upper = Array1::<f32>::zeros(3);
        for idx in 0..3 {
            lower[idx] = -3.0 + 6.0 * next_unit(&mut state);
            upper[idx] = lower[idx] + 0.02 + 0.98 * next_unit(&mut state);
        }
        let pre = BoundedTensor::new(lower.clone().into_dyn(), upper.clone().into_dyn()).unwrap();

        let mut a = Array2::<f32>::zeros((6, 3));
        let mut bias = Array1::<f32>::zeros(6);
        for row in 0..6 {
            for col in 0..3 {
                a[[row, col]] = -5.0 + 10.0 * next_unit(&mut state);
            }
            bias[row] = -1.0 + 2.0 * next_unit(&mut state);
        }
        let incoming = LinearBounds::new(a.clone(), bias.clone(), a, bias).unwrap();
        let (lse, constant, envelope) = objective_envelope_candidates(&layer, &pre, &incoming);
        let lse_concrete = lse.concretize_sound(&pre);
        let constant_concrete = constant.concretize_sound(&pre);
        let envelope_concrete = envelope.concretize_sound(&pre);

        for row in 0..6 {
            assert!(
                envelope_concrete.lower()[[row]] >= lse_concrete.lower()[[row]]
                    && envelope_concrete.lower()[[row]] >= constant_concrete.lower()[[row]],
                "case {case}, row {row}: lower envelope lost local dominance"
            );
            assert!(
                envelope_concrete.upper()[[row]] <= lse_concrete.upper()[[row]]
                    && envelope_concrete.upper()[[row]] <= constant_concrete.upper()[[row]],
                "case {case}, row {row}: upper envelope lost local dominance"
            );
        }

        for mask in 0..8usize {
            let sample = Array1::from_shape_fn(3, |idx| {
                if (mask >> idx) & 1 == 1 {
                    upper[idx]
                } else {
                    lower[idx]
                }
            });
            let softmax = utils::softmax_1d(&sample);
            for row in 0..6 {
                let true_value = incoming.lower_a.row(row).dot(&softmax) + incoming.lower_b[row];
                let lower_bound = envelope.lower_a.row(row).dot(&sample) + envelope.lower_b[row];
                let upper_bound = envelope.upper_a.row(row).dot(&sample) + envelope.upper_b[row];
                assert!(
                    lower_bound <= true_value + 2e-3,
                    "case {case}, row {row}, mask {mask}: lower {lower_bound} > true {true_value}"
                );
                assert!(
                    upper_bound + 2e-3 >= true_value,
                    "case {case}, row {row}, mask {mask}: upper {upper_bound} < true {true_value}"
                );
            }
        }
    }
}

#[test]
fn crown_1d_sound_bounds_contain_samples() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-1.0, 0.0, 0.5]);
    let pre_upper = Array1::from_vec(vec![1.0, 2.0, 1.5]);
    let pre =
        BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn()).unwrap();
    let bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Check at vertices
    for mask in 0..(1usize << 3) {
        let mut sample = Array1::<f32>::zeros(3);
        for i in 0..3 {
            sample[i] = if (mask >> i) & 1 == 1 {
                pre_upper[i]
            } else {
                pre_lower[i]
            };
        }
        let softmax = utils::softmax_1d(&sample);
        for i in 0..3 {
            let lb = result.lower_a.row(i).dot(&sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(&sample) + result.upper_b[i];
            // Sound mode: LSE affine bounds are mathematically valid. f32 arithmetic
            // on 3-dim softmax with range [-1,2] yields ~1e-5 precision.
            assert!(
                lb <= softmax[i] + 1e-5,
                "sound 1D lower violated: dim={}, lb={}, softmax={}, diff={}",
                i,
                lb,
                softmax[i],
                lb - softmax[i]
            );
            assert!(
                ub >= softmax[i] - 1e-5,
                "sound 1D upper violated: dim={}, ub={}, softmax={}, diff={}",
                i,
                ub,
                softmax[i],
                softmax[i] - ub
            );
        }
    }
}

/// Regression (talker-softmax overflow family): at the talker attention QK^T
/// scale (±2.7e4), the row span drives a residual ratio-chord intermediate past
/// the f64 exp domain, so `softmax_lse_affine_bounds` returns `None` and the
/// caller must FAIL CLOSED to the sound constant `[0, 1]` bounds — never leaking
/// an inf/NaN coefficient. This test pins that overflow-fail-closed path: the
/// returned coefficients are finite, and the (trivial, since `[0,1]`) enclosure
/// holds. A non-trivial affine relaxation is exercised separately by
/// [`crown_1d_sound_finite_domain_relaxation_is_tighter_than_trivial`], which
/// stays within the LSE finite domain so `softmax_lse_affine_bounds` returns
/// `Some(..)`; the two together cover both the overflow fallback and the real
/// relaxation. (The overflow-free `exp(t - shift)`, `shift = max_i pre_upper[i]`
/// numerator evaluation is exact by shift-invariance `softmax(x)=softmax(x-c)`;
/// only the ratio-chord denominator term is not shift-bounded and overflows here.)
#[test]
fn crown_1d_sound_wide_qkt_overflow_fails_closed_to_finite_sound_bounds() {
    let layer = SoftmaxLayer::new(-1);
    // Talker QK^T IBP intermediates span ±2.7e4 at the failing epsilons; a naive
    // f32 exp of these overflows. Mix signs and magnitudes across the row.
    let pre_lower = Array1::from_vec(vec![-2.7e4, -1.0e4, -5.0, 1.3e4]);
    let pre_upper = Array1::from_vec(vec![-1.3e4, 0.0, 1.0e4, 2.7e4]);
    let pre =
        BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn()).unwrap();
    let bounds = LinearBounds::identity(4);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("wide-input softmax CROWN must not error");

    // Anti-overflow guard: no coefficient or bias may be inf/NaN.
    assert!(
        result.lower_a.iter().all(|v| v.is_finite()),
        "lower_a has non-finite coefficient"
    );
    assert!(
        result.upper_a.iter().all(|v| v.is_finite()),
        "upper_a has non-finite coefficient"
    );
    assert!(
        result.lower_b.iter().all(|v| v.is_finite()),
        "lower_b non-finite"
    );
    assert!(
        result.upper_b.iter().all(|v| v.is_finite()),
        "upper_b non-finite"
    );

    // Sound enclosure at box vertices + center (affine relaxation validity).
    let n = 4;
    let mut samples = vec![(&pre_lower + &pre_upper) / 2.0];
    for mask in 0..(1usize << n) {
        let mut s = pre_lower.clone();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                s[i] = pre_upper[i];
            }
        }
        samples.push(s);
    }
    for sample in &samples {
        let softmax = utils::softmax_1d(sample);
        for i in 0..n {
            let lb = result.lower_a.row(i).dot(sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(sample) + result.upper_b[i];
            assert!(
                lb.is_finite() && ub.is_finite(),
                "concretized bound non-finite at dim {i}"
            );
            assert!(
                lb <= softmax[i] + 1e-5,
                "wide lower bound {lb} > softmax {} at dim {i}",
                softmax[i]
            );
            assert!(
                ub + 1e-5 >= softmax[i],
                "wide upper bound {ub} < softmax {} at dim {i}",
                softmax[i]
            );
            // softmax in [0, 1]; sound bounds must respect that envelope.
            assert!(
                lb <= 1.0 + 1e-5 && ub >= -1e-5,
                "wide bounds escape [0,1] envelope at dim {i}: [{lb}, {ub}]"
            );
        }
    }
}

/// Companion to the overflow-fallback regression above: a WIDE but
/// finite-domain row (span 12, well inside the f64 exp domain) so
/// `softmax_lse_affine_bounds` returns `Some(..)` and the NON-TRIVIAL affine
/// relaxation actually runs. Asserts the concretized bounds (a) are finite,
/// (b) soundly enclose the true softmax at every box vertex + center, and
/// (c) are STRICTLY TIGHTER than the trivial `[0, 1]` fallback on at least one
/// dim — proving this test exercises the real relaxation, not the constant
/// fallback (which would make the enclosure assertions vacuous).
#[test]
fn crown_1d_sound_finite_domain_relaxation_is_tighter_than_trivial() {
    let layer = SoftmaxLayer::new(-1);
    let pre_lower = Array1::from_vec(vec![-6.0f32, -3.0, 0.0, 3.0]);
    let pre_upper = Array1::from_vec(vec![-3.0f32, 0.0, 3.0, 6.0]);
    let pre =
        BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn()).unwrap();
    let bounds = LinearBounds::identity(4);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("finite-domain softmax CROWN must not error");

    assert!(
        result
            .lower_a
            .iter()
            .chain(result.upper_a.iter())
            .all(|v| v.is_finite())
            && result
                .lower_b
                .iter()
                .chain(result.upper_b.iter())
                .all(|v| v.is_finite()),
        "finite-domain relaxation produced a non-finite coefficient/bias"
    );

    let n = 4;
    let mut samples = vec![(&pre_lower + &pre_upper) / 2.0];
    for mask in 0..(1usize << n) {
        let mut s = pre_lower.clone();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                s[i] = pre_upper[i];
            }
        }
        samples.push(s);
    }
    // Track the tightest concretized interval per dim over the box corners, to
    // prove strict improvement over the trivial [0,1] fallback.
    let mut min_lb = [f32::INFINITY; 4];
    let mut max_ub = [f32::NEG_INFINITY; 4];
    for sample in &samples {
        let softmax = utils::softmax_1d(sample);
        for i in 0..n {
            let lb = result.lower_a.row(i).dot(sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(sample) + result.upper_b[i];
            assert!(lb.is_finite() && ub.is_finite(), "non-finite at dim {i}");
            assert!(
                lb <= softmax[i] + 1e-5,
                "lower bound {lb} > softmax {} at dim {i}",
                softmax[i]
            );
            assert!(
                ub + 1e-5 >= softmax[i],
                "upper bound {ub} < softmax {} at dim {i}",
                softmax[i]
            );
        }
    }
    // Non-vacuity: the affine relaxation must beat the trivial [0,1] fallback
    // somewhere — at least one dim's guaranteed upper is < 1 or lower is > 0 by
    // a real margin. If this fails, the test degenerated to the constant
    // fallback and the enclosure asserts above would be meaningless.
    for sample in &samples {
        for i in 0..n {
            let lb = result.lower_a.row(i).dot(sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(sample) + result.upper_b[i];
            min_lb[i] = min_lb[i].min(lb);
            max_ub[i] = max_ub[i].max(ub);
        }
    }
    let strictly_tighter = (0..n).any(|i| max_ub[i] < 1.0 - 1e-3 || min_lb[i] > 1e-3);
    assert!(
        strictly_tighter,
        "finite-domain relaxation is no tighter than the trivial [0,1] fallback \
         (min_lb={min_lb:?}, max_ub={max_ub:?}) — enclosure assertions would be vacuous"
    );
}

// =========================================================================
// propagate_linear_with_bounds — 1D heuristic
// =========================================================================

#[test]
fn crown_1d_heuristic_bounds_lower_le_upper() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(4);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();

    // Concretize at center and check lb <= ub.
    // Heuristic mode uses sampling-based relaxation; ordering should hold to ~1e-3
    // even with sampling noise (softmax outputs are in [0, 1]).
    let center = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
    for i in 0..4 {
        let lb = result.lower_a.row(i).dot(&center) + result.lower_b[i];
        let ub = result.upper_a.row(i).dot(&center) + result.upper_b[i];
        assert!(
            lb <= ub + 1e-3,
            "heuristic center: lb[{}]={} > ub[{}]={} (gap={})",
            i,
            lb,
            i,
            ub,
            lb - ub
        );
    }
}

// =========================================================================
// propagate_linear_with_bounds — 2D
// =========================================================================

#[test]
fn crown_2d_rowwise_sound() {
    let layer = SoftmaxLayer::new(-1); // axis=-1 → row-wise
                                       // [2, 3] tensor → 2 rows of 3 elements each
    let pre_lower_vals = vec![-1.0f32, 0.0, 0.5, -0.5, 0.5, 1.0];
    let pre_upper_vals = vec![1.0f32, 2.0, 1.5, 0.5, 1.5, 2.0];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), pre_lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), pre_upper_vals.clone()).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(6); // 2*3 = 6 flattened

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Check vertex soundness: enumerate all 2^6 = 64 vertices of the input box,
    // compute true row-wise softmax, and verify CROWN bounds contain it.
    let n = 6;
    for mask in 0..(1usize << n) {
        let mut sample = Array1::<f32>::zeros(n);
        for i in 0..n {
            sample[i] = if (mask >> i) & 1 == 1 {
                pre_upper_vals[i]
            } else {
                pre_lower_vals[i]
            };
        }
        // Row-wise softmax: apply to each row of 3 independently
        let row0 = Array1::from_vec(vec![sample[0], sample[1], sample[2]]);
        let row1 = Array1::from_vec(vec![sample[3], sample[4], sample[5]]);
        let sm0 = utils::softmax_1d(&row0);
        let sm1 = utils::softmax_1d(&row1);
        let true_softmax = [sm0[0], sm0[1], sm0[2], sm1[0], sm1[1], sm1[2]];

        for (i, &ts) in true_softmax.iter().enumerate().take(n) {
            let lb = result.lower_a.row(i).dot(&sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(&sample) + result.upper_b[i];
            // Sound mode with 6-dim (2x3) input: f32 accumulation over 6 terms
            // yields ~1e-5 precision for softmax values in [0, 1].
            assert!(
                lb <= ts + 1e-5,
                "2D rowwise vertex lower violated: dim={}, mask={}, lb={}, softmax={}, diff={}",
                i,
                mask,
                lb,
                ts,
                lb - ts
            );
            assert!(
                ub >= ts - 1e-5,
                "2D rowwise vertex upper violated: dim={}, mask={}, ub={}, softmax={}, diff={}",
                i,
                mask,
                ub,
                ts,
                ts - ub
            );
        }
    }
}

#[test]
fn crown_2d_colwise_sound() {
    let layer = SoftmaxLayer::new(0); // axis=0 → column-wise
                                      // [3, 2] tensor → 2 columns of 3 elements, softmax along axis 0
    let pre_lower_vals = vec![-1.0f32, 0.0, 0.5, -0.5, 0.5, 1.0];
    let pre_upper_vals = vec![1.0f32, 2.0, 1.5, 0.5, 1.5, 2.0];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), pre_lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), pre_upper_vals.clone()).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(6);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Check vertex soundness: enumerate all 2^6 = 64 vertices.
    // Column-wise softmax: apply along axis 0, i.e., softmax of each column.
    // Layout [3,2] flattened row-major: [r0c0, r0c1, r1c0, r1c1, r2c0, r2c1]
    // Column 0: indices [0, 2, 4], Column 1: indices [1, 3, 5]
    let n = 6;
    for mask in 0..(1usize << n) {
        let mut sample = Array1::<f32>::zeros(n);
        for i in 0..n {
            sample[i] = if (mask >> i) & 1 == 1 {
                pre_upper_vals[i]
            } else {
                pre_lower_vals[i]
            };
        }
        // Column-wise softmax: softmax along axis 0 for each column
        let col0 = Array1::from_vec(vec![sample[0], sample[2], sample[4]]);
        let col1 = Array1::from_vec(vec![sample[1], sample[3], sample[5]]);
        let sm0 = utils::softmax_1d(&col0);
        let sm1 = utils::softmax_1d(&col1);
        // Map back to flattened order: [r0c0, r0c1, r1c0, r1c1, r2c0, r2c1]
        let true_softmax = [sm0[0], sm1[0], sm0[1], sm1[1], sm0[2], sm1[2]];

        for (i, &ts) in true_softmax.iter().enumerate().take(n) {
            let lb = result.lower_a.row(i).dot(&sample) + result.lower_b[i];
            let ub = result.upper_a.row(i).dot(&sample) + result.upper_b[i];
            // Sound mode with 6-dim (3x2) input: same precision as rowwise.
            assert!(
                lb <= ts + 1e-5,
                "2D colwise vertex lower violated: dim={}, mask={}, lb={}, softmax={}, diff={}",
                i,
                mask,
                lb,
                ts,
                lb - ts
            );
            assert!(
                ub >= ts - 1e-5,
                "2D colwise vertex upper violated: dim={}, mask={}, ub={}, softmax={}, diff={}",
                i,
                mask,
                ub,
                ts,
                ts - ub
            );
        }
    }
}

#[test]
fn objective_envelope_routes_through_explicit_axis_1_2d() {
    let layer = SoftmaxLayer::new(1);
    let (pre, bounds) = grouped_objective_envelope_fixture(&[2, 5]);
    let baseline = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
    })
    .expect("axis=1 baseline");
    let envelope = with_softmax_objective_envelope_for_test(true, || {
        layer.propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
    })
    .expect("axis=1 envelope");

    assert_grouped_envelope_dominates_and_is_sound(&baseline, &envelope, &pre);
}

#[test]
fn objective_envelope_routes_through_explicit_axis_2_nd() {
    let layer = SoftmaxLayer::new(2);
    let (pre, bounds) = grouped_objective_envelope_fixture(&[1, 2, 5]);
    let baseline = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
    })
    .expect("axis=2 baseline");
    let envelope = with_softmax_objective_envelope_for_test(true, || {
        layer.propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
    })
    .expect("axis=2 envelope");

    assert_grouped_envelope_dominates_and_is_sound(&baseline, &envelope, &pre);
}

#[test]
fn crown_2d_empty_rows_preserves_bias_and_stays_finite() {
    let layer = SoftmaxLayer::new(-1); // axis=-1 -> row-wise, num_groups=rows
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[0, 3]), vec![]).expect("shape [0,3]"),
        ArrayD::from_shape_vec(IxDyn(&[0, 3]), vec![]).expect("shape [0,3]"),
    )
    .expect("bounded tensor");
    let bounds = LinearBounds::new(
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![0.5, -1.25]),
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![1.5, 0.75]),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("propagate empty rows");

    assert_eq!(result.lower_a.shape(), &[2, 0]);
    assert_eq!(result.upper_a.shape(), &[2, 0]);
    assert_eq!(result.lower_b, bounds.lower_b);
    assert_eq!(result.upper_b, bounds.upper_b);
    assert!(result.lower_b.iter().all(|v| v.is_finite()));
    assert!(result.upper_b.iter().all(|v| v.is_finite()));
}

#[test]
fn crown_2d_empty_cols_preserves_bias_and_stays_finite() {
    let layer = SoftmaxLayer::new(0); // axis=0 -> column-wise, num_groups=cols
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 0]), vec![]).expect("shape [3,0]"),
        ArrayD::from_shape_vec(IxDyn(&[3, 0]), vec![]).expect("shape [3,0]"),
    )
    .expect("bounded tensor");
    let bounds = LinearBounds::new(
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![-0.75, 2.0]),
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![0.25, 3.0]),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("propagate empty cols");

    assert_eq!(result.lower_a.shape(), &[2, 0]);
    assert_eq!(result.upper_a.shape(), &[2, 0]);
    assert_eq!(result.lower_b, bounds.lower_b);
    assert_eq!(result.upper_b, bounds.upper_b);
    assert!(result.lower_b.iter().all(|v| v.is_finite()));
    assert!(result.upper_b.iter().all(|v| v.is_finite()));
}

#[test]
fn crown_nd_empty_dim_preserves_bias_and_stays_finite() {
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 0, 3]), vec![]).expect("shape [2,0,3]"),
        ArrayD::from_shape_vec(IxDyn(&[2, 0, 3]), vec![]).expect("shape [2,0,3]"),
    )
    .expect("bounded tensor");
    let bounds = LinearBounds::new(
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![-0.625, 1.125]),
        Array2::zeros((2, 0)),
        Array1::from_vec(vec![0.75, 2.25]),
    )
    .unwrap();

    for soundness in [
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Heuristic,
    ] {
        let layer = SoftmaxLayer::new(2); // axis=2 -> N-D path with num_groups=2*0=0
        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre, soundness)
            .expect("propagate N-D empty dim");

        assert_eq!(result.lower_a.shape(), &[2, 0]);
        assert_eq!(result.upper_a.shape(), &[2, 0]);
        assert_eq!(result.lower_b, bounds.lower_b);
        assert_eq!(result.upper_b, bounds.upper_b);
        assert!(result.lower_b.iter().all(|v| v.is_finite()));
        assert!(result.upper_b.iter().all(|v| v.is_finite()));
    }
}

// =========================================================================
// Error cases
// =========================================================================

#[test]
fn crown_0d_rejects() {
    let layer = SoftmaxLayer::new(-1);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[]), vec![0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::new(
        Array2::zeros((1, 1)),
        Array1::zeros(1),
        Array2::zeros((1, 1)),
        Array1::zeros(1),
    )
    .unwrap();
    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap_err();
    assert!(
        format!("{}", err).contains("at least 1D"),
        "expected 1D error, got: {}",
        err
    );
}

#[test]
fn crown_shape_mismatch_1d() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(5); // wrong: expects 5 but pre has 3

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap_err();
    assert!(
        matches!(err, ny_core::NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {:?}",
        err
    );
}

#[test]
fn crown_non_finite_heuristic_falls_back() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, f32::INFINITY, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Heuristic)
        .unwrap();

    // Non-finite falls back to constant bounds from [0, 1]
    assert!(
        result.lower_a.iter().all(|&v| v.abs() < 1e-6),
        "non-finite heuristic: lower_a should be zero (constant bounds)"
    );
}

#[test]
fn crown_non_finite_sound_falls_back() {
    let layer = SoftmaxLayer::new(-1);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, f32::INFINITY, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();

    // Non-finite sound falls back to IBP constant bounds
    assert!(
        result.lower_a.iter().all(|&v| v.abs() < 1e-6),
        "non-finite sound: lower_a should be zero"
    );
}

// =========================================================================
// apply_affine_bounds
// =========================================================================

#[test]
fn apply_affine_identity_bounds_pass_through() {
    let layer = SoftmaxLayer::new(-1);
    let n = 3;
    let lower_a = Array2::eye(n);
    let lower_b = Array1::zeros(n);
    let upper_a = Array2::eye(n);
    let upper_b = Array1::zeros(n);

    let bounds = LinearBounds::identity(n);
    let result = layer
        .apply_affine_bounds(&bounds, &lower_a, &lower_b, &upper_a, &upper_b)
        .unwrap();

    // Identity bounds through identity affine → should be identity
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (result.lower_a[[i, j]] - expected).abs() < 1e-6,
                "lower_a[{},{}]={} != {}",
                i,
                j,
                result.lower_a[[i, j]],
                expected
            );
            assert!(
                (result.upper_a[[i, j]] - expected).abs() < 1e-6,
                "upper_a[{},{}]={} != {}",
                i,
                j,
                result.upper_a[[i, j]],
                expected
            );
        }
    }
}

#[test]
fn apply_affine_negative_coefficients_swap_bounds() {
    let layer = SoftmaxLayer::new(-1);
    let n = 2;
    // Lower affine: 2*I, Upper affine: 3*I
    let lower_a = Array2::from_diag(&Array1::from_vec(vec![2.0, 2.0]));
    let lower_b = Array1::from_vec(vec![-0.1, -0.1]);
    let upper_a = Array2::from_diag(&Array1::from_vec(vec![3.0, 3.0]));
    let upper_b = Array1::from_vec(vec![0.1, 0.1]);

    // Input bounds with negative coefficients → should swap
    let bounds = LinearBounds::new(
        Array2::from_diag(&Array1::from_vec(vec![-1.0, -1.0])),
        Array1::zeros(n),
        Array2::from_diag(&Array1::from_vec(vec![-1.0, -1.0])),
        Array1::zeros(n),
    )
    .unwrap();

    let result = layer
        .apply_affine_bounds(&bounds, &lower_a, &lower_b, &upper_a, &upper_b)
        .unwrap();

    // la < 0: new_lower uses upper_a/upper_b
    // ua < 0: new_upper uses lower_a/lower_b
    assert!(
        (result.lower_a[[0, 0]] - (-3.0)).abs() < 1e-6,
        "negative la should use upper: {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.upper_a[[0, 0]] - (-2.0)).abs() < 1e-6,
        "negative ua should use lower: {}",
        result.upper_a[[0, 0]]
    );
}

// =========================================================================
// Batched CROWN
// =========================================================================

#[test]
fn batched_crown_heuristic_basic() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 0.5, -0.5, 0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 1.5, 0.5, 1.5, 2.0]).unwrap(),
    )
    .unwrap();

    // Batched bounds: [2, 3, 3] — 2 batch elements, 3 outputs, 3 inputs each
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_elem(IxDyn(&[2, 3, 3]), 0.0),
        ArrayD::zeros(IxDyn(&[2, 3])),
        ArrayD::from_elem(IxDyn(&[2, 3, 3]), 0.0),
        ArrayD::zeros(IxDyn(&[2, 3])),
        vec![3],
        vec![3],
    );
    // Set to identity for each batch
    let mut la = batched_bounds.lower_a.clone();
    let mut ua = batched_bounds.upper_a.clone();
    for b in 0..2 {
        for i in 0..3 {
            la[[b, i, i]] = 1.0;
            ua[[b, i, i]] = 1.0;
        }
    }
    let (_, lb, _, ub, isp, osp) = batched_bounds.into_parts();
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(la, lb, ua, ub, isp, osp);

    let result = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap();

    // Output shape should match
    assert_eq!(result.lower_a.shape(), &[2, 3, 3]);
    assert_eq!(result.lower_b.shape(), &[2, 3]);
}

#[test]
fn batched_crown_sound_basic() {
    let layer = SoftmaxLayer::new(-1);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 0.5, -0.5, 0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 1.5, 0.5, 1.5, 2.0]).unwrap(),
    )
    .unwrap();

    let mut la = ArrayD::zeros(IxDyn(&[2, 3, 3]));
    let mut ua = ArrayD::zeros(IxDyn(&[2, 3, 3]));
    for b in 0..2 {
        for i in 0..3 {
            la[[b, i, i]] = 1.0;
            ua[[b, i, i]] = 1.0;
        }
    }
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la,
        ArrayD::zeros(IxDyn(&[2, 3])),
        ua,
        ArrayD::zeros(IxDyn(&[2, 3])),
        vec![3],
        vec![3],
    );

    let result = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Sound,
        )
        .unwrap();

    assert_eq!(result.lower_a.shape(), &[2, 3, 3]);
    assert_eq!(result.lower_b.shape(), &[2, 3]);
    // All values should be finite
    assert!(
        result.lower_a.iter().all(|v| v.is_finite()),
        "batched sound lower_a has non-finite"
    );
    assert!(
        result.upper_a.iter().all(|v| v.is_finite()),
        "batched sound upper_a has non-finite"
    );
}

#[test]
fn objective_envelope_routes_through_batched_sound_path() {
    let layer = SoftmaxLayer::new(-1);
    let mut lower = Vec::with_capacity(10);
    let mut upper = Vec::with_capacity(10);
    for _ in 0..2 {
        lower.extend_from_slice(&OBJECTIVE_ENVELOPE_LOGIT_LOWER);
        upper.extend_from_slice(&OBJECTIVE_ENVELOPE_LOGIT_UPPER);
    }
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 5]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 5]), upper).unwrap(),
    )
    .unwrap();

    let mut a = ArrayD::zeros(IxDyn(&[2, 4, 5]));
    for batch in 0..2 {
        for row in 0..4 {
            for col in 0..5 {
                a[[batch, row, col]] = OBJECTIVE_ENVELOPE_ROWS[row][col];
            }
        }
    }
    let incoming = BatchedLinearBounds::from_parts_unchecked(
        a.clone(),
        ArrayD::zeros(IxDyn(&[2, 4])),
        a,
        ArrayD::zeros(IxDyn(&[2, 4])),
        vec![2, 5],
        vec![2, 4],
    );
    let baseline = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_batched_with_bounds(
            &incoming,
            &pre,
            VerificationSoundnessMode::Sound,
        )
    })
    .expect("batched baseline");
    let envelope = with_softmax_objective_envelope_for_test(true, || {
        layer.propagate_linear_batched_with_bounds(
            &incoming,
            &pre,
            VerificationSoundnessMode::Sound,
        )
    })
    .expect("batched envelope");
    let baseline_concrete = baseline
        .concretize_sound(&pre)
        .expect("baseline concretize");
    let envelope_concrete = envelope
        .concretize_sound(&pre)
        .expect("envelope concretize");

    let mut strict = false;
    for batch in 0..2 {
        for row in 0..4 {
            let baseline_lower = baseline_concrete.lower()[[batch, row]];
            let baseline_upper = baseline_concrete.upper()[[batch, row]];
            let envelope_lower = envelope_concrete.lower()[[batch, row]];
            let envelope_upper = envelope_concrete.upper()[[batch, row]];
            assert!(envelope_lower >= baseline_lower);
            assert!(envelope_upper <= baseline_upper);
            strict |= envelope_lower > baseline_lower || envelope_upper < baseline_upper;
        }

        for mask in 0..32usize {
            let logits = Array1::from_shape_fn(5, |idx| {
                if (mask >> idx) & 1 == 1 {
                    OBJECTIVE_ENVELOPE_LOGIT_UPPER[idx]
                } else {
                    OBJECTIVE_ENVELOPE_LOGIT_LOWER[idx]
                }
            });
            let softmax = utils::softmax_1d(&logits);
            for (row, objective) in OBJECTIVE_ENVELOPE_ROWS.iter().enumerate() {
                let true_value: f32 = objective
                    .iter()
                    .zip(softmax.iter())
                    .map(|(&coefficient, &value)| coefficient * value)
                    .sum();
                let lower_bound = envelope
                    .lower_a
                    .slice(ndarray::s![batch, row, ..])
                    .dot(&logits)
                    + envelope.lower_b[[batch, row]];
                let upper_bound = envelope
                    .upper_a
                    .slice(ndarray::s![batch, row, ..])
                    .dot(&logits)
                    + envelope.upper_b[[batch, row]];
                assert!(lower_bound <= true_value + 1e-3);
                assert!(upper_bound + 1e-3 >= true_value);
            }
        }
    }
    assert!(strict, "batched envelope made no strict improvement");
}

#[test]
fn batched_crown_shape_mismatch() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0; 6]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap(),
    )
    .unwrap();

    // Mismatch: softmax_size=4 but pre has 3
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(IxDyn(&[2, 4, 4])),
        ArrayD::zeros(IxDyn(&[2, 4])),
        ArrayD::zeros(IxDyn(&[2, 4, 4])),
        ArrayD::zeros(IxDyn(&[2, 4])),
        vec![4],
        vec![4],
    );

    let err = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap_err();
    assert!(
        matches!(err, ny_core::NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {:?}",
        err
    );
}

// =========================================================================
// apply_affine_bounds — mixed-sign coefficients
// =========================================================================

#[test]
fn apply_affine_mixed_sign_coefficients() {
    // Test that when bounds.lower_a has a mix of positive and negative entries
    // in the same row, each entry independently selects lower_a or upper_a.
    let layer = SoftmaxLayer::new(-1);
    let n = 2;
    // Affine bounds for softmax:
    //   lower: 2*x + [-0.1, -0.1]
    //   upper: 3*x + [0.1, 0.1]
    let lower_a = Array2::from_diag(&Array1::from_vec(vec![2.0, 2.0]));
    let lower_b = Array1::from_vec(vec![-0.1, -0.1]);
    let upper_a = Array2::from_diag(&Array1::from_vec(vec![3.0, 3.0]));
    let upper_b = Array1::from_vec(vec![0.1, 0.1]);

    // Input bounds with mixed signs: la[0,0]=1.0, la[0,1]=-0.5
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((n, n), vec![1.0, -0.5, 0.0, 1.0]).unwrap(),
        Array1::zeros(n),
        Array2::from_shape_vec((n, n), vec![1.0, -0.5, 0.0, 1.0]).unwrap(),
        Array1::zeros(n),
    )
    .unwrap();

    let result = layer
        .apply_affine_bounds(&bounds, &lower_a, &lower_b, &upper_a, &upper_b)
        .unwrap();

    // Row 0: la[0,0]=1.0 (positive) → uses lower_a[0,0]=2.0 → contributes 1.0*2.0=2.0 to [0,0]
    //         la[0,1]=-0.5 (negative) → uses upper_a[1,1]=3.0 → contributes -0.5*3.0=-1.5 to [0,1]
    assert!(
        (result.lower_a[[0, 0]] - 2.0).abs() < 1e-6,
        "mixed: lower_a[0,0]={}, expected 2.0 (pos coeff uses lower)",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_a[[0, 1]] - (-1.5)).abs() < 1e-6,
        "mixed: lower_a[0,1]={}, expected -1.5 (neg coeff uses upper)",
        result.lower_a[[0, 1]]
    );
    // Row 0 bias: la[0,0]=1.0 → +1.0*lower_b[0]=-0.1; la[0,1]=-0.5 → -0.5*upper_b[1]=+(-0.05)
    let expected_lb = 1.0 * (-0.1) + (-0.5) * 0.1;
    assert!(
        (result.lower_b[0] - expected_lb).abs() < 1e-6,
        "mixed: lower_b[0]={}, expected {}",
        result.lower_b[0],
        expected_lb
    );

    // Row 1: la[1,0]=0.0 (zero, skipped), la[1,1]=1.0 (positive) → uses lower_a[1,1]=2.0
    assert!(
        result.lower_a[[1, 0]].abs() < 1e-6,
        "mixed: lower_a[1,0]={}, expected 0 (zero coeff skipped)",
        result.lower_a[[1, 0]]
    );
    assert!(
        (result.lower_a[[1, 1]] - 2.0).abs() < 1e-6,
        "mixed: lower_a[1,1]={}, expected 2.0",
        result.lower_a[[1, 1]]
    );
}

// =========================================================================
// Batched CROWN — vertex soundness
// =========================================================================

#[test]
fn batched_crown_sound_vertex_soundness() {
    // Verify that batched sound CROWN bounds actually contain softmax
    // at all vertices of each batch element's input domain.
    let layer = SoftmaxLayer::new(-1);
    let n = 3;
    let pre_lower_vals = vec![-1.0, 0.0, 0.5, -0.5, 0.5, 1.0];
    let pre_upper_vals = vec![1.0, 2.0, 1.5, 0.5, 1.5, 2.0];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, n]), pre_lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, n]), pre_upper_vals.clone()).unwrap(),
    )
    .unwrap();

    // Identity batched bounds
    let mut la = ArrayD::zeros(IxDyn(&[2, n, n]));
    let mut ua = ArrayD::zeros(IxDyn(&[2, n, n]));
    for b in 0..2 {
        for i in 0..n {
            la[[b, i, i]] = 1.0;
            ua[[b, i, i]] = 1.0;
        }
    }
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la,
        ArrayD::zeros(IxDyn(&[2, n])),
        ua,
        ArrayD::zeros(IxDyn(&[2, n])),
        vec![n],
        vec![n],
    );

    let result = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Sound,
        )
        .unwrap();

    // Check vertex soundness for each batch
    for b in 0..2 {
        let batch_lower: Vec<f32> = (0..n).map(|i| pre_lower_vals[b * n + i]).collect();
        let batch_upper: Vec<f32> = (0..n).map(|i| pre_upper_vals[b * n + i]).collect();

        for mask in 0..(1usize << n) {
            let mut sample = Array1::<f32>::zeros(n);
            for i in 0..n {
                sample[i] = if (mask >> i) & 1 == 1 {
                    batch_upper[i]
                } else {
                    batch_lower[i]
                };
            }
            let softmax = utils::softmax_1d(&sample);

            for i in 0..n {
                // Extract the batch b's result coefficients
                let result_la: Array1<f32> = (0..n).map(|k| result.lower_a[[b, i, k]]).collect();
                let result_ua: Array1<f32> = (0..n).map(|k| result.upper_a[[b, i, k]]).collect();
                let lb = result_la.dot(&sample) + result.lower_b[[b, i]];
                let ub = result_ua.dot(&sample) + result.upper_b[[b, i]];
                assert!(
                    lb <= softmax[i] + 1e-3,
                    "batch {}, dim {}, mask {}: lower {} > softmax {}",
                    b,
                    i,
                    mask,
                    lb,
                    softmax[i]
                );
                assert!(
                    ub >= softmax[i] - 1e-3,
                    "batch {}, dim {}, mask {}: upper {} < softmax {}",
                    b,
                    i,
                    mask,
                    ub,
                    softmax[i]
                );
            }
        }
    }
}

// =========================================================================
// Axis error cases (resolve_axis_i32 integration)
// =========================================================================

#[test]
fn crown_axis_out_of_range_returns_error() {
    // axis=5 for 1D input should error (not silently use last axis)
    let layer = SoftmaxLayer {
        axis: 5,
        sound: true,
    };
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap_err();
    assert!(
        format!("{}", err).contains("axis"),
        "expected axis error, got: {}",
        err
    );
}

#[test]
fn crown_negative_axis_out_of_range_returns_error() {
    // axis=-3 for 1D input should error
    let layer = SoftmaxLayer {
        axis: -3,
        sound: true,
    };
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(3);

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap_err();
    assert!(
        format!("{}", err).contains("axis"),
        "expected axis error, got: {}",
        err
    );
}

// =========================================================================
// Batched CROWN — non-finite fallback
// =========================================================================

#[test]
fn batched_crown_non_finite_sound_falls_back() {
    let layer = SoftmaxLayer::new(-1);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![f32::NEG_INFINITY, 0.0, 1.0, -1.0, 0.0, 0.5],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, f32::INFINITY, 2.0, 1.0, 1.0, 1.5])
            .unwrap(),
    )
    .unwrap();

    let mut la = ArrayD::zeros(IxDyn(&[2, 3, 3]));
    let mut ua = ArrayD::zeros(IxDyn(&[2, 3, 3]));
    for b in 0..2 {
        for i in 0..3 {
            la[[b, i, i]] = 1.0;
            ua[[b, i, i]] = 1.0;
        }
    }
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la,
        ArrayD::zeros(IxDyn(&[2, 3])),
        ua,
        ArrayD::zeros(IxDyn(&[2, 3])),
        vec![3],
        vec![3],
    );

    // The non-finite values in the pre-activation should cause the sound
    // path to fall back to IBP constant bounds for the overall call.
    // It should not panic or produce NaN.
    let result = layer.propagate_linear_batched_with_bounds(
        &batched_bounds,
        &pre,
        VerificationSoundnessMode::Sound,
    );
    // The batched path delegates to 1d_sound per batch. Batch 0 has inf,
    // batch 1 is finite. The 1d_sound path checks for non-finite per batch
    // element. This may fail or succeed depending on implementation.
    // The key invariant: no panic, no NaN.
    match result {
        Ok(r) => {
            assert!(
                r.lower_a.iter().all(|v| !v.is_nan()),
                "batched non-finite: lower_a has NaN"
            );
            assert!(
                r.upper_a.iter().all(|v| !v.is_nan()),
                "batched non-finite: upper_a has NaN"
            );
        }
        Err(_) => {
            // Error is acceptable for non-finite input in batched path
        }
    }
}

// =========================================================================
// Cross-group f64 bias accumulation (#2489)
// =========================================================================

#[test]
fn crown_2d_many_groups_f64_bias_accumulation_2489() {
    // Regression test for #2489: verify that cross-group bias accumulation
    // uses f64 intermediates. With 8 row-wise groups, f32 accumulation can
    // lose precision from alternating-sign cancellation. The f64 path
    // preserves precision and applies directed rounding.
    let n = 3; // softmax dimension
    let num_groups = 8; // rows = independent softmax groups (kept small for test speed)
    let layer = SoftmaxLayer::new(-1); // row-wise

    // Create [8, 3] pre-activation bounds
    let mut lower_vals = Vec::with_capacity(num_groups * n);
    let mut upper_vals = Vec::with_capacity(num_groups * n);
    for g in 0..num_groups {
        // Vary bounds per group to create diverse bias contributions
        let offset = (g as f32) * 0.01;
        for j in 0..n {
            lower_vals.push(-1.0 + offset + (j as f32) * 0.1);
            upper_vals.push(1.0 + offset + (j as f32) * 0.1);
        }
    }

    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[num_groups, n]), lower_vals).expect("shape [8,3]"),
        ArrayD::from_shape_vec(IxDyn(&[num_groups, n]), upper_vals).expect("shape [8,3]"),
    )
    .expect("bounded tensor");
    let total = num_groups * n;
    let bounds = LinearBounds::identity(total);

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre, VerificationSoundnessMode::Sound)
        .expect("propagate 8 groups");

    // Key invariant: all biases must be finite (no NaN from accumulation overflow)
    assert!(
        result.lower_b.iter().all(|v| v.is_finite()),
        "cross-group lower_b has non-finite after 8 groups"
    );
    assert!(
        result.upper_b.iter().all(|v| v.is_finite()),
        "cross-group upper_b has non-finite after 8 groups"
    );
    // Soundness: lower_b <= upper_b element-wise
    for i in 0..total {
        assert!(
            result.lower_b[i] <= result.upper_b[i] + 1e-3,
            "cross-group bias inversion at dim {}: lower_b={} > upper_b={}",
            i,
            result.lower_b[i],
            result.upper_b[i],
        );
    }
}

// =========================================================================
// Flat-grouped softmax CROWN backward — helpers
// =========================================================================

/// Helper: verify that `BatchedLinearBounds` has finite values and lower_b <= upper_b.
fn assert_bounds_finite_and_ordered(result: &BatchedLinearBounds) {
    for (name, arr) in [
        ("lower_a", &result.lower_a),
        ("upper_a", &result.upper_a),
        ("lower_b", &result.lower_b),
        ("upper_b", &result.upper_b),
    ] {
        assert!(arr.iter().all(|v| v.is_finite()), "{} has non-finite", name);
    }
    for (lb, ub) in result.lower_b.iter().zip(result.upper_b.iter()) {
        assert!(*lb <= *ub + 1e-3, "lower_b={lb} > upper_b={ub}");
    }
}

/// Helper: verify flat-grouped softmax CROWN backward is sound at all vertices.
///
/// For each vertex of the input box, computes per-group softmax and verifies
/// `result.lower_a @ x + result.lower_b <= A_orig @ softmax_grouped(x) + b_orig`
/// and similarly for upper bounds.
#[allow(clippy::too_many_arguments)] // vertex-soundness checker needs all params
fn assert_flat_grouped_vertex_soundness(
    result: &BatchedLinearBounds,
    a_orig: &[f32],
    b_orig: &[f32],
    pre_lower: &[f32],
    pre_upper: &[f32],
    num_groups: usize,
    softmax_size: usize,
    out_dim: usize,
    tol: f32,
) {
    let total_in = num_groups * softmax_size;
    for mask in 0..(1u32 << total_in) {
        let x: Vec<f32> = (0..total_in)
            .map(|i| {
                if (mask >> i) & 1 == 1 {
                    pre_upper[i]
                } else {
                    pre_lower[i]
                }
            })
            .collect();
        // Per-group softmax
        let mut sm = vec![0.0f32; total_in];
        for g in 0..num_groups {
            let s = g * softmax_size;
            let group_sm = utils::softmax_1d(&Array1::from(x[s..s + softmax_size].to_vec()));
            sm[s..s + softmax_size].copy_from_slice(group_sm.as_slice().unwrap());
        }
        for i in 0..out_dim {
            let true_val: f32 = (0..total_in)
                .map(|j| a_orig[i * total_in + j] * sm[j])
                .sum::<f32>()
                + b_orig[i];
            let lb: f32 = (0..total_in)
                .map(|j| result.lower_a[[i, j]] * x[j])
                .sum::<f32>()
                + result.lower_b[[i]];
            let ub: f32 = (0..total_in)
                .map(|j| result.upper_a[[i, j]] * x[j])
                .sum::<f32>()
                + result.upper_b[[i]];
            assert!(
                lb <= true_val + tol,
                "lower violated: out={i}, mask={mask}: lb={lb}, true={true_val}",
            );
            assert!(
                ub >= true_val - tol,
                "upper violated: out={i}, mask={mask}: ub={ub}, true={true_val}",
            );
        }
    }
}

// =========================================================================
// Flat-grouped softmax CROWN backward — tests
// =========================================================================

#[test]
fn objective_envelope_routes_through_flat_grouped_sound_path() {
    let layer = SoftmaxLayer::new(-1);
    let (pre, linear) = grouped_objective_envelope_fixture(&[2, 5]);
    let a_values: Vec<f32> = linear.lower_a.iter().copied().collect();
    let incoming = BatchedLinearBounds::from_parts_unchecked(
        linear.lower_a.clone().into_dyn(),
        linear.lower_b.clone().into_dyn(),
        linear.upper_a.clone().into_dyn(),
        linear.upper_b.into_dyn(),
        vec![2, 5],
        vec![4],
    );
    let baseline = with_softmax_objective_envelope_for_test(false, || {
        layer.propagate_linear_batched_with_bounds(
            &incoming,
            &pre,
            VerificationSoundnessMode::Sound,
        )
    })
    .expect("flat-grouped baseline");
    let envelope = with_softmax_objective_envelope_for_test(true, || {
        layer.propagate_linear_batched_with_bounds(
            &incoming,
            &pre,
            VerificationSoundnessMode::Sound,
        )
    })
    .expect("flat-grouped envelope");
    let baseline_concrete = baseline
        .concretize_sound(&pre)
        .expect("baseline concretize");
    let envelope_concrete = envelope
        .concretize_sound(&pre)
        .expect("envelope concretize");

    let mut strict = false;
    for row in 0..4 {
        let baseline_lower = baseline_concrete.lower()[[row]];
        let baseline_upper = baseline_concrete.upper()[[row]];
        let envelope_lower = envelope_concrete.lower()[[row]];
        let envelope_upper = envelope_concrete.upper()[[row]];
        assert!(envelope_lower + 1e-6 >= baseline_lower);
        assert!(envelope_upper <= baseline_upper + 1e-6);
        strict |= envelope_lower > baseline_lower + 1e-5 || envelope_upper + 1e-5 < baseline_upper;
    }
    assert!(strict, "flat-grouped envelope made no strict improvement");

    assert_flat_grouped_vertex_soundness(
        &envelope,
        &a_values,
        &[0.0; 4],
        &OBJECTIVE_ENVELOPE_LOGIT_LOWER.repeat(2),
        &OBJECTIVE_ENVELOPE_LOGIT_UPPER.repeat(2),
        2,
        5,
        4,
        1e-3,
    );
}

/// Test 3: Flat-grouped heuristic basic — lower <= upper, finite, correct shape.
///
/// Design: designs/2026-03-03-flat-grouped-softmax-crown-testing.md Test 3.
/// Part of #3247.
#[test]
fn flat_grouped_heuristic_lower_le_upper() {
    let layer = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    let num_groups = 2usize;
    let softmax_size = 3usize;
    let out_dim = 2usize;
    let total_in = num_groups * softmax_size; // 6

    // Pre-activation: [2, 3]
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[num_groups, softmax_size]),
            vec![-1.0, 0.0, 0.5, -0.5, 0.5, 1.0],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[num_groups, softmax_size]),
            vec![1.0, 2.0, 1.5, 0.5, 1.5, 2.0],
        )
        .unwrap(),
    )
    .unwrap();

    // Block-diagonal A: [2, 6]
    // Row 0: [1, 0.5, -0.5, 0, 0, 0] operates on group 0
    // Row 1: [0, 0, 0, 0.5, 1, -0.5] operates on group 1
    let la = ArrayD::from_shape_vec(
        IxDyn(&[out_dim, total_in]),
        vec![1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, -0.5],
    )
    .unwrap();
    let lb = ArrayD::from_shape_vec(IxDyn(&[out_dim]), vec![0.1, -0.1]).unwrap();
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la.clone(),
        lb.clone(),
        la,
        lb,
        vec![total_in],
        vec![out_dim],
    );

    let result = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Heuristic,
        )
        .unwrap();

    assert_eq!(result.lower_a.shape(), &[out_dim, total_in]);
    assert_eq!(result.lower_b.shape(), &[out_dim]);
    assert_bounds_finite_and_ordered(&result);

    // Concretize at center: lower <= upper.
    // Heuristic mode: sampling noise means ordering tolerance ~1e-3 (not 1e-5).
    let center = [0.0f32, 1.0, 1.0, 0.0, 1.0, 1.5];
    for i in 0..out_dim {
        let lb_val: f32 = (0..total_in)
            .map(|j| result.lower_a[[i, j]] * center[j])
            .sum::<f32>()
            + result.lower_b[[i]];
        let ub_val: f32 = (0..total_in)
            .map(|j| result.upper_a[[i, j]] * center[j])
            .sum::<f32>()
            + result.upper_b[[i]];
        assert!(
            lb_val <= ub_val + 1e-3,
            "center: lb[{i}]={lb_val} > ub[{i}]={ub_val} (gap={})",
            lb_val - ub_val,
        );
    }
}

/// Test 4: Detection heuristic false-positive guard — dense (non-block-diagonal) A
/// that passes the heuristic still produces SOUND bounds.
///
/// The flat-grouped path decomposes `A @ softmax_all(x) = Σ_g A[:, g*s:(g+1)*s] @ softmax_g(x_g)`
/// which is valid because softmax is applied independently per group. The bounds may be
/// wider (less tight) than optimal for non-block-diagonal A, but are always sound.
///
/// Design: designs/2026-03-03-flat-grouped-softmax-crown-testing.md Test 4.
/// Part of #3247.
#[test]
fn flat_grouped_detection_false_positive_sound() {
    let layer = SoftmaxLayer::new(-1);
    let num_groups = 2usize;
    let softmax_size = 2usize;
    let out_dim = 2usize;
    let total_in = num_groups * softmax_size; // 4

    // Pre-activation: [2, 2]
    let pre_lower = vec![-1.0f32, 0.5, -0.5, 1.0];
    let pre_upper = vec![1.0f32, 1.5, 0.5, 2.0];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[num_groups, softmax_size]), pre_lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[num_groups, softmax_size]), pre_upper.clone()).unwrap(),
    )
    .unwrap();

    // Dense A: all entries nonzero (NOT block-diagonal)
    // This triggers flat-grouped because: a_shape=[2,4], pre_shape=[2,2],
    // a_in_dim=4 != pre_softmax_size=2, 4%2==0.
    let a_vals = vec![1.0f32, 0.5, -0.3, 0.7, 0.2, -0.4, 0.8, 1.1];
    let bias_vals = vec![0.1f32, -0.2];
    let la = ArrayD::from_shape_vec(IxDyn(&[out_dim, total_in]), a_vals.clone()).unwrap();
    let lb = ArrayD::from_shape_vec(IxDyn(&[out_dim]), bias_vals.clone()).unwrap();
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la.clone(),
        lb.clone(),
        la,
        lb,
        vec![total_in],
        vec![out_dim],
    );

    let result = layer
        .propagate_linear_batched_with_bounds(
            &batched_bounds,
            &pre,
            VerificationSoundnessMode::Sound,
        )
        .unwrap();

    // Verify soundness at all 16 vertices
    assert_flat_grouped_vertex_soundness(
        &result,
        &a_vals,
        &bias_vals,
        &pre_lower,
        &pre_upper,
        num_groups,
        softmax_size,
        out_dim,
        1e-3,
    );
}

/// Test 5: Round-trip via `flatten_to_block_diagonal` — the actual production code path.
///
/// BilinearCrown creates 3D batched bounds, flattens to block-diagonal 2D, then
/// softmax receives the flat bounds and triggers the flat-grouped path.
/// Verifies this end-to-end flow produces sound results at all vertices.
///
/// Design: designs/2026-03-03-flat-grouped-softmax-crown-testing.md Test 5.
/// Part of #3247.
#[test]
fn flat_grouped_round_trip_via_flatten() {
    let layer = SoftmaxLayer::new(-1);
    let num_groups = 2usize;
    let softmax_size = 2usize;
    let out_dim = 2usize;

    // Pre-activation: [2, 2]
    let pre_lower = vec![-1.0f32, 0.0, -0.5, 0.5];
    let pre_upper = vec![1.0f32, 2.0, 0.5, 1.5];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[num_groups, softmax_size]), pre_lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[num_groups, softmax_size]), pre_upper.clone()).unwrap(),
    )
    .unwrap();

    // Create 3D batched identity bounds: [2, 2, 2]
    let mut la = ArrayD::zeros(IxDyn(&[num_groups, out_dim, softmax_size]));
    let mut ua = ArrayD::zeros(IxDyn(&[num_groups, out_dim, softmax_size]));
    for b in 0..num_groups {
        for i in 0..out_dim.min(softmax_size) {
            la[[b, i, i]] = 1.0;
            ua[[b, i, i]] = 1.0;
        }
    }
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        la,
        ArrayD::zeros(IxDyn(&[num_groups, out_dim])),
        ua,
        ArrayD::zeros(IxDyn(&[num_groups, out_dim])),
        vec![softmax_size],
        vec![out_dim],
    );

    // Flatten to block-diagonal (production path)
    let flat_bounds = batched_bounds.flatten_to_block_diagonal().unwrap();

    let total_out = num_groups * out_dim; // 4
    let total_in = num_groups * softmax_size; // 4

    assert_eq!(flat_bounds.lower_a.shape(), &[total_out, total_in]);
    assert_eq!(flat_bounds.lower_b.shape(), &[total_out]);
    let a_orig: Vec<f32> = flat_bounds.lower_a.iter().copied().collect();
    let b_orig: Vec<f32> = flat_bounds.lower_b.iter().copied().collect();

    // Call softmax backward on the flat bounds → triggers flat-grouped path
    let result = layer
        .propagate_linear_batched_with_bounds(&flat_bounds, &pre, VerificationSoundnessMode::Sound)
        .unwrap();
    assert_eq!(result.lower_a.shape(), &[total_out, total_in]);
    assert_eq!(result.lower_b.shape(), &[total_out]);

    // Verify soundness at all 2^4 = 16 vertices
    assert_flat_grouped_vertex_soundness(
        &result,
        &a_orig,
        &b_orig,
        &pre_lower,
        &pre_upper,
        num_groups,
        softmax_size,
        total_out,
        1e-3,
    );
}
