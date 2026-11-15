// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Historical test names containing "rebuild" are retained for the regression
// census. Their assertions now require preservation rather than reconstruction.
use super::*;
use crate::network::raw_root_cut::raw_root_cut_test_support::{strict_fixture, ExactToyBackend};
use crate::network::raw_root_cut::{evaluate_raw_root_cut, quote_raw_root_cut, RawRootCutLimits};
use ndarray::{arr1, Array1};
use num_rational::BigRational as Q;
use std::time::Duration;

fn exact(v: f32) -> Q {
    Q::from_float(f64::from(v)).unwrap()
}
fn box_of(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).unwrap()
}
fn unequal_alpha(pre: &BoundedTensor) -> GraphAlphaState {
    let mut alpha = GraphAlphaState::new();
    alpha.add_relu_node("relu", pre, false).unwrap();
    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(0.25);
    upper.fill(0.75);
    alpha
}
fn checked(layer: GpuCrownLayer, pre: &BoundedTensor) -> Option<GpuCrownLayer> {
    let mut poll = || true;
    let mut guard = Guard::new(Instant::now() + Duration::from_secs(30), &mut poll, false);
    let result = checked_relu_candidate(layer, pre, &mut guard)?;
    valid_relu(&result, pre, &mut guard)?;
    guard.check()?;
    Some(result)
}
fn ordinary_words(layer: &GpuCrownLayer) -> Vec<u32> {
    let GpuCrownLayer::Activation {
        lower_slope,
        upper_slope,
        lower_intercept,
        upper_intercept,
        ..
    } = layer
    else {
        panic!("expected ordinary relaxation")
    };
    [lower_slope, upper_slope, lower_intercept, upper_intercept]
        .into_iter()
        .flat_map(|v| v.iter().map(|x| x.to_bits()))
        .collect()
}

#[test]
fn unequal_alpha_rebuild_has_exact_piecewise_enclosure_and_reuses_buffers() {
    // Endpoints and zero are ALL vertices of each ReLU affine piece. Exact
    // rational inequalities there prove enclosure on the complete intervals.
    let pre = box_of(
        &[-3., 1., -4., -2f32.powi(-120)],
        &[2., 3., -1., 2f32.powi(-119)],
    );
    let alpha = unequal_alpha(&pre);
    let dual = bake_relu_layer("relu", &pre, Some(&alpha)).unwrap();
    let GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
        ..
    } = &dual
    else {
        panic!("fixture must exercise distinct alpha paths")
    };
    let original_words: Vec<_> = [
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
    ]
    .into_iter()
    .flat_map(|v| v.iter().map(|x| x.to_bits()))
    .collect();
    let pointers = [
        lower_pos_slope.as_ptr(),
        cross_slope.as_ptr(),
        upper_neg_slope.as_ptr(),
        cross_intercept.as_ptr(),
    ];
    let result = checked(dual, &pre).unwrap();
    let GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope: lower_slope,
        cross_slope: upper_slope,
        upper_neg_slope,
        cross_intercept: upper_intercept,
        ..
    } = &result
    else {
        panic!()
    };
    assert_eq!(
        pointers,
        [
            lower_slope.as_ptr(),
            upper_slope.as_ptr(),
            upper_neg_slope.as_ptr(),
            upper_intercept.as_ptr()
        ]
    );
    let final_words: Vec<_> = [lower_slope, upper_slope, upper_neg_slope, upper_intercept]
        .into_iter()
        .flat_map(|v| v.iter().map(|x| x.to_bits()))
        .collect();
    assert_eq!(
        original_words, final_words,
        "every original alpha/chord bit is retained"
    );
    for (i, (&lo, &hi)) in pre.lower().iter().zip(pre.upper().iter()).enumerate() {
        for x in [lo, hi, 0f32.max(lo).min(hi)] {
            let truth = exact(x.max(0.));
            let lower = exact(lower_slope[i]) * exact(x);
            let other_minorant = exact(upper_neg_slope[i]) * exact(x);
            let upper = exact(upper_slope[i]) * exact(x) + exact(upper_intercept[i]);
            assert!(lower <= truth && other_minorant <= truth && truth <= upper);
        }
    }
    // The retained two-path advice is not modified in its owner.
    let (lower, upper) = alpha.relu_alpha_pair("relu").unwrap();
    assert!(lower.iter().all(|&x| x == 0.25));
    assert!(upper.iter().all(|&x| x == 0.75));
}

#[test]
fn supported_ordinary_alpha_is_bit_preserved_and_bad_ordinary_alpha_still_refuses() {
    let pre = box_of(&[-3.], &[2.]);
    let mut alpha = unequal_alpha(&pre);
    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(0.25);
    upper.fill(0.25);
    let ordinary = bake_relu_layer("relu", &pre, Some(&alpha)).unwrap();
    let before = ordinary_words(&ordinary);
    let after = checked(ordinary, &pre).unwrap();
    assert_eq!(ordinary_words(&after), before);
    assert_eq!(before[0], 0.25f32.to_bits());

    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(1.25);
    upper.fill(1.25);
    assert!(checked(bake_relu_layer("relu", &pre, Some(&alpha)).unwrap(), &pre).is_none());
}

#[test]
fn dual_rebuild_rejects_wrong_shape_and_every_transient_cancellation() {
    let n = POLL_WORDS + 1;
    let pre = BoundedTensor::new(
        Array1::from_elem(n, -1.).into_dyn(),
        Array1::from_elem(n, 1.).into_dyn(),
    )
    .unwrap();
    let alpha = unequal_alpha(&pre);
    let mut malformed = bake_relu_layer("relu", &pre, Some(&alpha)).unwrap();
    let GpuCrownLayer::ActivationReluDualAlpha {
        upper_neg_slope, ..
    } = &mut malformed
    else {
        panic!()
    };
    upper_neg_slope.pop();
    assert!(checked(malformed, &pre).is_none());

    let run = |fail_at: Option<usize>| {
        let mut count = 0;
        let mut poll = || {
            count += 1;
            Some(count) != fail_at
        };
        let result = {
            let mut guard = Guard::new(Instant::now() + Duration::from_secs(30), &mut poll, false);
            checked_relu_candidate(
                bake_relu_layer("relu", &pre, Some(&alpha)).unwrap(),
                &pre,
                &mut guard,
            )
            .and_then(|layer| {
                valid_relu(&layer, &pre, &mut guard)?;
                guard.check()?;
                Some(layer)
            })
        };
        (result, count)
    };
    let (complete, calls) = run(None);
    assert!(complete.is_some() && calls >= 5);
    for at in 1..=calls {
        assert!(
            run(Some(at)).0.is_none(),
            "partial rebuild escaped at poll {at}"
        );
    }
}

#[test]
fn genuine_raw_root_call_with_dual_advice_proves_diamond_under_exact_work_caps() {
    let (graph, input, bounds) = strict_fixture();
    let alpha = unequal_alpha(bounds.get("pre").unwrap());
    assert!(matches!(
        bake_relu_layer("relu", bounds.get("pre").unwrap(), Some(&alpha)),
        Some(GpuCrownLayer::ActivationReluDualAlpha { .. })
    ));
    let end = Instant::now() + Duration::from_secs(30);
    let quote = |work| {
        quote_raw_root_cut(
            &graph,
            &input,
            &bounds,
            Some(&alpha),
            &[1.],
            (-0.75, 1.25),
            "relu",
            [0, 1],
            1.,
            RawRootCutLimits {
                max_nodes: 16,
                max_scratch_bytes: 16 << 20,
                max_work: work,
            },
            end,
            &mut || true,
        )
    };
    let generous = quote(100_000_000).unwrap();
    let two_work = generous.work();
    assert_eq!(generous.candidate_count(), 2);
    assert_eq!(quote(two_work).unwrap().candidate_count(), 2);
    let one = quote(two_work - 1).unwrap();
    assert_eq!(one.candidate_count(), 1);
    let one_work = one.work();
    assert!(quote(one_work - 1).is_none());
    let paid = quote(one_work).unwrap();
    assert_eq!(paid.candidate_count(), 1);
    let result = evaluate_raw_root_cut(paid, &ExactToyBackend::default(), &mut || true).unwrap();
    // Independent exact original graph: F=5/4-max(|u|,|v|), min=1/4.
    // This exercises fresh support, facet checking, full source replay and the
    // opaque completed-result constructor; no nominal scalar enters authority.
    assert_eq!(result.bounds(), (0.25, 1.25));
    assert_eq!((result.support_calls(), result.cut_calls()), (1, 1));
}

#[test]
fn dual_alpha_checks_both_minorants_and_exact_chord_without_repair() {
    let pre = box_of(&[-1.], &[1.]);
    let make = || GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope: vec![0.25],
        cross_slope: vec![0.5],
        upper_neg_slope: vec![0.75],
        cross_intercept: vec![0.5],
        num_neurons: 1,
    };
    assert!(checked(make(), &pre).is_some());
    for which in 0..2 {
        for bad in [-0.25, 1.25, f32::NAN, f32::INFINITY] {
            let mut layer = make();
            if let GpuCrownLayer::ActivationReluDualAlpha {
                lower_pos_slope,
                upper_neg_slope,
                ..
            } = &mut layer
            {
                if which == 0 {
                    lower_pos_slope[0] = bad;
                } else {
                    upper_neg_slope[0] = bad;
                }
            }
            assert!(checked(layer, &pre).is_none());
        }
    }
    let mut layer = make();
    if let GpuCrownLayer::ActivationReluDualAlpha {
        cross_intercept, ..
    } = &mut layer
    {
        cross_intercept[0] = ny_tensor::next_down_f32(0.5);
    }
    assert!(
        checked(layer, &pre).is_none(),
        "one-ULP invalid endpoint may not be repaired or tolerated"
    );
}
