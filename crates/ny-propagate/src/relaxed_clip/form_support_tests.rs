// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::array;
use ny_core::NyError;

use super::{
    relaxed_clip, relaxed_clip_internal_scalar, relaxed_clip_single_spec_row_fast,
    RelaxedClipOptions, SingleSpecRowClipScratch,
};

fn fixture() -> (
    ndarray::ArrayD<f32>,
    ndarray::ArrayD<f32>,
    ndarray::ArrayD<f32>,
    ndarray::ArrayD<f32>,
    ndarray::ArrayD<f32>,
) {
    (
        array![[0.0]].into_dyn(),
        array![[1.0]].into_dyn(),
        array![[[1.0]]].into_dyn(),
        array![[0.0]].into_dyn(),
        array![[0.5]].into_dyn(),
    )
}

#[test]
fn certified_lower_form_remains_available() {
    let (x_l, x_u, l_a, bias, thresholds) = fixture();
    let (new_l, new_u) =
        relaxed_clip(&x_l, &x_u, &l_a, &bias, &thresholds, 1, true).expect("certified lower form");
    assert!(new_l[[0, 0]] <= new_u[[0, 0]]);
    assert!(new_l[[0, 0]] >= x_l[[0, 0]]);
    assert!(new_u[[0, 0]] <= x_u[[0, 0]]);
}

#[test]
fn public_upper_form_fails_closed() {
    let (x_l, x_u, l_a, bias, thresholds) = fixture();
    let error = relaxed_clip(&x_l, &x_u, &l_a, &bias, &thresholds, 1, false)
        .expect_err("uncertified upper form must be refused");
    assert!(matches!(error, NyError::SoundnessRefusal(message) if message.contains("upper-form")));
}

#[test]
fn private_scalar_upper_form_cannot_bypass_the_guard() {
    let (x_l, x_u, l_a, bias, thresholds) = fixture();
    let error = relaxed_clip_internal_scalar(
        &x_l,
        &x_u,
        &l_a,
        &bias,
        &thresholds,
        RelaxedClipOptions {
            num_iterations: 1,
            is_lower: false,
            preserve_infeasible: false,
        },
    )
    .expect_err("scalar implementation must retain the form guard");
    assert!(matches!(error, NyError::SoundnessRefusal(_)));
}

#[test]
fn specialized_row_upper_form_cannot_bypass_the_guard() {
    let mut x_l = vec![0.0];
    let mut x_u = vec![1.0];
    let mut scratch = SingleSpecRowClipScratch::new();
    let error = relaxed_clip_single_spec_row_fast(
        &mut x_l,
        &mut x_u,
        &[1.0],
        &[0.0],
        &[0.5],
        1,
        1,
        1,
        false,
        &mut scratch,
    )
    .expect_err("specialized row implementation must retain the form guard");
    assert!(matches!(error, NyError::SoundnessRefusal(_)));
}
