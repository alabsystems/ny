// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness fuzzing for `compose_linear_mix()`.
//!
//! Verifies the end-to-end soundness property: for any concrete voice outputs
//! and gains within their bounded intervals, the true mixed output at each ear
//! falls within the composed bounds.
//!
//! Reference: Moore, R.E. (1966). *Interval Analysis*. Prentice-Hall.
//! 4-corner interval multiplication: [a,b]*[c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
//!
//! Part of #3517 composition soundness hardening.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_core::{MethodUsed, SoundnessProvenance};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use crate::composition::certificate::BoundCertificate;
use crate::composition::mixer::{compose_linear_mix, MixerSpec};

fn certificate(
    model_id: &str,
    output_bounds: BoundedTensor,
    actual_method: MethodUsed,
) -> BoundCertificate {
    BoundCertificate::try_new(
        model_id,
        output_bounds,
        actual_method,
        SoundnessProvenance::sound(),
    )
    .expect("supported method")
}

/// Strategy: generate a valid bounded interval [lower, upper] with lower <= upper.
fn bounded_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
    (-range..=range).prop_flat_map(move |a| {
        (-range..=range).prop_map(move |b| {
            let lo = a.min(b);
            let hi = a.max(b);
            (lo, hi)
        })
    })
}

/// Strategy: generate a pan coefficient in [-1.0, 1.0].
fn pan_coeff() -> impl Strategy<Value = f32> {
    -1.0f32..=1.0
}

/// Sample a concrete f32 value uniformly within [lower, upper].
/// For proptest: use the t parameter ∈ [0.0, 1.0] to interpolate.
fn sample_in_interval(lower: f32, upper: f32, t: f32) -> f32 {
    if lower == upper {
        return lower;
    }
    let result = lower + (upper - lower) * t;
    result.clamp(lower, upper)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Two-voice mixer soundness: for any concrete voice outputs and gains
    /// within their bounded intervals, the true mixed output at each ear
    /// must fall within the composed bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_two_voice_mixer(
        voice_a_bounds in bounded_interval(10.0),
        voice_b_bounds in bounded_interval(10.0),
        gain_a_bounds in bounded_interval(2.0),
        gain_b_bounds in bounded_interval(2.0),
        pan_a_left in pan_coeff(),
        pan_a_right in pan_coeff(),
        pan_b_left in pan_coeff(),
        pan_b_right in pan_coeff(),
        // Concrete sample points within each interval
        t_va in 0.0f32..=1.0,
        t_vb in 0.0f32..=1.0,
        t_ga in 0.0f32..=1.0,
        t_gb in 0.0f32..=1.0,
    ) {
        let (va_l, va_u) = voice_a_bounds;
        let (vb_l, vb_u) = voice_b_bounds;
        let (ga_l, ga_u) = gain_a_bounds;
        let (gb_l, gb_u) = gain_b_bounds;

        let cert_a = certificate(
            "voice_a",
            BoundedTensor::new_unchecked(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![va_l]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![va_u]).unwrap(),
            ).unwrap(),
            MethodUsed::Crown,
        );
        let cert_b = certificate(
            "voice_b",
            BoundedTensor::new_unchecked(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![vb_l]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![vb_u]).unwrap(),
            ).unwrap(),
            MethodUsed::Ibp,
        );

        let spec = MixerSpec {
            voice_gains: HashMap::from([
                ("voice_a".to_string(), BoundedTensor::new_unchecked(
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![ga_l]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![ga_u]).unwrap(),
                ).unwrap()),
                ("voice_b".to_string(), BoundedTensor::new_unchecked(
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![gb_l]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![gb_u]).unwrap(),
                ).unwrap()),
            ]),
            spatial_pan: HashMap::from([
                ("voice_a".to_string(), (pan_a_left, pan_a_right)),
                ("voice_b".to_string(), (pan_b_left, pan_b_right)),
            ]),
        };

        let (left, right) = compose_linear_mix(&[cert_a, cert_b], &spec).unwrap();

        // Sample concrete points within each interval
        let va_concrete = sample_in_interval(va_l, va_u, t_va);
        let vb_concrete = sample_in_interval(vb_l, vb_u, t_vb);
        let ga_concrete = sample_in_interval(ga_l, ga_u, t_ga);
        let gb_concrete = sample_in_interval(gb_l, gb_u, t_gb);

        // True mixed output at each ear (f64 for precision)
        let true_left = (pan_a_left as f64) * (ga_concrete as f64) * (va_concrete as f64)
            + (pan_b_left as f64) * (gb_concrete as f64) * (vb_concrete as f64);
        let true_right = (pan_a_right as f64) * (ga_concrete as f64) * (va_concrete as f64)
            + (pan_b_right as f64) * (gb_concrete as f64) * (vb_concrete as f64);

        let left_l = left.lower().as_slice().unwrap()[0] as f64;
        let left_u = left.upper().as_slice().unwrap()[0] as f64;
        let right_l = right.lower().as_slice().unwrap()[0] as f64;
        let right_u = right.upper().as_slice().unwrap()[0] as f64;

        prop_assert!(
            left_l <= true_left,
            "left lower bound unsound: composed_lower={left_l} > true={true_left}\n\
             voices=({va_concrete},{vb_concrete}), gains=({ga_concrete},{gb_concrete}), \
             pans_l=({pan_a_left},{pan_b_left})"
        );
        prop_assert!(
            left_u >= true_left,
            "left upper bound unsound: composed_upper={left_u} < true={true_left}\n\
             voices=({va_concrete},{vb_concrete}), gains=({ga_concrete},{gb_concrete}), \
             pans_l=({pan_a_left},{pan_b_left})"
        );
        prop_assert!(
            right_l <= true_right,
            "right lower bound unsound: composed_lower={right_l} > true={true_right}\n\
             voices=({va_concrete},{vb_concrete}), gains=({ga_concrete},{gb_concrete}), \
             pans_r=({pan_a_right},{pan_b_right})"
        );
        prop_assert!(
            right_u >= true_right,
            "right upper bound unsound: composed_upper={right_u} < true={true_right}\n\
             voices=({va_concrete},{vb_concrete}), gains=({ga_concrete},{gb_concrete}), \
             pans_r=({pan_a_right},{pan_b_right})"
        );
    }

    /// Five-voice mixer soundness: stress test with a realistic voice count.
    /// Verifies the same containment property as the two-voice test but with
    /// more voices to catch accumulation errors.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_five_voice_mixer(
        // Voice bounds (5 voices)
        v0_bounds in bounded_interval(5.0),
        v1_bounds in bounded_interval(5.0),
        v2_bounds in bounded_interval(5.0),
        v3_bounds in bounded_interval(5.0),
        v4_bounds in bounded_interval(5.0),
        // Gain bounds (all positive for realism — audio gains are non-negative)
        g0_bounds in (0.0f32..=2.0, 0.0f32..=2.0).prop_map(|(a, b)| (a.min(b), a.max(b))),
        g1_bounds in (0.0f32..=2.0, 0.0f32..=2.0).prop_map(|(a, b)| (a.min(b), a.max(b))),
        g2_bounds in (0.0f32..=2.0, 0.0f32..=2.0).prop_map(|(a, b)| (a.min(b), a.max(b))),
        g3_bounds in (0.0f32..=2.0, 0.0f32..=2.0).prop_map(|(a, b)| (a.min(b), a.max(b))),
        g4_bounds in (0.0f32..=2.0, 0.0f32..=2.0).prop_map(|(a, b)| (a.min(b), a.max(b))),
        // Pan coefficients (left, right) for each voice
        pan0 in (pan_coeff(), pan_coeff()),
        pan1 in (pan_coeff(), pan_coeff()),
        pan2 in (pan_coeff(), pan_coeff()),
        pan3 in (pan_coeff(), pan_coeff()),
        pan4 in (pan_coeff(), pan_coeff()),
        // Concrete sample points [0.0, 1.0] for interpolation
        t_v in prop::array::uniform5(0.0f32..=1.0),
        t_g in prop::array::uniform5(0.0f32..=1.0),
    ) {
        let voice_bounds = [v0_bounds, v1_bounds, v2_bounds, v3_bounds, v4_bounds];
        let gain_bounds = [g0_bounds, g1_bounds, g2_bounds, g3_bounds, g4_bounds];
        let pans = [pan0, pan1, pan2, pan3, pan4];
        let names: Vec<String> = (0..5).map(|i| format!("voice_{i}")).collect();

        let certificates: Vec<BoundCertificate> = names.iter()
            .zip(voice_bounds.iter())
            .map(|(name, &(lo, hi))| certificate(
                name,
                BoundedTensor::new_unchecked(
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
                ).unwrap(),
                MethodUsed::Crown,
            ))
            .collect();

        let voice_gains: HashMap<String, BoundedTensor> = names.iter()
            .zip(gain_bounds.iter())
            .map(|(name, &(lo, hi))| {
                (name.clone(), BoundedTensor::new_unchecked(
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
                ).unwrap())
            })
            .collect();

        let spatial_pan: HashMap<String, (f32, f32)> = names.iter()
            .zip(pans.iter())
            .map(|(name, &pan)| (name.clone(), pan))
            .collect();

        let spec = MixerSpec { voice_gains, spatial_pan };
        let (left, right) = compose_linear_mix(&certificates, &spec).unwrap();

        // Compute true mixed output with concrete sample points
        let mut true_left = 0.0_f64;
        let mut true_right = 0.0_f64;
        for i in 0..5 {
            let (vl, vu) = voice_bounds[i];
            let (gl, gu) = gain_bounds[i];
            let (pan_l, pan_r) = pans[i];
            let v_concrete = sample_in_interval(vl, vu, t_v[i]) as f64;
            let g_concrete = sample_in_interval(gl, gu, t_g[i]) as f64;
            true_left += (pan_l as f64) * g_concrete * v_concrete;
            true_right += (pan_r as f64) * g_concrete * v_concrete;
        }

        let left_l = left.lower().as_slice().unwrap()[0] as f64;
        let left_u = left.upper().as_slice().unwrap()[0] as f64;
        let right_l = right.lower().as_slice().unwrap()[0] as f64;
        let right_u = right.upper().as_slice().unwrap()[0] as f64;

        prop_assert!(
            left_l <= true_left,
            "5-voice left lower unsound: {left_l} > {true_left}"
        );
        prop_assert!(
            left_u >= true_left,
            "5-voice left upper unsound: {left_u} < {true_left}"
        );
        prop_assert!(
            right_l <= true_right,
            "5-voice right lower unsound: {right_l} > {true_right}"
        );
        prop_assert!(
            right_u >= true_right,
            "5-voice right upper unsound: {right_u} < {true_right}"
        );
    }

    /// Per-element gain mixer soundness: verify correctness when gains are
    /// per-sample vectors instead of scalars.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_per_element_gain(
        v0 in bounded_interval(5.0),
        v1 in bounded_interval(5.0),
        g0 in bounded_interval(2.0),
        g1 in bounded_interval(2.0),
        pan_left in pan_coeff(),
        pan_right in pan_coeff(),
        t_v0 in 0.0f32..=1.0,
        t_v1 in 0.0f32..=1.0,
        t_g0 in 0.0f32..=1.0,
        t_g1 in 0.0f32..=1.0,
    ) {
        let (v0_l, v0_u) = v0;
        let (v1_l, v1_u) = v1;
        let (g0_l, g0_u) = g0;
        let (g1_l, g1_u) = g1;

        let cert = certificate(
            "voice",
            BoundedTensor::new_unchecked(
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![v0_l, v1_l]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![v0_u, v1_u]).unwrap(),
            ).unwrap(),
            MethodUsed::Crown,
        );

        let spec = MixerSpec {
            voice_gains: HashMap::from([(
                "voice".to_string(),
                BoundedTensor::new_unchecked(
                    ArrayD::from_shape_vec(IxDyn(&[2]), vec![g0_l, g1_l]).unwrap(),
                    ArrayD::from_shape_vec(IxDyn(&[2]), vec![g0_u, g1_u]).unwrap(),
                ).unwrap(),
            )]),
            spatial_pan: HashMap::from([("voice".to_string(), (pan_left, pan_right))]),
        };

        let (left, right) = compose_linear_mix(&[cert], &spec).unwrap();

        // Verify each element independently
        let concrete_v = [
            sample_in_interval(v0_l, v0_u, t_v0),
            sample_in_interval(v1_l, v1_u, t_v1),
        ];
        let concrete_g = [
            sample_in_interval(g0_l, g0_u, t_g0),
            sample_in_interval(g1_l, g1_u, t_g1),
        ];

        for j in 0..2 {
            let true_left = (pan_left as f64) * (concrete_g[j] as f64) * (concrete_v[j] as f64);
            let true_right = (pan_right as f64) * (concrete_g[j] as f64) * (concrete_v[j] as f64);

            let ll = left.lower().as_slice().unwrap()[j] as f64;
            let lu = left.upper().as_slice().unwrap()[j] as f64;
            let rl = right.lower().as_slice().unwrap()[j] as f64;
            let ru = right.upper().as_slice().unwrap()[j] as f64;

            prop_assert!(
                ll <= true_left,
                "per-elem[{j}] left lower unsound: {ll} > {true_left}"
            );
            prop_assert!(
                lu >= true_left,
                "per-elem[{j}] left upper unsound: {lu} < {true_left}"
            );
            prop_assert!(
                rl <= true_right,
                "per-elem[{j}] right lower unsound: {rl} > {true_right}"
            );
            prop_assert!(
                ru >= true_right,
                "per-elem[{j}] right upper unsound: {ru} < {true_right}"
            );
        }
    }
}
