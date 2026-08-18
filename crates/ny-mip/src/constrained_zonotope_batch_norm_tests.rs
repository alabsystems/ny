// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{array, Array2};
use num_traits::Zero;
use std::time::{Duration, Instant};

use super::*;

fn limits() -> ConstrainedZonotopeBatchNormLimits {
    ConstrainedZonotopeBatchNormLimits {
        max_value_count: 10_000,
        max_rank: 8,
        max_channel_count: 1_000,
        max_alpha_dim: 1_000,
        max_generator_nonzeros: 100_000,
        max_parameter_elements: 4_000,
        max_coordinate_visits: 20_000,
        max_generator_visits: 200_000,
        max_interval_products: 1_000_000,
        max_constraint_count: 100_000,
        max_constraint_elements: 100_000,
    }
}

fn certificate_limits() -> ConstrainedZonotopeBatchNormAffineCertificateLimits {
    ConstrainedZonotopeBatchNormAffineCertificateLimits {
        max_rank: 8,
        max_channel_count: 1_000,
        max_parameter_elements: 6_000,
    }
}

fn rat(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite test value")
}

fn coefficient(domain: &ConstrainedZonotope64, generator: usize, value: usize) -> f64 {
    domain.generators()[generator]
        .entries()
        .find_map(|(index, coefficient)| (index == value).then_some(coefficient))
        .unwrap_or(0.0)
}

#[test]
fn actual_cgan_tail_surrogate_errors_reject_the_old_graph_underbounds() {
    // BatchNormalization_24 channels 88 and 29 from the authored cGAN model.
    // The graph's rounded certificates underbound the rigorous scale error on
    // channel 88 and the rigorous bias error on channel 29, respectively.
    let shape = [2];
    let gamma = [0.972_180_664_539_337_2, 1.032_762_765_884_399_4];
    let beta = [0.033_720_735_460_519_79, -0.016_135_465_353_727_34];
    let promoted_f32_subnormal_4 = f64::from(f32::from_bits(4));
    let mean = [0.038_430_672_138_929_37, promoted_f32_subnormal_4];
    let variance = [0.007_082_602_009_177_208, promoted_f32_subnormal_4];
    let nominal_scale = [1.082_151_293_754_577_6, 1.154_663_920_402_526_9];
    let nominal_bias = [-0.007_867_064_327_001_572, -0.016_135_465_353_727_34];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 0.800_000_011_920_929,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };

    let certificate = certify_batch_norm_affine_surrogate_unwired(
        spec,
        &nominal_scale,
        &nominal_bias,
        certificate_limits(),
    )
    .unwrap();
    assert_eq!(certificate.channels().len(), 2);
    assert!(certificate.conservative_live_bytes() > 4 * BATCH_NORM_RETAINED_RATIONAL_BYTES);

    let old_graph_scale_error = 7.591_057_227_251_952e-10;
    assert!(
        certificate.channels()[0].scale_error() > &rat(old_graph_scale_error),
        "the actual channel-88 graph scale_err is unsound and must be rejected"
    );
    let old_graph_bias_error = f64::from(f32::from_bits(1));
    assert!(
        certificate.channels()[1].bias_error() > &rat(old_graph_bias_error),
        "the actual channel-29 graph bias_err is unsound and must be rejected"
    );
    assert!(
        certificate.channels()[1].bias_error()
            > &(BigRational::from_integer(4.into()) * rat(f64::from(f32::from_bits(1)))),
        "the rigorous bias gap exceeds four promoted binary32 minimum subnormals"
    );
}

#[test]
fn declared_surrogate_certificate_is_exact_for_negative_scale_and_mixed_mean_signs() {
    let shape = [2];
    let gamma = [-1.0, 2.0];
    let beta = [0.5, -0.25];
    let mean = [-2.0, 3.0];
    let variance = [8.0, 8.0];
    let nominal_scale = [-0.25, 0.75];
    let nominal_bias = [-0.25, -2.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let certificate = certify_batch_norm_affine_surrogate_unwired(
        spec,
        &nominal_scale,
        &nominal_bias,
        certificate_limits(),
    )
    .unwrap();
    assert_eq!(certificate.sqrt_refinements(), 0);
    // Exact authored affines are (-1/3, -1/6) and (2/3, -9/4).
    assert_eq!(
        certificate.channels()[0].scale_error(),
        &BigRational::new(1.into(), 12.into())
    );
    assert_eq!(
        certificate.channels()[0].bias_error(),
        &BigRational::new(1.into(), 12.into())
    );
    assert_eq!(
        certificate.channels()[1].scale_error(),
        &BigRational::new(1.into(), 12.into())
    );
    assert_eq!(
        certificate.channels()[1].bias_error(),
        &BigRational::new(1.into(), 4.into())
    );
}

#[test]
fn declared_surrogate_certificate_firewall_is_exact_and_fail_closed() {
    let shape = [1];
    let gamma = [-1.0];
    let beta = [0.5];
    let mean = [-2.0];
    let variance = [8.0];
    let nominal_scale = [-0.25];
    let nominal_bias = [-0.25];
    let original = (gamma, beta, mean, variance, nominal_scale, nominal_bias);
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let start = Instant::now();
    let deadline = start + Duration::from_mins(1);
    let baseline = 4_096;
    let first = certify_batch_norm_affine_surrogate_unwired_with_clock(
        spec,
        &nominal_scale,
        &nominal_bias,
        certificate_limits(),
        ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
        |_| start,
    )
    .unwrap();
    let required = first.report().peak_live_bytes();
    assert!(required > baseline + first.value().conservative_live_bytes());
    let at_boundary = certify_batch_norm_affine_surrogate_unwired_with_clock(
        spec,
        &nominal_scale,
        &nominal_bias,
        certificate_limits(),
        ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
        |_| start,
    )
    .unwrap();
    assert_eq!(at_boundary.report().peak_live_bytes(), required);
    assert_eq!(at_boundary.value(), first.value());

    assert_eq!(
        certify_batch_norm_affine_surrogate_unwired_with_clock(
            spec,
            &nominal_scale,
            &nominal_bias,
            certificate_limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required - 1),
            |_| start,
        ),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required,
                limit: required - 1,
            }
        ))
    );
    let seam = "BatchNorm surrogate-certificate publication";
    assert!(matches!(
        certify_batch_norm_affine_surrogate_unwired_with_clock(
            spec,
            &nominal_scale,
            &nominal_bias,
            certificate_limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
            move |checkpoint| if checkpoint == seam { deadline } else { start },
        ),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
        )) if checkpoint == seam
    ));
    assert_eq!(
        (gamma, beta, mean, variance, nominal_scale, nominal_bias,),
        original,
        "declined certification must not mutate authored or declared data"
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_exact_square_enclosure(
    input: &ConstrainedZonotope64,
    output: &ConstrainedZonotope64,
    shape: &[usize],
    channel_axis: usize,
    gamma: &[f64],
    beta: &[f64],
    mean: &[f64],
    exact_denominator: &[f64],
) {
    let elements_per_channel = shape[channel_axis + 1..].iter().product::<usize>();
    for value_index in 0..input.value_dim() {
        let channel = (value_index / elements_per_channel) % shape[channel_axis];
        let scale = rat(gamma[channel]) / rat(exact_denominator[channel]);
        let bias = rat(beta[channel]) - rat(mean[channel]) * scale.clone();
        let exact_center = scale.clone() * rat(input.center()[value_index]) + bias;
        let mut required = (exact_center - rat(output.center()[value_index])).abs();
        required += scale.clone().abs() * rat(input.box_remainder()[value_index]);
        for generator in 0..input.alpha_dim() {
            let exact_coefficient = scale.clone() * rat(coefficient(input, generator, value_index));
            required +=
                (exact_coefficient - rat(coefficient(output, generator, value_index))).abs();
        }
        assert!(
            rat(output.box_remainder()[value_index]) >= required,
            "coordinate {value_index} under-enclosed its exact BatchNorm image"
        );
    }
}

#[test]
fn nchw_positive_negative_and_zero_scales_preserve_predicates() {
    let shape = [1, 3, 2, 2];
    let center: Vec<f64> = (0_i32..12).map(|value| f64::from(value) / 4.0).collect();
    let remainder: Vec<f64> = (0..12)
        .map(|index| if index % 2 == 0 { 0.125 } else { 0.0 })
        .collect();
    let input = ConstrainedZonotope64::try_new(
        center,
        vec![
            vec![(0, 0.25), (3, -0.5), (4, 0.75), (8, -0.125), (11, 0.5)],
            vec![(1, -0.25), (5, 0.5), (9, 1.0)],
        ],
        array![[1.0, -0.25], [-1.0, 0.5]],
        vec![0.75, 1.0],
        remainder,
    )
    .unwrap();
    let gamma = [2.0, -4.0, 0.0];
    let beta = [0.5, -0.25, 3.0];
    let mean = [0.25, -0.5, 7.0];
    let variance = [3.0, 3.0, 3.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 1,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let (output, plan) = constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();

    assert_eq!(output.constraints(), input.constraints());
    assert_eq!(output.rhs(), input.rhs());
    assert_eq!(output.alpha_dim(), input.alpha_dim());
    assert_eq!(output.generators().len(), input.generators().len());
    assert_eq!(plan.input_rank, 4);
    assert_eq!(plan.channel_axis, 1);
    assert_eq!(plan.outer_count, 1);
    assert_eq!(plan.channel_count, 3);
    assert_eq!(plan.elements_per_channel, 4);
    assert_eq!(plan.value_count, 12);
    assert_eq!(plan.parameter_elements, 12);
    assert_eq!(plan.coordinate_visits, 24);
    assert_eq!(plan.input_generator_nonzeros, 8);
    assert_eq!(plan.generator_visits, 16);
    assert_eq!(plan.output_generator_nonzeros, 5);

    // The third channel has exact zero gamma: it is the constant beta and has
    // no retained sparse entries at those coordinates.
    assert_eq!(&output.center()[8..12], &[3.0; 4]);
    for generator in output.generators() {
        assert!(generator.entries().all(|(index, _)| index < 8));
    }
    assert_exact_square_enclosure(
        &input,
        &output,
        &shape,
        1,
        &gamma,
        &beta,
        &mean,
        &[2.0, 2.0, 2.0],
    );
}

#[test]
fn flattened_channel_major_layout_repeats_each_channel_affine() {
    let shape = [2, 3];
    let input = ConstrainedZonotope64::try_new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![vec![(0, 0.5), (2, -0.25), (3, 1.0), (5, 0.125)]],
        Array2::zeros((0, 1)),
        Vec::new(),
        vec![0.0, 0.25, 0.0, 0.5, 0.0, 0.0],
    )
    .unwrap();
    let gamma = [6.0, -3.0];
    let beta = [1.0, 2.0];
    let mean = [0.5, -1.0];
    let variance = [8.0, 8.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let (output, plan) = constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();

    assert_eq!(plan.outer_count, 1);
    assert_eq!(plan.elements_per_channel, 3);
    // First block: scale 2, bias 0.  Second: scale -1, bias 1.
    assert_eq!(output.center(), &[2.0, 4.0, 6.0, -3.0, -4.0, -5.0]);
    assert_exact_square_enclosure(
        &input,
        &output,
        &shape,
        0,
        &gamma,
        &beta,
        &mean,
        &[3.0, 3.0],
    );
}

#[test]
fn coefficient_error_channels_are_load_bearing() {
    let shape = [2];
    let input = ConstrainedZonotope64::try_new(
        vec![1.0, 0.0],
        Vec::new(),
        Array2::zeros((0, 0)),
        Vec::new(),
        vec![0.0, 0.0],
    )
    .unwrap();
    let gamma = [1.0, 1.0];
    let beta = [0.0, 0.0];
    let mean = [0.0, 1.0];
    let variance = [8.0, 8.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let (affines, refinements) = certify_channel_affines(spec).unwrap();
    assert_eq!(refinements, 0);
    assert!(affines[0].scale_error > 0.0);
    assert_eq!(affines[0].bias_error, 0.0);
    assert!(affines[1].bias_error > 0.0);

    let (output, _) = constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();
    let true_scale = BigRational::new(1.into(), 3.into());
    let scale_only_required = (true_scale.clone() - rat(output.center()[0])).abs();
    let bias_only_required = (-true_scale - rat(output.center()[1])).abs();
    assert!(scale_only_required > BigRational::zero());
    assert!(bias_only_required > BigRational::zero());
    assert!(
        rat(output.box_remainder()[0]) >= scale_only_required,
        "scale_err * xmag was required for the unit input"
    );
    assert!(
        rat(output.box_remainder()[1]) >= bias_only_required,
        "bias_err was required for the zero input"
    );
}

#[test]
fn mixed_scale_and_subnormal_products_remain_enclosed() {
    let shape = [3];
    let min_subnormal = f64::from_bits(1);
    let huge = 2.0_f64.powi(500);
    let tiny = 2.0_f64.powi(-500);
    let input = ConstrainedZonotope64::try_new(
        vec![1.0, 1.0, tiny],
        vec![vec![(0, 1.0), (1, 2.0), (2, tiny)]],
        Array2::zeros((0, 1)),
        Vec::new(),
        vec![min_subnormal, 0.0, tiny],
    )
    .unwrap();
    let gamma = [f64::MIN_POSITIVE, f64::from_bits(3), huge];
    let beta = [0.0, min_subnormal, -0.5];
    let mean = [0.0, 0.0, tiny];
    let variance = [3.0, 3.0, 3.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let (affines, _) = certify_channel_affines(spec).unwrap();
    assert_eq!(affines[0].scale.to_bits(), 0x0008_0000_0000_0000);
    assert_eq!(affines[1].scale.to_bits(), 2);
    assert!(
        affines[1].scale_error >= min_subnormal,
        "the 3/2-subnormal tie must have a nonzero certified error"
    );

    let (output, _) = constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();
    assert!(output
        .center()
        .iter()
        .chain(output.box_remainder())
        .all(|value| value.is_finite()));
    assert_exact_square_enclosure(
        &input,
        &output,
        &shape,
        0,
        &gamma,
        &beta,
        &mean,
        &[2.0, 2.0, 2.0],
    );
}

fn mixed_dyadic(seed: u64, index: usize, denominator: f64) -> f64 {
    let mut value = seed
        ^ (u64::try_from(index)
            .unwrap()
            .wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let signed = i32::try_from(value % 33).unwrap() - 16;
    f64::from(signed) / denominator
}

#[test]
fn deterministic_dyadic_matrix_covers_batched_nchw_outer_repetition() {
    let shape = [2, 3, 2];
    let exact_denominator = [1.0, 3.0, 5.0];
    let variance = [0.0, 8.0, 24.0];

    for seed in 0_u64..64 {
        let center: Vec<f64> = (0..12)
            .map(|index| mixed_dyadic(seed, index, 8.0))
            .collect();
        let remainder: Vec<f64> = (0..12)
            .map(|index| mixed_dyadic(seed ^ 0xa5a5, index, 128.0).abs())
            .collect();
        let generators: Vec<Vec<(usize, f64)>> = (0..2)
            .map(|generator| {
                (0..12)
                    .filter_map(|index| {
                        let coefficient = mixed_dyadic(seed ^ 0x5a5a, generator * 12 + index, 16.0);
                        (coefficient != 0.0).then_some((index, coefficient))
                    })
                    .collect()
            })
            .collect();
        let input = ConstrainedZonotope64::try_new(
            center,
            generators,
            Array2::zeros((0, 2)),
            Vec::new(),
            remainder,
        )
        .unwrap();
        let gamma = [
            mixed_dyadic(seed, 40, 4.0),
            mixed_dyadic(seed, 41, 4.0),
            mixed_dyadic(seed, 42, 4.0),
        ];
        let beta = [
            mixed_dyadic(seed, 43, 8.0),
            mixed_dyadic(seed, 44, 8.0),
            mixed_dyadic(seed, 45, 8.0),
        ];
        let mean = [
            mixed_dyadic(seed, 46, 8.0),
            mixed_dyadic(seed, 47, 8.0),
            mixed_dyadic(seed, 48, 8.0),
        ];
        let spec = ConstrainedZonotopeBatchNormSpec {
            input_shape: &shape,
            channel_axis: 1,
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        };
        let (output, plan) =
            constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();
        assert_eq!(plan.outer_count, 2);
        assert_eq!(plan.elements_per_channel, 2);
        assert_exact_square_enclosure(
            &input,
            &output,
            &shape,
            1,
            &gamma,
            &beta,
            &mean,
            &exact_denominator,
        );
    }
}

#[test]
fn malformed_parameters_statistics_layout_and_semantics_fail_closed() {
    let input = ConstrainedZonotope64::try_new(
        vec![1.0],
        Vec::new(),
        Array2::zeros((0, 0)),
        Vec::new(),
        vec![0.0],
    )
    .unwrap();
    let shape = [1];
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let base = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };

    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                mode: ConstrainedZonotopeBatchNormMode::Training,
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::UnsupportedSemantics { .. })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &[],
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::InvalidSpec { .. })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                channel_axis: 1,
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::InvalidSpec { .. })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &[1, 0],
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::InvalidSpec { .. })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &[2],
                gamma: &[1.0, 1.0],
                beta: &[0.0, 0.0],
                mean: &[0.0, 0.0],
                variance: &[0.0, 0.0],
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::Shape {
            field: "input domain",
            ..
        })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec { gamma: &[], ..base },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::Shape { field: "gamma", .. })
    ));

    for (field, bad_spec) in [
        (
            "gamma",
            ConstrainedZonotopeBatchNormSpec {
                gamma: &[f64::NAN],
                ..base
            },
        ),
        (
            "beta",
            ConstrainedZonotopeBatchNormSpec {
                beta: &[f64::INFINITY],
                ..base
            },
        ),
        (
            "mean",
            ConstrainedZonotopeBatchNormSpec {
                mean: &[f64::NEG_INFINITY],
                ..base
            },
        ),
        (
            "variance",
            ConstrainedZonotopeBatchNormSpec {
                variance: &[f64::NAN],
                ..base
            },
        ),
    ] {
        assert!(matches!(
            constrained_zonotope_batch_norm_unwired(&input, bad_spec, limits()),
            Err(ConstrainedZonotopeBatchNormError::NonFinite {
                field: actual,
                index: 0,
            }) if actual == field
        ));
    }

    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            constrained_zonotope_batch_norm_unwired(
                &input,
                ConstrainedZonotopeBatchNormSpec { epsilon, ..base },
                limits(),
            ),
            Err(ConstrainedZonotopeBatchNormError::InvalidEpsilon)
        ));
    }
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                variance: &[-f64::MIN_POSITIVE],
                ..base
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::InvalidVariance { index: 0 })
    ));
}

#[test]
fn checked_shape_and_arithmetic_overflow_fail_closed() {
    let input = ConstrainedZonotope64::try_new(
        vec![1.0],
        Vec::new(),
        Array2::zeros((0, 0)),
        Vec::new(),
        vec![0.0],
    )
    .unwrap();
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let overflowing_shape = [usize::MAX, 2];
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &overflowing_shape,
                channel_axis: 0,
                gamma: &gamma,
                beta: &beta,
                mean: &mean,
                variance: &variance,
                epsilon: 1.0,
                mode: ConstrainedZonotopeBatchNormMode::Inference,
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "BatchNorm input value count"
        })
    ));

    let one_shape = [1];
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &one_shape,
                channel_axis: 0,
                gamma: &gamma,
                beta: &beta,
                mean: &mean,
                variance: &[f64::MAX],
                epsilon: f64::MAX,
                mode: ConstrainedZonotopeBatchNormMode::Inference,
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
            operation: "variance plus epsilon"
        })
    ));
    assert!(matches!(
        constrained_zonotope_batch_norm_unwired(
            &input,
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &one_shape,
                channel_axis: 0,
                gamma: &[f64::MAX],
                beta: &beta,
                mean: &mean,
                variance: &variance,
                epsilon: f64::from_bits(1),
                mode: ConstrainedZonotopeBatchNormMode::Inference,
            },
            limits(),
        ),
        Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
            operation: "BatchNorm nominal scale"
        })
    ));
}

fn cap_input() -> ConstrainedZonotope64 {
    ConstrainedZonotope64::try_new(
        vec![1.0, 2.0],
        vec![vec![(0, 0.25), (1, -0.5)]],
        array![[1.0], [-1.0]],
        vec![1.0, 1.0],
        vec![0.125, 0.0],
    )
    .unwrap()
}

fn assert_limit(
    result: Result<
        (ConstrainedZonotope64, ConstrainedZonotopeBatchNormPlan),
        ConstrainedZonotopeBatchNormError,
    >,
    resource: &'static str,
) {
    assert!(matches!(
        result,
        Err(ConstrainedZonotopeBatchNormError::ResourceLimit {
            resource: actual,
            ..
        }) if actual == resource
    ));
}

#[test]
fn every_declared_resource_cap_fails_closed() {
    let input = cap_input();
    let shape = [1, 2];
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &shape,
        channel_axis: 0,
        gamma: &gamma,
        beta: &beta,
        mean: &mean,
        variance: &variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };

    let mut capped = limits();
    capped.max_rank = 1;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "input rank",
    );

    let mut capped = limits();
    capped.max_value_count = 1;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "value count",
    );

    let mut capped = limits();
    capped.max_channel_count = 0;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "channel count",
    );

    let mut capped = limits();
    capped.max_alpha_dim = 0;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "alpha dimension",
    );

    let mut capped = limits();
    capped.max_generator_nonzeros = 1;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "input generator nonzeros",
    );

    let mut capped = limits();
    capped.max_parameter_elements = 3;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "parameter elements",
    );

    let mut capped = limits();
    capped.max_coordinate_visits = 3;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "coordinate visits",
    );

    let mut capped = limits();
    capped.max_generator_visits = 3;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "generator visits",
    );

    let mut capped = limits();
    capped.max_constraint_count = 1;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "constraint count",
    );

    let mut capped = limits();
    capped.max_constraint_elements = 1;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "constraint matrix elements",
    );

    let mut capped = limits();
    capped.max_interval_products = 4;
    assert_limit(
        constrained_zonotope_batch_norm_unwired(&input, spec, capped),
        "interval products",
    );
}

#[test]
fn nonstandard_constraint_layout_is_preserved_logically() {
    let constraints = array![[1.0, 2.0], [-3.0, 4.0]].reversed_axes();
    assert!(constraints.as_slice().is_none());
    let input = ConstrainedZonotope64::try_new(
        vec![1.0, -2.0],
        vec![vec![(0, 0.5)], vec![(1, -0.25)]],
        constraints,
        vec![5.0, 6.0],
        vec![0.0, 0.0],
    )
    .unwrap();
    let shape = [2];
    let gamma = [1.0, 1.0];
    let beta = [0.0, 0.0];
    let mean = [0.0, 0.0];
    let variance = [0.0, 0.0];
    let (output, _) = constrained_zonotope_batch_norm_unwired(
        &input,
        ConstrainedZonotopeBatchNormSpec {
            input_shape: &shape,
            channel_axis: 0,
            gamma: &gamma,
            beta: &beta,
            mean: &mean,
            variance: &variance,
            epsilon: 1.0,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        },
        limits(),
    )
    .unwrap();
    assert_eq!(output.constraints(), input.constraints());
    assert_eq!(output.rhs(), input.rhs());
}

fn budget_spec<'a>(
    shape: &'a [usize],
    gamma: &'a [f64],
    beta: &'a [f64],
    mean: &'a [f64],
    variance: &'a [f64],
) -> ConstrainedZonotopeBatchNormSpec<'a> {
    ConstrainedZonotopeBatchNormSpec {
        input_shape: shape,
        channel_axis: 0,
        gamma,
        beta,
        mean,
        variance,
        epsilon: 1.0,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    }
}

#[test]
fn budgeted_batch_norm_checks_peak_boundary_overflow_and_admission() {
    let input = cap_input();
    let shape = [1, 2];
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let spec = budget_spec(&shape, &gamma, &beta, &mean, &variance);
    let start = Instant::now();
    let deadline = start + Duration::from_mins(1);
    let baseline = 37;

    let first = constrained_zonotope_batch_norm_unwired_with_clock(
        &input,
        spec,
        limits(),
        ConstrainedZonotopeCallBudget::new(deadline, baseline, usize::MAX),
        |_| start,
    )
    .unwrap();
    let legacy = constrained_zonotope_batch_norm_unwired(&input, spec, limits()).unwrap();
    assert_eq!(first.value(), &legacy);
    let required = first.report().peak_live_bytes();
    assert!(required > baseline);

    let at_boundary = constrained_zonotope_batch_norm_unwired_with_clock(
        &input,
        spec,
        limits(),
        ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
        |_| start,
    )
    .unwrap();
    assert_eq!(at_boundary.report().peak_live_bytes(), required);

    assert_eq!(
        constrained_zonotope_batch_norm_unwired_with_clock(
            &input,
            spec,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required - 1),
            |_| start,
        ),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required,
                limit: required - 1,
            }
        ))
    );

    assert!(matches!(
        constrained_zonotope_batch_norm_unwired_with_clock(
            &input,
            spec,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, usize::MAX, usize::MAX),
            |_| start,
        ),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "aggregate peak-live bytes"
            }
        ))
    ));

    assert!(matches!(
        constrained_zonotope_batch_norm_unwired_with_clock(
            &input,
            spec,
            limits(),
            ConstrainedZonotopeCallBudget::new(start, 0, usize::MAX),
            |_| start,
        ),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "admission"
            }
        ))
    ));
}

#[test]
fn batch_norm_deadline_refuses_every_major_seam_and_never_publishes_partial_output() {
    let input = cap_input();
    let original = input.clone();
    let shape = [1, 2];
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let spec = budget_spec(&shape, &gamma, &beta, &mean, &variance);
    let seams = [
        "BatchNorm coefficient certification complete",
        "BatchNorm input-magnitude phase complete",
        "BatchNorm coordinate transform complete",
        "BatchNorm generator transform complete",
        "BatchNorm constraint clone complete",
        "BatchNorm domain materialization complete",
        "BatchNorm publication",
    ];

    for seam in seams {
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);
        let result = constrained_zonotope_batch_norm_unwired_with_clock(
            &input,
            spec,
            limits(),
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == seam {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == seam
        ));
        assert_eq!(
            input, original,
            "a declined call at {seam} mutated its input"
        );
    }
}

#[test]
fn batch_norm_deadline_polls_within_coordinate_generator_and_constraint_phases() {
    const ITEMS: usize = crate::CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL;
    let start = Instant::now();
    let deadline = start + Duration::from_mins(1);
    let channel_shape = [ITEMS];
    let channel_parameters = vec![1.0; ITEMS];
    let channel_zeros = vec![0.0; ITEMS];
    let channel_spec = budget_spec(
        &channel_shape,
        &channel_parameters,
        &channel_zeros,
        &channel_zeros,
        &channel_zeros,
    );
    let channel_input = ConstrainedZonotope64::try_new(
        vec![0.0; ITEMS],
        Vec::new(),
        Array2::zeros((0, 0)),
        Vec::new(),
        vec![0.0; ITEMS],
    )
    .unwrap();
    let channel_limits = ConstrainedZonotopeBatchNormLimits {
        max_value_count: ITEMS,
        max_rank: 1,
        max_channel_count: ITEMS,
        max_alpha_dim: 0,
        max_generator_nonzeros: 0,
        max_parameter_elements: ITEMS * 4,
        max_coordinate_visits: ITEMS * 2,
        max_generator_visits: 0,
        max_interval_products: ITEMS,
        max_constraint_count: 0,
        max_constraint_elements: 0,
    };
    let result = constrained_zonotope_batch_norm_unwired_with_clock(
        &channel_input,
        channel_spec,
        channel_limits,
        ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        move |checkpoint| {
            if checkpoint == "BatchNorm rational coefficient allocation" {
                deadline
            } else {
                start
            }
        },
    );
    assert!(matches!(
        result,
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "BatchNorm rational coefficient allocation"
            }
        ))
    ));

    let shape = [1, ITEMS];
    let gamma = [1.0];
    let beta = [0.0];
    let mean = [0.0];
    let variance = [0.0];
    let spec = budget_spec(&shape, &gamma, &beta, &mean, &variance);
    let input = ConstrainedZonotope64::try_new(
        vec![0.0; ITEMS],
        vec![(0..ITEMS).map(|index| (index, 1.0)).collect()],
        Array2::zeros((0, 1)),
        Vec::new(),
        vec![0.0; ITEMS],
    )
    .unwrap();
    let large_limits = ConstrainedZonotopeBatchNormLimits {
        max_value_count: ITEMS,
        max_rank: 2,
        max_channel_count: 1,
        max_alpha_dim: 1,
        max_generator_nonzeros: ITEMS,
        max_parameter_elements: 4,
        max_coordinate_visits: ITEMS * 2,
        max_generator_visits: ITEMS * 2,
        max_interval_products: ITEMS * 4,
        max_constraint_count: ITEMS,
        max_constraint_elements: ITEMS,
    };

    for phase in [
        "BatchNorm input-magnitude coordinates",
        "BatchNorm coordinate transform",
        "BatchNorm generator transform",
    ] {
        let result = constrained_zonotope_batch_norm_unwired_with_clock(
            &input,
            spec,
            large_limits,
            ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
            move |checkpoint| {
                if checkpoint == phase {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }
            )) if checkpoint == phase
        ));
    }

    let constraint_input = ConstrainedZonotope64::try_new(
        vec![0.0],
        vec![Vec::new(); 128],
        Array2::zeros((128, 128)),
        vec![0.0; 128],
        vec![0.0],
    )
    .unwrap();
    let constraint_shape = [1];
    let constraint_spec = budget_spec(&constraint_shape, &gamma, &beta, &mean, &variance);
    let mut constraint_limits = large_limits;
    constraint_limits.max_value_count = 1;
    constraint_limits.max_rank = 1;
    constraint_limits.max_alpha_dim = 128;
    constraint_limits.max_generator_nonzeros = 0;
    constraint_limits.max_coordinate_visits = 2;
    constraint_limits.max_generator_visits = 0;
    constraint_limits.max_interval_products = 4;
    constraint_limits.max_constraint_count = 128;
    constraint_limits.max_constraint_elements = ITEMS;
    let result = constrained_zonotope_batch_norm_unwired_with_clock(
        &constraint_input,
        constraint_spec,
        constraint_limits,
        ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        move |checkpoint| {
            if checkpoint == "BatchNorm constraint-matrix clone" {
                deadline
            } else {
                start
            }
        },
    );
    assert!(matches!(
        result,
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "BatchNorm constraint-matrix clone"
            }
        ))
    ));

    let rhs_input = ConstrainedZonotope64::try_new(
        vec![0.0],
        vec![Vec::new()],
        Array2::zeros((ITEMS, 1)),
        vec![0.0; ITEMS],
        vec![0.0],
    )
    .unwrap();
    let mut rhs_limits = constraint_limits;
    rhs_limits.max_alpha_dim = 1;
    rhs_limits.max_constraint_count = ITEMS;
    let rhs_clone = constrained_zonotope_batch_norm_unwired_with_clock(
        &rhs_input,
        constraint_spec,
        rhs_limits,
        ConstrainedZonotopeCallBudget::new(deadline, 0, usize::MAX),
        move |checkpoint| {
            if checkpoint == "BatchNorm right-hand-side clone" {
                deadline
            } else {
                start
            }
        },
    );
    assert!(matches!(
        rhs_clone,
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(
            ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "BatchNorm right-hand-side clone"
            }
        ))
    ));
}
