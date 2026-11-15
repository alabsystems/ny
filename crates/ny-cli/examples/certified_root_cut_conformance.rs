// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Explicit real-CUDA conformance, never a skip on unsupported hardware.
//! This exercises public numerical capabilities on an authored exact model.
//! Private original-root provenance/one-shot admission is tested separately.
//! The 60-second per-call validation allowance is not a solver runtime claim.

use ny_core::{
    CertifiedWeightError, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed,
    GpuResnetSegment, NyError, ResidentLocalL1LowerCutCarrier, ResidentLowerCutCarrier,
    ResidentLowerCutChannel, ResidentLowerCutRow, ResidentTripletLowerCutCarrier,
    ResidentTwoLocalL1LowerCutCarrier, ResidentTwoPairLowerCutCarrier, Result,
};
use ny_cuda::CudaGemmEngine;
use ny_propagate::multineuron::{ExactRelu2FacetCertificate, ExactRelu2Support, Octahedron2};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DIM: usize = 4096;
const WIDTH: usize = 4096;
const CALL_SECONDS: u64 = 60;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(CALL_SECONDS)
}

fn seed(a: Vec<f32>, width: usize, rows: usize) -> GpuCrownSeed {
    assert_eq!(a.len(), width * rows);
    let a: Arc<[f32]> = a.into();
    let b: Arc<[f32]> = vec![0.0; rows].into();
    GpuCrownSeed {
        lower_a: Arc::clone(&a),
        upper_a: a,
        lower_b: Arc::clone(&b),
        upper_b: b,
        current_dim: width,
        num_specs: rows,
    }
}

struct Diamond {
    weight: Arc<[f32]>,
    first_bias: Arc<[f32]>,
    prefix: Vec<GpuResnetSegment>,
    full: Vec<GpuResnetSegment>,
    lower: Vec<f32>,
    upper: Vec<f32>,
}

impl Diamond {
    fn first(&self, bias: Arc<[f32]>) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::clone(&self.weight),
            bias: Some(bias),
            out_features: WIDTH,
            in_features: DIM,
            cert_err: CertifiedWeightError::EXACT,
        }
    }

    fn new() -> Self {
        assert_eq!(DIM % 2, 0);
        assert_eq!(WIDTH * DIM, 1 << 24);
        // u and v are independent means of the two halves of the ORIGINAL
        // 4096-dimensional input. Every input coordinate affects the output.
        // u,v range independently over [-1,1]; endpoints are attained by
        // setting an entire half-box to -1 or +1. All weights are exact f32.
        let mut w = vec![0.0; WIDTH * DIM];
        let one_over_dim = 1.0 / DIM as f32;
        for j in 0..DIM {
            w[j] = one_over_dim; // z0=(u+v)/2
            w[DIM + j] = if j < DIM / 2 {
                one_over_dim
            } else {
                -one_over_dim
            };
            if j < DIM / 2 {
                w[2 * DIM + j] = 2.0 * one_over_dim;
            }
        }
        let mut b = vec![0.0; WIDTH];
        b[2] = 2.0; // z2=u+2 is stable positive over the original box.
        let mut this = Self {
            weight: w.into(),
            first_bias: b.into(),
            prefix: Vec::new(),
            full: Vec::new(),
            lower: vec![-1.0; DIM],
            upper: vec![1.0; DIM],
        };
        this.prefix.push(GpuResnetSegment::Chain(vec![
            this.first(Arc::clone(&this.first_bias))
        ]));
        let mut lower_slope = vec![0.0; WIDTH];
        let mut upper_slope = vec![0.0; WIDTH];
        let mut upper_intercept = vec![0.0; WIDTH];
        lower_slope[2] = 1.0;
        upper_slope[0] = 0.5;
        upper_slope[1] = 0.5;
        upper_slope[2] = 1.0;
        upper_intercept[0] = 0.5;
        upper_intercept[1] = 0.5;
        let mut final_weight = vec![0.0; WIDTH];
        final_weight[..3].copy_from_slice(&[-2.0, -2.0, 1.0]);
        this.full.push(GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Linear {
                weight: final_weight.into(),
                bias: Some(Arc::from([-0.75])),
                out_features: 1,
                in_features: WIDTH,
                cert_err: CertifiedWeightError::EXACT,
            },
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept: vec![0.0; WIDTH],
                upper_intercept,
                num_neurons: WIDTH,
            },
            this.first(Arc::clone(&this.first_bias)),
        ]));
        this
    }
}

fn support_seed() -> GpuCrownSeed {
    // Four different, genuine objectives: z0, z1, z0+z1=u, z0-z1=v.
    let mut a = vec![0.0; 4 * WIDTH];
    a[0] = 1.0;
    a[WIDTH + 1] = 1.0;
    a[2 * WIDTH] = 1.0;
    a[2 * WIDTH + 1] = 1.0;
    a[3 * WIDTH] = 1.0;
    a[3 * WIDTH + 1] = -1.0;
    seed(a, WIDTH, 4)
}

fn check_encloses(result: &GpuCrownResult, exact: &[(f64, f64)]) {
    assert_eq!(result.lower_bounds.len(), exact.len());
    assert_eq!(result.upper_bounds.len(), exact.len());
    for (row, &(l, u)) in exact.iter().enumerate() {
        let lo = f64::from(result.lower_bounds[row]);
        let hi = f64::from(result.upper_bounds[row]);
        assert!(
            lo.is_finite() && hi.is_finite() && lo <= l && hi >= u,
            "row {row}: [{lo},{hi}] does not enclose independent exact [{l},{u}]"
        );
    }
}

fn certify(result: &GpuCrownResult) -> ExactRelu2FacetCertificate {
    check_encloses(result, &[(-1.0, 1.0); 4]);
    for (&l, &u) in result.lower_bounds.iter().zip(&result.upper_bounds) {
        assert!(
            l > -1.001 && u < 1.001,
            "vacuous checked support cannot pass"
        );
    }
    let l = &result.lower_bounds;
    let u = &result.upper_bounds;
    // The original source and these four actual objectives establish P.
    // No ideal/fabricated support replaces the GPU result.
    let p = Octahedron2::from_bounds(
        f64::from(l[0]),
        f64::from(u[0]),
        f64::from(l[1]),
        f64::from(u[1]),
        f64::from(l[2]),
        f64::from(u[2]),
        f64::from(l[3]),
        f64::from(u[3]),
    );
    let checker =
        ExactRelu2Support::new(&p).expect("actual support must form a finite nonempty domain");
    let certificate = checker
        .certify_normal_certificate([-1.0, -1.0, 2.0, 2.0])
        .expect("exact rational fixed-normal support certification");
    assert_eq!(certificate.support_domain(), &p);
    let facet = certificate.facet();
    assert_eq!(facet.a, [-1.0, -1.0, 2.0, 2.0]);
    assert!(facet.b >= 1.0 && facet.b < 1.001);
    certificate
}

fn carrier(
    certificate: &ExactRelu2FacetCertificate,
    bias_error: f32,
    width: usize,
    end: Instant,
) -> Result<ResidentLowerCutCarrier> {
    let f = certificate.facet();
    // Exactly lambda=1. These five f32 channel centers are EXACT products
    // of lambda and the stored certified facet; unary negation is exact.
    // The optional 1/8 source error deliberately widens only the bias channel.
    let row = ResidentLowerCutRow::try_new(
        vec![1.0],
        [
            ResidentLowerCutChannel::try_new(f.a[0], 0.0)?,
            ResidentLowerCutChannel::try_new(f.a[1], 0.0)?,
        ],
        [
            ResidentLowerCutChannel::try_new(f.a[2], 0.0)?,
            ResidentLowerCutChannel::try_new(f.a[3], 0.0)?,
        ],
        ResidentLowerCutChannel::try_new(-f.b, bias_error)?,
    )?;
    ResidentLowerCutCarrier::try_new(0, width, [0, 1], vec![row], end)
}

fn deadline_refusal<T>(result: Result<T>) {
    assert!(
        matches!(result, Err(NyError::DeadlineExceeded(_))),
        "expired public call must return typed DeadlineExceeded"
    );
}

// Same real-device gate and every original coordinate participates. This
// weighted case is separate from private original-root/query admission tests.
fn weighted_conformance(device: &CudaGemmEngine) -> Result<()> {
    let mut w = vec![0.; WIDTH * DIM];
    for j in 0..DIM {
        if j < DIM / 2 {
            w[j] = 2. / DIM as f32;
            w[DIM + j] = 2. / DIM as f32;
            w[2 * DIM + j] = 6. / DIM as f32;
        } else {
            w[DIM + j] = 0.5 / DIM as f32;
            w[2 * DIM + j] = 1. / DIM as f32;
        }
    }
    let mut b = vec![0.; WIDTH];
    b[..3].copy_from_slice(&[0.5, -0.25, 4.]);
    let first = GpuCrownLayer::Linear {
        weight: w.into(),
        bias: Some(b.into()),
        out_features: WIDTH,
        in_features: DIM,
        cert_err: CertifiedWeightError::EXACT,
    };
    let prefix = [GpuResnetSegment::Chain(vec![first.clone()])];
    let mut ls = vec![0.; WIDTH];
    ls[2] = 1.;
    let mut us = vec![0.; WIDTH];
    us[0] = 0.75;
    us[1] = ny_tensor::next_up_f32(0.4);
    us[2] = 1.;
    let mut ui = vec![0.; WIDTH];
    ui[0] = 0.375;
    ui[1] = ny_tensor::next_up_f32(1.5 * us[1]);
    // Independent endpoint checks for the STORED upper chord. ReLU is affine
    // on each side of zero, so these three exact dyadic checks enclose it all.
    for x in [-1.5f64, 0., 1.] {
        assert!(f64::from(us[1]) * x + f64::from(ui[1]) >= x.max(0.));
    }
    let mut final_weight = vec![0.; WIDTH];
    final_weight[..3].copy_from_slice(&[-2., -4., 1.]);
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: final_weight.into(),
            bias: Some(Arc::from([-0.4375])),
            out_features: 1,
            in_features: WIDTH,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: ls,
            upper_slope: us,
            lower_intercept: vec![0.; WIDTH],
            upper_intercept: ui,
            num_neurons: WIDTH,
        },
        first,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    // x,y are independent half-input means; z1=x+1/2,z2=x+y/4-1/4.
    // F=57/16-|z1|-2|z2| has exact range[1/16,49/16]. Max of the convex
    // absolute sum is7/2 at box vertices; min is1/2 since z1-z2>=1/2,
    // attained at x=0,y=1. The exact triangle floor is-11/80.
    check_encloses(&baseline, &[(0.0625, 3.0625), (-3.0625, -0.0625)]);
    assert!(baseline.lower_bounds[0] <= -0.1375 && baseline.lower_bounds[0] > -0.139);
    let support = device.crown_backward_gpu_resnet_support4_with_deadline(
        &prefix,
        &support_seed(),
        &lower,
        &upper,
        deadline(),
    )?;
    let expected = [(-0.5, 1.5), (-1.5, 1.), (-2., 2.5), (0.5, 1.)];
    check_encloses(&support, &expected);
    for (i, &(l, u)) in expected.iter().enumerate() {
        assert!(f64::from(support.lower_bounds[i]) > l - 0.001);
        assert!(f64::from(support.upper_bounds[i]) < u + 0.001);
    }
    let l = &support.lower_bounds;
    let u = &support.upper_bounds;
    let p = Octahedron2::from_bounds(
        f64::from(l[0]),
        f64::from(u[0]),
        f64::from(l[1]),
        f64::from(u[1]),
        f64::from(l[2]),
        f64::from(u[2]),
        f64::from(l[3]),
        f64::from(u[3]),
    );
    let checker = ExactRelu2Support::new(&p).expect("actual weighted support must be consistent");
    let uniform = checker
        .certify_normal_certificate([-1., -1., 2., 2.])
        .expect("fresh uniform support");
    let weighted = checker
        .certify_normal_certificate([-0.5, -1., 1., 2.])
        .expect("fresh Q8 weighted support");
    assert_eq!(weighted.support_domain(), &p);
    assert!(uniform.facet().b >= 2.5 && uniform.facet().b < 2.501);
    assert!(weighted.facet().b >= 1.75 && weighted.facet().b < 1.751);
    println!("case=weighted-actual-checked-support lower={:?} upper={:?} normal={:?} certified_intercept={}",
        support.lower_bounds,support.upper_bounds,weighted.facet().a,weighted.facet().b);
    let objective = seed(vec![1.], 1, 1);
    let make = |certificate: &ExactRelu2FacetCertificate,
                lambda: f32,
                error: f32,
                end: Instant|
     -> Result<ResidentLowerCutCarrier> {
        let f = certificate.facet();
        let product = |v: f32| -> Result<f32> {
            let result = lambda * v;
            assert_eq!(f64::from(result), f64::from(lambda) * f64::from(v));
            Ok(result)
        };
        let row = ResidentLowerCutRow::try_new(
            vec![lambda],
            [
                ResidentLowerCutChannel::try_new(product(f.a[0])?, 0.)?,
                ResidentLowerCutChannel::try_new(product(f.a[1])?, 0.)?,
            ],
            [
                ResidentLowerCutChannel::try_new(product(f.a[2])?, 0.)?,
                ResidentLowerCutChannel::try_new(product(f.a[3])?, 0.)?,
            ],
            ResidentLowerCutChannel::try_new(-product(f.b)?, error)?,
        )?;
        ResidentLowerCutCarrier::try_new(0, WIDTH, [0, 1], vec![row], end)
    };
    for lambda in [0.5, 1.] {
        let end = deadline();
        let cut = make(&uniform, lambda, 0., end)?;
        let got = device.crown_backward_gpu_resnet_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        assert!(
            got < -0.18,
            "uniform half/full must stay negative: {lambda} {got}"
        );
        println!("case=weighted-uniform-lower-cut lambda={lambda} lower={got}");
    }
    for error in [0., 0.015625] {
        let end = deadline();
        let cut = make(&weighted, 2., error, end)?;
        let got = device.crown_backward_gpu_resnet_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        let exact_augmented = 3.5625 - 2. * f64::from(weighted.facet().b) - f64::from(error);
        assert!(got.is_finite() && f64::from(got) <= exact_augmented && got <= 0.0625);
        assert!(f64::from(got) > exact_augmented - 0.001 && got > 0.04);
        assert!(got > baseline.lower_bounds[0] + 0.17);
        println!("case=weighted-actual-lower-cut lambda=2 carrier_bias_abs_error={error} lower={got} exact_augmented_lower={exact_augmented}");
    }
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let cut = make(&weighted, 2., 0., expired)?;
    deadline_refusal(device.crown_backward_gpu_resnet_lower_cut_with_deadline(
        &full, &objective, &lower, &upper, &cut, expired,
    ));
    println!("case=weighted-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=1 completed_public_cut_calls=4 expected_refusals=1 exact_graph_min=0.0625 benchmark=false private_root_consumer=false");
    Ok(())
}

// Four independent quarter-input means x,y,u,v. Every original input affects
// F=9/4-max(|x|,|y|)-max(|u|,|v|), with exact range [1/4,9/4].
// The public numerical transport does not establish private root provenance.
fn two_pair_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(device.provides_bounded_two_pair_lower_cut());
    assert_eq!(DIM % 4, 0);
    assert_eq!(WIDTH * DIM, 1 << 24);
    let mut w = vec![0.; WIDTH * DIM];
    for j in 0..DIM {
        let quarter = j / (DIM / 4);
        let k = 2. / DIM as f32;
        if quarter < 2 {
            w[j] = k;
            w[DIM + j] = if quarter == 0 { k } else { -k };
        } else {
            w[2 * DIM + j] = k;
            w[3 * DIM + j] = if quarter == 2 { k } else { -k };
        }
        if quarter == 0 || quarter == 2 {
            w[4 * DIM + j] = 2. * k;
        }
    }
    let mut b = vec![0.; WIDTH];
    b[4] = 4.; // stable z4=x+u+4 ranges over [2,6]
    let first = GpuCrownLayer::Linear {
        weight: w.into(),
        bias: Some(b.into()),
        out_features: WIDTH,
        in_features: DIM,
        cert_err: CertifiedWeightError::EXACT,
    };
    let prefix = [GpuResnetSegment::Chain(vec![first.clone()])];
    let mut ls = vec![0.; WIDTH];
    ls[4] = 1.;
    let mut us = vec![0.; WIDTH];
    us[..4].fill(0.5);
    us[4] = 1.;
    let mut ui = vec![0.; WIDTH];
    ui[..4].fill(0.5);
    let mut fw = vec![0.; WIDTH];
    fw[..5].copy_from_slice(&[-2., -2., -2., -2., 1.]);
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: fw.into(),
            bias: Some(Arc::from([-1.75])),
            out_features: 1,
            in_features: WIDTH,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: ls,
            upper_slope: us,
            lower_intercept: vec![0.; WIDTH],
            upper_intercept: ui,
            num_neurons: WIDTH,
        },
        first,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 2.25), (-2.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -1.75 && baseline.lower_bounds[0] > -1.751);
    println!("case=two-pair-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_triangle_min=-1.75",baseline.lower_bounds,baseline.upper_bounds);

    // Exactly four genuine relations: z0+z1=x,z0-z1=y,z2+z3=u,z2-z3=v.
    // Each marginal zi in [-1,1] follows independently from the ORIGINAL
    // dyadic input means. No extra scalar/padded support calls are hidden.
    let mut a = vec![0.; 4 * WIDTH];
    for n in 0..2 {
        a[2 * n * WIDTH + 2 * n] = 1.;
        a[2 * n * WIDTH + 2 * n + 1] = 1.;
        a[(2 * n + 1) * WIDTH + 2 * n] = 1.;
        a[(2 * n + 1) * WIDTH + 2 * n + 1] = -1.;
    }
    let relations = seed(a, WIDTH, 4);
    let support = device.crown_backward_gpu_resnet_support4_with_deadline(
        &prefix,
        &relations,
        &lower,
        &upper,
        deadline(),
    )?;
    check_encloses(&support, &[(-1., 1.); 4]);
    let certificate = |n: usize| {
        let p = Octahedron2::from_bounds(
            -1.,
            1.,
            -1.,
            1.,
            f64::from(support.lower_bounds[2 * n]),
            f64::from(support.upper_bounds[2 * n]),
            f64::from(support.lower_bounds[2 * n + 1]),
            f64::from(support.upper_bounds[2 * n + 1]),
        );
        let c = ExactRelu2Support::new(&p)
            .unwrap()
            .certify_normal_certificate([-1., -1., 2., 2.])
            .unwrap();
        assert_eq!(c.support_domain(), &p);
        assert!(c.facet().b >= 1. && c.facet().b < 1.001);
        c
    };
    let facets = [certificate(0), certificate(1)];
    println!("case=two-pair-actual-support lower={:?} upper={:?} marginal_origin=original_box facet_intercepts={:?}",
        support.lower_bounds,support.upper_bounds,[facets[0].facet().b,facets[1].facet().b]);
    let objective = seed(vec![1.], 1, 1);

    // All six current uniform single-pair choices at lambda=1/2 and 1.
    // Cross-diamond projections are the exact square [-1,1]^2 because their
    // input quarters are independent; their sum/difference range is [-2,2].
    // These analytic controls add no GPU support query and are not substituted
    // for either ACTUAL within-diamond relation used by the joint certificate.
    let square = Octahedron2::from_bounds(-1., 1., -1., 1., -2., 2., -2., 2.);
    let cross = ExactRelu2Support::new(&square)
        .unwrap()
        .certify_normal_certificate([-1., -1., 2., 2.])
        .unwrap();
    assert_eq!(cross.facet().b, 2.);
    let mut single_max = f32::NEG_INFINITY;
    for i in 0..4 {
        for j in i + 1..4 {
            let f = if i / 2 == j / 2 {
                facets[i / 2].facet()
            } else {
                cross.facet()
            };
            for lambda in [0.5f32, 1.] {
                let channel = |v: f32| {
                    let product = lambda * v;
                    assert_eq!(f64::from(product), f64::from(lambda) * f64::from(v));
                    ResidentLowerCutChannel::try_new(product, 0.)
                };
                let row = ResidentLowerCutRow::try_new(
                    vec![lambda],
                    [channel(f.a[0])?, channel(f.a[1])?],
                    [channel(f.a[2])?, channel(f.a[3])?],
                    channel(-f.b)?,
                )?;
                let end = deadline();
                let cut = ResidentLowerCutCarrier::try_new(0, WIDTH, [i, j], vec![row], end)?;
                let got = device.crown_backward_gpu_resnet_lower_cut_with_deadline(
                    &full, &objective, &lower, &upper, &cut, end,
                )?;
                let exact = -1.75 + f64::from(lambda) * (2. - f64::from(f.b));
                assert!(f64::from(got) <= exact && f64::from(got) > exact - 0.001 && got < 0.);
                single_max = single_max.max(got);
                println!(
                    "case=two-pair-single-control first={i} second={j} lambda={lambda} lower={got}"
                );
            }
        }
    }
    assert!(single_max <= -0.75);
    let make = |error: f32, end: Instant| -> Result<ResidentTwoPairLowerCutCarrier> {
        let bias = joint_bias_channel(
            &[(1., -facets[0].facet().b), (1., -facets[1].facet().b)],
            error,
        )?;
        ResidentTwoPairLowerCutCarrier::try_new(
            0,
            WIDTH,
            [[0, 1], [2, 3]],
            [1.; 2],
            [ResidentLowerCutChannel::try_new(-1., 0.)?; 4],
            [ResidentLowerCutChannel::try_new(2., 0.)?; 4],
            bias,
            end,
        )
    };
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut successful_joint_calls = 0;
    for error in [0., 0.125] {
        let end = deadline();
        let cut = make(error, end)?;
        let exact = rat(2.25) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
        let got = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        successful_joint_calls += 1;
        assert!(got.is_finite() && rat(got) <= exact && rat(got) > exact - rat(0.001));
        assert!(got > 0.12 && got <= 0.25 && got > single_max + 0.86);
        println!("case=two-pair-joint-cut carrier_extra_bias_error={error} carrier_bias={} carrier_bias_abs_error={} lower={got}",cut.bias().value(),cut.bias().source_abs_error());
    }
    // Unsafe original graph: shift F by -1/2, so its attained minimum is -1/4.
    let mut unsafe_full = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_full[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
        unreachable!()
    };
    *bias = Some(Arc::from([-2.25]));
    let end = deadline();
    let cut = make(0., end)?;
    let got = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &unsafe_full,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    successful_joint_calls += 1;
    let exact = rat(1.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(got.is_finite() && rat(got) <= exact && rat(got) > exact - rat(0.001) && got <= -0.25);
    println!("case=two-pair-unsafe-shift shift=-0.5 exact_graph_min=-0.25 lower={got}");

    let end = deadline();
    let cut = make(0., end)?;
    let zero = ResidentTwoPairLowerCutCarrier::try_new(
        0,
        WIDTH,
        [[0, 1], [2, 3]],
        [0.; 2],
        [ResidentLowerCutChannel::try_new(0., 0.)?; 4],
        [ResidentLowerCutChannel::try_new(0., 0.)?; 4],
        ResidentLowerCutChannel::try_new(0., 0.)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &zero, end
        ),
        Err(NyError::UnsupportedOp(_))
    ));
    let bad_error = ResidentTwoPairLowerCutCarrier::try_new(
        0,
        WIDTH,
        cut.ordered_pairs(),
        [1.; 2],
        *cut.pre(),
        *cut.post(),
        ResidentLowerCutChannel::try_new(-2., ny_core::CROWN_COEFF_MAX)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &bad_error, end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let cut = make(0., expired)?;
    deadline_refusal(
        device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, expired,
        ),
    );
    println!("case=two-pair-refusals zero_multiplier=true unsafe_error=true expired_joint=true");
    assert_eq!(successful_joint_calls, 3);
    println!("case=two-pair-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=1 completed_single_pair_calls=12 completed_joint_calls=3 expected_refusals=3 exact_graph_min=0.25 benchmark=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

// An independent directed interval reduction of ORIGINAL f32 products. Do not
// add separately rounded pair centers. Enforce IEEE before any arithmetic;
// exact rational checks below independently audit the resulting declaration.
fn joint_bias_channel(products: &[(f32, f32)], extra: f32) -> Result<ResidentLowerCutChannel> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(extra.is_finite() && extra >= 0.);
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for &(a, b) in products {
        assert!(a.is_finite() && b.is_finite());
        let p = f64::from(a) * f64::from(b); // exact binary32 product in binary64
        lo = (lo + p).next_down();
        hi = (hi + p).next_up();
    }
    let center = (0.5 * lo + 0.5 * hi) as f32;
    let radius = ((f64::from(center) - lo)
        .abs()
        .next_up()
        .max((hi - f64::from(center)).abs().next_up())
        + f64::from(extra))
    .next_up();
    let mut error = radius as f32;
    if f64::from(error) < radius {
        error = ny_tensor::next_up_f32(error);
    }
    let channel = ResidentLowerCutChannel::try_new(center, error)?;
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let exact = products
        .iter()
        .fold(rat(0.), |s, &(a, b)| s + rat(a) * rat(b));
    assert!(rat(center) - rat(error) <= exact.clone() - rat(extra));
    assert!(rat(center) + rat(error) >= exact + rat(extra));
    Ok(channel)
}

// The following cases preserve both optimized ReLU paths. They exercise only
// public numerical APIs; original graph/alpha snapshot authority is separate.
fn dual_alpha_asymmetric_support(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    let mut weight = vec![0.; WIDTH * DIM];
    for j in 0..DIM {
        weight[j] = 1. / DIM as f32;
        weight[DIM + j] = 1. / DIM as f32;
    }
    let mut bias = vec![0.; WIDTH];
    bias[1] = 2.;
    let first = GpuCrownLayer::Linear {
        weight: weight.into(),
        bias: Some(bias.into()),
        out_features: WIDTH,
        in_features: DIM,
        cert_err: CertifiedWeightError::EXACT,
    };
    let mut alpha_l = vec![0.; WIDTH];
    let mut alpha_u = vec![0.; WIDTH];
    let mut chord = vec![0.; WIDTH];
    let mut intercept = vec![0.; WIDTH];
    alpha_l[1] = 1.;
    alpha_u[..2].fill(1.);
    chord[..2].copy_from_slice(&[0.5, 1.]);
    intercept[0] = 0.5;
    // The four genuine objectives are w-R(w), R(w), w, and |w|. The first
    // upper path MUST use alpha_U=1: using alpha_L=0 instead returns upper1.
    let mut output = vec![0.; 4 * WIDTH];
    output[0] = -1.;
    output[1] = 1.;
    output[WIDTH] = 1.;
    output[2 * WIDTH + 1] = 1.;
    output[3 * WIDTH] = 2.;
    output[3 * WIDTH + 1] = -1.;
    let layers = vec![
        GpuCrownLayer::Linear {
            weight: output.into(),
            bias: Some(Arc::from([-2., 0., -2., 2.])),
            out_features: 4,
            in_features: WIDTH,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: alpha_l.clone(),
            cross_slope: chord.clone(),
            upper_neg_slope: alpha_u,
            cross_intercept: intercept.clone(),
            num_neurons: WIDTH,
        },
        first,
    ];
    let dual = [GpuResnetSegment::Chain(layers)];
    let mut ordinary = dual.clone();
    let GpuResnetSegment::Chain(layers) = &mut ordinary[0] else {
        unreachable!()
    };
    layers[1] = GpuCrownLayer::Activation {
        lower_slope: alpha_l,
        upper_slope: chord,
        lower_intercept: vec![0.; WIDTH],
        upper_intercept: intercept,
        num_neurons: WIDTH,
    };
    let identity = seed(
        vec![
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ],
        4,
        4,
    );
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let kept = device.crown_backward_gpu_resnet_support4_with_deadline(
        &dual,
        &identity,
        &lower,
        &upper,
        deadline(),
    )?;
    let symmetric = device.crown_backward_gpu_resnet_support4_with_deadline(
        &ordinary,
        &identity,
        &lower,
        &upper,
        deadline(),
    )?;
    let truth = [(-1., 0.), (0., 1.), (-1., 1.), (0., 1.)];
    check_encloses(&kept, &truth);
    check_encloses(&symmetric, &truth);
    assert!(kept.upper_bounds[0] >= 0. && kept.upper_bounds[0] < 0.001);
    assert!(symmetric.upper_bounds[0] >= 1. && symmetric.upper_bounds[0] < 1.001);
    assert!(symmetric.upper_bounds[0] > kept.upper_bounds[0] + 0.99);
    println!("case=dual-alpha-asymmetric-support lower={:?} upper={:?} symmetric_upper={:?} alpha_l=0 alpha_u=1 exact_first_upper=0", kept.lower_bounds, kept.upper_bounds, symmetric.upper_bounds);

    let mut malformed = dual.clone();
    let GpuResnetSegment::Chain(layers) = &mut malformed[0] else {
        unreachable!()
    };
    let GpuCrownLayer::ActivationReluDualAlpha {
        upper_neg_slope, ..
    } = &mut layers[1]
    else {
        unreachable!()
    };
    upper_neg_slope.pop();
    assert!(matches!(
        device.crown_backward_gpu_resnet_support4_with_deadline(
            &malformed,
            &identity,
            &lower,
            &upper,
            deadline(),
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let mut unsafe_source = dual.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_source[0] else {
        unreachable!()
    };
    let GpuCrownLayer::ActivationReluDualAlpha {
        upper_neg_slope, ..
    } = &mut layers[1]
    else {
        unreachable!()
    };
    upper_neg_slope[0] = f32::MAX;
    assert!(matches!(
        device.crown_backward_gpu_resnet_support4_with_deadline(
            &unsafe_source,
            &identity,
            &lower,
            &upper,
            deadline(),
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    deadline_refusal(device.crown_backward_gpu_resnet_support4_with_deadline(
        &dual, &identity, &lower, &upper, expired,
    ));
    Ok(())
}

fn dual_alpha_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    dual_alpha_asymmetric_support(device)?;
    assert_eq!(DIM, 4096);
    assert_eq!(WIDTH * DIM, 1 << 24);
    // Five independent dyadic means u,v,s,t,w. The first four use512 original
    // coordinates each; w uses the final2048. Every input affects this model.
    // z=[(u+v)/2,(u-v)/2,(s+t)/2,(s-t)/2,w,u+s-3w+6].
    let mut weight = vec![0.; WIDTH * DIM];
    for j in 0..DIM {
        match j / 512 {
            0 => {
                weight[j] = 1. / 1024.;
                weight[DIM + j] = 1. / 1024.;
                weight[5 * DIM + j] = 1. / 512.;
            }
            1 => {
                weight[j] = 1. / 1024.;
                weight[DIM + j] = -1. / 1024.;
            }
            2 => {
                weight[2 * DIM + j] = 1. / 1024.;
                weight[3 * DIM + j] = 1. / 1024.;
                weight[5 * DIM + j] = 1. / 512.;
            }
            3 => {
                weight[2 * DIM + j] = 1. / 1024.;
                weight[3 * DIM + j] = -1. / 1024.;
            }
            _ => {
                weight[4 * DIM + j] = 1. / 2048.;
                weight[5 * DIM + j] = -3. / 2048.;
            }
        }
    }
    let mut bias = vec![0.; WIDTH];
    bias[5] = 6.; // stable source range[1,11]
    let first = GpuCrownLayer::Linear {
        weight: weight.into(),
        bias: Some(bias.into()),
        out_features: WIDTH,
        in_features: DIM,
        cert_err: CertifiedWeightError::EXACT,
    };
    let prefix = [GpuResnetSegment::Chain(vec![first.clone()])];
    let mut alpha_l = vec![0.; WIDTH];
    let mut alpha_u = vec![0.; WIDTH];
    let mut chord = vec![0.; WIDTH];
    let mut intercept = vec![0.; WIDTH];
    alpha_l[..4].fill(0.5);
    alpha_u[..4].fill(0.5);
    alpha_l[4] = 1.;
    alpha_u[4] = 0.;
    alpha_l[5] = 1.;
    alpha_u[5] = 1.;
    chord[..5].fill(0.5);
    chord[5] = 1.;
    intercept[..5].fill(0.5);
    let mut output = vec![0.; WIDTH];
    output[..6].copy_from_slice(&[-2., -2., -2., -2., 3., 1.]);
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: output.into(),
            bias: Some(Arc::from([-3.75])),
            out_features: 1,
            in_features: WIDTH,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope: alpha_l,
            cross_slope: chord.clone(),
            upper_neg_slope: alpha_u,
            cross_intercept: intercept.clone(),
            num_neurons: WIDTH,
        },
        first,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    // F=9/4-max(|u|,|v|)-max(|s|,|t|)+3(R(w)-w).
    // Its exact range is[1/4,21/4], attained at box/breakpoint assignments.
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 5.25), (-5.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -1.75 && baseline.lower_bounds[0] > -1.751);
    println!(
        "case=dual-alpha-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_graph_max=5.25",
        baseline.lower_bounds, baseline.upper_bounds
    );
    let mut relations = vec![0.; 4 * WIDTH];
    for n in 0..2 {
        relations[2 * n * WIDTH + 2 * n] = 1.;
        relations[2 * n * WIDTH + 2 * n + 1] = 1.;
        relations[(2 * n + 1) * WIDTH + 2 * n] = 1.;
        relations[(2 * n + 1) * WIDTH + 2 * n + 1] = -1.;
    }
    let support = device.crown_backward_gpu_resnet_support4_with_deadline(
        &prefix,
        &seed(relations, WIDTH, 4),
        &lower,
        &upper,
        deadline(),
    )?;
    check_encloses(&support, &[(-1., 1.); 4]);
    let certificate = |n: usize| {
        let p = Octahedron2::from_bounds(
            -1.,
            1.,
            -1.,
            1.,
            f64::from(support.lower_bounds[2 * n]),
            f64::from(support.upper_bounds[2 * n]),
            f64::from(support.lower_bounds[2 * n + 1]),
            f64::from(support.upper_bounds[2 * n + 1]),
        );
        let c = ExactRelu2Support::new(&p)
            .unwrap()
            .certify_normal_certificate([-1., -1., 2., 2.])
            .unwrap();
        assert_eq!(c.support_domain(), &p);
        assert!(c.facet().b >= 1. && c.facet().b < 1.001);
        c
    };
    let certificates = [certificate(0), certificate(1)];
    println!("case=dual-alpha-actual-support lower={:?} upper={:?} facet_intercepts={:?} marginal_origin=original_box",support.lower_bounds,support.upper_bounds,[certificates[0].facet().b,certificates[1].facet().b]);
    let make = |end: Instant| -> Result<ResidentTwoPairLowerCutCarrier> {
        ResidentTwoPairLowerCutCarrier::try_new(
            0,
            WIDTH,
            [[0, 1], [2, 3]],
            [1.; 2],
            [ResidentLowerCutChannel::try_new(-1., 0.)?; 4],
            [ResidentLowerCutChannel::try_new(2., 0.)?; 4],
            joint_bias_channel(
                &[
                    (1., -certificates[0].facet().b),
                    (1., -certificates[1].facet().b),
                ],
                0.,
            )?,
            end,
        )
    };
    let objective = seed(vec![1.], 1, 1);
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut rebuilt = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut rebuilt[0] else {
        unreachable!()
    };
    let mut old_lower = vec![0.; WIDTH];
    old_lower[5] = 1.;
    layers[1] = GpuCrownLayer::Activation {
        lower_slope: old_lower,
        upper_slope: chord,
        lower_intercept: vec![0.; WIDTH],
        upper_intercept: intercept,
        num_neurons: WIDTH,
    };
    let end = deadline();
    let cut = make(end)?;
    let lost = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &rebuilt, &objective, &lower, &upper, &cut, end,
    )?;
    let rebuilt_exact = rat(-0.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(lost) <= rebuilt_exact && rat(lost) > rebuilt_exact - rat(0.001) && lost < -2.74);
    println!("case=dual-alpha-rebuilt-control lower={lost} alpha_l_w=0 exact_graph_min=0.25");
    let end = deadline();
    let cut = make(end)?;
    let kept = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &full, &objective, &lower, &upper, &cut, end,
    )?;
    let exact = rat(2.25) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(kept) <= exact && rat(kept) > exact - rat(0.001));
    assert!(kept > 0.24 && kept <= 0.25 && kept > lost + 2.99);
    println!("case=dual-alpha-joint-cut mode=exact lower={kept} alpha_l_w=1 alpha_u_w=0 carrier_bias={} carrier_bias_abs_error={} exact_graph_min=0.25",cut.bias().value(),cut.bias().source_abs_error());

    let mut source_error = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    let rho = 1. / 1024.;
    let delta = 1. / 64.;
    *cert_err = CertifiedWeightError {
        weight_rel_err: rho,
        bias_abs_err: delta,
    };
    // These are MODEL output-weight/bias errors, not carrier uncertainty.
    // For every declared output weight, worst independent error subtracts
    // rho*sum(|W_i|h_i)+delta. Here that exact family minimum is1/4-12rho-delta:
    // use u<=max(|u|,|v|),s<=max(|s|,|t|),R(w)-w>=0 andrho<1;
    // equality at u=s=1,v=t=w=0. Source target facets are unchanged.
    let exact_family_min = 0.25 - 12. * f64::from(rho) - f64::from(delta);
    assert_eq!(exact_family_min, 0.22265625);
    let end = deadline();
    let cut = make(end)?;
    let charged = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &source_error,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    assert!(charged.is_finite() && f64::from(charged) <= exact_family_min && charged > 0.18);
    assert!(charged < kept - 0.02);
    println!("case=dual-alpha-joint-cut mode=source-error lower={charged} model_weight_relative_error={rho} model_bias_abs_error={delta} exact_family_min={exact_family_min}");

    let mut unsafe_full = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_full[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
        unreachable!()
    };
    *bias = Some(Arc::from([-4.25]));
    let end = deadline();
    let cut = make(end)?;
    let negative = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
        &unsafe_full,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    let exact = rat(1.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(negative) <= exact && rat(negative) > exact - rat(0.001));
    assert!(negative <= -0.25 && negative > -0.251);
    println!("case=dual-alpha-unsafe-shift shift=-0.5 lower={negative} exact_graph_min=-0.25");
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    cert_err.weight_rel_err = -1.;
    let end = deadline();
    let cut = make(end)?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &source_error,
            &objective,
            &lower,
            &upper,
            &cut,
            end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let cut = make(expired)?;
    deadline_refusal(
        device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, expired,
        ),
    );
    println!("case=dual-alpha-refusals malformed_upper_alpha=true unsafe_upper_alpha=true invalid_model_error=true expired_support=true expired_joint=true");
    println!("case=dual-alpha-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=3 completed_joint_calls=4 expected_refusals=5 benchmark=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

// Independent original-source truth for the three-neuron cycle. All rational
// operations below use the original dyadic source coefficients. The identity
// sum|z|=max(a,b,c)-min(a,b,c) proves the whole-box range, not just the vertices.
fn triplet_cycle_truth() {
    use num_rational::BigRational;
    let rat = |x: f32| BigRational::from_float(f64::from(x)).unwrap();
    let mut minimum = rat(100.);
    let mut maximum = rat(-100.);
    let mut pair_max = [rat(0.), rat(0.), rat(0.)];
    for mask in 0..8 {
        let x: [BigRational; 3] =
            std::array::from_fn(|i| rat(if mask & (1 << i) == 0 { -1. } else { 1. }));
        let z = [
            (&x[0] - &x[1]) / rat(2.),
            (&x[1] - &x[2]) / rat(2.),
            (&x[2] - &x[0]) / rat(2.),
        ];
        assert_eq!(&z[0] + &z[1] + &z[2], rat(0.));
        let absolute: [BigRational; 3] = std::array::from_fn(|i| {
            if z[i] < rat(0.) {
                -z[i].clone()
            } else {
                z[i].clone()
            }
        });
        let span = x.iter().max().unwrap() - x.iter().min().unwrap();
        assert_eq!(&absolute[0] + &absolute[1] + &absolute[2], span);
        let value = rat(2.25) - span;
        minimum = minimum.min(value.clone());
        maximum = maximum.max(value);
        for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            pair_max[k] = pair_max[k].clone().max(&absolute[i] + &absolute[j]);
        }
    }
    assert_eq!(minimum, rat(0.25));
    assert_eq!(maximum, rat(2.25));
    assert_eq!(pair_max, [rat(2.), rat(2.), rat(2.)]);
    assert_eq!(minimum - rat(0.5), rat(-0.25));
}

fn triplet_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(device.provides_bounded_triplet_lower_cut());
    assert!(device.provides_bounded_support4());
    assert_eq!(DIM, 4096);
    assert_eq!(WIDTH * DIM, 1 << 24);
    triplet_cycle_truth();
    // Every original coordinate contributes through one independent mean:
    // a uses 2048 coordinates; b and c each use 1024. The three rows are
    // (a-b)/2,(b-c)/2,(c-a)/2. All source coefficients are exact binary32.
    let mut weight = vec![0.; WIDTH * DIM];
    for j in 0..DIM {
        if j < DIM / 2 {
            weight[j] = 1. / 4096.;
            weight[2 * DIM + j] = -1. / 4096.;
        } else if j < 3 * DIM / 4 {
            weight[j] = -1. / 2048.;
            weight[DIM + j] = 1. / 2048.;
        } else {
            weight[DIM + j] = -1. / 2048.;
            weight[2 * DIM + j] = 1. / 2048.;
        }
    }
    let first = GpuCrownLayer::Linear {
        weight: weight.into(),
        bias: Some(vec![0.; WIDTH].into()),
        out_features: WIDTH,
        in_features: DIM,
        cert_err: CertifiedWeightError::EXACT,
    };
    let prefix = [GpuResnetSegment::Chain(vec![first.clone()])];
    let mut lower_alpha = vec![0.; WIDTH];
    lower_alpha[..3].fill(0.5);
    let mut chord = vec![0.; WIDTH];
    chord[..3].fill(0.5);
    let mut intercept = vec![0.; WIDTH];
    intercept[..3].fill(0.5);
    let mut final_weight = vec![0.; WIDTH];
    final_weight[..3].fill(-2.);
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: final_weight.into(),
            bias: Some(Arc::from([2.25])),
            out_features: 1,
            in_features: WIDTH,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: lower_alpha,
            upper_slope: chord,
            lower_intercept: vec![0.; WIDTH],
            upper_intercept: intercept,
            num_neurons: WIDTH,
        },
        first,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 2.25), (-2.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -0.75 && baseline.lower_bounds[0] > -0.751);
    println!(
        "case=triplet-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_triangle_min=-0.75",
        baseline.lower_bounds, baseline.upper_bounds
    );

    // Four genuine signed objectives plus their opposite endpoints cover all
    // eight signs. The +++ relation is identically zero in this source; it is
    // an actual relation row, not a padded objective or a size-gate bypass.
    let signs = [[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]];
    let mut a = vec![0.; 4 * WIDTH];
    for row in 0..4 {
        a[row * WIDTH..row * WIDTH + 3].copy_from_slice(&signs[row]);
    }
    let relations = seed(a, WIDTH, 4);
    let support = device.crown_backward_gpu_resnet_support4_with_deadline(
        &prefix,
        &relations,
        &lower,
        &upper,
        deadline(),
    )?;
    check_encloses(&support, &[(0., 0.), (-2., 2.), (-2., 2.), (-2., 2.)]);
    let mut bound = 0.0f32;
    for row in 0..4 {
        let l = support.lower_bounds[row];
        let u = support.upper_bounds[row];
        assert!(ny_core::is_crown_coeff_safe(l) && ny_core::is_crown_coeff_safe(u) && l <= u);
        let exact = if row == 0 { 0. } else { 2. };
        assert!(l > -exact - 0.001 && u < exact + 0.001);
        bound = bound.max(u).max(-l);
    }
    assert!((2. ..2.001).contains(&bound));
    // The sign identity and actual outward endpoints imply sum|z|<=bound.
    // No ideal B=2 replaces these GPU results in the new triplet carrier.
    println!("case=triplet-actual-support lower={:?} upper={:?} l1_bound={bound} weights=[1.0,1.0,1.0] ordered_neurons=[0,1,2]", support.lower_bounds, support.upper_bounds);
    let objective = seed(vec![1.], 1, 1);

    // Independent exact original-cycle pair projection: |zi|<=1, |zj|<=1,
    // |zi+zj|<=1 and |zi-zj|<=2. Each pair attains (+1,-1), so b=2.
    // These analytic comparison certificates require no additional GPU sweep.
    let pair_domain = Octahedron2::from_bounds(-1., 1., -1., 1., -1., 1., -2., 2.);
    let pair_certificate = ExactRelu2Support::new(&pair_domain)
        .unwrap()
        .certify_normal_certificate([-1., -1., 2., 2.])
        .unwrap();
    assert_eq!(pair_certificate.facet().b, 2.);
    let mut single_max = f32::NEG_INFINITY;
    for i in 0..3 {
        for j in i + 1..3 {
            for lambda in [0.5f32, 1.] {
                let row = ResidentLowerCutRow::try_new(
                    vec![lambda],
                    [ResidentLowerCutChannel::try_new(-lambda, 0.)?; 2],
                    [ResidentLowerCutChannel::try_new(2. * lambda, 0.)?; 2],
                    ResidentLowerCutChannel::try_new(-2. * lambda, 0.)?,
                )?;
                let end = deadline();
                let cut = ResidentLowerCutCarrier::try_new(0, WIDTH, [i, j], vec![row], end)?;
                let got = device.crown_backward_gpu_resnet_lower_cut_with_deadline(
                    &full, &objective, &lower, &upper, &cut, end,
                )?;
                assert!(got <= -0.75 && got > -0.751);
                single_max = single_max.max(got);
                println!("case=triplet-single-control first={i} second={j} lambda={lambda} lower={got} support_origin=original_cycle_exact_hexagon");
            }
        }
    }
    assert!(single_max < 0.);
    let make = |extra: f32, end: Instant| -> Result<ResidentTripletLowerCutCarrier> {
        // This helper enforces IEEE before reducing ORIGINAL f32 operands and
        // independently checks the center/error interval with BigRational.
        let bias = joint_bias_channel(&[(1., -bound)], extra)?;
        ResidentTripletLowerCutCarrier::try_new(
            0,
            WIDTH,
            [0, 1, 2],
            1.,
            [ResidentLowerCutChannel::try_new(-1., 0.)?; 3],
            [ResidentLowerCutChannel::try_new(2., 0.)?; 3],
            bias,
            end,
        )
    };
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut completed = 0;
    for extra in [0., 0.125] {
        let end = deadline();
        let cut = make(extra, end)?;
        let got = device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        completed += 1;
        let exact = rat(2.25) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
        assert!(got.is_finite() && rat(got) <= exact.clone() && rat(got) > exact - rat(0.001));
        assert!(got > 0.12 && got <= 0.25 && got > single_max + 0.86);
        let mode = if extra == 0. {
            "exact"
        } else {
            "carrier-error"
        };
        println!("case=triplet-lower-cut mode={mode} carrier_extra_bias_error={extra} carrier_bias={} carrier_bias_abs_error={} lower={got} exact_graph_min=0.25", cut.bias().value(), cut.bias().source_abs_error());
    }

    let mut source_error = full.clone();
    let rho = 1. / 1024.;
    let delta = 1. / 64.;
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    *cert_err = CertifiedWeightError {
        weight_rel_err: rho,
        bias_abs_err: delta,
    };
    let end = deadline();
    let cut = make(0., end)?;
    let charged = device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
        &source_error,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    // For T=sum|z|<=2, F=9/4-T and sum|W|ReLU(z)=T. The exact model
    // parameter-ball minimum is 1/4-2rho-delta, attained when T=2.
    let exact_family_min = 0.25 - 2. * rho - delta;
    assert_eq!(
        rat(exact_family_min),
        rat(0.25) - rat(2.) * rat(rho) - rat(delta)
    );
    assert!(charged <= exact_family_min && charged > 0.20);
    println!("case=triplet-lower-cut mode=source-error lower={charged} model_weight_relative_error={rho} model_bias_abs_error={delta} exact_family_min={exact_family_min}");
    let mut unsafe_full = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_full[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
        unreachable!()
    };
    *bias = Some(Arc::from([1.75]));
    let end = deadline();
    let cut = make(0., end)?;
    let negative = device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
        &unsafe_full,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    let exact = rat(1.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(negative) <= exact && rat(negative) > exact - rat(0.001));
    assert!(negative <= -0.25 && negative > -0.251);
    println!("case=triplet-unsafe-shift shift=-0.5 lower={negative} exact_graph_min=-0.25");

    let end = deadline();
    let cut = make(0., end)?;
    let zero = ResidentTripletLowerCutCarrier::try_new(
        0,
        WIDTH,
        [0, 1, 2],
        0.,
        [ResidentLowerCutChannel::try_new(0., 0.)?; 3],
        [ResidentLowerCutChannel::try_new(0., 0.)?; 3],
        ResidentLowerCutChannel::try_new(0., 0.)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &zero, end
        ),
        Err(NyError::UnsupportedOp(_))
    ));
    let unsafe_error = ResidentTripletLowerCutCarrier::try_new(
        0,
        WIDTH,
        [0, 1, 2],
        1.,
        *cut.pre(),
        *cut.post(),
        ResidentLowerCutChannel::try_new(-bound, ny_core::CROWN_COEFF_MAX)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &unsafe_error,
            end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let wrong_width = ResidentTripletLowerCutCarrier::try_new(
        0,
        WIDTH - 1,
        [0, 1, 2],
        1.,
        *cut.pre(),
        *cut.post(),
        cut.bias(),
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &wrong_width,
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    cert_err.weight_rel_err = -1.;
    assert!(matches!(
        device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &source_error,
            &objective,
            &lower,
            &upper,
            &cut,
            end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let expired_cut = make(0., expired)?;
    deadline_refusal(
        device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &expired_cut,
            expired,
        ),
    );
    println!("case=triplet-refusals zero_multiplier=true unsafe_carrier_error=true wrong_width=true invalid_model_error=true expired_triplet=true");
    assert_eq!(completed, 4);
    println!("case=triplet-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=1 completed_single_pair_calls=6 completed_triplet_calls=4 expected_refusals=5 exact_graph_min=0.25 benchmark=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

// Exact source-box reduction for this authored dyadic control. This is an
// independent conformance oracle, not the private local-kernel/root authority.
// Original stored entries are authenticated in units of2^-10 before summing.
fn local_l1_authored_box(weight: &[f32], bias: &[f32], indices: &[usize]) -> (f32, Vec<[f32; 2]>) {
    ny_core::require_f64_interval_proof_environment().unwrap();
    assert_eq!(weight.len(), 10 * DIM);
    assert_eq!(bias.len(), 10);
    assert!((2..=8).contains(&indices.len()));
    assert!(indices.windows(2).all(|p| p[0] < p[1]));
    assert!(indices.iter().all(|&i| i < 8));
    let integer = |v: f32| -> i64 {
        assert!(v.is_finite() && v.abs() <= 8.);
        let scaled = f64::from(v) * 1024.;
        let exact = scaled as i64;
        assert_eq!(exact as f64, scaled);
        exact
    };
    let forms = 1usize << (indices.len() - 1);
    let mut endpoints = Vec::with_capacity(forms);
    let mut bound = 0i64;
    for mask in 0..forms {
        let sign = |position: usize| -> i64 {
            if position == 0 || mask & (1 << (position - 1)) != 0 {
                1
            } else {
                -1
            }
        };
        let center: i64 = indices
            .iter()
            .enumerate()
            .map(|(k, &i)| sign(k) * integer(bias[i]))
            .sum();
        let mut radius = 0i64;
        for col in 0..DIM {
            let coefficient: i64 = indices
                .iter()
                .enumerate()
                .map(|(k, &i)| sign(k) * integer(weight[i * DIM + col]))
                .sum();
            radius = radius.checked_add(coefficient.abs()).unwrap();
        }
        let lo = center.checked_sub(radius).unwrap();
        let hi = center.checked_add(radius).unwrap();
        bound = bound.max(-lo).max(hi);
        // These actual integer numerators fit the exact f32 mantissa. This
        // conversion introduces no uncharged rounding in the fixture oracle.
        assert!(lo.abs() < (1 << 24) && hi.abs() < (1 << 24));
        endpoints.push([lo as f32 / 1024., hi as f32 / 1024.]);
    }
    assert!(bound < (1 << 24));
    (bound as f32 / 1024., endpoints)
}

fn local_l1_four_diamond_truth() {
    use num_rational::BigRational;
    let rat = |v: f32| BigRational::from_float(f64::from(v)).unwrap();
    // For each independent pair |(u+v)/2|+|(u-v)/2|=max(|u|,|v|).
    // This identity bounds the entire cube; vertices establish attainability.
    let mut minimum = rat(100.);
    for mask in 0..256 {
        let x: [BigRational; 8] =
            std::array::from_fn(|i| rat(if mask & (1 << i) == 0 { -1. } else { 1. }));
        let abs = |v: BigRational| if v < rat(0.) { -v } else { v };
        let mut total = rat(0.);
        for pair in 0..4 {
            let a = (&x[2 * pair] + &x[2 * pair + 1]) / rat(2.);
            let b = (&x[2 * pair] - &x[2 * pair + 1]) / rat(2.);
            let sum = abs(a) + abs(b);
            assert_eq!(
                sum,
                abs(x[2 * pair].clone()).max(abs(x[2 * pair + 1].clone()))
            );
            total += sum;
        }
        minimum = minimum.min(rat(4.25) - total);
    }
    assert_eq!(minimum, rat(0.25));
    assert_eq!(minimum - rat(0.5), rat(-0.25));
    // All means zero give the exact maximum17/4; every original input
    // contributes to one of the eight independent512-coordinate means.
}

fn local_l1_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(device.provides_bounded_local_l1_lower_cut());
    assert!(device.provides_bounded_two_pair_lower_cut());
    assert!(device.provides_bounded_triplet_lower_cut());
    assert_eq!(DIM * DIM, 1 << 24);
    local_l1_four_diamond_truth();
    // Genuine identity Conv with16,777,216 stored f32 coefficients. The
    // selected original affine region is a subsequent Linear10x4096.
    let mut identity = vec![0.; DIM * DIM];
    for i in 0..DIM {
        identity[i * DIM + i] = 1.;
    }
    let feeder = GpuCrownLayer::Conv2d {
        weight_col: identity.into(),
        bias_expanded: Some(vec![0.; DIM].into()),
        out_channels: DIM,
        in_channels: DIM,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        out_h: 1,
        out_w: 1,
        in_h: 1,
        in_w: 1,
        cert_err: CertifiedWeightError::EXACT,
    };
    let mut weight = vec![0.; 10 * DIM];
    for pair in 0..4 {
        for col in pair * 1024..pair * 1024 + 512 {
            weight[(2 * pair) * DIM + col] = 1. / 1024.;
            weight[(2 * pair + 1) * DIM + col] = 1. / 1024.;
            weight[8 * DIM + col] = 1. / 512.;
        }
        for col in pair * 1024 + 512..(pair + 1) * 1024 {
            weight[(2 * pair) * DIM + col] = 1. / 1024.;
            weight[(2 * pair + 1) * DIM + col] = -1. / 1024.;
        }
    }
    for col in 0..512 {
        weight[9 * DIM + col] = 1. / 512.;
    }
    for col in 0..DIM {
        assert_eq!(
            (0..8).filter(|&row| weight[row * DIM + col] != 0.).count(),
            2
        );
    }
    let mut bias = vec![0.; 10];
    bias[8] = 5.;
    bias[9] = 2.;
    let (bound, endpoints) = local_l1_authored_box(&weight, &bias, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(endpoints.len(), 128);
    assert_eq!(bound, 4.);
    // The original selected source bytes produce the128 exact signed supports.
    // No ideal bound replaces returned endpoints or a missing kernel result.
    println!("case=local-l1-original-box signed_endpoints={endpoints:?} local_bound={bound} active_len=8 ordered_neurons=[0,1,2,3,4,5,6,7] weights=[1,1,1,1,1,1,1,1] source=authored_exact_linear_box private_kernel=false support4_calls=0");
    let mut pair_bounds = [[0.; 8]; 8];
    for a in 0..8 {
        for b in a + 1..8 {
            pair_bounds[a][b] = local_l1_authored_box(&weight, &bias, &[a, b]).0;
            assert_eq!(pair_bounds[a][b], if a / 2 == b / 2 { 1. } else { 2. });
        }
    }
    let mut triplets = Vec::new();
    for a in 0..8 {
        for b in a + 1..8 {
            for c in b + 1..8 {
                let indices = [a, b, c];
                let local = local_l1_authored_box(&weight, &bias, &indices).0;
                let contains_pair = a / 2 == b / 2 || b / 2 == c / 2;
                assert_eq!(local, if contains_pair { 2. } else { 3. });
                triplets.push((indices, local));
            }
        }
    }
    let mut pairs = 0;
    for a in 0..8 {
        for b in a + 1..8 {
            for c in a + 1..8 {
                for d in c + 1..8 {
                    if b == c || b == d {
                        continue;
                    }
                    // All210 matchings are checked against their actual source-box bounds.
                    let full_floor = -3.75 + (2. - pair_bounds[a][b]) + (2. - pair_bounds[c][d]);
                    assert!(full_floor <= -1.75);
                    pairs += 1;
                }
            }
        }
    }
    assert_eq!(pairs, 210);
    assert_eq!(triplets.len(), 56);
    let mut slopes = vec![0.5; 10];
    slopes[8] = 1.;
    slopes[9] = 1.;
    let mut intercept = vec![0.5; 10];
    intercept[8] = 0.;
    intercept[9] = 0.;
    let mut final_weight = vec![-2.; 10];
    final_weight[8] = 1.;
    final_weight[9] = 0.;
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: final_weight.into(),
            bias: Some(Arc::from([-0.75])),
            out_features: 1,
            in_features: 10,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: slopes.clone(),
            upper_slope: slopes,
            lower_intercept: vec![0.; 10],
            upper_intercept: intercept,
            num_neurons: 10,
        },
        GpuCrownLayer::Linear {
            weight: weight.into(),
            bias: Some(bias.into()),
            out_features: 10,
            in_features: DIM,
            cert_err: CertifiedWeightError::EXACT,
        },
        feeder,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let objective = seed(vec![1.], 1, 1);
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 4.25), (-4.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -3.75 && baseline.lower_bounds[0] > -3.751);
    println!("case=local-l1-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_triangle_min=-3.75",baseline.lower_bounds,baseline.upper_bounds);
    // The six best two-pair choices each contain two genuine diamond pairs.
    // Every other matching has a larger exact box B and is provably no better.
    // Full scale dominates all independent half/full choices on this fixture:
    // each scalar gain is lambda*(number_of_channels-B), with lambda<=1.
    let mut pair_calls = 0;
    for a in 0..4 {
        for b in a + 1..4 {
            let indices = [[2 * a, 2 * a + 1], [2 * b, 2 * b + 1]];
            let rhs = pair_bounds[2 * a][2 * a + 1] + pair_bounds[2 * b][2 * b + 1];
            let end = deadline();
            let cut = ResidentTwoPairLowerCutCarrier::try_new(
                0,
                10,
                indices,
                [1.; 2],
                [ResidentLowerCutChannel::try_new(-1., 0.)?; 4],
                [ResidentLowerCutChannel::try_new(2., 0.)?; 4],
                joint_bias_channel(&[(1., -rhs)], 0.)?,
                end,
            )?;
            let got = device.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
                &full, &objective, &lower, &upper, &cut, end,
            )?;
            assert!(got <= -1.75 && got > -1.751);
            pair_calls += 1;
            println!("case=local-l1-two-pair-control pairs={indices:?} scales=[1,1] lower={got} exact_box_bound={rhs} comparison=all_best_source_pairs");
        }
    }
    let mut triplet_calls = 0;
    for (indices, rhs) in triplets {
        if rhs != 2. {
            continue;
        }
        let end = deadline();
        let cut = ResidentTripletLowerCutCarrier::try_new(
            0,
            10,
            indices,
            1.,
            [ResidentLowerCutChannel::try_new(-1., 0.)?; 3],
            [ResidentLowerCutChannel::try_new(2., 0.)?; 3],
            joint_bias_channel(&[(1., -rhs)], 0.)?,
            end,
        )?;
        let got = device.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        assert!(got <= -2.75 && got > -2.751);
        triplet_calls += 1;
        println!("case=local-l1-triplet-control ordered_neurons={indices:?} multiplier=1 lower={got} exact_box_bound={rhs} comparison=all_best_source_triplets");
    }
    assert_eq!(pair_calls, 6);
    assert_eq!(triplet_calls, 24);
    let make = |extra: f32, end: Instant| -> Result<ResidentLocalL1LowerCutCarrier> {
        let bias = joint_bias_channel(&[(1., -bound)], extra)?;
        ResidentLocalL1LowerCutCarrier::try_new(
            0,
            10,
            8,
            [0, 1, 2, 3, 4, 5, 6, 7],
            1.,
            [ResidentLowerCutChannel::try_new(-1., 0.)?; 8],
            [ResidentLowerCutChannel::try_new(2., 0.)?; 8],
            bias,
            end,
        )
    };
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut completed = 0;
    for extra in [0., 0.125] {
        let end = deadline();
        let cut = make(extra, end)?;
        let got = device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        let exact = rat(4.25) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
        assert!(rat(got) <= exact.clone() && rat(got) > exact - rat(0.001));
        assert!(got > 0.12 && got <= 0.25);
        completed += 1;
        let mode = if extra == 0. {
            "exact"
        } else {
            "carrier-error"
        };
        println!("case=local-l1-cut mode={mode} lower={got} carrier_extra_bias_error={extra} carrier_bias={} carrier_bias_abs_error={} exact_graph_min=0.25",cut.bias().value(),cut.bias().source_abs_error());
    }
    let mut source_error = full.clone();
    let rho = 1. / 1024.;
    let delta = 1. / 64.;
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    *cert_err = CertifiedWeightError {
        weight_rel_err: rho,
        bias_abs_err: delta,
    };
    let end = deadline();
    let cut = make(0., end)?;
    let charged = device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
        &source_error,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    // Selected local Linear remains exact. Only the downstream output affine
    // has declared uncertainty. T=sum|z|<=4,U=sum(u)<=4, and sum|W|h=T+2U+5.
    // Thus exact parameter-family min=.25-17rho-delta, at u=1,v=0 in each pair.
    let exact_family_min = 0.25 - 17. * rho - delta;
    assert_eq!(
        rat(exact_family_min),
        rat(0.25) - rat(17.) * rat(rho) - rat(delta)
    );
    assert!(charged <= exact_family_min && charged > 0.15);
    println!("case=local-l1-cut mode=source-error lower={charged} model_weight_relative_error={rho} model_bias_abs_error={delta} exact_family_min={exact_family_min}");
    let mut unsafe_full = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_full[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
        unreachable!()
    };
    *bias = Some(Arc::from([-1.25]));
    let end = deadline();
    let cut = make(0., end)?;
    let negative = device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
        &unsafe_full,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    let exact = rat(3.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(negative) <= exact && rat(negative) > exact - rat(0.001));
    assert!(negative <= -0.25 && negative > -0.251);
    println!("case=local-l1-unsafe-shift shift=-0.5 lower={negative} exact_graph_min=-0.25");

    let end = deadline();
    let cut = make(0., end)?;
    assert!(matches!(
        ResidentLocalL1LowerCutCarrier::try_new(
            0,
            10,
            8,
            [0, 1, 2, 3, 4, 5, 6, 7],
            0.,
            *cut.pre(),
            *cut.post(),
            cut.bias(),
            end
        ),
        Err(NyError::NumericalInstability(_))
    ));
    let mut pad_pre = *cut.pre();
    let mut pad_post = *cut.post();
    pad_pre[7] = ResidentLowerCutChannel::try_new(0., 0.)?;
    pad_post[7] = ResidentLowerCutChannel::try_new(0., 0.)?;
    // Slot7 remains a nonzero index while active_len7: constructor refusal.
    assert!(matches!(
        ResidentLocalL1LowerCutCarrier::try_new(
            0,
            10,
            7,
            [0, 1, 2, 3, 4, 5, 6, 7],
            1.,
            pad_pre,
            pad_post,
            cut.bias(),
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let bad_error = ResidentLocalL1LowerCutCarrier::try_new(
        0,
        10,
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        1.,
        *cut.pre(),
        *cut.post(),
        ResidentLowerCutChannel::try_new(-bound, ny_core::CROWN_COEFF_MAX)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &bad_error, end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let wrong_width = ResidentLocalL1LowerCutCarrier::try_new(
        0,
        11,
        8,
        [0, 1, 2, 3, 4, 5, 6, 7],
        1.,
        *cut.pre(),
        *cut.post(),
        cut.bias(),
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &wrong_width,
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let mut bad_seed = objective.clone();
    bad_seed.upper_a = Arc::from([2.]);
    assert!(matches!(
        device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full, &bad_seed, &lower, &upper, &cut, end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    cert_err.weight_rel_err = -1.;
    assert!(matches!(
        device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &source_error,
            &objective,
            &lower,
            &upper,
            &cut,
            end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let expired_cut = make(0., expired)?;
    deadline_refusal(
        device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &expired_cut,
            expired,
        ),
    );
    println!("case=local-l1-refusals zero_multiplier_constructor=true noncanonical_padding_constructor=true unsafe_carrier_error=true wrong_width=true asymmetric_seed=true invalid_model_error=true expired_local=true");
    assert_eq!(completed, 4);
    println!("case=local-l1-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=0 completed_joint_calls=6 completed_triplet_calls=24 completed_local_l1_calls=4 expected_constructor_refusals=2 expected_backend_refusals=5 expected_refusals=7 exact_graph_min=0.25 benchmark=false private_kernel=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

// Independent authored-source oracle. It authenticates exact stored dyadic
// coefficients, not private kernel/map/owner authority. No support4 is called.
fn two_local_l1_authored_box(
    weight: &[f32],
    bias: &[f32],
    indices: &[usize; 8],
) -> (f32, Vec<[f32; 2]>) {
    ny_core::require_f64_interval_proof_environment().unwrap();
    assert_eq!(weight.len(), 17 * DIM);
    assert_eq!(bias.len(), 17);
    assert!(indices.windows(2).all(|p| p[0] < p[1]) && indices.iter().all(|&i| i < 16));
    let integer = |v: f32| -> i64 {
        assert!(v.is_finite() && v.abs() <= 32.);
        let scaled = f64::from(v) * 512.;
        let exact = scaled as i64;
        assert_eq!(exact as f64, scaled);
        exact
    };
    let mut endpoints = Vec::with_capacity(128);
    let mut bound = 0i64;
    for mask in 0..128 {
        let sign = |k: usize| -> i64 {
            if k == 0 || mask & (1 << (k - 1)) != 0 {
                1
            } else {
                -1
            }
        };
        let center: i64 = indices
            .iter()
            .enumerate()
            .map(|(k, &i)| sign(k) * integer(bias[i]))
            .sum();
        let mut radius = 0i64;
        for col in 0..DIM {
            let coefficient: i64 = indices
                .iter()
                .enumerate()
                .map(|(k, &i)| sign(k) * integer(weight[i * DIM + col]))
                .sum();
            radius = radius.checked_add(coefficient.abs()).unwrap();
        }
        let lo = center.checked_sub(radius).unwrap();
        let hi = center.checked_add(radius).unwrap();
        assert!(lo.abs() < (1 << 24) && hi.abs() < (1 << 24));
        endpoints.push([lo as f32 / 512., hi as f32 / 512.]);
        bound = bound.max(-lo).max(hi);
    }
    assert!(bound < (1 << 24));
    (bound as f32 / 512., endpoints)
}

fn two_local_l1_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(device.provides_bounded_two_local_l1_lower_cut());
    assert!(device.provides_bounded_local_l1_lower_cut());
    assert_eq!(DIM * DIM, 1 << 24);
    let mut identity = vec![0.; DIM * DIM];
    for i in 0..DIM {
        identity[i * DIM + i] = 1.;
    }
    let feeder = GpuCrownLayer::Conv2d {
        weight_col: identity.into(),
        bias_expanded: Some(vec![0.; DIM].into()),
        out_channels: DIM,
        in_channels: DIM,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        out_h: 1,
        out_w: 1,
        in_h: 1,
        in_w: 1,
        cert_err: CertifiedWeightError::EXACT,
    };
    // Sixteen disjoint256-coordinate means; every original input participates.
    // z2k=(uk+vk)/2,z2k+1=(uk-vk)/2; t=sum(u)+17 is positive throughout.
    let mut weight = vec![0.; 17 * DIM];
    for pair in 0..8 {
        for col in pair * 512..pair * 512 + 256 {
            weight[(2 * pair) * DIM + col] = 1. / 512.;
            weight[(2 * pair + 1) * DIM + col] = 1. / 512.;
            weight[16 * DIM + col] = 1. / 256.;
        }
        for col in pair * 512 + 256..(pair + 1) * 512 {
            weight[(2 * pair) * DIM + col] = 1. / 512.;
            weight[(2 * pair + 1) * DIM + col] = -1. / 512.;
        }
    }
    for col in 0..DIM {
        assert_eq!(
            (0..16).filter(|&row| weight[row * DIM + col] != 0.).count(),
            2
        );
    }
    let mut bias = vec![0.; 17];
    bias[16] = 17.;
    let groups = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]];
    let (b0, e0) = two_local_l1_authored_box(&weight, &bias, &groups[0]);
    let (b1, e1) = two_local_l1_authored_box(&weight, &bias, &groups[1]);
    assert_eq!((b0, b1), (4., 4.));
    assert_eq!((e0.len(), e1.len()), (128, 128));
    println!("case=two-local-l1-original-box group_0_signed_endpoints={e0:?} group_1_signed_endpoints={e1:?} group_0_bound={b0} group_1_bound={b1} active_lens=[8,8] ordered_neurons={groups:?} weights=[[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]] source=authored_exact_linear_box private_kernel=false support4_calls=0");
    // Each diamond contributes max(|u|,|v|)<=1 on the WHOLE square.
    // Thus F=33/4-T has range[1/4,33/4], attained at all u=1,v=0 /all0.
    // A subset of m cells with p complete diamonds has exact B=m-p.
    // Its actual chord lower is -31/4+lambda*p. Full scale dominates half,
    // and p<=4. Exhaust all memberships; only four representative GPU folds
    // are claimed below. All70 maximizing groups are not device-executed.
    let mut subsets = 0;
    let mut best_count = 0;
    let mut max_pairs = 0;
    for mask in 0u32..(1 << 16) {
        let m = mask.count_ones();
        if !(3..=8).contains(&m) {
            continue;
        }
        let pairs = (0..8).filter(|&k| ((mask >> (2 * k)) & 3) == 3).count();
        assert!(pairs <= 4);
        subsets += 1;
        max_pairs = max_pairs.max(pairs);
        if pairs == 4 {
            best_count += 1;
        }
    }
    assert_eq!((subsets, best_count, max_pairs), (39066, 70, 4));
    println!("case=two-local-l1-single-family-census subset_count=39066 maximizing_subsets=70 best_exact_single_lower=-3.75 exact_ordinary_lower=-7.75 proof=original_diamond_membership_and_chord representative_gpu_calls=4 all_subsets_gpu_executed=false");
    let mut slopes = vec![0.5; 17];
    slopes[16] = 1.;
    let mut intercept = vec![0.5; 17];
    intercept[16] = 0.;
    let mut final_weight = vec![-2.; 17];
    final_weight[16] = 1.;
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: final_weight.into(),
            bias: Some(Arc::from([-8.75])),
            out_features: 1,
            in_features: 17,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: slopes.clone(),
            upper_slope: slopes,
            lower_intercept: vec![0.; 17],
            upper_intercept: intercept,
            num_neurons: 17,
        },
        GpuCrownLayer::Linear {
            weight: weight.into(),
            bias: Some(bias.into()),
            out_features: 17,
            in_features: DIM,
            cert_err: CertifiedWeightError::EXACT,
        },
        feeder,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let objective = seed(vec![1.], 1, 1);
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 8.25), (-8.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -7.75 && baseline.lower_bounds[0] > -7.751);
    println!("case=two-local-l1-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_graph_max=8.25 exact_triangle_min=-7.75",baseline.lower_bounds,baseline.upper_bounds);
    let mut single_calls = 0;
    for group in 0..2 {
        for lambda in [0.5, 1.] {
            let bound = if group == 0 { b0 } else { b1 };
            let end = deadline();
            let cut = ResidentLocalL1LowerCutCarrier::try_new(
                0,
                17,
                8,
                groups[group],
                lambda,
                [ResidentLowerCutChannel::try_new(-lambda, 0.)?; 8],
                [ResidentLowerCutChannel::try_new(2. * lambda, 0.)?; 8],
                joint_bias_channel(&[(lambda, -bound)], 0.)?,
                end,
            )?;
            let got = device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
                &full, &objective, &lower, &upper, &cut, end,
            )?;
            let exact = -7.75 + 4. * lambda;
            assert!(got <= exact && got > exact - 0.001);
            single_calls += 1;
            println!("case=two-local-l1-single-control group={group} ordered_neurons={:?} multiplier={lambda} lower={got} exact_lower={exact} comparison=representative_best_single_group",groups[group]);
        }
    }
    let make = |extra: f32, end: Instant| -> Result<ResidentTwoLocalL1LowerCutCarrier> {
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            17,
            [8, 8],
            groups,
            [1., 1.],
            [[ResidentLowerCutChannel::try_new(-1., 0.)?; 8]; 2],
            [[ResidentLowerCutChannel::try_new(2., 0.)?; 8]; 2],
            joint_bias_channel(&[(1., -b0), (1., -b1)], extra)?,
            end,
        )
    };
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut completed = 0;
    for extra in [0., 0.125] {
        let end = deadline();
        let cut = make(extra, end)?;
        let got = device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        let exact = rat(8.25) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
        assert!(rat(got) <= exact.clone() && rat(got) > exact - rat(0.001));
        assert!(got > 0.12 && got <= 0.25);
        completed += 1;
        let mode = if extra == 0. {
            "exact"
        } else {
            "carrier-error"
        };
        println!("case=two-local-l1-cut mode={mode} lower={got} carrier_extra_bias_error={extra} carrier_bias={} carrier_bias_abs_error={} exact_graph_min=0.25",cut.bias().value(),cut.bias().source_abs_error());
    }
    let mut source_error = full.clone();
    let rho = 1. / 4096.;
    let delta = 1. / 128.;
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    *cert_err = CertifiedWeightError {
        weight_rel_err: rho,
        bias_abs_err: delta,
    };
    let end = deadline();
    let cut = make(0., end)?;
    let charged = device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
        &source_error,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    // Only downstream output affine uncertainty: selected Linear stays exact.
    // sum|W|h=T+2U+17, T<=8,U<=8; exact family min=.25-41rho-delta.
    // Attainment: all u=1,v=0 and each output parameter at its negative error.
    let exact_family_min = 0.25 - 41. * rho - delta;
    assert_eq!(
        rat(exact_family_min),
        rat(0.25) - rat(41.) * rat(rho) - rat(delta)
    );
    assert_eq!(rat(exact_family_min), rat(951.) / rat(4096.));
    assert!(charged <= exact_family_min && charged > 0.1);
    println!("case=two-local-l1-cut mode=source-error lower={charged} model_weight_relative_error={rho} model_bias_abs_error={delta} exact_family_min={exact_family_min}");
    let mut unsafe_full = full.clone();
    let GpuResnetSegment::Chain(layers) = &mut unsafe_full[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
        unreachable!()
    };
    *bias = Some(Arc::from([-9.25]));
    let end = deadline();
    let cut = make(0., end)?;
    let negative = device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
        &unsafe_full,
        &objective,
        &lower,
        &upper,
        &cut,
        end,
    )?;
    completed += 1;
    let exact = rat(7.75) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
    assert!(rat(negative) <= exact && rat(negative) > exact - rat(0.001));
    assert!(negative <= -0.25 && negative > -0.251);
    println!("case=two-local-l1-unsafe-shift shift=-0.5 lower={negative} exact_graph_min=-0.25");

    let end = deadline();
    let cut = make(0., end)?;
    assert!(matches!(
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            17,
            [8, 8],
            groups,
            [1., 0.],
            *cut.pre(),
            *cut.post(),
            cut.bias(),
            end
        ),
        Err(NyError::NumericalInstability(_))
    ));
    assert!(matches!(
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            17,
            [8, 8],
            [groups[0], groups[0]],
            [1., 1.],
            *cut.pre(),
            *cut.post(),
            cut.bias(),
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let mut pad_pre = *cut.pre();
    let mut pad_post = *cut.post();
    pad_pre[1][7] = ResidentLowerCutChannel::try_new(0., 0.)?;
    pad_post[1][7] = ResidentLowerCutChannel::try_new(0., 0.)?;
    assert!(matches!(
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            17,
            [8, 7],
            groups,
            [1., 1.],
            pad_pre,
            pad_post,
            cut.bias(),
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let bad_error = ResidentTwoLocalL1LowerCutCarrier::try_new(
        0,
        17,
        [8, 8],
        groups,
        [1., 1.],
        *cut.pre(),
        *cut.post(),
        ResidentLowerCutChannel::try_new(-b0 - b1, ny_core::CROWN_COEFF_MAX)?,
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &bad_error, end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let wrong_width = ResidentTwoLocalL1LowerCutCarrier::try_new(
        0,
        18,
        [8, 8],
        groups,
        [1., 1.],
        *cut.pre(),
        *cut.post(),
        cut.bias(),
        end,
    )?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &wrong_width,
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let mut bad_seed = objective.clone();
    bad_seed.upper_a = Arc::from([2.]);
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full, &bad_seed, &lower, &upper, &cut, end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let GpuResnetSegment::Chain(layers) = &mut source_error[0] else {
        unreachable!()
    };
    let GpuCrownLayer::Linear { cert_err, .. } = &mut layers[0] else {
        unreachable!()
    };
    cert_err.weight_rel_err = -1.;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &source_error,
            &objective,
            &lower,
            &upper,
            &cut,
            end
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let expired_cut = make(0., expired)?;
    deadline_refusal(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &expired_cut,
            expired,
        ),
    );
    println!("case=two-local-l1-refusals zero_multiplier_constructor=true cross_group_alias_constructor=true noncanonical_padding_constructor=true unsafe_carrier_error=true wrong_width=true asymmetric_seed=true invalid_model_error=true expired_local=true");
    assert_eq!((single_calls, completed), (4, 4));
    println!("case=two-local-l1-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=0 completed_single_local_calls=4 completed_two_local_calls=4 expected_constructor_refusals=3 expected_backend_refusals=5 expected_refusals=8 exact_graph_min=0.25 benchmark=false private_kernel=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

// Authored stored-row oracle only; no private local-kernel/source seal authority.
fn nine_cycle_authored_box(
    weight: &[f32],
    bias: &[f32],
    indices: &[usize],
) -> (f32, Vec<[f32; 2]>) {
    ny_core::require_f64_interval_proof_environment().unwrap();
    assert_eq!(weight.len(), 10 * DIM);
    assert_eq!(bias.len(), 10);
    assert!((3..=8).contains(&indices.len()) && indices.windows(2).all(|p| p[0] < p[1]));
    assert!(indices.iter().all(|&i| i < 9));
    let integer = |v: f32| -> i64 {
        assert!(v.is_finite() && v.abs() <= 32.);
        let scaled = f64::from(v) * 2048.;
        let exact = scaled as i64;
        assert_eq!(exact as f64, scaled);
        exact
    };
    let mut endpoints = Vec::new();
    let mut bound = 0i64;
    for mask in 0usize..(1 << (indices.len() - 1)) {
        let sign = |k: usize| -> i64 {
            if k == 0 || mask & (1 << (k - 1)) != 0 {
                1
            } else {
                -1
            }
        };
        let center: i64 = indices
            .iter()
            .enumerate()
            .map(|(k, &i)| sign(k) * integer(bias[i]))
            .sum();
        let mut radius = 0i64;
        for col in 0..DIM {
            let coefficient: i64 = indices
                .iter()
                .enumerate()
                .map(|(k, &i)| sign(k) * integer(weight[i * DIM + col]))
                .sum();
            radius = radius.checked_add(coefficient.abs()).unwrap();
        }
        let lo = center.checked_sub(radius).unwrap();
        let hi = center.checked_add(radius).unwrap();
        assert!(lo.abs() < (1 << 24) && hi.abs() < (1 << 24));
        endpoints.push([lo as f32 / 2048., hi as f32 / 2048.]);
        bound = bound.max(-lo).max(hi);
    }
    assert!(bound < (1 << 24));
    (bound as f32 / 2048., endpoints)
}

fn nine_cycle_two_local_conformance(device: &CudaGemmEngine) -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    assert!(device.provides_bounded_two_local_l1_lower_cut());
    assert!(device.provides_bounded_local_l1_lower_cut());
    assert_eq!(DIM * DIM, 1 << 24);
    let mut identity = vec![0.; DIM * DIM];
    for i in 0..DIM {
        identity[i * DIM + i] = 1.;
    }
    let feeder = GpuCrownLayer::Conv2d {
        weight_col: identity.into(),
        bias_expanded: Some(vec![0.; DIM].into()),
        out_channels: DIM,
        in_channels: DIM,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        out_h: 1,
        out_w: 1,
        in_h: 1,
        in_w: 1,
        cert_err: CertifiedWeightError::EXACT,
    };
    let sizes = [1024usize, 1024, 512, 512, 256, 256, 256, 128, 128];
    let mut starts = [0usize; 9];
    for i in 1..9 {
        starts[i] = starts[i - 1] + sizes[i - 1];
    }
    assert_eq!(starts[8] + sizes[8], DIM);
    let mut weight = vec![0.; 10 * DIM];
    for cycle in 0..3 {
        let u = 3 * cycle;
        let v = u + 1;
        let w = u + 2;
        for col in starts[u]..starts[u] + sizes[u] {
            weight[u * DIM + col] = 0.5 / sizes[u] as f32;
            weight[w * DIM + col] = -0.5 / sizes[u] as f32;
        }
        for col in starts[v]..starts[v] + sizes[v] {
            weight[u * DIM + col] = 0.5 / sizes[v] as f32;
            weight[v * DIM + col] = 0.5 / sizes[v] as f32;
            weight[9 * DIM + col] = 1. / sizes[v] as f32;
        }
        for col in starts[w]..starts[w] + sizes[w] {
            weight[v * DIM + col] = 0.5 / sizes[w] as f32;
            weight[w * DIM + col] = 0.5 / sizes[w] as f32;
            weight[9 * DIM + col] = 1. / sizes[w] as f32;
        }
    }
    for col in 0..DIM {
        assert_eq!(
            (0..9).filter(|&row| weight[row * DIM + col] != 0.).count(),
            2
        );
    }
    let mut bias = vec![0.; 10];
    bias[9] = 7.;
    let groups = [[0, 1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]];
    let (b0, e0) = nine_cycle_authored_box(&weight, &bias, &groups[0][..6]);
    let (b1, e1) = nine_cycle_authored_box(&weight, &bias, &groups[1][..3]);
    let single_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let (single_b, _) = nine_cycle_authored_box(&weight, &bias, &single_ids);
    assert_eq!((b0, b1, single_b), (4., 2., 6.));
    assert_eq!((e0.len(), e1.len()), (32, 4));
    // Each cycle has six cube corners with L1=2 and two zero corners.
    // Convexity and source independence give whole-box T<=6. Any m-cell
    // subset has B=m-c, c=number of complete cycles. A single<=8 has c<=2.
    let mut subset_count = 0;
    let mut maximizing_count = 0;
    for mask in 0u32..512 {
        let m = mask.count_ones();
        if !(3..=8).contains(&m) {
            continue;
        }
        let c = (0..3).filter(|&k| (mask >> (3 * k)) & 7 == 7).count();
        assert!(c <= 2);
        subset_count += 1;
        maximizing_count += usize::from(c == 2);
    }
    assert_eq!((subset_count, maximizing_count), (465, 21));
    println!("case=nine-cycle-original-box active_lens=[6,3] ordered_neurons={groups:?} group_0_signed_endpoints={e0:?} group_1_signed_endpoints={e1:?} group_0_bound={b0} group_1_bound={b1} single_eight_bound={single_b} subset_count=465 best_single_groups=21 best_exact_single_lower=-0.75 representative_single_gpu_calls=2 all_subsets_gpu_executed=false support4_calls=0 private_kernel=false");
    let mut slopes = vec![0.5; 10];
    slopes[9] = 1.;
    let mut intercept = vec![0.5; 10];
    intercept[9] = 0.;
    let mut final_weight = vec![-2.; 10];
    final_weight[9] = 1.;
    let full = [GpuResnetSegment::Chain(vec![
        GpuCrownLayer::Linear {
            weight: final_weight.into(),
            bias: Some(Arc::from([-0.75])),
            out_features: 1,
            in_features: 10,
            cert_err: CertifiedWeightError::EXACT,
        },
        GpuCrownLayer::Activation {
            lower_slope: slopes.clone(),
            upper_slope: slopes,
            lower_intercept: vec![0.; 10],
            upper_intercept: intercept,
            num_neurons: 10,
        },
        GpuCrownLayer::Linear {
            weight: weight.into(),
            bias: Some(bias.into()),
            out_features: 10,
            in_features: DIM,
            cert_err: CertifiedWeightError::EXACT,
        },
        feeder,
    ])];
    let lower = vec![-1.; DIM];
    let upper = vec![1.; DIM];
    let objective = seed(vec![1.], 1, 1);
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &full,
        &seed(vec![1., -1.], 1, 2),
        &lower,
        &upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 6.25), (-6.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -2.75 && baseline.lower_bounds[0] > -2.751);
    println!("case=nine-cycle-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_graph_max=6.25 exact_triangle_min=-2.75", baseline.lower_bounds, baseline.upper_bounds);
    let mut single_calls = 0;
    for lambda in [0.5, 1.] {
        let end = deadline();
        let cut = ResidentLocalL1LowerCutCarrier::try_new(
            0,
            10,
            8,
            single_ids,
            lambda,
            [ResidentLowerCutChannel::try_new(-lambda, 0.)?; 8],
            [ResidentLowerCutChannel::try_new(2. * lambda, 0.)?; 8],
            joint_bias_channel(&[(lambda, -single_b)], 0.)?,
            end,
        )?;
        let got = device.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &full, &objective, &lower, &upper, &cut, end,
        )?;
        let exact = -2.75 + 2. * lambda;
        assert!(got <= exact && got > exact - 0.001);
        single_calls += 1;
        println!("case=nine-cycle-single-control active_len=8 ordered_neurons={single_ids:?} multiplier={lambda} lower={got} exact_lower={exact} comparison=one_best_single_group");
    }
    let make = |width: usize, end: Instant| -> Result<ResidentTwoLocalL1LowerCutCarrier> {
        let zero = ResidentLowerCutChannel::try_new(0., 0.)?;
        let mut pre = [[zero; 8]; 2];
        let mut post = [[zero; 8]; 2];
        for (g, count) in [6usize, 3].into_iter().enumerate() {
            for i in 0..count {
                pre[g][i] = ResidentLowerCutChannel::try_new(-1., 0.)?;
                post[g][i] = ResidentLowerCutChannel::try_new(2., 0.)?;
            }
        }
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            width,
            [6, 3],
            groups,
            [1., 1.],
            pre,
            post,
            joint_bias_channel(&[(1., -b0), (1., -b1)], 0.)?,
            end,
        )
    };
    let rat = |v: f32| num_rational::BigRational::from_float(f64::from(v)).unwrap();
    let mut completed = 0;
    for shift in [0., -0.5] {
        let mut actual = full.clone();
        let GpuResnetSegment::Chain(layers) = &mut actual[0] else {
            unreachable!()
        };
        let GpuCrownLayer::Linear { bias, .. } = &mut layers[0] else {
            unreachable!()
        };
        *bias = Some(Arc::from([-0.75 + shift]));
        let end = deadline();
        let cut = make(10, end)?;
        let got = device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &actual, &objective, &lower, &upper, &cut, end,
        )?;
        let exact =
            rat(6.25 + shift) + rat(cut.bias().value()) - rat(cut.bias().source_abs_error());
        assert!(rat(got) <= exact.clone() && rat(got) > exact - rat(0.001));
        assert!(got <= 0.25 + shift && got > 0.249 + shift);
        completed += 1;
        println!("case=nine-cycle-two-local shift={shift} lower={got} exact_graph_min={} active_lens=[6,3] carrier_bias={} carrier_bias_abs_error={} support4_calls=0", 0.25 + shift, cut.bias().value(), cut.bias().source_abs_error());
    }
    let end = deadline();
    let cut = make(10, end)?;
    let mut bad_padding = groups;
    bad_padding[0][6] = 9;
    assert!(matches!(
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            0,
            10,
            [6, 3],
            bad_padding,
            [1., 1.],
            *cut.pre(),
            *cut.post(),
            cut.bias(),
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let wrong_width = make(11, end)?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
            &full,
            &objective,
            &lower,
            &upper,
            &wrong_width,
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    println!("case=nine-cycle-refusals noncanonical_inactive_index=true wrong_width=true expected_constructor_refusals=1 expected_backend_refusals=1");
    assert_eq!((single_calls, completed), (2, 2));
    println!("case=nine-cycle-completed actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=0 completed_single_local_calls=2 completed_two_local_calls=2 expected_refusals=2 exact_graph_min=0.25 benchmark=false private_kernel=false private_root_consumer=false loader_provenance=false");
    Ok(())
}

fn main() -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    // No global factory, optional device lookup, fallback backend, skip, or
    // test-only size-gate override occurs anywhere in this executable.
    let device = CudaGemmEngine::new()?;
    assert!(device.provides_bounded_support4() && device.provides_bounded_lower_cut());
    println!("device={} transport={} original_input_dim={DIM} hidden_dim={WIDTH} one_row_dense_macs={} call_deadline_s={CALL_SECONDS}",
        device.device_name(), device.deadline_f64_transport_name(), DIM * WIDTH);
    let d = Diamond::new();
    // Independent all-domain identity, with both endpoints attainable:
    // F=5/4+u-2R((u+v)/2)-2R((u-v)/2)=5/4-max(|u|,|v|).
    // Therefore F ranges over [1/4,5/4]. The original triangle rows give
    // F >= -3/4 and F <= 9/4, in exact arithmetic.
    // The two baseline objectives F and -F are distinct, never row padding.
    let baseline = device.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &d.full,
        &seed(vec![1.0, -1.0], 1, 2),
        &d.lower,
        &d.upper,
        &[],
        &[],
        deadline(),
    )?;
    check_encloses(&baseline, &[(0.25, 1.25), (-1.25, -0.25)]);
    assert!(baseline.lower_bounds[0] <= -0.75 && baseline.lower_bounds[0] > -0.751);
    println!("case=ordinary-two-objective-baseline lower={:?} upper={:?} exact_graph_min=0.25 exact_triangle_min=-0.75", baseline.lower_bounds, baseline.upper_bounds);

    let support_seed = support_seed();
    let support = device.crown_backward_gpu_resnet_support4_with_deadline(
        &d.prefix,
        &support_seed,
        &d.lower,
        &d.upper,
        deadline(),
    )?;
    let certificate = certify(&support);
    println!(
        "case=actual-checked-support lower={:?} upper={:?} normal={:?} certified_intercept={}",
        support.lower_bounds,
        support.upper_bounds,
        certificate.facet().a,
        certificate.facet().b
    );
    let objective = seed(vec![1.0], 1, 1);
    let mut lower_results = Vec::new();
    for error in [0.0, 0.125] {
        let end = deadline();
        let cut = carrier(&certificate, error, WIDTH, end)?;
        let got = device.crown_backward_gpu_resnet_lower_cut_with_deadline(
            &d.full, &objective, &d.lower, &d.upper, &cut, end,
        )?;
        // The lambda=1 channels cancel both crossing ReLUs exactly. The
        // resulting full-domain affine function is constant 5/4-b. For the
        // widened bias interval its independent worst endpoint is 5/4-b-e.
        // These f64 operations are exact dyadic arithmetic, not a CPU CROWN
        // comparison, and include the ACTUAL outward certified intercept.
        let exact_lower = 1.25 - f64::from(certificate.facet().b) - f64::from(error);
        assert!(
            got.is_finite() && f64::from(got) <= exact_lower && got <= 0.25,
            "returned {got} exceeds exact augmented lower {exact_lower}"
        );
        assert!(
            f64::from(got) > exact_lower - 0.001,
            "unexpectedly vacuous cut enclosure"
        );
        assert!(
            got > 0.12,
            "positive exact/error-charged root improvement required"
        );
        lower_results.push(got);
        println!("case=actual-lower-cut lambda=1 source_bias_abs_error={error} lower={got} exact_augmented_lower={exact_lower}");
    }
    assert!(lower_results[0] > 0.24);
    assert!(lower_results[0] > baseline.lower_bounds[0] + 0.99);
    assert!(
        lower_results[1] < lower_results[0] - 0.12,
        "source-error lane must materially affect the returned enclosure"
    );

    // All refusals call the same public real-CUDA object; none accepts a
    // shadow scalar or changes the threshold to reach the implementation.
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    deadline_refusal(device.crown_backward_gpu_resnet_support4_with_deadline(
        &d.prefix,
        &support_seed,
        &d.lower,
        &d.upper,
        expired,
    ));
    let expired_cut = carrier(&certificate, 0.0, WIDTH, expired)?;
    deadline_refusal(device.crown_backward_gpu_resnet_lower_cut_with_deadline(
        &d.full,
        &objective,
        &d.lower,
        &d.upper,
        &expired_cut,
        expired,
    ));

    let mut bad_bias = d.first_bias.to_vec();
    bad_bias[0] = ny_core::CROWN_COEFF_MAX; // finite transported sentinel
    let bad_source = [GpuResnetSegment::Chain(vec![d.first(bad_bias.into())])];
    assert!(matches!(
        device.crown_backward_gpu_resnet_support4_with_deadline(
            &bad_source,
            &support_seed,
            &d.lower,
            &d.upper,
            deadline()
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    let end = deadline();
    let wrong_width = carrier(&certificate, 0.0, WIDTH - 1, end)?;
    assert!(matches!(
        device.crown_backward_gpu_resnet_lower_cut_with_deadline(
            &d.full,
            &objective,
            &d.lower,
            &d.upper,
            &wrong_width,
            end
        ),
        Err(NyError::InvalidSpec(_))
    ));
    let mut asymmetric = support_seed.upper_a.to_vec();
    asymmetric[0] = 0.5;
    let bad_seed = GpuCrownSeed {
        upper_a: asymmetric.into(),
        ..support_seed
    };
    assert!(matches!(
        device.crown_backward_gpu_resnet_support4_with_deadline(
            &d.prefix,
            &bad_seed,
            &d.lower,
            &d.upper,
            deadline()
        ),
        Err(NyError::SoundnessRefusal(_))
    ));
    println!("case=refusals expired_support=true expired_cut=true unsafe_source=true wrong_width=true asymmetric_seed=true");
    drop(bad_source); // release its retained Arc to the original large weight
    drop(d);
    weighted_conformance(&device)?;
    two_pair_conformance(&device)?;
    dual_alpha_conformance(&device)?;
    triplet_conformance(&device)?;
    local_l1_conformance(&device)?;
    two_local_l1_conformance(&device)?;
    nine_cycle_two_local_conformance(&device)?;
    println!("completed=true actual_cuda_required=true completed_public_baseline_calls=1 completed_public_support_calls=1 completed_public_cut_calls=2 expected_refusals=5 benchmark=false private_root_consumer=false loader_provenance=false scope=original_diamond");
    Ok(())
}
