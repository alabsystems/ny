// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Ordinary and cancellable per-column passes share the underflow-safe error
//! arithmetic. Completed passes agree bit-for-bit on the tested normal-source
//! functionals. Gradual-underflow cases intentionally widen the legacy relative
//! budget, and independent exact dyadic endpoints test those repairs.

use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::NyError;

use super::{BackwardEngine, LaneDir, PassOut, Seed, DEADLINE_MAX_CELLS};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::gpu_conv::GpuConvRoute;
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

fn fixture() -> TwinNet {
    let spec = TwinSpec {
        n_in: 4,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.75, -0.375],
                bias: vec![0.0625, -0.03125],
                bias_err: vec![0.01, 0.02],
                weight_rel_err: 0.003,
                kernel: (2, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 2, 2),
                oshape: (2, 2, 2),
            }, // t1
            TwinOpSpec::Relu { input: 1 }, // t2
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![0.7, -0.25],
                shift: vec![0.125, 0.07],
                scale_rel_err: 0.003,
                shift_err: vec![0.02, 0.01],
                shape: (2, 2, 2),
            }, // t3
            TwinOpSpec::Conv {
                input: 3,
                weight: vec![0.5, -0.75, 0.0, 0.375],
                bias: vec![0.02, -0.01],
                bias_err: vec![0.03, 0.04],
                weight_rel_err: 0.01,
                kernel: (2, 2, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (2, 2, 2),
                oshape: (2, 2, 2),
            }, // t4
            TwinOpSpec::Add { lhs: 4, rhs: 2 }, // t5, shared branch merge
            TwinOpSpec::Relu { input: 5 }, // t6
            TwinOpSpec::Flatten { input: 6 }, // t7
            TwinOpSpec::Gemm {
                input: 7,
                weight: (0..24).map(|i| ((i % 7) as f64 - 3.0) / 8.0).collect(),
                bias: vec![0.125, -0.0625, 0.3],
                bias_err: vec![0.02, 0.03, 0.04],
                weight_rel_err: 0.02,
                shape: (3, 8),
            }, // t8 = y
            TwinOpSpec::Relu { input: 8 }, // t9
            TwinOpSpec::Gemm {
                input: 9,
                weight: vec![1.0, -0.5, 0.375, -0.75, 0.25, 0.125],
                bias: vec![0.2, -0.1],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 3),
            },
        ],
    };
    TwinNet::compile(&spec).expect("source fixture")
}

fn seed(columns: usize, error: bool) -> Seed {
    let s = Array2::from_shape_fn((3, columns), |(j, c)| {
        let value = ((j * 17 + c * 11) % 19) as f64 - 9.0;
        if value == 0.0 && c % 2 == 1 {
            -0.0
        } else {
            value / 8.0
        }
    });
    let e =
        error.then(|| Array2::from_shape_fn(s.dim(), |(j, c)| ((j * 3 + c) % 7) as f64 / 4096.0));
    Seed { s, e }
}

fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what} shape");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what} element {i}");
    }
}

fn assert_pass_bits(a: &PassOut, b: &PassOut) {
    assert_eq!(a.a.dim(), b.a.dim());
    assert_bits(
        a.a.as_slice().unwrap(),
        b.a.as_slice().unwrap(),
        "coefficients",
    );
    match (&a.e, &b.e) {
        (Some(a), Some(b)) => {
            assert_eq!(a.dim(), b.dim());
            assert_bits(
                a.as_slice().unwrap(),
                b.as_slice().unwrap(),
                "coefficient errors",
            );
        }
        (None, None) => {}
        _ => panic!("error lane mismatch"),
    }
    assert_bits(&a.b, &b.b, "bias");
    assert_bits(&a.eb, &b.eb, "bias errors");
    assert!(a.coll.is_none() && b.coll.is_none());
    assert!(a.coll_rows.is_none() && b.coll_rows.is_none());
    assert!(a.unst_rows.is_none() && b.unst_rows.is_none());
}

#[test]
fn deadline_pass_matches_complete_signed_folded_residual_functionals_bitwise() {
    let net = fixture();
    let root = RootGates::build(&net, &[-0.6; 4], &[0.7; 4], RoundMode::Outward, None)
        .expect("qualified source root");
    let eng = BackwardEngine::new(&net, &root)
        .with_err_lane(ErrLane::PerCol)
        .with_gpu_conv_route(GpuConvRoute::Off);
    for columns in [1, 2, 7, 16, 33, 64] {
        for error in [false, true] {
            let seed = seed(columns, error);
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                let ordinary = eng.run(&seed, None, dir, None, false).expect("ordinary");
                let checked = eng
                    .run_with_deadline(&seed, dir, Instant::now() + Duration::from_secs(30))
                    .expect("completed cancellable pass");
                assert_pass_bits(&ordinary, &checked);
                // Identical input concretization remains the exact same bound.
                assert_bits(
                    &eng.concretize(&ordinary, dir == LaneDir::Lower),
                    &eng.concretize(&checked, dir == LaneDir::Lower),
                    "bounds",
                );
            }
        }
    }
}

#[test]
fn deadline_pass_returns_no_rows_after_expiry_and_source_is_reusable() {
    let net = fixture();
    let root = RootGates::build(&net, &[-0.6; 4], &[0.7; 4], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
    let seed = seed(16, true);
    let before = eng.run(&seed, None, LaneDir::Lower, None, false).unwrap();
    let refused = eng.run_with_deadline(&seed, LaneDir::Lower, Instant::now());
    assert!(matches!(refused, Err(NyError::DeadlineExceeded(_))));
    let after = eng.run(&seed, None, LaneDir::Lower, None, false).unwrap();
    assert_pass_bits(&before, &after);
}

#[test]
fn deadline_pass_refuses_uncancellable_lanes_and_malformed_seed_errors() {
    let net = fixture();
    let root = RootGates::build(&net, &[-0.6; 4], &[0.7; 4], RoundMode::Outward, None).unwrap();
    let future = Instant::now() + Duration::from_secs(30);
    for lane in [ErrLane::PerChannel, ErrLane::Envelope] {
        let eng = BackwardEngine::new(&net, &root)
            .with_err_lane(lane)
            .with_err_min_cols(1);
        assert!(matches!(
            eng.run_with_deadline(&seed(4, true), LaneDir::Lower, future),
            Err(NyError::UnsupportedConfiguration(_))
        ));
    }
    let eng = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
    assert!(matches!(
        eng.run_with_deadline(&seed(65, false), LaneDir::Lower, future),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    let mut wrong = seed(4, true);
    wrong.e.as_mut().unwrap()[[0, 0]] = -0.125;
    assert!(matches!(
        eng.run_with_deadline(&wrong, LaneDir::Lower, future),
        Err(NyError::InvalidSpec(_))
    ));
    wrong.e = Some(Array2::zeros((3, 2)));
    assert!(matches!(
        eng.run_with_deadline(&wrong, LaneDir::Lower, future),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    let mut nonfinite = seed(4, false);
    nonfinite.s[[0, 1]] = f64::NAN;
    assert!(matches!(
        eng.run_with_deadline(&nonfinite, LaneDir::Lower, future),
        Err(NyError::NumericalInstability(_))
    ));
}

#[test]
fn deadline_pass_refuses_oversized_scratch_before_tensor_arithmetic() {
    let mut net = fixture();
    let root = RootGates::build(&net, &[-0.6; 4], &[0.7; 4], RoundMode::Outward, None).unwrap();
    net.tsize[1] = DEADLINE_MAX_CELLS / 4 + 1;
    let eng = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
    assert!(matches!(
        eng.run_with_deadline(
            &seed(4, false),
            LaneDir::Lower,
            Instant::now() + Duration::from_secs(30)
        ),
        Err(NyError::UnsupportedConfiguration(_))
    ));
}

fn subnormal_amplified_net(weight: f64) -> TwinNet {
    let amplification = 2.0f64.powi(256);
    TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![amplification],
                shift: vec![0.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0],
                shape: (1, 1, 1),
            },
            TwinOpSpec::Flatten { input: 3 },
            TwinOpSpec::Gemm {
                input: 4,
                weight: vec![weight; 4],
                bias: vec![0.0; 4],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (4, 1),
            },
            TwinOpSpec::Relu { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![0.0; 8],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 4),
            },
        ],
    })
    .expect("amplified finite source")
}

#[test]
fn deadline_pass_charges_each_lost_nominal_product_before_source_amplification() {
    let eta = f64::from_bits(1);
    let net = subnormal_amplified_net(eta);
    let root = RootGates::build(&net, &[0.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
    let seed = Seed {
        s: Array2::from_elem((4, 1), -3.0 / 8.0),
        e: None,
    };
    let pass = eng
        .run_with_deadline(
            &seed,
            LaneDir::Lower,
            Instant::now() + Duration::from_secs(30),
        )
        .expect("finite complete rows");
    // eta * 2^256 is exactly 2^-818; -1.5 * 2^-818 is representable.
    let ordinary = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("corrected source rows");
    assert_pass_bits(&ordinary, &pass);
    let exact = -1.5 * (eta * 2.0f64.powi(256));
    assert!(exact < 0.0);
    assert!(pass.e.as_ref().unwrap()[[0, 0]] >= exact.abs());
    assert!(
        eng.concretize_lower(&pass)[0] <= exact,
        "the exact negative functional must remain enclosed"
    );
}

#[test]
fn deadline_pass_repairs_lost_positive_error_products_before_source_amplification() {
    let eta = f64::from_bits(1);
    let net = subnormal_amplified_net(3.0 / 8.0);
    let root = RootGates::build(&net, &[0.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
    let seed = Seed {
        s: Array2::zeros((4, 1)),
        e: Some(Array2::from_elem((4, 1), eta)),
    };
    let pass = eng
        .run_with_deadline(
            &seed,
            LaneDir::Lower,
            Instant::now() + Duration::from_secs(30),
        )
        .expect("nonzero uncertainty on zero rows");
    let ordinary = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("corrected source rows");
    assert_pass_bits(&ordinary, &pass);
    let exact_radius = 1.5 * (eta * 2.0f64.powi(256));
    assert!(
        pass.e.as_ref().unwrap()[[0, 0]] >= exact_radius,
        "four individually underflowing error terms must remain enclosed"
    );
    assert!(eng.concretize_lower(&pass)[0] <= -exact_radius);
}

#[test]
fn exception_bias_error_prices_rounding_on_the_updated_running_bias() {
    // Exact dyadics: 1 + 2^-54 rounds back to 1, losing the entire new term.
    // A term-magnitude-only bound 4u*2^-54 is much smaller than that loss.
    let bpart = 2.0f64.powi(-54);
    let updated = 1.0 + bpart;
    assert_eq!(updated, 1.0);
    let old_term_charge = 4.0 * super::UNIT * bpart;
    assert!(old_term_charge < bpart);
    let corrected = super::exception_bias_error(0.0, updated, 0.0, 1.0, 1.0, bpart, bpart, true);
    assert!(
        corrected >= bpart,
        "the exact lost dyadic term must be covered by the running-bias carry"
    );
}

#[test]
fn envelope_gate_dominates_repaired_subnormal_error_on_zero_nominal_rows() {
    use crate::margin_row::errlane::{env_gate_update, EnvErr};
    let eta = f64::from_bits(1);
    let mut env = EnvErr {
        v: vec![eta],
        grp: 1,
    };
    env_gate_update(&mut env, &[0.0], &[3.0 / 8.0]);
    // The exact positive error product is 3eta/8; nearest rounding loses it.
    // UP first retains eta, and the established slack adds another eta.
    let per_column = super::slack16(super::underflow_product_up(eta, 3.0 / 8.0));
    assert_eq!(per_column.to_bits(), 2);
    assert!(env.v[0] >= per_column);
}

#[test]
fn envelope_nominal_dust_majorant_covers_smaller_columns_with_normal_max_product() {
    use crate::margin_row::errlane::env_nominal_dust_up;
    let eta = f64::from_bits(1);
    let slope = 3.0 / 8.0;
    assert_eq!(super::nominal_product_dust(1.0, slope), 0.0);
    assert_eq!(super::nominal_product_dust(eta, slope), eta);
    let repaired_column = super::add_nominal_dust(0.0, eta, slope, true);
    assert!(env_nominal_dust_up(0.0, 1.0, slope) >= repaired_column);
    assert_eq!(env_nominal_dust_up(0.0, 0.0, slope), 0.0);
    assert_eq!(env_nominal_dust_up(0.0, 1.0, 0.0), 0.0);
}

#[test]
fn envelope_passes_dominate_amplified_underflow_without_changing_nominal_rows() {
    let eta = f64::from_bits(1);
    let amplification = 2.0f64.powi(256);
    for nominal_loss in [false, true] {
        let net = subnormal_amplified_net(if nominal_loss { eta } else { 3.0 / 8.0 });
        let root = RootGates::build(&net, &[0.0], &[1.0], RoundMode::Outward, None).unwrap();
        let col = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::PerCol);
        let seed = Seed {
            s: if nominal_loss {
                Array2::from_elem((4, 16), -3.0 / 8.0)
            } else {
                Array2::zeros((4, 16))
            },
            e: (!nominal_loss).then(|| Array2::from_elem((4, 16), eta)),
        };
        for dir in [LaneDir::Lower, LaneDir::Upper] {
            let per_column = col.run(&seed, None, dir, None, false).unwrap();
            for lane in [ErrLane::Envelope, ErrLane::PerChannel] {
                let env = BackwardEngine::new(&net, &root).with_err_lane(lane);
                let pass = env.run(&seed, None, dir, None, false).unwrap();
                assert_bits(
                    pass.a.as_slice().unwrap(),
                    per_column.a.as_slice().unwrap(),
                    "nominal envelope rows",
                );
                assert_bits(&pass.b, &per_column.b, "nominal envelope bias");
                for (&e, &c) in pass
                    .e
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(per_column.e.as_ref().unwrap())
                {
                    assert!(
                        e >= c,
                        "{lane:?}/{dir:?}: envelope must dominate repaired error"
                    );
                    assert!(
                        e >= 1.5 * (eta * amplification),
                        "exact dyadic uncertainty must survive amplification"
                    );
                }
                for (&e, &c) in pass.eb.iter().zip(&per_column.eb) {
                    assert!(e >= c, "{lane:?}/{dir:?}: envelope bias must dominate");
                }
                if nominal_loss && matches!(dir, LaneDir::Lower) {
                    assert!(env.concretize_lower(&pass)[0] <= -1.5 * (eta * amplification));
                }
            }
        }
    }
}
