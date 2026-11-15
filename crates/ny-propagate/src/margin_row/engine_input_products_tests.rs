// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Producer-entry controls. The original network is evaluated independently in
//! rational arithmetic; published affine rows retain each direction's own
//! coefficients, bias and full pointwise error.

use super::*;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

fn fixture(fold_error: bool) -> TwinNet {
    TwinNet::compile(&TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.5],
                bias: vec![0.125],
                bias_err: if fold_error { vec![0.001] } else { vec![0.0] },
                weight_rel_err: if fold_error { 0.003 } else { 0.0 },
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 2),
                oshape: (1, 1, 2),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![0.75],
                shift: vec![-0.25],
                scale_rel_err: if fold_error { 0.003 } else { 0.0 },
                shift_err: if fold_error { vec![0.002] } else { vec![0.0] },
                shape: (1, 1, 2),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![1.0, -0.5, 0.25, 0.75],
                bias: vec![0.125, -0.25],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![1.0, -1.0, -0.5, 0.75],
                bias: vec![0.0, 0.125],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    })
    .unwrap()
}

fn make_root(net: &TwinNet, mode: RoundMode) -> RootGates {
    let _source = super::super::head_group_source::HeadGroupSourceGuard::enter(true);
    RootGates::build(net, &[-0.75, -0.5], &[0.75, 1.0], mode, None).unwrap()
}

fn engine<'a>(net: &'a TwinNet, root: &'a RootGates) -> BackwardEngine<'a> {
    BackwardEngine::new(net, root)
        .with_err_lane(ErrLane::PerCol)
        .with_gpu_conv_route(GpuConvRoute::Off)
}

fn identity(targets: &[usize]) -> Seed {
    let mut s = Array2::zeros((2, targets.len()));
    for (col, &target) in targets.iter().enumerate() {
        s[[target, col]] = 1.0;
    }
    Seed { s, e: None }
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn bits_eq(a: &[f64], b: &[f64]) {
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

fn pass_eq(a: &PassOut, b: &PassOut) {
    assert_eq!(a.a.dim(), b.a.dim());
    bits_eq(a.a.as_slice().unwrap(), b.a.as_slice().unwrap());
    bits_eq(
        a.e.as_ref().unwrap().as_slice().unwrap(),
        b.e.as_ref().unwrap().as_slice().unwrap(),
    );
    bits_eq(&a.b, &b.b);
    bits_eq(&a.eb, &b.eb);
    assert!(a.coll.is_none() && b.coll.is_none());
    assert!(a.coll_rows.is_none() && b.coll_rows.is_none());
    assert!(a.unst_rows.is_none() && b.unst_rows.is_none());
}

#[test]
fn engine_input_products_timed_prefix_matches_legacy_signed_folded_rows_in_both_directions() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(true);
    let root = make_root(&net, RoundMode::Outward);
    let eng = engine(&net, &root);
    let seed = Seed {
        s: ndarray::array![[1.0, -0.5, -0.0], [-0.25, 0.75, 0.125]],
        e: Some(ndarray::array![[0.001, 0.0, 0.002], [0.0, 0.003, 0.001]]),
    };
    for op in [1, 4] {
        for dir in [LaneDir::Lower, LaneDir::Upper] {
            let legacy = eng.run_prefix(&seed, op, dir).unwrap();
            let no_override = eng.run_prefix_dom(&seed, op, dir, None).unwrap();
            let timed = eng
                .run_prefix_with_deadline(&seed, op, dir, deadline())
                .unwrap();
            pass_eq(&legacy, &no_override);
            pass_eq(&legacy, &timed);
        }
    }
}

fn rat(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

fn original_second_preactivation(x: f64) -> BigRational {
    let pre = rat(0.5) * rat(x) + rat(0.125);
    let relu = pre.max(BigRational::zero());
    rat(0.75) * &relu - rat(0.25) + relu
}

#[test]
fn engine_input_products_explicit_rows_contain_original_residual_network_on_declared_box() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(false);
    let root = make_root(&net, RoundMode::Outward);
    let eng = engine(&net, &root);
    let targets = [1, 0];
    let seed = identity(&targets);
    let at = deadline();
    let pair = eng
        .run_prefix_input_rows(
            &seed,
            4,
            &targets,
            0,
            super::super::root_box::ROOT_SOURCE_BYTES_CAP,
            at,
        )
        .unwrap();
    pair.validate_scope(&net, &root, 4, &targets).unwrap();
    assert_eq!(pair.columns(), targets.len());
    assert_eq!(pair.deadline(), at);
    assert_eq!(pair.n_in(), net.n_in);
    for x in [-0.75, -0.25, 0.25, 0.75] {
        for y in [-0.5, 0.0, 0.5, 1.0] {
            let input = [x, y];
            for (col, &target) in targets.iter().enumerate() {
                assert_eq!(pair.target(col), Some(target));
                let original = original_second_preactivation(input[target]);
                for upper in [false, true] {
                    let row = if upper {
                        pair.upper(col)
                    } else {
                        pair.lower(col)
                    }
                    .unwrap();
                    let line = row.entries().fold(rat(row.bias()), |sum, (i, a, _)| {
                        sum + rat(a) * rat(input[i])
                    });
                    let eta = rat(row.pointwise_error(at).unwrap());
                    assert!(
                        if upper {
                            line + eta >= original
                        } else {
                            line - eta <= original
                        },
                        "column={col}, target={target}, upper={upper}, input={input:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn engine_input_products_pointwise_error_uses_declared_domain_when_root_xabs_is_zero() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(true);
    let mut root = make_root(&net, RoundMode::Outward);
    root.xabs.fill(0.0);
    let eng = engine(&net, &root);
    let targets = [0, 1];
    let seed = identity(&targets);
    let at = deadline();
    let pair = eng
        .run_prefix_input_rows(
            &seed,
            4,
            &targets,
            0,
            super::super::root_box::ROOT_SOURCE_BYTES_CAP,
            at,
        )
        .unwrap();
    for col in 0..2 {
        for row in [pair.lower(col).unwrap(), pair.upper(col).unwrap()] {
            let exact_charge = row.entries().fold(rat(row.bias_error()), |sum, (i, _, e)| {
                sum + rat(e) * rat(root.lo[i].abs().max(root.hi[i].abs()))
            });
            let published = row.pointwise_error(at).unwrap();
            assert!(
                published > row.bias_error(),
                "coefficient errors on nonzero inputs cannot disappear"
            );
            assert!(rat(published) >= exact_charge);
        }
    }
}

#[test]
fn engine_input_products_refuse_relabelled_seed_error_shape_targets_and_nearest_scope() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(false);
    let root = make_root(&net, RoundMode::Outward);
    let eng = engine(&net, &root);
    let targets = [0, 1];
    let mut wrong = identity(&targets);
    wrong.s[[0, 0]] = 0.5;
    assert!(eng
        .run_prefix_input_rows(&wrong, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
    let mut wrong_error = identity(&targets);
    wrong_error.e = Some(Array2::from_elem((2, 2), 0.01));
    assert!(eng
        .run_prefix_input_rows(&wrong_error, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
    let mut wrong_shape = identity(&targets);
    wrong_shape.e = Some(Array2::zeros((1, 2)));
    assert!(eng
        .run_prefix_input_rows(&wrong_shape, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
    let seed = identity(&targets);
    for bad_targets in [[0, 0], [0, 2]] {
        assert!(eng
            .run_prefix_input_rows(&seed, 4, &bad_targets, 0, usize::MAX, deadline())
            .is_err());
    }
    assert!(eng
        .run_prefix_input_rows(&seed, 3, &targets, 0, usize::MAX, deadline())
        .is_err());
    let nearest_root = make_root(&net, RoundMode::Parity);
    assert!(engine(&net, &nearest_root)
        .run_prefix_input_rows(&seed, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
    let mut missing_source = make_root(&net, RoundMode::Outward);
    missing_source.head_group_source = None;
    assert!(engine(&net, &missing_source)
        .run_prefix_input_rows(&seed, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
    // Every scalar gate value below is finite and the constant upper line is
    // valid on its forged [0,0] interval. Only the completed original prefix
    // generation can establish whether that interval really encloses z(x).
    let mut changed_gates = make_root(&net, RoundMode::Outward);
    let gate = &mut changed_gates.layers[0];
    for values in [
        &mut gate.l,
        &mut gate.u,
        &mut gate.alpha,
        &mut gate.s,
        &mut gate.c,
        &mut gate.ms,
    ] {
        values.fill(0.0);
    }
    assert!(engine(&net, &changed_gates)
        .run_prefix_input_rows(&seed, 4, &targets, 0, usize::MAX, deadline())
        .is_err());
}

#[test]
fn engine_input_products_cap_expiry_and_unqualified_envelope_refuse_without_mutation() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(false);
    let root = make_root(&net, RoundMode::Outward);
    let eng = engine(&net, &root);
    let targets = [0, 1];
    let seed = identity(&targets);
    let seed_before = seed.s.clone();
    let root_before = root.lo.clone();
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        eng.run_prefix_input_rows(&seed, 4, &targets, 0, usize::MAX, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        eng.run_prefix_input_rows(&seed, 4, &targets, 0, 1, deadline()),
        Err(NyError::CpuMemoryExceeded {
            site: "margin_row dense input-row product tile",
            budget_bytes: 1,
            ..
        })
    ));
    let envelope = engine(&net, &root).with_err_lane(ErrLane::PerChannel);
    assert!(matches!(
        envelope.run_prefix_with_deadline(&seed, 4, LaneDir::Lower, deadline()),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    let armed = engine(&net, &root).with_gpu_conv_route(GpuConvRoute::Global);
    assert!(matches!(
        armed.run_prefix_input_rows(&seed, 4, &targets, 0, usize::MAX, deadline()),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_eq!(seed.s, seed_before);
    assert_eq!(root.lo, root_before);
}

#[test]
fn engine_input_products_owner_scope_and_conservative_census_do_not_relabel_other_roots() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(false);
    let root = make_root(&net, RoundMode::Outward);
    let eng = engine(&net, &root);
    let targets = [0, 1];
    let seed = identity(&targets);
    let floor = input_product_resident_floor(&net, &root).unwrap();
    let estimate = eng
        .dense_prefix_input_product_bytes(&seed, 4, floor)
        .unwrap();
    assert!(estimate > floor + seed.s.len() * 8 + 2 * net.n_in * targets.len() * 16);
    assert_eq!(
        eng.dense_prefix_input_product_bytes(&seed, 4, floor + 4096)
            .unwrap(),
        estimate + 4096
    );
    assert!(eng
        .dense_prefix_input_product_bytes(&seed, 4, usize::MAX)
        .is_none());
    assert_eq!(
        input_product_resident_floor_with_deadline(&net, &root, deadline()).unwrap(),
        floor
    );
    let at = deadline();
    let pair = eng
        .run_prefix_input_rows(&seed, 4, &targets, floor, usize::MAX, at)
        .unwrap();
    let same_values_other_owner = make_root(&net, RoundMode::Outward);
    assert!(pair
        .validate_scope(&net, &same_values_other_owner, 4, &targets)
        .is_err());
    assert!(pair.validate_scope(&net, &root, 4, &[1, 0]).is_err());
    assert!(pair.validate_scope(&net, &root, 1, &targets).is_err());
}

#[test]
fn engine_input_products_timed_scratch_reconciles_capacity_and_preserves_exact_source_bits() {
    let _env = crate::tests::lock_env_shared();
    assert!(matches!(
        reconcile_timed_scratch_capacity(2, 3),
        Err(NyError::CpuMemoryExceeded {
            required_bytes: 24,
            budget_bytes: 16,
            site: "margin_row deadline scratch capacity",
        })
    ));
    assert!(reconcile_timed_scratch_capacity(2, usize::MAX).is_err());
    assert!(reconcile_timed_scratch_capacity(usize::MAX, usize::MAX).is_err());
    reconcile_timed_scratch_capacity(6, 6).unwrap();

    let source = ndarray::array![
        [0.0, -0.0, f64::from_bits(1)],
        [-f64::from_bits(1), f64::MIN_POSITIVE, -1.0]
    ];
    let copied = clone_with_deadline(&source, Some(deadline())).unwrap();
    bits_eq(source.as_slice().unwrap(), copied.as_slice().unwrap());
    let (owned, offset) = copied.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    assert_eq!(owned.len(), 6);
    assert_eq!(owned.capacity(), 6);

    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        zeros_with_deadline((usize::MAX, 2), Some(expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(matches!(
        zeros_with_deadline((usize::MAX, 2), Some(deadline())),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert!(matches!(
        zeros_with_deadline((DEADLINE_MAX_CELLS + 1, 1), Some(deadline())),
        Err(NyError::UnsupportedConfiguration(_))
    ));
}

#[test]
fn engine_input_products_dense_request_keeps_completed_builder_deadline() {
    let _env = crate::tests::lock_env_shared();
    let net = fixture(false);
    let original = deadline();
    let _source = super::super::head_group_source::HeadGroupSourceGuard::enter(true);
    let root = RootGates::build_retaining(
        &net,
        &[-0.75, -0.5],
        &[0.75, 1.0],
        RoundMode::Outward,
        Some(original),
        None,
        &[],
    )
    .unwrap()
    .0;
    let targets = [0, 1];
    let seed = identity(&targets);
    let pair = engine(&net, &root)
        .run_prefix_input_rows(
            &seed,
            4,
            &targets,
            0,
            usize::MAX,
            original + Duration::from_secs(600),
        )
        .unwrap();
    assert_eq!(pair.deadline(), original);
}
