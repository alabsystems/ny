// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Independent original-coordinate controls for the first borrowed products.
//! Rational evaluation uses authored weights and explicit image coordinates,
//! never a production convolution, scatter, gate or concretization routine.

use super::*;
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::patch::PatchPlan;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::time::Duration;

const KERNEL: [f64; 9] = [0.125, -0.25, 0.125, 0.5, -0.75, 0.25, 0.0, 0.125, -0.125];

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn head(input: usize, rows: usize) -> Vec<TwinOpSpec> {
    let mut weight = vec![0.0; 2 * rows];
    weight[0] = 1.0;
    weight[2 * rows - 1] = -1.0;
    vec![
        TwinOpSpec::Flatten { input },
        TwinOpSpec::Gemm {
            input: input + 1,
            weight,
            bias: vec![0.0, 0.125],
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (2, rows),
        },
        TwinOpSpec::Relu { input: input + 2 },
        TwinOpSpec::Gemm {
            input: input + 3,
            weight: vec![1.0, -1.0, -0.5, 0.75],
            bias: vec![0.0, 0.0],
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (2, 2),
        },
    ]
}

fn residual_spec(folded: bool) -> TwinSpec {
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: KERNEL.to_vec(),
            bias: vec![0.125],
            bias_err: vec![if folded { 0.001 } else { 0.0 }],
            weight_rel_err: if folded { 0.003 } else { 0.0 },
            kernel: (1, 1, 3, 3),
            stride: (1, 1),
            pads: (1, 1, 1, 1),
            ishape: (1, 5, 5),
            oshape: (1, 5, 5),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![0.75],
            shift: vec![-0.25],
            scale_rel_err: if folded { 0.002 } else { 0.0 },
            shift_err: vec![if folded { 0.001 } else { 0.0 }],
            shape: (1, 5, 5),
        },
        TwinOpSpec::Add { lhs: 3, rhs: 2 },
        TwinOpSpec::Relu { input: 4 },
    ];
    ops.extend(head(5, 25));
    TwinSpec { n_in: 25, ops }
}

fn residual_fixture(folded: bool) -> TwinNet {
    TwinNet::compile(&residual_spec(folded)).unwrap()
}

fn root(net: &TwinNet, lo: &[f64], hi: &[f64], at: Option<Instant>) -> RootGates {
    let _authority = HeadGroupSourceGuard::enter(true);
    RootGates::build_prec_store(net, lo, hi, RoundMode::Outward, at, false, false).unwrap()
}

fn products<'a>(
    plan: &PatchPlan<'_>,
    net: &'a TwinNet,
    root: &'a RootGates,
    targets: &'a [usize],
    at: Instant,
) -> crate::margin_row::patch::PatchProducts<'a> {
    let floor = plan.product_resident_floor(root, at).unwrap();
    let required = plan
        .product_required_bytes(floor, 2 * targets.len())
        .unwrap();
    plan.prefix_products(net, root, targets, at, floor, required)
        .unwrap()
        .unwrap()
}

fn original_residual_preactivation(input: &[f64], target: usize) -> Q {
    let (y, x) = (target / 5, target % 5);
    let mut pre = q(0.125);
    for ky in 0..3 {
        for kx in 0..3 {
            let iy = y as isize + ky as isize - 1;
            let ix = x as isize + kx as isize - 1;
            if (0..5).contains(&iy) && (0..5).contains(&ix) {
                pre += q(KERNEL[3 * ky + kx]) * q(input[iy as usize * 5 + ix as usize]);
            }
        }
    }
    let activation = pre.max(Q::zero());
    q(0.75) * &activation - q(0.25) + activation
}

fn check_line(row: InputRowView<'_>, input: &[f64], original: &Q, upper: bool, at: Instant) {
    let line = row.entries().fold(q(row.bias()), |sum, (index, a, _)| {
        sum + q(a) * q(input[index])
    });
    let eta = q(row.pointwise_error(at).unwrap());
    assert!(if upper {
        line + eta >= *original
    } else {
        line - eta <= *original
    });
}

#[test]
fn input_rows_products_patch_original_coordinates_residual_border_and_rational_containment() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let lo = vec![-0.75; 25];
    let hi = vec![1.0; 25];
    let root = root(&net, &lo, &hi, None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [24, 0, 12];
    let at = deadline();
    let out = products(&plan, &net, &root, &targets, at);
    out.rows.validate_scope(&net, &root, 4, &targets).unwrap();
    assert_eq!(out.rows.deadline(), at);
    assert_eq!(out.rows.n_in(), 25);
    for (col, &target) in targets.iter().enumerate() {
        assert_eq!(out.rows.target(col), Some(target));
        let (y, x) = (target / 5, target % 5);
        let expected: Vec<_> = (0usize..25)
            .filter(|&i| (i / 5).abs_diff(y) <= 1 && (i % 5).abs_diff(x) <= 1)
            .collect();
        assert_eq!(
            out.rows
                .lower(col)
                .unwrap()
                .entries()
                .map(|(i, _, _)| i)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            out.rows
                .upper(col)
                .unwrap()
                .entries()
                .map(|(i, _, _)| i)
                .collect::<Vec<_>>(),
            expected
        );
    }
    // Vertices, asymmetric interior patterns and one-coordinate changes are
    // independent original input assignments, including negative inputs.
    for pattern in 0..29 {
        let input: Vec<_> = (0..25)
            .map(|i| match pattern {
                0 => lo[i],
                1 => hi[i],
                2 => {
                    if i % 2 == 0 {
                        lo[i]
                    } else {
                        hi[i]
                    }
                }
                3 => {
                    if i % 3 == 0 {
                        -0.25
                    } else {
                        0.5
                    }
                }
                _ => {
                    if i == pattern - 4 {
                        hi[i]
                    } else {
                        lo[i]
                    }
                }
            })
            .collect();
        for (col, &target) in targets.iter().enumerate() {
            let original = original_residual_preactivation(&input, target);
            check_line(out.rows.lower(col).unwrap(), &input, &original, false, at);
            check_line(out.rows.upper(col).unwrap(), &input, &original, true, at);
            assert!(q(out.bounds.lb[col]) <= original && q(out.bounds.ub[col]) >= original);
        }
    }
}

#[test]
fn input_rows_products_patch_full_error_charge_and_zero_domain_magnitude_cache() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(true);
    let mut root = root(&net, &[-0.75; 25], &[1.0; 25], None);
    root.xabs.fill(0.0);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0, 12, 24];
    let at = deadline();
    let out = products(&plan, &net, &root, &targets, at);
    for col in 0..targets.len() {
        for row in [out.rows.lower(col).unwrap(), out.rows.upper(col).unwrap()] {
            let exact = row.entries().fold(q(row.bias_error()), |sum, (i, _, e)| {
                sum + q(e) * q(root.lo[i].abs().max(root.hi[i].abs()))
            });
            assert!(q(row.pointwise_error(at).unwrap()) >= exact);
            assert!(row.pointwise_error(at).unwrap() > row.bias_error());
        }
        let original = original_residual_preactivation(&[1.0; 25], targets[col]);
        assert!(q(out.bounds.lb[col]) <= original && q(out.bounds.ub[col]) >= original);
    }
}

fn pointwise_fixture(underflow: bool) -> TwinNet {
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![if underflow { 1e300 } else { 1.0 }],
            bias: vec![if underflow { 0.0 } else { 1.5 }],
            bias_err: vec![0.0],
            weight_rel_err: 0.0,
            kernel: (1, 1, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (1, 5, 5),
            oshape: (1, 5, 5),
        },
        TwinOpSpec::Relu { input: 1 },
    ];
    if underflow {
        ops.push(TwinOpSpec::Conv {
            input: 2,
            weight: vec![f64::from_bits(1)],
            bias: vec![0.0],
            bias_err: vec![0.0],
            weight_rel_err: 0.0,
            kernel: (1, 1, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (1, 5, 5),
            oshape: (1, 5, 5),
        });
    }
    let current = ops.len();
    ops.push(TwinOpSpec::ChannelAffine {
        input: current,
        scale: vec![if underflow { 0.5 } else { 1.0 }],
        shift: vec![0.0],
        scale_rel_err: 0.0,
        shift_err: vec![0.0],
        shape: (1, 5, 5),
    });
    let current = ops.len();
    ops.push(TwinOpSpec::Relu { input: current });
    ops.extend(head(ops.len(), 25));
    TwinNet::compile(&TwinSpec { n_in: 25, ops }).unwrap()
}

#[test]
fn input_rows_products_independent_constants_preserve_upper_bias_below_lower_bias() {
    let _env = crate::tests::lock_env_shared();
    let net = pointwise_fixture(false);
    let root = root(&net, &[-2.0; 25], &[-1.0; 25], None);
    let plan = PatchPlan::build(&net, 3, 0.5).unwrap();
    let targets = [0, 24];
    let at = deadline();
    let out = products(&plan, &net, &root, &targets, at);
    for col in 0..2 {
        let lower = out.rows.lower(col).unwrap();
        let upper = out.rows.upper(col).unwrap();
        assert!(upper.bias() < lower.bias());
        for x in [-2.0, -1.75, -1.5, -1.25, -1.0] {
            let original = (q(x) + q(1.5)).max(Q::zero());
            check_line(out.rows.lower(col).unwrap(), &[x; 25], &original, false, at);
            check_line(out.rows.upper(col).unwrap(), &[x; 25], &original, true, at);
        }
    }
}

#[test]
fn input_rows_products_nominal_subnormal_dust_survives_later_large_original_weight() {
    let _env = crate::tests::lock_env_shared();
    let net = pointwise_fixture(true);
    let root = root(&net, &[-1.0; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0, 24];
    let at = deadline();
    let out = products(&plan, &net, &root, &targets, at);
    for col in 0..2 {
        assert!(out
            .rows
            .lower(col)
            .unwrap()
            .entries()
            .any(|(_, _, e)| e > 0.0));
        for x in [-1.0, 0.0, 0.5, 1.0] {
            let original = (q(1e300) * q(x)).max(Q::zero()) * q(f64::from_bits(1)) * q(0.5);
            check_line(out.rows.lower(col).unwrap(), &[x; 25], &original, false, at);
            check_line(out.rows.upper(col).unwrap(), &[x; 25], &original, true, at);
            assert!(q(out.bounds.lb[col]) <= original && q(out.bounds.ub[col]) >= original);
        }
    }
}

#[test]
fn input_rows_products_refuse_missing_baked_stale_gate_model_target_and_shape_authority() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let root = root(&net, &[-0.75; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0, 12];
    let at = deadline();
    let out = products(&plan, &net, &root, &targets, at);
    let other_root = self::root(&net, &[-0.75; 25], &[1.0; 25], None);
    assert!(out
        .rows
        .validate_scope(&net, &other_root, 4, &targets)
        .is_err());
    assert!(out.rows.validate_scope(&net, &root, 4, &[12, 0]).is_err());
    assert!(out.rows.validate_scope(&net, &root, 1, &targets).is_err());
    let missing = {
        let mut copy = self::root(&net, &[-0.75; 25], &[1.0; 25], None);
        copy.head_group_source = None;
        copy
    };
    assert!(InputRowScope::new(&net, &missing, 4, &targets, at).is_err());
    let mut changed = self::root(&net, &[-0.75; 25], &[1.0; 25], None);
    changed.layers[0].l.fill(0.0);
    changed.layers[0].u.fill(0.0);
    // Plausible scalar endpoints still carry no authority to replace the
    // actual enclosures captured at completed original construction.
    assert!(InputRowScope::new(&net, &changed, 4, &targets, at).is_err());
    let mut changed = self::root(&net, &[-0.75; 25], &[1.0; 25], None);
    changed.layers[0].alpha[0] = if changed.layers[0].alpha[0] == 0.0 {
        1.0
    } else {
        0.0
    };
    assert!(InputRowScope::new(&net, &changed, 4, &targets, at).is_err());
    let mut changed = self::root(&net, &[-0.75; 25], &[1.0; 25], None);
    changed.layers[0].s.pop();
    assert!(InputRowScope::new(&net, &changed, 4, &targets, at).is_err());
    for targets in [&[0, 0][..], &[25][..], &[][..]] {
        assert!(InputRowScope::new(&net, &root, 4, targets, at).is_err());
    }
    let cloned_net = residual_fixture(false);
    assert!(plan
        .prefix_products(&cloned_net, &root, &targets, at, usize::MAX, 1 << 24)
        .is_none());
    let mut stale_net = residual_fixture(false);
    let stale_root = self::root(&stale_net, &[-0.75; 25], &[1.0; 25], None);
    let TwinOp::Conv(conv) = &mut stale_net.ops[0] else {
        panic!("original fixture conv")
    };
    conv.wmat[0] = 0.25;
    assert!(InputRowScope::new(&stale_net, &stale_root, 4, &targets, at).is_err());
    let mut wrong_plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    wrong_plan.geo.frames[0].as_mut().unwrap().ox += 1;
    assert!(wrong_plan
        .prefix_products(&net, &root, &targets, at, usize::MAX, 1 << 24)
        .is_none());
    let _source = HeadGroupSourceGuard::enter(true);
    let baked = RootGates::build_retaining(
        &net,
        &[-0.75; 25],
        &[1.0; 25],
        RoundMode::Outward,
        None,
        None,
        &[(0, 0, 1)],
    )
    .unwrap()
    .0;
    assert!(InputRowScope::new(&net, &baked, 4, &targets, at).is_err());
}

#[test]
fn input_rows_products_inherited_original_deadline_cannot_be_renewed() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let original = deadline();
    let root = root(&net, &[-0.75; 25], &[1.0; 25], Some(original));
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0];
    let out = products(
        &plan,
        &net,
        &root,
        &targets,
        original + Duration::from_secs(600),
    );
    assert_eq!(out.rows.deadline(), original);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(out.rows.lower(0).unwrap().pointwise_error(expired).is_err());
    let floor = plan.product_resident_floor(&root, original).unwrap();
    assert!(matches!(
        plan.prefix_products(&net, &root, &targets, expired, floor, 1 << 24),
        Some(Err(NyError::DeadlineExceeded(_)))
    ));
}

#[test]
fn input_rows_products_total_live_floor_and_exact_local_cap_refuse_before_walk() {
    let _env = crate::tests::lock_env_shared();
    let net = residual_fixture(false);
    let root = root(&net, &[-0.75; 25], &[1.0; 25], None);
    let plan = PatchPlan::build(&net, 4, 0.5).unwrap();
    let targets = [0, 12];
    let at = deadline();
    let floor = plan.product_resident_floor(&root, at).unwrap();
    let required = plan.product_required_bytes(floor, 4).unwrap();
    assert!(matches!(
        plan.prefix_products(&net, &root, &targets, at, floor, required - 1),
        Some(Err(NyError::CpuMemoryExceeded { .. }))
    ));
    assert!(matches!(
        plan.prefix_products(&net, &root, &targets, at, floor - 1, required),
        Some(Err(NyError::CpuMemoryExceeded { .. }))
    ));
    assert!(matches!(
        plan.prefix_products(&net, &root, &targets, at, floor, usize::MAX),
        Some(Err(NyError::CpuMemoryExceeded { .. }))
    ));
    assert!(plan
        .prefix_products(&net, &root, &targets, at, floor, required)
        .unwrap()
        .is_ok());
}

#[test]
fn input_rows_products_zero_signed_zero_error_lanes_and_shape_nonfinite_refusals() {
    let at = deadline();
    // These raw data fixtures exercise validation only. They do not possess
    // a producer token and cannot enter the sealed production constructors.
    let valid = Rows {
        a: Coefficients::F64(ContiguousMatrix::from_array(ndarray::arr2(&[[0.0, -0.0]])).unwrap()),
        e: ContiguousMatrix::from_array(ndarray::arr2(&[[-0.0, 0.0]])).unwrap(),
        b: Biases::F64(vec![0.0, -0.0]),
        eb: vec![0.0, -0.0],
        matrix_bytes: 0,
    };
    validate_rows(&valid, 1, 2, at).unwrap();
    let invalid_shape = Rows {
        e: ContiguousMatrix::from_array(Array2::zeros((2, 2))).unwrap(),
        ..valid
    };
    assert!(validate_rows(&invalid_shape, 1, 2, at).is_err());
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let rows = Rows {
            a: Coefficients::F64(ContiguousMatrix::from_array(Array2::zeros((1, 1))).unwrap()),
            e: ContiguousMatrix::from_array(ndarray::arr2(&[[bad]])).unwrap(),
            b: Biases::F64(vec![0.0]),
            eb: vec![0.0],
            matrix_bytes: 0,
        };
        assert!(validate_rows(&rows, 1, 1, at).is_err());
    }
    let rows = Rows {
        a: Coefficients::F64(
            ContiguousMatrix::from_array(ndarray::arr2(&[[f64::INFINITY]])).unwrap(),
        ),
        e: ContiguousMatrix::from_array(Array2::zeros((1, 1))).unwrap(),
        b: Biases::F64(vec![0.0]),
        eb: vec![0.0],
        matrix_bytes: 0,
    };
    assert!(validate_rows(&rows, 1, 1, at).is_err());
}

#[test]
fn input_rows_products_original_mapping_padding_skips_and_overflow_refusals() {
    let frame = Frame {
        c: 2,
        h: 5,
        w: 5,
        sy: 1,
        sx: 1,
        oy: -1,
        ox: -1,
        ph: 3,
        pw: 3,
        full: false,
    };
    for pos in [(0, 0), (2, 2), (4, 4)] {
        let mut expected_rows = Vec::new();
        for row in 0..18 {
            let channel = row / 9;
            let y = pos.0 as isize - 1 + (row % 9 / 3) as isize;
            let x = pos.1 as isize - 1 + (row % 3) as isize;
            let expected = if (0..5).contains(&y) && (0..5).contains(&x) {
                expected_rows.push(row);
                Some(channel * 25 + y as usize * 5 + x as usize)
            } else {
                None
            };
            assert_eq!(input_index(&frame, pos, row).unwrap(), expected);
        }
        for start in 0..=18 {
            assert_eq!(
                next_real_row(&frame, pos, start),
                expected_rows.iter().copied().find(|&row| row >= start)
            );
        }
    }
    assert!(input_index(&frame, (usize::MAX, 0), 0).is_err());
    let frame = Frame {
        oy: isize::MIN,
        ..frame
    };
    assert!(input_index(&frame, (0, 0), 17).unwrap().is_none());
}

#[test]
fn input_rows_products_sealed_transfer_preserves_allocation_and_rejects_unpaid_origin_capacity() {
    let matrix = ndarray::arr2(&[[1.25, -0.0], [0.0, -2.5]]);
    let pointer = matrix.as_ptr();
    let (sealed, bytes) = seal_matrix(matrix, deadline()).unwrap();
    assert_eq!(sealed.as_ptr(), pointer);
    assert_eq!(bytes, 4 * size_of::<f64>());
    assert_eq!(sealed[[0, 1]].to_bits(), (-0.0f64).to_bits());
    let mut values = Vec::with_capacity(64);
    values.extend([0.0, 1.0]);
    let excess = Array2::from_shape_vec((1, 2), values).unwrap();
    assert!(matches!(
        seal_matrix(excess, deadline()),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    let shifted = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]).slice_move(ndarray::s![1.., ..]);
    assert!(seal_matrix(shifted, deadline()).is_err());
    let mut transposed = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    transposed.swap_axes(0, 1);
    assert!(seal_matrix(transposed, deadline()).is_err());
}

#[path = "input_rows_storage_tests.rs"]
mod storage;

mod owned {
    use super::*;
    use crate::certify::residency::{CancellationToken, MemoryAuthority};
    use crate::certify::work::{WorkLimits, WorkReserve};
    use crate::margin_row::support_program::{OriginalModel, OriginalRoot};
    use crate::margin_row::{ErrLane, RootPrecisionRuntimePolicy};

    const CAP: usize = 128 << 20;

    fn ledger() -> (WorkBudget, CancellationToken, MemoryAuthority) {
        let memory = MemoryAuthority::new(2 * CAP);
        let cancel = CancellationToken::default();
        let budget = WorkBudget::new_in(
            &memory,
            &cancel,
            WorkLimits {
                deadline: None,
                scratch_bytes: CAP,
                persistent_bytes: CAP,
                max_work: 1_000_000_000,
                max_proposals: 0,
                certificate_reserve: Duration::ZERO,
                reserve: WorkReserve::default(),
            },
        )
        .unwrap();
        (budget, cancel, memory)
    }

    fn prefix(budget: &WorkBudget, at: Option<Instant>) -> PreparedPrefix {
        let model = OriginalModel::compile(residual_spec(false), budget, at).unwrap();
        let root = OriginalRoot::new(
            model,
            &[-0.75; 25],
            &[1.0; 25],
            RootPrecisionRuntimePolicy {
                err_lane: ErrLane::PerCol,
                ..RootPrecisionRuntimePolicy::default()
            },
            at,
        )
        .unwrap();
        let input = root.input_box().unwrap();
        let pre = root.derive(0, &input, None).unwrap();
        let empty = root.empty_prefix().unwrap();
        empty.commit_relu(&pre, 1, None).unwrap().0
    }

    fn rows(prefix: &PreparedPrefix, targets: &[usize], at: Instant) -> InputRowPair<'static> {
        crate::margin_row::engine::run_prepared_prefix_input_rows(prefix, targets, Some(at))
            .unwrap()
    }

    #[test]
    fn input_rows_owned_scope_pays_target_copy_and_retains_exact_original_owner_until_last_drop() {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, memory) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let before = budget.receipt().unwrap().live_persistent_bytes;
        let mut targets = vec![24, 0, 12];
        let scope = InputRowScope::from_prepared(
            &prefix,
            OriginalRowDemand {
                relu_op: 4,
                targets: &targets,
            },
            at + Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(scope.deadline(), at);
        assert!(scope.prepared_prefix().unwrap().same_prefix(&prefix));
        assert!(scope.borrowed_root().is_none());
        assert!(scope.budget().unwrap().shares_ledger(&budget));
        assert_eq!(scope.lo(), &[-0.75; 25]);
        assert_eq!(scope.hi(), &[1.0; 25]);
        assert!(budget.receipt().unwrap().live_persistent_bytes > before);
        targets.fill(3);
        drop(targets);
        drop(prefix);
        assert_eq!(scope.targets(), &[24, 0, 12]);
        assert_eq!(scope.net().n_in, 25);
        scope.checkpoint(at).unwrap();
        drop(scope);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }

    #[test]
    fn input_rows_owned_scope_invalid_original_ids_operations_and_expiry_release_target_claims() {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, _) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let baseline = budget.receipt().unwrap().live_persistent_bytes;
        for (op, targets) in [
            (4, vec![]),
            (4, vec![1, 1]),
            (4, vec![25]),
            (1, vec![0]),
            (5, vec![0]),
        ] {
            assert!(InputRowScope::from_prepared(
                &prefix,
                OriginalRowDemand {
                    relu_op: op,
                    targets: &targets
                },
                at
            )
            .is_err());
            assert_eq!(budget.receipt().unwrap().live_persistent_bytes, baseline);
        }
        assert!(InputRowScope::from_prepared(
            &prefix,
            OriginalRowDemand {
                relu_op: 4,
                targets: &[0],
            },
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("test clock supports an expired deadline")
        )
        .is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, baseline);
    }

    #[test]
    fn input_rows_owned_dense_persistent_original_coordinate_lines_contain_independent_rational_members(
    ) {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, _) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let targets = vec![24, 0, 12];
        let pair = rows(&prefix, &targets, at);
        pair.validate_prepared_prefix(&prefix, 4, &targets).unwrap();
        assert!(
            pair.resident_bytes().unwrap()
                >= pair.owned_payload_bytes().unwrap() + INPUT_ROWS_OWNER_BYTES
        );
        assert!(pair.borrowed_root().is_none());
        drop(targets);
        drop(prefix);
        for pattern in 0..5 {
            let input: Vec<_> = (0..25)
                .map(|i| match pattern {
                    0 => -0.75,
                    1 => 1.0,
                    2 if i % 2 == 0 => -0.75,
                    3 if i % 3 == 0 => -0.25,
                    _ => 0.5,
                })
                .collect();
            for col in 0..pair.columns() {
                let original = original_residual_preactivation(&input, pair.target(col).unwrap());
                check_line(pair.lower(col).unwrap(), &input, &original, false, at);
                check_line(pair.upper(col).unwrap(), &input, &original, true, at);
            }
        }
    }

    #[test]
    fn input_rows_owned_stale_prefix_generation_foreign_root_and_changed_target_order_refuse() {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, _) = ledger();
        let at = deadline();
        let one = prefix(&budget, Some(at));
        let targets = [24, 0, 12];
        let pair = rows(&one, &targets, at);
        assert!(pair
            .validate_prepared_prefix(&one, 4, &[0, 24, 12])
            .is_err());
        assert!(pair.validate_prepared_prefix(&one, 1, &targets).is_err());
        let foreign = prefix(&budget, Some(at));
        assert!(pair
            .validate_prepared_prefix(&foreign, 4, &targets)
            .is_err());
        let root = one.root();
        let input = root.input_box().unwrap();
        let pre = root.derive(0, &input, None).unwrap();
        let relu = root.derive(1, &pre, None).unwrap();
        let affine = root.derive(2, &relu, None).unwrap();
        let add = root.derive(3, &affine, Some(&relu)).unwrap();
        let (two, _) = one.commit_relu(&add, 4, None).unwrap();
        assert!(pair.validate_prepared_prefix(&two, 4, &targets).is_err());
        pair.validate_prepared_prefix(&one, 4, &targets).unwrap();
    }

    #[test]
    fn input_rows_owned_actual_output_claim_rejects_missing_foreign_equal_limits_and_underfunded_owners(
    ) {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, _) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let mut pair = rows(&prefix, &[0, 12], at);
        // Adversarial private-field mutations exist only in this control;
        // the genuine claim remains alive throughout every refusal.
        let genuine = pair.resident.take().unwrap();
        assert!(pair.validate_prepared_prefix(&prefix, 4, &[0, 12]).is_err());
        let (foreign_budget, _, _) = ledger();
        let foreign = foreign_budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: genuine.bytes(),
                    ..WorkRequest::default()
                },
            )
            .unwrap()
            .commit_persistent()
            .unwrap();
        pair.resident = Some(foreign);
        assert!(pair.validate_prepared_prefix(&prefix, 4, &[0, 12]).is_err());
        let short = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: INPUT_ROWS_OWNER_BYTES,
                    ..WorkRequest::default()
                },
            )
            .unwrap()
            .commit_persistent()
            .unwrap();
        pair.resident = Some(short);
        assert!(pair.validate_prepared_prefix(&prefix, 4, &[0, 12]).is_err());
        pair.resident = Some(genuine);
        pair.validate_prepared_prefix(&prefix, 4, &[0, 12]).unwrap();
        assert_eq!(foreign_budget.receipt().unwrap().live_persistent_bytes, 0);
    }

    #[test]
    fn input_rows_owned_real_f32_transfer_keeps_same_claim_scope_errors_and_one_last_owner_release()
    {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, memory) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let targets = [24, 0, 12];
        let pair = rows(&prefix, &targets, at);
        let held = pair.resident.as_ref().unwrap().clone();
        let held_bytes = held.bytes();
        let required = pair.f32_storage_required_bytes(0, at).unwrap();
        let before = budget.receipt().unwrap();
        let delivered = pair.into_f32(0, required, at).unwrap();
        assert_eq!(delivered.rows.backing(), InputRowsBacking::F32);
        assert_eq!(delivered.rows.resident_bytes(), Some(held_bytes));
        assert!(delivered.receipt.coefficient_cells_stored > 0);
        assert!(delivered.receipt.bias_cells_stored > 0);
        assert!(delivered.receipt.new_payload_bytes < delivered.receipt.old_payload_bytes);
        delivered
            .rows
            .validate_prepared_prefix(&prefix, 4, &targets)
            .unwrap();
        let after = budget.receipt().unwrap();
        assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
        assert_eq!(after.live_scratch_bytes, 0);
        assert!(after.work_reserved > before.work_reserved);
        for input in [vec![-0.75; 25], vec![1.0; 25], vec![0.125; 25]] {
            for (col, &target) in targets.iter().enumerate() {
                let original = original_residual_preactivation(&input, target);
                check_line(
                    delivered.rows.lower(col).unwrap(),
                    &input,
                    &original,
                    false,
                    at,
                );
                check_line(
                    delivered.rows.upper(col).unwrap(),
                    &input,
                    &original,
                    true,
                    at,
                );
            }
        }
        drop(prefix);
        drop(delivered);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, held_bytes);
        drop(held);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }

    #[test]
    fn input_rows_owned_f32_integer_cap_cannot_substitute_for_original_shared_overlap_admission() {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, _) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let pair = rows(&prefix, &[0], at);
        let required = pair.f32_storage_required_bytes(0, at).unwrap();
        let occupied = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    scratch_bytes: CAP,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        assert!(pair.into_f32(0, required, at).is_err());
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, CAP);
        drop(occupied);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }

    // Insert in input_rows_tests.rs::owned, with its original helper ledger.

    #[test]
    fn input_rows_owned_native_complete_accounting_observation() {
        let _env = crate::tests::lock_env_shared();
        let (budget, _, memory) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let targets = [24, 0, 12];
        let pair = rows(&prefix, &targets, at);
        assert!(matches!(pair.storage, Storage::Dense { .. }));
        let held = pair.resident.as_ref().unwrap().clone();
        let held_bytes = held.bytes();
        let source_bytes = pair.scope.resident_floor(at).unwrap();
        let floor_before = pair.resident_floor(at).unwrap();
        assert_eq!(floor_before, source_bytes + held_bytes);
        let payload_before = pair.owned_payload_bytes().unwrap();
        assert!(held_bytes >= payload_before + INPUT_ROWS_OWNER_BYTES);
        let (coefficient_cells, bias_cells) = pair.nominal_cells().unwrap();
        let nominal_bytes = (coefficient_cells + bias_cells) * size_of::<f32>();
        let overlap_bytes = nominal_bytes + F32_TRANSFER_FIXED_BYTES;
        let required = pair.f32_storage_required_bytes(0, at).unwrap();
        assert_eq!(required, floor_before + overlap_bytes);
        let before = budget.receipt().unwrap();
        let transfer = pair.into_f32(0, required, at).unwrap();
        let payload_after = transfer.rows.owned_payload_bytes().unwrap();
        let floor_after = transfer.rows.resident_floor(at).unwrap();
        assert_eq!(floor_after, floor_before);
        assert_eq!(transfer.rows.resident_bytes(), Some(held_bytes));
        assert_eq!(transfer.receipt.old_payload_bytes, payload_before);
        assert_eq!(transfer.receipt.new_payload_bytes, payload_after);
        assert_eq!(transfer.receipt.new_nominal_allocation_bytes, nominal_bytes);
        assert_eq!(transfer.receipt.peak_admitted_bytes, required);
        transfer
            .rows
            .validate_prepared_prefix(&prefix, 4, &targets)
            .unwrap();
        let after = budget.receipt().unwrap();
        assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
        assert_eq!(after.live_scratch_bytes, 0);
        assert_eq!(
            after.peak_leased_bytes,
            before
                .peak_leased_bytes
                .max(before.live_persistent_bytes + overlap_bytes)
        );
        assert!(after.work_reserved > before.work_reserved);
        let changed_cells = transfer.receipt.changed_cells;
        drop(prefix);
        drop(transfer);
        let retained_after_drop = budget.receipt().unwrap().live_persistent_bytes;
        assert_eq!(retained_after_drop, held_bytes);
        drop(held);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        println!(
            "{{\"phase\":\"input_rows_accounting_prepared\",\"rows_bytes\":{},\"storage_bytes\":{},\"pair_bytes\":{},\"transfer_bytes\":{},\"owner_envelope_bytes\":{},\"f32_transfer_fixed_bytes\":{},\"source_bytes\":{source_bytes},\"target_bytes\":{},\"payload_before\":{payload_before},\"payload_after\":{payload_after},\"floor_before\":{floor_before},\"floor_after\":{floor_after},\"coefficient_cells\":{coefficient_cells},\"bias_cells\":{bias_cells},\"nominal_allocation_bytes\":{nominal_bytes},\"peak_admitted_bytes\":{required},\"changed_cells\":{changed_cells},\"held_bytes\":{held_bytes},\"live_persistent_bytes\":{},\"work_delta\":{},\"ledger_peak_before\":{},\"ledger_peak_after\":{},\"retained_after_drop\":{retained_after_drop}}}",
            size_of::<Rows>(),
            size_of::<Storage>(),
            size_of::<InputRowPair<'static>>(),
            size_of::<InputRowsF32Transfer<'static>>(),
            INPUT_ROWS_OWNER_BYTES,
            F32_TRANSFER_FIXED_BYTES,
            targets.len() * size_of::<usize>(),
            before.live_persistent_bytes,
            after.work_reserved - before.work_reserved,
            before.peak_leased_bytes,
            after.peak_leased_bytes,
        );
    }

    #[test]
    fn input_rows_owned_pointwise_scan_spends_original_work_and_cancelled_views_cannot_renew_deadline(
    ) {
        let _env = crate::tests::lock_env_shared();
        let (budget, cancel, _) = ledger();
        let at = deadline();
        let prefix = prefix(&budget, Some(at));
        let pair = rows(&prefix, &[0], at);
        let before = budget.receipt().unwrap().work_reserved;
        pair.lower(0)
            .unwrap()
            .pointwise_error(at + Duration::from_secs(60))
            .unwrap();
        let first = budget.receipt().unwrap().work_reserved;
        assert!(first > before);
        pair.lower(0).unwrap().pointwise_error(at).unwrap();
        assert_eq!(
            budget.receipt().unwrap().work_reserved - first,
            first - before
        );
        assert!(pair
            .lower(0)
            .unwrap()
            .pointwise_error(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap()
            )
            .is_err());
        cancel.cancel();
        assert!(pair
            .lower(0)
            .unwrap()
            .pointwise_error(at + Duration::from_secs(60))
            .is_err());
        assert!(pair.validate_prepared_prefix(&prefix, 4, &[0]).is_err());
        drop(pair);
        drop(prefix);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }
}
