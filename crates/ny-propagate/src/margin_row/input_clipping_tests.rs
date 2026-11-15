// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::engine::{BackwardEngine, Seed};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::gpu_conv::GpuConvRoute;
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use ndarray::arr2;
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::time::Duration;

fn fixture() -> TwinNet {
    TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![1.5],
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
                scale: vec![1.0],
                shift: vec![0.0],
                scale_rel_err: 0.0,
                shift_err: vec![0.0],
                shape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![1.0, -1.0],
                bias: vec![0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    })
    .unwrap()
}

fn root(net: &TwinNet) -> RootGates {
    let _source = HeadGroupSourceGuard::enter(true);
    RootGates::build_prec_store(
        net,
        &[-2.0],
        &[-1.0],
        RoundMode::Outward,
        None,
        false,
        false,
    )
    .unwrap()
}

fn pair<'a>(net: &'a TwinNet, root: &'a RootGates, targets: &'a [usize]) -> InputRowPair<'a> {
    BackwardEngine::new(net, root)
        .with_err_lane(ErrLane::PerCol)
        .with_gpu_conv_route(GpuConvRoute::Off)
        .run_prefix_input_rows(
            &Seed {
                s: arr2(&[[1.0]]),
                e: None,
            },
            3,
            targets,
            0,
            1 << 24,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
}

fn constraint<'pair, 'source>(
    rows: &'pair InputRowPair<'source>,
    net: &TwinNet,
    root: &RootGates,
    side: InputSplitSide,
) -> ScopedInputHalfspace<'pair, 'source> {
    input_halfspace(
        rows,
        net,
        root,
        3,
        &[0],
        0,
        side,
        0,
        1 << 20,
        Instant::now() + Duration::from_secs(100),
    )
    .unwrap()
}

#[test]
fn input_clip_products_preserve_crossed_constant_order_and_cover_exact_original_phases() {
    let _env = ny_test_utils::env::lock_env_shared();
    let net = fixture();
    let root = root(&net);
    let targets = [0];
    let rows = pair(&net, &root, &targets);
    assert!(rows.upper(0).unwrap().bias() < rows.lower(0).unwrap().bias());
    let inactive = constraint(&rows, &net, &root, InputSplitSide::Inactive);
    let active = constraint(&rows, &net, &root, InputSplitSide::Active);
    // Independent exact phase vertices of z(x)=max(x+3/2,0). Both
    // halfspaces must retain the shared zero boundary; the active premise
    // holds on this entire box because the requested preactivation is z.
    for x in [-2.0, -1.75, -1.5, -1.25, -1.0] {
        let qx = Q::from_float(x).unwrap();
        let exact = (qx.clone() + Q::from_float(1.5).unwrap()).max(Q::zero());
        let eval =
            |h: &Halfspace| Q::from_float(h.a[0]).unwrap() * &qx + Q::from_float(h.d).unwrap();
        if exact.is_zero() {
            assert!(eval(inactive.halfspace()) <= Q::zero());
        }
        assert!(eval(active.halfspace()) <= Q::zero());
    }
    // A midpoint+absolute-deviation bias would weaken the inactive
    // constant from 1.5 to 1.0 and keep this impossible inactive point.
    let h = inactive.halfspace();
    assert!(
        Q::from_float(h.a[0]).unwrap() * Q::from_float(-1.25).unwrap()
            + Q::from_float(h.d).unwrap()
            > Q::zero()
    );
    assert_eq!(inactive.original_target(), 0);
    assert_eq!(inactive.side(), InputSplitSide::Inactive);
    assert_eq!(active.side(), InputSplitSide::Active);
}

#[test]
fn input_clip_products_charge_each_direction_and_keep_original_deadline() {
    let _env = ny_test_utils::env::lock_env_shared();
    let net = fixture();
    let root = root(&net);
    let targets = [0];
    let rows = pair(&net, &root, &targets);
    for side in [InputSplitSide::Inactive, InputSplitSide::Active] {
        let h = constraint(&rows, &net, &root, side);
        let row = if side == InputSplitSide::Inactive {
            rows.lower(0)
        } else {
            rows.upper(0)
        }
        .unwrap();
        let b = Q::from_float(if side == InputSplitSide::Inactive {
            row.bias()
        } else {
            -row.bias()
        })
        .unwrap();
        let mut error = Q::from_float(row.bias_error()).unwrap();
        for (i, _, e) in row.entries() {
            error += Q::from_float(e).unwrap()
                * Q::from_float(root.lo[i].abs().max(root.hi[i].abs())).unwrap();
        }
        assert!(Q::from_float(h.halfspace().d).unwrap() <= b - &error);
        assert!(Q::from_float(h.charged_error()).unwrap() >= error);
        assert_eq!(h.deadline(), rows.deadline());
    }
}

#[test]
fn input_clip_products_refuse_foreign_owner_row_and_expired_phase() {
    let _env = ny_test_utils::env::lock_env_shared();
    let net = fixture();
    let root = root(&net);
    let targets = [0];
    let rows = pair(&net, &root, &targets);
    let foreign = self::root(&net);
    assert!(input_halfspace(
        &rows,
        &net,
        &foreign,
        3,
        &targets,
        0,
        InputSplitSide::Inactive,
        0,
        1 << 20,
        rows.deadline()
    )
    .is_err());
    assert!(input_halfspace(
        &rows,
        &net,
        &root,
        3,
        &targets,
        1,
        InputSplitSide::Inactive,
        0,
        1 << 20,
        rows.deadline()
    )
    .is_err());
    assert!(matches!(
        input_halfspace(
            &rows,
            &net,
            &root,
            3,
            &targets,
            0,
            InputSplitSide::Inactive,
            0,
            1 << 20,
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn input_clip_products_output_cap_is_atomic_and_accounts_for_resident_base() {
    let _env = ny_test_utils::env::lock_env_shared();
    let net = fixture();
    let root = root(&net);
    let targets = [0];
    let rows = pair(&net, &root, &targets);
    let resident = 8192.max(rows.resident_floor(rows.deadline()).unwrap());
    let needed = input_halfspace_bytes(resident, net.n_in).unwrap();
    assert!(matches!(
        input_halfspace(
            &rows,
            &net,
            &root,
            3,
            &targets,
            0,
            InputSplitSide::Inactive,
            resident,
            needed - 1,
            rows.deadline()
        ),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    let h = input_halfspace(
        &rows,
        &net,
        &root,
        3,
        &targets,
        0,
        InputSplitSide::Inactive,
        resident,
        needed,
        rows.deadline(),
    )
    .unwrap();
    assert_eq!(h.halfspace().a.len(), net.n_in);
    assert!(input_halfspace_bytes(resident, h.halfspace().a.capacity()).unwrap() <= needed);
    assert!(input_halfspace_bytes(usize::MAX, net.n_in).is_none());
}

#[test]
fn input_clip_products_zero_caller_census_cannot_hide_live_source_and_rows() {
    let _env = ny_test_utils::env::lock_env_shared();
    let net = fixture();
    let root = root(&net);
    let targets = [0];
    let rows = pair(&net, &root, &targets);
    let producer_floor = rows.resident_floor(rows.deadline()).unwrap();
    assert!(producer_floor > 0);
    let output_only = input_halfspace_bytes(0, net.n_in).unwrap();
    assert!(output_only < input_halfspace_bytes(producer_floor, net.n_in).unwrap());
    assert!(matches!(
        input_halfspace(
            &rows,
            &net,
            &root,
            3,
            &targets,
            0,
            InputSplitSide::Inactive,
            0,
            output_only,
            rows.deadline()
        ),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    assert_eq!(rows.target(0), Some(0));
}
