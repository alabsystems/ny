// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine producer controls. No test mints row arrays, source seals or gates.

use super::super::compiler::OriginalModel;
use super::super::construction::{build_prepared_root, PreparedConstructionOptions};
use super::super::prefix::OriginalRoot;
use super::super::publication::share_prepared_root;
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkBudget, WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(2 * CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}
fn conv(n: usize, input: usize, weight: f64, bias: f64, fold: bool) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![bias],
        bias_err: vec![if fold { 1.0 / 1024.0 } else { 0.0 }],
        weight_rel_err: if fold { 1.0 / 512.0 } else { 0.0 },
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, n),
        oshape: (1, 1, n),
    }
}
fn fixture(n: usize, bias: f64, fold: bool) -> TwinSpec {
    TwinSpec {
        n_in: n,
        ops: vec![
            conv(n, 0, 1.0, bias, fold),
            TwinOpSpec::Relu { input: 1 },
            conv(n, 2, 1.0, -0.1, fold),
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25; n],
                bias: vec![0.5],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.25],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}
fn source(
    spec: TwinSpec,
    budget: &WorkBudget,
    lo: f64,
    hi: f64,
    f32: bool,
    tile: usize,
) -> PreparedRootSource {
    let n = spec.n_in;
    let model = OriginalModel::compile(spec, budget, Some(at())).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![lo; n],
        &vec![hi; n],
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let constructed = build_prepared_root(
        root,
        PreparedConstructionOptions {
            keep_input_rows: true,
            row_f32: f32,
            prefer_patch: false,
            dense_tile: tile,
            ..PreparedConstructionOptions::default()
        },
        Some(at()),
    )
    .unwrap();
    share_prepared_root(constructed, Some(at())).unwrap()
}
fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}
fn eval_halfspace(h: &Halfspace, x: &[f64]) -> BigRational {
    h.a.iter()
        .zip(x)
        .fold(r(h.d), |sum, (a, x)| sum + r(*a) * r(*x))
}

#[test]
fn prepared_retained_deterministic_ties_topk_and_strict_global_advice_budget() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(fixture(5, -0.25, false), &budget, -1.0, 1.0, false, 2);
    let row_bytes = 2 * 6 * size_of::<f32>();
    let retained = build_prepared_retained(
        &src,
        RetainCfg {
            per_layer: 3,
            budget_bytes: row_bytes * 4 + row_bytes - 1,
        },
        Some(at()),
    )
    .unwrap();
    assert_eq!(retained.layers().len(), 2);
    assert_eq!(retained.layers()[0].eligible(), 5);
    assert_eq!(retained.layers()[0].ids(), [0, 1, 2]);
    assert_eq!(retained.layers()[0].unstable_positions(), [0, 1, 2]);
    assert_eq!(retained.layers()[1].ids(), [0]);
    assert_eq!(retained.layers()[1].unstable_positions(), [0]);
    assert_eq!(retained.payload_bytes(), row_bytes * 4);
    assert!(retained.payload_bytes() <= retained.cfg().budget_bytes);
    let repeated = build_prepared_retained(&src, retained.cfg(), None).unwrap();
    assert_eq!(retained.digest(), repeated.digest());
    let changed = build_prepared_retained(
        &src,
        RetainCfg {
            per_layer: 3,
            budget_bytes: row_bytes * 3,
        },
        None,
    )
    .unwrap();
    assert_ne!(retained.digest(), changed.digest());
    for layer in retained.layers() {
        assert_eq!(layer.naug(), 6);
        for (pos, &id) in layer.ids().iter().enumerate() {
            let (rows, col) = retained.row(layer.layer(), id, None).unwrap();
            let lower = rows.lower(col).unwrap();
            let upper = rows.upper(col).unwrap();
            assert_eq!(layer.lower_advice()[(pos + 1) * 6 - 1], lower.bias() as f32);
            assert_eq!(layer.upper_advice()[(pos + 1) * 6 - 1], upper.bias() as f32);
            for (input, coefficient, _) in lower.entries() {
                assert_eq!(layer.lower_advice()[pos * 6 + input], coefficient as f32);
            }
        }
    }
}

#[test]
fn prepared_retained_topk_uses_actual_original_gate_slack_before_sorting_ids() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut spec = fixture(3, 0.0, false);
    for (op, weights, bias) in [
        (0, vec![1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0], 0.0),
        (2, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], -0.1),
    ] {
        spec.ops[op] = TwinOpSpec::Conv {
            input: if op == 0 { 0 } else { 2 },
            weight: weights,
            bias: vec![bias; 3],
            bias_err: vec![0.0; 3],
            weight_rel_err: 0.0,
            kernel: (3, 3, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (3, 1, 1),
            oshape: (3, 1, 1),
        };
    }
    let src = source(spec, &budget, -1.0, 1.0, false, 2);
    let retained = build_prepared_retained(
        &src,
        RetainCfg {
            per_layer: 2,
            budget_bytes: CAP,
        },
        None,
    )
    .unwrap();
    let gate = src.construction().prefix().gate(0).unwrap();
    let scores: Vec<_> = gate
        .unst
        .iter()
        .map(|&id| gate.c[id] * (gate.u[id] - gate.l[id]))
        .collect();
    assert!(scores[1] > scores[2] && scores[2] > scores[0]);
    assert_eq!(retained.layers()[0].ids(), [1, 2]);
    assert_eq!(retained.layers()[0].unstable_positions(), [1, 2]);
}

#[test]
fn prepared_retained_zero_budget_and_zero_topk_preserve_eligible_census_without_packs() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(fixture(3, -0.25, false), &budget, -1.0, 1.0, false, 1);
    for cfg in [
        RetainCfg {
            per_layer: 7,
            budget_bytes: 0,
        },
        RetainCfg {
            per_layer: 0,
            budget_bytes: CAP,
        },
    ] {
        let retained = build_prepared_retained(&src, cfg, None).unwrap();
        assert_eq!(retained.payload_bytes(), 0);
        assert!(retained
            .layers()
            .iter()
            .all(|layer| layer.ids().is_empty() && layer.eligible() == 3));
        assert!(retained.row(0, 0, None).is_err());
    }
}

#[test]
fn prepared_clip_all_final_ids_paid_ordered_filter_shares_exact_original_planes() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(fixture(5, -0.25, false), &budget, -1.0, 1.0, false, 2);
    let all = build_prepared_clip_rows(&src, None).unwrap();
    assert_eq!(all.len(), 10);
    let selected = [(0, 1), (0, 4), (1, 0), (1, 3)];
    let filtered = all.filter(&selected, None).unwrap();
    assert_eq!(filtered.ids().collect::<Vec<_>>(), selected);
    assert_ne!(all.digest(), filtered.digest());
    assert!(filtered.source().same_source(&src));
    for &(layer, target) in &selected {
        let (original, col) = all.row(layer, target, None).unwrap();
        let (shared, shared_col) = filtered.row(layer, target, None).unwrap();
        assert!(std::ptr::eq(original, shared));
        assert_eq!(col, shared_col);
        assert_eq!(shared.target(col), Some(target));
    }
    for bad in [
        vec![(0, 1), (0, 1)],
        vec![(1, 0), (0, 1)],
        vec![(0, 5)],
        vec![(2, 0)],
    ] {
        assert!(all.filter(&bad, None).is_err());
    }
    assert!(filtered.row(0, 0, None).is_err());
    let empty = filtered.filter(&[], None).unwrap();
    assert!(empty.is_empty());
    assert!(empty.resident_bytes() > 0);
}

#[test]
fn prepared_clip_independent_upper_bias_below_lower_bias_keeps_every_original_phase_member() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(fixture(1, -1.5, false), &budget, 1.0, 2.0, false, 1);
    let clip = build_prepared_clip_rows(&src, None).unwrap();
    let (rows, col) = clip.row(1, 0, None).unwrap();
    assert!(rows.upper(col).unwrap().bias() < rows.lower(col).unwrap().bias());
    let requests = [
        PreparedSplit {
            layer: 1,
            target: 0,
            side: InputSplitSide::Inactive,
        },
        PreparedSplit {
            layer: 1,
            target: 0,
            side: InputSplitSide::Active,
        },
    ];
    let batch = clip.halfspaces(&requests, None).unwrap();
    assert_eq!(batch.splits(), requests);
    assert_eq!(batch.source_digest(), clip.digest());
    let reversed = clip.halfspaces(&[requests[1], requests[0]], None).unwrap();
    assert_ne!(batch.digest(), reversed.digest());
    let preactivation = |x: f64| (r(x) - r(1.5)).max(BigRational::zero()) - r(0.1);
    for x in [1.0, 1.25, 1.5, 1.55, 1.6, 1.75, 2.0] {
        let exact = preactivation(x);
        if exact <= BigRational::zero() {
            assert!(eval_halfspace(&batch.halfspaces()[0], &[x]) <= BigRational::zero());
        }
        if exact >= BigRational::zero() {
            assert!(eval_halfspace(&batch.halfspaces()[1], &[x]) <= BigRational::zero());
        }
    }
    assert!(batch
        .charged_errors()
        .iter()
        .all(|eta| eta.is_finite() && *eta >= 0.0));
}

#[test]
fn prepared_clip_fold_uncertainty_and_actual_f32_storage_use_original_domain_row_errors() {
    let _env = crate::tests::lock_env_shared();
    for f32 in [false, true] {
        let (budget, _, _) = resources();
        let src = source(fixture(1, -1.5, true), &budget, 1.0, 2.0, f32, 1);
        let clip = build_prepared_clip_rows(&src, None).unwrap();
        let (rows, col) = clip.row(1, 0, None).unwrap();
        assert_eq!(
            rows.backing(),
            if f32 {
                InputRowsBacking::F32
            } else {
                InputRowsBacking::F64
            }
        );
        let requests = [
            PreparedSplit {
                layer: 1,
                target: 0,
                side: InputSplitSide::Inactive,
            },
            PreparedSplit {
                layer: 1,
                target: 0,
                side: InputSplitSide::Active,
            },
        ];
        let batch = clip.halfspaces(&requests, None).unwrap();
        for (i, view) in [rows.lower(col).unwrap(), rows.upper(col).unwrap()]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                batch.charged_errors()[i].to_bits(),
                view.pointwise_error(at()).unwrap().to_bits()
            );
            assert!(batch.charged_errors()[i] >= 1.0 / 1024.0);
        }
        for x in [1.0, 1.25, 1.5, 1.6, 1.75, 2.0] {
            for signs in 0..16 {
                let sign = |bit| {
                    if signs & (1 << bit) == 0 {
                        r(-1.0)
                    } else {
                        r(1.0)
                    }
                };
                let h = (r(x) * (r(1.0) + sign(0) * r(1.0 / 512.0)) - r(1.5)
                    + sign(1) * r(1.0 / 1024.0))
                .max(BigRational::zero());
                let y =
                    h * (r(1.0) + sign(2) * r(1.0 / 512.0)) - r(0.1) + sign(3) * r(1.0 / 1024.0);
                if y <= BigRational::zero() {
                    assert!(eval_halfspace(&batch.halfspaces()[0], &[x]) <= BigRational::zero());
                }
                if y >= BigRational::zero() {
                    assert!(eval_halfspace(&batch.halfspaces()[1], &[x]) <= BigRational::zero());
                }
            }
        }
    }
}

#[test]
fn prepared_clip_content_key_binds_model_domain_row_backing_and_ordered_selection() {
    let _env = crate::tests::lock_env_shared();
    let mut digests = Vec::new();
    for (bias, lo, hi, f32) in [
        (-0.25, -1.0, 1.0, false),
        (-0.3, -1.0, 1.0, false),
        (-0.25, -0.9, 1.0, false),
        (-0.25, -1.0, 1.0, true),
    ] {
        let (budget, _, _) = resources();
        let src = source(fixture(2, bias, false), &budget, lo, hi, f32, 1);
        let all = build_prepared_clip_rows(&src, None).unwrap();
        let repeated = build_prepared_clip_rows(&src, None).unwrap();
        assert_eq!(all.digest(), repeated.digest());
        let first = all.filter(&[(0, 0)], None).unwrap();
        let second = all.filter(&[(0, 1)], None).unwrap();
        assert_ne!(first.digest(), second.digest());
        digests.push(*all.digest());
    }
    for (i, digest) in digests.iter().enumerate() {
        assert!(digests
            .iter()
            .enumerate()
            .all(|(j, other)| i == j || digest != other));
    }
}

#[test]
fn prepared_clip_and_retention_reject_foreign_equal_content_source_identity() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let first = source(fixture(2, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let second = source(fixture(2, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let clip = build_prepared_clip_rows(&first, None).unwrap();
    let other = build_prepared_clip_rows(&second, None).unwrap();
    assert_eq!(clip.digest(), other.digest());
    assert!(!first.same_source(&second));
    assert!(clip.validate_source(&second, None).is_err());
    let retained = build_prepared_retained(&first, RetainCfg::default(), None).unwrap();
    assert!(retained.validate_source(&second, None).is_err());
    let batch = clip
        .halfspaces(
            &[PreparedSplit {
                layer: 0,
                target: 0,
                side: InputSplitSide::Active,
            }],
            None,
        )
        .unwrap();
    assert!(batch.validate_source(&second, None).is_err());
    clip.validate_source(&first, None).unwrap();
    retained.validate_source(&first, None).unwrap();
    batch.validate_source(&first, None).unwrap();
}

#[test]
fn prepared_clip_batch_retains_last_original_source_and_halfspace_claim() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let src = source(fixture(2, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let clip = build_prepared_clip_rows(&src, None).unwrap();
    let retained = build_prepared_retained(&src, RetainCfg::default(), None).unwrap();
    let batch = clip
        .halfspaces(
            &[PreparedSplit {
                layer: 1,
                target: 1,
                side: InputSplitSide::Active,
            }],
            None,
        )
        .unwrap();
    let batch_claim = batch.resident_bytes();
    assert!(batch_claim >= (size_of::<Halfspace>() + 2 * size_of::<f64>()));
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let clip_alias = clip.clone();
    let retained_alias = retained.clone();
    let batch_alias = batch.clone();
    assert!(clip.same_owner(&clip_alias));
    assert!(retained.same_owner(&retained_alias));
    assert!(batch.same_owner(&batch_alias));
    assert_eq!(retained.layers().as_ptr(), retained_alias.layers().as_ptr());
    assert_eq!(
        batch.halfspaces().as_ptr(),
        batch_alias.halfspaces().as_ptr()
    );
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
    drop(src);
    assert_eq!(live(&memory), before);
    let removable = clip.resident_bytes() + retained.resident_bytes();
    drop((clip, retained));
    assert_eq!(live(&memory), before);
    drop((clip_alias, retained_alias));
    assert_eq!(live(&memory), before - removable);
    batch.validate_source(batch.source(), None).unwrap();
    assert_eq!(batch.halfspaces().len(), 1);
    assert!(live(&memory) > batch_claim);
    drop(batch);
    assert!(live(&memory) > 0);
    drop(batch_alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn prepared_clip_halfspace_cap_expiry_and_cancellation_refuse_before_output_allocation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let src = source(fixture(2, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let clip = build_prepared_clip_rows(&src, None).unwrap();
    let request = [PreparedSplit {
        layer: 0,
        target: 0,
        side: InputSplitSide::Inactive,
    }];
    let r0 = budget.receipt().unwrap();
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    assert!(clip
        .halfspaces(
            &request,
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .unwrap()
            )
        )
        .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, r0.admitted);
    let ceiling = r0.live_persistent_bytes + r0.live_scratch_bytes;
    budget.tighten_total_byte_limit(ceiling).unwrap();
    assert!(clip.halfspaces(&request, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, r0.admitted);
    assert_eq!(live(&memory), ceiling);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    cancel.cancel();
    assert!(clip.filter(&[(0, 0)], None).is_err());
    assert!(clip.halfspaces(&request, None).is_err());
    assert_eq!(live(&memory), ceiling);
}

#[test]
fn prepared_clip_missing_requested_row_delivery_and_unknown_split_refuse_honestly() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let model = OriginalModel::compile(fixture(1, -0.25, false), &budget, Some(at())).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-1.0],
        &[1.0],
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let incomplete =
        build_prepared_root(root, PreparedConstructionOptions::default(), Some(at())).unwrap();
    let src = share_prepared_root(incomplete, Some(at())).unwrap();
    assert!(build_prepared_clip_rows(&src, None).is_err());
    assert!(build_prepared_retained(&src, RetainCfg::default(), None).is_err());
    let actual = source(fixture(1, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let clip = build_prepared_clip_rows(&actual, None).unwrap();
    for (layer, target) in [(0, 1), (2, 0)] {
        assert!(clip
            .halfspaces(
                &[PreparedSplit {
                    layer,
                    target,
                    side: InputSplitSide::Active
                }],
                None
            )
            .is_err());
    }
}

#[test]
fn prepared_owners_preserve_the_earliest_successful_consumer_deadline() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let src = source(fixture(1, -0.25, false), &budget, -1.0, 1.0, false, 1);
    let early = src
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .unwrap();
    let clip = build_prepared_clip_rows(&src, Some(early)).unwrap();
    assert_eq!(clip.deadline(None).unwrap(), early);
    let filtered = clip
        .filter(&[(0, 0)], Some(early + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(filtered.deadline(None).unwrap(), early);
    let earliest = early.checked_sub(Duration::from_secs(1)).unwrap();
    let batch = filtered
        .halfspaces(
            &[PreparedSplit {
                layer: 0,
                target: 0,
                side: InputSplitSide::Active,
            }],
            Some(earliest),
        )
        .unwrap();
    assert_eq!(batch.deadline(None).unwrap(), earliest);
    let retained = build_prepared_retained(&src, RetainCfg::default(), Some(earliest)).unwrap();
    assert_eq!(retained.deadline(Some(early)).unwrap(), earliest);
}

#[test]
fn prepared_ranker_unrepresentable_f32_advice_refuses_and_releases_all_new_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut spec = fixture(1, 0.0, false);
    spec.ops[0] = conv(1, 0, 1e100, 0.0, false);
    let src = source(spec, &budget, -1.0, 1.0, false, 1);
    let before = live(&memory);
    assert!(build_prepared_retained(&src, RetainCfg::default(), None).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    // The genuine binary64 clip rows remain useful and source-owned.
    let clip = build_prepared_clip_rows(&src, None).unwrap();
    assert!(clip
        .row(0, 0, None)
        .unwrap()
        .0
        .lower(0)
        .unwrap()
        .entries()
        .any(|(_, a, _)| a.abs() > f32::MAX as f64));
}

#[test]
fn prepared_clip_solved_original_layers_deliver_empty_complete_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let src = source(fixture(2, 2.0, false), &budget, -1.0, 1.0, false, 1);
    assert!(src.construction().coverage_complete(None).unwrap());
    let clip = build_prepared_clip_rows(&src, None).unwrap();
    assert!(clip.is_empty());
    let retained = build_prepared_retained(&src, RetainCfg::default(), None).unwrap();
    assert!(retained
        .layers()
        .iter()
        .all(|layer| layer.eligible() == 0 && layer.ids().is_empty()));
    let batch = clip.halfspaces(&[], None).unwrap();
    assert!(batch.halfspaces().is_empty());
    drop((src, clip, retained, batch));
    assert_eq!(live(&memory), 0);
}
