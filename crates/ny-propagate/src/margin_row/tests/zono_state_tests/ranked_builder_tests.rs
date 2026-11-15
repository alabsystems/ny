// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Streaming construction preserves the complete adaptive objective, the
//! selected row bits, global generator indices and exact static-order ties.

use super::*;
use crate::margin_row::net::TwinOp;
use crate::margin_row::zono_state::build::estimate_ranked_state_bytes;

fn assert_float_bits(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn assert_ranked_matches_full(
    full: &FrozenDualState,
    compact: &FrozenDualState,
    originals: &[usize],
) {
    assert_eq!(compact.t, full.t);
    assert_eq!(compact.class, full.class);
    assert_eq!(compact.n_gens, full.n_gens);
    assert_eq!(compact.n_in_gens, full.n_in_gens);
    assert_eq!(compact.c_obj.to_bits(), full.c_obj.to_bits());
    assert_eq!(compact.r_obj.to_bits(), full.r_obj.to_bits());
    assert_float_bits(&compact.d, &full.d);
    assert_eq!(compact.root_bound().to_bits(), full.root_bound().to_bits());
    assert_eq!(compact.num_unstable(), originals.len());
    assert_eq!(compact.a.len(), originals.len() * compact.n_gens);
    for (k, &original) in originals.iter().enumerate() {
        assert_float_bits(compact.row(k), full.row(original));
        for (left, right) in [
            (&compact.c_in, &full.c_in),
            (&compact.r, &full.r),
            (&compact.lam, &full.lam),
            (&compact.mu, &full.mu),
            (&compact.lo, &full.lo),
            (&compact.hi, &full.hi),
        ] {
            assert_eq!(left[k].to_bits(), right[original].to_bits());
        }
        assert_eq!(compact.enc[k], full.enc[original]);
        assert_eq!(compact.layer[k], full.layer[original]);
        assert_eq!(compact.site[k], full.site[original]);
        assert_eq!(compact.neuron[k], full.neuron[original]);
    }
    assert_eq!(compact.stats.passes, full.stats.passes);
    assert_eq!(compact.stats.columns, full.stats.columns);
    assert!(compact.bytes() <= compact.stats.estimate_bytes);
}

#[test]
fn frozen_dual_ranked_builder_matches_full_topk_bits_and_global_symbols() {
    for fixture in fixtures() {
        let frame = frame_of(&fixture);
        for &class in fixture.adv.iter().take(2) {
            // Identical pass chunking is required for exact BLAS arithmetic.
            for chunk_cols in [1, 8, 256] {
                let opts = FrozenDualOptions {
                    chunk_cols,
                    ..FrozenDualOptions::default()
                };
                let full =
                    FrozenDualState::build(&fixture.net, &frame, class, &opts).expect("full state");
                let order = static_order(&full);
                for max_rows in [0, 1, 7, 32, 64, usize::MAX] {
                    let ranked =
                        FrozenDualState::build_ranked(&fixture.net, &frame, class, &opts, max_rows)
                            .expect("ranked state");
                    let count = max_rows
                        .min(crate::margin_row::zono_bab::FROZEN_DUAL_MAX_DEPTH)
                        .min(order.len());
                    assert_eq!(ranked.total_unstable, full.num_unstable());
                    assert_eq!(ranked.original_rows, order[..count]);
                    assert_ranked_matches_full(&full, &ranked.state, &ranked.original_rows);
                    assert_eq!(
                        ranked.state.stats.estimate_bytes,
                        estimate_ranked_state_bytes(&frame, chunk_cols, max_rows)
                    );
                    // Problem construction uses compact row IDs while the
                    // copied rows retain their original generator columns.
                    let compact_order: Vec<usize> = (0..count).collect();
                    let compact_problem = FrozenDualProblem::new(&ranked.state, &compact_order)
                        .expect("compact problem");
                    let full_problem = FrozenDualProblem::new(&full, &order[..count])
                        .expect("full problem prefix");
                    assert_eq!(
                        compact_problem.root_bound().to_bits(),
                        full_problem.root_bound().to_bits()
                    );
                    for side in [Side::Off, Side::On] {
                        let depth = count.min(4);
                        let sides = vec![side; depth];
                        let l0 = vec![0.25; depth];
                        let l1 = vec![0.5; depth];
                        let compact_bound = compact_problem.eval_node(
                            &sides,
                            &l0,
                            &l1,
                            &mut NodeScratch::default(),
                            None,
                        );
                        let full_bound = full_problem.eval_node(
                            &sides,
                            &l0,
                            &l1,
                            &mut NodeScratch::default(),
                            None,
                        );
                        assert_eq!(compact_bound.value.to_bits(), full_bound.value.to_bits());
                        assert_eq!(
                            compact_bound.certified.to_bits(),
                            full_bound.certified.to_bits()
                        );
                        assert_eq!(compact_bound.charge.to_bits(), full_bound.charge.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn frozen_dual_ranked_builder_keeps_more_than_32_exact_rows() {
    let fixture = fx(
        "ranked_wide",
        conv_bn_spec(3, 2, 2, 32, 5.0, 1e-13, 1e-6, 128, 6, 0xC064),
        1.0,
        64,
        false,
    );
    let frame = frame_of(&fixture);
    let opts = FrozenDualOptions {
        chunk_cols: 8,
        ..FrozenDualOptions::default()
    };
    let full = FrozenDualState::build(&fixture.net, &frame, fixture.adv[0], &opts)
        .expect("wide full state");
    assert!(full.num_unstable() > 64);
    let ranked = FrozenDualState::build_ranked(&fixture.net, &frame, fixture.adv[0], &opts, 64)
        .expect("wide ranked state");
    assert_eq!(ranked.state.num_unstable(), 64);
    assert_eq!(ranked.original_rows, static_order(&full)[..64]);
    assert_ranked_matches_full(&full, &ranked.state, &ranked.original_rows);
    assert!(ranked.state.bytes() < full.bytes());
}

#[test]
fn frozen_dual_ranked_builder_preserves_original_row_ties() {
    let mut fixture = fixture0();
    let Some(TwinOp::Gemm { weight, bias, .. }) = fixture.net.ops.last_mut() else {
        panic!("fixture final gemm");
    };
    weight.fill(0.0);
    bias.fill(0.0);
    let frame = frame_of(&fixture);
    let opts = FrozenDualOptions {
        chunk_cols: 3,
        ..FrozenDualOptions::default()
    };
    let full = FrozenDualState::build(&fixture.net, &frame, fixture.adv[0], &opts)
        .expect("zero-objective state");
    assert!(full.num_unstable() > 1);
    let scores = split_scores(&full);
    assert!(scores.iter().all(|v| v.to_bits() == scores[0].to_bits()));
    let ranked = FrozenDualState::build_ranked(&fixture.net, &frame, fixture.adv[0], &opts, 7)
        .expect("ranked ties");
    let expected: Vec<usize> = (0..7.min(full.num_unstable())).collect();
    assert_eq!(ranked.original_rows, expected);
    assert_ranked_matches_full(&full, &ranked.state, &ranked.original_rows);
}

#[test]
fn frozen_dual_ranked_builder_refuses_budget_deadline_and_wrong_class() {
    let fixture = fixture0();
    let frame = frame_of(&fixture);
    let estimate = estimate_ranked_state_bytes(&frame, 8, 7);
    let opts = FrozenDualOptions {
        chunk_cols: 8,
        max_state_bytes: estimate - 1,
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualState::build_ranked(&fixture.net, &frame, fixture.adv[0], &opts, 7),
        Err(ZonoRefusal::Memory { estimate_bytes, budget_bytes })
            if estimate_bytes == estimate && budget_bytes < estimate_bytes
    ));
    let opts = FrozenDualOptions {
        deadline: Some(std::time::Instant::now()),
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualState::build_ranked(&fixture.net, &frame, fixture.adv[0], &opts, 7),
        Err(ZonoRefusal::Deadline)
    ));
    assert!(matches!(
        FrozenDualState::build_ranked(
            &fixture.net,
            &frame,
            fixture.t,
            &FrozenDualOptions::default(),
            7,
        ),
        Err(ZonoRefusal::Shape(_))
    ));
}

#[test]
fn frozen_dual_ranked_estimate_accounts_for_global_symbols_and_bounded_scoring() {
    let mut frame = frame_of(&fixture0());
    frame.n_unstable = 2048;
    frame.n_gens = 4096;
    let ranked = estimate_ranked_state_bytes(&frame, 8, 32);
    let full = crate::margin_row::zono_state::estimate_state_bytes(&frame, 8);
    assert!(ranked < full);
    assert_eq!(
        estimate_ranked_state_bytes(&frame, 8, 512),
        estimate_ranked_state_bytes(&frame, 8, usize::MAX)
    );
    let fewer = estimate_ranked_state_bytes(&frame, 8, 1);
    assert!(fewer < ranked);
    // Retained rows continue to carry coefficients for every global symbol.
    let before = ranked;
    frame.n_gens *= 2;
    assert!(estimate_ranked_state_bytes(&frame, 8, 32) > before);
}
