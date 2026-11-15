// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::work::{WorkBudget, WorkRefusal};
use num_rational::BigRational;
use std::time::{Duration, Instant};

fn head<'a>(vocab: usize, dim: usize, weights: &'a [f64], bias: &'a [f64]) -> AffineHead<'a> {
    AffineHead {
        vocab_size: vocab,
        hidden_size: dim,
        weights,
        bias,
        arithmetic: HeadArithmetic::ExactReal,
    }
}
fn options() -> VocabOptions {
    VocabOptions {
        leaf_size: 2,
        ..VocabOptions::default()
    }
}
fn original_margin(h: AffineHead<'_>, point: &[f64], target: usize, row: usize) -> BigRational {
    let q = |x: f64| BigRational::from_float(x).expect("finite exact binary64");
    let mut margin = q(h.bias[row]) - q(h.bias[target]);
    for ((&wj, &wt), &x) in h.row(row).iter().zip(h.row(target)).zip(point) {
        margin += (q(wj) - q(wt)) * q(x);
    }
    margin
}
fn vertices(lo: &[f64], hi: &[f64]) -> Vec<Vec<f64>> {
    (0..1usize << lo.len())
        .map(|bits| {
            (0..lo.len())
                .map(|k| if bits & (1 << k) == 0 { lo[k] } else { hi[k] })
                .collect()
        })
        .collect()
}

#[test]
fn decoder_d2_complete_vocabulary_group_replays_and_binds_original_head() {
    let weights = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let bias = [0.0, -1.5, -1.5, -1.5];
    let h = head(4, 2, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-1.0, -1.0],
        hi: &[1.0, 1.0],
    };
    let report =
        verify_next_token(h, hidden, &[0.0, 0.0], 0, &options()).expect("group certificate");
    assert!(report.is_certified());
    assert_eq!(report.evidence.groups.len(), 1);
    assert_eq!(report.evidence.groups[0].members, [1, 2, 3]);
    let g = report.evidence.groups[0]
        .group_bounds
        .as_ref()
        .expect("group geometry used");
    assert!(
        g.residual_l1_linf < g.residual_componentwise,
        "matching l1/linf pair is stronger"
    );
    assert!(g.upper < 0.0);
    let replay = replay_next_token(h, hidden, &[0.0, 0.0], 0, &report.evidence, &options())
        .expect("original-head replay");
    assert!(replay.is_certified());
    let NextTokenOutcome::Certified(cert) = replay.outcome else {
        panic!("certificate");
    };
    assert_eq!(cert.target(), 0);
    assert_eq!(cert.covered_competitors(), 3);
    assert!(cert.maximum_competitor_upper().expect("nonempty cover") < 0.0);
    let mut changed = weights;
    changed[2] = 0.125;
    assert!(replay_next_token(
        head(4, 2, &changed, &bias),
        hidden,
        &[0.0, 0.0],
        0,
        &report.evidence,
        &options()
    )
    .is_err());
    assert!(replay_next_token(h, hidden, &[0.1, 0.0], 0, &report.evidence, &options()).is_err());
    assert!(replay_next_token(
        h,
        HiddenBox {
            lo: hidden.lo,
            hi: &[1.1, 1.0]
        },
        &[0.0, 0.0],
        0,
        &report.evidence,
        &options()
    )
    .is_err());
}

#[test]
fn decoder_d2_componentwise_candidate_beats_mismatched_box_norm_cost() {
    let weights = [0.0, 0.0, 0.0, 0.0, 100.0, 0.0];
    let bias = [0.0, -1.0, -1.0];
    let h = head(3, 2, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.001, -1.0],
        hi: &[0.001, 1.0],
    };
    let report = verify_next_token(h, hidden, &[0.0, 0.0], 0, &options()).expect("anisotropic box");
    assert!(report.is_certified());
    let bounds = report.evidence.groups[0]
        .group_bounds
        .as_ref()
        .expect("group");
    for other in [
        bounds.residual_l1_linf,
        bounds.residual_l2_l2,
        bounds.residual_linf_l1,
    ] {
        assert!(bounds.residual_componentwise < other);
    }
    assert!(bounds.upper < -0.89);
    assert!(
        replay_next_token(h, hidden, &[0.0, 0.0], 0, &report.evidence, &options())
            .expect("replay")
            .is_certified()
    );
}

#[test]
fn decoder_d2_asymmetric_anchor_support_is_not_a_midpoint_radius() {
    let weights = [0.0, 1.0];
    let bias = [0.0, -3.1];
    let h = head(2, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[2.0],
        hi: &[3.0],
    };
    for anchor in [2.9, 100.0] {
        let report =
            verify_next_token(h, hidden, &[anchor], 0, &options()).expect("arbitrary anchor");
        assert!(report.is_certified());
        let b = report.evidence.groups[0]
            .group_bounds
            .as_ref()
            .expect("group");
        assert!(b.anchor_upper < 3.0 - anchor + 1e-11);
        assert!(b.upper < -0.099999999);
        assert!(
            replay_next_token(h, hidden, &[anchor], 0, &report.evidence, &options())
                .expect("replay")
                .is_certified()
        );
    }
}

#[test]
fn decoder_d2_target_is_absent_from_every_group_and_residual_radius() {
    let weights = [1000.0, 0.0, 0.0, 0.0];
    let bias = [0.0, -1.0, -1.0, -1.0];
    let h = head(4, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[1.0],
        hi: &[2.0],
    };
    let r = verify_next_token(h, hidden, &[1.5], 0, &options()).expect("target excluded");
    assert!(r.is_certified());
    for group in &r.evidence.groups {
        assert!(!group.members.contains(&0));
        assert_ne!(group.anchor_token, 0);
        let b = group.group_bounds.as_ref().expect("group");
        assert!(
            b.radius_l1 < 1e-10 && b.radius_linf < 1e-10,
            "target contaminated radius"
        );
    }
}

#[test]
fn decoder_d2_no_shortlist_can_hide_a_distant_unsafe_competitor() {
    let weights = vec![0.0; 19];
    let mut bias = vec![-1.0; 19];
    bias[0] = 0.0;
    bias[18] = 0.25;
    let h = head(19, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[0.0],
        hi: &[0.0],
    };
    let r = verify_next_token(h, hidden, &[0.0], 0, &options()).expect("full vocabulary");
    assert!(!r.is_certified());
    assert_eq!(r.unresolved, [18]);
    assert!(r.stats.groups_visited > 1 && r.stats.dense_rows > 0);
    assert!(original_margin(h, &[0.0], 0, 18) > BigRational::from_integer(0.into()));
    let mut omitted = r.evidence;
    omitted.groups.retain(|g| !g.members.contains(&18));
    assert!(replay_next_token(h, hidden, &[0.0], 0, &omitted, &options()).is_err());
}

#[test]
fn decoder_d2_coverage_poisoning_duplicate_target_and_false_radius_refuse() {
    let weights = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let bias = [0.0, -1.5, -1.5, -1.5];
    let h = head(4, 2, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-1.0, -1.0],
        hi: &[1.0, 1.0],
    };
    let original = verify_next_token(h, hidden, &[0.0, 0.0], 0, &options())
        .expect("certificate")
        .evidence;
    for mutation in 0..7 {
        let mut evidence = original.clone();
        match mutation {
            0 => {
                evidence.groups[0].members[1] = evidence.groups[0].members[0];
            }
            1 => {
                evidence.groups[0].members[0] = 0;
            }
            2 => {
                evidence.groups[0].anchor_token = 0;
            }
            3 => {
                evidence.groups[0].group_bounds.as_mut().unwrap().radius_l1 = 0.0;
            }
            4 => {
                evidence.groups[0]
                    .group_bounds
                    .as_mut()
                    .unwrap()
                    .residual_componentwise = 0.0;
            }
            5 => {
                evidence.groups[0].upper = -100.0;
            }
            _ => {
                evidence.groups[0].group_bounds.as_mut().unwrap().radius_l2 = f64::NAN;
            }
        }
        assert!(
            replay_next_token(h, hidden, &[0.0, 0.0], 0, &evidence, &options()).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn decoder_d2_exact_ties_remain_unknown_and_singleton_vocabulary_is_vacuous() {
    let weights = [1.0, 1.0];
    let bias = [0.0, 0.0];
    let h = head(2, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[1.0],
        hi: &[1.0],
    };
    let r = verify_next_token(h, hidden, &[1.0], 0, &options()).expect("tie");
    assert!(!r.is_certified());
    assert_eq!(r.unresolved, [1]);
    assert!(
        !replay_next_token(h, hidden, &[1.0], 0, &r.evidence, &options())
            .expect("tie replay")
            .is_certified()
    );
    let single = verify_next_token(head(1, 1, &[1.0], &[0.0]), hidden, &[1.0], 0, &options())
        .expect("singleton");
    let NextTokenOutcome::Certified(cert) = single.outcome else {
        panic!("vacuous certificate");
    };
    assert_eq!(cert.covered_competitors(), 0);
    assert_eq!(cert.maximum_competitor_upper(), None);
}

#[test]
fn decoder_d2_dense_and_groups_dominate_exact_rational_original_vertices() {
    let weights = [
        1e9,
        0.1,
        -0.2,
        1e9 + 0.25,
        -0.3,
        0.7,
        1e9 - 0.125,
        0.9,
        -0.125,
        1e9 + 0.5,
        0.7,
        0.2,
    ];
    let bias = [3.0, 0.0, 0.25, -0.1];
    let h = head(4, 3, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.75, 0.2, -1.3],
        hi: &[0.4, 1.1, 0.9],
    };
    let anchor = [0.2, 0.8, -0.4];
    let dense =
        verify_dense_next_token(h, hidden, &anchor, 0, &options()).expect("direct reference");
    let r = verify_next_token(h, hidden, &anchor, 0, &options()).expect("groups");
    let points = vertices(hidden.lo, hidden.hi);
    for evidence in [&dense.evidence, &r.evidence] {
        for group in &evidence.groups {
            let bound = BigRational::from_float(group.upper).unwrap();
            for &id in &group.members {
                for point in &points {
                    assert!(
                        original_margin(h, point, 0, id) <= bound,
                        "group {:?}, competitor {id}",
                        group.members
                    );
                }
            }
        }
    }
    assert!(dense.is_certified());
    assert!(r.is_certified());
    assert!(
        replay_next_token(h, hidden, &anchor, 0, &r.evidence, &options())
            .expect("replay")
            .is_certified()
    );
}

#[test]
fn decoder_d2_sequential_binary64_contract_covers_cancellation_and_binds_scope() {
    let weights = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let bias = [0.0, 0.25];
    let mut h = head(2, 3, &weights, &bias);
    h.arithmetic = HeadArithmetic::SequentialBinary64;
    let point = [1e16, 1.0, -1e16];
    let hidden = HiddenBox {
        lo: &point,
        hi: &point,
    };
    let r = verify_next_token(h, hidden, &point, 0, &options()).expect("execution enclosure");
    assert!(!r.is_certified());
    let mut target = bias[0];
    for (&w, &x) in h.row(0).iter().zip(&point) {
        target += w * x;
    }
    let observed = bias[1] - target;
    assert!(observed > 0.0);
    assert!(numeric::dense_upper(h, hidden, 0, 1).expect("bound") >= observed);
    let mut reference = h;
    reference.arithmetic = HeadArithmetic::ExactReal;
    assert!(replay_next_token(reference, hidden, &point, 0, &r.evidence, &options()).is_err());
}

#[test]
fn decoder_d2_json_preserves_exact_outward_group_summaries() {
    // Directed operations produce adjacent/subnormal floats. A decimal parser
    // with approximate rounding must not change independent replay evidence.
    let weights = vec![0.0; 256 * 2];
    let mut bias = vec![0.0; 256];
    bias[17] = 1.0;
    let h = head(256, 2, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-1.0, -0.25],
        hi: &[0.5, 2.0],
    };
    let opts = options();
    let report = verify_next_token(h, hidden, &[0.0, 0.0], 17, &opts).expect("full cover");
    let json = report.evidence.to_json_v1().expect("v1 export");
    let parsed: VocabEvidence = serde_json::from_str(&json).expect("parse");
    assert_eq!(
        report.evidence, parsed,
        "all checked summaries must survive JSON exactly"
    );
    assert!(
        replay_next_token_json(h, hidden, &[0.0, 0.0], 17, json.as_bytes(), &opts)
            .expect("original group summary replay")
            .is_certified()
    );
}

#[test]
fn decoder_d2_json_evidence_is_untrusted_and_admitted_before_parse() {
    let weights = [0.0, 1.0];
    let bias = [2.0, 0.0];
    let h = head(2, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.1],
        hi: &[0.1],
    };
    let opts = options();
    let r = verify_next_token(h, hidden, &[0.0], 0, &opts).expect("certificate");
    let json = r.evidence.to_json().expect("export");
    assert!(
        replay_next_token_json(h, hidden, &[0.0], 0, json.as_bytes(), &opts)
            .expect("JSON replay")
            .is_certified()
    );
    let tiny = VocabOptions {
        max_evidence_bytes: 3,
        ..opts.clone()
    };
    assert!(matches!(
        replay_next_token_json(h, hidden, &[0.0], 0, json.as_bytes(), &tiny),
        Err(DecoderRefusal::Resource(_))
    ));
    let value = json.replace("\"schema_version\":2", "\"schema_version\":99");
    assert!(replay_next_token_json(h, hidden, &[0.0], 0, value.as_bytes(), &opts).is_err());
}

#[test]
fn decoder_d2_shared_budget_lifetime_clones_and_late_proposals_preserve_certificates() {
    let budget =
        WorkBudget::new(None, 1 << 20, 16 << 20, 1 << 24, 32, Duration::ZERO).expect("budget");
    let opts = VocabOptions {
        budget: Some(budget.clone()),
        ..options()
    };
    let weights = [0.0, 1.0];
    let bias = [2.0, 0.0];
    let h = head(2, 1, &weights, &bias);
    let tree = prepare_vocab_tree(h, &[0.0], 0, &opts).expect("prepare");
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    let r = tree
        .verify(
            HiddenBox {
                lo: &[-0.1],
                hi: &[0.1],
            },
            &opts,
        )
        .expect("query");
    let NextTokenOutcome::Certified(cert) = &r.outcome else {
        panic!("certificate");
    };
    let retained = cert.clone();
    let late = VocabOptions {
        deadline: Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        ..opts.clone()
    };
    assert!(matches!(
        tree.verify(
            HiddenBox {
                lo: &[-0.1],
                hi: &[0.1]
            },
            &late
        ),
        Err(DecoderRefusal::Deadline)
    ));
    assert!(retained.maximum_competitor_upper().unwrap() < 0.0);
    drop(r);
    drop(tree);
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn decoder_d2_shape_finiteness_overflow_and_resource_admission_fail_closed() {
    let w = [0.0, 1.0];
    let b = [2.0, 0.0];
    let h = head(2, 1, &w, &b);
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    for opts in [
        VocabOptions {
            max_persistent_bytes: 1,
            ..options()
        },
        VocabOptions {
            max_terms: 0,
            ..options()
        },
        VocabOptions {
            max_hidden: 0,
            ..options()
        },
        VocabOptions {
            max_nodes: 0,
            ..options()
        },
    ] {
        assert!(verify_next_token(h, hidden, &[0.0], 0, &opts).is_err());
    }
    assert!(verify_next_token(
        h,
        HiddenBox {
            lo: &[1.0],
            hi: &[-1.0]
        },
        &[0.0],
        0,
        &options()
    )
    .is_err());
    assert!(verify_next_token(h, hidden, &[f64::NAN], 0, &options()).is_err());
    assert!(verify_next_token(
        head(2, 1, &[0.0, f64::INFINITY], &b),
        hidden,
        &[0.0],
        0,
        &options()
    )
    .is_err());
    assert!(verify_next_token(
        head(usize::MAX, 2, &[], &[]),
        hidden,
        &[0.0, 0.0],
        0,
        &options()
    )
    .is_err());
    let starved = WorkBudget::new(None, 1 << 20, 1 << 20, 0, 10, Duration::ZERO).unwrap();
    let opts = VocabOptions {
        budget: Some(starved),
        ..options()
    };
    assert!(matches!(
        verify_next_token(h, hidden, &[0.0], 0, &opts),
        Err(DecoderRefusal::Budget(WorkRefusal::WorkLimit))
    ));
}

#[test]
fn decoder_d2_cached_warm_group_query_reduces_fixed_work_against_dense_reference() {
    let vocab = 257;
    let dim = 8;
    let mut weights = vec![1.0; vocab * dim];
    weights[..dim].fill(0.0);
    let mut bias = vec![0.0; vocab];
    bias[0] = 100.0;
    let h = head(vocab, dim, &weights, &bias);
    let lo = vec![-1.0; dim];
    let hi = vec![1.0; dim];
    let anchor = vec![0.0; dim];
    let opts = VocabOptions::default();
    let tree = prepare_vocab_tree(h, &anchor, 0, &opts).expect("charged cold preparation");
    let hidden = HiddenBox { lo: &lo, hi: &hi };
    let dense = verify_dense_next_token(h, hidden, &anchor, 0, &opts).expect("dense reference");
    let warm = tree.verify(hidden, &opts).expect("warm query");
    assert!(dense.is_certified() && warm.is_certified());
    assert_eq!(warm.stats.groups_visited, 1);
    assert_eq!(warm.stats.dense_rows, 0);
    assert!(warm.stats.query_terms < dense.stats.preprocessing_terms);
    assert!(warm.stats.preprocessing_terms > 0 && warm.stats.persistent_bytes > 0);
    let replay = replay_next_token(h, hidden, &anchor, 0, &warm.evidence, &opts)
        .expect("full original-head replay");
    assert!(replay.is_certified());
    assert!(replay.stats.query_terms >= vocab as u64 * dim as u64);
    eprintln!("[decoder-d2] cold_terms={} warm_terms={} dense_candidate_terms={} original_replay_terms={} resident_bytes={}",
        warm.stats.preprocessing_terms,warm.stats.query_terms,dense.stats.preprocessing_terms,replay.stats.query_terms,warm.stats.persistent_bytes);
}

#[test]
fn decoder_d2_replay_and_dense_fallback_can_use_final_certificate_reserve() {
    let weights = [0.0, 1.0];
    let bias = [2.0, 0.0];
    let h = head(2, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.1],
        hi: &[0.1],
    };
    let opts = options();
    let evidence = verify_next_token(h, hidden, &[0.0], 0, &opts)
        .unwrap()
        .evidence;
    // No sleeps: optional work is already late, final proof work is not.
    let budget = WorkBudget::new(
        Some(Instant::now() + Duration::from_secs(60)),
        1 << 20,
        16 << 20,
        1 << 24,
        32,
        Duration::from_secs(120),
    )
    .unwrap();
    let opts = VocabOptions {
        budget: Some(budget),
        ..opts
    };
    assert!(matches!(
        prepare_vocab_tree(h, &[0.0], 0, &opts),
        Err(DecoderRefusal::Budget(WorkRefusal::Deadline))
    ));
    assert!(replay_next_token(h, hidden, &[0.0], 0, &evidence, &opts)
        .unwrap()
        .is_certified());
    assert!(verify_dense_next_token(h, hidden, &[0.0], 0, &opts)
        .unwrap()
        .is_certified());
}

#[test]
fn decoder_d2_replay_charges_evidence_spare_capacities_before_cloning() {
    let weights = [0.0, 1.0];
    let bias = [2.0, 0.0];
    let h = head(2, 1, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.1],
        hi: &[0.1],
    };
    let opts = options();
    let mut evidence = verify_next_token(h, hidden, &[0.0], 0, &opts)
        .unwrap()
        .evidence;
    evidence.groups[0].members.reserve_exact(10_000);
    let limited = VocabOptions {
        max_persistent_bytes: 16 << 10,
        ..opts
    };
    assert!(matches!(
        replay_next_token(h, hidden, &[0.0], 0, &evidence, &limited),
        Err(DecoderRefusal::Resource(_))
    ));
}

fn assert_model_tree_cover(report: &ModelTreeReport, vocab: usize, target: usize) {
    let mut ids: Vec<usize> = report
        .report
        .evidence
        .groups
        .iter()
        .flat_map(|group| {
            assert!(!group.members.contains(&target));
            assert_ne!(group.anchor_token, target);
            assert!(group.members.contains(&group.anchor_token));
            group.members.iter().copied()
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..vocab).filter(|&id| id != target).collect::<Vec<_>>()
    );
}

#[test]
fn decoder_n2_changing_targets_reuse_geometry_with_complete_original_replay() {
    // Each row j uniquely wins near x=2j, without a target-specific tree.
    let vocab = 35;
    let weights: Vec<f64> = (0..vocab).map(|j| j as f64).collect();
    let bias: Vec<f64> = (0..vocab).map(|j| -(j as f64).powi(2)).collect();
    let h = head(vocab, 1, &weights, &bias);
    let opts = VocabOptions {
        leaf_size: 4,
        ..VocabOptions::default()
    };
    let tree = prepare_model_vocab_tree(h, &opts).expect("model-level cold preparation");
    let cold = tree.preparation_stats().clone();
    assert!(cold.geometry_terms > vocab as u64);
    for target in [0, 1, 17, 34] {
        let anchor = [2.0 * target as f64];
        let lo = [anchor[0] - 0.01];
        let hi = [anchor[0] + 0.02];
        let hidden = HiddenBox { lo: &lo, hi: &hi };
        let report = tree
            .verify(hidden, &anchor, target, &opts)
            .expect("changing target");
        assert!(report.is_certified());
        assert_model_tree_cover(&report, vocab, target);
        assert!(report.stats.cached_geometry_reuses > 0);
        assert!(report.stats.subset_geometry_terms <= (opts.leaf_size - 1) as u64);
        assert!(report.stats.subset_geometry_terms < cold.geometry_terms);
        assert_eq!(report.stats.anchor_terms, (vocab - 1) as u64);
        assert!(report.stats.coverage_terms >= (vocab - 1) as u64);
        let dense =
            verify_dense_next_token(h, hidden, &anchor, target, &opts).expect("dense oracle");
        assert!(dense.is_certified());
        let replay = tree
            .replay(hidden, &anchor, target, &report.report.evidence, &opts)
            .expect("independent original-head replay");
        assert!(replay.is_certified());
        assert!(replay.stats.query_terms >= vocab as u64);
        assert_eq!(tree.preparation_stats(), &cold);
        eprintln!(
            "[decoder-n2] target={target} cold_geometry={} warm_subset_geometry={} anchor_terms={} cached_reuses={} dense_terms={} cover_terms={} replay_terms={}",
            cold.geometry_terms, report.stats.subset_geometry_terms, report.stats.anchor_terms,
            report.stats.cached_geometry_reuses, report.stats.dense_terms,
            report.stats.coverage_terms, replay.stats.query_terms
        );
    }
}

#[test]
fn decoder_n2_target_anchor_and_asymmetric_boxes_do_not_contaminate_geometry() {
    let weights = [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let bias = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let h = head(9, 1, &weights, &bias);
    let opts = VocabOptions {
        leaf_size: 3,
        ..options()
    };
    let tree = prepare_model_vocab_tree(h, &opts).unwrap();
    let hidden = HiddenBox {
        lo: &[1.0],
        hi: &[2.0],
    };
    // Target 0 is the cold root anchor and leftmost leaf anchor. Every emitted
    // geometry excludes that enormous target row, even far from the box center.
    for anchor in [1.8, 100.0] {
        let r = tree.verify(hidden, &[anchor], 0, &opts).unwrap();
        assert!(r.is_certified());
        assert_model_tree_cover(&r, 9, 0);
        for group in &r.report.evidence.groups {
            let g = group
                .group_bounds
                .as_ref()
                .expect("zero competitor residual");
            assert!(g.radius_l1 < 1e-10 && g.radius_linf < 1e-10);
        }
        assert!(tree
            .replay(hidden, &[anchor], 0, &r.report.evidence, &opts)
            .unwrap()
            .is_certified());
    }
}

#[test]
fn decoder_n2_cached_bounds_dominate_exact_rational_float_coefficient_vertices() {
    let weights = [
        1e9,
        0.1,
        -0.2,
        1e9 + 0.25,
        -0.3,
        0.7,
        1e9 - 0.125,
        0.9,
        -0.125,
        1e9 + 0.5,
        0.7,
        0.2,
        1e9 - 0.3,
        0.2,
        0.9,
    ];
    let bias = [3.0, 0.0, 0.25, -0.1, 0.7];
    let h = head(5, 3, &weights, &bias);
    let hidden = HiddenBox {
        lo: &[-0.75, 0.2, -1.3],
        hi: &[0.4, 1.1, 0.9],
    };
    let opts = options();
    let tree = prepare_model_vocab_tree(h, &opts).unwrap();
    for target in 0..5 {
        let anchor = [0.2, 0.8, -0.4];
        let r = tree.verify(hidden, &anchor, target, &opts).unwrap();
        assert_model_tree_cover(&r, 5, target);
        let dense = verify_dense_next_token(h, hidden, &anchor, target, &opts).unwrap();
        assert_eq!(r.is_certified(), dense.is_certified());
        for group in &r.report.evidence.groups {
            let upper = BigRational::from_float(group.upper).unwrap();
            for &id in &group.members {
                for point in vertices(hidden.lo, hidden.hi) {
                    assert!(original_margin(h, &point, target, id) <= upper);
                }
            }
        }
        let replay = tree
            .replay(hidden, &anchor, target, &r.report.evidence, &opts)
            .unwrap();
        assert_eq!(r.is_certified(), replay.is_certified());
    }
}

#[test]
fn decoder_n2_duplicate_rows_biases_ties_and_unsafe_targets_remain_complete() {
    let weights = [0.1; 7];
    let bias = [0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.25];
    let h = head(7, 1, &weights, &bias);
    let opts = options();
    let tree = prepare_model_vocab_tree(h, &opts).unwrap();
    let hidden = HiddenBox {
        lo: &[-2.0],
        hi: &[3.0],
    };
    let tied_and_unsafe = tree.verify(hidden, &[0.4], 0, &opts).unwrap();
    assert!(!tied_and_unsafe.is_certified());
    assert_model_tree_cover(&tied_and_unsafe, 7, 0);
    assert!(tied_and_unsafe.report.unresolved.contains(&1));
    assert!(tied_and_unsafe.report.unresolved.contains(&6));
    let winner = tree.verify(hidden, &[0.4], 6, &opts).unwrap();
    assert!(winner.is_certified());
    assert_model_tree_cover(&winner, 7, 6);
    assert!(tree
        .replay(hidden, &[0.4], 6, &winner.report.evidence, &opts)
        .unwrap()
        .is_certified());
    let single = prepare_model_vocab_tree(head(1, 1, &[0.1], &[0.0]), &opts).unwrap();
    let r = single.verify(hidden, &[100.0], 0, &opts).unwrap();
    assert!(r.is_certified() && r.report.evidence.groups.is_empty());
}

#[test]
fn decoder_n2_coverage_poison_and_changed_scope_refuse_independent_replay() {
    let weights = [0.0; 9];
    let bias = [2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let h = head(9, 1, &weights, &bias);
    let opts = options();
    let tree = prepare_model_vocab_tree(h, &opts).unwrap();
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let r = tree.verify(hidden, &[0.25], 0, &opts).unwrap();
    assert!(r.is_certified());
    for poison in 0..5 {
        let mut e = r.report.evidence.clone();
        match poison {
            0 => {
                e.groups[0].members[0] = 0;
            }
            1 => {
                let duplicated = e.groups[0].members[0];
                e.groups[1].members[0] = duplicated;
            }
            2 => {
                e.groups.pop();
            }
            3 => {
                e.groups[0].anchor_token = 0;
            }
            _ => {
                e.groups[0].group_bounds.as_mut().unwrap().radius_linf = 123.0;
            }
        }
        assert!(
            tree.replay(hidden, &[0.25], 0, &e, &opts).is_err(),
            "poison {poison}"
        );
    }
    assert!(tree
        .replay(hidden, &[0.5], 0, &r.report.evidence, &opts)
        .is_err());
    assert!(tree
        .replay(
            HiddenBox {
                lo: hidden.lo,
                hi: &[2.0]
            },
            &[0.25],
            0,
            &r.report.evidence,
            &opts
        )
        .is_err());
    assert!(tree
        .replay(hidden, &[0.25], 1, &r.report.evidence, &opts)
        .is_err());
    let mut changed_bias = bias;
    changed_bias[8] = 3.0;
    let changed = head(9, 1, &weights, &changed_bias);
    assert!(replay_next_token(changed, hidden, &[0.25], 0, &r.report.evidence, &opts).is_err());
    let mut changed_arithmetic = h;
    changed_arithmetic.arithmetic = HeadArithmetic::SequentialBinary64;
    assert!(replay_next_token(
        changed_arithmetic,
        hidden,
        &[0.25],
        0,
        &r.report.evidence,
        &opts
    )
    .is_err());
    let changed_tree = prepare_model_vocab_tree(changed, &opts).unwrap();
    assert_ne!(tree.head_identity(), changed_tree.head_identity());
}

#[test]
fn decoder_n2_admission_and_cross_query_residency_preserve_prior_certificates() {
    let weights = [0.0; 9];
    let bias = [2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let h = head(9, 1, &weights, &bias);
    let budget = WorkBudget::new(None, 1 << 20, 16 << 20, 1 << 28, 64, Duration::ZERO).unwrap();
    let opts = VocabOptions {
        budget: Some(budget.clone()),
        ..options()
    };
    let tree = prepare_model_vocab_tree(h, &opts).unwrap();
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        tree.reserved_bytes()
    );
    let hidden = HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    };
    let r = tree.verify(hidden, &[0.0], 0, &opts).unwrap();
    let NextTokenOutcome::Certified(cert) = &r.report.outcome else {
        panic!("certificate");
    };
    let retained = cert.clone();
    let resident = budget.receipt().unwrap().live_persistent_bytes;
    assert_eq!(resident, tree.reserved_bytes() + r.report.reserved_bytes());
    for cap in [
        VocabOptions {
            max_scratch_bytes: 1,
            ..opts.clone()
        },
        VocabOptions {
            max_persistent_bytes: 1,
            ..opts.clone()
        },
        VocabOptions {
            max_terms: 0,
            ..opts.clone()
        },
        VocabOptions {
            max_nodes: 1,
            ..opts.clone()
        },
        VocabOptions {
            max_evidence_bytes: 1,
            ..opts.clone()
        },
        VocabOptions {
            deadline: Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            ..opts.clone()
        },
    ] {
        assert!(tree.verify(hidden, &[0.0], 0, &cap).is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, resident);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
    assert!(tree.verify(hidden, &[f64::NAN], 0, &opts).is_err());
    assert!(tree.verify(hidden, &[0.0], 9, &opts).is_err());
    assert!(prepare_model_vocab_tree(
        h,
        &VocabOptions {
            max_persistent_bytes: 1,
            ..options()
        }
    )
    .is_err());
    assert!(prepare_model_vocab_tree(head(9, 1, &[f64::NAN; 9], &bias), &options()).is_err());
    drop(r);
    drop(tree);
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    assert!(retained.maximum_competitor_upper().unwrap() < 0.0);
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}
