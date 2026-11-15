// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn affine(bf: f64, bg: f64) -> Stage {
    Stage::Affine {
        rows: 1,
        cols: 1,
        wf: vec![1.0],
        wg: vec![1.0],
        bf: vec![bf],
        bg: vec![bg],
    }
}

fn chain(shift: f64, output_bias: f64) -> PairedChain {
    PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            affine(shift, 0.0),
            Stage::Relu { width: 1 },
            affine(0.0, 0.0),
            Stage::Relu { width: 1 },
            affine(output_bias, 0.0),
        ],
    }
}

fn domain() -> BoxDomain {
    BoxDomain {
        lo: vec![-1.0],
        hi: vec![1.0],
    }
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn worker_limit_is_typed_and_bounded() {
    assert!(ParallelWorkers::new(0).is_none());
    assert!(ParallelWorkers::new(17).is_none());
    for n in 1..=16 {
        assert_eq!(ParallelWorkers::new(n).unwrap().get(), n);
    }
}

#[test]
fn every_worker_count_closes_certified_root_without_open_cover() {
    assert!(suitable_fp_environment());
    let c = chain(1.0 / 1024.0, 0.0);
    for n in [1, 2, 4, 8, 16] {
        let r = c.prove_parallel(
            &domain(),
            1.0 / 512.0,
            deadline(),
            1,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        assert!(r.proved && r.closed == 1 && r.open == 0 && r.evaluated == 1);
        assert_eq!(r.reason, "covered");
    }
}

#[test]
fn unsafe_bias_and_global_cap_retain_the_exact_unclosed_frontier() {
    let c = chain(0.0, 0.5);
    for n in [1, 2, 4, 8, 16] {
        let r = c.prove_parallel(
            &domain(),
            0.125,
            deadline(),
            31,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        // Every visited node has a real violation and splits into BOTH
        // children. After 31 split nodes, exactly 32 leaves remain open,
        // regardless of worker scheduling or partial final batch size.
        assert!(
            !r.proved && r.closed == 0 && r.open == 32 && r.evaluated == 31,
            "workers {n}: {r:?}"
        );
        assert_eq!(r.reason, "domain_cap");
    }
}

#[test]
fn closing_safe_siblings_does_not_exhaust_cover_of_a_violating_sibling() {
    let shift = 1.0 / 1024.0;
    let c = chain(shift, 0.0);
    // The first split has a wholly inactive left child, and a right child
    // containing the real violation. This checks cover regardless of batch
    // traversal order without assuming a heuristic visits grandchildren.
    let root = BoxDomain {
        lo: vec![-1.0],
        hi: vec![0.5],
    };
    for n in [1, 2, 4, 16] {
        let r = c.prove_parallel(
            &root,
            shift / 2.0,
            deadline(),
            128,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        assert!(
            !r.proved && r.open > 0 && r.closed > 0 && r.evaluated <= 128,
            "workers {n}: {r:?}"
        );
    }
}

#[test]
fn one_worker_preserves_serial_dfs_cap_accounting_and_result() {
    let c = chain(1.0 / 1024.0, 0.0);
    let serial = c.prove(&domain(), 1.0 / 512.0, deadline(), 17, false);
    let parallel = c.prove_parallel(
        &domain(),
        1.0 / 512.0,
        deadline(),
        17,
        false,
        ParallelWorkers::new(1).unwrap(),
    );
    assert_eq!(serial.proved, parallel.proved);
    assert_eq!(serial.evaluated, parallel.evaluated);
    assert_eq!(serial.closed, parallel.closed);
    assert_eq!(serial.open, parallel.open);
    assert_eq!(serial.max_depth, parallel.max_depth);
}

#[test]
fn expired_deadline_and_zero_cap_never_prove_or_dispatch_domains() {
    let c = chain(0.0, 0.0);
    for n in [1, 4, 16] {
        let workers = ParallelWorkers::new(n).unwrap();
        let expired = c.prove_parallel(&domain(), 1.0, Instant::now(), 1000, true, workers);
        assert!(!expired.proved && expired.open == 1 && expired.evaluated == 0);
        assert_eq!(expired.reason, "deadline");
        let capped = c.prove_parallel(&domain(), 1.0, deadline(), 0, true, workers);
        assert!(!capped.proved && capped.open == 1 && capped.evaluated == 0);
    }
}

#[test]
fn refused_point_and_numeric_bound_cannot_manufacture_empty_cover() {
    let c = chain(0.0, 0.5);
    let point = BoxDomain {
        lo: vec![0.0],
        hi: vec![0.0],
    };
    for n in [1, 4, 16] {
        let r = c.prove_parallel(
            &point,
            0.125,
            deadline(),
            1000,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        assert!(!r.proved && r.open == 1 && r.evaluated == 1 && r.reason == "point_unclosed");
    }
    let huge = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![Stage::Affine {
            rows: 1,
            cols: 1,
            wf: vec![1e100],
            wg: vec![1e100],
            bf: vec![0.0],
            bg: vec![0.0],
        }],
    };
    let point = BoxDomain {
        lo: vec![1e100],
        hi: vec![1e100],
    };
    let r = huge.prove_parallel(
        &point,
        1.0,
        deadline(),
        4,
        true,
        ParallelWorkers::new(4).unwrap(),
    );
    assert!(!r.proved && r.open == 1 && r.reason == "bound_refused");
}

#[test]
fn merge_order_and_global_cap_are_repeatable_for_fixed_workers() {
    let c = chain(1.0 / 1024.0, 0.0);
    let workers = ParallelWorkers::new(4).unwrap();
    let first = c.prove_parallel(&domain(), 1.0 / 2048.0, deadline(), 63, true, workers);
    for _ in 0..3 {
        let r = c.prove_parallel(&domain(), 1.0 / 2048.0, deadline(), 63, true, workers);
        assert_eq!(
            (
                r.proved,
                r.evaluated,
                r.closed,
                r.open,
                r.max_depth,
                r.reason
            ),
            (
                first.proved,
                first.evaluated,
                first.closed,
                first.open,
                first.max_depth,
                first.reason
            )
        );
    }
}

#[test]
fn every_fourth_depth_uses_root_relative_width_and_ignores_zero_root_axes() {
    let root = BoxDomain {
        lo: vec![-100.0, -1.0, 7.0],
        hi: vec![100.0, 1.0, 7.0],
    };
    let domain = BoxDomain {
        lo: vec![-100.0, -1.0, 7.0],
        hi: vec![-99.0, 1.0, 7.0],
    };
    let bounds = BoundReport {
        lower: vec![0.0],
        upper: vec![1.0],
        split_scores: vec![1e9, 1.0, 1e30],
    };
    for depth in 0..12 {
        assert_eq!(
            split_axis(&domain, &root, &bounds, depth),
            Ok(if depth % 4 == 3 { 1 } else { 0 })
        );
    }
    // Relative widths, rather than absolute units, select axis one even
    // though the root span of axis zero is one hundred times larger.
    let equal_fractions = BoxDomain {
        lo: vec![-100.0, -1.0, 7.0],
        hi: vec![0.0, 0.0, 7.0],
    };
    assert_eq!(split_axis(&equal_fractions, &root, &bounds, 3), Ok(1));
}

#[test]
fn exhausted_gradient_axis_falls_back_to_an_interior_midpoint_axis() {
    let root = BoxDomain {
        lo: vec![1.0, 0.0, 9.0],
        hi: vec![2.0, 2.0, 9.0],
    };
    let mut domain = BoxDomain {
        lo: vec![1.0, 0.0, 9.0],
        hi: vec![1.0_f64.next_up(), 2.0, 9.0],
    };
    let bounds = BoundReport {
        lower: vec![0.0],
        upper: vec![1.0],
        split_scores: vec![1e20, 1.0, 1e30],
    };
    for depth in 0..8 {
        assert_eq!(split_axis(&domain, &root, &bounds, depth), Ok(1));
    }
    domain.hi[1] = domain.lo[1];
    assert_eq!(split_axis(&domain, &root, &bounds, 0), Err("unsplittable"));
    domain.hi[0] = domain.lo[0];
    assert_eq!(
        split_axis(&domain, &root, &bounds, 0),
        Err("point_unclosed")
    );
}

#[test]
fn refined_workers_close_a_safe_root_with_total_elapsed_accounting() {
    let c = chain(1.0 / 1024.0, 0.0);
    for n in [1, 4, 16] {
        let r = c.prove_refined_parallel(
            &domain(),
            1.0 / 512.0,
            deadline(),
            1,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        assert!(
            r.proved && r.closed == 1 && r.open == 0 && r.evaluated == 1,
            "{r:?}"
        );
        assert_eq!(r.reason, "covered");
        assert!(r.elapsed > Duration::ZERO);
    }
}

#[test]
fn refined_search_cannot_prove_an_unsafe_bias_or_discard_capped_children() {
    let c = chain(0.0, 0.5);
    for n in [1, 4, 16] {
        let r = c.prove_refined_parallel(
            &domain(),
            0.125,
            deadline(),
            31,
            true,
            ParallelWorkers::new(n).unwrap(),
        );
        assert!(
            !r.proved && r.closed == 0 && r.open == 32 && r.evaluated == 31,
            "{r:?}"
        );
        assert_eq!(r.reason, "domain_cap");
    }
}

#[test]
fn expired_refused_and_zero_budget_refinement_retain_the_original_root() {
    let workers = ParallelWorkers::new(4).unwrap();
    let c = chain(0.0, 0.0);
    let expired = c.prove_refined_parallel(&domain(), 1.0, Instant::now(), 4, true, workers);
    assert!(!expired.proved && expired.open == 1 && expired.evaluated == 0);
    assert_eq!(expired.reason, "deadline");
    let capped = c.prove_refined_parallel(&domain(), 1.0, deadline(), 0, true, workers);
    assert!(!capped.proved && capped.open == 1 && capped.evaluated == 0);
    let huge = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![Stage::Affine {
            rows: 1,
            cols: 1,
            wf: vec![1e100],
            wg: vec![1e100],
            bf: vec![0.0],
            bg: vec![0.0],
        }],
    };
    let point = BoxDomain {
        lo: vec![1e100],
        hi: vec![1e100],
    };
    let refused = huge.prove_refined_parallel(&point, 1.0, deadline(), 4, true, workers);
    assert!(!refused.proved && refused.open == 1 && refused.evaluated == 0);
    assert_eq!(refused.reason, "refinement_refused");
}

#[test]
fn public_refined_search_imports_no_reference_and_rebuilds_for_each_source_root() {
    // The public entry accepts the program, root and budgets, and no caller-
    // authored enclosure. Its reference is private and local to this call.
    let entry: fn(
        &PairedChain,
        &BoxDomain,
        f64,
        Instant,
        u64,
        bool,
        ParallelWorkers,
    ) -> SearchReport = PairedChain::prove_refined_parallel;
    let workers = ParallelWorkers::new(4).unwrap();
    let shift = 1.0 / 1024.0;
    let c = chain(shift, 0.0);
    let inactive = BoxDomain {
        lo: vec![-1.0],
        hi: vec![-0.5],
    };
    let active = BoxDomain {
        lo: vec![0.5],
        hi: vec![1.0],
    };
    let safe = entry(&c, &inactive, shift / 2.0, deadline(), 1, true, workers);
    assert!(safe.proved);
    // Reusing the first call's zero reference here would manufacture a false
    // closure. The new reference must come from the second, violating root.
    let unsafe_root = entry(&c, &active, shift / 2.0, deadline(), 1, true, workers);
    assert!(!unsafe_root.proved && unsafe_root.open == 2 && unsafe_root.evaluated == 1);
    let other_source = chain(0.0, 0.5);
    let unsafe_source = entry(
        &other_source,
        &inactive,
        0.125,
        deadline(),
        1,
        true,
        workers,
    );
    assert!(!unsafe_source.proved && unsafe_source.open == 2 && unsafe_source.evaluated == 1);
}
