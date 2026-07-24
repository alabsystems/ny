// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::beta_crown::bab_cuts::{CutKind, CutMetadata, CutTerm, CuttingPlane};
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::{BoundedTensor, Network};
use ndarray::{arr1, arr2, Array1};

fn make_test_network(layers: Vec<Layer>) -> Network {
    Network {
        layers,
        gpu_crown_cache: std::sync::Mutex::new(None),
    }
}

fn make_cut(terms: Vec<(usize, usize, f32)>, bias: f32) -> CuttingPlane {
    CuttingPlane {
        terms: terms
            .into_iter()
            .map(|(layer, neuron, coeff)| CutTerm {
                layer_idx: layer,
                neuron_idx: neuron,
                coefficient: coeff,
            })
            .collect(),
        bias,
        lambda: 0.01,
        lambda_grad: 0.0,
        lambda_m: 0.0,
        lambda_v: 0.0,
        source_depth: 2,
        metadata: CutMetadata::new(0, CutKind::Verified),
    }
}

fn build_two_relu_network_with_bounds() -> (Network, Vec<BoundedTensor>) {
    let linear1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1_f32, -0.1])),
    )
    .unwrap();
    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.3], [0.3, 1.0]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();

    let network = make_test_network(vec![
        Layer::Linear(linear1),
        Layer::ReLU(ReLULayer),
        Layer::Linear(linear2),
        Layer::ReLU(ReLULayer),
    ]);

    let bounds_linear1 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -0.5]).into_dyn(),
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();
    let bounds_relu1 = BoundedTensor::new(
        Array1::from_vec(vec![0.0, 0.0]).into_dyn(),
        Array1::from_vec(vec![1.0, 0.5]).into_dyn(),
    )
    .unwrap();
    let bounds_linear2 = BoundedTensor::new(
        Array1::from_vec(vec![-0.8, -0.3]).into_dyn(),
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();
    let bounds_relu2 = BoundedTensor::new(
        Array1::from_vec(vec![0.0, 0.0]).into_dyn(),
        Array1::from_vec(vec![0.8, 0.3]).into_dyn(),
    )
    .unwrap();

    (
        network,
        vec![bounds_linear1, bounds_relu1, bounds_linear2, bounds_relu2],
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_sibling_cuts_basic() {
    // Cut A: z0 + z1 >= 1 (both active)
    // Cut B: z0 - z1 >= 0 (z0 active, z1 inactive - flipped sign)
    // Expected parent: z0 >= 0 (remove differing term z1, bias = 1 - 1 = 0)
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![(0, 0, 1.0), (0, 1, -1.0)], 0.0);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);

    let count = pool.merge_cuts();

    // Should merge into one parent cut
    assert_eq!(count, 1, "Expected 1 cut after merge, got {}", count);
    assert_eq!(pool.cuts[0].terms.len(), 1, "Parent cut should have 1 term");
    assert_eq!(pool.cuts[0].terms[0].neuron_idx, 0, "Should keep z0");
    assert!(
        pool.cuts[0].terms[0].coefficient > 0.0,
        "z0 should be positive"
    );
    // Parent bias should be original bias - 1 = 1.0 - 1.0 = 0.0
    assert!(
        (pool.cuts[0].bias - 0.0).abs() < 1e-6,
        "Parent bias should be 0.0, got {}",
        pool.cuts[0].bias
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_generate_proactive_cuts_returns_result_2998() {
    let (network, layer_bounds) = build_two_relu_network_with_bounds();
    let mut pool = CutPool::new(100);

    let generated = pool
        .generate_proactive_cuts(&network, &layer_bounds, 50)
        .expect("valid proactive cut generation should return Ok(count)");

    assert!(
        generated > 0,
        "expected proactive cuts for unstable neurons"
    );
    assert!(
        pool.cuts.iter().all(|cut| cut.source_depth == 0),
        "proactive cuts must be tagged with source_depth=0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_no_siblings() {
    // Two cuts that are not siblings (differ by more than one coefficient)
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![(0, 0, -1.0), (0, 1, -1.0)], -1.0);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);

    let count = pool.merge_cuts();

    // Should not merge (differ by two coefficients)
    assert_eq!(count, 2, "Expected 2 cuts (no merge), got {}", count);
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_exhausts_duplicate_single_term_siblings() {
    // Regression: the sibling index must retain every cut with a matching
    // signature. If it only remembers one sibling, one duplicate pair merges
    // away and the second pair gets stranded.
    let cut_a1 = make_cut(vec![(0, 0, 1.0)], 0.0);
    let cut_a2 = make_cut(vec![(0, 0, 1.0)], 0.0);
    let cut_b1 = make_cut(vec![(0, 0, -1.0)], -1.0);
    let cut_b2 = make_cut(vec![(0, 0, -1.0)], -1.0);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a1);
    pool.cuts.push(cut_a2);
    pool.cuts.push(cut_b1);
    pool.cuts.push(cut_b2);

    let count = pool.merge_cuts();

    assert_eq!(
        count, 0,
        "Expected all duplicate sibling pairs to merge away"
    );
    assert!(
        pool.cuts.is_empty(),
        "Expected no residual cuts after merging"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_empty_pool() {
    let mut pool = CutPool::new(10);
    let count = pool.merge_cuts();
    assert_eq!(count, 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_single_cut() {
    let cut = make_cut(vec![(0, 0, 1.0)], 0.5);
    let mut pool = CutPool::new(10);
    pool.cuts.push(cut);

    let count = pool.merge_cuts();
    assert_eq!(count, 1, "Single cut should remain unchanged");
}

#[ntest::timeout(10000)]
#[test]
fn test_prune_redundant_after_merge() {
    // Create cuts where after merge, some children become redundant
    // Cut A: z0 + z1 >= 1
    // Cut B: z0 - z1 >= 0
    // Cut C: z0 + z1 + z2 >= 2 (child of the parent z0)
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![(0, 0, 1.0), (0, 1, -1.0)], 0.0);
    let cut_c = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)], 2.0);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);
    pool.cuts.push(cut_c);

    let count = pool.merge_cuts();

    // A+B merge into parent with just z0
    // Parent (z0) is a parent of C (z0, z1, z2) since z0 is in C with same sign
    // Therefore C should be pruned as redundant
    // Result: only the parent cut remains
    assert_eq!(count, 1, "Expected exactly 1 cut after merge and prune");
    assert_eq!(
        pool.cuts[0].terms.len(),
        1,
        "Parent should have 1 term (z0)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_is_parent_of() {
    let parent = make_cut(vec![(0, 0, 1.0)], 0.0);
    let child = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], 1.0);
    let non_child = make_cut(vec![(0, 0, -1.0), (0, 1, 1.0)], 0.0);

    assert!(
        CutPool::is_parent_of(&parent, &child),
        "parent should be parent of child"
    );
    assert!(
        !CutPool::is_parent_of(&parent, &non_child),
        "parent should not be parent of non_child (different sign)"
    );
    assert!(
        !CutPool::is_parent_of(&child, &parent),
        "child should not be parent of parent"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cut_signature_ordering() {
    // Same terms in different order should produce same signature
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, -1.0)], 0.0);
    let cut_b = make_cut(vec![(0, 1, -1.0), (0, 0, 1.0)], 0.0);

    let sig_a = CutPool::cut_signature(&cut_a);
    let sig_b = CutPool::cut_signature(&cut_b);

    assert_eq!(
        sig_a, sig_b,
        "Signatures should match regardless of term order"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_iterative_merge_dedup_identical_parents() {
    // Test deduplication of identical parent cuts:
    // Cut A: z0 + z1 >= 1
    // Cut B: z0 - z1 >= 0
    // Cut C: z0 + z2 >= 1
    // Cut D: z0 - z2 >= 0
    // After merge: A+B -> parent(z0), C+D -> parent(z0)
    // Deduplication should keep only one parent cut
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], 1.0);
    let cut_b = make_cut(vec![(0, 0, 1.0), (0, 1, -1.0)], 0.0);
    let cut_c = make_cut(vec![(0, 0, 1.0), (0, 2, 1.0)], 1.0);
    let cut_d = make_cut(vec![(0, 0, 1.0), (0, 2, -1.0)], 0.0);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);
    pool.cuts.push(cut_c);
    pool.cuts.push(cut_d);

    let count = pool.merge_cuts();

    // A+B -> parent(z0), C+D -> parent(z0)
    // Deduplication should keep exactly 1 copy
    assert_eq!(count, 1, "Expected exactly 1 cut after merge and dedup");
    assert_eq!(
        pool.cuts[0].terms.len(),
        1,
        "Parent should have 1 term (z0)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_deduplicate_cuts() {
    // Direct test for deduplicate_cuts helper
    let cut_a = make_cut(vec![(0, 0, 1.0)], 0.0);
    let cut_b = make_cut(vec![(0, 0, 1.0)], 1.0); // Same terms, different bias
    let cut_c = make_cut(vec![(0, 1, 1.0)], 0.0); // Different term

    let cuts = vec![cut_a, cut_b, cut_c];
    let deduped = CutPool::deduplicate_cuts(cuts);

    // cut_a and cut_b have same signature, should keep only first
    assert_eq!(deduped.len(), 2, "Should have 2 unique cuts");
    assert_eq!(deduped[0].bias, 0.0, "First cut should be cut_a");
    assert_eq!(deduped[1].terms[0].neuron_idx, 1, "Second should be cut_c");
}

#[ntest::timeout(10000)]
#[test]
fn test_nan_lambda_cut_evicted_as_hard_stale_2598() {
    // A cut with NaN lambda must be eligible for hard-stale eviction.
    // Before the fix, `NaN.abs() < cut_lambda_min` returned false (IEEE 754),
    // making NaN-lambda cuts immortal in the pool.
    let mut pool = CutPool::new(1);
    // Shorten hard_stale threshold for the test.
    pool.cut_hard_stale_iters = 5;

    // Insert a NaN-lambda cut at iter 0.
    let mut nan_cut = make_cut(vec![(0, 0, 1.0)], 0.5);
    nan_cut.lambda = f32::NAN;
    nan_cut.metadata = CutMetadata::new(0, CutKind::NearMiss);
    pool.cuts.push(nan_cut);
    pool.cuts_live_by_kind[CutPool::kind_index(CutKind::NearMiss)] += 1;

    // Advance iter well past hard_stale threshold.
    pool.iter_counter.store(100, Ordering::Relaxed);

    // Adding a new cut should evict the NaN cut.
    let healthy_cut = make_cut(vec![(0, 1, 2.0)], 1.0);
    let added = pool.add_cut(healthy_cut);
    assert!(
        added,
        "healthy cut should be accepted after evicting NaN cut"
    );
    assert_eq!(
        pool.cuts.len(),
        1,
        "pool should have exactly 1 cut after eviction"
    );
    assert!(
        !pool.cuts[0].lambda.is_nan(),
        "remaining cut should be the healthy one (lambda={}, not NaN)",
        pool.cuts[0].lambda
    );
    assert!(
        pool.cuts_evicted_stale > 0,
        "eviction should be counted as stale eviction"
    );
}

/// Regression test for #3148: the create_parent_cut guard must reject
/// non-finite parent biases. We inject -Inf bias via direct struct push
/// (bypassing CuttingPlane::new validation) to verify the guard fires.
/// Without the guard, merge would propagate -Inf into the parent cut.
#[ntest::timeout(10000)]
#[test]
fn test_merge_f32_min_bias_no_overflow_3148() {
    // Use -Inf bias to actually trigger the non-finite guard.
    // Note: f32::MIN - 1.0 == f32::MIN (1.0 is below ULP), so f32::MIN
    // would NOT trigger the guard. -Inf - 1.0 == -Inf which IS non-finite.
    let cut_a = make_cut(vec![(0, 0, 1.0), (0, 1, 1.0)], f32::NEG_INFINITY);
    let cut_b = make_cut(vec![(0, 0, 1.0), (0, 1, -1.0)], f32::NEG_INFINITY);

    let mut pool = CutPool::new(10);
    pool.cuts.push(cut_a);
    pool.cuts.push(cut_b);

    let count = pool.merge_cuts();

    // The guard in create_parent_cut returns None for non-finite bias,
    // so no parent is added. Pool should contain no cuts with non-finite bias.
    for cut in &pool.cuts {
        assert!(
            cut.bias.is_finite(),
            "No cut should have non-finite bias after merge, got {}",
            cut.bias
        );
    }
    assert!(count <= 2, "Expected at most 2 cuts, got {}", count);
}
