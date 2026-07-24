// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for GPU BaB cached-lA compaction contracts.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::super::domain_conversion::processed_from_backward_results;
use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::GraphBabDomain;

fn make_tagged_cached_la(coeff: f32) -> CachedLinearBounds {
    let mut lower_a = HashMap::new();
    let mut upper_a = HashMap::new();
    let mut lower_b = HashMap::new();
    let mut upper_b = HashMap::new();
    lower_a.insert("node".to_string(), arr2(&[[coeff]]));
    upper_a.insert("node".to_string(), arr2(&[[coeff]]));
    lower_b.insert("node".to_string(), arr1(&[coeff]));
    upper_b.insert("node".to_string(), arr1(&[coeff]));
    CachedLinearBounds {
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    }
}

/// `processed_from_backward_results` consumes a dense kept-order cache vector
/// with no interior holes. This helper models the safe prefix contract the GPU
/// BaB compaction step must uphold before handing cached lA to domain
/// conversion: once a kept child is missing cache, all later kept caches must
/// be dropped to avoid misalignment.
fn compact_kept_cached_la_prefix(
    captured: Vec<Option<CachedLinearBounds>>,
    keep_mask: &[bool],
) -> Vec<Arc<CachedLinearBounds>> {
    let mut compacted = Vec::new();
    let mut saw_missing_kept_cache = false;

    for (cached_la, &kept) in captured.into_iter().zip(keep_mask.iter()) {
        if !kept {
            continue;
        }

        match cached_la {
            Some(cache) if !saw_missing_kept_cache => compacted.push(Arc::new(cache)),
            Some(_) => break,
            None => saw_missing_kept_cache = true,
        }
    }

    compacted
}

fn three_child_domains() -> (BoundedTensor, Vec<GraphBabDomain>) {
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
    let children = (0..3)
        .map(|_| GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input_bounds, false).unwrap())
        .collect();
    (input_bounds, children)
}

#[ntest::timeout(5000)]
#[test]
fn test_cached_la_compaction_contract_skips_refinement_verified_1916() -> Result<()> {
    let captured = vec![
        Some(make_tagged_cached_la(1.0)),
        Some(make_tagged_cached_la(2.0)),
        Some(make_tagged_cached_la(3.0)),
    ];
    let keep_mask = vec![true, false, true];
    let compacted = compact_kept_cached_la_prefix(captured, &keep_mask);
    let (_input_bounds, children) = three_child_domains();

    let processed = processed_from_backward_results(
        vec![HashMap::new(); 3],
        &children,
        &[-0.5, -0.1, -0.3],
        &[0.5, 0.1, 0.3],
        &keep_mask,
        &[],
        Some(compacted),
    )?;

    assert_eq!(processed.metadata.len(), 2);
    let la_a = processed.metadata[0].cached_la.as_ref().unwrap();
    let la_c = processed.metadata[1].cached_la.as_ref().unwrap();
    assert_eq!(la_a.lower_a["node"][[0, 0]], 1.0, "child A lA coeff");
    assert_eq!(la_c.lower_a["node"][[0, 0]], 3.0, "child C lA coeff");

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_cached_la_compaction_contract_truncates_after_missing_kept_cache_1916() -> Result<()> {
    let captured = vec![
        Some(make_tagged_cached_la(1.0)),
        None,
        Some(make_tagged_cached_la(3.0)),
    ];
    let keep_mask = vec![true, true, true];
    let compacted = compact_kept_cached_la_prefix(captured, &keep_mask);
    let (_input_bounds, children) = three_child_domains();

    assert_eq!(compacted.len(), 1, "only the aligned prefix should remain");
    assert_eq!(compacted[0].lower_a["node"][[0, 0]], 1.0);

    let processed = processed_from_backward_results(
        vec![HashMap::new(); 3],
        &children,
        &[-0.5, -0.1, -0.3],
        &[0.5, 0.1, 0.3],
        &keep_mask,
        &[],
        Some(compacted),
    )?;

    assert_eq!(processed.metadata.len(), 3);
    assert_eq!(
        processed.metadata[0].cached_la.as_ref().unwrap().lower_a["node"][[0, 0]],
        1.0,
        "first kept child retains aligned cache"
    );
    assert!(
        processed.metadata[1].cached_la.is_none(),
        "middle kept child should not receive a shifted cache"
    );
    assert!(
        processed.metadata[2].cached_la.is_none(),
        "later kept child should drop cache rather than inherit a misaligned one"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_cached_la_compaction_contract_preserves_identity_1916() -> Result<()> {
    let captured = vec![
        Some(make_tagged_cached_la(1.0)),
        Some(make_tagged_cached_la(2.0)),
        Some(make_tagged_cached_la(3.0)),
    ];
    let keep_mask = vec![true, true, true];
    let compacted = compact_kept_cached_la_prefix(captured, &keep_mask);
    let (_input_bounds, children) = three_child_domains();

    let processed = processed_from_backward_results(
        vec![HashMap::new(); 3],
        &children,
        &[-0.5, -0.1, -0.3],
        &[0.5, 0.1, 0.3],
        &keep_mask,
        &[],
        Some(compacted),
    )?;

    assert_eq!(processed.metadata.len(), 3);
    for (idx, expected) in [1.0_f32, 2.0, 3.0].into_iter().enumerate() {
        let la = processed.metadata[idx].cached_la.as_ref().unwrap();
        assert_eq!(la.lower_a["node"][[0, 0]], expected, "child {idx} lA coeff");
    }

    Ok(())
}
