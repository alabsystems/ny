// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::Axis;
use ny_core::{NyError, Result};

use crate::batched_domain::{ConstraintTuple, DomainMetadata, PickedDomains, ProcessedDomains};
use crate::beta_crown::config::BetaCrownConfig;

use super::shared::array_element_at;

/// Create two input-split child domains directly from `PickedDomains` batch.
///
/// For each domain at `idx`, bisects the selected input dimension at the midpoint:
/// - Left child: `input_uppers[split_dim] = midpoint`
/// - Right child: `input_lowers[split_dim] = midpoint`
///
/// Unlike ReLU splitting, input-split children:
/// - Do NOT inherit layer bounds (must be recomputed via fresh forward pass)
/// - Do NOT add ReLU constraints (constraints remain empty)
/// - Do NOT inherit alpha/beta state (invalidated by changed input bounds)
///
/// # Arguments
/// * `idx` - Index into the `PickedDomains` batch
/// * `picked` - The batched picked domains
/// * `split_dim` - Flat input dimension index to bisect
/// * `verify_upper` - Whether we verify upper bound (affects priority sign)
///
/// # Returns
/// `(left_child, right_child)` — either is `None` if the split dimension has zero width.
///
/// # Reference
/// - Design: `designs/2026-02-10-input-split-domain-list.md` Phase 1 §1.3
/// - alpha-beta-CROWN: `input_split/branching_domains.py:UnsortedInputDomainList`
/// - Issue: #1891
pub fn branch_input_split_from_picked(
    idx: usize,
    picked: &PickedDomains,
    split_dim: usize,
    verify_upper: bool,
) -> Result<(Option<ProcessedDomains>, Option<ProcessedDomains>)> {
    let metadata = picked.metadata.get(idx).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "branch_input_split_from_picked: metadata missing for idx {idx}"
        ))
    })?;

    let parent_input_lower = picked
        .input_lowers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();
    let parent_input_upper = picked
        .input_uppers
        .index_axis(Axis(0), idx)
        .to_owned()
        .into_dyn();

    let flat_len = parent_input_lower.len();
    if split_dim >= flat_len {
        return Err(NyError::InvalidSpec(format!(
            "branch_input_split_from_picked: split_dim {split_dim} >= input len {flat_len}"
        )));
    }

    let lower_val = array_element_at(
        &parent_input_lower.view(),
        split_dim,
        "branch_input_split_from_picked: input lower",
    )?;
    let upper_val = array_element_at(
        &parent_input_upper.view(),
        split_dim,
        "branch_input_split_from_picked: input upper",
    )?;

    let width = upper_val - lower_val;
    if !width.is_finite() || width <= 0.0 {
        return Ok((None, None));
    }

    let midpoint = lower_val + (upper_val - lower_val) / 2.0;
    let child_depth = metadata.depth + 1;
    let child_constraints: Vec<ConstraintTuple> = Vec::new();

    let _priority = BetaCrownConfig::domain_priority_for_mode(
        verify_upper,
        metadata.lower_bound,
        metadata.upper_bound,
    )?;

    let make_child = |child_input_lower: ndarray::ArrayD<f32>,
                      child_input_upper: ndarray::ArrayD<f32>|
     -> Result<ProcessedDomains> {
        Ok(ProcessedDomains {
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            input_lowers: child_input_lower.insert_axis(Axis(0)),
            input_uppers: child_input_upper.insert_axis(Axis(0)),
            global_lbs: vec![metadata.lower_bound],
            global_ubs: vec![metadata.upper_bound],
            metadata: vec![DomainMetadata::new(
                metadata.lower_bound,
                metadata.upper_bound,
                child_depth,
                child_constraints.clone(),
                None,
                None,
            )?],
            keep_mask: vec![true],
        })
    };

    let mut left_upper = parent_input_upper.clone();
    let left_len = left_upper.len();
    *left_upper.iter_mut().nth(split_dim).ok_or_else(|| {
        NyError::InternalError(format!(
            "branch_input_split_from_picked: left_upper nth({split_dim}) failed for len {left_len}",
        ))
    })? = midpoint;
    let left = make_child(parent_input_lower.clone(), left_upper)?;

    let mut right_lower = parent_input_lower;
    let right_len = right_lower.len();
    *right_lower.iter_mut().nth(split_dim).ok_or_else(|| {
        NyError::InternalError(format!(
            "branch_input_split_from_picked: right_lower nth({split_dim}) failed for len {right_len}",
        ))
    })? = midpoint;
    let right = make_child(right_lower, parent_input_upper)?;

    Ok((Some(left), Some(right)))
}

/// Select the best input dimension to split for a single picked domain.
///
/// Scores each input dimension by its width (`upper - lower`). Returns the
/// flat dimension index with the largest width, and the midpoint for splitting.
///
/// Phase 1 uses width-only scoring. Phase 2+ will add CROWN-coefficient
/// weighting (`|lA| * width / 2`) via the `la_coefficients` parameter.
///
/// # Reference
/// - alpha-beta-CROWN `input_split/branching_heuristics.py`: `sb` method
///   scores `|lA| * (x_U - x_L) / 2`
/// - Design: `designs/2026-02-10-input-split-domain-list.md` §1.2
/// - Issue: #1891
#[cfg(test)]
pub fn select_input_split_dimension(picked: &PickedDomains, idx: usize) -> Result<(usize, f32)> {
    let input_lower = picked.input_lowers.index_axis(Axis(0), idx);
    let input_upper = picked.input_uppers.index_axis(Axis(0), idx);

    let flat_len = input_lower.len();
    if flat_len == 0 {
        return Err(NyError::InvalidSpec(
            "select_input_split_dimension: empty input".to_string(),
        ));
    }

    let mut best_dim = 0;
    let mut best_width = f32::NEG_INFINITY;

    let input_lower_dyn = input_lower.into_dyn();
    let input_upper_dyn = input_upper.into_dyn();

    for dim in 0..flat_len {
        let lower = array_element_at(&input_lower_dyn, dim, "select_input_split_dimension: lower")?;
        let upper = array_element_at(&input_upper_dyn, dim, "select_input_split_dimension: upper")?;
        let width = upper - lower;
        if width.is_finite() && width > best_width {
            best_width = width;
            best_dim = dim;
        }
    }

    if !best_width.is_finite() || best_width <= 0.0 {
        return Err(NyError::InvalidSpec(
            "select_input_split_dimension: all dimensions have zero or infinite width".to_string(),
        ));
    }

    let lower = array_element_at(
        &input_lower_dyn,
        best_dim,
        "select_input_split_dimension: best lower",
    )?;
    let upper = array_element_at(
        &input_upper_dyn,
        best_dim,
        "select_input_split_dimension: best upper",
    )?;
    let midpoint = lower + (upper - lower) / 2.0;

    Ok((best_dim, midpoint))
}
