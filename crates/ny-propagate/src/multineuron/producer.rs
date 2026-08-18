// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound joint-bound producer for a real neuron group in the LIVE network
//! (design §1.2 / increment 2, item 1).
//!
//! Increment 1 only had [`Octahedron2::from_affine`], the toy producer for a
//! pre-activation that is a *single affine map* of the input. On a deep net the
//! group's pre-activations `x_i, x_j` are separated from the input by earlier
//! (ReLU-relaxed) layers, so the coupling bounds `x_i ± x_j` must come from a
//! genuine CROWN/α backward, not a closed-form affine range.
//!
//! [`combined_row_octahedron`] produces the sound octahedral `P ⊇ Z` for a
//! both-unstable pair at a ReLU's pre-activation node by running ONE spec-guided
//! CROWN backward whose spec rows are the four coupling directions
//! `e_i, e_j, e_i+e_j, e_i−e_j` over that node — reusing NY's proven
//! [`SpecCrownRequest`] backward (which relaxes every intervening ReLU with the
//! supplied `α` exactly as production does). The eight octahedral bounds are then
//! rounded OUTWARD (`next_up_f32` on uppers, `next_down_f32` on lowers) so the
//! stored `P` certifiably contains the true reachable pre-activation set `Z`
//! after the f32 backward accumulation (Invariant P1).
//!
//! # Why the spec-row backward is a valid enclosure of `x_i ± x_j`
//!
//! `SpecCrownRequest` with spec matrix `C` computes a certified enclosure of
//! `C · out(u)` over `u ∈ U`, where `out` is the (retargeted) output node's
//! value. Setting the output to the pre-activation node and `C`'s rows to the
//! coupling directions makes row `r` a certified `[lower_r, upper_r]` enclosure
//! of `(C · x)_r = x_i`, `x_j`, `x_i+x_j`, or `x_i−x_j`. Every intervening ReLU
//! is over-approximated (α-CROWN triangle), so the enclosure is *sound* (a
//! superset), never an under-approximation — exactly Invariant P1's requirement.
//!
//! Soundness does NOT depend on the backward being tight: a looser `[l,u]` only
//! enlarges `P`, which only enlarges the relaxation (fewer/looser facets), never
//! excludes a reachable point. This is verified LIVE by
//! `producer_combined_row_encloses_on_real_backward` (sample real inputs → true
//! pre-activations → assert inside `P` and inside every emitted coupling facet).

use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::Octahedron2;
use crate::bounds::GraphAlphaState;
use crate::network::SpecCrownRequest;
use crate::GraphNetwork;

/// Cooperative deadline checkpoints for joint-bound construction.
///
/// The typed stage is also the narrow deterministic test seam: tests inject a
/// checker that fails at an exact boundary, while production checks the real
/// request-local [`Instant`] at the same boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProducerDeadlineStage {
    BeforeRequestConstruction,
    AfterInputValidation,
    AfterSpecConstruction,
    AfterCrownBackward,
    AfterBoundsValidation,
    AfterLowerConversion,
    AfterUpperConversion,
    BeforeSinglePublish,
    AfterBatchPairValidation(usize),
    AfterBatchPairSpecConstruction(usize),
    BeforeBatchPairConversion(usize),
    AfterBatchPairConversion(usize),
    BeforeBatchPublish,
}

impl std::fmt::Display for ProducerDeadlineStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeRequestConstruction => write!(formatter, "before request construction"),
            Self::AfterInputValidation => write!(formatter, "after input validation"),
            Self::AfterSpecConstruction => write!(formatter, "after spec construction"),
            Self::AfterCrownBackward => write!(formatter, "after CROWN backward"),
            Self::AfterBoundsValidation => write!(formatter, "after bounds validation"),
            Self::AfterLowerConversion => write!(formatter, "after lower-bound conversion"),
            Self::AfterUpperConversion => write!(formatter, "after upper-bound conversion"),
            Self::BeforeSinglePublish => write!(formatter, "before publishing one octahedron"),
            Self::AfterBatchPairValidation(pair) => {
                write!(formatter, "after validating batch pair {pair}")
            }
            Self::AfterBatchPairSpecConstruction(pair) => {
                write!(formatter, "after constructing spec for batch pair {pair}")
            }
            Self::BeforeBatchPairConversion(pair) => {
                write!(formatter, "before converting batch pair {pair}")
            }
            Self::AfterBatchPairConversion(pair) => {
                write!(formatter, "after converting batch pair {pair}")
            }
            Self::BeforeBatchPublish => write!(formatter, "before publishing batched octahedra"),
        }
    }
}

fn check_deadline(
    deadline: Option<Instant>,
    operation: &str,
    stage: ProducerDeadlineStage,
) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "{operation}: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

/// Sound octahedral `P ⊇ Z` for the both-unstable pair `(i, j)` at the ReLU
/// pre-activation node `pre_node` (design §1.2, Invariant P1).
///
/// `pre_node` must be a node whose flattened output has `> max(i, j)` elements
/// (the ReLU's input / pre-activation). `alpha_state` and `node_bounds` are the
/// SAME objects production feeds its backward, so the intervening-ReLU relaxation
/// matches the deployed bound exactly. `engine` may be `None` (CPU backward).
///
/// The eight bounds are rounded OUTWARD from the f32 backward result, so
/// `Z ⊆ P` holds certifiably. Returns the assembled [`Octahedron2`]; the caller
/// checks `both_unstable()` and calls [`super::coupling_facets`].
///
/// Cost note (production): this clones the graph to retarget the output, which
/// is cheap for the group producer's small candidate count but O(nodes) per
/// call. Increment 3 should batch all groups at a layer into one spec matrix
/// (stack every pair's four rows) so a single backward serves the whole layer.
#[allow(clippy::too_many_arguments)]
pub fn combined_row_octahedron(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    i: usize,
    j: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<Octahedron2> {
    combined_row_octahedron_with_deadline(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        i,
        j,
        engine,
        None,
    )
}

/// Deadline-bounded form of [`combined_row_octahedron`].
///
/// The same private deadline is forwarded into the CROWN request and polled
/// through validation, spec construction, post-backward validation, conversion,
/// and immediately before publication. An expired result is discarded rather
/// than published as a cut-production candidate.
#[allow(clippy::too_many_arguments)]
pub fn combined_row_octahedron_with_deadline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    i: usize,
    j: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Octahedron2> {
    let mut check = |stage| check_deadline(deadline, "combined_row_octahedron", stage);
    combined_row_octahedron_with_checker(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        i,
        j,
        engine,
        deadline,
        &mut check,
    )
}

#[allow(clippy::too_many_arguments)]
fn combined_row_octahedron_with_checker<F>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    i: usize,
    j: usize,
    engine: Option<&dyn GemmEngine>,
    crown_deadline: Option<Instant>,
    check: &mut F,
) -> Result<Octahedron2>
where
    F: FnMut(ProducerDeadlineStage) -> Result<()>,
{
    check(ProducerDeadlineStage::BeforeRequestConstruction)?;
    if i == j {
        return Err(NyError::InvalidSpec(
            "combined_row_octahedron requires two distinct neurons".into(),
        ));
    }

    // The pre-activation node's flattened width bounds i, j.
    let n_pre = node_bounds
        .and_then(|m| m.get(pre_node))
        .map(|bt| bt.flatten().len())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "combined_row_octahedron: pre-activation bounds for '{pre_node}' not available"
            ))
        })?;
    if i >= n_pre || j >= n_pre {
        return Err(NyError::InvalidSpec(format!(
            "combined_row_octahedron: neuron index out of range (i={i}, j={j}, n_pre={n_pre})"
        )));
    }
    check(ProducerDeadlineStage::AfterInputValidation)?;

    // Retarget a graph clone at the pre-activation node (the backward starts
    // there); the four spec rows are the coupling directions. Downstream nodes
    // are simply not ancestors of `pre_node`, so the backward ignores them.
    let mut probe = graph.clone();
    probe.set_output(pre_node);

    let mut spec = Array2::<f32>::zeros((4, n_pre));
    spec[[0, i]] = 1.0; // x_i
    spec[[1, j]] = 1.0; // x_j
    spec[[2, i]] = 1.0; // x_i + x_j
    spec[[2, j]] = 1.0;
    spec[[3, i]] = 1.0; // x_i - x_j
    spec[[3, j]] = -1.0;
    check(ProducerDeadlineStage::AfterSpecConstruction)?;

    let bounds = SpecCrownRequest::new(&probe, input, &spec, engine)
        .node_bounds_opt(node_bounds)
        .alpha_state_opt(Some(alpha_state))
        .deadline_opt(crown_deadline)
        .run()?;
    check(ProducerDeadlineStage::AfterCrownBackward)?;

    let lo = bounds.lower();
    let hi = bounds.upper();
    if lo.len() != 4 || hi.len() != 4 {
        return Err(NyError::InvalidSpec(format!(
            "combined_row_octahedron: expected 4 spec rows, got {}",
            lo.len()
        )));
    }
    check(ProducerDeadlineStage::AfterBoundsValidation)?;
    let lo: Vec<f64> = lo.iter().map(|&x| x as f64).collect();
    check(ProducerDeadlineStage::AfterLowerConversion)?;
    let hi: Vec<f64> = hi.iter().map(|&x| x as f64).collect();
    check(ProducerDeadlineStage::AfterUpperConversion)?;

    // OUTWARD rounding (Invariant P1): uppers up, lowers down, in f32 (the
    // certified representation). The spec backward already produced a sound f32
    // enclosure; the extra nudge covers the f64→f32→f64 hand-off in assembly and
    // keeps the discipline identical to `Octahedron2::from_affine`.
    let out_up = |x: f64| next_up_f32(x as f32) as f64;
    let out_dn = |x: f64| next_down_f32(x as f32) as f64;

    let octahedron = Octahedron2::from_bounds(
        out_dn(lo[0]), // l1
        out_up(hi[0]), // u1
        out_dn(lo[1]), // l2
        out_up(hi[1]), // u2
        out_dn(lo[2]), // s_lo
        out_up(hi[2]), // s_hi
        out_dn(lo[3]), // d_lo
        out_up(hi[3]), // d_hi
    );
    check(ProducerDeadlineStage::BeforeSinglePublish)?;
    Ok(octahedron)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn combined_row_octahedron_with_checker_for_test<F>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    i: usize,
    j: usize,
    engine: Option<&dyn GemmEngine>,
    mut check: F,
) -> Result<Octahedron2>
where
    F: FnMut(ProducerDeadlineStage) -> Result<()>,
{
    combined_row_octahedron_with_checker(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        i,
        j,
        engine,
        None,
        &mut check,
    )
}

/// BATCHED sound octahedral producer (increment 3, §5.3 efficiency): assemble a
/// sound `Octahedron2 ⊇ Z` for EVERY pair in `pairs` from ONE spec-guided CROWN
/// backward.
///
/// Each pair `(i, j)` contributes four rows `x_i, x_j, x_i+x_j, x_i−x_j` to a
/// single `4·|pairs|`-row spec matrix over `pre_node`, so the whole candidate
/// set costs one backward through the intervening layers instead of one per
/// pair. Every row is assembled exactly as [`combined_row_octahedron`] (same
/// four coupling directions, same OUTWARD rounding), so each returned octahedron
/// is soundly `Z ⊆ P` by the identical Invariant-P1 argument — batching only
/// stacks independent rows of the same certified backward, it changes no math.
///
/// Returns one `Octahedron2` per input pair, in the same order. `pairs` must be
/// distinct-neuron pairs with indices `< flatten(pre_node).len()`.
#[allow(clippy::too_many_arguments)]
pub fn combined_rows_octahedra(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    pairs: &[(usize, usize)],
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<Octahedron2>> {
    combined_rows_octahedra_with_deadline(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        pairs,
        engine,
        None,
    )
}

/// Deadline-bounded form of [`combined_rows_octahedra`].
///
/// The deadline is checked before allocating the batched spec, forwarded into
/// the underlying CROWN request, polled for every validated/spec-built/converted
/// pair, and checked again immediately before the complete vector is returned.
/// Therefore expiry returns [`NyError::DeadlineExceeded`] with no partial vector
/// and cannot publish a carrier assembled after its request budget expired.
#[allow(clippy::too_many_arguments)]
pub fn combined_rows_octahedra_with_deadline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    pairs: &[(usize, usize)],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<Vec<Octahedron2>> {
    let mut check = |stage| check_deadline(deadline, "combined_rows_octahedra", stage);
    combined_rows_octahedra_with_checker(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        pairs,
        engine,
        deadline,
        &mut check,
    )
}

#[allow(clippy::too_many_arguments)]
fn combined_rows_octahedra_with_checker<F>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    pairs: &[(usize, usize)],
    engine: Option<&dyn GemmEngine>,
    crown_deadline: Option<Instant>,
    check: &mut F,
) -> Result<Vec<Octahedron2>>
where
    F: FnMut(ProducerDeadlineStage) -> Result<()>,
{
    check(ProducerDeadlineStage::BeforeRequestConstruction)?;
    if pairs.is_empty() {
        check(ProducerDeadlineStage::BeforeBatchPublish)?;
        return Ok(Vec::new());
    }
    let n_pre = node_bounds
        .and_then(|m| m.get(pre_node))
        .map(|bt| bt.flatten().len())
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "combined_rows_octahedra: pre-activation bounds for '{pre_node}' not available"
            ))
        })?;
    for (pair, &(i, j)) in pairs.iter().enumerate() {
        if i == j {
            return Err(NyError::InvalidSpec(
                "combined_rows_octahedra requires two distinct neurons per pair".into(),
            ));
        }
        if i >= n_pre || j >= n_pre {
            return Err(NyError::InvalidSpec(format!(
                "combined_rows_octahedra: neuron index out of range (i={i}, j={j}, n_pre={n_pre})"
            )));
        }
        check(ProducerDeadlineStage::AfterBatchPairValidation(pair))?;
    }
    check(ProducerDeadlineStage::AfterInputValidation)?;

    let mut probe = graph.clone();
    probe.set_output(pre_node);

    let n_rows = 4 * pairs.len();
    let mut spec = Array2::<f32>::zeros((n_rows, n_pre));
    for (p, &(i, j)) in pairs.iter().enumerate() {
        let base = 4 * p;
        spec[[base, i]] = 1.0; // x_i
        spec[[base + 1, j]] = 1.0; // x_j
        spec[[base + 2, i]] = 1.0; // x_i + x_j
        spec[[base + 2, j]] = 1.0;
        spec[[base + 3, i]] = 1.0; // x_i - x_j
        spec[[base + 3, j]] = -1.0;
        check(ProducerDeadlineStage::AfterBatchPairSpecConstruction(p))?;
    }
    check(ProducerDeadlineStage::AfterSpecConstruction)?;

    let bounds = SpecCrownRequest::new(&probe, input, &spec, engine)
        .node_bounds_opt(node_bounds)
        .alpha_state_opt(Some(alpha_state))
        .deadline_opt(crown_deadline)
        .run()?;
    check(ProducerDeadlineStage::AfterCrownBackward)?;

    let lo = bounds.lower();
    let hi = bounds.upper();
    if lo.len() != n_rows || hi.len() != n_rows {
        return Err(NyError::InvalidSpec(format!(
            "combined_rows_octahedra: expected {n_rows} spec rows, got {}",
            lo.len()
        )));
    }
    check(ProducerDeadlineStage::AfterBoundsValidation)?;
    let out_up = |x: f32| next_up_f32(x) as f64;
    let out_dn = |x: f32| next_down_f32(x) as f64;

    let mut result = Vec::with_capacity(pairs.len());
    for p in 0..pairs.len() {
        check(ProducerDeadlineStage::BeforeBatchPairConversion(p))?;
        let base = 4 * p;
        result.push(Octahedron2::from_bounds(
            out_dn(lo[base]),     // l1
            out_up(hi[base]),     // u1
            out_dn(lo[base + 1]), // l2
            out_up(hi[base + 1]), // u2
            out_dn(lo[base + 2]), // s_lo
            out_up(hi[base + 2]), // s_hi
            out_dn(lo[base + 3]), // d_lo
            out_up(hi[base + 3]), // d_hi
        ));
        check(ProducerDeadlineStage::AfterBatchPairConversion(p))?;
    }
    check(ProducerDeadlineStage::BeforeBatchPublish)?;
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn combined_rows_octahedra_with_checker_for_test<F>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    node_bounds: Option<&HashMap<String, BoundedTensor>>,
    pre_node: &str,
    pairs: &[(usize, usize)],
    engine: Option<&dyn GemmEngine>,
    mut check: F,
) -> Result<Vec<Octahedron2>>
where
    F: FnMut(ProducerDeadlineStage) -> Result<()>,
{
    combined_rows_octahedra_with_checker(
        graph,
        input,
        alpha_state,
        node_bounds,
        pre_node,
        pairs,
        engine,
        None,
        &mut check,
    )
}
