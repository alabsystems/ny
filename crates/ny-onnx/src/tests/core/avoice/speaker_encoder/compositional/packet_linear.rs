// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direct raw-input boundary certificates for the ECAPA MFA seam (#3499).
//!
//! This module implements Packet 1 of the alternative design:
//! `designs/2026-03-13-issue-3499-linear-boundary-certificates-alternative.md`
//!
//! Instead of chaining Stage A → box → Stage B → box → Stage C → box → concat,
//! this approach extracts three prefix graphs (input→x2, input→x3, input→x4)
//! all rooted at `NETWORK_INPUT`, runs spec-guided CROWN with an identity spec
//! on each to obtain `LinearBounds` certificates, then concatenates the three
//! certificates exactly at the MFA seam. The concatenated certificate is
//! concretized once on the original input domain.
//!
//! Key improvement: x3 and x4 no longer inherit boxed x2/x3 input —
//! all certificates are computed directly against the raw mel input.

use super::boundary::{discover_ecapa_composition_boundary, EcapaCompositionBoundary};
use super::packet_bc::core::{ensure_bounded_tensor_finite_and_ordered, total_bound_width};
use super::subgraph::extract_single_input_subgraph;
use super::*;
use ny_propagate::LinearBounds;
use std::time::{Duration, Instant};

mod composition;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Raw-input prefix graph extraction
// ---------------------------------------------------------------------------

/// Extract three prefix graphs rooted at `NETWORK_INPUT` for each block output.
///
/// Unlike the stage-local extraction (input→x2, x2→x3, x3→x4), all three
/// prefix graphs share the same root: `NETWORK_INPUT`. This eliminates the
/// compounding re-boxing at each stage boundary.
fn extract_ecapa_raw_input_prefix_graphs(
    graph: &GraphNetwork,
    boundary: &EcapaCompositionBoundary,
) -> Result<[GraphNetwork; 3], String> {
    let prefix_x2 = extract_single_input_subgraph(
        graph,
        ny_propagate::NETWORK_INPUT,
        &boundary.block_outputs[0],
    )?;
    let prefix_x3 = extract_single_input_subgraph(
        graph,
        ny_propagate::NETWORK_INPUT,
        &boundary.block_outputs[1],
    )?;
    let prefix_x4 = extract_single_input_subgraph(
        graph,
        ny_propagate::NETWORK_INPUT,
        &boundary.block_outputs[2],
    )?;
    Ok([prefix_x2, prefix_x3, prefix_x4])
}

// ---------------------------------------------------------------------------
// LinearBounds extraction per prefix graph
// ---------------------------------------------------------------------------

/// Result of extracting CROWN bounds for a single prefix graph.
struct PrefixBoundsResult {
    /// Concrete bounds from spec-guided CROWN (always available, flat shape [N]).
    /// When LinearBounds is None, this is the IBP-applied-spec fallback.
    concrete: BoundedTensor,
    /// Linear coefficient bounds extracted from CROWN backward.
    /// None when the backward produces non-finite results and falls back to IBP.
    linear: Option<LinearBounds>,
    /// Number of output dimensions for this prefix (flat).
    output_dim: usize,
    /// Original tensor shape of the prefix output (e.g., [512, 5]).
    /// Needed for reshaping the flat MFA bounds back to the expected shape.
    output_shape: Vec<usize>,
}

/// Collect CROWN-IBP tightened node bounds and then attempt `LinearBounds`
/// extraction via identity-spec CROWN on a prefix graph.
///
/// Steps:
/// 1. IBP forward on the prefix graph for base node bounds
/// 2. CROWN-IBP tightening with a deadline
/// 3. Identity-spec CROWN with the tightened node bounds → concrete bounds
///    plus optional `LinearBounds`
///
/// Returns concrete bounds (always) and optional `LinearBounds` certificate.
/// When the CROWN backward produces non-finite results (common for deep Conv1d
/// graphs with few tightened intermediates), the LinearBounds is None and the
/// concrete output is the IBP-applied-spec fallback.
fn extract_prefix_bounds(
    prefix_graph: &GraphNetwork,
    input: &BoundedTensor,
    label: &str,
    crown_ibp_budget_secs: u64,
) -> Result<PrefixBoundsResult, String> {
    let output_name = prefix_graph.output_name().to_string();
    let start = Instant::now();

    // Step 1: IBP forward for base bounds.
    let ibp_bounds = prefix_graph
        .collect_node_bounds(input)
        .map_err(|e| format!("{label}: IBP failed: {e}"))?;
    let ibp_output = ibp_bounds
        .get(&output_name)
        .ok_or_else(|| format!("{label}: IBP missing output '{output_name}'"))?;
    let ibp_output_dim = ibp_output.len();
    let output_shape = ibp_output.shape().to_vec();
    eprintln!(
        "{label}: IBP output_dim={ibp_output_dim}, shape={output_shape:?}, max_width={:.6}",
        ibp_output.max_width()
    );

    // Step 2: CROWN-IBP tightening with bounded budget.
    let crown_ibp_deadline = Instant::now() + Duration::from_secs(crown_ibp_budget_secs);
    let crown_ibp_result = prefix_graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            input,
            ibp_bounds,
            Some(crown_ibp_deadline),
        )
        .map_err(|e| format!("{label}: CROWN-IBP tightening failed: {e}"))?;
    let tightened_count = crown_ibp_result
        .provenance
        .values()
        .filter(|p| matches!(p, BoundsProvenance::Crown))
        .count();
    eprintln!(
        "{label}: CROWN-IBP tightened {tightened_count}/{} nodes ({:.1}s)",
        crown_ibp_result.bounds.len(),
        start.elapsed().as_secs_f32(),
    );

    // Step 3: Identity-spec CROWN with tightened node bounds.
    // No deadline: LinearBounds extraction is all-or-nothing — if the backward
    // pass hits a deadline, it falls back to IBP and returns None for
    // LinearBounds. We rely on the ntest timeout for overall time bounding.
    let identity_spec = ndarray::Array2::eye(ibp_output_dim);
    let (concrete, linear_bounds_opt) = prefix_graph
        .propagate_crown_with_specs_and_node_bounds_and_linear(
            input,
            &identity_spec,
            None,
            &crown_ibp_result.bounds,
        )
        .map_err(|e| format!("{label}: identity-spec CROWN failed: {e}"))?;

    match &linear_bounds_opt {
        Some(lb) => eprintln!(
            "{label}: LinearBounds extracted: outputs={}, inputs={} ({:.1}s total)",
            lb.num_outputs(),
            lb.num_inputs(),
            start.elapsed().as_secs_f32(),
        ),
        None => eprintln!(
            "{label}: CROWN backward fell back to IBP (non-finite coefficients) — \
             using concrete bounds ({:.1}s total)",
            start.elapsed().as_secs_f32(),
        ),
    }

    Ok(PrefixBoundsResult {
        concrete,
        linear: linear_bounds_opt,
        output_dim: ibp_output_dim,
        output_shape, // recorded from step 1 IBP
    })
}

// ---------------------------------------------------------------------------
// LinearBounds concatenation
// ---------------------------------------------------------------------------

/// Concatenate three `LinearBounds` certificates by vertically stacking
/// coefficient matrices and bias vectors.
///
/// Given:
///   x2 in [L2_a @ x + L2_b, U2_a @ x + U2_b]
///   x3 in [L3_a @ x + L3_b, U3_a @ x + U3_b]
///   x4 in [L4_a @ x + L4_b, U4_a @ x + U4_b]
///
/// Returns:
///   concat(x2,x3,x4) in [cat(L2_a,L3_a,L4_a) @ x + cat(L2_b,L3_b,L4_b),
///                         cat(U2_a,U3_a,U4_a) @ x + cat(U2_b,U3_b,U4_b)]
///
/// Sound because each row is an independent linear bound over the same input x.
fn concat_linear_bounds(certs: [LinearBounds; 3]) -> Result<LinearBounds, String> {
    // Verify all certificates share the same input dimension.
    let num_inputs = certs[0].num_inputs();
    for (i, cert) in certs.iter().enumerate() {
        if cert.num_inputs() != num_inputs {
            return Err(format!(
                "LinearBounds input dimension mismatch: cert[0]={num_inputs}, cert[{i}]={}",
                cert.num_inputs()
            ));
        }
    }

    let [c0, c1, c2] = certs;
    let (la0, lb0, ua0, ub0) = c0.into_parts();
    let (la1, lb1, ua1, ub1) = c1.into_parts();
    let (la2, lb2, ua2, ub2) = c2.into_parts();

    let lower_a = ndarray::concatenate(ndarray::Axis(0), &[la0.view(), la1.view(), la2.view()])
        .map_err(|e| format!("concat lower_a failed: {e}"))?;
    let upper_a = ndarray::concatenate(ndarray::Axis(0), &[ua0.view(), ua1.view(), ua2.view()])
        .map_err(|e| format!("concat upper_a failed: {e}"))?;
    let lower_b = ndarray::concatenate(ndarray::Axis(0), &[lb0.view(), lb1.view(), lb2.view()])
        .map_err(|e| format!("concat lower_b failed: {e}"))?;
    let upper_b = ndarray::concatenate(ndarray::Axis(0), &[ub0.view(), ub1.view(), ub2.view()])
        .map_err(|e| format!("concat upper_b failed: {e}"))?;

    LinearBounds::new(lower_a, lower_b, upper_a, upper_b)
        .map_err(|e| format!("concat LinearBounds validation failed: {e}"))
}

// ---------------------------------------------------------------------------
// Direct-boundary MFA bounds
// ---------------------------------------------------------------------------

/// Result of the direct-boundary MFA certificate computation.
#[derive(Debug, Clone)]
struct DirectBoundaryMfaResult {
    /// Concretized MFA bounds from the direct-boundary certificate.
    mfa_bounds: BoundedTensor,
    /// Per-prefix output dimensions [dim_x2, dim_x3, dim_x4].
    prefix_output_dims: [usize; 3],
    /// Total MFA output dimension (sum of prefix dims).
    total_mfa_dim: usize,
    /// Whether the linear certificate path was used (true) or the concrete
    /// fallback (false). The linear path concatenates LinearBounds certificates
    /// and concretizes once; the concrete fallback concatenates BoundedTensors
    /// from individual spec-guided CROWN calls (IBP-equivalent when CROWN
    /// backward produces non-finite results).
    used_linear_path: bool,
    /// The MFA LinearBounds certificate before concretization.
    ///
    /// Available when `used_linear_path == true`. This maps raw mel input to
    /// the flat MFA dimension. Used by Packet 2 (linear composition) to
    /// compose with suffix LinearBounds for tighter end-to-end scalar bounds.
    mfa_linear: Option<LinearBounds>,
}

/// Concatenate three flat `BoundedTensor`s (each shaped `[N_i]`) into one `[sum(N_i)]`.
///
/// Used as fallback when LinearBounds extraction fails on one or more prefixes.
fn concat_concrete_bounds(bounds: [&BoundedTensor; 3]) -> Result<BoundedTensor, String> {
    let lowers: Result<Vec<_>, _> = bounds
        .iter()
        .map(|b| {
            b.lower()
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| format!("lower not 1D: {e}"))
        })
        .collect();
    let lowers = lowers?;
    let uppers: Result<Vec<_>, _> = bounds
        .iter()
        .map(|b| {
            b.upper()
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| format!("upper not 1D: {e}"))
        })
        .collect();
    let uppers = uppers?;

    let lower_views: Vec<_> = lowers.iter().map(|a| a.view()).collect();
    let upper_views: Vec<_> = uppers.iter().map(|a| a.view()).collect();

    let lower = ndarray::concatenate(ndarray::Axis(0), &lower_views)
        .map_err(|e| format!("concat lower failed: {e}"))?;
    let upper = ndarray::concatenate(ndarray::Axis(0), &upper_views)
        .map_err(|e| format!("concat upper failed: {e}"))?;

    BoundedTensor::new(lower.into_dyn(), upper.into_dyn())
        .map_err(|e| format!("concat_concrete_bounds validation failed: {e}"))
}

/// Build direct raw-input MFA bounds by extracting CROWN certificates
/// for each ECAPA block output and concatenating them at the MFA seam.
///
/// Two paths:
/// - **Linear path**: all three prefixes yield `LinearBounds` -> concatenate
///   coefficient matrices -> single concretization on raw input.
/// - **Concrete fallback**: one or more prefixes fall back to IBP concrete
///   bounds -> concatenate the flat BoundedTensors directly.
fn run_ecapa_direct_boundary_mfa_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    crown_ibp_budget_secs: u64,
) -> Result<DirectBoundaryMfaResult, String> {
    let boundary = discover_ecapa_composition_boundary(graph)?;
    let [prefix_x2, prefix_x3, prefix_x4] =
        extract_ecapa_raw_input_prefix_graphs(graph, &boundary)?;

    eprintln!(
        "direct-boundary: prefix graphs extracted: x2={} nodes, x3={} nodes, x4={} nodes",
        prefix_x2.num_nodes(),
        prefix_x3.num_nodes(),
        prefix_x4.num_nodes(),
    );

    let r_x2 = extract_prefix_bounds(&prefix_x2, input, "prefix_x2", crown_ibp_budget_secs)?;
    let r_x3 = extract_prefix_bounds(&prefix_x3, input, "prefix_x3", crown_ibp_budget_secs)?;
    let r_x4 = extract_prefix_bounds(&prefix_x4, input, "prefix_x4", crown_ibp_budget_secs)?;

    let dims = [r_x2.output_dim, r_x3.output_dim, r_x4.output_dim];
    let total_dim: usize = dims.iter().sum();

    // Prefer linear certificate path when all three prefixes produced LinearBounds.
    let all_have_linear = r_x2.linear.is_some() && r_x3.linear.is_some() && r_x4.linear.is_some();
    let (mfa_bounds, used_linear_path, mfa_linear) = if all_have_linear {
        // Safe to unwrap: checked above.
        let l2 = r_x2.linear.unwrap();
        let l3 = r_x3.linear.unwrap();
        let l4 = r_x4.linear.unwrap();
        eprintln!("direct-boundary: all 3 prefixes yielded LinearBounds — using linear path");
        let mfa_cert = concat_linear_bounds([l2, l3, l4])?;
        eprintln!(
            "direct-boundary: MFA certificate: outputs={}, inputs={}",
            mfa_cert.num_outputs(),
            mfa_cert.num_inputs()
        );
        let bounds = mfa_cert.concretize(input);
        (bounds, true, Some(mfa_cert))
    } else {
        let linear_count = [&r_x2, &r_x3, &r_x4]
            .iter()
            .filter(|r| r.linear.is_some())
            .count();
        eprintln!(
            "direct-boundary: {linear_count}/3 prefixes yielded LinearBounds — \
             falling back to concrete concatenation"
        );
        let bounds = concat_concrete_bounds([&r_x2.concrete, &r_x3.concrete, &r_x4.concrete])?;
        (bounds, false, None)
    };

    ensure_bounded_tensor_finite_and_ordered(&mfa_bounds, "direct-boundary MFA bounds")?;

    // Design Section C: reshape flat [total_dim] back to the MFA tensor shape.
    // The suffix pipeline expects the original concat shape (e.g., [1536, 5]),
    // not a flat vector. Compute target shape by concatenating individual block
    // output shapes along the boundary's concat axis.
    let mfa_shape = {
        let mut shape = r_x2.output_shape.clone();
        // Resolve the raw ONNX concat axis (possibly negative) against the
        // prefix output rank (see boundary.rs / subgraph.rs).
        let axis = ny_core::resolve_axis(boundary.concat_axis, shape.len(), "ECAPA MFA concat")
            .map_err(|e| format!("MFA concat axis resolution failed: {e}"))?;
        shape[axis] = r_x2.output_shape[axis] + r_x3.output_shape[axis] + r_x4.output_shape[axis];
        shape
    };
    let mfa_bounds = mfa_bounds
        .reshape(&mfa_shape)
        .map_err(|e| format!("MFA reshape to {mfa_shape:?} failed: {e}"))?;

    eprintln!(
        "direct-boundary: MFA bounds shape={:?}, max_width={:.6}, total_width={:.6} (path={})",
        mfa_bounds.shape(),
        mfa_bounds.max_width(),
        total_bound_width(&mfa_bounds),
        if used_linear_path {
            "linear"
        } else {
            "concrete-fallback"
        },
    );

    Ok(DirectBoundaryMfaResult {
        mfa_bounds,
        prefix_output_dims: dims,
        total_mfa_dim: total_dim,
        used_linear_path,
        mfa_linear,
    })
}
