// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN and alpha-CROWN precheck for disjunctive clause screening.
//!
//! Before running expensive per-clause BaB, these functions run a single
//! bound propagation pass and check which clauses are provably unsatisfiable.
//! This avoids N separate CROWN passes for N-clause disjunctions.

use ndarray::{Array2, ArrayD};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::bounds::AlphaCrownConfig;
use ny_propagate::layers::LinearLayer;
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::{debug, info};

use super::BetaCrownModel;

/// Run a single CROWN pass and check which clauses are provably unsatisfiable.
///
/// For Sequential models with relational constraints, uses spec-guided CROWN:
/// a C-matrix Linear layer is appended so that ONE CROWN pass (with shared
/// CROWN-IBP intermediate bounds) checks all clauses simultaneously. This
/// gives tighter bounds than per-output CROWN (accounts for output correlations)
/// and avoids N separate CROWN-IBP computations in the per-clause BaB fallback.
///
/// For malbeware 16-25 (16 hidden layers, 24 clauses), this reduces the
/// precheck from ~192s (24 × 8s per-clause CROWN-IBP) to ~8s (1 shared
/// CROWN-IBP pass). #3218
///
/// Returns a boolean vector where `true` means the clause at that index is
/// verified (provably UNSAT) by CROWN bounds alone. If CROWN propagation
/// fails for any reason, returns all-false (no pre-verification).
/// Sound IBP output bounds for a model over an input box (forward-only, no clone /
/// no CROWN backward) — the cheap per-box screen that scales to the tens of
/// thousands of clauses nn4sys lindex/mscn produce.
pub(super) fn ibp_output_bounds(
    model_net: &BetaCrownModel,
    box_input: &BoundedTensor,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Option<BoundedTensor> {
    match model_net {
        BetaCrownModel::Sequential(network) => network
            .propagate_ibp_with_engine(box_input, gemm_engine)
            .ok(),
        BetaCrownModel::Graph(graph) => {
            graph.propagate_ibp_with_engine(box_input, gemm_engine).ok()
        }
    }
}

/// Whether interval output bounds prove a single (unsafe-region) constraint can
/// NEVER hold over the box. Uses the sound IBP lower/upper bounds.
///
/// Const-threshold arms compare in f64: the f32 bound widens losslessly to
/// f64, so `(l as f64) > k` is exact against the spec's f64 constant. The
/// previous `k as f32` round-to-NEAREST both lost up to half an ULP of margin
/// AND could round the constant toward the bound (a half-ULP wrong-"impossible"
/// hole on adversarially tight thresholds).
fn constraint_provably_false(output: &BoundedTensor, c: &OutputConstraint) -> bool {
    let lo = output.lower();
    let hi = output.upper();
    let l = |i: usize| lo.iter().nth(i).copied();
    let u = |i: usize| hi.iter().nth(i).copied();
    match c {
        OutputConstraint::LessEqConst(i, k) => l(*i).is_some_and(|v| f64::from(v) > *k),
        OutputConstraint::LessThanConst(i, k) => l(*i).is_some_and(|v| f64::from(v) >= *k),
        OutputConstraint::GreaterEqConst(i, k) => u(*i).is_some_and(|v| f64::from(v) < *k),
        OutputConstraint::GreaterThanConst(i, k) => u(*i).is_some_and(|v| f64::from(v) <= *k),
        OutputConstraint::LessEq(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a > b),
        OutputConstraint::LessThan(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a >= b),
        OutputConstraint::GreaterEq(i, j) => matches!((u(*i), l(*j)), (Some(a), Some(b)) if a < b),
        OutputConstraint::GreaterThan(i, j) => {
            matches!((u(*i), l(*j)), (Some(a), Some(b)) if a <= b)
        }
        _ => false,
    }
}

/// A clause (conjunction of unsafe constraints) is provably impossible over the box
/// iff ANY of its constraints is provably false there.
pub(super) fn clause_provably_unsat(output: &BoundedTensor, clause: &[OutputConstraint]) -> bool {
    !clause.is_empty() && clause.iter().any(|c| constraint_provably_false(output, c))
}

/// Whether a constraint is one of the variants `constraint_provably_false` can
/// ever discharge from interval output bounds. Clauses with NO such constraint
/// can never be proven by the box-refinement screen, so refining their box is
/// wasted budget.
pub(super) fn constraint_is_interval_checkable(c: &OutputConstraint) -> bool {
    matches!(
        c,
        OutputConstraint::LessEqConst(..)
            | OutputConstraint::LessThanConst(..)
            | OutputConstraint::GreaterEqConst(..)
            | OutputConstraint::GreaterThanConst(..)
            | OutputConstraint::LessEq(..)
            | OutputConstraint::LessThan(..)
            | OutputConstraint::GreaterEq(..)
            | OutputConstraint::GreaterThan(..)
    )
}

/// Largest f32 that is <= the f64 value (directed rounding for a sound lower
/// endpoint). Exactly-representable values (0.0, 1.0, …) convert without
/// widening — the unconditional `next_down_f32(v as f32)` this replaces cost a
/// full ULP per point axis, and across the ~150 point axes of an nn4sys mscn
/// clause box that self-inflicted widening dominated the IBP output width
/// (~3e-5), pushing tight band margins below the provable floor.
pub(super) fn f64_to_f32_floor(v: f64) -> f32 {
    let f = v as f32; // round-to-nearest
    if f as f64 <= v {
        f
    } else {
        ny_tensor::next_down_f32(f)
    }
}

/// Smallest f32 that is >= the f64 value (directed rounding for a sound upper
/// endpoint). See `f64_to_f32_floor`.
pub(super) fn f64_to_f32_ceil(v: f64) -> f32 {
    let f = v as f32;
    if f as f64 >= v {
        f
    } else {
        ny_tensor::next_up_f32(f)
    }
}

/// Tighten an input box to a clause's per-clause input sub-box (directed
/// f64→f32 rounding so the restricted box still encloses the clause's exact
/// real domain). A no-op for axes the clause does not constrain.
pub(super) fn tighten_input_to_box(
    input: &BoundedTensor,
    clause_box: &std::collections::BTreeMap<usize, (f64, f64)>,
) -> BoundedTensor {
    let mut lower = input.lower().clone();
    let mut upper = input.upper().clone();
    if let (Some(lo), Some(hi)) = (lower.as_slice_mut(), upper.as_slice_mut()) {
        for (&idx, &(lb, ub)) in clause_box {
            if idx < lo.len() {
                lo[idx] = lo[idx].max(f64_to_f32_floor(lb));
                hi[idx] = hi[idx].min(f64_to_f32_ceil(ub));
            }
        }
    }
    BoundedTensor::new(lower, upper).unwrap_or_else(|_| input.clone())
}

pub(super) fn crown_precheck_clauses(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Vec<bool> {
    // Per-clause INPUT-box screening (nn4sys lindex/mscn): each clause constrains a
    // tiny input sub-box, so a single shared CROWN pass over the GLOBAL input hull is
    // far too loose to verify any clause (the output ranges across every band). When
    // per-clause boxes are present, screen each clause over its OWN box, where the
    // bound is tight enough to prove the clause impossible. Only the constraints
    // matter per clause here; the input restriction is what makes it provable.
    //
    // The batched box-refinement engine groups clauses that share a box (mscn
    // band pairs: Y<=lo and Y>=hi over one box → ONE bound pass decides both
    // rows), runs IBP/CROWN passes over waves of boxes in parallel, and — the
    // part the old serial loop lacked — bisects the widest input axis of any
    // box whose bounds are not yet tight enough, re-screening the sub-boxes.
    // Fresh per-box bounds converge under bisection (unlike the per-clause BaB
    // fallback, whose shared-root intermediate bounds do not), so this decides
    // the mscn cardinality bands that previously burned the whole budget.
    //
    // GRAPH MODELS ONLY: the bisection engine exists because per-clause BaB
    // does not converge on the nn4sys Mul-heavy DAGs it was built for. For
    // SEQUENTIAL models with per-clause boxes (acasxu prop_6-class input-box
    // disjunctions: few clauses, low-dim WIDE boxes), the naive widest-axis
    // bisection is the wrong tool — measured on ACASXU_run2a_1_1/prop_6 it
    // closed 0/8 clauses in 94s (1.9M nodes) while starving the downstream
    // per-clause input-split BaB, which decides each disjunct in seconds.
    // Sequential specs get ONE cheap spec-guided CROWN pass per clause over
    // that clause's own box; survivors go to per-clause BaB with the budget.
    if per_clause_input_bounds.iter().any(|b| !b.is_empty()) {
        if matches!(model_net, BetaCrownModel::Graph(_)) {
            return super::disjunctive_box_refine::refine_clause_boxes(
                model_net,
                input,
                clauses,
                per_clause_input_bounds,
                gemm_engine,
                deadline,
            );
        }
        return clauses
            .iter()
            .enumerate()
            .map(|(idx, clause)| {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    return false; // UNPROVEN on deadline — never verified
                }
                let clause_input = per_clause_input_bounds
                    .get(idx)
                    .filter(|b| !b.is_empty())
                    .map_or_else(|| input.clone(), |b| tighten_input_to_box(input, b));
                crown_precheck_clauses(
                    model_net,
                    &clause_input,
                    std::slice::from_ref(clause),
                    &[],
                    gemm_engine,
                    deadline,
                )
                .first()
                .copied()
                .unwrap_or(false)
            })
            .collect();
    }

    // Try spec-guided CROWN first (tighter bounds, shared intermediate bounds).
    match model_net {
        BetaCrownModel::Sequential(network) => {
            if let Some(result) =
                try_spec_guided_crown_precheck(network, input, clauses, gemm_engine, deadline)
            {
                return result;
            }
        }
        BetaCrownModel::Graph(graph) => {
            if let Some(result) =
                try_spec_guided_crown_precheck_graph(graph, input, clauses, gemm_engine, deadline)
            {
                return result;
            }
        }
    }

    // Fallback: per-output CROWN bounds check.
    crown_precheck_per_output(model_net, input, clauses, gemm_engine, deadline)
}

/// Spec-guided CROWN precheck for Sequential models with relational constraints.
///
/// Builds a C-matrix from the clause constraints and appends it as a Linear
/// layer to the network. One `propagate_crown` call then produces bounds on
/// each constraint directly (e.g., bounds on Y_k - Y_0 for `GreaterEq(k, 0)`),
/// which are tighter than using individual output bounds because the CROWN
/// backward pass exploits correlations between Y_k and Y_0.
///
/// Returns `None` if the clauses contain non-relational constraints or if the
/// spec-guided approach fails (falls back to per-output precheck).
fn try_spec_guided_crown_precheck(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Option<Vec<bool>> {
    // Compute num_outputs from constraint indices (validated by VNN-LIB parser).
    // More robust than inferring from the last linear layer's weight shape,
    // since the network may end with a non-linear layer (ReLU, etc.).
    let num_outputs = clauses
        .iter()
        .flatten()
        .map(|c| c.max_output_index())
        .max()
        .map(|m| m + 1)?; // None if no constraints → fall back to per-output

    // Build C-matrix: one row per clause, encoding the relational constraint.
    // Each clause must have exactly one relational constraint (GreaterEq/LessEq).
    // For clauses with multiple constraints, we pick the FIRST relational one
    // (the others are checked by the per-output fallback if needed).
    let num_clauses = clauses.len();
    let mut c_matrix = Array2::<f32>::zeros((num_clauses, num_outputs));
    let mut row_constraint_type: Vec<SpecRowKind> = Vec::with_capacity(num_clauses);

    for (row, clause) in clauses.iter().enumerate() {
        // Find a relational constraint in this clause.
        let kind = build_spec_row(clause, num_outputs, c_matrix.row_mut(row))?;
        row_constraint_type.push(kind);
    }

    // Build augmented network: original + C-matrix linear layer.
    let spec_layer = LinearLayer::new(c_matrix, None).ok()?;
    let mut augmented = network.clone();
    augmented.add_layer(Layer::Linear(spec_layer));

    // Run CROWN on augmented network (shared CROWN-IBP for all clauses).
    let output = augmented
        .propagate_crown_with_engine_and_deadline(input, gemm_engine, deadline)
        .ok()?;

    let flat_lower = output.lower().as_slice().unwrap_or(&[]);
    let flat_upper = output.upper().as_slice().unwrap_or(&[]);

    // Check each clause from the spec-guided bounds.
    let result: Vec<bool> = row_constraint_type
        .iter()
        .enumerate()
        .map(|(row, kind)| {
            let lo = flat_lower.get(row).copied().unwrap_or(f32::NEG_INFINITY);
            let up = flat_upper.get(row).copied().unwrap_or(f32::INFINITY);
            kind.is_unsatisfiable(lo, up)
        })
        .collect();

    let verified_count = result.iter().filter(|&&v| v).count();
    debug!(
        verified = verified_count,
        total = num_clauses,
        "Spec-guided CROWN pre-check results"
    );

    Some(result)
}

/// Spec-guided CROWN precheck for Graph models with relational constraints.
///
/// Uses `propagate_crown_with_specs_and_engine_with_node_bounds` to pass the
/// C-matrix directly to the Graph CROWN backward pass, using IBP intermediate
/// bounds instead of the expensive O(N²) CROWN-IBP pass. For malbeware 16-25
/// (Conv→ReLU→Flatten→Gemm, 61K ReLU neurons), this avoids ~34 billion scalar
/// ops in the CROWN-IBP Dense backward through Conv2d. #3218
fn try_spec_guided_crown_precheck_graph(
    graph: &ny_propagate::GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Option<Vec<bool>> {
    let num_outputs = clauses
        .iter()
        .flatten()
        .map(|c| c.max_output_index())
        .max()
        .map(|m| m + 1)?;

    let num_clauses = clauses.len();
    let mut c_matrix = Array2::<f32>::zeros((num_clauses, num_outputs));
    let mut row_constraint_type: Vec<SpecRowKind> = Vec::with_capacity(num_clauses);

    for (row, clause) in clauses.iter().enumerate() {
        let kind = build_spec_row(clause, num_outputs, c_matrix.row_mut(row))?;
        row_constraint_type.push(kind);
    }

    // Collect cheap IBP node bounds (forward pass only, no CROWN-IBP).
    // This avoids the O(N²) CROWN-IBP intermediate tightening that
    // dominates graph CROWN time for Conv2d models with large spatial outputs.
    // For malbeware 16-25 (61K Conv2d output neurons), CROWN-IBP takes ~120s
    // while IBP forward takes <1s. #3218
    let node_bounds = graph.collect_node_bounds(input).ok()?;

    // Spec-guided CROWN backward with pre-computed IBP bounds.
    // This runs ONE backward pass with the C-matrix, using IBP intermediates
    // instead of the expensive CROWN-IBP Dense backward through Conv2d.
    let output = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
            input,
            &c_matrix,
            gemm_engine,
            &node_bounds,
            deadline,
        )
        .ok()?;

    let flat_lower = output.lower().as_slice().unwrap_or(&[]);
    let flat_upper = output.upper().as_slice().unwrap_or(&[]);

    let result: Vec<bool> = row_constraint_type
        .iter()
        .enumerate()
        .map(|(row, kind)| {
            let lo = flat_lower.get(row).copied().unwrap_or(f32::NEG_INFINITY);
            let up = flat_upper.get(row).copied().unwrap_or(f32::INFINITY);
            kind.is_unsatisfiable(lo, up)
        })
        .collect();

    let verified_count = result.iter().filter(|&&v| v).count();
    info!(
        verified = verified_count,
        total = num_clauses,
        "Spec-guided Graph CROWN pre-check (IBP intermediates, bypassed CROWN-IBP)"
    );

    Some(result)
}

/// Kind of spec row for a clause's constraint.
pub(crate) enum SpecRowKind {
    /// Row encodes a difference of outputs: unsatisfiable if upper < 0.
    ///
    /// For GreaterEq(i,j): row = Y_i - Y_j, UNSAT when upper(Y_i - Y_j) < 0.
    /// For LessEq(i,j): row = Y_j - Y_i, UNSAT when upper(Y_j - Y_i) < 0.
    DiffUpperNeg,
}

impl SpecRowKind {
    pub(crate) fn is_unsatisfiable(&self, _lower: f32, upper: f32) -> bool {
        match self {
            // Constraint is unsatisfiable when the upper bound on the encoded
            // difference is strictly negative. For GreaterEq(i,j) with row
            // Y_i - Y_j: upper < 0 means Y_i < Y_j always. For LessEq(i,j)
            // with row Y_j - Y_i: upper < 0 means Y_j < Y_i always. #3384
            SpecRowKind::DiffUpperNeg => upper < 0.0,
        }
    }
}

/// Build one row of the C-matrix from a clause's constraints.
///
/// Returns the kind of unsatisfiability check needed, or `None` if the
/// clause doesn't contain a supported relational constraint.
pub(crate) fn build_spec_row(
    clause: &[OutputConstraint],
    num_outputs: usize,
    mut row: ndarray::ArrayViewMut1<f32>,
) -> Option<SpecRowKind> {
    for constraint in clause {
        match constraint {
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                if *i < num_outputs && *j < num_outputs {
                    row[*i] = 1.0;
                    row[*j] = -1.0;
                    return Some(SpecRowKind::DiffUpperNeg);
                }
            }
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                if *i < num_outputs && *j < num_outputs {
                    // Encode Y_j - Y_i: UNSAT when upper(Y_j - Y_i) < 0,
                    // meaning Y_j < Y_i always, so Y_i <= Y_j can never hold. #3384
                    row[*j] = 1.0;
                    row[*i] = -1.0;
                    return Some(SpecRowKind::DiffUpperNeg);
                }
            }
            _ => continue, // Skip constant constraints, use per-output check
        }
    }
    None // No relational constraint found
}

/// Fallback: per-output CROWN precheck using individual output bounds.
/// This is the slow path — triggers full CROWN (potentially with CROWN-IBP).
fn crown_precheck_per_output(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Vec<bool> {
    let no_precheck = vec![false; clauses.len()];

    // Run a single CROWN forward pass to get output bounds.
    let output_bounds = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_crown_with_engine_and_deadline(input, gemm_engine, deadline)
        }
        BetaCrownModel::Graph(graph) => graph
            .propagate_crown_with_engine_and_deadline(input, gemm_engine, deadline)
            .map(|result| result.bounds),
    };

    let output = match output_bounds {
        Ok(bounds) => bounds,
        Err(e) => {
            debug!(error = %e, "CROWN pre-check failed, falling back to per-clause BaB");
            return no_precheck;
        }
    };

    let lower = output.lower();
    let upper = output.upper();
    // Diagnostic (#cgan-conv-err-compose): the per-output precheck verdict is
    // decided by these root bounds; surface them so a band-property miss is
    // attributable (tight-map vs realized-root divergence).
    if lower.len() <= 8 {
        info!(
            lower = ?lower.iter().collect::<Vec<_>>(),
            upper = ?upper.iter().collect::<Vec<_>>(),
            input_lower = ?(input.lower().len() <= 8).then(|| input.lower().iter().collect::<Vec<_>>()),
            input_upper = ?(input.upper().len() <= 8).then(|| input.upper().iter().collect::<Vec<_>>()),
            input_shape = ?input.shape(),
            "CROWN per-output precheck root bounds"
        );
    }

    // Check each clause: a clause is unsatisfiable if ANY constraint in it
    // is provably false given the CROWN output bounds.
    clauses
        .iter()
        .map(|clause| is_clause_unsatisfiable(clause, lower, upper))
        .collect()
}

/// Check if a clause (conjunction of constraints) is provably unsatisfiable
/// given output lower/upper bounds.
///
/// A clause is unsatisfiable if at least one constraint in it cannot hold
/// within the computed output intervals. Uses directed rounding for f64→f32
/// constant conversion to maintain soundness (never falsely declares a
/// clause unsatisfiable). Matches the logic in
/// `crates/ny-cli/src/commands/verify/result.rs:evaluate_property_status`.
fn is_clause_unsatisfiable(
    clause: &[OutputConstraint],
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
) -> bool {
    let flat_lower = lower.as_slice().unwrap_or(&[]);
    let flat_upper = upper.as_slice().unwrap_or(&[]);

    let get_lower = |idx: usize| -> Option<f32> { flat_lower.get(idx).copied() };
    let get_upper = |idx: usize| -> Option<f32> { flat_upper.get(idx).copied() };

    for constraint in clause {
        // Check if this constraint is satisfiable. If NOT satisfiable,
        // the whole clause is unsatisfiable (conjunctive within clause).
        let satisfiable = match constraint {
            OutputConstraint::LessEq(i, j) => {
                // Y_i <= Y_j is satisfiable if lower(Y_i) <= upper(Y_j)
                match (get_lower(*i), get_upper(*j)) {
                    (Some(li), Some(uj)) => li <= uj,
                    _ => true, // Out of bounds → assume satisfiable (conservative)
                }
            }
            OutputConstraint::GreaterEq(i, j) => {
                // Y_i >= Y_j is satisfiable if upper(Y_i) >= lower(Y_j)
                match (get_upper(*i), get_lower(*j)) {
                    (Some(ui), Some(lj)) => ui >= lj,
                    _ => true,
                }
            }
            OutputConstraint::LessThan(i, j) => match (get_lower(*i), get_upper(*j)) {
                (Some(li), Some(uj)) => li < uj,
                _ => true,
            },
            OutputConstraint::GreaterThan(i, j) => match (get_upper(*i), get_lower(*j)) {
                (Some(ui), Some(lj)) => ui > lj,
                _ => true,
            },
            // Directed rounding for f64→f32 constant conversion: round in the
            // direction that makes satisfiability easier to achieve, so we never
            // falsely declare a clause unsatisfiable (which would incorrectly
            // produce a "safe" verdict). Matches verify/result.rs #2658.
            OutputConstraint::LessEqConst(i, c) => {
                // lower(Y_i) <= c: round c UP so the check is conservative.
                match get_lower(*i) {
                    Some(li) => li <= ny_tensor::next_up_f32(*c as f32),
                    None => true,
                }
            }
            OutputConstraint::GreaterEqConst(i, c) => {
                // upper(Y_i) >= c: round c DOWN so the check is conservative.
                match get_upper(*i) {
                    Some(ui) => ui >= ny_tensor::next_down_f32(*c as f32),
                    None => true,
                }
            }
            OutputConstraint::LessThanConst(i, c) => match get_lower(*i) {
                Some(li) => li < ny_tensor::next_up_f32(*c as f32),
                None => true,
            },
            OutputConstraint::GreaterThanConst(i, c) => match get_upper(*i) {
                Some(ui) => ui > ny_tensor::next_down_f32(*c as f32),
                None => true,
            },
            _ => true, // conservatively assume unknown variants are satisfiable
        };

        if !satisfiable {
            return true; // Clause is unsatisfiable
        }
    }

    false // All constraints are satisfiable → clause might hold
}

/// Run alpha-CROWN with optimized alpha parameters and re-check unverified clauses.
///
/// Only runs alpha-CROWN if there are unverified clauses remaining after the basic
/// CROWN precheck. Skips CNN models (Conv2d/MaxPool2d) where alpha-CROWN falls back
/// to basic CROWN anyway. Uses a fast config (15 iterations, IBP intermediates) for
/// FC models where alpha-CROWN actually runs (~0.1s for MNIST FC 256x4).
///
/// `already_verified` is the result from the CROWN precheck — clauses marked `true`
/// are skipped (already verified). Returns an updated verification vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn alpha_crown_precheck_clauses(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    already_verified: &[bool],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Vec<bool> {
    // Skip if all clauses already verified by basic CROWN.
    if already_verified.iter().all(|&v| v) {
        return already_verified.to_vec();
    }

    // Per-clause INPUT-box screening (nn4sys lindex/mscn): refine each still-unverified
    // clause over its OWN tiny input box, where alpha-CROWN is tight, instead of the
    // global hull. See `crown_precheck_clauses`.
    if per_clause_input_bounds.iter().any(|b| !b.is_empty()) {
        return clauses
            .iter()
            .enumerate()
            .map(|(idx, clause)| {
                if already_verified.get(idx).copied().unwrap_or(false) {
                    return true;
                }
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    return false;
                }
                let tightened = match per_clause_input_bounds.get(idx) {
                    Some(b) if !b.is_empty() => tighten_input_to_box(input, b),
                    _ => input.clone(),
                };
                alpha_crown_precheck_clauses(
                    model_net,
                    &tightened,
                    std::slice::from_ref(clause),
                    &[false],
                    &[],
                    gemm_engine,
                    deadline,
                )
                .first()
                .copied()
                .unwrap_or(false)
            })
            .collect();
    }

    // Skip for Graph models: Graph models are created for CNN DAGs (Conv2d
    // auto-routes to Graph). Alpha-CROWN on Graph models runs the full CROWN
    // backward through Conv2d layers without meaningful alpha optimization
    // (patches mode doesn't support per-neuron alpha). The CROWN precheck
    // with spec-guided IBP intermediates already captures the same bounds.
    // For malbeware 16-25, this avoids a wasteful 444s alpha-CROWN pass. #3218
    if matches!(model_net, BetaCrownModel::Graph(_)) {
        info!("Skipping alpha-CROWN pre-check: Graph model (Conv2d DAG)");
        return already_verified.to_vec();
    }

    // Skip for Sequential CNN models: alpha-CROWN falls back to basic CROWN when
    // Conv2d/ConvTranspose2d/MaxPool2d layers are detected (alpha_crown.rs:145-156),
    // producing identical bounds to the CROWN precheck already run above.
    if let BetaCrownModel::Sequential(network) = model_net {
        use ny_propagate::layers::Layer;
        if network.layers().iter().any(|l| {
            matches!(
                l,
                Layer::Conv2d(_) | Layer::ConvTranspose2d(_) | Layer::MaxPool2d(_)
            )
        }) {
            debug!("Skipping alpha-CROWN pre-check: Conv2d/MaxPool2d layers detected");
            return already_verified.to_vec();
        }
    }

    let remaining = already_verified.iter().filter(|&&v| !v).count();
    debug!(
        remaining,
        "Running alpha-CROWN pre-check on unverified clauses"
    );

    // Fast alpha-CROWN config: 15 iterations is enough for FC models where
    // alpha-CROWN actually runs (MNIST FC 256x4/256x6). CNN models are skipped
    // above since alpha-CROWN falls back to CROWN for Conv2d layers.
    // Deep FC models (>8 ReLU layers, e.g., malbeware 16-25) are handled by
    // adaptive_skip in the propagation path — alpha-CROWN provides no benefit
    // for deep networks where bounds are fundamentally loose. #3218
    let alpha_config = AlphaCrownConfig {
        iterations: 15,
        learning_rate: 0.1,
        lr_decay: 0.98,
        fix_interm_bounds: true,
        adaptive_skip: true, // Skip for deep models (>8 ReLU layers) — no benefit
        deadline,
        ..AlphaCrownConfig::default()
    };

    let output_bounds = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_alpha_crown_with_config_and_engine(input, &alpha_config, gemm_engine)
        }
        BetaCrownModel::Graph(graph) => {
            graph.propagate_alpha_crown_with_config_and_engine(input, &alpha_config, gemm_engine)
        }
    };

    let output = match output_bounds {
        Ok(bounds) => bounds,
        Err(e) => {
            debug!(error = %e, "Alpha-CROWN pre-check failed, using CROWN results");
            return already_verified.to_vec();
        }
    };

    let lower = output.lower();
    let upper = output.upper();

    // Re-check all clauses with tighter alpha-CROWN bounds.
    // Preserve any clauses already verified by basic CROWN.
    let mut result = already_verified.to_vec();
    let mut newly_verified = 0;
    for (idx, clause) in clauses.iter().enumerate() {
        if result[idx] {
            continue; // Already verified by basic CROWN
        }
        if is_clause_unsatisfiable(clause, lower, upper) {
            result[idx] = true;
            newly_verified += 1;
        }
    }

    let total_verified = result.iter().filter(|&&v| v).count();
    if newly_verified > 0 || total_verified > 0 {
        debug!(
            newly_verified,
            total_verified,
            total = clauses.len(),
            "Alpha-CROWN pre-check clause results"
        );
    }

    result
}
