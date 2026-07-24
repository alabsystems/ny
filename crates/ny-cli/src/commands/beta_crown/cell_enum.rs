// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cell-enumeration verification driver for piecewise-constant "trunc gate"
//! models (cctsdb_yolo_2023, #cctsdb Phase C).
//!
//! # Strategy (mirrors the winning alpha-beta-CROWN custom driver)
//!
//! The model's only non-degenerate (free) inputs reach the network EXCLUSIVELY
//! through Cast-to-int (round-toward-zero `Trunc`) nodes. The network is
//! therefore PIECEWISE-CONSTANT over the integer cells of the free inputs:
//! for a free value `x` in `[v, v+1)` (with `v >= 0`), every node sees exactly
//! `trunc(x) = v`. Fixing the free inputs to any representative point strictly
//! inside a cell evaluates the ENTIRE cell exactly.
//!
//! Per cell the effective input is a point, so a sound f64 interval forward
//! ([`GraphNetwork::propagate_ibp_f64_cell`]) yields an enclosure tight enough
//! to decide the property:
//! - every cell definitely-safe        => UNSAT (verified);
//! - some cell definitely-violating    => SAT with the cell representative as
//!   a concrete witness (re-confirmed in-process, then again by the vnncomp
//!   ONNX-Runtime trusted-oracle gate downstream);
//! - any cell indeterminate            => fall through to the normal pipeline
//!   (sound: never conclude UNSAT past an undecided cell);
//! - deadline hit with partial coverage => Timeout (NEVER unsat from partial
//!   coverage).
//!
//! # Detection (sound, conservative — falls through on ANY doubt)
//!
//! A spec/graph qualifies iff:
//! 1. every non-point input dimension (in exact f64 vnnlib bounds) is finite
//!    with `lo >= 0` (trunc == floor on the non-negative axis, so integer
//!    cells are `[v, v+1)`; negative ranges fail closed);
//! 2. each free dimension feeds ONLY `Gather`/`Slice` selections whose every
//!    consumer is a `Trunc` node (or feeds a `Trunc` directly); consumers with
//!    unknown read-sets conservatively read ALL dims;
//! 3. the total cell count is within [`CELL_BUDGET`];
//! 4. the spec has plain global input bounds (no per-clause boxes, no dual
//!    networks) and at least one output constraint.
//!
//! Disable with `NY_NO_CELL_ENUM=1` (batteries-included: default ON).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownResult, GraphNetwork, Interval64, Layer, NETWORK_INPUT,
};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use super::BetaCrownModel;

/// Maximum number of integer cells the driver will enumerate.
const CELL_BUDGET: usize = 16_384;

/// One integer cell of the free-input lattice.
#[derive(Debug, Clone)]
struct Cell {
    /// Representative point per free dim (index-aligned with `free_dims`).
    /// `trunc(rep) == cell value` and `rep` lies inside the spec box — see
    /// [`cell_representative`] for why evaluating at `rep` is EXACT for the
    /// whole cell.
    reps: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellVerdict {
    Safe,
    Violated,
    Indeterminate,
    /// Not evaluated (deadline or early SAT stop).
    Skipped,
}

/// Try the cell-enumeration driver. `None` => not applicable / undecided:
/// the caller MUST continue with the normal pipeline unchanged.
pub(super) fn try_cell_enumeration(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
    deadline: Instant,
) -> Option<BetaCrownResult> {
    if std::env::var_os("NY_NO_CELL_ENUM").is_some_and(|v| v == "1") {
        return None;
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return None;
    };
    let start = Instant::now();

    let plan = detect(graph, input_shape, vnnlib)?;
    info!(
        "Cell enumeration qualifies: {} free dim(s) {:?}, {} cells",
        plan.free_dims.len(),
        plan.free_dims,
        plan.cells.len()
    );

    // Evaluate cells in parallel chunks; every cell is independent. Early-stop
    // on a definite violation or on deadline (skipped cells stay Skipped —
    // partial coverage can NEVER conclude unsat).
    let stop = AtomicBool::new(false);
    let deadline_hit = AtomicBool::new(false);
    let verdicts: Vec<CellVerdict> = plan
        .cells
        .par_iter()
        .map(|cell| {
            if stop.load(Ordering::Relaxed) {
                return CellVerdict::Skipped;
            }
            if Instant::now() >= deadline {
                deadline_hit.store(true, Ordering::Relaxed);
                stop.store(true, Ordering::Relaxed);
                return CellVerdict::Skipped;
            }
            let verdict = evaluate_cell(graph, &plan, cell, vnnlib);
            if verdict == CellVerdict::Violated {
                stop.store(true, Ordering::Relaxed);
            }
            verdict
        })
        .collect();

    let violated_idx = verdicts.iter().position(|v| *v == CellVerdict::Violated);
    let indeterminate = verdicts
        .iter()
        .filter(|v| **v == CellVerdict::Indeterminate)
        .count();
    let skipped = verdicts
        .iter()
        .filter(|v| **v == CellVerdict::Skipped)
        .count();
    let safe = verdicts.len() - indeterminate - skipped - usize::from(violated_idx.is_some());
    let elapsed = start.elapsed();
    // tracing goes to stderr, so this never corrupts the captured JSON verdict.
    info!(
        "Cell enumeration: {} cells in {:.1}s ({:.1} ms/cell wall): {} safe, {} violated, \
         {} indeterminate, {} skipped",
        verdicts.len(),
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / (verdicts.len() - skipped).max(1) as f64,
        safe,
        usize::from(violated_idx.is_some()),
        indeterminate,
        skipped
    );

    // SAT path: confirm the violating cell with a concrete forward before
    // claiming. The emitted witness is ALSO re-checked by the vnncomp
    // trusted-oracle ONNX-Runtime gate downstream.
    if let Some(idx) = violated_idx {
        if let Some(result) = confirm_violation(model_net, &plan, &plan.cells[idx], vnnlib, start) {
            return Some(result);
        }
        warn!(
            "Cell enumeration: interval-violating cell {:?} failed concrete confirmation; \
             treating as indeterminate",
            plan.cells[idx].reps
        );
        // Fall through as undecided — never claim SAT without a confirmed point.
        return None;
    }

    if deadline_hit.load(Ordering::Relaxed) || skipped > 0 {
        // Partial coverage: NEVER unsat. Report timeout (no budget remains).
        return Some(BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: verdicts.len() - skipped,
            domains_verified: safe,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: elapsed,
            output_bounds: None,
        });
    }

    if indeterminate > 0 {
        info!(
            "Cell enumeration: {} indeterminate cell(s) — falling through to the normal \
             pipeline (sound)",
            indeterminate
        );
        return None;
    }

    // Full coverage, all cells definitely safe: UNSAT.
    Some(BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: verdicts.len(),
        domains_verified: verdicts.len(),
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: elapsed,
        output_bounds: None,
    })
}

/// Everything needed to run the enumeration.
struct CellPlan {
    input_shape: Vec<usize>,
    /// Exact f64 point value per input dim (free dims overridden per cell).
    base_values: Vec<f64>,
    free_dims: Vec<usize>,
    cells: Vec<Cell>,
}

/// Structural + spec detection. `None` => not applicable (fall through).
fn detect(graph: &GraphNetwork, input_shape: &[usize], vnnlib: &VnnLibSpec) -> Option<CellPlan> {
    // Spec shape: single global input box, real output constraints.
    if vnnlib.dual_network.is_some() {
        return None;
    }
    if vnnlib
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return None;
    }
    if vnnlib.output_constraints.is_empty() && vnnlib.output_constraint_clauses.is_empty() {
        return None;
    }
    let n: usize = input_shape.iter().product();
    if vnnlib.input_bounds.len() != n || n == 0 {
        return None;
    }

    // Free dims: non-degenerate in EXACT f64 spec bounds. All must be finite
    // with lo >= 0 (trunc == floor there; see module docs).
    let mut free_dims = Vec::new();
    let mut base_values = Vec::with_capacity(n);
    for (idx, &(lo, hi)) in vnnlib.input_bounds.iter().enumerate() {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return None;
        }
        base_values.push(lo);
        if lo < hi {
            if lo < 0.0 {
                return None;
            }
            free_dims.push(idx);
        }
    }
    if free_dims.is_empty() {
        return None;
    }

    // Structural check: every free dim reaches the graph ONLY through
    // selections whose consumers are all Trunc (or a Trunc directly).
    if !free_dims_gated_by_trunc(graph, n, &free_dims) {
        debug!("Cell enumeration: free dims are not exclusively Trunc-gated; falling through");
        return None;
    }

    // Cells: cartesian product of per-dim integer values v in
    // [floor(lo), floor(hi)] — for lo >= 0 these are exactly the trunc images
    // of [lo, hi], and cell v covers [v, v+1) ∩ [lo, hi].
    let mut per_dim_values: Vec<Vec<f64>> = Vec::with_capacity(free_dims.len());
    let mut total: usize = 1;
    for &dim in &free_dims {
        let (lo, hi) = vnnlib.input_bounds[dim];
        let v_min = lo.floor();
        let v_max = hi.floor();
        let count = (v_max - v_min) as usize + 1;
        total = total.checked_mul(count)?;
        if total > CELL_BUDGET {
            debug!("Cell enumeration: {total} cells exceed budget {CELL_BUDGET}");
            return None;
        }
        let mut values = Vec::with_capacity(count);
        let mut v = v_min;
        // Deliberate float walk: v steps through the exact integer-valued f64
        // trunc-cell labels floor(lo)..=floor(hi) (budget-checked above).
        #[allow(clippy::while_float)]
        while v <= v_max {
            values.push(v);
            v += 1.0;
        }
        per_dim_values.push(values);
    }

    // Cartesian product via odometer.
    let mut cells = Vec::with_capacity(total);
    let mut odo = vec![0usize; free_dims.len()];
    'outer: loop {
        let mut reps = Vec::with_capacity(free_dims.len());
        for (slot, &dim) in free_dims.iter().enumerate() {
            let v = per_dim_values[slot][odo[slot]];
            let (lo, hi) = vnnlib.input_bounds[dim];
            reps.push(cell_representative(v, lo, hi)?);
        }
        cells.push(Cell { reps });
        // Advance odometer.
        let mut pos = free_dims.len();
        loop {
            if pos == 0 {
                break 'outer;
            }
            pos -= 1;
            odo[pos] += 1;
            if odo[pos] < per_dim_values[pos].len() {
                break;
            }
            odo[pos] = 0;
            if pos == 0 {
                break 'outer;
            }
        }
    }

    Some(CellPlan {
        input_shape: input_shape.to_vec(),
        base_values,
        free_dims,
        cells,
    })
}

/// A representative point for cell `v` of range `[lo, hi]` (`lo >= 0`).
///
/// EXACTNESS: the network sees the free input only through `trunc`, and
/// `trunc(x) = v` for every real `x` in `[v, v+1)` (non-negative axis). Any
/// representative in `[max(lo, v), min(hi, v+1))` therefore induces the same
/// value at every Trunc output as EVERY other point of the cell — evaluating
/// at the representative decides the entire cell. `v + 0.5` is preferred
/// (exactly representable, comfortably inside); the fallbacks handle cells
/// clipped by the box. Returns `None` if no valid representative exists
/// (degenerate box), which fails the whole detection closed.
fn cell_representative(v: f64, lo: f64, hi: f64) -> Option<f64> {
    for candidate in [v + 0.5, v, lo, hi] {
        if candidate >= lo && candidate <= hi && candidate >= v && candidate < v + 1.0 {
            debug_assert_eq!(candidate.trunc(), v);
            return Some(candidate);
        }
    }
    None
}

/// Check that every free input dim feeds ONLY Trunc nodes, possibly through
/// one level of exact selection (Gather with constant indices / Slice) whose
/// consumers are all Trunc.
fn free_dims_gated_by_trunc(graph: &GraphNetwork, n: usize, free_dims: &[usize]) -> bool {
    let free: std::collections::HashSet<usize> = free_dims.iter().copied().collect();

    // Consumer map: node name -> consumer node names.
    let mut consumers: HashMap<&str, Vec<&str>> = HashMap::new();
    for name in graph.node_names() {
        let Some(node) = graph.node(name) else {
            return false;
        };
        for input in node.inputs() {
            consumers.entry(input.as_str()).or_default().push(name);
        }
    }

    let input_consumers = consumers
        .get(NETWORK_INPUT)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    if input_consumers.is_empty() {
        // No consumer reads the input at all: the graph cannot depend on the
        // free dims, but this is so unusual that we fail closed.
        return false;
    }

    for &consumer_name in input_consumers {
        let Some(node) = graph.node(consumer_name) else {
            return false;
        };
        let reads = input_read_set(node.layer(), n);
        let reads_free = match &reads {
            ReadSet::All => true,
            ReadSet::Dims(dims) => dims.iter().any(|d| free.contains(d)),
        };
        if !reads_free {
            continue; // touches only fixed (point) dims
        }
        match node.layer() {
            // Trunc applied directly to (a selection of) the input: the free
            // dims pass through trunc immediately.
            Layer::Trunc(_) => continue,
            // Exact selection: OK iff EVERY consumer of its output is Trunc
            // and it is not itself the network output.
            Layer::Gather(_) | Layer::Slice(_) => {
                if graph.output_name() == consumer_name {
                    return false;
                }
                let Some(next) = consumers.get(consumer_name) else {
                    // No consumers: selection output is dead — free dims
                    // cannot influence the output through it.
                    continue;
                };
                let all_trunc = next.iter().all(|next_name| {
                    graph
                        .node(next_name)
                        .is_some_and(|next_node| matches!(next_node.layer(), Layer::Trunc(_)))
                });
                if !all_trunc {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

enum ReadSet {
    All,
    Dims(Vec<usize>),
}

/// Which input dims a direct NETWORK_INPUT consumer reads (rank-1 input of
/// length `n`). Unknown layers conservatively read ALL dims.
fn input_read_set(layer: &Layer, n: usize) -> ReadSet {
    match layer {
        Layer::Gather(gather) => {
            let axis_ok = gather.axis_raw() == 0 || gather.axis_raw() == -1;
            match (axis_ok, gather.constant_indices()) {
                (true, Some(indices)) => {
                    let mut dims = Vec::with_capacity(indices.len());
                    for &raw in indices.iter() {
                        let idx = if raw < 0 { raw + n as i64 } else { raw };
                        if idx < 0 || idx >= n as i64 {
                            return ReadSet::All; // out of range: fail closed
                        }
                        dims.push(idx as usize);
                    }
                    ReadSet::Dims(dims)
                }
                _ => ReadSet::All,
            }
        }
        Layer::Slice(slice) => {
            if slice.axis == 0 || slice.axis == -1 {
                // ONNX clamps: read set is start..min(end, n). An end sentinel
                // (usize::MAX) or end-offset variant clamps to n — a SUPERSET
                // of the true reads, which is conservative for detection.
                let end = slice.end.min(n);
                let start = slice.start.min(end);
                ReadSet::Dims((start..end).collect())
            } else {
                ReadSet::All
            }
        }
        _ => ReadSet::All,
    }
}

/// Build the exact per-cell f64 input box (all dims points).
fn cell_input(plan: &CellPlan, cell: &Cell) -> Option<Interval64> {
    let mut values = plan.base_values.clone();
    for (slot, &dim) in plan.free_dims.iter().enumerate() {
        values[dim] = cell.reps[slot];
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&plan.input_shape), values).ok()?;
    Some(Interval64::point(arr))
}

/// Evaluate a single cell to a verdict via the sound f64 interval forward.
fn evaluate_cell(
    graph: &GraphNetwork,
    plan: &CellPlan,
    cell: &Cell,
    vnnlib: &VnnLibSpec,
) -> CellVerdict {
    let Some(input) = cell_input(plan, cell) else {
        return CellVerdict::Indeterminate;
    };
    let out = match graph.propagate_ibp_f64_cell(&input) {
        Ok(out) => out,
        Err(e) => {
            debug!("cell {:?}: f64 forward undecided: {e}", cell.reps);
            return CellVerdict::Indeterminate;
        }
    };
    let lower: Vec<f64> = out.lower.iter().copied().collect();
    let upper: Vec<f64> = out.upper.iter().copied().collect();
    if lower.len() != vnnlib.num_outputs {
        return CellVerdict::Indeterminate;
    }
    if box_definitely_safe(&lower, &upper, vnnlib) {
        CellVerdict::Safe
    } else if box_definitely_violated(&lower, &upper, vnnlib) {
        CellVerdict::Violated
    } else {
        CellVerdict::Indeterminate
    }
}

/// Clauses of the unsafe region (each clause a conjunction). Falls back to a
/// single clause of the flat constraint list.
fn unsafe_clauses(vnnlib: &VnnLibSpec) -> Vec<&[OutputConstraint]> {
    if vnnlib.output_constraint_clauses.is_empty() {
        vec![vnnlib.output_constraints.as_slice()]
    } else {
        vnnlib
            .output_constraint_clauses
            .iter()
            .map(|clause| clause.as_slice())
            .collect()
    }
}

/// A constraint DEFINITELY holds for every output in the box.
fn constraint_definite(c: &OutputConstraint, lower: &[f64], upper: &[f64]) -> bool {
    let l = |i: usize| lower.get(i).copied();
    let u = |i: usize| upper.get(i).copied();
    match c {
        OutputConstraint::LessEq(i, j) => matches!((u(*i), l(*j)), (Some(a), Some(b)) if a <= b),
        OutputConstraint::LessThan(i, j) => matches!((u(*i), l(*j)), (Some(a), Some(b)) if a < b),
        OutputConstraint::GreaterEq(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a >= b),
        OutputConstraint::GreaterThan(i, j) => {
            matches!((l(*i), u(*j)), (Some(a), Some(b)) if a > b)
        }
        OutputConstraint::LessEqConst(i, c) => u(*i).is_some_and(|a| a <= *c),
        OutputConstraint::LessThanConst(i, c) => u(*i).is_some_and(|a| a < *c),
        OutputConstraint::GreaterEqConst(i, c) => l(*i).is_some_and(|a| a >= *c),
        OutputConstraint::GreaterThanConst(i, c) => l(*i).is_some_and(|a| a > *c),
        _ => false, // unknown variants can never be proven to hold
    }
}

/// A constraint is IMPOSSIBLE for every output in the box.
fn constraint_impossible(c: &OutputConstraint, lower: &[f64], upper: &[f64]) -> bool {
    let l = |i: usize| lower.get(i).copied();
    let u = |i: usize| upper.get(i).copied();
    match c {
        OutputConstraint::LessEq(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a > b),
        OutputConstraint::LessThan(i, j) => matches!((l(*i), u(*j)), (Some(a), Some(b)) if a >= b),
        OutputConstraint::GreaterEq(i, j) => matches!((u(*i), l(*j)), (Some(a), Some(b)) if a < b),
        OutputConstraint::GreaterThan(i, j) => {
            matches!((u(*i), l(*j)), (Some(a), Some(b)) if a <= b)
        }
        OutputConstraint::LessEqConst(i, c) => l(*i).is_some_and(|a| a > *c),
        OutputConstraint::LessThanConst(i, c) => l(*i).is_some_and(|a| a >= *c),
        OutputConstraint::GreaterEqConst(i, c) => u(*i).is_some_and(|a| a < *c),
        OutputConstraint::GreaterThanConst(i, c) => u(*i).is_some_and(|a| a <= *c),
        _ => false, // unknown variants can never be refuted
    }
}

/// The whole output box avoids the unsafe region (cell proven safe).
pub(super) fn box_definitely_safe(lower: &[f64], upper: &[f64], vnnlib: &VnnLibSpec) -> bool {
    let clauses = unsafe_clauses(vnnlib);
    if clauses.iter().any(|clause| clause.is_empty()) {
        return false; // an empty clause is trivially satisfiable: never safe
    }
    if vnnlib.is_disjunction {
        // Unsafe = OR of clauses: safe iff EVERY clause is impossible.
        clauses.iter().all(|clause| {
            clause
                .iter()
                .any(|c| constraint_impossible(c, lower, upper))
        })
    } else {
        // Unsafe = AND of clauses: safe iff ANY clause is impossible.
        clauses.iter().any(|clause| {
            clause
                .iter()
                .any(|c| constraint_impossible(c, lower, upper))
        })
    }
}

/// The whole output box lies inside the unsafe region (cell violates).
fn box_definitely_violated(lower: &[f64], upper: &[f64], vnnlib: &VnnLibSpec) -> bool {
    let clauses = unsafe_clauses(vnnlib);
    if clauses.iter().any(|clause| clause.is_empty()) {
        return false;
    }
    let clause_definite =
        |clause: &&[OutputConstraint]| clause.iter().all(|c| constraint_definite(c, lower, upper));
    if vnnlib.is_disjunction {
        clauses.iter().any(clause_definite)
    } else {
        clauses.iter().all(clause_definite)
    }
}

/// Concrete output satisfies the unsafe region (clause-aware).
pub(super) fn concrete_violates(output: &ArrayD<f32>, vnnlib: &VnnLibSpec) -> bool {
    let clauses = unsafe_clauses(vnnlib);
    if clauses.iter().any(|clause| clause.is_empty()) {
        return false;
    }
    let clause_holds =
        |clause: &&[OutputConstraint]| super::verify::check_unsafe_counterexample(output, clause);
    if vnnlib.is_disjunction {
        clauses.iter().any(clause_holds)
    } else {
        clauses.iter().all(clause_holds)
    }
}

/// Confirm an interval-violating cell with a concrete forward and build the
/// `Violated` result whose witness the vnncomp ORT gate re-checks.
fn confirm_violation(
    model_net: &BetaCrownModel,
    plan: &CellPlan,
    cell: &Cell,
    vnnlib: &VnnLibSpec,
    start: Instant,
) -> Option<BetaCrownResult> {
    // f32 witness point: nearest-rounded per dim. The free-dim representative
    // v + 0.5 is exactly representable; fixed dims land within the widened
    // input box the rest of the pipeline uses.
    let mut values_f32: Vec<f32> = plan.base_values.iter().map(|&v| v as f32).collect();
    for (slot, &dim) in plan.free_dims.iter().enumerate() {
        values_f32[dim] = cell.reps[slot] as f32;
    }
    let point = ArrayD::from_shape_vec(IxDyn(&plan.input_shape), values_f32.clone()).ok()?;
    let input = ny_tensor::BoundedTensor::concrete(point).ok()?;
    let output = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_concrete_point(&input, None).ok()?
        }
        BetaCrownModel::Graph(graph) => graph.propagate_concrete_point(&input, None, None).ok()?,
    };
    let output_center = output.center();
    if !concrete_violates(&output_center, vnnlib) {
        return None;
    }
    Some(BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: values_f32,
            output: output_center.iter().copied().collect(),
        },
        domains_explored: 1,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start.elapsed(),
        output_bounds: None,
    })
}

#[cfg(test)]
mod tests;
