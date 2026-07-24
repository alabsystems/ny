// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Increment 5c — the pinned-column FOLD pass for the DAG-MIP escalation.
//
// MEASURED motivation (nn4sys mscn_128d/cardinality): the per-clause encoding
// is ~10,000 columns / ~8,000 rows with only ~50 ReLU binaries, and ~95% of
// the INPUT columns are PINNED (degenerate `[v, v]` bounds from the vnnlib
// clause box), with pinnedness propagating through the exact affine equality
// rows — yet ay sees the full 10K-column LP and times out. This pass
// substitutes provably-pinned columns into the rows BEFORE the problem is
// handed to the solver, shrinking it by one to two orders of magnitude.
//
// SEMANTIC CONTRACT (soundness-critical — a fold that changes the feasible
// set is a potential false-unsat): the folded system's feasible set must be
// the ORIGINAL system's feasible set projected onto the surviving columns,
// except where a bound is explicitly rounded OUTWARD (which only ENLARGES the
// feasible set — an enlargement can only turn a should-be-infeasible clause
// into a fail-closed fallback, never fabricate an infeasibility). Concretely:
//
//  1. A column is PINNED iff the constraint system forces it to a single f64
//     value `v` at EVERY feasible point:
//       * seed — its bounds are exactly `[v, v]`, `v` finite (the vnnlib
//         clause box pins; ReLU stable-inactive `[0, 0]` outputs);
//       * derived — a finite EQUALITY row over it and otherwise-pinned
//         columns solves to `x = v` in exact rational arithmetic with `v`
//         exactly f64-representable and inside the column's own bounds
//         (checked by exact comparison). The defining row's content is then
//         carried entirely by the pin, so the row is dropped.
//     Integer (ReLU indicator) columns only ever pin to exactly 0.0 or 1.0
//     (anything else is refused — fail open, the solver keeps the row).
//     OUTPUT columns are NEVER pinned (rule 5): the violation-threshold rows
//     reference them, and excluding them outright is simpler and safer than
//     folding thresholds.
//
//  2. EXACT ARITHMETIC: every `a·v` product and every partial sum is
//     accumulated in `num_rational::BigRational` (each finite f64 is an exact
//     rational, so products/sums NEVER round — the same exact seam ny-mip's
//     ay backend already uses for witness values). The single f64 rounding in
//     the whole pass is the final conversion of a folded row bound, and it is
//     DIRECTED OUTWARD per row sense:
//       * `Σ ≤ ub` rows: the folded ub rounds UP (toward +inf);
//       * `Σ ≥ lb` rows: the folded lb rounds DOWN (toward −inf);
//       * equality rows: if the exact folded rhs is not EXACTLY
//         f64-representable the row is kept UNFOLDED verbatim (an equality
//         with a rounded rhs would CHANGE the feasible set in an unsound
//         direction — deliberately not weakened into a two-sided interval
//         either, per the inc5c spec) and every pinned column it references
//         is kept (with its `[v, v]` bounds, so other rows' folds of the same
//         column stay exact).
//     Directed rounding is implemented by exact rational comparison
//     (`rat_to_f64_down`/`_up` verify and nudge the approximation), never by
//     trusting a float op.
//
//  3. A row whose every column is pinned becomes a CONSTANT comparison,
//     evaluated in exact rational arithmetic: satisfied → dropped (implied by
//     the pins); VIOLATED → [`FoldOutcome::ProvedInfeasible`]. The folder
//     NEVER turns that into a verdict: `try_graph_mip_escalation` treats it
//     as fail-closed for the WHOLE escalation (returns `None`), because the
//     LG3 posture is that ONLY ay's independently verified certificate
//     justifies an unsat — a folder-proved infeasibility would bypass that
//     gate. (On real instances it should be near-impossible anyway: the true
//     forward trajectory is feasible.)
//
//  4. A pinned column is REMOVED only when every row that references it was
//     folded (its `a·v` moved into the row constant) or dropped; a pinned
//     column referenced by a kept-verbatim row survives with `[v, v]` bounds
//     (exact — for a derived pin this re-expresses the dropped defining
//     row). Column indices are remapped DENSELY and the encoding's
//     input/output/binary column maps are remapped consistently
//     (`binary_widths` stays aligned with `binary_vars`).
//
// FAIL-CLOSED POSTURE: anything the pass cannot fold EXACTLY (or weaken
// outward) is simply kept — a partially-folded system is fine. Any internal
// inconsistency returns `Err` and the caller solves the UNFOLDED original,
// which is byte-identical to the pre-inc5c behavior.

use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, bail, Result};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_mip::ir::{Col, MilpProblem};
use tracing::info;

use super::graph_mip::GraphMipEncoding;

/// Size/action statistics of one fold pass (diagnostics only — no decision
/// rides on these).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FoldStats {
    pub cols_before: usize,
    pub cols_after: usize,
    pub rows_before: usize,
    pub rows_after: usize,
    pub binaries_before: usize,
    pub binaries_after: usize,
    /// Columns pinned by degenerate `[v, v]` bounds (the vnnlib clause box).
    pub pinned_from_bounds: usize,
    /// Columns pinned by solving a single-variable exact equality row.
    pub pinned_derived: usize,
    /// Rows kept VERBATIM because their folded equality rhs is not exactly
    /// f64-representable (the fail-closed lane of contract rule 2).
    pub rows_kept_unfolded: usize,
    /// Fully-pinned rows that evaluated (exactly) to a satisfied constant
    /// comparison and were dropped.
    pub rows_dropped_constant: usize,
}

/// A successfully folded encoding plus the maps a caller (or test oracle)
/// needs to relate it back to the original.
#[derive(Debug)]
pub(crate) struct FoldedEncoding {
    /// The folded encoding. `node_cols` is intentionally EMPTY: per-node
    /// introspection does not survive column removal — [`Self::col_map`] is
    /// the introspection surface for folded problems.
    pub encoding: GraphMipEncoding,
    /// Original column index → surviving column (`None` = folded away).
    /// Consumed by the fold parity oracle (tests) and future callers that
    /// need to project a full-column assignment onto the folded problem.
    #[allow(dead_code)]
    pub col_map: Vec<Option<Col>>,
    /// Original column index → its proven pinned value (`Some` for every
    /// pinned column, whether removed or kept with `[v, v]` bounds). Same
    /// consumers as `col_map`.
    #[allow(dead_code)]
    pub pins: Vec<Option<f64>>,
    pub stats: FoldStats,
}

/// Result of [`fold_pinned_columns`].
#[derive(Debug)]
pub(crate) enum FoldOutcome {
    /// The folded (possibly unchanged) system.
    Folded(Box<FoldedEncoding>),
    /// A fully-pinned row evaluated to a VIOLATED constant comparison in
    /// exact arithmetic. NOT a verdict: the caller must fail closed (see the
    /// module header, rule 3).
    ProvedInfeasible {
        /// Index of the violated row in the ORIGINAL problem.
        row: usize,
    },
}

/// Exact rational value of a finite f64 (`None` for NaN/±inf).
fn rat_of(x: f64) -> Option<BigRational> {
    BigRational::from_float(x)
}

/// Greatest f64 `x` with `x <= q` (directed rounding DOWN, verified by exact
/// rational comparison — the `to_f64` approximation is a starting point, not
/// a trusted value). Returns `-inf` when no finite f64 qualifies (or on any
/// pathology) — for a row LOWER bound, `-inf` is the weakest (open) side, so
/// the failure mode is outward, never inward.
fn rat_to_f64_down(q: &BigRational) -> f64 {
    let mut x = match q.to_f64() {
        Some(v) if v.is_finite() => v,
        // q likely above the finite range: f64::MAX <= q is then exact.
        Some(v) if v == f64::INFINITY => f64::MAX,
        // q below the finite range (or NaN pathology): -inf is always <= q.
        _ => return f64::NEG_INFINITY,
    };
    // Soundness first: walk DOWN until x <= q holds by exact comparison.
    let mut steps = 0usize;
    while rat_of(x).is_none_or(|r| r > *q) {
        x = x.next_down();
        if x == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        steps += 1;
        if steps > 1024 {
            return f64::NEG_INFINITY; // give up in the sound (outward) direction
        }
    }
    // Quality second: climb back UP while still <= q (tightest sound value).
    loop {
        let up = x.next_up();
        if !up.is_finite() {
            break;
        }
        match rat_of(up) {
            Some(r) if r <= *q => x = up,
            _ => break,
        }
        steps += 1;
        if steps > 2048 {
            break; // still sound, merely less tight
        }
    }
    x
}

/// Least f64 `x` with `x >= q` (directed rounding UP) — mirror of
/// [`rat_to_f64_down`]; the failure mode is `+inf` (outward for a row UPPER
/// bound).
fn rat_to_f64_up(q: &BigRational) -> f64 {
    -rat_to_f64_down(&-q)
}

/// `Some(x)` iff the exact rational `q` is EXACTLY representable as the
/// finite f64 `x`. Anything else — including values outside the finite range
/// — is `None` (callers fail closed).
fn rat_to_f64_exact(q: &BigRational) -> Option<f64> {
    let down = rat_to_f64_down(q);
    if down.is_finite() && rat_of(down).as_ref() == Some(q) {
        Some(down)
    } else {
        None
    }
}

/// Exact satisfaction of the constant comparison `lb <= s <= ub` (f64 bounds,
/// exact rational activity). `None` = uninterpretable bounds (NaN) — the
/// caller must keep the row (fail closed).
fn constant_row_satisfied(s: &BigRational, lb: f64, ub: f64) -> Option<bool> {
    if lb.is_nan() || ub.is_nan() {
        return None;
    }
    let lb_ok = lb == f64::NEG_INFINITY || rat_of(lb).is_some_and(|q| q <= *s);
    let ub_ok = ub == f64::INFINITY || rat_of(ub).is_some_and(|q| *s <= q);
    Some(lb_ok && ub_ok)
}

/// What happens to one original row in the folded problem.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RowPlan {
    /// Dropped: a satisfied constant row, or an equality row whose content is
    /// carried entirely by a derived pin.
    Drop,
    /// Kept byte-identically (no pinned references, or the fail-closed
    /// unfoldable lane — every pinned column it references is then kept).
    Verbatim,
    /// Kept with the pinned terms moved into the constant side: coefficients
    /// on pinned columns are removed and the bounds become `[lb, ub]`
    /// (exact for equalities, outward-rounded for inequalities).
    Folded { lb: f64, ub: f64 },
}

/// Fold provably-pinned columns into the row constants (module header for the
/// full contract). Pure function of the encoding — the original is left
/// untouched, so an `Err` caller can always fall back to solving it.
pub(crate) fn fold_pinned_columns(enc: &GraphMipEncoding) -> Result<FoldOutcome> {
    let problem = &enc.problem;
    let cols = problem.cols();
    let rows = problem.rows();
    let n_cols = cols.len();
    let n_rows = rows.len();

    // Rule 5: output columns are never pinned (hence never folded) — the
    // violation-threshold rows reference them.
    let mut is_output = vec![false; n_cols];
    for c in &enc.output_vars {
        *is_output
            .get_mut(c.0)
            .ok_or_else(|| anyhow!("output column {} out of range ({n_cols} cols)", c.0))? = true;
    }
    // Structural integrity up front: every row reference must be in range.
    for (r, row) in rows.iter().enumerate() {
        if row.coeffs.iter().any(|&(c, _)| c >= n_cols) {
            bail!("row {r} references a column beyond {n_cols}");
        }
    }

    // ── Pins: seed from degenerate `[v, v]` bounds (contract rule 1) ────────
    let mut pin: Vec<Option<f64>> = vec![None; n_cols];
    let mut pinned_from_bounds = 0usize;
    for (i, spec) in cols.iter().enumerate() {
        if is_output[i] {
            continue;
        }
        let v = spec.lb;
        // Exact f64 equality of the bounds; finite; integers only at 0/1
        // (rule 4 — anything else stays for the solver's integrality logic).
        if v.is_finite() && spec.lb == spec.ub && (!spec.integer || v == 0.0 || v == 1.0) {
            pin[i] = Some(v);
            pinned_from_bounds += 1;
        }
    }

    // ── Fixpoint: constant-row elimination + single-variable equality pin
    //    derivation (worklist over rows whose unpinned reference count drops
    //    to <= 1). Every derivation is exact-rational and fail-open on any
    //    obstacle (see the match arms).
    let mut dropped = vec![false; n_rows];
    let mut rows_dropped_constant = 0usize;
    let mut pinned_derived = 0usize;

    // Column → referencing rows (one entry per occurrence, so the count
    // bookkeeping below stays correct for duplicate references).
    let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); n_cols];
    let mut unpinned_refs: Vec<usize> = vec![0; n_rows];
    for (r, row) in rows.iter().enumerate() {
        for &(c, _) in &row.coeffs {
            col_rows[c].push(r as u32);
            if pin[c].is_none() {
                unpinned_refs[r] += 1;
            }
        }
    }
    let mut queue: VecDeque<usize> = (0..n_rows).filter(|&r| unpinned_refs[r] <= 1).collect();
    let mut queued = vec![false; n_rows];
    for &r in &queue {
        queued[r] = true;
    }

    while let Some(r) = queue.pop_front() {
        queued[r] = false;
        if dropped[r] || unpinned_refs[r] > 1 {
            continue;
        }
        let row = &rows[r];
        if row.lb.is_nan() || row.ub.is_nan() {
            continue; // uninterpretable bounds: keep verbatim (fail closed)
        }
        // Exact pinned-term sum S = Σ a·v and the single unpinned reference
        // (if any). A non-finite coefficient makes the row unfoldable.
        let mut s = BigRational::zero();
        let mut free: Option<(usize, BigRational)> = None;
        let mut foldable = true;
        for &(c, w) in &row.coeffs {
            let Some(wq) = rat_of(w) else {
                foldable = false;
                break;
            };
            match pin[c] {
                Some(v) => match rat_of(v) {
                    Some(vq) => s += wq * vq,
                    None => {
                        foldable = false;
                        break;
                    }
                },
                // At most one occurrence by the unpinned_refs <= 1 guard.
                None => free = Some((c, wq)),
            }
        }
        if !foldable {
            continue; // kept verbatim at materialization
        }

        // A zero-coefficient single variable constrains nothing: the row is a
        // constant comparison with respect to the feasible set.
        let effective_free = match free {
            Some((_, ref a)) if a.is_zero() => None,
            other => other,
        };
        match effective_free {
            None => {
                // Constant row: exact evaluation (contract rule 3).
                match constant_row_satisfied(&s, row.lb, row.ub) {
                    Some(true) => {
                        dropped[r] = true;
                        rows_dropped_constant += 1;
                    }
                    Some(false) => return Ok(FoldOutcome::ProvedInfeasible { row: r }),
                    None => {} // NaN bounds — unreachable (checked above); keep
                }
            }
            Some((c, a)) => {
                // Derive a pin ONLY from a finite equality row with a nonzero
                // exact coefficient; every other shape stays for the solver.
                if !(row.lb == row.ub && row.lb.is_finite()) || is_output[c] {
                    continue;
                }
                let Some(rhsq) = rat_of(row.lb) else { continue };
                let vq = (rhsq - &s) / &a;
                // Exactly-representable pins only (a non-representable value
                // cannot be stored as a f64 pin — fail closed, keep the row).
                let Some(v) = rat_to_f64_exact(&vq) else {
                    continue;
                };
                let spec = &cols[c];
                if spec.integer && !(v == 0.0 || v == 1.0) {
                    continue; // rule 4 — never pin a binary off {0, 1}
                }
                // The pin must sit INSIDE the column's own bounds (exact
                // comparison). Outside ⇒ the system is jointly infeasible,
                // but the folder only reports infeasibility for constant
                // rows (rule 3) — leave row + column for the solver.
                let lb_ok =
                    spec.lb == f64::NEG_INFINITY || rat_of(spec.lb).is_some_and(|q| q <= vq);
                let ub_ok = spec.ub == f64::INFINITY || rat_of(spec.ub).is_some_and(|q| vq <= q);
                if !(lb_ok && ub_ok) {
                    continue;
                }
                pin[c] = Some(v);
                pinned_derived += 1;
                // The row's content is now carried entirely by the pin.
                dropped[r] = true;
                // Re-examine every row that references the newly pinned col.
                for &rr in &col_rows[c] {
                    let rr = rr as usize;
                    unpinned_refs[rr] -= 1;
                    if !dropped[rr] && unpinned_refs[rr] <= 1 && !queued[rr] {
                        queued[rr] = true;
                        queue.push_back(rr);
                    }
                }
            }
        }
    }

    // ── Materialization: per-row fold plan + must-keep marks ────────────────
    let mut plans: Vec<RowPlan> = Vec::with_capacity(n_rows);
    let mut must_keep = vec![false; n_cols];
    let mut rows_kept_unfolded = 0usize;
    for (r, row) in rows.iter().enumerate() {
        if dropped[r] {
            plans.push(RowPlan::Drop);
            continue;
        }
        if !row.coeffs.iter().any(|&(c, _)| pin[c].is_some()) {
            plans.push(RowPlan::Verbatim); // nothing to fold
            continue;
        }
        // Fail-closed verbatim lane: keeps the row byte-identical AND keeps
        // every pinned column it references (with `[v, v]` bounds), so the
        // row's meaning is unchanged and other rows' folds stay exact.
        let keep_verbatim = |rows_kept_unfolded: &mut usize, must_keep: &mut Vec<bool>| {
            for &(c, _) in &row.coeffs {
                if pin[c].is_some() {
                    must_keep[c] = true;
                }
            }
            *rows_kept_unfolded += 1;
            RowPlan::Verbatim
        };
        // Uninterpretable bounds or a non-finite coefficient anywhere: the
        // row is unfoldable.
        if row.lb.is_nan() || row.ub.is_nan() || row.coeffs.iter().any(|&(_, w)| !w.is_finite()) {
            let plan = keep_verbatim(&mut rows_kept_unfolded, &mut must_keep);
            plans.push(plan);
            continue;
        }
        // Exact pinned-term sum (pins are finite by construction).
        let mut s = BigRational::zero();
        for &(c, w) in &row.coeffs {
            if let Some(v) = pin[c] {
                s += rat_of(w).expect("finite coeff") * rat_of(v).expect("finite pin");
            }
        }
        if row.lb == row.ub {
            // Equality row (contract rule 2): exact folded rhs or verbatim.
            if !row.lb.is_finite() {
                // lb == ub == ±inf is malformed; refuse to interpret it.
                let plan = keep_verbatim(&mut rows_kept_unfolded, &mut must_keep);
                plans.push(plan);
                continue;
            }
            let rhsq = rat_of(row.lb).expect("finite rhs") - &s;
            match rat_to_f64_exact(&rhsq) {
                Some(rhs) => plans.push(RowPlan::Folded { lb: rhs, ub: rhs }),
                None => {
                    let plan = keep_verbatim(&mut rows_kept_unfolded, &mut must_keep);
                    plans.push(plan);
                }
            }
        } else {
            // Inequality row: outward directed rounding per side (`>=` side
            // DOWN, `<=` side UP) — only ever WEAKENS the row.
            let lb = if row.lb == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                rat_to_f64_down(&(rat_of(row.lb).expect("finite lb") - &s))
            };
            let ub = if row.ub == f64::INFINITY {
                f64::INFINITY
            } else {
                rat_to_f64_up(&(rat_of(row.ub).expect("finite ub") - &s))
            };
            plans.push(RowPlan::Folded { lb, ub });
        }
    }

    // ── Column removal + dense remap ────────────────────────────────────────
    // Removable ⇔ pinned ∧ not an output ∧ not referenced by a verbatim-kept
    // row ∧ objective-free (a nonzero obj would silently shift the objective;
    // this encoder never sets one, but fail closed anyway).
    let mut new_problem = MilpProblem::new();
    let mut col_map: Vec<Option<Col>> = vec![None; n_cols];
    for (i, spec) in cols.iter().enumerate() {
        let removable = pin[i].is_some() && !is_output[i] && !must_keep[i] && spec.obj == 0.0;
        if removable {
            continue;
        }
        // A kept-but-pinned column carries its pin as `[v, v]` bounds: for a
        // bounds-pin that IS the original bound; for a derived pin this
        // re-expresses the dropped defining equality row exactly (the pin was
        // verified inside the original bounds).
        let (lb, ub) = match pin[i] {
            Some(v) => (v, v),
            None => (spec.lb, spec.ub),
        };
        let col = if spec.integer {
            new_problem.add_integer_col(spec.obj, lb, ub)
        } else {
            new_problem.add_col(spec.obj, lb, ub)
        };
        col_map[i] = Some(col);
    }

    let remap = |c: usize| -> Result<Col> {
        col_map[c].ok_or_else(|| anyhow!("folded column {c} still referenced by a kept row"))
    };
    for (r, plan) in plans.iter().enumerate() {
        let row = &rows[r];
        match *plan {
            RowPlan::Drop => {}
            RowPlan::Verbatim => {
                let coeffs: Vec<(Col, f64)> = row
                    .coeffs
                    .iter()
                    .map(|&(c, w)| remap(c).map(|col| (col, w)))
                    .collect::<Result<_>>()?;
                new_problem.add_row(row.lb, row.ub, coeffs);
            }
            RowPlan::Folded { lb, ub } => {
                let coeffs: Vec<(Col, f64)> = row
                    .coeffs
                    .iter()
                    .filter(|&&(c, _)| pin[c].is_none())
                    .map(|&(c, w)| remap(c).map(|col| (col, w)))
                    .collect::<Result<_>>()?;
                if coeffs.is_empty() {
                    // All-pinned rows are resolved by the fixpoint; reaching
                    // here means the pass lost a row's content — fail closed.
                    bail!("folded row {r} lost every column reference");
                }
                new_problem.add_row(lb, ub, coeffs);
            }
        }
    }

    // ── Remap the encoding's column handles consistently (rule 5) ───────────
    let input_vars: Vec<Col> = enc.input_vars.iter().filter_map(|c| col_map[c.0]).collect();
    let output_vars: Vec<Col> = enc
        .output_vars
        .iter()
        .map(|c| {
            col_map[c.0].ok_or_else(|| anyhow!("output column {} was folded (forbidden)", c.0))
        })
        .collect::<Result<_>>()?;
    let (binary_vars, binary_widths): (Vec<Col>, Vec<f64>) = enc
        .binary_vars
        .iter()
        .zip(&enc.binary_widths)
        .filter_map(|(c, &w)| col_map[c.0].map(|nc| (nc, w)))
        .unzip();

    let stats = FoldStats {
        cols_before: n_cols,
        cols_after: new_problem.num_cols(),
        rows_before: n_rows,
        rows_after: new_problem.num_rows(),
        binaries_before: enc.binary_vars.len(),
        binaries_after: binary_vars.len(),
        pinned_from_bounds,
        pinned_derived,
        rows_kept_unfolded,
        rows_dropped_constant,
    };
    Ok(FoldOutcome::Folded(Box::new(FoldedEncoding {
        encoding: GraphMipEncoding {
            problem: new_problem,
            input_vars,
            output_vars,
            binary_vars,
            binary_widths,
            // Per-node introspection does not survive column removal; the
            // col_map below is the folded problem's introspection surface.
            // (nor do the leaf-pin identities — the folded encoding is the
            // whole-net escalation surface, not the leaf lane.)
            binary_keys: Vec::new(),
            node_cols: HashMap::new(),
        },
        col_map,
        pins: pin,
        stats,
    })))
}

/// Apply the fold for one escalation clause ([`try_graph_mip_escalation`]).
///
///  * `Some(folded)` — the shrunk encoding to hand to the solver;
///  * `Some(original)` — the fold errored internally: solve the UNFOLDED
///    original (byte-identical to pre-inc5c behavior — sound fallback);
///  * `None` — the folder proved a constant row VIOLATED: the WHOLE
///    escalation must fail closed (LG3: only ay's verified certificate
///    justifies unsat; the folder never emits verdicts — module header,
///    rule 3).
pub(crate) fn fold_for_escalation(
    enc: GraphMipEncoding,
    clause_idx: usize,
) -> Option<GraphMipEncoding> {
    match fold_pinned_columns(&enc) {
        Ok(FoldOutcome::Folded(folded)) => {
            let s = folded.stats;
            info!(
                "graph-MIP escalation: clause {} fold: {}→{} cols, {}→{} rows, {}→{} binaries \
                 ({} bound-pinned + {} derived-pinned cols; {} constant rows dropped, {} rows \
                 kept unfolded)",
                clause_idx + 1,
                s.cols_before,
                s.cols_after,
                s.rows_before,
                s.rows_after,
                s.binaries_before,
                s.binaries_after,
                s.pinned_from_bounds,
                s.pinned_derived,
                s.rows_dropped_constant,
                s.rows_kept_unfolded,
            );
            Some(folded.encoding)
        }
        Ok(FoldOutcome::ProvedInfeasible { row }) => {
            info!(
                "graph-MIP escalation: clause {} fold evaluated constant row {row} as VIOLATED \
                 (exact arithmetic). The folder never emits verdicts — only ay's verified \
                 certificate justifies unsat (LG3) — so the whole escalation fails closed",
                clause_idx + 1
            );
            None
        }
        Err(e) => {
            info!(
                "graph-MIP escalation: clause {} fold failed ({e:#}); solving the UNFOLDED \
                 encoding (sound fallback)",
                clause_idx + 1
            );
            Some(enc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built encoding wrapper: `outputs` marks output columns by index;
    /// every other column doubles as an "input" (exercising the input remap).
    fn enc_of(problem: MilpProblem, outputs: &[usize]) -> GraphMipEncoding {
        let n = problem.num_cols();
        GraphMipEncoding {
            problem,
            input_vars: (0..n).filter(|i| !outputs.contains(i)).map(Col).collect(),
            output_vars: outputs.iter().map(|&i| Col(i)).collect(),
            binary_vars: Vec::new(),
            binary_widths: Vec::new(),
            binary_keys: Vec::new(),
            node_cols: HashMap::new(),
        }
    }

    fn fold(enc: &GraphMipEncoding) -> FoldedEncoding {
        match fold_pinned_columns(enc).expect("fold must not error") {
            FoldOutcome::Folded(folded) => *folded,
            FoldOutcome::ProvedInfeasible { row } => {
                panic!("unexpected ProvedInfeasible at row {row}")
            }
        }
    }

    /// (a) Fully exact fold on a hand-built system: folded rows/bounds match
    /// hand-computed values; the column remap is dense and consistent; a
    /// satisfied constant row is dropped.
    #[test]
    fn fold_exact_hand_system() {
        let mut p = MilpProblem::new();
        let x0 = p.add_col(0.0, 2.0, 2.0); // pinned 2.0
        let x1 = p.add_col(0.0, -10.0, 10.0); // free
        let x2 = p.add_col(0.0, 0.5, 0.5); // pinned 0.5
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
                                                                  // r0 (equality): 3·x0 + x1 − y = −1  →  x1 − y = −7 (exact).
        p.add_row(-1.0, -1.0, [(x0, 3.0), (x1, 1.0), (y, -1.0)]);
        // r1 (inequality): 2·x2 + x1 ≤ 5  →  x1 ≤ 4 (exact).
        p.add_row(f64::NEG_INFINITY, 5.0, [(x2, 2.0), (x1, 1.0)]);
        // r2 (constant): x0 − x2 ∈ [1, 3] → 1.5 ∈ [1, 3]: satisfied, dropped.
        p.add_row(1.0, 3.0, [(x0, 1.0), (x2, -1.0)]);
        let enc = enc_of(p, &[y.0]);

        let folded = fold(&enc);
        let s = folded.stats;
        assert_eq!((s.cols_before, s.cols_after), (4, 2));
        assert_eq!((s.rows_before, s.rows_after), (3, 2));
        assert_eq!(s.pinned_from_bounds, 2);
        assert_eq!(s.rows_dropped_constant, 1);
        assert_eq!(s.rows_kept_unfolded, 0);

        // Dense remap: x1 → Col(0), y → Col(1); pinned cols removed.
        assert_eq!(folded.col_map, vec![None, Some(Col(0)), None, Some(Col(1))]);
        assert_eq!(folded.pins, vec![Some(2.0), None, Some(0.5), None]);
        assert_eq!(folded.encoding.input_vars, vec![Col(0)]); // only x1 survives
        assert_eq!(folded.encoding.output_vars, vec![Col(1)]);

        let rows = folded.encoding.problem.rows();
        assert_eq!(rows.len(), 2);
        // r0 folded: x1 − y = −7.
        assert_eq!(rows[0].coeffs, vec![(0, 1.0), (1, -1.0)]);
        assert_eq!((rows[0].lb, rows[0].ub), (-7.0, -7.0));
        // r1 folded: x1 ≤ 4.
        assert_eq!(rows[1].coeffs, vec![(0, 1.0)]);
        assert_eq!((rows[1].lb, rows[1].ub), (f64::NEG_INFINITY, 4.0));
        // Surviving column bounds are untouched.
        let cols = folded.encoding.problem.cols();
        assert_eq!((cols[0].lb, cols[0].ub), (-10.0, 10.0));
    }

    /// Derived pins chain through exact equality rows: x pinned → h derived
    /// (−h = −3 − x) → downstream row folds h away too.
    #[test]
    fn fold_derives_pins_through_equality_chain() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 2.0, 2.0); // pinned 2.0
        let h = p.add_col(0.0, -100.0, 100.0); // derived: h = x + 3 = 5
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
                                                                  // h − x = 3 (equality, single unpinned var h after x's pin).
        p.add_row(3.0, 3.0, [(h, 1.0), (x, -1.0)]);
        // y − h ≥ −1 → y ≥ 4 (folds h's derived pin).
        p.add_row(-1.0, f64::INFINITY, [(y, 1.0), (h, -1.0)]);
        let enc = enc_of(p, &[y.0]);

        let folded = fold(&enc);
        assert_eq!(folded.pins[h.0], Some(5.0));
        assert_eq!(folded.stats.pinned_derived, 1);
        assert_eq!(folded.stats.cols_after, 1); // only y survives
        let rows = folded.encoding.problem.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].coeffs, vec![(0, 1.0)]);
        assert_eq!((rows[0].lb, rows[0].ub), (4.0, f64::INFINITY));
    }

    /// (b) Inexact accumulation on an INEQUALITY row: 0.1·0.1 is not exactly
    /// f64-representable, so the folded ub must round OUTWARD (up) by at most
    /// one ulp — verified against the exact rational product — and the pinned
    /// column still folds away.
    #[test]
    fn fold_inexact_inequality_rounds_outward() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.1, 0.1); // pinned to the f64 nearest 0.1
        let z = p.add_col(0.0, -10.0, 10.0);
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
                                                                  // 0.1·x + z ≤ 0.05 → z ≤ 0.05 − 0.1·0.1 (inexact → outward).
        p.add_row(f64::NEG_INFINITY, 0.05, [(x, 0.1), (z, 1.0)]);
        p.add_row(0.0, 0.0, [(z, 1.0), (y, -1.0)]); // keep z/y related
        let enc = enc_of(p, &[y.0]);

        let exact = rat_of(0.05).unwrap() - rat_of(0.1).unwrap() * rat_of(0.1).unwrap();
        // The exact value must NOT be f64-representable for this test to bite.
        assert!(
            rat_to_f64_exact(&exact).is_none(),
            "test fixture must be inexact"
        );

        let folded = fold(&enc);
        assert_eq!(folded.col_map[x.0], None, "pinned col folds away");
        let row = &folded.encoding.problem.rows()[0];
        assert_eq!(row.coeffs.len(), 1);
        let ub = row.ub;
        // Outward (never inward), and tight to one ulp.
        assert!(rat_of(ub).unwrap() >= exact, "folded ub must round UP");
        assert!(
            rat_of(ub.next_down()).unwrap() < exact,
            "folded ub must be the TIGHTEST sound f64"
        );
    }

    /// (c) An EQUALITY row whose folded rhs is not exactly f64-representable
    /// fails closed: the row is kept verbatim and the pinned column survives
    /// with its `[v, v]` bounds.
    #[test]
    fn fold_inexact_equality_fails_closed() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.1, 0.1); // pinned
        let z = p.add_col(0.0, -10.0, 10.0);
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
                                                                  // 0.1·x + z = 0.05: folded rhs 0.05 − 0.01… is inexact → keep row.
        p.add_row(0.05, 0.05, [(x, 0.1), (z, 1.0)]);
        // A second, exactly foldable row on x: x + y ≤ 3 → y ≤ 2.9 exactly?
        // (0.1 IS a f64 value; 3 − 0.1 rounds — use outwardness assertion.)
        p.add_row(f64::NEG_INFINITY, 3.0, [(x, 1.0), (y, 1.0)]);
        let enc = enc_of(p, &[y.0]);

        let folded = fold(&enc);
        assert_eq!(folded.stats.rows_kept_unfolded, 1);
        // x kept (must-keep from the verbatim row), with its [v, v] bounds.
        let xc = folded.col_map[x.0].expect("x must survive");
        let spec = folded.encoding.problem.cols()[xc.0];
        assert_eq!((spec.lb, spec.ub), (0.1, 0.1));
        // The equality row is byte-identical (same coeffs, same rhs).
        let rows = folded.encoding.problem.rows();
        assert_eq!(rows[0].coeffs.len(), 2);
        assert_eq!((rows[0].lb, rows[0].ub), (0.05, 0.05));
        // The second row still folded x's term outward.
        assert_eq!(rows[1].coeffs.len(), 1);
        let exact = rat_of(3.0).unwrap() - rat_of(0.1).unwrap();
        assert!(rat_of(rows[1].ub).unwrap() >= exact);
    }

    /// (d) A VIOLATED constant row returns the distinguished
    /// `ProvedInfeasible` — and the escalation wrapper maps it to `None`
    /// (fail closed for the whole escalation).
    #[test]
    fn fold_violated_constant_row_fails_closed() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 2.0, 2.0); // pinned 2.0
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
        p.add_row(5.0, 5.0, [(x, 1.0)]); // 2.0 = 5.0: VIOLATED
        p.add_row(0.0, 0.0, [(x, 1.0), (y, -1.0)]);
        let enc = enc_of(p, &[y.0]);

        match fold_pinned_columns(&enc).expect("fold must not error") {
            FoldOutcome::ProvedInfeasible { row } => assert_eq!(row, 0),
            other => panic!("expected ProvedInfeasible, got {other:?}"),
        }
        // The escalation treats it as fail-closed (None), never a verdict.
        assert!(fold_for_escalation(enc, 0).is_none());
    }

    /// (4) A binary pinned to exactly 1.0 folds out of its big-M rows with
    /// exact constants, and `binary_vars`/`binary_widths` stay aligned.
    #[test]
    fn fold_binary_pin_simplifies_big_m_rows() {
        let l = -3.0f64;
        let u = 2.0f64;
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, l, u);
        let y = p.add_col(0.0, 0.0, u);
        let z_pinned = p.add_integer_col(0.0, 1.0, 1.0); // fix_col'd binary
        let z_free = p.add_integer_col(0.0, 0.0, 1.0);
        let out = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
                                                                    // Big-M rows on the pinned binary:
                                                                    // y − x − l·z ≤ −l  →  y − x ≤ −l + l·1 = 0 (exact).
        p.add_row(f64::NEG_INFINITY, -l, [(y, 1.0), (x, -1.0), (z_pinned, -l)]);
        // y − u·z ≤ 0  →  y ≤ u (exact).
        p.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z_pinned, -u)]);
        // Keep the free binary + output alive.
        p.add_row(f64::NEG_INFINITY, 1.0, [(out, 1.0), (z_free, 1.0)]);
        let mut enc = enc_of(p, &[out.0]);
        enc.binary_vars = vec![z_pinned, z_free];
        enc.binary_widths = vec![5.0, 7.0];

        let folded = fold(&enc);
        assert_eq!(folded.pins[z_pinned.0], Some(1.0));
        assert_eq!(folded.col_map[z_pinned.0], None);
        // binary maps: only the free binary survives, width still aligned.
        assert_eq!(folded.encoding.binary_vars.len(), 1);
        assert_eq!(folded.encoding.binary_widths, vec![7.0]);
        let rows = folded.encoding.problem.rows();
        // y − x ≤ 0 and y ≤ 2 with exact constants.
        assert_eq!((rows[0].lb, rows[0].ub), (f64::NEG_INFINITY, 0.0));
        assert_eq!(rows[0].coeffs.len(), 2);
        assert_eq!((rows[1].lb, rows[1].ub), (f64::NEG_INFINITY, 2.0));
        assert_eq!(rows[1].coeffs.len(), 1);
    }

    /// A binary with fractional degenerate bounds is NEVER pinned (rule 4).
    #[test]
    fn fold_refuses_fractional_binary_pin() {
        let mut p = MilpProblem::new();
        let z = p.add_integer_col(0.0, 0.5, 0.5); // malformed: fractional pin
        let y = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY); // output
        p.add_row(0.0, 0.0, [(z, 1.0), (y, -1.0)]);
        let mut enc = enc_of(p, &[y.0]);
        enc.binary_vars = vec![z];
        enc.binary_widths = vec![1.0];

        let folded = fold(&enc);
        assert_eq!(folded.pins[z.0], None);
        assert_eq!(folded.stats.cols_after, 2);
        assert_eq!(folded.encoding.problem.rows().len(), 1);
    }

    /// (5) Output columns are NEVER folded, even with degenerate bounds; the
    /// threshold rows referencing them survive.
    #[test]
    fn fold_never_touches_output_columns() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 1.0, 1.0); // pinned input
        let y = p.add_col(0.0, 4.0, 4.0); // output with DEGENERATE bounds
        p.add_row(3.0, 3.0, [(y, 1.0), (x, -1.0)]); // y − x = 3
        p.add_row(f64::NEG_INFINITY, 5.0, [(y, 1.0)]); // threshold row
        let enc = enc_of(p, &[y.0]);

        let folded = fold(&enc);
        assert_eq!(folded.pins[y.0], None, "outputs are never pinned");
        let yc = folded.col_map[y.0].expect("output survives");
        assert_eq!(folded.encoding.output_vars, vec![yc]);
        let rows = folded.encoding.problem.rows();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].lb, rows[0].ub), (4.0, 4.0)); // y = 3 + 1 folded exactly
        assert_eq!((rows[1].lb, rows[1].ub), (f64::NEG_INFINITY, 5.0));
    }

    /// Directed-rounding helpers: exact values round to themselves in both
    /// directions; inexact values straddle within one ulp.
    #[test]
    fn rational_directed_rounding_is_exact_and_tight() {
        for v in [0.0, 1.0, -2.5, 0.1, f64::MAX, -f64::MAX, 5e-324] {
            let q = rat_of(v).unwrap();
            assert_eq!(rat_to_f64_down(&q), v);
            assert_eq!(rat_to_f64_up(&q), v);
            assert_eq!(rat_to_f64_exact(&q), Some(v));
        }
        // 0.1 + 0.2 exactly (as rationals of the f64 values) is not a f64.
        let q = rat_of(0.1).unwrap() + rat_of(0.2).unwrap();
        let down = rat_to_f64_down(&q);
        let up = rat_to_f64_up(&q);
        assert!(rat_of(down).unwrap() < q && q < rat_of(up).unwrap());
        assert_eq!(down.next_up(), up, "straddle must be one ulp wide");
        assert_eq!(rat_to_f64_exact(&q), None);
        // Beyond the finite range: outward extremes.
        let huge = rat_of(f64::MAX).unwrap() * rat_of(2.0).unwrap();
        assert_eq!(rat_to_f64_up(&huge), f64::INFINITY);
        assert_eq!(rat_to_f64_down(&huge), f64::MAX);
        assert_eq!(rat_to_f64_exact(&huge), None);
    }
}
