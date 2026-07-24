// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// ay SMT backend for the solver-neutral MILP IR (P0 baseline).
//
// Lowers a [`crate::ir::MilpProblem`] to SMT-LIB2 QF_LRA — ReLU indicator
// binaries become 0/1 disjunctions over Real variables, which plays to ay's
// DPLL(T) case-splitting instead of forcing a LIRA mix — and drives the
// external `ay` binary (discovered via `$NY_AY`, else `ay` on `$PATH`, the
// same convention as the gt/cnf routes). The lowering is EXACT: every f64
// bound and coefficient is emitted as the precise rational it denotes, so an
// ay `unsat` speaks about the same problem the float backends solve, and the
// certificate lane (docs/AY_MIP_P0.md, gate G0) certifies the real thing.
//
// Soundness contract (identical to the HiGHS/SCIP arms): a `Sat` witness is
// only a candidate — callers revalidate it with a concrete forward pass;
// `Unsat` requires the solver's proven infeasibility; `unknown`, a timeout,
// or any lowering/parse failure degrades to `Timeout`/`Error`, never `Unsat`.
//
// Reference: designs/2026-07-12-gurobi-class-milp-for-ny.md (ay repo), P0.

mod lower;
mod parse;
mod run;

use crate::error::MipError;
use crate::ir::{Col, MilpProblem};
use crate::solver::MipResult;

pub(crate) use lower::{to_smtlib, ObjSense, ObjectiveSpec};

type Result<T> = std::result::Result<T, MipError>;

/// Feasibility check via the external ay solver.
///
/// `input_vars` / `output_vars` select which columns are extracted into the
/// witness on `sat` (matching the HiGHS/SCIP arms). The warm-start seed is
/// accepted for signature parity but ignored — SMT-LIB has no primal seeding
/// surface; it is a performance hint, never a correctness requirement.
pub(crate) fn check_feasibility(
    problem: &MilpProblem,
    timeout_secs: f64,
    input_vars: &[Col],
    output_vars: &[Col],
    warm_start_cols: Option<&[f64]>,
) -> Result<MipResult> {
    if warm_start_cols.is_some() {
        tracing::debug!("ay backend: warm-start seed ignored (no SMT-LIB seeding surface)");
    }
    solve(problem, timeout_secs, None, input_vars, output_vars)
}

/// Optimize a single column via ay's OMT surface.
///
/// On `sat` the reported objective is the target column's value in the
/// optimum model (parity with the HiGHS oracle arm, which reads the target
/// column rather than the model objective).
///
/// Only ay's `(minimize)` lane is used: ay 0.11.0's maximize lane silently
/// returns wrong optima for equality-defined variables in every spelling
/// probed (repro in the ay design doc, P0 findings), while minimize is
/// exact. Maximize therefore lowers to minimizing a fresh auxiliary column
/// `neg` constrained by `target + neg = 0`; the target's model value is
/// still read directly, and the caller's outward rounding absorbs the
/// (sub-f32-ULP) binary-search epsilon of ay's iterative lane.
pub(crate) fn optimize_col(
    problem: &MilpProblem,
    timeout_secs: f64,
    objective: ObjectiveSpec,
    input_vars: &[Col],
    output_vars: &[Col],
) -> Result<MipResult> {
    match objective.sense {
        ObjSense::Minimize => solve(
            problem,
            timeout_secs,
            Some(Objective {
                minimize: objective.col,
                report: objective.col,
            }),
            input_vars,
            output_vars,
        ),
        ObjSense::Maximize => {
            let mut negated = problem.clone();
            let neg = negated.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
            negated.add_row(0.0, 0.0, [(objective.col, 1.0), (neg, 1.0)]);
            solve(
                &negated,
                timeout_secs,
                Some(Objective {
                    minimize: neg,
                    report: objective.col,
                }),
                input_vars,
                output_vars,
            )
        }
    }
}

/// Internal solve-time objective: which column ay minimizes, and which
/// column's model value is reported as the objective.
#[derive(Debug, Clone, Copy)]
struct Objective {
    minimize: Col,
    report: Col,
}

fn solve(
    problem: &MilpProblem,
    timeout_secs: f64,
    objective: Option<Objective>,
    input_vars: &[Col],
    output_vars: &[Col],
) -> Result<MipResult> {
    let script = to_smtlib(
        problem,
        objective.map(|o| ObjectiveSpec {
            col: o.minimize,
            sense: ObjSense::Minimize,
        }),
    )?;
    let output = match run::run_ay(&script, timeout_secs)? {
        run::AyRun::Completed(output) => output,
        run::AyRun::TimedOut => return Ok(MipResult::Timeout),
    };
    match parse::parse_verdict(&output) {
        parse::Verdict::Sat => {
            let values = parse::parse_values(&output, problem.num_cols())?;
            let extract =
                |cols: &[Col]| -> Vec<f64> { cols.iter().map(|&c| values[c.0]).collect() };
            let objective_value = objective.map_or(0.0, |o| values[o.report.0]);
            Ok(MipResult::Sat {
                objective: objective_value,
                output_values: extract(output_vars),
                input_values: extract(input_vars),
                // The frozen P0 subprocess lane reports `sat` only for a
                // COMPLETED solve (interrupted -> `unknown` -> Timeout), and
                // carries no certificate evidence — no rigorous dual bound.
                dual_bound: None,
            })
        }
        // The subprocess lane carries no certificate evidence.
        parse::Verdict::Unsat => Ok(MipResult::Unsat { certified: false }),
        // `unknown` is inconclusive: degrade to Timeout (aggregation treats it
        // soundly — never contributes to an Unsat verdict).
        parse::Verdict::Unknown => Ok(MipResult::Timeout),
        parse::Verdict::Missing => Ok(MipResult::Error(format!(
            "ay produced no verdict; first output bytes: {:?}",
            &output[..output.len().min(200)]
        ))),
    }
}
