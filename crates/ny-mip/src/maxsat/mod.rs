// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! WCNF to binary MILP using NY's existing solver and exact integer coefficients.
//!
//! A returned point is only an incumbent, independently replayed against the
//! original clauses. Integral optimization status is never an optimality proof.
//! This module does not export an UNSAT claim or invoke another solver process.

mod parse;
#[cfg(test)]
mod tests;

use std::io::BufRead;
use std::time::Instant;

use crate::ir::{Col, MilpProblem};
use crate::{MipBackend, MipConfig, MipParts, MipResult, MipSolver};

/// Largest consecutive nonnegative integer exactly representable as f64.
pub const MAX_EXACT_INTEGER: u64 = 1_u64 << 53;
/// Fixed parser/encoding bounds; exceeding one returns unsupported/unknown.
pub const MAX_VARIABLES: usize = 2_000_000;
/// Maximum retained clauses, including empty clauses.
pub const MAX_CLAUSES: usize = 4_000_000;
/// Maximum retained literal occurrences, before any simplification.
pub const MAX_LITERALS: usize = 32_000_000;

/// Parsing, budget, encoding, or candidate-replay refusal.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An original input could not be read.
    #[error("WCNF input: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed or unsupported input is not approximated.
    #[error("unsupported WCNF: {0}")]
    Unsupported(String),
    /// The one caller-owned deadline expired.
    #[error("WCNF deadline reached")]
    Deadline,
    /// A solver point failed independent original-clause replay.
    #[error("candidate rejected: {0}")]
    Candidate(String),
    /// Existing NY MIP API failed before a usable result.
    #[error("NY MIP: {0}")]
    Mip(#[from] crate::MipError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Clause {
    weight: Option<u64>,
    literals: Vec<i32>,
}

/// Original parsed clause data, retained separately from the MILP encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wcnf {
    variables: usize,
    clauses: Vec<Clause>,
    soft_total: u64,
}

/// A checked feasible assignment; `cost` is recomputed from original clauses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Assignment for original variables 1 through n, including unused ones.
    pub assignment: Vec<bool>,
    /// Exact sum of weights of falsified original soft clauses.
    pub cost: u64,
}

/// No variant claims an optimum or proves hard-clause infeasibility.
#[derive(Debug)]
pub enum Outcome {
    /// A point passed independent original-clause replay.
    Incumbent(Candidate),
    /// No independently validated point was returned within the supported lane.
    Unknown(String),
}

fn before_deadline(deadline: Instant) -> Result<(), Error> {
    if Instant::now() >= deadline {
        Err(Error::Deadline)
    } else {
        Ok(())
    }
}

impl Wcnf {
    /// Read legacy `p wcnf` or headerless `h`/integer-weight WCNF without
    /// truncation, weight rounding, normalization, or clause deletion.
    pub fn read(reader: impl BufRead, deadline: Instant) -> Result<Self, Error> {
        parse::read(reader, deadline)
    }

    /// Number of original variables, including declared but unused variables.
    pub fn variables(&self) -> usize {
        self.variables
    }

    /// Check strict Boolean coordinates and replay every original clause.
    pub fn validate_values(&self, values: &[f64]) -> Result<Candidate, Error> {
        if values.len() != self.variables {
            return Err(Error::Candidate("assignment length differs".into()));
        }
        let assignment: Vec<_> = values
            .iter()
            .map(|&v| {
                if v == 0.0 {
                    Ok(false)
                } else if v == 1.0 {
                    Ok(true)
                } else {
                    Err(Error::Candidate(
                        "non-Boolean or non-finite coordinate".into(),
                    ))
                }
            })
            .collect::<Result<_, _>>()?;
        let cost = self.cost(&assignment)?;
        Ok(Candidate { assignment, cost })
    }

    fn cost(&self, assignment: &[bool]) -> Result<u64, Error> {
        if assignment.len() != self.variables {
            return Err(Error::Candidate("assignment length differs".into()));
        }
        let mut cost = 0_u64;
        for clause in &self.clauses {
            let satisfied = clause
                .literals
                .iter()
                .any(|&literal| assignment[literal.unsigned_abs() as usize - 1] == (literal > 0));
            if !satisfied {
                let weight = clause
                    .weight
                    .ok_or_else(|| Error::Candidate("hard clause is false".into()))?;
                cost = cost
                    .checked_add(weight)
                    .ok_or_else(|| Error::Candidate("cost overflow".into()))?;
            }
        }
        Ok(cost)
    }

    /// Construct NY's existing MILP IR. Positive literals contribute x;
    /// negative literals contribute 1-x. A soft clause gets a binary r with
    /// sum(literals)+r >= 1, and z = sum(weight*r) is the sole objective.
    pub fn encode(&self, deadline: Instant) -> Result<MipParts, Error> {
        before_deadline(deadline)?;
        let mut problem = MilpProblem::new();
        let inputs: Vec<_> = (0..self.variables)
            .map(|_| problem.add_integer_col(0.0, 0.0, 1.0))
            .collect();
        let mut objective_terms = Vec::new();
        for (index, clause) in self.clauses.iter().enumerate() {
            if index % 1024 == 0 {
                before_deadline(deadline)?;
            }
            let mut negative = 0_i64;
            let mut terms = Vec::with_capacity(clause.literals.len() + 1);
            for (position, &literal) in clause.literals.iter().enumerate() {
                if position % 4096 == 0 {
                    before_deadline(deadline)?;
                }
                let col = inputs[literal.unsigned_abs() as usize - 1];
                terms.push((
                    col,
                    if literal > 0 {
                        1.0
                    } else {
                        negative += 1;
                        -1.0
                    },
                ));
            }
            if let Some(weight) = clause.weight {
                let relaxed = problem.add_integer_col(0.0, 0.0, 1.0);
                terms.push((relaxed, 1.0));
                objective_terms.push((relaxed, -(weight as f64)));
            }
            // MAX_LITERALS bounds this constant far below 2^53.
            problem.add_row((1 - negative) as f64, f64::INFINITY, terms);
        }
        let objective = problem.add_col(0.0, 0.0, self.soft_total as f64);
        objective_terms.push((objective, 1.0));
        problem.add_row(0.0, 0.0, objective_terms);
        before_deadline(deadline)?;
        Ok(MipParts {
            num_cols: problem.num_cols(),
            problem,
            input_vars: inputs,
            output_vars: vec![objective],
            // These metadata fields describe ReLU-specific phase advice.
            // Integrality lives in the IR; WCNF variables are not ReLU phases.
            binary_vars: Vec::<Col>::new(),
            binary_widths: Vec::new(),
        })
    }

    /// Search once through NY's production in-process MIP route. The one
    /// absolute deadline covers parsing (caller), encoding, and optimization.
    /// Backend objective, dual bound and UNSAT status confer no output claim.
    pub fn solve(&self, deadline: Instant) -> Result<Outcome, Error> {
        let parts = self.encode(deadline)?;
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if remaining < 0.001 {
            return Err(Error::Deadline);
        }
        let config = MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            timeout_secs: remaining,
            ay_hard_deadline: Some(deadline),
            ..MipConfig::default()
        };
        self.interpret_result(MipSolver::new(parts, config).minimize_output(0)?, deadline)
    }

    fn interpret_result(&self, result: MipResult, deadline: Instant) -> Result<Outcome, Error> {
        before_deadline(deadline)?;
        match result {
            MipResult::Sat { input_values, .. } => {
                let candidate = self.validate_values(&input_values)?;
                before_deadline(deadline)?;
                Ok(Outcome::Incumbent(candidate))
            }
            MipResult::Unsat { .. } => Ok(Outcome::Unknown(
                "no incumbent; infeasibility certificate is not exported by this frontend".into(),
            )),
            MipResult::Timeout => Ok(Outcome::Unknown(
                "NY deadline or iteration limit; no returned incumbent".into(),
            )),
            MipResult::Error(reason) => Ok(Outcome::Unknown(reason)),
        }
    }
}
