// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for VNNLib constraint normalization.

use std::collections::BTreeMap;

pub(super) const COEFF_EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum VarKind {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LinearVar {
    pub kind: VarKind,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearExpr {
    pub(super) coeffs: BTreeMap<LinearVar, f64>,
    pub(super) constant: f64,
}

impl LinearExpr {
    pub fn constant(value: f64) -> Self {
        Self {
            coeffs: BTreeMap::new(),
            constant: value,
        }
    }

    pub fn var(var: LinearVar) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(var, 1.0);
        Self {
            coeffs,
            constant: 0.0,
        }
    }

    #[cfg(test)]
    pub fn coeff(&self, kind: VarKind, index: usize) -> f64 {
        self.coeffs
            .get(&LinearVar { kind, index })
            .copied()
            .unwrap_or(0.0)
    }

    pub fn constant_term(&self) -> f64 {
        self.constant
    }

    pub fn terms(&self) -> impl Iterator<Item = (&LinearVar, &f64)> {
        self.coeffs.iter()
    }

    pub(super) fn normalize(mut self) -> Self {
        self.coeffs.retain(|_, v| v.abs() > COEFF_EPS);
        if self.constant.abs() <= COEFF_EPS {
            self.constant = 0.0;
        }
        self
    }

    pub(super) fn add(mut self, other: &LinearExpr) -> Self {
        for (var, coeff) in &other.coeffs {
            let entry = self.coeffs.entry(*var).or_insert(0.0);
            *entry += coeff;
        }
        self.constant += other.constant;
        self.normalize()
    }

    pub(super) fn sub(mut self, other: &LinearExpr) -> Self {
        for (var, coeff) in &other.coeffs {
            let entry = self.coeffs.entry(*var).or_insert(0.0);
            *entry -= coeff;
        }
        self.constant -= other.constant;
        self.normalize()
    }

    pub(super) fn scale(mut self, factor: f64) -> Self {
        if factor.abs() <= COEFF_EPS {
            return Self::constant(0.0);
        }
        for coeff in self.coeffs.values_mut() {
            *coeff *= factor;
        }
        self.constant *= factor;
        self.normalize()
    }

    pub(super) fn as_constant(&self) -> Option<f64> {
        if self.coeffs.is_empty() {
            Some(self.constant)
        } else {
            None
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Relation {
    LessEq,
    GreaterEq,
    Equal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearConstraint {
    pub expr: LinearExpr,
    pub relation: Relation,
    pub is_strict: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoolExpr<T> {
    Atom(T),
    And(Vec<BoolExpr<T>>),
    Or(Vec<BoolExpr<T>>),
}

#[derive(Clone, Debug)]
pub struct NormalizeOptions {
    pub max_clauses: usize,
    pub max_clause_len: usize,
}

impl Default for NormalizeOptions {
    fn default() -> Self {
        Self {
            max_clauses: 200_000,
            max_clause_len: 4096,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedConstraints {
    pub clauses: Vec<Vec<LinearConstraint>>,
    /// Per-clause input bounds extracted from mixed input+output `and` clauses.
    /// Outer vec: one per clause (parallel to `clauses`).
    /// Inner: input variable index → (lower, upper) bound for that clause.
    /// Empty map means no per-clause input bounds (use global bounds).
    pub per_clause_input_bounds: Vec<BTreeMap<usize, (f64, f64)>>,
}

impl NormalizedConstraints {
    #[cfg(test)]
    pub fn is_disjunction(&self) -> bool {
        self.clauses.len() > 1
    }
}
