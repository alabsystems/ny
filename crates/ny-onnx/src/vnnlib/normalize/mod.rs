// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNNLib constraint normalization: parsing, linearization, and DNF/CNF conversion.

mod convert;
mod parse;
mod types;

pub use convert::{normalize_output_constraints, to_output_constraint_clauses};
pub use types::{NormalizeOptions, Relation, VarKind};

#[cfg(test)]
pub use convert::{to_cnf, to_dnf, to_output_constraints};
#[cfg(test)]
pub use types::BoolExpr;

pub(super) use parse::parse_linear_constraint;
