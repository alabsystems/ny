// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-precision — the specification's ROUTING SHAPE (ROUTER_PROGRAM
//! §2.2 FACTS). DATA ONLY, plus the pure archetype mapping.
//!
//! A [`SpecShape`] is what the lanes route a row on, read off the parsed
//! VNN-LIB property: how many output atoms and clauses it has, whether the
//! clauses are a top-level OR, whether clauses carry their own input boxes,
//! whether an atom compares two outputs, and the two predicates the parsed
//! specification type itself defines (`has_multi_constraint_disjunction`,
//! `has_boxed_clause_disjunction`). It is extracted ONCE, by ONE function
//! (`ny-cli`'s `commands::beta_crown::spec_shape::shape_of`, over the SAME
//! parser the lanes use), and that one extraction feeds both the router (an
//! [`super::facts::InstanceFacts`] field at router start) and the lanes'
//! own route checks ([`super::admission`]), so a router admission and the lane
//! it describes read the same bits.
//!
//! There is no name and no path here: a shape is a function of the
//! property's content only.

use serde::{Deserialize, Serialize};

use super::decl::SpecArchetype;

/// The routing shape of an affine (single-network, parsed) VNN-LIB property.
///
/// Every field is a count or a flag of the parsed specification; see the
/// extractor (`ny-cli` `spec_shape::shape_of`) for the exact source of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SpecShape {
    /// Output atoms (`VnnLibSpec::output_constraints.len()`).
    pub atoms: u64,
    /// Output clauses (`VnnLibSpec::output_constraint_clauses.len()`).
    pub clauses: u64,
    /// Atoms of the widest clause (`atoms` when the clauses are not grouped).
    pub max_clause_atoms: u64,
    /// Clauses that carry their own non-empty input box.
    pub boxed_clauses: u64,
    /// The clauses are a top-level OR (`VnnLibSpec::is_disjunction`).
    pub disjunction: bool,
    /// Some atom compares two outputs (`classify_constraints(..).has_relational`).
    pub relational_atoms: bool,
    /// Every clause is exactly one atom comparing two outputs, and every such
    /// atom shares its SMALLER side (the true class of an argmax-margin
    /// robustness property: `Y_j >= Y_t` for each `j`).
    pub argmax_margin: bool,
    /// `VnnLibSpec::has_multi_constraint_disjunction()`.
    pub multi_clause_disjunction: bool,
    /// `VnnLibSpec::has_boxed_clause_disjunction()`.
    pub boxed_clause_disjunction: bool,
    /// The sequential lane's same-LHS relational reduction applies
    /// (`normalize_same_lhs_reduction(..).is_some()`).
    pub same_lhs_reducible: bool,
    /// Declared input variables (`VnnLibSpec::num_inputs`).
    pub inputs: u64,
    /// Inputs whose box is not a point (`upper > lower`).
    pub perturbed_inputs: u64,
    /// Declared output variables (`VnnLibSpec::num_outputs`).
    pub outputs: u64,
}

impl SpecShape {
    /// The clause count as the lanes enumerate clauses: the grouped clauses
    /// when the parser grouped them, else one clause of every atom (none when
    /// there is no atom).
    #[must_use]
    pub const fn effective_clauses(&self) -> u64 {
        if self.clauses > 0 {
            self.clauses
        } else if self.atoms > 0 {
            1
        } else {
            0
        }
    }

    /// The specification archetype of this affine shape. The same rule as the
    /// seal pipeline's offline classifier (`scripts/seal/seallib.py`
    /// `vnnlib_spec`), so the router's archetype and the envelope the seal
    /// records speak one vocabulary: argmax-margin robustness first, then a
    /// single clause (a conjunction of output half-spaces), else a
    /// disjunction. Relational and nonlinear properties have no affine shape.
    #[must_use]
    pub const fn archetype(&self) -> SpecArchetype {
        if self.argmax_margin {
            SpecArchetype::Robustness
        } else if self.effective_clauses() <= 1 {
            SpecArchetype::Halfspace
        } else {
            SpecArchetype::Disjunctive
        }
    }

    /// `+`-joined flag tokens (`-` when none), for logs and the census.
    #[must_use]
    pub fn flag_tokens(&self) -> String {
        let flags: Vec<&str> = [
            (self.disjunction, "or"),
            (self.multi_clause_disjunction, "multi"),
            (self.boxed_clause_disjunction, "boxed"),
            (self.relational_atoms, "rel"),
            (self.argmax_margin, "argmax"),
            (self.same_lhs_reducible, "same_lhs"),
        ]
        .into_iter()
        .filter_map(|(on, token)| on.then_some(token))
        .collect();
        if flags.is_empty() {
            "-".to_owned()
        } else {
            flags.join("+")
        }
    }
}
