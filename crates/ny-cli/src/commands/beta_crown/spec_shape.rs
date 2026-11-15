// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-precision — THE extractor of a property's routing shape
//! ([`ny_core::router::spec::SpecShape`]), and the router's start-time read of
//! it.
//!
//! [`shape_of`] is the ONE implementation both sides call:
//!
//! * the lanes' route checks (`dispatch.rs` `routes_to_relational_verifier`,
//!   `verify/mod.rs` `verify_relational_constraints_with_ledger` and
//!   `should_upgrade_sequential_conjunction_to_graph`, `verify/graph.rs`
//!   `verify_graph_relational`, `model_load.rs` `load_model`) call
//!   `ny_core::router::admission` on `shape_of(<the spec they hold>)`;
//! * the router reads the SAME shape at start ([`read_spec`]) from the
//!   property the scored entry point was handed, parsed by the SAME parser
//!   the lanes load it with (`ny_onnx::vnnlib::load_vnnlib`), and composes the
//!   same checks into each routable consumer's site.
//!
//! Every field reads the parsed specification through the lanes' own
//! accessors: `VnnLibSpec::has_multi_constraint_disjunction` /
//! `has_boxed_clause_disjunction`, the constraint planner's
//! `classify_constraints`, the sequential lane's same-LHS reduction, the
//! perturbed-input count the VGG policy reads. Nothing here reads a name.

use std::path::Path;

use ny_core::router::decl::SpecArchetype;
use ny_core::router::spec::SpecShape;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};

/// A property above this size is not read at router start (its shape is
/// `unknown`, which defers every specification check to the consumer's own
/// call site): the read is a second full parse inside the timed run.
pub(crate) const SPEC_READ_MAX_BYTES: u64 = 16 << 20;

/// The smaller side of an atom comparing two outputs (`Y_i >= Y_j` and
/// `Y_j <= Y_i` both name `j`), `None` for a threshold atom.
const fn smaller_output(atom: &OutputConstraint) -> Option<usize> {
    match atom {
        OutputConstraint::GreaterEq(_, small) | OutputConstraint::GreaterThan(_, small) => {
            Some(*small)
        }
        OutputConstraint::LessEq(small, _) | OutputConstraint::LessThan(small, _) => Some(*small),
        _ => None,
    }
}

/// Every clause is exactly one atom comparing two outputs, and all of them
/// share their smaller side (the seal pipeline's offline robustness rule).
fn argmax_margin(vnnlib: &VnnLibSpec) -> bool {
    if vnnlib.output_constraint_clauses.is_empty() {
        // Ungrouped atoms are ONE conjunctive clause.
        return matches!(vnnlib.output_constraints.as_slice(), [atom] if smaller_output(atom).is_some());
    }
    let mut common: Option<usize> = None;
    for clause in &vnnlib.output_constraint_clauses {
        let [atom] = clause.as_slice() else {
            return false;
        };
        let Some(small) = smaller_output(atom) else {
            return false;
        };
        match common {
            None => common = Some(small),
            Some(seen) if seen == small => {}
            Some(_) => return false,
        }
    }
    common.is_some()
}

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// THE routing shape of a parsed property (see the module docs).
#[must_use]
pub(crate) fn shape_of(vnnlib: &VnnLibSpec) -> SpecShape {
    let atoms = count(vnnlib.output_constraints.len());
    SpecShape {
        atoms,
        clauses: count(vnnlib.output_constraint_clauses.len()),
        max_clause_atoms: vnnlib
            .output_constraint_clauses
            .iter()
            .map(|clause| count(clause.len()))
            .max()
            .unwrap_or(atoms),
        boxed_clauses: count(
            vnnlib
                .per_clause_input_bounds
                .iter()
                .filter(|bounds| !bounds.is_empty())
                .count(),
        ),
        disjunction: vnnlib.is_disjunction,
        relational_atoms: super::constraint_plan::classify_constraints(vnnlib).has_relational,
        argmax_margin: argmax_margin(vnnlib),
        multi_clause_disjunction: vnnlib.has_multi_constraint_disjunction(),
        boxed_clause_disjunction: vnnlib.has_boxed_clause_disjunction(),
        same_lhs_reducible: super::verify::same_lhs_reducible(vnnlib),
        inputs: count(vnnlib.num_inputs),
        perturbed_inputs: count(super::count_perturbed_inputs(vnnlib)),
        outputs: count(vnnlib.num_outputs),
    }
}

/// D45 (`vnncomp.rs`, the non-linear lanes' entry): the affine parser refused
/// the property because it cannot REPRESENT it (non-linear arithmetic, strict
/// input constraints, an input/output mix, a list expression) — not because
/// the file is broken. The router's `nonlinear` archetype is exactly this
/// refusal; the entry point calls the same function.
#[must_use]
pub(crate) fn affine_parser_refused_for_representability(msg: &str) -> bool {
    msg.contains("Non-linear arithmetic")
        || msg.contains("Strict input constraints are unsupported")
        || msg.contains("Constraint mixes input and output variables")
        || msg.contains("Unsupported list expression")
}

/// What the router reads about the property at start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecRead {
    /// An affine single-network property and its routing shape.
    Affine(SpecShape),
    /// The affine parser refused it for representability (D45).
    Nonlinear,
    /// It declares two or more networks.
    Relational,
    /// Not read, or not classifiable (the reason).
    Unknown(&'static str),
}

impl SpecRead {
    /// The archetype the router records (`None` when unknown).
    #[must_use]
    pub(crate) const fn archetype(&self) -> Option<SpecArchetype> {
        match self {
            Self::Affine(shape) => Some(shape.archetype()),
            Self::Nonlinear => Some(SpecArchetype::Nonlinear),
            Self::Relational => Some(SpecArchetype::Relational),
            Self::Unknown(_) => None,
        }
    }

    /// The affine shape, when there is one.
    #[must_use]
    pub(crate) const fn shape(&self) -> Option<SpecShape> {
        match self {
            Self::Affine(shape) => Some(*shape),
            _ => None,
        }
    }

    /// One TSV-safe token: `<archetype> c=<clauses> a=<atoms> bx=<boxed>
    /// <flags>`, `nonlinear`, `relational` or `unknown:<why>`.
    #[must_use]
    pub(crate) fn token(&self) -> String {
        match self {
            Self::Affine(shape) => format!(
                "{} c={} a={} bx={} {}",
                shape.archetype().token(),
                shape.clauses,
                shape.atoms,
                shape.boxed_clauses,
                shape.flag_tokens()
            ),
            Self::Nonlinear => "nonlinear".to_owned(),
            Self::Relational => "relational".to_owned(),
            Self::Unknown(why) => format!("unknown:{why}"),
        }
    }
}

/// The router's start-time read of the property at `path`: its size, then
/// the lanes' parser, then the shape. Never panics outward (a parser panic is
/// `unknown:spec-parse-panic`); deterministic for a given file content.
#[must_use]
pub(crate) fn read_spec(path: &Path) -> SpecRead {
    match std::fs::metadata(path) {
        Err(_) => return SpecRead::Unknown("spec-unreadable"),
        Ok(meta) if meta.len() > SPEC_READ_MAX_BYTES => return SpecRead::Unknown("spec>16MiB"),
        Ok(_) => {}
    }
    let parsed = std::panic::catch_unwind(|| ny_onnx::vnnlib::load_vnnlib(path));
    match parsed {
        Err(_) => SpecRead::Unknown("spec-parse-panic"),
        Ok(Ok(spec)) if spec.dual_network.is_some() => SpecRead::Relational,
        Ok(Ok(spec)) => SpecRead::Affine(shape_of(&spec)),
        Ok(Err(error)) if affine_parser_refused_for_representability(&error.to_string()) => {
            SpecRead::Nonlinear
        }
        Ok(Err(_)) => SpecRead::Unknown("spec-unparsed"),
    }
}
