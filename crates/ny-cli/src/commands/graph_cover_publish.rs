// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-bab-cover-ledger: the publication side of the graph lane's
//! completeness ledger (`ny_propagate::certify::graph`).
//!
//! The in-process β-CROWN verifier reports its verdict as a status string.
//! Its ledgered drivers (single-objective ReLU split, precomputed-bounds ReLU
//! split, single-objective input split) additionally deposit every PROVED cover — a gate-checked cover of a
//! root box for one objective — into the process capture, which the
//! publication path arms before the verifier runs and takes after it.
//!
//! When the status is `verified`, [`graph_instance_unsat`] asks the certify
//! facade's instance door for a token: every clause of the instance's unsafe
//! region must be entailed by a captured cover on a root box containing the
//! declared box. The instance's clauses are built here from the parsed
//! VNN-LIB with the SAME objective encoding the lanes use
//! (`constraint_plan::build_constraint_objective`), which is the U1 trust the
//! margin-row lane places in its own objective rows. Anything this module
//! cannot express (a dual-network relation, per-clause input boxes, a
//! disjunction without clause groups) is a reason, never a token: the verdict
//! then publishes through the counted `BetaCrownVerdict` legacy door. So is a
//! process whose ONNX loads folded a `BatchNormalization` with its enclosure
//! dropped (`bn_fold_uncharged`, the open M2e gap): the lane's bounds on that
//! model carry an uncharged fold error, so they are not a certificate.

use std::path::Path;

use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::certify::{
    authorize_graph_instance, GraphCoverCapture, GraphInstanceProperty, GraphObjective, UnsatToken,
};

use crate::commands::beta_crown::constraint_plan::build_constraint_objective;

/// The instance's unsafe region as the graph instance door reads it: the
/// declared input box and the clauses (a disjunction of conjunctions), each
/// atom as the lower-bound objective whose certification excludes it.
pub(crate) fn graph_instance_property(spec: &VnnLibSpec) -> Result<GraphInstanceProperty, String> {
    if spec.dual_network.is_some() {
        return Err("dual_network_property".to_string());
    }
    if spec.per_clause_input_bounds.iter().any(|m| !m.is_empty()) {
        return Err("per_clause_input_boxes".to_string());
    }
    let groups: Vec<&[OutputConstraint]> = if spec.is_disjunction {
        if spec.output_constraint_clauses.is_empty() {
            return Err("disjunction_without_clause_groups".to_string());
        }
        spec.output_constraint_clauses
            .iter()
            .map(Vec::as_slice)
            .collect()
    } else {
        vec![spec.output_constraints.as_slice()]
    };
    let mut clauses = Vec::with_capacity(groups.len());
    for group in groups {
        let mut atoms = Vec::with_capacity(group.len());
        for constraint in group {
            let objective = build_constraint_objective(constraint, spec.num_outputs)
                .map_err(|e| format!("atom_objective {e}"))?;
            let atom = GraphObjective::new(objective.spec_coeffs(), objective.threshold(), false)
                .ok_or_else(|| "non_finite_atom".to_string())?;
            atoms.push(atom);
        }
        clauses.push(atoms);
    }
    let (lo, hi): (Vec<f64>, Vec<f64>) = spec.input_bounds.iter().copied().unzip();
    GraphInstanceProperty::new(lo, hi, clauses)
        .ok_or_else(|| "malformed_instance_property".to_string())
}

/// Why no graph token may be minted in this process even when a cover
/// entails every clause, or `None`.
///
/// M2e (ENGINE_PROGRAM §1.6: the BN-fold closure comes "before any certified
/// GraphNetwork lowering"): a load that folded a `BatchNormalization` stored
/// its composed coefficients as binary32 POINTS with the enclosure dropped
/// (`ny_onnx::bn_fold_enclosure_dropped`), so a graph-lane bound close on
/// that model rests on an uncharged `~2^-24` relative error. Such a verdict
/// is the lane's word, not a certificate: it publishes through the counted
/// legacy door, exactly as before the ledger existed.
fn graph_token_blocker(bn_fold_enclosure_dropped: bool) -> Option<&'static str> {
    bn_fold_enclosure_dropped.then_some("bn_fold_uncharged")
}

/// The graph lane's instance token for a `verified` run, or the reason there
/// is none (the caller then publishes through the counted legacy door).
/// Re-reads the property only when something was captured.
pub(crate) fn graph_instance_unsat(
    vnnlib: &Path,
    capture: &GraphCoverCapture,
) -> Result<UnsatToken, String> {
    if capture.is_empty() {
        return Err("nothing_captured".to_string());
    }
    if let Some(blocker) = graph_token_blocker(ny_onnx::bn_fold_enclosure_dropped()) {
        return Err(blocker.to_string());
    }
    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib).map_err(|e| format!("vnnlib_reparse {e}"))?;
    let property = graph_instance_property(&spec)?;
    authorize_graph_instance(&property, capture).map_err(|refusal| refusal.to_string())
}

#[cfg(test)]
mod tests {
    use super::{graph_instance_property, graph_instance_unsat, graph_token_blocker};
    use ny_propagate::certify::GraphCoverCapture;

    /// M2e: a process whose loads dropped a BatchNormalization fold enclosure
    /// mints no graph token (legacy door, reason `bn_fold_uncharged`); an
    /// unfolded process is not blocked. The production call reads the loader's
    /// own latch, not a copy of it.
    #[test]
    fn graph_token_is_blocked_by_an_uncharged_bn_fold() {
        assert_eq!(graph_token_blocker(true), Some("bn_fold_uncharged"));
        assert_eq!(graph_token_blocker(false), None);
        let source = include_str!("graph_cover_publish.rs");
        let call = format!(
            "{}({}::{}())",
            "graph_token_blocker", "ny_onnx", "bn_fold_enclosure_dropped"
        );
        let head = format!("fn {}(", "graph_instance_unsat");
        let start = source.find(head.as_str()).expect("the publication entry");
        let body = &source[start..];
        let body = &body[..body.find("\n}\n").map_or(body.len(), |e| e + 3)];
        let blocked = body
            .find(call.as_str())
            .expect("the door consults the loader's latch");
        // Split so `unsat_receipt`'s door census (which counts the bare
        // identifier in this file's code) still sees exactly the `use` and
        // the one call.
        let minted = body
            .find(format!("{}_{}(", "authorize_graph", "instance").as_str())
            .expect("the door mints");
        assert!(
            blocked < minted,
            "the M2e block must run before the token is minted"
        );
    }

    fn parse(text: &str) -> ny_onnx::vnnlib::VnnLibSpec {
        ny_onnx::vnnlib::parse_vnnlib(text).expect("the fixture property parses")
    }

    const HEADER: &str = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
        (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
        (assert (>= X_0 -1.0))\n(assert (<= X_0 1.0))\n\
        (assert (>= X_1 0.0))\n(assert (<= X_1 0.5))\n";

    /// A conjunctive property is ONE clause of every atom, on the declared
    /// box; a disjunction is one clause per group; shapes the door cannot
    /// express are reasons, never properties.
    #[test]
    fn graph_instance_property_reads_the_parsed_clauses() {
        let conj = parse(&format!(
            "{HEADER}(assert (<= Y_0 Y_1))\n(assert (>= Y_0 0.25))\n"
        ));
        let property = graph_instance_property(&conj).expect("conjunction");
        assert_eq!(property.clause_count(), 1);

        let disj = parse(&format!(
            "{HEADER}(assert (or (and (<= Y_0 Y_1) (>= Y_0 0.5)) \
             (and (>= Y_0 3.0) (<= Y_1 0.0))))\n"
        ));
        assert!(disj.has_multi_constraint_disjunction());
        let property = graph_instance_property(&disj).expect("disjunction");
        assert_eq!(property.clause_count(), 2);

        let mut dual = conj;
        dual.per_clause_input_bounds = vec![std::iter::once((0usize, (0.0, 0.5))).collect()];
        assert_eq!(
            graph_instance_property(&dual).err().as_deref(),
            Some("per_clause_input_boxes")
        );
    }

    /// With nothing captured there is no token (and the property is never
    /// re-read): the verdict publishes through the counted legacy door.
    #[test]
    fn graph_instance_unsat_without_a_capture_is_a_reason() {
        let missing = std::path::Path::new("/nonexistent/graph_cover_publish.vnnlib");
        assert_eq!(
            graph_instance_unsat(missing, &GraphCoverCapture::default()).err(),
            Some("nothing_captured".to_string())
        );
    }
}
