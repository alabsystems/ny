// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! History serialization/deserialization between GraphSplitHistory and ConstraintTuple format.

use crate::batched_domain::ConstraintTuple;
use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint, GraphSplitHistory};
use ny_core::{NyError, Result};

/// Reconstruct a GraphSplitHistory from constraint tuples.
///
/// DomainMetadata stores constraints as `(node_name, neuron_idx, is_active, split_point)` tuples:
/// - ReLU constraints: `split_point = None`, `is_active = true` (x >= 0) or `false` (x <= 0)
/// - GenBaB constraints: `split_point = Some(pt)`, `is_active = true` (x >= pt) or `false` (x <= pt)
///
/// This function converts them back to a full GraphSplitHistory with constraint objects,
/// preserving the original split order.
pub fn history_from_constraints(constraints: &[ConstraintTuple]) -> Result<GraphSplitHistory> {
    let mut history = GraphSplitHistory::new();
    for (node_name, neuron_idx, is_active, split_point) in constraints {
        match split_point {
            None => {
                // ReLU constraint
                history.add_constraint(GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx: *neuron_idx,
                    is_active: *is_active,
                    score: 0.0,
                });
            }
            Some(pt) => {
                // GenBaB constraint — validate split_point via constructor (#3011)
                let constraint =
                    GenBabConstraint::new(node_name.clone(), *neuron_idx, *pt, *is_active, 0.0)?;
                history.add_genbab_constraint(constraint);
            }
        }
    }
    Ok(history)
}

/// Serialize constraints from a GraphSplitHistory to ConstraintTuple format.
///
/// Interleaves ReLU and GenBaB constraints by split order, matching the format
/// expected by `DomainMetadata::constraints`.
///
/// This is the inverse of `history_from_constraints`.
///
/// # Errors
/// Returns `NyError::InvalidSpec` if `split_count` under-consumes constraints,
/// which would silently drop trailing entries and widen domains (#2248, #2313).
///
/// # Reference
/// Same algorithm and guard as `batched_domain::types::serialize_constraints`.
pub(crate) fn constraints_from_history(
    history: &GraphSplitHistory,
) -> Result<Vec<ConstraintTuple>> {
    let mut result =
        Vec::with_capacity(history.constraints.len() + history.genbab_constraints.len());
    let mut relu_idx = 0;
    let mut genbab_idx = 0;

    for split_id in 0..history.split_count {
        let next_genbab_split = history.genbab_split_ids.get(genbab_idx).copied();

        if next_genbab_split == Some(split_id) {
            // Emit all GenBaB constraints with this split_id
            while history.genbab_split_ids.get(genbab_idx) == Some(&split_id) {
                let c = &history.genbab_constraints[genbab_idx];
                // SOUNDNESS (#mul-genbab): the 4-tuple ConstraintTuple cannot carry
                // `input_index`. A second-input MulBinary/BilinearCrown split
                // (input_index == Some(1)) would be reconstructed as input 0 and
                // misrouted to the wrong input node — a hard index error or a
                // silent clamp of the wrong neuron (excluding reachable values).
                // Refuse to serialize it lossily; the caller treats this as an
                // unresolved domain (PropagationFailure), never a false Verified.
                // The CPU GenBaB path keeps the full GraphSplitHistory and is
                // unaffected.
                if c.input_index().unwrap_or(0) != 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "constraints_from_history: GenBaB constraint on '{}' targets \
                         input_index={:?}, which the ConstraintTuple format cannot \
                         represent — refusing lossy serialization (#mul-genbab)",
                        c.node_name, c.input_index,
                    )));
                }
                result.push((
                    c.node_name.clone(),
                    c.neuron_idx,
                    c.is_upper_branch,
                    Some(c.split_point),
                ));
                genbab_idx += 1;
            }
        } else {
            // This split_id is a ReLU constraint
            if relu_idx < history.constraints.len() {
                let c = &history.constraints[relu_idx];
                result.push((c.node_name.clone(), c.neuron_idx, c.is_active, None));
                relu_idx += 1;
            }
        }
    }

    // Soundness guard (#2248/#2313): reject histories where split_count under-consumes constraints.
    if relu_idx != history.constraints.len() || genbab_idx != history.genbab_constraints.len() {
        return Err(NyError::InvalidSpec(format!(
            "constraints_from_history: not all constraints consumed \
             (split_count={}, relu={}/{}, genbab={}/{})",
            history.split_count,
            relu_idx,
            history.constraints.len(),
            genbab_idx,
            history.genbab_constraints.len(),
        )));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for #2313: constraints_from_history must return an error
    /// when split_count is too low and ReLU constraints would be silently dropped.
    #[test]
    fn rejects_wrong_split_count_relu_2313() {
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu0".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        });
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu0".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 0.0,
        });
        assert_eq!(history.split_count, 2);

        // Corrupt split_count to under-consume
        history.split_count = 1;

        let err = constraints_from_history(&history)
            .expect_err("mismatched split_count should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("not all constraints consumed"),
            "unexpected error: {msg}"
        );
    }

    /// Regression test for #2313: constraints_from_history must return an error
    /// when split_count is too low and GenBaB constraints would be silently dropped.
    #[test]
    fn rejects_wrong_split_count_genbab_2313() {
        let mut history = GraphSplitHistory::new();
        history.add_genbab_constraint(
            GenBabConstraint::new("node0".to_string(), 0, 0.5, true, 0.0)
                .expect("finite test constraint"),
        );
        history.add_genbab_constraint(
            GenBabConstraint::new("node0".to_string(), 1, -0.3, false, 0.0)
                .expect("finite test constraint"),
        );
        assert_eq!(history.split_count, 2);

        // Corrupt split_count to under-consume
        history.split_count = 1;

        let err = constraints_from_history(&history)
            .expect_err("mismatched split_count should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("not all constraints consumed"),
            "unexpected error: {msg}"
        );
    }

    /// Soundness (#mul-genbab): the 4-tuple ConstraintTuple cannot represent a
    /// second-input MulBinary/BilinearCrown split. `constraints_from_history` must
    /// REFUSE to serialize it lossily rather than silently misroute it to input 0.
    #[test]
    fn rejects_genbab_input_index_serialization_mul_genbab() {
        let mut history = GraphSplitHistory::new();
        history.add_genbab_constraint(
            GenBabConstraint::new("mul0".to_string(), 0, 0.5, true, 0.0)
                .expect("finite test constraint")
                .with_input_index(1),
        );

        let err = constraints_from_history(&history)
            .expect_err("input_index=1 GenBaB constraint must not serialize lossily");
        let msg = err.to_string();
        assert!(
            msg.contains("input_index") && msg.contains("cannot represent"),
            "unexpected error: {msg}"
        );

        // input_index=0 (default for unary activations) serializes fine.
        let mut history0 = GraphSplitHistory::new();
        history0.add_genbab_constraint(
            GenBabConstraint::new("gelu0".to_string(), 0, 0.5, true, 0.0)
                .expect("finite test constraint"),
        );
        let tuples = constraints_from_history(&history0)
            .expect("input_index=0 GenBaB constraint serializes");
        assert_eq!(tuples.len(), 1);
        assert_eq!(tuples[0].0, "gelu0");
    }

    /// Valid history round-trips correctly.
    #[test]
    fn roundtrip_valid_history() {
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu0".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        });
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 3,
            is_active: false,
            score: 0.5,
        });

        let tuples = constraints_from_history(&history).unwrap();
        assert_eq!(tuples.len(), 2);
        assert_eq!(tuples[0].0, "relu0");
        assert_eq!(tuples[0].1, 0);
        assert!(tuples[0].2);
        assert!(tuples[0].3.is_none());
        assert_eq!(tuples[1].0, "relu1");
        assert_eq!(tuples[1].1, 3);
        assert!(!tuples[1].2);

        // Round-trip back
        let restored = history_from_constraints(&tuples).unwrap();
        assert_eq!(restored.split_count, 2);
        assert_eq!(restored.constraints.len(), 2);
    }
}
