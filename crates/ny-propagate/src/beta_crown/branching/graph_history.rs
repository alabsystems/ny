// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::graph_constraints::{
    GenBabConstraint, GraphConstraint, GraphNeuronConstraint, NormInvRmsConstraint,
};
use ny_core::{nan_propagating_max, nan_propagating_min};
use std::collections::HashMap;
use std::mem::size_of;

/// Hard cap for the collision-free split-history identity retained by a
/// verdict-adjacent provenance token.  Refusing an unexpectedly large identity
/// only disables a tightening proposal; the inherited domain remains sound.
const PROVENANCE_HISTORY_MAX_BYTES: usize = 16 * 1024 * 1024;

/// History of split decisions for a GraphNetwork domain.
///
/// Supports both ReLU constraints (binary split at 0) and GenBaB constraints
/// (arbitrary split points for general nonlinearities).
#[derive(Debug, Clone, Default)]
pub struct GraphSplitHistory {
    /// ReLU constraints applied in this domain (split at 0).
    pub constraints: Vec<GraphNeuronConstraint>,
    /// GenBaB constraints applied in this domain (arbitrary split points).
    pub genbab_constraints: Vec<GenBabConstraint>,
    /// Split decision id for each GenBaB constraint (same length as genbab_constraints).
    pub genbab_split_ids: Vec<usize>,
    /// Number of split decisions applied (may differ from constraint count for range splits).
    pub split_count: usize,
    /// O(1) lookup cache for ReLU constraints: node_name -> (neuron_idx -> is_active).
    /// Two-level map enables zero-allocation lookup via `&str` key (no String clone needed).
    relu_lookup: HashMap<String, HashMap<usize, bool>>,
    /// O(1) lookup cache for GenBaB constraints: node_name -> (neuron_idx -> (lower, upper)).
    /// Mirrors `relu_lookup` structure. Lower bound comes from `is_upper_branch=true` (the neuron
    /// is above split_point), upper bound from `is_upper_branch=false` (below split_point).
    /// Replaces the O(D) linear scan in `is_genbab_constrained` (#4315).
    genbab_lookup: HashMap<String, HashMap<usize, (Option<f32>, Option<f32>)>>,
    /// GenBaB norm-branching `inv_rms` clamps (#norm-genbab). One entry per
    /// applied norm split; stacked splits on the same node intersect (see
    /// `norm_inv_rms_overrides`).
    norm_inv_rms_constraints: Vec<NormInvRmsConstraint>,
}

impl GraphSplitHistory {
    /// Create empty split history.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            genbab_constraints: Vec::new(),
            genbab_split_ids: Vec::new(),
            split_count: 0,
            relu_lookup: HashMap::new(),
            genbab_lookup: HashMap::new(),
            norm_inv_rms_constraints: Vec::new(),
        }
    }

    /// Add a GenBaB norm `inv_rms` constraint (#norm-genbab).
    pub fn add_norm_inv_rms_constraint(&mut self, constraint: NormInvRmsConstraint) {
        self.norm_inv_rms_constraints.push(constraint);
        self.split_count += 1;
    }

    /// Create a new history with an additional norm `inv_rms` constraint.
    pub fn with_norm_inv_rms_constraint(&self, constraint: NormInvRmsConstraint) -> Self {
        let mut new = self.clone();
        new.add_norm_inv_rms_constraint(constraint);
        new
    }

    /// Whether the history carries any norm `inv_rms` constraint.
    pub fn has_norm_inv_rms_constraints(&self) -> bool {
        !self.norm_inv_rms_constraints.is_empty()
    }

    /// Accumulate all norm `inv_rms` constraints into per-(node, group) windows
    /// by INTERSECTION (#norm-genbab).
    ///
    /// Each node maps to a vector of per-group windows (indexed by normalization
    /// group / batch row); `None` leaves a group unconstrained (full IBP range).
    /// Stacked splits on the same (node, group) — a BaB path that branched the
    /// same group twice — each narrow the window; the effective window is their
    /// intersection (`max` of los, `min` of his). Returns `None` when there are
    /// no norm constraints, so the common (unbranched) path pays no allocation.
    ///
    /// SOUNDNESS: intersection only narrows. A group's effective window is the
    /// conjunction of every clamp on the BaB path to this domain, the `inv_rms`
    /// sub-range this leaf is responsible for FOR THAT GROUP. Splitting one
    /// group at a time avoids the join gap a shared window would create (see
    /// `InvRmsOverride`). The decomposed RmsNorm backward further intersects each
    /// window with the group's own IBP `inv_rms` interval, so the certified
    /// range can never widen.
    pub fn norm_inv_rms_overrides(&self) -> Option<HashMap<String, Vec<Option<(f32, f32)>>>> {
        if self.norm_inv_rms_constraints.is_empty() {
            return None;
        }
        let mut map: HashMap<String, Vec<Option<(f32, f32)>>> = HashMap::new();
        for c in &self.norm_inv_rms_constraints {
            let windows = map.entry(c.node_name.clone()).or_default();
            if windows.len() <= c.group_index {
                windows.resize(c.group_index + 1, None);
            }
            let slot = &mut windows[c.group_index];
            *slot = Some(match *slot {
                Some((lo, hi)) => (
                    nan_propagating_max(lo, c.inv_rms_lo),
                    nan_propagating_min(hi, c.inv_rms_hi),
                ),
                None => (c.inv_rms_lo, c.inv_rms_hi),
            });
        }
        Some(map)
    }

    /// Add a ReLU constraint to the history.
    pub fn add_constraint(&mut self, constraint: GraphNeuronConstraint) {
        self.relu_lookup
            .entry(constraint.node_name.clone())
            .or_default()
            .insert(constraint.neuron_idx, constraint.is_active);
        self.constraints.push(constraint);
        self.split_count += 1;
    }

    /// Add a GenBaB constraint to the history.
    pub fn add_genbab_constraint(&mut self, constraint: GenBabConstraint) {
        Self::update_genbab_lookup(&mut self.genbab_lookup, &constraint);
        self.genbab_constraints.push(constraint);
        self.genbab_split_ids.push(self.split_count);
        self.split_count += 1;
    }

    /// Add multiple GenBaB constraints that correspond to a single split decision.
    ///
    /// Used for range splits that apply both a lower and upper bound to the same neuron.
    pub fn add_genbab_constraints_for_split<I>(&mut self, constraints: I)
    where
        I: IntoIterator<Item = GenBabConstraint>,
    {
        let mut iter = constraints.into_iter();
        let Some(first) = iter.next() else {
            return;
        };
        let split_id = self.split_count;
        self.split_count += 1;
        Self::update_genbab_lookup(&mut self.genbab_lookup, &first);
        self.genbab_constraints.push(first);
        self.genbab_split_ids.push(split_id);
        for constraint in iter {
            Self::update_genbab_lookup(&mut self.genbab_lookup, &constraint);
            self.genbab_constraints.push(constraint);
            self.genbab_split_ids.push(split_id);
        }
    }

    /// Add a generic constraint (either ReLU or GenBaB).
    pub fn add_generic_constraint(&mut self, constraint: GraphConstraint) {
        match constraint {
            GraphConstraint::Relu(c) => self.add_constraint(c),
            GraphConstraint::GenBab(c) => self.add_genbab_constraint(c),
        }
    }

    /// Get the depth (total number of splits).
    pub fn depth(&self) -> usize {
        self.split_count
    }

    /// Check if a neuron is already constrained (ReLU only).
    /// O(1) complexity via two-level HashMap lookup, zero allocation.
    pub fn is_constrained(&self, node_name: &str, neuron_idx: usize) -> Option<bool> {
        self.relu_lookup
            .get(node_name)
            .and_then(|neurons| neurons.get(&neuron_idx).copied())
    }

    /// Check if a neuron has GenBaB constraints.
    ///
    /// Returns (lower_bound, upper_bound) if constrained, where each bound may be None.
    /// O(1) complexity via two-level HashMap lookup, mirroring `relu_lookup` (#4315).
    pub fn is_genbab_constrained(
        &self,
        node_name: &str,
        neuron_idx: usize,
    ) -> Option<(Option<f32>, Option<f32>)> {
        self.genbab_lookup
            .get(node_name)
            .and_then(|neurons| neurons.get(&neuron_idx).copied())
    }

    /// Return the set of node names that carry at least one constraint
    /// (ReLU or GenBaB) in this history.
    ///
    /// Used as the seed set for downstream-reachability when deciding which
    /// intermediate node bounds may be reused verbatim from a parent domain:
    /// only nodes downstream of a constrained node can have bounds that differ
    /// from the unconstrained forward pass. See
    /// `GraphNetwork::descendants_inclusive` and
    /// `compute_constrained_forward_bounds`.
    pub fn constrained_node_names(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        names.extend(self.relu_lookup.keys().cloned());
        names.extend(self.genbab_lookup.keys().cloned());
        // Norm inv_rms splits change the RmsNorm node's CROWN relaxation (and
        // hence everything downstream), so its node must be re-propagated.
        names.extend(
            self.norm_inv_rms_constraints
                .iter()
                .map(|c| c.node_name.clone()),
        );
        names
    }

    /// Update the genbab_lookup cache for a single constraint.
    ///
    /// `is_upper_branch=true` means x >= split_point, setting the lower bound.
    /// `is_upper_branch=false` means x <= split_point, setting the upper bound.
    fn update_genbab_lookup(
        lookup: &mut HashMap<String, HashMap<usize, (Option<f32>, Option<f32>)>>,
        constraint: &GenBabConstraint,
    ) {
        let entry = lookup
            .entry(constraint.node_name.clone())
            .or_default()
            .entry(constraint.neuron_idx)
            .or_insert((None, None));
        if constraint.is_upper_branch {
            entry.0 = Some(constraint.split_point);
        } else {
            entry.1 = Some(constraint.split_point);
        }
    }

    /// Check if a neuron is constrained by either ReLU or GenBaB constraints.
    ///
    /// Returns `true` if the neuron has any constraint (ReLU binary or GenBaB split-point).
    /// Use this instead of `is_constrained()` when both constraint types should be considered,
    /// e.g. when determining whether a neuron should be skipped for branching. (#2399)
    pub fn is_any_constrained(&self, node_name: &str, neuron_idx: usize) -> bool {
        if self.is_constrained(node_name, neuron_idx).is_some() {
            return true;
        }
        self.is_genbab_constrained(node_name, neuron_idx).is_some()
    }

    /// Create a new history with an additional ReLU constraint.
    pub fn with_constraint(&self, constraint: GraphNeuronConstraint) -> Self {
        let mut new = self.clone();
        new.add_constraint(constraint);
        new
    }

    /// Create a new history with an additional GenBaB constraint.
    pub fn with_genbab_constraint(&self, constraint: GenBabConstraint) -> Self {
        let mut new = self.clone();
        new.add_genbab_constraint(constraint);
        new
    }

    /// Create a new history with multiple GenBaB constraints that represent a single split.
    pub fn with_genbab_constraints_for_split<I>(&self, constraints: I) -> Self
    where
        I: IntoIterator<Item = GenBabConstraint>,
    {
        let mut new = self.clone();
        new.add_genbab_constraints_for_split(constraints);
        new
    }

    /// Iterate over all constraints as generic `GraphConstraint` entries.
    ///
    /// Returns ReLU constraints first, then GenBaB constraints.
    pub fn iter_all(&self) -> impl Iterator<Item = GraphConstraint> + '_ {
        self.constraints
            .iter()
            .map(|c| GraphConstraint::Relu(c.clone()))
            .chain(
                self.genbab_constraints
                    .iter()
                    .map(|c| GraphConstraint::GenBab(c.clone())),
            )
    }

    /// Check if the history has any GenBaB constraints.
    pub fn has_genbab_constraints(&self) -> bool {
        !self.genbab_constraints.is_empty()
    }

    /// True iff EVERY split on the path to this domain is a pure ReLU-at-0
    /// literal (`z >= 0` / `z <= 0` on one neuron), i.e. the history contains
    /// no GenBaB arbitrary-split-point constraint and no norm `inv_rms` clamp.
    ///
    /// This is the PURITY GUARD for graph-engine conflict-clause learning
    /// (`conflict_clauses_graph`): the region-inclusion subsumption argument
    /// only holds when the domain's region is exactly `root_box ∩` the
    /// half-spaces named by its literal set. GenBaB and norm constraints carry
    /// private split points / windows that the (node, neuron, phase) literal
    /// vocabulary cannot express, so any such entry must disable both record
    /// and prune for the domain (fail closed).
    ///
    /// O(1): uses the maintained per-kind vectors plus `split_count`. The
    /// `split_count == constraints.len()` equality is a fail-closed catch-all —
    /// every split kind increments `split_count`, but only ReLU splits push to
    /// `constraints`, so a history touched by ANY split kind this method does
    /// not know about (including future ones) fails the equality and is
    /// treated as impure.
    pub fn is_pure_relu_at_zero(&self) -> bool {
        self.genbab_constraints.is_empty()
            && self.norm_inv_rms_constraints.is_empty()
            && self.split_count == self.constraints.len()
    }

    /// Return a collision-free, bit-exact identity of the semantic split path.
    ///
    /// This is deliberately an owned byte encoding rather than a `u64` hash: a
    /// hash collision must never let an affine enclosure produced for one BaB
    /// domain authorize clipping in another.  Lookup maps are derived caches and
    /// are omitted; every source field that defines the domain is encoded in
    /// order, including advisory scores, GenBaB split ids/input indices, and norm
    /// windows.  `None` is a fail-closed resource refusal.
    pub(crate) fn exact_provenance_identity(&self) -> Option<Vec<u8>> {
        fn add(total: &mut usize, amount: usize) -> Option<()> {
            *total = total.checked_add(amount)?;
            (*total <= PROVENANCE_HISTORY_MAX_BYTES).then_some(())
        }

        fn add_name(total: &mut usize, name: &str) -> Option<()> {
            add(total, size_of::<u64>())?;
            add(total, name.len())
        }

        fn push_u64(out: &mut Vec<u8>, value: usize) -> Option<()> {
            out.extend_from_slice(&u64::try_from(value).ok()?.to_le_bytes());
            Some(())
        }

        fn push_name(out: &mut Vec<u8>, name: &str) -> Option<()> {
            push_u64(out, name.len())?;
            out.extend_from_slice(name.as_bytes());
            Some(())
        }

        // Header/version plus vector lengths and split_count.
        let mut bytes = 8usize;
        add(&mut bytes, 5 * size_of::<u64>())?;
        for c in &self.constraints {
            add(&mut bytes, 1 + 1 + size_of::<u64>() + size_of::<u32>())?;
            add_name(&mut bytes, &c.node_name)?;
        }
        for c in &self.genbab_constraints {
            add(
                &mut bytes,
                1 + 1 + 1 + size_of::<u64>() * 2 + size_of::<u32>() * 2,
            )?;
            add_name(&mut bytes, &c.node_name)?;
        }
        add(
            &mut bytes,
            self.genbab_split_ids.len().checked_mul(size_of::<u64>())?,
        )?;
        for c in &self.norm_inv_rms_constraints {
            add(&mut bytes, 1 + size_of::<u64>() + size_of::<u32>() * 3)?;
            add_name(&mut bytes, &c.node_name)?;
        }

        let mut out = Vec::new();
        out.try_reserve_exact(bytes).ok()?;
        out.extend_from_slice(b"NYHIST01");
        push_u64(&mut out, self.split_count)?;
        push_u64(&mut out, self.constraints.len())?;
        push_u64(&mut out, self.genbab_constraints.len())?;
        push_u64(&mut out, self.genbab_split_ids.len())?;
        push_u64(&mut out, self.norm_inv_rms_constraints.len())?;

        for c in &self.constraints {
            out.push(1);
            push_name(&mut out, &c.node_name)?;
            push_u64(&mut out, c.neuron_idx)?;
            out.push(u8::from(c.is_active));
            out.extend_from_slice(&c.score.to_bits().to_le_bytes());
        }
        for c in &self.genbab_constraints {
            out.push(2);
            push_name(&mut out, &c.node_name)?;
            push_u64(&mut out, c.neuron_idx)?;
            out.extend_from_slice(&c.split_point.to_bits().to_le_bytes());
            out.push(u8::from(c.is_upper_branch));
            out.extend_from_slice(&c.score.to_bits().to_le_bytes());
            match c.input_index {
                Some(index) => {
                    out.push(1);
                    push_u64(&mut out, index)?;
                }
                None => {
                    out.push(0);
                    push_u64(&mut out, 0)?;
                }
            }
        }
        for &split_id in &self.genbab_split_ids {
            push_u64(&mut out, split_id)?;
        }
        for c in &self.norm_inv_rms_constraints {
            out.push(3);
            push_name(&mut out, &c.node_name)?;
            push_u64(&mut out, c.group_index)?;
            out.extend_from_slice(&c.inv_rms_lo.to_bits().to_le_bytes());
            out.extend_from_slice(&c.inv_rms_hi.to_bits().to_le_bytes());
            out.extend_from_slice(&c.score.to_bits().to_le_bytes());
        }
        debug_assert_eq!(out.len(), bytes);
        Some(out)
    }

    /// Get the node name of the most recent branch (regardless of ReLU vs GenBaB).
    ///
    /// Uses `split_count` and `genbab_split_ids` to determine whether the last
    /// split was ReLU or GenBaB, then returns the corresponding node name.
    ///
    /// Returns `None` if the history is empty (root domain).
    pub fn last_branch_node(&self) -> Option<&str> {
        if self.split_count == 0 {
            return None;
        }
        let last_split_id = self.split_count - 1;
        if let Some(&genbab_last_id) = self.genbab_split_ids.last() {
            if genbab_last_id == last_split_id {
                return self.genbab_constraints.last().map(|c| c.node_name.as_str());
            }
        }
        self.constraints.last().map(|c| c.node_name.as_str())
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    #[test]
    fn exact_identity_accepts_cap_boundary_and_refuses_one_byte_over() {
        let empty_name_history = GraphSplitHistory::new().with_constraint(
            GraphNeuronConstraint::new(String::new(), 0, true, 0.0).expect("constraint"),
        );
        let fixed_bytes = empty_name_history
            .exact_provenance_identity()
            .expect("small history identity")
            .len();
        assert!(fixed_bytes < PROVENANCE_HISTORY_MAX_BYTES);

        let mut at_cap = GraphSplitHistory::new();
        at_cap.add_constraint(
            GraphNeuronConstraint::new(
                "x".repeat(PROVENANCE_HISTORY_MAX_BYTES - fixed_bytes),
                0,
                true,
                0.0,
            )
            .expect("constraint"),
        );
        let identity = at_cap
            .exact_provenance_identity()
            .expect("exact cap is accepted");
        assert_eq!(identity.len(), PROVENANCE_HISTORY_MAX_BYTES);
        drop(identity);
        drop(at_cap);

        let mut over_cap = GraphSplitHistory::new();
        over_cap.add_constraint(
            GraphNeuronConstraint::new(
                "x".repeat(PROVENANCE_HISTORY_MAX_BYTES - fixed_bytes + 1),
                0,
                true,
                0.0,
            )
            .expect("constraint"),
        );
        assert!(over_cap.exact_provenance_identity().is_none());
    }
}
