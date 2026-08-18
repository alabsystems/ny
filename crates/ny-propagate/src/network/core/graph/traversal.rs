// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph traversal algorithms.

use std::collections::{HashMap, HashSet};

use ny_core::{NyError, Result};

use super::{GraphNetwork, GraphNode, NETWORK_INPUT};

impl GraphNetwork {
    /// Borrow a previously initialized execution order without initializing it.
    ///
    /// The retained-BaB v1 static composer is a pre-open, default-dark consumer:
    /// a cold cache is an ordinary refusal and must never trigger an allocation-
    /// bearing topological sort after provider selection or phase opening.
    pub(crate) fn retained_v1_exec_order_if_cached(&self) -> Option<&[String]> {
        self.cached_exec_order.get().map(Vec::as_slice)
    }

    /// Borrow the graph execution order, caching the computed topological sort
    /// until the graph structure mutates.
    pub fn exec_order(&self) -> Result<&[String]> {
        if let Some(exec_order) = self.cached_exec_order.get() {
            return Ok(exec_order.as_slice());
        }

        let exec_order = self.compute_topological_sort()?;
        let _ = self.cached_exec_order.set(exec_order);
        self.cached_exec_order
            .get()
            .map(Vec::as_slice)
            .ok_or_else(|| {
                NyError::InternalError("exec_order cache missing after initialization".to_string())
            })
    }

    /// Perform topological sort to get valid execution order.
    ///
    /// Returns node names in order such that all dependencies come before dependents.
    /// Returns an error if the graph contains cycles. Prefer [`Self::exec_order`]
    /// when an owned `Vec` is not required.
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        Ok(self.exec_order()?.to_vec())
    }

    /// Borrow the cached per-node ancestors map, computing it on first access.
    ///
    /// For each node, ancestors are all nodes reachable backward (excluding
    /// `NETWORK_INPUT`), returned in topological order. The map is computed
    /// incrementally in O(N*(N+E)) total by walking exec_order forward and
    /// unioning parent ancestor sets (#2220 Packet A, #2237 F1).
    pub(crate) fn all_ancestors(&self) -> Result<&HashMap<String, Vec<String>>> {
        if let Some(cached) = self.cached_ancestors.get() {
            return Ok(cached);
        }

        let exec_order = self.exec_order()?;
        // For each node, compute its ancestor set incrementally:
        // ancestors(node) = {node} ∪ ∪_{parent ∈ inputs(node)} ancestors(parent)
        // Since exec_order is topological, all parents are processed before children.
        let mut ancestor_sets: HashMap<String, HashSet<String>> =
            HashMap::with_capacity(exec_order.len());

        for node_name in exec_order {
            let mut ancestors = HashSet::new();
            ancestors.insert(node_name.clone());

            if let Some(node) = self.nodes.get(node_name) {
                for input_name in &node.inputs {
                    if input_name == NETWORK_INPUT {
                        continue;
                    }
                    if let Some(parent_ancestors) = ancestor_sets.get(input_name) {
                        ancestors.extend(parent_ancestors.iter().cloned());
                    }
                }
            }

            ancestor_sets.insert(node_name.clone(), ancestors);
        }

        // Convert sets to topologically-ordered vecs
        let mut result: HashMap<String, Vec<String>> = HashMap::with_capacity(ancestor_sets.len());
        for node_name in exec_order {
            if let Some(set) = ancestor_sets.get(node_name) {
                let mut ordered = Vec::with_capacity(set.len());
                for name in exec_order {
                    if set.contains(name) {
                        ordered.push(name.clone());
                    }
                }
                result.insert(node_name.clone(), ordered);
            }
        }

        let _ = self.cached_ancestors.set(result);
        self.cached_ancestors.get().ok_or_else(|| {
            NyError::InternalError("ancestors cache missing after initialization".to_string())
        })
    }

    /// Compute the set of nodes reachable *downstream* (forward) from any of the
    /// given seed nodes, including the seeds themselves.
    ///
    /// A node `n` is in the returned set iff `n` is one of `seeds` or at least one
    /// of `n`'s inputs is (transitively) in the set. Because `exec_order` is
    /// topological, a single forward sweep suffices: every input of `n` is
    /// processed before `n`.
    ///
    /// Used by constrained forward-bound caching (#issue: graph BaB upstream
    /// inheritance): when a BaB split adds a constraint at a single node, only
    /// nodes downstream of that node can change. Bounds for all other nodes are
    /// provably identical to the parent's and may be reused verbatim.
    ///
    /// The seed set is treated conservatively: any seed name that is not present
    /// in the graph (or is `NETWORK_INPUT`) still contributes its own membership,
    /// and a `NETWORK_INPUT` seed marks *every* node as downstream (the whole
    /// network depends on the input) — i.e. nothing may be reused. This keeps the
    /// caller sound even when the split node cannot be resolved.
    pub(crate) fn descendants_inclusive(&self, seeds: &[String]) -> Result<HashSet<String>> {
        let exec_order = self.exec_order()?;
        let seed_set: HashSet<&str> = seeds.iter().map(String::as_str).collect();

        // If the network input itself is a seed, every node is downstream.
        let input_is_seed = seed_set.contains(NETWORK_INPUT);

        let mut downstream: HashSet<String> = HashSet::with_capacity(exec_order.len());
        for node_name in exec_order {
            let is_seed = seed_set.contains(node_name.as_str());
            let mut affected = is_seed || input_is_seed;
            if !affected {
                if let Some(node) = self.nodes.get(node_name) {
                    affected = node.inputs.iter().any(|input_name| {
                        input_name == NETWORK_INPUT && input_is_seed
                            || downstream.contains(input_name.as_str())
                    });
                }
            }
            if affected {
                downstream.insert(node_name.clone());
            }
        }

        // Defensive: ensure every seed name appears in the result even if it was
        // absent from exec_order (e.g. a malformed history). Conservative: an
        // unknown seed only adds itself, which can never cause unsound *reuse*.
        for s in seeds {
            if s != NETWORK_INPUT {
                downstream.insert(s.clone());
            }
        }

        Ok(downstream)
    }

    fn compute_topological_sort(&self) -> Result<Vec<String>> {
        let mut visited = HashSet::with_capacity(self.nodes.len());
        let mut temp_mark = HashSet::with_capacity(self.nodes.len());
        let mut result = Vec::with_capacity(self.nodes.len());

        fn visit(
            name: &str,
            nodes: &super::TrackedStringMap<GraphNode>,
            visited: &mut HashSet<String>,
            temp_mark: &mut HashSet<String>,
            result: &mut Vec<String>,
        ) -> Result<()> {
            if visited.contains(name) {
                return Ok(());
            }
            if temp_mark.contains(name) {
                return Err(NyError::InvalidSpec(format!(
                    "Cycle detected in graph at node: {}",
                    name
                )));
            }
            if name == NETWORK_INPUT {
                return Ok(());
            }

            let node = nodes.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Dangling input reference: node '{}' is referenced as a dependency \
                     but does not exist in the graph",
                    name
                ))
            })?;

            temp_mark.insert(name.to_string());

            for input_name in &node.inputs {
                visit(input_name, nodes, visited, temp_mark, result).map_err(|e| {
                    // If the child error is about a dangling reference, augment
                    // with consumer node context so diagnostics show the edge.
                    match &e {
                        NyError::InvalidSpec(msg) if msg.contains("Dangling input") => {
                            NyError::InvalidSpec(format!("{} (referenced by node '{}')", msg, name))
                        }
                        _ => e,
                    }
                })?;
            }

            temp_mark.remove(name);
            visited.insert(name.to_string());
            result.push(name.to_string());
            Ok(())
        }

        // Sort keys for deterministic topological ordering
        // (HashMap iteration order is non-deterministic)
        let mut sorted_keys: Vec<&String> = self.nodes.keys().collect();
        sorted_keys.sort();
        for name in sorted_keys {
            visit(name, &self.nodes, &mut visited, &mut temp_mark, &mut result)?;
        }

        Ok(result)
    }
}
