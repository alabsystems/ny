// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph kFSB candidate-selection pieces (#kfsb-multi, barrier 2).
//!
//! Hoisted out of the single-objective GPU BaB lane (`gpu_bab::kfsb`) so the
//! multi-objective wave-batched selector
//! (`graph::multi_objective::batched::kfsb_multi`) uses the SAME candidate
//! type and top-k ∪ backup-top-k filter — one definition, two lanes.

use std::collections::HashSet;

use crate::beta_crown::config::KfsbReduceOp;

/// One kFSB branching candidate with its pre-scores (main = heuristic score
/// ranking, backup = the intercept channel used by `BranchingHeuristic::Kfsb`).
#[derive(Clone)]
pub(in crate::beta_crown::engine) struct GraphKfsbCandidate {
    pub node_name: String,
    pub neuron_idx: usize,
    pub main_score: f32,
    pub backup_score: f32,
}

/// Filter the pre-scored candidates down to the child-evaluation set:
/// top-`k` by main score UNION (when `use_backup`) top-`k` by backup score,
/// deduplicated, order-preserving. `scored` must already be sorted by main
/// score descending (NaN last). Extracted verbatim from the single-objective
/// lane (`select_graph_kfsb_eval_candidates`); `use_backup` mirrors its
/// `BranchingHeuristic::Kfsb` check.
pub(in crate::beta_crown::engine) fn select_graph_kfsb_eval_candidates(
    scored: &[GraphKfsbCandidate],
    k: usize,
    use_backup: bool,
) -> Vec<GraphKfsbCandidate> {
    let mut eval_candidates = Vec::new();
    let mut seen = HashSet::new();

    for candidate in scored.iter().take(k) {
        if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
            eval_candidates.push(candidate.clone());
        }
    }

    if use_backup {
        let mut backup_ranked = scored.to_vec();
        backup_ranked.sort_by(|a, b| {
            crate::cmp_utils::nan_propagating_cmp(&a.backup_score, &b.backup_score)
        });
        for candidate in backup_ranked.into_iter().take(k) {
            if seen.insert((candidate.node_name.clone(), candidate.neuron_idx)) {
                eval_candidates.push(candidate);
            }
        }
    }

    eval_candidates
}

/// Fold the two child bound values into the kFSB candidate score, honoring the
/// configured reduce op (α,β-CROWN's `branching:reduceop` parity knob — `Min`
/// is the classic conservative choice; `Max` rewards ONE-SIDED verifiers,
/// where one child is fully verified/infeasible and only the other survives).
#[inline]
pub(in crate::beta_crown::engine) fn kfsb_reduce(
    op: KfsbReduceOp,
    active: f32,
    inactive: f32,
) -> f32 {
    match op {
        KfsbReduceOp::Min => active.min(inactive),
        KfsbReduceOp::Max => active.max(inactive),
        // Bit-identical anchor: `f32::midpoint` rounds differently at overflow/subnormal
        // edges, and this mean folds two child bound scores that steer branch selection.
        #[allow(clippy::manual_midpoint)]
        KfsbReduceOp::Mean => (active + inactive) / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{select_graph_kfsb_eval_candidates, GraphKfsbCandidate};

    /// A pool of `n` candidates whose main scores STRICTLY decrease (rank 0
    /// highest, all distinct) and sorted by main score descending — the input
    /// contract of `select_graph_kfsb_eval_candidates`. `backup_score` is
    /// CONSTANT, so the (stable) backup re-sort preserves the main order and the
    /// backup top-k is the SAME set as the main top-k: the union adds nothing,
    /// isolating the `k`-cut on the main channel to exactly `min(k, n)`.
    fn descending_pool(n: usize) -> Vec<GraphKfsbCandidate> {
        (0..n)
            .map(|i| GraphKfsbCandidate {
                node_name: "relu".to_string(),
                neuron_idx: i,
                main_score: (n - i) as f32, // strictly decreasing, distinct
                backup_score: 0.0,          // constant ⇒ backup top-k ⊆ main top-k
            })
            .collect()
    }

    /// GUARD (#kfsb-multi `NY_MO_KFSB_K`): with a candidate pool LARGER than
    /// either `k`, `k=3` selects STRICTLY FEWER candidates than `k=7`. This
    /// pins the `k` knob as live — the earlier "min3 ≡ min7 identical table"
    /// was candidate-pool SATURATION (a pool ≤ k, so both `k`s admit the whole
    /// pool), NOT an inert knob. Pure function, no solver.
    #[test]
    fn kfsb_eval_candidate_k_strictly_binds_on_large_pool() {
        let pool = descending_pool(12); // > 7, so neither k saturates
        let k3 = select_graph_kfsb_eval_candidates(&pool, 3, true);
        let k7 = select_graph_kfsb_eval_candidates(&pool, 7, true);

        // Constant backup ⇒ the union is a no-op, so each k admits exactly its
        // main top-k.
        assert_eq!(
            k3.len(),
            3,
            "k=3 admits exactly its top-3 (backup adds none)"
        );
        assert_eq!(
            k7.len(),
            7,
            "k=7 admits exactly its top-7 (backup adds none)"
        );
        assert!(
            k3.len() < k7.len(),
            "k must bind: k=3 must select strictly fewer than k=7 on a pool of 12"
        );

        // k=3's selection is the prefix of k=7's (both are the main-score top-k).
        for (a, b) in k3.iter().zip(k7.iter()) {
            assert_eq!(
                (a.node_name.as_str(), a.neuron_idx),
                (b.node_name.as_str(), b.neuron_idx),
                "k=3 top-k must be the leading prefix of k=7 top-k"
            );
        }

        // Sanity: a pool NOT larger than k saturates — this is the earlier
        // "min3 ≡ min7" regime, and correctly does NOT distinguish the knobs.
        let small = descending_pool(3);
        assert_eq!(select_graph_kfsb_eval_candidates(&small, 3, true).len(), 3);
        assert_eq!(select_graph_kfsb_eval_candidates(&small, 7, true).len(), 3);
    }
}
