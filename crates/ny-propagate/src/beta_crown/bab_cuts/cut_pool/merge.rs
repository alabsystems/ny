// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cut merging and deduplication.
//!
//! Implements BICCOS-style cut merging: sibling cuts that differ by exactly
//! one coefficient sign are merged into a stronger parent cut. After merging,
//! redundant child cuts are pruned.
//!
//! Reference: alpha-beta-CROWN `complete_verifier/cuts/infered_cuts.py:212-343`

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use super::CutPool;
use crate::beta_crown::bab_cuts::{CutMetadata, CutTerm, CuttingPlane};

impl CutPool {
    /// Merge sibling cuts to reduce redundancy and create stronger parent cuts.
    ///
    /// This implements BICCOS-style cut merging. Two cuts are siblings if they differ
    /// by exactly one coefficient sign (+1.0 vs -1.0 at the same position). When merged:
    /// - The differing term is removed
    /// - Bias is adjusted by -1
    /// - The parent cut is stronger (fewer constraints = more general)
    ///
    /// Example:
    /// - Cut A: z1 + z2 >= 1 (terms: [(0,0,+1), (0,1,+1)], bias: 1)
    /// - Cut B: z1 - z2 >= 0 (terms: [(0,0,+1), (0,1,-1)], bias: 0)
    /// - Parent: z1 >= 0.5 (terms: [(0,0,+1)], bias adjusted)
    ///
    /// The function iteratively merges until no more siblings are found, then prunes
    /// cuts whose parent already exists in the pool.
    ///
    /// # Returns
    /// Number of cuts after merging (may be less than before).
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/cuts/infered_cuts.py:212-343`
    pub fn merge_cuts(&mut self) -> usize {
        // Fast path: nothing to merge
        if self.cuts.len() < 2 {
            return self.cuts.len();
        }

        let mut changed = true;
        while changed {
            changed = false;

            // Build lookup: cut signature -> all cut indices with that signature.
            let mut sig_to_idx: HashMap<Vec<(usize, usize, i8)>, Vec<usize>> = HashMap::new();
            for (idx, cut) in self.cuts.iter().enumerate() {
                sig_to_idx
                    .entry(Self::cut_signature(cut))
                    .or_default()
                    .push(idx);
            }

            // Find mergeable sibling pairs and produce merged/kept cuts
            let (new_cuts, any_merged) = Self::find_and_merge_siblings(&self.cuts, &sig_to_idx);
            changed |= any_merged;

            // Deduplicate, sort by complexity, and prune redundant child cuts
            let deduped = Self::deduplicate_cuts(new_cuts);
            let mut to_prune = deduped;
            to_prune.sort_by_key(|c| c.terms.len());
            let pruned = Self::prune_redundant_cuts(to_prune);

            // Detect content changes via signature comparison
            let old_sigs: HashSet<_> = self.cuts.iter().map(Self::cut_signature).collect();
            let new_sigs: HashSet<_> = pruned.iter().map(Self::cut_signature).collect();
            if old_sigs != new_sigs {
                changed = true;
            }

            self.cuts = pruned;
            self.rebuild_live_counts();
        }

        self.cuts.len()
    }

    /// Scan `cuts` for sibling pairs (differ by exactly one coefficient sign),
    /// merge each pair into a parent cut, and keep unmerged cuts unchanged.
    ///
    /// Returns the new cut list and whether any merges occurred.
    fn find_and_merge_siblings(
        cuts: &[CuttingPlane],
        sig_to_idx: &HashMap<Vec<(usize, usize, i8)>, Vec<usize>>,
    ) -> (Vec<CuttingPlane>, bool) {
        let mut processed: HashSet<usize> = HashSet::new();
        let mut new_cuts: Vec<CuttingPlane> = Vec::new();
        let mut any_merged = false;

        for (idx, cut) in cuts.iter().enumerate() {
            if processed.contains(&idx) {
                continue;
            }
            if let Some(sibling_idx) =
                Self::try_merge_cut(cut, idx, sig_to_idx, &processed, &mut new_cuts)
            {
                processed.insert(idx);
                processed.insert(sibling_idx);
                any_merged = true;
            } else {
                new_cuts.push(cut.clone());
                processed.insert(idx);
            }
        }

        (new_cuts, any_merged)
    }

    /// Try to merge `cut` with a sibling. If successful, pushes the parent
    /// cut into `new_cuts` and returns the sibling index.
    fn try_merge_cut(
        cut: &CuttingPlane,
        idx: usize,
        sig_to_idx: &HashMap<Vec<(usize, usize, i8)>, Vec<usize>>,
        processed: &HashSet<usize>,
        new_cuts: &mut Vec<CuttingPlane>,
    ) -> Option<usize> {
        for (term_idx, term) in cut.terms.iter().enumerate() {
            if term.coefficient < 0.0 {
                continue;
            }
            let sibling_sig = Self::sibling_signature(cut, term_idx);
            if let Some(sibling_candidates) = sig_to_idx.get(&sibling_sig) {
                if let Some(sibling_idx) = sibling_candidates
                    .iter()
                    .copied()
                    .find(|&sibling_idx| sibling_idx != idx && !processed.contains(&sibling_idx))
                {
                    if let Some(parent) =
                        Self::create_parent_cut(cut, term_idx).filter(|p| !p.terms.is_empty())
                    {
                        new_cuts.push(parent);
                    }
                    return Some(sibling_idx);
                }
            }
        }
        None
    }

    /// Create a signature for a cut (sorted list of normalized term tuples).
    pub(super) fn cut_signature(cut: &CuttingPlane) -> Vec<(usize, usize, i8)> {
        let mut sig: Vec<(usize, usize, i8)> = cut
            .terms
            .iter()
            .map(|t| {
                let coeff_sign = if t.coefficient > 0.0 {
                    1
                } else if t.coefficient < 0.0 {
                    -1
                } else {
                    0
                };
                (t.layer_idx, t.neuron_idx, coeff_sign)
            })
            .collect();
        sig.sort_unstable();
        sig
    }

    /// Create a sibling signature (flip coefficient at term_idx).
    fn sibling_signature(cut: &CuttingPlane, flip_idx: usize) -> Vec<(usize, usize, i8)> {
        let mut sig: Vec<(usize, usize, i8)> = cut
            .terms
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let coeff_sign = if i == flip_idx {
                    -1 // Flip the sign
                } else if t.coefficient > 0.0 {
                    1
                } else if t.coefficient < 0.0 {
                    -1
                } else {
                    0
                };
                (t.layer_idx, t.neuron_idx, coeff_sign)
            })
            .collect();
        sig.sort_unstable();
        sig
    }

    /// Create a parent cut by removing the term at the given index.
    /// Returns `None` if the derived bias is non-finite (#3148).
    fn create_parent_cut(cut: &CuttingPlane, remove_idx: usize) -> Option<CuttingPlane> {
        let terms: Vec<CutTerm> = cut
            .terms
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != remove_idx)
            .map(|(_, t)| *t)
            .collect();

        // Bias adjustment: parent bias = original bias - 1
        // This follows the BICCOS merging rule from alpha-beta-CROWN
        let bias = cut.bias - 1.0;

        // Skip merge if bias arithmetic produced non-finite (#3148).
        // Extreme-bias cuts (near f32::MIN) can overflow to -Inf on subtraction.
        if !bias.is_finite() {
            return None;
        }

        // Inherit lambda from parent or use small default. Both bias and lambda
        // are guaranteed valid since the source CuttingPlane is already validated.
        CuttingPlane::new(
            terms,
            bias,
            cut.lambda.max(0.01), // Inherit lambda or use small default
            cut.source_depth.saturating_sub(1),
            CutMetadata::new(
                cut.metadata.created_iter.load(Ordering::Relaxed),
                cut.metadata.cut_kind(),
            ),
        )
        .ok()
    }

    /// Deduplicate cuts with identical signatures.
    ///
    /// When multiple cuts have the same signature (same terms with same coefficient signs),
    /// keep only the first one encountered. This handles the case where multiple sibling
    /// pairs merge into identical parent cuts.
    pub(super) fn deduplicate_cuts(cuts: Vec<CuttingPlane>) -> Vec<CuttingPlane> {
        let mut seen_sigs: HashSet<Vec<(usize, usize, i8)>> = HashSet::new();
        let mut result: Vec<CuttingPlane> = Vec::new();

        for cut in cuts {
            let sig = Self::cut_signature(&cut);
            if seen_sigs.insert(sig) {
                result.push(cut);
            }
        }

        result
    }

    /// Prune cuts that are redundant (a more general parent cut already exists).
    ///
    /// A cut C is redundant if there exists a parent cut P where:
    /// - Every term in P appears in C with the same coefficient sign
    /// - P has fewer terms (is more general)
    ///
    /// Optimization (#2326 Finding 2): For each candidate cut, build a HashSet
    /// of its terms so that `is_parent_of` lookups are O(1) per parent term
    /// instead of O(T_child). Total complexity: O(n^2 * T_parent) instead of
    /// O(n^2 * T_parent * T_child).
    fn prune_redundant_cuts(cuts: Vec<CuttingPlane>) -> Vec<CuttingPlane> {
        // Already sorted by term count (ascending), so parents come first
        let mut result: Vec<CuttingPlane> = Vec::new();

        for cut in cuts {
            if cut.terms.is_empty() {
                continue; // Skip degenerate cuts
            }

            // Build a HashSet of (layer_idx, neuron_idx, is_positive) for O(1) lookups.
            // This converts the inner loop of is_parent_of from O(T_child) to O(1).
            let child_term_set: HashSet<(usize, usize, bool)> = cut
                .terms
                .iter()
                .map(|t| (t.layer_idx, t.neuron_idx, t.coefficient > 0.0))
                .collect();

            let mut is_redundant = false;
            for parent in &result {
                if Self::is_parent_of_with_set(parent, &child_term_set, cut.terms.len()) {
                    is_redundant = true;
                    break;
                }
            }

            if !is_redundant {
                result.push(cut);
            }
        }

        result
    }

    /// Check if `parent` is a parent of `child` using a pre-built HashSet for O(1) lookups.
    ///
    /// This is the optimized version of `is_parent_of` that avoids the O(T_child)
    /// linear scan per parent term. Total: O(T_parent) per check instead of
    /// O(T_parent * T_child).
    ///
    /// Reference: Issue #2326 Finding 2
    fn is_parent_of_with_set(
        parent: &CuttingPlane,
        child_term_set: &HashSet<(usize, usize, bool)>,
        child_term_count: usize,
    ) -> bool {
        if parent.terms.len() >= child_term_count {
            return false; // Parent must be strictly smaller
        }

        // Every term in parent must appear in child with same sign
        parent
            .terms
            .iter()
            .all(|pt| child_term_set.contains(&(pt.layer_idx, pt.neuron_idx, pt.coefficient > 0.0)))
    }

    /// Check if `parent` is a parent of `child` (child's terms contain all of parent's terms).
    ///
    /// Retained for use in contexts where building a HashSet is not worthwhile
    /// (e.g., single-pair checks in tests). For batch pruning, use
    /// `is_parent_of_with_set` via `prune_redundant_cuts`.
    #[cfg(test)]
    pub(super) fn is_parent_of(parent: &CuttingPlane, child: &CuttingPlane) -> bool {
        if parent.terms.len() >= child.terms.len() {
            return false; // Parent must be strictly smaller
        }

        // Every term in parent must appear in child with same sign
        for pt in &parent.terms {
            let found = child.terms.iter().any(|ct| {
                ct.layer_idx == pt.layer_idx
                    && ct.neuron_idx == pt.neuron_idx
                    && (ct.coefficient > 0.0) == (pt.coefficient > 0.0)
            });
            if !found {
                return false;
            }
        }

        true
    }
}
