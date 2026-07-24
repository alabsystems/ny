// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BICCOS cut merging for graph cutting planes.
//!
//! Merges sibling cuts that differ by exactly one variable's sign into parent
//! cuts, then deduplicates and prunes redundant cuts. Based on the BICCOS
//! algorithm from alpha-beta-CROWN (`complete_verifier/cuts/infered_cuts.py:212-343`).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use super::super::merge_index::{dedup_signatures, sign_i8, ParentSubsetIndex};
use crate::beta_crown::bab_cuts::{CutMetadata, GraphCutTerm, GraphCuttingPlane};

use super::GraphCutPool;

type GraphCutSignatureKey<'a> = (&'a str, usize, i8);

impl GraphCutPool {
    /// Merge sibling cuts that differ by exactly one variable's sign.
    pub fn merge_cuts(&mut self) -> usize {
        if self.cuts.len() < 2 {
            return self.cuts.len();
        }

        let mut changed = true;
        while changed {
            let prepared_signatures: Vec<Vec<GraphCutSignatureKey<'_>>> =
                self.cuts.iter().map(Self::cut_signature_ref).collect();
            let sig_to_idx = Self::build_signature_index(&prepared_signatures);
            let (new_cuts, merged_any) = self.collect_merge_pass(&prepared_signatures, &sig_to_idx);
            let pruned = Self::dedup_and_prune(new_cuts);

            let changed_by_signature = {
                let old_sigs: HashSet<_> = prepared_signatures.iter().cloned().collect();
                let new_sigs: HashSet<_> = pruned.iter().map(Self::cut_signature_ref).collect();
                old_sigs != new_sigs
            };
            changed = merged_any || changed_by_signature;

            self.cuts = pruned;
        }

        self.cuts.len()
    }

    fn build_signature_index<'a>(
        signatures: &[Vec<GraphCutSignatureKey<'a>>],
    ) -> HashMap<Vec<GraphCutSignatureKey<'a>>, Vec<usize>> {
        let mut sig_to_idx: HashMap<Vec<GraphCutSignatureKey<'a>>, Vec<usize>> =
            HashMap::with_capacity(signatures.len());
        for (idx, signature) in signatures.iter().cloned().enumerate() {
            sig_to_idx.entry(signature).or_default().push(idx);
        }
        sig_to_idx
    }

    fn collect_merge_pass<'a>(
        &self,
        prepared_signatures: &[Vec<GraphCutSignatureKey<'a>>],
        sig_to_idx: &HashMap<Vec<GraphCutSignatureKey<'a>>, Vec<usize>>,
    ) -> (Vec<GraphCuttingPlane>, bool) {
        let mut processed: HashSet<usize> = HashSet::new();
        let mut new_cuts: Vec<GraphCuttingPlane> = Vec::new();
        let mut merged_any = false;

        for (idx, cut) in self.cuts.iter().enumerate() {
            if processed.contains(&idx) {
                continue;
            }

            let cut_signature = &prepared_signatures[idx];
            let mut merged = false;
            for (term_idx, term) in cut.terms.iter().enumerate() {
                if term.coefficient < 0.0 {
                    continue;
                }

                let sibling_sig = Self::sibling_signature(cut_signature, term);
                if let Some(sibling_candidates) = sig_to_idx.get(&sibling_sig) {
                    if let Some(sibling_idx) =
                        sibling_candidates.iter().copied().find(|&sibling_idx| {
                            sibling_idx != idx && !processed.contains(&sibling_idx)
                        })
                    {
                        processed.insert(idx);
                        processed.insert(sibling_idx);

                        if let Some(parent) = Self::create_parent_cut(cut, term_idx)
                            .filter(|parent| !parent.terms.is_empty())
                        {
                            new_cuts.push(parent);
                        }

                        merged = true;
                        merged_any = true;
                        break;
                    }
                }
            }

            if !merged && !processed.contains(&idx) {
                new_cuts.push(cut.clone());
                processed.insert(idx);
            }
        }

        (new_cuts, merged_any)
    }

    fn dedup_and_prune(cuts: Vec<GraphCuttingPlane>) -> Vec<GraphCuttingPlane> {
        let dedup_keep = {
            let signatures: Vec<Vec<GraphCutSignatureKey<'_>>> =
                cuts.iter().map(Self::cut_signature_ref).collect();
            dedup_signatures(signatures.iter().cloned())
        };
        let mut deduped: Vec<GraphCuttingPlane> = cuts
            .into_iter()
            .zip(dedup_keep)
            .filter_map(|(cut, keep)| keep.then_some(cut))
            .collect();

        deduped.sort_by_key(|cut| cut.terms.len());
        let prune_keep = {
            let signatures: Vec<Vec<GraphCutSignatureKey<'_>>> =
                deduped.iter().map(Self::cut_signature_ref).collect();
            Self::prune_redundant_keep(&signatures)
        };
        deduped
            .into_iter()
            .zip(prune_keep)
            .filter_map(|(cut, keep)| keep.then_some(cut))
            .collect()
    }

    fn cut_signature_ref(cut: &GraphCuttingPlane) -> Vec<GraphCutSignatureKey<'_>> {
        let mut sig: Vec<GraphCutSignatureKey<'_>> = cut.terms.iter().map(Self::term_key).collect();
        sig.sort_unstable();
        sig
    }

    fn sibling_signature<'a>(
        signature: &'a [GraphCutSignatureKey<'a>],
        flipped_term: &GraphCutTerm,
    ) -> Vec<GraphCutSignatureKey<'a>> {
        let mut sig = signature.to_vec();
        let flipped_sign = sign_i8(flipped_term.coefficient);
        if let Some(key) = sig.iter_mut().find(|key| {
            key.0 == flipped_term.node_name
                && key.1 == flipped_term.neuron_idx
                && key.2 == flipped_sign
        }) {
            key.2 = -1;
        }
        sig.sort_unstable();
        sig
    }

    fn term_key(term: &GraphCutTerm) -> GraphCutSignatureKey<'_> {
        (
            term.node_name.as_str(),
            term.neuron_idx,
            sign_i8(term.coefficient),
        )
    }

    fn create_parent_cut(cut: &GraphCuttingPlane, remove_idx: usize) -> Option<GraphCuttingPlane> {
        let terms: Vec<GraphCutTerm> = cut
            .terms
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != remove_idx)
            .map(|(_, t)| t.clone())
            .collect();

        let bias = cut.bias - 1.0;
        if !bias.is_finite() {
            return None;
        }

        GraphCuttingPlane::new(
            terms,
            bias,
            cut.lambda.max(0.01),
            cut.source_depth.saturating_sub(1),
            CutMetadata::new(
                cut.metadata.created_iter.load(Ordering::Relaxed),
                cut.metadata.cut_kind(),
            ),
        )
        .ok()
    }

    fn prune_redundant_keep(signatures: &[Vec<GraphCutSignatureKey<'_>>]) -> Vec<bool> {
        let mut keep = vec![true; signatures.len()];
        let mut parent_index = ParentSubsetIndex::new();

        for (idx, signature) in signatures.iter().enumerate() {
            if signature.is_empty() {
                keep[idx] = false;
                continue;
            }

            if parent_index.has_parent_subset(signature) {
                keep[idx] = false;
            } else {
                parent_index.insert(signature);
            }
        }

        keep
    }
}
