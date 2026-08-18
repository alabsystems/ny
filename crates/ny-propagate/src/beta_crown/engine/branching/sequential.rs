// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential-network branching heuristics and CROWN-based scoring.

use super::*;

use ny_core::{nan_propagating_max, nan_propagating_min};

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn select_split_neuron(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize, f32)>> {
        match self.config.branching_heuristic {
            BranchingHeuristic::LargestBoundWidth => {
                self.select_largest_width_neuron(network, domain)
            }
            BranchingHeuristic::Sequential => self.select_sequential_neuron(network, domain),
            BranchingHeuristic::BoundImpact => self.select_babsr_neuron(network, domain),
            BranchingHeuristic::FilteredSmartBranching => {
                self.select_fsb_neuron(network, input, domain)
            }
            BranchingHeuristic::Kfsb => {
                // kFSB with both BaBSR alpha score and intercept backup
                self.select_kfsb_neuron(network, input, domain, false)
            }
            BranchingHeuristic::KfsbInterceptOnly => {
                // kFSB with intercept-only scoring
                self.select_kfsb_neuron(network, input, domain, true)
            }
            BranchingHeuristic::InputSplit => {
                // Signal that we should use input splitting instead of ReLU splitting
                Ok(None)
            }
            BranchingHeuristic::GenBaB(_) => {
                // GenBaB is designed for GraphNetwork, not sequential Network
                // Use LargestBoundWidth as fallback for sequential networks
                self.select_largest_width_neuron(network, domain)
            }
        }
    }

    /// Select the unstable zero-threshold neuron (ReLU or Sign) with largest
    /// pre-activation bound width.
    fn select_largest_width_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize, f32)>> {
        let mut best: Option<(usize, usize, f32)> = None;

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !is_zero_threshold_binary_activation(layer) {
                continue;
            }

            // Get pre-activation bounds for this ReLU/Sign
            // Pre-activation is from the previous layer's output
            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            // Find unstable neurons (l < 0 < u) that aren't already constrained
            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];

                // Check if unstable
                if l >= 0.0 || u <= 0.0 {
                    continue;
                }

                // Check if already constrained
                if domain
                    .history
                    .is_constrained(layer_idx, neuron_idx)
                    .is_some()
                {
                    continue;
                }

                let width = u - l;
                // Skip NaN widths — corrupt bounds must not be selected (#2588).
                if width.is_nan() {
                    continue;
                }
                if best.map_or(true, |b| width > b.2) {
                    best = Some((layer_idx, neuron_idx, width));
                }
            }
        }

        Ok(best)
    }

    /// Select the unstable neuron using BaBSR heuristic (Bound-impact scoring).
    ///
    /// BaBSR scores each unstable zero-threshold neuron (ReLU or Sign) using
    /// the reference-style signed-lA score kernel with producer bias.
    fn select_babsr_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize, f32)>> {
        // First, collect all unstable neurons with their basic info
        let mut candidates: Vec<(usize, usize, f32, f32)> = Vec::new(); // (layer, neuron, lb, ub)

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !is_zero_threshold_binary_activation(layer) {
                continue;
            }

            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];

                // Check if unstable and not constrained
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    candidates.push((layer_idx, neuron_idx, l, u));
                }
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        let babsr_scores = self.compute_babsr_scores(network, domain, KfsbReduceOp::Min)?;

        let mut best: Option<(usize, usize, f32)> = None;

        for (layer_idx, neuron_idx, lb, ub) in candidates {
            let score = babsr_scores
                .get(&(layer_idx, neuron_idx))
                .copied()
                .unwrap_or_else(|| {
                    debug!(
                        layer = layer_idx,
                        neuron = neuron_idx,
                        lower = lb,
                        upper = ub,
                        "BaBSR sequential: no score parts for neuron, using 0.0 fallback"
                    );
                    BabsrScoreParts::default()
                })
                .main_score;

            // Skip NaN-scored neurons — corrupt scores must not be selected (#2588).
            if score.is_nan() {
                continue;
            }

            if best.map_or(true, |b| score > b.2) {
                best = Some((layer_idx, neuron_idx, score));
            }
        }

        trace!(
            "BaBSR selected neuron layer={}, idx={}, score={:.4}",
            best.map(|b| b.0).unwrap_or(0),
            best.map(|b| b.1).unwrap_or(0),
            best.map(|b| b.2).unwrap_or(0.0)
        );

        Ok(best)
    }

    /// Select the unstable ReLU neuron using KFSB-style heuristic.
    ///
    /// KFSB (K-FSB) uses two scoring methods to select branching candidates:
    /// 1) BaBSR score: coefficient magnitude * intercept (higher is better)
    /// 2) Intercept-only score: pure triangle relaxation gap (higher is better)
    ///
    /// Strategy:
    /// 1) Rank by BaBSR and take top-k candidates.
    /// 2) Rank by intercept-only and take top-k candidates.
    /// 3) Merge both sets (deduplicate) and evaluate all candidates.
    /// 4) Choose the split that maximizes worst-child improvement.
    ///
    /// The intercept-only scoring helps find neurons where the relaxation gap
    /// is large even if the CROWN coefficient is small.
    fn select_fsb_neuron(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize, f32)>> {
        let k = self.config.fsb_candidates;
        if k == 0 {
            return self.select_babsr_neuron(network, domain);
        }

        // Collect unstable, unconstrained neurons with their scores.
        let mut candidates: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
        // (layer_idx, neuron_idx, lb, ub, intercept)

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !is_zero_threshold_binary_activation(layer) {
                continue;
            }
            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    // Triangle relaxation intercept: measures looseness
                    let intercept = relu_intercept_score(l, u);
                    candidates.push((layer_idx, neuron_idx, l, u, intercept));
                }
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        let babsr_scores = self.compute_babsr_scores(network, domain, KfsbReduceOp::Min)?;

        // Compute both scores for all candidates
        let mut scored: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(candidates.len());
        // (layer_idx, neuron_idx, babsr_score, intercept_score)
        for (layer_idx, neuron_idx, lb, ub, _) in &candidates {
            let score_parts = babsr_scores
                .get(&(*layer_idx, *neuron_idx))
                .copied()
                .unwrap_or_else(|| {
                    debug!(
                        layer = layer_idx,
                        neuron = neuron_idx,
                        lower = lb,
                        upper = ub,
                        "FSB sequential: no score parts for neuron, using 0.0 fallback"
                    );
                    BabsrScoreParts::default()
                });
            scored.push((
                *layer_idx,
                *neuron_idx,
                score_parts.main_score,
                score_parts.backup_score,
            ));
        }

        // Get top-k by BaBSR score (higher is better - "alpha" score, NaN last — #2995)
        let mut babsr_ranked = scored.clone();
        babsr_ranked.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.2, &b.2));

        // Get top-k by intercept-only score (lower/more negative is better - "backup" score)
        // Per alpha-beta-CROWN kfsb.py: intercept list uses `largest=False` for k-smallest
        let mut intercept_ranked = scored.clone();
        intercept_ranked.sort_by(|a, b| crate::cmp_utils::nan_propagating_cmp(&a.3, &b.3));

        // Merge candidates from both rankings, deduplicate by (layer, neuron)
        let mut eval_candidates: Vec<(usize, usize, f32, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        // Add top-k from BaBSR
        for &(layer_idx, neuron_idx, babsr_score, intercept_score) in babsr_ranked.iter().take(k) {
            if seen.insert((layer_idx, neuron_idx)) {
                eval_candidates.push((layer_idx, neuron_idx, babsr_score, intercept_score));
            }
        }

        // Add top-k from intercept-only (may add up to k more, but often overlaps)
        for &(layer_idx, neuron_idx, babsr_score, intercept_score) in
            intercept_ranked.iter().take(k)
        {
            if seen.insert((layer_idx, neuron_idx)) {
                eval_candidates.push((layer_idx, neuron_idx, babsr_score, intercept_score));
            }
        }

        // Evaluate all candidates by computing child bounds
        let mut best: Option<(usize, usize, f32, f32)> = None; // (layer, neuron, fsb_score, babsr_score)
        for &(layer_idx, neuron_idx, babsr_score, _intercept_score) in &eval_candidates {
            let active = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, true,
            )?;
            let inactive = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, false,
            )?;

            let fsb_score = match self.config.fsb_worst_case_score(active, inactive) {
                Some(score) => score,
                None => continue,
            };

            // Skip NaN-scored candidates — corrupt scores must not win (#2588).
            if fsb_score.is_nan() {
                continue;
            }

            let is_better = best
                .map(|(_, _, best_fsb, best_babsr)| {
                    fsb_score > best_fsb + 1e-6
                        || ((fsb_score - best_fsb).abs() <= 1e-6
                            && !babsr_score.is_nan()
                            && (best_babsr.is_nan() || babsr_score > best_babsr))
                })
                .unwrap_or(true);
            if is_better {
                best = Some((layer_idx, neuron_idx, fsb_score, babsr_score));
            }
        }

        if let Some((layer_idx, neuron_idx, fsb_score, babsr_score)) = best {
            trace!(
                "KFSB selected neuron layer={}, idx={}, fsb_score={:.4}, babsr_score={:.4} (eval={}/{})",
                layer_idx,
                neuron_idx,
                fsb_score,
                babsr_score,
                eval_candidates.len(),
                scored.len()
            );
            return Ok(Some((layer_idx, neuron_idx, fsb_score)));
        }

        // Fallback: best BaBSR if evaluation failed for all candidates.
        Ok(babsr_ranked
            .first()
            .map(|(l, n, score, _)| (*l, *n, *score)))
    }

    /// kFSB: k-Filtered Smart Branching with configurable reduce operation.
    ///
    /// Matches alpha-beta-CROWN's kfsb heuristic:
    /// - Uses both BaBSR (alpha) score and intercept (backup) score
    /// - Evaluates top-k candidates from each ranking
    /// - Combines branch scores using configurable reduce_op (min/max/mean)
    /// - Falls back to random selection when candidates exhausted
    fn select_kfsb_neuron(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domain: &BabDomain,
        intercept_only: bool,
    ) -> Result<Option<(usize, usize, f32)>> {
        let k = self.config.fsb_candidates;
        if k == 0 {
            return self.select_babsr_neuron(network, domain);
        }

        // Collect unstable, unconstrained neurons with their scores.
        let mut candidates: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
        // (layer_idx, neuron_idx, lb, ub, intercept)

        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !is_zero_threshold_binary_activation(layer) {
                continue;
            }
            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];
            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    // Triangle relaxation intercept: measures looseness
                    let intercept = relu_intercept_score(l, u);
                    candidates.push((layer_idx, neuron_idx, l, u, intercept));
                }
            }
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        let mut scored: Vec<(usize, usize, f32, f32)> = Vec::with_capacity(candidates.len());
        // (layer_idx, neuron_idx, babsr_score, intercept_score)

        if intercept_only {
            let intercept_only_scores =
                self.compute_babsr_intercept_only_scores(network, domain)?;
            for (layer_idx, neuron_idx, lb, ub, _) in &candidates {
                let main_score = intercept_only_scores
                    .get(&(*layer_idx, *neuron_idx))
                    .copied()
                    .unwrap_or_else(|| {
                        debug!(
                            layer = layer_idx,
                            neuron = neuron_idx,
                            lower = lb,
                            upper = ub,
                            "kFSB sequential intercept-only: no score for neuron, using 0.0 fallback"
                        );
                        0.0
                    });
                scored.push((*layer_idx, *neuron_idx, main_score, 0.0));
            }
        } else {
            let babsr_scores =
                self.compute_babsr_scores(network, domain, self.config.kfsb_reduce_op)?;
            for (layer_idx, neuron_idx, lb, ub, _) in &candidates {
                let score_parts = babsr_scores
                    .get(&(*layer_idx, *neuron_idx))
                    .copied()
                    .unwrap_or_else(|| {
                        debug!(
                            layer = layer_idx,
                            neuron = neuron_idx,
                            lower = lb,
                            upper = ub,
                            "kFSB sequential: no score parts for neuron, using 0.0 fallback"
                        );
                        BabsrScoreParts::default()
                    });
                scored.push((
                    *layer_idx,
                    *neuron_idx,
                    score_parts.main_score,
                    score_parts.backup_score,
                ));
            }
        }

        // Get top-k by BaBSR/main score (higher is better, NaN last — #2995)
        let mut main_ranked = scored.clone();
        main_ranked.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.2, &b.2));

        // Merge candidates from both rankings, deduplicate
        let mut eval_candidates: Vec<(usize, usize, f32, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        for &(layer_idx, neuron_idx, main_score, intercept_score) in main_ranked.iter().take(k) {
            if seen.insert((layer_idx, neuron_idx)) {
                eval_candidates.push((layer_idx, neuron_idx, main_score, intercept_score));
            }
        }

        if !intercept_only {
            // Backup ranking keeps the #2539 ascending-order contract for regular kFSB.
            let mut intercept_ranked = scored.clone();
            intercept_ranked.sort_by(|a, b| crate::cmp_utils::nan_propagating_cmp(&a.3, &b.3));

            for &(layer_idx, neuron_idx, main_score, intercept_score) in
                intercept_ranked.iter().take(k)
            {
                if seen.insert((layer_idx, neuron_idx)) {
                    eval_candidates.push((layer_idx, neuron_idx, main_score, intercept_score));
                }
            }
        }

        // Evaluate candidates with configurable reduce operation
        let reduce_op = self.config.kfsb_reduce_op;
        let mut best: Option<(usize, usize, f32, f32)> = None;

        for &(layer_idx, neuron_idx, main_score, _) in &eval_candidates {
            let active = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, true,
            )?;
            let inactive = self.estimate_child_bounds_after_split(
                network, input, domain, layer_idx, neuron_idx, false,
            )?;

            // Extract the relevant bound (lower for safety, upper for adversarial)
            let active_val = self.config.child_bound_value(active);
            let inactive_val = self.config.child_bound_value(inactive);

            if active_val == f32::NEG_INFINITY && inactive_val == f32::NEG_INFINITY {
                continue;
            }

            // Apply reduce operation
            let kfsb_score = match reduce_op {
                KfsbReduceOp::Min => active_val.min(inactive_val), // Conservative (default)
                KfsbReduceOp::Max => active_val.max(inactive_val), // Optimistic
                KfsbReduceOp::Mean => f32::midpoint(active_val, inactive_val), // Balanced
            };

            // Skip NaN-scored candidates — corrupt scores must not win (#2588).
            if kfsb_score.is_nan() {
                continue;
            }

            let is_better = best
                .map(|(_, _, best_score, best_main)| {
                    kfsb_score > best_score + 1e-6
                        || ((kfsb_score - best_score).abs() <= 1e-6
                            && !main_score.is_nan()
                            && (best_main.is_nan() || main_score > best_main))
                })
                .unwrap_or(true);

            if is_better {
                best = Some((layer_idx, neuron_idx, kfsb_score, main_score));
            }
        }

        if let Some((layer_idx, neuron_idx, kfsb_score, main_score)) = best {
            trace!(
                "kFSB selected neuron layer={}, idx={}, score={:.4}, main={:.4} (eval={}/{}, reduce={:?})",
                layer_idx,
                neuron_idx,
                kfsb_score,
                main_score,
                eval_candidates.len(),
                scored.len(),
                reduce_op
            );
            return Ok(Some((layer_idx, neuron_idx, kfsb_score)));
        }

        // Random fallback (per alpha-beta-CROWN kfsb.py line 210-226)
        // When all evaluations fail, pick randomly from unstable neurons
        if !candidates.is_empty() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            domain.depth().hash(&mut hasher);
            let idx = (hasher.finish() as usize) % candidates.len();
            let (layer_idx, neuron_idx, _, _, _) = candidates[idx];
            trace!(
                "kFSB fallback: random selection layer={}, idx={} (from {} candidates)",
                layer_idx,
                neuron_idx,
                candidates.len()
            );
            return Ok(Some((layer_idx, neuron_idx, 0.0)));
        }

        Ok(None)
    }

    /// Estimate child bounds after applying a single ReLU split constraint, without joint optimization.
    ///
    /// This is intended for branching heuristics (FSB) where we want a cheap estimate.
    pub(in crate::beta_crown::engine) fn estimate_child_bounds_after_split(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        layer_idx: usize,
        neuron_idx: usize,
        is_active: bool,
    ) -> Result<Option<(f32, f32)>> {
        let constraint = NeuronConstraint {
            layer_idx,
            neuron_idx,
            is_active,
            score: 0.0,
        };
        let new_history = parent.history.with_constraint(constraint);

        let mut new_layer_bounds = parent.layer_bounds.clone();

        // Apply constraint to pre-activation bounds (same tightening as create_child_domain).
        if layer_idx > 0 && layer_idx <= new_layer_bounds.len() {
            let pre_bounds = &new_layer_bounds[layer_idx - 1];
            let lower = pre_bounds.lower().clone();
            let upper = pre_bounds.upper().clone();

            let shape = lower.shape().to_vec();
            let lower_len = lower.len();
            let upper_len = upper.len();
            let mut lower_flat = lower
                .into_shape_clone(ndarray::IxDyn(&[lower_len]))
                .map_err(|_| ny_core::NyError::ShapeMismatch {
                    expected: vec![lower_len],
                    got: shape.clone(),
                })?;
            let mut upper_flat = upper
                .into_shape_clone(ndarray::IxDyn(&[upper_len]))
                .map_err(|_| ny_core::NyError::ShapeMismatch {
                    expected: vec![upper_len],
                    got: shape.clone(),
                })?;

            if is_active {
                // NaN-safe: propagate NaN instead of silently clamping to 0.0 (#2643)
                lower_flat[[neuron_idx]] = nan_propagating_max(lower_flat[[neuron_idx]], 0.0);
            } else {
                upper_flat[[neuron_idx]] = nan_propagating_min(upper_flat[[neuron_idx]], 0.0);
            }

            if lower_flat[[neuron_idx]] > upper_flat[[neuron_idx]] {
                return Ok(None);
            }

            let lower_new = lower_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|_| ny_core::NyError::ShapeMismatch {
                    expected: shape.clone(),
                    got: vec![lower_len],
                })?;
            let upper_new = upper_flat
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|_| ny_core::NyError::ShapeMismatch {
                    expected: shape.clone(),
                    got: vec![upper_len],
                })?;
            new_layer_bounds[layer_idx - 1] = Arc::new(BoundedTensor::new(lower_new, upper_new)?);
        }

        let beta_state = BetaState::from_history(&new_history)?;
        let alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &new_history,
            )
        } else {
            DomainAlphaState::empty()
        };

        let empty_cuts = CutPool::new(0);
        // Note: FSB uses CPU-only for cheap estimates during branching heuristics
        let bounds = self.compute_bounds_with_alpha_beta(
            network,
            input,
            &new_history,
            &new_layer_bounds,
            &beta_state,
            &alpha_state,
            &empty_cuts,
            None, // CPU-only for branching heuristics
        )?;

        Ok(Some((bounds.lower_scalar(), bounds.upper_scalar())))
    }

    /// Compute CROWN coefficients (sensitivities) for all neurons.
    ///
    /// Returns a map from (layer_idx, neuron_idx) to the sum of absolute output
    /// sensitivities |lA[output, neuron]|.
    #[cfg(test)]
    pub(in crate::beta_crown::engine) fn compute_crown_coefficients(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<std::collections::HashMap<(usize, usize), f32>> {
        let mut coeffs = std::collections::HashMap::new();

        if network.layers.is_empty() {
            return Ok(coeffs);
        }

        // Get output dimension from last layer bounds
        let output_dim = domain.layer_bounds.last().map(|b| b.len()).ok_or_else(|| {
            ny_core::NyError::InternalError("compute_crown_coefficients: layer_bounds empty".into())
        })?;

        // Start with identity at output layer
        let mut current_coeffs = Array2::<f32>::eye(output_dim);

        // Backward pass through layers (output to input)
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Get pre-activation bounds as reference (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                // Use first layer bounds if available, else this is special case
                if !domain.layer_bounds.is_empty() {
                    domain.layer_bounds[0].as_ref()
                } else {
                    continue;
                }
            } else if layer_idx <= domain.layer_bounds.len() {
                domain.layer_bounds[layer_idx - 1].as_ref()
            } else {
                continue;
            };

            match layer {
                Layer::Linear(linear) => {
                    // Linear backward: coeffs = coeffs @ W
                    current_coeffs = current_coeffs.dot(linear.weight());
                }
                Layer::ReLU(_) => {
                    // For ReLU, record coefficients for all neurons
                    let flat = pre_bounds.flatten();
                    let num_neurons = current_coeffs.ncols().min(flat.len());

                    for neuron_idx in 0..num_neurons {
                        // Sum of absolute coefficients across all outputs
                        let sum_abs_coeff: f32 = current_coeffs
                            .column(neuron_idx)
                            .iter()
                            .map(|c| c.abs())
                            .sum();
                        coeffs.insert((layer_idx, neuron_idx), sum_abs_coeff);
                    }

                    // Apply ReLU relaxation slopes to coefficients
                    let mut new_coeffs =
                        Array2::<f32>::zeros((current_coeffs.nrows(), num_neurons));
                    for neuron_idx in 0..num_neurons {
                        let l = flat.lower()[[neuron_idx]];
                        let u = flat.upper()[[neuron_idx]];

                        let slope = relu_upper_slope(l, u);

                        for output_idx in 0..current_coeffs.nrows() {
                            new_coeffs[[output_idx, neuron_idx]] =
                                current_coeffs[[output_idx, neuron_idx]] * slope;
                        }
                    }
                    current_coeffs = new_coeffs;
                }
                Layer::Sign(_) => {
                    // For Sign, record coefficients (same structure as ReLU) and
                    // apply Sign fixed-CROWN proxy slopes for backward propagation.
                    // Part of #3769: Sign neurons are zero-threshold branching candidates.
                    let flat = pre_bounds.flatten();
                    let num_neurons = current_coeffs.ncols().min(flat.len());

                    for neuron_idx in 0..num_neurons {
                        let sum_abs_coeff: f32 = current_coeffs
                            .column(neuron_idx)
                            .iter()
                            .map(|c| c.abs())
                            .sum();
                        coeffs.insert((layer_idx, neuron_idx), sum_abs_coeff);
                    }

                    let mut new_coeffs =
                        Array2::<f32>::zeros((current_coeffs.nrows(), num_neurons));
                    for neuron_idx in 0..num_neurons {
                        let l = flat.lower()[[neuron_idx]];
                        let u = flat.upper()[[neuron_idx]];
                        let slope = sign_fixed_crown_proxy_slope(l, u);
                        for output_idx in 0..current_coeffs.nrows() {
                            new_coeffs[[output_idx, neuron_idx]] =
                                current_coeffs[[output_idx, neuron_idx]] * slope;
                        }
                    }
                    current_coeffs = new_coeffs;
                }
                other => {
                    // Unsupported layer type — stop backward propagation here.
                    // Coefficients computed for layers above this point are still
                    // valid branching scores. Continuing with identity passthrough
                    // would produce incorrect scores for layers below, potentially
                    // causing suboptimal branching decisions (#2271).
                    tracing::debug!(
                        "compute_crown_coefficients: stopping at unsupported layer {} ({})",
                        layer_idx,
                        other.layer_type(),
                    );
                    break;
                }
            }
        }

        Ok(coeffs)
    }

    pub(in crate::beta_crown::engine) fn compute_babsr_scores(
        &self,
        network: &Network,
        domain: &BabDomain,
        reduce_op: KfsbReduceOp,
    ) -> Result<std::collections::HashMap<(usize, usize), BabsrScoreParts>> {
        let mut scores = std::collections::HashMap::new();

        if network.layers.is_empty() {
            return Ok(scores);
        }

        let output_dim = domain.layer_bounds.last().map(|b| b.len()).ok_or_else(|| {
            ny_core::NyError::InternalError("compute_babsr_scores: layer_bounds empty".into())
        })?;
        let mut current_coeffs = Array2::<f32>::eye(output_dim);

        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                if !domain.layer_bounds.is_empty() {
                    domain.layer_bounds[0].as_ref()
                } else {
                    continue;
                }
            } else if layer_idx <= domain.layer_bounds.len() {
                domain.layer_bounds[layer_idx - 1].as_ref()
            } else {
                continue;
            };

            match layer {
                Layer::Linear(linear) => {
                    current_coeffs = current_coeffs.dot(linear.weight());
                }
                Layer::ReLU(_) | Layer::Sign(_) => {
                    let flat = pre_bounds.flatten();
                    let num_neurons = current_coeffs.ncols().min(flat.len());
                    let bias_flat = self
                        .sequential_preact_bias(network, layer_idx, pre_bounds.shape())
                        .and_then(|bias| bias.into_shape_with_order((flat.len(),)).ok());
                    if bias_flat.is_none() && layer_idx > 0 {
                        debug!(
                            activation_layer = layer_idx,
                            producer_layer = layer_idx - 1,
                            producer_type = network.layers[layer_idx - 1].layer_type(),
                            "BaBSR sequential: unrecoverable producer bias, using 0.0 fallback"
                        );
                    }

                    for neuron_idx in 0..num_neurons {
                        let lower = flat.lower()[[neuron_idx]];
                        let upper = flat.upper()[[neuron_idx]];
                        let bias = bias_flat
                            .as_ref()
                            .map(|flat_bias| flat_bias[neuron_idx])
                            .unwrap_or(0.0);
                        scores.insert(
                            (layer_idx, neuron_idx),
                            compute_babsr_score_parts(
                                current_coeffs.column(neuron_idx),
                                lower,
                                upper,
                                bias,
                                reduce_op,
                            ),
                        );
                    }

                    let is_sign = matches!(layer, Layer::Sign(_));
                    let mut new_coeffs =
                        Array2::<f32>::zeros((current_coeffs.nrows(), num_neurons));
                    for neuron_idx in 0..num_neurons {
                        let l = flat.lower()[[neuron_idx]];
                        let u = flat.upper()[[neuron_idx]];
                        let slope = if is_sign {
                            sign_fixed_crown_proxy_slope(l, u)
                        } else {
                            relu_upper_slope(l, u)
                        };
                        for output_idx in 0..current_coeffs.nrows() {
                            new_coeffs[[output_idx, neuron_idx]] =
                                current_coeffs[[output_idx, neuron_idx]] * slope;
                        }
                    }
                    current_coeffs = new_coeffs;
                }
                other => {
                    tracing::debug!(
                        "compute_babsr_scores: stopping at unsupported layer {} ({})",
                        layer_idx,
                        other.layer_type(),
                    );
                    break;
                }
            }
        }

        Ok(scores)
    }

    pub(in crate::beta_crown::engine) fn compute_babsr_intercept_only_scores(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<std::collections::HashMap<(usize, usize), f32>> {
        let mut scores = std::collections::HashMap::new();

        if network.layers.is_empty() {
            return Ok(scores);
        }

        let output_dim = domain.layer_bounds.last().map(|b| b.len()).ok_or_else(|| {
            ny_core::NyError::InternalError(
                "compute_babsr_intercept_only_scores: layer_bounds empty".into(),
            )
        })?;
        let mut current_coeffs = Array2::<f32>::eye(output_dim);

        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                if !domain.layer_bounds.is_empty() {
                    domain.layer_bounds[0].as_ref()
                } else {
                    continue;
                }
            } else if layer_idx <= domain.layer_bounds.len() {
                domain.layer_bounds[layer_idx - 1].as_ref()
            } else {
                continue;
            };

            match layer {
                Layer::Linear(linear) => {
                    current_coeffs = current_coeffs.dot(linear.weight());
                }
                Layer::ReLU(_) | Layer::Sign(_) => {
                    let flat = pre_bounds.flatten();
                    let num_neurons = current_coeffs.ncols().min(flat.len());
                    for neuron_idx in 0..num_neurons {
                        scores.insert(
                            (layer_idx, neuron_idx),
                            compute_babsr_intercept_only_score(
                                current_coeffs.column(neuron_idx),
                                flat.lower()[[neuron_idx]],
                                flat.upper()[[neuron_idx]],
                            ),
                        );
                    }

                    let is_sign = matches!(layer, Layer::Sign(_));
                    let mut new_coeffs =
                        Array2::<f32>::zeros((current_coeffs.nrows(), num_neurons));
                    for neuron_idx in 0..num_neurons {
                        let l = flat.lower()[[neuron_idx]];
                        let u = flat.upper()[[neuron_idx]];
                        let slope = if is_sign {
                            sign_fixed_crown_proxy_slope(l, u)
                        } else {
                            relu_upper_slope(l, u)
                        };
                        for output_idx in 0..current_coeffs.nrows() {
                            new_coeffs[[output_idx, neuron_idx]] =
                                current_coeffs[[output_idx, neuron_idx]] * slope;
                        }
                    }
                    current_coeffs = new_coeffs;
                }
                other => {
                    tracing::debug!(
                        "compute_babsr_intercept_only_scores: stopping at unsupported layer {} ({})",
                        layer_idx,
                        other.layer_type(),
                    );
                    break;
                }
            }
        }

        Ok(scores)
    }

    /// Select neurons in sequential order.
    fn select_sequential_neuron(
        &self,
        network: &Network,
        domain: &BabDomain,
    ) -> Result<Option<(usize, usize, f32)>> {
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if !is_zero_threshold_binary_activation(layer) {
                continue;
            }

            if layer_idx == 0 || layer_idx > domain.layer_bounds.len() {
                continue;
            }
            let pre_bounds = &domain.layer_bounds[layer_idx - 1];

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];

                // Check if unstable and not constrained
                if l < 0.0
                    && u > 0.0
                    && domain
                        .history
                        .is_constrained(layer_idx, neuron_idx)
                        .is_none()
                {
                    return Ok(Some((layer_idx, neuron_idx, 0.0)));
                }
            }
        }
        Ok(None)
    }
}
