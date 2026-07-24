// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential and parallel domain-processing entrypoints for BaB.

use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use tracing::debug;

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::domain::{BabDomain, DomainProcessingConfig};
use crate::faer_parallelism::RayonTaskGuard;

use super::super::input_split::InputSplitChildren;
use super::super::BetaCrownVerifier;
use super::DomainProcessingResult;

impl BetaCrownVerifier {
    /// Process a single domain sequentially, returning child domains and failure info.
    ///
    /// #1865: Now returns `DomainProcessingResult` so the caller can detect when
    /// child creation fails and avoid falsely claiming Verified.
    // Justification: Domain processing requires network, input, domain, threshold,
    // cut pool, engine, and deadline as independent parameters. The sequential path
    // takes different parameters than the parallel path (no DomainProcessingConfig).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn process_domain_sequential(
        &self,
        network: &crate::Network,
        input: &BoundedTensor,
        domain: &BabDomain,
        threshold: f32,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> DomainProcessingResult {
        // Check if using input splitting (Conv layers now support ReLU splitting via CROWN backward)
        if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) {
            return match self
                .create_input_split_children(network, input, domain, threshold, deadline, engine)
            {
                Ok(InputSplitChildren::Split(children)) => DomainProcessingResult {
                    children,
                    had_propagation_failure: false,
                    had_no_branch: false,
                    had_unsplittable: false,
                },
                // No splittable input dimension — the domain stays unexplored
                // and must be flagged, or the loop would drain the queue and
                // falsely claim Verified.
                Ok(InputSplitChildren::Unsplittable) => DomainProcessingResult {
                    children: Vec::new(),
                    had_propagation_failure: false,
                    had_no_branch: false,
                    had_unsplittable: true,
                },
                Err(e) => {
                    // #1865: input split failed — sub-region unexplored.
                    debug!("[#1865] input split failed: {e}");
                    DomainProcessingResult {
                        children: Vec::new(),
                        had_propagation_failure: true,
                        had_no_branch: false,
                        had_unsplittable: false,
                    }
                }
            };
        }

        let mut children = Vec::with_capacity(2);
        let mut had_propagation_failure = false;

        // Select neuron to split
        let split_neuron = match self.select_split_neuron(network, input, domain) {
            Ok(Some(neuron)) => neuron,
            Ok(None) => {
                debug!("No unstable neurons to split, domain unresolved");
                return DomainProcessingResult {
                    children,
                    had_propagation_failure: false,
                    had_no_branch: true,
                    had_unsplittable: false,
                };
            }
            Err(e) => {
                // #1865: neuron selection failed — sub-region unexplored.
                debug!("[#1865] select_split_neuron failed: {e}");
                return DomainProcessingResult {
                    children,
                    had_propagation_failure: true,
                    had_no_branch: false,
                    had_unsplittable: false,
                };
            }
        };
        let (layer_idx, neuron_idx, score) = split_neuron;

        // Create active child
        match self.create_child_domain(
            network, input, domain, layer_idx, neuron_idx, true, score, threshold, cut_pool, engine,
        ) {
            Ok(Some(child)) => children.push(child),
            Ok(None) => { /* infeasible — correctly pruned */ }
            Err(e) => {
                // #1865: active child creation failed — sub-region unexplored.
                debug!("[#1865] active child creation failed: {e}");
                had_propagation_failure = true;
            }
        }

        // Create inactive child
        match self.create_child_domain(
            network, input, domain, layer_idx, neuron_idx, false, score, threshold, cut_pool,
            engine,
        ) {
            Ok(Some(child)) => children.push(child),
            Ok(None) => { /* infeasible — correctly pruned */ }
            Err(e) => {
                // #1865: inactive child creation failed — sub-region unexplored.
                debug!("[#1865] inactive child creation failed: {e}");
                had_propagation_failure = true;
            }
        }

        DomainProcessingResult {
            children,
            had_propagation_failure,
            had_no_branch: false,
            had_unsplittable: false,
        }
    }

    /// Process a single domain with parallel child creation, returning child domains and failure info.
    ///
    /// #1865: Now returns `DomainProcessingResult` so the caller can detect when
    /// child creation fails and avoid falsely claiming Verified.
    ///
    /// Note: When cuts are enabled, parallel child creation is disabled because
    /// lambda optimization modifies the shared cut pool. This is a known limitation;
    /// future work could use per-domain lambda copies.
    pub(crate) fn process_domain_parallel(
        &self,
        network: &crate::Network,
        input: &BoundedTensor,
        domain: &BabDomain,
        config: &DomainProcessingConfig,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> DomainProcessingResult {
        // Check if using input splitting (Conv layers now support ReLU splitting via CROWN backward)
        if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) {
            return match self.create_input_split_children(
                network,
                input,
                domain,
                config.threshold,
                config.deadline,
                engine,
            ) {
                Ok(InputSplitChildren::Split(children)) => DomainProcessingResult {
                    children,
                    had_propagation_failure: false,
                    had_no_branch: false,
                    had_unsplittable: false,
                },
                // No splittable input dimension — the domain stays unexplored
                // and must be flagged, or the loop would drain the queue and
                // falsely claim Verified.
                Ok(InputSplitChildren::Unsplittable) => DomainProcessingResult {
                    children: Vec::new(),
                    had_propagation_failure: false,
                    had_no_branch: false,
                    had_unsplittable: true,
                },
                Err(e) => {
                    // #1865: input split failed — sub-region unexplored.
                    debug!("[#1865] input split failed (parallel): {e}");
                    DomainProcessingResult {
                        children: Vec::new(),
                        had_propagation_failure: true,
                        had_no_branch: false,
                        had_unsplittable: false,
                    }
                }
            };
        }

        // Select neuron to split
        let split_neuron = match self.select_split_neuron(network, input, domain) {
            Ok(Some(neuron)) => neuron,
            Ok(None) => {
                debug!("No unstable neurons to split, domain unresolved");
                return DomainProcessingResult {
                    children: Vec::new(),
                    had_propagation_failure: false,
                    had_no_branch: true,
                    had_unsplittable: false,
                };
            }
            Err(e) => {
                // #1865: neuron selection failed — sub-region unexplored.
                debug!("[#1865] select_split_neuron failed (parallel): {e}");
                return DomainProcessingResult {
                    children: Vec::new(),
                    had_propagation_failure: true,
                    had_no_branch: false,
                    had_unsplittable: false,
                };
            }
        };
        let (layer_idx, neuron_idx, score) = split_neuron;

        // When cuts are enabled, use sequential processing to avoid concurrent mutation
        // of lambda values in the cut pool. This is a known limitation.
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;

        if config.use_parallel_children && !has_cuts {
            // Create both children in parallel using rayon (no cuts).
            // GemmEngine is Sync+Send, so GPU acceleration is safe to use here.
            // #1865: Track propagation failures from parallel child creation.
            let results: Vec<Result<Option<BabDomain>>> = [true, false]
                .par_iter()
                .map(|&is_active| {
                    let _rayon_task_guard = RayonTaskGuard::new();
                    let mut empty_pool = CutPool::new(0); // No cuts for parallel path
                    self.create_child_domain(
                        network,
                        input,
                        domain,
                        layer_idx,
                        neuron_idx,
                        is_active,
                        score,
                        config.threshold,
                        &mut empty_pool,
                        engine, // GPU engine usable in rayon parallel path
                    )
                })
                .collect();

            let mut children = Vec::with_capacity(2);
            let mut had_propagation_failure = false;
            for result in results {
                match result {
                    Ok(Some(child)) => children.push(child),
                    Ok(None) => { /* infeasible — correctly pruned */ }
                    Err(e) => {
                        // #1865: child creation failed — sub-region unexplored.
                        debug!("[#1865] parallel child creation failed: {e}");
                        had_propagation_failure = true;
                    }
                }
            }

            DomainProcessingResult {
                children,
                had_propagation_failure,
                had_no_branch: false,
                had_unsplittable: false,
            }
        } else {
            // Sequential child creation (required when cuts are enabled)
            // GPU engine can be used here for acceleration
            self.process_domain_sequential(
                network,
                input,
                domain,
                config.threshold,
                cut_pool,
                engine,
                config.deadline,
            )
        }
    }
}
