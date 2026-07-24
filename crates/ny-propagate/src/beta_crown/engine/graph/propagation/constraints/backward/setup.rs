// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Setup and warm-start seeding for constrained backward CROWN.
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::GraphAlphaCrownIntermediate;
use crate::LinearBounds;

use super::super::super::super::super::BetaCrownVerifier;
use super::{BackwardMode, BackwardParams, ConstrainedBackwardSetup, ConstrainedBackwardState};

impl BetaCrownVerifier {
    pub(super) fn initialize_constrained_backward<'graph, 'mode>(
        &self,
        params: &'graph BackwardParams<'graph>,
        mode: &'mode BackwardMode,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    ) -> Result<ConstrainedBackwardSetup<'graph, 'mode>> {
        let intermediate = match mode {
            BackwardMode::StoringIntermediates { .. } => Some(GraphAlphaCrownIntermediate::new()),
            BackwardMode::Standard => None,
        };
        let mode_lookups = match mode {
            BackwardMode::StoringIntermediates { lookups } => Some(lookups.as_ref()),
            BackwardMode::Standard => None,
        };

        let output_name = params.graph.output_name();
        let output_node = if output_name.is_empty() {
            params
                .exec_order
                .last()
                .map(String::as_str)
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            output_name
        };
        let ibp_output = bounds_cache.get(output_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        // Raw network output width — the *input* side of the spec/objective map,
        // used only to validate the seed's column count.
        let raw_output_dim = ibp_output.len();
        let input_dim = params.constrained_input.len();

        // Seed selection: spec_matrix (multi-row) > objective (single-row) > identity.
        // spec_matrix support added for batched multi-objective CROWN (#4306).
        //
        // `output_dim` (the SETUP field) is the number of backward output ROWS —
        // i.e. the seed's row count (num_specs / 1 / raw). This is the dimension of
        // the accumulated bound and the zero-coefficient bias wrapper. It differs
        // from `raw_output_dim` whenever a spec maps N raw outputs to M≠N specs
        // (e.g. 200-class TinyImageNet → 199 robustness comparisons). Using
        // raw_output_dim here was an off-by-N bug that panicked the NETWORK_INPUT
        // bias accumulation on residual conv nets during child-domain β-CROWN.
        let (output_shape, initial_lb, output_dim) = if let Some(spec) = params.spec_matrix {
            let num_specs = spec.nrows();
            let spec_dim = spec.ncols();
            if spec_dim != raw_output_dim {
                return Err(NyError::shape_mismatch(
                    vec![spec_dim],
                    vec![raw_output_dim],
                ));
            }
            let bias = Array1::zeros(num_specs);
            (
                vec![num_specs],
                LinearBounds::new(spec.clone(), bias.clone(), spec.clone(), bias)?,
                num_specs,
            )
        } else if let Some(objective) = params.objective {
            if objective.len() != raw_output_dim {
                return Err(NyError::shape_mismatch(
                    vec![objective.len()],
                    vec![raw_output_dim],
                ));
            }
            let a = Array2::from_shape_vec((1, raw_output_dim), objective.to_vec()).map_err(
                |error| {
                    NyError::InvalidSpec(format!("Failed to build objective coefficients: {error}"))
                },
            )?;
            (
                vec![1usize],
                // Migrated from from_parts_unchecked: objective is caller-provided
                // f32 data that should be validated. See #3438.
                LinearBounds::new(a.clone(), Array1::zeros(1), a, Array1::zeros(1))?,
                1,
            )
        } else {
            (
                ibp_output.shape().to_vec(),
                LinearBounds::identity(raw_output_dim),
                raw_output_dim,
            )
        };

        let mut node_crown_bounds = crate::network::CrownMergeAccumulator::new();
        let warm_started = if self.config.enable_la_warm_start {
            match (params.seed_cache, params.context.history.last_branch_node()) {
                (Some(cache), Some(branch_node)) => {
                    if let Some(warm_lb) = cache.linear_bounds(branch_node) {
                        node_crown_bounds.insert(
                            branch_node.to_string(),
                            params.patches_policy.seed_bounds(warm_lb),
                        );
                        debug!(
                            branch_node,
                            "Constrained CROWN lA warm-start: seeded backward pass at last branch node"
                        );
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        };
        if !warm_started {
            node_crown_bounds.insert(
                output_node.to_string(),
                params.patches_policy.seed_bounds(initial_lb),
            );
        }

        Ok(ConstrainedBackwardSetup {
            output_node,
            output_shape,
            output_dim,
            input_dim,
            mode_lookups,
            state: ConstrainedBackwardState {
                node_crown_bounds,
                intermediate,
                captured_linear_bounds: params.capture_linear_bounds.then(HashMap::new),
                input_accumulated: false,
            },
        })
    }
}
