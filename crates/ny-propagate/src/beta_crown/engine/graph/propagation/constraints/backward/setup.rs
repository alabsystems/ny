// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Setup and warm-start seeding for constrained backward CROWN.
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::GraphAlphaCrownIntermediate;
use crate::LinearBounds;

use super::super::super::super::super::BetaCrownVerifier;
use super::{BackwardMode, BackwardParams, ConstrainedBackwardSetup, ConstrainedBackwardState};

fn finite_constrained_seed(
    rows: usize,
    cols: usize,
    retained_source_bytes: usize,
    mut coefficient: impl FnMut(usize, usize) -> f32,
    deadline: std::time::Instant,
) -> Result<LinearBounds> {
    const SITE: &str = "constrained backward finite seed";
    let mut poll = crate::bounds::patches::PatchesMaterializationDeadline::new(Some(deadline));
    poll.checkpoint(SITE)?;
    let coefficient_elements = rows.saturating_mul(cols);
    let destination_elements = coefficient_elements
        .checked_mul(2)
        .and_then(|elements| {
            rows.checked_mul(2)
                .and_then(|bias| elements.checked_add(bias))
        })
        .unwrap_or(usize::MAX);
    let nominal_required_bytes =
        retained_source_bytes.saturating_add(destination_elements.saturating_mul(size_of::<f32>()));
    let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    let memory_error = |required_bytes| NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site: SITE,
    };
    if nominal_required_bytes > budget_bytes {
        return Err(memory_error(nominal_required_bytes));
    }
    let reconcile = |allocated_capacity: usize, remaining_elements: usize| -> Result<()> {
        let required_bytes = retained_source_bytes.saturating_add(
            allocated_capacity
                .saturating_add(remaining_elements)
                .saturating_mul(size_of::<f32>()),
        );
        if required_bytes > budget_bytes {
            Err(memory_error(required_bytes))
        } else {
            Ok(())
        }
    };

    let mut lower_a = Vec::new();
    lower_a
        .try_reserve_exact(coefficient_elements)
        .map_err(|_| memory_error(nominal_required_bytes))?;
    reconcile(
        lower_a.capacity(),
        coefficient_elements.saturating_add(rows.saturating_mul(2)),
    )?;
    let mut upper_a = Vec::new();
    upper_a
        .try_reserve_exact(coefficient_elements)
        .map_err(|_| memory_error(nominal_required_bytes))?;
    reconcile(
        lower_a.capacity().saturating_add(upper_a.capacity()),
        rows.saturating_mul(2),
    )?;
    for row in 0..rows {
        for col in 0..cols {
            let value = coefficient(row, col);
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "constrained backward finite seed contains NaN or Inf".into(),
                ));
            }
            lower_a.push(value);
            upper_a.push(value);
            poll.work(2, SITE)?;
        }
    }

    let mut lower_b = Vec::new();
    lower_b
        .try_reserve_exact(rows)
        .map_err(|_| memory_error(nominal_required_bytes))?;
    reconcile(
        lower_a
            .capacity()
            .saturating_add(upper_a.capacity())
            .saturating_add(lower_b.capacity()),
        rows,
    )?;
    let mut upper_b = Vec::new();
    upper_b
        .try_reserve_exact(rows)
        .map_err(|_| memory_error(nominal_required_bytes))?;
    reconcile(
        lower_a
            .capacity()
            .saturating_add(upper_a.capacity())
            .saturating_add(lower_b.capacity())
            .saturating_add(upper_b.capacity()),
        0,
    )?;
    for _ in 0..rows {
        lower_b.push(0.0);
        upper_b.push(0.0);
        poll.work(2, SITE)?;
    }
    poll.checkpoint(SITE)?;

    let lower_a = Array2::from_shape_vec((rows, cols), lower_a).map_err(|error| {
        NyError::InternalError(format!("{SITE}: lower seed shape failed: {error}"))
    })?;
    let upper_a = Array2::from_shape_vec((rows, cols), upper_a).map_err(|error| {
        NyError::InternalError(format!("{SITE}: upper seed shape failed: {error}"))
    })?;
    LinearBounds::from_prevalidated_parts(
        lower_a,
        Array1::from_vec(lower_b),
        upper_a,
        Array1::from_vec(upper_b),
    )
}

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
            let bounds = if let Some(deadline) = params.deadline {
                finite_constrained_seed(
                    num_specs,
                    spec_dim,
                    spec.len().saturating_mul(size_of::<f32>()),
                    |row, col| spec[[row, col]],
                    deadline,
                )?
            } else {
                let bias = Array1::zeros(num_specs);
                LinearBounds::new(spec.clone(), bias.clone(), spec.clone(), bias)?
            };
            (vec![num_specs], bounds, num_specs)
        } else if let Some(objective) = params.objective {
            if objective.len() != raw_output_dim {
                return Err(NyError::shape_mismatch(
                    vec![objective.len()],
                    vec![raw_output_dim],
                ));
            }
            let bounds = if let Some(deadline) = params.deadline {
                finite_constrained_seed(
                    1,
                    raw_output_dim,
                    objective.len().saturating_mul(size_of::<f32>()),
                    |_, col| objective[col],
                    deadline,
                )?
            } else {
                let a = Array2::from_shape_vec((1, raw_output_dim), objective.to_vec()).map_err(
                    |error| {
                        NyError::InvalidSpec(format!(
                            "Failed to build objective coefficients: {error}"
                        ))
                    },
                )?;
                LinearBounds::new(a.clone(), Array1::zeros(1), a, Array1::zeros(1))?
            };
            (
                vec![1usize],
                // Caller-provided objective coefficients are validated on both
                // the legacy and cooperative finite construction paths.
                bounds,
                1,
            )
        } else {
            let bounds =
                LinearBounds::try_identity_with_deadline(raw_output_dim, params.deadline, 0)?;
            (ibp_output.shape().to_vec(), bounds, raw_output_dim)
        };

        let mut node_crown_bounds = crate::network::CrownMergeAccumulator::new();
        // A verifier timeout is ordinary Dense constrained-CROWN authority, not
        // by itself permission to disable the historical lA warm start. Keep
        // the opaque cache reconstruction outside the explicitly bounded
        // engine facade, and bracket the legacy Dense clone so expiry remains
        // terminal before publication.
        let explicitly_bounded = params.deadline.is_some()
            && params
                .context
                .engine
                .is_some_and(|engine| engine.forbids_unbounded_cpu_fallback());
        let warm_started = if self.config.enable_la_warm_start && !explicitly_bounded {
            match (params.seed_cache, params.context.history.last_branch_node()) {
                (Some(cache), Some(branch_node)) => {
                    super::super::ensure_constrained_propagation_deadline(
                        params.deadline,
                        "before constrained lA warm-start reconstruction",
                    )?;
                    if let Some(warm_lb) = cache.linear_bounds(branch_node) {
                        super::super::ensure_constrained_propagation_deadline(
                            params.deadline,
                            "after constrained lA warm-start reconstruction",
                        )?;
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
            gpu_suffix_runtime_refused: false,
            state: ConstrainedBackwardState {
                node_crown_bounds,
                intermediate,
                captured_linear_bounds: params.capture_linear_bounds.then(HashMap::new),
                input_accumulated: false,
            },
        })
    }
}
