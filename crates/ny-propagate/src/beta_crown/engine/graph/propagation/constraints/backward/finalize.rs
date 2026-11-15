// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Final concretization for constrained backward CROWN.
//!
//! The legacy post-hoc graph cut contribution used to be applied here. It was
//! deleted: it was never a certified GCP-CROWN fold (the scalar was added after
//! concretization, outside the backward relaxation) and
//! `BetaCrownConfig::cut_proof_authority_enabled()` had already made it
//! statically unreachable.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::batched_domain::CachedLinearBounds;
use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::bounds::GraphAlphaCrownIntermediate;
use crate::{LinearBounds, NETWORK_INPUT};

use super::super::super::super::super::BetaCrownVerifier;
use super::{
    BackwardCrownResult, BackwardParams, ConstrainedBackwardSetup, ConstrainedBackwardState,
};

/// The rows the constrained walk was seeded with (setup.rs precedence: the
/// spec matrix, else the objective row; the identity seed has no rows).
#[derive(Clone, Copy)]
enum SeedRows<'a> {
    Spec(&'a ndarray::Array2<f32>),
    Objective(&'a [f32]),
}

/// #finalize-objective: map the output node's forward box through the seeded
/// rows with the certified directed-rounding interval dot product
/// (`objectives::objective_bounds`, the reducer the #2904 batched fallback
/// uses), one entry per row, in the seed's output shape. A row the box does
/// not fit (never the case after setup.rs validated the seed width) yields
/// the conservative `(-inf, +inf)` rather than a guess.
fn map_output_box_through_seed_rows(
    output_box: &BoundedTensor,
    rows: SeedRows<'_>,
    output_shape: &[usize],
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    let row_count = match rows {
        SeedRows::Spec(spec) => spec.nrows(),
        SeedRows::Objective(_) => 1,
    };
    if output_shape.iter().product::<usize>() != row_count {
        return Err(NyError::InternalError(format!(
            "constrained IBP fallback: seed has {row_count} rows but the result shape is \
             {output_shape:?}"
        )));
    }
    let mut lower = Vec::with_capacity(row_count);
    let mut upper = Vec::with_capacity(row_count);
    let mut push = |row: &[f32]| {
        let (lo, hi) = crate::beta_crown::engine::graph::objective_bounds(output_box, row)
            .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        lower.push(lo);
        upper.push(hi);
    };
    match rows {
        SeedRows::Objective(objective) => push(objective),
        SeedRows::Spec(spec) => {
            for row in spec.rows() {
                super::super::ensure_constrained_propagation_deadline(
                    deadline,
                    "during constrained IBP fallback spec-row projection",
                )?;
                match row.as_slice() {
                    Some(slice) => push(slice),
                    None => push(&row.to_vec()),
                }
            }
        }
    }
    let shape = ndarray::IxDyn(output_shape);
    BoundedTensor::new_allow_infinite(
        ndarray::ArrayD::from_shape_vec(shape.clone(), lower).map_err(|error| {
            NyError::InternalError(format!("constrained IBP fallback lower shape: {error}"))
        })?,
        ndarray::ArrayD::from_shape_vec(shape, upper).map_err(|error| {
            NyError::InternalError(format!("constrained IBP fallback upper shape: {error}"))
        })?,
    )
}

fn constrained_capture_logical_memory_bytes(
    intermediate: Option<&GraphAlphaCrownIntermediate>,
    captured_linear_bounds: Option<&HashMap<String, LinearBounds>>,
    deadline: Option<Instant>,
    phase: &'static str,
) -> Result<usize> {
    super::super::ensure_constrained_propagation_deadline(deadline, phase)?;
    let mut bytes = intermediate.map_or(0, GraphAlphaCrownIntermediate::logical_memory_bytes);
    if let Some(linear_bounds_map) = captured_linear_bounds {
        for (index, (key, bounds)) in linear_bounds_map.iter().enumerate() {
            bytes = bytes
                .saturating_add(key.len())
                .saturating_add(bounds.memory_bytes());
            if index % 4096 == 4095 {
                super::super::ensure_constrained_propagation_deadline(deadline, phase)?;
            }
        }
    }
    super::super::ensure_constrained_propagation_deadline(deadline, phase)?;
    Ok(bytes)
}

impl BetaCrownVerifier {
    pub(super) fn finalize_constrained_backward(
        &self,
        params: &BackwardParams<'_>,
        is_standard: bool,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: ConstrainedBackwardSetup<'_, '_>,
    ) -> Result<BackwardCrownResult> {
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained backward finalization",
        )?;
        let ConstrainedBackwardSetup {
            output_node,
            output_shape,
            state,
            ..
        } = setup;
        let ConstrainedBackwardState {
            mut node_crown_bounds,
            mut intermediate,
            mut captured_linear_bounds,
            captured_frontiers,
            input_accumulated,
        } = state;

        let output_bounds = if input_accumulated {
            let retained_capture_bytes = if params.deadline.is_some() {
                constrained_capture_logical_memory_bytes(
                    intermediate.as_ref(),
                    captured_linear_bounds.as_ref(),
                    params.deadline,
                    "during constrained final capture retained-memory scan",
                )?
            } else {
                0
            };
            let final_cb = node_crown_bounds
                .take_with_deadline_and_resident(
                    NETWORK_INPUT,
                    params.deadline,
                    retained_capture_bytes,
                )?
                .ok_or_else(|| NyError::InvalidSpec("No linear bounds at input".to_string()))?;
            // #finalize-objective: the input block is the whole bound only if
            // every contribution reached it. A block still pending at a node
            // the walk never processed (one not in `exec_order` below its
            // producer) would be dropped from the concretization, tightening
            // the bound by that path's entire contribution. The full
            // topological order never leaves one; refuse rather than publish.
            if !node_crown_bounds.is_empty() {
                return Err(NyError::SoundnessRefusal(
                    "constrained backward reached the input with a contribution still \
                     pending at an unprocessed node"
                        .to_string(),
                ));
            }
            let final_lb = match final_cb {
                CrownBounds::Dense(bounds) => bounds,
                CrownBounds::Patches(bounds) => {
                    if params.deadline.is_some() {
                        bounds.to_dense_with_deadline_and_resident_for_purpose(
                            params.deadline,
                            retained_capture_bytes,
                            PatchesMaterializationPurpose::NetworkInputTerminal,
                        )?
                    } else {
                        bounds.to_dense_for_purpose(
                            PatchesMaterializationPurpose::NetworkInputTerminal,
                        )?
                    }
                }
            };
            if is_standard
                && params.deadline.is_none()
                && tracing::enabled!(tracing::Level::DEBUG)
                && params.context.history.constraints.len() >= 12
            {
                let gap = (final_lb.upper_a() - final_lb.lower_a())
                    .mapv(f32::abs)
                    .sum();
                let b_gap = (final_lb.upper_b() - final_lb.lower_b())
                    .mapv(f32::abs)
                    .sum();
                if gap > 1e-6 || b_gap > 1e-6 {
                    debug!(
                        "[#1817] CROWN backward A-gap={:.6}, b-gap={:.6} constraints={}",
                        gap,
                        b_gap,
                        params.context.history.constraints.len()
                    );
                }
            }
            let staged_intermediate_bounds = if intermediate.is_some() {
                Some(if params.deadline.is_some() {
                    final_lb.try_clone_with_deadline(params.deadline, retained_capture_bytes)?
                } else {
                    final_lb.clone()
                })
            } else {
                None
            };
            let retained_capture_bytes = retained_capture_bytes.saturating_add(
                staged_intermediate_bounds
                    .as_ref()
                    .map_or(0, LinearBounds::memory_bytes),
            );
            let staged_linear_bounds = if captured_linear_bounds.is_some() {
                let mut staged = if params.deadline.is_some() {
                    final_lb.try_clone_with_deadline(params.deadline, retained_capture_bytes)?
                } else {
                    final_lb.clone()
                };
                // #warm-cache-err: `CachedLinearBounds` stores no coefficient
                // error; fold it over this domain's input box before
                // publication, or leave the input block out of the cache.
                CachedLinearBounds::fold_capture_coeff_err(
                    &mut staged,
                    Some(params.constrained_input),
                    params.deadline,
                )?
                .then_some(staged)
            } else {
                None
            };
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "before constrained final-bound capture publication",
            )?;
            if let (Some(intermediate), Some(staged)) =
                (intermediate.as_mut(), staged_intermediate_bounds)
            {
                intermediate.final_bounds = staged;
            }
            if let (Some(linear_bounds_map), Some(staged)) =
                (captured_linear_bounds.as_mut(), staged_linear_bounds)
            {
                linear_bounds_map.insert(NETWORK_INPUT.to_string(), staged);
            }
            let concretized = if params.deadline.is_some() {
                let retained_capture_bytes = constrained_capture_logical_memory_bytes(
                    intermediate.as_ref(),
                    captured_linear_bounds.as_ref(),
                    params.deadline,
                    "during constrained pre-concretize retained-memory scan",
                )?;
                final_lb.concretize_sound_with_deadline_and_resident(
                    params.constrained_input,
                    params.deadline,
                    retained_capture_bytes,
                )?
            } else {
                final_lb.concretize_sound_with_deadline(params.constrained_input, None)?
            };
            let output = if params.deadline.is_some() {
                concretized.into_reshape_with_poll(&output_shape, || {
                    super::super::ensure_constrained_propagation_deadline(
                        params.deadline,
                        "during constrained final output reshape",
                    )
                })?
            } else {
                concretized.reshape(&output_shape)?
            };
            super::super::ensure_constrained_propagation_deadline(
                params.deadline,
                "before constrained final-bound publication",
            )?;
            output
        } else {
            if is_standard {
                debug!(
                    "[#1817] CROWN backward did NOT reach input, falling back to IBP output bounds"
                );
            }
            let cached_output = bounds_cache_mut.get(output_node).ok_or_else(|| {
                NyError::InvalidSpec(format!("Output node {} not found", output_node))
            })?;
            // #finalize-objective: the walk was seeded with the spec rows `C`
            // or the objective row `c` (setup.rs), so its result must bound
            // `C·out` / `c·out`. The forward box of the output node bounds
            // `out` itself; it is mapped through the same rows, outward, before
            // it may stand in for the backward result. Publishing the raw box
            // was unsound for any row other than a unit selector (`c = [-1]`
            // put the lower bound at `min out` where the truth is `-max out`)
            // and the wrong shape for a multi-row spec. The output entry stays
            // in the cache (it is only read here).
            let seed_rows = match (params.spec_matrix, params.objective) {
                (Some(spec), _) => Some(SeedRows::Spec(spec)),
                (None, Some(objective)) => Some(SeedRows::Objective(objective)),
                (None, None) => None,
            };
            if let Some(rows) = seed_rows {
                let mapped = map_output_box_through_seed_rows(
                    cached_output,
                    rows,
                    &output_shape,
                    params.deadline,
                )?;
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "before constrained IBP fallback publication",
                )?;
                mapped
            } else {
                // Identity seed: the output box IS the bound.
                // #cone-delta increment 2 residual copy: this IBP fallback
                // (backward did not reach the input) must hand out an OWNED
                // tensor. The Arc is usually uniquely held here (the output
                // node is always in the recompute cone), so the finite route
                // moves it. Only the legacy unbounded route may use the opaque
                // shared-Arc deep clone.
                if params.deadline.is_some() && Arc::strong_count(cached_output) != 1 {
                    return Err(NyError::UnsupportedConfiguration(format!(
                        "finite constrained IBP fallback requires unique output ownership; shared {} clone is not cooperatively pollable",
                        output_node
                    )));
                }
                super::super::ensure_constrained_propagation_deadline(
                    params.deadline,
                    "before constrained IBP fallback publication",
                )?;
                let cached_output = bounds_cache_mut
                    .remove(output_node)
                    .expect("output presence was checked before removal");
                if params.deadline.is_some() {
                    Arc::try_unwrap(cached_output).map_err(|_| {
                        NyError::UnsupportedConfiguration(format!(
                            "finite constrained IBP fallback lost unique ownership of {}",
                            output_node
                        ))
                    })?
                } else {
                    Arc::unwrap_or_clone(cached_output)
                }
            }
        };

        let captured_la = if input_accumulated {
            captured_linear_bounds.map(|map| {
                let cache = CachedLinearBounds::from_linear_bounds_map(map);
                // #la-warm-frontier: `None` unless armed (bit-identical dark).
                match captured_frontiers {
                    Some(frontiers) if !frontiers.is_empty() => cache.with_frontiers(frontiers),
                    _ => cache,
                }
            })
        } else {
            None
        };
        super::super::ensure_constrained_propagation_deadline(
            params.deadline,
            "before constrained backward result publication",
        )?;
        Ok(BackwardCrownResult {
            output_bounds,
            intermediate,
            captured_la,
        })
    }
}
