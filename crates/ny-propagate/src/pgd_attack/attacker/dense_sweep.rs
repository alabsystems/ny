// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense deterministic grid sweep for low-effective-dimension input boxes
//! (#dense-sweep).
//!
//! Some benchmark boxes pin all but a handful of input dims (cctsdb_yolo has
//! exactly 2 dims with nonzero width). Gradient-guided restarts are the wrong
//! tool there: a dense grid over the varying dims evaluates the ENTIRE box at
//! a resolution PGD restarts cannot match, deterministically, in a few batched
//! forward passes. This pre-PGD phase:
//!
//! 1. lays a uniform grid over the varying dims (pinned dims stay at their
//!    exact bound value), sized so the initial pass spends about half the
//!    point budget (128×128 for 2 dims at the default budget);
//! 2. refines around the top-k best cells with progressively halved local
//!    grids until a violation is found or the budget/deadline runs out.
//!
//! Every probe point is constructed inside `[lower, upper]` coordinate-wise,
//! so a hit is strictly in-box by construction. ATTACK-ONLY: a hit is a
//! counterexample CANDIDATE that flows through the exact same downstream
//! validation (concrete re-check + trusted-ORT witness gate) as any PGD
//! candidate — nothing here can affect a sound verdict.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::Network;

use super::PgdAttacker;

/// Refinement keeps this many best cells per round.
const REFINE_TOP_K: usize = 16;
/// Per-dim points of each local refinement grid (odd, so the center point —
/// the previous best — is re-evaluated exactly).
const REFINE_LOCAL_POINTS: usize = 5;
/// Hard cap on refinement rounds (each round halves the local radius, so 12
/// rounds shrink a cell by 4096× — far below f32 resolution of typical boxes).
const REFINE_MAX_ROUNDS: usize = 12;
/// Forward-evaluation chunk size (bounds peak memory of a batched forward).
const SWEEP_CHUNK: usize = 4096;
/// Per-dim resolution cap for the initial grid pass.
const MAX_GRID_PER_DIM: usize = 512;

/// One scored sweep point: varying-dim coordinates plus its objective value
/// (larger = closer to violation, sign-normalized for the attack direction).
struct ScoredPoint {
    coords: Vec<f32>,
    score: f32,
}

impl PgdAttacker<'_> {
    /// Dense low-effective-dimension sweep (#dense-sweep): pre-PGD phase for
    /// `attack()`. Returns `Ok(Some((input, output, evals)))` when a grid or
    /// refinement point violates the property, `Ok(None)` when the gate does
    /// not fire (feature off / too many varying dims) or no violation was
    /// found within the point budget and deadline.
    pub(super) fn try_dense_low_dim_sweep(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        output_idx: usize,
        threshold: f32,
        verify_upper_bound: bool,
    ) -> Result<Option<(ArrayD<f32>, ArrayD<f32>, usize)>> {
        if !self.config.dense_low_dim_sweep {
            return Ok(None);
        }
        let lower = input_bounds.lower();
        let upper = input_bounds.upper();
        // Flat indices of dims with nonzero width — the effective dimensions.
        let varying: Vec<usize> = lower
            .iter()
            .zip(upper.iter())
            .enumerate()
            .filter(|(_, (l, u))| u > l)
            .map(|(i, _)| i)
            .collect();
        if varying.is_empty() || varying.len() > self.config.dense_sweep_max_dims {
            return Ok(None);
        }
        let budget = self.config.dense_sweep_points;
        if budget < 4 {
            return Ok(None);
        }

        let d = varying.len();
        let lo: Vec<f32> = varying
            .iter()
            .map(|&i| lower.iter().nth(i).copied().unwrap())
            .collect();
        let hi: Vec<f32> = varying
            .iter()
            .map(|&i| upper.iter().nth(i).copied().unwrap())
            .collect();

        // Base point: every dim at its lower bound; pinned dims have
        // lower == upper, so this is their exact (in-box) value.
        let base: ArrayD<f32> = lower.to_owned();

        // Sign-normalized objective: larger = closer to violation.
        let score_of = |value: f32| if verify_upper_bound { value } else { -value };
        let violates = |value: f32| {
            if verify_upper_bound {
                value >= threshold
            } else {
                value <= threshold
            }
        };

        // --- Phase 1: uniform grid, ~half the budget. -----------------------
        let per_dim = (((budget / 2) as f64).powf(1.0 / d as f64).floor() as usize)
            .clamp(2, MAX_GRID_PER_DIM);
        let mut evals = 0usize;
        let mut best: Vec<ScoredPoint> = Vec::new();

        let grid_coord = |dim: usize, step: usize| -> f32 {
            let t = step as f32 / (per_dim - 1) as f32;
            (lo[dim] + (hi[dim] - lo[dim]) * t).clamp(lo[dim], hi[dim])
        };
        let phase1: Vec<Vec<f32>> = (0..per_dim.pow(d as u32))
            .map(|mut idx| {
                (0..d)
                    .map(|dim| {
                        let step = idx % per_dim;
                        idx /= per_dim;
                        grid_coord(dim, step)
                    })
                    .collect()
            })
            .collect();

        if let Some(hit) = self.sweep_eval_chunked(
            network, &base, &varying, &phase1, output_idx, &score_of, &violates, &mut best,
            &mut evals,
        )? {
            return Ok(Some((hit.0, hit.1, evals)));
        }

        // --- Phase 2: top-k local refinement, halving the radius each round. -
        let mut radius: Vec<f32> = (0..d)
            .map(|dim| (hi[dim] - lo[dim]) / (per_dim - 1) as f32)
            .collect();
        for _round in 0..REFINE_MAX_ROUNDS {
            if evals >= budget || self.config.past_deadline() {
                break;
            }
            let centers: Vec<Vec<f32>> = best.iter().map(|p| p.coords.clone()).collect();
            if centers.is_empty() {
                break;
            }
            let mut local: Vec<Vec<f32>> = Vec::new();
            for center in &centers {
                for mut idx in 0..REFINE_LOCAL_POINTS.pow(d as u32) {
                    let coords: Vec<f32> = (0..d)
                        .map(|dim| {
                            let step = idx % REFINE_LOCAL_POINTS;
                            idx /= REFINE_LOCAL_POINTS;
                            let t = step as f32 / (REFINE_LOCAL_POINTS - 1) as f32;
                            (center[dim] - radius[dim] + 2.0 * radius[dim] * t)
                                .clamp(lo[dim], hi[dim])
                        })
                        .collect();
                    local.push(coords);
                }
            }
            local.truncate(budget.saturating_sub(evals));
            if local.is_empty() {
                break;
            }
            if let Some(hit) = self.sweep_eval_chunked(
                network, &base, &varying, &local, output_idx, &score_of, &violates, &mut best,
                &mut evals,
            )? {
                return Ok(Some((hit.0, hit.1, evals)));
            }
            for r in radius.iter_mut() {
                *r *= 0.5;
            }
        }

        Ok(None)
    }

    /// Evaluate `coords_list` (varying-dim coordinates) in batched chunks
    /// through the attack's existing concrete forward machinery. Returns the
    /// first violating (input, output) pair, and maintains the global top-k
    /// `best` list otherwise.
    #[allow(clippy::too_many_arguments)] // sweep state threaded through one helper
    fn sweep_eval_chunked(
        &self,
        network: &Network,
        base: &ArrayD<f32>,
        varying: &[usize],
        coords_list: &[Vec<f32>],
        output_idx: usize,
        score_of: &dyn Fn(f32) -> f32,
        violates: &dyn Fn(f32) -> bool,
        best: &mut Vec<ScoredPoint>,
        evals: &mut usize,
    ) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
        let materialize = |coords: &[f32]| -> ArrayD<f32> {
            let mut point = base.clone();
            for (&flat_idx, &v) in varying.iter().zip(coords.iter()) {
                if let Some(cell) = point.iter_mut().nth(flat_idx) {
                    *cell = v;
                }
            }
            point
        };

        for chunk in coords_list.chunks(SWEEP_CHUNK) {
            if self.config.past_deadline() {
                return Ok(None);
            }
            let mut batch_shape = vec![chunk.len()];
            batch_shape.extend_from_slice(base.shape());
            let mut batch = ArrayD::zeros(IxDyn(&batch_shape));
            for (i, coords) in chunk.iter().enumerate() {
                batch
                    .index_axis_mut(Axis(0), i)
                    .assign(&materialize(coords));
            }
            let outputs = self.evaluate_batch(network, &batch)?;
            *evals += chunk.len();

            for (i, coords) in chunk.iter().enumerate() {
                let out_view = outputs.index_axis(Axis(0), i);
                let Some(value) = out_view.iter().nth(output_idx).copied() else {
                    continue;
                };
                if !value.is_finite() {
                    continue;
                }
                if violates(value) {
                    let input = materialize(coords);
                    // Confirm with the TRUE single-point forward (the batched
                    // eval is the same machinery, but keep the accept check on
                    // the exact path every attack verdict candidate uses).
                    let output = self.evaluate(network, &input)?;
                    *evals += 1;
                    if let Some(confirmed) = output.iter().nth(output_idx).copied() {
                        if violates(confirmed) {
                            return Ok(Some((input, output)));
                        }
                    }
                    continue;
                }
                let score = score_of(value);
                best.push(ScoredPoint {
                    coords: coords.clone(),
                    score,
                });
            }
            // Keep only the global top-k candidate cells.
            best.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            best.truncate(REFINE_TOP_K);
        }
        Ok(None)
    }
}
