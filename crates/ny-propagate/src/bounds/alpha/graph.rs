// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph/DAG alpha-CROWN state (`GraphAlphaState`).
//!
//! Stores learnable alpha parameters for unstable ReLU neurons in DAG-structured
//! graph models. Uses node names as keys (BTreeMap for deterministic iteration
//! order). Also holds monotone S-shaped and Sqrt tangent-point alpha bundles.

use crate::layers::trigonometric::{
    sigmoid_crossing_default_tangents, tanh_crossing_default_tangents,
};
use ndarray::{Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::contiguous_flat_slice;

use super::super::alpha_reciprocal::ReciprocalAlpha;
use super::super::alpha_s_shaped::MonotoneSShapedAlpha;
use super::super::alpha_sqrt::SqrtAlpha;
use super::shared::{
    extract_contiguous_bounds, init_alpha_from_bounds, update_alphas_adam, update_alphas_sgd,
};
use super::AdamParams;

/// One spec row's view of a ReLU node's lower-path α (#spec-axis-alpha).
///
/// `Base` borrows the shared vector untouched — the bit-identical fallback
/// for rows without an active δ slot. `Materialized` owns the clamped
/// `α_base + δ_slot` for an active row. Callers treat both as a slice.
#[derive(Debug)]
pub(crate) enum RowAlpha<'state> {
    Base(&'state Array1<f32>),
    Materialized(Array1<f32>),
}

impl RowAlpha<'_> {
    pub(crate) fn as_array(&self) -> &Array1<f32> {
        match self {
            Self::Base(alpha) => alpha,
            Self::Materialized(alpha) => alpha,
        }
    }
}

/// Alpha state for DAG/GraphNetwork models.
///
/// Unlike `AlphaState` which uses indices, `GraphAlphaState` uses node names
/// as keys, since DAG models have named nodes rather than sequential layer indices.
#[derive(Debug, Clone)]
pub struct GraphAlphaState {
    /// Alpha values per ReLU node for the **lower bound path** (alpha[0]).
    /// Key is the node name.
    /// Each Array1 has length equal to the number of neurons in that ReLU node.
    /// BTreeMap ensures deterministic iteration order, which is required for
    /// reproducible SPSA gradient estimation (RNG consumption order must be
    /// consistent across runs). See #1976.
    pub(crate) alphas: std::collections::BTreeMap<String, Array1<f32>>,
    /// Alpha values per ReLU node for the **upper bound path** (alpha[1]). (#3393)
    pub(crate) alphas_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Mask for unstable neurons (l < 0 < u). Only these neurons have optimizable alpha.
    pub(crate) unstable_mask: std::collections::BTreeMap<String, Array1<bool>>,
    /// Momentum for gradient updates (velocity) - used by SGD with momentum.
    pub(crate) velocity: std::collections::BTreeMap<String, Array1<f32>>,
    /// First moment estimate (mean of gradients) for Adam optimizer.
    pub(crate) adam_m: std::collections::BTreeMap<String, Array1<f32>>,
    /// Second moment estimate (uncentered variance) for Adam optimizer.
    pub(crate) adam_v: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path optimizer state (#3393).
    pub(crate) velocity_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path first moment estimate (#3393).
    pub(crate) adam_m_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Upper-path second moment estimate (#3393).
    pub(crate) adam_v_upper: std::collections::BTreeMap<String, Array1<f32>>,
    /// Tangent-point alpha bundles for monotone Sigmoid/Tanh DAG nodes.
    pub(crate) monotone_s_shaped_alphas: std::collections::BTreeMap<String, MonotoneSShapedAlpha>,
    /// Tangent-point alpha bundles for positive-domain Sqrt DAG nodes.
    pub(crate) sqrt_alphas: std::collections::BTreeMap<String, SqrtAlpha>,
    /// Tangent-point alpha bundles for non-zero-domain Reciprocal DAG nodes.
    pub(crate) reciprocal_alphas: std::collections::BTreeMap<String, ReciprocalAlpha>,
    /// Original spatial shape for channel-only alpha nodes.
    /// When `full_conv_alpha: False`, conv-output ReLU alpha has length C instead
    /// of C*H*W. This map stores [C, H, W] for nodes using channel-only alpha,
    /// enabling expansion before backward pass and reduction after gradient
    /// computation. Absent key = full alpha (no expansion needed).
    /// Reference: `backward_bound.py:868-938`, `relu.py:658-664`.
    pub(crate) spatial_shapes: std::collections::BTreeMap<String, Vec<usize>>,
    /// Negative cache for GPU-suffix offload attempts (perf only, no bound
    /// impact). Per-node entries record that suffix extractability is a GRAPH
    /// STRUCTURE from a node to the input, which never changes across alpha
    /// iterations — yet on suffix-ineligible graphs (vit attention: MatMul/
    /// Softmax/Transpose never decompose) every backward pass re-attempted the
    /// full extraction walk on every node (measured: 102 wasted walks per pass
    /// per iteration). A node lands here after BOTH the unary-chain extraction
    /// AND the resnet decomposition declined it; seed-dependent rejections
    /// (non-finite coefficients) are NOT cached. A reserved internal entry also
    /// disables later GPU attempts for this state after any backend error or
    /// malformed receipt, so CPU fallback is atomic for the solve. `Arc` lets
    /// cheap state clones share both decisions for the same graph.
    pub(crate) gpu_suffix_ineligible:
        std::sync::Arc<std::sync::RwLock<std::collections::BTreeSet<String>>>,
    /// Spec-axis δ rows (#spec-axis-alpha, `docs/SPEC_AXIS_ALPHA_DESIGN.md`).
    ///
    /// One shared α vector per ReLU serves every output margin today, and the
    /// margins fight over it (MEASURED: joint ascent degrades per-spec
    /// bounds, `CIFAR100_ROOT_ALPHA_DEGRADES_SPEC_BOUNDS_2026-07-26.md`).
    /// The spec axis parameterizes per-margin deviations from the shared
    /// baseline: `α_eff(node, row) = clamp01(α_base(node) + δ(node, slot))`
    /// for the K ACTIVE slots in [`Self::spec_slot_rows`]; every other row
    /// reads `α_base` untouched. δ is stored per node as a `K × num_alpha`
    /// array, row-indexed by slot order.
    ///
    /// Empty maps ⇒ the accessors fall back to the shared vectors
    /// bit-identically — slice 1 lands this state dark, with no compose or
    /// optimizer consumer (design §5.1).
    pub(crate) spec_deltas: std::collections::BTreeMap<String, Array2<f32>>,
    /// Which C-matrix rows own the δ slots, in slot order. `spec_slot_rows[k]`
    /// is the spec row served by δ row `k` in every [`Self::spec_deltas`]
    /// entry. Empty ⇒ no active slots anywhere.
    pub(crate) spec_slot_rows: Vec<usize>,
    /// Adam first-moment state for δ rows (slice 2b), keyed like
    /// [`Self::spec_deltas`]. Optimizer-only: never read by a backward pass,
    /// so `clone_for_backward` leaves it empty like the six base maps.
    pub(crate) spec_adam_m: std::collections::BTreeMap<String, Array2<f32>>,
    /// Adam second-moment state for δ rows (slice 2b).
    pub(crate) spec_adam_v: std::collections::BTreeMap<String, Array2<f32>>,
}

impl GraphAlphaState {
    /// Create empty state.
    pub fn new() -> Self {
        Self {
            alphas: std::collections::BTreeMap::new(),
            alphas_upper: std::collections::BTreeMap::new(),
            unstable_mask: std::collections::BTreeMap::new(),
            velocity: std::collections::BTreeMap::new(),
            adam_m: std::collections::BTreeMap::new(),
            adam_v: std::collections::BTreeMap::new(),
            velocity_upper: std::collections::BTreeMap::new(),
            adam_m_upper: std::collections::BTreeMap::new(),
            adam_v_upper: std::collections::BTreeMap::new(),
            monotone_s_shaped_alphas: std::collections::BTreeMap::new(),
            sqrt_alphas: std::collections::BTreeMap::new(),
            reciprocal_alphas: std::collections::BTreeMap::new(),
            spatial_shapes: std::collections::BTreeMap::new(),
            gpu_suffix_ineligible: std::sync::Arc::new(std::sync::RwLock::new(
                std::collections::BTreeSet::new(),
            )),
            spec_deltas: std::collections::BTreeMap::new(),
            spec_slot_rows: Vec::new(),
            spec_adam_m: std::collections::BTreeMap::new(),
            spec_adam_v: std::collections::BTreeMap::new(),
        }
    }

    /// Clone only the state a CROWN **backward pass** reads, leaving the six
    /// optimizer-state maps (`velocity`/`adam_m`/`adam_v` and their `_upper`
    /// variants) EMPTY.
    ///
    /// SPSA gradient estimation evaluates the objective at perturbed alpha
    /// values: it builds `2 * num_samples` perturbed copies of the alpha state
    /// and runs a backward pass on each. The backward pass
    /// (`run_target_backward_pass` and everything it calls) reads ONLY the seven
    /// alpha/shape fields — `alphas`, `alphas_upper`, `unstable_mask`,
    /// `monotone_s_shaped_alphas`, `sqrt_alphas`, `reciprocal_alphas`,
    /// `spatial_shapes`. The six optimizer-state maps are touched only by the
    /// Adam/SGD `update*`/`add_relu_node` paths, which a perturbation copy never
    /// reaches. A full `clone()` would deep-copy those six maps `2 * num_samples`
    /// times per optimization iteration for nothing.
    ///
    /// NUMERICALLY IDENTICAL: every field the backward pass reads is cloned
    /// bit-for-bit; the omitted maps are provably never read on that path, so the
    /// computed bounds (and their exact f32 bits) are unchanged. The result is
    /// suitable ONLY for a read-only backward pass — never route it into an
    /// optimizer `update*` step, which expects the optimizer maps populated.
    #[must_use]
    pub(crate) fn clone_for_backward(&self) -> Self {
        Self {
            alphas: self.alphas.clone(),
            alphas_upper: self.alphas_upper.clone(),
            unstable_mask: self.unstable_mask.clone(),
            // Optimizer state is not read by a backward pass — leave empty.
            velocity: std::collections::BTreeMap::new(),
            adam_m: std::collections::BTreeMap::new(),
            adam_v: std::collections::BTreeMap::new(),
            velocity_upper: std::collections::BTreeMap::new(),
            adam_m_upper: std::collections::BTreeMap::new(),
            adam_v_upper: std::collections::BTreeMap::new(),
            monotone_s_shaped_alphas: self.monotone_s_shaped_alphas.clone(),
            sqrt_alphas: self.sqrt_alphas.clone(),
            reciprocal_alphas: self.reciprocal_alphas.clone(),
            spatial_shapes: self.spatial_shapes.clone(),
            // Share the negative cache: suffix eligibility is a graph-structure
            // property, identical for every perturbation copy.
            gpu_suffix_ineligible: std::sync::Arc::clone(&self.gpu_suffix_ineligible),
            // δ rows are read by the backward pass once slice 2 wires the
            // compose, so perturbation copies must carry them.
            spec_deltas: self.spec_deltas.clone(),
            spec_slot_rows: self.spec_slot_rows.clone(),
            // δ optimizer state is update-only, like the six base maps.
            spec_adam_m: std::collections::BTreeMap::new(),
            spec_adam_v: std::collections::BTreeMap::new(),
        }
    }

    /// Consume a state returned by a warm-bound call and retain only data read
    /// by a later DAG warm-start.
    ///
    /// `collect_alpha_crown_bounds_dag_warm_with_engine` creates a fresh child
    /// state and therefore resets the six ReLU optimizer maps before copying
    /// the parent's lower/upper alpha values. Keeping those maps on every queued
    /// input-split domain can multiply queue memory without affecting the next
    /// bound. All warm-read parameters, masks, tangent bundles, spatial shapes,
    /// and the shared GPU-suffix cache are preserved exactly.
    #[must_use]
    pub(crate) fn into_warm_start_seed(mut self) -> Self {
        self.velocity.clear();
        self.adam_m.clear();
        self.adam_v.clear();
        self.velocity_upper.clear();
        self.adam_m_upper.clear();
        self.adam_v_upper.clear();
        self
    }

    /// Initialize alpha state from pre-activation bounds for a single ReLU node.
    ///
    /// For unstable neurons (l < 0 < u), initializes alpha using the adaptive heuristic:
    /// alpha = 1 if u > -l, else 0.
    ///
    /// When `channel_only_alpha` is true and the pre-activation has spatial dimensions
    /// (ndim >= 3, i.e., [C, H, W]), alpha is reduced to per-channel (length C) by
    /// taking per-channel worst-case bounds (min lower, max upper). This is the
    /// `full_conv_alpha: False` mode from the reference (`backward_bound.py:868-938`).
    pub fn add_relu_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
        channel_only_alpha: bool,
    ) -> Result<()> {
        let shape = pre_activation.shape();
        let use_channel_only = channel_only_alpha && shape.len() >= 3;
        tracing::debug!(
            node = node_name,
            shape = ?shape,
            channel_only = use_channel_only,
            "alpha init"
        );

        let (alpha, mask, num_alpha) = if use_channel_only {
            // Channel-only: reduce [C, H, W] bounds to [C] by taking worst-case per channel.
            // Reference: get_unstable_locations(..., channel_only=True, conv=True)
            let channels = shape[0];
            let spatial: usize = shape[1..].iter().product();

            let (lower_std, upper_std) = extract_contiguous_bounds(&pre_activation.flatten())?;
            let lower_arr = contiguous_flat_slice(&lower_std);
            let upper_arr = contiguous_flat_slice(&upper_std);

            // Reshape to [C, spatial] and reduce per channel
            let lower_2d = Array2::from_shape_vec((channels, spatial), lower_arr.to_vec())
                .map_err(|_e| ny_core::NyError::ShapeMismatch {
                    expected: vec![channels, spatial],
                    got: vec![lower_arr.len()],
                })?;
            let upper_2d = Array2::from_shape_vec((channels, spatial), upper_arr.to_vec())
                .map_err(|_e| ny_core::NyError::ShapeMismatch {
                    expected: vec![channels, spatial],
                    got: vec![upper_arr.len()],
                })?;

            let channel_lower: Array1<f32> = lower_2d.map_axis(ndarray::Axis(1), |row| {
                row.iter().copied().fold(f32::INFINITY, f32::min)
            });
            let channel_upper: Array1<f32> = upper_2d.map_axis(ndarray::Axis(1), |row| {
                row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            });

            let (alpha, mask) = init_alpha_from_bounds(
                channel_lower.as_slice().expect("contiguous"),
                channel_upper.as_slice().expect("contiguous"),
            );
            self.spatial_shapes
                .insert(node_name.to_string(), shape.to_vec());
            (alpha, mask, channels)
        } else {
            let pre_flat = pre_activation.flatten();
            let num_neurons = pre_flat.len();
            let (lower_std, upper_std) = extract_contiguous_bounds(&pre_flat)?;
            let lower_arr = contiguous_flat_slice(&lower_std);
            let upper_arr = contiguous_flat_slice(&upper_std);
            let (alpha, mask) = init_alpha_from_bounds(lower_arr.as_ref(), upper_arr.as_ref());
            (alpha, mask, num_neurons)
        };

        // Dual alpha (#3393): upper path initialized identically to lower path.
        self.alphas_upper
            .insert(node_name.to_string(), alpha.clone());
        self.alphas.insert(node_name.to_string(), alpha);
        self.unstable_mask.insert(node_name.to_string(), mask);
        self.velocity
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_m
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_v
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.velocity_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_m_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        self.adam_v_upper
            .insert(node_name.to_string(), Array1::<f32>::zeros(num_alpha));
        Ok(())
    }

    /// Lower-path alpha values for a specific ReLU node.
    pub fn alpha(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.alphas.get(node_name)
    }

    /// Effective lower-path α for one ReLU node as seen by one spec row
    /// (#spec-axis-alpha, design §2).
    ///
    /// `Base` is the exact shared slice — the SAME reference `alpha()`
    /// returns, so a row without an active δ slot (or a node without δ rows)
    /// is bit-identical to today by construction, not by arithmetic: no
    /// addition, no clamp, no copy happens on that path. Only an active slot
    /// materializes `clamp01(α_base + δ_slot)`.
    ///
    /// Slice 1 lands this accessor with no compose consumer; slice 2's
    /// dense-walk wiring calls it per active row.
    pub(crate) fn alpha_for_row(&self, node_name: &str, spec_row: usize) -> Option<RowAlpha<'_>> {
        let base = self.alphas.get(node_name)?;
        let Some(slot) = self.slot_for_spec_row(spec_row) else {
            return Some(RowAlpha::Base(base));
        };
        let Some(deltas) = self.spec_deltas.get(node_name) else {
            return Some(RowAlpha::Base(base));
        };
        // Malformed slot state falls back to the baseline DELIBERATELY
        // (never a debug_assert): a shape mismatch here means the optimizer
        // and the walk disagree about the slot table, and the sound answer
        // to disagreement is the shared α that both always agree on.
        if deltas.nrows() != self.spec_slot_rows.len() || deltas.ncols() != base.len() {
            return Some(RowAlpha::Base(base));
        }
        let effective = base
            .iter()
            .zip(deltas.row(slot).iter())
            // Π[0,1](α_base + δ): δ is stored unclamped so δ=0 reproduces the
            // baseline bit-for-bit (0.0 + x == x in IEEE for finite x, and
            // the clamp of an in-range α is itself).
            .map(|(&alpha, &delta)| (alpha + delta).clamp(0.0, 1.0))
            .collect::<Array1<f32>>();
        Some(RowAlpha::Materialized(effective))
    }

    /// Slot index owning `spec_row`, if any (#spec-axis-alpha).
    ///
    /// A duplicated row id in the slot table is malformed state; the FIRST
    /// occurrence wins deterministically (BTree-ordered construction upstream
    /// makes duplicates impossible to build through the public path, but the
    /// lookup must not depend on that).
    pub(crate) fn slot_for_spec_row(&self, spec_row: usize) -> Option<usize> {
        self.spec_slot_rows.iter().position(|&row| row == spec_row)
    }

    /// True when ANY node carries δ rows — the cheap outer gate the walk
    /// checks before building per-node row tables.
    pub(crate) fn has_spec_deltas(&self) -> bool {
        !self.spec_slot_rows.is_empty() && !self.spec_deltas.is_empty()
    }

    /// Adam ascent step on one node's δ rows from per-slot gradients
    /// (slice 2b; design §3).
    ///
    /// Semantics mirror `update_alphas_adam` for the base vectors, with the
    /// δ-specific differences the external review required:
    /// - projection keeps `α_base + δ` inside `[0, 1]`, i.e.
    ///   `δ_i ∈ [-α_base_i, 1 - α_base_i]` — NOT a bare `[0,1]` clamp;
    /// - a non-finite δ or gradient resets that entry to 0 (the BASELINE,
    ///   never the 0.5 the shared sanitize uses — 0.5 would silently move a
    ///   row off its parity anchor) and zeroes its moments;
    /// - only unstable neurons update (the mask is the same one base α uses).
    ///
    /// Ascent direction matches the base updater's convention: gradients are
    /// `∂bound/∂α` and the caller wants the LOWER bound maximized.
    pub(crate) fn update_spec_deltas_adam(
        &mut self,
        node_name: &str,
        gradients: &Array2<f32>,
        step: usize,
        params: &AdamParams,
    ) {
        let Some(base) = self.alphas.get(node_name) else {
            return;
        };
        let base = base.clone();
        let Some(mask) = self.unstable_mask.get(node_name).cloned() else {
            return;
        };
        let Some(deltas) = self.spec_deltas.get_mut(node_name) else {
            return;
        };
        if gradients.dim() != deltas.dim() || base.len() != deltas.ncols() {
            return; // malformed ⇒ leave δ untouched (fail-closed, like reads)
        }
        let slots = deltas.nrows();
        let width = deltas.ncols();
        let moment_m = self
            .spec_adam_m
            .entry(node_name.to_string())
            .or_insert_with(|| Array2::zeros((slots, width)));
        let moment_v = self
            .spec_adam_v
            .entry(node_name.to_string())
            .or_insert_with(|| Array2::zeros((slots, width)));
        if moment_m.dim() != deltas.dim() || moment_v.dim() != deltas.dim() {
            // Slot table changed shape since the moments were built (e.g.
            // reassignment): restart the optimizer state rather than mixing
            // moments across unrelated slots.
            *moment_m = Array2::zeros((slots, width));
            *moment_v = Array2::zeros((slots, width));
        }
        let t = step.max(1) as f32;
        let bias1 = 1.0 - params.beta1.powi(t as i32);
        let bias2 = 1.0 - params.beta2.powi(t as i32);
        for slot in 0..slots {
            for i in 0..width {
                if !mask[i % mask.len()] {
                    continue;
                }
                let gradient = gradients[[slot, i]];
                let delta = deltas[[slot, i]];
                if !gradient.is_finite() || !delta.is_finite() {
                    deltas[[slot, i]] = 0.0;
                    moment_m[[slot, i]] = 0.0;
                    moment_v[[slot, i]] = 0.0;
                    continue;
                }
                let m = params.beta1 * moment_m[[slot, i]] + (1.0 - params.beta1) * gradient;
                let v =
                    params.beta2 * moment_v[[slot, i]] + (1.0 - params.beta2) * gradient * gradient;
                moment_m[[slot, i]] = m;
                moment_v[[slot, i]] = v;
                let m_hat = m / bias1;
                let v_hat = v / bias2;
                // Ascent (+): maximize the row's lower bound.
                let stepped =
                    delta + params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);
                let low = -base[i];
                let high = 1.0 - base[i];
                deltas[[slot, i]] = if stepped.is_finite() {
                    stepped.clamp(low, high)
                } else {
                    moment_m[[slot, i]] = 0.0;
                    moment_v[[slot, i]] = 0.0;
                    0.0
                };
            }
        }
    }

    /// Reassign δ slot `slot` to `new_row` (#spec-axis-alpha slice 2e: lazy
    /// acquisition when the margin lane binds an unslotted row).
    ///
    /// The slot's δ row and Adam moments reset to zero across every node —
    /// the incoming row starts at the parity anchor, and the outgoing row's
    /// corrections never leak to it (the design's slot-reset rule). No-op on
    /// an out-of-range slot or when `new_row` already owns any slot.
    pub(crate) fn reassign_spec_slot(&mut self, slot: usize, new_row: usize) -> bool {
        if slot >= self.spec_slot_rows.len() || self.spec_slot_rows.contains(&new_row) {
            return false;
        }
        self.spec_slot_rows[slot] = new_row;
        for deltas in self.spec_deltas.values_mut() {
            if slot < deltas.nrows() {
                deltas.row_mut(slot).fill(0.0);
            }
        }
        for moments in self
            .spec_adam_m
            .values_mut()
            .chain(self.spec_adam_v.values_mut())
        {
            if slot < moments.nrows() {
                moments.row_mut(slot).fill(0.0);
            }
        }
        true
    }

    /// All K materialized effective-α rows for one node, in slot order, at
    /// BASE width (channel width when channel-only — expansion happens at the
    /// consumer, once per active row, mirroring the shared-α path's
    /// `expand_alpha` call order). `None` when the node has no well-formed δ.
    pub(crate) fn materialized_spec_alphas(&self, node_name: &str) -> Option<Vec<Array1<f32>>> {
        let base = self.alphas.get(node_name)?;
        let deltas = self.spec_deltas.get(node_name)?;
        if deltas.nrows() != self.spec_slot_rows.len() || deltas.ncols() != base.len() {
            return None; // malformed ⇒ consumer stays on the shared path
        }
        Some(
            (0..deltas.nrows())
                .map(|slot| {
                    base.iter()
                        .zip(deltas.row(slot).iter())
                        .map(|(&alpha, &delta)| (alpha + delta).clamp(0.0, 1.0))
                        .collect::<Array1<f32>>()
                })
                .collect(),
        )
    }

    /// Upper-path alpha values for a specific ReLU node (#3393).
    pub fn alpha_upper(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.alphas_upper.get(node_name)
    }

    /// Length of the alpha vector for one ReLU node.
    #[must_use]
    pub(crate) fn relu_len(&self, node_name: &str) -> Option<usize> {
        self.alphas.get(node_name).map(Array1::len)
    }

    /// Unstable mask for one ReLU node.
    #[must_use]
    pub(crate) fn relu_unstable_mask(&self, node_name: &str) -> Option<&Array1<bool>> {
        self.unstable_mask.get(node_name)
    }

    /// Lower/upper alpha pair for one ReLU node.
    #[must_use]
    pub(crate) fn relu_alpha_pair(&self, node_name: &str) -> Option<(&Array1<f32>, &Array1<f32>)> {
        Some((
            self.alphas.get(node_name)?,
            self.alphas_upper.get(node_name)?,
        ))
    }

    /// Mutable lower/upper alpha pair for one ReLU node.
    pub(crate) fn relu_alpha_pair_mut(
        &mut self,
        node_name: &str,
    ) -> Option<(&mut Array1<f32>, &mut Array1<f32>)> {
        let lower = self.alphas.get_mut(node_name)?;
        let upper = self.alphas_upper.get_mut(node_name)?;
        Some((lower, upper))
    }

    /// Register monotone Sigmoid tangent-point alpha state for one DAG node.
    pub fn add_sigmoid_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha =
            MonotoneSShapedAlpha::from_bounds(pre_activation, sigmoid_crossing_default_tangents)?;
        self.monotone_s_shaped_alphas
            .insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Register monotone Tanh tangent-point alpha state for one DAG node.
    pub fn add_tanh_node(&mut self, node_name: &str, pre_activation: &BoundedTensor) -> Result<()> {
        let alpha =
            MonotoneSShapedAlpha::from_bounds(pre_activation, tanh_crossing_default_tangents)?;
        self.monotone_s_shaped_alphas
            .insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Register positive-domain Sqrt tangent-point alpha state for one DAG node.
    ///
    /// Kept `pub(crate)` per #3773 design: all observed call sites are crate-internal
    /// DAG wiring helpers. The existing `pub` on `add_sigmoid_node`/`add_tanh_node`
    /// is acknowledged visibility debt (#2611).
    pub(crate) fn add_sqrt_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha = SqrtAlpha::from_bounds(pre_activation)?;
        self.sqrt_alphas.insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Tangent-point alpha bundle for a monotone Sigmoid/Tanh DAG node.
    #[must_use]
    pub(crate) fn monotone_s_shaped_alpha(&self, node_name: &str) -> Option<&MonotoneSShapedAlpha> {
        self.monotone_s_shaped_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG node.
    pub(crate) fn monotone_s_shaped_alpha_mut(
        &mut self,
        node_name: &str,
    ) -> Option<&mut MonotoneSShapedAlpha> {
        self.monotone_s_shaped_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG monotone alpha node names.
    pub(crate) fn monotone_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.monotone_s_shaped_alphas.keys()
    }

    /// Tangent-point alpha bundle for one DAG Sqrt node.
    #[must_use]
    pub(crate) fn sqrt_alpha(&self, node_name: &str) -> Option<&SqrtAlpha> {
        self.sqrt_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG Sqrt node.
    pub(crate) fn sqrt_alpha_mut(&mut self, node_name: &str) -> Option<&mut SqrtAlpha> {
        self.sqrt_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG Sqrt alpha node names.
    pub(crate) fn sqrt_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.sqrt_alphas.keys()
    }

    /// Register non-zero-domain Reciprocal tangent-point alpha state for one DAG node.
    pub(crate) fn add_reciprocal_node(
        &mut self,
        node_name: &str,
        pre_activation: &BoundedTensor,
    ) -> Result<()> {
        let alpha = ReciprocalAlpha::from_bounds(pre_activation)?;
        self.reciprocal_alphas.insert(node_name.to_string(), alpha);
        Ok(())
    }

    /// Tangent-point alpha bundle for one DAG Reciprocal node.
    #[must_use]
    pub(crate) fn reciprocal_alpha(&self, node_name: &str) -> Option<&ReciprocalAlpha> {
        self.reciprocal_alphas.get(node_name)
    }

    /// Mutable tangent-point alpha bundle for one DAG Reciprocal node.
    pub(crate) fn reciprocal_alpha_mut(&mut self, node_name: &str) -> Option<&mut ReciprocalAlpha> {
        self.reciprocal_alphas.get_mut(node_name)
    }

    /// Deterministic iterator over DAG Reciprocal alpha node names.
    pub(crate) fn reciprocal_alpha_names(&self) -> impl Iterator<Item = &String> {
        self.reciprocal_alphas.keys()
    }

    /// Absolute velocity buffer for one ReLU node.
    #[must_use]
    pub(crate) fn relu_velocity(&self, node_name: &str) -> Option<&Array1<f32>> {
        self.velocity.get(node_name)
    }

    /// Update alpha values using gradient descent with optional momentum.
    ///
    /// Delegates to `update_alphas_sgd` for the core optimization loop.
    pub fn update(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(vel) = self.velocity.get_mut(node_name) else {
            return;
        };

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update: gradient length {} != alpha length {} for '{}' (#1937), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Count total number of unstable neurons.
    pub fn num_unstable(&self) -> usize {
        self.unstable_mask
            .values()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum()
    }

    /// Get all ReLU node names.
    pub fn relu_nodes(&self) -> impl Iterator<Item = &str> {
        self.alphas.keys().map(|s| s.as_str())
    }

    /// Update alpha values using Adam optimizer.
    ///
    /// Delegates to `update_alphas_adam` for the core optimization loop.
    pub fn update_adam(&mut self, node_name: &str, gradient: &Array1<f32>, params: &AdamParams) {
        let Some(alpha) = self.alphas.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(m) = self.adam_m.get_mut(node_name) else {
            return;
        };
        let Some(v) = self.adam_v.get_mut(node_name) else {
            return;
        };

        // Guard against length mismatch from fallback gradients (#1937).
        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_adam: gradient length {} != alpha length {} for '{}' (#1937), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }

    /// Update upper-path alpha values using gradient descent with optional momentum. (#3393)
    pub fn update_upper(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) {
        let Some(alpha) = self.alphas_upper.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(vel) = self.velocity_upper.get_mut(node_name) else {
            return;
        };

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_upper: gradient length {} != alpha length {} for '{}' (#3393), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_sgd(alpha, gradient, mask, vel, learning_rate, momentum);
    }

    /// Update upper-path alpha values using Adam optimizer. (#3393)
    pub fn update_adam_upper(
        &mut self,
        node_name: &str,
        gradient: &Array1<f32>,
        params: &AdamParams,
    ) {
        let Some(alpha) = self.alphas_upper.get_mut(node_name) else {
            return;
        };
        let Some(mask) = self.unstable_mask.get(node_name) else {
            return;
        };
        let Some(m) = self.adam_m_upper.get_mut(node_name) else {
            return;
        };
        let Some(v) = self.adam_v_upper.get_mut(node_name) else {
            return;
        };

        if gradient.len() != alpha.len() {
            tracing::warn!(
                "GraphAlphaState::update_adam_upper: gradient length {} != alpha length {} for '{}' (#3393), skipping",
                gradient.len(), alpha.len(), node_name
            );
            return;
        }

        update_alphas_adam(alpha, gradient, mask, m, v, params);
    }
}

impl Default for GraphAlphaState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod warm_start_seed_tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn consuming_warm_start_seed_drops_only_reset_optimizer_maps() {
        let pre_activation = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.5_f32]).into_dyn(),
            arr1(&[1.0_f32, 1.5_f32]).into_dyn(),
        )
        .expect("valid ReLU bounds");
        let mut state = GraphAlphaState::new();
        state
            .add_relu_node("relu", &pre_activation, false)
            .expect("ReLU state should initialize");
        state
            .spatial_shapes
            .insert("relu".to_string(), vec![1, 1, 2]);

        let expected_alphas = state.alphas.clone();
        let expected_alphas_upper = state.alphas_upper.clone();
        let expected_unstable_mask = state.unstable_mask.clone();
        let expected_spatial_shapes = state.spatial_shapes.clone();
        let expected_cache = std::sync::Arc::clone(&state.gpu_suffix_ineligible);
        assert!(
            [
                &state.velocity,
                &state.adam_m,
                &state.adam_v,
                &state.velocity_upper,
                &state.adam_m_upper,
                &state.adam_v_upper,
            ]
            .iter()
            .all(|map| !map.is_empty()),
            "fixture must populate every reset optimizer map"
        );

        let seed = state.into_warm_start_seed();

        assert_eq!(seed.alphas, expected_alphas);
        assert_eq!(seed.alphas_upper, expected_alphas_upper);
        assert_eq!(seed.unstable_mask, expected_unstable_mask);
        assert_eq!(seed.spatial_shapes, expected_spatial_shapes);
        assert!(std::sync::Arc::ptr_eq(
            &seed.gpu_suffix_ineligible,
            &expected_cache
        ));
        assert!(seed.velocity.is_empty());
        assert!(seed.adam_m.is_empty());
        assert!(seed.adam_v.is_empty());
        assert!(seed.velocity_upper.is_empty());
        assert!(seed.adam_m_upper.is_empty());
        assert!(seed.adam_v_upper.is_empty());
    }
}

#[cfg(test)]
mod spec_axis_tests {
    use ndarray::{arr1, Array2};

    use super::*;

    fn relu_state() -> GraphAlphaState {
        let pre_activation = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5, -2.0, -0.25]).into_dyn(),
            arr1(&[0.5_f32, 1.5, 1.0, 0.75]).into_dyn(),
        )
        .expect("valid unstable ReLU bounds");
        let mut state = GraphAlphaState::new();
        state
            .add_relu_node("relu", &pre_activation, false)
            .expect("ReLU state should initialize");
        state
    }

    /// #spec-axis-alpha design §2: a row without an active slot must see the
    /// SHARED vector itself — the same allocation, not an equal copy — so the
    /// fallback path is bit-identical by construction.
    #[test]
    fn a_row_without_an_active_slot_borrows_the_shared_alpha_itself() {
        let state = relu_state();
        let shared = state.alpha("relu").expect("alpha present");
        match state.alpha_for_row("relu", 7).expect("row view") {
            RowAlpha::Base(alpha) => assert!(
                std::ptr::eq(alpha, shared),
                "fallback must borrow the shared vector, not copy it"
            ),
            RowAlpha::Materialized(_) => {
                panic!("no slot table exists; nothing may materialize (#spec-axis-alpha)")
            }
        }
    }

    /// δ = 0 on an ACTIVE slot must reproduce the baseline bit-for-bit —
    /// the parity anchor the whole parameterization rests on (design §2:
    /// `0.0 + x == x` in IEEE for finite x, and clamping an in-range α is
    /// the identity).
    #[test]
    fn a_zero_delta_active_slot_reproduces_the_baseline_bitwise() {
        let mut state = relu_state();
        let width = state.alpha("relu").expect("alpha").len();
        state.spec_slot_rows = vec![3];
        state
            .spec_deltas
            .insert("relu".to_string(), Array2::<f32>::zeros((1, width)));

        let shared = state.alpha("relu").expect("alpha").clone();
        let row_view = state.alpha_for_row("relu", 3).expect("row view");
        let effective = row_view.as_array();
        assert_eq!(effective.len(), shared.len());
        for (index, (effective_bits, shared_bits)) in effective
            .iter()
            .map(|value| value.to_bits())
            .zip(shared.iter().map(|value| value.to_bits()))
            .enumerate()
        {
            assert_eq!(
                effective_bits, shared_bits,
                "δ=0 must be bitwise-identical to α_base at neuron {index} (#iter0-alpha-parity)"
            );
        }
    }

    /// A nonzero δ moves only its own slot's row and stays inside [0,1].
    #[test]
    fn a_nonzero_delta_moves_only_its_slot_and_clamps_into_the_unit_interval() {
        let mut state = relu_state();
        let width = state.alpha("relu").expect("alpha").len();
        state.spec_slot_rows = vec![0, 5];
        let mut deltas = Array2::<f32>::zeros((2, width));
        // Slot 1 (spec row 5): push far past both clamp edges.
        deltas.row_mut(1).fill(2.0);
        deltas[[1, 0]] = -2.0;
        state.spec_deltas.insert("relu".to_string(), deltas);

        let shared = state.alpha("relu").expect("alpha").clone();

        // Row 0 has slot 0 with δ=0: baseline.
        let row0 = state.alpha_for_row("relu", 0).expect("row view");
        assert_eq!(
            row0.as_array()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            shared.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Row 5 (slot 1): clamped to the unit interval.
        let row5 = state.alpha_for_row("relu", 5).expect("row view");
        let effective = row5.as_array();
        assert_eq!(effective[0], 0.0, "α + (-2) clamps to 0");
        for value in effective.iter().skip(1) {
            assert_eq!(*value, 1.0, "α + 2 clamps to 1");
        }

        // A row with NO slot still borrows the baseline.
        match state.alpha_for_row("relu", 9).expect("row view") {
            RowAlpha::Base(_) => {}
            RowAlpha::Materialized(_) => panic!("row 9 owns no slot"),
        }
    }

    /// SPSA perturbation copies run backward passes, which will consult δ
    /// once slice 2 wires the compose — `clone_for_backward` must carry it.
    #[test]
    fn clone_for_backward_carries_spec_deltas() {
        let mut state = relu_state();
        let width = state.alpha("relu").expect("alpha").len();
        state.spec_slot_rows = vec![2];
        state.spec_deltas.insert(
            "relu".to_string(),
            Array2::<f32>::from_elem((1, width), 0.25),
        );

        let copy = state.clone_for_backward();
        assert_eq!(copy.spec_slot_rows, vec![2]);
        assert_eq!(
            copy.spec_deltas.get("relu").map(|d| d[[0, 0]]),
            Some(0.25),
            "perturbation copies must see the same δ the optimizer holds"
        );
    }
}

#[cfg(test)]
mod spec_axis_optimizer_tests {
    use ndarray::{arr1, Array2};

    use super::super::AdamParams;
    use super::*;

    fn relu_state_with_slots(slots: &[usize]) -> GraphAlphaState {
        let pre_activation = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5, -2.0, -0.25]).into_dyn(),
            arr1(&[0.5_f32, 1.5, 1.0, 0.75]).into_dyn(),
        )
        .expect("valid unstable ReLU bounds");
        let mut state = GraphAlphaState::new();
        state
            .add_relu_node("relu", &pre_activation, false)
            .expect("ReLU state should initialize");
        state.spec_slot_rows = slots.to_vec();
        let width = state.alphas["relu"].len();
        state
            .spec_deltas
            .insert("relu".to_string(), Array2::zeros((slots.len(), width)));
        state
    }

    fn params() -> AdamParams {
        AdamParams {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 1,
        }
    }

    /// The projection keeps `α_base + δ` in [0,1]: δ is clamped to
    /// `[-base_i, 1-base_i]`, never a bare unit clamp (design §3).
    #[test]
    fn delta_projection_is_relative_to_the_baseline_not_the_unit_interval() {
        let mut state = relu_state_with_slots(&[0]);
        let base = state.alphas["relu"].clone();
        let width = base.len();
        // Enormous positive gradient drives δ to its ceiling.
        let gradients = Array2::from_elem((1, width), 1.0e6_f32);
        for step in 1..=50 {
            state.update_spec_deltas_adam("relu", &gradients, step, &params());
        }
        let deltas = &state.spec_deltas["relu"];
        for i in 0..width {
            let effective = base[i] + deltas[[0, i]];
            assert!(
                (0.0..=1.0).contains(&effective),
                "α_base + δ must stay in [0,1]; neuron {i}: base={} δ={}",
                base[i],
                deltas[[0, i]]
            );
            // Ceiling is exactly 1 - base (ascent saturates there).
            assert!(
                (effective - 1.0).abs() < 1e-6,
                "sustained ascent must saturate the ceiling at neuron {i}, got {effective}"
            );
        }
    }

    /// A non-finite gradient resets the entry to δ=0 — the BASELINE, never
    /// the 0.5 the shared sanitize uses — and zeroes its moments.
    #[test]
    fn nonfinite_gradient_resets_delta_to_the_baseline_and_clears_moments() {
        let mut state = relu_state_with_slots(&[0]);
        let width = state.alphas["relu"].len();
        // One clean step to build nonzero δ and moments.
        state.update_spec_deltas_adam("relu", &Array2::from_elem((1, width), 1.0), 1, &params());
        assert!(state.spec_deltas["relu"][[0, 0]] != 0.0);
        // Now poison neuron 0.
        let mut gradients = Array2::from_elem((1, width), 1.0_f32);
        gradients[[0, 0]] = f32::NAN;
        state.update_spec_deltas_adam("relu", &gradients, 2, &params());
        assert_eq!(
            state.spec_deltas["relu"][[0, 0]],
            0.0,
            "poisoned entry must return to the parity anchor δ=0"
        );
        assert_eq!(state.spec_adam_m["relu"][[0, 0]], 0.0);
        assert_eq!(state.spec_adam_v["relu"][[0, 0]], 0.0);
        // Neuron 2 has base α = 0 (u < -l), so ascent has real headroom
        // there — neuron 1's base is 1.0 and its δ ceiling is 0, which is
        // the projection doing its job, not a lost update.
        assert!(
            state.spec_deltas["relu"][[0, 2]] != 0.0,
            "a base-0 neuron keeps its clean update"
        );
    }

    /// Shape disagreement between gradients and δ leaves δ untouched —
    /// fail-closed like the read side, never a panic.
    #[test]
    fn malformed_gradient_shapes_leave_delta_untouched() {
        let mut state = relu_state_with_slots(&[0, 3]);
        let width = state.alphas["relu"].len();
        let before = state.spec_deltas["relu"].clone();
        state.update_spec_deltas_adam("relu", &Array2::from_elem((1, width), 1.0), 1, &params());
        assert_eq!(
            state.spec_deltas["relu"], before,
            "a 1-slot gradient against a 2-slot table must be refused"
        );
        state.update_spec_deltas_adam(
            "relu",
            &Array2::from_elem((2, width + 1), 1.0),
            1,
            &params(),
        );
        assert_eq!(state.spec_deltas["relu"], before, "width mismatch refused");
    }
}

#[cfg(test)]
mod spec_slot_reassignment_tests {
    use ndarray::{arr1, Array2};

    use super::*;

    /// Lazy acquisition (#spec-axis-alpha 2e): reassignment moves the slot to
    /// the new row, resets its δ and moments to the parity anchor, and never
    /// touches other slots; a row that already owns a slot is refused.
    #[test]
    fn reassignment_resets_the_slot_to_the_parity_anchor_and_isolates_neighbors() {
        let pre = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 1.5]).into_dyn(),
        )
        .expect("bounds");
        let mut state = GraphAlphaState::new();
        state.add_relu_node("relu", &pre, false).expect("relu");
        state.spec_slot_rows = vec![7, 9];
        state
            .spec_deltas
            .insert("relu".to_string(), Array2::from_elem((2, 2), 0.25));
        state
            .spec_adam_m
            .insert("relu".to_string(), Array2::from_elem((2, 2), 0.5));
        state
            .spec_adam_v
            .insert("relu".to_string(), Array2::from_elem((2, 2), 0.5));

        assert!(!state.reassign_spec_slot(0, 9), "row 9 already owns slot 1");
        assert!(!state.reassign_spec_slot(5, 3), "out-of-range slot refused");
        assert!(state.reassign_spec_slot(0, 3));
        assert_eq!(state.spec_slot_rows, vec![3, 9]);
        let deltas = &state.spec_deltas["relu"];
        assert!(
            deltas.row(0).iter().all(|&d| d == 0.0),
            "incoming row starts clean"
        );
        assert!(
            deltas.row(1).iter().all(|&d| d == 0.25),
            "neighbor slot untouched"
        );
        assert!(state.spec_adam_m["relu"].row(0).iter().all(|&m| m == 0.0));
        assert!(state.spec_adam_v["relu"].row(1).iter().all(|&v| v == 0.5));
    }
}
