// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Seeded GPU suffix acceleration for per-target graph alpha/CROWN backward.

use super::target_backward_patches::resolve_preactivation;
use crate::bounds::patches::PatchesMaterializationPurpose;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::core::{
    extract_relu_gpu_layer_with_alpha, try_extract_single_gpu_layer, GraphNetwork,
    GraphTargetShapeContract, NETWORK_INPUT,
};
use crate::network::CrownMergeAccumulator;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, GpuCrownLayer, GpuCrownSeed, NyError, Result};
use ny_tensor::{add_down_f32, add_up_f32, BoundedTensor, RepairStrategy};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

// ---------------------------------------------------------------------------
// Precomputed GPU suffix plan
// ---------------------------------------------------------------------------

/// One entry in the precomputed GPU suffix plan. Stores the extracted
/// `GpuCrownLayer` for this node and the name of its unary input (or `None`
/// when the input is `NETWORK_INPUT`, i.e. the chain terminates).
struct GpuSuffixPlanNode {
    next_input: Option<String>,
    layer: GpuCrownLayer,
}

/// A target-local plan that records which nodes form GPU-eligible unary tails.
///
/// Built once per `propagate_crown_to_node_core` call in O(N) topological
/// time. The hot backward loop then consults the plan for O(1) eligibility
/// checks instead of re-walking the suffix at every step.
pub(super) struct GpuSuffixPlan {
    entries: HashMap<String, GpuSuffixPlanNode>,
}

pub(super) fn take_only_gpu_layer(mut gpu_layers: Vec<GpuCrownLayer>) -> Option<GpuCrownLayer> {
    if gpu_layers.len() == 1 {
        gpu_layers.pop()
    } else {
        None
    }
}

impl GpuSuffixPlan {
    /// Build the plan by walking `relevant_nodes` in topological (forward)
    /// order. A node is eligible if:
    /// 1. it is unary (exactly one input)
    /// 2. it has no active non-ReLU alpha (S-shaped/Sqrt)
    /// 3. its layer can be extracted to a `GpuCrownLayer`
    /// 4. its unary input is either `NETWORK_INPUT` or already in the plan
    pub(super) fn build(
        relevant_nodes: &[String],
        graph: &GraphNetwork,
        input: &BoundedTensor,
        crown_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
    ) -> Self {
        let mut entries = HashMap::new();

        for node_name in relevant_nodes {
            // Guard 1: no S-shaped/sqrt alpha on GPU
            if suffix_has_active_non_relu_alpha(node_name, alpha_state) {
                continue;
            }

            let Some(node) = graph.nodes.get(node_name) else {
                continue;
            };

            // Guard 2: must be unary
            if node.inputs().len() != 1 {
                continue;
            }

            let Ok(input_name) = node.require_unary_input() else {
                continue;
            };

            // Guard 3: input must be NETWORK_INPUT or already eligible
            let terminates = input_name == NETWORK_INPUT;
            if !terminates && !entries.contains_key(input_name) {
                continue;
            }

            // Guard 4: pre-activation resolution must succeed
            let Ok(pre_activation) =
                resolve_preactivation(input, input_name, crown_bounds, ibp_bounds)
            else {
                continue;
            };

            // Guard 5: GPU layer extraction must succeed
            let mut gpu_layers = Vec::with_capacity(1);
            if try_extract_single_gpu_layer_with_alpha(
                &node.layer,
                pre_activation,
                alpha_state,
                node_name,
                &mut gpu_layers,
            )
            .is_none()
                || gpu_layers.is_empty()
            {
                continue;
            }

            let layer_count = gpu_layers.len();
            let Some(layer) = take_only_gpu_layer(gpu_layers) else {
                debug!(
                    node_name = node_name,
                    layer_count,
                    "Graph alpha GPU suffix skipped: extraction produced unexpected GPU layer count"
                );
                continue;
            };
            entries.insert(
                node_name.clone(),
                GpuSuffixPlanNode {
                    next_input: if terminates {
                        None
                    } else {
                        Some(input_name.to_string())
                    },
                    layer,
                },
            );
        }

        GpuSuffixPlan { entries }
    }

    /// Check if `node_name` is in the plan (O(1)).
    pub(super) fn contains(&self, node_name: &str) -> bool {
        self.entries.contains_key(node_name)
    }

    /// Materialize the full suffix layer vector by following `next_input`
    /// links from `start_node` to the terminal node (whose input is
    /// `NETWORK_INPUT`). Returns `None` if `start_node` is not in the plan.
    pub(super) fn materialize_suffix(&self, start_node: &str) -> Option<Vec<GpuCrownLayer>> {
        if !self.entries.contains_key(start_node) {
            return None;
        }
        let mut layers = Vec::new();
        let mut current = start_node;
        loop {
            let entry = self.entries.get(current)?;
            layers.push(entry.layer.clone());
            match &entry.next_input {
                None => break,
                Some(next) => current = next,
            }
        }
        Some(layers)
    }
}

/// Check if a node has active NON-ReLU alpha (S-shaped or Sqrt).
/// ReLU alpha is handled by the alpha-aware GPU extraction; only S-shaped and
/// Sqrt alpha require CPU fallback (not yet ported to GPU).
fn suffix_has_active_non_relu_alpha(
    node_name: &str,
    alpha_state: Option<&GraphAlphaState>,
) -> bool {
    let Some(alpha_state) = alpha_state else {
        return false;
    };
    alpha_state.monotone_s_shaped_alpha(node_name).is_some()
        || alpha_state.sqrt_alpha(node_name).is_some()
        || alpha_state.reciprocal_alpha(node_name).is_some()
}

/// Extract a GPU layer descriptor for a single node, using alpha-aware ReLU
/// slopes when alpha state is present.
///
/// For ReLU nodes with active alpha, builds the `Activation` descriptor from
/// optimized alpha values rather than the fixed heuristic. All other layer
/// types delegate to [`try_extract_single_gpu_layer`].
fn try_extract_single_gpu_layer_with_alpha(
    layer: &Layer,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphAlphaState>,
    node_name: &str,
    gpu_layers: &mut Vec<GpuCrownLayer>,
) -> Option<()> {
    if let Layer::ReLU(_) = layer {
        if let Some(alpha_state) = alpha_state {
            if let Some((alpha_lower, alpha_upper)) = alpha_state.relu_alpha_pair(node_name) {
                if let Some(unstable_mask) = alpha_state.relu_unstable_mask(node_name) {
                    let pre_l = pre_activation.lower().as_slice()?;
                    let pre_u = pre_activation.upper().as_slice()?;
                    // Expand channel-only alpha to full spatial for GPU extraction.
                    let al_expanded = alpha_state.expand_alpha(node_name, alpha_lower);
                    let au_expanded = alpha_state.expand_alpha(node_name, alpha_upper);
                    let mask_expanded = if alpha_state.spatial_shape(node_name).is_some() {
                        alpha_state.expand_mask(node_name, unstable_mask)
                    } else {
                        unstable_mask.clone()
                    };
                    let gpu_layer = extract_relu_gpu_layer_with_alpha(
                        pre_l,
                        pre_u,
                        al_expanded.as_slice()?,
                        au_expanded.as_slice()?,
                        mask_expanded.as_slice()?,
                    );
                    gpu_layers.push(gpu_layer);
                    return Some(());
                }
            }
        }
    }
    // Not a ReLU with active alpha — fall back to standard extraction.
    try_extract_single_gpu_layer(layer, pre_activation, gpu_layers)
}

/// Copy a seed coefficient array into an `Arc<[f32]>` with exactly ONE
/// full-size copy (#gpu-descriptor-single-copy).
///
/// `Arc::<[f32]>::from(&[f32])` allocates the ref-counted slice and memcpys the
/// contents straight in. The `.iter().copied().collect::<Vec<_>>().into()` this
/// replaces copied twice: it built a whole `Vec` first, and std's
/// `impl From<Vec<T>> for Arc<[T]>` then allocated the `Arc`, copied the `Vec`
/// into it and freed the `Vec`. A seed carries four such arrays — the two
/// `num_specs x current_dim` coefficient matrices and the two per-spec bias
/// vectors — so one dispatch sheds four full-size copies and four
/// allocate/free pairs.
///
/// BIT-IDENTICAL on both branches, and no float is touched either way. ndarray
/// documents `as_slice` as returning `Some` only when the array is contiguous
/// AND in standard order, in which case "the element order in the slice
/// corresponds to the logical order of the array's elements" — i.e. exactly the
/// order `iter()` walked, so the fast branch memcpys the same values into the
/// same positions. Any other layout takes the original iterator path verbatim.
fn arc_from_array<D: ndarray::Dimension>(array: &ndarray::Array<f32, D>) -> Arc<[f32]> {
    match array.as_slice() {
        Some(contiguous) => Arc::from(contiguous),
        None => array.iter().copied().collect::<Vec<f32>>().into(),
    }
}

/// The relation a seeded GPU suffix starts from.
///
/// `bounds` is the target's linear relation over `node_name`'s OUTPUT.
/// `node_output_box` is a sound enclosure of that output, the value the
/// columns of `bounds` multiply. The finisher reads the box only when `bounds`
/// carries certified coefficient error (#gpu-suffix-coeff-err). It then charges
/// the error into the seed's bias over the box, and it refuses when the box is
/// absent or the charge is not finite.
#[derive(Clone, Copy)]
pub(super) struct GpuSuffixSeedRelation<'a> {
    pub(super) bounds: &'a LinearBounds,
    pub(super) node_output_box: Option<&'a BoundedTensor>,
}

/// Finish a per-target graph backward pass on GPU using the precomputed
/// suffix plan. Looks up `node_name` in the plan in O(1) and
/// materializes the cached layer vector if eligible, then runs the existing
/// seeded GPU dispatch path unchanged.
fn try_finish_target_gpu_suffix_with_plan(
    input: &BoundedTensor,
    node_name: &str,
    relation: GpuSuffixSeedRelation<'_>,
    plan: &GpuSuffixPlan,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    target_contract: &GraphTargetShapeContract,
) -> Result<Option<BoundedTensor>> {
    // Building the host seed below scans and copies every coefficient and both
    // input endpoints before the GPU call.  That legacy preparation is neither
    // fallible nor cooperatively pollable, so a backend's own deadline support
    // is insufficient to authorize this shortcut.  Keep finite requests on the
    // CPU path, whose materialization/reduction kernels observe the same
    // absolute deadline.  The ordinary no-deadline GPU suffix is unchanged.
    // #gpu-suffix-expiry set-mate: decided by `PATCHES_FINITE_EXPIRY`, which
    // SHIPS ARMED (`DefaultSpec::Bool(true)`) — refusal is by EXPIRY in every
    // scored run; `NY_PATCHES_FINITE_EXPIRY=0` restores the presence refusal.
    if let Some(limit) = deadline {
        if Instant::now() >= limit {
            return Err(NyError::DeadlineExceeded(
                "Graph alpha GPU suffix deadline expired before host seed preparation".into(),
            ));
        }
        if crate::sound_gpu_gate::gpu_suffix_declines_under_finite_authority(limit) {
            return Ok(None);
        }
    }

    // Soundness gate (#vnncomp-gpu-crown-soundness): under the gate, route to the
    // SOUND seeded GPU-resident backward when available; else CPU sound fallback.
    let Some((gpu, use_sound)) =
        crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline)
    else {
        return Ok(None);
    };

    // O(1) plan lookup — ineligible nodes bail immediately.
    if !plan.contains(node_name) {
        return Ok(None);
    }

    // #gpu-suffix-coeff-err (SOUNDNESS: ACCOUNTS, else REFUSES).
    //
    // `GpuCrownSeed` transports only the midpoint relation and has no field for the
    // certified per-coefficient error planes a `LinearBounds` may carry;
    // `crown_backward_gpu_seeded_sound` treats the seed as exact. Seeding an
    // err-carrying relation unchanged would publish a bound TIGHTER than the
    // relation justifies (the defect found 2026-08-29).
    //
    // A blanket refusal is not the fix either: alpha-aware ReLU relaxations carry
    // coefficient error by construction, so refusing every err-carrying seed turns
    // the alpha-aware GPU suffix off on a CUDA host (gpu_suffix_tests' alpha-aware
    // `Activation` assertions pin that capability).
    //
    // So the error is CHARGED before the seed is built. For true coefficients
    // `Ã ∈ [A − E, A + E]` and `z` in this node's output box,
    // `Ã·z ∈ A·z ± Σ_j E_ij·max(|z_lo_j|, |z_hi_j|)`; `fold_coeff_err_over_box_eager`
    // moves that penalty outward into the bias with directed rounding and clears
    // the error. The relation the device receives is then EXACT over every
    // reachable `z`, which is precisely what the seed contract assumes. This is the
    // `ACCOUNTS` class of `host_side_gpu_crown_seed_construction_sites_are_pinned`.
    // Rows whose penalty is non-finite stay carried by the fold, and a missing box
    // cannot pay at all: both REFUSE before any host seed copy, so the CPU backward
    // (which carries and discharges the error itself) takes the node.
    let charged_seed_relation;
    let node_lb = if relation.bounds.has_coeff_err() {
        let Some(node_box) = relation.node_output_box else {
            debug!(
                node_name = node_name,
                "Graph alpha GPU suffix skipped: relation carries certified coefficient error \
                 and no node-output box is available to charge it"
            );
            return Ok(None);
        };
        let mut staged = relation.bounds.clone();
        staged.fold_coeff_err_over_box_eager(node_box);
        if staged.has_coeff_err() {
            debug!(
                node_name = node_name,
                "Graph alpha GPU suffix skipped: certified coefficient error could not be \
                 charged into a finite seed bias over the node-output box"
            );
            return Ok(None);
        }
        charged_seed_relation = staged;
        &charged_seed_relation
    } else {
        relation.bounds
    };

    if node_lb.lower_a().iter().any(|value| !value.is_finite())
        || node_lb.upper_a().iter().any(|value| !value.is_finite())
        || node_lb.lower_b().iter().any(|value| !value.is_finite())
        || node_lb.upper_b().iter().any(|value| !value.is_finite())
    {
        debug!(
            node_name = node_name,
            "Graph alpha GPU suffix skipped: non-finite seed coefficients or bias"
        );
        return Ok(None);
    }

    let Some(gpu_layers) = plan.materialize_suffix(node_name) else {
        return Ok(None);
    };

    let seed = GpuCrownSeed {
        lower_a: arc_from_array(node_lb.lower_a()),
        upper_a: arc_from_array(node_lb.upper_a()),
        lower_b: arc_from_array(node_lb.lower_b()),
        upper_b: arc_from_array(node_lb.upper_b()),
        num_specs: node_lb.num_outputs(),
        current_dim: node_lb.num_inputs(),
    };
    let input_lower: Vec<f32> = input.lower().iter().copied().collect();
    let input_upper: Vec<f32> = input.upper().iter().copied().collect();

    let seeded = if use_sound {
        gpu.crown_backward_gpu_seeded_sound(&gpu_layers, &seed, &input_lower, &input_upper)
    } else {
        gpu.crown_backward_gpu_seeded(&gpu_layers, &seed, &input_lower, &input_upper)
    };
    let gpu_result = match seeded {
        Ok(result) => result,
        Err(error) => {
            debug!(
                node_name = node_name,
                error = %error,
                "Graph alpha GPU suffix failed; falling back to CPU backward"
            );
            return Ok(None);
        }
    };

    if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&gpu_result, seed.num_specs) {
        debug!(
            node_name = node_name,
            "Graph alpha GPU suffix produced malformed bounds; falling back to CPU backward"
        );
        return Ok(None);
    }

    let (Ok(lower), Ok(upper)) = (
        ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.lower_bounds),
        ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.upper_bounds),
    ) else {
        return Ok(None);
    };
    let Some(bounds) = BoundedTensor::new(lower, upper).ok() else {
        return Ok(None);
    };
    let Some(bounds) = target_contract
        .restore_concrete(bounds, "Graph alpha-CROWN GPU suffix restore")
        .ok()
    else {
        return Ok(None);
    };
    Ok(Some(bounds))
}

#[allow(clippy::too_many_arguments)] // extends try_finish_target_gpu_suffix with merge accumulator
pub(super) fn try_finish_target_gpu_suffix_with_pending_input(
    input: &BoundedTensor,
    node_name: &str,
    relation: GpuSuffixSeedRelation<'_>,
    plan: &GpuSuffixPlan,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    target_contract: &GraphTargetShapeContract,
    node_crown_bounds: &mut CrownMergeAccumulator,
) -> Result<Option<BoundedTensor>> {
    let has_only_pending_input_contribution = node_crown_bounds.has_only_key(NETWORK_INPUT);
    if !node_crown_bounds.is_empty() && !has_only_pending_input_contribution {
        return Ok(None);
    }

    let Some(mut bounds) = try_finish_target_gpu_suffix_with_plan(
        input,
        node_name,
        relation,
        plan,
        engine,
        deadline,
        target_contract,
    )?
    else {
        return Ok(None);
    };

    if has_only_pending_input_contribution {
        let pending_input_contribution =
            node_crown_bounds.take(NETWORK_INPUT)?.ok_or_else(|| {
                NyError::InvalidSpec(
                    "Graph alpha GPU suffix expected pending input contribution".to_string(),
                )
            })?;
        let pending_input_contribution = target_contract.restore_concrete(
            pending_input_contribution
                .into_dense_with_deadline_for_purpose(
                    deadline,
                    PatchesMaterializationPurpose::NetworkInputTerminal,
                )?
                .concretize_sound(input),
            "Graph alpha-CROWN pending input contribution restore",
        )?;
        bounds = add_concrete_bounds(
            bounds,
            &pending_input_contribution,
            "Graph alpha-CROWN pending input contribution merge",
        )?;
    }

    Ok(Some(bounds))
}

pub(super) fn add_concrete_bounds(
    lhs: BoundedTensor,
    rhs: &BoundedTensor,
    context: &'static str,
) -> Result<BoundedTensor> {
    if lhs.shape() != rhs.shape() {
        return Err(NyError::shape_mismatch(
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }

    // DIRECTED. This is the SAME operation as `BoundedTensor::add`
    // (ny-tensor/src/bounded_tensor/ops.rs) — an element-wise add of two
    // bounded tensors — and it must round the same way. A plain `+` on the two
    // arrays is IEEE round-to-NEAREST, so `lhs_lo + rhs_lo` can land up to half
    // an ULP ABOVE the true sum (a lower bound that excludes the value it
    // bounds) and `lhs_hi + rhs_hi` up to half an ULP BELOW it. Nothing
    // downstream recovers that half ULP: `new_repaired(.., Conservative)` below
    // repairs NaN and fixes inversions, and never widens a finite endpoint.
    //
    // BOTH halves move outward — lower rounds DOWN, upper rounds UP.
    // `add_down_f32`/`add_up_f32` step only when the addition was inexact, so
    // an exactly-representable sum is bit-identical to the old `+`; and both
    // preserve the ±Inf/NaN behaviour the repair policy below documents
    // (`-inf + finite` stays `-inf`, `inf + (-inf)` stays NaN and is widened).
    let lower_sum = ndarray::Zip::from(lhs.lower())
        .and(rhs.lower())
        .map_collect(|&x, &y| add_down_f32(x, y));
    let upper_sum = ndarray::Zip::from(lhs.upper())
        .and(rhs.upper())
        .map_collect(|&x, &y| add_up_f32(x, y));

    // Both operands can legitimately contain ±Inf (GPU suffix uses Widen strategy,
    // concretize_sound uses new_allow_infinite). The sum can produce:
    // - Inf + finite = Inf (valid conservative bound)
    // - Inf + (-Inf) = NaN (proves nothing, must be repaired)
    // Conservative strategy repairs NaN → -inf (lower) / +inf (upper) and passes
    // ±Inf endpoints and finite values through unchanged. A non-finite endpoint
    // carries no proven bound in that direction, and any finite substitute
    // (FALLBACK_BOUND included) is a strict subset of that unbounded interval —
    // an unsound tightening — so the repair must widen, never clamp.
    let non_finite_count = lower_sum.iter().filter(|v| !v.is_finite()).count()
        + upper_sum.iter().filter(|v| !v.is_finite()).count();
    if non_finite_count > 0 {
        debug!(
            non_finite_count,
            context,
            "add_concrete_bounds: {non_finite_count} non-finite sum endpoints (NaN widened to ±inf, ±Inf preserved)"
        );
    }
    BoundedTensor::new_repaired(lower_sum, upper_sum, RepairStrategy::Conservative)
}

/// Pins for [`arc_from_array`] (#gpu-descriptor-single-copy).
#[cfg(test)]
mod seed_buffer_tests {
    use super::arc_from_array;
    use ndarray::{Array2, ShapeBuilder};

    /// `arc_from_array` reproduces the array's LOGICAL (row-major) element
    /// order on both of its branches.
    ///
    /// Fails closed on the branch that matters: the fast branch is only correct
    /// because `as_slice` refuses anything but standard layout. Widening it to
    /// the underlying buffer — `as_slice_memory_order`, the neighbouring
    /// accessor — would hand the column-major array back in storage order,
    /// `[1, 4, 2, 5, 3, 6]`, and this test would fail on the second
    /// `assert_eq!`. The `as_slice().is_none()` assertion proves the fallback
    /// is really the path under test rather than the fast branch again.
    #[test]
    fn arc_from_array_preserves_logical_order_on_both_branches() {
        let row_major = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("row-major shape should be valid");
        assert!(
            row_major.as_slice().is_some(),
            "row-major owned array must take the single-copy branch"
        );
        assert_eq!(
            arc_from_array(&row_major).as_ref(),
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // The same logical matrix held column-major.
        let column_major =
            Array2::<f32>::from_shape_vec((2, 3).f(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
                .expect("column-major shape should be valid");
        assert_eq!(column_major, row_major, "fixtures must be the same matrix");
        assert!(
            column_major.as_slice().is_none(),
            "column-major array must fall back to the iterator branch"
        );
        assert_eq!(
            arc_from_array(&column_major).as_ref(),
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}

/// Soundness pins for the two host-side seams in this file
/// (#gpu-suffix-coeff-err, and the directed-rounding half of the concrete
/// merge). Both drive the PRODUCTION wiring, not a kernel with
/// self-consistent inputs.
#[cfg(test)]
mod gpu_suffix_soundness_tests {
    use super::*;
    use ndarray::Array2;
    use ny_core::{CertifiedWeightError, GpuCrownBackward, GpuCrownResult, NaiveCpuGemmEngine};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const DIM: usize = 2;
    const NODE: &str = "gpu_suffix_seed_node";

    /// A GPU CROWN backward that answers every seeded dispatch with a canned,
    /// publishable interval, counts how many times it was asked, and keeps the
    /// last seed it was handed.
    ///
    /// The COUNT is the observable for a refusal: a refusal that happens before
    /// dispatch is distinguishable from every other reason the finisher can
    /// return `Ok(None)` only by the dispatch never occurring. The SEED is the
    /// observable for a charge: the device treats it as exact, so whatever
    /// coefficient error the relation carried must already be in its bias.
    struct CountingSeedEngine {
        seeded_calls: AtomicUsize,
        last_seed: Mutex<Option<GpuCrownSeed>>,
        rows: usize,
    }

    impl CountingSeedEngine {
        fn new(rows: usize) -> Self {
            Self {
                seeded_calls: AtomicUsize::new(0),
                last_seed: Mutex::new(None),
                rows,
            }
        }

        fn seeded_calls(&self) -> usize {
            self.seeded_calls.load(Ordering::SeqCst)
        }

        fn last_seed(&self) -> GpuCrownSeed {
            self.last_seed
                .lock()
                .expect("seed recorder lock")
                .clone()
                .expect("a seeded dispatch recorded its seed")
        }

        fn canned(&self, seed: &GpuCrownSeed) -> Result<GpuCrownResult> {
            self.seeded_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_seed.lock().expect("seed recorder lock") = Some(seed.clone());
            Ok(GpuCrownResult {
                lower_bounds: vec![-1.0; self.rows],
                upper_bounds: vec![1.0; self.rows],
            })
        }
    }

    impl GemmEngine for CountingSeedEngine {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for CountingSeedEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            panic!("the seeded suffix finisher must never take the unseeded dispatch");
        }

        fn crown_backward_gpu_seeded(
            &self,
            _layers: &[GpuCrownLayer],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            self.canned(seed)
        }

        /// Both dispatch routes are covered, and both increment the SAME
        /// counter, so the test does not depend on the state of the
        /// process-global sound-GPU gate (which selects between them).
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn crown_backward_gpu_seeded_sound(
            &self,
            _layers: &[GpuCrownLayer],
            seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            self.canned(seed)
        }
    }

    /// A one-node plan whose suffix terminates at `NETWORK_INPUT`, built
    /// directly so the test exercises the finisher rather than plan discovery.
    fn one_node_plan() -> GpuSuffixPlan {
        let mut entries = HashMap::new();
        entries.insert(
            NODE.to_string(),
            GpuSuffixPlanNode {
                next_input: None,
                layer: GpuCrownLayer::Linear {
                    weight: Arc::<[f32]>::from(vec![1.0_f32, 0.0, 0.0, 1.0]),
                    bias: None,
                    out_features: DIM,
                    in_features: DIM,
                    cert_err: CertifiedWeightError::EXACT,
                },
            },
        );
        GpuSuffixPlan { entries }
    }

    fn unit_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[DIM]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[DIM]), 1.0_f32),
        )
        .expect("unit input box")
    }

    /// #gpu-suffix-coeff-err: the finisher never hands the device an
    /// err-carrying relation as if it were exact. It CHARGES the certified
    /// coefficient error into the seed's bias over the node's own output box,
    /// and REFUSES before dispatch when that charge cannot be paid.
    ///
    /// This drives the production entry point with a real plan and a real
    /// engine. The arms differ only in `set_coeff_err` and in the node-output
    /// box the caller supplies:
    ///
    /// - CONTROL: an exact relation reaches the device with its bias untouched,
    ///   which proves the dispatch is genuinely reachable in this fixture, so a
    ///   `None` in a refusal arm cannot be an artifact of some other guard.
    /// - REFUSES (no box): fails if the refusal is removed, because the device
    ///   would then be seeded with the unpaid relation (`seeded_calls()` 1).
    /// - ACCOUNTS: fails if the charge is removed, because the seed's bias
    ///   would stay 0 where `-Σ_j E_ij·max(|z_lo_j|, |z_hi_j|)` is required. The
    ///   node box `[-2, 2]` is wider than the network input `[-1, 1]`, so a
    ///   charge taken over the wrong box (the input) also fails.
    /// - REFUSES (box cannot pay): an unbounded box leaves the error carried,
    ///   and a box of the wrong width cannot be mapped onto the columns.
    ///
    /// History: the 2026-08-29 version of this test pinned an unconditional
    /// refusal and was excluded from every run, because that refusal turned the
    /// alpha-aware GPU suffix off (alpha-aware ReLU relaxations carry
    /// coefficient error by construction). Charging over the node box keeps
    /// the suffix and is the `ACCOUNTS` class of
    /// `host_side_gpu_crown_seed_construction_sites_are_pinned`.
    #[test]
    fn gpu_suffix_finisher_charges_seed_coeff_err_over_the_node_output_box_or_refuses() {
        let input = unit_box();
        let plan = one_node_plan();
        let contract = GraphTargetShapeContract::from_bounds(NODE, &input);
        let finish = |relation: GpuSuffixSeedRelation<'_>, engine: &CountingSeedEngine| {
            try_finish_target_gpu_suffix_with_plan(
                &input,
                NODE,
                relation,
                &plan,
                Some(engine),
                None,
                &contract,
            )
            .expect("the finisher answers Ok: a GPU result or a CPU fallback, never an error")
        };

        // CONTROL — an exact relation reaches the device, bias untouched.
        let exact = LinearBounds::identity(DIM);
        assert!(
            !exact.has_coeff_err(),
            "control precondition: the identity relation is exact"
        );
        let control_engine = CountingSeedEngine::new(DIM);
        let control = finish(
            GpuSuffixSeedRelation {
                bounds: &exact,
                node_output_box: None,
            },
            &control_engine,
        );
        assert!(
            control.is_some(),
            "control: the fixture must actually reach the seeded GPU dispatch, \
             otherwise the refusal arms below prove nothing"
        );
        assert_eq!(
            control_engine.seeded_calls(),
            1,
            "control: exactly one seeded dispatch was expected"
        );
        let control_seed = control_engine.last_seed();
        assert_eq!(&control_seed.lower_b[..], &[0.0_f32; DIM][..]);
        assert_eq!(&control_seed.upper_b[..], &[0.0_f32; DIM][..]);

        // The err-carrying relation: identity coefficients, certified error E.
        let err_entry = 1.0e-3_f32;
        let mut err_carrying = LinearBounds::identity(DIM);
        let err = Array2::<f32>::from_elem((DIM, DIM), err_entry);
        err_carrying.set_coeff_err(err.clone(), err);
        assert!(
            err_carrying.has_coeff_err(),
            "precondition: the relation now carries a certified enclosure"
        );

        // REFUSES — no node-output box, so the charge cannot be computed.
        let unboxed_engine = CountingSeedEngine::new(DIM);
        let unboxed = finish(
            GpuSuffixSeedRelation {
                bounds: &err_carrying,
                node_output_box: None,
            },
            &unboxed_engine,
        );
        assert!(
            unboxed.is_none(),
            "an err-carrying relation with no node-output box must fall back to the CPU \
             backward: the device seed has no channel for `lower_a_err`/`upper_a_err` and \
             treats the coefficients it receives as exact"
        );
        assert_eq!(
            unboxed_engine.seeded_calls(),
            0,
            "the refusal must happen BEFORE dispatch — a device that has already been \
             handed an under-charged seed has already computed a bound tighter than the \
             relation justifies"
        );

        // ACCOUNTS — the charge is paid into the bias over the node box.
        const NODE_MAG: f32 = 2.0;
        let node_box = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[DIM]), -NODE_MAG),
            ArrayD::from_elem(IxDyn(&[DIM]), NODE_MAG),
        )
        .expect("node-output box");
        let charged_engine = CountingSeedEngine::new(DIM);
        let charged = finish(
            GpuSuffixSeedRelation {
                bounds: &err_carrying,
                node_output_box: Some(&node_box),
            },
            &charged_engine,
        );
        assert!(
            charged.is_some(),
            "a payable charge keeps the GPU suffix: refusing here is the capability loss \
             the charge exists to avoid"
        );
        assert_eq!(charged_engine.seeded_calls(), 1);
        let seed = charged_engine.last_seed();
        assert_eq!(
            &seed.lower_a[..],
            exact.lower_a().as_slice().expect("contiguous identity"),
            "the charge moves the bias only; the nominal coefficients are unchanged"
        );
        assert_eq!(
            &seed.upper_a[..],
            exact.upper_a().as_slice().expect("contiguous identity"),
            "the charge moves the bias only; the nominal coefficients are unchanged"
        );
        // Σ_j E_ij·max(|z_lo_j|, |z_hi_j|) in exact f64 (every term is exact).
        let penalty = DIM as f64 * f64::from(err_entry) * f64::from(NODE_MAG);
        for row in 0..DIM {
            let lower = f64::from(seed.lower_b[row]);
            let upper = f64::from(seed.upper_b[row]);
            assert!(
                lower <= -penalty,
                "row {row}: seed lower bias {lower:e} does not pay the certified charge \
                 {penalty:e} over the node box, so the exact-seed contract is violated"
            );
            assert!(
                upper >= penalty,
                "row {row}: seed upper bias {upper:e} does not pay the certified charge \
                 {penalty:e} over the node box"
            );
            // The charge is the box penalty plus directed rounding, not a
            // conservative blow-up that would make this route useless.
            assert!(
                lower >= -penalty * (1.0 + 1.0e-6) && upper <= penalty * (1.0 + 1.0e-6),
                "row {row}: charge [{lower:e}, {upper:e}] is far wider than the penalty \
                 {penalty:e}"
            );
        }

        // REFUSES — the box cannot pay (unbounded) or cannot be mapped (width).
        let unbounded_box = BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[DIM]), f32::NEG_INFINITY),
            ArrayD::from_elem(IxDyn(&[DIM]), NODE_MAG),
        )
        .expect("unbounded node-output box");
        let wrong_width_box = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[DIM + 1]), -NODE_MAG),
            ArrayD::from_elem(IxDyn(&[DIM + 1]), NODE_MAG),
        )
        .expect("wrong-width node-output box");
        for (label, unpayable) in [
            ("unbounded", &unbounded_box),
            ("wrong-width", &wrong_width_box),
        ] {
            let engine = CountingSeedEngine::new(DIM);
            let refused = finish(
                GpuSuffixSeedRelation {
                    bounds: &err_carrying,
                    node_output_box: Some(unpayable),
                },
                &engine,
            );
            assert!(
                refused.is_none(),
                "{label} box: an unpayable charge must fall back to the CPU backward"
            );
            assert_eq!(
                engine.seeded_calls(),
                0,
                "{label} box: the refusal must happen before dispatch"
            );
        }
    }

    /// `add_concrete_bounds` rounds each endpoint OUTWARD, like its sibling
    /// `BoundedTensor::add`.
    ///
    /// Fixture: `d = 3 * 2^-25` is exactly three quarters of an ULP at 1.0, so
    /// round-to-nearest carries `1.0 + d` UP to the next float — strictly above
    /// the true sum, which is exactly the lower bound that excludes the value
    /// it bounds. The negated pair does the mirror image to the upper endpoint.
    /// Both preconditions are asserted, so this cannot silently degrade into a
    /// test of exactly-representable sums.
    #[test]
    fn add_concrete_bounds_rounds_both_endpoints_outward() {
        // 3 * 2^-25 = 1.5 * 2^-24, exactly representable in f32.
        const THREE_QUARTER_ULP: f32 = 8.940_697e-8;
        assert_eq!(
            THREE_QUARTER_ULP.to_bits(),
            0x33C0_0000,
            "fixture must be exactly 3 * 2^-25"
        );

        let exact_lower = 1.0_f64 + f64::from(THREE_QUARTER_ULP);
        let exact_upper = -1.0_f64 - f64::from(THREE_QUARTER_ULP);
        assert!(
            f64::from(1.0_f32 + THREE_QUARTER_ULP) > exact_lower,
            "precondition: plain round-to-nearest overshoots this lower sum"
        );
        assert!(
            f64::from(-1.0_f32 - THREE_QUARTER_ULP) < exact_upper,
            "precondition: plain round-to-nearest undershoots this upper sum"
        );

        // Index 0 exercises the lower endpoint, index 1 the upper one.
        let lhs = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, -1.0])
                .expect("lhs lower shape is valid"),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, -1.0])
                .expect("lhs upper shape is valid"),
        )
        .expect("lhs is a valid degenerate interval");
        let rhs = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![THREE_QUARTER_ULP, -THREE_QUARTER_ULP])
                .expect("rhs lower shape is valid"),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![THREE_QUARTER_ULP, -THREE_QUARTER_ULP])
                .expect("rhs upper shape is valid"),
        )
        .expect("rhs is a valid degenerate interval");

        let merged = add_concrete_bounds(lhs, &rhs, "directed rounding pin")
            .expect("the merge must succeed on finite operands");
        let lower: Vec<f32> = merged.lower().iter().copied().collect();
        let upper: Vec<f32> = merged.upper().iter().copied().collect();

        assert!(
            f64::from(lower[0]) <= exact_lower,
            "lower endpoint must round DOWN: got {} > true sum {}",
            lower[0],
            exact_lower
        );
        assert!(
            f64::from(upper[1]) >= exact_upper,
            "upper endpoint must round UP: got {} < true sum {}",
            upper[1],
            exact_upper
        );
    }
}
