// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fast CROWN propagation paths.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::layers::Layer;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, instrument, warn};

use super::{crown_backward_step_patches, CrownStepResult, Network};

fn initialize_fast_crown_bounds(
    has_conv2d: bool,
    output_shape: &[usize],
    output_dim: usize,
    label: &str,
) -> Result<Option<CrownBounds>> {
    if has_conv2d && output_shape.len() == 3 {
        let spatial = (output_shape[0], output_shape[1], output_shape[2]);
        debug!("{label}: Patches mode — 3D spatial output {:?}", spatial);
        return Ok(Some(CrownBounds::Patches(Box::new(
            PatchesLinearBounds::identity(spatial, spatial),
        ))));
    }
    if let Some(estimate) =
        super::dense_identity_budget_estimate("initial_dense_identity", output_dim)
    {
        super::log_dense_materialization_budget_fallback(label, estimate, None, None);
        return Ok(None);
    }
    Ok(Some(CrownBounds::Dense(LinearBounds::identity(output_dim))))
}

impl Network {
    #[inline]
    fn ibp_fallback_with_constant_linear(
        &self,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, LinearBounds)> {
        let concrete = self.propagate_ibp(input)?;
        let output_flat = concrete.flatten();
        let output_dim = output_flat.len();
        let input_dim = input.len();

        let lower_b = ndarray::Array1::from_vec(output_flat.lower().iter().copied().collect());
        let upper_b = ndarray::Array1::from_vec(output_flat.upper().iter().copied().collect());

        let linear_bounds = LinearBounds::new_or_conservative(
            ndarray::Array2::zeros((output_dim, input_dim)),
            lower_b,
            ndarray::Array2::zeros((output_dim, input_dim)),
            upper_b,
        )?;
        Ok((concrete, linear_bounds))
    }

    /// Propagate bounds using fast CROWN with IBP intermediate bounds.
    ///
    /// This method is 3-10x faster than standard CROWN by using simple IBP bounds
    /// for intermediate layers instead of running CROWN-IBP tightening passes.
    /// The tradeoff is slightly looser bounds, but this is often acceptable for:
    /// - Initial bound computation before α-optimization
    /// - Cases where verification succeeds easily
    /// - Performance-critical code paths
    ///
    /// # REQUIRES
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    ///
    /// # ENSURES
    /// - Output bounds contain all possible network outputs for inputs in `input`
    /// - Bounds may be looser than `propagate_crown()` but computation is 3-10x faster
    /// - Soundness: for any `x` where `input.contains(x)`, `output.contains(network(x))`
    ///
    /// Algorithm:
    /// 1. Run IBP forward to collect intermediate bounds (fast)
    /// 2. Run CROWN backward using IBP bounds for ReLU relaxation
    /// 3. Concretize final linear bounds using input bounds
    /// 4. Intersect with IBP forward bounds to ensure output is at least as tight as IBP (#2990)
    #[inline]
    #[instrument(skip(self, input), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    pub fn propagate_crown_fast(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
            ));
        }

        // Step 1: Collect IBP intermediate bounds (fast, no CROWN-IBP overhead)
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "CROWN-fast: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize CROWN bounds — use Patches mode for CNN networks
        // with 3D spatial output. Same logic as propagate_crown_with_engine.
        // Part of #2613: extends Patches optimization to the fast CROWN path.
        let has_conv2d = self.layers.iter().any(|l| matches!(l, Layer::Conv2d(_)));
        let mut crown_bounds = match initialize_fast_crown_bounds(
            has_conv2d,
            &output_shape,
            output_dim,
            "CROWN-fast",
        )? {
            Some(bounds) => bounds,
            None => return self.propagate_ibp(input),
        };

        // Step 3: Propagate backward through each layer using Patches dispatch
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match crown_backward_step_patches(
                layer,
                &mut crown_bounds,
                pre_activation,
                None,
                i,
                "CROWN-fast",
                None, // no deadline in CROWN-fast path
            )? {
                CrownStepResult::Continue => {}
                CrownStepResult::IbpFallback(fallback) => {
                    warn!("CROWN-fast: {}", fallback.details);
                    return self.propagate_ibp(input);
                }
            }
        }
        // Step 4: Convert to Dense (no-op when already Dense) and concretize
        if let Some(estimate) =
            super::dense_materialization_budget_estimate(&crown_bounds, "final_concretization")?
        {
            super::log_dense_materialization_budget_fallback("CROWN-fast", estimate, None, None);
            return self.propagate_ibp(input);
        }
        let linear_bounds = crown_bounds.into_dense()?;
        let concrete_bounds = linear_bounds.concretize_sound(input);
        let concrete_bounds = concrete_bounds.reshape(&output_shape)?;
        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        super::tighten_crown_output(concrete_bounds, output_bounds, "CROWN-fast")
    }

    /// Propagate fast CROWN returning (concrete_bounds, linear_bounds) at input.
    ///
    /// Used by Clip-and-Verify which needs linear coefficients for input tightening.
    /// Delegates to `propagate_crown_with_linear_and_engine` with `engine: None`.
    pub fn propagate_crown_with_linear(
        &self,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, LinearBounds)> {
        self.propagate_crown_with_linear_and_engine(input, None)
    }

    /// Fast CROWN with linear, with optional GemmEngine for GPU backward (#3598).
    pub fn propagate_crown_with_linear_and_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, LinearBounds)> {
        if self.layers.is_empty() {
            let dim = input.len();
            return Ok((input.clone(), LinearBounds::identity(dim)));
        }
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
            ));
        }

        // Step 1: Collect IBP intermediate bounds
        let layer_bounds = self.collect_ibp_bounds(input)?;
        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        // Step 2: Initialize CROWN bounds — use Patches mode for CNN networks
        // with 3D spatial output. Part of #2613: extends Patches to fast-with-linear.
        let has_conv2d = self.layers.iter().any(|l| matches!(l, Layer::Conv2d(_)));
        let mut crown_bounds = match initialize_fast_crown_bounds(
            has_conv2d,
            &output_shape,
            output_dim,
            "fast CROWN with linear",
        )? {
            Some(bounds) => bounds,
            None => return self.ibp_fallback_with_constant_linear(input),
        };

        // Step 3: Propagate backward through each layer using Patches dispatch.
        // Uses crown_backward_step_patches which handles Conv2d Patches, activation
        // Patches, pooling Patches, and Dense fallback for all other layers.
        // This replaces the previous per-layer match arms, deduplicating dispatch.
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match crown_backward_step_patches(
                layer,
                &mut crown_bounds,
                pre_activation,
                engine,
                i,
                "fast CROWN with linear",
                None, // no deadline in fast-CROWN-with-linear path
            )? {
                CrownStepResult::Continue => {}
                CrownStepResult::IbpFallback(fallback) => {
                    warn!("fast CROWN with linear: {}", fallback.details);
                    return self.ibp_fallback_with_constant_linear(input);
                }
            }
        }

        // Step 4: Convert to Dense (no-op when already Dense) and concretize.
        // LinearBounds are always needed for the return value (callers use
        // the A-matrices for Clip-and-Verify input tightening).
        if let Some(estimate) =
            super::dense_materialization_budget_estimate(&crown_bounds, "final_concretization")?
        {
            super::log_dense_materialization_budget_fallback(
                "fast CROWN with linear",
                estimate,
                None,
                None,
            );
            return self.ibp_fallback_with_constant_linear(input);
        }
        let linear_bounds = crown_bounds.into_dense()?;
        let concrete = linear_bounds
            .concretize_sound(input)
            .reshape(&output_shape)?;
        // Tightening heuristic: prefer IBP if CROWN degraded to non-finite bounds (#2287).
        // Divergent fallback — returns (BoundedTensor, LinearBounds), so cannot use
        // tighten_crown_output directly for this branch.
        if super::has_degraded_bounds(&concrete) {
            warn!(
                "fast CROWN with linear: falling back to IBP — output contains non-finite bounds"
            );
            return self.ibp_fallback_with_constant_linear(input);
        }

        // Step 5: Intersect concrete bounds with IBP forward bounds (#3043 dedup).
        // Note: linear_bounds are returned unmodified — they represent the A matrices
        // and biases of the linear relaxation, not the concrete interval bounds.
        let concrete =
            super::tighten_crown_output(concrete, output_bounds, "fast CROWN with linear")?;

        Ok((concrete, linear_bounds))
    }
}

#[cfg(test)]
mod tests {
    use super::Network;
    use crate::bounds::LinearBounds;
    use crate::layers::{
        GatherLayer, Layer, LinearLayer, ReduceMeanLayer, ReduceSumLayer, SkipMergeLayer,
    };
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use ny_core::Result;
    use ny_tensor::BoundedTensor;

    /// Assert linear bounds A-matrix coefficients match expected 2-row block-diagonal pattern.
    fn assert_block_diagonal_coefficients(
        lb: &LinearBounds,
        cols: usize,
        split: usize,
        scale: f32,
        label: &str,
    ) {
        for col in 0..cols {
            let exp0 = if col < split { scale } else { 0.0 };
            let exp1 = if col >= split { scale } else { 0.0 };
            assert!(
                (lb.lower_a[[0, col]] - exp0).abs() < 1e-6,
                "{label} lower_a[0,{col}]: expected {exp0}, got {}",
                lb.lower_a[[0, col]]
            );
            assert!(
                (lb.upper_a[[0, col]] - exp0).abs() < 1e-6,
                "{label} upper_a[0,{col}]: expected {exp0}, got {}",
                lb.upper_a[[0, col]]
            );
            assert!(
                (lb.lower_a[[1, col]] - exp1).abs() < 1e-6,
                "{label} lower_a[1,{col}]: expected {exp1}, got {}",
                lb.lower_a[[1, col]]
            );
            assert!(
                (lb.upper_a[[1, col]] - exp1).abs() < 1e-6,
                "{label} upper_a[1,{col}]: expected {exp1}, got {}",
                lb.upper_a[[1, col]]
            );
        }
        assert!(
            lb.lower_b.iter().all(|&b| b.abs() < 1e-6),
            "{label} lower_b not near zero: {:?}",
            lb.lower_b
        );
        assert!(
            lb.upper_b.iter().all(|&b| b.abs() < 1e-6),
            "{label} upper_b not near zero: {:?}",
            lb.upper_b
        );
    }

    #[test]
    fn propagate_crown_with_linear_smoke() -> Result<()> {
        let weight = arr2(&[[2.0f32]]);
        let bias = arr1(&[1.0f32]);
        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias))?));

        let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
        let (output, linear_bounds) = network.propagate_crown_with_linear(&input)?;

        assert_eq!(output.len(), 1);
        // Directed rounding on bias (#2164) widens bounds by 1 ULP — check soundness + proximity
        let (lo, hi) = (output.lower()[[0]], output.upper()[[0]]);
        assert!(lo <= 1.0 && (lo - 1.0).abs() < 1e-6, "lower={lo}");
        assert!(hi >= 3.0 && (hi - 3.0).abs() < 1e-6, "upper={hi}");
        assert_eq!(linear_bounds.lower_a[[0, 0]], 2.0);
        assert_eq!(linear_bounds.upper_a[[0, 0]], 2.0);
        let (lb, ub) = (linear_bounds.lower_b[[0]], linear_bounds.upper_b[[0]]);
        assert!(lb <= 1.0 && (lb - 1.0).abs() < 1e-6, "lower_b={lb}");
        assert!(ub >= 1.0 && (ub - 1.0).abs() < 1e-6, "upper_b={ub}");
        Ok(())
    }

    #[test]
    fn propagate_crown_with_linear_skip_merge_identity() -> Result<()> {
        let mut network = Network::new();
        network.add_layer(Layer::SkipMerge(SkipMergeLayer::new()));

        let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
        let (output, linear_bounds) = network.propagate_crown_with_linear(&input)?;

        // Concretized bounds use directed rounding (next_down_f32/next_up_f32),
        // so exact 0.0 becomes -1e-45. Check soundness containment + tightness.
        assert!(
            output.lower()[[0]] <= 0.0,
            "lower bound must be <= 0.0, got {}",
            output.lower()[[0]]
        );
        assert!(
            output.upper()[[0]] >= 1.0,
            "upper bound must be >= 1.0, got {}",
            output.upper()[[0]]
        );
        assert!(
            (output.lower()[[0]] - 0.0).abs() < 1e-6,
            "skip_merge lower bound tightness: expected ~0.0, got {}",
            output.lower()[[0]]
        );
        assert!(
            (output.upper()[[0]] - 1.0).abs() < 1e-6,
            "skip_merge upper bound tightness: expected ~1.0, got {}",
            output.upper()[[0]]
        );
        // Linear coefficients are exact (no directed rounding on coefficients).
        assert_eq!(linear_bounds.lower_a[[0, 0]], 1.0);
        assert_eq!(linear_bounds.upper_a[[0, 0]], 1.0);
        assert_eq!(linear_bounds.lower_b[[0]], 0.0);
        assert_eq!(linear_bounds.upper_b[[0]], 0.0);
        Ok(())
    }

    #[test]
    fn propagate_crown_with_linear_reduce_mean_expands_input_coefficients() -> Result<()> {
        let mut network = Network::new();
        network.add_layer(Layer::ReduceMean(ReduceMeanLayer::new(vec![-1], true)));

        let lower = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let upper = arr2(&[[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]]).into_dyn();
        let input = BoundedTensor::new(lower, upper)?;

        let (output, linear_bounds) = network.propagate_crown_with_linear(&input)?;

        assert_eq!(output.shape(), &[2, 1]);
        assert!(
            (output.lower()[[0, 0]] - 2.0).abs() < 1e-6,
            "reduce_mean lower[0] tightness: expected ~2.0, got {}",
            output.lower()[[0, 0]]
        );
        assert!(
            (output.upper()[[0, 0]] - 3.0).abs() < 1e-6,
            "reduce_mean upper[0] tightness: expected ~3.0, got {}",
            output.upper()[[0, 0]]
        );
        assert!(
            (output.lower()[[1, 0]] - 5.0).abs() < 1e-6,
            "reduce_mean lower[1] tightness: expected ~5.0, got {}",
            output.lower()[[1, 0]]
        );
        assert!(
            (output.upper()[[1, 0]] - 6.0).abs() < 1e-6,
            "reduce_mean upper[1] tightness: expected ~6.0, got {}",
            output.upper()[[1, 0]]
        );
        assert_eq!(linear_bounds.num_outputs(), 2);
        assert_eq!(linear_bounds.num_inputs(), input.len());
        assert_block_diagonal_coefficients(&linear_bounds, 6, 3, 1.0f32 / 3.0, "reduce_mean");
        Ok(())
    }

    #[test]
    fn propagate_crown_with_linear_reduce_sum_expands_input_coefficients() -> Result<()> {
        let mut network = Network::new();
        network.add_layer(Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)));

        let lower = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let upper = arr2(&[[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]]).into_dyn();
        let input = BoundedTensor::new(lower, upper)?;

        let (output, linear_bounds) = network.propagate_crown_with_linear(&input)?;

        assert_eq!(output.shape(), &[2, 1]);
        // Directed rounding (concretize_sound) widens bounds by 1 ULP.
        // Check soundness containment: output must contain true range.
        assert!(
            output.lower()[[0, 0]] <= 6.0,
            "lower[0] must be <= 6.0, got {}",
            output.lower()[[0, 0]]
        );
        assert!(
            output.upper()[[0, 0]] >= 9.0,
            "upper[0] must be >= 9.0, got {}",
            output.upper()[[0, 0]]
        );
        assert!(
            output.lower()[[1, 0]] <= 15.0,
            "lower[1] must be <= 15.0, got {}",
            output.lower()[[1, 0]]
        );
        assert!(
            output.upper()[[1, 0]] >= 18.0,
            "upper[1] must be >= 18.0, got {}",
            output.upper()[[1, 0]]
        );
        // Tightness: bounds should be within small tolerance of true values.
        assert!(
            (output.lower()[[0, 0]] - 6.0).abs() < 1e-4,
            "reduce_sum lower[0] tightness: expected ~6.0, got {}",
            output.lower()[[0, 0]]
        );
        assert!(
            (output.upper()[[0, 0]] - 9.0).abs() < 1e-4,
            "reduce_sum upper[0] tightness: expected ~9.0, got {}",
            output.upper()[[0, 0]]
        );
        assert!(
            (output.lower()[[1, 0]] - 15.0).abs() < 1e-4,
            "reduce_sum lower[1] tightness: expected ~15.0, got {}",
            output.lower()[[1, 0]]
        );
        assert!(
            (output.upper()[[1, 0]] - 18.0).abs() < 1e-4,
            "reduce_sum upper[1] tightness: expected ~18.0, got {}",
            output.upper()[[1, 0]]
        );
        assert_eq!(linear_bounds.num_outputs(), 2);
        assert_eq!(linear_bounds.num_inputs(), input.len());
        assert_block_diagonal_coefficients(&linear_bounds, 6, 3, 1.0, "reduce_sum");
        Ok(())
    }

    #[test]
    fn propagate_crown_with_linear_gather_produces_selection_matrix_bounds() -> Result<()> {
        // Gather(axis=0, indices=[0, 2]) on input [3] produces output [2].
        // CROWN backward produces a selection matrix (exact, no relaxation).
        // Reference: #3400 — Gather CROWN backward implementation.
        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Gather(GatherLayer::new(0, Some(indices), vec![])));

        let input = BoundedTensor::new(
            arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
            arr1(&[10.0f32, 20.0, 30.0]).into_dyn(),
        )?;

        let (concrete, linear_bounds) = network.propagate_crown_with_linear(&input)?;
        let linear_concrete = linear_bounds.concretize(&input);
        let concrete_flat = concrete.flatten();

        assert_eq!(concrete.shape(), &[2]);
        assert_eq!(linear_bounds.num_inputs(), input.len());
        assert_eq!(linear_bounds.num_outputs(), concrete_flat.len());

        // Gather is a selection matrix: output[0] = input[0], output[1] = input[2].
        // Verify selection matrix via block-diagonal helper: row 0 selects col 0, row 1 skips.
        // Manual check for the non-standard selection pattern [0, _, 2]:
        let expected = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for (row, expected_row) in expected.iter().enumerate() {
            for (col, &exp) in expected_row.iter().enumerate() {
                assert!(
                    (linear_bounds.lower_a[[row, col]] - exp).abs() < 1e-6,
                    "gather lower_a[{row},{col}]: expected {exp}, got {}",
                    linear_bounds.lower_a[[row, col]]
                );
            }
        }

        // Concretization should match concrete bounds exactly (Gather is exact).
        assert_eq!(linear_concrete.lower(), concrete_flat.lower());
        assert_eq!(linear_concrete.upper(), concrete_flat.upper());

        // Concrete bounds: output[0] = input[0] ∈ [1, 10], output[1] = input[2] ∈ [3, 30].
        assert!(
            (concrete.lower()[[0]] - 1.0).abs() < 1e-4,
            "gather concrete lower[0]: expected ~1.0, got {}",
            concrete.lower()[[0]]
        );
        assert!(
            (concrete.upper()[[0]] - 10.0).abs() < 1e-4,
            "gather concrete upper[0]: expected ~10.0, got {}",
            concrete.upper()[[0]]
        );
        assert!(
            (concrete.lower()[[1]] - 3.0).abs() < 1e-4,
            "gather concrete lower[1]: expected ~3.0, got {}",
            concrete.lower()[[1]]
        );
        assert!(
            (concrete.upper()[[1]] - 30.0).abs() < 1e-4,
            "gather concrete upper[1]: expected ~30.0, got {}",
            concrete.upper()[[1]]
        );
        Ok(())
    }

    #[test]
    fn propagate_crown_with_linear_non_finite_concretization_falls_back_to_ibp_constant_linear(
    ) -> Result<()> {
        // Large linear coefficient makes CROWN concretization overflow (+inf) for input upper=2.
        // The fast-with-linear path must detect this and return the IBP+constant-linear fallback.
        let weight = arr2(&[[f32::MAX]]);
        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(weight, None)?));

        let input = BoundedTensor::new(arr1(&[0.0f32]).into_dyn(), arr1(&[2.0]).into_dyn())?;
        let (concrete, linear_bounds) = network.propagate_crown_with_linear(&input)?;
        let linear_concrete = linear_bounds.concretize(&input);
        let concrete_flat = concrete.flatten();

        // IBP lower: f32::MAX * 0 = 0 — finite and exact. IBP upper:
        // f32::MAX * 2 overflows f32, and the true output at x=2 really is
        // above every finite f32, so the sound fallback upper is +inf
        // (a finite clamp here would understate the reachable output).
        assert!(
            concrete_flat.lower()[[0]].is_finite(),
            "IBP fallback lower must be finite, got {}",
            concrete_flat.lower()[[0]]
        );
        assert!(
            concrete_flat.lower()[[0]] <= 0.0,
            "IBP fallback lower must bound f(0) = 0, got {}",
            concrete_flat.lower()[[0]]
        );
        assert_eq!(
            concrete_flat.upper()[[0]],
            f32::INFINITY,
            "IBP fallback upper must preserve +inf: f32::MAX * 2 overflows f32"
        );
        assert!(
            linear_bounds.lower_a.iter().all(|&v| v == 0.0),
            "IBP fallback lower_a must be all zeros (constant linear)"
        );
        assert!(
            linear_bounds.upper_a.iter().all(|&v| v == 0.0),
            "IBP fallback upper_a must be all zeros (constant linear)"
        );
        assert_eq!(linear_bounds.lower_b[0], concrete_flat.lower()[[0]]);
        assert_eq!(linear_bounds.upper_b[0], concrete_flat.upper()[[0]]);
        // Re-concretizing the constant linear can only widen: `concretize`
        // repairs any element with a non-finite endpoint to [-inf, +inf]
        // (here upper_b = +inf drags the concretized lower to -inf), so
        // assert containment of the concrete fallback rather than equality.
        assert!(
            linear_concrete.lower()[[0]] <= concrete_flat.lower()[[0]],
            "constant-linear concretization lower {} must contain fallback lower {}",
            linear_concrete.lower()[[0]],
            concrete_flat.lower()[[0]]
        );
        assert_eq!(
            linear_concrete.upper()[[0]],
            f32::INFINITY,
            "constant-linear concretization must preserve the +inf fallback upper"
        );
        Ok(())
    }
}
