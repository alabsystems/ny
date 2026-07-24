// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN propagation entry points for sequential networks.
//!
//! Moved out of the sequential core module to break bidirectional dependency
//! (#2380).

use crate::bounds::AlphaCrownConfig;
use crate::network::Network;
use crate::network::NetworkAlphaCrownExt;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

impl Network {
    /// Propagate bounds through the network using α-CROWN with optimized parameters.
    ///
    /// α-CROWN extends CROWN by making the lower bound slope (α) for unstable ReLUs
    /// learnable and optimizing it via gradient descent to tighten bounds.
    ///
    /// # REQUIRES
    /// - `input` shape must match network's expected input dimension
    /// - `input.lower()[i] <= input.upper()[i]` for all elements (well-formed bounds)
    /// - Network must contain only supported layers (Linear, ReLU, GELU, etc.)
    ///
    /// # ENSURES
    /// - Output bounds contain all possible network outputs for inputs in `input`
    /// - Bounds are at least as tight as `propagate_crown()` (α-optimization tightens)
    /// - Soundness: for any `x` where `input.contains(x)`, `output.contains(network(x))`
    ///
    /// Algorithm:
    /// 1. Run IBP to collect pre-activation bounds
    /// 2. Initialize α state (from heuristic)
    /// 3. For each optimization iteration:
    ///    a. Run CROWN backward with current α values
    ///    b. Concretize to get bounds
    ///    c. Compute gradients ∂bounds/∂α
    ///    d. Update α via gradient descent
    /// 4. Return the tightest bounds found
    #[inline]
    pub fn propagate_alpha_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        NetworkAlphaCrownExt::propagate_alpha_crown_impl(self, input)
    }

    /// α-CROWN with optional GEMM acceleration engine.
    #[inline]
    pub fn propagate_alpha_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkAlphaCrownExt::propagate_alpha_crown_with_engine_impl(self, input, engine)
    }

    /// α-CROWN with custom configuration (no acceleration engine).
    pub fn propagate_alpha_crown_with_config(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor> {
        NetworkAlphaCrownExt::propagate_alpha_crown_with_config_impl(self, input, config)
    }

    /// α-CROWN with custom configuration and optional GEMM acceleration engine.
    pub fn propagate_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        NetworkAlphaCrownExt::propagate_alpha_crown_with_config_and_engine_impl(
            self, input, config, engine,
        )
    }

    /// α-CROWN with directed rounding for soundness.
    ///
    /// Same as `propagate_alpha_crown` but applies 1-ULP widening to the
    /// final bounds, ensuring floating-point rounding errors do not produce
    /// unsound results.
    pub fn propagate_alpha_crown_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_sound_with_engine(input, None)
    }

    /// α-CROWN with directed rounding for soundness and optional GEMM acceleration engine.
    pub fn propagate_alpha_crown_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let bounds = self.propagate_alpha_crown_with_engine(input, engine)?;
        Ok(bounds.round_for_soundness())
    }
}
