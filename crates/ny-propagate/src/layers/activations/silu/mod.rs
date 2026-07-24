// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

mod math;
mod relaxation;

// Re-export public items from submodules.
#[cfg(test)]
pub(crate) use math::silu_critical_point;
pub use math::silu_eval;
pub use relaxation::silu_sound_linear_relaxation;

/// SiLU (Swish) activation: y = x * sigmoid(x).
#[derive(Debug, Clone, Default)]
pub struct SiLULayer;

impl SiLULayer {
    /// Create a new SiLU layer.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate SiLU at a point: x * sigmoid(x)
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        silu_eval(x)
    }
}

impl BoundPropagation for SiLULayer {
    /// IBP for SiLU: y = x * sigmoid(x)
    ///
    /// SiLU is non-monotonic with a single global minimum near x ≈ -1.28.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut lower_vals = input.lower().clone();
        let mut upper_vals = input.upper().clone();

        ndarray::Zip::from(&mut lower_vals)
            .and(&mut upper_vals)
            .and(input.lower())
            .and(input.upper())
            .for_each(|out_l, out_u, &in_l, &in_u| {
                let (min_val, max_val) = math::silu_min_max(in_l, in_u);

                *out_l = min_val;
                *out_u = max_val;
            });

        BoundedTensor::new(lower_vals, upper_vals)
    }
    impl_elementwise_activation!(
        @trait_methods
        SiLULayer,
        NyError::InvalidSpec(
            "SiLU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl SiLULayer {
    impl_elementwise_activation!(
        @inherent_methods
        SiLULayer,
        silu_sound_linear_relaxation,
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("SiLU", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests;
