// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Re-exports conversion types from `ny-build` and keeps the
//! `OnnxModel` compatibility delegate.

use ny_core::Result;
use ny_propagate::Layer;

pub use ny_build::ConvertContext;

use super::{LayerSpec, OnnxModel};

impl OnnxModel {
    /// Borrow the subset of model state used by conversion logic.
    pub fn convert_context(&self) -> ConvertContext<'_> {
        ConvertContext::new(&self.weights, &self.tensor_shapes, &self.constant_tensors)
            .with_model_unbatched(ny_build::model_is_unbatched(&self.network.inputs))
    }

    /// Convert a single [`LayerSpec`] into a bound-propagation [`Layer`].
    ///
    /// Dispatches on `spec.layer_type` to the appropriate converter. Returns
    /// [`NyError::UnsupportedOp`] for unrecognized layer types.
    ///
    /// Kept as a compatibility delegate while callers migrate to
    /// [`ConvertContext::convert_layer`] (#1752).
    pub fn convert_layer(&self, spec: &LayerSpec) -> Result<Layer> {
        self.convert_context().convert_layer(spec)
    }
}
