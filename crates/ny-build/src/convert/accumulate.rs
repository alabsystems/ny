// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Additive indexed-accumulation conversions: `ScatterAdd` and `IndexAdd`.
//!
//! Both ops share the `(data, index, src)` operand triple and an `axis`
//! attribute; they differ only in how `index` addresses the scatter axis:
//!
//! * `ScatterAdd` — `index` is shaped like `src`, and each src element carries
//!   its own destination coordinate along `axis`
//!   (`out[i][index[i][j][k]][k] += src[i][j][k]` for `axis == 1`).
//! * `IndexAdd` — `index` is 1-D of length `src.shape()[axis]`, so a whole
//!   src slice shares one destination (`out[index[i], j, ..] += src[i, j, ..]`).
//!
//! ## Why the additive family needs no duplicate-index union
//!
//! `+` is associative and commutative OVER THE REALS, so a destination hit by
//! several index entries receives the SUM of the contributions regardless of
//! visit order. That is a single well-defined value, unlike the overwriting
//! `ScatterND` (see `convert/scatter_nd.rs`), whose duplicate destinations are
//! order-dependent and must be enclosed by a union of the candidate writes.
//! `ScatterAddLayer` / `IndexAddLayer` accumulate the lower bounds into the
//! lower output and the upper bounds into the upper output, which is the
//! interval image of that real sum.
//!
//! The associativity argument is about the REALS and does not survive into
//! f32: f32 `+` is round-to-nearest, so a long accumulation both rounds and
//! depends on visit order (`1e8 + 1.0` rounds back to `1e8`). The identity of
//! the two endpoint passes makes that worse, not better — an undirected `+=`
//! shifts BOTH endpoints the same way and translates the interval off the true
//! value instead of widening it. The layers therefore round each accumulation
//! OUTWARD for the endpoint it is building (`add_down_f32` below,
//! `add_up_f32` above; the dynamic-index hull folds in f64 with directed adds
//! and lands with one directed cast). So the enclosure claim above holds, but
//! it rests on that rounding discipline — see the rounding-contract section of
//! `ny-propagate`'s `layers/transform/accumulate.rs` — not on f32 arithmetic
//! being exact.
//!
//! ## Constant folding and the dynamic-index fallback
//!
//! Any of the three operands may be a build-time constant; each resolved value
//! is embedded into the layer and the corresponding activation input drops out
//! (`activation_input_count()` tracks the remainder, and the graph builder
//! filters constants out of the node's inputs the same way).
//!
//! A NON-constant `index` is left dynamic on purpose rather than refused: the
//! layers' `ResolvedIndices::Dynamic` path never reads the index tensor and
//! instead widens `data` by the total possible additive contribution of `src`
//! (`sum of min(src_lower, 0)` below, `sum of max(src_upper, 0)` above). Every
//! index assignment routes each src element to exactly one destination, so
//! every realizable output lies inside that hull — sound for ANY index, just
//! loose.

use ny_core::{NyError, Result};
use ny_propagate::layers::{IndexAddLayer, ScatterAddLayer};
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec, LegacyBatchAxisPolicy};

/// Operand slot names, used only for error messages.
const OPERANDS: [&str; 3] = ["data", "index", "src"];

impl ConvertContext<'_> {
    pub(crate) fn convert_scatter_add(&self, spec: &LayerSpec) -> Result<Layer> {
        let (axis, data, indices, src) = self.resolve_accumulate_operands("ScatterAdd", spec)?;
        debug!(
            "Converting ScatterAdd '{}' axis={} (data_const={}, index_const={}, src_const={})",
            spec.name,
            axis,
            data.is_some(),
            indices.is_some(),
            src.is_some()
        );
        Ok(Layer::ScatterAdd(ScatterAddLayer::new(
            axis, data, indices, src,
        )))
    }

    pub(crate) fn convert_index_add(&self, spec: &LayerSpec) -> Result<Layer> {
        let (axis, data, indices, src) = self.resolve_accumulate_operands("IndexAdd", spec)?;
        if let Some(indices) = &indices {
            if indices.ndim() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "IndexAdd {} requires a 1-D index tensor, got rank {} (shape {:?}); a \
                     src-shaped index is ScatterAdd, not IndexAdd",
                    spec.name,
                    indices.ndim(),
                    indices.shape()
                )));
            }
        }
        debug!(
            "Converting IndexAdd '{}' axis={} (data_const={}, index_const={}, src_const={})",
            spec.name,
            axis,
            data.is_some(),
            indices.is_some(),
            src.is_some()
        );
        Ok(Layer::IndexAdd(IndexAddLayer::new(
            axis, data, indices, src,
        )))
    }

    /// Shared operand resolution for the additive scatter family.
    ///
    /// Returns `(axis, data_constant, indices, src_constant)`.
    #[allow(clippy::type_complexity)]
    fn resolve_accumulate_operands(
        &self,
        op: &str,
        spec: &LayerSpec,
    ) -> Result<(
        i64,
        Option<ndarray::ArrayD<f32>>,
        Option<ndarray::ArrayD<i64>>,
        Option<ndarray::ArrayD<f32>>,
    )> {
        if spec.inputs.len() != 3 {
            return Err(NyError::ModelLoad(format!(
                "{op} {} requires 3 inputs (data, index, src), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        // The axis is REQUIRED, with no default. ONNX's ScatterElements would
        // default it to 0, but axis 0 is precisely the (possibly stripped)
        // batch axis this converter refuses to guess at — see
        // `remap_axis_trailing`. A missing attribute is a translator bug, so
        // fail closed instead of silently accumulating along the wrong axis.
        let onnx_axis = match spec.attributes.get("axis") {
            Some(AttributeValue::Int(axis)) => *axis,
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "{op} {} has a non-integer 'axis' attribute ({other:?})",
                    spec.name
                )));
            }
            None => {
                return Err(NyError::ModelLoad(format!(
                    "{op} {} is missing the required 'axis' attribute; there is no safe default \
                     (ONNX's axis-0 default names the batch dimension)",
                    spec.name
                )));
            }
        };
        // Trailing-relative remap, matching Gather: the runtime data tensor may
        // or may not have kept its leading batch dim, and a negative axis names
        // the same semantic dimension either way. The layers resolve it against
        // the actual runtime rank. Already-negative axes (what the trace bridge
        // emits) pass through unchanged.
        let axis = self.remap_axis_trailing(
            op,
            &spec.name,
            &spec.inputs[0],
            onnx_axis,
            LegacyBatchAxisPolicy::RejectZero,
        )?;

        let data = self.constant_value(&spec.inputs[0]);
        self.reject_unmaterialized_constant(op, spec, 0, data.is_some())?;

        let indices =
            self.discrete_constant_i64(&spec.inputs[1], &format!("{op} {} index", spec.name))?;
        self.reject_unmaterialized_constant(op, spec, 1, indices.is_some())?;

        let src = self.constant_value(&spec.inputs[2]);
        self.reject_unmaterialized_constant(op, spec, 2, src.is_some())?;

        Ok((axis, data, indices, src))
    }

    /// Fail closed when an operand is MARKED constant but its value could not
    /// be materialized.
    ///
    /// The graph builder drops every input in `constant_tensors` from a node's
    /// activation inputs (`find_activation_inputs`), while the layer counts a
    /// non-embedded operand as an activation input it still expects to receive.
    /// A tensor that is constant-by-name but value-less therefore desynchronizes
    /// the two and surfaces later as an opaque arity error at propagation time.
    /// Catch it here, where the operand can actually be named.
    fn reject_unmaterialized_constant(
        &self,
        op: &str,
        spec: &LayerSpec,
        position: usize,
        materialized: bool,
    ) -> Result<()> {
        let name = &spec.inputs[position];
        if materialized || !self.is_constant(name) {
            return Ok(());
        }
        Err(NyError::UnsupportedConfiguration(format!(
            "{op} {}: {} input '{}' is marked constant but its value is not available at build \
             time; refusing rather than emitting a layer whose operand arity cannot be satisfied",
            spec.name, OPERANDS[position], name
        )))
    }
}

#[cfg(test)]
#[path = "accumulate_tests.rs"]
mod accumulate_tests;
