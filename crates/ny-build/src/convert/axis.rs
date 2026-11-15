// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Axis interpretation shared by ONNX conversion families.

use ny_core::{NyError, Result};
use tracing::debug;

use super::ConvertContext;
use crate::{AttributeValue, LayerSpec, INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR};

/// Legacy behavior for ONNX axis 0 when the recorded rank of the data tensor
/// is unknown and the node carries authenticated internal-lowering provenance.
/// For an ONNX-authored node with an unknown recorded rank neither variant
/// applies: the axis cannot be resolved and conversion refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LegacyBatchAxisPolicy {
    /// Keep axis 0 with a debug note (historic reductions / Concat behavior).
    KeepZeroWarn,
    /// Reject axis 0 as targeting the nonexistent batch dim.
    RejectZero,
}

/// Validate the source-semantics marker for the narrow legacy-axis branch.
///
/// Missing provenance is an ordinary `false`; malformed provenance is a load
/// error. This marker only selects the axis convention used while converting a
/// [`LayerSpec`]. It confers no solver, proof, certificate, or verdict authority.
pub(crate) fn legacy_axis_authority(spec: &LayerSpec) -> Result<bool> {
    match spec.attributes.get(INTERNAL_LEGACY_AXIS_AUTHORITY_ATTR) {
        None => Ok(false),
        Some(AttributeValue::Int(1)) => Ok(true),
        Some(other) => Err(NyError::ModelLoad(format!(
            "node '{}' has malformed internal legacy-axis authority {other:?}; expected integer 1",
            spec.name
        ))),
    }
}

impl ConvertContext<'_> {
    /// Remap an ONNX axis of the activation tensor `data_name` to the
    /// TRAILING-RELATIVE (negative) internal axis encoding, which is correct
    /// under BOTH internal runtime layouts of a batched model.
    ///
    /// Background (#pensieve ReduceSum no-op miscompile): ny's internal
    /// (unbatched) convention is NOT rank-uniform. For an ONNX tensor of rank
    /// `r`, the runtime tensor is either
    ///   - the ONNX tensor with its leading batch dim stripped (rank `r-1`,
    ///     e.g. Split/Slice outputs), or
    ///   - the ONNX tensor verbatim (rank `r`, leading size-1 retained, e.g.
    ///     Flatten and rank-2 Gemm outputs, weights/constants).
    ///
    /// The legacy blanket `axis >= 1 → axis - 1` guess is only correct for the
    /// first layout; on the second it selects the WRONG dimension — on
    /// pensieve `ReduceSum(axes=[1])` on a runtime `[1, n]` tensor became a
    /// size-1-axis no-op, so the graph bounded a different function than the
    /// ONNX semantics (w = p/p = 1).
    ///
    /// Both layouts share the ONNX tensor's TRAILING dims, so expressing the
    /// axis from the end (`onnx_axis - r`, negative) selects the same semantic
    /// dimension in either layout; layers resolve negative axes against the
    /// actual runtime rank at propagation time.
    ///
    /// Fail-closed where information exists: with a KNOWN recorded rank, an
    /// out-of-range axis or the (possibly stripped) batch axis 0 of a rank>1
    /// tensor — where no single encoding is correct for both layouts — REFUSES
    /// conversion with an error rather than guessing.
    ///
    /// UNKNOWN recorded rank REFUSES for every ONNX-authored node. The
    /// previous wording here — "real ONNX models get recorded shapes from
    /// load-time shape inference, so this branch is reached (in practice) only
    /// by ny-SYNTHESIZED internal subgraphs" — was a measurement claim, and it
    /// is false. `tensor_shapes` is built (`ny-onnx parse/metadata.rs`
    /// `build_tensor_shapes`) from graph inputs/outputs, proto `value_info`,
    /// initializer shapes and the ORT inference result. A PyTorch export
    /// typically carries NO `value_info` for intermediates, and the ORT step is
    /// best-effort — `infer_file_shapes`/`infer_byte_shapes` swallow any error
    /// into `warn!("ONNX Runtime shape inference skipped: …")` and, since
    /// `shape_infer::static_infer`, the ORT-free exact-rule table, which leaves
    /// every tensor it cannot prove unrecorded. So a subprocess crash, a
    /// timeout, or a model ORT will not load can still leave an ordinary ONNX
    /// model with unrecorded intermediates, and the
    /// old branch then applied the `axis - 1` guess to genuinely ONNX-authored
    /// axes: exactly the defect described above, where the graph bounds a
    /// DIFFERENT FUNCTION than the model. Inferring "ny synthesized this" from
    /// the ABSENCE of a shape is backwards, so the branch now demands positive
    /// evidence instead.
    ///
    /// [`legacy_axis_authority`] validates that evidence from an explicit
    /// internal attribute. Only nodes with that authority keep the legacy
    /// adjustment (`axis - 1`, warn); for them the stripped-batch convention
    /// is what the lowering wrote against, so the adjustment is correct by
    /// construction. `zero_policy` selects the per-op legacy behavior for axis 0
    /// (historic per-op semantics preserved exactly). Every other node with an
    /// unknown recorded rank returns [`NyError::UnsupportedOp`], which
    /// `graph/builder.rs` replaces with a conservative `OpaqueSkip`
    /// (`[-inf, +inf]`); the sequential builder (`propagate.rs`) surfaces it as
    /// a load failure. Both outcomes cost a verdict and neither can tighten a
    /// bound.
    ///
    /// The `ReduceL2` lowering's synthesized `ReduceSum` deliberately carries
    /// no authority: its axes remain user-authored ONNX axes, so an unknown rank
    /// refuses rather than guesses.
    ///
    /// Unbatched models (#cctsdb B5) keep ONNX axes verbatim. Negative ONNX
    /// axes are already trailing-relative and pass through (range-checked when
    /// the rank is known).
    pub(crate) fn remap_axis_trailing(
        &self,
        op: &str,
        node_name: &str,
        data_name: &str,
        onnx_axis: i64,
        zero_policy: LegacyBatchAxisPolicy,
        legacy_axis_authorized: bool,
    ) -> Result<i64> {
        if self.model_unbatched {
            // Unbatched model: no tensor ever carried a batch axis; ONNX axes
            // describe the internal tensors verbatim. They still must name a
            // real dimension when the authored rank is known.
            if let Some(rank) = self.recorded_onnx_rank(data_name) {
                let rank_i = i64::try_from(rank).map_err(|_| {
                    NyError::ModelLoad(format!(
                        "{op} '{node_name}': recorded rank {rank} for input '{data_name}' does not fit i64"
                    ))
                })?;
                if onnx_axis < -rank_i || onnx_axis >= rank_i {
                    return Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': ONNX axis {onnx_axis} out of range for input \
                         '{data_name}' of recorded rank {rank}"
                    )));
                }
            }
            return Ok(onnx_axis);
        }
        let rank = self.recorded_onnx_rank(data_name);
        if onnx_axis < 0 {
            if let Some(rank) = rank {
                if onnx_axis < -(rank as i64) {
                    return Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': ONNX axis {onnx_axis} out of range for input \
                         '{data_name}' of recorded rank {rank}"
                    )));
                }
            }
            return Ok(onnx_axis);
        }
        let Some(rank) = rank else {
            // No recorded ONNX rank. `axis - 1` is correct ONLY when the
            // runtime tensor is the ONNX tensor with its leading batch dim
            // STRIPPED; on a retained-rank tensor it selects a different
            // dimension and the graph bounds a different function (#pensieve
            // ReduceSum no-op; the mscn deep-set Concat in `concat.rs`).
            // A MISSING shape says nothing about which layout applies, so the
            // legacy adjustment is admissible only where ny itself authored
            // the node against the stripped-batch convention. That requires
            // POSITIVE evidence — see the doc comment.
            if !legacy_axis_authorized {
                // Preserve the historic axis-0 refusal verbatim for
                // `RejectZero` ops so this change adds no new admitted path:
                // that case already failed closed, with this exact message.
                if onnx_axis == 0 && zero_policy == LegacyBatchAxisPolicy::RejectZero {
                    return Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': axis=0 targets batch dimension which does not exist \
                         in unbatched mode"
                    )));
                }
                return Err(NyError::UnsupportedOp(format!(
                    "{op} '{node_name}': input '{data_name}' has no recorded ONNX shape, so ONNX \
                     axis {onnx_axis} cannot be re-expressed trailing-relative and the legacy \
                     batch-squeeze adjustment would be a guess about the runtime layout; \
                     refusing conversion (re-export with value_info, or make load-time shape \
                     inference succeed, to recover this node)"
                )));
            }
            // Authenticated ny-authored lowering (see `legacy_axis_authority`):
            // the stripped-batch convention is the one it wrote against, so
            // the legacy adjustment is correct by construction here.
            if onnx_axis == 0 {
                return match zero_policy {
                    LegacyBatchAxisPolicy::KeepZeroWarn => {
                        debug!(
                            "{op} '{node_name}': ONNX axis 0 (batch dim) kept as 0 in unbatched \
                             mode (unknown recorded rank; ny-lowered node, legacy behavior)"
                        );
                        Ok(0)
                    }
                    LegacyBatchAxisPolicy::RejectZero => Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': axis=0 targets batch dimension which does not exist \
                         in unbatched mode"
                    ))),
                };
            }
            debug!(
                "{op} '{node_name}': input '{data_name}' has no recorded ONNX shape; keeping \
                 legacy batch-squeeze adjustment {} -> {} (ny-lowered-node compatibility)",
                onnx_axis,
                onnx_axis - 1
            );
            return Ok(onnx_axis - 1);
        };
        let rank_i = rank as i64;
        if onnx_axis >= rank_i {
            return Err(NyError::ModelLoad(format!(
                "{op} '{node_name}': ONNX axis {onnx_axis} out of range for input '{data_name}' \
                 of recorded rank {rank}"
            )));
        }
        if onnx_axis == 0 && rank > 1 {
            // ONNX axis 0 of a rank>1 tensor is the (possibly stripped) batch
            // axis: on a stripped runtime tensor the op is a batch no-op, on a
            // retained one it targets the leading size-1 dim. No single axes
            // encoding expresses both; refuse (consistent with Gather).
            return Err(NyError::UnsupportedOp(format!(
                "{op} '{node_name}': ONNX axis 0 targets the batch dimension of rank-{rank} \
                 input '{data_name}', which is ambiguous in unbatched mode; refusing conversion"
            )));
        }
        Ok(onnx_axis - rank_i)
    }
}
