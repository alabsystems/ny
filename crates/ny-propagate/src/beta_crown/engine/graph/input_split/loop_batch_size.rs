// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime batch-size clamp for reordered input-split BaB.

use std::mem::size_of;

use ny_core::{NyError, Result};

/// Configured ceiling on the number of parent domains picked per reordered
/// input-split BaB iteration.
///
/// The effective batch size is still clamped at runtime by
/// [`MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES`] to keep Packet A's shared IBP
/// prescreen workset bounded for larger inputs.
pub(crate) const INPUT_SPLIT_LOOP_BATCH_SIZE_CAP: usize = 1_000_000;

/// Conservative bound on the transient child-input workset used by Packet A.
///
/// The reordered input-split prescreen now retains two live copies of child
/// inputs: the per-child queue staging buffer and the stacked batched IBP
/// input. With two children per parent, that is `4 * input_elems * 2 * sizeof(f32)`
/// bytes per picked parent. Keeping this under 256 MiB preserves the reference
/// `lsnc_relu` 1M-parent batch on 6-element inputs while automatically
/// clamping larger models.
pub(crate) const MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputSplitLoopBatchClampReason {
    None,
    ConfigCap,
    MemoryCap,
}

impl InputSplitLoopBatchClampReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ConfigCap => "config_cap",
            Self::MemoryCap => "memory_cap",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InputSplitLoopBatchDecision {
    pub(crate) requested_batch_size: usize,
    pub(crate) effective_batch_size: usize,
    pub(crate) clamp_reason: InputSplitLoopBatchClampReason,
}

#[inline]
pub(crate) fn input_split_loop_batch_size(
    requested_batch_size: usize,
    input_elems: usize,
) -> Result<InputSplitLoopBatchDecision> {
    let requested_batch_size = requested_batch_size.max(1);
    let config_limited_batch_size = requested_batch_size.min(INPUT_SPLIT_LOOP_BATCH_SIZE_CAP);

    // Zero-sized inputs are unexpected here; charging one scalar keeps the
    // helper deterministic and avoids division-by-zero in the memory guard.
    let input_elems = input_elems.max(1);
    let bytes_per_child = input_elems
        .checked_mul(2)
        .and_then(|value| value.checked_mul(size_of::<f32>()))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "input split loop batch size overflowed child workset bytes for input_elems={input_elems}"
            ))
        })?;
    let bytes_per_parent = bytes_per_child
        .checked_mul(2)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "input split loop batch size overflowed parent workset bytes for input_elems={input_elems}"
            ))
        })?;
    let memory_limited_batch_size = (MAX_INPUT_SPLIT_CHILD_WORKSET_BYTES / bytes_per_parent).max(1);
    let effective_batch_size = config_limited_batch_size.min(memory_limited_batch_size);
    let clamp_reason = if effective_batch_size < config_limited_batch_size {
        InputSplitLoopBatchClampReason::MemoryCap
    } else if config_limited_batch_size < requested_batch_size {
        InputSplitLoopBatchClampReason::ConfigCap
    } else {
        InputSplitLoopBatchClampReason::None
    };

    Ok(InputSplitLoopBatchDecision {
        requested_batch_size,
        effective_batch_size,
        clamp_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_split_loop_batch_size_keeps_lsnc_scale_cap_4353() {
        let decision =
            input_split_loop_batch_size(INPUT_SPLIT_LOOP_BATCH_SIZE_CAP, 6).expect("valid input");

        assert_eq!(
            decision.requested_batch_size,
            INPUT_SPLIT_LOOP_BATCH_SIZE_CAP
        );
        assert_eq!(
            decision.effective_batch_size,
            INPUT_SPLIT_LOOP_BATCH_SIZE_CAP
        );
        assert_eq!(decision.clamp_reason, InputSplitLoopBatchClampReason::None);
    }

    #[test]
    fn test_input_split_loop_batch_size_clamps_large_inputs_by_memory_4353() {
        let decision =
            input_split_loop_batch_size(INPUT_SPLIT_LOOP_BATCH_SIZE_CAP, 784).expect("valid input");

        assert_eq!(
            decision.requested_batch_size,
            INPUT_SPLIT_LOOP_BATCH_SIZE_CAP
        );
        assert!(
            decision.effective_batch_size < INPUT_SPLIT_LOOP_BATCH_SIZE_CAP,
            "expected memory clamp below the 1M configured ceiling, got {}",
            decision.effective_batch_size
        );
        assert_eq!(
            decision.clamp_reason,
            InputSplitLoopBatchClampReason::MemoryCap
        );
    }

    #[test]
    fn test_input_split_loop_batch_size_reports_config_cap_4353() {
        let decision = input_split_loop_batch_size(INPUT_SPLIT_LOOP_BATCH_SIZE_CAP + 1, 6)
            .expect("valid input");

        assert_eq!(
            decision.effective_batch_size,
            INPUT_SPLIT_LOOP_BATCH_SIZE_CAP
        );
        assert_eq!(
            decision.clamp_reason,
            InputSplitLoopBatchClampReason::ConfigCap
        );
    }

    #[test]
    fn test_input_split_loop_batch_size_rejects_overflowing_input_sizes_4353() {
        let error = input_split_loop_batch_size(1, usize::MAX)
            .expect_err("overflowing input size should fail fast");

        assert!(
            error.to_string().contains("overflowed"),
            "unexpected overflow error: {error}"
        );
    }
}
