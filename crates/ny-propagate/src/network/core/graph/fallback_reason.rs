// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::types::CrownIbpFallbackReason;
use ny_core::NyError;

/// Map recoverable graph-CROWN dispatch errors to the structured fallback
/// provenance used by both spec-guided and plain DAG-CROWN paths.
pub(crate) fn graph_crown_dispatch_fallback_reason(
    err: &NyError,
) -> Option<CrownIbpFallbackReason> {
    match err {
        NyError::ShapeMismatch { .. } => Some(CrownIbpFallbackReason::ShapeMismatch),
        // #3795: structural match replaces fragile string matching on "per-node deadline exceeded"
        NyError::DeadlineExceeded(_) => Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded),
        NyError::UnsupportedOp(_)
        | NyError::UnsupportedConfiguration(_)
        | NyError::NumericalInstability(_) => Some(CrownIbpFallbackReason::CrownPropagationError),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::graph_crown_dispatch_fallback_reason;
    use crate::types::CrownIbpFallbackReason;
    use ny_core::NyError;

    #[test]
    fn graph_crown_dispatch_fallback_reason_maps_correctly() {
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::UnsupportedOp(
                "not supported".to_string()
            )),
            Some(CrownIbpFallbackReason::CrownPropagationError)
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::UnsupportedConfiguration(
                "missing input_shape".to_string()
            )),
            Some(CrownIbpFallbackReason::CrownPropagationError)
        );
        // #3795: DeadlineExceeded variant replaces UnsupportedConfiguration string matching
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::DeadlineExceeded(
                "Conv2d CROWN backward: per-node deadline exceeded".to_string()
            )),
            Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::NumericalInstability(
                "NaN detected".to_string()
            )),
            Some(CrownIbpFallbackReason::CrownPropagationError)
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::ShapeMismatch {
                expected: vec![186],
                got: vec![1],
            }),
            Some(CrownIbpFallbackReason::ShapeMismatch)
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::InvalidSpec(
                "shape mismatch".to_string()
            )),
            None
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::InternalError(
                "internal bug".to_string()
            )),
            None
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::SoundnessRefusal(
                "heuristic bounds refused".to_string()
            )),
            None
        );
        assert_eq!(
            graph_crown_dispatch_fallback_reason(&NyError::UnsupportedLayer(
                "custom layer".to_string()
            )),
            None
        );
    }
}
