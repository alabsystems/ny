// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::layers::LayerNormCrownMode;

/// Structure describing a single encoder block's boundaries.
#[derive(Debug, Clone)]
pub struct WhisperBlockInfo {
    /// Index of the block (0-3 for Whisper-tiny).
    pub index: usize,
    /// First ONNX LayerSpec index (inclusive).
    pub start_layer_idx: usize,
    /// Last ONNX LayerSpec index (exclusive).
    pub end_layer_idx: usize,
    /// Number of layers in this block.
    pub num_layers: usize,
}

/// Structure describing the Whisper encoder layout.
#[derive(Debug, Clone)]
pub struct WhisperEncoderStructure {
    /// Stem layers (Conv1, GELU, Conv2, GELU, positional embedding).
    pub stem_end_idx: usize,
    /// Information about each encoder block.
    pub blocks: Vec<WhisperBlockInfo>,
    /// Start of the final LayerNorm (ln_post).
    pub ln_post_start_idx: usize,
}

/// Per-LayerNorm site row-collapse stats from decomposed normalization CROWN.
///
/// Reports how many rows collapsed to fused LayerNorm IBP because the decomposed
/// CROWN result was looser than the fused baseline. Part of #318.
#[derive(Debug, Clone)]
pub struct NormalizationRowStats {
    /// ONNX node name of the LayerNorm site.
    pub site_name: String,
    /// Number of rows that collapsed to fused IBP fallback.
    pub fallback_rows: usize,
    /// Total number of rows processed.
    pub total_rows: usize,
}

/// Execution details returned by GPU-aware compositional APIs.
///
/// The `used_*` fields report execution, not requested configuration. The
/// current Whisper encoder compatibility backend runs CPU graph IBP, reports
/// both flags as `false`, leaves normalization stats empty, and aliases all
/// stage widths to the final graph output width. The decoder compatibility APIs
/// currently fail closed and do not construct this type.
#[derive(Debug, Clone)]
pub struct GpuCompositionalDetails {
    /// Reported max width of attention delta bounds.
    pub attention_delta_width: f32,
    /// Reported max width after the first residual.
    pub x_attn_width: f32,
    /// Reported max width of the MLP delta.
    pub mlp_delta_width: f32,
    /// Final output width.
    pub output_width: f32,
    /// Whether the intermediate-looking stage widths are actual measurements.
    ///
    /// False for the Whisper full-graph IBP compatibility fallback, whose
    /// stage fields alias `output_width`.
    pub stage_metrics_available: bool,
    /// Whether attention actually executed on a GPU.
    pub used_gpu_attention: bool,
    /// Whether attention actually used zonotope propagation.
    pub used_zonotope_attention: bool,
    /// Sequence length of the input.
    pub seq_len: usize,
    /// Per-LayerNorm row-collapse stats from an executed block-wise CROWN lane.
    ///
    /// Empty when that lane did not execute, including every current Whisper
    /// encoder compatibility fallback.
    pub normalization_row_stats: Vec<NormalizationRowStats>,
}

/// Details returned by the compositional compatibility API.
///
/// The current Whisper encoder backend executes one complete graph-IBP pass,
/// so its three intermediate-looking fields all alias `output_width`.
#[derive(Debug, Clone)]
pub struct CompositionalVerificationDetails {
    /// Max width of attention delta bounds
    pub attention_delta_width: f32,
    /// Max width after first residual (x + attn_delta)
    pub x_attn_width: f32,
    /// Max width of MLP delta bounds
    pub mlp_delta_width: f32,
    /// Final output width
    pub output_width: f32,
    /// Whether the intermediate-looking stage widths are actual measurements.
    ///
    /// False for the current Whisper full-graph IBP compatibility fallback.
    pub stage_metrics_available: bool,
}

/// Details produced by an executing multi-block sequential verifier.
///
/// Sequential Whisper verification is currently unavailable and returns
/// [`ny_core::NyError::UnsupportedConfiguration`] instead of constructing this
/// type.
#[derive(Debug, Clone)]
pub struct MultiBlockDetails {
    /// Number of blocks in the verification request.
    pub num_blocks: usize,
    /// Per-block attention/MLP details.
    pub block_details: Vec<GpuCompositionalDetails>,
    /// Whether the stem was included.
    pub included_stem: bool,
    /// Whether the final LayerNorm was included.
    pub included_ln_post: bool,
    /// Total verification time in milliseconds.
    pub total_time_ms: u64,
    /// Output width after the stem, if included.
    pub stem_output_width: Option<f32>,
    /// Output width after the final LayerNorm, if included.
    pub ln_post_output_width: Option<f32>,
    /// Final output width.
    pub final_output_width: f32,
    /// Number of blocks actually completed.
    pub blocks_completed: usize,
    /// Whether execution terminated early due to bound overflow.
    pub early_terminated: bool,
    /// Block index where overflow was first detected.
    pub overflow_at_block: Option<usize>,
    /// Reason for early termination, if applicable.
    pub termination_reason: Option<String>,
}

/// Compatibility configuration for Whisper block and sequential verification.
///
/// The current direct-block backend runs conservative CPU graph IBP.
/// Sequential verification is unavailable and returns
/// `UnsupportedConfiguration`. GPU, heuristic LayerNorm, zonotope, LayerNorm
/// CROWN, block-wise CROWN, and overflow-control fields are retained requests.
/// Non-default values for those unavailable requests are rejected by the
/// direct-block backend rather than silently selecting another execution lane.
///
/// Direct block defaults are conservative. Forward-mode LayerNorm remains in
/// the configuration for API compatibility but is rejected because its
/// heuristic bounds have no machine-readable soundness provenance.
///
/// # Factory Methods
///
/// - `default()` - Conservative LayerNorm for direct block IBP
/// - `conservative()` - Conservative LayerNorm for direct block IBP
/// - `strict()` - Conservative preset with retained overflow requests
/// - `diagnostic()` - Conservative preset with retained overflow requests
/// - `tightest_attention()` - Heuristic forward-mode config with a retained zonotope request
/// - `deep_transformer()` - Heuristic forward-mode deep-stack preset
/// - `sound_tight()` - Conservative LayerNorm with a retained zonotope request
#[derive(Debug, Clone)]
pub struct MultiBlockConfig {
    /// Retained maximum-width request for a future sequential backend.
    ///
    /// A non-default value is rejected by the current direct-block backend.
    /// Default: `f32::MAX`.
    pub max_bound_width: f32,
    /// Retained overflow-termination request for a future sequential backend.
    ///
    /// A true value is rejected by the current direct-block backend. Default: false.
    pub terminate_on_overflow: bool,
    /// Retained continue-after-overflow request for a future sequential backend.
    ///
    /// A true value is rejected by the current direct-block backend. Default: false.
    pub continue_after_overflow: bool,
    /// Retained overflow clamp value for a future sequential backend.
    ///
    /// A non-default value is rejected by the current direct-block backend.
    /// Default: `1e30`.
    pub overflow_clamp_value: f32,
    /// Retained heuristic LayerNorm forward-mode request.
    ///
    /// The current direct-block backend rejects `true` and returns no bounds;
    /// untagged heuristic bounds must not escape a verification-named API.
    /// Sequential verification is unavailable. Default: false.
    pub layernorm_forward_mode: bool,
    /// Retained LayerNorm CROWN request.
    ///
    /// The current direct-block backend does not execute CROWN, so a value
    /// other than the compatibility default is rejected. Default: `Cut`.
    pub layernorm_crown_mode: LayerNormCrownMode,
    /// Request zonotope propagation for a future attention backend.
    ///
    /// The current [`WhisperModel`](super::WhisperModel) compatibility backend
    /// rejects this request and returns no bounds. The field remains in the
    /// public configuration surface for API compatibility.
    /// Default: false.
    pub use_zonotope_attention: bool,
    /// Compatibility knob for future sequential zonotope backends.
    ///
    /// The current compatibility backend has no zonotope execution lane, so a
    /// true value is rejected. The field remains in the public config surface
    /// so a future backend can opt into explicit
    /// block-boundary resets without another API break.
    /// Default: false.
    pub reset_zonotope_between_blocks: bool,
    /// Request decomposed-normalization CROWN for a future compositional MLP
    /// backend. The current `WhisperModel` compatibility backend rejects this
    /// request and returns no bounds. Default: false.
    pub use_crown_block_wise: bool,
}

impl Default for MultiBlockConfig {
    fn default() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: false,
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false,
            reset_zonotope_between_blocks: false,
            use_crown_block_wise: false,
        }
    }
}

impl MultiBlockConfig {
    /// Create a direct-block graph-IBP config with conservative LayerNorm bounds.
    pub fn conservative() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: false, // Conservative: strictly sound but may explode
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false,
            reset_zonotope_between_blocks: false,
            use_crown_block_wise: false,
        }
    }

    /// Create a conservative config carrying unavailable strict-overflow requests.
    ///
    /// Sequential verification is currently unavailable, and the current
    /// direct-block backend rejects the non-default overflow fields.
    pub fn strict() -> Self {
        Self {
            max_bound_width: 1e20,
            terminate_on_overflow: true,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: false,
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false,
            reset_zonotope_between_blocks: false,
            use_crown_block_wise: false,
        }
    }

    /// Create a conservative config carrying unavailable diagnostic requests.
    ///
    /// Sequential verification is currently unavailable, and the current
    /// direct-block backend rejects the non-default overflow fields.
    pub fn diagnostic() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: true,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: false, // Conservative for explosion diagnosis
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false,
            reset_zonotope_between_blocks: false,
            use_crown_block_wise: false,
        }
    }

    /// Create the legacy heuristic forward-mode configuration.
    #[deprecated(
        since = "0.1.0",
        note = "Heuristic LayerNorm is unavailable; use MultiBlockConfig::conservative()"
    )]
    pub fn tight_bounds() -> Self {
        Self::default().with_layernorm_forward_mode(true)
    }

    /// Create a heuristic forward-mode config carrying the retained zonotope request.
    ///
    /// No current backend accepts this preset: sequential verification is
    /// unavailable and direct-block verification rejects its zonotope request.
    pub fn tightest_attention() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: true,
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: true,
            reset_zonotope_between_blocks: true,
            use_crown_block_wise: false,
        }
    }

    /// Create a legacy heuristic forward-mode deep-transformer config.
    ///
    /// No current backend accepts this preset: sequential verification is
    /// unavailable and direct-block verification rejects its zonotope/reset
    /// requests.
    pub fn deep_transformer() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: true,
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: true,
            reset_zonotope_between_blocks: true,
            use_crown_block_wise: false,
        }
    }

    /// Create a conservative LayerNorm config carrying the zonotope request.
    ///
    /// No current backend accepts this preset: sequential verification is
    /// unavailable and direct-block verification rejects its zonotope/reset
    /// requests.
    pub fn sound_tight() -> Self {
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: false,
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: true,
            reset_zonotope_between_blocks: true,
            use_crown_block_wise: false,
        }
    }

    /// Set the retained maximum-width request.
    ///
    /// The current direct-block backend rejects non-default values, and
    /// sequential verification is unavailable.
    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_bound_width = max_width;
        self
    }

    /// Set the retained heuristic LayerNorm forward-mode request.
    ///
    /// The current direct-block backend rejects `true` without returning
    /// bounds. Sequential verification is also unavailable.
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.layernorm_forward_mode = enabled;
        self
    }

    /// Set the retained LayerNorm CROWN request.
    ///
    /// The current direct-block backend rejects non-default values.
    pub fn with_layernorm_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.layernorm_crown_mode = mode;
        self
    }

    /// Set the retained overflow-termination request.
    ///
    /// The current direct-block backend rejects a true value.
    pub fn with_terminate_on_overflow(mut self, terminate: bool) -> Self {
        self.terminate_on_overflow = terminate;
        self
    }

    /// Enable or disable the retained zonotope-attention request.
    ///
    /// The current direct-block backend rejects a true value.
    pub fn with_zonotope_attention(mut self, enabled: bool) -> Self {
        self.use_zonotope_attention = enabled;
        self
    }

    /// Enable or disable the compatibility reset flag for future shared-zonotope
    /// sequential backends.
    ///
    /// The current direct-block backend rejects a true value.
    pub fn with_reset_zonotope_between_blocks(mut self, enabled: bool) -> Self {
        self.reset_zonotope_between_blocks = enabled;
        self
    }

    /// Enable or disable the retained block-wise-CROWN request.
    ///
    /// The current direct-block backend rejects a true value.
    pub fn with_crown_block_wise(mut self, enabled: bool) -> Self {
        self.use_crown_block_wise = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_conservative_layernorm(config: &MultiBlockConfig, label: &str) {
        assert!(
            !config.layernorm_forward_mode,
            "{label} must not silently opt into heuristic LayerNorm"
        );
        assert_eq!(config.layernorm_crown_mode, LayerNormCrownMode::Cut);
        assert!(!config.use_zonotope_attention);
        assert!(!config.reset_zonotope_between_blocks);
        assert!(!config.use_crown_block_wise);
    }

    #[test]
    fn default_and_conservative_presets_pin_safe_direct_block_requests() {
        for (label, config) in [
            ("default", MultiBlockConfig::default()),
            ("conservative", MultiBlockConfig::conservative()),
        ] {
            assert_conservative_layernorm(&config, label);
            assert_eq!(config.max_bound_width, f32::MAX);
            assert!(!config.terminate_on_overflow);
            assert!(!config.continue_after_overflow);
            assert_eq!(config.overflow_clamp_value, 1e30);
        }
    }

    #[test]
    fn strict_preset_does_not_enable_heuristic_layernorm() {
        let config = MultiBlockConfig::strict();
        assert_conservative_layernorm(&config, "strict");
        assert!(config.terminate_on_overflow);
    }
}
