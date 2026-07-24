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

/// Details from GPU-accelerated compositional verification.
#[derive(Debug, Clone)]
pub struct GpuCompositionalDetails {
    /// Max width of attention delta bounds
    pub attention_delta_width: f32,
    /// Max width after first residual (x + attn_delta)
    pub x_attn_width: f32,
    /// Max width of MLP delta bounds
    pub mlp_delta_width: f32,
    /// Final output width
    pub output_width: f32,
    /// Whether GPU was used for attention
    pub used_gpu_attention: bool,
    /// Whether zonotope was used for attention (correlation-aware bounds)
    pub used_zonotope_attention: bool,
    /// Sequence length of input
    pub seq_len: usize,
    /// Per-LayerNorm site row-collapse stats (empty when `use_crown_block_wise` is false).
    /// Part of #318.
    pub normalization_row_stats: Vec<NormalizationRowStats>,
}

/// Details from compositional verification showing intermediate bound widths.
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
}

/// Details from multi-block sequential verification.
#[derive(Debug, Clone)]
pub struct MultiBlockDetails {
    /// Number of blocks verified
    pub num_blocks: usize,
    /// Per-block details (attention/MLP widths for each block)
    pub block_details: Vec<GpuCompositionalDetails>,
    /// Whether stem was included
    pub included_stem: bool,
    /// Whether final LayerNorm (ln_post) was included
    pub included_ln_post: bool,
    /// Total verification time in milliseconds
    pub total_time_ms: u64,
    /// Output width after stem (if included)
    pub stem_output_width: Option<f32>,
    /// Output width after ln_post (if included)
    pub ln_post_output_width: Option<f32>,
    /// Final output width
    pub final_output_width: f32,
    /// Number of blocks actually completed (may be < num_blocks if early termination)
    pub blocks_completed: usize,
    /// Whether early termination occurred due to bound overflow
    pub early_terminated: bool,
    /// Block index where overflow was first detected (if any)
    pub overflow_at_block: Option<usize>,
    /// Reason for early termination (if applicable)
    pub termination_reason: Option<String>,
}

/// Configuration for multi-block sequential verification.
///
/// # Default Configuration
///
/// The default config uses forward-mode LayerNorm (`layernorm_forward_mode: true`),
/// which provides dramatically tighter bounds (up to 1e31x improvement on multi-block
/// transformers) compared to conservative mode. This is appropriate for typical
/// verification scenarios with small perturbations (eps < 0.1).
///
/// When zonotope attention is enabled, the attention-prefix seam in
/// `verify_block_compositional_gpu_with_config()` stays pinned to the
/// conservative attention LayerNorm output as part of `#318`. Forward mode still
/// applies to the suffix graph and the rest of the block.
///
/// When `use_crown_block_wise` is enabled, the MLP-side LayerNorm inside
/// `verify_block_compositional_gpu_with_config()` is also pinned to
/// conservative semantics. `layernorm_forward_mode` continues to govern the
/// attention side and the non-block-wise MLP path.
///
/// For users requiring strictly mathematically sound bounds (at the cost of
/// potentially useless results due to bound explosion), use `MultiBlockConfig::conservative()`.
///
/// # Factory Methods
///
/// - `default()` - Forward-mode LayerNorm for practical verification (recommended)
/// - `conservative()` - Strictly sound bounds, may explode on multi-block transformers
/// - `strict()` - Like default but terminates early on overflow
/// - `diagnostic()` - Continues through overflow for analysis
/// - `tightest_attention()` - Forward-mode block config + zonotope attention with a conservative attention-prefix seam
/// - `deep_transformer()` - `tightest_attention()` preset retained for deep-stack workflows
/// - `sound_tight()` - Conservative LayerNorm across the full block + zonotope attention
#[derive(Debug, Clone)]
pub struct MultiBlockConfig {
    /// Maximum allowed bound width before early termination.
    /// If bounds exceed this threshold, verification stops and returns Unknown.
    /// Default: f32::MAX (no threshold - continue until overflow)
    pub max_bound_width: f32,
    /// Whether to terminate early when NaN or Infinity is detected in bounds.
    /// Default: false (preserves original behavior of `verify_encoder_sequential`)
    pub terminate_on_overflow: bool,
    /// Whether to continue verification even after overflow (for diagnostics).
    /// When true, bounds will be clamped to prevent NaN propagation.
    /// Default: false (stop on first overflow for soundness)
    pub continue_after_overflow: bool,
    /// Bound value to clamp to when continue_after_overflow is true.
    /// Default: 1e30
    pub overflow_clamp_value: f32,
    /// Use forward mode for LayerNorm IBP: compute mean/std from center point.
    /// This dramatically reduces bound explosion (up to 1e31x tighter bounds on
    /// multi-block transformers) but may not be perfectly sound for large perturbations.
    ///
    /// When `use_zonotope_attention` is enabled, the attention-prefix seam inside
    /// `verify_block_compositional_gpu_with_config()` still uses the conservative
    /// attention LayerNorm output for the shared-source zonotope suffix. This flag
    /// continues to control the non-zonotope paths, the zonotope suffix graph, and
    /// the rest of the block.
    ///
    /// When `use_crown_block_wise` is also enabled,
    /// `verify_block_compositional_gpu_with_config()` keeps the MLP-side
    /// LayerNorm conservative by design for `#318` stability. In
    /// `verify_encoder_sequential_with_config()`, later blocks are also forced
    /// to conservative LayerNorm before the block verifier runs, so this flag's
    /// attention-side effect is limited to direct block calls and block 0.
    ///
    /// Default: true (forward mode for practical verification)
    pub layernorm_forward_mode: bool,
    /// LayerNorm CROWN mode for per-position CROWN in the MLP subgraph.
    /// `Cut` is strictly sound; `Sampling` uses heuristic sampling and is not
    /// provably sound. Default: Cut.
    pub layernorm_crown_mode: LayerNormCrownMode,
    /// Use zonotope propagation for the attention suffix instead of pure IBP.
    /// Zonotopes track Q/K correlations through shared error symbols, giving
    /// tighter bounds for Q@K^T. The zonotope path currently roots that suffix
    /// at a conservative attention LayerNorm seam, then applies the configured
    /// LayerNorm mode to the suffix graph and the rest of the block. Provides
    /// additional tightening over pure forward-mode attention IBP at extra cost.
    /// Default: false (forward-mode LN provides the bulk of improvement)
    pub use_zonotope_attention: bool,
    /// Compatibility knob for future sequential zonotope backends.
    ///
    /// The current compositional Whisper verifier already rebuilds zonotope
    /// attention from each block's interval-valued input, so toggling this flag
    /// does not change today's results. It remains in the public config surface
    /// so a future shared-zonotope multi-block backend can opt into explicit
    /// block-boundary resets without another API break.
    /// Default: true (preserve the intended deep-transformer preset)
    pub reset_zonotope_between_blocks: bool,
    /// Enable decomposed-normalization CROWN inside the compositional block
    /// verifier's MLP subgraph.
    /// When true, `verify_block_compositional_gpu_with_config()` switches the
    /// MLP leg to `propagate_crown_within_graph_per_position_with_stats()`
    /// while attention still runs through the compositional IBP/zonotope
    /// routes. `verify_encoder_sequential_with_config()` intentionally keeps
    /// using the compositional block verifier because full-block CROWN through
    /// Whisper attention still degenerates at MatMul boundaries.
    /// Part of #318.
    /// Default: false
    pub use_crown_block_wise: bool,
}

impl Default for MultiBlockConfig {
    fn default() -> Self {
        // Default uses forward-mode LayerNorm for dramatically tighter bounds.
        // This provides up to 1e31x improvement on multi-block transformers.
        // For strictly sound (but potentially useless) bounds, use conservative().
        Self {
            max_bound_width: f32::MAX,
            terminate_on_overflow: false, // Match original verify_encoder_sequential
            continue_after_overflow: false, // Don't clamp, just let NaN propagate
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: true, // Forward mode for practical verification
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false, // IBP is sufficient with forward-mode LN
            reset_zonotope_between_blocks: true, // Compatibility default for future zonotope reuse
            use_crown_block_wise: false,
        }
    }
}

impl MultiBlockConfig {
    /// Create a conservative config with strictly sound LayerNorm bounds.
    ///
    /// WARNING: Conservative mode causes extreme bound explosion on multi-block
    /// transformers (bounds grow ~10^10 per block). Use only when strict mathematical
    /// soundness is required and you accept that results may be useless.
    ///
    /// For practical verification, use `default()` which enables forward-mode LayerNorm.
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

    /// Create a strict config that terminates early on any overflow.
    ///
    /// Uses forward-mode LayerNorm (like default) but stops verification
    /// if bounds exceed 1e20 or become NaN/Infinity.
    pub fn strict() -> Self {
        Self {
            max_bound_width: 1e20,
            terminate_on_overflow: true,
            continue_after_overflow: false,
            overflow_clamp_value: 1e30,
            layernorm_forward_mode: true, // Forward mode for practical bounds
            layernorm_crown_mode: LayerNormCrownMode::Cut,
            use_zonotope_attention: false,
            reset_zonotope_between_blocks: false,
            use_crown_block_wise: false,
        }
    }

    /// Create a diagnostic config that continues through overflow for analysis.
    ///
    /// Uses conservative LayerNorm (not forward mode) to help diagnose bound
    /// explosion patterns. Bounds are clamped to prevent NaN propagation.
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

    /// Alias for `default()` - forward-mode LayerNorm for tight bounds.
    ///
    /// Note: As of iteration #323, forward-mode LayerNorm is the default.
    /// This method is retained for backwards compatibility.
    #[deprecated(
        since = "0.1.0",
        note = "Use default() instead - forward-mode LN is now the default"
    )]
    pub fn tight_bounds() -> Self {
        Self::default()
    }

    /// Create a config optimized for tightest attention bounds using zonotope.
    /// Uses forward-mode LayerNorm for the block overall, but keeps the
    /// zonotope attention prefix seam conservative in
    /// `verify_block_compositional_gpu_with_config()`. The sequential verifier
    /// already rebuilds zonotope state per block, so the reset flag is
    /// currently a compatibility knob rather than an active transformation.
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

    /// Create a config optimized for deep transformers (28+ layers).
    /// Uses the same conservative zonotope attention prefix seam as
    /// `tightest_attention()`. In the current compositional verifier this is
    /// behaviorally identical, because each block already reconstructs a fresh
    /// zonotope from its interval input.
    ///
    /// Recommended for:
    /// - Qwen3, LLaMA, GPT models with many decoder layers
    /// - Models where bounds saturate before reaching the final layer
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

    /// Create a sound config that still applies zonotope tightening.
    ///
    /// Uses conservative (sound) LayerNorm bounds across the full block and
    /// enables zonotope attention. The reset flag stays enabled as a
    /// compatibility default for future shared-zonotope sequential backends.
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

    /// Set maximum bound width threshold.
    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_bound_width = max_width;
        self
    }

    /// Enable or disable forward mode for LayerNorm IBP.
    /// Forward mode uses center point for mean/std, giving dramatically tighter bounds.
    /// When zonotope attention is enabled, this does not relax the conservative
    /// attention-prefix seam. When `use_crown_block_wise` is enabled, the
    /// direct block verifier still keeps the MLP-side LayerNorm conservative
    /// even if this flag is `true`; multi-block encoder runs additionally force
    /// later blocks to conservative LayerNorm before dispatch.
    pub fn with_layernorm_forward_mode(mut self, enabled: bool) -> Self {
        self.layernorm_forward_mode = enabled;
        self
    }

    /// Set LayerNorm CROWN mode for per-position CROWN in the MLP subgraph.
    pub fn with_layernorm_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.layernorm_crown_mode = mode;
        self
    }

    /// Enable or disable early termination on overflow (NaN/Infinity).
    pub fn with_terminate_on_overflow(mut self, terminate: bool) -> Self {
        self.terminate_on_overflow = terminate;
        self
    }

    /// Enable or disable zonotope propagation for the attention suffix graph.
    /// Zonotopes track Q/K correlations for tighter Q@K^T bounds while keeping
    /// the attention-prefix seam conservative.
    pub fn with_zonotope_attention(mut self, enabled: bool) -> Self {
        self.use_zonotope_attention = enabled;
        self
    }

    /// Enable or disable the compatibility reset flag for future shared-zonotope
    /// sequential backends.
    ///
    /// The current compositional Whisper verifier already rebuilds zonotope
    /// attention from interval bounds at each block, so toggling this flag
    /// currently has no behavioral effect.
    pub fn with_reset_zonotope_between_blocks(mut self, enabled: bool) -> Self {
        self.reset_zonotope_between_blocks = enabled;
        self
    }

    /// Enable or disable the decomposed-norm CROWN path for the compositional
    /// verifier's MLP subgraph.
    /// This does not switch Whisper encoder verification to a full attention +
    /// MLP backward CROWN pass; it only changes the MLP leg inside
    /// `verify_block_compositional_gpu_with_config()`, where the MLP-side
    /// LayerNorm stays conservative while the attention side keeps the caller's
    /// block-local policy. Part of #318.
    pub fn with_crown_block_wise(mut self, enabled: bool) -> Self {
        self.use_crown_block_wise = enabled;
        self
    }
}
