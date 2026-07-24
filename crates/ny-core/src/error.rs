// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Macro to create a ShapeMismatch error.
#[macro_export]
macro_rules! shape_mismatch_err {
    ($expected:expr, $got:expr) => {{
        let exp: Vec<usize> = $expected;
        let got: Vec<usize> = $got;
        $crate::NyError::shape_mismatch(exp, got)
    }};
}

/// Error types for ny operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NyError {
    /// Tensor shape mismatch during bound propagation.
    ///
    /// Occurs when tensor operations receive inputs with incompatible shapes.
    /// Common causes: incorrect model conversion, mismatched layer dimensions,
    /// or specification bounds that don't match model input shape.
    ///
    /// # Handling
    /// - Verify input specification matches model's expected input shape
    /// - Check ONNX model conversion for shape inference errors
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Layer type not supported by bound propagation engine.
    ///
    /// Returned when the model contains a layer type that ny does not
    /// implement bound propagation for (e.g., custom training-only layers).
    ///
    /// # Handling
    /// - Replace with supported equivalent layer if possible
    /// - Use `CustomOpSchemaRegistry` to register custom bound propagation
    #[error("Unsupported layer type: {0}")]
    UnsupportedLayer(String),

    /// ONNX operation not supported by bound propagation.
    ///
    /// Returned when attempting to propagate through an ONNX operation that
    /// ny does not have a `BoundPropagation` implementation for. Callers
    /// often catch this to fall back to IBP bounds. Different from
    /// `UnsupportedLayer` (layer type not implemented at all) and
    /// `UnsupportedConfiguration` (valid operation in unsupported mode).
    ///
    /// # Handling
    /// - Check if operation can be replaced with supported equivalent
    /// - Use `CustomOpSchemaRegistry` for custom operations
    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    /// Model file loading or parsing failed.
    ///
    /// Returned when ONNX/SafeTensors/CoreML model cannot be loaded.
    /// Wraps underlying I/O and parsing errors.
    ///
    /// # Handling
    /// - Verify file path exists and is readable
    /// - Check model file format is supported (ONNX opset version)
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    /// Invalid verification specification (VNN-LIB parsing error).
    ///
    /// Returned when parsing VNN-LIB specifications fails due to syntax errors,
    /// invalid constraints, or unsupported specification features.
    ///
    /// # Handling
    /// - Validate VNN-LIB syntax against specification
    /// - Check variable names match model input/output names
    #[error("Invalid specification: {0}")]
    InvalidSpec(String),

    /// Numerical instability detected during computation.
    ///
    /// Returned when operations would produce NaN/Inf due to numerical issues,
    /// before they corrupt the verification result.
    ///
    /// # Handling
    /// - Check for near-zero divisors in the model
    /// - Reduce input bound width to improve numerical conditioning
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Feature, mode, or configuration not supported.
    ///
    /// Returned when a valid operation is attempted in an unsupported mode or
    /// configuration (e.g., self-attention on sequential networks, missing input
    /// shapes for convolution, GPU backend unavailable). Different from
    /// `UnsupportedOp` (operation lacks bound propagation implementation) and
    /// `UnsupportedLayer` (layer type not implemented at all).
    ///
    /// # Handling
    /// - Check documentation for supported configurations
    /// - File issue if feature should be supported
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),

    /// Soundness policy refused an operation.
    ///
    /// Returned when a layer has a CROWN implementation but refuses to use it
    /// because the current mode requires sound bounds and the implementation
    /// is heuristic. Unlike `UnsupportedOp`, this error MUST NOT trigger
    /// silent fallback to IBP — the user explicitly requested sound verification.
    ///
    /// # Handling
    /// - Change the layer's mode to allow heuristic bounds (e.g., `--layernorm-mode sampling`)
    /// - Use a verification method that doesn't need CROWN for this layer (e.g., IBP-only)
    /// - Cut CROWN propagation at this layer's boundary (e.g., `--layernorm-mode cut`)
    #[error("Soundness refusal: {0}")]
    SoundnessRefusal(String),

    /// Invalid configuration parameter.
    ///
    /// Returned when optimizer or verification configuration contains values
    /// outside valid ranges (e.g., negative learning rates, NaN parameters).
    /// These are user-facing errors from CLI flags, YAML config, or presets.
    ///
    /// # Handling
    /// - Check the error message for which parameter is invalid and the valid range
    /// - Fix the CLI flag, YAML config file, or preset definition
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Internal invariant violation.
    ///
    /// Returned when an internal assumption is violated (e.g., array reshape
    /// failure, non-contiguous memory layout). These indicate logic bugs
    /// rather than user-facing errors.
    ///
    /// # Handling
    /// - Report as a bug with the error message
    #[error("Internal error: {0}")]
    InternalError(String),

    /// GPU memory budget exceeded (#3515).
    ///
    /// Returned when the estimated GPU memory footprint for CROWN backward
    /// exceeds the configured budget. The caller should fall back to CPU
    /// CROWN backward or reduce the spec batch size.
    ///
    /// # Handling
    /// - Caller falls back to CPU (see `crown.rs` GPU backward fallback)
    /// - Increase budget via `NY_GPU_MEMORY_BUDGET_MB` env var
    #[error(
        "GPU memory exceeded: requires {required_bytes} bytes but budget is {budget_bytes} bytes"
    )]
    GpuMemoryExceeded {
        required_bytes: usize,
        budget_bytes: usize,
    },

    /// CPU dense-materialization budget exceeded (#3550).
    ///
    /// Returned when the estimated CPU memory for a batched dense identity or
    /// Patches-to-Dense conversion exceeds `NY_DENSE_BUDGET_MB`.
    /// The caller should fall back to unbatched CROWN or IBP bounds.
    ///
    /// # Handling
    /// - Sequential batched `Network`: falls back to unbatched `propagate_crown()`
    /// - Graph batched CROWN: falls back to unbatched `propagate_crown_with_provenance()`
    /// - Block-wise CROWN: uses IBP block width (`crown_successful = false`)
    /// - Streaming batched CROWN: falls back to regular streaming CROWN
    /// - Increase budget via `NY_DENSE_BUDGET_MB` env var
    #[error(
        "CPU memory exceeded at {site}: requires {required_bytes} bytes but budget is {budget_bytes} bytes"
    )]
    CpuMemoryExceeded {
        required_bytes: usize,
        budget_bytes: usize,
        site: &'static str,
    },

    /// Per-node deadline exceeded during CROWN backward propagation (#3795).
    ///
    /// Returned when a single-node backward step (e.g., Conv2d transpose GEMM)
    /// exceeds the per-node time budget. The caller should fall back to IBP
    /// bounds for that target node instead of stalling the entire pass.
    ///
    /// # Handling
    /// - Graph CROWN: fall back to IBP for the target node
    /// - Alpha-CROWN: fall back to IBP for the exceeded node, continue with others
    /// - BaB loop: treat as degraded but not failed
    #[error("Deadline exceeded: {0}")]
    DeadlineExceeded(String),

    /// Domain is infeasible: constraint interactions produce empty intersection.
    ///
    /// Returned during constrained forward CROWN when split constraints create
    /// contradictory intermediate bounds (lower > upper for some neuron). An
    /// infeasible domain is empty — no input satisfies all constraints
    /// simultaneously. The BaB loop should treat this as "trivially verified"
    /// (the property holds vacuously on the empty set), not as a propagation
    /// failure.
    ///
    /// Reference: alpha-beta-CROWN `infeasible_bounds_constraints` in
    /// `input_split/bounding.py:78-82,149-152`.
    ///
    /// # Handling
    /// - BaB loop: prune the domain (don't add to queue, don't mark as failure)
    /// - Callers MUST NOT count this as a propagation failure (#2926)
    #[error("Infeasible domain: {0}")]
    InfeasibleDomain(String),

    /// Error occurred in a specific network layer.
    ///
    /// Wrapper that adds layer context to an underlying error, enabling
    /// precise error localization in deep networks.
    ///
    /// # Handling
    /// - Inspect the `source` error for root cause
    /// - Use `layer_index` and `layer_type` to locate problematic layer
    #[error("Layer {layer_index} ({layer_type}) failed: {source}")]
    LayerError {
        layer_index: usize,
        layer_type: String,
        #[source]
        source: Box<NyError>,
    },
}

impl NyError {
    /// Create a ShapeMismatch error. Identical shapes are a bug indicator, but
    /// should not panic in production error paths.
    #[track_caller]
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>) -> Self {
        NyError::ShapeMismatch { expected, got }
    }

    /// Returns true if this error represents a per-node deadline exceeded (#3795).
    ///
    /// Used by fallback classification to structurally identify deadline
    /// timeouts without fragile string matching.
    pub fn is_deadline_exceeded(&self) -> bool {
        matches!(self, NyError::DeadlineExceeded(_))
    }

    /// Returns true if this error represents a CPU dense-materialization budget
    /// overflow (#conv-crown-oom).
    ///
    /// Used by CROWN-IBP fallback classification to structurally identify a
    /// memory-cap trip (e.g. the Conv2d backward coefficient-buffer backstop) so
    /// the affected target degrades to sound IBP bounds instead of aborting.
    pub fn is_cpu_memory_exceeded(&self) -> bool {
        matches!(self, NyError::CpuMemoryExceeded { .. })
    }

    /// Returns true if this error represents an infeasible domain.
    ///
    /// Used by the BaB loop to distinguish infeasible domains (trivially
    /// verified — empty set) from genuine propagation failures (#2926).
    pub fn is_infeasible_domain(&self) -> bool {
        matches!(self, NyError::InfeasibleDomain(_))
    }
}

/// Convenience alias for `std::result::Result<T, NyError>`.
pub type Result<T> = std::result::Result<T, NyError>;
