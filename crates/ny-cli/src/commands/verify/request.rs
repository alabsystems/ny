// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Request builder for the `ny verify` command.

use std::path::PathBuf;

use crate::{BackendArg, LayerNormModeArg, LayerNormNormModeArg, MulBinaryRelaxationArg};

/// Immutable request for one verification run.
#[derive(Debug, Clone)]
pub(crate) struct VerificationConfig {
    pub(crate) model: PathBuf,
    pub(crate) property: Option<PathBuf>,
    pub(crate) epsilon: f32,
    pub(crate) method: String,
    pub(crate) mul_binary_relaxation: MulBinaryRelaxationArg,
    pub(crate) max_iterations: usize,
    pub(crate) tolerance: f32,
    pub(crate) timeout: u64,
    pub(crate) backend: BackendArg,
    pub(crate) gpu: bool,
    pub(crate) native: bool,
    pub(crate) conservative_layernorm: bool,
    pub(crate) layernorm_mode: LayerNormModeArg,
    pub(crate) layernorm_norm_mode: LayerNormNormModeArg,
    pub(crate) layer_by_layer: bool,
    pub(crate) block_wise: bool,
    pub(crate) progress: bool,
    pub(crate) progress_json: bool,
    pub(crate) max_blocks: usize,
    pub(crate) checkpoint: Option<PathBuf>,
    pub(crate) json: bool,
    pub(crate) strict: bool,
    pub(crate) require_sound: bool,
    pub(crate) allow_heuristic_logsoftmax: bool,
    pub(crate) allow_heuristic_softmax: bool,
    pub(crate) peel_off_last_softmax_layer: bool,
    pub(crate) allow_unknown: bool,
    pub(crate) double_fp: bool,
    pub(crate) shrink_eps: Option<f64>,
}

impl VerificationConfig {
    pub(crate) fn builder(
        model: PathBuf,
        epsilon: f32,
        method: String,
    ) -> VerificationConfigBuilder {
        VerificationConfigBuilder {
            model,
            property: None,
            epsilon,
            method,
            mul_binary_relaxation: MulBinaryRelaxationArg::Mccormick,
            max_iterations: 100,
            tolerance: 1e-4,
            timeout: 60,
            backend: BackendArg::Cpu,
            gpu: false,
            native: false,
            conservative_layernorm: false,
            layernorm_mode: LayerNormModeArg::Sound,
            layernorm_norm_mode: LayerNormNormModeArg::Standard,
            layer_by_layer: false,
            block_wise: false,
            progress: false,
            progress_json: false,
            max_blocks: 0,
            checkpoint: None,
            json: false,
            strict: false,
            require_sound: false,
            allow_heuristic_logsoftmax: false,
            allow_heuristic_softmax: false,
            peel_off_last_softmax_layer: false,
            allow_unknown: false,
            double_fp: false,
            shrink_eps: None,
        }
    }

    #[must_use]
    pub(crate) fn use_block_wise(&self) -> bool {
        self.block_wise || self.checkpoint.is_some()
    }
}

pub(crate) struct VerificationConfigBuilder {
    model: PathBuf,
    property: Option<PathBuf>,
    epsilon: f32,
    method: String,
    mul_binary_relaxation: MulBinaryRelaxationArg,
    max_iterations: usize,
    tolerance: f32,
    timeout: u64,
    backend: BackendArg,
    gpu: bool,
    native: bool,
    conservative_layernorm: bool,
    layernorm_mode: LayerNormModeArg,
    layernorm_norm_mode: LayerNormNormModeArg,
    layer_by_layer: bool,
    block_wise: bool,
    progress: bool,
    progress_json: bool,
    max_blocks: usize,
    checkpoint: Option<PathBuf>,
    json: bool,
    strict: bool,
    require_sound: bool,
    allow_heuristic_logsoftmax: bool,
    allow_heuristic_softmax: bool,
    peel_off_last_softmax_layer: bool,
    allow_unknown: bool,
    double_fp: bool,
    shrink_eps: Option<f64>,
}

impl VerificationConfigBuilder {
    pub(crate) fn property(mut self, property: Option<PathBuf>) -> Self {
        self.property = property;
        self
    }

    pub(crate) fn verification(
        mut self,
        mul_binary_relaxation: MulBinaryRelaxationArg,
        max_iterations: usize,
        tolerance: f32,
        timeout: u64,
    ) -> Self {
        self.mul_binary_relaxation = mul_binary_relaxation;
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self.timeout = timeout;
        self
    }

    pub(crate) fn backend(mut self, backend: BackendArg, gpu: bool) -> Self {
        self.backend = backend;
        self.gpu = gpu;
        self
    }

    pub(crate) fn native(mut self, native: bool) -> Self {
        self.native = native;
        self
    }

    pub(crate) fn layernorm(
        mut self,
        conservative_layernorm: bool,
        layernorm_mode: LayerNormModeArg,
        layernorm_norm_mode: LayerNormNormModeArg,
    ) -> Self {
        self.conservative_layernorm = conservative_layernorm;
        self.layernorm_mode = layernorm_mode;
        self.layernorm_norm_mode = layernorm_norm_mode;
        self
    }

    pub(crate) fn modes(
        mut self,
        layer_by_layer: bool,
        block_wise: bool,
        progress: bool,
        progress_json: bool,
        max_blocks: usize,
        checkpoint: Option<PathBuf>,
    ) -> Self {
        self.layer_by_layer = layer_by_layer;
        self.block_wise = block_wise;
        self.progress = progress;
        self.progress_json = progress_json;
        self.max_blocks = max_blocks;
        self.checkpoint = checkpoint;
        self
    }

    pub(crate) fn output(
        mut self,
        json: bool,
        strict: bool,
        require_sound: bool,
        allow_unknown: bool,
    ) -> Self {
        self.json = json;
        self.strict = strict;
        self.require_sound = require_sound;
        self.allow_unknown = allow_unknown;
        self
    }

    pub(crate) fn heuristics(
        mut self,
        allow_heuristic_logsoftmax: bool,
        allow_heuristic_softmax: bool,
        peel_off_last_softmax_layer: bool,
    ) -> Self {
        self.allow_heuristic_logsoftmax = allow_heuristic_logsoftmax;
        self.allow_heuristic_softmax = allow_heuristic_softmax;
        self.peel_off_last_softmax_layer = peel_off_last_softmax_layer;
        self
    }

    pub(crate) fn double_fp(mut self, double_fp: bool, shrink_eps: Option<f64>) -> Self {
        self.double_fp = double_fp;
        self.shrink_eps = shrink_eps;
        self
    }

    pub(crate) fn build(self) -> VerificationConfig {
        VerificationConfig {
            model: self.model,
            property: self.property,
            epsilon: self.epsilon,
            method: self.method,
            mul_binary_relaxation: self.mul_binary_relaxation,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            timeout: self.timeout,
            backend: self.backend,
            gpu: self.gpu,
            native: self.native,
            conservative_layernorm: self.conservative_layernorm,
            layernorm_mode: self.layernorm_mode,
            layernorm_norm_mode: self.layernorm_norm_mode,
            layer_by_layer: self.layer_by_layer,
            block_wise: self.block_wise,
            progress: self.progress,
            progress_json: self.progress_json,
            max_blocks: self.max_blocks,
            checkpoint: self.checkpoint,
            json: self.json,
            strict: self.strict,
            require_sound: self.require_sound,
            allow_heuristic_logsoftmax: self.allow_heuristic_logsoftmax,
            allow_heuristic_softmax: self.allow_heuristic_softmax,
            peel_off_last_softmax_layer: self.peel_off_last_softmax_layer,
            allow_unknown: self.allow_unknown,
            double_fp: self.double_fp,
            shrink_eps: self.shrink_eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VerificationConfig;
    use std::path::PathBuf;

    #[test]
    fn checkpoint_enables_block_wise_execution() {
        let config =
            VerificationConfig::builder(PathBuf::from("model.onnx"), 0.01, "crown".to_string())
                .modes(
                    false,
                    false,
                    false,
                    false,
                    0,
                    Some(PathBuf::from("checkpoint.json")),
                )
                .build();

        assert!(
            config.use_block_wise(),
            "checkpoint-backed verify runs must continue to take the block-wise path"
        );
    }

    #[test]
    fn disabled_block_wise_without_checkpoint_stays_disabled() {
        let config =
            VerificationConfig::builder(PathBuf::from("model.onnx"), 0.01, "crown".to_string())
                .modes(false, false, false, false, 0, None)
                .build();

        assert!(
            !config.use_block_wise(),
            "plain sequential verify runs should not switch to block-wise mode"
        );
    }
}
