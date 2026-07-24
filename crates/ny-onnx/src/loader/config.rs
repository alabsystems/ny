// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::LayerSpec;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Shape-inference strategy for ONNX loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapeInferencePolicy {
    /// Run ONNX Runtime shape inference before graph conversion.
    #[default]
    Ort,
    /// Skip ONNX Runtime shape inference and rely on proto-declared shapes only.
    Skip,
}

/// Where ONNX Runtime shape inference executes when the policy is
/// [`ShapeInferencePolicy::Ort`].
///
/// ORT shape inference is a blocking FFI call into ONNX Runtime's C++ that can
/// panic, abort, or fault on malformed models. Rust panic recovery cannot
/// contain native aborts or faults, so callers that must never crash (e.g.
/// competition verify lanes) can delegate the call to a child process instead:
/// the child dying is then observed as an ordinary inference failure and
/// degrades to the same graceful no-inferred-shapes fallback as any other
/// shape-inference error.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ShapeInferBackend {
    /// Run ORT shape inference in this process (default; no configuration
    /// needed, identical to historical behavior).
    #[default]
    InProcess,
    /// Run ORT shape inference in a child process: spawn `exe __shape-infer`,
    /// stream the model bytes over stdin, and read the inferred shape table
    /// from stdout (see `shape_infer::subprocess` for the wire protocol).
    /// Any subprocess failure — spawn error, crash/abort, timeout, or garbage
    /// output — is reported as an inference error, which loaders degrade to
    /// the no-inferred-shapes fallback. It is never treated as fatal and
    /// shapes are never fabricated.
    Subprocess {
        /// Binary that serves the `__shape-infer` protocol (typically
        /// `std::env::current_exe()` of the `ny` CLI).
        exe: PathBuf,
    },
}

/// ONNX conversion-time optimization flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxOptimizationFlag {
    /// Collapse single-consumer affine MatMul/Add chains into one Linear layer.
    MergeLinear,
}

/// Configuration for ONNX model loading.
#[derive(Clone, Default)]
pub struct OnnxLoadConfig {
    pub custom_ops: CustomOpRegistry,
    shape_inference: ShapeInferencePolicy,
    shape_infer_backend: ShapeInferBackend,
    optimization_flags: HashSet<OnnxOptimizationFlag>,
    capture_raw_float32_initializer_provenance: bool,
}

impl OnnxLoadConfig {
    /// Creates a config with the given custom op registry.
    pub fn new(custom_ops: CustomOpRegistry) -> Self {
        Self {
            custom_ops,
            shape_inference: ShapeInferencePolicy::default(),
            shape_infer_backend: ShapeInferBackend::default(),
            optimization_flags: HashSet::new(),
            capture_raw_float32_initializer_provenance: false,
        }
    }

    /// Returns the built-in defaults merged with user-provided custom ops.
    pub fn merged_registry(&self) -> CustomOpRegistry {
        CustomOpRegistry::defaults().merge(&self.custom_ops)
    }

    /// Returns a copy of this config with the requested shape-inference policy.
    #[must_use]
    pub fn with_shape_inference_policy(mut self, policy: ShapeInferencePolicy) -> Self {
        self.shape_inference = policy;
        self
    }

    /// Returns the configured shape-inference policy.
    pub fn shape_inference_policy(&self) -> ShapeInferencePolicy {
        self.shape_inference
    }

    /// Returns a copy of this config with the requested shape-inference
    /// execution backend (in-process vs. subprocess).
    #[must_use]
    pub fn with_shape_infer_backend(mut self, backend: ShapeInferBackend) -> Self {
        self.shape_infer_backend = backend;
        self
    }

    /// Returns the configured shape-inference execution backend.
    pub fn shape_infer_backend(&self) -> &ShapeInferBackend {
        &self.shape_infer_backend
    }

    /// Returns a copy of this config with one optimization flag enabled.
    #[must_use]
    pub fn with_optimization_flag(mut self, flag: OnnxOptimizationFlag) -> Self {
        self.optimization_flags.insert(flag);
        self
    }

    /// Returns a copy of this config with each requested optimization flag enabled.
    #[must_use]
    pub fn with_optimization_flags(
        mut self,
        flags: impl IntoIterator<Item = OnnxOptimizationFlag>,
    ) -> Self {
        self.optimization_flags.extend(flags);
        self
    }

    /// Returns whether the given optimization flag is enabled.
    pub fn has_optimization_flag(&self, flag: OnnxOptimizationFlag) -> bool {
        self.optimization_flags.contains(&flag)
    }

    /// Returns a copy of this config with loader-sealed raw ONNX FLOAT
    /// initializer and finalized-network provenance enabled or disabled.
    ///
    /// This is disabled by default. Enabling it adds exact-bit fingerprinting
    /// and private topology snapshots intended for explicitly qualified proof
    /// consumers; ordinary model loads pay none of that work or storage.
    #[must_use]
    pub fn with_raw_float32_initializer_provenance(mut self, enabled: bool) -> Self {
        self.capture_raw_float32_initializer_provenance = enabled;
        self
    }

    /// Returns whether loader-sealed raw FLOAT provenance is enabled.
    #[must_use]
    pub fn raw_float32_initializer_provenance_enabled(&self) -> bool {
        self.capture_raw_float32_initializer_provenance
    }
}

/// Registry of custom ONNX op handlers for model loading.
#[derive(Clone)]
pub struct CustomOpRegistry {
    handlers: Vec<Arc<dyn CustomOpHandler>>,
}

impl Default for CustomOpRegistry {
    fn default() -> Self {
        Self::defaults()
    }
}

impl CustomOpRegistry {
    /// Built-in custom-op handlers shipped with ny-onnx (empty by default).
    pub fn defaults() -> Self {
        Self {
            handlers: default_custom_handlers(),
        }
    }

    /// Registry containing only the provided handlers (no built-in defaults).
    pub fn from_handlers(handlers: Vec<Arc<dyn CustomOpHandler>>) -> Self {
        Self { handlers }
    }

    /// Appends a handler to this registry.
    pub fn register(&mut self, handler: Arc<dyn CustomOpHandler>) {
        self.handlers.push(handler);
    }

    /// Returns a new registry containing handlers from both `self` and `custom`.
    pub fn merge(&self, custom: &CustomOpRegistry) -> Self {
        let mut merged = Vec::with_capacity(self.handlers.len() + custom.handlers.len());
        merged.extend(self.handlers.iter().cloned());
        merged.extend(custom.handlers.iter().cloned());
        Self { handlers: merged }
    }

    /// Returns the registered handlers.
    pub fn handlers(&self) -> &[Arc<dyn CustomOpHandler>] {
        &self.handlers
    }
}

/// Handler that converts an ONNX node proto into a ny [`LayerSpec`].
///
/// Handlers participate only during ONNX loading. The returned `LayerSpec`
/// should describe a built-in layer type; ny does not preserve a
/// runtime custom-op node after conversion.
pub trait CustomOpHandler: Send + Sync {
    fn try_convert(&self, node: &onnx_proto::NodeProto) -> Option<LayerSpec>;

    fn try_convert_with_context(
        &self,
        node: &onnx_proto::NodeProto,
        opset_version: Option<i64>,
    ) -> Option<LayerSpec> {
        let _ = opset_version;
        self.try_convert(node)
    }

    fn supports(&self, op_type: &str) -> bool {
        let _ = op_type;
        false
    }

    fn supports_with_context(
        &self,
        op_type: &str,
        domain: &str,
        opset_version: Option<i64>,
    ) -> bool {
        let _ = domain;
        let _ = opset_version;
        self.supports(op_type)
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

fn default_custom_handlers() -> Vec<Arc<dyn CustomOpHandler>> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FirstHandler;
    struct SecondHandler;

    impl CustomOpHandler for FirstHandler {
        fn try_convert(&self, _node: &onnx_proto::NodeProto) -> Option<LayerSpec> {
            None
        }
    }

    impl CustomOpHandler for SecondHandler {
        fn try_convert(&self, _node: &onnx_proto::NodeProto) -> Option<LayerSpec> {
            None
        }
    }

    #[test]
    fn registry_merge_preserves_order() {
        let base = CustomOpRegistry::from_handlers(vec![Arc::new(FirstHandler)]);
        let extra = CustomOpRegistry::from_handlers(vec![Arc::new(SecondHandler)]);
        let merged = base.merge(&extra);
        let handler_names: Vec<&str> = merged
            .handlers()
            .iter()
            .map(|handler| handler.name())
            .collect();
        assert_eq!(
            handler_names,
            vec![
                "ny_onnx::loader::config::tests::FirstHandler",
                "ny_onnx::loader::config::tests::SecondHandler"
            ]
        );
    }

    #[test]
    fn config_defaults_to_ort_shape_inference() {
        let config = OnnxLoadConfig::default();
        assert_eq!(config.shape_inference_policy(), ShapeInferencePolicy::Ort);
        assert!(!config.raw_float32_initializer_provenance_enabled());
    }

    #[test]
    fn config_shape_inference_builder_overrides_default() {
        let config =
            OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip);
        assert_eq!(config.shape_inference_policy(), ShapeInferencePolicy::Skip);
    }

    #[test]
    fn config_defaults_to_in_process_shape_infer_backend() {
        let config = OnnxLoadConfig::default();
        assert_eq!(config.shape_infer_backend(), &ShapeInferBackend::InProcess);
    }

    #[test]
    fn config_shape_infer_backend_builder_overrides_default() {
        let exe = PathBuf::from("/path/to/ny");
        let config = OnnxLoadConfig::default()
            .with_shape_infer_backend(ShapeInferBackend::Subprocess { exe: exe.clone() });
        assert_eq!(
            config.shape_infer_backend(),
            &ShapeInferBackend::Subprocess { exe }
        );
        // The backend is orthogonal to the policy: setting it must not change
        // the policy default.
        assert_eq!(config.shape_inference_policy(), ShapeInferencePolicy::Ort);
    }

    #[test]
    fn config_optimization_flag_builder_enables_flag() {
        let config =
            OnnxLoadConfig::default().with_optimization_flag(OnnxOptimizationFlag::MergeLinear);
        assert!(config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear));
    }

    #[test]
    fn config_builder_composition_preserves_existing_fields() {
        let registry = CustomOpRegistry::from_handlers(vec![Arc::new(FirstHandler)]);
        let config = OnnxLoadConfig::new(registry)
            .with_shape_inference_policy(ShapeInferencePolicy::Skip)
            .with_raw_float32_initializer_provenance(true)
            .with_optimization_flags([OnnxOptimizationFlag::MergeLinear]);

        assert_eq!(config.shape_inference_policy(), ShapeInferencePolicy::Skip);
        assert_eq!(config.merged_registry().handlers().len(), 1);
        assert!(config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear));
        assert!(config.raw_float32_initializer_provenance_enabled());
    }
}
