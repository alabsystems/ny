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
    ///
    /// File models with external tensor data use model-authored shapes instead:
    /// ONNX Runtime cannot consume NY's capability-scoped file handles and
    /// must not reopen untrusted sidecar paths through ambient authority.
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
/// degrades to the same ORT-free static shape fallback as any other
/// shape-inference error (exact rules only; every tensor they cannot prove
/// stays unknown, as it did before that fallback existed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeInferBackend {
    /// Run ORT shape inference in this process (the default wherever the
    /// engine reports failure rather than aborting; identical to historical
    /// behavior).
    InProcess,
    /// Run ORT shape inference in a child process: spawn `exe __shape-infer`,
    /// stream the model bytes over stdin, and read the inferred shape table
    /// from stdout (see `shape_infer::subprocess` for the wire protocol).
    /// Any subprocess failure — spawn error, crash/abort, timeout, or garbage
    /// output — is reported as an inference error, which loaders degrade to
    /// the ORT-free static shape fallback. It is never treated as fatal, and
    /// nothing from the failed exchange is ever used.
    Subprocess {
        /// Binary that serves the `__shape-infer` protocol (typically
        /// `std::env::current_exe()` of the `ny` CLI).
        exe: PathBuf,
    },
}

/// The default backend depends on WHICH ENGINE this build links, because the
/// two engines fail differently.
///
/// The C++ runtime reports a model it cannot handle as an error, which the
/// in-process worker thread turns into an ordinary inference failure — so
/// in-process is both cheaper and sufficient there, and stays the default.
///
/// tract can instead panic across the C ABI while translating a graph, and a
/// panic in a function that cannot unwind ABORTS: no Rust boundary in this
/// process can contain it. That is not hypothetical — it is what ACASXU does,
/// measured across the 2021 through 2026 VNN-COMP suites, and ACASXU is the
/// most-run benchmark family there is. A child process is the only boundary
/// that holds, and this module's whole purpose is to turn such a death into an
/// ordinary inference error that loaders degrade from gracefully. So on a
/// tract build the containment is the default rather than something a caller
/// must know to ask for: a verifier that aborts on the canonical benchmark is
/// worse than one that reports it could not infer shapes.
///
/// `current_exe` failing leaves nothing to spawn, so that falls back to
/// in-process rather than pretending containment exists.
impl Default for ShapeInferBackend {
    fn default() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        if let Ok(exe) = std::env::current_exe() {
            return Self::Subprocess { exe };
        }
        Self::InProcess
    }
}

/// ONNX conversion-time optimization flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxOptimizationFlag {
    /// Collapse single-consumer affine MatMul/Add chains into one Linear layer.
    MergeLinear,
}

/// BatchNormalization rewrite policy for ONNX graph conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchNormFoldingPolicy {
    /// Preserve the historical loader behavior.
    ///
    /// Conv/Gemm folds are enabled and the extended ConvTranspose/cGAN folds
    /// remain governed by the `NY_BN_FOLD_EXT` process environment variable.
    #[default]
    LegacyEnvironment,
    /// Preserve every authored BatchNormalization node.
    ///
    /// This is a call-local override. It dominates `NY_BN_FOLD_EXT` and the
    /// loader does not consult or mutate that variable for this decision.
    PreserveRaw,
    /// Internal, call-local structural selection before authored graph rewrites.
    /// Preset serialization cannot request this mode; the frozen query policy
    /// selects it only when no explicit folding policy was supplied.
    #[serde(skip)]
    StructuralCertifiedRoot,
}

/// Actual policy used by the same synchronous model load, never a proposed flag.
#[derive(Clone, Copy, Debug)]
pub struct RootCutLoadReceipt {
    policy: BatchNormFoldingPolicy,
    structurally_admitted: bool,
}
impl RootCutLoadReceipt {
    pub(crate) fn new(policy: BatchNormFoldingPolicy, structurally_admitted: bool) -> Self {
        Self {
            policy,
            structurally_admitted,
        }
    }
    pub fn batch_norm_folding(&self) -> BatchNormFoldingPolicy {
        self.policy
    }
    pub fn certified_root_source(&self) -> bool {
        self.policy == BatchNormFoldingPolicy::PreserveRaw && self.structurally_admitted
    }
}

/// Configuration for ONNX model loading.
#[derive(Clone, Default)]
pub struct OnnxLoadConfig {
    pub custom_ops: CustomOpRegistry,
    shape_inference: ShapeInferencePolicy,
    shape_infer_backend: ShapeInferBackend,
    optimization_flags: HashSet<OnnxOptimizationFlag>,
    batch_norm_folding: BatchNormFoldingPolicy,
    capture_raw_float32_initializer_provenance: bool,
    require_authored_float32_initializers: bool,
    qdq_activation_relaxation: bool,
    multi_input_concat: bool,
}

impl OnnxLoadConfig {
    /// Creates a config with the given custom op registry.
    pub fn new(custom_ops: CustomOpRegistry) -> Self {
        Self {
            custom_ops,
            shape_inference: ShapeInferencePolicy::default(),
            shape_infer_backend: ShapeInferBackend::default(),
            optimization_flags: HashSet::new(),
            batch_norm_folding: BatchNormFoldingPolicy::default(),
            capture_raw_float32_initializer_provenance: false,
            require_authored_float32_initializers: false,
            qdq_activation_relaxation: false,
            multi_input_concat: false,
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

    /// Returns a copy of this config with the requested BatchNormalization
    /// rewrite policy.
    #[must_use]
    pub fn with_batch_norm_folding_policy(mut self, policy: BatchNormFoldingPolicy) -> Self {
        self.batch_norm_folding = policy;
        self
    }

    /// Returns the configured BatchNormalization rewrite policy.
    #[must_use]
    pub fn batch_norm_folding_policy(&self) -> BatchNormFoldingPolicy {
        self.batch_norm_folding
    }

    /// #soundness-spine M2e: the load configuration every CERTIFIED lowering of
    /// a `GraphNetwork` (ENGINE_PROGRAM §1.6, the BN-fold row) must start from.
    ///
    /// The Conv/Gemm <- BatchNormalization fold stores a binary32 POINT for
    /// each composed coefficient and DROPS the coefficient enclosure
    /// (`fusion/batch_norm_fold.rs`: "Nothing downstream absorbs this"), about
    /// `2^-24 · S` per folded layer. A certified lowering therefore keeps every
    /// authored BatchNormalization node: `BatchNormLayer` encloses the exact
    /// real affine in outward-rounded binary64 and its consumers charge
    /// `scale_err`/`bias_err` outward. This stays the rule until the licensed
    /// `kernel_err` channel carries the fold's enclosure into the conv/linear
    /// certificates, at which point the fold may be re-admitted with its error
    /// charged.
    ///
    /// WHO LOADS THROUGH IT (#spine-legacy-lanes). Every load that claims
    /// proof authority over the AUTHORED model apart from the β-CROWN graph
    /// lane's own preset load: the relational difference-network pair
    /// (`ny-cli` `load_certified_graph_network`), the cctsdb sub-box model, the
    /// non-linear interval branch-and-bound's f64 cell graph, the
    /// affine-invprop strict reload, and — via the preset predicate
    /// [`Self::admits_certified_graph_lowering`] — the forward-linear
    /// spec-alpha surrogate (cGAN presets set `batch_norm_folding:
    /// preserve_raw`). None of those lanes' models carries a
    /// BatchNormalization today, so the switch is byte-identical on their
    /// banks; it closes the fold for any future BN model.
    ///
    /// NOT applied to today's β-CROWN graph lane: refusing the fold there is
    /// the measured −15 banked `unsat` intervention (#bn-fold-restore), so it
    /// would DOWNGRADE banks. That live gap is reported, not closed, here. Its
    /// exposure (census 2026-09-29, `BatchNormalization` bytes in every model
    /// of `benchmarks/vnncomp2026`): BN models exist only in cgan (loaded
    /// `preserve_raw`), cifar100 and tinyimagenet (banked `unsat` rows are
    /// margin-row verdicts, whose extractor folds from the raw proto in
    /// binary64 with a charged error), traffic_signs (no banked `unsat`),
    /// vit_2023 (`unsat` unbanked) and challenging_certified_training_2026,
    /// whose 3 banked `unsat` rows are graph-lane verdicts on folded CNN7s —
    /// the only banked rows the uncharged fold can touch.
    #[must_use]
    pub fn certified_graph_lowering(self) -> Self {
        self.with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
    }

    /// Whether a model loaded with this configuration may feed a CERTIFIED
    /// `GraphNetwork` lowering (see [`Self::certified_graph_lowering`]): only
    /// when no BatchNormalization was folded with its enclosure dropped.
    #[must_use]
    pub fn admits_certified_graph_lowering(&self) -> bool {
        self.batch_norm_folding == BatchNormFoldingPolicy::PreserveRaw
    }

    /// Returns a copy of this config with the dark QDQ activation-relaxation
    /// lane enabled (see `model.qdq_activation_relaxation`).
    ///
    /// Default OFF: the loader refuses static-INT64->FLOAT Cast plumbing and
    /// activation QuantizeLinear exactly as before. ON arms BOTH halves of the
    /// mechanism: (a) a `Cast(to=FLOAT)` whose data input is a PROVEN-STATIC
    /// INT64 structural value is accepted as a deferred-fold obligation (the
    /// original refusal re-raises unless constant folding erases the Cast into
    /// a literal before propagation), and (b) an activation
    /// `QuantizeLinear -> DequantizeLinear` pair with identical parameters is
    /// fused into the certified `QdqPerturbationLayer` enclosure.
    #[must_use]
    pub fn with_qdq_activation_relaxation(mut self, enabled: bool) -> Self {
        self.qdq_activation_relaxation = enabled;
        self
    }

    /// Returns whether the dark QDQ activation-relaxation lane is enabled.
    #[must_use]
    pub fn qdq_activation_relaxation_enabled(&self) -> bool {
        self.qdq_activation_relaxation
    }

    /// Returns a copy of this config with the dark multi-input concat-view
    /// mechanism enabled (see `model.multi_input_concat`).
    ///
    /// Default OFF: a model with more than one declared graph input keeps
    /// today's fail-closed refusal (the input box is sized from the FIRST
    /// input only, and the VNN-LIB coordinate-count validation rejects the
    /// spec). ON, the graph builder concatenates all declared inputs, in
    /// declared order, into one flat network input and carves each input back
    /// out with exact Slice(+Reshape) view nodes; the CLI sizes the joint
    /// input box to the concatenated element count. Single-input models are
    /// byte-identical in both states. This flag rides `OnnxLoadConfig` purely
    /// as preset->loader transport (like the QDQ lane); the mechanism itself
    /// lives in ny-build's graph builder.
    #[must_use]
    pub fn with_multi_input_concat(mut self, enabled: bool) -> Self {
        self.multi_input_concat = enabled;
        self
    }

    /// Returns whether the dark multi-input concat-view mechanism is enabled.
    #[must_use]
    pub fn multi_input_concat_enabled(&self) -> bool {
        self.multi_input_concat
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

    /// Require every raw ONNX FLOAT initializer to survive loading unchanged.
    ///
    /// Enabling the admission guard also enables loader-private provenance
    /// capture. A mismatch or an empty/absent provenance set fails the load
    /// before any verifier can consume the model.
    #[must_use]
    pub fn with_require_authored_float32_initializers(mut self, required: bool) -> Self {
        self.require_authored_float32_initializers = required;
        if required {
            self.capture_raw_float32_initializer_provenance = true;
        }
        self
    }

    /// Whether unchanged authored FLOAT initializers are a load-time
    /// requirement.
    #[must_use]
    pub fn require_authored_float32_initializers(&self) -> bool {
        self.require_authored_float32_initializers
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
        assert_eq!(
            config.batch_norm_folding_policy(),
            BatchNormFoldingPolicy::LegacyEnvironment
        );
        assert!(!config.raw_float32_initializer_provenance_enabled());
        assert!(!config.require_authored_float32_initializers());
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
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_require_authored_float32_initializers(true)
            .with_optimization_flags([OnnxOptimizationFlag::MergeLinear]);

        assert_eq!(config.shape_inference_policy(), ShapeInferencePolicy::Skip);
        assert_eq!(
            config.batch_norm_folding_policy(),
            BatchNormFoldingPolicy::PreserveRaw
        );
        assert_eq!(config.merged_registry().handlers().len(), 1);
        assert!(config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear));
        assert!(config.raw_float32_initializer_provenance_enabled());
        assert!(config.require_authored_float32_initializers());
    }
}
